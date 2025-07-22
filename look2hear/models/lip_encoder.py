import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm


class LipDataset(Dataset):
    def __init__(self, npz_dirs, spk_list):
        """
        npz_dirs: str 或 List[str]，每个路径下都包含 .npz 文件
        """
        if isinstance(npz_dirs, str):
            npz_dirs = [npz_dirs]

        self.files = []
        for d in npz_dirs:
            self.files += sorted(glob.glob(os.path.join(d, "*.npz")))

        self.transform = transforms.Compose([
                         transforms.ToPILImage(),  # # 先转成 PIL 图像以支持 Resize
                         transforms.Resize((112, 112)),
                         transforms.ToTensor(),  # 转成 PyTorch 的 tensor 格式，并且把像素值 归一化到 [0, 1] 区间
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
        self.spk_list = self._load_spk(spk_list)
        # print("Loaded speaker list:", self.spk_list)
        # print("Total speakers:", len(self.spk_list))

    def _load_spk(self, spk_list_path):
        if spk_list_path is None:
            return []
        with open(spk_list_path) as f:
            lines = f.readlines()
        return [line.strip() for line in lines]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        filename = os.path.basename(file)
        data = np.load(file)
        video = data["data"]  # [T, 96, 96]
        spk_id = os.path.basename(filename).split('_')[0]
        # print("type:", type(spk_id))
        # print("spk_id:", spk_id)
        spk_idx_label = self.spk_list.index(spk_id)
        # print(spk_idx_label)

        frames = [self.transform(frame) for frame in video]  # 每帧：[1, 112, 112]
        frames = torch.stack(frames)              # [T,,3, 112, 112 ]
        # print(frames.shape)
        return frames, spk_idx_label

def collate_fn(batch):

    videos, labels = zip(*batch)
    max_frames = max([video.size(0) for video in videos])
    padded_videos = []
    for video in videos:
        frames = video.size(0)
        T, C, H, W = video.shape
        if frames < max_frames:
            padding = max_frames - frames
            padded_vi = torch.zeros((padding, C, H, W), dtype=video.dtype)
            padded_video = torch.cat([video, padded_vi], dim=0)
            # padded_video = F.pad(video, (0, 0, 0, padding), mode='constant', value=0)  # 填充在时间维度上
        elif frames > max_frames:
            padded_video = video[:max_frames]  # 如果帧数多，裁剪掉多余的部分
        else:
            padded_video = video    
        padded_videos.append(padded_video)
    
    # 将视频列表堆叠成一个张量，形状为 [B, T, C, H, W]
    padded_videos = torch.stack(padded_videos, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_videos, labels

# train: 2936 speakers
# val: 68 speakers
# test: 148 speakers


class ResBlock_2d(nn.Module):

    def __init__(self, in_dims, out_dims):
        super(ResBlock_2d, self).__init__()
        self.conv1 = nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_dims)
        self.batch_norm2 = nn.BatchNorm2d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        # print(y.shape)
        return self.maxpool(y)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):  # [B, C, T]
        return self.relu(self.norm(self.conv(x)))

class LipEncoderClassifier(nn.Module):
    def __init__(self, num_speakers, tcn_channels=256, lip_emb_dim=256):
        super().__init__()

        # 使用自定义的 ResBlock 替换 ResNet
        self.resblocks = nn.Sequential(
            ResBlock_2d(1, 64),  # 输入通道数为 1
            ResBlock_2d(64, 128),
            ResBlock_2d(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )

        self.temporal_conv = nn.Sequential(
            TCNBlock(256, 256, dilation=1),
            TCNBlock(256, tcn_channels, dilation=2),
            TCNBlock(tcn_channels, tcn_channels, dilation=4),
        )

        self.proj = nn.Linear(tcn_channels, lip_emb_dim)
        # self.classifier = nn.Linear(lip_emb_dim, num_speakers)

        # 分类器头部，用于预测 speaker ID
        self.classifier = nn.Linear(lip_emb_dim, num_speakers)  # 这个在训练的时候应该在·lip_encoder的外面

    def forward(self, x):  # [B, T, 1, 112, 112] 灰色图像
        # # x = x.unsqueeze(2)
        # if x.ndim == 4:
        #     x = x.unsqueeze(2)
        # B, T, C, H, W = x.shape
        # x = x.reshape(B*T, C, H, W)  # 展平为 (B*T, C, H, W)
        # feat = self.resblocks(x)  # 输出的特征
        # print(feat.shape)  # torch.Size([20, 512, 1, 1])
        
        # feat = feat.view(B, T, -1).transpose(1, 2)


        # tcn_out = self.temporal_conv(feat)
        # # print(tcn_out.shape)
        # pooled = tcn_out.mean(dim=2)
        # emb = self.proj(pooled)  # [B, lip_emb_dim]
        # # logits = self.classifier(emb)  # [B, num_speakers]
        if x.ndim == 4:
            x = x.unsqueeze(2)  # [B, T, 1, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)  # [B*T, C, H, W]
        
        feat = self.resblocks(x)  # [B*T, 256, 1, 1]
        feat = feat.view(B, T, -1).transpose(1, 2)  # [B, 256, T]

        tcn_out = self.temporal_conv(feat)  # [B, tcn_channels, T]

        emb = self.proj(tcn_out.transpose(1, 2))  # [B, T, tcn_channels] -> [B, T, lip_emb_dim]

        pooled = emb.mean(dim=1)
        logits = self.classifier(pooled)  # [B, num_speakers]

        
            
        return logits, emb

def main():

    batch_size = 16
    num_workers = 4
    num_epochs = 100
    learning_rate = 1e-4
    num_speakers = 3096 # 你统计的 speaker 数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 设置训练集、验证集、测试集的路径
    train_dirs = ["/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/train/video1", 
                "/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/train/video2", 
                "/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/test/video1", 
                "/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/test/video2",
                "/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/valid/video1",
                "/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/valid/video2"]
    valid_dirs = ["/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/valid/video1",
                  "/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/test/video1",
                  "/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/test/video2"]
    test_dirs = ["/home/xueke/dataset/LRS2/mvlrs_v1/raw_data/valid/video2"]

    train_spk_list = "/home/xueke/dataset/LRS2/mvlrs_v1/train+test_id1.spk"
    valid_spk_list = "/home/xueke/dataset/LRS2/mvlrs_v1/train+test_id1.spk"
    test_spk_list = "/home/xueke/dataset/LRS2/mvlrs_v1/train+test_id1.spk"

    train_dataset = LipDataset(npz_dirs=train_dirs, spk_list=train_spk_list)
    valid_dataset = LipDataset(npz_dirs=valid_dirs, spk_list=valid_spk_list)
    test_dataset = LipDataset(npz_dirs=test_dirs, spk_list=test_spk_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # # 测试一下，打印一个 batch 的 shape 和 label
    # for frames, label in train_loader:
    #     print("Batch frames shape:", frames.shape)  # [B, T, 3, H, W]
    #     print("Batch label shape:", label.shape)  # [B] 例如 [32] 表示 32 个样本的标签
    #     print("Batch label shape:", label)
    #     break

    model = LipEncoderClassifier(num_speakers=num_speakers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化最佳验证准确率和早停参数
    best_val_acc = 0.0
    patience = 15  # 假设设置最多容忍5个epoch的验证准确率无提升
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_loss, total_correct, total_samples = 0, 0, 0

        # for videos, labels in train_loader:
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Train", unit="batch") as pbar:
            for videos, labels in pbar:
                videos, labels = videos.to(device), labels.to(device)
                logits, _ = model(videos)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # # calculate accuracy
                # probs = th.softmax(spk_pred, dim=1)
                # est_label = probs.argmax(dim=1)
                # correct = (est_label==egs["spk_idx"]).sum().item()

                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += videos.size(0)
                 # 更新进度条的描述信息
                pbar.set_postfix(loss=running_loss / (pbar.n + 1), acc=total_correct / total_samples)

        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

         # 验证部分
        model.eval()
        with torch.no_grad():
            total_correct, total_samples = 0, 0
            with tqdm(valid_loader, desc=f"Epoch {epoch}/{num_epochs} Val", unit="batch") as pbar:
                for videos, labels in pbar:
                    videos, labels = videos.to(device), labels.to(device)
                    logits, _ = model(videos)
                    total_correct += (logits.argmax(1) == labels).sum().item()
                    total_samples += videos.size(0)
                    # 更新验证进度条的描述信息
                pbar.set_postfix(acc=total_correct / total_samples)

            val_acc = total_correct / total_samples
            print(f"[Epoch {epoch}] Val Acc: {val_acc:.4f}")

            # 如果验证集准确率提升，保存模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"Validation accuracy improved! Saving model...")
                save_path = os.path.join("/home/xueke/wav2vec_TCN/lip_encoder_checkpoint1", f"best_model_epoch_{epoch}.pt")
                torch.save(model.state_dict(),save_path)
            else:
                patience_counter += 1
                print(f"Validation accuracy did not improve. Patience counter: {patience_counter}/{patience}")

            # 早停：如果验证集准确率在若干个epoch内没有提升，则停止训练
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

if __name__ == '__main__':
    # main()
    model = LipEncoderClassifier(num_speakers=1000)
    x = torch.randn(2, 50, 112, 112)
    emb = model(x)
    print(emb.shape)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)
