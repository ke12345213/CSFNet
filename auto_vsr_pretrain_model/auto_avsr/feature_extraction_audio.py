import sys
sys.path.insert(0, "../")  # 将上一级目录加入 Python 模块路径，确保可以 import 工程外部的模块或 espnet 项目代码。


import os
import torch
import torchaudio

import argparse
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args(args=[])

from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from pytorch_lightning import LightningModule
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.batch_beam_search import BatchBeamSearch
from datamodule.transforms import TextTransform, VideoTransform

def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )


class ModelModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.modality = args.modality
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        self.model = E2E(len(self.token_list), self.modality, ctc_weight=getattr(args, "ctc_weight", 0.1))

    def forward(self, x, lengths, label):
        _, _, _, x = self.model(x.unsqueeze(-1), lengths, label) # 返回的是预测后的token——id的准确率，并不是对应的词错误率
        return x
    def forward_predicted(self, x):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        x = self.model.frontend(x.unsqueeze(-1))
        x = self.model.proj_encoder(x)
        enc_feat, _ = self.model.encoder(x, None)
        enc_feat = enc_feat[:, :50]
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

model_path = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/asr_trlrs3vox2_base.pth"
setattr(args, 'modality', 'audio')  # 设置 modality 为 "video"，

model = ModelModule(args)

ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
model.model.load_state_dict(ckpt)
model.freeze()

# x = torch.randn((50, 1, 88, 88))
# import torchvision
# vid = torchvision.io.read_video("/home/xueke/LRS2/mvlrs_v1/mouth_mp4/5535415699068794046_00004.mp4", pts_unit="sec", output_format="THWC")[0]
# vid = vid.permute((0, 3, 1, 2))
# transform = VideoTransform(subset="test")
# processed_video = transform(vid)
# processed_video = processed_video[:50]  # 2s
# print(processed_video.shape)
# 应该输出: torch.Size([T, 1, 88, 88])
# 因为灰度化后 C=1，随机裁剪后 H=W=88
# ==== 构造 label ====
text_transform = TextTransform()
example_text = "THE TRADITIONAL CHIP PAN OFTEN STAYS ON THE SHELF" 
example_text_2s = "THE TRADITIONAL CHIP PAN OFTEN STAYS ON THE"
label = text_transform.tokenize(example_text_2s)  # -> 1D tensor
print("label:", label.shape)
# ==== 音频长度读取 ====
audio_path = "/home/xueke/LRS2/mvlrs_v1/raw_audio/train/5535415699068794046_00002.wav"
waveform, sr = torchaudio.load(audio_path)  # waveform shape: [1, num_samples]
# waveform  = waveform[:, :32000]
print(waveform.shape)  # torch.Size([1, 41984])
length_in_samples = waveform.shape[1]
length = torch.tensor([length_in_samples])
print(length_in_samples)  # -> batch_size = 1
with torch.inference_mode():
    y = model.forward(waveform, length, label.unsqueeze(0))
    y_words = model.forward_predicted(waveform)
print("模型参数总数:", sum(p.numel() for p in model.parameters()))
print(y)
print(y_words)
#W WER 计算公式：wer = s+d+i/n
# 其中：S = 替换（substitutions）D = 删除（deletions）I = 插入（insertions）N = 参考句子的词数（reference word count）
# import numpy as np
# np.save("features.npy", y.cpu().numpy())
# y = torch.tensor(np.load("features.npy"))
# print(y.shape)



## 222222222222222222222222222222222222222222222
# import os
# import glob
# import numpy as np
# from tqdm import tqdm
# from torch.multiprocessing import Process, set_start_method

# input_dir = "/home/xueke/LRS2/mvlrs_v1/mouth_mp4"
# output_dir = "/home/xueke/LRS2/mvlrs_v1/feature"
# os.makedirs(output_dir, exist_ok=True)
# video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))

# def split_list(lst, n):
#     k, m = divmod(len(lst), n)
#     return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

# def process_files(file_list, gpu_id, model_path):
#     import torch
#     import argparse
#     import torchvision
#     from datamodule.transforms import VideoTransform
#     from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
#     from pytorch_lightning import LightningModule

#     class ModelModule(LightningModule):
#         def __init__(self, args):
#             super().__init__()
#             self.args = args
#             self.save_hyperparameters(args)
#             self.text_transform = TextTransform()
#             self.token_list = self.text_transform.token_list
#             self.model = E2E(len(self.token_list), args.modality, ctc_weight=getattr(args, "ctc_weight", 0.1))

#         def forward(self, x):
#             x = self.model.frontend(x.unsqueeze(0))
#             x = self.model.proj_encoder(x)
#             x, _ = self.model.encoder(x, None)
#             return x.squeeze(0)

#     torch.cuda.set_device(gpu_id)
#     parser = argparse.ArgumentParser()
#     args, _ = parser.parse_known_args(args=[])
#     setattr(args, 'modality', 'video')

#     model = ModelModule(args)
#     ckpt = torch.load(model_path, map_location=f"cuda:{gpu_id}")
#     model.model.load_state_dict(ckpt)
#     model.cuda().eval()

#     transform = VideoTransform(subset="test")

#     for video_path in tqdm(file_list, desc=f"GPU{gpu_id}"):
#         try:
#             filename_only = os.path.basename(video_path).replace(".mp4", ".npy")
#             save_path = os.path.join(output_dir, filename_only)
#             if os.path.exists(save_path):
#                 print(f"[GPU{gpu_id}] Skipped: {save_path}")
#                 continue

#             vid = torchvision.io.read_video(video_path, pts_unit="sec", output_format="THWC")[0]
#             if vid.size(0) == 0:
#                 print(f"[GPU{gpu_id}] Empty video: {video_path}")
#                 continue
#             vid = vid.permute((0, 3, 1, 2))  # [T, C, H, W]
#             vid = transform(vid)  # [T, 1, 88, 88]

#             with torch.inference_mode():
#                 y = model(vid.cuda())

#             np.save(save_path, y.cpu().numpy())
#             print(f"[GPU{gpu_id}] Saved: {save_path}, shape={y.shape}")
#         except Exception as e:
#             print(f"[GPU{gpu_id}] Error processing {video_path}: {e}")

# def main():
#     try:
#         set_start_method('spawn')
#     except RuntimeError:
#         pass

#     model_path = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/vsr_trlrs2lrs3vox2avsp_base.pth"
#     num_gpus = 4
#     file_chunks = split_list(video_files, num_gpus)
#     processes = []

#     for gpu_id in range(num_gpus):
#         p = Process(target=process_files, args=(file_chunks[gpu_id], gpu_id, model_path))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

# if __name__ == "__main__":
#     main()