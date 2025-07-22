import sys
sys.path.insert(0, "../")  # 将上一级目录加入 Python 模块路径，确保可以 import 工程外部的模块或 espnet 项目代码。


import os
import torch
import torchaudio

import argparse
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args(args=[])

from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
from pytorch_lightning import LightningModule
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.batch_beam_search import BatchBeamSearch
from datamodule.transforms import TextTransform, VideoTransform
from espnet.nets.scorers.ctc import CTCPrefixScorer

from argparse import Namespace
config_dict = {
    "adim": 768,
    "aheads": 12,
    "eunits": 3072,
    "elayers": 12,
    "transformer_input_layer": "conv3d",
    "dropout_rate": 0.1,
    "transformer_attn_dropout_rate": 0.1,
    "transformer_encoder_attn_layer_type": "rel_mha",
    "macaron_style": True,
    "use_cnn_module": True,
    "cnn_module_kernel": 31,
    "zero_triu": False,
    "a_upsample_ratio": 1,
    "relu_type": "swish",
    "ddim": 768,
    "dheads": 12,
    "dunits": 3072,
    "dlayers": 6,
    "lsm_weight": 0.1,
    "transformer_length_normalized_loss": False,
    "mtlalpha": 0.1,
    "ctc_type": "builtin",
    "rel_pos_type": "latest",

    "aux_adim": 768,
    "aux_aheads": 12,
    "aux_eunits": 3072,
    "aux_elayers": 12,
    "aux_transformer_input_layer": "conv1d",
    "aux_dropout_rate": 0.1,
    "aux_transformer_attn_dropout_rate": 0.1,
    "aux_transformer_encoder_attn_layer_type": "rel_mha",
    "aux_macaron_style": True,
    "aux_use_cnn_module": True,
    "aux_cnn_module_kernel": 31,
    "aux_zero_triu": False,
    "aux_a_upsample_ratio": 1,
    "aux_relu_type": "swish",
    "aux_dunits": 3072,
    "aux_dlayers": 6,
    "aux_lsm_weight": 0.1,
    "aux_transformer_length_normalized_loss": False,
    "aux_mtlalpha": 0.1,
    "aux_ctc_type": "builtin",
    "aux_rel_pos_type": "latest",

    "fusion_hdim": 8192,
    "fusion_norm": "batchnorm",
}
args = Namespace(**config_dict)

def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )

class ModelModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.adim = args.adim
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        self.model = E2E(len(self.token_list), args, ignore_id=-1)

    def forward(self, video, audio, video_lengths, audio_lengths, label):
        _, _, _, x = self.model(video, audio, video_lengths, audio_lengths, label)
        return x
    def forward_predicted(self, video, audio):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        video_feat, _ = self.model.encoder(video.unsqueeze(0).to(self.device), None)
        video_feat = video_feat[:, :50] # 先全局建模然后提取前50帧的特征
        audio_feat, _ = self.model.aux_encoder(audio.to(self.device), None)
        audio_feat = audio_feat[:, :50]
        audiovisual_feat = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))

        audiovisual_feat = audiovisual_feat.squeeze(0)
        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted


model_path = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/avsr_trlrwlrs2lrs3vox2avsp_base.pth"
# setattr(args, 'modality', 'audio')  # 设置 modality 为 "video"，

model = ModelModule(args)

ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
model.model.load_state_dict(ckpt)
model.freeze()

import torchvision
vid = torchvision.io.read_video("/home/xueke/LRS2/mvlrs_v1/mouth_mp4/5535415699068794046_00004.mp4", pts_unit="sec", output_format="THWC")[0]
vid = vid.permute((0, 3, 1, 2))  # torch.Size([65, 1, 88, 88])
transform = VideoTransform(subset="test")

def wer(reference: str, hypothesis: str) -> float:
    r = reference.strip().split()
    h = hypothesis.strip().split()
    n = len(r)
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): dp[i][0] = i
    for j in range(len(h) + 1): dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + 1,  # substitute
                               dp[i][j - 1] + 1,      # insert
                               dp[i - 1][j] + 1)      # delete
    return dp[len(r)][len(h)] / n if n > 0 else float('inf')


processed_video = transform(vid)

processed_video_50 = processed_video[:50]

length_video = processed_video_50.shape[0]  # 2s
length_video = torch.tensor([length_video])
print(processed_video.shape)
# 应该输出: torch.Size([T, 1, 88, 88])
# 因为灰度化后 C=1，随机裁剪后 H=W=88
# ==== 构造 label ====
text_transform = TextTransform()
example_text = "WHICH INVOLVES FIRING A POTATO DOWN A PIPE" 
example_text_2s = "WHICH INVOLVES FIRING A POTATO DOWN"
label = text_transform.tokenize(example_text_2s)  # -> 1D tensor
print("label:", label.shape)
# ==== 音频长度读取 ====
audio_path = "/home/xueke/LRS2/mvlrs_v1/raw_audio/train/5535415699068794046_00004.wav"
waveform, sr = torchaudio.load(audio_path)  # waveform shape: [1, num_samples]
waveforms = waveform
waveform  = waveform[:, :32000]

print(waveform.shape)  # torch.Size([1, 41984])
length_in_samples = waveform.shape[1]
audio_length = torch.tensor([length_in_samples])
print(length_in_samples)  # -> batch_size = 1
with torch.inference_mode():
    y = model.forward(processed_video_50.unsqueeze(0), waveform.unsqueeze(-1), length_video, audio_length, label.unsqueeze(0))
    y_words = model.forward_predicted(processed_video, waveforms.unsqueeze(-1))  # wavform的输入希望是 B, T, C
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