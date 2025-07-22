# import sys
# sys.path.insert(0, "../")  # 将上一级目录加入 Python 模块路径，确保可以 import 工程外部的模块或 espnet 项目代码。


# import os
# import torch
# import torchaudio
# import os
# from pathlib import Path
# import torch
# import torchvision
# from tqdm import tqdm
# import torch.multiprocessing as mp
# from torch.nn import DataParallel
# from typing import List
 
# import argparse
# parser = argparse.ArgumentParser()
# args, _ = parser.parse_known_args(args=[])

# from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
# from pytorch_lightning import LightningModule
# from espnet.nets.scorers.length_bonus import LengthBonus
# from espnet.nets.batch_beam_search import BatchBeamSearch
# from datamodule.transforms import TextTransform, VideoTransform

# def get_beam_search_decoder(
#     model,
#     token_list,
#     rnnlm=None,
#     rnnlm_conf=None,
#     penalty=0,
#     ctc_weight=0.1,
#     lm_weight=0.0,
#     beam_size=40,
# ):
#     sos = model.odim - 1
#     eos = model.odim - 1
#     scorers = model.scorers()

#     scorers["lm"] = None
#     scorers["length_bonus"] = LengthBonus(len(token_list))
#     weights = {
#         "decoder": 1.0 - ctc_weight,
#         "ctc": ctc_weight,
#         "lm": lm_weight,
#         "length_bonus": penalty,
#     }

#     return BatchBeamSearch(
#         beam_size=beam_size,
#         vocab_size=len(token_list),
#         weights=weights,
#         scorers=scorers,
#         sos=sos,
#         eos=eos,
#         token_list=token_list,
#         pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
#     )


# class ModelModule(LightningModule):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.save_hyperparameters(args)

#         self.modality = args.modality
#         self.text_transform = TextTransform()
#         self.token_list = self.text_transform.token_list

#         self.model = E2E(len(self.token_list), self.modality, ctc_weight=getattr(args, "ctc_weight", 0.1))

#     def forward(self, x, lengths, label):
#         _, _, _, x = self.model(x.unsqueeze(0), lengths, label) # 返回的是预测后的token——id的准确率，并不是对应的词错误率
#         return x
#     def forward_predicted(self, x):
#         self.beam_search = get_beam_search_decoder(self.model, self.token_list)
#         x = self.model.frontend(x.unsqueeze(0))
#         x = self.model.proj_encoder(x)
#         enc_feat, _ = self.model.encoder(x, None)
#         # print(enc_feat.shape)  # torch.Size([1, 63, 768])
#         enc_feat = enc_feat[:, :50]

#         enc_feat = enc_feat.squeeze(0)
#         nbest_hyps = self.beam_search(enc_feat)
#         nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
#         predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
#         predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
#         return predicted

# # model_path = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/vsr_trlrs2lrs3vox2avsp_base.pth"
# # setattr(args, 'modality', 'video')  # 设置 modality 为 "video"，

# # model = ModelModule(args)

# # ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
# # model.model.load_state_dict(ckpt)
# # model.freeze()

# # x = torch.randn((50, 1, 88, 88))
# import torchvision
# # vid = torchvision.io.read_video("/home/xueke/LRS2/mvlrs_v1/mouth_mp4/5535415699068794046_00004.mp4", pts_unit="sec", output_format="THWC")[0]
# # vid = vid.permute((0, 3, 1, 2))
# # transform = VideoTransform(subset="test")
# # processed_video = transform(vid)
# # processed_video = processed_video[:50]  # 2s
# # print(processed_video.shape)
# # # 应该输出: torch.Size([T, 1, 88, 88])
# # # 因为灰度化后 C=1，随机裁剪后 H=W=88
# # # ==== 构造 label ====
# # text_transform = TextTransform()
# # example_text = "WHICH INVOLVES FIRING A POTATO DOWN A PIPE" 
# # example_text_2s = "WHICH INVOLVES FIRING A POTATO DOWN"
# # label = text_transform.tokenize(example_text)  # -> 1D tensor
# # print("label:", label.shape)
# # lengths = torch.tensor([50])  # -> batch_size = 1
# # with torch.inference_mode():
# #     y = model.forward(processed_video, lengths, label.unsqueeze(0))
# #     y_words = model.forward_predicted(processed_video)
# # print("模型参数总数:", sum(p.numel() for p in model.parameters()))
# # print(y)
# # print(y_words)
# # #W WER 计算公式：wer = s+d+i/n
# # # 其中：S = 替换（substitutions）D = 删除（deletions）I = 插入（insertions）N = 参考句子的词数（reference word count）
# # def wer(reference: str, hypothesis: str) -> float:
# #     """
# #     计算 Word Error Rate（WER）

# #     :param reference: 参考文本（ground truth）
# #     :param hypothesis: 预测文本（识别输出）
# #     :return: WER 值，0~1 之间
# #     """
# #     r = reference.strip().split()
# #     h = hypothesis.strip().split()
# #     n = len(r)

# #     # 初始化编辑距离矩阵
# #     dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
# #     for i in range(len(r) + 1):
# #         dp[i][0] = i
# #     for j in range(len(h) + 1):
# #         dp[0][j] = j

# #     # 填充矩阵
# #     for i in range(1, len(r) + 1):
# #         for j in range(1, len(h) + 1):
# #             if r[i - 1] == h[j - 1]:
# #                 dp[i][j] = dp[i - 1][j - 1]
# #             else:
# #                 substitute = dp[i - 1][j - 1] + 1
# #                 insert = dp[i][j - 1] + 1
# #                 delete = dp[i - 1][j] + 1
# #                 dp[i][j] = min(substitute, insert, delete)

# #     return dp[len(r)][len(h)] / n if n > 0 else float('inf')

# # # ref = "I have a dream"
# # # hyp = "I had a cream"
# # print(f"WER: {wer(example_text_2s,y_words):.2%}")

# text_transform = TextTransform()
# transform = VideoTransform(subset="test")
# # wer_list = []

# def wer(reference: str, hypothesis: str) -> float:
#     r = reference.strip().split()
#     h = hypothesis.strip().split()
#     n = len(r)
#     dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)] 
#     for i in range(len(r) + 1): dp[i][0] = i
#     for j in range(len(h) + 1): dp[0][j] = j
#     for i in range(1, len(r) + 1):
#         for j in range(1, len(h) + 1):
#             if r[i - 1] == h[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1]
#             else:
#                 dp[i][j] = min(dp[i - 1][j - 1] + 1,  # substitute
#                                dp[i][j - 1] + 1,      # insert
#                                dp[i - 1][j] + 1)      # delete
#     return dp[len(r)][len(h)] / n if n > 0 else float('inf')


# def process_subset(rank: int, world_size: int, all_files: List[str], video_dir, text_dir, device):
#     torch.cuda.set_device(device)
#     local_files = all_files[rank::world_size]
#     local_wers = []

#     model = load_model(device)
#     model.eval()

#     for mp4_path in tqdm(local_files, desc=f"[GPU {rank}]"):
#         name = Path(mp4_path).stem
#         txt_path = text_dir / f"{name}.txt"
#         if not txt_path.exists():
#             continue

#         with open(txt_path, "r") as f:
#             text = f.read().strip()
#         if not text:
#             continue

#         try:
#             vid = torchvision.io.read_video(str(mp4_path), pts_unit="sec", output_format="THWC")[0]
#             vid = vid.permute((0, 3, 1, 2))  # TCHW
#             processed_video = transform(vid)[:50].to(device)
#             if processed_video.shape[0] < 10:
#                 continue
#         except Exception as e:
#             print(f"读取失败: {mp4_path}: {e}")
#             continue

#         lengths = torch.tensor([processed_video.shape[0]]).to(device)

#         with torch.inference_mode():
#             y_pred_text = model.forward_predicted(processed_video)
#             if isinstance(y_pred_text, (list, tuple)):
#                 y_pred_text = " ".join(y_pred_text)
#             elif isinstance(y_pred_text, torch.Tensor):
#                 y_pred_text = y_pred_text.item() if y_pred_text.dim() == 0 else ""
#             y_pred_text = str(y_pred_text).strip()

#         wer_score = wer(text, y_pred_text)
#         print(wer_score)
#         local_wers.append(wer_score)

#     return local_wers


# def main_worker(rank, world_size, all_files, video_dir, text_dir, return_dict):
#     device = f"cuda:{rank}"
#     local_wers = process_subset(rank, world_size, all_files, video_dir, text_dir, device)
#     return_dict[rank] = local_wers


# def run_multi_gpu():
#     video_dir = Path("/home/xueke/LRS2/mvlrs_v1/mouth_mp4")
#     text_dir = Path("/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/lrs2_2s_txt")
#     all_files = sorted([str(f) for f in video_dir.glob("*.mp4")])

#     world_size = 4  # 4 GPUs
#     mp.set_start_method('spawn', force=True)
#     manager = mp.Manager()
#     return_dict = manager.dict()

#     processes = []
#     for rank in range(world_size):
#         p = mp.Process(target=main_worker, args=(rank, world_size, all_files, video_dir, text_dir, return_dict))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     # 汇总所有 GPU 的结果
#     all_wers = []
#     for k in return_dict.keys():
#         all_wers.extend(return_dict[k])

#     print(f"\n✅ 共处理 {len(all_wers)} 个视频，平均 WER: {sum(all_wers) / len(all_wers):.2%}")

# def load_model(device):
#     setattr(args, 'modality', 'video')
#     model = ModelModule(args).to(device)
#     ckpt = torch.load("/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/vsr_trlrs2lrs3vox2avsp_base.pth", map_location="cpu")
#     model.model.load_state_dict(ckpt)
#     model.freeze()
#     return model


# if __name__ == "__main__":
#     run_multi_gpu()







# 222222222222222222222222222222222222222222222

import os
import glob
import numpy as np
from tqdm import tqdm
from torch.multiprocessing import Process, set_start_method
import sys
sys.path.append("/home/xueke/DPT_1d_main")

input_dir = "/home/xueke/dataset/Voxceleb2/vox2_dev_mouth_cv"
output_dir = "/home/xueke/dataset/Voxceleb2/feature_valid"
os.makedirs(output_dir, exist_ok=True)
video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def process_files(file_list, gpu_id, model_path):
    import torch
    import argparse
    import torchvision
    from datamodule.transforms import VideoTransform, TextTransform
    from auto_vsr_pretrain_model.auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
    from pytorch_lightning import LightningModule

    class ModelModule(LightningModule):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self.save_hyperparameters(args)
            self.text_transform = TextTransform()
            self.token_list = self.text_transform.token_list
            self.model = E2E(len(self.token_list), args.modality, ctc_weight=getattr(args, "ctc_weight", 0.1))

        def forward(self, x):
            x = self.model.frontend(x.unsqueeze(0))
            x = self.model.proj_encoder(x)
            x, _ = self.model.encoder(x, None)
            return x.squeeze(0)

    torch.cuda.set_device(gpu_id)
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args(args=[])
    setattr(args, 'modality', 'video')

    model = ModelModule(args)
    ckpt = torch.load(model_path, map_location=f"cuda:{gpu_id}")
    model.model.load_state_dict(ckpt)
    model.cuda().eval()

    transform = VideoTransform(subset="test")

    for video_path in tqdm(file_list, desc=f"GPU{gpu_id}"):
        try:
            filename_only = os.path.basename(video_path).replace(".mp4", ".npy")
            save_path = os.path.join(output_dir, filename_only)
            if os.path.exists(save_path):
                print(f"[GPU{gpu_id}] Skipped: {save_path}")
                continue

            vid = torchvision.io.read_video(video_path, pts_unit="sec", output_format="THWC")[0]
            if vid.size(0) == 0:
                print(f"[GPU{gpu_id}] Empty video: {video_path}")
                continue
            vid = vid.permute((0, 3, 1, 2))  # [T, C, H, W]
            vid = transform(vid)  # [T, 1, 88, 88]
            vid = vid[:50]

            with torch.inference_mode():
                y = model(vid.cuda())

            np.save(save_path, y.cpu().numpy())
            print(f"[GPU{gpu_id}] Saved: {save_path}, shape={y.shape}")
        except Exception as e:
            print(f"[GPU{gpu_id}] Error processing {video_path}: {e}")

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    model_path = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/vsr_trlrs2lrs3vox2avsp_base.pth"
    num_gpus = 4
    file_chunks = split_list(video_files, num_gpus)
    processes = []

    for gpu_id in range(num_gpus):
        p = Process(target=process_files, args=(file_chunks[gpu_id], gpu_id, model_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()