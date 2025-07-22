import os
import torch
import torchaudio
import torchvision
from tqdm import tqdm

from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
from pytorch_lightning import LightningModule
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.batch_beam_search import BatchBeamSearch
from datamodule.transforms import TextTransform, VideoTransform
from espnet.nets.scorers.ctc import CTCPrefixScorer
from datamodule.transforms import TextTransform, VideoTransform

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

# === 准备转换器与模型 ===
transform = VideoTransform(subset="test")
text_transform = TextTransform()

model = ModelModule(args)
model_path = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/avsr_trlrwlrs2lrs3vox2avsp_base.pth"
ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
model.model.load_state_dict(ckpt)
model.eval().cuda()

# === 数据路径 ===
video_dir = "/home/xueke/LRS2/mvlrs_v1/mouth_mp4"
audio_root = "/home/xueke/LRS2/mvlrs_v1/raw_audio"
text_dir = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/lrs2_2s_txt"

# === 定义 WER 函数 ===
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

# === 主循环 ===
results = []
wer_scores = []

for txt_file in tqdm(sorted(os.listdir(text_dir))):
    if not txt_file.endswith(".txt"):
        continue

    utt_id = txt_file[:-4]
    text_path = os.path.join(text_dir, txt_file)
    video_path = os.path.join(video_dir, utt_id + ".mp4")

    # 查找音频路径
    audio_path = None
    for subset in ["train", "val", "test"]:
        temp_audio = os.path.join(audio_root, subset, utt_id + ".wav")
        if os.path.exists(temp_audio):
            audio_path = temp_audio
            break
    if not os.path.exists(video_path) or audio_path is None:
        continue

    # 加载文本
    with open(text_path, "r") as f:
        text = f.read().strip()

    # 视频处理
    video = torchvision.io.read_video(video_path, pts_unit="sec", output_format="THWC")[0]
    video = video.permute(0, 3, 1, 2)  # TCHW
    video_tensor = transform(video)[:50]  # 前两秒
    video_len = torch.tensor([video_tensor.shape[0]])

    # 音频处理
    waveform, sr = torchaudio.load(audio_path)
    waveform_cut = waveform[:, :32000]  # 前两秒
    waveform_full = waveform
    audio_len = torch.tensor([waveform_cut.shape[1]])

    # 文本编码
    label = text_transform.tokenize(text)

    with torch.inference_mode():
        # 推理
        _ = model.forward(
            video_tensor.unsqueeze(0).cuda(),
            waveform_cut.unsqueeze(-1).cuda(),
            video_len.cuda(),
            audio_len.cuda(),
            label.unsqueeze(0).cuda()
        )
        pred_text = model.forward_predicted(video_tensor.cuda(), waveform_full.unsqueeze(-1).cuda())

    # 计算 WER
    wer_score = wer(text.upper(), pred_text.upper())
    wer_scores.append(wer_score)

    results.append({
        "utt_id": utt_id,
        "ref": text,
        "hyp": pred_text,
        "wer": wer_score,
    })

    print(f"{utt_id}\nREF: {text}\nHYP: {pred_text}\nWER: {wer_score:.2%}")
    print("=" * 60)

# === 统计平均 WER ===
avg_wer = sum(wer_scores) / len(wer_scores)
print(f"\n✅ 总共 {len(results)} 个样本，平均 WER: {avg_wer:.2%}")

# === 保存结果 ===
output_path = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/auto_avsr_av/avsr_batch_results.txt"
with open(output_path, "w") as f:
    for item in results:
        f.write(f"{item['utt_id']}\n")
        f.write(f"REF: {item['ref']}\n")
        f.write(f"HYP: {item['hyp']}\n")
        f.write(f"WER: {item['wer']:.2%}\n")
        f.write("=" * 60 + "\n")
    f.write(f"\n✅ Total {len(results)} samples, Avg WER: {avg_wer:.2%}\n")

print(f"\n📁 推理结果已保存到: {output_path}")


