import os
import torch
from glob import glob
import sentencepiece as spm

# 1111111111111111111111111
SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000.model",
)

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000_units.txt",
)

# === 你提供的 TextTransform 类 ===
class TextTransform:
    def __init__(self, sp_model_path=SP_MODEL_PATH, dict_path=DICT_PATH):
        self.spm = spm.SentencePieceProcessor(model_file=sp_model_path)
        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")

# === 路径配置 ===
input_txt_root = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/lrs2_2s_txt"
output_txt_root = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/lrs2_2s_tokenized"  # 输出保存路径


os.makedirs(output_txt_root, exist_ok=True)
text_transform = TextTransform()

# === 批量处理所有 .txt ===
txt_files = glob(os.path.join(input_txt_root, "*.txt"), recursive=True)
print(f"共找到 {len(txt_files)} 个文本文件，开始处理...")

for txt_path in txt_files:
    with open(txt_path, "r") as f:
        content = f.read().strip()

    if not content:
        continue  # 跳过空文本

    token_ids = text_transform.tokenize(content)
    token_id_str = " ".join(map(str, token_ids.tolist()))

    # # 构造输出文件名：5535415699068794046_00001.txt
    # rel_path = os.path.relpath(txt_path, input_txt_root)
    # parts = rel_path.split(os.sep)
    # if len(parts) < 2:
    #     continue
    # output_filename = f"{parts[-2]}_{parts[-1]}"
    output_filename = os.path.basename(txt_path)
    output_path = os.path.join(output_txt_root, output_filename)

    with open(output_path, "w") as out_f:
        out_f.write(token_id_str + "\n")

print("✅ 所有文件处理完毕！")



# 22222222222222222222222222222222
# import os
# from textgrid import TextGrid

# def extract_text_before_2s_and_save(textgrid_path, output_dir):
#     tg = TextGrid.fromFile(textgrid_path)  # 读取 .TextGrid 文件并解析成一个 TextGrid 对象 tg。
#     word_tier = tg.getFirst("words")  # 获取名为 "words" 的层（tier），即每个对齐好的单词所在的时间段。
#     words_in_2s = []
    
#     for interval in word_tier.intervals:
#         if interval.maxTime <= 2.0:
#             if interval.mark.strip():
#                 words_in_2s.append(interval.mark.strip())
#         elif interval.minTime < 2.0:
#             if interval.mark.strip():
#                 words_in_2s.append(interval.mark.strip())
#             break

#     sentence = " ".join(words_in_2s).upper()  # 把所有保留下来的单词连接成一句完整的句子（用空格分隔）。

#     # 构建输出文件路径
#     basename = os.path.splitext(os.path.basename(textgrid_path))[0]
#     output_path = os.path.join(output_dir, f"{basename}.txt")

#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(sentence + "\n")

#     print(f"Saved: {output_path}")

# # extract_text_before_2s_and_save("/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/alignment_output/5535415699068794046_00002.TextGrid", "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/alignment_output/")
# def process_all_textgrids(input_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     for file in os.listdir(input_dir):
#         if file.endswith(".TextGrid"):
#             textgrid_path = os.path.join(input_dir, file)
#             extract_text_before_2s_and_save(textgrid_path, output_dir)

# # 设置路径
# input_dir = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/alignment_output"
# output_dir = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/lrs2_2s_txt"

# # 执行处理
# process_all_textgrids(input_dir, output_dir)


# 33333333333333333333333333333333333333333333333333
# import os

# src_root = "/home/xueke/LRS2/mvlrs_v1/main"
# dst_root = "/home/xueke/DPT_1d_main/file_txt"

# os.makedirs(dst_root, exist_ok=True)

# for speaker_id in os.listdir(src_root):
#     speaker_dir = os.path.join(src_root, speaker_id)
#     if not os.path.isdir(speaker_dir):
#         continue

#     for fname in os.listdir(speaker_dir):
#         if fname.endswith(".txt"):
#             txt_path = os.path.join(speaker_dir, fname)
#             with open(txt_path, "r", encoding="utf-8") as f:
#                 lines = f.readlines()

#             if not lines or not lines[0].startswith("Text:"):
#                 print(f"跳过无效文件: {txt_path}")
#                 continue

#             text_line = lines[0].strip().replace("Text:", "").strip()
#             output_name = f"{speaker_id}_{fname}"
#             output_path = os.path.join(dst_root, output_name)

#             with open(output_path, "w", encoding="utf-8") as out_f:
#                 out_f.write(text_line + "\n")

#             print(f"保存: {output_path}")
