import sys
sys.path.insert(0, "../")  # 上一级目录（/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/）

import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import os
import torch
import torchvision

from preparation.data.data_module import AVSRDataLoader

class AVSRDataLoader(torch.nn.Module):
    def __init__(self, detector="retinaface"):
        super().__init__()
        if detector == "mediapipe":
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
        elif detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector(device="cuda:0")
            self.video_process = VideoProcess(convert_gray=False)

    def forward(self, data_filename):
        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        return video

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
    
video_dataloader = AVSRDataLoader(detector="retinaface")


# data_filename = "/home/xueke/dataset/LRS3/trainval/wUKAVcj9NVA/50001.mp4"
# output_path = "/home/xueke/dataset/LRS3/mouth_mp4/wUKAVcj9NVA_50001.mp4"
# data_filename = "/home/xueke/dataset/LRS3/trainval/6XS8TA4RBog/50017.mp4"
# output_path = "/home/xueke/dataset/LRS3/mouth_mp4/6XS8TA4RBog_50017.mp4"

# data_filename = "/home/xueke/dataset/LRS3/trainval/1KS10q6N2A8/50001.mp4"
# output_path = "/home/xueke/dataset/LRS3/mouth_mp4/1KS10q6N2A8_50001.mp4"

# preprocessed_video = video_dataloader(data_filename)

# print("Processed ROI video shape:", preprocessed_video.shape)
import glob

def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)

# save2vid(output_path, preprocessed_video, frames_per_second=25)
# 视频根目录
input_root = "/home/xueke/dataset/Voxceleb2/vox2_test_mp4"
output_root = "/home/xueke/dataset/Voxceleb2/vox2_dev_mouth_cv"

# 查找所有 mp4 视频
# video_files = sorted(glob.glob(os.path.join(input_root, "*", "*.mp4")))
video_files = sorted(glob.glob(os.path.join(input_root, "**", "*.mp4"), recursive=True)) # ** ➜ 匹配任意层级的子文件夹
# recursive=True ➜ 告诉 glob 开启递归模式

# save2vid("./output.mp4", preprocessed_video, frames_per_second=25)
def get_video_duration(path):
    video, _, info = torchvision.io.read_video(path, pts_unit="sec")
    return video.shape[0] / info['video_fps']

# for data_filename in video_files:
#     try:
#         # 获取视频时长信息
#         duration = get_video_duration(data_filename)
#         if duration < 2.0:
#             print(f"Skipped (too short <2s): {data_filename}")
#             continue

#         # 用 video_dataloader 处理
#         preprocessed_video = video_dataloader(data_filename)

#         # 构建输出路径
#         parent = os.path.basename(os.path.dirname(data_filename)) 
#         filename_only = os.path.basename(data_filename)  # eg. 5535415699068794046_00002.mp4
#         combined = f"{parent}_{filename_only}"
#         # print(combined)
#         output_path = os.path.join(output_root, combined)

#         # 保存
#         save2vid(output_path, preprocessed_video, frames_per_second=25)
#         print(f"Saved: {output_path} (shape={preprocessed_video.shape})")

#     except Exception as e:
#         print(f"Error processing {data_filename}: {e}")



def process_video(data_filename):
    try:
        # video_dataloader = AVSRDataLoader(detector="retinaface")

        # print(data_filename)
        # /home/xueke/dataset/Voxceleb2/vox2_dew_mp4/dev/id00012/21Uxsk56VDQ/00003.mp4
        parent_dir = os.path.dirname(os.path.dirname(data_filename))
        id_folder = os.path.basename(parent_dir)
        parent = os.path.basename(os.path.dirname(data_filename))
        filename_only = os.path.basename(data_filename)
        combined = f"{id_folder}_{parent}_{filename_only}"
        output_path = os.path.join(output_root, combined)

        if os.path.exists(output_path):
            return f"✅ Skipped: {output_path}"

        duration = get_video_duration(data_filename)
        if duration < 2.0:
            return f"⚠️ Too short: {data_filename}"

        preprocessed_video = video_dataloader(data_filename)
        save2vid(output_path, preprocessed_video, frames_per_second=25)
        return f"✅ Saved: {output_path} (shape={preprocessed_video.shape})"

    except Exception as e:
        return f"❌ Error: {data_filename}: {e}"
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
def main():

    root = "/home/xueke/dataset/Voxceleb2/vox2_test_mp4"
    scp_file = "/home/xueke/CTCNet/Datasets/Vox2/mix_2_spk_cv.scp"
    resultss = []
    with open(scp_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            # 每个有效的音频路径在奇数索引上（0, 2, 4,...）
            for i in range(0, len(parts), 2):
                wav_path = parts[i]
                # print(wav_path)
                # basename: id04423_NGFWaLAxU-o_00267.wav
                base = os.path.basename(wav_path)
                name_no_ext = os.path.splitext(base)[0]  # id04423_NGFWaLAxU-o_00267
                # print(name_no_ext)

                id_part = name_no_ext[:7]
                end_part = name_no_ext[-5:]
                middle_part = name_no_ext[8:-6]             
                # if len(parts) != 3:
                #     raise ValueError(f"Unexpected format: {name_no_ext}")
                # 拼成 idXXXXX/XXXX/XXXXXX.mp4
                relative_path = os.path.join(id_part, middle_part, end_part + ".mp4")
                full_path = os.path.join(root, relative_path)
                # print(full_path)

                resultss.append(full_path)
    # video_files = sorted(glob.glob(os.path.join(input_root, "**", "*.mp4"), recursive=True))
    video_files = resultss  # 直接替换
    num_workers = min(4, cpu_count())  # 或者改成你想用的并行数

    # 建议添加 tqdm 可视化
    from tqdm import tqdm
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_video, video_files), total=len(video_files)))

    for r in results:
        print(r)

# 关键入口，防止 CUDA 初始化失败
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # 必须是 spawn
    main()















# import cv2

# cap = cv2.VideoCapture("./output.mp4")
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# duration = frame_count / fps

# print(f"帧率: {fps}")
# print(f"总帧数: {frame_count}")
# print(f"时长: {duration} 秒")
# cap.release()