import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import cv2
import shutil
import zipfile
import json
from typing import Dict, Iterable, List, Iterator
import numbers
import inspect
from einops import rearrange
from torchsummary import summary
from torch.autograd import Variable
import glob
from PIL import Image
from torchvision import transforms
import torchvision
from .transform import get_preprocessing_pipelines

def convert_bgr2gray(data):
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)
def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)
    
class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned
    
class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BaseEncoder(torch.nn.Module):
    def unsqueeze_to_3D(self, x: torch.Tensor):
        if x.ndim == 1:
            return x.reshape(1, 1, -1)
        elif x.ndim == 2:
            return x.unsqueeze(1)
        else:
            return x

    def unsqueeze_to_2D(self, x: torch.Tensor):
        if x.ndim == 1:
            return x.reshape(1, -1)
        elif len(s := x.shape) == 3:
            assert s[1] == 1
            return x.reshape(s[0], -1)
        else:
            return x

    def pad(self, x: torch.Tensor, lcm: int):
        values_to_pad = int(x.shape[-1]) % lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padding = torch.zeros(
                list(appropriate_shape[:-1]) + [lcm - values_to_pad],
                dtype=x.dtype,
                device=x.device,
            )
            padded_x = torch.cat([x, padding], dim=-1)
            return padded_x
        else:
            return x

    def get_out_chan(self):
        return self.out_chan

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args

class STFTEncoder(BaseEncoder):
    def __init__(
        self,
        win: int,
        hop_length: int,
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(STFTEncoder, self).__init__()

        self.win = win
        self.hop_length = hop_length
        self.bias = bias

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x: torch.Tensor):
        x = self.unsqueeze_to_2D(x)

        spec = torch.stft(
            x,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True,
            # center=False,
        )

        spec = spec.transpose(1, 2).contiguous()
        # spec = torch.stack([spec.real, spec.imag], 1).transpose(2, 3).contiguous()  # B, 2, T, F
        return spec

class AVSpeechDataset1(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        is_train: bool = True
    ):
        super().__init__()
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            self.fps_len = int(segment * 25)
        self.n_src = n_src
        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []
        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        self.mix.append(mix_infos[i])
                        self.sources.append([src_inf[i] for src_inf in sources_infos])

            else:
                for i in range(len(mix_infos)):
                    self.mix.append(mix_infos[i])
                    self.sources.append([sources_infos[0][i], sources_infos[1][i]])
            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        if self.n_src == 1:
            rand_start = 0
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len

            mix_source, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            source = sf.read(
                self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
            )[0]
            source_mouth = self.lipreading_preprocessing_func(
                convert_bgr2gray(np.load(self.sources[idx][1])["data"])
            )[: self.fps_len]

            source = torch.from_numpy(source)
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)
            return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1]

        if self.n_src == 2:
            rand_start = 0
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len

            mix_source, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            sources = []
            for src in self.sources[idx]:
                # import pdb; pdb.set_trace()
                sources.append(
                    sf.read(src[0], start=rand_start, stop=stop, dtype="float32")[0]
                )
            # import pdb; pdb.set_trace()
            sources_mouths = torch.stack(
                [
                    torch.from_numpy(
                        self.lipreading_preprocessing_func(convert_bgr2gray(np.load(src[1])["data"]))
                    )
                    for src in self.sources[idx]
                ]
            )[: self.fps_len]
            # import pdb; pdb.set_trace()
            sources = torch.stack([torch.from_numpy(source) for source in sources])
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            return mixture, sources, sources_mouths, self.mix[idx][0].split("/")[-1]


class AVSpeechDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 1,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        is_train: bool = True
    ):
        super().__init__()
        self.stft_encoder = STFTEncoder(win=256, hop_length=128, bias=False)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),  # 自动将 [H, W] 转为 [1, H, W]，像素归一化到 [0, 1]
            transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图只用一个通道
        ])
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        self.i = []
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            self.fps_len = int(segment * 25)
        self.n_src = n_src
        self.test = self.seg_len is None  # 如果segment为None，test是true
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []
        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            print(len(mix_infos), len(sources_infos[0]), len(sources_infos[1]))
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len
        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        self.mix.append(mix_infos[i])
                        self.sources.append([src_inf[i] for src_inf in sources_infos])

            else:
                for i in range(len(mix_infos)):
                    self.mix.append(mix_infos[i])
                    self.sources.append([sources_infos[0][i], sources_infos[1][i]])
            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

    def __len__(self):
        return self.length
    
    # def _pad_tensor(self, tensor, target_shape):
    #         tensor = torch.tensor(tensor, dtype=torch.float32)
    #         pad_size = (target_shape[0] - tensor.shape[0],)
    #         padding_tensor = torch.zeros((pad_size[0],) + tensor.shape[1:], dtype=tensor.dtype)
    #         padded_tensor = torch.cat((tensor, padding_tensor), dim=0)
    #         padded_array = padded_tensor.numpy()
            
    #         return padded_array
    def _pad_tensor(self, tensor, target_shape):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        print(type(tensor))
        tensor = tensor.float()  # 保证类型一致
        # print(type(tensor))
        pad_len = target_shape[0] - tensor.shape[0]
        if pad_len > 0:
            padding_tensor = torch.zeros((pad_len,) + tensor.shape[1:], dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding_tensor), dim=0)
            # print(type(tensor))

        return tensor  # ✅ 返回的是 Tensor，不转 numpy

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        if self.n_src == 1:
            # rand_start = 0  # # # 数据集没有按照min生成所以可能会有大量的空的，所以还是从0开始比较好
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                # rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len - 640*5)
                max_frame = (self.mix[idx][1] - self.seg_len - 640 * 5) // 640
                if max_frame > 0:
                    rand_start = np.random.randint(0, max_frame) * 640
                else:
                    rand_start = 0
            if self.test:
                # stop = None
                stop = rand_start + 32000
            else:
                stop = rand_start + self.seg_len

            mix_source, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            # print(self.mix[idx][0])
            source = sf.read(
                self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
            )[0]

            # source_mouth = self.lipreading_preprocessing_func(
            #     np.load(self.sources[idx][1])["data"]
            # )[: 50]  # [start_frame:stop_frame]

            source_mouth = self.lipreading_preprocessing_func(
                convert_bgr2gray(np.load(self.sources[idx][1])["data"])
            )[: 50]

            source = torch.from_numpy(source)
            mixture = torch.from_numpy(mix_source)
            # source_mouth = torch.from_numpy(source_mouth)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)
            # return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1]
            return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1]


        if self.n_src == 2:
            rand_start = 0
            if self.test:
                stop = 32000
            else:
                stop = rand_start + self.seg_len

            mix_source, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            sources = []
            for src in self.sources[idx]:
                # import pdb; pdb.set_trace()
                sources.append(
                    sf.read(src[0], start=rand_start, stop=stop, dtype="float32")[0]
                )
            # # import pdb; pdb.set_trace()
            # sources_mouths = []
            # for src in self.sources[idx]:
            #     mouth_tensor = torch.from_numpy(
            #         self.lipreading_preprocessing_func(
            #             convert_bgr2gray(np.load(src[1])["data"])
            #         )
            #     )[:50]

            #     if mouth_tensor.shape[0] < 50:
            #         mouth_tensor = self._pad_tensor(mouth_tensor, (50, mouth_tensor.shape[1], mouth_tensor.shape[2]))
            #     else: 
            #         mouth_tensor = torch.tensor(mouth_tensor, dtype=torch.float32)

            #     sources_mouths.append(mouth_tensor)
            # sources_mouths = torch.stack(sources_mouths)
            sources_mouths = []
            for src in self.sources[idx]:
                mouth_tensor = torch.tensor(np.load(src[1]))  # [: 50]
                sources_mouths.append(mouth_tensor)
            sources_mouths = torch.stack(sources_mouths)

            text_dir = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/lrs2_2s_txt"
            filenames = []
            for src in self.sources[idx]:
                filename = os.path.splitext(os.path.basename(src[1]))[0]
                filenames.append(filename)
            filename1 = filenames[0]
            filename2 = filenames[1]
            frame1_path_text = os.path.join(text_dir, f"{filename1}.txt")   # batchsize只能是1
            with open(frame1_path_text, "r") as f:
                text1 = f.read().strip()
            frame2_path_text = os.path.join(text_dir, f"{filename2}.txt")   # batchsize只能是1
            with open(frame2_path_text, "r") as f:
                text2 = f.read().strip()

            video_path1 = os.path.join("/home/xueke/LRS2/mvlrs_v1/mouth_mp4/", f"{filename1}.mp4")
            vid1 = torchvision.io.read_video(video_path1, pts_unit="sec", output_format="THWC")[0]
            vid1 = vid1.permute((0, 3, 1, 2))  # torch.Size([65, 1, 88, 88])
            transform = VideoTransform(subset="test")
            video1 = transform(vid1)[:50] # {50, 1, 88, 88}

            video_path2 = os.path.join("/home/xueke/LRS2/mvlrs_v1/mouth_mp4/", f"{filename2}.mp4")
            vid2 = torchvision.io.read_video(video_path2, pts_unit="sec", output_format="THWC")[0]
            vid2 = vid2.permute((0, 3, 1, 2))  # torch.Size([65, 1, 88, 88])
            # transform = VideoTransform(subset="test")
            video2 = transform(vid2)[:50] # {50, 1, 88, 88}

            videos = torch.stack([video1, video2], dim=0)

            source_frames = []
            for src in self.sources[idx]: 
                filename1 = os.path.splitext(os.path.basename(src[1]))[0]
                frame1_path = os.path.join("/home/xueke/LRS2/mvlrs_v1/frames_112/val", filename1)
                # print(frame1_path, frame2_path)
                frame_paths1 = sorted(glob.glob(os.path.join(frame1_path, '*.png')))
                frames1 = []
                for frame_path in frame_paths1:
                    img = Image.open(frame_path)  # 不再加 convert('L')，因为你已经是灰度图
                    img = self.transform(img)     # [1, 112, 112]
                    frames1.append(img)
                frames1 = torch.stack(frames1)
                # print(frames1.shape)
                source_frames.append(frames1)
            source_frames = torch.stack(source_frames)          
            # sources_mouths = torch.stack(
            #     [
            #         torch.from_numpy(
            #             self.lipreading_preprocessing_func(convert_bgr2gray(np.load(src[1])["data"]))
            #         )
            #         for src in self.sources[idx]
            #     ]
            # )[:, : self.fps_len]
            # import pdb; pdb.set_trace()
            sources = torch.stack([torch.from_numpy(source) for source in sources])
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            # return mixture, sources, sources_mouths, self.mix[idx][0].split("/")[-1]
            source_stft0 = self.stft_encoder(sources[0])
            source_stft1 = self.stft_encoder(sources[1])
            source_stft = torch.cat([source_stft0, source_stft1], dim=0)

            # return mixture, sources, source_stft, sources_mouths, source_frames, text1, text2, videos
            return mixture, sources, source_stft, sources_mouths, source_frames

class AVSpeechDataset_test(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 1,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        is_train: bool = True
    ):
        super().__init__()
        self.stft_encoder = STFTEncoder(win=256, hop_length=128, bias=False)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),  # 自动将 [H, W] 转为 [1, H, W]，像素归一化到 [0, 1]
            transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图只用一个通道
        ])
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        self.i = []
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            self.fps_len = int(segment * 25)
        self.n_src = n_src
        self.test = self.seg_len is None  # 如果segment为None，test是true
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []
        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            print(len(mix_infos), len(sources_infos[0]), len(sources_infos[1]))
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len
        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        self.mix.append(mix_infos[i])
                        self.sources.append([src_inf[i] for src_inf in sources_infos])

            else:
                for i in range(len(mix_infos)):
                    self.mix.append(mix_infos[i])
                    self.sources.append([sources_infos[0][i], sources_infos[1][i]])
            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

    def __len__(self):
        return self.length
    
    # def _pad_tensor(self, tensor, target_shape):
    #         tensor = torch.tensor(tensor, dtype=torch.float32)
    #         pad_size = (target_shape[0] - tensor.shape[0],)
    #         padding_tensor = torch.zeros((pad_size[0],) + tensor.shape[1:], dtype=tensor.dtype)
    #         padded_tensor = torch.cat((tensor, padding_tensor), dim=0)
    #         padded_array = padded_tensor.numpy()
            
    #         return padded_array
    def _pad_tensor(self, tensor, target_shape):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        print(type(tensor))
        tensor = tensor.float()  # 保证类型一致
        # print(type(tensor))
        pad_len = target_shape[0] - tensor.shape[0]
        if pad_len > 0:
            padding_tensor = torch.zeros((pad_len,) + tensor.shape[1:], dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding_tensor), dim=0)
            # print(type(tensor))

        return tensor  # ✅ 返回的是 Tensor，不转 numpy

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        if self.n_src == 1:
            # rand_start = 0  # # # 数据集没有按照min生成所以可能会有大量的空的，所以还是从0开始比较好
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                # rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len - 640*5)
                max_frame = (self.mix[idx][1] - self.seg_len - 640 * 5) // 640
                if max_frame > 0:
                    rand_start = np.random.randint(0, max_frame) * 640
                else:
                    rand_start = 0
            if self.test:
                # stop = None
                stop = rand_start + 32000
            else:
                stop = rand_start + self.seg_len

            mix_source, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            # print(self.mix[idx][0])
            source = sf.read(
                self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
            )[0]

            # source_mouth = self.lipreading_preprocessing_func(
            #     np.load(self.sources[idx][1])["data"]
            # )[: 50]  # [start_frame:stop_frame]

            source_mouth = self.lipreading_preprocessing_func(
                convert_bgr2gray(np.load(self.sources[idx][1])["data"])
            )[: 50]

            source = torch.from_numpy(source)
            mixture = torch.from_numpy(mix_source)
            # source_mouth = torch.from_numpy(source_mouth)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)
            # return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1]
            return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1]


        if self.n_src == 2:
            rand_start = 0
            if self.test:
                stop = 32000
            else:
                stop = rand_start + self.seg_len

            mix_source, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            sources = []
            for src in self.sources[idx]:
                # import pdb; pdb.set_trace()
                sources.append(
                    sf.read(src[0], start=rand_start, stop=stop, dtype="float32")[0]
                )
            # # import pdb; pdb.set_trace()
            sources_mouths = []
            for src in self.sources[idx]:

                mouth_tensor = torch.tensor(np.load(src[1]))  # [: 50]
                sources_mouths.append(mouth_tensor)
            sources_mouths = torch.stack(sources_mouths)

            text_dir = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/lrs2_2s_txt"
            filenames = []
            for src in self.sources[idx]:
                filename = os.path.splitext(os.path.basename(src[1]))[0]
                filenames.append(filename)
            filename1 = filenames[0]
            filename2 = filenames[1]
            frame1_path_text = os.path.join(text_dir, f"{filename1}.txt")   # batchsize只能是1
            with open(frame1_path_text, "r") as f:
                text1 = f.read().strip()
            frame2_path_text = os.path.join(text_dir, f"{filename2}.txt")   # batchsize只能是1
            with open(frame2_path_text, "r") as f:
                text2 = f.read().strip()

            video_path1 = os.path.join("/home/xueke/LRS2/mvlrs_v1/mouth_mp4/", f"{filename1}.mp4")
            vid1 = torchvision.io.read_video(video_path1, pts_unit="sec", output_format="THWC")[0]
            vid1 = vid1.permute((0, 3, 1, 2))  # torch.Size([65, 1, 88, 88])
            transform = VideoTransform(subset="test")
            video1 = transform(vid1)[:50] # {50, 1, 88, 88}

            video_path2 = os.path.join("/home/xueke/LRS2/mvlrs_v1/mouth_mp4/", f"{filename2}.mp4")
            vid2 = torchvision.io.read_video(video_path2, pts_unit="sec", output_format="THWC")[0]
            vid2 = vid2.permute((0, 3, 1, 2))  # torch.Size([65, 1, 88, 88])
            # transform = VideoTransform(subset="test")
            video2 = transform(vid2)[:50] # {50, 1, 88, 88}

            videos = torch.stack([video1, video2], dim=0)

            source_frames = []
            for src in self.sources[idx]: 
                filename1 = os.path.splitext(os.path.basename(src[1]))[0]
                frame1_path = os.path.join("/home/xueke/LRS2/mvlrs_v1/frames_112/test", filename1)
                # print(frame1_path, frame2_path)
                frame_paths1 = sorted(glob.glob(os.path.join(frame1_path, '*.png')))
                frames1 = []
                for frame_path in frame_paths1:
                    img = Image.open(frame_path)  # 不再加 convert('L')，因为你已经是灰度图
                    img = self.transform(img)     # [1, 112, 112]
                    frames1.append(img)
                frames1 = torch.stack(frames1)
                # print(frames1.shape)
                source_frames.append(frames1)
            source_frames = torch.stack(source_frames)          
            # sources_mouths = torch.stack(
            #     [
            #         torch.from_numpy(
            #             self.lipreading_preprocessing_func(convert_bgr2gray(np.load(src[1])["data"]))
            #         )
            #         for src in self.sources[idx]
            #     ]
            # )[:, : self.fps_len]
            # import pdb; pdb.set_trace()
            sources = torch.stack([torch.from_numpy(source) for source in sources])
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            # return mixture, sources, sources_mouths, self.mix[idx][0].split("/")[-1]
            source_stft0 = self.stft_encoder(sources[0])
            source_stft1 = self.stft_encoder(sources[1])
            source_stft = torch.cat([source_stft0, source_stft1], dim=0)

            # return mixture, sources, source_stft, sources_mouths, source_frames, self.mix[idx][0].split("/")[-1], text1, text2, videos
            return mixture, sources, source_stft, sources_mouths, source_frames, self.mix[idx][0].split("/")[-1]



class AVSpeechDynamicDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        is_train: bool = True,
        spk_list: str = None
    ):
        super().__init__()

        self.stft_encoder = STFTEncoder(256, 128, bias=False)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),  # 自动将 [H, W] 转为 [1, H, W]，像素归一化到 [0, 1]
            transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图只用一个通道
        ])
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            self.fps_len = int(segment * 25)
        self.n_src = n_src
        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.spk_list = self._load_spk(spk_list)

        self.mix = []
        self.sources = []
        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

        elif self.n_src == 3: # 先这样改
            orig_len = len(mix_infos)
            # print(orig_len) 20000
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        self.mix.append(mix_infos[i])
                        self.sources.append([src_inf[i] for src_inf in sources_infos])
                        # print(self.sources)

            else:
                for i in range(len(mix_infos)):
                    self.mix.append(mix_infos[i])
                    self.sources.append([sources_infos[0][i], sources_infos[1][i]])
            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

    def __len__(self):
        return self.length

    def _pad_tensor(self, tensor, target_shape):
            tensor = torch.tensor(tensor, dtype=torch.float32)
            pad_size = (target_shape[0] - tensor.shape[0],)
            padding_tensor = torch.zeros((pad_size[0],) + tensor.shape[1:], dtype=tensor.dtype)
            padded_tensor = torch.cat((tensor, padding_tensor), dim=0)
            padded_array = padded_tensor.numpy()
            
            return padded_array

    def _load_spk(self, spk_list_path):
        if spk_list_path is None:
            return []
        with open(spk_list_path) as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        
        rand_start = 0
        if self.test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        
        s1_json = None
        s2_json = None
        while True:
            s1_json = random.choice(self.sources)
            # print(s1_json)
             #print(s1_json[0])
            spk = s1_json[0].split('/')[-2]
            # print(spk)
            spk_id = ["s1", "s2"].index(spk)  # s2的话就是1，s1的话就是0
            # print(spk_id)
            s1_split = s1_json[0].split('/')[-1].split('_')
            # print(s1_split)
            res = ["{}_{}".format(s1_split[0], s1_split[1]), "{}_{}".format(s1_split[3], s1_split[4])]
            s1_name = res[spk_id]
            # print(s1_name)
            
            s2_json = random.choice(self.sources)
            # print(s2_json[0])
            spk = s2_json[0].split('/')[-2]
            # print(spk)
            spk_id = ["s1", "s2"].index(spk)
            # print(spk_id)
            s2_split = s2_json[0].split('/')[-1].split('_')
            # print(s2_split)
            res = ["{}_{}".format(s2_split[0], s2_split[1]), "{}_{}".format(s2_split[3], s2_split[4])]
            s2_name = res[spk_id]
            # print(s2_name)
            
            if s1_name != s2_name:
                break
        
        s1 = sf.read(
            s1_json[0], start=rand_start, stop=stop, dtype="float32"
        )[0]
        # print(s1.dtype)   # 通常是 float32
        # print(s1.min(), s1.max())  # 通常在 -1 到 1 之间
        s2 = sf.read(
            s2_json[0], start=rand_start, stop=stop, dtype="float32"
        )[0]

        s1_mouth = torch.tensor(np.load(s1_json[1]))
        s2_mouth = torch.tensor(np.load(s2_json[1]))

        # s1_mouth = self.lipreading_preprocessing_func(
        #     convert_bgr2gray(np.load(s1_json[1])["data"])
        # )[: self.fps_len]
        # if s1_mouth.shape[0] < 50:
        #     s1_mouth = self._pad_tensor(s1_mouth, (50, s1_mouth.shape[1], s1_mouth.shape[2]))
        # else:
        #     s1_mouth = torch.tensor(s1_mouth, dtype=torch.float32)
        
        # s2_mouth = self.lipreading_preprocessing_func(
        #     convert_bgr2gray(np.load(s2_json[1])["data"])
        # )[: self.fps_len]
        # if s2_mouth.shape[0] < 50:
        #     s2_mouth = self._pad_tensor(s2_mouth, (50, s2_mouth.shape[1], s2_mouth.shape[2]))
        # else:
        #     s2_mouth = torch.tensor(s2_mouth, dtype=torch.float32)
        
        # s1_mouth = self.lipreading_preprocessing_func(
        #     np.load(s1_json[1])["data"]
        # )[: self.fps_len]
        
        # s2_mouth = self.lipreading_preprocessing_func(
        #     np.load(s2_json[1])["data"]
        # )[: self.fps_len]

        # nrc＝1的情况
        # sources_json = [s1_json, s2_json]
        # mouths = [s1_mouth, s2_mouth]
        # mouths_key = [s1_json[1],s2_json[1]]
        # sources = [torch.from_numpy(s1), torch.from_numpy(s2)]
        # select_idx = np.random.randint(0,2)  # 0, 1
        # mixture = torch.from_numpy(s1) + torch.from_numpy(s2)
        # source = sources[select_idx]
        # source_mouth = mouths[select_idx]
        # source_key = mouths_key[select_idx]

        # spk_id = os.path.basename(source_key).split('_')[0]
        # spk_idx = self.spk_list.index(spk_id)
        
        # if self.normalize_audio:  # 还没有进行归一化
        #     m_std = mixture.std(-1, keepdim=True)
        #     mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
        #     source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)
        # # print(mixture.shape, source.shape, source_mouth.shape)
        # return mixture, source, source_mouth, sources_json[select_idx][0].split("/")[-1], source_key, spk_idx 
    
        sources_json = [s1_json, s2_json]  # nrc==2
        # print(sources_json)
        source_mouth = torch.cat([s1_mouth.unsqueeze(0), s2_mouth.unsqueeze(0)], dim=0)
        # print(mouths_tensor.shape)  # torch.Size([2, 50, 88, 88])
        mouths_key = [s1_json[1],s2_json[1]]
        text_dir = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/lrs2_2s_txt"
        # print(mouths_key)
        filename1 = os.path.splitext(os.path.basename(s1_json[1]))[0]
        filename2 = os.path.splitext(os.path.basename(s2_json[1]))[0]
        frame1_path = os.path.join("/home/xueke/LRS2/mvlrs_v1/frames_112/train", filename1)
        frame1_path_text = os.path.join(text_dir, f"{filename1}.txt")   # batchsize只能是1

        video_path1 = os.path.join("/home/xueke/LRS2/mvlrs_v1/mouth_mp4/", f"{filename1}.mp4")
        vid1 = torchvision.io.read_video(video_path1, pts_unit="sec", output_format="THWC")[0]
        vid1 = vid1.permute((0, 3, 1, 2))  # torch.Size([65, 1, 88, 88])
        transform = VideoTransform(subset="test")
        video1 = transform(vid1)[:50] # {50, 1, 88, 88}

        video_path2 = os.path.join("/home/xueke/LRS2/mvlrs_v1/mouth_mp4/", f"{filename2}.mp4")
        vid2 = torchvision.io.read_video(video_path2, pts_unit="sec", output_format="THWC")[0]
        vid2 = vid2.permute((0, 3, 1, 2))  # torch.Size([65, 1, 88, 88])
        # transform = VideoTransform(subset="test")
        video2 = transform(vid2)[:50] # {50, 1, 88, 88}

        videos = torch.stack([video1, video2], dim=0)

        frame2_path = os.path.join("/home/xueke/LRS2/mvlrs_v1/frames_112/train", filename2)
        frame2_path_text = os.path.join(text_dir, f"{filename2}.txt")


        # print(frame1_path, frame2_path)
        frame_paths1 = sorted(glob.glob(os.path.join(frame1_path, '*.png')))
        frames1 = []
        for frame_path in frame_paths1:
            img = Image.open(frame_path)  # 不再加 convert('L')，因为你已经是灰度图
            img = self.transform(img)     # [1, 112, 112]
            frames1.append(img)

        frames1 = torch.stack(frames1)  # shape: [T, 1, 112, 112]
        frame_paths2 = sorted(glob.glob(os.path.join(frame2_path, '*.png')))
        frames2 = []
        for frame_path in frame_paths2:
            img = Image.open(frame_path)  # 不再加 convert('L')，因为你已经是灰度图
            img = self.transform(img)     # [1, 112, 112]
            frames2.append(img)

        frames2 = torch.stack(frames2)  # shape: [T, 1, 112, 112]
        # print(frames1.shape, frames2.shape)
        frames = torch.cat([frames1.unsqueeze(0), frames2.unsqueeze(0)], dim=0)

        s1_tensor = torch.from_numpy(s1)
        s2_tensor = torch.from_numpy(s2)
        sources = torch.cat([s1_tensor.unsqueeze(0), s2_tensor.unsqueeze(0)], dim=0)
        # print(sources1.shape)  # torch.Size([2, 32000])
        mixture = torch.from_numpy(s1) + torch.from_numpy(s2)
        spk_id0 = os.path.basename(mouths_key[0]).split('_')[0]
        spk_id1 = os.path.basename(mouths_key[1]).split('_')[0]
        spk_idx0 = self.spk_list.index(spk_id0)
        spk_idx1 = self.spk_list.index(spk_id1)
        spk_idx = [spk_idx0, spk_idx1]
        # print(spk_idx[0])
        # print(spk_idx[1])

        with open(frame1_path_text, "r") as f:
            text1 = f.read().strip()
        with open(frame2_path_text, "r") as f:
            text2 = f.read().strip()
        
        if self.normalize_audio:  # 还没有进行归一化
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)
        # print(mixture.shape, source.shape, source_mouth.shape)

        source_stft0 = self.stft_encoder(sources[0])
        source_stft1 = self.stft_encoder(sources[1])
        source_stft = torch.cat([source_stft0, source_stft1], dim=0)
        # print(source_stft.shape)

        # return mixture, sources, source_stft, source_mouth, [sources_json[0][0].split("/")[-1],sources_json[1][0].split("/")[-1]], spk_idx ,frames, text1, text2, videos
        return mixture, sources, source_stft, source_mouth, [sources_json[0][0].split("/")[-1],sources_json[1][0].split("/")[-1]], spk_idx ,frames

        
class AVSpeechDyanmicDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        spk_list: str,
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        # this line allows to access init params with 'self.hparams' attribute
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.spk_list = spk_list
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = AVSpeechDynamicDataset(
            json_dir = self.train_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            segment = self.segment,
            normalize_audio = self.normalize_audio,
            is_train=True,
            spk_list=self.spk_list,
        )
        self.data_val = AVSpeechDataset(
            json_dir = self.valid_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            # segment = self.segment,
            segment = None,
            normalize_audio = self.normalize_audio,
            is_train=False
        )
        self.data_test = AVSpeechDataset_test(
            json_dir = self.test_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            # segment = self.segment,
            segment = None,
            normalize_audio = self.normalize_audio,
            is_train=False
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test

if __name__ == "__main__":
    # from tqdm import tqdm
    # val_set = AVSpeechDynamicDataset(
    #     "/root/dataset/LRS2/mvlrs_v1/raw_data/train/",
    #     n_src=1,
    #     sample_rate=16000,
    #     # segment=conf["data"]["segment"],
    #     segment=2.0,
    #     normalize_audio=False
    # )
    # import torch
    # import numpy as np

    # class YourClass:
    #     def _pad_tensor(self, tensor, target_shape):
    #         # 将 NumPy 数组转换为 PyTorch 张量
    #         tensor = torch.tensor(tensor, dtype=torch.float32)
            
    #         # 计算需要填充的大小
    #         pad_size = (target_shape[0] - tensor.shape[0],)
            
    #         # 创建全零张量
    #         padding_tensor = torch.zeros((pad_size[0],) + tensor.shape[1:], dtype=tensor.dtype)
            
    #         # 拼接原始张量和填充张量
    #         padded_tensor = torch.cat((tensor, padding_tensor), dim=0)
            
    #         return padded_tensor

    # # 示例用法
    # your_class_instance = YourClass()
    # s1_mouth = np.random.randn(30, 64, 64)  # 假设 s1_mouth 的形状是 (30, 64, 64)
    # padded_s1_mouth = your_class_instance._pad_tensor(s1_mouth, (50, s1_mouth.shape[1], s1_mouth.shape[2]))

    # print(padded_s1_mouth.shape)  # 应该输出 (50, 64, 64)
    # for idx in tqdm(range(len(val_set))):
    #     mixture, sources, sources_mouths, _ = val_set[idx]
    #     import pdb; pdb.set_trace()

# # 可以输入：

# p idx：打印当前的 idx

# n：进入下一行

# s：如果当前行是个函数调用，就进去看细节

# l：查看上下文代码

# c：继续执行直到下一个断点或程序结束

# q：退出调试器
    from tqdm import tqdm
    # import fast_bss_eval
    val_set = AVSpeechDataset_test(
        "/home/xueke/LRS2/mvlrs_v1/mix_audio_2speakers/wav16k/min/tt/",
        n_src=2,
        sample_rate=16000,
        # segment=conf["data"]["segment"],
        segment=2.0,
        normalize_audio=False,
        # spk_list = "/home/xueke/LRS2/mvlrs_v1/frames_112/train_id.spk"
    )
    # for idx in tqdm(range(len(val_set))):
    #     mixture, sources, sources_mouths = val_set[idx]
    #     print(mixture.shape, sources.shape, sources_mouths.shape)  # 输出将根据__getitem__方法中的返回语句而定
    #     # import pdb; pdb.set_trace()
    #     # import pdb 导入 Python 的调试器模块 pdb。
    #     # pdb.set_trace() 设置一个断点，程序执行到这一行时会暂停，进入交互式调试模式。
    #     # 在 pdb 的交互式调试模式中，你可以使用以下命令：
    #     # c 或 continue：继续执行代码，直到下一个断点。
    #     # n 或 next：执行下一行代码。
    #     # s 或 step：进入函数内部。
    #     # p variable_name：打印变量的值。
    #     # l 或 list：显示当前代码上下文。
    #     # q 或 quit：退出调试器。
    # 初始化第一个样本作为基准 shape
    # ref_mixture, ref_sources,source_stft,  ref_sources_mouths, _, id, frames = val_set[0]
    ref_mixture, ref_sources, source_stft,  ref_sources_mouths, frames,id, text1, text2, videos = val_set[0]
    print(source_stft.shape)
    print(videos.shape)
    ref_mix_shape = ref_mixture.shape
    ref_src_shape = ref_sources.shape
    ref_mouth_shape = ref_sources_mouths.shape

    print(f"[参考 shape] mixture: {ref_mix_shape}, sources: {ref_src_shape}, mouths: {ref_mouth_shape}")
    # print(key)
    # print(frames.shape)  # [979, 1062] torch.Size([50, 1, 112, 112]) torch.Size([50, 1, 112, 112])

    # 遍历检测不一致
    for idx in tqdm(range(len(val_set))):
        try:
            mixture, sources,source_stft, sources_mouths, key_mix, id, text1, text2, videos = val_set[idx]
            # print(key_mix, id)
            # print(type(mixture), type(sources), type(sources_mouths))
            # sdr_baseline = -fast_bss_eval.sdr_pit_loss(mixture.unsqueeze(0).unsqueeze(0), sources.unsqueeze(0).unsqueeze(0)).mean()
            # print(sdr_baseline)
            if (mixture.shape != ref_mix_shape or 
                sources.shape != ref_src_shape or 
                sources_mouths.shape != ref_mouth_shape):
                print(f"[❌ idx {idx}] mismatch!")
                print(f"  mixture shape: {mixture.shape}")
                print(f"  sources shape: {sources.shape}")
                print(f"  mouths shape: {sources_mouths.shape}")
                
        except Exception as e:
            print(f"[💥 idx {idx}] Exception occurred: {e}")