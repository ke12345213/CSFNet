###
# Author: Kai Li
# Date: 2021-06-03 18:29:46
# LastEditors: Please set LastEditors
# LastEditTime: 2022-03-16 06:36:17
###

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
import librosa

from .transform import get_preprocessing_pipelines
def convert_bgr2gray(data):
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


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
            # print(self.sources[idx][0])
            # video_fps = 25
            # audio_sr = 16000
            # samples_per_frame = audio_sr // video_fps  # 640
            # start_frame = rand_start // samples_per_frame
            # stop_frame = stop // samples_per_frame
            # mix_source1, sr = librosa.load(self.mix[idx][0], sr=None)  # sr=None保持原始采样率
            # duration = librosa.get_duration(y=mix_source1, sr=sr)
            
            # duration_samples = int(duration * sr)  # 将时长转换为采样点数
            # # print(self.mix[idx][0])
            # print("duration_samples", duration_samples)
            # frames = duration_samples // 640  # 正常情况下这个是比叫多的。但是这个多出来的帧数并没有超过5帧，也就是0.2s
            # source_mouth1 = self.lipreading_preprocessing_func(
            #     convert_bgr2gray(np.load(self.sources[idx][1])["data"])
            # )
            # print(source_mouth1.shape[0])
            # # if abs(source_mouth1.shape[0] - frames) > 50:
            # #     self.i += 1  # 替换为你实际使用的变量名
            # if frames - source_mouth1.shape[0] > 5:
            #     self.i += 1  # 替换为你实际使用的变量名
            # print(self.i)
            

            source_mouth = self.lipreading_preprocessing_func(
                convert_bgr2gray(np.load(self.sources[idx][1])["data"])
            )[rand_start // 640: 50 + rand_start // 640]  # [start_frame:stop_frame]
            if source_mouth.shape[0] < 50:  # 错了三个，还好
                source_mouth = self._pad_tensor(source_mouth, (50, source_mouth.shape[1], source_mouth.shape[2]))
                self.i.append(self.sources[idx][1])
                # print(self.i)
            else:
                source_mouth = torch.tensor(source_mouth, dtype=torch.float32)
            # print(self.i)
            assert isinstance(source_mouth, torch.Tensor), f"type: {type(source_mouth)}"
            # print(source_mouth.shape)
            # print(type(source_mouth))
            # print(self.sources[idx][1])
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
            # rand_start = 0
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
                # rand_start = 0
            # print(idx, self.mix[idx][1], rand_start)
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
            # # import pdb; pdb.set_trace()
            sources_mouths = []
            for src in self.sources[idx]:
                mouth_tensor = torch.from_numpy(
                    self.lipreading_preprocessing_func(
                        convert_bgr2gray(np.load(src[1])["data"])
                    )
                )[:self.fps_len]

                if mouth_tensor.shape[0] < 50:
                    mouth_tensor = self._pad_tensor(mouth_tensor, (50, mouth_tensor.shape[1], mouth_tensor.shape[2]))

                sources_mouths.append(mouth_tensor)
            sources_mouths = torch.stack(sources_mouths)                
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

            return mixture, sources, sources_mouths

class AVSpeechDataset_valid(Dataset):
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
        
        tensor = tensor.float()  # 保证类型一致

        pad_len = target_shape[0] - tensor.shape[0]
        if pad_len > 0:
            padding_tensor = torch.zeros((pad_len,) + tensor.shape[1:], dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding_tensor), dim=0)

        return tensor  # ✅ 返回的是 Tensor，不转 numpy

    def __getitem__(self, idx: int):
        self.EPS = 1e-8
        if self.n_src == 1:
            rand_start = 0  # # # 
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len

            mix_source, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            # print(self.mix[idx][0])
            source = sf.read(
                self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
            )[0]
            # print(self.sources[idx][0])
            source_mouth = self.lipreading_preprocessing_func(
                convert_bgr2gray(np.load(self.sources[idx][1])["data"])
            )  # [: self.fps_len]
            # if source_mouth.shape[0] < 50:  # 错了三个，还好
               #  source_mouth = self._pad_tensor(source_mouth, (50, source_mouth.shape[1], source_mouth.shape[2]))
            # print(self.sources[idx][1])
            source = torch.from_numpy(source)
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                source = normalize_tensor_wav(source, eps=self.EPS, std=m_std)
            # return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1]
            return mixture, source, source_mouth


        if self.n_src == 2:
            rand_start = 0
            # if self.mix[idx][1] == self.seg_len or self.test:
            #     rand_start = 0
            # else:
            #     rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
            #     # rand_start = 0
            # # print(idx, self.mix[idx][1], rand_start)
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
                )   # 这个source已经提前对应了，不再需要改了
            # # import pdb; pdb.set_trace()
            sources_mouths = []
            for src in self.sources[idx]:
                mouth_tensor = torch.from_numpy(
                    self.lipreading_preprocessing_func(
                        convert_bgr2gray(np.load(src[1])["data"])
                    )
                )[:50]  #### 注意这里我是随便写的，没有用实际的值，这个视频是无效的也就是不能和音频对应，音频对应的话没有变法pad，同时这里也对应n=2，要慎重使用

                if mouth_tensor.shape[0] < 50:
                    mouth_tensor = self._pad_tensor(mouth_tensor, (50, mouth_tensor.shape[1], mouth_tensor.shape[2]))

                sources_mouths.append(mouth_tensor)
            sources_mouths = torch.stack(sources_mouths)                
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

            return mixture, sources, sources_mouths

class AVSpeechDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
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
        self.data_train = AVSpeechDataset(
            json_dir = self.train_dir,
            n_src = self.n_src,
            sample_rate = self.sample_rate,
            segment = self.segment,
            normalize_audio = self.normalize_audio,
            is_train=True
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
        self.data_test = AVSpeechDataset(
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
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=1,
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
    from tqdm import tqdm
    import fast_bss_eval
    val_set = AVSpeechDataset(
        "/root/dataset/LRS2/mvlrs_v1/raw_data/test/",
        n_src=1,
        sample_rate=16000,
        # segment=conf["data"]["segment"],
        segment=None,
        normalize_audio=False
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
    ref_mixture, ref_sources, ref_sources_mouths, _ = val_set[0]
    ref_mix_shape = ref_mixture.shape
    ref_src_shape = ref_sources.shape
    ref_mouth_shape = ref_sources_mouths.shape

    print(f"[参考 shape] mixture: {ref_mix_shape}, sources: {ref_src_shape}, mouths: {ref_mouth_shape}")

    # 遍历检测不一致
    for idx in tqdm(range(len(val_set))):
        try:
            mixture, sources, sources_mouths, _ = val_set[idx]
            # print(type(mixture), type(sources), type(sources_mouths))
            sdr_baseline = -fast_bss_eval.sdr_pit_loss(mixture.unsqueeze(0).unsqueeze(0), sources.unsqueeze(0).unsqueeze(0)).mean()
            print(sdr_baseline)
            if (mixture.shape != ref_mix_shape or 
                sources.shape != ref_src_shape or 
                sources_mouths.shape != ref_mouth_shape):
                print(f"[❌ idx {idx}] mismatch!")
                print(f"  mixture shape: {mixture.shape}")
                print(f"  sources shape: {sources.shape}")
                print(f"  mouths shape: {sources_mouths.shape}")
                
        except Exception as e:
            print(f"[💥 idx {idx}] Exception occurred: {e}")
