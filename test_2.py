import os
import random
from typing import Union
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy.io import wavfile
import warnings
# import torchaudio
warnings.filterwarnings("ignore")
import look2hear.models
import look2hear.videomodels
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from look2hear.models.DPT_1d import DPT_1d 
from look2hear.models.tfgridnet_v2 import tfgridnet_v2
from look2hear.models.tfgridnet_v2_a_o import tfgridnet_v2_a_o
from look2hear.models.tfgridnet_v2_Lip_o import tfgridnet_v2_Lip_o

from look2hear.models.tfgridnet_v2_step2 import tfgridnet_v2_step2


parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir",
                    default="/DPT_1d_main/configs/LRS2-tfgridnet.yaml",
                    help="Full path to save best validation model")

def remove_prefix_from_keys(state_dict, prefixes):
    new_state_dict = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break  
        new_state_dict[key] = value
    return new_state_dict


compute_metrics = ["si_sdr", "sdr"]
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main(config):
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress), 
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn
    )
    # import pdb; pdb.set_trace()
    # config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
    #     os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    # )
    # model_path = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "best_model.pth")
    model_path = "/home/xueke/DPT_1d_main/checkpoint_improve_tfgridnet_LRS2_SS_Lip_only/LRS2-restormer/epoch=86.ckpt"
    # import pdb; pdb.set_trace()
    # conf["train_conf"]["masknet"].update({"n_src": 2})
    # model =  getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(
    #     model_path,
    #     # sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
    #     **config["train_conf"]["audionet"]["audionet_config"],
    # )
    model = tfgridnet_v2_Lip_o(**config["train_conf"]["audionet"]["audionet_config"])
    conf = torch.load(model_path, map_location="cpu")
    conf["state_dict"] = remove_prefix_from_keys(conf["state_dict"], ['audio_model.', 'video_model.'])

    # for key, value in conf["state_dict"].items():  # Remove 'model.' prefix from key
        # print(key)
        # new_key = key.replace('model.', '', 1) 
        # new_state_dict[new_key] = value
        # print("new_state_dict", new_state_dict.keys())
        # self.model.load_state_dict(new_state_dict, strict=False)

    # incompatible_keys  = model.load_state_dict(new_state_dict, strict=False)
    # print("Unmatched keys:", incompatible_keys)

    model.load_state_dict(conf["state_dict"], strict=False)
    video_model = getattr(look2hear.videomodels, config["train_conf"]["videonet"]["videonet_name"])(
        **config["train_conf"]["videonet"]["videonet_config"],
    )
    video_model.load_state_dict(conf["state_dict"], strict=False)
    if config["train_conf"]["training"]["gpus"]:
        device = "cuda"
        model.to(device)
        video_model.to(device)
    model_device = next(model.parameters()).device
    datamodule: object = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
        **config["train_conf"]["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _ , test_set = datamodule.make_sets
   
    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join("/DPT_1d_main", "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    metrics = MetricsTracker(
        save_file=os.path.join(ex_save_dir, "metrics_LRS2_2mix_lip_only.csv"))
    video_model.eval()
    model.eval() 
    torch.no_grad().__enter__()
    with progress:
        for idx in progress.track(range(len(test_set))):
            # Forward the network on the mixture.
            mix, sources, source_stft, mouth, source_frames, key = tensors_to_device(test_set[idx],
                                                    device=model_device)
            # print(sources.shape)
            #  print(videos.shape)
            # mouth_emb = video_model(torch.from_numpy(mouth[None, None]).float().cuda())
            # if mouth.ndim == 4:
            #     mouth = mouth.unsqueeze(1)
            # mouth_emb = video_model(mouth[None, None].float().cuda())
            # print("mouth emb shape:", mouth.shape)  # torch.Size([2, 50, 88, 88])
            # mouth1 = mouth[0]
            # mouth2 = mouth[1]
            # if mouth1.ndim == 3:
            #     mouth1 = mouth1.unsqueeze(0).unsqueeze(1) # torch.Size([1, 1, 50, 88, 88])
            # if mouth2.ndim == 3:
            #     mouth2 = mouth2.unsqueeze(0).unsqueeze(1)
            # with torch.no_grad():
            #     mouth_emb1 = video_model(mouth1.float().cuda())
            #     mouth_emb2 = video_model(mouth2.float().cuda())
            #     mouth_emb1 = mouth_emb1.unsqueeze(1)
            #     mouth_emb2 = mouth_emb2.unsqueeze(1)
            #     mouth_emb = torch.cat([mouth_emb1, mouth_emb2], dim=1)
            #     # print(mouth_emb.shape)  # torch.Size([1, 2, 512, 50])
            # # print(mix.shape, source_frames.shape)  # torch.Size([32000]) torch.Size([2, 50, 1, 112, 112])
            # print(mouth.shape)  # torch.Size([2, 50, 768])
            mouth_emb = mouth.transpose(1, 2).unsqueeze(0) # torch.Size([2, 768, 50])
            stft, est_sources, _ = model(mix.unsqueeze(0), mouth_emb, source_frames.unsqueeze(0))
            mix_np = mix
            # print(mix_np.shape)
            sources_np = sources
             #print(sources_np.shape) # torch.Size([2, 32000])
            est_sources_np = est_sources.squeeze(0)
            # print(est_sources_np.shape) # # torch.Size([2, 32000])
            metrics(mix=mix_np,
                    clean=sources_np,
                    estimate=est_sources_np,
                    key=key)
            if idx % 50 == 0:
                metricscolumn.update(metrics.update())
    metrics.final()


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    # print(arg_dic)
    main(arg_dic)
