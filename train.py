import os
import sys
import torch
from torch import Tensor
import argparse
import json
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
import look2hear.videomodels
from look2hear.system import make_optimizer
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from rich import print, reconfigure
from collections.abc import MutableMapping
from look2hear.utils import print_only, MyRichProgressBar, RichProgressBarTheme

import warnings


########## L1loss大概是0.047，还是比较大
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, num_epochs, iter_per_epoch):
        self.base_lrs = {
            param_group["name"]: param_group["lr"]
            for param_group in optimizer.param_groups
        }
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

        self.init_lr()  # so that at first step we have the correct step size

    def get_lr(self, base_lr):
        if self.iter < self.warmup_iter:
            return base_lr * self.iter / self.warmup_iter
        else:
            decay_iter = self.total_iter - self.warmup_iter
            return (
                0.5
                * base_lr
                * (1 + np.cos(np.pi * (self.iter - self.warmup_iter) / decay_iter))
            )

    def update_param_groups(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.base_lrs[param_group["name"]])

    def step(self):
        self.update_param_groups()
        self.iter += 1

    def init_lr(self):
        self.update_param_groups()

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler_redu(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, num_epochs, iter_per_epoch, min_lr=1e-6, max_lr=1e-3):
        self.base_lrs = {
            param_group["name"]: param_group["lr"]
            for param_group in optimizer.param_groups
        }
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

        # Set the min and max learning rate for cosine warm-up
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        # Initialize ReduceLROnPlateau scheduler for post-warmup phase
        self.plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.9)

        # Perform the warmup initialization
        self.init_lr()

    def get_lr(self, base_lr):
        # First, calculate the warmup phase learning rate
        if self.iter < self.warmup_iter:
            return self.min_lr + (self.max_lr - self.min_lr) * (self.iter / self.warmup_iter)
        else:
            return self.max_lr

    def update_param_groups(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.base_lrs[param_group["name"]])

    def step(self, val_loss=None):
        # Update the learning rate based on warm-up or cosine annealing
        self.update_param_groups()
        
        # If we have completed warmup, start using ReduceLROnPlateau
        if self.iter >= self.warmup_iter:
            if val_loss is not None:
                # Apply the ReduceLROnPlateau scheduler based on validation loss
                self.plateau_scheduler.step(val_loss)

        # Increment the iteration counter
        self.iter += 1

    def init_lr(self):
        # Initialize the learning rate at the start of the training
        self.update_param_groups()


def main(config):
    print_only(
        "Instantiating datamodule <{}>".format(config["datamodule"]["data_name"])
    )
    datamodule: object = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader
    # Define model and optimizer
    # print(getattr(look2hear.models, config["audionet"]["audionet_name"]))

    print_only(
        "Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"])
    )
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        # sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )

    # # 动态获取类并实例化
    # model_class = getattr(look2hear.models, config["audionet"]["audionet_name"])实例化这个类
    # model = model_class(
    # sample_rate=config["datamodule"]["data_config"]["sample_rate"],
    # **config["audionet"]["audionet_config"]
    # )传入参数

    video_model = getattr(look2hear.videomodels, config["videonet"]["videonet_name"])(
        **config["videonet"]["videonet_config"],
    )  # 前面是获取指定的模型类，后面是实例化模型类
    # import pdb; pdb.set_trace()
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])
    # optimizer = torch.optim.AdamW([{'name': 'model', 
    #                                 'params': model.parameters(),
    #                                 'lr': config["optimizer"]["lr"]}], 
    #                                 weight_decay=config["optimizer"]["weight_decay"],
    #                                  betas=(0.9, 0.999)
    #                                  )
    # # print("eps=", config["optimizer"]["eps"])
    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"])
        )

        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
            optimizer=optimizer, **config["scheduler"]["sche_config"],
            # eta_min= 0.000001
        )
    #     scheduler = {
    #     "scheduler": [cosine_scheduler, plateau_scheduler],
    #     "interval": "epoch",
    #     "monitor": "val_loss",  # 用于 ReduceLROnPlateau
    #     "frequency": 1,
    #     }
    #     # print("eta_min =", config["scheduler"]["sche_config"]["T_max"])
    #     # print("type =", type(config["scheduler"]["sche_config"]["T_max"]))
    # print(len(train_loader)//8)
    # scheduler = WarmupCosineScheduler_redu(optimizer, 10, config["training"]["epochs"], len(train_loader)//4)
    # scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}



    # Just after instantiating, save the args. Easy loading in the future.
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "/home/xueke/DPT_1d_main", "checkpoint_improve_tfgridnet_LRS2_SS_face_only", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # Define Loss function.
    print_only(
        "Instantiating Loss, Train <{}>, Val <{}>".format(
            config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]
        )
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    print_only("Instantiating System <{}>".format(config["training"]["system"]))
    system = getattr(look2hear.system, config["training"]["system"])(
        audio_model=model,
        video_model=video_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    # callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))
    # MyRichProgressBar 是一个自定义的进度条类，添加一些自定义的行为或样式，自定义进度条的颜色、字体等。
    # 继承自 pytorch_lightning.callbacks.RichProgressBar，并且可以自定义进度条的主题样式

    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "gpu" if torch.cuda.is_available() else None

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    # comet_logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])
    # comet_logger = WandbLogger(
    #         name=config["exp"]["exp_name"], 
    #         save_dir=os.path.join(logger_dir, config["exp"]["exp_name"]), 
    #         project="CVPR2023-Fast",
    #         # offline=True
    # )

    trainer = pl.Trainer(
        precision=32,  # bf16 保留了 float32 的数值范围（因为它有相同的 8 位指数），所以更适合深度学习中需要大动态范围的操作。
        # fp16 精度更高，但数值范围小，可能出现溢出或精度损失问题。	由于动态范围大，bf16 更不容易因梯度爆炸等问题导致 NaN。
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=True),
        # limit_train_batches=1,  # Useful for fast experiment
        # limit_val_batches=1,
        # limit_test_batches=1,
        gradient_clip_val=5.0,
        # logger=comet_logger,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        # fast_dev_run=True,
    )
    trainer.fit(system)
    # trainer.fit(system, ckpt_path='/home/xueke/DPT_1d_main/checkpoint_improve_tfgridnet_LRS2_SS_step2/LRS2-restormer/epoch=28-16.65.ckpt')
    print_only("Finished Training")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    # to_save = system.audio_model.serialize()
    # torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    torch.save(system.audio_model.state_dict(), os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    # from pprint_only import pprint_only
    from look2hear.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_dir",
                        default="/home/xueke/DPT_1d_main/configs/LRS2-tfgridnet.yaml",
                        help="Full path to save best validation model",
                        )
    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
        # print(def_conf)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # print(parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    # pprint_only(arg_dic)
    main(arg_dic)
