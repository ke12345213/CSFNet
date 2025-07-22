###
# Author: Kai Li
# Date: 2022-05-26 18:09:54
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-10-04 16:00:58
###
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from look2hear.models.restormer_2d import STFTEncoder
# from look2hear.losses.pit_wrapper import PITLossWrapper
from look2hear.losses.matrix import pitL1Loss

# def compute_LMag(stft_s_hat_c, stft_s_c):
#     """
#     Args:
#         stft_s_hat_c: Tensor of shape [B, n_src, T, F], complex or real
#         stft_s_c:      Tensor of shape [B, n_src, T, F], complex or real
#     Returns:
#         LMag loss per sample: Tensor of shape [B]
#     """
#     # 取幅度谱 [B, n_src, T, F]
#     mag_s_hat = torch.abs(stft_s_hat_c)
#     mag_s = torch.abs(stft_s_c)

#     # L1 范数差值 [B, n_src, T, F]
#     diff = torch.abs(mag_s_hat - mag_s)

#     # 按 T 和 F 求和 -> [B, n_src]
#     numerator = diff.sum(dim=[2, 3])
#     denominator = mag_s.sum(dim=[2, 3]) + 1e-8

#     # 每个样本每个源的 LMag -> [B, n_src]
#     lmag = numerator / denominator

#     # 最终按源求平均 -> [B]
#     return lmag.mean(dim=1)

def compute_LMag(stft_s_hat_c, stft_s_c, eps=1e-8):
    """
    L1 normalized magnitude loss
    """
    mag_s_hat = torch.abs(stft_s_hat_c)
    mag_s = torch.abs(stft_s_c)

    diff = torch.abs(mag_s_hat - mag_s)  # [B, n_src, T, F]
    numerator = diff.sum(dim=[2, 3])
    denominator = torch.clamp(mag_s.sum(dim=[2, 3]), min=1e-3)

    lmag = numerator / denominator  # [B, n_src]
    lmag = torch.clamp(lmag, min=0.0, max=10.0)  # 防爆炸

    return lmag.mean(dim=1)  # [B]

from itertools import permutations


def pit_lmag_loss(estimates, targets):
    """
    Args:
        estimates: [B, n_src, T, F]
        targets:   [B, n_src, T, F]
    Returns:
        loss: scalar
    """
    B, n_src, T, F = estimates.shape
    perms = list(permutations(range(n_src)))  # e.g., [(0,1), (1,0)]

    losses = []

    for perm in perms:
        # permute estimates to match target source order
        perm_est = estimates[:, perm, :, :]  # [B, n_src, T, F]
        loss = compute_LMag(perm_est, targets)  # [B]
        losses.append(loss.unsqueeze(1))  # [B, 1]

    all_losses = torch.cat(losses, dim=1)  # [B, num_permutations]
    min_loss, _ = torch.min(all_losses, dim=1)  # [B]

    return min_loss.mean()  # scalar

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

def compute_LMag_mask(stft_s_hat_c, stft_s_c, mask=None, weight=5.0):
    """
    LMag loss with optional mask-based weighting.
    Args:
        stft_s_hat_c: Estimated STFT, shape [B, C, T, F]
        stft_s_c: Ground truth STFT, shape [B, C, T, F]
        mask: Optional mask, shape [B, 1, T, 1] (bool or float)
        weight: Weight multiplier for masked regions
    """
    mag_s_hat_c = torch.abs(stft_s_hat_c)
    mag_s_c = torch.abs(stft_s_c)

    mag_diff = torch.abs(mag_s_hat_c - mag_s_c)  # (B, C, T, F)

    if mask is not None:
        mask = mask.float()  # (B, 1, T, 1)
        mask = mask.expand_as(mag_diff)  # (B, C, T, F)
        weighted_diff = mag_diff * (1.0 + mask * (weight - 1.0))
    else:
        weighted_diff = mag_diff

    norm_mag_diff = torch.sum(weighted_diff)
    norm_mag_s_c = torch.sum(torch.abs(mag_s_c))
    return norm_mag_diff / (norm_mag_s_c + 1e-8)


def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioVisualLightningModule(pl.LightningModule):
    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.l1_loss_module = pitL1Loss(loss_weight=1.0, reduction='mean')
        # self.STFTEncoder = STFTEncoder(256, 128, bias=False)

    def forward(self, wav,  mouth=None, frame=None):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        # mouth1 = mouth[:, 0]
        # mouth2 = mouth[:, 1]

        # if mouth1.ndim == 4:
        #     mouth1 = mouth1.unsqueeze(1)
        # if mouth2.ndim == 4:
        #     mouth2 = mouth2.unsqueeze(1)
        # with torch.no_grad():
        #     mouth_emb1 = self.video_model(mouth1.type_as(wav))
        #     mouth_emb2 = self.video_model(mouth2.type_as(wav))
        #     mouth_emb1 = mouth_emb1.unsqueeze(1)
        #     mouth_emb2 = mouth_emb2.unsqueeze(1)
        #     mouth_emb = torch.cat([mouth_emb1, mouth_emb2], dim=1)
        # # if mouth.ndim == 4:
        # #     mouth = mouth.unsqueeze(1)
        # # mouth_emb = self.video_model(mouth.type_as(wav))
        # # mouth = mouth.squeeze(1)
        mouth = mouth.transpose(2, 3)
        return self.audio_model(wav, mouth, frame)

    def training_step(self, batch, batch_nb):
        mixtures, targets, targets_stft, mouth, source_key, spk_idx, frames = batch
        # print(targets.shape) # [2, 2, 32000]
        # print(text1)
        # print(text2)
    
        # est_stft, est_sources, a, b, mask_b, c, mask_c, d, mask_d, logits = self(mixtures, mouth)
        est_stft, est_sources, logits = self(mixtures, mouth, frames)

        # print(y_text1)
        # print(y_text2)
        # wer_score1_v_emb = wer(text1[0], y_text1)  # 效果不行，不兼容，可以删掉
        # wer_score2_v_emb = wer(text2[0], y_text2)  # 同样

        # wer_score1_av = wer(text1[0], a1)
        # wer_score2_av = wer(text2[0], a2)
        # print(est_stft.shape)  # torch.Size([4, 2, 2, 251, 129])
        if targets.ndim == 2:
            targets = targets.unsqueeze(1)
            
        loss1 = self.loss_func["train"](est_sources, targets) # 更在乎可懂度，但是频谱可能会有微小的误差，也就是频谱的局部误差容忍度高
        loss2 = pit_lmag_loss(est_stft, targets_stft)  # 这里应该把实部和虚部分别求L1loss，这样能显示的逼近幅度和相位的关系
        # 约束逐点误差，但是忽略听觉的关键音素，也就是优化了无关紧要的频谱细节，但是忽略了语音可懂度和质量
        # loss3 = compute_LMag_mask(b, a, mask=mask_b) + compute_LMag_mask(c, a, mask=mask_c) + compute_LMag_mask(d, a, mask=mask_d)

        # loss = 0.9*loss2 + 0.1*loss1
        # loss = 100*loss2 + loss1
        loss = loss1 + loss2

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "loss1",
            loss1,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "loss2",
            loss2,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        # self.log(
        #     "a",
        #     a1,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        #     logger=True,
        # )
        # self.log(
        #     "b",
        #     a2,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        #     logger=True,
        # )
        # self.log(
        #     "wer_score1_v_emb_tr",
        #     wer_score1_v_emb,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        #     logger=True,
        # )
        # self.log(
        #     "wer_score2_v_emb_tr",
        #     wer_score2_v_emb,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        #     logger=True,
        # )
        # self.log(
        #     "wer_score1_av_tr",
        #     wer_score1_av,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        #     logger=True,
        # )
        # self.log(
        #     "wer_score2_av_tr",
        #     wer_score2_av,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        #     logger=True,
        # )
        # self.log(
        #     "correct",
        #     correct,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        #     logger=True,
        # )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb, dataloader_idx):
        # cal val loss
        if dataloader_idx == 0:
            mixtures, targets, targets_stft, mouth, frames = batch
            # mixtures, targets, mouth = batch
            # est_stft, est_sources, a, b, mask_b, c, mask_c, d, mask_d, logits = self(mixtures, mouth)
            est_stft, est_sources, _ = self(mixtures, mouth, frames)   # targets是新加的

            # wer_score1_val_v_emb = wer(text1[0], y_text1)
            # wer_score2_val_v_emb = wer(text2[0], y_text2)

            # wer_score1_val_av = wer(text1[0], a1)
            # wer_score2_val_av = wer(text2[0], a2)
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)
            loss = self.loss_func["val"](est_sources, targets)
            # loss2 = self.l1_loss_module(est_stft, targets_stft)
            self.log(
                "val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            # self.log(
            #     "wer_score1_v_emb_val",
            #     wer_score1_val_v_emb,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            #     logger=True,
            # )
            # self.log(
            #     "wer_score2_v_emb_val",
            #     wer_score2_val_v_emb,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            #     logger=True,
            # )
            # self.log(
            #     "wer_score1_av_val",
            #     wer_score1_val_av,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            #     logger=True,
            # )
            # self.log(
            #     "wer_score2_av_val",
            #     wer_score2_val_av,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            #     logger=True,
            # )

            # self.log(
            #     "val_loss_l1_loss",
            #     loss2,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            #     logger=True,
            # )
            
            self.validation_step_outputs.append(loss)
            
            return {"val_loss": loss}

        # cal test loss
        if (self.trainer.current_epoch) % 10 == 0 and dataloader_idx == 1: ####################################
            mixtures, targets, targets_stft, mouth, frames, _ = batch
            # mixtures, targets, mouth = batch
            # est_stft, est_sources, a, b, mask_b, c, mask_c, d, mask_d, logits = self(mixtures, mouth)
            est_stft, est_sources, _ = self(mixtures, mouth, frames)

            # wer_score1_tt_v_emb = wer(text1[0], y_text1)
            # wer_score2_tt_v_emb = wer(text2[0], y_text2)


            # wer_score1_tt_av = wer(text1[0], a1)
            # wer_score2_tt_av = wer(text2[0], a2)
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)
            tloss = self.loss_func["val"](est_sources, targets)
            self.log(
                "test_loss",
                tloss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            # self.log(
            #     "wer_score1_tt_v_emb",
            #     wer_score1_tt_v_emb,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            #     logger=True,
            # )
            # self.log(
            #     "wer_score2_tt_v_emb",
            #     wer_score2_tt_v_emb,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            #     logger=True,
            # )
            # self.log(
            #     "wer_score1_tt_av",
            #     wer_score1_tt_av,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            #     logger=True,
            # )
            # self.log(
            #     "wer_score2_tt_av",
            #     wer_score2_tt_av,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            #     logger=True,
            # )
            self.test_step_outputs.append(tloss)
            return {"test_loss": tloss}

    def on_validation_epoch_end(self):
        # val
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # self.logger.experiment.log(
        #     {"learning_rate": self.optimizer.param_groups[0]["lr"], "epoch": self.current_epoch}
        # ) 
        # self.logger.experiment.log(
        #     {"val_pit_sisnr": -val_loss, "epoch": self.current_epoch}
        # )
        self.log('learning_rate', self.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('epoch', self.current_epoch, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pit_sisnr', -val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # test
        if (self.trainer.current_epoch) % 10 == 0:
            avg_loss = torch.stack(self.test_step_outputs).mean()
            test_loss = torch.mean(self.all_gather(avg_loss))
            # self.logger.experiment.log(
            #     {"test_pit_sisnr": -test_loss, "epoch": self.current_epoch}
            # )
            self.log('val_pit_sisnr', -val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('epoch', self.current_epoch, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()  # free memory
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return [self.val_loader, self.test_loader]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        # dic = flatten_dict(dic)
        # for k, v in dic.items():
        #     print(v)
        #     if v is None:
        #         dic[k] = str(v)
        #     elif isinstance(v, (list, tuple)):
        #         dic[k] = torch.tensor(v)
        # return dic

        dic = flatten_dict(dic)  # 假设 flatten_dict 用于展平嵌套字典
        for k, v in dic.items():
            # print(v)
            if v is None:
                dic[k] = str(v)  # 将 None 转换为字符串 "None"
            elif isinstance(v, (list, tuple)):
                if all(isinstance(i, (int, float, bool)) for i in v):  # 如果列表中的所有元素是数值类型
                    dic[k] = torch.tensor(v)  # 转换为 tensor
                else:
                    dic[k] = v  # 如果包含非数值（如字符串），保留原始列表
        return dic
