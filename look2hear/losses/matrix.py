###
# Author: Kai Li
# Date: 2021-06-09 16:43:09
# LastEditors: Please set LastEditors
# LastEditTime: 2021-12-03 17:52:13
###

import torch
from torch.nn.modules.loss import _Loss
from torch import nn as nn
from torch.nn import functional as F
import numpy as np



class PairwiseNegSDR(_Loss):
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, ests, targets):
        if targets.size() != ests.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead"
            )
        assert targets.size() == ests.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(ests, dim=2, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(ests, dim=2)
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -pair_wise_sdr


class SingleSrcNegSDR(_Loss):
    def __init__(
        self, sdr_type, zero_mean=True, take_log=True, reduction="none", EPS=1e-8
    ):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, ests, targets):
        if targets.size() != ests.size() or targets.ndim != 2:
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {targets.size()} and {ests.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=1, keepdim=True)
            mean_estimate = torch.mean(ests, dim=1, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(ests * targets, dim=1, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(targets ** 2, dim=1, keepdim=True) + self.EPS
            # [batch, time]
            scaled_target = dot * targets / s_target_energy
        else:
            # [batch, time]
            scaled_target = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = ests - targets
        else:
            e_noise = ests - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + self.EPS
        )
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses


class MultiSrcNegSDR(_Loss):
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, ests, targets):
        if targets.size() != ests.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_est = torch.mean(ests, dim=2, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_est
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src]
            pair_wise_dot = torch.sum(ests * targets, dim=2, keepdim=True)
            # [batch, n_src]
            s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + self.EPS
            # [batch, n_src, time]
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            # [batch, n_src, time]
            scaled_targets = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = ests - targets
        else:
            e_noise = ests - scaled_targets
        # [batch, n_src]
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
            torch.sum(e_noise ** 2, dim=2) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -torch.mean(pair_wise_sdr, dim=-1)


_reduction_modes = ['none', 'mean', 'sum']
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

from itertools import permutations

# 定义 L1Loss 类
class pitL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss with PIT support.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(pitL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, n_src, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, n_src, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, n_src, C, H, W). Element-wise
                weights. Default: None.
        """
        # pred = pred.unsqueeze(2)
        # target = target.unsqueeze(2)
        if target.ndim == 4 and pred.ndim == 4:
            target = target.unsqueeze(1)
            pred = pred.unsqueeze(1)
        if target.size() != pred.size() or target.ndim != 5:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, C, H, W], got {target.size()} and {pred.size()} instead"
            )

        batch_size, n_src, C, H, W = target.size()

        # Compute pairwise L1 loss
        pairwise_l1_loss = []
        for i in range(n_src):
            for j in range(n_src):
                loss = l1_loss(pred[:, i], target[:, j])  
                if weight is not None:
                    loss = loss * weight[:, i]
                pairwise_l1_loss.append(loss)

        # Reshape pairwise losses to [batch, n_src, n_src, C, H, W]
        pairwise_l1_loss = torch.stack(pairwise_l1_loss, dim=1).view(batch_size, n_src, n_src, C, H, W)

        # Compute the loss for all possible permutations of sources
        perm_losses = []
        for perm in permutations(range(n_src)):
            perm_loss = pairwise_l1_loss[:, range(n_src), perm].mean(dim=[1, 2, 3, 4])
            perm_losses.append(perm_loss)

        # Choose the permutation with the minimum loss
        min_loss, _ = torch.min(torch.stack(perm_losses, dim=1), dim=1)

        # Apply reduction
        if self.reduction == 'mean':
            min_loss = min_loss.mean()
        elif self.reduction == 'sum':
            min_loss = min_loss.sum()

        return self.loss_weight * min_loss

# aliases
pairwise_neg_sisdr = PairwiseNegSDR("sisdr")
pairwise_neg_sdsdr = PairwiseNegSDR("sdsdr")
pairwise_neg_snr = PairwiseNegSDR("snr")
singlesrc_neg_sisdr = SingleSrcNegSDR("sisdr")
singlesrc_neg_sdsdr = SingleSrcNegSDR("sdsdr")
singlesrc_neg_snr = SingleSrcNegSDR("snr")
multisrc_neg_sisdr = MultiSrcNegSDR("sisdr")
multisrc_neg_sdsdr = MultiSrcNegSDR("sdsdr")
multisrc_neg_snr = MultiSrcNegSDR("snr")
l1_loss_module = pitL1Loss(reduction='mean')
mix_loss  = PairwiseNegSDR("snr")

if __name__ == "__main__":
    # 创建一个示例场景
    batch_size = 2
    n_src = 2
    C = 3
    H = 10
    W = 10
    pred = torch.randn(batch_size, n_src, C, H, W)
    target = torch.randn(batch_size, n_src, C, H, W)
    l1_loss_module = pitL1Loss(loss_weight=1.0, reduction='mean')
    loss = l1_loss_module(pred, target)
    print("L1 Loss:", loss.item())
