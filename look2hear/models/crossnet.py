import torch
import torch.nn as nn
import torch.nn.functional as FF
from pdb import set_trace as stx
import numbers
import inspect
from einops import rearrange
from torchsummary import summary
from torch.autograd import Variable
from typing import Optional
from collections import OrderedDict
from typing import Tuple
# from .lip_encoder import LipEncoderClassifier

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BaseEncoder(nn.Module):
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

        spec = torch.stack([spec.real, spec.imag], 1).transpose(2, 3).contiguous()  # B, 2, T, F
        return spec

class BaseDecoder(nn.Module):
    def pad_to_input_length(self, separated_audio, input_frames):
        output_frames = separated_audio.shape[-1]
        return nn.functional.pad(separated_audio, [0, input_frames - output_frames])

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args

class STFTDecoder(BaseDecoder):
    def __init__(
        self,
        win: int,
        hop_length: int,
        in_chan: int,
        n_src: int,
        kernel_size: int = -1,
        stride: int = 1,
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(STFTDecoder, self).__init__()
        self.win = win
        self.hop_length = hop_length
        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.stride = stride
        self.bias = bias
        # mask_conv = nn.Conv2d(in_chan, 128, 1)
        # self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # self.mask_nl_class = nn.ReLU()

        if self.kernel_size > 0:
            self.decoder = nn.ConvTranspose2d(
                in_channels=self.in_chan,
                out_channels=2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            )
            torch.nn.init.xavier_uniform_(self.decoder.weight)
        else:
            self.decoder = nn.Identity()

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x, input_shape):
        # print(x.shape)
        # B, n_src, N, T, F
        batch_size, length = input_shape.shape[0], input_shape.shape[-1]
        x = x.view(batch_size * self.n_src, self.in_chan, x.shape[-2], x.shape[-1])  # B, n_src, N, T, F -> # B * n_src, N, T, F

        # decoded_separated_audio = self.decoder(d.view(batch_size * self.n_src, 128, d.shape[-2], d.shape[-1]))  # B * n_src, N, T, F - > B * n_src, 2, T, F
        decoded_separated_audio = self.decoder(x)
        # print(decoded_separated_audio.shape)
        
        spec = torch.complex(decoded_separated_audio[:, 0], decoded_separated_audio[:, 1])  # B*n_src, T, F
        # spec = torch.stack([spec.real, spec.imag], dim=-1)  # B*n_src, T, F
        spec = spec.transpose(1, 2).contiguous()  # B*n_src, F, T

        output = torch.istft(
            spec,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=length,
        )  # B*n_src, L

        output = output.view(batch_size, self.n_src, length)  # B, n_src, L

        return output

class STFTDecoder_1(BaseDecoder):
    def __init__(
        self,
        win: int,
        hop_length: int,
        in_chan: int,
        n_src: int,
        kernel_size: int = -1,
        stride: int = 1,
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(STFTDecoder_1, self).__init__()
        self.win = win
        self.hop_length = hop_length
        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.stride = stride
        self.bias = bias

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x, input_shape):
        # # print(x.shape)
        # # B, n_src, N, T, F
        batch_size, length = input_shape.shape[0], input_shape.shape[-1]
        # spec = torch.complex(x[:, 0], x[:, 1])
        spec = torch.complex(x[:, 0].float(), x[:, 1].float())
        spec = spec.transpose(1, 2).contiguous()  # B*n_src, F, T

        output = torch.istft(
            spec,
            n_fft=self.win,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            length=length,
        )  # B*n_src, L

        output = output.view(batch_size, self.n_src, length)  # B, n_src, L

        return output

def _get_activation_fn(activation):
    if activation == "relu":
        return FF.relu
    elif activation == "gelu":
        return FF.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class SingleBlock(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, nhead, dropout=0.1, bidirectional=True, eps=1e-8):
        super(SingleBlock, self).__init__()

        self.feature_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_direction = int(bidirectional) + 1

        self.att_freq = nn.MultiheadAttention(self.feature_dim, self.nhead, batch_first=True)
        self.norm_att_freq = nn.LayerNorm(self.feature_dim, eps=eps)
        self.rnn_freq = getattr(nn, rnn_type)(self.feature_dim, self.hidden_dim, 1, dropout=dropout,
                                         batch_first=True, bidirectional=bidirectional)
        self.fc_freq = nn.Linear(self.hidden_dim * self.num_direction, self.feature_dim)
        self.norm_freq = nn.LayerNorm(self.feature_dim, eps=eps)

        self.att_time = nn.MultiheadAttention(self.feature_dim, self.nhead, batch_first=True)
        self.norm_att_time = nn.LayerNorm(self.feature_dim, eps=eps)
        self.rnn_time = getattr(nn, rnn_type)(self.feature_dim, self.hidden_dim, 1, dropout=dropout,
                                              batch_first=True, bidirectional=bidirectional)
        self.fc_time = nn.Linear(self.hidden_dim * self.num_direction, self.feature_dim)
        self.norm_time = nn.LayerNorm(self.feature_dim, eps=eps)

    def forward(self, input):
        # input: (Batch, Channel, Freq, Time)
        input = input.permute(0, 1, 3, 2).contiguous()
        batch_size, channel, freq, time = input.size()

        input_freq = input.permute(0, 3, 2, 1).contiguous().view(batch_size * time, freq, -1)
        att_out_freq, _ = self.att_freq(input_freq, input_freq, input_freq)
        att_out_freq = att_out_freq + input_freq
        att_out_freq = self.norm_att_freq(att_out_freq)
        rnn_out_freq, _ = self.rnn_freq(att_out_freq)
        fc_out_freq = self.fc_freq(rnn_out_freq)
        fc_out_freq = fc_out_freq + att_out_freq
        fc_out_freq = self.norm_freq(fc_out_freq)
        fc_out_freq = fc_out_freq.contiguous().view(batch_size, time, freq, -1).permute(0, 3, 2, 1)

        input_time = fc_out_freq.permute(0, 2, 3, 1).contiguous().view(batch_size * freq, time, -1)
        att_out_time, _ = self.att_time(input_time, input_time, input_time)
        att_out_time = att_out_time + input_time
        att_out_time = self.norm_att_time(att_out_time)
        rnn_out_time, _ = self.rnn_time(att_out_time)
        fc_out_time = self.fc_time(rnn_out_time)
        fc_out_time = fc_out_time + att_out_time
        fc_out_time = self.norm_time(fc_out_time)
        fc_out_time = fc_out_time.contiguous().view(batch_size, freq, time, -1).permute(0, 3, 1, 2)

        fc_out_time = fc_out_time.permute(0, 1, 3, 2).contiguous()

        return fc_out_time

class Extractor(nn.Module):
    def __init__(self, in_channels, out1, out_channels, hidden_dim, num_blocks, block_type='mDPTNet'):
        super(Extractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.block_type = block_type
        

        # self.ln = nn.LayerNorm(in_channels)
        self.ln = nn.GroupNorm(1, self.in_channels, eps=1e-8)
        self.conv1 = nn.Conv2d(in_channels, out1, kernel_size=1)
        # self.relu1 = nn.ReLU()

        if block_type == 'mDPTNet':
            self.blocks = nn.ModuleList([SingleBlock('LSTM', out1, hidden_dim, 4) for _ in range(num_blocks)])

        self.conv2 = nn.Conv2d(out1, out_channels, kernel_size=1)
        # self.relu2 = nn.ReLU()

    def forward(self, x):

        x = self.ln(x)
        x = self.conv1(x)
        # x = self.relu1(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv2(x)
        # x = self.relu2(x)

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class multi_OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=2, embed_dim=64, bias=False):
        super(multi_OverlapPatchEmbed, self).__init__()

        # self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_1x1 = nn.Conv2d(in_c, embed_dim, kernel_size=1, dilation=1, padding=0)
        self.conv_3x3_d1 = nn.Conv2d(in_c, embed_dim, kernel_size=3, dilation=1, padding=1)
        self.conv_3x3_d2 = nn.Conv2d(in_c, embed_dim, kernel_size=3, dilation=2, padding=2)
        self.conv_3x3_d3 = nn.Conv2d(in_c, embed_dim, kernel_size=3, dilation=3, padding=3)

    def forward(self, x):
        out1 = self.conv_1x1(x)
        out2 = self.conv_3x3_d1(x)
        out3 = self.conv_3x3_d2(x)
        out4 = self.conv_3x3_d3(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)  # 256
        return out


class multi_OverlapPatchEmbed_v(nn.Module):
    def __init__(self, in_c=512, embed_dim=64, bias=False):
        super(multi_OverlapPatchEmbed_v, self).__init__()

        # self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_1x1 = nn.Conv1d(in_c, embed_dim, kernel_size=1, dilation=1, padding=0)
        self.conv_3x3_d1 = nn.Conv1d(in_c, embed_dim, kernel_size=3, dilation=1, padding=1)
        self.conv_3x3_d2 = nn.Conv1d(in_c, embed_dim, kernel_size=3, dilation=2, padding=2)
        self.conv_3x3_d3 = nn.Conv1d(in_c, embed_dim, kernel_size=3, dilation=3, padding=3)

    def forward(self, x):
        out1 = self.conv_1x1(x)
        out2 = self.conv_3x3_d1(x)
        out3 = self.conv_3x3_d2(x)
        out4 = self.conv_3x3_d3(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)  # 256
        return out
##########################################################################

class MultiRangeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels_each):
        super(MultiRangeConv2d, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels_each, kernel_size=1, dilation=1, padding=0)
        self.conv_3x3_d1 = nn.Conv2d(in_channels, out_channels_each, kernel_size=3, dilation=1, padding=1)
        self.conv_3x3_d2 = nn.Conv2d(in_channels, out_channels_each, kernel_size=3, dilation=2, padding=2)
        self.conv_3x3_d3 = nn.Conv2d(in_channels, out_channels_each, kernel_size=3, dilation=3, padding=3)

    def forward(self, x):
        out1 = self.conv_1x1(x)
        out2 = self.conv_3x3_d1(x)
        out3 = self.conv_3x3_d2(x)
        out4 = self.conv_3x3_d3(x)

        out = torch.cat([out1, out2, out3, out4], dim=1)  # concatenate on channel dim
        return out


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):  # 类似于通道选择性，SENet可以换一下
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)

class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)


class MuGIBlock(nn.Module):
    def __init__(self, c, shared_b=False):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        # print(inp_l.shape, inp_r.shape)
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r


class HubertTimeMask2D(nn.Module):
    def __init__(self, channels, mask_prob=0.065, mask_length=10):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_embedding = nn.Parameter(torch.randn(1, channels, 1, 1))  # learnable token

    def compute_mask_indices(self, B, T, device):
        # 返回形状 (B, T)，表示每个 batch 中时间维度哪些位置被 mask
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            num_mask = int(self.mask_prob * T / self.mask_length)
            attempts = 0
            mask_indices = []
            while len(mask_indices) < num_mask and attempts < T:
                start = torch.randint(0, T - self.mask_length + 1, (1,), device=device).item()
                if all(not (start <= idx < start + self.mask_length) for idx in mask_indices):
                    mask_indices.extend(range(start, start + self.mask_length))
                attempts += 1
            mask[b, mask_indices[:T]] = True
        return mask  # (B, T)

    def forward(self, x):
        # x: (B, C, T, F)
        B, C, T, F = x.size()
        mask = self.compute_mask_indices(B, T, x.device)  # (B, T)
        mask = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1) => broadcast 到 (B, C, T, F)
        mask_embed = self.mask_embedding.expand(B, C, T, F)
        x_masked = torch.where(mask, mask_embed, x)
        return x_masked, mask



from typing import *
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter, init
from einops import rearrange
from einops.layers.torch import Rearrange
import math 
import difflib
from torch.nn import *
import math
import numpy as np

class LayerNorm(nn.LayerNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        # """
        # Arg s:
        #     seq_last (bool): whether the sequence dim is the last dim
        # """
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)  # [B, H, Seq] -> [B, Seq, H], or [B,H,w,h] -> [B,h,w,H]
        o = super().forward(input)
        if self.seq_last:
            o = o.transpose(-1, 1)
        return o


class GlobalLayerNorm(nn.Module):

    def __init__(self, dim_hidden: int, seq_last: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.seq_last = seq_last
        self.eps = eps

        if seq_last:
            self.weight = Parameter(torch.empty([dim_hidden, 1]))
            self.bias = Parameter(torch.empty([dim_hidden, 1]))
        else:
            self.weight = Parameter(torch.empty([dim_hidden]))
            self.bias = Parameter(torch.empty([dim_hidden]))
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # """
        # Args:
        #     input (Tensor): shape [B, Seq, H] or [B, H, Seq]
        # """
        var, mean = torch.var_mean(input, dim=(1, 2), unbiased=False, keepdim=True)

        output = (input - mean) / torch.sqrt(var + self.eps)
        output = output * self.weight + self.bias
        return output

    def extra_repr(self) -> str:
        return '{dim_hidden}, seq_last={seq_last}, eps={eps}'.format(**self.__dict__)


class BatchNorm1d(nn.Module):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__()
        self.seq_last = seq_last
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if not self.seq_last:
            input = input.transpose(-1, -2)  # [B, Seq, H] -> [B, H, Seq]
        o = self.bn.forward(input)  # accepts [B, H, Seq]
        if not self.seq_last:
            o = o.transpose(-1, -2)
        return o


class GroupNorm(nn.GroupNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last == False:
            input = input.transpose(-1, -2)  # [B, Seq, H] -> [B, H, Seq]
        o = super().forward(input)  # accepts [B, H, Seq]
        if self.seq_last == False:
            o = o.transpose(-1, -2)
        return o


class GroupBatchNorm(nn.Module):
    # """Applies Group Batch Normalization over a group of inputs

    # see: `Changsheng Quan, Xiaofei Li. NBC2: Multichannel Speech Separation with Revised Narrow-band Conformer. arXiv:2212.02076.`

    # """

    dim_hidden: int
    group_size: int
    eps: float
    affine: bool
    seq_last: bool
    share_along_sequence_dim: bool

    def __init__(
        self,
        dim_hidden: int,
        group_size: Optional[int],
        share_along_sequence_dim: bool = False,
        seq_last: bool = False,
        affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        # """
        # Args:
        #     dim_hidden (int): hidden dimension
        #     group_size (int): the size of group, optional
        #     share_along_sequence_dim (bool): share statistics along the sequence dimension. Defaults to False.
        #     seq_last (bool): whether the shape of input is [B, Seq, H] or [B, H, Seq]. Defaults to False, i.e. [B, Seq, H].
        #     affine (bool): affine transformation. Defaults to True.
        #     eps (float): Defaults to 1e-5.
        # """
        super(GroupBatchNorm, self).__init__()

        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.seq_last = seq_last
        self.share_along_sequence_dim = share_along_sequence_dim
        if self.affine:
            if seq_last:
                self.weight = Parameter(torch.empty([dim_hidden, 1]))
                self.bias = Parameter(torch.empty([dim_hidden, 1]))
            else:
                self.weight = Parameter(torch.empty([dim_hidden]))
                self.bias = Parameter(torch.empty([dim_hidden]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: Tensor, group_size: int = None) -> Tensor:
        """
        Args:
            x: shape [B, Seq, H] if seq_last=False, else shape [B, H, Seq] , where B = num of groups * group size.
            group_size: the size of one group. if not given anywhere, the input must be 4-dim tensor with shape [B, group_size, Seq, H] or [B, group_size, H, Seq]
        """
        if self.group_size != None:
            assert group_size == None or group_size == self.group_size, (group_size, self.group_size)
            group_size = self.group_size

        if group_size is not None:
            assert (x.shape[0] // group_size) * group_size, f'batch size {x.shape[0]} is not divisible by group size {group_size}'

        original_shape = x.shape
        if self.seq_last == False:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, Seq, H = x.shape
            else:
                B, Seq, H = x.shape
                x = x.reshape(B // group_size, group_size, Seq, H)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 3), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)
        else:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, H, Seq = x.shape
            else:
                B, H, Seq = x.shape
                x = x.reshape(B // group_size, group_size, H, Seq)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 2), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)

        return output

    def extra_repr(self) -> str:
        return '{dim_hidden}, {group_size}, share_along_sequence_dim={share_along_sequence_dim}, seq_last={seq_last}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)



def new_norm(norm_type: str, dim_hidden: int, seq_last: bool, group_size: int = None, num_groups: int = None) -> nn.Module:
    if norm_type.upper() == 'LN':
        norm = LayerNorm(normalized_shape=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == 'GBN':
        norm = GroupBatchNorm(dim_hidden=dim_hidden, seq_last=seq_last, group_size=group_size, share_along_sequence_dim=False)
    elif norm_type == 'GBNShare':
        norm = GroupBatchNorm(dim_hidden=dim_hidden, seq_last=seq_last, group_size=group_size, share_along_sequence_dim=True)
    elif norm_type.upper() == 'BN':
        norm = BatchNorm1d(num_features=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == 'GN':
        norm = GroupNorm(num_groups=num_groups, num_channels=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == 'GLN':
        norm = GlobalLayerNorm(dim_hidden, seq_last=seq_last)
    else:
        raise Exception(f"Unknown norm type: {norm_type}")
    return norm

def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler

class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        with torch.autocast(device_type = "cuda", enabled = False):
            if x.ndim == 4:
                _, C, _, _ = x.shape
                stat_dim = (1,)
            else:
                raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
            mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
            std_ = torch.sqrt(torch.clamp(x.var(dim=stat_dim, unbiased=False, keepdim=True), self.eps))  # [B,1,T,F]
            x_hat = (x - mu_) / (std_ )
                
            x_hat = x_hat * self.gamma + self.beta

            return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        with torch.autocast(device_type = "cuda", enabled = False):
            
            if x.ndim == 4:
                stat_dim = (1, 3)
            else:
                raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
            mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
            std_ = torch.sqrt(torch.clamp(x.var(dim=stat_dim, unbiased=False, keepdim=True), self.eps))  # [B,1,T,F]
            x_hat = (x - mu_) / (std_)
            
            x_hat = x_hat * self.gamma + self.beta
            
            
            return x_hat

class GlobalMultiheadSlefAttentionModule(nn.Module):
    def __init__(self,
                 embed_dim: int = 64,
                 n_freqs: int = 129,
                 num_heads: int = 4,
                 approx_qk_dim: int = 512,
                 activation: int = "prelu",
                 eps=1e-5,
                 **kwargs

                 ) -> None:

        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.in_reshape = Rearrange("B F T C -> B C T F")
        
        
        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert embed_dim % num_heads == 0
        
        self.attn_conv_Q = nn.Sequential(
                    nn.Conv2d(embed_dim , num_heads*E, 1),
                    get_layer(activation)(),
                    Rearrange("B (num_heads E) T F -> (num_heads B) E T F", num_heads = num_heads),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                    Rearrange("Bh C T F -> Bh T (C F)"),
                )
        
        self.attn_conv_K = nn.Sequential(
                    nn.Conv2d(embed_dim , num_heads*E, 1),
                    get_layer(activation)(),
                    Rearrange("B (num_heads E) T F -> (num_heads B) E T F", num_heads = num_heads),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                    Rearrange("Bh C T F -> Bh T (C F)"),
                )
        
        
        self.attn_conv_V = nn.Sequential(
                    nn.Conv2d(embed_dim , num_heads*(embed_dim // num_heads), 1),
                    get_layer(activation)(),
                    Rearrange("B (num_heads E) T F -> (num_heads B) E T F", num_heads = num_heads),
                    LayerNormalization4DCF((embed_dim // num_heads, n_freqs), eps=eps),
                    Rearrange("hB C T F -> hB T (C F)"),
                )
        
        self.v_reshape = Rearrange("(h B) T (e F) -> B (h e) T F", F = n_freqs, h = self.num_heads)
        
        self.attn_concat_proj = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((embed_dim, n_freqs), eps=eps),
                Rearrange("B C T F -> B F T C")
            )
        
        
        
    def __getitem__(self, key):
        return getattr(self, key)

    def forward(self, x, average_attn_weights=False, attn_mask=None):

        batch_a = self.in_reshape(x)
       
        Q = self.attn_conv_Q(batch_a)  # [Bxh, C, T, F]
        K = self.attn_conv_K(batch_a)  # [Bxh, C, T, F]
        V = self.attn_conv_V(batch_a)  # [Bh, T, CxF]
        
        V = torch.nn.functional.scaled_dot_product_attention(Q,K,V) # [Bxh, T, C*F]
        
        V = self.v_reshape(V)

        batch = self.attn_concat_proj(V)  # [B, C, T, F])

        out = batch + x

        return out, None

class LinearGroup(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups 
        # print(in_features, out_features, num_groups)
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"

class CrossNetLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_ffn: int,
            dim_squeeze: int,
            num_freqs: int,
            num_heads: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full: nn.Module = None,
            gmhsa: bool = True,
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        # print("norms =", norms)
        # print("norms =", norms[3])
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # # narrow-band block
        # # MHSA module
        # self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        # self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        # self.dropout_mhsa = nn.Dropout(dropout[0])
        
        # cross multihead self attention
        self.gmhsa = gmhsa 
        if self.gmhsa:
            self.norm_crossmhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
            self.global_mhsa = GlobalMultiheadSlefAttentionModule(embed_dim=dim_hidden, num_heads=num_heads, n_freqs= num_freqs, batch_first=True)
            self.dropout_crossmhsa = nn.Dropout(dropout[0])
        
        
        # T-ConvFFN module
        self.tconvffn = nn.ModuleList([
            new_norm(norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            new_norm(norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
        ])
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """

        x = x + self._fconv(self.fconv1, x)
        
        x = x + self._full(x)
        
        x = x + self._fconv(self.fconv2, x)
        
        if self.gmhsa:
            x, attn = self._csa(x, att_mask)
        else:
            attn = None

        x = x + self._tconvffn(x)
        
        return x, attn

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(B * F, T, H)
        need_weights = False
        x, attn = self.mhsa(x, x, x, average_attn_weights=False, attn_mask=attn_mask, need_weights = need_weights)
        x = x.reshape(B, F, T, H).contiguous()
        return self.dropout_mhsa(x)
    
    def _csa(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        
        B, F, T, H = x.shape
        x = self.norm_crossmhsa(x)
        
        
        # x = rearrange(x, "B F T H -> (B H) T F")
        x, attn = self.global_mhsa(x, average_attn_weights=False, attn_mask=attn_mask)
        # x = rearrange(x, "(B H) T F -> B F T H", F = F)
        return self.dropout_crossmhsa(x), attn

    def _tconvffn(self, x: Tensor) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2).contiguous()  # [B,F,H,T]
        x = x.reshape(B * F, H0, T).contiguous()
        for m in self.tconvffn:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=F)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T).contiguous()
        x = x.transpose(-1, -2).contiguous()  # [B,F,T,H]
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B,T,H,F]
        x = x.reshape(B * T, H, F).contiguous()
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F).contiguous()
            x = x.transpose(1, 3).contiguous()  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3).contiguous()  # [B,T,H',F]
            x = x.reshape(B * T, -1, F).contiguous()

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"

class crossnet(nn.Module):
    def __init__(self, 
        num_layers = 8,
        dim_squeeze = 4,
        dim_hidden = 192,
        dim_ffn = 384,
        num_heads = 4,
        num_freqs = 129,
        dropout = (0, 0, 0),
        kernel_size = (5, 3),
        conv_groups = (8, 8),
        norms = ("LN", "LN", "GN", "LN", "LN", "LN"),
        padding = 'zeros',
        full_share = 0, 
        gmhsa = True,
        win = 256,
        hop_length = 128,
        inp_channels=2, 
        out_channels=2, 
        dim = 64,
        bias = False,
        vpre_channels=512,
        num_source = 1,
        
    ):

        super(crossnet, self).__init__()
        self.num_source = num_source
        self.win = win
        self.hop_length = hop_length

        self.pre_v = multi_OverlapPatchEmbed_v(vpre_channels, dim)  # 256

        self.stft_encoder = STFTEncoder(win, hop_length, bias)

        self.patch_embed = multi_OverlapPatchEmbed(inp_channels, dim)  # 256

        # self.masker_10 = HubertTimeMask2D(channels=dim*4, mask_prob=0.01, mask_length=10) # x 是 shape (B, C, T, F)  # 10% mask
        # self.masker_30 = HubertTimeMask2D(channels=dim*4, mask_prob=0.03, mask_length=10) # x 是 shape (B, C, T, F)  # 30% mask
        self.masker_45 = HubertTimeMask2D(channels=dim*4, mask_prob=0.3, mask_length=10) # x 是 shape (B, C, T, F)  # 45% mask

        self.blocks_inter = DualStreamSeq(*[MuGIBlock(dim*4) for _ in range(2)])
        # self.blocks_inter_10 = DualStreamSeq(*[MuGIBlock(dim*4) for _ in range(2)])
        # self.blocks_inter_30 = DualStreamSeq(*[MuGIBlock(dim*4) for _ in range(2)])
        self.blocks_inter_45 = DualStreamSeq(*[MuGIBlock(dim*4) for _ in range(2)])
        self.conv_all_v = nn.Conv2d(dim*8, dim*4, kernel_size=1, stride=1, bias=False)

        # model = LipEncoderClassifier(num_speakers=3096)
        # checkpoint = torch.load('/home/xueke/DPT_1d_main/lip_encoder_checkpoint1/best_model_epoch_67.pt')
        # model.load_state_dict(checkpoint['model'])

        # for p in model.parameters():
        #     p.requires_grad = False

        # self.conv = nn.Conv2d(dim*8, dim*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv = nn.Conv2d(dim*8, dim*3, kernel_size=3, stride=1,padding=1, bias=False)
        layers = []
        for l in range(num_layers):
            layer = CrossNetLayer(
                dim_hidden=dim_hidden,
                dim_ffn=dim_ffn,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
                gmhsa=gmhsa,
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        if self.num_source == 1:
            self.decoder = nn.Linear(in_features=dim_hidden, out_features=out_channels)

            self.stft_decoder = STFTDecoder_1(win, hop_length, in_chan=dim, n_src=num_source, kernel_size=3, stride=1, bias=bias)
        elif self.num_source == 2:
            self.output1 = nn.Conv2d(int(dim**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
            self.output2 = nn.Conv2d(int(dim**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
            self.stft_decoder12 = STFTDecoder(win, hop_length, in_chan=dim, n_src=num_source, kernel_size=3, stride=1, bias=bias)
        else:
            raise ValueError("Unsupported number of sources: {}".format(self.num_source))

    def forward(self, x, v):

        # lip_embedding
        # logits, emb = model(mouth)  # B * f *3*  112*112  这个mouth形状到底应该是什么样的，还需斟酌,但是如果用修改后的数据的话应该是120，且是灰度图，不是彩色图，
        # emb （B*256）


        v0 = self.pre_v(v)  # torch.Size([4, 256, 251, 129])   torch.Size([4, 256, 50])

        feature_map = self.stft_encoder(x)  # torch.Size([4, 2, 251, 129])
        B, C, T, F = feature_map.size()
        inp_enc_level1 = self.patch_embed(feature_map) # torch.Size([4, 256, 251, 129])

        v0 = FF.interpolate(v0, size=T, mode='linear', align_corners=True)  # [B, C, 251]
        v0 = v0.unsqueeze(-1)  # [B, C, 251, 1]
        v0 = v0.repeat(1, 1, 1, F)  # [B, 256, 251, 129]

        # mask
        # inp_enc_level1_mask_10 = self.masker_10(inp_enc_level1)
        # inp_enc_level1_mask_30 = self.masker_30(inp_enc_level1)
        inp_enc_level1_mask_45, mask = self.masker_45(inp_enc_level1)

        # inp_enc_level1_10, v0_10 = self.blocks_inter_10(inp_enc_level1_mask_10, v0)
        # inp_enc_level1_30, v0_30 = self.blocks_inter_30(inp_enc_level1_mask_30, v0)
        inp_enc_level1_45, v0_45 = self.blocks_inter_45(inp_enc_level1_mask_45, v0)

        inp_enc_level1_pre, v0_pre = self.blocks_inter(inp_enc_level1, v0)

        # inp_enc_level_all = torch.cat([inp_enc_level1_pre, inp_enc_level1_10, inp_enc_level1_30, inp_enc_level1_45], dim=1)
        v0_all = torch.cat([v0_pre, v0_45], dim=1)
        # inp_enc_level_all = self.conv_all_a(inp_enc_level_all)
        v0_all = self.conv_all_v(v0_all)

        # inp_enc_level_all = inp_enc_level1_pre + inp_enc_level1_10 + inp_enc_level1_30 + inp_enc_level1_45 + global_a
        # # inp_enc_level_all = inp_enc_level_all + self.gate(inp_enc_level_all)
        # v0_all = v0_pre + v0_10 + v0_30 + v0_45 + global_v
        # # v0_all = v0_all + self.gate(v0_all)
        inp_enc_level_all = inp_enc_level1_pre
        # v0_all = v0_pre

        xx =  torch.nn.functional.relu(self.conv(torch.cat([inp_enc_level_all, v0_all], dim=1)))
        xx = xx.permute(0, 3, 2, 1).contiguous()  # B, F, T, H
        # print(xx.shape)  # torch.Size([2, 129, 251, 192])

        # # 假设 freq_feat: [B, F, T, C]
        # B, F, T, C = xx.shape  # 比如 [2, 80, 100, 1]
        # lip_emb = emb.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 256]
        # lip_emb = lip_emb.expand(-1, F, T, -1)                 # [B, F, T, 256]
        # # 拼接特征
        # # fused = torch.cat([freq_feat, emb], dim=-1)   # [B, F, T, C+256]

        for m in self.layers:
            xx, attn = m(xx)
        y = self.decoder(xx)
        # print(y.shape)
        y = y.permute(0, 3, 2, 1).contiguous() # B,H,T, F
        stft_out =  y
        source = self.stft_decoder(y, x)
        # print(inp_enc_level1_pre)
        # print(inp_enc_level1_45)
        # loss_all = compute_LMag(inp_enc_level1_45, inp_enc_level1_pre, mask=mask, weight=5.0)
        # print(loss_all) 

        return stft_out, source, inp_enc_level1_pre, inp_enc_level1_45, mask



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = crossnet().to(device)
    # print(model)
    input_signal = torch.randn(2,1,32000).to(device)  # 输入图像的形状 (C, H, W)
    v = torch.randn(2,512,50).to(device)
    # summary(model, input_signal, device='cuda')
    import time
    start_time = time.time()
    out1,out2, a,b, mask = model(input_signal, v)
    end_time = time.time()
    print(out1.shape)
    print(out2.shape)

    # 输出信息
    print("模型参数总数:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("输出形状:", out1.shape)
    print(f"前向传播耗时: {end_time - start_time:.4f} 秒")
