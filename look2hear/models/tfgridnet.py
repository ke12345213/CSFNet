import torch
import torch.nn as nn
import torch.nn.functional as FF
from pdb import set_trace as stx
import inspect
from einops import rearrange
from torchsummary import summary
from typing import Optional
from collections import OrderedDict
from typing import Sequence, Tuple, Union, Dict, Optional, Tuple
# from lip_encoder import LipEncoderClassifier
import math
from typing import Dict, List, Optional, Tuple
from torch.nn import init
from torch.nn.parameter import Parameter
from abc import ABC, abstractmethod
import difflib
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor


def is_torch_complex_tensor(c):
    return not isinstance(c, ComplexTensor) and torch.is_complex(c)
def is_complex(c):
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)
EPS = torch.finfo(torch.double).eps

def new_complex_like(
    ref: Union[torch.Tensor, ComplexTensor],
    real_imag: Tuple[torch.Tensor, torch.Tensor],
):
    if isinstance(ref, ComplexTensor):
        return ComplexTensor(*real_imag)
    elif is_torch_complex_tensor(ref):
        return torch.complex(*real_imag)
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )

class AbsSeparator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        raise NotImplementedError

    def forward_streaming(
        self,
        input_frame: torch.Tensor,
        buffer=None,
    ):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        raise NotImplementedError

from packaging.version import parse as V

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

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = FF.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SingleBlock(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, nhead, dropout=0, bidirectional=True, eps=1e-8):
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

        return fc_out_time

class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()

        self.att_freq = nn.MultiheadAttention(emb_dim, n_head, batch_first=True)
        self.norm_att_freq = nn.LayerNorm(emb_dim, eps=eps)
        self.att_time = nn.MultiheadAttention(emb_dim, n_head, batch_first=True)
        self.norm_att_time = nn.LayerNorm(emb_dim, eps=eps)

        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )
        self.FFN = FeedForward(emb_dim, 4, bias=True)  # 64*4  256

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = FF.pad(x, (0, Q - old_Q, 0, T - old_T))

        

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        )  # [BT, C, Q]
        intra_rnn = intra_rnn.transpose(1, 2)
        att_out_freq, _ = self.att_freq(intra_rnn, intra_rnn, intra_rnn)
        att_out_freq = att_out_freq + intra_rnn
        att_out_freq = self.norm_att_freq(att_out_freq)
        intra_rnn = att_out_freq.transpose(1, 2)
        intra_rnn = FF.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BF, C, T]

        inter_rnn = inter_rnn.transpose(1, 2)
        att_out_time, _ = self.att_time(inter_rnn, inter_rnn, inter_rnn)
        att_out_time = att_out_time + inter_rnn
        att_out_time = self.norm_att_time(att_out_time)
        inter_rnn = att_out_time.transpose(1, 2)
        inter_rnn = FF.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # [B, C, T, Q]

        # attention
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = FF.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, old_T, -1]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        out = batch + inter_rnn

        out1 = self.FFN(out)
        out = out1 + out
        return out


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
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
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
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


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
        win_length: int = 512,         # 32ms @16kHz
        hop_length: int = 128,         # 8ms @16kHz
        n_fft: int = 256,              # FFT 点数（决定频率维度）
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(STFTEncoder, self).__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.bias = bias

        hann = torch.hann_window(self.win_length)
        sqrt_hann = torch.sqrt(hann)
        self.register_buffer("window", sqrt_hann, persistent=False)

    def forward(self, x: torch.Tensor):
        x = self.unsqueeze_to_2D(x)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            return_complex=True,
        )

        # B, 2, T, F 格式
        spec = torch.stack([spec.real, spec.imag], 1).transpose(2, 3).contiguous()
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

class STFTDecoder_1(BaseDecoder):
    def __init__(
        self,
        win_length: int = 512,
        hop_length: int = 128,
        n_fft: int = 256,
        in_chan: int = 2,
        n_src: int = 2,
        kernel_size: int = -1,
        stride: int = 1,
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(STFTDecoder_1, self).__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.in_chan = in_chan
        self.n_src = n_src
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2 if self.kernel_size > 0 else 0
        self.stride = stride
        self.bias = bias

        hann = torch.hann_window(self.win_length)
        sqrt_hann = torch.sqrt(hann)
        self.register_buffer("window", sqrt_hann, persistent=False)

    def forward(self, x, input_shape):
        B, n_src, N, T, F = x.shape
        batch_size, length = input_shape.shape[0], input_shape.shape[-1]
        
        x = x.view(B * n_src, N, T, F)  # (B*n_src, 2, T, F)
        spec = torch.complex(x[:, 0], x[:, 1])  # complex spectrum
        spec = spec.transpose(1, 2).contiguous()  # (B*n_src, F, T)

        output = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            length=length,
        )  # (B*n_src, L)

        output = output.view(batch_size, self.n_src, length)  # (B, n_src, L)

        return output


def _get_activation_fn(activation):
    if activation == "relu":
        return FF.relu
    elif activation == "gelu":
        return FF.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


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


class multi_OverlapPatchEmbed_1D(nn.Module):
    def __init__(self, in_dim=2, embed_dim=64, bias=False):
        super(multi_OverlapPatchEmbed_1D, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_dim, embed_dim, kernel_size=1, padding=0, bias=bias)
        self.conv_3_d1 = nn.Conv1d(in_dim, embed_dim, kernel_size=3, padding=1, dilation=1, bias=bias)
        self.conv_3_d2 = nn.Conv1d(in_dim, embed_dim, kernel_size=3, padding=2, dilation=2, bias=bias)
        self.conv_3_d3 = nn.Conv1d(in_dim, embed_dim, kernel_size=3, padding=3, dilation=3, bias=bias)

    def forward(self, x):
        # 输入 [B, C, T, F]
        B, C, T, F = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()         # [B, F, C, T]
        x = x.reshape(B*F, C, T)        # [B*F, C, T]

        out1 = self.conv_1x1(x)
        out2 = self.conv_3_d1(x)
        out3 = self.conv_3_d2(x)
        out4 = self.conv_3_d3(x)

        out = torch.cat([out1, out2, out3, out4], dim=1)  # [B*F, 4*embed_dim, T 
        out = out.permute(0, 2, 1).contiguous()  # [B*F, T, 4*embed_dim 
        out = out.reshape(B, F, T, -1).contiguous()
        out = out.permute(0, 3, 2, 1).contiguous()
        return out


class multi_OverlapPatchEmbed_v(nn.Module):
    def __init__(self, in_c=1024, embed_dim=64, bias=False):
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
         # 保证 weight 和 bias 与 x 在同一个 device
        # weight = self.weight.to(x.device)
        # bias = self.bias.to(x.device)
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEWeightModule(nn.Module):  # SE block
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight

class PSAModule(nn.Module):
    def __init__(self, inplanes, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16], reduction=8):
        super(PSAModule, self).__init__()
        self.split_channel = planes // 4

        self.conv_1 = conv(inplanes, self.split_channel, kernel_size=conv_kernels[0],
                           padding=conv_kernels[0] // 2, stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplanes, self.split_channel, kernel_size=conv_kernels[1],
                           padding=conv_kernels[1] // 2, stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplanes, self.split_channel, kernel_size=conv_kernels[2],
                           padding=conv_kernels[2] // 2, stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplanes, self.split_channel, kernel_size=conv_kernels[3],
                           padding=conv_kernels[3] // 2, stride=stride, groups=conv_groups[3])

        self.se = SEWeightModule(self.split_channel, reduction=8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.size(0)

        # stage 1: multi-scale conv
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.stack([x1, x2, x3, x4], dim=1)  # shape: [B, 4, C//4, H, W]

        # stage 2: SE attention for each path
        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        attn = torch.stack([x1_se, x2_se, x3_se, x4_se], dim=1)  # [B, 4, C//4, 1, 1]
        attn = self.softmax(attn)

        # stage 3: apply attention
        feats_weighted = feats * attn  # [B, 4, C//4, H, W]

        # stage 4: concatenate along channel dimension
        out = torch.cat([feats_weighted[:, i, :, :, :] for i in range(4)], dim=1)

        return out

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 分别对 Q, K, V 做 projection
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim_q, dim_out, kernel_size=1, bias=bias),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=dim_out, bias=bias),
        )
        self.kv_proj = nn.Sequential(
            nn.Conv2d(dim_kv, dim_out * 2, kernel_size=1, bias=bias),
            nn.Conv2d(dim_out * 2, dim_out * 2, kernel_size=3, padding=1, groups=dim_out * 2, bias=bias),
        )

        self.project_out = nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias)

    def forward(self, q_input, kv_input):
        B, _, H, W = q_input.shape

        # 1. 生成 Q、K、V
        q = self.q_proj(q_input)
        kv = self.kv_proj(kv_input)
        k, v = kv.chunk(2, dim=1)

        # 2. reshape 成多头注意力形式
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 3. 归一化
        q = FF.normalize(q, dim=-1)
        k = FF.normalize(k, dim=-1)

        # 4. Cross-Attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v  # [B, heads, C, H*W]

        # 5. reshape 回去
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        # 6. 输出映射
        out = self.project_out(out)
        return out

class global_channel_attention(nn.Module):
    def __init__(self, channels_a=64, channels_v=384, reduction=16):
        super(global_channel_attention, self).__init__()
        self.LN_a  = LayerNorm2d(channels_a)
        self.LN_v  = LayerNorm2d(channels_v)
        self.conv1_a = nn.Conv2d(channels_a, channels_a, kernel_size=1)
        self.conv1_v = nn.Conv2d(channels_v, channels_v, kernel_size=1)
        self.psa_a = PSAModule(channels_a, channels_a, reduction=2)
        self.psa_v = PSAModule(channels_v, channels_v, reduction=reduction)
        self.cross_attention_v = CrossAttention(dim_q=channels_a, dim_kv=channels_v, dim_out=channels_a)
        self.cross_attention_a = CrossAttention(dim_q=channels_v, dim_kv=channels_a, dim_out=channels_v)
        self.conv2_a = nn.Conv2d(channels_a, channels_a, kernel_size=1)
        self.conv2_v = nn.Conv2d(channels_v, channels_v, kernel_size=1)

    def forward(self, x, v):
        # 原始特征克隆
        x1 = x.clone()
        v1 = v.clone()

        # 1. LayerNorm
        x = self.LN_a(x)
        v = self.LN_v(v)

        # 2. 通道变换 + PSA 注意力
        x = self.conv1_a(x)
        x2 = self.psa_a(x)      # 提取后得到 x 的增强特征 x2

        v = self.conv1_v(v)
        v2 = self.psa_v(v)      # 得到 v 的增强特征 v2

        # 3. Cross-Attention：用 v 作为 KV 融合到 x，反之亦然
        x = self.cross_attention_v(x2, v)  # 理论上还可以直接将混合语音和唇语特征
        # 直接分隔开，先让网络提前知道要分离的通道数
        x = x + x2              # 残差连接

        v = self.cross_attention_a(v2, x)
        v = v + v2              # 残差连接

        # 4. 输出映射 & 总残差连接
        x = self.conv2_a(x)
        x_out = x + x1

        v = self.conv2_v(v)
        v_out = v + v1
        return x_out, v_out



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

class tfgridnet(nn.Module):
    def __init__(self, 
        num_layers = 6,  # 12
        win = 256,
        hop_length = 128,
        n_fft = 256,
        inp_channels=2, 
        out_channels=2, 
        dim = 48,
        bias = False,
        vpre_channels=512,
        num_source = 2,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=64,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        
    ):

        super(tfgridnet, self).__init__()
        self.num_source = num_source
        self.win = win
        self.hop_length = hop_length
        self.num_layers = num_layers
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

        self.pre_v = multi_OverlapPatchEmbed_v(vpre_channels, dim)  # 192
        self.stft_encoder = STFTEncoder(win, hop_length, n_fft, bias)
        # self.patch_embed = multi_OverlapPatchEmbed(inp_channels, dim)  # 192  #
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(inp_channels, emb_dim, ks, padding=padding),  # 48
            nn.GroupNorm(1, emb_dim, eps=eps),  # 48
        ) 
        self.conv1 = nn.Sequential(
            nn.Conv2d(emb_dim+dim*8, emb_dim, ks, padding=padding),  # 48
            nn.GroupNorm(1, emb_dim, eps=eps),  # 48
        )
        # self.conv_v0 = nn.Conv2d(dim*8, dim*4, kernel_size=1)

        # self.lipencoder = LipEncoderClassifier(num_speakers=2936, tcn_channels=256, lip_emb_dim=256)
        # state_dict = torch.load('/home/xueke/wav2vec_TCN/lip_encoder_checkpoint1/best_model_epoch_58.pt', map_location='cuda', weights_only=True)  # 或 'cuda'
        # # self.lipencoder.load_state_dict(state_dict)
        # self.lipencoder.load_state_dict(state_dict, strict=False)
        # for param in self.lipencoder.parameters():
        #     param.requires_grad = True
        # self.classifier = nn.Linear(256, 2936)
        # self.conv_v0_lip0 = nn.Conv1d(dim*8, dim*3, kernel_size=1)
        # self.conv_v0_lip1 = nn.Conv1d(dim*8, dim*3, kernel_size=1)


        # self.masker_10 = HubertTimeMask2D(channels=dim*4, mask_prob=0.15, mask_length=10) # x 是 shape (B, C, T, F)  # 10% mask
        # self.masker_30 = HubertTimeMask2D(channels=dim*4, mask_prob=0.3, mask_length=10) # x 是 shape (B, C, T, F)  # 30% mask
        # self.masker_45 = HubertTimeMask2D(channels=dim*4, mask_prob=0.45, mask_length=10) # x 是 shape (B, C, T, F)  # 45% mask

        # self.blocks_inter = DualStreamSeq(*[MuGIBlock(dim*4) for _ in range(2)])
        # self.blocks_inter_10 = DualStreamSeq(*[MuGIBlock(dim*4) for _ in range(2)])
        # self.blocks_inter_30 = DualStreamSeq(*[MuGIBlock(dim*4) for _ in range(2)])
        # self.blocks_inter_45 = DualStreamSeq(*[MuGIBlock(dim*4) for _ in range(2)])
        # self.conv_all_v = nn.Conv2d(dim*16, dim*4, kernel_size=1, stride=1, bias=False)
        self.inter = nn.ModuleList([
            global_channel_attention(emb_dim, dim * 8) for _ in range(1)
        ])

        # self.conv = nn.Conv2d(dim*8, dim*4, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv = nn.Conv2d(dim*8, dim*3, kernel_size=1, stride=1,padding=0, bias=False)
        self.blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
            )
        self.deconv = nn.ConvTranspose2d(emb_dim, num_source * out_channels, ks, padding=padding)
        self.stft_decoder = STFTDecoder_1(win, hop_length, n_fft, in_chan=dim, n_src=num_source, kernel_size=3, stride=1, bias=bias)

    def forward(self, x, v, face=None):  # 这里应该是face
        # print(x.shape)
        # print(v[:, 0].shape)
        x= x.unsqueeze(1)
        mix_std_ = torch.std(x, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        x = x / mix_std_

        # print(face.shape)  # torch.Size([B, 2, 50, 1, 112, 112])
        # # print(face[:, 0].shape)
        # lip_emb0 = self.lipencoder(face[:, 0].float())
        # lip_emb1 = self.lipencoder(face[:, 1].float())
        # logits0 = self.classifier(lip_emb0)
        # logits1 = self.classifier(lip_emb1)
        # logits = [logits0, logits1]

        v00 = self.pre_v(v[:, 0])  # torch.Size([4, 192, 251, 129])   torch.Size([4, 192, 50])
        v01 = self.pre_v(v[:, 1])

        # lip_emb0 = lip_emb0.unsqueeze(-1)
        # lip_emb0 = lip_emb0.repeat(1, 1, v00.shape[-1])  # torch.Size([4, 256, 50])
        # lip_emb1 = lip_emb1.unsqueeze(-1)
        # lip_emb1 = lip_emb1.repeat(1, 1, v01.shape[-1])  # torch.Size([4, 256, 50])

        feature_map = self.stft_encoder(x)  # torch.Size([4, 2, 251, 129])
        B, C, T, F = feature_map.size()
        inp_enc_level1 = self.conv(feature_map) # torch.Size([4, 256, 251, 129])

        # v00 = self.conv_v0_lip0(torch.cat([lip_emb0, v00], dim=1))  # 512-192
        # v01 = self.conv_v0_lip1(torch.cat([lip_emb1, v01], dim=1))  # 512-192

        v00 = FF.interpolate(v00, size=T, mode='linear', align_corners=True)  # [B, C, 251]
        v00 = v00.unsqueeze(-1)  # [B, C, 251, 1]
        v00 = v00.repeat(1, 1, 1, F)  # [B, 256, 251, 129]  # 目前先用这个，还有一种方法是unfold，也就是lip-stft

        v01 = FF.interpolate(v01, size=T, mode='linear', align_corners=True)  # [B, C, 251]
        v01 = v01.unsqueeze(-1)  # [B, C, 251, 1]
        v01 = v01.repeat(1, 1, 1, F)  # [B, 256, 251, 129]  # 目前先用这个，还有一种方法是unfold，也就是lip-stft
        # v0 = self.conv_v0(torch.cat([v00, v01], dim=1))


        # # mask
        # inp_enc_level1_mask_10, mask10 = self.masker_10(inp_enc_level1)
        # inp_enc_level1_mask_30, mask30 = self.masker_30(inp_enc_level1)
        # inp_enc_level1_mask_45, mask45 = self.masker_45(inp_enc_level1)

        # inp_enc_level1_10, v0_10 = self.blocks_inter_10(inp_enc_level1_mask_10, v0)
        # inp_enc_level1_30, v0_30 = self.blocks_inter_30(inp_enc_level1_mask_30, v0)
        # inp_enc_level1_45, v0_45 = self.blocks_inter_45(inp_enc_level1_mask_45, v0)

        # inp_enc_level1_pre, v0_pre = self.blocks_inter(inp_enc_level1, v0)

        # # inp_enc_level_all = torch.cat([inp_enc_level1_pre, inp_enc_level1_10, inp_enc_level1_30, inp_enc_level1_45], dim=1)
        # v0_all = torch.cat([v0_pre, v0_10, v0_30, v0_45], dim=1)
        # v0_all = self.conv_all_v(v0_all)
        # inp_enc_level_all = inp_enc_level1_pre
        v0_all = torch.cat([v00, v01], dim=1)
        for i in self.inter:
            inp_enc_level1, v0_all = i(inp_enc_level1, v0_all)

        batch = self.conv1(torch.cat([inp_enc_level1, v0_all], dim=1))
        for ii in range(self.num_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]
        y = self.deconv(batch) # [B, n_srcs*2, T, F]
        y = y.view([B, self.num_source, 2, T, F])
        
        source = self.stft_decoder(y, x)
        # print(source.shape)
        source = mix_std_ * source

        stft_out0 = self.stft_encoder(source[:,0])
        stft_out0 = torch.complex(stft_out0[:, 0], stft_out0[:, 1])
        # print(stft_out0.shape)
        stft_out1 = self.stft_encoder(source[:,1])
        stft_out1 = torch.complex(stft_out1[:, 0], stft_out1[:, 1])
        stft_out_spec = torch.stack([stft_out0, stft_out1], dim=1)


        return stft_out_spec, source


if __name__ == '__main__':
    import os
    import glob
    from PIL import Image
    from torchvision import transforms
    transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),  # 自动将 [H, W] 转为 [1, H, W]，像素归一化到 [0, 1]
            transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图只用一个通道
        ])
    frame_paths1 = sorted(glob.glob(os.path.join("/home/xueke/LRS2/mvlrs_v1/frames_112/train/5535415699068794046_00002", '*.png')))
    frames1 = []
    for frame_path in frame_paths1:
        img = Image.open(frame_path)  # 不再加 convert('L')，因为你已经是灰度图
        img = transform(img)     # [1, 112, 112]
        frames1.append(img)
    frames1 = torch.stack(frames1)  # shape: [T, 1, 112, 112]

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tfgridnet_v2().to(device)
    # print(model)
    input_signal = torch.randn(2,32000).to(device)  # 输入图像的形状 (C, H, W)
    v = torch.randn(2,2,512,50).to(device)
    mouth = torch.randn(2,2,50,1,112,112).to(device)
    frames1 = frames1.unsqueeze(0).to(device)
    # summary(model, input_signal, device='cuda')
    import time
    start_time = time.time()
    # out1,out2, a,b,c, d,e,_,_,_ = model(input_signal, v, mouth)
    out1,out2 = model(input_signal, v, mouth)  # [1, T, 1, 112, 112]
    # print(logits[0].shape)
    # print(logits[1].shape)
    # criterion = nn.CrossEntropyLoss()
    # labels = torch.tensor([1]).to(device)
    # loss = criterion(logits, labels)
    # print(loss)
    # print(b.shape)  # torch.Size([2, 256, 251, 129])
    # print(c.shape) # torch.Size([2, 1, 251, 1]) mask
    end_time = time.time()
    print(out1.shape)
    print(out2.shape)

    # 输出信息
    print("模型参数总数:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("输出形状:", out1.shape)
    print(f"前向传播耗时: {end_time - start_time:.4f} 秒")

    # B, H, W = 2, 32, 32  # Batch size and spatial resolution
    # x = torch.randn(B, 64, H, W)     # 语音/音频特征
    # v = torch.randn(B, 384, H, W)    # 视频/视觉特征

    # model1 = global_channel_attention(channels_a=64, channels_v=384)
    # x_out, v_out = model1(x, v)
    # print("模型参数总数:", sum(p.numel() for p in model1.parameters() if p.requires_grad))
    # print("x_out shape:", x_out.shape)  # should be (B, 64, H, W)
    # print("v_out shape:", v_out.shape)  # should be (B, 384, H, W)
