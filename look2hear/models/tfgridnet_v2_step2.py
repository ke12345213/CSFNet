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
from .lip_encoder import LipEncoderClassifier
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

# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()

#         hidden_features = int(dim*ffn_expansion_factor)

#         self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
#         self.norm = nn.GroupNorm(1, hidden_features, eps=1e-6)
#         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = FF.gelu(x1) * x2
#         x = self.norm(x)
#         x = self.project_out(x)
#         return x
    

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

        # self.att_freq = nn.MultiheadAttention(emb_dim, n_head, batch_first=True)
        # self.norm_att_freq = nn.LayerNorm(emb_dim, eps=eps)
        # self.att_time = nn.MultiheadAttention(emb_dim, n_head, batch_first=True)
        # self.norm_att_time = nn.LayerNorm(emb_dim, eps=eps)

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
        # self.FFN = FeedForward(emb_dim, 4, bias=True)  # 64*4  256

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
        # branch 1
        intra_rnn = FF.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        print(intra_rnn.shape)
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        print(intra_rnn.shape)
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

        # inter_rnn = inter_rnn.transpose(1, 2)
        # att_out_time, _ = self.att_time(inter_rnn, inter_rnn, inter_rnn)
        # att_out_time = att_out_time + inter_rnn
        # att_out_time = self.norm_att_time(att_out_time)
        # inter_rnn = att_out_time.transpose(1, 2)

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

        # out1 = self.FFN(out)
        # assert not torch.isnan(out1).any(), "NaN detected in FFN output"
        # out = out1 + out
        return out


class MultiRangeGridNetBlock(nn.Module):
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
        
        # -------------------- intra模块多分支 --------------------
        # 分支1：不使用unfold，直接LSTM处理原始序列
        self.intra_branch1_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_branch1_rnn = nn.LSTM(
            emb_dim, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_branch1_linear = nn.Conv1d(
            hidden_channels * 2, emb_dim, kernel_size=1
        )
        
        # 分支2：使用unfold，ks=4, hs=1
        self.intra_branch2_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_branch2_rnn = nn.LSTM(
            emb_dim * 4, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_branch2_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, 4, stride=1
        )
        
        # 分支3：使用unfold，ks=8, hs=1
        self.intra_branch3_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_branch3_rnn = nn.LSTM(
            emb_dim * 8, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_branch3_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, 8, stride=1
        )
        
        # # 分支4：使用unfold，ks=8, hs=2
        # self.intra_branch4_norm = LayerNormalization4D(emb_dim, eps=eps)
        # self.intra_branch4_rnn = nn.LSTM(
        #     emb_dim * 8, hidden_channels, 1, batch_first=True, bidirectional=True
        # )
        # self.intra_branch4_linear = nn.ConvTranspose1d(
        #     hidden_channels * 2, emb_dim, 8, stride=2
        # )
        
        # intra分支融合层
        self.intra_fusion_conv = nn.Conv2d(emb_dim * 3, emb_dim, kernel_size=1)
        self.intra_fusion_norm = LayerNormalization4D(emb_dim, eps=eps)
        
        # -------------------- inter模块多分支 --------------------
        # 分支1：不使用unfold，直接LSTM处理原始序列
        self.inter_branch1_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_branch1_rnn = nn.LSTM(
            emb_dim, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_branch1_linear = nn.Conv1d(
            hidden_channels * 2, emb_dim, kernel_size=1
        )
        
        # 分支2：使用unfold，ks=4, hs=1
        self.inter_branch2_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_branch2_rnn = nn.LSTM(
            emb_dim * 4, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_branch2_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, 4, stride=1
        )
        
        # 分支3：使用unfold，ks=8, hs=1
        self.inter_branch3_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_branch3_rnn = nn.LSTM(
            emb_dim * 8, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.inter_branch3_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, 8, stride=1
        )
        
        # # 分支4：使用unfold，ks=8, hs=2
        # self.inter_branch4_norm = LayerNormalization4D(emb_dim, eps=eps)
        # self.inter_branch4_rnn = nn.LSTM(
        #     emb_dim * 8, hidden_channels, 1, batch_first=True, bidirectional=True
        # )
        # self.inter_branch4_linear = nn.ConvTranspose1d(
        #     hidden_channels * 2, emb_dim, 8, stride=2
        # )
        
        # inter分支融合层
        self.inter_fusion_conv = nn.Conv2d(emb_dim * 3, emb_dim, kernel_size=1)
        self.inter_fusion_norm = LayerNormalization4D(emb_dim, eps=eps)
        
        # 原始模型中的注意力机制部分保持不变
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)
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
        
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.emb_ks = 8
        self.emb_hs = 1

    def forward(self, x):
        B, C, old_T, old_Q = x.shape
        
        # 动态填充确保尺寸合适
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = FF.pad(x, (0, Q - old_Q, 0, T - old_T))
        
        # 保存原始输入用于最终残差连接
        residual = x
        
        # -------------------- intra模块前向传播 --------------------
        intra_input = x
        
        # 分支1：不使用unfold，直接LSTM处理
        intra_b1 = self.intra_branch1_norm(intra_input)
        intra_b1 = intra_b1.permute(0, 2, 3, 1).contiguous().view(B * T, Q, C)
        intra_b1, _ = self.intra_branch1_rnn(intra_b1)  # [BT, Q, H*2]
        intra_b1 = intra_b1.transpose(1, 2)  # [BT, H*2, Q]
        intra_b1 = self.intra_branch1_linear(intra_b1)  # [BT, C, Q]
        intra_b1 = intra_b1.view(B, T, C, Q).permute(0, 2, 1, 3).contiguous()
        
        # 分支2：ks=4, hs=1
        intra_b2 = self.intra_branch2_norm(intra_input)
        intra_b2 = intra_b2.transpose(1, 2).contiguous().view(B * T, C, Q)
        intra_b2 = FF.unfold(intra_b2[..., None], (4, 1), stride=(1, 1))
        intra_b2 = intra_b2.transpose(1, 2)  # [BT, -1, C*4]
        intra_b2, _ = self.intra_branch2_rnn(intra_b2)
        intra_b2 = intra_b2.transpose(1, 2)
        intra_b2 = self.intra_branch2_linear(intra_b2)  # [BT, C, Q]
        intra_b2 = intra_b2.view(B, T, C, Q).transpose(1, 2).contiguous()
        
        # 分支3：ks=8, hs=1
        intra_b3 = self.intra_branch3_norm(intra_input)
        intra_b3 = intra_b3.transpose(1, 2).contiguous().view(B * T, C, Q)
        intra_b3 = FF.unfold(intra_b3[..., None], (8, 1), stride=(1, 1))
        intra_b3 = intra_b3.transpose(1, 2)  # [BT, -1, C*8]
        intra_b3, _ = self.intra_branch3_rnn(intra_b3)
        intra_b3 = intra_b3.transpose(1, 2)
        intra_b3 = self.intra_branch3_linear(intra_b3)  # [BT, C, Q]
        intra_b3 = intra_b3.view(B, T, C, Q).transpose(1, 2).contiguous()
        
        # # 分支4：ks=8, hs=2
        # intra_b4 = self.intra_branch4_norm(intra_input)
        # intra_b4 = intra_b4.transpose(1, 2).contiguous().view(B * T, C, Q)
        # intra_b4 = FF.unfold(intra_b4[..., None], (8, 1), stride=(2, 1))
        # intra_b4 = intra_b4.transpose(1, 2)  # [BT, -1, C*8]
        # intra_b4, _ = self.intra_branch4_rnn(intra_b4)
        # intra_b4 = intra_b4.transpose(1, 2)
        # intra_b4 = self.intra_branch4_linear(intra_b4)  # [BT, C, Q']
        
        # 融合intra分支
        intra_concat = torch.cat([intra_b1, intra_b2, intra_b3], dim=1)
        intra_fused = self.intra_fusion_conv(intra_concat)
        intra_fused = self.intra_fusion_norm(intra_fused)
        
        # intra模块残差连接
        intra_output = intra_fused + intra_input
        
        # -------------------- inter模块前向传播 --------------------
        inter_input = intra_output
        
        # 分支1：不使用unfold，直接LSTM处理
        inter_b1 = self.inter_branch1_norm(inter_input)
        inter_b1 = inter_b1.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        inter_b1 = inter_b1.transpose(1, 2)
        inter_b1, _ = self.inter_branch1_rnn(inter_b1)  # [BQ, T, H*2]
        inter_b1 = inter_b1.transpose(1, 2)  # [BQ, H*2, T]
        inter_b1 = self.inter_branch1_linear(inter_b1)  # [BQ, C, T]
        inter_b1 = inter_b1.view(B, Q, C, T).permute(0, 2, 3, 1).contiguous()
        
        # 分支2：ks=4, hs=1
        inter_b2 = self.inter_branch2_norm(inter_input)
        inter_b2 = inter_b2.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        inter_b2 = FF.unfold(inter_b2[..., None], (4, 1), stride=(1, 1))
        inter_b2 = inter_b2.transpose(1, 2)  # [BQ, -1, C*4]
        inter_b2, _ = self.inter_branch2_rnn(inter_b2)
        inter_b2 = inter_b2.transpose(1, 2)
        inter_b2 = self.inter_branch2_linear(inter_b2)  # [BQ, C, T]
        inter_b2 = inter_b2.view(B, Q, C, T).permute(0, 2, 3, 1).contiguous()
        
        # 分支3：ks=8, hs=1
        inter_b3 = self.inter_branch3_norm(inter_input)
        inter_b3 = inter_b3.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        inter_b3 = FF.unfold(inter_b3[..., None], (8, 1), stride=(1, 1))
        inter_b3 = inter_b3.transpose(1, 2)  # [BQ, -1, C*8]
        inter_b3, _ = self.inter_branch3_rnn(inter_b3)
        inter_b3 = inter_b3.transpose(1, 2)
        inter_b3 = self.inter_branch3_linear(inter_b3)  # [BQ, C, T]
        inter_b3 = inter_b3.view(B, Q, C, T).permute(0, 2, 3, 1).contiguous()
        
        # # 分支4：ks=8, hs=2
        # inter_b4 = self.inter_branch4_norm(inter_input)
        # inter_b4 = inter_b4.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        # inter_b4 = FF.unfold(inter_b4[..., None], (8, 1), stride=(2, 1))
        # inter_b4 = inter_b4.transpose(1, 2)  # [BQ, -1, C*8]
        # inter_b4, _ = self.inter_branch4_rnn(inter_b4)
        # inter_b4 = inter_b4.transpose(1, 2)
        # inter_b4 = self.inter_branch4_linear(inter_b4)  # [BQ, C, T']
        # inter_b4 = inter_b4.view(B, Q, C, T).permute(0, 2, 3, 1).contiguous()
        
        # 融合inter分支
        inter_concat = torch.cat([inter_b1, inter_b2, inter_b3], dim=1)
        inter_fused = self.inter_fusion_conv(inter_concat)
        inter_fused = self.inter_fusion_norm(inter_fused)
        
        # inter模块残差连接
        inter_output = inter_fused + inter_input
        
        # 裁剪回原始尺寸
        inter_output = inter_output[..., :old_T, :old_Q]
        
        # 原始模型中的注意力机制部分
        batch = inter_output
        
        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))
            all_K.append(self["attn_conv_K_%d" % ii](batch))
            all_V.append(self["attn_conv_V_%d" % ii](batch))

        Q = torch.cat(all_Q, dim=0)
        K = torch.cat(all_K, dim=0)
        V = torch.cat(all_V, dim=0)

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)
        V = V.transpose(1, 2)
        old_shape = V.shape
        V = V.flatten(start_dim=2)
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)
        attn_mat = FF.softmax(attn_mat, dim=2)
        V = torch.matmul(attn_mat, V)

        V = V.reshape(old_shape)
        V = V.transpose(1, 2)
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])
        batch = batch.transpose(0, 1)
        batch = batch.contiguous().view([B, self.n_head * emb_dim, old_T, -1])
        batch = self["attn_concat_proj"](batch)

        # 最终输出与残差连接（使用原始尺寸的残差）
        out = batch + residual[..., :old_T, :old_Q]
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

class CABlock(nn.Module):  # 类似于通道选择性，SENet可以换一下
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )
    def forward(self, x):
        return x * self.ca(x)

class global_channel_attention(nn.Module):
    def __init__(self, channels_a=192, channels_v=192, channels_l=192, reduction=8):
        super(global_channel_attention, self).__init__()
        self.LN_a  = LayerNorm2d(channels_a)
        self.LN_v  = LayerNorm2d(channels_v)
        self.LN_l  = LayerNorm2d(channels_l)
        self.conv1_a = nn.Conv2d(channels_a, channels_a*3, kernel_size=1)
        self.conv1_v = nn.Conv2d(channels_v, channels_v*3, kernel_size=1)
        self.conv1_l = nn.Conv2d(channels_l, channels_l*3, kernel_size=1)
        self.conv1_dw_a = nn.Conv2d(channels_a*3, channels_a*3, kernel_size=3, padding=1, groups=channels_a * 3)
        self.conv1_dw_v = nn.Conv2d(channels_v*3, channels_v*3, kernel_size=3, padding=1, groups=channels_v * 3)
        self.conv1_dw_l = nn.Conv2d(channels_l*3, channels_l*3, kernel_size=3, padding=1, groups=channels_l * 3)

        # self.psa_a = PSAModule(channels_a, channels_a, reduction=reduction)
        # self.psa_v = PSAModule(channels_v, channels_v, reduction=reduction)
        # self.psa_l = PSAModule(channels_l, channels_l, reduction=reduction)

        self.ca_a = CABlock(channels_a)
        self.ca_v = CABlock(channels_v)
        self.ca_l = CABlock(channels_l)

        self.conv2_a = nn.Conv2d(channels_a, channels_a, kernel_size=1)
        self.conv2_v = nn.Conv2d(channels_v, channels_v, kernel_size=1)
        self.conv2_l = nn.Conv2d(channels_l, channels_l, kernel_size=1)

        self.conv3_a = nn.Conv2d(channels_a, channels_a*3, kernel_size=1)
        self.conv3_v = nn.Conv2d(channels_v, channels_v*3, kernel_size=1)
        self.conv3_l = nn.Conv2d(channels_l, channels_l*3, kernel_size=1)

        self.conv4_a = nn.Conv2d(channels_a, channels_a, kernel_size=1)
        self.conv4_v = nn.Conv2d(channels_v, channels_v, kernel_size=1)
        self.conv4_l = nn.Conv2d(channels_l, channels_l, kernel_size=1)

        # self.cross_attention_v = CrossAttention(dim_q=channels_a, dim_kv=channels_v, dim_out=channels_a)
        # self.cross_attention_a = CrossAttention(dim_q=channels_v, dim_kv=channels_a, dim_out=channels_v)
        # self.conv2_a = nn.Conv2d(channels_a, channels_a, kernel_size=1)
        # self.conv2_v = nn.Conv2d(channels_v, channels_v, kernel_size=1)

    def forward(self, x, v, l):
        # 原始特征克隆
        x1 = x.clone()
        v1 = v.clone()
        l1 = l.clone()

        # 1. LayerNorm
        x = self.LN_a(x)
        v = self.LN_v(v)
        l = self.LN_l(l)

        x = self.conv1_a(x)
        x= self.conv1_dw_a(x)
        x_1, x_2, x_3 = torch.chunk(x, 3, dim=1)  # 192
        # x2 = self.psa_a(x)      # 提取后得到 x 的增强特征 x2

        v = self.conv1_v(v)
        v = self.conv1_dw_v(v)  
        v_1, v_2, v_3 = torch.chunk(v, 3, dim=1) # 192
        # v2 = self.psa_v(v)      # 得到 v 的增强特征 v2

        l = self.conv1_l(l)
        l = self.conv1_dw_l(l)
        l_1, l_2, l_3 = torch.chunk(l, 3, dim=1) # 192

        # # gated_fusion  # 这个对每个模态的依赖性太强了，不太好
        # x_out = x_1 * v_1 * l_1
        # v_out = v_2 * x_2 * l_2
        # l_out = l_3 * x_3 * v_3

        # 改进版
        x_out = x_1 + v_1 + l_1 + x_1 * v_1 + x_1 * l_1 + v_1 * l_1
        v_out = v_2 + x_2 + l_2 + v_2 * x_2 + v_2 * l_2 + x_2 * l_2
        l_out = l_3 + x_3 + v_3 + l_3 * x_3 + l_3 * v_3 + x_3 * v_3 

        x_out = self.ca_a(x_out)
        v_out = self.ca_v(v_out)
        l_out = self.ca_l(l_out)

        x_out = self.conv2_a(x_out)
        v_out = self.conv2_v(v_out)
        l_out = self.conv2_l(l_out)

        x_out1 = x_out + x1
        v_out1 = v_out + v1
        l_out1 = l_out + l1  # 残差连接

        x_out2 = self.LN_a(x_out1)
        v_out2 = self.LN_v(v_out1)
        l_out2 = self.LN_l(l_out1)

        x_out2 = self.conv3_a(x_out2)
        v_out2 = self.conv3_v(v_out2)
        l_out2 = self.conv3_l(l_out2)

        x_out2_1, x_out2_2, x_out2_3 = torch.chunk(x_out2, 3, dim=1)
        v_out2_1, v_out2_2, v_out2_3 = torch.chunk(v_out2, 3, dim=1)
        l_out2_1, l_out2_2, l_out2_3 = torch.chunk(l_out2, 3, dim=1)

        # x_out2 = x_out2_1 * v_out2_1  * l_out2_1
        # v_out2 = v_out2_2 * x_out2_2  * l_out2_2
        # l_out2 = l_out2_3 * x_out2_3  * v_out2_3

        # 改进版 
        x_out2 = (
            x_out2_1 + v_out2_1 + l_out2_1
            + x_out2_1 * v_out2_1
            + x_out2_1 * l_out2_1
            + v_out2_1 * l_out2_1
        )
        v_out2 = (
            v_out2_2 + x_out2_2 + l_out2_2
            + v_out2_2 * x_out2_2
            + v_out2_2 * l_out2_2
            + x_out2_2 * l_out2_2
        )
        l_out2 = (
            l_out2_3 + x_out2_3 + v_out2_3
            + l_out2_3 * x_out2_3
            + l_out2_3 * v_out2_3
            + x_out2_3 * v_out2_3
        )

        x_out2 = self.conv4_a(x_out2)
        v_out2 = self.conv4_v(v_out2)
        l_out2 = self.conv4_l(l_out2)

        x_out2 = x_out2 + x_out1
        v_out2 = v_out2 + v_out1
        l_out2 = l_out2 + l_out1
        # # 3. Cross-Attention：用 v 作为 KV 融合到 x，反之亦然
        # x = self.cross_attention_v(x2, v)  # 理论上还可以直接将混合语音和唇语特征
        # # 直接分隔开，先让网络提前知道要分离的通道数
        # x = x + x2              # 残差连接

        # v = self.cross_attention_a(v2, x)
        # v = v + v2              # 残差连接

        # # 4. 输出映射 & 总残差连接
        # x = self.conv2_a(x)
        # x_out = x + x1

        # v = self.conv2_v(v)
        # v_out = v + v1
        return x_out2, v_out2, l_out2


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

import torch as th
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Resnet block for speaker encoder to obtain speaker embedding
    ref to 
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        and
        https://github.com/Jungjee/RawNet/blob/master/PyTorch/model_RawNet.py
    """
    def __init__(self, in_dims, out_dims):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.maxpool = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        return self.maxpool(y)

class ChannelwiseLayerNorm(nn.LayerNorm):
    """
    Channel-wise layer normalization based on nn.LayerNorm
    Input: 3D tensor with [batch_size(N), channel_size(C), frame_num(T)]
    Output: 3D tensor with same shape
    """

    def __init__(self, *args, **kwargs):
        super(ChannelwiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} requires a 3D tensor input".format(
                self.__name__))
        x = th.transpose(x, 1, 2)
        x = super().forward(x)
        x = th.transpose(x, 1, 2)
        return x


class tfgridnet_v2(nn.Module):
    def __init__(self, 
        num_layers = 6,  # 12
        win = 256,
        hop_length = 128,
        n_fft = 256,
        inp_channels=2, 
        out_channels=2, 
        dim = 48,
        bias = False,
        vpre_channels=768,
        num_source = 2,
        lstm_hidden_units=128,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        
    ):

        super(tfgridnet_v2, self).__init__()
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
            # nn.Conv2d(inp_channels, emb_dim, ks, padding=padding),  # 48
            multi_OverlapPatchEmbed(inp_channels, emb_dim),
            nn.GroupNorm(1, emb_dim*4, eps=eps),
            nn.PReLU()  # 48
        ) 
        self.conv1 = nn.Sequential(
            nn.Conv2d(emb_dim*4+dim*8, emb_dim, ks, padding=padding),  # 48
            nn.GroupNorm(1, emb_dim, eps=eps),
            nn.PReLU()  # 48
        )
        # self.conv_v0 = nn.Conv2d(dim*8, dim*4, kernel_size=1)

        self.lipencoder = LipEncoderClassifier(num_speakers=2936, tcn_channels=256, lip_emb_dim=192)
        # self.classifier = nn.Linear(192, 2936)

        # 改 fusion conv
        self.fusion_conv0 = nn.Sequential(
            nn.Conv2d(dim*12, dim*4, kernel_size=1),
            nn.GroupNorm(1, dim*4, eps=eps),
            nn.PReLU()
        )

        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(dim*12, dim*4, kernel_size=1),
            nn.GroupNorm(1, dim*4, eps=eps),
            nn.PReLU()
        )
        self.blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.blocks.append(
                MultiRangeGridNetBlock(
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

        logits0, lip_emb0 = self.lipencoder(face[:, 0].float()) # torch.Size([4, 50, 192]) 
        logits1, lip_emb1 = self.lipencoder(face[:, 1].float()) # torch.Size([4, 50, 192])
        # utt_emb0 = lip_emb0.mean(dim=1)  # [B, D]  —— 全局平均池化
        # utt_emb1 = lip_emb1.mean(dim=1)  # [B, D]  —— 全局平均池化
        # logits0 = self.classifier(utt_emb0)
        # logits1 = self.classifier(utt_emb1)
        logits = [logits0, logits1]

        # lip_emb0 = self.lipencoder(face[:, 0].float()) # torch.Size([4, 50, 192]) 
        # lip_emb1 = self.lipencoder(face[:, 1].float()) # torch.Size([4, 50, 192])
        # utt_emb0 = lip_emb0.mean(dim=1)  # [B, D]  —— 全局平均池化
        # utt_emb1 = lip_emb1.mean(dim=1)  # [B, D]  —— 全局平均池化
        # logits0 = self.classifier(utt_emb0)
        # logits1 = self.classifier(utt_emb1)
        # logits = [logits0, logits1]

        feature_map = self.stft_encoder(x)  # torch.Size([4, 2, 251, 129])
        assert not torch.isnan(feature_map).any(), "NaN in stft_encoder output"
        B, C, T, F = feature_map.size()
        inp_enc_level1 = self.conv(feature_map) # torch.Size([4, 256, 251, 129])
        assert not torch.isnan(inp_enc_level1).any(), "NaN in conv"

        
        v00 = self.pre_v(v[:, 0])  # torch.Size([4, 192, 251, 129])   torch.Size([4, 192, 50])
        v01 = self.pre_v(v[:, 1])

        lip_emb0 = lip_emb0.transpose(1, 2)
        lip_emb0 = FF.interpolate(lip_emb0, size=T, mode='linear', align_corners=True)
        lip_emb0 = lip_emb0.unsqueeze(-1)
        lip_emb0 = lip_emb0.repeat(1, 1, 1, F)

        lip_emb1 = lip_emb1.transpose(1, 2)
        lip_emb1 = FF.interpolate(lip_emb1, size=T, mode='linear', align_corners=True) 
        lip_emb1 = lip_emb1.unsqueeze(-1)
        lip_emb1 = lip_emb1.repeat(1, 1, 1, F)  # torch.Size([4, 192, 50])

        v00 = FF.interpolate(v00, size=T, mode='linear', align_corners=True)  # [B, C, 251]
        v00 = v00.unsqueeze(-1)  # [B, C, 251, 1]
        v00 = v00.repeat(1, 1, 1, F)  # [B, 192, 251, 129]  # 目前先用这个，还有一种方法是unfold，也就是lip-stft

        v01 = FF.interpolate(v01, size=T, mode='linear', align_corners=True)  # [B, C, 251]
        v01 = v01.unsqueeze(-1)  # [B, C, 251, 1]
        v01 = v01.repeat(1, 1, 1, F)  # [B, 192, 251, 129]

        # inp_enc_level1_1 = inp_enc_level1
        # inp_enc_level1_2 = inp_enc_level1  
        # for i in self.inter0:
        #     inp_enc_level1_1, v00, lip_emb0 = i(inp_enc_level1_1, v00, lip_emb0)
        # for i in self.inter1:
        #     inp_enc_level1_2, v01, lip_emb1 = i(inp_enc_level1_2, v01, lip_emb1)


        fusion0 = torch.cat([inp_enc_level1, v00, lip_emb0], dim=1)  # [B, 512, T, F] 192+192+192  # 576
        fusion1 = torch.cat([inp_enc_level1, v01, lip_emb1], dim=1)  # [B, 512, T, F] 192+192+192  # 576

        fusion0 = self.fusion_conv0(fusion0)  # nn.Conv2d(768, 192, 1)
        fusion1 = self.fusion_conv1(fusion1)  # [B, 192, T, F]
       
        batch = self.conv1(torch.cat([inp_enc_level1, fusion0, fusion1], dim=1))
        for ii in range(self.num_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]
        y = self.deconv(batch) # [B, n_srcs*2, T, F]
        assert not torch.isnan(y).any(), "NaN in est_sources"
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


        return stft_out_spec, source, logits

import sys
import os
sys.path.append("/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/auto_avsr_av")
import argparse
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args(args=[])

from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.nets_utils import (
    make_non_pad_mask
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E as E2E_av
from pytorch_lightning import LightningModule
from datamodule.transforms import TextTransform
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.scorers.length_bonus import LengthBonus
from argparse import Namespace

def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """
    Args:
        pad_outputs: Tensor of shape (B * T, D)
        pad_targets: LongTensor of shape (B * T)
    Returns:
        float: accuracy ignoring ignore_label
    """
    # 获取预测标签
    pad_pred = pad_outputs.argmax(dim=-1)  # shape: (B * T,)

    # 构建 mask，忽略 ignore_id
    mask = pad_targets != ignore_label

    # 计算准确率
    correct = (pad_pred[mask] == pad_targets[mask]).sum()
    total = mask.sum()

    if total == 0:
        return 0.0  # 避免除以 0
    else:
        return float(correct) / float(total)

config_dict = {
    "adim": 768,
    "aheads": 12,
    "eunits": 3072,
    "elayers": 12,
    "transformer_input_layer": "conv3d",
    "dropout_rate": 0.1,
    "transformer_attn_dropout_rate": 0.1,
    "transformer_encoder_attn_layer_type": "rel_mha",
    "macaron_style": True,
    "use_cnn_module": True,
    "cnn_module_kernel": 31,
    "zero_triu": False,
    "a_upsample_ratio": 1,
    "relu_type": "swish",
    "ddim": 768,
    "dheads": 12,
    "dunits": 3072,
    "dlayers": 6,
    "lsm_weight": 0.1,
    "transformer_length_normalized_loss": False,
    "mtlalpha": 0.1,
    "ctc_type": "builtin",
    "rel_pos_type": "latest",

    "aux_adim": 768,
    "aux_aheads": 12,
    "aux_eunits": 3072,
    "aux_elayers": 12,
    "aux_transformer_input_layer": "conv1d",
    "aux_dropout_rate": 0.1,
    "aux_transformer_attn_dropout_rate": 0.1,
    "aux_transformer_encoder_attn_layer_type": "rel_mha",
    "aux_macaron_style": True,
    "aux_use_cnn_module": True,
    "aux_cnn_module_kernel": 31,
    "aux_zero_triu": False,
    "aux_a_upsample_ratio": 1,
    "aux_relu_type": "swish",
    "aux_dunits": 3072,
    "aux_dlayers": 6,
    "aux_lsm_weight": 0.1,
    "aux_transformer_length_normalized_loss": False,
    "aux_mtlalpha": 0.1,
    "aux_ctc_type": "builtin",
    "aux_rel_pos_type": "latest",

    "fusion_hdim": 8192,
    "fusion_norm": "batchnorm",
}

args = Namespace(**config_dict)

class MLPHead(torch.nn.Module):
    def __init__(self, idim, hdim, odim, norm="batchnorm"):
        super(MLPHead, self).__init__()
        self.norm = norm

        self.fc1 = torch.nn.Linear(idim, hdim)
        if norm == "batchnorm":
            self.bn1 = torch.nn.BatchNorm1d(hdim)
        elif norm == "layernorm":
            self.norm1 = torch.nn.LayerNorm(hdim)
        self.nonlin1 = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(hdim, odim)

    def forward(self, x):
        x = self.fc1(x)
        if self.norm == "batchnorm":
            x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        elif self.norm == "layernorm":
            x = self.norm1(x)
        x = self.nonlin1(x)
        x = self.fc2(x)
        return x

def get_beam_search_decoder_av(model, token_list, ctc_weight=0.1, beam_size=40):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )

class ModelModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # self.modality = args.modality
        self.adim = args.adim
        self.fusion = args.fusion_norm
        self.sos = self.adim - 1
        self.eos = self.adim - 1
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.ignore_id = -1

        self.model = E2E_av(len(self.token_list), args, ignore_id=-1)
        self.fusion = MLPHead(
            idim=self.adim + self.adim,
            hdim=8192,
            odim=self.adim,
            norm=self.fusion,
        )

    def forward(self, x, a, label=None):
        # x, _ = self.model.encoder(x, None) # torch.Size([4, 50, 768]) 4是batch
        a, _ = self.model.aux_encoder(a, None)  # torch.Size([4, 50, 768])
        f = self.fusion(torch.cat((x, a), dim=-1)) # torch.Size([4, 50, 768]
        return x, a, f

    def forward_predicted_v_embed(self, x, a, label=None):
        self.beam_search = get_beam_search_decoder_av(self.model, self.token_list)
        # x, _ = self.model.encoder(x, None) # torch.Size([4, 50, 768]) 4是batch
        a, _ = self.model.aux_encoder(a, None)  # torch.Size([4, 50, 768])
        f = self.fusion(torch.cat((x, a), dim=-1)) # torch.Size([4, 50, 768]
        audiovisual_feat = f.squeeze(0)
        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def forward_predicted(self, video, audio):  # 先不进行解码，节约时间
        # self.beam_search = get_beam_search_decoder_av(self.model, self.token_list)
        video_feat, _ = self.model.encoder(video.unsqueeze(0), None)
        # video_feat = video_feat[:, :50] # 先全局建模然后提取前50帧的特征
        audio_feat, _ = self.model.aux_encoder(audio, None)
        # audio_feat = audio_feat[:, :50]
        audiovisual_feat = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))

        # audiovisual_feat = audiovisual_feat.squeeze(0)
        # nbest_hyps = self.beam_search(audiovisual_feat)
        # nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        # predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        # predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        # return audiovisual_feat, predicted
        return audiovisual_feat
    
   
    # def forward(self, x, a):
    #     # x, _ = self.model.encoder(x, None)
    #     # print(x.shape)  # torch.Size([4, 50, 768]) 4是batch
    #     a, _ = self.model.aux_encoder(a, None)  
    #     # print(a.shape)  # # torch.Size([4, 50, 768])
    #     f = self.model.fusion(torch.cat((x, a), dim=-1))
    #     # print(f.shape)  # torch.Size([4, 50, 768])
    #     return x, a, f
    
# model_path = "/root/IIANet-main/avsr_trlrwlrs2lrs3vox2avsp_base.pth"
# model = ModelModule(args)
# ckpt = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True)
# model.model.load_state_dict(ckpt)
# model.freeze()

# # x = torch.randn((4, 50, 1, 88, 88))
# x = torch.randn((4, 50, 768))
# a = torch.randn((4, 32000, 1)) 
# with torch.inference_mode():
#     video, audio, y = model(x, a)
# print(y.size())

import torch
import torch.nn.functional as F

def si_snr(est_source, true_source, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR)

    Args:
        est_source (Tensor): 估计源 [B, T]
        true_source (Tensor): 真实源 [B, T]
        eps (float): 避免除以零的微小值

    Returns:
        Tensor: 每个样本的 SI-SNR [B]
    """
    B, T = true_source.size()

    # 去均值
    est_source = est_source - torch.mean(est_source, dim=1, keepdim=True)
    true_source = true_source - torch.mean(true_source, dim=1, keepdim=True)

    # 计算投影目标分量 s_target
    dot = torch.sum(est_source * true_source, dim=1, keepdim=True)  # [B, 1]
    s_target = dot * true_source / (torch.sum(true_source ** 2, dim=1, keepdim=True) + eps)  # [B, T]

    # 计算噪声分量 e_noise
    e_noise = est_source - s_target

    # SI-SNR = 10 * log10(||s_target||² / ||e_noise||²)
    target_power = torch.sum(s_target ** 2, dim=1)  # [B]
    noise_power = torch.sum(e_noise ** 2, dim=1) + eps  # [B]

    si_snr_val = 10 * torch.log10(target_power / noise_power + eps)  # [B]
    return si_snr_val


# import sys
# # sys.path.insert(0, "../../../")  # 注意是当前文件，将上一级目录加入 Python 模块路径，确保可以 import 工程外部的模块或 espnet 项目代码。
# sys.path.append("/home/xueke/DPT_1d_main/")
# import argparse
# parser = argparse.ArgumentParser()
# args1, _ = parser.parse_known_args(args=[])

# from auto_vsr_pretrain_model.auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer import E2E as E2E_asr
# from pytorch_lightning import LightningModule
# from auto_vsr_pretrain_model.auto_avsr.espnet.nets.scorers.length_bonus import LengthBonus
# from auto_vsr_pretrain_model.auto_avsr.espnet.nets.batch_beam_search import BatchBeamSearch
# from auto_vsr_pretrain_model.auto_avsr.datamodule.transforms import TextTransform, VideoTransform

# # def si_snr(est_source, true_source, eps=1e-8):
# #     """
# #     Scale-Invariant Signal-to-Noise Ratio (SI-SNR)

# #     Args:
# #         est_source (Tensor): 估计源 [B, T]
# #         true_source (Tensor): 真实源 [B, T]
# #         eps (float): 避免除以零的微小值

# #     Returns:
# #         Tensor: 每个样本的 SI-SNR [B]
# #     """
# #     B, T = true_source.size()

# #     # 去均值
# #     est_source = est_source - torch.mean(est_source, dim=1, keepdim=True)
# #     true_source = true_source - torch.mean(true_source, dim=1, keepdim=True)

# #     # 计算投影目标分量 s_target
# #     dot = torch.sum(est_source * true_source, dim=1, keepdim=True)  # [B, 1]
# #     s_target = dot * true_source / (torch.sum(true_source ** 2, dim=1, keepdim=True) + eps)  # [B, T]

# #     # 计算噪声分量 e_noise
# #     e_noise = est_source - s_target

# #     # SI-SNR = 10 * log10(||s_target||² / ||e_noise||²)
# #     target_power = torch.sum(s_target ** 2, dim=1)  # [B]
# #     noise_power = torch.sum(e_noise ** 2, dim=1) + eps  # [B]

# #     si_snr_val = 10 * torch.log10(target_power / noise_power + eps)  # [B]
# #     return si_snr_val


# def get_beam_search_decoder(
#     model,
#     token_list,
#     rnnlm=None,
#     rnnlm_conf=None,
#     penalty=0,
#     ctc_weight=0.1,
#     lm_weight=0.0,
#     beam_size=40,
# ):
#     sos = model.odim - 1
#     eos = model.odim - 1
#     scorers = model.scorers()

#     scorers["lm"] = None
#     scorers["length_bonus"] = LengthBonus(len(token_list))
#     weights = {
#         "decoder": 1.0 - ctc_weight,
#         "ctc": ctc_weight,
#         "lm": lm_weight,
#         "length_bonus": penalty,
#     }

#     return BatchBeamSearch(
#         beam_size=beam_size,
#         vocab_size=len(token_list),
#         weights=weights,
#         scorers=scorers,
#         sos=sos,
#         eos=eos,
#         token_list=token_list,
#         pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
#     )

# def wer(reference: str, hypothesis: str) -> float:
#     r = reference.strip().split()
#     h = hypothesis.strip().split()
#     n = len(r)
#     dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
#     for i in range(len(r) + 1): dp[i][0] = i
#     for j in range(len(h) + 1): dp[0][j] = j
#     for i in range(1, len(r) + 1):
#         for j in range(1, len(h) + 1):
#             if r[i - 1] == h[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1]
#             else:
#                 dp[i][j] = min(dp[i - 1][j - 1] + 1,  # substitute
#                                dp[i][j - 1] + 1,      # insert
#                                dp[i - 1][j] + 1)      # delete
#     return dp[len(r)][len(h)] / n if n > 0 else float('inf')

# class ModelModule_a(LightningModule):
#     def __init__(self, args1):
#         super().__init__()
#         self.args = args1
#         self.save_hyperparameters(args1)

#         self.modality = args1.modality
#         self.text_transform = TextTransform()
#         self.token_list = self.text_transform.token_list

#         self.model = E2E_asr(len(self.token_list), self.modality, ctc_weight=getattr(args1, "ctc_weight", 0.1))

#     def forward(self, x, lengths, label):
#         _, _, _, x = self.model(x.unsqueeze(-1), lengths, label) # 返回的是预测后的token——id的准确率，并不是对应的词错误率
#         return x
#     def forward_predicted(self, x):
#         self.beam_search = get_beam_search_decoder(self.model, self.token_list)
#         x = self.model.frontend(x.unsqueeze(-1))
#         x = self.model.proj_encoder(x)
#         enc_feat, _ = self.model.encoder(x, None)
#         enc_feat = enc_feat[:, :50]
#         enc_feat = enc_feat.squeeze(0)
#         nbest_hyps = self.beam_search(enc_feat)
#         nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
#         predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
#         predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
#         return predicted

class tfgridnet_v2_step2(nn.Module):
    def __init__(self, 
        num_layers = 6,  # 12
        win = 256,
        hop_length = 128,
        n_fft = 256,
        inp_channels=2, 
        out_channels=2, 
        dim = 48,
        bias = False,
        vpre_channels=768,
        num_source = 2,
        lstm_hidden_units=128,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        
    ):
        super(tfgridnet_v2_step2, self).__init__()
        self.num_source = num_source
        self.win = win
        self.hop_length = hop_length
        self.num_layers = num_layers
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

        self.tfgridnet1 = tfgridnet_v2(num_layers, win,hop_length,n_fft,inp_channels, out_channels, dim,
        bias,
        vpre_channels,
        num_source,
        lstm_hidden_units,
        attn_n_head,
        attn_approx_qk_dim,
        emb_dim,
        emb_ks,
        emb_hs,
        activation="prelu",
        eps=1.0e-5,)

        self.tfgridnet2 = tfgridnet_v2(num_layers, win,hop_length,n_fft,inp_channels, out_channels, dim,
        bias,
        vpre_channels,
        num_source,
        lstm_hidden_units,
        attn_n_head,
        attn_approx_qk_dim,
        emb_dim,
        emb_ks,
        emb_hs,
        activation="prelu",
        eps=1.0e-5,)

        model_path = "/home/xueke/DPT_1d_main/checkpoint_improve_tfgridnet_LRS2_SS/LRS2-restormer/epoch=113-16.5.ckpt"
        ckpt1 = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True)
        state_dict = ckpt1["state_dict"]
        # print(state_dict.keys())
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("audio_model."):
                new_k = k[len("audio_model."):]  # 去掉前缀
                new_state_dict[new_k] = v
        self.tfgridnet1.load_state_dict(new_state_dict, strict=False)  # video_model不需要管s
        # self.tfgridnet1.eval()
        for p in self.tfgridnet1.parameters():
            p.requires_grad = False

        self.tfgridnet2.load_state_dict(new_state_dict, strict=False)  # video_model不需要管s

        config_dict = {
            "adim": 768,
            "aheads": 12,
            "eunits": 3072,
            "elayers": 12,
            "transformer_input_layer": "conv3d",
            "dropout_rate": 0.1,
            "transformer_attn_dropout_rate": 0.1,
            "transformer_encoder_attn_layer_type": "rel_mha",
            "macaron_style": True,
            "use_cnn_module": True,
            "cnn_module_kernel": 31,
            "zero_triu": False,
            "a_upsample_ratio": 1,
            "relu_type": "swish",
            "ddim": 768,
            "dheads": 12,
            "dunits": 3072,
            "dlayers": 6,
            "lsm_weight": 0.1,
            "transformer_length_normalized_loss": False,
            "mtlalpha": 0.1,
            "ctc_type": "builtin",
            "rel_pos_type": "latest",

            "aux_adim": 768,
            "aux_aheads": 12,
            "aux_eunits": 3072,
            "aux_elayers": 12,
            "aux_transformer_input_layer": "conv1d",
            "aux_dropout_rate": 0.1,
            "aux_transformer_attn_dropout_rate": 0.1,
            "aux_transformer_encoder_attn_layer_type": "rel_mha",
            "aux_macaron_style": True,
            "aux_use_cnn_module": True,
            "aux_cnn_module_kernel": 31,
            "aux_zero_triu": False,
            "aux_a_upsample_ratio": 1,
            "aux_relu_type": "swish",
            "aux_dunits": 3072,
            "aux_dlayers": 6,
            "aux_lsm_weight": 0.1,
            "aux_transformer_length_normalized_loss": False,
            "aux_mtlalpha": 0.1,
            "aux_ctc_type": "builtin",
            "aux_rel_pos_type": "latest",

            "fusion_hdim": 8192,
            "fusion_norm": "batchnorm",
        }
        args = Namespace(**config_dict)
        self.model_av = ModelModule(args)
        model_path_av = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/avsr_trlrwlrs2lrs3vox2avsp_base.pth"
        ckpt = torch.load(model_path_av, map_location=lambda storage, loc: storage, weights_only=True)
        self.model_av.model.load_state_dict(ckpt)
        # self.model_av.freeze()
        for p in self.model_av.parameters():
            p.requires_grad = False

        self.a = nn.Parameter(torch.tensor(1.0))  # 初始保留视觉影响力
        self.b = nn.Parameter(torch.tensor(0.0))  # 初始让模型学着慢慢信任融合信息

    def forward(self, x, v, face=None, targets=None, videos=None):  # 这里应该是face
        stft_out_spec, source, logits = self.tfgridnet1.forward(x, v, face)  # torch.Size([2, 2, 32000])

        # 怎么把这两个两两对齐
        # print(targets.shape)
        si_snr1_1 = si_snr(source[:,0], targets[:,0])
        si_snr2_1 = si_snr(source[:,1], targets[:,1])
        si_snr1_avg = (si_snr1_1 + si_snr2_1) / 2
        # print(si_snr1_avg)
        si_snr1_2 = si_snr(source[:,1], targets[:,0])
        si_snr2_2 = si_snr(source[:,0], targets[:,1])
        si_snr1_avg2 = (si_snr1_2 + si_snr2_2) / 2
        # print(si_snr1_avg2)
        if si_snr1_avg > si_snr1_avg2:
            source1 = source[:,0].unsqueeze(-1)  # torch.Size([2, 32000, 1])
            source2 = source[:,1].unsqueeze(-1)  # torch.Size([2, 32000, 1])
        else:
            source1 = source[:,1].unsqueeze(-1)  # torch.Size([2, 32000, 1])
            source2 = source[:,0].unsqueeze(-1)  # torch.Size([2, 32000, 1])

        v1 = v[:, 0].transpose(1, 2)  # (2, 50, 768)
        v2 = v[:, 1].transpose(1, 2)  # (2, 50, 768)

        video1 = videos[:,0]  #  # torch.Size([1, 2, 65, 1, 88, 88])
        video2 = videos[:,1]  #  # torch.Size([1, 2, 65, 1, 88, 88])
        # 第一种
        with torch.no_grad():
            # video1_1, audio1_1, fusied_y1_1 = self.model_av.forward(v1, source1)  # 从解码的角度证明这个也是错的
            # predicted_v_embedd_a_1 = self.model_av.forward_predicted_v_embed(v1, source1)  # 从解码的角度证明这个是错的
            # f1, predicted_v_a_1 = self.model_av.forward_predicted(video1.squeeze(0), source1) 
            f1 = self.model_av.forward_predicted(video1.squeeze(0), source1)   

            # video2_1, audio2_1, fusied_y2_1 = self.model_av.forward(v2, source2)
            # predicted_v_embedd_a_2 = self.model_av.forward_predicted_v_embed(v2, source2)
            # f2, predicted_v_a_2 = self.model_av.forward_predicted(video2.squeeze(0), source2)
            f2 = self.model_av.forward_predicted(video2.squeeze(0), source2)

        # fusied_y = torch.stack((fusied_y1_1, fusied_y2_1), dim=1) 
        # print(fusied_y.shape) # torch.Size([1, 2, 50, 768])
        fusied_y = torch.stack((f1, f2), dim=1)
        # print(fusied_y.shape)
        fusied_y = fusied_y.transpose(2, 3)

        # step1
        # out = self.a * v + self.b * fusied_y  # 不能固定，固定的话fusied_y的权重会固定，不会变了，训练就没有动态的意义
        out = fusied_y
        stft_out_spec_1, source_1, logits_1 = self.tfgridnet2.forward(x, out, face)

        # si_snr1_1 = si_snr(source_1[:,0], targets[:,0])
        # si_snr2_1 = si_snr(source_1[:,1], targets[:,1])
        # si_snr1_avg = (si_snr1_1 + si_snr2_1) / 2
        # si_snr1_2 = si_snr(source_1[:,1], targets[:,0])
        # si_snr2_2 = si_snr(source_1[:,0], targets[:,1])
        # si_snr1_avg2 = (si_snr1_2 + si_snr2_2) / 2
        # # print(si_snr1_avg2)
        # if si_snr1_avg > si_snr1_avg2:
        #     source1 = source_1[:,0].unsqueeze(-1)  # torch.Size([2, 32000, 1])
        #     source2 = source_1[:,1].unsqueeze(-1)  # torch.Size([2, 32000, 1])
        # else:
        #     source1 = source_1[:,1].unsqueeze(-1)  # torch.Size([2, 32000, 1])
        #     source2 = source_1[:,0].unsqueeze(-1)  # torch.Size([2, 32000, 1])
        # model_path = "/home/xueke/DPT_1d_main/auto_vsr_pretrain_model/asr_trlrs3vox2_base.pth"
        # setattr(args1, 'modality', 'audio')  # 设置 modality 为 "video"，
        # asr_model = ModelModule_a(args1)
        # asr_model.to("cuda")
        # ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        # asr_model.model.load_state_dict(ckpt)
        # asr_model.freeze()
        # with torch.inference_mode():
        #     # y1 = self.asr_model.forward(source1, length, label1.unsqueeze(0))
        #     # print(source1.shape)# torch.Size([1, 32000, 1])
        #     y_words1 = asr_model.forward_predicted(source1.squeeze(-1))
        #     y_words2 = asr_model.forward_predicted(source2.squeeze(-1))  # 这个暂时用不到了0.2db的提升对于wer来说没有提升。

        return stft_out_spec_1, source_1, logits_1, self.a, self.b


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
    # frame_paths1 = sorted(glob.glob(os.path.join("/root/dataset/LRS2/frame_112/train/5535415699068794046_00002", '*.png')))
    # frames1 = []
    # for frame_path in frame_paths1:
    #     img = Image.open(frame_path)  # 不再加 convert('L')，因为你已经是灰度图
    #     img = transform(img)     # [1, 112, 112]
    #     frames1.append(img)
    # frames1 = torch.stack(frames1)  # shape: [T, 1, 112, 112]

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tfgridnet_v2_step2().to(device)
    # print(model)
    input_signal = torch.randn(1,32000).to(device)  # 输入图像的形状 (C, H, W)
    v = torch.randn(1,2,768,50).to(device)
    mouth = torch.randn(1,2,50,1,112,112).to(device)
    target = torch.randn(1, 2, 32000).to(device)
    # frames1 = frames1.unsqueeze(0).to(device)
    label1 = torch.tensor([4498, 10, 33, 1]).to(device)
    label2 = torch.tensor([4498, 10, 33, 1]).to(device)
    videos = torch.randn(1,2,50,1,88,88).to(device) # torch.Size([1, 2, 65, 1, 88, 88])
    # summary(model, input_signal, device='cuda')
    import time
    start_time = time.time()
    # out1,out2, a,b,c, d,e,_,_,_ = model(input_signal, v, mouth)
    out1, out2, logits, a, b,_,_ = model(input_signal, v, mouth, target, videos)  # [1, T, 1, 112, 112]
    print(logits[0].shape)
    print(logits[1].shape)
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
