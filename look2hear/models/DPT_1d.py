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

        # d = self.mask_net(x)
        # # print(d.shape)
        # d = d.view(batch_size, self.n_src, -1, d.shape[-2], d.shape[-1])
        # # print(d.shape)
        # d = self.mask_nl_class(d) # mask: torch.Size([4, 2, 48, 251, 129])
        # d = d * encoder_out.unsqueeze(1) # 

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
        # mask_conv = nn.Conv2d(in_chan*2, 128, 1)
        # self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # self.mask_nl_class = nn.ReLU()

        # if self.kernel_size > 0:
        #     self.decoder = nn.ConvTranspose2d(
        #         in_channels=in_chan*2,
        #         out_channels=2,
        #         kernel_size=self.kernel_size,
        #         stride=self.stride,
        #         padding=self.padding,
        #         bias=self.bias,
        #     )
        #     torch.nn.init.xavier_uniform_(self.decoder.weight)
        # else:
        #     self.decoder = nn.Identity()

        self.register_buffer("window", torch.hann_window(self.win), False)

    def forward(self, x, input_shape):
        # # print(x.shape)
        # # B, n_src, N, T, F
        batch_size, length = input_shape.shape[0], input_shape.shape[-1]
        # x = x.view(batch_size * self.n_src, self.in_chan*2, x.shape[-2], x.shape[-1])  # B, n_src, N, T, F -> # B * n_src, N, T, F

        # # d = self.mask_net(x)
        # # # print(d.shape)
        # # d = d.view(batch_size, self.n_src, -1, d.shape[-2], d.shape[-1])
        # # # print(d.shape)
        # # d = self.mask_nl_class(d) # mask: torch.Size([4, 2, 48, 251, 129])
        # # d = d * encoder_out.unsqueeze(1) # 

        # # decoded_separated_audio = self.decoder(d.view(batch_size * self.n_src, 128, d.shape[-2], d.shape[-1]))  # B * n_src, N, T, F - > B * n_src, 2, T, F
        # # print(decoded_separated_audio.shape)
        # decoded_separated_audio = self.decoder(x)
        
        # spec = torch.complex(decoded_separated_audio[:, 0], decoded_separated_audio[:, 1])  # B*n_src, T, F
        # # spec = torch.stack([spec.real, spec.imag], dim=-1)  # B*n_src, T, F
        # spec = spec.transpose(1, 2).contiguous()  # B*n_src, F, T
        spec = torch.complex(x[:, 0], x[:, 1])
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

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
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

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

from functools import partial
class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 1
        # med_planes = outplanes // expansion
        med_planes = outplanes * expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x

class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 1
        med_planes = inplanes * expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


##########################################################################
class conformerBlock(nn.Module):  # 可以改成conformer
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, ):
        super(conformerBlock, self).__init__()

        self.cnn_block = ConvBlock(inplanes=dim, outplanes=dim, res_conv=False, stride=1, groups=1)
        self.med_block = nn.Sequential(*[Med_ConvBlock(inplanes=dim, groups=dim) for i in range(2)])
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.fusion = ConvBlock(inplanes=dim, outplanes=dim, res_conv=True, stride=1, groups=1)

    def forward(self, x):
        x1, x2 = self.cnn_block(x)
        xt = x + x2
        xt = xt + self.attn(self.norm1(xt))
        xt = xt + self.ffn(self.norm2(xt))
        resudial = xt
        x1 = self.med_block(x1)
        xt = self.fusion(x1, xt, return_x_2=False)
        xt = resudial + xt

        return xt

class TransformerBlock(nn.Module):  # 可以改成conformer
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, ):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class classical_Transformer(nn.Module):
    def __init__(self, dim, num_heads, dim_feedforward=768, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # 前馈网络（FFN）
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)

        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        if activation == "relu":
            self.activation = FF.relu
        elif activation == "gelu":
            self.activation = FF.gelu
        else:
            raise RuntimeError(f"Unsupported activation: {activation}")

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力 + 残差连接 + LayerNorm
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络 + 残差连接 + LayerNorm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class DPT(nn.Module):
    def __init__(self, 
        dim = 64,
        num_blocks = [1,1,1,2], 
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
    ):

        super(DPT, self).__init__()
        
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) 
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) 
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) 
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) 
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1)) 
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        # self.encoder_level1 = nn.Sequential(*[classical_Transformer(dim=dim, num_heads=heads[0], dim_feedforward=dim*4) for i in range(num_blocks[0])])
        
        # self.down1_2 = Downsample(dim) 
        # self.encoder_level2 = nn.Sequential(*[classical_Transformer(dim=int(dim*2**1), num_heads=heads[1], dim_feedforward=int(dim*2**1*4)) for i in range(num_blocks[1])])
        
        # self.down2_3 = Downsample(int(dim*2**1)) 
        # self.encoder_level3 = nn.Sequential(*[classical_Transformer(dim=int(dim*2**2), num_heads=heads[2], dim_feedforward=int(dim*2**2*4)) for i in range(num_blocks[2])])

        # self.down3_4 = Downsample(int(dim*2**2)) 
        # self.latent = nn.Sequential(*[classical_Transformer(dim=int(dim*2**3), num_heads=heads[3], dim_feedforward=int(dim*2**3*4)) for i in range(num_blocks[3])])
        
        # self.up4_3 = Upsample(int(dim*2**3)) 
        # self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        # self.decoder_level3 = nn.Sequential(*[classical_Transformer(dim=int(dim*2**2), num_heads=heads[2], dim_feedforward=int(dim*2**2*4)) for i in range(num_blocks[2])])

        # self.up3_2 = Upsample(int(dim*2**2)) 
        # self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        # self.decoder_level2 = nn.Sequential(*[classical_Transformer(dim=int(dim*2**1), num_heads=heads[1], dim_feedforward=int(dim*2**1*4)) for i in range(num_blocks[1])])
        
        # self.up2_1 = Upsample(int(dim*2**1)) 
        # self.decoder_level1 = nn.Sequential(*[classical_Transformer(dim=int(dim*2**1), num_heads=heads[1], dim_feedforward=int(dim*2**1*4)) for i in range(num_blocks[0])])


    def forward(self, x):
         
        out_enc_level1 = self.encoder_level1(x)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        B3, C3, T3, F3 = out_enc_level3.size()
        inp_dec_level3 = inp_dec_level3[:, :, :T3, :F3]
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        B2, C2, T2, F2 = out_enc_level2.size()
        inp_dec_level2 = inp_dec_level2[:, :, :T2, :F2]
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        B1, C1, T1, F1 = out_enc_level1.size()
        inp_dec_level1 = inp_dec_level1[:, :, :T1, :F1]
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return out_dec_level1

class Recurrent(nn.Module):
    def __init__(self, dim, num_blocks, heads, ffn_expansion_factor, _iter=1):
        super().__init__()
        self.unet = DPT(dim,
                        num_blocks, 
                        heads,
                        ffn_expansion_factor)
        self.iter = _iter
        self.conv = nn.Conv2d(dim*2, dim, 1, 1)
    
        self.concat_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, groups=dim), nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.unet(x)
            else:
                x = self.conv(x)
                x = self.concat_block(mixture + x) 
                x = self.unet(x)
        return x

from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.normalization import LayerNorm

def _get_activation_fn(activation):
    if activation == "relu":
        return FF.relu
    elif activation == "gelu":
        return FF.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class Downsample1(nn.Module):
    def __init__(self, n_feat):
        super(Downsample1, self).__init__()

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, stride=1, bias=False),
        #                           nn.PixelUnshuffle(2))
        self.body = nn.PixelUnshuffle(2)

    def forward(self, x):
        B, C, T, F = x.size()
    
        pad_h = (2 - T % 2) % 2
        pad_w = (2 - F % 2) % 2
    
        pad = nn.ReflectionPad2d((0, pad_w, 0, pad_h))
        x = pad(x)
        return self.body(x)

class Upsample1(nn.Module):
    def __init__(self, n_feat):
        super(Upsample1, self).__init__()

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, stride=1, bias=False),
        #                           nn.PixelShuffle(2))
        self.body = nn.PixelShuffle(2)

    def forward(self, x):
        return self.body(x)

class TransformerEncoderLayer(Module):

    def __init__(self, d_model, nhead, hidden_size, dim_feedforward, dropout, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of improved part
        self.lstm = LSTM(d_model, hidden_size, 1, bidirectional=True)
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_size*2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = FF.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear(self.dropout(self.activation(self.lstm(src)[0])))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SingleTransformer(nn.Module):
    """
    Container module for a single Transformer layer.
    args: input_size: int, dimension of the input feature. The input should have shape (batch, seq_len, input_size).
    """
    def __init__(self, input_size, hidden_size, dropout):
        super(SingleTransformer, self).__init__()
        self.transformer = TransformerEncoderLayer(d_model=input_size, nhead=4, hidden_size=hidden_size,
                                                   dim_feedforward=hidden_size*2, dropout=dropout)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        transformer_output = self.transformer(output.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        return transformer_output

class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out

""" CONV - (BN) - RELU - CONV - (BN) """
class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=3, reduction=False, bias=True, # 'reduction' is just for placeholder
                 norm=False, act=nn.ReLU(True), downscale=False):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1)
        )
        
        self.downscale = None
        if downscale:
            self.downscale = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=2)

    def forward(self, x):
        res = x
        out = self.body(x)
        if self.downscale is not None:
            res = self.downscale(res)
        out += res

        return out 

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y, y

class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=True,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=2 if downscale else 1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out, ca = self.body(x)
        if self.downscale:
            res = self.downConv(res)
        out += res

        if self.return_ca:
            return out, ca
        else:
            return out

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()

        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act)
            for _ in range(n_resblocks)]  # 12
        modules_body.append(ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class STFTMaskNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_groups=5, blocks_per_group=12):
        super(STFTMaskNet, self).__init__()
        
        # 多个 ResidualGroup，每个组中有多个 RCAB 块
        self.res_groups = nn.Sequential(*[
            ResidualGroup(
                Block=RCAB,
                n_resblocks=blocks_per_group,
                n_feat=base_channels,
                kernel_size=3,
                reduction=16,
                act=nn.ReLU(True),
                norm=False
            )
            for _ in range(num_groups)
        ])

    def forward(self, x):
        x = self.res_groups(x)        # 多层残差组 # (B, C, F, T)
        return x


class mDPTNetBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mDPTNetBlock, self).__init__()

        self.mha_freq =  SingleTransformer(input_size, hidden_size, dropout=0)
        self.mha_time = SingleTransformer(input_size, hidden_size, dropout=0)
    def forward(self, x):
        B, C, T, F = x.size()
        x_freq = x.permute(0, 3, 2, 1).contiguous().view(B * F, T, C)  # (B*F, T, C)
        x_freq = self.mha_freq(x_freq)
        x_freq = x_freq.view(B, F, T, C).permute(0, 3, 2, 1).contiguous()  # (B, C, T, F)
        x_time = x_freq.permute(0, 2, 3, 1).contiguous().view(B * T, F, C)  # (B*T, F, C)
        x_time = self.mha_time(x_time)
        x_time = x_time.view(B, T, F, C).permute(0, 3, 1, 2).contiguous()  # (B, C, T, F)

        return x_time 

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
        # self.downsample = Downsample

        self.down = Downsample1(out1)
        self.conv_2d = nn.Conv2d(out1*4, out1*4, kernel_size=3, stride=1, padding=1)
        self.res_groups = nn.Sequential(*[
            ResidualGroup(
                Block=RCAB,
                n_resblocks=6,
                n_feat=out1*4,
                kernel_size=3,
                reduction=16,
                act=nn.ReLU(True),
                norm=False
            )
            for _ in range(3)
        ])
        self.conv_2d_out = nn.Conv2d(out1*4, out1*4, kernel_size=3, stride=1, padding=1)
        self.up = Upsample1(int(out1*2**2)) 
        
        # self.relu1 = nn.ReLU()

        if block_type == 'mDPTNet':
            self.blocks = nn.ModuleList([mDPTNetBlock(out1, hidden_dim) for _ in range(num_blocks)])

        self.conv2 = nn.Conv2d(out1, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(out1, out_channels, kernel_size=1)

        # self.down1_2 = Downsample(out1)
        # if block_type == 'mDPTNet':
        #     self.blocks1 = nn.ModuleList([mDPTNetBlock(out1*2, hidden_dim) for _ in range(4)])
        # self.up2_1 = Upsample(int(out1*2**1)) 
    def forward(self, x):

        x = self.ln(x)
        x = self.conv1(x)  # 64
        prme_x = x
        for block in self.blocks:
            x = block(x)
        x = self.conv2(x)

        x1 = self.down(prme_x)
        x1 = self.conv_2d(x1)# torch.Size([2, 256, 126, 65])
        # print(x1.shape)  # torch.Size([2, 256, 126, 65])
        x2 = self.res_groups(x1)
        x2 += x1
        x2 = self.conv_2d_out(x2)
        x3 = self.up(x2)
        x3 = x3[:, :, :x.shape[2], :x.shape[3]]
        # print(x3.shape)
        x3 = self.conv3(x3)

        return x, x3


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
##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        B, C, T, F = x.size()
    
        pad_h = (2 - T % 2) % 2
        pad_w = (2 - F % 2) % 2
    
        pad = nn.ReflectionPad2d((0, pad_w, 0, pad_h))
        x = pad(x)
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class CMA_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(CMA_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_channel, out_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        _, _, h, w = rgb.size()

        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)

        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(
            -2, -1
        )
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))

        attn = torch.matmul(q, k) * self.scale
        m = attn.softmax(dim=-1)

        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(
            -2, -1
        )
        z = torch.matmul(m, v)
        z = z.view(z.size(0), h, w, -1)
        z = z.permute(0, 3, 1, 2).contiguous()

        output = rgb + self.conv4(z)

        return output


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

class LipSTFT(nn.Module):
    def __init__(self, segment_size=256, hop_size=128, fft_bins=129, center=True):
        super().__init__()
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.center = center
        self.freq_proj = nn.Linear(segment_size, fft_bins)

    def forward(self, x):
        """
        x: [B, C, T] -- e.g., [4, 64, 32000]
        Return: [B, C', F, T'] -- e.g., [4, 2, 129, 251]
        """
        B, C, T = x.shape

        # 2. padding
        if self.center:
            pad = self.segment_size // 2
            x = FF.pad(x, (pad, pad), mode='constant', value=0)

        # 3. unfold 滑窗分段
        x = x.unfold(dimension=2, size=self.segment_size, step=self.hop_size)  # [B, C', T', segment_size]
        # print(x.shape)
        # x = x.permute(0, 1, 3, 2)  # [B, C', segment_size, T']
        # print(x.shape)  # torch.Size([4, 64, 256, 251])

        # 4. 类似 FFT：线性层模拟频率变换
        x = self.freq_proj(x)  # [B, C', F, T']

        return x


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
        return torch.where(mask, mask_embed, x)

class GateNet(nn.Module):
    def __init__(self, in_channels=2, mid_channels=8):
        super(GateNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出范围在 [0, 1]
        )

    def forward(self, x):
        return self.net(x)  # 输出 shape: (B, 1, F, T)

##########################################################################
##---------- Restormer -----------------------
class DPT_1d(nn.Module):
    def __init__(self, 
        win = 256,
        hop_length = 128,
        encoder_channels = 128,
        inp_channels=2, 
        out_channels=2, 
        dim = 64,
        num_blocks = [1,1,1,1], 
        num_refinement_blocks = 1,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        vpre_channels=512,
        vin_channels=64,
        num_source = 1,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(DPT_1d, self).__init__()
        self.num_source = num_source
        self.win = win
        self.hop_length = hop_length

        # self.pre_v = multi_OverlapPatchEmbed_v(vpre_channels, vin_channels)
        self.pre_v = multi_OverlapPatchEmbed_v(vpre_channels, dim)  # 256
        self.v_stft = LipSTFT(win, hop_length)
        self.stft_encoder = STFTEncoder(win, hop_length, bias)

        self.patch_embed = multi_OverlapPatchEmbed(inp_channels, dim)  # 256

        # self.masker_10 = HubertTimeMask2D(channels=dim*4, mask_prob=0.01, mask_length=10) # x 是 shape (B, C, T, F)  # 10% mask
        # self.masker_30 = HubertTimeMask2D(channels=dim*4, mask_prob=0.03, mask_length=10) # x 是 shape (B, C, T, F)  # 30% mask
        # self.masker_45 = HubertTimeMask2D(channels=dim*4, mask_prob=0.065, mask_length=10) # x 是 shape (B, C, T, F)  # 45% mask

        self.blocks_inter = DualStreamSeq(*[MuGIBlock(dim*4) for _ in range(2)])

        # self.conv = nn.Conv2d(dim*8, dim*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv = nn.Conv2d(dim*8, dim*4, kernel_size=1, stride=1, bias=False)
        self.extractor = Extractor(dim*4, dim, dim*4, dim*2, num_blocks=6, block_type="mDPTNet")

        self.gate_net = GateNet(dim*8, dim*4)
        
        # self.sm = Recurrent(dim, num_blocks, heads, ffn_expansion_factor)        
        # self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        ###########################
        if self.num_source == 1:
            self.decoder = nn.Conv2d(dim*4, 2, kernel_size=1)
            # self.output = nn.Conv2d(int(dim*2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
            self.stft_decoder = STFTDecoder_1(win, hop_length, in_chan=dim, n_src=num_source, kernel_size=3, stride=1, bias=bias)
        elif self.num_source == 2:
            self.output1 = nn.Conv2d(int(dim**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
            self.output2 = nn.Conv2d(int(dim**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
            self.stft_decoder12 = STFTDecoder(win, hop_length, in_chan=dim, n_src=num_source, kernel_size=3, stride=1, bias=bias)
        else:
            raise ValueError("Unsupported number of sources: {}".format(self.num_source))
        
    @staticmethod
    def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
        from . import get
        conf = torch.load(pretrained_model_conf_or_path, map_location="cpu")
        # Attempt to find the model and instantiate it.
        # import pdb; pdb.set_trace()
        model_class = get(conf["model_name"])
        # model_class = get("AVFRCNN")
        model = model_class(*args, **kwargs)
        model.load_state_dict(conf["state_dict"])
        # model = BaseAVModel.load_state_dict_in(model, conf["state_dict"])
        return model

    def forward(self, x, v):
        
        v0 = FF.interpolate(v, size=x.shape[-1], mode='linear', align_corners=False)  # 4*512 *32000
        # v0_segments, v0_rest = self.split_feature(v0, 256)  # torch.Size([4, 512, 256, 252])
        v0_stft = self.v_stft(v0)  # torch.Size([4, 512, 251, 129])
        v0 = self.pre_v(v0_stft)  # torch.Size([4, 256, 251, 129])

        feature_map = self.stft_encoder(x)  # torch.Size([4, 2, 251, 129])
        inp_enc_level1 = self.patch_embed(feature_map) # torch.Size([4, 256, 251, 129])

        # # mask
        # inp_enc_level1_mask_10 = self.masker_10(inp_enc_level1)
        # inp_enc_level1_mask_30 = self.masker_30(inp_enc_level1)
        # inp_enc_level1_mask_45 = self.masker_45(inp_enc_level1)

        # inp_enc_level1_10, v0_10 = self.blocks_inter(inp_enc_level1_mask_10, v0)
        # inp_enc_level1_30, v0_30 = self.blocks_inter(inp_enc_level1_mask_30, v0)
        # inp_enc_level1_45, v0_45 = self.blocks_inter(inp_enc_level1_mask_45, v0)

        inp_enc_level1_pre, v0_pre = self.blocks_inter(inp_enc_level1, v0)

        # inp_enc_level_all = inp_enc_level1_pre + inp_enc_level1_10 + inp_enc_level1_30 + inp_enc_level1_45
        # v0_all = v0_pre + v0_10 + v0_30 + v0_45
        inp_enc_level_all = inp_enc_level1_pre
        v0_all = v0_pre

        inp_enc_level1_mix =  torch.nn.functional.relu(self.conv(torch.cat([inp_enc_level_all, v0_all], dim=1)))
        # print(inp_enc_level1.shape)
        mask_1d, mask_2d = self.extractor(inp_enc_level1_mix)
        # print(mask_2d.shape, mask_1d.shape)
        gate = self.gate_net(torch.cat([mask_1d, mask_2d], dim=1))
        fused_mask = gate * mask_1d + (1 - gate) * mask_2d
        masks = torch.nn.functional.relu(fused_mask)
        masked_output = inp_enc_level1 * masks

        dec_output = self.decoder(masked_output)
        stft_out =  dec_output
        # out_dec_level1 =self.sm(inp_enc_level1) 
        # out_dec_level1 = self.refinement(out_dec_level1)
        source = self.stft_decoder(dec_output, x)

        return stft_out, source


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    # # # 实例化模型
    # lip = torch.randn(4, 64, 32000)  # [B, C, T]
    # model = LipSTFT(segment_size=256, hop_size=128, fft_bins=129)
    # out = model(lip)  # -> [4, 2, 129, 251]
    # print(out.shape)  # ✅ torch.Size([4, 2, 129, 251])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPT_1d().to(device)
    input_signal = torch.randn(2,1,32000).to(device)  # 输入图像的形状 (C, H, W)
    v = torch.randn(2,512,50).to(device)
    # summary(model, input_signal, device='cuda')
    import time
    start_time = time.time()
    out1,out2 = model(input_signal, v)
    end_time = time.time()
    print(out1.shape)
    print(out2.shape)

    # 输出信息
    print("模型参数总数:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("输出形状:", out1.shape)
    print(f"前向传播耗时: {end_time - start_time:.4f} 秒")
    # model1 = STFTMaskNet()
    # x = torch.randn(1, 1, 256, 128)  # batch=8, freq bins=256, time steps=128
    # mask = model1(x) 
    # print("模型参数总数:", sum(p.numel() for p in model1.parameters() if p.requires_grad))  
    # print(mask.shape)              # 输出 (8, 1, 256, 128)
