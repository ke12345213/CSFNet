import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==== 1D Pixel shuffle & unshuffle ====

class PixelUnshuffle1D(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.r = downscale_factor

    def forward(self, x):
        B, C, T = x.shape
        r = self.r
        assert T % r == 0
        out_t = T // r
        x = x.view(B, C, out_t, r)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * r, out_t)
        return x

class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        B, C, T = x.shape
        r = self.r
        assert C % r == 0
        out_c = C // r
        x = x.view(B, out_c, r, T)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, out_c, T * r)
        return x

# ==== LayerNorm ====

class LayerNorm1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        # print(x.shape)
        return x.permute(0, 2, 1)

# ==== FeedForward ====

class FeedForward1D(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.in_proj = nn.Conv1d(dim, hidden * 2, 1)
        self.dwconv = nn.Conv1d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.out_proj = nn.Conv1d(hidden, dim, 1)

    def forward(self, x):
        x = self.in_proj(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.out_proj(x)

# ==== Attention ====

class Attention1D(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.temp = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv1d(dim, dim * 3, 1)
        self.qkv_dw = nn.Conv1d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3)
        self.out_proj = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        B, C, T = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (h c) t -> b h c t', h=self.heads)
        k = rearrange(k, 'b (h c) t -> b h c t', h=self.heads)
        v = rearrange(v, 'b (h c) t -> b h c t', h=self.heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temp
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h c t -> b (h c) t')
        return self.out_proj(out)

# ==== Transformer Block ====

class TransformerBlock1D(nn.Module):
    def __init__(self, dim, heads, ffn_expansion_factor):
        super().__init__()
        self.norm1 = LayerNorm1D(dim)
        self.attn = Attention1D(dim, heads)
        self.norm2 = LayerNorm1D(dim)
        self.ffn = FeedForward1D(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ==== Down / Up ====

class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            PixelUnshuffle1D(2),  
            nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.body(x)

class Upsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            PixelShuffle1D(2)
        )

    def forward(self, x):
        return self.body(x)

# ==== Restormer 1D ====

class restormer_1d(nn.Module):
    def __init__(self, 
        inp_channels=1,
        out_channels=1,
        dim=64,
        num_blocks=[1,2,2,4],
        num_blocks_v=[1,2,2,4],
        num_refinement_blocks = 4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        vpre_channels = 512,
        vin_channels = 64,
        num_sources=2,
    ):
        super().__init__()

        self.num_sources = num_sources

        # a
        self.patch_embed = nn.Conv1d(inp_channels, dim, kernel_size=3, padding=1)
        # v
        self.pre_v = nn.Conv1d(vpre_channels, vin_channels, kernel_size=3, padding=1)

        self.encoder1 = nn.Sequential(*[TransformerBlock1D(dim, heads[0], ffn_expansion_factor) for _ in range(num_blocks[0])])
        self.encoder1_v = nn.Sequential(*[TransformerBlock1D(vin_channels, heads[0], ffn_expansion_factor) for _ in range(num_blocks_v[0])])
        self.cross_1 = nn.Conv1d(vin_channels+dim, dim, kernel_size=3, padding=1)
        self.down1 = Downsample1D(dim)

        self.encoder2 = nn.Sequential(*[TransformerBlock1D(dim*2, heads[1], ffn_expansion_factor) for _ in range(num_blocks[1])])
        self.encoder2_v = nn.Sequential(*[TransformerBlock1D(vin_channels*2, heads[1], ffn_expansion_factor) for _ in range(num_blocks_v[1])])
        self.cross_2 = nn.Conv1d(vin_channels*2+dim*2, dim*2, kernel_size=3, padding=1)
        self.down2 = Downsample1D(dim*2)

        self.encoder3 = nn.Sequential(*[TransformerBlock1D(dim*4, heads[2], ffn_expansion_factor) for _ in range(num_blocks[2])])
        self.encoder3_v = nn.Sequential(*[TransformerBlock1D(vin_channels*4, heads[2], ffn_expansion_factor) for _ in range(num_blocks_v[2])])
        self.cross_3 = nn.Conv1d(vin_channels*4+dim*4, dim*4, kernel_size=3, padding=1)
        self.down3 = Downsample1D(dim*4)

        self.conv = nn.Conv1d(dim*16, dim*8, kernel_size=3, padding=1)
        self.latent = nn.Sequential(*[TransformerBlock1D(dim*8, heads[3], ffn_expansion_factor) for _ in range(num_blocks[3])])
        self.latent_v = nn.Sequential(*[TransformerBlock1D(vin_channels*8, heads[3], ffn_expansion_factor) for _ in range(num_blocks_v[3])])

        self.up3 = Upsample1D(dim*8)
        self.reduce3 = nn.Conv1d(dim*8, dim*4, 1)
        self.reduce3_v = nn.Conv1d(vin_channels*8, vin_channels*4, 1)
        self.decoder3 = nn.Sequential(*[TransformerBlock1D(dim*4, heads[2], ffn_expansion_factor) for _ in range(num_blocks[2])])
        self.decoder3_v = nn.Sequential(*[TransformerBlock1D(vin_channels*4, heads[2], ffn_expansion_factor) for _ in range(num_blocks_v[2])])
        self.cross_4 = nn.Conv1d(vin_channels*4+dim*4, dim*4, kernel_size=3, padding=1)

        self.up2 = Upsample1D(dim*4)
        self.reduce2 = nn.Conv1d(dim*4, dim*2, 1)
        self.reduce2_v = nn.Conv1d(vin_channels*4, vin_channels*2, 1)
        self.decoder2 = nn.Sequential(*[TransformerBlock1D(dim*2, heads[1], ffn_expansion_factor) for _ in range(num_blocks[1])])
        self.decoder2_v = nn.Sequential(*[TransformerBlock1D(vin_channels*2, heads[1], ffn_expansion_factor) for _ in range(num_blocks_v[1])])
        self.cross_5 = nn.Conv1d(vin_channels*2+dim*2, dim*2, kernel_size=3, padding=1)

        self.up1 = Upsample1D(dim*2)
        self.decoder1 = nn.Sequential(*[TransformerBlock1D(dim*2, heads[0], ffn_expansion_factor) for _ in range(num_blocks[0])])
        self.decoder1_v = nn.Sequential(*[TransformerBlock1D(vin_channels*2, heads[0], ffn_expansion_factor) for _ in range(num_blocks_v[0])])
        self.cross_6 = nn.Conv1d(vin_channels*2+dim*2, dim*2, kernel_size=3, padding=1)
        self.refinement = nn.Sequential(*[TransformerBlock1D(dim*2, heads[0], ffn_expansion_factor) for _ in range(num_refinement_blocks)])
        mask_conv = nn.Conv1d(dim*2, num_sources * dim, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        self.mask_nl_class = nn.ReLU()

        # self.output = nn.Conv1d(dim*2, out_channels, 3, padding=1)
        self.output = nn.ConvTranspose1d(dim, out_channels, 3, padding=1)

    def forward(self, x, v):


        # 两种策略，第一种是先插值到4000，然后逐步向上采样，第二种是先卷积到64，然后插值到32000，然后下采样
        # 第一种
        # print(x.shape)  # torch.Size([4, 32000])
        # print(v.shape)  # torch.Size([4, 512, 50])
        x = x.unsqueeze(1)
        v0 = F.interpolate(v, size=x.shape[-1], mode='linear', align_corners=False)
        v1 = self.pre_v(v0)  # 64 32000

        x1 = self.patch_embed(x)  # encoder
        s = x1.clone()
        # print(s.shape)# ([2, 64, 32000])
        
        v1 = self.encoder1_v(v1)# 64 32000
        x1 = self.encoder1(x1)  # 64 32000
        
        x_v_1 = self.cross_1(torch.cat([v1, x1], dim=1))

        x2 = self.encoder2(self.down1(x1 + x_v_1)) # 128 16000  # 这个残差不对
        # print(x2.shape)
        v2 = self.encoder2_v(self.down1(v1))  # 64 32000

        x_v_2 = self.cross_2(torch.cat([v2, x2], dim=1))
        x3 = self.encoder3(self.down2(x2 + x_v_2)) # 256 8000
        # print(x3.shape)
        v3 = self.encoder3_v(self.down2(v2)) # 256 8000
        x_v_3 = self.cross_3(torch.cat([v3, x3], dim=1))  # 256 8000

        x4 = self.latent(self.down3(x3 + x_v_3))  # 512 4000

        v4 = F.interpolate(v, size=x4.shape[-1], mode='linear', align_corners=False)
        v4 = self.latent_v(self.conv(torch.cat([self.down3(v3), v4], dim=1))) # 512 4000

        uv3 = self.up3(v4) # 256 8000
        
        

        d3 = self.up3(x4) # 256 8000
        d3 = self.reduce3(torch.cat([d3, x3], dim=1))
        uv3 = self.reduce3_v(torch.cat([uv3, v3], dim=1))
        uv3 = self.decoder3_v(uv3)
        d3 = self.decoder3(d3)
        d_nv3 = self.cross_4(torch.cat([d3, uv3], dim=1))

        d2 = self.up2(d3+d_nv3)
        d2 = self.reduce2(torch.cat([d2, x2], dim=1))
        uv2 = self.up2(uv3)# 128 16000
        uv2 = self.reduce2_v(torch.cat([uv2, v2], dim=1))
        uv2 = self.decoder2_v(uv2)
        d2 = self.decoder2(d2)
        d_nv2 = self.cross_5(torch.cat([d2, uv2], dim=1))

        d1 = self.up1(d2+d_nv2) # 64 
        # print(d1.shape)
        d1 = torch.cat([d1, x1], dim=1)
        uv1 = self.up1(uv2) # 64 32000
        # uv1 = self.reduce1_v(torch.cat([uv1, v1], dim=1)) # 64 32000
        uv1 = self.decoder1_v(torch.cat([uv1, v1], dim=1)) # 128 32000
        # print(d1.shape)
        d1 = self.decoder1(d1) # 128 32000
        d_nv1 = self.cross_6(torch.cat([d1, uv1], dim=1))  # 128 32000
        d0= self.refinement(d1+d_nv1)

        d = self.mask_net(d0)
        d = d.view(x.shape[0], self.num_sources, -1, x.shape[-1])
        d = self.mask_nl_class(d)
        # print(d.shape) # torch.Size([2, 2, 32000, 64])

        d = d * s.unsqueeze(1)
        B, N, C, T = d.shape
        d = d.view(B * N, C, T)             # 合并 batch 和 speaker  
        out = self.output(d) # decoder
        out = out.view(B, N, T)
        return out


if __name__ == '__main__':
    model = restormer_1d(
        inp_channels=1, 
        out_channels=1, 
        dim = 64,
        num_blocks = [2,2,2,2], 
        num_blocks_v = [1,1,1,1], 
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
    )
    x = torch.randn(2, 1, 32000)  # (B, C, T)
    v = torch.randn(2, 512, 50)  # (B, C, T)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)
    y = v.to(device)
    # 计时前向传播
    import time
    start_time = time.time()
    out = model(x, y)
    end_time = time.time()

    # 输出信息
    print("模型参数总数:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("输出形状:", out.shape)
    print(f"前向传播耗时: {end_time - start_time:.4f} 秒")
