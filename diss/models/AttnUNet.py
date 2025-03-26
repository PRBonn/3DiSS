import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride, padding=padding),
            nn.GroupNorm(32, outc),
            nn.SiLU(True)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(inc, outc, kernel_size=ks, stride=stride, dilation=dilation, padding=padding),
            nn.GroupNorm(32, outc),
            nn.SiLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride, padding=padding),
            nn.GroupNorm(32, outc),
            nn.SiLU(inplace=True),
            nn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1, padding=1),
            nn.GroupNorm(32, outc)
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride, padding=0),
                nn.GroupNorm(32, outc)
            )

        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.silu(self.net(x) + self.downsample(x))
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, y=None):
        x = x.permute(0,2,3,4,1)
        in_shape = x.shape
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if y is not None:
            y = y.permute(0,2,3,4,1)
            y = y.reshape(y.shape[0], -1, y.shape[-1])


        for attn, ff in self.layers:
            x = attn(x, y, False if y is None else True) + x
            x = ff(x) + x
        return self.norm(x).reshape(in_shape).permute(0,4,1,2,3)

class LatentDiffuser(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.dim = kwargs.get('latent_dim', 256)
        self.mid_attn = kwargs.get('mid_attn', False)
        self.D = kwargs.get('D', 3)
        cs = [self.dim, int(1.5*self.dim), int(2*self.dim), int(1.5*self.dim), self.dim]

        self.stem = nn.Sequential(
            nn.Conv3d(cs[0], cs[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(cs[0]),
            nn.SiLU(True),
        )

        self.t_proj1  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(True),
            nn.Linear(cs[1], cs[4]),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=3, stride=(2,2,1), dilation=1, padding=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, padding=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, padding=1),
        )

        self.t_proj2  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(True),
            nn.Linear(cs[1], cs[1]),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=3, stride=(2,2,1), dilation=1, padding=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, padding=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, padding=1),
        )

        self.t_mid_proj  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(True),
            nn.Linear(cs[1], cs[2]),
        )

        if self.mid_attn:
            self.mid = Transformer(cs[2], 16, 8, cs[2], int(cs[2]/2), True)
        else:
            self.mid = nn.Sequential(
                BasicConvolutionBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(cs[2], 2*cs[2], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(2*cs[2], 2*cs[2], ks=3, stride=1, dilation=1, padding=1),
                BasicConvolutionBlock(2*cs[2], 2*cs[2], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(2*cs[2], cs[2], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, padding=1),
            )

        self.t_up_proj1  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(inplace=True),
            nn.Linear(cs[1], cs[2]),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=(4,4,3), stride=(2,2,1), dilation=1, padding=1),
            nn.Sequential(
                ResidualBlock(cs[3]+cs[1], cs[3], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, padding=1),
            )
        ])

        self.t_up_proj2  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(inplace=True),
            nn.Linear(cs[1], cs[3]),
        )

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=(4,4,3), stride=(2,2,1), dilation=1, padding=1),
            nn.Sequential(
                ResidualBlock(cs[4]+cs[0], cs[4], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, padding=1),
            )
        ])

        self.last  = nn.Sequential(
            nn.Linear(cs[4], 2*cs[4]),
            nn.SiLU(inplace=True),
            nn.Linear(2*cs[4], cs[0]),
        )
 
    def get_timestep_embedding(self, timesteps):
        half_dim = self.dim / 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(torch.device('cuda'))
        emb = timesteps * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], int(2*self.dim / 2)])
        return emb

    def forward(self, ae_latent, t, condition=None):
        temp_emb = self.get_timestep_embedding(t[:,None])

        x0 = self.stem(ae_latent)

        t1 = self.t_proj1(temp_emb)[...,None,None,None]
        x1 = self.stage1(x0 * t1) 

        t2 = self.t_proj2(temp_emb)[...,None,None,None]
        x2 = self.stage2(x1 * t2)

        t_attn = self.t_mid_proj(temp_emb)[...,None,None,None]
        x_mid = self.mid(x2 * t_attn)

        t_up_1 = self.t_up_proj1(temp_emb)[...,None,None,None]
        y1 = self.up1[0](x_mid * t_up_1)
        y1 = torch.cat((y1, x1), 1)
        y1 = self.up1[1](y1)
 
        t_up_2 = self.t_up_proj2(temp_emb)[...,None,None,None]
        y2 = self.up2[0](y1 * t_up_2)
        y2 = torch.cat((y2, x0), 1)
        y2 = self.up2[1](y2)

        y3 = self.last(y2.permute(0,2,3,4,1)).permute(0,4,1,2,3)

        return y3 

class LatentCondDiffuser(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.dim = kwargs.get('latent_dim', 256)
        self.mid_attn = kwargs.get('mid_attn', False)
        self.D = kwargs.get('D', 3)
        self.cond_diff = kwargs.get('cond_diff', False)
        cs = [self.dim, int(1.5*self.dim), int(2*self.dim), int(1.5*self.dim), self.dim]

        self.stem = nn.Sequential(
            nn.Conv3d(cs[0], cs[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(cs[0]),
            nn.SiLU(True),
        )

        self.t_proj1  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(True),
            nn.Linear(cs[1], cs[4]),
        )

        self.cond_proj1 = None if not self.cond_diff else nn.Sequential(
                nn.Conv3d(cs[0], cs[1], kernel_size=3, stride=1, dilation=1, padding=1),
                nn.SiLU(True),
                nn.Conv3d(cs[1], cs[0], kernel_size=3, stride=1, dilation=1, padding=1),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=3, stride=(2,2,1), dilation=1, padding=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, padding=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, padding=1),
        )

        self.t_proj2  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(True),
            nn.Linear(cs[1], cs[1]),
        )

        self.cond_proj2 = None if not self.cond_diff else nn.Sequential(
                nn.Conv3d(cs[0], cs[1], kernel_size=3, stride=(2,2,1), dilation=1, padding=1),
                nn.SiLU(True),
                nn.Conv3d(cs[1], cs[1], kernel_size=3, stride=1, dilation=1, padding=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=3, stride=(2,2,1), dilation=1, padding=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, padding=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, padding=1),
        )

        self.t_mid_proj  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(True),
            nn.Linear(cs[1], cs[2]),
        )

        self.cond_proj_mid = None if not self.cond_diff else nn.Sequential(
                nn.Conv3d(cs[0], cs[1], kernel_size=3, stride=(4,4,1), dilation=1, padding=1),
                nn.SiLU(True),
                nn.Conv3d(cs[1], cs[2], kernel_size=3, stride=1, dilation=1, padding=1),
        )

        self.merge_cond_mid = None if not self.cond_diff else nn.MultiheadAttention(cs[2], 4, batch_first=True)

        if self.mid_attn:
            self.mid = Transformer(cs[2], 16, 8, cs[2], int(cs[2]/2), True)
        else:
            self.mid = nn.Sequential(
                BasicConvolutionBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(cs[2], 2*cs[2], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(2*cs[2], 2*cs[2], ks=3, stride=1, dilation=1, padding=1),
                BasicConvolutionBlock(2*cs[2], 2*cs[2], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(2*cs[2], cs[2], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, padding=1),
            )

        self.t_up_proj1  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(inplace=True),
            nn.Linear(cs[1], cs[2]),
        )

        self.cond_proj_up1 = None if not self.cond_diff else nn.Sequential(
                nn.Conv3d(cs[0], cs[1], kernel_size=3, stride=(4,4,1), dilation=1, padding=1),
                nn.SiLU(True),
                nn.Conv3d(cs[1], cs[2], kernel_size=3, stride=1, dilation=1, padding=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=(4,4,3), stride=(2,2,1), dilation=1, padding=1),
            nn.Sequential(
                ResidualBlock(cs[3]+cs[1], cs[3], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, padding=1),
            )
        ])

        self.t_up_proj2  = nn.Sequential(
            nn.Linear(cs[0], cs[1]),
            nn.SiLU(inplace=True),
            nn.Linear(cs[1], cs[3]),
        )

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=(4,4,3), stride=(2,2,1), dilation=1, padding=1),
            nn.Sequential(
                ResidualBlock(cs[4]+cs[0], cs[4], ks=3, stride=1, dilation=1, padding=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, padding=1),
            )
        ])

        self.cond_proj_up2 = None if not self.cond_diff else nn.Sequential(
                nn.Conv3d(cs[0], cs[1], kernel_size=3, stride=(2,2,1), dilation=1, padding=1),
                nn.SiLU(True),
                nn.Conv3d(cs[1], cs[3], kernel_size=3, stride=1, dilation=1, padding=1),
        )

        self.last  = nn.Sequential(
            nn.Linear(cs[4], 2*cs[4]),
            nn.SiLU(inplace=True),
            nn.Linear(2*cs[4], cs[0]),
        )

    def get_timestep_embedding(self, timesteps):
        half_dim = self.dim / 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(torch.device('cuda'))
        # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
        emb = timesteps * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # zero pad
            # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], int(2*self.dim / 2)])
        return emb

    def merge_cond(self, x, cond, conv_proj, attn=None):
        cond = conv_proj(cond)

        if attn is not None:
            q_ = x.permute(0,2,3,4,1).view(x.shape[0],-1, x.shape[1])
            vk_ = cond.permute(0,2,3,4,1).view(cond.shape[0],-1, cond.shape[1])
            merged, _ = attn(q_, vk_, vk_)
            x = merged.permute(0,2,1).view(x.shape)
        else:
            x = x * cond

        return x

    def forward(self, ae_latent, t, cond=None):
        temp_emb = self.get_timestep_embedding(t[:,None])

        x0 = self.stem(ae_latent)
        if self.cond_diff:
            x0 = self.merge_cond(x0, cond, self.cond_proj1, None)

        t1 = self.t_proj1(temp_emb)[...,None,None,None]
        x1 = self.stage1(x0 * t1)
        if self.cond_diff:
            x1 = self.merge_cond(x1, cond, self.cond_proj2, None)

        t2 = self.t_proj2(temp_emb)[...,None,None,None]
        x2 = self.stage2(x1 * t2)
        if self.cond_diff:
            x2 = self.merge_cond(x2, cond, self.cond_proj_mid, self.merge_cond_mid)

        t_attn = self.t_mid_proj(temp_emb)[...,None,None,None]
        x_mid = self.mid(x2 * t_attn)
        if self.cond_diff:
            x_mid = self.merge_cond(x_mid, cond, self.cond_proj_up1, None)

        t_up_1 = self.t_up_proj1(temp_emb)[...,None,None,None]
        y1 = self.up1[0](x_mid * t_up_1)
        y1 = torch.cat((y1, x1), 1)
        y1 = self.up1[1](y1)
        if self.cond_diff:
            y1 = self.merge_cond(y1, cond, self.cond_proj_up2, None)

        t_up_2 = self.t_up_proj2(temp_emb)[...,None,None,None]
        y2 = self.up2[0](y1 * t_up_2)
        y2 = torch.cat((y2, x0), 1)
        y2 = self.up2[1](y2)

        y3 = self.last(y2.permute(0,2,3,4,1)).permute(0,4,1,2,3)

        return y3
