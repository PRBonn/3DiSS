import time
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

__all__ = ['MinkUNet']

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(inc,
                                 outc,
                                 kernel_size=ks,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(outc,
                                 outc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=1,
                                 dimension=D),
            ME.MinkowskiBatchNorm(outc)
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                ME.MinkowskiConvolution(inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
                ME.MinkowskiBatchNorm(outc)
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class MinkEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        #cs = [32, 64, 128, 256, 256, 128, 96]
        cs = [32, 64, 128, 128]
        cs = [int(cr * x) for x in cs]
        self.train_range = kwargs.get('train_range', [[-25.6, 25.6], [-25.6, 25.6], [-2.2, 4.2]])
        self.val_range = kwargs.get('val_range', [[-25.6, 25.6], [-25.6, 25.6], [-2.2, 4.2]])
        self.resolution = kwargs.get('resolution', 0.1)
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=(2,2,1), dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D)
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def get_target_prune(self, x, target_key, kernel_size=1):
        target_prune = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        cm = x.coordinate_manager
        strided_target_key = cm.stride(
            target_key,
            x.tensor_stride[0],
        )
        kernel_map = cm.kernel_map(
            x.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )

        for k, curr_in in kernel_map.items():
            target_prune[curr_in[0].long()] = 1

        return target_prune

    def range_to_voxel(self, sparse_tensor, scene_range):
        b = len(sparse_tensor.C[:,0].unique())
        f = sparse_tensor.F.shape[-1]
        w = int((scene_range[0][1] - scene_range[0][0]) / (self.resolution * sparse_tensor.tensor_stride[0]))# + 4
        d = int((scene_range[1][1] - scene_range[1][0]) / (self.resolution * sparse_tensor.tensor_stride[1]))# + 4
        h = int((scene_range[2][1] - scene_range[2][0]) / (self.resolution * sparse_tensor.tensor_stride[2]))# + 4

        return torch.Size((b,f,w,d,h))

    def forward(self, x, training=True, encoder=False):
        ################################# Encoder ################################################
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        dense_shape = self.range_to_voxel(x3, self.train_range if training else self.val_range)
        dense_x, _, stride = x3.dense(dense_shape)
        return dense_x

class MinkUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        out_channels = kwargs.get('out_channels', 3)
        #cs = [32, 64, 128, 256, 256, 128, 96]
        cs = [32, 64, 128, 128, 128, 128, 64]
        cs = [int(cr * x) for x in cs]
        self.train_range = kwargs.get('train_range', [[-25.6, 25.6], [-25.6, 25.6], [-2.2, 4.2]])
        self.val_range = kwargs.get('val_range', [[-25.6, 25.6], [-25.6, 25.6], [-2.2, 4.2]])
        self.resolution = kwargs.get('resolution', 0.1)
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.latent_stride = torch.tensor([8, 8, 4], dtype=torch.int32)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=(4,4,2), dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D)
        )

        self.dense_projection = nn.Sequential(
            ME.MinkowskiLinear(cs[3], cs[2]),
            ME.MinkowskiLeakyReLU(0.1, inplace=True),
            ME.MinkowskiLinear(cs[2], cs[3]),
        )

        self.up0_prune_class = nn.Sequential(
                ME.MinkowskiConvolution(cs[3], 1, kernel_size=3, bias=True, stride=1, dimension=3),
                ME.MinkowskiSigmoid(),
            )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[4], cs[4], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.up1_prune_class = nn.Sequential(
                ME.MinkowskiConvolution(cs[4], 1, kernel_size=3, bias=True, stride=1, dimension=3),
                ME.MinkowskiSigmoid(),
            )

        # stride=4 doesn't work for the generative conv, therefore we have two deconv with stride=2
        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=(2,2,1), D=self.D),
            nn.Sequential(
                ResidualBlock(cs[5], cs[5], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.up2_prune_class = nn.Sequential(
                ME.MinkowskiConvolution(cs[5], 1, kernel_size=3, bias=True, stride=1, dimension=3),
                ME.MinkowskiSigmoid(),
            )

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, D=self.D),
            nn.Sequential(
                ResidualBlock(cs[6], cs[6], ks=3, stride=1,
                              dilation=1, D=self.D),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

        self.up3_prune_class = nn.Sequential(
                ME.MinkowskiConvolution(cs[6], 1, kernel_size=3, bias=True, stride=1, dimension=3),
                ME.MinkowskiSigmoid(),
            )

        self.pruning = ME.MinkowskiPruning()

        self.last  = nn.Sequential(
            ME.MinkowskiLinear(cs[6], 20),
            ME.MinkowskiLeakyReLU(0.1, inplace=True),
            ME.MinkowskiLinear(20, out_channels),
            ME.MinkowskiSigmoid(),
        )

        self.hidden = nn.Sequential(
            nn.Conv3d(cs[3], cs[2], kernel_size=1, stride=1, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(cs[2], cs[3], kernel_size=1, stride=1, dilation=1),
        )

        self.mean_head = nn.Sequential(
            nn.Conv3d(cs[3], cs[2], kernel_size=1, stride=1, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(cs[2], cs[3], kernel_size=1, stride=1, dilation=1),
        )

        self.logvar_head = nn.Sequential(
            nn.Conv3d(cs[3], cs[2], kernel_size=1, stride=1, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(cs[2], cs[3], kernel_size=1, stride=1, dilation=1),
        )

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def get_target_prune(self, x, target_key, kernel_size=1):
        target_prune = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        cm = x.coordinate_manager
        strided_target_key = cm.stride(
            target_key,
            x.tensor_stride,
        )
        kernel_map = cm.kernel_map(
            x.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )

        for k, curr_in in kernel_map.items():
            target_prune[curr_in[0].long()] = 1

        return target_prune

    def sparse_to_dense(self, dense_feats, target_coords=None):
        dense_feats = dense_feats.permute(0,2,3,4,1)
        b, w, d, h, f = dense_feats.shape
        dense_feats = dense_feats.reshape(-1, f)

        # build new grid
        grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(w),torch.arange(d),torch.arange(h))
        grid_coords = torch.cat((grid_x[...,None], grid_y[...,None], grid_z[...,None]), -1).reshape(-1,3)
        batched_coords = ME.utils.batched_coordinates([grid_coords]*b, dtype=torch.float32, device=torch.device('cuda'))

        # from the grid we multiply by the stride to have the strided voxel coords from the original input tensor
        batched_coords[:,1:] *= self.latent_stride.cuda()

        dense_tensor = ME.SparseTensor(
                features=dense_feats,
                coordinates=batched_coords,
                tensor_stride=self.latent_stride.cuda(),
                device=torch.device('cuda'),
            )

        if target_coords is not None:
            cm = dense_tensor.coordinate_manager
            x_target, _ = cm.insert_and_map(target_coords, string_id='target')
        else:
            x_target = None

        return dense_tensor, x_target

    def forward_decoder(self, diff_latent, target_coords=None, training=False):
        # used to generate novel scenes
        # decode a diffused latent
        diff_latent, target_coord_map = self.sparse_to_dense(diff_latent, target_coords)
        x2_proj = self.dense_projection(diff_latent)

        x2_cls = self.up0_prune_class(x2_proj).F[:,0]
        x2_mask = x2_cls > 0.5
        x2_target = self.get_target_prune(x2_proj, target_coord_map) if training else None
        x2_mask = x2_mask + x2_target if training else x2_mask
        x2_prune = self.pruning(x2_proj, x2_mask)

        y1 = self.up1[0](x2_prune)
        y1 = self.up1[1](y1)

        y1_cls = self.up1_prune_class(y1).F[:,0]
        y1_mask = y1_cls > 0.5
        y1_target = self.get_target_prune(y1, target_coord_map) if training else None
        y1_mask = y1_mask + y1_target if training else y1_mask
        y1_prune = self.pruning(y1, y1_mask)

        y2 = self.up2[0](y1_prune)
        y2 = self.up2[1](y2)

        y2_cls = self.up2_prune_class(y2).F[:,0]
        y2_mask = y2_cls > 0.3
        y2_target = self.get_target_prune(y2, target_coord_map) if training else None
        y2_mask = y2_mask + y2_target if training else y2_mask
        y2_prune = self.pruning(y2, y2_mask)

        y3 = self.up3[0](y2_prune)
        y3 = self.up3[1](y3)

        y3_cls = self.up3_prune_class(y3).F[:,0]
        y3_mask = y3_cls > 0.3
        y3_target = self.get_target_prune(y3, target_coord_map) if training else None
        y3_mask = y3_mask + y3_target if training else y3_mask
        y3_prune = self.pruning(y3, y3_mask)

        y4 = self.last(y3_prune)
        y4_target = self.get_target_prune(y4, target_coord_map) if training else None

        if training:
            return y4, [y1, y2, y3, y4], [x2_cls, y1_cls, y2_cls, y3_cls], [x2_target, y1_target, y2_target, y3_target, y4_target]
        else:
            return y4

    def vae_reparametrization(self, mean, log_var, shape):
        std = torch.exp(0.5 * log_var)
        return mean + std * torch.randn(shape, device=mean.device)

    def range_to_voxel(self, sparse_tensor, scene_range):
        b = len(sparse_tensor.C[:,0].unique())
        f = sparse_tensor.F.shape[-1]
        w = int((scene_range[0][1] - scene_range[0][0]) / (self.resolution * sparse_tensor.tensor_stride[0]))# + 4
        d = int((scene_range[1][1] - scene_range[1][0]) / (self.resolution * sparse_tensor.tensor_stride[1]))# + 4
        h = int((scene_range[2][1] - scene_range[2][0]) / (self.resolution * sparse_tensor.tensor_stride[2]))# + 4

        return torch.Size((b,f,w,d,h))

    def forward(self, x, training=True, encoder=False):
        ################################# Encoder ################################################
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)

        ################################# VAE Reparametrization ################################## 
        # x2 is our latent space!
        dense_shape = self.range_to_voxel(x2, self.train_range if training else self.val_range)
        x2_dense, _, stride = x2.dense(dense_shape)
        assert torch.all(stride.cuda() == self.latent_stride.cuda()), f'Latent stride mismatch {stride} != {self.latent_stride} (self.latent_stride incorrect)'
        x2_hidden = self.hidden(x2_dense)
        x2_mean = self.mean_head(x2_hidden)
        x2_logvar = self.logvar_head(x2_hidden)
        x2_reparam = self.vae_reparametrization(x2_mean, x2_logvar, x2_dense.shape)
        ################################# Decoder ################################################
        if encoder:
            return x2_reparam, stride

        torch.cuda.empty_cache()

        x2_dense, target_coord_map = self.sparse_to_dense(x2_reparam, x.C)
        x2_dense = self.dense_projection(x2_dense)

        x2_cls = self.up0_prune_class(x2_dense).F[:,0]
        x2_mask = x2_cls > 0.5
        x2_target = self.get_target_prune(x2_dense, target_coord_map)
        x2_mask = x2_mask + x2_target if training else x2_mask
        x2_prune = self.pruning(x2_dense, x2_mask)

        y1 = self.up1[0](x2_prune)
        y1 = self.up1[1](y1)

        y1_cls = self.up1_prune_class(y1).F[:,0]
        y1_mask = y1_cls > 0.5
        y1_target = self.get_target_prune(y1, target_coord_map)
        y1_mask = y1_mask + y1_target if training else y1_mask
        y1_prune = self.pruning(y1, y1_mask)

        y2 = self.up2[0](y1_prune)
        y2 = self.up2[1](y2)

        y2_cls = self.up2_prune_class(y2).F[:,0]
        y2_mask = y2_cls > 0.3
        y2_target = self.get_target_prune(y2, target_coord_map)
        y2_mask = y2_mask + y2_target if training else y2_mask
        y2_prune = self.pruning(y2, y2_mask)

        y3 = self.up3[0](y2_prune)
        y3 = self.up3[1](y3)

        y3_cls = self.up3_prune_class(y3).F[:,0]
        y3_mask = y3_cls > 0.3
        y3_target = self.get_target_prune(y3, target_coord_map)
        y3_mask = y3_mask + y3_target if training else y3_mask
        y3_prune = self.pruning(y3, y3_mask)

        y4 = self.last(y3_prune)
        y4_target = self.get_target_prune(y4, target_coord_map)
        torch.cuda.empty_cache()

        return [x2_reparam, x2_mean, x2_logvar], [x2, y1, y2, y3, y4], [x2_cls, y1_cls, y2_cls, y3_cls], [x2_target, y1_target, y2_target, y3_target, y4_target]

