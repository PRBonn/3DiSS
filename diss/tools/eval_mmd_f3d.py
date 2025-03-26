import torch
import numpy as np
import click
import os
import open3d as o3d
from diss.models.sem_vae import AutoEncoder
from diss.utils.collations import points_to_tensor
from diss.utils.eval_utils import MinkUNet
from random import shuffle
from tqdm import tqdm
import yaml

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True


class F3DMetric:
    # Adapted implementation from https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/fid.py#L182-L464
    def __init__(self, feature_dim=128):
        self.num_real = torch.tensor(0, device=torch.device('cuda')).long()
        self.real_sum = torch.zeros(feature_dim, device=torch.device('cuda'))
        self.real_cov_sum = torch.zeros((feature_dim,feature_dim), device=torch.device('cuda'))

        self.num_gen = torch.tensor(0, device=torch.device('cuda')).long()
        self.gen_sum = torch.zeros(feature_dim, device=torch.device('cuda'))
        self.gen_cov_sum = torch.zeros((feature_dim,feature_dim), device=torch.device('cuda'))

    def update(self, real, gen): 
        if len(real.shape) > 2:
            b,f,w,h,d = real.shape
            real = real.permute(0,2,3,4,1).reshape(-1,f)
        if len(gen.shape) > 2:
            b,f,w,h,d = gen.shape
            gen = gen.permute(0,2,3,4,1).reshape(-1,f)
 
        self.num_real += real.size(0)
        self.real_sum += torch.sum(real, dim=0)
        self.real_cov_sum += torch.matmul(real.T, real)

        self.num_gen += gen.size(0)
        self.gen_sum += torch.sum(gen, dim=0)
        self.gen_cov_sum += torch.matmul(gen.T, gen)

    def compute(self):
        real_mean = (self.real_sum / self.num_real).unsqueeze(0)
        gen_mean = (self.gen_sum / self.num_gen).unsqueeze(0)

        real_cov_num = self.real_cov_sum - self.num_real * torch.matmul(real_mean.T, real_mean)
        real_cov = real_cov_num / (self.num_real - 1)

        gen_cov_num = self.gen_cov_sum - self.num_gen * torch.matmul(gen_mean.T, gen_mean)
        gen_cov = gen_cov_num / (self.num_gen - 1)

        mean_norm = (real_mean - gen_mean).square().sum(dim=-1)
        trace_sum = real_cov.trace() + gen_cov.trace()
        sigma_mm = torch.matmul(real_cov.double(), gen_cov.double())
        eigenvals = torch.linalg.eigvals(sigma_mm)
        sqrt_eigenvals_sum = eigenvals.sqrt().real.float().sum(dim=-1)

        f3d = mean_norm + trace_sum - 2 * sqrt_eigenvals_sum

        return f3d.item()

class MMDMetric:
    # Borrowed implementation from https://pytorch.org/ignite/_modules/ignite/metrics/maximum_mean_discrepancy.html#MaximumMeanDiscrepancy
    def __init__(self, metric_batch=64, multi_bandwidth=False):
        self.multi_bandwidth = multi_bandwidth
        self.metric_batch = metric_batch
        self._num_batches = 0
        self._xx_sum = torch.tensor(0.0, device=torch.device('cuda'))
        self._yy_sum = torch.tensor(0.0, device=torch.device('cuda'))
        self._xy_sum = torch.tensor(0.0, device=torch.device('cuda'))
        self.x_samples = []
        self.y_samples = []
    
    def _update(self):
        x = torch.vstack(self.x_samples)
        y = torch.vstack(self.y_samples)
 
        xx, yy, zz = torch.mm(x, x.T), torch.mm(y, y.T), torch.mm(x, y.T)
        # compute sigma
        sigmas = [torch.median(xx), torch.median(yy), torch.median(zz)] if self.multi_bandwidth else [torch.median(zz)]

        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.T + rx - 2.0 * xx
        dyy = ry.T + ry - 2.0 * yy
        dxy = rx.T + ry - 2.0 * zz

        XX, YY, XY = torch.zeros_like(dxx), torch.zeros_like(dyy), torch.zeros_like(dxy)

        for sigma in sigmas:
            XX += torch.exp(-0.5 * dxx / sigma)
            YY += torch.exp(-0.5 * dyy / sigma)
            XY += torch.exp(-0.5 * dxy / sigma)

        n = x.shape[0]
        XX = (XX.sum() - n) / (n * (n-1))
        YY = (YY.sum() - n) / (n * (n-1))
        XY = XY.sum() / (n * n)

        self._xx_sum += XX
        self._yy_sum += YY
        self._xy_sum += XY
        self._num_batches += 1

        self.x_samples = []
        self.y_samples = []

    def update(self, x, y):
        if len(x.shape) > 2:
            b,f,w,h,d = x.shape
            x = x.reshape(x.size(0),-1)
        if len(y.shape) > 2:
            b,f,w,h,d = y.shape
            y = y.reshape(y.size(0),-1)

        self.x_samples.append(x)
        self.y_samples.append(y)

        if len(self.x_samples) < self.metric_batch:
            return

        self._update()

    def compute(self):
        if len(self.x_samples) != 0:
            self._update()
        mmd2 = (self._xx_sum + self._yy_sum - 2.0 * self._xy_sum).clamp(min=0.0) / self._num_batches
        return mmd2.sqrt().item()

def coords_to_voxel(pcd, resolution=0.1):
    pcd = (pcd / resolution).trunc()
    pcd -= pcd.min(0).values

    return pcd.int(), pcd * resolution

def load_pcd(pcd_file, xyz_range):
    if pcd_file.endswith('.npz'):
        points = torch.tensor(np.load(pcd_file)['arr_0']).cuda()
    elif pcd_file.endswith('.npy'):
        points = torch.tensor(np.load(pcd_file)).cuda()
    elif pcd_file.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = torch.tensor(np.array(pcd.points)).cuda()
        points[:,:3] = torch.round(points[:,:3] / 0.1) * 0.1
    elif pcd_file.endswith('.label'):
        original_shape = (256, 256, 32)
        voxel_map = np.fromfile(pcd_file, dtype=np.uint16).reshape(original_shape)
        voxels = voxel_map > 0
        occupied_voxels = np.argwhere(voxels)
        p_set = occupied_voxels * 0.2
        p_set[:,0] += xyz_range[0][0]# - 0.2
        p_set[:,1] += xyz_range[1][0]# - 0.2
        p_set[:,2] += xyz_range[2][0]# - 0.2
        # resolution of SSC labels from semantickitti
        points = torch.tensor(p_set).cuda()

    return filter_fov(points, xyz_range)

def filter_fov(points, xyz_range):
    grid_fov = (points[:,0] > xyz_range[0][0]) & (points[:,0] < xyz_range[0][1]) &\
                (points[:,1] > xyz_range[1][0]) & (points[:,1] < xyz_range[1][1]) &\
                (points[:,2] > xyz_range[2][0]) & (points[:,2] < xyz_range[2][1])

    return points[grid_fov]


@click.command()
@click.option('--fake', '-f', type=str, default='', help='path to the generated data')
@click.option('--real', '-r', type=str, default='', help='path to the real data')
@click.option('--vae_weights', '-w', type=str, default='', help='path to the VAE weights')
@click.option('--num_samples', '-n', type=int, default=8000, help='Number of samples to evaluate')
@click.option('--resolution', type=float, default=0.1, help='Evaluation resolution')
@click.option('--exp_name', '-exp', type=str, default='', help='Experiment name to save the output file')
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../config/vae_config.yaml'))
def eval_samples(fake, real, vae_weights, num_samples, resolution, exp_name, config):
    set_deterministic()
    cfg = yaml.safe_load(open(config))
    fake_files = os.listdir(os.path.join(fake))
    np.random.shuffle(fake_files)
    fake_files = fake_files[:num_samples]

    real_files = os.listdir(os.path.join(real))
    # it is important to shuffle otherwise all the real samples will be to similar since they would be sequential
    set_deterministic()
    np.random.shuffle(real_files)
    real_files = real_files[:num_samples]
    model = MinkUNet(in_channels=3)
    ckpt = torch.load(vae_weights)
    if 'model' in ckpt.keys():
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model = model.cuda()
    model.eval()

    f3d_real_real = F3DMetric(feature_dim=256)
    mmd_real_real = MMDMetric()
    f3d_real_fake = F3DMetric(feature_dim=256)
    mmd_real_fake = MMDMetric()

    for i, (fake_file, real_file) in tqdm(enumerate(zip(fake_files, real_files))):
        pcd_real = load_pcd(os.path.join(real, real_file), cfg['data']['xyz_range_val'])
        pcd_real[:,:3] = torch.round(pcd_real[:,:3] / resolution) * resolution
        pcd_real = filter_fov(pcd_real, cfg['data']['xyz_range_val'])
        # all generative methods are voxelized, so we voxelize the coords of the real pcd so evaluation compares only the data distribution
        # without being influenced by the noisy points from the real pcd
        vox_real, pcd_real = coords_to_voxel(pcd_real[:,:3], resolution)
        batched_pcds = points_to_tensor([vox_real], [pcd_real[:,:3]], resolution, -1) 
        with torch.no_grad():
            latent_real = model(batched_pcds).dense(torch.Size((1,256,32,32,4)))[0]
        torch.cuda.empty_cache()
        f3d_real_real.update(latent_real, latent_real)
        mmd_real_real.update(latent_real, latent_real)

        pcd_fake = load_pcd(os.path.join(fake, fake_file), cfg['data']['xyz_range_val'])
        vox_fake, pcd_fake = coords_to_voxel(pcd_fake[:,:3], resolution)
        batched_pcds = points_to_tensor([vox_fake], [pcd_fake[:,:3]], resolution, -1) 
        with torch.no_grad():
            latent_fake = model(batched_pcds).dense(torch.Size((1,256,32,32,4)))[0]
        torch.cuda.empty_cache()
        f3d_real_fake.update(latent_real, latent_fake)
        mmd_real_fake.update(latent_real, latent_fake)

        if (i+1) % 512 == 0:
            print(f'Sanity check - [F3D]: {f3d_real_real.compute()}\t[MMD]: {mmd_real_real.compute()}')
            print(f'[F3D]: {f3d_real_fake.compute()}\t[MMD]: {mmd_real_fake.compute()}')

    print(f'Sanity check - [F3D]: {f3d_real_real.compute()}\t[MMD]: {mmd_real_real.compute()}')
    print(f'[F3D]: {f3d_real_fake.compute()}\t[MMD]: {mmd_real_fake.compute()}')

    with open(f'EVAL/{exp_name}.txt', 'w') as f:
        f.write(f'Generated data path: {fake}\nReal data path: {real}\n\nModel weights: {vae_weights}\nNum samples: {num_samples}\tResolution: {resolution}\n\n')
        f.write(f'[F3D]: {f3d_real_fake.compute()}\t[MMD]: {mmd_real_fake.compute()}')



if __name__ == "__main__":
    eval_samples()
