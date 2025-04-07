import numpy as np
import MinkowskiEngine as ME
import torch
from diss.models.minkunet import MinkUNet, MinkEncoder
from diss.models.AttnUNet import LatentCondDiffuser
from diss.utils.collations import points_to_tensor
import open3d as o3d
from diffusers import DPMSolverMultistepScheduler
from pytorch_lightning.core.lightning import LightningModule
import yaml
import os
from tqdm import tqdm
from natsort import natsorted
from diss.utils.data_map import color_map
import click

class DiSS(LightningModule):
    def __init__(self, diff_path, vae_path, denoising_steps, cond_weight, condition):
        super().__init__()
        ckpt_diff = torch.load(diff_path)
        ckpt_vae = torch.load(vae_path)
        self.save_hyperparameters(ckpt_diff['hyper_parameters'])
        self.condition = condition
        assert denoising_steps <= self.hparams['diff']['t_steps'], \
        f"The number of denoising steps cannot be bigger than T={self.hparams['diff']['t_steps']} (you've set '-T {denoising_steps}')"

        self.cond_enc = MinkEncoder(in_channels=3,
                resolution=self.hparams['data']['resolution'],
                train_range=self.hparams['data']['xyz_range_train'],
                val_range=self.hparams['data']['xyz_range_val'],
        )

        self.model = MinkUNet(
                in_channels=4, out_channels=self.hparams['model']['out_dim'],
                resolution=self.hparams['data']['resolution'],
                train_range=self.hparams['data']['xyz_range_train'],
                val_range=self.hparams['data']['xyz_range_val'],
        )
        self.latent_diff = LatentCondDiffuser(latent_dim=128, mid_attn=self.hparams['model']['mid_attn'], cond_diff=True if condition == 'single_scan' else False)
        print('Loading diffusion weights...')
        self.load_state_dict(ckpt_diff['state_dict'], strict=False)
        print('Loading VAE weights...')
        self.load_state_dict(ckpt_vae['state_dict'], strict=False)

        self.model.eval()
        self.cond_enc.eval()
        self.latent_diff.eval()
        self.cuda()

        # for fast sampling
        self.hparams['diff']['s_steps'] = denoising_steps
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.hparams['diff']['t_steps'],
                trained_betas=self.zero_snr_betas(),
                prediction_type='v_prediction',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )

        self.dpm_scheduler.set_timesteps(self.hparams['diff']['s_steps'])
        self.scheduler_to_cuda()
        self.sqrt_alphas_cumprod = torch.sqrt(self.dpm_scheduler.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.dpm_scheduler.alphas_cumprod)
        self.snr = (self.sqrt_alphas_cumprod / self.sqrt_one_minus_alphas_cumprod) ** 2
        self.snr_weights = torch.stack([self.snr, self.hparams['diff']['gamma'] * torch.ones_like(self.snr)], dim=1).min(dim=1)[0]
        self.snr_weights = self.snr_weights / (self.snr + 1)

        self.hparams['diff']['w_cond'] = cond_weight
        self.w_uncond = self.hparams['diff']['w_cond']
        
        exp_dir = diff_path.split('/')[-1].split('.')[0].replace('=','') + f'_{condition}'
        os.makedirs(f'./results/{exp_dir}', exist_ok=True)
        with open(f'./results/{exp_dir}/exp_config.yaml', 'w+') as exp_config:
            yaml.dump(self.hparams, exp_config)

    def zero_snr_betas(self):
        # define betas and alphas
        betas = torch.linspace(self.hparams['diff']['beta_start'], self.hparams['diff']['beta_end'], self.hparams['diff']['t_steps'])
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # get alphas_sqrt min and max
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # rescale alphas_sqrt to have zero value at T
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # get rescaled alphas and betas
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])

        return 1 - alphas

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

        # reset scheduler for new sample otherwise it will keep stored the last sample from the previous batch
        self.dpm_scheduler.model_outputs = [None] * self.dpm_scheduler.config.solver_order
        self.dpm_scheduler.lower_order_nums = 0
        self.dpm_scheduler._step_index = None

    def preprocess_scan(self, points, xyz_range):
        #points[:,2] += 0.5
        grid_fov = (points[:,0] > xyz_range[0][0]) & (points[:,0] < xyz_range[0][1]) &\
                (points[:,1] > xyz_range[1][0]) & (points[:,1] < xyz_range[1][1]) &\
                (points[:,2] > xyz_range[2][0]) & (points[:,2] < xyz_range[2][1])

        x_fov = torch.tensor(points[grid_fov])
        x_fov[:,:3] = (x_fov[:,:3] / self.hparams['data']['resolution']).trunc()
        _, mapping = ME.utils.sparse_quantize(coordinates=x_fov[:,:3], return_index=True)
        x_fov = x_fov[mapping]
        x_fov[:,:3] -= x_fov[:,:3].min(0).values
    
        return x_fov[:,:3], torch.tensor(points[grid_fov][mapping])

    def devoxelize(self, points):
        points = points * self.hparams['data']['resolution']
        points[:,0] += self.hparams['data']['xyz_range_val'][0][0]
        points[:,1] += self.hparams['data']['xyz_range_val'][1][0]
        points[:,2] += self.hparams['data']['xyz_range_val'][2][0]

        return points

    def decode_to_pcd(self, decoded_latent, batch_id=0, vis=False):
        batch_idx = decoded_latent.C[:,0] == batch_id
        sem_pred = decoded_latent.F[batch_idx].max(dim=1)[1].detach().cpu().numpy()

        # devoxelize the coordinates
        points = self.devoxelize(decoded_latent.C[batch_idx,1:].cpu().detach().numpy())

        if vis:
            # save it as .ply to be visualized
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            color_array = np.array(list(color_map.values()))
            colors = color_array[sem_pred,::-1]
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors)/255.)
        else:
            # save it as .npy to be used for training
            pcd = np.concatenate((points, sem_pred[:,None]), axis=-1)

        return pcd

    def complete_scan(self, scan, vis=True):
        coords, feats = self.preprocess_scan(scan, self.hparams['data']['xyz_range_val'])

        latent_shape = torch.Size((1,128,64,64,16))
        noise_in = torch.randn(latent_shape, device=torch.device('cuda'))
        x0_completion = self.completion_loop(noise_in, [(coords,), (feats,)])
        decoded_x0 = self.model.forward_decoder(x0_completion)
        complete_scan_x0 = self.decode_to_pcd(decoded_x0)

        return complete_scan_x0

    def uncond_sample(self, vis=True):
        latent_shape = torch.Size((1,128,64,64,16))
        noise_in = torch.randn(latent_shape, device=torch.device('cuda'))
        x0 = self.uncond_loop(noise_in)
        decoded_x0 = self.model.forward_decoder(x0)
        pcd_x0 = self.decode_to_pcd(decoded_x0)

        return pcd_x0

    def classfree_forward(self, latent, cond, uncond, t):
        with torch.no_grad():
            x_cond = self.latent_diff(latent, t, cond)
            x_uncond = self.latent_diff(latent, t, uncond)

        x_out = x_uncond + self.hparams['diff']['w_cond'] * (x_cond - x_uncond)
        return x_out


    def cond_tokens(self, cond):
        x_cond = points_to_tensor(cond[0], cond[1], self.hparams['data']['resolution'], -1)
        x_cond = self.cond_enc(x_cond)

        x_uncond = points_to_tensor(
               [torch.zeros_like(c) for c in cond[0]],
               [torch.zeros_like(c) for c in cond[1]],
               self.hparams['data']['resolution'], -1
           )
        x_uncond = self.cond_enc(x_uncond)

        return x_cond, x_uncond

    def completion_loop(self, x_t, cond):
        self.scheduler_to_cuda()
        x_cond, x_uncond = self.cond_tokens(cond)
        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones((len(x_t),)).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()

            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            x_t_feats = self.dpm_scheduler.step(noise_t, t[0], x_t)['prev_sample']
            x_t = x_t_feats

            torch.cuda.empty_cache()

        return x_t

    def uncond_loop(self, x_t):
        self.scheduler_to_cuda()
        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones((len(x_t),)).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()

            with torch.no_grad():
                noise_t = self.latent_diff(x_t, t)
            x_t_feats = self.dpm_scheduler.step(noise_t, t[0], x_t)['prev_sample']
            x_t = x_t_feats

            torch.cuda.empty_cache()

        return x_t

    def diff_sample(self, scan=None, vis=True):
        if self.condition == 'uncond':
            return self.uncond_sample()

        elif self.condition == 'single_scan':
            return self.complete_scan(scan)


def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")

def cond_loop(diff, path, exp_dir):
    pcds = os.listdir(path)

    for pcd_path in tqdm(pcds):
        pcd_file = os.path.join(path, pcd_path)
        points = load_pcd(pcd_file)
        #points[:,2] += 0.3
    
        diff_scan_x0 = diff.diff_sample(points)
        #diff_scan_x0.estimate_normals()

        np.savez_compressed(f'./results/{exp_dir}/x0/{pcd_path.split(".")[0]}.npz', diff_scan_x0)
        np.savez_compressed(f'./results/{exp_dir}/cond/{pcd_path.split(".")[0]}.npz', points)

def uncond_loop(diff, num_samples, exp_dir):
    for i in range(num_samples):
        diff_x0 = diff.diff_sample()
        np.savez_compressed(f'./results/{exp_dir}/x0/{i}.npz', diff_x0)

@click.command()
@click.option('--path', '-p', type=str, help='path to the condition scans')
@click.option('--diff', '-d', type=str, default='checkpoints/diff_net.ckpt', help='path to the diffusion weights')
@click.option('--vae', '-v', type=str, default='checkpoints/vae_net.ckpt', help='path to the VAE weights')
@click.option('--denoising_steps', '-T', type=int, default=1000, help='number of denoising steps (default: 1000)')
@click.option('--cond_weight', '-s', type=float, default=2.0, help='conditioning weight (default: 2.0)')
@click.option('--condition', '-cond', type=str, default='uncond', help='path to the condition scans')
@click.option('--num_samples', '-n', type=int, default=10, help='number of uncondtional samples to be generated (default: 10)')
def main(path, diff, vae, denoising_steps, cond_weight, condition, num_samples):
    exp_dir = diff.split('/')[-1].split('.')[0].replace('=','') + f'_{condition}'

    diff = DiSS(diff, vae, denoising_steps, cond_weight, condition)

    os.makedirs(f'./results/{exp_dir}/x0', exist_ok=True)
    if condition == 'uncond':
        uncond_loop(diff, num_samples, exp_dir)
    elif condition == 'single_scan':
        os.makedirs(f'./results/{exp_dir}/cond', exist_ok=True)
        cond_loop(diff, path, exp_dir)

if __name__ == '__main__':
    main()
