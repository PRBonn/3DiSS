import torch
import torch.nn as nn
import torch.nn.functional as F
from diss.models.minkunet import MinkUNet
from diss.models.AttnUNet import LatentDiffuser
from diss.utils.collations import points_to_tensor
from utils.data_map import content
from diffusers import DPMSolverMultistepScheduler
import MinkowskiEngine as ME
from tqdm import tqdm
from diss.utils.data_map import color_map
import numpy as np
import open3d as o3d
from os import makedirs, path
import time

#from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningModule


class DiffLatent(LightningModule):
    def __init__(self, hparams:dict):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)
        self.gamma = self.hparams['diff']['gamma']

        self.model = MinkUNet(
                in_channels=4, out_channels=self.hparams['model']['out_dim'],
                resolution=self.hparams['data']['resolution'],
                train_range=self.hparams['data']['xyz_range_train'],
                val_range=self.hparams['data']['xyz_range_val'],
        )
        self.latent_diff = LatentDiffuser(latent_dim=128, mid_attn=self.hparams['model']['mid_attn']) 
        self.latent_diff_ema = LatentDiffuser(latent_dim=128, mid_attn=self.hparams['model']['mid_attn'])

        self.latent_sizes = {'x2': [], 'y1': [], 'y2': [], 'y3': [], 'time': []}
        self.count = 0

        for param, param_ema in zip(self.latent_diff.parameters(), self.latent_diff_ema.parameters()):
            param_ema.data.copy_(param.data)
            param_ema.requires_grad = False

        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.hparams['diff']['t_steps'],
                trained_betas=self.zero_snr_betas(),
                prediction_type='v_prediction',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )

        self.dpm_scheduler.set_timesteps(self.hparams['diff']['s_steps'])
        self.scheduler_to_cuda(self.device)

        sqrt_acp = torch.sqrt(self.dpm_scheduler.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_acp)
        sqrt_om_acp = torch.sqrt(1. - self.dpm_scheduler.alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_om_acp)

        # w calculation
        # compute Signal-to-Noise Ratio
        snr = (self.sqrt_alphas_cumprod / self.sqrt_one_minus_alphas_cumprod) ** 2
        # truncate it with gamma
        snr_w = torch.stack([snr, self.gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0]
        # compute weights for v-parameterization
        snr_w = snr_w / (snr + 1)
        self.register_buffer("snr_weights", snr_w)

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


    @torch.no_grad()
    def _update_ema(self):
        for param, param_ema in zip(self.latent_diff.parameters(), self.latent_diff_ema.parameters()):
            param_ema.data = param_ema.data * self.hparams['train']['ema_rate'] + param.data * (1. - self.hparams['train']['ema_rate'])

    def scheduler_to_cuda(self, device):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.to(device)
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.to(device)
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.to(device)
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.to(device)
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.to(device)
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.to(device)
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.to(device)
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.to(device)

        # reset scheduler for new sample otherwise it will keep stored the last sample from the previous batch
        self.dpm_scheduler.model_outputs = [None] * self.dpm_scheduler.config.solver_order
        self.dpm_scheduler.lower_order_nums = 0
        self.dpm_scheduler._step_index = None

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None,None,None] * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None,None,None,None] * noise

    def get_v(self, x, noise, t):
        return self.sqrt_alphas_cumprod[t][:,None,None,None,None] * noise - self.sqrt_one_minus_alphas_cumprod[t][:,None,None,None,None] * x

    def getDiffusionLoss(self, x, y, t):
        mse_loss = F.mse_loss(x, y, reduction='none')
        # apply weights batch-wise
        mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape)))) * self.snr_weights[t]
        
        return mse_loss.mean()

    def forward_vae(self, x:torch.Tensor, training=True):
        return self.model(x, training)

    def forward_diff(self, feats, t, ema=False):
        return self.latent_diff_ema(feats, t) if ema else self.latent_diff(feats, t)

    def training_step(self, batch:dict, batch_idx):
        self._update_ema()
        x_occupancy = points_to_tensor(batch['coords'], batch['feats'], self.hparams['data']['resolution'], -1, self.global_rank, batch['filename'])
        # get the auto-encoder latent and pass it to the diffusion process
        with torch.no_grad():
            latent_args, occupancy_pred, pred_prune, target_prune  = self.forward_vae(x_occupancy, training=False)
            occupancy_latent, latent_mean, latent_logvar = latent_args
        torch.cuda.empty_cache()
        # diffusion part
        t = torch.randint(0, self.hparams['diff']['t_steps'], size=(len(batch['feats']),), device=occupancy_latent.device)
        noise = torch.randn(occupancy_latent.shape, device=occupancy_latent.device)
        noisy_latent = self.q_sample(occupancy_latent, t, noise)
        torch.cuda.empty_cache()

        pred_noise = self.forward_diff(noisy_latent, t)
        torch.cuda.empty_cache()
        loss = self.getDiffusionLoss(pred_noise, self.get_v(occupancy_latent, noise, t), t)
        torch.cuda.empty_cache()

        self.log('train/loss', loss)

        return loss

    def p_sample_loop(self, x_t, ema=False):
        self.scheduler_to_cuda(device=x_t.device)
        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            torch.cuda.empty_cache()
            t = torch.ones((len(x_t),), device=x_t.device).long() * self.dpm_scheduler.timesteps[t]

            noise_t = self.forward_diff(x_t, t, ema)
            x_t_feats = self.dpm_scheduler.step(noise_t, t[0], x_t)['prev_sample']
            x_t = x_t_feats

            torch.cuda.empty_cache()

        return x_t

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

    def infer_step(self, batch, batch_idx):
        makedirs(f'{self.logger.log_dir}/generated_pcd/x0/', exist_ok=True)
        makedirs(f'{self.logger.log_dir}/generated_pcd/real_data/', exist_ok=True)
        self.model.eval()
        self.latent_diff.eval()
        self.latent_diff_ema.eval()
        with torch.no_grad():
            x_occupancy = points_to_tensor(batch['coords'], batch['feats'], self.hparams['data']['resolution'], -1, self.global_rank, batch['filename']) 
            # purning/occupancy loss
            latent_args, occupancy_pred, pred_prune, target_prune = self.forward_vae(x_occupancy, training=False)
            before_time = time.time()
            _, latent_size = self.model.forward_decoder(latent_args[0])
            after_time = time.time()
            self.latent_sizes['time'].append(after_time - before_time)
            self.latent_sizes['x2'].append(latent_size[0])
            self.latent_sizes['y1'].append(latent_size[1])
            self.latent_sizes['y2'].append(latent_size[2])
            self.latent_sizes['y3'].append(latent_size[3])
            self.count += 1

            print(f'time avg size: {np.mean(self.latent_sizes["time"])}')
            print(f'x2 avg size: {np.mean(self.latent_sizes["x2"])}')
            print(f'y1 avg size: {np.mean(self.latent_sizes["y1"])}')
            print(f'y2 avg size: {np.mean(self.latent_sizes["y2"])}')
            print(f'y3 avg size: {np.mean(self.latent_sizes["y3"])}')

            if self.count == 100:
                import ipdb; ipdb.set_trace()


            torch.cuda.empty_cache()

            # diffusion part
            #bsize = len(batch['coords'])
            #latent_shape = torch.Size((bsize,128,64,64,16))
            #noise_in = torch.randn(latent_shape, device=torch.device('cuda'))
            #x0_pred = self.p_sample_loop(noise_in, ema=False)

            #decoded_x0 = self.model.forward_decoder(x0_pred)
            #torch.cuda.empty_cache()

            #for i in range(bsize):
            #    pcd_x0 = self.decode_to_pcd(decoded_x0, i)
            #    np.savez_compressed(f'{self.logger.log_dir}/generated_pcd/x0/{batch_idx*bsize + i}.npz', pcd_x0)

            #    pcd_gt = np.concatenate((batch['coords'][i].cpu().numpy(), batch['feats'][i].cpu().numpy()), axis=-1)
            #    pcd_gt[:,:3] = self.devoxelize(pcd_gt[:,:3])
            #    np.savez_compressed(f'{self.logger.log_dir}/generated_pcd/real_data/{batch_idx*bsize + i}.npz', pcd_gt)

        torch.cuda.empty_cache()

        return 0.

    def validation_step(self, batch:dict, batch_idx):
        loss = self.infer_step(batch, batch_idx)
        self.log('val/loss', loss)

        return loss

    def test_step(self, batch:dict, batch_idx):
        loss = self.infer_step(batch, batch_idx)
        self.log('test/loss', loss)

        return {'loss':loss}

    def configure_optimizers(self): 
        #optimizer = torch.optim.SGD(self.latent_diff.parameters(), lr=self.hparams['train']['lr'], momentum=0.9)
        optimizer = torch.optim.AdamW(self.latent_diff.parameters(), lr=self.hparams['train']['lr'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)

        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 50,
        }


        return [optimizer], [scheduler]

