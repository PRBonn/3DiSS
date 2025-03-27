import torch
import torch.nn as nn
import torch.nn.functional as F
from diss.models.minkunet import MinkUNet
from diss.utils.collations import points_to_tensor
from diss.utils.data_map import content
import open3d as o3d
import numpy as np
from diss.utils.data_map import color_map
from torchmetrics.classification import MulticlassJaccardIndex
from diffusers import DPMSolverMultistepScheduler

from pytorch_lightning.core.lightning import LightningModule


class AutoEncoder(LightningModule):
    def __init__(self, hparams:dict):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)
        self.sem_weights = 1/torch.tensor(list(content.values()), device=torch.device('cuda'))
        self.sem_weights /= self.sem_weights.max()
        self.iou = MulticlassJaccardIndex(num_classes=20, ignore_index=0).cuda()

        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.hparams['diff']['t_steps'],
                trained_betas=self.zero_snr_betas(),
                prediction_type='v_prediction',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.dpm_scheduler.alphas_cumprod).cuda()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.dpm_scheduler.alphas_cumprod).cuda()

        self.model = MinkUNet(
                in_channels=4, out_channels=self.hparams['model']['out_dim'],
                resolution=self.hparams['data']['resolution'],
                train_range=self.hparams['data']['xyz_range_train'],
                val_range=self.hparams['data']['xyz_range_val'],
            )

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

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None,None,None].cuda() * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None,None,None,None].cuda() * noise

    def getLoss(self, x:torch.Tensor, y:torch.Tensor):
        return F.binary_cross_entropy(x, y)

    def getLatentLoss(self, mean, logvar):
        loss = - 0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp())
        # normalize by the number of voxels
        loss /= mean.view(mean.shape[0], mean.shape[1], -1).shape[-1]

        return loss

    def getSemLoss(self, x, y):
        # during the first 25 epochs use weights to force the model to consider all classes
        if self.current_epoch < self.hparams['train']['max_epoch'] / 2 and not self.hparams['train']['refine']:
           loss = F.cross_entropy(x, y, ignore_index=0, weight=self.sem_weights.cuda())
        # the last 25 epochs ignore the weights so the model can optimize to achieve highest IoU
        else:
           loss = F.cross_entropy(x, y, ignore_index=0)

        return loss

    def matchSem(self, pred, target):
        # create the full grid from the prediction
        pred_feats = pred.dense()[0].permute(0,2,3,4,1)

        target_coords = target.C.T.long()
        # select the grid feats at the target coords
        pred_sem = pred_feats[target_coords[0], target_coords[1], target_coords[2], target_coords[3]]
        torch.cuda.empty_cache()

        return pred_sem, target.F[:,-1]

    def forward(self, x:torch.Tensor, training=True, encoder=False):
        return self.model(x, training, encoder)

    def decode_to_pcd(self, decoded_latent, batch_id=0):
        # convert the VAE decoded pred into a point cloud
        batch_idx = decoded_latent.C[:,0] == batch_id
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(decoded_latent.C[batch_idx,1:].cpu().detach().numpy())

        sem_pred = decoded_latent.F[batch_idx].max(dim=1)[1].detach().cpu().numpy()
        color_array = np.array(list(color_map.values()))
        colors = color_array[sem_pred,::-1]
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors)/255.)

        return pcd

    def compute_iou(self, pred, target):
        target_ = target.dense()[0][:,-1,...][:,None,...].max(1)[1].shape
        pred_ = pred.dense()[0].max(1)[1].shape
        
        # densify everything to biggest shape between pred and target
        shape = torch.Size([ max(t,p) for t,p in zip(target_,pred_)])
        
        target = target.dense(torch.Size([shape[0],4,shape[1],shape[2],shape[3]]))[0][:,-1,...][:,None,...].permute(0,2,3,4,1)
        pred = pred.dense(torch.Size([shape[0],20,shape[1],shape[2],shape[3]]))[0].permute(0,2,3,4,1).max(-1)[1][...,None]
        
        return self.iou.update(pred, target)

    def getDiceLoss(self, x, y):
        tp_index = ((x == 1) & (y == 1))

        nominator = 2 * tp_index.sum()
        fp = (x[~tp_index] == 1).sum()
        fn = (x[y == 1] == 0).sum()
        denominator = nominator + fp + fn

        return 1 - (nominator / denominator)

    def on_training_epoch_end(self, outputs):
        self.iou.reset()

    def on_validation_epoch_end(self):
        self.iou.reset()

    def train_vae(self, batch:dict, batch_idx):
        # pytorch does not know how to handle sparse operations in the first iteration
        # so we run the first iteration with fewer data points to avoid over allocating memory
        if self.global_step == 0:
            batch['coords'] = (batch['coords'][0],)
            batch['feats'] = (batch['feats'][0],)
        x_occupancy = points_to_tensor(batch['coords'], batch['feats'], self.hparams['data']['resolution'], self.global_step)
        latent_args, occupancy_pred, pred_prune, target_prune  = self.forward(x_occupancy)
        occupancy_latent, latent_mean, latent_logvar = latent_args

        self.compute_iou(occupancy_pred[-1], x_occupancy)

        # purning/occupancy loss
        loss_prune0 = self.getLoss(pred_prune[0], target_prune[0].float())
        loss_prune0 += self.getDiceLoss((pred_prune[0] > 0.5).int(), target_prune[0].int())
        loss_prune1 = self.getLoss(pred_prune[1], target_prune[1].float())
        loss_prune1 += self.getDiceLoss((pred_prune[1] > 0.5).int(), target_prune[1].int())
        loss_prune2 = self.getLoss(pred_prune[2], target_prune[2].float())
        loss_prune2 += self.getDiceLoss((pred_prune[2] > 0.5).int(), target_prune[2].int())
        loss_prune3 = self.getLoss(pred_prune[3], target_prune[3].float())
        loss_prune3 += self.getDiceLoss((pred_prune[3] > 0.5).int(), target_prune[3].int())

        loss_prune = loss_prune0 + loss_prune1 + 2*loss_prune2 + 3*loss_prune3
        # semantic prediction loss
        pred_sem, target_sem = self.matchSem(occupancy_pred[-1], x_occupancy)
        loss_sem = self.getSemLoss(pred_sem, target_sem.long())
        # KL latent loss (approx to a gaussian)
        loss_latent = self.getLatentLoss(latent_mean, latent_logvar)

        loss = self.hparams['train']['prune_w'] * loss_prune + self.hparams['train']['sem_w']*loss_sem + self.hparams['train']['kl_w']*loss_latent

        torch.cuda.empty_cache()

        self.log('train/loss_prune', loss_prune)
        self.log('train/loss_sem', loss_sem)
        self.log('train/loss_latent', loss_latent)
        self.log('train/latent_mean', latent_mean.mean())
        self.log('train/latent_logvar', latent_logvar.mean())
        self.log('train/latent_std', torch.exp(0.5*latent_logvar).mean())
        self.log('train/loss', loss)
        self.log('train/miou', self.iou.compute())

        return loss

    def refine_decoder(self, batch, batch_idx):
        # pytorch does not know how to handle sparse operations in the first iteration
        # so we run the first iteration with fewer data points to avoid over allocating memory
        if self.global_step == 0:
            batch['coords'] = (batch['coords'][0],)
            batch['feats'] = (batch['feats'][0],)
        x_occupancy = points_to_tensor(batch['coords'], batch['feats'], self.hparams['data']['resolution'], self.global_step)
        with torch.no_grad():
            occupancy_latent, stride = self.forward(x_occupancy, encoder=True)

        # add noise to latent
        t = torch.randint(0, int(self.hparams['diff']['t_steps']/10), size=(len(batch['feats']),)).cuda()
        noise = torch.randn(occupancy_latent.shape, device=torch.device('cuda'))
        noisy_latent = self.q_sample(occupancy_latent, t, noise)

        decoded, occupancy_pred, pred_prune, target_prune  = self.model.forward_decoder(noisy_latent, x_occupancy.C, training=True)

        self.compute_iou(occupancy_pred[-1], x_occupancy)

        # purning/occupancy loss
        loss_prune0 = self.getLoss(pred_prune[0], target_prune[0].float())
        loss_prune0 += self.getDiceLoss((pred_prune[0] > 0.5).int(), target_prune[0].int())
        loss_prune1 = self.getLoss(pred_prune[1], target_prune[1].float())
        loss_prune1 += self.getDiceLoss((pred_prune[1] > 0.5).int(), target_prune[1].int())
        loss_prune2 = self.getLoss(pred_prune[2], target_prune[2].float())
        loss_prune2 += self.getDiceLoss((pred_prune[2] > 0.5).int(), target_prune[2].int())
        loss_prune3 = self.getLoss(pred_prune[3], target_prune[3].float())
        loss_prune3 += self.getDiceLoss((pred_prune[3] > 0.5).int(), target_prune[3].int())

        loss_prune = loss_prune0 + loss_prune1 + 2*loss_prune2 + 3*loss_prune3
        # semantic prediction loss
        pred_sem, target_sem = self.matchSem(occupancy_pred[-1], x_occupancy)
        loss_sem = self.getSemLoss(pred_sem, target_sem.long())

        loss = self.hparams['train']['prune_w'] * loss_prune + self.hparams['train']['sem_w']*loss_sem

        torch.cuda.empty_cache()

        self.log('train/loss_prune', loss_prune)
        self.log('train/loss_sem', loss_sem)
        self.log('train/loss', loss)
        self.log('train/miou', self.iou.compute())

        return loss

    def training_step(self, batch:dict, batch_idx):
        if self.hparams['train']['refine']:
            loss = self.refine_decoder(batch, batch_idx)
            self.log('train/refine_loss', loss)
        else:
            loss = self.train_vae(batch, batch_idx)
            self.log('train/diff_loss', loss)

        return loss

    def validation_step(self, batch:dict, batch_idx):
        x_occupancy = points_to_tensor(batch['coords'], batch['feats'], self.hparams['data']['resolution'], -1) 
        # purning/occupancy loss
        latent_args, occupancy_pred, pred_prune, target_prune = self.forward(x_occupancy, training=False)
        occupancy_latent, latent_mean, latent_logvar = latent_args

        self.compute_iou(occupancy_pred[-1], x_occupancy)

        loss_prune0 = self.getLoss(pred_prune[0], target_prune[0].float())
        loss_prune0 += self.getDiceLoss((pred_prune[0] > 0.5).int(), target_prune[0].int())
        loss_prune1 = self.getLoss(pred_prune[1], target_prune[1].float())
        loss_prune1 += self.getDiceLoss((pred_prune[1] > 0.5).int(), target_prune[1].int())
        loss_prune2 = self.getLoss(pred_prune[2], target_prune[2].float())
        loss_prune2 += self.getDiceLoss((pred_prune[2] > 0.5).int(), target_prune[2].int())
        loss_prune3 = self.getLoss(pred_prune[3], target_prune[3].float())
        loss_prune3 += self.getDiceLoss((pred_prune[3] > 0.5).int(), target_prune[3].int())

        loss_prune = loss_prune0 + loss_prune1 + 2 * loss_prune2 + 3 * loss_prune3
        # semantic prediction loss
        #loss_sem = self.getSemLoss(occupancy_pred[-1].F[target_prune[-1]], x_occupancy.F[:,0].long())
        # KL latent loss (approx to a gaussian)
        loss_latent = self.getLatentLoss(latent_mean, latent_logvar)

        loss = loss_prune + 0.001*loss_latent

        torch.cuda.empty_cache()

        self.log('val/loss_prune', loss_prune)
        #self.log('val/loss_sem', loss_sem)
        self.log('val/loss_latent', loss_latent)
        self.log('val/latent_mean', latent_mean.mean())
        self.log('train/latent_logvar', latent_logvar.mean())
        self.log('train/latent_std', torch.exp(0.5*latent_logvar).mean())
        self.log('val/loss', loss)
        self.log('val/miou', self.iou.compute())

        return loss

    def test_step(self, batch:dict, batch_idx):
        x_occupancy = points_to_tensor(batch['coords'], batch['feats'], self.hparams['data']['resolution'], self.global_step)
        # purning/occupancy loss
        occupancy_latent, occupancy_pred, pred_prune, target_prune = self.forward(x_occupancy, training=False)
        loss_prune0 = self.getLoss(pred_prune[0], target_prune[0].float())
        loss_prune1 = self.getLoss(pred_prune[1], target_prune[1].float())
        loss_prune2 = self.getLoss(pred_prune[2], target_prune[2].float())
        loss_prune3 = self.getLoss(pred_prune[3], target_prune[3].float())
        loss_prune = loss_prune0 + loss_prune1 + loss_prune2 + loss_prune3
        # semantic prediction loss
        #loss_sem = self.getSemLoss(occupancy_pred[-1].F[target_prune[-1]], x_occupancy.F[:,0].long())
        # KL latent loss (approx to a gaussian)
        loss_latent = self.getLatentLoss(occupancy_pred[-1].F, torch.randn(occupancy_pred[-1].F.shape, device=self.device))

        loss = loss_prune + loss_latent

        torch.cuda.empty_cache()

        self.log('val/loss_prune', loss_prune)
        #self.log('val/loss_sem', loss_sem)
        self.log('val/loss_latent', loss_latent)
        self.log('val/latent_mean', occupancy_latent.F.mean())
        self.log('val/latent_std', occupancy_latent.F.std())
        self.log('val/loss', loss)

        return loss

    def list_decoder_params(self):
        # during refinement we just optimize the decoder params
        optim_weights = []
        optim_weights += list(self.model.dense_projection.parameters())
        optim_weights += list(self.model.up0_prune_class.parameters())
        optim_weights += list(self.model.up1.parameters())
        optim_weights += list(self.model.up1_prune_class.parameters())
        optim_weights += list(self.model.up2.parameters())
        optim_weights += list(self.model.up2_prune_class.parameters())
        optim_weights += list(self.model.up3.parameters())
        optim_weights += list(self.model.up3_prune_class.parameters())
        optim_weights += list(self.model.last.parameters())

        return optim_weights

    def configure_optimizers(self):
        optim_weights = self.list_decoder_params() if self.hparams['train']['refine'] else self.parameters()
        optimizer = torch.optim.Adam(optim_weights, lr=self.hparams['train']['lr'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 5,
        }

        return [optimizer], [scheduler]
