import click
from os.path import join, dirname, abspath
from os import makedirs
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import yaml
import numpy as np

import diss.datasets.datasets as datasets
import diss.models.diff_latent as diff_latent
import diss.models.scan_cond_diff as scan_cond_diff

def set_deterministic():
    np.random.seed(16)
    torch.manual_seed(16)
    torch.cuda.manual_seed(16)
    torch.backends.cudnn.deterministic = True

def get_diff_model(condition, diff_weights, cfg):
    diff_models = {
            'uncond': diff_latent.DiffLatent,
            'single_scan': scan_cond_diff.ScanCondDiffLatent,
        }

    if diff_weights is None:
        model = diff_models[condition](cfg)
    else:
        model = diff_models[condition].load_from_checkpoint(diff_weights,hparams=cfg)

    return model


@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/diff.yaml'))
@click.option('--vae_weights',
              '-v',
              type=str,
              help='path to pretrained vae weights (.ckpt).',
              default=None)
@click.option('--diff_weights',
              '-d',
              type=str,
              help='path to diffusion checkpoint (.ckpt).',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--condition',
              '-cond',
              type=str,
              help='type of conditioning (possibilities: uncond, single_scan)',
              default='uncond')
@click.option('--test', '-t', is_flag=True, help='test mode')
def main(config, vae_weights, diff_weights, checkpoint, condition, test):
    set_deterministic()
    cfg = yaml.safe_load(open(config))
    cfg['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())
    print('\033[92m' + f'\nDIFFUSION TRAINING CONDITION: {condition.upper()}\n' + '\033[0m')

    #Load data and model
    data = datasets.KittiDataModule(cfg) if condition == 'uncond' else datasets.CondKittiDataModule(cfg)
    model = get_diff_model(condition, diff_weights, cfg)

    # load pre-trained vae weights
    vae_ckpt = torch.load(vae_weights)
    model.load_state_dict(vae_ckpt['state_dict'], strict=False)

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(monitor='val/loss',
                                 filename=cfg['experiment']['id']+'_{epoch:02d}',
                                 mode='min',
                                 save_top_k=-1)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    #Save git diff to keep track of all changes
    makedirs(tb_logger.log_dir, exist_ok=True)
    with open(f'{tb_logger.log_dir}/project.diff', 'w+') as diff_file:
        repo_diff = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE)
        diff_file.write(repo_diff.stdout.decode('utf-8'))

    print(cfg)

    #Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      max_epochs=cfg['train']['max_epoch'],
                      callbacks=[lr_monitor, checkpoint_saver],
                      log_every_n_steps=100,
                      check_val_every_n_epoch=10,
                      num_sanity_val_steps=0,
                      accelerator='ddp',
                      )

    if not test:
        trainer.fit(model, data)
    else:
        trainer.test(model, data)

if __name__ == "__main__":
    main()
