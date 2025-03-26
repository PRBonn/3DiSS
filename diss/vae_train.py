import click
from os.path import join, dirname, abspath
from os import makedirs
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import torch
import yaml

import diss.datasets.datasets as datasets
import diss.models.sem_vae as sem_vae

def set_deterministic():
    np.random.seed(16)
    torch.manual_seed(16)
    torch.cuda.manual_seed(16)
    torch.backends.cudnn.deterministic = True

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/vae_config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
def main(config,weights,checkpoint):
    set_deterministic()
    cfg = yaml.safe_load(open(config))
    cfg['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())

    #Load data and model
    data = datasets.KittiDataModule(cfg)
    if weights is None:
        model = sem_vae.AutoEncoder(cfg)
    else:
        model = sem_vae.AutoEncoder.load_from_checkpoint(weights,hparams=cfg)

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
                      check_val_every_n_epoch=5,
                      num_sanity_val_steps=0,
                      accelerator='ddp',
                      )

    # Train!
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
