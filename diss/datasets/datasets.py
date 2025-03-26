import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from diss.datasets.dataloader.SemanticKITTI import KITTISet
from diss.datasets.dataloader.SemanticKITTICond import CondKITTISet
from diss.utils.collations import SparseCollation
import warnings

warnings.filterwarnings('ignore')

class CondKittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseCollation()

        data_set = CondKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            split=self.cfg['data']['train'],
            resolution=self.cfg['data']['resolution'],
            xyz_range=self.cfg['data']['xyz_range_train'],
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseCollation()

        data_set = CondKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            split=self.cfg['data']['validation'],
            resolution=self.cfg['data']['resolution'],
            xyz_range=self.cfg['data']['xyz_range_val'],
            )
        loader = DataLoader(data_set, batch_size=1,#self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseCollation()

        data_set = CondKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            split=self.cfg['data']['train'],
            resolution=self.cfg['data']['resolution'],
            xyz_range=self.cfg['data']['xyz_range_val'],
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class KittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseCollation()

        data_set = KITTISet(
            data_dir=self.cfg['data']['data_dir'],
            split=self.cfg['data']['train'],
            resolution=self.cfg['data']['resolution'],
            xyz_range=self.cfg['data']['xyz_range_train'],
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = SparseCollation()

        data_set = KITTISet(
            data_dir=self.cfg['data']['data_dir'],
            split=self.cfg['data']['validation'],
            resolution=self.cfg['data']['resolution'],
            xyz_range=self.cfg['data']['xyz_range_val'],
            )
        loader = DataLoader(data_set, batch_size=1,#self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = SparseCollation()

        data_set = KITTISet(
            data_dir=self.cfg['data']['data_dir'],
            split=self.cfg['data']['train'],
            resolution=self.cfg['data']['resolution'],
            xyz_range=self.cfg['data']['xyz_range_val'],
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

dataloaders = {
    'KITTI': KittiDataModule,
    'CondKITTI': CondKittiDataModule,
}
