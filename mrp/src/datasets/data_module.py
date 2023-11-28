import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from . import image_dataset, image_dataset_mg, image_dataset_duke, image_dataset_hybrid, image_dataset_ispy2
from .. import builder

class INBDataModule(pl.LightningDataModule):
    def __init__(self, cfg, args):
        super().__init__()
        self.args = args
        self.cfg = cfg
        if self.cfg.modal=='MRI':
            if self.cfg.datasetname == 'duke':
                self.dataset = image_dataset_duke.INBImageDataset
            if self.cfg.datasetname == 'ispy2':
                self.dataset = image_dataset_ispy2.INBImageDataset
            if self.cfg.datasetname == 'inhouse':
                self.dataset = image_dataset.INBImageDataset
            if self.cfg.datasetname == 'duke_inhouse':
                self.dataset = image_dataset_hybrid.INBImageDataset
        else:
            self.dataset = image_dataset_mg.INBImageDataset

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, self.args, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "valid")
        dataset = self.dataset(self.cfg, self.args, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=1,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, self.args, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.test.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

