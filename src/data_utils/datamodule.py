import torch_geometric

import pytorch_lightning as pl
from typing import Optional

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch
import random

seed = 1#from confing

class GraphDataModule(pl.LightningDataModule):
    def __init__(self ,dataset , batch_size , num_workers, ratio = [0.7,0.2, 0.1]):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ratio = ratio

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
    def setup(self, stage:Optional[str] = None):
        
        total_size = len(self.dataset)
        train_size ,val_size  = int(total_size*self.ratio[0]) ,int(total_size*self.ratio[1]) 
        test_size = total_size - (train_size + val_size)

        generator = torch.Generator().manual_seed(seed)

        self.train_ds , self.val_ds ,self.test_ds = random_split(
            self.dataset ,[train_size, val_size, test_size] , generator=generator
        )


    def train_dataloader(self):
        return DataLoader(self.train_ds,
                           batch_size=self.batch_size,
                           shuffle=True,
                           num_workers = self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_ds,
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers = self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_ds,
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers = self.num_workers)

