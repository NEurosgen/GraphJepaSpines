import torch_geometric

import pytorch_lightning as pl
from typing import Optional

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch
import random
import os



class GraphDataSet(Dataset):
    def __init__(self, path, transform=None):
        super().__init__(None, transform)
        self.transform = transform
        self.path = path
        self.file_names = [f for f in os.listdir(self.path) if f.endswith('.pt')]
    
    def len(self):
        return len(self.file_names)
    
    def get(self, idx):
        file_path = os.path.join(self.path, self.file_names[idx])
        # weights_only=False is needed for loading torch_geometric Data objects in PyTorch 2.6+
        data = torch.load(file_path, weights_only=False)
        if self.transform is not None:
            data = self.transform(data)
        return data





class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size: int, num_workers: int = 4, seed: int = 42, ratio: list = None):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.ratio = ratio if ratio is not None else [0.7, 0.2, 0.1]

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
    def setup(self, stage:Optional[str] = None):
        
        total_size = len(self.dataset)
        train_size ,val_size  = int(total_size*self.ratio[0]) ,int(total_size*self.ratio[1]) 
        test_size = total_size - (train_size + val_size)

        generator = torch.Generator().manual_seed(self.seed)

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

