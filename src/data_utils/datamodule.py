import torch_geometric

import pytorch_lightning as pl
from typing import Optional

from torch.utils.data import random_split, DataLoader
from torch_geometric.data import Dataset
import torch
import random
import os
import re
import pandas as pd
from pathlib import Path

MAP = {
 '23P': 0,
 '4P': 0,
 '5P-IT': 0,
 '5P-NP': 0,
 '5P-PT': 0,
 '6P-CT': 0,
 '6P-IT': 0,
 'BC': 1,
 'BPC': 1,
 'MC': 1,
 'NGC': 1,
}

def get_class(df , file_path) -> int:
    neuron_id = int(re.findall(r'\d+', file_path.stem)[0])
    result = df.loc[df['segment_id']==neuron_id,'cell_type']
    if not result.empty:
        cell_type_value = MAP.get(result.values[0], 1)  
    else:
        cell_type_value = 1
    return cell_type_value



class GraphDataSet(Dataset):
    def __init__(self, path, transform=None, save_cache=False, class_path = None):
        super().__init__(None, None) 
        self.my_transform = transform
        self.path = path
        self.file_names = [f for f in os.listdir(self.path) if f.endswith('.pt')]

        self.cache = dict()
        self.save_cache = save_cache
        self.df = pd.read_csv(class_path) if class_path is not None else None

    
    def len(self):
        return len(self.file_names)
    
    def _load_to_cache(self, idx):
        file_path = Path(os.path.join(self.path, self.file_names[idx]))
        out = torch.load(file_path, weights_only=False)
        seg_id = int(re.findall(r'\d+', file_path.stem)[0])
        out.segment_id = torch.tensor(seg_id, dtype=torch.long)
        if self.df is not None:
            cell_type_value = get_class(self.df, file_path)
            out.y = cell_type_value
        self.cache[idx] = out
        return out

    def get(self, idx):
        if self.save_cache and idx in self.cache:
            out = self.cache[idx]
        else:
            out = self._load_to_cache(idx)
            
        if self.my_transform is not None:
            return self.my_transform(out)
        
        return out


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size: int, num_workers: int = 4, seed: int = 42, ratio: list = None, collate_fn = None):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.ratio = ratio if ratio is not None else [0.7, 0.2, 0.1]
        self.collate_fn = collate_fn

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None):
        indices = torch.arange(len(self.dataset))
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(self.dataset), generator=generator)
        
        train_size = int(len(self.dataset) * self.ratio[0])
        val_size = int(len(self.dataset) * self.ratio[1])
        
        self.train_ds = self.dataset[perm[:train_size]]
        self.val_ds = self.dataset[perm[train_size:train_size+val_size]]
        self.test_ds = self.dataset[perm[train_size+val_size:]]

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                           batch_size=self.batch_size,
                           shuffle=True,
                           num_workers=self.num_workers,
                           persistent_workers=self.num_workers > 0, 
                           pin_memory=True,
                           collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds,
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers,
                           persistent_workers=self.num_workers > 0, 
                           pin_memory=True,
                           collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds,
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers,
                           persistent_workers=self.num_workers > 0,
                           pin_memory=True,
                           collate_fn=self.collate_fn)

