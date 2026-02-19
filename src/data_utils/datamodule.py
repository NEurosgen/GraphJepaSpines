import torch_geometric

import pytorch_lightning as pl
from typing import Optional, Callable, Dict

from torch.utils.data import random_split, DataLoader
from torch_geometric.data import Dataset
import torch
import random
import os
import re
import pandas as pd
from pathlib import Path


class GraphDataSet(Dataset):
    def __init__(self, path, get_class: Callable = None, transform=None, save_cache=False):
        super().__init__(None, None) 
        self.my_transform = transform
        self.path = path
        self.file_names = [f for f in os.listdir(self.path) if f.endswith('.pt')]
        self.get_class = get_class
        self.cache = dict()
        self.save_cache = save_cache
       

    
    def len(self):
        return len(self.file_names)
    
    def _load_to_cache(self, idx):
        file_path = Path(os.path.join(self.path, self.file_names[idx]))
        out = torch.load(file_path, weights_only=False)
        seg_id = int(re.findall(r'\d+', file_path.stem)[0])
        out.segment_id = torch.tensor(seg_id, dtype=torch.long)
        if self.get_class is not None:
            out.y = self.get_class(file_path)
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


def make_folder_class_getter(folder_to_label: Dict[str, int]) -> Callable:
    """
    Создаёт get_class функцию, которая определяет класс графа
    по имени родительской папки.

    Args:
        folder_to_label: маппинг имя_папки -> числовой_label.
            Сравнение регистронезависимое.
            Пример: {"ab": 0, "wt": 1}

    Returns:
        Callable[[Path], torch.Tensor]: функция file_path -> label tensor
    """
    mapping = {k.lower(): v for k, v in folder_to_label.items()}

    def get_class(file_path: Path) -> torch.Tensor:
        folder_name = Path(file_path).parent.name.lower()
        if folder_name not in mapping:
            raise ValueError(
                f"Folder '{folder_name}' not in mapping {mapping}. "
                f"File: {file_path}"
            )
        return torch.tensor(mapping[folder_name], dtype=torch.long)

    return get_class
