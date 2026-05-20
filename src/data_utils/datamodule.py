import logging

import pytorch_lightning as pl
from typing import Optional, Callable

from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
import torch
import re
from pathlib import Path


class GraphDataSet(Dataset):
    def __init__(self, path, get_class: Callable = None, transform=None, save_cache=False):
        super().__init__(None, None) 
        self.my_transform = transform
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} don't exist.")
        if not self.path.is_dir():
            raise NotADirectoryError(f"Path {self.path} must be a directory.")
        self.file_paths = sorted(self.path.rglob('*.pt'))
        if not self.file_paths:
            raise FileNotFoundError(f"In the directory {self.path} and its subdirectories, no '.pt' files were found.")
        self.get_class = get_class
        self.cache = dict()
        self.save_cache = save_cache
       

    
    def len(self):
        return len(self.file_paths)
    
    def _load_file(self, idx):
        file_path = self.file_paths[idx]
        out = torch.load(file_path, weights_only=False)
        out.path = file_path.relative_to(self.path)
        if self.get_class is not None:
            try:
                out.y = self.get_class(file_path=file_path, out=out)
            except TypeError:
                out.y = self.get_class(file_path)
                
        if hasattr(out, 'segment_id') and isinstance(out.segment_id, str):
            match = re.search(r'\d+', out.segment_id)
            if match:
                out.segment_id = torch.tensor(int(match.group(0)), dtype=torch.long)
        elif not hasattr(out, 'segment_id'):
            try:
                seg_id = int(re.findall(r'\d+', file_path.stem)[0])
                out.segment_id = torch.tensor(seg_id, dtype=torch.long)
            except Exception:
                pass

        return out

    def get(self, idx):
        if self.save_cache and idx in self.cache:
            return self.cache[idx]
            
        out = self._load_file(idx)
        
        if self.my_transform is not None:
            out = self.my_transform(out)
            
        if self.save_cache:
            self.cache[idx] = out
            
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
        # Разогрев кэша в главном потоке: если save_cache=True, мы один раз переводим все графы в RAM
        # с примененными статическими трансформациями. Когда DataLoader запустит num_workers, 
        # все воркеры получат доступ к уже заполненному кэшу в RAM.
        if self.dataset.save_cache:
            logging.info("Pre-loading dataset into RAM cache...")
            for i in range(len(self.dataset)):
                _ = self.dataset[i]

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





