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
        self.path = Path(path)
        # Рекурсивный поиск .pt файлов (поддержка подпапок вроде ab/, wt/)
        self.file_paths = sorted(self.path.rglob('*.pt'))
        self.get_class = get_class
        self.cache = dict()
        self.save_cache = save_cache
       

    
    def len(self):
        return len(self.file_paths)
    
    def _load_file(self, idx):
        file_path = self.file_paths[idx]
        out = torch.load(file_path, weights_only=False)
        
        if self.get_class is not None:
            try:
                out.y = self.get_class(file_path=file_path, out=out)
            except TypeError:
                out.y = self.get_class(file_path)
                
        if hasattr(out, 'segment_id') and isinstance(out.segment_id, str):
            match = re.search(r'\d+', out.segment_id)
            if match:
                out.segment_id = torch.tensor(int(match.group(0)), dtype=torch.long)
        else:
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
            import logging
            logging.info("Pre-loading dataset into RAM cache...")
            for i in range(len(self.dataset)):
                _ = self.dataset[i]
                
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

    def get_class(file_path: Path, **kwargs) -> torch.Tensor:
        folder_name = Path(file_path).parent.name.lower()
        if folder_name not in mapping:
            raise ValueError(
                f"Folder '{folder_name}' not in mapping {mapping}. "
                f"File: {file_path}"
            )
        return torch.tensor(mapping[folder_name], dtype=torch.long)

    return get_class

def make_minnie65_class_getter(csv_path: str) -> Callable:
    """
    Создаёт функцию для определения класса графа по segment_id 
    для датасета minnie65. Неизвестные классы мапятся в -1.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['segment_id', 'cell_type'])
    mapping = {str(int(row['segment_id'])): row['cell_type'] for _, row in df.iterrows()}
    
    class_map = {
        '23P': 0, '4P': 0, '5P-IT': 0, '5P-NP': 0, '5P-PT': 0,
        '6P-CT': 0, '6P-IT': 0, 'BC': 1, 'BPC': 1, 'MC': 1, 'NGC': 1
    }
    
    def get_class(file_path: Path, out=None, **kwargs) -> torch.Tensor:
        segment_id = None
        if out is not None and hasattr(out, 'segment_id') and isinstance(out.segment_id, str):
            match = re.search(r'\d+', out.segment_id)
            if match:
                segment_id = match.group(0)
                
        if segment_id is None:
            filename = Path(file_path).name
            match = re.search(r'\d+', filename)
            if not match:
                raise ValueError(f"Could not find segment_id in filename: {filename}")
            segment_id = match.group(0)
            
        if segment_id not in mapping:
            # If not in CSV, return -1 (to be filtered out)
            return torch.tensor(-1, dtype=torch.long)
            
        cell_type = mapping[segment_id]
        if cell_type not in class_map:
            # Unknown cell type, return -1
            return torch.tensor(-1, dtype=torch.long)
            
        return torch.tensor(class_map[cell_type], dtype=torch.long)
        
    return get_class
