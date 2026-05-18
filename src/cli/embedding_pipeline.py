import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class EmbeddingSet:
    """Контейнер для хранения эмбеддингов и сопутствующих метаданных."""
    embeddings: torch.Tensor
    labels: torch.Tensor
    segment_ids: Optional[torch.Tensor] = None

    def save(self, path: str | os.PathLike) -> None:
        """Сохраняет эмбеддинги на диск."""
        state = {
            "embeddings": self.embeddings,
            "labels": self.labels,
            "segment_ids": self.segment_ids,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | os.PathLike, device: torch.device = torch.device('cpu')) -> "EmbeddingSet":
        """Загружает эмбеддинги с диска."""
        state = torch.load(path, map_location=device, weights_only=True)
        return cls(
            embeddings=state["embeddings"],
            labels=state["labels"],
            segment_ids=state["segment_ids"]
        )

    def pool_by_segment(self, pooling_type: Literal["mean", "sum", "amin", "amax"] = "mean") -> "EmbeddingSet":
        """
        Векторизованная группировка эмбеддингов по segment_ids.
        Возвращает новый экземпляр EmbeddingSet с агрегированными данными.
        """
        if self.segment_ids is None:
            raise ValueError("segment_ids отсутствуют, пулинг невозможен.")
            
        if self.embeddings.numel() == 0:
            return EmbeddingSet(self.embeddings.clone(), self.labels.clone(), self.segment_ids.clone())
            
        unique_segments, inverse_indices = torch.unique(self.segment_ids, return_inverse=True)
        num_segments = unique_segments.size(0)
        
        pooled_x = torch.zeros(
            (num_segments, self.embeddings.size(1)), 
            dtype=self.embeddings.dtype, 
            device=self.embeddings.device
        )
        pooled_x.scatter_reduce_(
            dim=0, 
            index=inverse_indices.unsqueeze(1).expand_as(self.embeddings), 
            src=self.embeddings, 
            reduce=pooling_type, 
            include_self=False
        )
        
        pooled_y = torch.empty(num_segments, dtype=self.labels.dtype, device=self.labels.device)
        pooled_y.scatter_(dim=0, index=inverse_indices, src=self.labels)
            
        return EmbeddingSet(embeddings=pooled_x, labels=pooled_y, segment_ids=unique_segments)


class EmbeddingExtractor:
    """Класс для извлечения эмбеддингов из датасетов с использованием энкодера."""
    
    def __init__(self, encoder: nn.Module, device: torch.device):
        self.encoder = encoder.to(device)
        self.device = device

    def extract_from_graph_dataset(
        self, 
        dataset: Dataset, 
        batch_size: int = 128,
        num_workers: int = 4,
        ignore_class: int = -1,
        desc: str = "Extracting"
    ) -> EmbeddingSet:
        """
        Извлекает эмбеддинги из графового датасета.
        Поддерживает батчи с наличием или отсутствием segment_id.
        """
        self.encoder.eval()
        
        embeddings_list, labels_list, segment_ids_list = [], [], []
        has_segments = False
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=lambda x: Batch.from_data_list(x)
        )
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                valid_mask = batch.y != ignore_class
                if not valid_mask.any():
                    continue
                    
                batch = batch.to(self.device)
                pooled_emb = self.encoder(batch)
                
                embeddings_list.append(pooled_emb[valid_mask].cpu())
                labels_list.append(batch.y[valid_mask].cpu())
                
                if hasattr(batch, 'segment_id') and batch.segment_id is not None:
                    has_segments = True
                    segment_ids_list.append(batch.segment_id[valid_mask].cpu())
                
        if len(embeddings_list) == 0:
            out_channels = getattr(self.encoder, 'out_channels', 0)
            return EmbeddingSet(
                embeddings=torch.empty((0, out_channels)), 
                labels=torch.empty(0), 
                segment_ids=torch.empty(0) if has_segments else None
            )
            
        return EmbeddingSet(
            embeddings=torch.cat(embeddings_list),
            labels=torch.cat(labels_list),
            segment_ids=torch.cat(segment_ids_list) if has_segments else None
        )
    

def main():
    encoder = load_encoder_from_folder(encoder_folder)
    encoder.eval().requires_grad_(False).to(device)
    gen_normalize = GenNormalize(transforms=[], mask_transform=None)

    get_class_fn = make_minnie65_class_getter(dm_cfg.dataset.class_path)

    print(f"Loading dataset from: {dataset_path}")
    ds = GraphDataSet(path=dataset_path, get_class=get_class_fn, transform=gen_normalize)
    macro_mean, macro_std = compute_macro_stats(ds)
    encoder_graph = GraphLatent(
        encoder=encoder,
        macro_mean=macro_mean,
        macro_std=macro_std,
        pooling=global_add_pool,
    ).to(device)
    extractor = EmbeddingExtractor(encoder=encoder, device=device)
    emb_set = extractor.extract_from_graph_dataset(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    if pooling_type is not None:
        if emb_set.segment_ids is not None:
            print(f"Выполняется пулинг (тип: {pooling_type})...")
            emb_set = emb_set.pool_by_segment(pooling_type=pooling_type)
        else:
            print("Предупреждение: pooling_type указан, но segment_ids отсутствуют в датасете. Пулинг пропущен.")

    # Создаем директорию для сохранения, если она не существует
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    emb_set.save(save_path)