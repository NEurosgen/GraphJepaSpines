import gc
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Any

import torch
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch
from tqdm import tqdm

from src.data_utils.datamodule import GraphDataSet
from src.data_utils.stats import compute_macro_stats
from src.models.encoder import GraphLatent
from src.models.loader_model import load_encoder_from_folder

def pool_by_segment(
    embeddings: torch.Tensor, 
    labels: torch.Tensor, 
    segment_ids: torch.Tensor, 
    pooling_type: Literal["mean", "sum", "amin", "amax"] = "mean"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Векторизованная группировка эмбеддингов по segment_ids.
    """
    if embeddings.numel() == 0:
        return embeddings, labels
        
    unique_segments, inverse_indices = torch.unique(segment_ids, return_inverse=True)
    num_segments = unique_segments.size(0)
    pooled_x = torch.zeros(
        (num_segments, embeddings.size(1)), 
        dtype=embeddings.dtype, 
        device=embeddings.device
    )
    pooled_x.scatter_reduce_(
        dim=0, 
        index=inverse_indices.unsqueeze(1).expand_as(embeddings), 
        src=embeddings, 
        reduce=pooling_type, 
        include_self=False
    )
    
    pooled_y = torch.empty(num_segments, dtype=labels.dtype, device=labels.device)
    pooled_y.scatter_(dim=0, index=inverse_indices, src=labels)
        
    return pooled_x, pooled_y

def extract_from_dataset(
    dataset: GraphDataSet, 
    encoder_graph: nn.Module, 
    device: torch.device, 
    batch_size: int = 128,
    num_workers: int = 4,
    ignore_class: int = -1,
    desc: str = "Extracting"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    embeddings, labels, segment_ids = [], [], []
    encoder_graph.eval()
    
    loader = torch.utils.data.DataLoader(
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
                
            batch = batch.to(device)
            pooled_emb = encoder_graph(batch)
            
            embeddings.append(pooled_emb[valid_mask].cpu())
            labels.append(batch.y[valid_mask].cpu())
            segment_ids.append(batch.segment_id[valid_mask].cpu())
            
    if len(embeddings) == 0:
        # Корректное определение размерности пустого тензора
        out_channels = getattr(encoder_graph.encoder, 'out_channels', 0)
        return torch.empty((0, out_channels)), torch.empty(0), torch.empty(0)
        
    return torch.cat(embeddings), torch.cat(labels), torch.cat(segment_ids)

@dataclass
class DatamoduleConfig:
    extra: dict = field(default_factory=dict)   
 
    def get(self, key: str, default: Any = None) -> Any:            
        return self.extra.get(key, default)
 
    def __getattr__(self, key: str) -> Any:
        try:
            return self.extra[key]
        except KeyError:
            raise AttributeError(key)

@dataclass
class EmbeddingConfig:
    stats_path: str
    class_csv_path: str
    pooling_level: Literal["neuron", "graph"] = "graph"
    pooling_type: Literal["mean", "max", "sum"] = "mean"
    sigma: float = 1.0
    batch_size: int = 128
    num_workers: int = 4
    ignore_class: int = -1
    global_pool_fn: Callable = global_add_pool

class EmbeddingPipeline:
    def __init__(
        self,
        emb_cfg: EmbeddingConfig,
        dm_cfg: DatamoduleConfig,
        build_class_getter: Callable,
        cell_type_map: Optional[dict[str, int]] = None,
    ):
        self.emb_cfg = emb_cfg
        self.dm_cfg = dm_cfg
        self.build_class_getter = build_class_getter
        self.cell_type_map = cell_type_map

    def run(
        self,
        encoder_folder: str,
        dataset_path: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        encoder, encoder_graph = self._load_encoder(encoder_folder, dataset_path, device)
 
        try:
            emb_all, y_all, seg_all = extract_from_dataset(
                dataset=self._build_dataset(dataset_path),
                encoder_graph=encoder_graph,
                device=device,
                batch_size=self.emb_cfg.batch_size,
                num_workers=self.emb_cfg.num_workers,
                ignore_class=self.emb_cfg.ignore_class,
                desc="Extracting All"
            )
        finally:
            self._cleanup(encoder, encoder_graph)
 
        return self._pool(emb_all, y_all, seg_all)

    def _load_encoder(self, encoder_folder: str, dataset_path: str, device: torch.device) -> tuple[nn.Module, nn.Module]:
        print(f"[EmbeddingPipeline] Загрузка энкодера: {encoder_folder}")
        encoder = load_encoder_from_folder(encoder_folder)
        encoder.eval().requires_grad_(False).to(device)
 
        ds = self._build_dataset(dataset_path)
        macro_mean, macro_std = compute_macro_stats(ds)
 
        encoder_graph = GraphLatent(
            encoder=encoder,
            macro_mean=macro_mean,
            macro_std=macro_std,
            pooling=self.emb_cfg.global_pool_fn, 
            sigma=self.emb_cfg.sigma,
        ).to(device)
 
        return encoder, encoder_graph
 
    def _build_dataset(self, dataset_path: str) -> GraphDataSet:
      
        get_class = self.build_class_getter(
            self.emb_cfg.class_csv_path,
            self.cell_type_map,
        )
 
        print(f"[EmbeddingPipeline] Загрузка датасета: {dataset_path}")
        return GraphDataSet(path=dataset_path, get_class=get_class, transform=None)
 
    def _pool(
        self,
        emb_all: torch.Tensor,
        y_all: torch.Tensor,
        seg_all: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        level = self.emb_cfg.pooling_level
        if level == "neuron":
            ptype = self.emb_cfg.pooling_type
            
            reduce_map = {"mean": "mean", "max": "amax", "sum": "sum"}
            scatter_ptype = reduce_map.get(ptype, "mean")
            
            print(f"[EmbeddingPipeline] Пулинг: level={level}, type={scatter_ptype}")
            return pool_by_segment(emb_all, y_all, seg_all, scatter_ptype)
        return emb_all, y_all
 
    @staticmethod
    def _cleanup(*models: nn.Module):
        for m in models:
            del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()