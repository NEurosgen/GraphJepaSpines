import pytest
import torch
from torch_geometric.data import Data, Batch
from src.data_utils.transforms import create_mask_collate_fn, GenNormalize, MaskData

def test_create_mask_collate_fn():
    # Test without transform
    collate_fn = create_mask_collate_fn(transform=None)
    data_list = [
        Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0,1],[1,0]])),
        Data(x=torch.ones(4, 2), edge_index=torch.tensor([[0,1],[1,0]]))
    ]
    batch = collate_fn(data_list)
    assert isinstance(batch, Batch)
    assert batch.num_nodes == 7
    
    # Test with MaskData transform
    mask_transform = MaskData(mask_ratio=0.5)
    dyn_transform = GenNormalize(transforms=[], mask_transform=mask_transform)
    collate_fn_with_mask = create_mask_collate_fn(transform=dyn_transform)
    
    ctx_batch, tgt_batch = collate_fn_with_mask(data_list)
    assert isinstance(ctx_batch, Batch)
    assert isinstance(tgt_batch, Batch)
    assert ctx_batch.num_nodes + tgt_batch.num_nodes == 7
