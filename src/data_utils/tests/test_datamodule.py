import pytest
import torch
import os
from pathlib import Path
from torch_geometric.data import Data
from src.data_utils.datamodule import GraphDataSet, GraphDataModule

@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Creates a temporary directory with dummy .pt files representing graphs."""
    dataset_dir = tmp_path / "dummy_dataset"
    dataset_dir.mkdir()
    
    # Create 10 dummy files
    for i in range(10):
        # file names like "graph_0.pt", "graph_1.pt", ...
        file_path = dataset_dir / f"graph_{i}.pt"
        data = Data(
            x=torch.ones(5, 3) * i,
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            y=torch.tensor([i]),
            segment_id=f"seg_{i}" # String to test regex
        )
        torch.save(data, file_path)
        
    return dataset_dir

def test_graph_dataset_len(mock_dataset_dir):
    ds = GraphDataSet(path=mock_dataset_dir)
    assert len(ds) == 10

def test_graph_dataset_get_segment_id_regex(mock_dataset_dir):
    ds = GraphDataSet(path=mock_dataset_dir)
    data = ds[0]
    
    # It should have extracted the integer from "seg_0" or the filename
    assert hasattr(data, 'segment_id')
    assert isinstance(data.segment_id, torch.Tensor)

def test_graph_dataset_caching(mock_dataset_dir):
    ds = GraphDataSet(path=mock_dataset_dir, save_cache=True)
    
    assert len(ds.cache) == 0
    
    data = ds[0] # Should load from disk and cache
    assert len(ds.cache) == 1
    assert 0 in ds.cache
    
    # Modify cache manually to verify it reads from cache next time
    ds.cache[0].x = torch.zeros(5, 3)
    data2 = ds[0]
    assert torch.allclose(data2.x, torch.zeros(5, 3))

def test_graph_dataset_custom_get_class(mock_dataset_dir):
    # Dummy getter that always returns 42
    def custom_getter(file_path, out=None):
        return torch.tensor([42])
        
    ds = GraphDataSet(path=mock_dataset_dir, get_class=custom_getter)
    data = ds[0]
    assert data.y.item() == 42

def test_graph_datamodule_setup(mock_dataset_dir):
    ds = GraphDataSet(path=mock_dataset_dir)
    
    # 10 samples total, ratio [0.7, 0.2, 0.1] -> 7 train, 2 val, 1 test
    dm = GraphDataModule(dataset=ds, batch_size=2, num_workers=0, ratio=[0.7, 0.2, 0.1])
    dm.setup()
    
    assert len(dm.train_ds) == 7
    assert len(dm.val_ds) == 2
    assert len(dm.test_ds) == 1

def test_graph_datamodule_dataloaders(mock_dataset_dir):
    from torch_geometric.data import Batch
    ds = GraphDataSet(path=mock_dataset_dir)
    dm = GraphDataModule(dataset=ds, batch_size=2, num_workers=0, ratio=[0.5, 0.3, 0.2], collate_fn=Batch.from_data_list)
    dm.setup()
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    # Check if we get a Batch object with 2 graphs
    assert hasattr(batch, 'batch')
    assert batch.num_graphs == 2
    
    val_loader = dm.val_dataloader()
    batch_val = next(iter(val_loader))
    assert batch_val.num_graphs == 2 # (since we have 3 val items, first batch is 2)
