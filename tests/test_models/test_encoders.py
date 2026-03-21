import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool
from src.models.encoder import GraphGinEncoder, GraphGcnEncoder, GraphLatent

@pytest.fixture
def dummy_batch():
    x = torch.randn(10, 5) # 10 nodes total, 5 features
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 0, 2, 1, 4, 3, 6, 5, 8, 7]
    ])
    edge_attr = torch.randn(10) # GCN expects 1D edge_weight
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
    
    macro_metrics = torch.randn(4, 7) # 4 graphs, 7 macro metrics
    
    data = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx)
    data.macro_metrics = macro_metrics
    return data

def test_graph_gin_encoder(dummy_batch):
    in_channels = 5
    out_channels = 16
    encoder = GraphGinEncoder(in_channels=in_channels, out_channels=out_channels, num_layers=2)
    
    out = encoder(dummy_batch.x, dummy_batch.edge_index, dummy_batch.edge_attr)
    assert out.shape == (10, out_channels)
    
    # Check gradients
    loss = out.sum()
    loss.backward()
    assert encoder.proj.weight.grad is None # Because we have self.proj.requires_grad_(False)
    assert encoder.layers[0].model.nn[0].weight.grad is not None

def test_graph_gcn_encoder(dummy_batch):
    in_channels = 5
    out_channels = 16
    encoder = GraphGcnEncoder(in_channels=in_channels, out_channels=out_channels, num_layers=2)
    
    out = encoder(dummy_batch.x, dummy_batch.edge_index, dummy_batch.edge_attr)
    assert out.shape == (10, out_channels)

def test_graph_latent(dummy_batch):
    in_channels = 5
    out_channels = 16
    encoder = GraphGcnEncoder(in_channels=in_channels, out_channels=out_channels, num_layers=1)
    
    macro_mean = torch.zeros(1, 7)
    macro_std = torch.ones(1, 7)
    
    latent_model = GraphLatent(
        encoder=encoder, 
        macro_mean=macro_mean, 
        macro_std=macro_std, 
        pooling=global_add_pool,
        sigma=1.0
    )
    
    out = latent_model(dummy_batch)
    # 4 graphs in batch. output should be [4, out_channels + num_macro]
    assert out.shape == (4, out_channels + 7)
