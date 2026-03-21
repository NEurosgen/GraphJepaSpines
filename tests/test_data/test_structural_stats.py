import pytest
import torch
from torch_geometric.data import Data
from src.data_utils.transforms import (
    LaplacianPE,
    CentralityEncoding,
    RandomWalkPE,
    ConcatStructuralPE
)

@pytest.fixture
def dummy_graph():
    x = torch.ones(5, 3) # 5 nodes, 3 features
    # A simple chain graph: 0-1-2-3-4
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ])
    return Data(x=x, edge_index=edge_index)

@pytest.fixture
def disconnected_graph():
    x = torch.ones(5, 3)
    edge_index = torch.empty((2, 0), dtype=torch.long) # No edges
    return Data(x=x, edge_index=edge_index)

def test_laplacian_pe(dummy_graph, disconnected_graph):
    # Test valid graph
    k = 3
    transform = LaplacianPE(k=k)
    out = transform(dummy_graph)
    assert out.x.shape == (5, 3 + k) # 3 original + 3 PE
    
    # Test disconnected graph
    out_disc = transform(disconnected_graph)
    assert out_disc.x.shape == (5, 3 + k) # Handles gracefully with zeroes

def test_centrality_encoding(dummy_graph, disconnected_graph):
    transform = CentralityEncoding()
    
    out = transform(dummy_graph)
    assert out.x.shape == (5, 3 + 1) # Centrality is 1 column
    
    out_disc = transform(disconnected_graph)
    assert out_disc.x.shape == (5, 3 + 1)
    assert torch.all(out_disc.x[:, -1] == 0) # All degrees are 0

def test_random_walk_pe(dummy_graph, disconnected_graph):
    walk_length = 4
    transform = RandomWalkPE(walk_length=walk_length)
    
    out = transform(dummy_graph)
    assert out.x.shape == (5, 3 + walk_length)
    
    out_disc = transform(disconnected_graph)
    assert out_disc.x.shape == (5, 3 + walk_length)

def test_concat_structural_pe(dummy_graph):
    dummy_graph.laplacian_pe = torch.randn(5, 4)
    dummy_graph.centrality_pe = torch.randn(5, 1)
    dummy_graph.random_walk_pe = torch.randn(5, 8)
    
    transform = ConcatStructuralPE()
    out = transform(dummy_graph)
    assert out.x.shape == (5, 3 + 4 + 1 + 8)
