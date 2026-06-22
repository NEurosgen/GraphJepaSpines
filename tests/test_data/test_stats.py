import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from src.data_utils.structural_stats import ThesisMacroMetrics
from src.data_utils.stats import compute_macro_stats, extract_macro_features

@pytest.fixture
def mock_disjoint_triangles():
    """Creates a graph with two disconnected triangles (6 nodes)."""
    # Nodes 0,1,2 form a triangle. Nodes 3,4,5 form a triangle.
    x = torch.ones(6, 1)
    
    # Edges (undirected)
    # Triangle 1: (0,1), (1,0), (1,2), (2,1), (0,2), (2,0)
    # Triangle 2: (3,4), (4,3), (4,5), (5,4), (3,5), (5,3)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 2, 3, 4, 4, 5, 3, 5],
        [1, 0, 2, 1, 2, 0, 4, 3, 5, 4, 5, 3]
    ], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, num_nodes=6)

def test_thesis_macro_metrics(mock_disjoint_triangles):
    metric_module = ThesisMacroMetrics()
    data = metric_module(mock_disjoint_triangles)
    
    assert hasattr(data, 'macro_metrics')
    metrics = data.macro_metrics.squeeze()
    
    # 1. Avg subgraph size = 3
    assert torch.allclose(metrics[0], torch.tensor(3.0))
    # 2. Avg intra distance = 1.0 (all nodes are connected to each other in triangle)
    assert torch.allclose(metrics[1], torch.tensor(1.0))
    # 3. Modularity - Louvain will find 2 communities, modularity is > 0.
    assert metrics[2] > 0.0
    # 4. Clustering coeff = 1.0
    assert torch.allclose(metrics[3], torch.tensor(1.0))
    # 5. Num nodes = 6
    assert torch.allclose(metrics[4], torch.tensor(6.0))
    # 6. Num edges = 6 (in networkx, undirected)
    assert torch.allclose(metrics[5], torch.tensor(6.0))
    # 7. Density = 2 * 6 / (6 * 5) = 12 / 30 = 0.4
    assert torch.allclose(metrics[6], torch.tensor(0.4))

def test_compute_macro_stats():
    # Create dummy dataset containing mock graphs with different macro metrics
    class DummyDataset:
        def __init__(self):
            self.data = []
            for i in range(10):
                d = Data(x=torch.ones(1, 1))
                d.macro_metrics = torch.ones(1, 7) * i
                self.data.append(d)
                
        def __len__(self):
            return 10
            
        def __getitem__(self, idx):
            return self.data[idx]

    ds = DummyDataset()
    macro_mean, macro_std = compute_macro_stats(ds, max_samples=10)
    
    # Metrics are from 0 to 9.
    # Mean of 0..9 is 4.5
    assert torch.allclose(macro_mean, torch.ones(1, 7) * 4.5)
    # Std of 0..9 is ~3.02765
    assert torch.allclose(macro_std, torch.ones(1, 7) * np.std(np.arange(10), ddof=1), atol=1e-4)

def test_extract_macro_features():
    # If a graph has macro_metrics
    data = Data(x=torch.ones(1, 1))
    data.macro_metrics = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]])
    
    macro_mean = torch.zeros(1, 7)
    macro_std = torch.ones(1, 7) * 10.0
    
    features = extract_macro_features(data, macro_mean, macro_std)
    
    # Expected: (metrics - 0) / 10 = [1, 2, 3, 4, 5, 6, 7]
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    assert torch.allclose(features, expected)
    
    # Test fallback if no macro metrics
    data_empty = Data(x=torch.ones(1, 1))
    features_empty = extract_macro_features(data_empty, macro_mean, macro_std)
    
    assert features_empty.shape == (1, 7)
    assert torch.allclose(features_empty, torch.zeros(1, 7))
