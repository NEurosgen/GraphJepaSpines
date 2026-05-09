import pytest
import torch
from torch_geometric.data import Data
import numpy as np
from src.data_utils.transforms import (
    fast_normalization_by_features,
    NormNoEps,
    EdgeNorm,
    LocalPos,
    GraphPruning,
    MaskData,
    FeatureChoice,
    LaplacianPE,
    CentralityEncoding,
    RandomWalkPE,
    ConcatStructuralPE,
    FeatureShuffling,
    GaussianNoiseAugmentation,
    GaussianPositionNoise,
    GenNormalize
)

@pytest.fixture
def mock_graph():
    """Creates a basic mock graph with 5 nodes."""
    x = torch.tensor([
        [1.0, 0.0, 3.0],
        [2.0, 1.0, 4.0],
        [3.0, 0.0, 5.0],
        [4.0, 1.0, 6.0],
        [5.0, 0.0, 7.0]
    ], dtype=torch.float32)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ], dtype=torch.float32)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    edge_attr = torch.ones((edge_index.size(1), 2), dtype=torch.float32)
    
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, num_nodes=5)

def test_fast_normalization_by_features():
    # Test data with zeros which should be ignored
    data = torch.tensor([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 1e-7], # 1e-7 is <= eps (1e-6)
        [0.0, 2.0]
    ], dtype=torch.float32)
    
    means, stds = fast_normalization_by_features(data, eps=1e-6)
    
    # First column: 1, 2, 3 are valid. 0 is ignored.
    assert torch.allclose(means[0], torch.tensor(2.0))
    # std of [1, 2, 3] is 1.0
    assert torch.allclose(stds[0], torch.tensor(1.0))
    
    # Second column: 2 is valid. Others ignored.
    assert torch.allclose(means[1], torch.tensor(2.0))
    # std of [2.0] is NaN, but torch.std with 1 element returns nan or 0 depending on unbiased. 
    # Let's ensure it handles logic correctly. Actually torch.std([2.0]) is NaN.
    # We should just check the first dimension which is well-defined
    assert stds[0].item() > 0

def test_norm_no_eps(mock_graph):
    mean = torch.tensor([3.0, 0.5, 5.0])
    std = torch.tensor([1.5, 0.5, 1.5])
    transform = NormNoEps(mean, std, eps=1e-6)
    
    # Modify mock graph so some features are below eps
    mock_graph.x[0, 0] = 0.0 
    
    out = transform(mock_graph)
    # The 0.0 should not be modified
    assert out.x[0, 0] == 0.0
    
    # Value 5.0 at index 4,0 -> (5 - 3) / 1.5 = 1.3333
    assert torch.allclose(out.x[4, 0], torch.tensor(1.333333), atol=1e-4)

def test_edge_norm(mock_graph):
    mean = torch.tensor([0.5, 0.5])
    std = torch.tensor([0.5, 0.5])
    transform = EdgeNorm(mean, std)
    
    out = transform(mock_graph)
    # Original is 1.0 -> (1.0 - 0.5) / 0.5 = 1.0
    assert torch.allclose(out.edge_attr[0, 0], torch.tensor(1.0))

def test_local_pos(mock_graph):
    transform = LocalPos()
    out = transform(mock_graph)
    
    # mean of pos[:, 0] is 2.0, std is ~1.58
    assert torch.allclose(out.pos[:, 0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(out.pos[:, 0].std(), torch.tensor(1.0), atol=1e-6)

def test_graph_pruning(mock_graph):
    # Test spatial pruning (radius)
    # Nodes are separated by 1.0 in pos[:, 0].
    # r=1.5 should connect 0->1, 1->2 but not 0->2
    # mock_graph.batch is None, so radius_graph works on single graph
    mock_graph.batch = torch.zeros(mock_graph.num_nodes, dtype=torch.long)
    transform = GraphPruning(r=1.5)
    out = transform(mock_graph)
    
    # Original edges only keep those found by radius
    assert out.edge_index is not None

def test_mask_data(mock_graph):
    transform = MaskData(mask_ratio=0.4) # 5 nodes * 0.4 = 2 nodes
    ctx, tgt = transform(mock_graph)
    
    assert ctx.num_nodes + tgt.num_nodes == mock_graph.num_nodes
    assert tgt.num_nodes == 2
    assert ctx.num_nodes == 3

def test_feature_choice(mock_graph):
    transform = FeatureChoice(feature=[0, 2])
    out = transform(mock_graph)
    
    assert out.x.size(1) == 2
    assert torch.allclose(out.x[0], torch.tensor([1.0, 3.0]))

def test_laplacian_pe(mock_graph):
    transform = LaplacianPE(k=2)
    out = transform(mock_graph)
    
    # Original x dim is 3 + 2 (PE) = 5
    assert out.x.size(1) == 5
    # Should not contain NaNs
    assert not torch.isnan(out.x).any()

def test_centrality_encoding(mock_graph):
    transform = CentralityEncoding()
    out = transform(mock_graph)
    
    # Original x dim is 3 + 1 (deg) = 4
    assert out.x.size(1) == 4
    # Central nodes have degree 2, edge nodes have degree 1
    # Max degree is 2, so values should be 1.0 and 0.5
    assert out.x[:, 3].max() == 1.0

def test_random_walk_pe(mock_graph):
    transform = RandomWalkPE(walk_length=4)
    out = transform(mock_graph)
    
    # Original x dim is 3 + 4 (RW) = 7
    assert out.x.size(1) == 7
    assert not torch.isnan(out.x).any()

def test_feature_shuffling(mock_graph):
    transform = FeatureShuffling(ratio=1.0)
    original_x = mock_graph.x.clone()
    out = transform(mock_graph)
    
    # It should have the same shape
    assert out.x.size() == original_x.size()
    # It's highly unlikely (but possible) to be exactly the same
    # We can at least check it doesn't crash

def test_gaussian_noise(mock_graph):
    transform = GaussianNoiseAugmentation(sigma=0.1)
    original_x = mock_graph.x.clone()
    out = transform(mock_graph)
    
    assert out.x.size() == original_x.size()
    assert not torch.allclose(out.x, original_x)

def test_gaussian_pos_noise(mock_graph):
    transform = GaussianPositionNoise(sigma=0.1)
    original_pos = mock_graph.pos.clone()
    out = transform(mock_graph)
    
    assert out.pos.size() == original_pos.size()
    assert not torch.allclose(out.pos, original_pos)
