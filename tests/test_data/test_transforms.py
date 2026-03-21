import pytest
import torch
from torch_geometric.data import Data
from src.data_utils.transforms import (
    MaskData,
    GraphPruning,
    NormNoEps,
    EdgeNorm,
    LocalPos,
    FeatureChoice,
    GenNormalize
)

@pytest.fixture
def dummy_data():
    x = torch.randn(10, 5) # 10 nodes, 5 features
    pos = torch.randn(10, 3) # 10 nodes, 3D positions
    edge_index = torch.randint(0, 10, (2, 20)) # 20 random edges
    edge_attr = torch.randn(20, 2) # 20 edges, 2 features
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

def test_mask_data(dummy_data):
    transform = MaskData(mask_ratio=0.3)
    context, target = transform(dummy_data)
    
    # Check if they are valid PyG Data objects
    assert isinstance(context, Data)
    assert isinstance(target, Data)
    
    # Context + target nodes should roughly equal the total (though subgraph might drop isolated ones but indices sum to num_nodes)
    assert context.num_nodes + target.num_nodes == dummy_data.num_nodes

def test_graph_pruning(dummy_data):
    # Test KNN pruning
    knn_prune = GraphPruning(k=2)
    out_knn = knn_prune(dummy_data)
    assert out_knn.num_nodes == dummy_data.num_nodes
    # Since we are filtering by existing edges, we just check no crashes and output is Data
    assert isinstance(out_knn, Data)
    
    # Test Radius pruning
    rad_prune = GraphPruning(r=1.5)
    out_rad = rad_prune(dummy_data)
    assert out_rad.num_nodes == dummy_data.num_nodes
    assert isinstance(out_rad, Data)

def test_norm_no_eps(dummy_data):
    mean = torch.zeros(5)
    std = torch.ones(5)
    transform = NormNoEps(mean=mean, std=std)
    
    # Values close to 0 should be ignored (not normalized)
    dummy_data.x[0, :] = 1e-8
    out = transform(dummy_data)
    
    assert torch.allclose(out.x[0, :], torch.tensor(1e-8)) # 0s are unchanged due to mask
    assert out.x.shape == (10, 5)

def test_edge_norm(dummy_data):
    mean = torch.zeros(2)
    std = torch.ones(2)
    transform = EdgeNorm(mean=mean, std=std)
    out = transform(dummy_data)
    assert out.edge_attr.shape == (20, 2)

def test_local_pos(dummy_data):
    transform = LocalPos()
    out = transform(dummy_data)
    assert out.pos.shape == (10, 3)
    # Mean of centered pos should be roughly 0
    assert torch.allclose(out.pos.mean(dim=0), torch.zeros(3), atol=1e-5)

def test_feature_choice(dummy_data):
    transform = FeatureChoice(feature=[0, 2, 4])
    out = transform(dummy_data)
    assert out.x.shape == (10, 3)

def test_gen_normalize(dummy_data):
    gen_norm = GenNormalize(transforms=[LocalPos()])
    out = gen_norm(dummy_data)
    assert isinstance(out, Data)
    
    gen_norm_mask = GenNormalize(transforms=[LocalPos()], mask_transform=MaskData(mask_ratio=0.5))
    ctx, tgt = gen_norm_mask(dummy_data)
    assert isinstance(ctx, Data)
    assert isinstance(tgt, Data)
