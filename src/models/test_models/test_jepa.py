import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from src.models.jepa import CrossAttentionPredictor, GraphJepa, sigreg
from src.models.encoder import GraphGinEncoder

@pytest.fixture
def dummy_jepa_data():
    class DummyContextTarget:
        def __init__(self, num_nodes):
            self.x = torch.randn(num_nodes, 5)
            self.pos = torch.randn(num_nodes, 3)
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_attr = torch.empty((0, 2))
            
    context = DummyContextTarget(10)
    target = DummyContextTarget(5)
    return context, target

def test_cross_attention_predictor():
    hidden_dim = 16
    pos_dim = 3
    predictor = CrossAttentionPredictor(hidden_dim=hidden_dim, pos_dim=pos_dim)
    
    context_emb = torch.randn(10, hidden_dim)
    context_pos = torch.randn(10, pos_dim)
    target_pos = torch.randn(5, pos_dim)
    
    pred = predictor(context_emb=context_emb, context_pos=context_pos, target_pos=target_pos)
    
    # Target positions predict target embeddings
    assert pred.shape == (5, hidden_dim)

def test_graph_jepa(dummy_jepa_data):
    context, target = dummy_jepa_data
    
    in_channels = 5
    hidden_dim = 16
    
    encoder = GraphGinEncoder(in_channels=in_channels, out_channels=hidden_dim, num_layers=1)
    predictor = CrossAttentionPredictor(hidden_dim=hidden_dim, pos_dim=3)
    
    jepa = GraphJepa(encoder=encoder, predictor=predictor, ema=0.99)
    
    # 1. Test EMA update
    initial_teacher_weight = jepa.teach_encoder.layers[0].norm.gamma.clone()
    
    # Change student weight artificially
    with torch.no_grad():
        jepa.student_encoder.layers[0].norm.gamma += 1.0
        
    jepa._ema()
    
    # Teacher weight should be updated
    updated_teacher_weight = jepa.teach_encoder.layers[0].norm.gamma
    assert not torch.allclose(initial_teacher_weight, updated_teacher_weight)
    
    # 2. Test forward pass loss calculation
    loss = jepa(context, target)
    assert loss.dim() == 0 # Scalar loss
    assert not torch.isnan(loss)
    
    loss.backward()
    assert jepa.student_encoder.layers[0].norm.gamma.grad is not None
    assert jepa.teach_encoder.layers[0].norm.gamma.grad is None # Teacher params require_grad=False

def test_sigreg():
    x = torch.randn(10, 16)
    reg_loss = sigreg(x, num_slices=64)
    assert reg_loss.dim() == 1
    assert reg_loss.size(0) == 64
    assert not torch.isnan(reg_loss).any()
