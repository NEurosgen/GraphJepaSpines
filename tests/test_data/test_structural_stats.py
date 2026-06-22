import pytest
import torch
from torch_geometric.data import Data
from src.data_utils.structural_stats import ThesisMacroMetrics


@pytest.fixture
def chain_graph():
    """Цепочка 0-1-2-3-4 (5 узлов, 4 неориентированных ребра)."""
    x = torch.ones(5, 3)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4],
         [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )
    return Data(x=x, edge_index=edge_index, num_nodes=5)


@pytest.fixture
def disconnected_graph():
    """5 изолированных узлов без рёбер."""
    x = torch.ones(5, 3)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=5)


def test_macro_metrics_shape_and_counts(chain_graph):
    out = ThesisMacroMetrics()(chain_graph)
    assert hasattr(out, "macro_metrics")
    m = out.macro_metrics
    assert m.shape == (1, 7)

    metrics = m.squeeze()
    # 5. num_nodes = 5
    assert torch.allclose(metrics[4], torch.tensor(5.0))
    # 6. num_edges = 4 (8 направленных / 2)
    assert torch.allclose(metrics[5], torch.tensor(4.0))


def test_macro_metrics_disconnected(disconnected_graph):
    out = ThesisMacroMetrics()(disconnected_graph)
    metrics = out.macro_metrics.squeeze()
    assert metrics.shape == (7,)
    # 5. num_nodes = 5
    assert torch.allclose(metrics[4], torch.tensor(5.0))
    # 6. num_edges = 0
    assert torch.allclose(metrics[5], torch.tensor(0.0))
    # 4. clustering = 0 и 7. density = 0 без рёбер
    assert torch.allclose(metrics[3], torch.tensor(0.0))
    assert torch.allclose(metrics[6], torch.tensor(0.0))
