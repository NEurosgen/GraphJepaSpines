"""
Shared fixtures for explainer tests.

Provides synthetic graphs with deterministic, obvious explainability patterns
and a small trained model that can actually distinguish between graph classes.
"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_add_pool

from src.models.encoder import GraphGcnEncoder
from src.explainer.models import GraphExplainerWrapper


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def _make_star_graph(num_leaves: int = 5, node_feat_dim: int = 4, label: int = 0) -> Data:
    """
    Star graph: hub node 0 connected to all leaf nodes.
    Node features: feature 0 is near -2.0 (class 0 marker),
    other features are small random noise.
    """
    hub = 0
    src = [hub] * num_leaves + list(range(1, num_leaves + 1))
    dst = list(range(1, num_leaves + 1)) + [hub] * num_leaves
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    num_nodes = num_leaves + 1
    x = torch.randn(num_nodes, node_feat_dim) * 0.01
    # Class 0 discriminator: feature 0 is strongly negative
    x[:, 0] = -2.0 + torch.randn(num_nodes) * 0.1

    edge_attr = torch.ones(edge_index.size(1), 1)
    pos = torch.randn(num_nodes, 3)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=torch.tensor(label, dtype=torch.long),
    )


def _make_chain_graph(length: int = 6, node_feat_dim: int = 4, label: int = 1) -> Data:
    """
    Chain (path) graph: 0 — 1 — 2 — ... — (length-1).
    Node features: feature 0 is near +2.0 (class 1 marker),
    other features are small random noise.
    """
    src = list(range(length - 1)) + list(range(1, length))
    dst = list(range(1, length)) + list(range(length - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    x = torch.randn(length, node_feat_dim) * 0.01
    # Class 1 discriminator: feature 0 is strongly positive
    x[:, 0] = 2.0 + torch.randn(length) * 0.1

    edge_attr = torch.ones(edge_index.size(1), 1)
    pos = torch.randn(length, 3)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=torch.tensor(label, dtype=torch.long),
    )


def _make_house_graph(node_feat_dim: int = 4, label: int = 1) -> Data:
    """
    House graph: square base (0-1-2-3) + triangle roof (2-3-4).
    The triangle sub-structure is the 'important' motif.
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # square base
        (2, 4), (3, 4),                    # triangle roof
    ]
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    num_nodes = 5
    x = torch.randn(num_nodes, node_feat_dim)
    edge_attr = torch.ones(edge_index.size(1), 1)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.5, 0.0],
    ])

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        pos=pos, y=torch.tensor(label, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NODE_FEAT_DIM = 4
EMBED_DIM = 8
NUM_MACRO = 3  # small number for tests
NUM_CLASSES = 2


@pytest.fixture
def node_feat_dim():
    return NODE_FEAT_DIM


@pytest.fixture
def embed_dim():
    return EMBED_DIM


@pytest.fixture
def num_macro():
    return NUM_MACRO


@pytest.fixture
def simple_encoder():
    """Tiny GCN encoder for testing."""
    return GraphGcnEncoder(in_channels=NODE_FEAT_DIM, out_channels=EMBED_DIM, num_layers=1)


@pytest.fixture
def simple_classifier():
    """Linear classifier: (embed_dim + num_macro) → num_classes."""
    return nn.Linear(EMBED_DIM + NUM_MACRO, NUM_CLASSES)


@pytest.fixture
def star_graph():
    return _make_star_graph(num_leaves=5, node_feat_dim=NODE_FEAT_DIM, label=0)


@pytest.fixture
def chain_graph():
    return _make_chain_graph(length=6, node_feat_dim=NODE_FEAT_DIM, label=1)


@pytest.fixture
def house_graph():
    return _make_house_graph(node_feat_dim=NODE_FEAT_DIM, label=1)


@pytest.fixture
def two_class_graphs():
    """
    Returns (class_0_graphs, class_1_graphs) — 20 samples per class.
    Class 0 = star topology + feature 0 is -2.0.
    Class 1 = chain topology + feature 0 is +2.0.
    """
    torch.manual_seed(42)
    class_0 = [_make_star_graph(num_leaves=5, node_feat_dim=NODE_FEAT_DIM, label=0) for _ in range(20)]
    class_1 = [_make_chain_graph(length=6, node_feat_dim=NODE_FEAT_DIM, label=1) for _ in range(20)]
    return class_0, class_1


@pytest.fixture
def wrapper(simple_encoder, simple_classifier):
    """Untrained GraphExplainerWrapper for shape / smoke tests."""
    return GraphExplainerWrapper(
        jepa_model=simple_encoder,
        classifier=simple_classifier,
        num_node_features=NODE_FEAT_DIM,
        sigma=1.0,
    )


@pytest.fixture(scope="session")
def trained_wrapper():
    """
    GraphExplainerWrapper trained ~300 steps on synthetic data so it
    actually distinguishes the two classes.  Returns (wrapper, all_data).

    Session-scoped so we only train once across all tests.
    """
    torch.manual_seed(0)

    # Generate data
    class_0 = [_make_star_graph(num_leaves=5, node_feat_dim=NODE_FEAT_DIM, label=0) for _ in range(20)]
    class_1 = [_make_chain_graph(length=6, node_feat_dim=NODE_FEAT_DIM, label=1) for _ in range(20)]
    all_data = class_0 + class_1

    encoder = GraphGcnEncoder(in_channels=NODE_FEAT_DIM, out_channels=EMBED_DIM, num_layers=2)
    classifier = nn.Linear(EMBED_DIM + NUM_MACRO, NUM_CLASSES)

    # Unfreeze for training
    for p in encoder.parameters():
        p.requires_grad = True
    for p in classifier.parameters():
        p.requires_grad = True

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(300):
        epoch_loss = 0.0
        for data in all_data:
            optimizer.zero_grad()

            edge_attr_exp = torch.exp(-data.edge_attr ** 2)
            node_emb = encoder(data.x, data.edge_index, edge_attr_exp.squeeze(-1))
            batch_idx = torch.zeros(data.x.size(0), dtype=torch.long)
            graph_emb = global_add_pool(node_emb, batch_idx)

            # Macro features: deterministic per class (class 0 → 0.0, class 1 → 1.0)
            macro = torch.full((1, NUM_MACRO), float(data.y.item()))
            combined = torch.cat([graph_emb, macro], dim=-1)
            logits = classifier(combined)
            loss = loss_fn(logits, data.y.unsqueeze(0))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Verify the model actually learned
    correct = 0
    with torch.no_grad():
        for data in all_data:
            edge_attr_exp = torch.exp(-data.edge_attr ** 2)
            node_emb = encoder(data.x, data.edge_index, edge_attr_exp.squeeze(-1))
            batch_idx = torch.zeros(data.x.size(0), dtype=torch.long)
            graph_emb = global_add_pool(node_emb, batch_idx)
            macro = torch.full((1, NUM_MACRO), float(data.y.item()))
            combined = torch.cat([graph_emb, macro], dim=-1)
            pred = classifier(combined).argmax(dim=-1).item()
            if pred == data.y.item():
                correct += 1

    accuracy = correct / len(all_data)
    assert accuracy >= 0.8, f"Trained model accuracy is only {accuracy:.1%}, expected ≥80%"

    # Now wrap with frozen params
    w = GraphExplainerWrapper(
        jepa_model=encoder,
        classifier=classifier,
        num_node_features=NODE_FEAT_DIM,
        sigma=1.0,
    )
    return w, all_data
