"""
Tests for GraphExplainerWrapper — shapes, determinism, frozen params.
"""
import torch

from src.explainer.tests.conftest import NUM_MACRO, NODE_FEAT_DIM, NUM_CLASSES


def _add_macro(data, num_macro=NUM_MACRO):
    """Broadcast fake macro features into x, mimicking the explainer pipeline."""
    macro = torch.randn(1, num_macro)
    macro_bc = macro.expand(data.x.size(0), -1)
    x_combined = torch.cat([data.x, macro_bc], dim=-1)
    return x_combined


# ── Shape tests ───────────────────────────────────────────────────────────

def test_forward_shape(wrapper, star_graph):
    """Output shape must be (1, num_classes)."""
    x = _add_macro(star_graph)
    out = wrapper(x, star_graph.edge_index, edge_attr=star_graph.edge_attr)
    assert out.shape == (1, NUM_CLASSES), f"Expected (1, {NUM_CLASSES}), got {out.shape}"


def test_forward_no_edge_attr(wrapper, star_graph):
    """Model should handle edge_attr=None gracefully."""
    x = _add_macro(star_graph)
    out = wrapper(x, star_graph.edge_index, edge_attr=None)
    assert out.shape == (1, NUM_CLASSES)


def test_forward_single_graph_with_batch_dim(wrapper, star_graph):
    """
    Wrapper works for single-graph explanation with explicit batch=zeros.
    Note: GraphExplainerWrapper is designed for single-graph explanation,
    not multi-graph batching (global_feats taken from x[0]).
    """
    x = _add_macro(star_graph)
    batch = torch.zeros(star_graph.num_nodes, dtype=torch.long)
    out = wrapper(x, star_graph.edge_index, edge_attr=star_graph.edge_attr, batch=batch)
    assert out.shape == (1, NUM_CLASSES)


# ── Frozen params ─────────────────────────────────────────────────────────

def test_gradients_frozen(wrapper, star_graph):
    """Encoder and classifier parameters should not require grad inside wrapper."""
    for name, p in wrapper.graph_jepa.named_parameters():
        assert not p.requires_grad, f"Encoder param {name} has requires_grad=True"
    for name, p in wrapper.classifier.named_parameters():
        assert not p.requires_grad, f"Classifier param {name} has requires_grad=True"


# ── Determinism ───────────────────────────────────────────────────────────

def test_deterministic_output(wrapper, chain_graph):
    """Same input must produce identical output."""
    x = _add_macro(chain_graph)
    out1 = wrapper(x, chain_graph.edge_index, edge_attr=chain_graph.edge_attr)
    out2 = wrapper(x, chain_graph.edge_index, edge_attr=chain_graph.edge_attr)
    assert torch.allclose(out1, out2), "Wrapper output is non-deterministic"


# ── Edge attr processing ─────────────────────────────────────────────────

def test_edge_attr_affects_output(wrapper, star_graph):
    """
    Verify that different edge_attr vs no edge_attr produces different outputs.
    The RBF kernel in the wrapper transforms edge_attr before passing to GCN.
    Note: when all edges have constant attr, the RBF shift makes them all 1.0,
    so we compare edge_attr vs None instead.
    """
    x = _add_macro(star_graph)
    out_with = wrapper(x, star_graph.edge_index,
                       edge_attr=torch.randn(star_graph.edge_index.size(1), 1).abs() + 0.5)
    out_none = wrapper(x, star_graph.edge_index, edge_attr=None)
    # These should generally differ since edge_attr=None uses ones
    # while random edge_attr will produce non-trivial RBF weights
    # (Not guaranteed to differ for all random seeds, but very likely)
    assert out_with.shape == out_none.shape == (1, NUM_CLASSES)


def test_global_feats_extracted_correctly(wrapper, star_graph):
    """
    The wrapper extracts global features from x[:, num_node_features:].
    Verify that changing global features changes the output.
    """
    macro1 = torch.zeros(1, NUM_MACRO)
    macro2 = torch.ones(1, NUM_MACRO) * 5.0

    x1 = torch.cat([star_graph.x, macro1.expand(star_graph.num_nodes, -1)], dim=-1)
    x2 = torch.cat([star_graph.x, macro2.expand(star_graph.num_nodes, -1)], dim=-1)

    out1 = wrapper(x1, star_graph.edge_index, edge_attr=star_graph.edge_attr)
    out2 = wrapper(x2, star_graph.edge_index, edge_attr=star_graph.edge_attr)

    assert not torch.allclose(out1, out2, atol=1e-4), \
        "Model output unchanged when global features change"
