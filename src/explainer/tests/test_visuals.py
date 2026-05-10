"""
Smoke tests for visualization utilities (visuals.py).

These tests verify that plotting functions don't crash and produce output files,
and that helper functions return correct values.
"""
import os
import torch
import pytest

from src.explainer.visuals import feature_name, set_neurips_style, plot_custom_graph
from src.explainer.tests.conftest import _make_star_graph, _make_house_graph, NODE_FEAT_DIM


# ── feature_name ──────────────────────────────────────────────────────────

def test_feature_name_local_known():
    """Known local feature names should be returned correctly."""
    assert feature_name(0, num_node_features=21) == 'head_area'
    assert feature_name(5, num_node_features=21) == 'head_volume'
    assert feature_name(20, num_node_features=21) == 'spine_volume'


def test_feature_name_local_unknown():
    """Node feature indices beyond the known names should get generic names."""
    assert feature_name(25, num_node_features=30) == 'Node Feature 25'


def test_feature_name_macro():
    """Indices beyond num_node_features should be labeled as macro features."""
    assert feature_name(21, num_node_features=21) == 'Macro Feature 0'
    assert feature_name(23, num_node_features=21) == 'Macro Feature 2'


def test_feature_name_zero_node_features():
    """When num_node_features=0, all features are macro."""
    assert feature_name(0, num_node_features=0) == 'Macro Feature 0'
    assert feature_name(3, num_node_features=0) == 'Macro Feature 3'


# ── set_neurips_style ────────────────────────────────────────────────────

def test_set_neurips_style_runs():
    """set_neurips_style should not raise."""
    set_neurips_style()  # Just verify no exception


# ── plot_custom_graph ────────────────────────────────────────────────────

def test_plot_custom_graph_no_masks(tmp_path):
    """plot_custom_graph should create a file without masks."""
    data = _make_star_graph(num_leaves=4, node_feat_dim=NODE_FEAT_DIM)
    save_path = str(tmp_path / "test_graph.png")

    plot_custom_graph(data, node_mask=None, edge_mask=None, save_path=save_path,
                      title="Test Star Graph")

    assert os.path.exists(save_path), f"Plot file was not created at {save_path}"
    assert os.path.getsize(save_path) > 0, "Plot file is empty"


def test_plot_custom_graph_with_masks(tmp_path):
    """plot_custom_graph should work with node_mask and edge_mask."""
    data = _make_house_graph(node_feat_dim=NODE_FEAT_DIM)
    save_path = str(tmp_path / "test_explained.png")

    node_mask = torch.rand(data.num_nodes, NODE_FEAT_DIM)
    edge_mask = torch.rand(data.num_edges)

    plot_custom_graph(data, node_mask=node_mask, edge_mask=edge_mask,
                      save_path=save_path, title="Test Explanation")

    assert os.path.exists(save_path), f"Plot file was not created at {save_path}"
    assert os.path.getsize(save_path) > 0, "Plot file is empty"


def test_plot_custom_graph_no_pos(tmp_path):
    """plot_custom_graph should fall back to spring layout when pos=None."""
    data = _make_star_graph(num_leaves=3, node_feat_dim=NODE_FEAT_DIM)
    data.pos = None  # Remove positions
    save_path = str(tmp_path / "test_no_pos.png")

    plot_custom_graph(data, save_path=save_path, title="No positions")

    assert os.path.exists(save_path)
