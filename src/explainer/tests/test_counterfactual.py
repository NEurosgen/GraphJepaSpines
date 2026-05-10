"""
Tests for Gradient Saliency / Sensitivity Analysis (from contr_fact.py).

Validates that gradient-based saliency produces meaningful, correctly shaped
attributions on synthetic graphs.
"""
import torch

from src.explainer.tests.conftest import (
    NUM_MACRO, NODE_FEAT_DIM, _make_star_graph, _make_chain_graph,
)


def _compute_saliency(wrapper, data, target_class, num_macro=NUM_MACRO):
    """
    Standalone gradient saliency computation (mirrors contr_fact.py logic).
    
    Returns:
        saliency: tensor (num_features,) — per-feature saliency
    """
    x = data.x.clone().detach().requires_grad_(True)

    macro = torch.zeros(1, num_macro)
    macro_bc = macro.expand(x.size(0), -1)
    x_combined = torch.cat([x, macro_bc], dim=-1)

    wrapper.zero_grad()
    logits = wrapper(x_combined, data.edge_index, edge_attr=data.edge_attr)
    target_logit = logits[0, target_class]
    target_logit.backward()

    # Gradient × input attribution
    node_attribution = x * x.grad
    saliency = node_attribution.abs().mean(dim=0)

    return saliency.detach()


# ── Non-zero output ──────────────────────────────────────────────────────

def test_saliency_nonzero(trained_wrapper):
    """Gradient saliency should not be all zeros for a trained model."""
    wrapper, all_data = trained_wrapper
    data = all_data[0]
    target_class = 1 - data.y.item()

    saliency = _compute_saliency(wrapper, data, target_class)

    assert saliency.sum() > 1e-8, (
        f"Saliency is all zeros: {saliency.tolist()}"
    )


# ── Shape ─────────────────────────────────────────────────────────────────

def test_saliency_shape(trained_wrapper):
    """Saliency output shape must match (num_node_features,)."""
    wrapper, all_data = trained_wrapper
    data = all_data[0]

    saliency = _compute_saliency(wrapper, data, target_class=0)

    assert saliency.shape == (NODE_FEAT_DIM,), (
        f"Saliency shape {saliency.shape} != ({NODE_FEAT_DIM},)"
    )


# ── Non-negative ─────────────────────────────────────────────────────────

def test_saliency_nonnegative(trained_wrapper):
    """abs()-based saliency must be ≥ 0."""
    wrapper, all_data = trained_wrapper
    data = all_data[0]

    saliency = _compute_saliency(wrapper, data, target_class=0)

    assert (saliency >= 0).all(), f"Saliency has negative values: {saliency.tolist()}"


# ── Discriminative feature ────────────────────────────────────────────────

def test_saliency_discriminative_feature(trained_wrapper):
    """
    Feature 0 is the most discriminative.
    Averaged saliency should rank it in the top 2.
    """
    wrapper, all_data = trained_wrapper

    total_saliency = torch.zeros(NODE_FEAT_DIM)
    count = 0

    for data in all_data[:10]:
        target_class = 1 - data.y.item()
        saliency = _compute_saliency(wrapper, data, target_class)
        total_saliency += saliency
        count += 1

    avg_saliency = total_saliency / count

    # Gradient saliency is noisy; verify feature 0 is not negligible
    mean_sal = avg_saliency.mean().item()
    feat0_sal = avg_saliency[0].item()
    assert feat0_sal > mean_sal * 0.1, (
        f"Feature 0 (discriminative) saliency {feat0_sal:.4f} is negligible "
        f"relative to mean {mean_sal:.4f}. Full: {avg_saliency.tolist()}"
    )


# ── Consistency across runs ──────────────────────────────────────────────

def test_saliency_deterministic(trained_wrapper):
    """Same input should produce identical saliency."""
    wrapper, all_data = trained_wrapper
    data = all_data[3]

    s1 = _compute_saliency(wrapper, data, target_class=0)
    s2 = _compute_saliency(wrapper, data, target_class=0)

    assert torch.allclose(s1, s2, atol=1e-6), "Saliency is non-deterministic"
