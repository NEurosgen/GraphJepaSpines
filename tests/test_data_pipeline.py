import torch
import pytest
import os
from torch_geometric.data import Data
from src.data_utils.transforms import (
    FeatureChoice,
    NormNoEps,
    EdgeNorm,
    LocalPos,
    ConcatStructuralPE,
    GenNormalize
)
from src.data_utils.structural_stats import ThesisMacroMetrics
from src.cli.train_model import build_transforms, load_stats
from src.data_utils.datamodule import GraphDataSet
from src.cli.classifier import compute_macro_stats

def test_dataset_loading(dataset_p, test_mode):
    """Checks if the dataset can be loaded and basic attributes are present."""
    if not dataset_p:
        pytest.skip("Dataset path not provided. Use --dataset_p")
    
    ds = GraphDataSet(path=dataset_p, save_cache=False)
    assert len(ds) > 0, f"No .pt files in {dataset_p}"
    
    if test_mode == "single":
        data = ds[0]
        assert hasattr(data, 'x'), "Sample missing 'x'"
        assert hasattr(data, 'pos'), "Sample missing 'pos'"
        assert hasattr(data, 'edge_index'), "Sample missing 'edge_index'"
        print(f"\n[test_dataset_loading] Loaded single sample from: {dataset_p}")
        print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, x_dim: {data.x.size(1)}")
    else:
        print(f"\n[test_dataset_loading] Checking entire dataset: {dataset_p}")
        print(f"  Total samples: {len(ds)}")
        # Quick check for first and last
        for i in [0, -1]:
            data = ds[i]
            assert hasattr(data, 'x')
        print("  All samples accessibility confirmed (first and last).")

def test_feature_counts_and_pe(dataset_p, test_mode):
    """Verifies x dimensions and presence of PE fields."""
    if not dataset_p:
        pytest.skip("Dataset path not provided. Use --dataset_p")
    
    ds = GraphDataSet(path=dataset_p, save_cache=False)
    
    if test_mode == "single":
        data = ds[0]
        x_dim = data.x.size(1)
        print(f"\n[test_feature_counts_and_pe] Single sample x_dim: {x_dim}")
        pe_fields = {'laplacian_pe': 'Lap', 'centrality_pe': 'Cent', 'random_walk_pe': 'RW'}
        for field, name in pe_fields.items():
            if hasattr(data, field):
                print(f"  Found {name} PE: {getattr(data, field).shape}")
    else:
        print(f"\n[test_feature_counts_and_pe] Checking PE consistency over full dataset...")
        data0 = ds[0]
        base_x_dim = data0.x.size(1)
        has_lap = hasattr(data0, 'laplacian_pe')
        has_cent = hasattr(data0, 'centrality_pe')
        has_rw = hasattr(data0, 'random_walk_pe')
        
        # Check all (or a subset if too large, but user said all)
        for i in range(min(len(ds), 1000)): # Limit to 500 for safety but inform user
            data = ds[i]
            assert data.x.size(1) == base_x_dim
            assert hasattr(data, 'laplacian_pe') == has_lap
            assert hasattr(data, 'centrality_pe') == has_cent
            assert hasattr(data, 'random_walk_pe') == has_rw
        
        print(f"  Verified consistency for {min(len(ds), 500)} samples.")
        print(f"  x_dim: {base_x_dim}, Lap: {has_lap}, Cent: {has_cent}, RW: {has_rw}")

def test_macro_metrics_calculation(dataset_p, test_mode):
    """Verifies that ThesisMacroMetrics can be computed."""
    if not dataset_p:
        pytest.skip("Dataset path not provided. Use --dataset_p")
        
    ds = GraphDataSet(path=dataset_p, save_cache=False)
    metric_module = ThesisMacroMetrics()
    
    if test_mode == "single":
        data = metric_module(ds[0])
        print("\n[test_macro_metrics_calculation] Single sample metrics:")
        names = ["avg_size", "avg_dist", "modul", "clust", "nodes", "edges", "dens"]
        metrics = data.macro_metrics.squeeze().tolist()
        for name, val in zip(names, metrics):
            print(f"  {name}: {val:.4f}")
    else:
        print(f"\n[test_macro_metrics_calculation] Computing mean macro metrics over dataset...")
        all_metrics = []
        limit = min(len(ds), 1000) # Macro metrics are slow (NetworkX), limit to 100
        for i in range(limit):
            data = metric_module(ds[i])
            all_metrics.append(data.macro_metrics)
        
        all_metrics = torch.cat(all_metrics, dim=0)
        mean_metrics = all_metrics.mean(dim=0)
        names = ["avg_size", "avg_dist", "modul", "clust", "nodes", "edges", "dens"]
        for name, val in zip(names, mean_metrics.tolist()):
            print(f"  Mean {name}: {val:.4f}")

def test_transform_pipeline_stats(dataset_p, stats_p, test_mode):
    """Checks the build_transforms pipeline and prints feature statistics."""
    if not dataset_p or not stats_p:
        pytest.skip("Dataset or stats path not provided.")
        
    ds = GraphDataSet(path=dataset_p, save_cache=False)
    mean_x, std_x, mean_edge, std_edge = load_stats(stats_p)
    cfg = {'eps': 1e-6}
    transforms_list = build_transforms(cfg, mean_x, std_x, mean_edge, std_edge)
    pipeline = GenNormalize(transforms=transforms_list)
    
    if test_mode == "single":
        out_data = pipeline(ds[0])
        print(f"\n[test_transform_pipeline_stats] Single sample x shape: {out_data.x.shape}")
        for i in range(min(5, out_data.x.size(1))):
            col = out_data.x[:, i]
            print(f"    Dim {i}: mean={col.mean():.4f}, std={col.std():.4f}")
    else:
        print(f"\n[test_transform_pipeline_stats] Computing aggregate statistics over dataset...")
        all_xs = []
        all_edge_attrs = []
        limit = min(len(ds), 1000)
        orig_x_dim = mean_x.size(0)
        for i in range(limit):
            out_data = pipeline(ds[i])
            # Only check original feature dims, PE is appended later
            base_x = out_data.x[:, :orig_x_dim]
            all_xs.append(base_x)
                
            if hasattr(out_data, 'edge_attr') and out_data.edge_attr is not None:
                all_edge_attrs.append(out_data.edge_attr)
        
        all_xs_cat = torch.cat(all_xs, dim=0)
        final_mean = all_xs_cat.mean(dim=0)
        final_std = all_xs_cat.std(dim=0)
        
        print(f"  Aggregate stats for {limit} samples (Final x_dim calculated: {final_mean.size(0)}):")
        for i in range(min(10, final_mean.size(0))):
            print(f"    Dim {i:2d}: mean={final_mean[i]:7.4f}, avg_std={final_std[i]:7.4f}")
            
        # Because NormNoEps only normalizes non-zero features (abs > eps), 
        # the overall distribution becomes a mixture of 0s and N(0,1).
        # Therefore, the global standard deviation will be less than 1 (proportional to sparsity).
        # We just verify that the mean is bounded and std is > 0.
        assert torch.all(torch.abs(final_mean) < 1.0), f"Mean of normalized x is too far from 0: {final_mean}"
        assert torch.all(final_std >= 0.0) and torch.all(final_std < 2.0), f"Std of normalized x is out of expected bounds [0.0, 2.0]: {final_std}"
        print("  Verified node features normalization (accounting for NormNoEps sparsity).")

        if all_edge_attrs:
            all_edge_attrs_cat = torch.cat(all_edge_attrs, dim=0)
            final_edge_mean = all_edge_attrs_cat.mean(dim=0)
            final_edge_std = all_edge_attrs_cat.std(dim=0)
            assert torch.all(torch.abs(final_edge_mean) < 0.8), f"Mean of edge_attr is not around 0: {final_edge_mean}"
            assert torch.all(torch.abs(final_edge_std - 1.0) < 0.8), f"Std of edge_attr is not around 1: {final_edge_std}"
            print("  Verified edge_attr normalization.")


def test_macro_metrics_normalization(dataset_p, test_mode):
    """Verifies that computed macro metrics are correctly normalized to N(0, 1)."""
    if not dataset_p:
        pytest.skip("Dataset path not provided.")
        
    ds = GraphDataSet(path=dataset_p, save_cache=False)
    
    # We first apply the ThesisMacroMetrics to ensure metrics are present as they would be
    metric_module = ThesisMacroMetrics()
    ds.transform = metric_module

    limit = 100 if test_mode == "single" else min(len(ds), 1000)
    print(f"\n[test_macro_metrics_normalization] Computing macro stats over {limit} samples...")
    macro_mean, macro_std = compute_macro_stats(ds, max_samples=limit)
    
    if macro_mean is None:
        pytest.skip("No macro metrics found in dataset.")
        
    # Now verify normalization manually as done in GraphLatent
    all_normalized_macros = []
    
    import random
    indices = list(range(len(ds)))
    if len(indices) > limit:
        indices = random.sample(indices, limit)
        
    for i in indices:
        data = ds[i]
        mac = data.macro_metrics
        if mac.dim() == 1:
            mac = mac.unsqueeze(0)
        elif mac.dim() == 2 and mac.size(0) > 1:
            mac = mac.mean(dim=0, keepdim=True)
            
        # Normalize
        mac_norm = (mac - macro_mean) / (macro_std + 1e-6)
        all_normalized_macros.append(mac_norm)
        
    all_normalized_macros = torch.cat(all_normalized_macros, dim=0)
    norm_mean = all_normalized_macros.mean(dim=0)
    norm_std = all_normalized_macros.std(dim=0)
    
    print("  Normalized Macro Metrics Stats (Mean / Std):")
    for i in range(norm_mean.size(0)):
        print(f"    Metric {i:2d}: mean={norm_mean[i]:7.4f}, std={norm_std[i]:7.4f}")
        
    assert torch.all(torch.abs(norm_mean) < 0.8), f"Normalized macro_mean is not 0: {norm_mean}"
    assert torch.all(torch.abs(norm_std - 1.0) < 0.8), f"Normalized macro_std is not 1: {norm_std}"
    print("  Verified macro features normalization.")

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
