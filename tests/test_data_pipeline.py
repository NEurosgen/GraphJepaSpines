import torch
import pytest
from src.data_utils.transforms import GenNormalize, build_canonical_transform
from src.data_utils.structural_stats import ThesisMacroMetrics
from src.data_utils.stats import load_feature_stats, compute_macro_stats
from src.data_utils.datamodule import GraphDataSet


def test_dataset_loading(dataset_p, test_mode):
    """Checks if the dataset can be loaded and basic attributes are present."""
    if not dataset_p:
        pytest.skip("Dataset path not provided. Use --dataset_p")

    ds = GraphDataSet(path=dataset_p, save_cache=False)
    assert len(ds) > 0, f"No .pt files in {dataset_p}"

    # x и pos — общий знаменатель всех датасетов; рёбра могут отсутствовать
    # (например h01) и строятся канонизацией из pos.
    if test_mode == "single":
        data = ds[0]
        assert hasattr(data, "x"), "Sample missing 'x'"
        assert hasattr(data, "pos"), "Sample missing 'pos'"
        print(f"\n[test_dataset_loading] Loaded single sample from: {dataset_p}")
        print(f"  Nodes: {data.num_nodes}, x_dim: {data.x.size(1)}")
    else:
        print(f"\n[test_dataset_loading] Checking entire dataset: {dataset_p}")
        print(f"  Total samples: {len(ds)}")
        for i in [0, -1]:
            data = ds[i]
            assert hasattr(data, "x") and hasattr(data, "pos")
        print("  All samples accessibility confirmed (first and last).")


def test_feature_dim_consistency(dataset_p, test_mode):
    """Verifies node-feature dimensionality is consistent across the dataset."""
    if not dataset_p:
        pytest.skip("Dataset path not provided. Use --dataset_p")

    ds = GraphDataSet(path=dataset_p, save_cache=False)
    base_x_dim = ds[0].x.size(1)

    if test_mode == "single":
        print(f"\n[test_feature_dim_consistency] Single sample x_dim: {base_x_dim}")
    else:
        for i in range(min(len(ds), 1000)):
            assert ds[i].x.size(1) == base_x_dim
        print(f"\n[test_feature_dim_consistency] x_dim={base_x_dim} consistent "
              f"over {min(len(ds), 1000)} samples.")


def test_macro_metrics_calculation(dataset_p, test_mode):
    """Verifies that ThesisMacroMetrics can be computed (optional, behind use_macro flag)."""
    if not dataset_p:
        pytest.skip("Dataset path not provided. Use --dataset_p")

    ds = GraphDataSet(path=dataset_p, save_cache=False)
    metric_module = ThesisMacroMetrics()
    names = ["avg_size", "avg_dist", "modul", "clust", "nodes", "edges", "dens"]

    if test_mode == "single":
        data = metric_module(ds[0])
        print("\n[test_macro_metrics_calculation] Single sample metrics:")
        for name, val in zip(names, data.macro_metrics.squeeze().tolist()):
            print(f"  {name}: {val:.4f}")
    else:
        limit = min(len(ds), 1000)
        all_metrics = [metric_module(ds[i]).macro_metrics for i in range(limit)]
        mean_metrics = torch.cat(all_metrics, dim=0).mean(dim=0)
        print("\n[test_macro_metrics_calculation] Mean macro metrics:")
        for name, val in zip(names, mean_metrics.tolist()):
            print(f"  Mean {name}: {val:.4f}")


def test_canonical_transform_stats(dataset_p, stats_p, test_mode):
    """Checks the build_canonical_transform pipeline and prints feature statistics."""
    if not dataset_p or not stats_p:
        pytest.skip("Dataset or stats path not provided.")

    ds = GraphDataSet(path=dataset_p, save_cache=False)
    mean_x, std_x = load_feature_stats(stats_p)
    cfg = {"eps": 1e-6, "r": 1.5, "use_macro": False}
    pipeline = GenNormalize(transforms=build_canonical_transform(cfg, mean_x, std_x))

    if test_mode == "single":
        out_data = pipeline(ds[0])
        print(f"\n[test_canonical_transform_stats] Single sample x shape: {out_data.x.shape}")
        for i in range(min(5, out_data.x.size(1))):
            col = out_data.x[:, i]
            print(f"    Dim {i}: mean={col.mean():.4f}, std={col.std():.4f}")
        # после канонизации граф всегда имеет рёбра (из pos) и дистанции [E, 1]
        assert out_data.edge_index is not None and out_data.edge_index.numel() > 0
        assert out_data.edge_attr.size(1) == 1
    else:
        all_xs = []
        limit = min(len(ds), 1000)
        orig_x_dim = mean_x.size(0)
        for i in range(limit):
            out_data = pipeline(ds[i])
            all_xs.append(out_data.x[:, :orig_x_dim])

        all_xs_cat = torch.cat(all_xs, dim=0)
        final_mean = all_xs_cat.mean(dim=0)
        final_std = all_xs_cat.std(dim=0)

        print(f"\n[test_canonical_transform_stats] Aggregate stats over {limit} samples:")
        for i in range(min(10, final_mean.size(0))):
            print(f"    Dim {i:2d}: mean={final_mean[i]:7.4f}, std={final_std[i]:7.4f}")

        # NormNoEps нормализует только ненулевые признаки, поэтому глобальный std < 1
        # (смесь нулей и N(0,1)). Проверяем ограниченность среднего и std.
        assert torch.all(torch.abs(final_mean) < 1.0), f"mean too far from 0: {final_mean}"
        assert torch.all((final_std >= 0.0) & (final_std < 2.0)), f"std out of bounds: {final_std}"
        print("  Verified node-feature normalization (accounting for NormNoEps sparsity).")


def test_macro_metrics_normalization(dataset_p, test_mode):
    """Verifies that computed macro metrics are correctly normalized to ~N(0, 1)."""
    if not dataset_p:
        pytest.skip("Dataset path not provided.")

    ds = GraphDataSet(path=dataset_p, save_cache=False)
    ds.transform = ThesisMacroMetrics()

    limit = 100 if test_mode == "single" else min(len(ds), 1000)
    print(f"\n[test_macro_metrics_normalization] Computing macro stats over {limit} samples...")
    macro_mean, macro_std = compute_macro_stats(ds, max_samples=limit)
    if macro_mean is None:
        pytest.skip("No macro metrics found in dataset.")

    import random
    indices = list(range(len(ds)))
    if len(indices) > limit:
        indices = random.sample(indices, limit)

    all_normalized = []
    for i in indices:
        mac = ds[i].macro_metrics
        if mac.dim() == 1:
            mac = mac.unsqueeze(0)
        elif mac.dim() == 2 and mac.size(0) > 1:
            mac = mac.mean(dim=0, keepdim=True)
        all_normalized.append((mac - macro_mean) / (macro_std + 1e-6))

    all_normalized = torch.cat(all_normalized, dim=0)
    norm_mean = all_normalized.mean(dim=0)
    norm_std = all_normalized.std(dim=0)

    print("  Normalized Macro Metrics Stats (Mean / Std):")
    for i in range(norm_mean.size(0)):
        print(f"    Metric {i:2d}: mean={norm_mean[i]:7.4f}, std={norm_std[i]:7.4f}")

    assert torch.all(torch.abs(norm_mean) < 0.8), f"normalized macro_mean not 0: {norm_mean}"
    assert torch.all(torch.abs(norm_std - 1.0) < 0.8), f"normalized macro_std not 1: {norm_std}"
    print("  Verified macro features normalization.")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
