"""
Pipeline: extract embeddings from a pre-trained encoder and a prepared dataset,
then train a classifier using Stratified K-Fold Cross Validation.
"""

import gc
from pathlib import Path
from src.data_utils.stats import compute_macro_stats, load_feature_stats
import hydra
import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from torch_geometric.nn import global_add_pool
from src.cli.embedding_pipeline import EmbeddingExtractor, EmbeddingSet, train_cv, summarize_cv
from src.data_utils.datamodule import GraphDataSet
from src.data_utils.transforms import GenNormalize, build_canonical_transform
from src.models.encoder import GraphLatent
from src.models.loader_model import load_encoder_from_folder

torch.set_float32_matmul_precision("high")


def get_class_9009(file_path) -> torch.Tensor:
    mapping = {"ab": 0, "wt": 1}
    folder_name = Path(file_path).parent.name.lower()
    if folder_name not in mapping:
        raise ValueError(
            f"Folder '{folder_name}' not in mapping {mapping}. File: {file_path}"
        )
    return torch.tensor(mapping[folder_name], dtype=torch.long)


def extract_all_embeddings(cfg: DictConfig, device: torch.device):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule

    print(f"Loading encoder from: {cls_cfg.checkpoint_path}")
    encoder = load_encoder_from_folder(cls_cfg.checkpoint_path)
    


    mean_x, std_x = load_feature_stats(cls_cfg.stats_path)
    gen_normalize = GenNormalize(
        build_canonical_transform(dm_cfg, mean_x, std_x),
        mask_transform=None,
    )

    print(f"Loading dataset from: {cls_cfg.raw_path}")
    ds = GraphDataSet(path=cls_cfg.raw_path, get_class=get_class_9009, transform=gen_normalize)
    macro_mean, macro_std = compute_macro_stats(ds)
    encoder_graph = GraphLatent(
        encoder=encoder,
        macro_mean=macro_mean,
        macro_std=macro_std,
        pooling=global_add_pool,
    ).to(device)
    extractor = EmbeddingExtractor(encoder=encoder_graph, device=device)
    emb_set: EmbeddingSet = extractor.extract_from_graph_dataset(
        dataset=ds,
        batch_size=cls_cfg.get("batch_size", 128),
        num_workers=dm_cfg.get("num_workers", 4),
        desc="Extracting All",
    )

    x_all, y_all = emb_set.embeddings, emb_set.labels

    del encoder, extractor, emb_set
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return x_all, y_all





@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_splits = cfg.get("n_splits", 5)

    print("=" * 60)
    print(" Cross-Validation Evaluation Pipeline (9009)")
    print(f" Encoder : {cfg.classifier.checkpoint_path}")
    print(f" Dataset : {cfg.classifier.path}")
    print(f" Folds   : {n_splits}")
    print("=" * 60)

    print("\n[1/2] Extracting Embeddings...")
    x_all, y_all = extract_all_embeddings(cfg, device)
    print(f"Features: {x_all.shape}, Labels: {y_all.shape}")

    print(f"\n[2/2] Training & Evaluating with {n_splits}-Fold CV...")
    save_path = cfg.classifier.get("save_classifier_path", None)
    fold_metrics = train_cv(cfg, x_all, y_all, class_names=["ab", "wt"], save_path=save_path)

    if not fold_metrics:
        print("No metrics collected!")
        return

    summarize_cv(fold_metrics)


if __name__ == "__main__":
    main()
