"""
Pipeline: extract embeddings from a pre-trained encoder and a prepared dataset,
then train a classifier using Stratified K-Fold Cross Validation.
"""

import gc
import torch
import hydra
import pytorch_lightning as L
from omegaconf import DictConfig

from torch_geometric.nn import global_add_pool
from src.cli.embedding_pipeline import EmbeddingExtractor, EmbeddingSet, train_cv, summarize_cv
from src.data_utils.datamodule import GraphDataSet
from src.data_utils.stats import compute_macro_stats, load_feature_stats
from src.data_utils.transforms import GenNormalize, build_canonical_transform
from src.models.encoder import GraphLatent
from src.models.loader_model import load_encoder_from_folder
from src.cli.inference.minnie65.minnie65_get_class import make_minnie65_class_getter
torch.set_float32_matmul_precision("high")







def extract_all_embeddings(cfg: DictConfig, encoder_folder: str, dataset_path: str, device: torch.device):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule # Ошибка важная 

    print(f"Loading encoder from: {encoder_folder}")
    encoder = load_encoder_from_folder(encoder_folder)
    encoder.eval().requires_grad_(False).to(device)

    mean_x, std_x = load_feature_stats(cls_cfg.stats_path)
    gen_normalize = GenNormalize(
        build_canonical_transform(dm_cfg, mean_x, std_x),
        mask_transform=None,
    )

    get_class_fn = make_minnie65_class_getter(dm_cfg.dataset.class_path)

    print(f"Loading dataset from: {dataset_path}")
    ds = GraphDataSet(path=dataset_path, get_class=get_class_fn, transform=gen_normalize)

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
        batch_size=cls_cfg["batch_size"],
        num_workers=dm_cfg.get("num_workers", 4),
        desc="Extracting All"
    )

    pooling_level = cls_cfg["pooling_level"]
    if pooling_level == "neuron":
        pooling_type = cls_cfg["pooling_type"]
        print(f"Pooling level: {pooling_level}, type: {pooling_type}")
        pooled_set = emb_set.pool_by_segment(pooling_type=pooling_type)
        print(f"  Группировка по нейрону: {emb_set.embeddings.shape[0]} веток -> "
              f"{pooled_set.embeddings.shape[0]} нейронов")
    else:
        pooled_set = emb_set
        print(f"  Уровень '{pooling_level}': без пулинга по нейрону, "
              f"{pooled_set.embeddings.shape[0]} объектов (сплит будет по веткам!)")


    x_pooled, y_pooled = pooled_set.embeddings, pooled_set.labels

    del encoder, encoder_graph, extractor, emb_set
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return x_pooled, y_pooled





@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_path = cfg.classifier.checkpoint_path
    dataset_path = cfg.classifier.raw_path
    n_splits     = cfg.classifier.get("n_splits", 3)

    print("=" * 60)
    print(f" Cross-Validation Evaluation Pipeline")
    print(f" Encoder : {encoder_path}")
    print(f" Dataset : {dataset_path}")
    print(f" Folds   : {n_splits}")
    print("=" * 60)

    print("\n[1/2] Extracting Embeddings...")
    x_pooled, y_pooled = extract_all_embeddings(cfg, encoder_path, dataset_path, device)
    print(f"Features: {x_pooled.shape}, Labels: {y_pooled.shape}")

    print(f"\n[2/2] Training & Evaluating with {n_splits}-Fold CV...")
    fold_metrics = train_cv(cfg, x_pooled, y_pooled)

    if not fold_metrics:
        print("No metrics collected!")
        return

    summarize_cv(fold_metrics)


if __name__ == "__main__":
    main()