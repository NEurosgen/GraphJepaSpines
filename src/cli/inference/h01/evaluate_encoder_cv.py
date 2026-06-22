"""
Linear-probe оценка энкодера на h01 (человек), neuron-level.

Энкодер берётся из cfg.classifier.checkpoint_path. Датасет h01 канонизируется на
лету (NormNoEps со СВОИМИ статистиками h01 + построение графа из pos), эмбеддинги
пулятся по нейрону (segment_id), затем repeated Stratified K-fold + 95% CI.

Задача выбирается через cfg.transfer.task:
  celltype -> бинарно exc/inh (по умолчанию)
  layer    -> слой коры (7 классов)
num_classes выставляется автоматически под задачу.

Запуск:
  python -m src.cli.inference.h01.evaluate_encoder_cv                 # celltype
  python -m src.cli.inference.h01.evaluate_encoder_cv transfer.task=layer
"""
import gc

import hydra
import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from torch_geometric.nn import global_add_pool

from src.cli.embedding_pipeline import EmbeddingExtractor, EmbeddingSet, train_cv, summarize_cv
from src.data_utils.datamodule import GraphDataSet
from src.data_utils.stats import compute_macro_stats, load_feature_stats
from src.data_utils.transforms import GenNormalize, build_canonical_transform
from src.models.encoder import GraphLatent
from src.models.loader_model import load_encoder_from_folder
from src.cli.inference.h01.h01_get_class import make_h01_celltype_getter, make_h01_layer_getter

torch.set_float32_matmul_precision("high")


TASKS = {
    "celltype": dict(
        getter=make_h01_celltype_getter, path_key="celltype_path",
        num_classes=2, class_names=["exc", "inh"],
    ),
    "layer": dict(
        getter=make_h01_layer_getter, path_key="layer_path",
        num_classes=7, class_names=["L1", "L2", "L3", "L4", "L5", "L6", "WM"],
    ),
}


def extract_all_embeddings(cfg: DictConfig, task_cfg: dict, device: torch.device):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule

    print(f"Loading encoder from: {cls_cfg.checkpoint_path}")
    encoder = load_encoder_from_folder(cls_cfg.checkpoint_path)
    encoder.eval().requires_grad_(False).to(device)

    mean_x, std_x = load_feature_stats(cfg.h01.stats_sph_path)
    gen_normalize = GenNormalize(
        build_canonical_transform(dm_cfg, mean_x, std_x),
        mask_transform=None,
    )

    get_class = task_cfg["getter"](cfg.h01[task_cfg["path_key"]])
    print(f"Loading dataset from: {cfg.h01.path_sph}")
    ds = GraphDataSet(path=cfg.h01.path_sph, get_class=get_class, transform=gen_normalize)

    macro_mean, macro_std = compute_macro_stats(ds)  # None при use_macro=false
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
        desc="Extracting h01",
    )

    # neuron-level: агрегируем эмбеддинги ветвей/шипиков по segment_id (= нейрон)
    pooled = emb_set.pool_by_segment(pooling_type=cls_cfg["pooling_type"])
    print(f"  Группировка по нейрону: {emb_set.embeddings.shape[0]} веток -> "
          f"{pooled.embeddings.shape[0]} нейронов")
    x_all, y_all = pooled.embeddings, pooled.labels

    del encoder, encoder_graph, extractor, emb_set
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    #print("!!!!!!!!!!!!", y_all.shape)
    return x_all, y_all


@hydra.main(version_base="1.3", config_path="../../../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = cfg.transfer.get("task", "celltype") #Убрать
    if task not in TASKS:
        raise ValueError(f"transfer.task='{task}' не поддерживается, ожидается одно из {list(TASKS)}")
    task_cfg = TASKS[task]
    cfg.classifier.num_classes = task_cfg["num_classes"]  # авто под задачу

    print("=" * 60)
    print(" h01 linear-probe evaluation (neuron-level)")
    print(f" Encoder : {cfg.classifier.checkpoint_path}")
    print(f" Task    : {task} ({task_cfg['num_classes']} classes)")
    print("=" * 60)

    print("\n[1/2] Extracting Embeddings...")
    x_all, y_all = extract_all_embeddings(cfg, task_cfg, device)
    print(f"Features: {tuple(x_all.shape)}, Labels: {tuple(y_all.shape)}")

    print(f"\n[2/2] Training & Evaluating ({cfg.classifier.n_repeats}x{cfg.classifier.n_splits}-Fold CV)...")
    fold_metrics = train_cv(cfg, x_all, y_all, class_names=task_cfg["class_names"])

    if not fold_metrics:
        print("No metrics collected!")
        return

    summarize_cv(fold_metrics)


if __name__ == "__main__":
    main()
