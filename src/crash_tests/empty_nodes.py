import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
from hydra.utils import instantiate
import torch
import shutil
import gc 
import os
from pathlib import Path
from tqdm import tqdm

from src.cli.train_model import get_datamodule
from src.models.jepa import JepaLight
from src.data_utils.transforms import (
    GraphPruning, LaplacianPE, CentralityEncoding,
    RandomWalkPE, LocalPos, GaussianNoiseAugmentation
)

torch.set_float32_matmul_precision('high')

SAVE_DIR = Path("crash_tests/empty_nodes")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def train_multiple_sigma(cfg: DictConfig):
    """
    Crash test: train JEPA models with additive Gaussian noise N(0, sigma)
    on node features to evaluate how much the model relies on structural
    graph properties versus morphological node features.
    """
    sigma_values = [10, 100, 10000]
    
    print("=" * 60)
    print("  Gaussian Noise Crash Test")
    print(f"  Sigma values: {sigma_values}")
    print("=" * 60)

    for sigma in sigma_values:
        print(f"\n{'=' * 50}")
        print(f"   Starting iteration for sigma = {sigma}")
        print(f"{'=' * 50}")

        # --- Dataset preparation ---
        out_dir = Path(cfg.datamodule.dataset.path)

        if out_dir.exists():
            print(f"Removing old prepared dataset {out_dir}...")
            shutil.rmtree(out_dir)

        dataset_path = Path(cfg.datamodule.dataset.raw_path)
        file_paths = sorted(dataset_path.rglob('*.pt'))

        OmegaConf.set_struct(cfg, False)
        knn_k = cfg.datamodule.get('knn', -1)
        radius_r = cfg.datamodule.get('r', -1.0)
        mutual = cfg.datamodule.get('mutual_knn', False)
        OmegaConf.set_struct(cfg, True)

        pruning = GraphPruning(k=knn_k, r=radius_r, mutual=mutual)
        noise_aug = GaussianNoiseAugmentation(sigma=sigma)

        se_cfg = cfg.datamodule.get('structural_encoding', {})
        lap_k = se_cfg.get('laplacian_k', 0)
        centrality = se_cfg.get('centrality', False)
        rw_steps = se_cfg.get('random_walk_steps', 0)

        print(f"Preparing dataset with sigma={sigma}...")
        for file_path in tqdm(file_paths, leave=False, desc=f"Preparing (sigma={sigma})"):
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            data = pruning(LocalPos()(data))
            data = noise_aug(data)
            x_dim_original = data.x.size(1)

            if lap_k > 0:
                lap_pe_module = LaplacianPE(k=lap_k)
                data = lap_pe_module(data)
                data.laplacian_pe = data.x[:, x_dim_original:]
                data.x = data.x[:, :x_dim_original]

            if centrality:
                cent_module = CentralityEncoding()
                data = cent_module(data)
                data.centrality_pe = data.x[:, x_dim_original:]
                data.x = data.x[:, :x_dim_original]

            if rw_steps > 0:
                rw_module = RandomWalkPE(walk_length=rw_steps)
                data = rw_module(data)
                data.random_walk_pe = data.x[:, x_dim_original:]
                data.x = data.x[:, :x_dim_original]

            rel_path = file_path.relative_to(dataset_path)
            out_file = out_dir / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, out_file)

        print("Done preparing. Initializing Model...")

        # --- Model initialization ---
        L.seed_everything(cfg.seed, workers=True)
        model = instantiate(cfg.network, _recursive_=True)
        model_module = JepaLight(cfg=cfg, model=model, debug=False)

        # --- Trainer setup ---
        sigma_save_dir = SAVE_DIR / f"sigma_{sigma}"
        sigma_save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = L.callbacks.ModelCheckpoint(
            dirpath=str(sigma_save_dir),
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename=f"jepa-sigma_{sigma}-{{epoch:02d}}-{{val_loss:.4f}}"
        )

        logger = L.loggers.TensorBoardLogger(
            save_dir=cfg.get("log_dir", "lightning_logs"),
            name=f"jepa_sigma_{sigma}"
        )

        trainer = L.Trainer(
            **cfg.trainer,
            logger=logger,
            callbacks=[checkpoint_callback],
            deterministic=True
        )

        datamodule = get_datamodule(cfg.datamodule)

        # --- Training ---
        print(f"Starting Training for sigma = {sigma}")
        trainer.fit(model_module, datamodule=datamodule)
        print(f"Finished Training for sigma = {sigma}\n")

        # --- Memory cleanup ---
        print(f"Clearing RAM and GPU memory for sigma = {sigma}...")
        del model
        del model_module
        del trainer
        del datamodule
        del checkpoint_callback
        del logger

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    train_multiple_sigma()