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
    RandomWalkPE, LocalPos, GaussianPositionNoise
)

torch.set_float32_matmul_precision('high')

SAVE_DIR = Path("src/crash_tests/experiments_shufle")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def train_position_noise(cfg: DictConfig):
    """
    Experiment 2: Train JEPA encoders with Gaussian noise N(0, sigma)
    added to node positions (data.pos) to evaluate how much the model
    relies on spine position vs other structural features.

    Models are saved in standard Lightning format:
        SAVE_DIR/jepa_pos_sigma_{sigma}/version_X/
            checkpoints/*.ckpt
            hparams.yaml
            events.out.tfevents.*
    """
    sigma_values = [1000, 10000]

    print("=" * 60)
    print("  Experiment 2: Position Noise Crash Test")
    print(f"  Sigma values: {sigma_values}")
    print(f"  Save dir: {SAVE_DIR}")
    print("=" * 60)

    for sigma in sigma_values:
        print(f"\n{'=' * 50}")
        print(f"   Starting iteration for pos_sigma = {sigma}")
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
        pos_noise = GaussianPositionNoise(sigma=sigma)

        se_cfg = cfg.datamodule.get('structural_encoding', {})
        lap_k = se_cfg.get('laplacian_k', 0)
        centrality = se_cfg.get('centrality', False)
        rw_steps = se_cfg.get('random_walk_steps', 0)

        print(f"Preparing dataset with pos_sigma={sigma}...")
        for file_path in tqdm(file_paths, leave=False, desc=f"Preparing (pos_sigma={sigma})"):
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            # LocalPos normalizes positions, then we add noise AFTER normalization
            data = LocalPos()(data)
            # Add Gaussian noise to normalized positions
            data = pos_noise(data)
            # Prune graph based on (noisy) positions
            data = pruning(data)
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
        # Use TensorBoardLogger with save_dir=SAVE_DIR so that
        # the standard Lightning layout is created automatically:
        #   SAVE_DIR/name/version_X/checkpoints/*.ckpt
        #   SAVE_DIR/name/version_X/hparams.yaml
        #   SAVE_DIR/name/version_X/events.out.tfevents.*
        logger = L.loggers.TensorBoardLogger(
            save_dir=str(SAVE_DIR),
            name=f"jepa_pos_sigma_{sigma}"
        )

        # Do NOT set dirpath — let Lightning save checkpoints
        # inside the logger's version directory automatically
        checkpoint_callback = L.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename=f"jepa-pos_sigma_{sigma}-{{epoch:02d}}-{{val_loss:.4f}}"
        )

        trainer = L.Trainer(
            **cfg.trainer,
            logger=logger,
            callbacks=[checkpoint_callback],
            deterministic=True
        )

        datamodule = get_datamodule(cfg.datamodule)

        # --- Training ---
        print(f"Starting Training for pos_sigma = {sigma}")
        trainer.fit(model_module, datamodule=datamodule)
        print(f"Finished Training for pos_sigma = {sigma}\n")

        # --- Memory cleanup ---
        print(f"Clearing RAM and GPU memory for pos_sigma = {sigma}...")
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
    train_position_noise()
