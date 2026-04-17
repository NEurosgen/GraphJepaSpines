import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
from hydra.utils import instantiate
import torch
import shutil
import gc  # Добавлен импорт сборщика мусора
from pathlib import Path

# Import the logic functions directly from our own modules
from src.cli.prepare_dataset import main as prepare_main
from src.cli.train_model import get_datamodule
from src.models.jepa import JepaLight

torch.set_float32_matmul_precision('high')

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def train_multiple_r(cfg: DictConfig):
    # Definition of densities to iterate through
    print("Warning: iterating through r_values and shuffle_ratios, clearing dataset folder each time.")
    r_values = [0, 1 , 1.5 ,2 ,3]  # Replace with the actual array values you want to test!
    shuffle_ratios = [ 0] # Spectrum of randomness
    
    for shuffle_ratio in shuffle_ratios:
        for r in r_values:
            print(f"=========================================")
            print(f"   Starting iteration for r = {r}, shuffle = {shuffle_ratio}   ")
            print(f"=========================================")
            
            # We need to manually set the config parameters
            OmegaConf.set_struct(cfg, False)
            cfg.datamodule.r = r
            cfg.datamodule.shuffle_ratio = shuffle_ratio
            OmegaConf.set_struct(cfg, True)
            
            # Prepare the dataset again based on the updated config
            out_dir = Path(cfg.datamodule.dataset.path)
            
            # Because we overwrite in the same structure, clean the existing _prepared directory 
            if out_dir.exists():
                print(f"Removing old prepared dataset {out_dir} to replace with new ones...")
                shutil.rmtree(out_dir)
                
            print("Running dataset preparation...")
            import os
            from tqdm import tqdm
            from src.data_utils.transforms import GraphPruning, LaplacianPE, CentralityEncoding, RandomWalkPE, LocalPos, FeatureShuffling
            
            dataset_path = Path(cfg.datamodule.dataset.raw_path)
            file_paths = sorted(dataset_path.rglob('*.pt'))
            
            knn_k = cfg.datamodule.get('knn', -1)
            radius_r = cfg.datamodule.get('r', -1.0)
            mutual = cfg.datamodule.get('mutual_knn', False)
            pruning = GraphPruning(k=knn_k, r=radius_r, mutual=mutual)
            shuffling = FeatureShuffling(ratio=cfg.datamodule.get('shuffle_ratio', 0.0))
            
            se_cfg = cfg.datamodule.get('structural_encoding', {})
            lap_k = se_cfg.get('laplacian_k', 0)
            centrality = se_cfg.get('centrality', False)
            rw_steps = se_cfg.get('random_walk_steps', 0)
            
            for file_path in tqdm(file_paths, leave=False, desc=f"Preparing dataset r={r}, shuffle={shuffle_ratio}"):
                data = torch.load(file_path, map_location='cpu', weights_only=False)
                data = pruning(LocalPos()(data))
                data = shuffling(data) # Apply feature shuffling
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
                
            print("Done preparing. Initializing Model...)")
            
            # Model initialization
            L.seed_everything(cfg.seed, workers=True)
            model = instantiate(cfg.network, _recursive_=True)
            
            model_module = JepaLight(cfg=cfg, model=model, debug=False)
            
            # Setup trainer
            checkpoint_callback = L.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                filename=f"jepa-r_{r}-sh_{shuffle_ratio}-{{epoch:02d}}-{{val_loss:.4f}}"
            )

            logger = L.loggers.TensorBoardLogger(
                save_dir=cfg.get("log_dir", "lightning_logs"), 
                name=f"jepa_r_{r}_sh_{shuffle_ratio}" # Keep different tensorboard models separate
            )

            trainer = L.Trainer(
                **cfg.trainer,
                logger=logger,
                callbacks=[checkpoint_callback],
                deterministic=True
            )

            datamodule = get_datamodule(cfg.datamodule)
            
            print(f"Starting Training for r = {r}, shuffle = {shuffle_ratio}")
            trainer.fit(model_module, datamodule=datamodule)
            print(f"Finished Training for r = {r}, shuffle = {shuffle_ratio}\n")

            # --- Блок очистки памяти ---
            print(f"Clearing RAM and GPU memory for r = {r}, shuffle = {shuffle_ratio}...")
            
            # Удаляем ссылки на тяжелые объекты
            del model
            del model_module
            del trainer
            del datamodule
            del checkpoint_callback
            del logger
            
            # Принудительно вызываем сборщик мусора для очистки RAM
            gc.collect()
            
            # Очищаем кэш GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() # Очистка IPC памяти, полезно при workers > 0 в DataLoader
            # ---------------------------

if __name__ == "__main__":
    train_multiple_r()