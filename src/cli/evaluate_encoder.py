import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import re
import pytorch_lightning as L

from src.cli.evaluate_all_encoders import prepare_dataset_for_r, extract_embeddings_for_model, train_classifier

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    
    checkpoint_path = cfg.classifier.get("checkpoint_path", None)
    if not checkpoint_path:
        print("Error: classifier.checkpoint_path must be specified.")
        return
        
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Error: Path {path} does not exist.")
        return
        
    # Determine r and shuffle_ratio from path
    detected_r = None
    detected_sh = None
    
    # Try to find r_ value in path segments
    # We look at all parts of the path to find the folder name assigned by the logger
    for part in list(path.parts):
        r_match = re.search(r'jepa_r_(-?\d+\.?\d*)', part)
        if r_match:
            detected_r = float(r_match.group(1))
        sh_match = re.search(r'_sh_(\d+\.?\d*)', part)
        if sh_match:
            detected_sh = float(sh_match.group(1))

    if detected_r is not None:
        print(f"Detected r={detected_r} from path")
    if detected_sh is not None:
        print(f"Detected sh={detected_sh} from path")
        
    # Logic: Path detection has priority over config default (0), 
    # but we allow explicit overrides if they are passed (though Hydra makes this tricky).
    # For now, if detected, use it.
    r_val = detected_r if detected_r is not None else cfg.datamodule.get("r", 0)
    sh_val = detected_sh if detected_sh is not None else cfg.datamodule.get("shuffle_ratio", 0.0)
    print(f"Final parameters: r={r_val}, sh={sh_val}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Prepare dataset
    print(f"[1/3] Preparing dataset with r={r_val}, sh={sh_val} ...")
    prepare_dataset_for_r(cfg, r_val, sh_val)
    
    # 2. Extract embeddings
    print(f"[2/3] Extracting embeddings from {path} ...")
    emb_data = extract_embeddings_for_model(cfg, str(path), device)
    
    # 3. Train classifier
    print(f"[3/3] Training classifier ...")
    metrics = train_classifier(cfg, emb_data, r_val, sh_val)
    
    print("\n" + "="*30)
    print("      FINAL RESULTS")
    print("="*30)
    print(f"  test_acc: {metrics.get('test_acc', float('nan')):.4f}")
    print(f"  test_f1:  {metrics.get('test_f1', float('nan')):.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
