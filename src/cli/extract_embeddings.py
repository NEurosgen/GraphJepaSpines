import torch
import hydra
import os
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from torch_geometric.nn import global_add_pool

from src.models.loader_model import load_encoder_from_folder
from src.models.encoder import GraphLatent
from src.data_utils.datamodule import GraphDataSet, make_minnie65_class_getter
from src.data_utils.transforms import GenNormalize
from src.cli.train_model import load_stats, build_transforms
from src.data_utils.stats import compute_macro_stats
from torch_geometric.data import Batch

def extract_from_dataset(dataset, encoder_graph, device, desc="Extracting"):
    embeddings = []
    labels = []
    segment_ids = []
    
    encoder_graph.eval()
    
    # Process one by one (or batch if possible, but one by one is simple since we just extract)
    # We will use DataLoader to speed up with num_workers
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=4,
        collate_fn=lambda x: Batch.from_data_list(x)
    )
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            # Filter unknown classes (-1)
            valid_mask = batch.y != -1
            if not valid_mask.any():
                continue
                
            batch = batch.to(device)
            # encoder_graph output is pooled embedding
            pooled_emb = encoder_graph(batch)
            
            embeddings.append(pooled_emb[valid_mask].cpu())
            labels.append(batch.y[valid_mask].cpu())
            segment_ids.append(batch.segment_id[valid_mask].cpu())
            
    if len(embeddings) == 0:
        return torch.empty((0, encoder_graph.encoder.out_channels)), torch.empty(0), torch.empty(0)
        
    return torch.cat(embeddings), torch.cat(labels), torch.cat(segment_ids)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load encoder
    encoder = load_encoder_from_folder(cls_cfg.checkpoint_path)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.to(device)
    
    # 2. Setup stats and transforms
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)
    
    # 3. Setup getter and Dataset
    csv_path = dm_cfg.dataset.class_path
    get_class_fn = make_minnie65_class_getter(csv_path)
    
    ds_path = cls_cfg.path if Path(cls_cfg.path).exists() else dm_cfg.dataset.path
    print(f"Using dataset path: {ds_path}")
    
    ds = GraphDataSet(
        path=ds_path,
        get_class=get_class_fn,
        transform=gen_normalize,
    )
    
    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)
    
    encoder_graph = GraphLatent(
        encoder=encoder, 
        macro_mean=macro_mean, 
        macro_std=macro_std, 
        pooling=global_add_pool, 
        sigma=cls_cfg.get("sigma", 1.0)
    ).to(device)
    
    # To split identically to training, we use random_split with fixed seed
    generator = torch.Generator().manual_seed(dm_cfg.seed)
    perm = torch.randperm(len(ds), generator=generator)
    
    ratio = dm_cfg.get("ratio", [0.7, 0.2, 0.1])
    train_size = int(len(ds) * ratio[0])
    val_size = int(len(ds) * ratio[1])
    
    train_ds = ds[perm[:train_size]]
    val_ds = ds[perm[train_size:train_size+val_size]]
    test_ds = ds[perm[train_size+val_size:]]
    
    # Extract
    emb_train, y_train, seg_train = extract_from_dataset(train_ds, encoder_graph, device, "Train")
    emb_val, y_val, seg_val = extract_from_dataset(val_ds, encoder_graph, device, "Val")
    emb_test, y_test, seg_test = extract_from_dataset(test_ds, encoder_graph, device, "Test")
    
    save_path = Path(cls_cfg.extracted_embeddings_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        'train': {'x': emb_train, 'y': y_train, 'seg': seg_train},
        'val': {'x': emb_val, 'y': y_val, 'seg': seg_val},
        'test': {'x': emb_test, 'y': y_test, 'seg': seg_test},
    }
    
    torch.save(output, save_path)
    print(f"[{emb_train.shape} dim] Embeddings saved to {save_path}")

if __name__ == "__main__":
    main()
