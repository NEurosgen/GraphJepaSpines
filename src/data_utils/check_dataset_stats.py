import torch
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
from collections import Counter

# Import our custom dataset and getter
from src.data_utils.datamodule import GraphDataSet, make_minnie65_class_getter

def main():
    # Load config to get paths
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    ds_path = cfg['datamodule']['dataset']['path']
    # The config path is slightly wrong, using verified path
    csv_path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/public_cave_ground_truth_cell_types_with_nucleus.csv"
    
    if not os.path.exists(csv_path):
        # Fallback if I misnamed it
        csv_path = cfg['datamodule']['dataset']['class_path']

    print(f"Dataset path: {ds_path}")
    print(f"CSV path: {csv_path}")
    
    # Initialize getter
    get_class_fn = make_minnie65_class_getter(csv_path)
    
    # Initialize dataset
    ds = GraphDataSet(path=ds_path, get_class=get_class_fn)
    
    print(f"Total files found: {len(ds)}")
    
    # Mapping for labels to names
    class_map_inv = {
        0: '23P', 1: '4P', 2: '5P-IT', 3: '5P-NP', 4: '5P-PT',
        5: '6P-CT', 6: '6P-IT', 7: 'BC', 8: 'BPC', 9: 'MC', 10: 'NGC',
        -1: 'Unknown/Filtered'
    }
    
    counts = Counter()
    unknown_ids = []
    
    for i in tqdm(range(len(ds)), desc="Counting classes"):
        # We must use ds[i] to trigger GraphDataSet's loading which passes 'out' to getter
        item = ds[i]
        label = item.y.item()
        counts[label] += 1
        if label == -1:
            unknown_ids.append(item.segment_id if hasattr(item, 'segment_id') else ds.file_paths[i].name)
        
    print("\nClass Distribution:")
    # ... (same as before)
    
    # After the distribution table
    if unknown_ids:
        print("\nSome Unknown IDs (first 10):")
        for uid in unknown_ids[:10]:
            print(f" - {uid}")
    print(f"{'Label':<6} | {'Class Name':<20} | {'Count':<10}")
    print("-" * 42)
    
    for label in sorted(counts.keys()):
        name = class_map_inv.get(label, "Unknown")
        count = counts[label]
        print(f"{label:<6} | {name:<20} | {count:<10}")
        
    total_valid = sum(count for lbl, count in counts.items() if lbl != -1)
    print("-" * 42)
    print(f"Total labeled: {total_valid}")
    print(f"Total unknown: {counts[-1]}")

if __name__ == "__main__":
    main()
