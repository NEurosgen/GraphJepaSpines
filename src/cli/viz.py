import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from sklearn.manifold import TSNE


from src.models.loader_model import load_classifier
from src.cli.evaluate_all_encoders import EmbeddingsLightModule

def pool_by_segment(embeddings, labels, segment_ids, pooling_type="mean"):
    """
    Groups embeddings by segment_ids and applies mean or add pooling.
    Returns (pooled_embeddings, labels).
    """
    if len(embeddings) == 0:
        return embeddings, labels
        
    unique_segments = torch.unique(segment_ids)
    
    pooled_x = []
    pooled_y = []
    
    for seg_id in unique_segments:
        mask = segment_ids == seg_id
        x_seg = embeddings[mask]
        y_seg = labels[mask][0] # All labels should be the same
        
        if pooling_type == "mean":
            x_pool = x_seg.mean(dim=0)
        else: # add
            x_pool = x_seg.sum(dim=0)
            
        pooled_x.append(x_pool)
        pooled_y.append(y_seg)
        
    return torch.stack(pooled_x), torch.tensor(pooled_y, dtype=torch.long)

def visualize_embeddings(
    embeddings, 
    labels, 
    method='tsne', 
    class_names=None, 
    title='Latent Space Visualization'
):
    print(f"Computing {method.upper()} projection...")
    if method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(random_state=42)
    else:
        if method == 'umap':
            print("UMAP not available or incompatible. Falling back to t-SNE.")
            method = 'tsne'
        reducer = TSNE(n_components=2, perplexity=15.0, random_state=42, init='pca', learning_rate='auto')
    
    embedding_2d = reducer.fit_transform(embeddings)

    print("Plotting...")
    if class_names is None:
        num_classes = len(np.unique(labels))
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    label_names = [class_names[idx] if idx < len(class_names) else f"Class {idx}" for idx in labels]

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=embedding_2d[:, 0], 
        y=embedding_2d[:, 1], 
        hue=label_names, 
        palette='tab10',
        s=50, 
        alpha=0.7,
        linewidth=0.5,
        edgecolor='white'
    )

    plt.title(title, fontsize=16)
    plt.xlabel(f'{method.upper()} 1', fontsize=12)
    plt.ylabel(f'{method.upper()} 2', fontsize=12)
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = Path("visualizations")
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / f"latent_space_{method}.png", dpi=800, bbox_inches='tight')
    print(f"Saved visualization to {save_path / f'latent_space_{method}.png'}")
    plt.show()

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cls_cfg = cfg.classifier
    embeddings_path = cls_cfg.get("extracted_embeddings_path", "data/embeddings/minnie65_embeddings.pt")
    checkpoint_path = cls_cfg.get("classifier_checkpoint_path")
    
    print(f"Loading embeddings from {embeddings_path}...")
    emb_data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    
    # We'll use the train set for visualization as it's usually larger
    x = emb_data['train']['x']
    y = emb_data['train']['y']
    
    # Check if we need to pool by segment
    if 'seg' in emb_data['train']:
        seg = emb_data['train']['seg']
        pooling_level = cls_cfg.get("pooling_level", "graph")
        if pooling_level == "neuron":
            pooling_type = cls_cfg.get("pooling_type", "mean")
            print(f"Pooling by segment using {pooling_type} (level: {pooling_level})...")
            x, y = pool_by_segment(x, y, seg, pooling_type)
    
    # Map many classes to two types if needed
    num_classes = len(torch.unique(y))
    print(f"Number of classes in data: {num_classes}")
    
    if num_classes == 2:
        class_names = ["Excitatory", "Inhibitory"]
    else:
        class_names = [
            '23P', '4P', '5P-IT', '5P-NP', '5P-PT', '6P-CT', '6P-IT', 
            'BC', 'BPC', 'MC', 'NGC'
        ]

    # Option 1: Visualize raw embeddings
    visualize_embeddings(
        x.cpu().numpy(), 
        y.cpu().numpy(), 
        method='tsne', 
        class_names=class_names,
        title='Latent Space (Raw Embeddings)'
    )
    
    # Option 2: Visualize penultimate features from classifier
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading classifier from {checkpoint_path}...")
        try:
            model = load_classifier(checkpoint_path).to(device)
            model.eval()
            
            # Use forward hook to get penultimate features
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    # input is (output_of_previous_layer,)
                    activation[name] = input[0].detach().cpu().numpy()
                return hook

            # Find the Linear layer in the Sequential head
            # LinearClassifier.head is nn.Sequential(norm, dropout, linear)
            last_layer = None
            if hasattr(model, 'classifier') and hasattr(model.classifier, 'head'):
                for layer in model.classifier.head:
                    if isinstance(layer, torch.nn.Linear):
                        last_layer = layer
                        break
            
            if last_layer:
                handle = last_layer.register_forward_hook(get_activation('penultimate'))
                
                print("Extracting penultimate features...")
                with torch.no_grad():
                    # Check if model expects batch object or raw tensor
                    if hasattr(model, 'forward_with_embeddings'): # hypothetical
                         _ = model.forward_with_embeddings(x.to(device))
                    else:
                        # ClassifierLightModule.forward(batch) calls encoder_graph(batch)
                        # We might need to call only the classifier part
                        if hasattr(model, 'classifier'):
                            _ = model.classifier(x.to(device))
                        else:
                            _ = model(x.to(device))
                
                handle.remove()
                features = activation['penultimate']
                
                visualize_embeddings(
                    features, 
                    y.cpu().numpy(), 
                    method='umap', 
                    class_names=class_names,
                    title='Latent Space (Classifier Penultimate Features)'
                )
            else:
                print("Could not find Linear layer for hook.")
        except Exception as e:
            print(f"Error loading classifier or extracting features: {e}")
    else:
        print(f"Checkpoint path {checkpoint_path} not found or not provided.")

if __name__ == "__main__":
    main()