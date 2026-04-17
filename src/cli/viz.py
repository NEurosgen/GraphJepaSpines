import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from src.models.classificator import LinearClassifier
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


def visualize_classifier_embeddings_umap(
    embeddings_path: str,
    checkpoint_path: str,
    num_classes: int,
    in_channels: int,
    class_names: list = None
):
    """
    Извлекает эмбеддинги с предпоследнего слоя обученного классификатора 
    и визуализирует их с помощью UMAP.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Загрузка тестовых данных (сохраненных на шаге 2 пайплайна)
    print(f"Загрузка данных из {embeddings_path}...")
    emb_data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    x_test = emb_data['train']['x'].to(device)
    y_test = emb_data['train']['y'].numpy()
    seg_test = emb_data['train']['seg']
    

    x_test, y_test = pool_by_segment(x_test, y_test, seg_test, 'add')

    # 2. Инициализация и загрузка обученной модели
    print(f"Загрузка модели из {checkpoint_path}...")
    classifier_head = LinearClassifier(in_channels=in_channels, num_classes=num_classes)
    
    model = EmbeddingsLightModule.load_from_checkpoint(
        checkpoint_path,
        classifier=classifier_head,
        lr=1e-3, wd=1e-5, max_epochs=1, num_classes=num_classes
    ).to(device)
    model.eval()

    # 3. Извлечение признаков с помощью forward hook
    # Перехватываем входной тензор для самого последнего модуля (слоя классификации)
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # input - это кортеж. Берем первый элемент как тензор признаков
            activation[name] = input[0].detach().cpu().numpy()
        return hook

    # Находим последний слой в классификаторе. 
    # Если LinearClassifier - это просто nn.Linear, берем его.
    # Если это nn.Sequential, берем последний слой.
    last_layer = list(model.classifier.modules())[-1]
    handle = last_layer.register_forward_hook(get_activation('penultimate_features'))

    print("Прогон данных через классификатор...")
    with torch.no_grad():
        _ = model(x_test)
        
    handle.remove()
    features = activation['penultimate_features']
    #scaler = StandardScaler()
    #features_scaled = scaler.fit_transform(features)
    # 4. Снижение размерности с помощью UMAP
    print("Вычисление UMAP проекции...")
    reducer = TSNE(
        n_components=2,
        perplexity=8.0, 
        random_state=42,
        init='random',            # Инициализация через PCA делает алгоритм стабильнее
        learning_rate='auto',

    )
    
    # t-SNE сразу обучается и трансформирует данные
    embedding_2d = reducer.fit_transform(features)

    # 5. Визуализация
    print("Построение графика...")
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    # Формируем список строковых меток для каждого объекта
    labels = [class_names[idx] for idx in y_test]

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=embedding_2d[:, 0], 
        y=embedding_2d[:, 1], 
        hue=labels, 
        palette='tab20' if num_classes > 10 else 'tab10',
        s=30, 
        alpha=0.8,
        linewidth=0
    )

    plt.title('UMAP проекция эмбеддингов классификатора', fontsize=14)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    
    # Выносим легенду за пределы графика
    plt.legend(title='Классы', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.tight_layout()
    plt.show()

# Пример использования:
if __name__ == "__main__":
    # Замените пути и параметры на актуальные после завершения обучения
    visualize_classifier_embeddings_umap(
        embeddings_path="data/embeddings/r_0_embeddings.pt",
        checkpoint_path="/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/emb_classifier_r_0_sh_0/version_28/checkpoints/cls-r_0-sh_0-epoch=633-val_acc=0.9477.ckpt",
        num_classes=11,
        in_channels=39, # Укажите размерность x_test.shape[1]
        class_names=['23P', '4P', '5P-IT', '5P-NP', '5P-PT', '6P-CT', '6P-IT', 'BC', 'BPC', 'MC', 'NGC']
    )