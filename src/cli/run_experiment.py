"""
Скрипт для извлечения эмбеддингов из обученной модели GraphJEPA
и визуализации латентного пространства с помощью PCA, t-SNE и UMAP.
"""

import torch
import torch.serialization
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Импорты для методов снижения размерности
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    print("\n[WARNING] UMAP library not found. Install with `pip install umap-learn`.")
    umap = None

# PyTorch 2.6+ требует явного разрешения для OmegaConf классов
from omegaconf import DictConfig, ListConfig
torch.serialization.add_safe_globals([DictConfig, ListConfig])

from src.data_utils.transforms import GenNormalize, FeatureChoice, NormNoEps, EdgeNorm
from src.data_utils.datamodule import GraphDataModule, GraphDataSet
# from src.models.jepa import JepaLight # Не используется напрямую в этом скрипте


def load_stats(path: str):
    """Загружает статистики нормализации для признаков узлов и рёбер."""
    mean_x = torch.load(path + "means.pt")
    std_x = torch.load(path + "stds.pt")
    mean_edge = torch.load(path + "mean_edge.pt")
    std_edge = torch.load(path + "std_edge.pt")
    return mean_x, std_x, mean_edge, std_edge


def get_datamodule(path: str, stats_path: str, batch_size: int = 1, features: list = None):
    """Создаёт DataModule для датасета графов.
    
    Args:
        path: Путь к датасету
        stats_path: Путь к статистикам нормализации
        batch_size: Размер батча
        features: Список индексов фич для выбора (если None - используются все)
    """
    mean_x, std_x, mean_edge, std_edge = load_stats(stats_path)
    
    # Создаём pipeline трансформаций
    transforms = []
    
    if features is not None:
        # Сначала выбираем нужные фичи
        transforms.append(FeatureChoice(features))
        # Нормализуем только выбранные фичи (срезаем статистики)
        mean_x = mean_x[features]
        std_x = std_x[features]
    
    # Добавляем нормализацию
    transforms.append(NormNoEps(mean_x, std_x))
    transforms.append(EdgeNorm(mean_edge, std_edge))
    
    # Собираем в GenNormalize (без mask_transform для inference)
    norm = GenNormalize(transforms=transforms, mask_transform=None)
    
    ds = GraphDataSet(path=path, transform=norm)
    
    # Collate function для PyG Data объектов
    from torch_geometric.data import Batch
    def collate_fn(data_list):
        return Batch.from_data_list(data_list)
    
    datamodule = GraphDataModule(
        ds, 
        batch_size,
        num_workers=0,  # Используем 0 для inference
        seed=42,
        ratio=[1, 0, 0],  # Все данные в train split для inference
        collate_fn=collate_fn
    )
    return datamodule


def extract_embeddings(encoder, datamodule: GraphDataModule, 
                       label: int, sigma: float = 1.0, device: str = 'cuda'):
    """
    Извлекает эмбеддинги из датасета используя обученный энкодер.
    """
    encoder = encoder.to(device)
    encoder.eval()
    
    embeddings_list = []
    labels_list = []
    filenames_list = []
    
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting embeddings (class {label})"):
            # Batch может быть tuple (context, target) или один граф
            if isinstance(batch, tuple):
                context_batch, _ = batch
            else:
                context_batch = batch
            
            context_batch = context_batch.to(device)
            
            # Применяем RBF преобразование к edge_attr (как в training_step)
            edge_attr = context_batch.edge_attr
            if edge_attr is not None:
                edge_attr = torch.exp(-edge_attr**2 / sigma**2)
            
            # Получаем эмбеддинги через энкодер
            emb = encoder(
                context_batch.x, 
                context_batch.edge_index, 
                edge_attr
            )
            
            # Агрегируем эмбеддинги узлов в эмбеддинг графа (mean pooling)
            if hasattr(context_batch, 'batch') and context_batch.batch is not None:
                # Несколько графов в батче
                from torch_geometric.nn import global_mean_pool
                graph_emb = global_mean_pool(emb, context_batch.batch)
            else:
                # Один граф
                graph_emb = emb.mean(dim=0, keepdim=True)
            
            embeddings_list.append(graph_emb.cpu())
            
            # Добавляем метки
            batch_size = graph_emb.size(0)
            labels_list.extend([label] * batch_size)
            
            # Добавляем имена файлов если есть
            if hasattr(context_batch, 'file_name'):
                if isinstance(context_batch.file_name, list):
                    filenames_list.extend(context_batch.file_name)
                else:
                    filenames_list.append(context_batch.file_name)
            else:
                filenames_list.extend([f"graph_{i}" for i in range(batch_size)])
    
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = np.array(labels_list)
    
    return embeddings, labels, filenames_list


# --- НОВЫЙ БЛОК ВИЗУАЛИЗАЦИИ ---

def plot_scatter(X_2d, labels, title, save_path):
    """
    Вспомогательная функция для отрисовки 2D графика рассеяния.
    """
    plt.figure(figsize=(11, 9))
    sns.set_theme(style="whitegrid")
    
    # Преобразуем числовые метки обратно в текстовые для легенды
    label_names = ['AB (class 0)' if l == 0 else 'WT (class 1)' for l in labels]
    
    scatter = sns.scatterplot(
        x=X_2d[:, 0], 
        y=X_2d[:, 1],
        hue=label_names,
        palette=sns.color_palette("deep", len(np.unique(labels))),
        style=label_names,
        s=60,
        alpha=0.8,
        edgecolor='w'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(title="Classes", title_fontsize=12, fontsize=11)
    
    # Убираем рамки сверху и справа для чистоты
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранен график: {save_path}")


def visualize_embeddings(embeddings: torch.Tensor, labels: np.ndarray, output_dir: str,tag = ''):
    """
    Выполняет PCA, t-SNE и UMAP проекции и сохраняет графики.
    """
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ ЛАТЕНТНОГО ПРОСТРАНСТВА")
    print("="*60)
    
    # Убедимся, что директория для сохранения существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Переводим в numpy для sklearn/umap
    X = embeddings.numpy()
    
    # 1. PCA (Principal Component Analysis) - Линейный метод
    print("\n📊 Вычисление PCA (2 компоненты)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    title_pca = f"PCA Projection (Explained Variance: {explained_variance[0]+explained_variance[1]:.2%})"
    plot_scatter(X_pca, labels, title_pca, os.path.join(output_dir, tag + "visualization_pca.png"))
    
    # 2. t-SNE (t-distributed Stochastic Neighbor Embedding) - Нелинейный, вероятностный
    print("\n🗺️ Вычисление t-SNE...")
    # Параметр perplexity влияет на баланс внимания между локальными и глобальными аспектами
    # Обычно выбирают между 5 и 50. Чем больше данных, тем больше можно ставить.
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)/10), max_iter=1500, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)
    plot_scatter(X_tsne, labels, "t-SNE Projection", os.path.join(output_dir,tag + "visualization_tsne.png"))
    
    # 3. UMAP (Uniform Manifold Approximation and Projection) - Нелинейный, топологический
    # Обычно быстрее t-SNE и лучше сохраняет глобальную структуру
    if umap is not None:
        print("\n🌌 Вычисление UMAP...")
        # n_neighbors: баланс локальной (меньше) и глобальной (больше) структуры (default=15)
        # min_dist: насколько плотно могут группироваться точки (default=0.1)
        umap_reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42, n_jobs=-1)
        X_umap = umap_reducer.fit_transform(X)
        plot_scatter(X_umap, labels, "UMAP Projection", os.path.join(output_dir,tag + "visualization_umap.png"))
    else:
        print("   Пропуск UMAP (библиотека не установлена).")

# --- КОНЕЦ НОВОГО БЛОКА ---


def main():
    # Пути к данным
    stats_path = '/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/data/stats_9009/'
    path_ab = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/graph_dataset_output_ab"
    path_wt = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/notebooks/graph_dataset_output_wt"
    checkpoint_path = "/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/lightning_logs/version_142/checkpoints/epoch=62-step=139923.ckpt"
    
    # Фичи для моделей обученных с FeatureChoice (из main ветки)
    # Если модель обучена со всеми фичами - установить features = None
    features = [0, 4, 5, 6, 7, 13, 14, 15, 17, 19, 20]
    
    # Папка для сохранения результатов визуализации
    tag = "jepa_feature_"

    output_base_path = '/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/exp/'
    visualization_dir = os.path.join(output_base_path, "visualizations")

    # Устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Загружаем чекпоинт напрямую
    print("\n📦 Загрузка чекпоинта...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Смотрим структуру state_dict чтобы понять размерности энкодера
    state_dict = checkpoint['state_dict']
    
    # Определяем префикс энкодера (разные для LeJEPA и GraphJepa)
    encoder_prefix = None
    for key in state_dict.keys():
        if 'encoder.proj.weight' in key:
            if 'student_encoder' in key:
                encoder_prefix = 'model.student_encoder.'
            else:
                encoder_prefix = 'model.encoder.'
            break
    
    if encoder_prefix is None:
        raise ValueError("Не удалось найти энкодер в чекпоинте")
    
    # Находим proj.weight чтобы определить размерности
    proj_key = f'{encoder_prefix}proj.weight'
    proj_weight = state_dict[proj_key]
    out_channels, in_channels = proj_weight.shape
    
    # Считаем количество слоёв
    num_layers = 0
    for key in state_dict.keys():
        if f'{encoder_prefix}layers' in key and '.model' in key:
            layer_idx = int(key.split('layers.')[1].split('.')[0])
            num_layers = max(num_layers, layer_idx + 1)
    
    print(f"   Обнаружено: in={in_channels}, out={out_channels}, layers={num_layers}")
    print(f"   Префикс энкодера: {encoder_prefix}")
    
    # Создаём энкодер
    # (Этот импорт должен работать, если ваш проект структурирован так же, как в оригинале)
    try:
        from src.models.encoder import GraphGcnEncoder
    except ImportError:
         print("\nОШИБКА ИМПОРТА: Убедитесь, что скрипт запускается из корня проекта, чтобы 'src' был доступен.")
         exit(1)

    encoder = GraphGcnEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers
    )
    
    # Извлекаем веса энкодера
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(encoder_prefix):
            new_key = key[len(encoder_prefix):]
            encoder_state_dict[new_key] = value
    
    encoder.load_state_dict(encoder_state_dict)
    encoder = encoder.to(device)
    encoder.eval()
    print("✅ Энкодер загружен успешно!")
    
    # Создаём DataModules для обоих классов
    print("\n📂 Создание DataModules...")
    dm_ab = get_datamodule(path_ab, stats_path, batch_size=32, features=features)
    dm_wt = get_datamodule(path_wt, stats_path, batch_size=32, features=features)
    
    # Извлекаем эмбеддинги
    print("\n🔄 Извлечение эмбеддингов...")
    embeddings_ab, labels_ab, files_ab = extract_embeddings(
        encoder, dm_ab, label=0, device=device
    )
    embeddings_wt, labels_wt, files_wt = extract_embeddings(
        encoder, dm_wt, label=1, device=device
    )
    
    # Объединяем эмбеддинги обоих классов
    all_embeddings = torch.cat([embeddings_ab, embeddings_wt], dim=0)
    all_labels = np.concatenate([labels_ab, labels_wt])
    all_files = files_ab + files_wt
    
    print(f"\n📊 Статистика датасета:")
    print(f"   Класс AB (label=0): {len(labels_ab)} графов")
    print(f"   Класс WT (label=1): {len(labels_wt)} графов")
    print(f"   Всего: {len(all_labels)} графов")
    print(f"   Размерность эмбеддингов: {all_embeddings.shape[1]}")
    
    # --- ЗАПУСК ВИЗУАЛИЗАЦИИ ВМЕСТО ЭСТИМАТОРОВ ---
    visualize_embeddings(all_embeddings, all_labels, visualization_dir,tag = tag)
    
    # Сохраняем сами эмбеддинги и метки на всякий случай
    print(f"\n💾 Сохранение сырых эмбеддингов в {output_base_path}...")
    torch.save({
        'embeddings': all_embeddings,
        'labels': all_labels,
        'files': all_files
    }, os.path.join(visualization_dir, tag + 'embeddings_raw.pt'))
    
    print(f"\n✅ Готово! Графики сохранены в: {visualization_dir}")
    
    return all_embeddings, all_labels


if __name__ == "__main__":
    main()