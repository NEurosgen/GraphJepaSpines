import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import TypedDict, List, Tuple, Optional, Dict
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


class EncodedData(TypedDict):
    """Структура для хранения закодированных данных."""
    embedding: torch.Tensor
    label: int
    file_name: str          


class BaseEstimator(ABC):
    def __init__(self, data: Dict, **kwargs):
        self.data = data
        self._embeddings = None
        self._labels = None
    
    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            emb = self.data.get('embedding')
            if emb is None:
                emb = self.data.get('embeddings')
            if isinstance(emb, torch.Tensor):
                self._embeddings = emb.detach().cpu().numpy()
            else:
                self._embeddings = np.array(emb)
        return self._embeddings
    
    @property
    def labels(self) -> Optional[np.ndarray]:
        if self._labels is None:
            labels = self.data.get('label') or self.data.get('labels')
            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    self._labels = labels.detach().cpu().numpy()
                else:
                    self._labels = np.array(labels)
        return self._labels

    @abstractmethod
    def estimate(self, model=None) -> Dict:
        """
        Вычисляет метрики качества пространства.
        
        Args:
            model: Опциональный энкодер (не используется, данные уже закодированы)
            
        Returns:
            Словарь с метриками
        """
        pass


class ClusterQualityEstimator(BaseEstimator):
    """
    Оценивает качество кластеризации в латентном пространстве.
    
    Метрики:
    - silhouette_score: от -1 до 1, чем выше - лучше разделение кластеров
    - davies_bouldin: чем ниже - лучше (отношение внутрикластерных к межкластерным расстояниям)
    """
    
    def __init__(self, data: Dict, sample_size: int = 5000, **kwargs):
        super().__init__(data, **kwargs)
        self.sample_size = sample_size
    
    def estimate(self, model=None) -> Dict:
        X = self.embeddings
        y = self.labels
        
        if y is None:
            raise ValueError("ClusterQualityEstimator требует метки классов (labels)")
    
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return {'silhouette': 0.0, 'davies_bouldin': float('inf')}
        
        if len(X) > self.sample_size:
            indices = np.random.choice(len(X), self.sample_size, replace=False)
            X = X[indices]
            y = y[indices]
        
        try:
            silhouette = silhouette_score(X, y)
            davies_bouldin = davies_bouldin_score(X, y)
        except Exception as e:
            print(f"Ошибка при вычислении метрик кластеризации: {e}")
            return {'silhouette': 0.0, 'davies_bouldin': float('inf')}
        
        return {
            'silhouette': float(silhouette),
            'davies_bouldin': float(davies_bouldin)
        }


class RecallAtKEstimator(BaseEstimator):
    def __init__(self, data: Dict, k_values: Tuple[int, ...] = (1, 5, 10), **kwargs):
        super().__init__(data, **kwargs)
        self.k_values = k_values
    
    def estimate(self, model=None) -> Dict:
        """
        Вычисляет Recall@K для различных значений K.
        
        Returns:
            dict: {'recall@1': float, 'recall@5': float, 'recall@10': float}
        """
        X = self.embeddings
        y = self.labels
        
        if y is None:
            raise ValueError("RecallAtKEstimator требует метки классов (labels)")
        
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        
        similarity = X_norm @ X_norm.T
        np.fill_diagonal(similarity, -np.inf)
        
        results = {}
        for k in self.k_values:
            if k >= len(X):
                results[f'recall@{k}'] = 1.0
                continue
                
            top_k_indices = np.argsort(-similarity, axis=1)[:, :k]
            

            correct = 0
            for i in range(len(X)):
                if y[i] in y[top_k_indices[i]]:
                    correct += 1
            
            results[f'recall@{k}'] = correct / len(X)
        
        return results


class IsotropyEstimator(BaseEstimator):
    """
    Оценивает степень изотропии пространства через спектральный анализ.
    
    Идеально изотропное пространство имеет равномерно распределенную
    дисперсию по всем главным компонентам.
    
    Метрики:
    - isotropy: 0 = анизотропное (все в одном направлении), 1 = изотропное
    - explained_variance_ratio: доля объясненной дисперсии top-K компонентами
    - effective_dim: эффективная размерность (entropy-based)
    """
    
    def __init__(self, data: Dict, n_components: int = 10, **kwargs):
        super().__init__(data, **kwargs)
        self.n_components = n_components
    
    def estimate(self, model=None) -> Dict:

        X = self.embeddings
        X_centered = X - X.mean(axis=0)
        n_components = min(self.n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_components)
        pca.fit(X_centered)
        
        explained_var = pca.explained_variance_ratio_
        singular_values = pca.singular_values_
        sv_norm = singular_values / (singular_values.sum() + 1e-8)
        entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-8))
        max_entropy = np.log(len(singular_values))
        isotropy = entropy / max_entropy if max_entropy > 0 else 0.0
        effective_dim = np.exp(entropy)
        explained_top_k = np.sum(explained_var[:min(3, len(explained_var))])
        
        return {
            'isotropy': float(isotropy),
            'explained_variance_top3': float(explained_top_k),
            'effective_dim': float(effective_dim),
            'singular_values': singular_values.tolist()
        }


class RankMeEstimator(BaseEstimator):
    """
    Вычисляет RankMe - эффективный ранг матрицы эмбеддингов.
    
    RankMe = exp(-sum(p_i * log(p_i))), где p_i = sigma_i / sum(sigma)
    
    Высокий RankMe означает, что пространство использует больше измерений,
    что коррелирует с качеством представлений.
    
    Ref: "The Rank of Deep Networks" (Garrido et al., 2023)
    """
    
    def estimate(self, model=None) -> Dict:
        """
        Вычисляет RankMe и связанные метрики.
        
        Returns:
            dict: {'rank_me': float, 'condition_number': float, 'nuclear_norm': float}
        """
        X = self.embeddings
        X_centered = X - X.mean(axis=0)
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        S_norm = S / (S.sum() + 1e-8)
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-8))
        rank_me = np.exp(entropy)
        condition_number = S[0] / (S[-1] + 1e-8)
        nuclear_norm = S.sum()
        
        return {
            'rank_me': float(rank_me),
            'condition_number': float(condition_number),
            'nuclear_norm': float(nuclear_norm)
        }


class UniformityEstimator(BaseEstimator):
    """
    Оценивает равномерность распределения на гиперсфере.
    
    Uniformity = log(mean(exp(-2 * ||z_i - z_j||^2)))
    
    Низкие значения (более отрицательные) означают лучшую равномерность.
    
    Ref: "Understanding Contrastive Representation Learning" (Wang & Isola, 2020)
    """
    
    def __init__(self, data: Dict, t: float = 2.0, sample_size: int = 5000, **kwargs):
        super().__init__(data, **kwargs)
        self.t = t
        self.sample_size = sample_size
    
    def estimate(self, model=None) -> Dict:
        """
        Вычисляет uniformity loss.
        
        Returns:
            dict: {'uniformity': float}
        """
        X = self.embeddings
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        
        if len(X_norm) > self.sample_size:
            indices = np.random.choice(len(X_norm), self.sample_size, replace=False)
            X_norm = X_norm[indices]
        
        sq_distances = pdist(X_norm, metric='sqeuclidean')
        
        uniformity = np.log(np.mean(np.exp(-self.t * sq_distances)) + 1e-8)
        
        return {
            'uniformity': float(uniformity)
        }


class AlignmentEstimator(BaseEstimator):
    """
    Оценивает согласованность позитивных пар.
    
    Alignment = mean(||z_i - z_j||^2) для позитивных пар
    
    Низкие значения означают, что похожие объекты близки в пространстве.
    
    Ref: "Understanding Contrastive Representation Learning" (Wang & Isola, 2020)
    """
    
    def __init__(self, data: Dict, **kwargs):
        super().__init__(data, **kwargs)
    
    def estimate(self, model=None) -> Dict:
        """
        Вычисляет alignment для объектов одного класса.
        
        Returns:
            dict: {'alignment': float, 'intra_class_distance': float}
        """
        X = self.embeddings
        y = self.labels
        
        if y is None:
            raise ValueError("AlignmentEstimator требует метки классов (labels)")
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        total_distance = 0.0
        count = 0
        
        unique_labels = np.unique(y)
        for label in unique_labels:
            mask = y == label
            X_class = X_norm[mask]
            if len(X_class) < 2:
                continue
            pairwise_dist = pdist(X_class, metric='sqeuclidean')
            total_distance += np.sum(pairwise_dist)
            count += len(pairwise_dist)
        
        alignment = total_distance / count if count > 0 else 0.0
        
        return {
            'alignment': float(alignment),
            'intra_class_distance': float(np.sqrt(alignment))
        }


class CompositeEstimator(BaseEstimator):
    """
    Объединяет несколько эстиматоров для полного анализа пространства.
    """
    
    def __init__(self, data: Dict, estimators: List[str] = None, **kwargs):
        super().__init__(data, **kwargs)
        if estimators is None:
            estimators = ['rank_me', 'isotropy', 'uniformity']
        
        self._estimators = []
        estimator_map = {
            'cluster': ClusterQualityEstimator,
            'recall': RecallAtKEstimator,
            'isotropy': IsotropyEstimator,
            'rank_me': RankMeEstimator,
            'uniformity': UniformityEstimator,
            'alignment': AlignmentEstimator
        }
        
        for name in estimators:
            if name in estimator_map:
                self._estimators.append((name, estimator_map[name](data, **kwargs)))
    
    def estimate(self, model=None) -> Dict:
        """
        Вычисляет все метрики.
        
        Returns:
            dict: Объединенный словарь всех метрик
        """
        results = {}
        for name, estimator in self._estimators:
            try:
                metrics = estimator.estimate(model)
                results.update(metrics)
            except Exception as e:
                print(f"Ошибка в {name}: {e}")
        
        return results