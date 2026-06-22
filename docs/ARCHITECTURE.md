# GraphJEPA Spines — Архитектура и документация проекта

> Документ описывает текущее состояние research-проекта: что реализовано, как
> устроены модули, как запускать пайплайны, и где находятся слабые места /
> точки роста. Цель — дать «карту» кодовой базы, чтобы можно было осознанно
> выбирать направление дальнейшего развития.

Дата составления: 2026-06-09. Ветка: `main`.

---

## 1. О чём проект

Цель — построить **self-supervised** латентное пространство для дендритных
шипиков (dendritic spines) на основе их геометрии, **без использования
разметки**. В качестве базовой архитектуры выбрана **JEPA (Joint Embedding
Predictive Architecture)** — она избегает дорогого декодера (как в
автоэнкодерах / VGAE) и фокусируется на семантике латентного пространства.

Качество выученного пространства затем проверяется через **linear probe**
(линейный классификатор поверх замороженного энкодера) на размеченных
датасетах, а также через набор **unsupervised-метрик** (RankMe, isotropy,
uniformity и т.д.).

### Модальность данных

Каждый объект — **граф фрагмента дендрита**:
- **Узлы** — отдельные шипики, признак узла = вектор геометрических дескрипторов
  (используются «сферические» / spherical descriptors).
- **Рёбра** — пространственная близость; атрибут ребра = евклидово расстояние.
- **Позиции** (`data.pos`) — 3D-координаты шипиков, используются предиктором и
  для построения графа по радиусу.

> ⚠️ Пайплайн «сырой меш → граф `.pt`» **в репозитории отсутствует**. Проект
> работает с уже построенными графами `.pt`; нормализация и построение рёбер
> делаются внутри пайплайна на лету (см. §3). Билдер графов h01 —
> `src/data_utils/build_h01_dendrite_graph.py`.

---

## 2. Карта репозитория

```
GIT_Graph_refactor/
├── configs/                  # Hydra-конфиги (главный + по датасетам)
│   ├── config.yaml           # корневой конфиг (network / training / datamodule / classifier)
│   ├── 9009.yaml             # пути датасета 9009 (ab/wt мыши)
│   ├── minnie65.yaml         # пути датасета Minnie65 (типы клеток)
│   └── human_descriptors.yaml# пути датасета человеческих шипиков (возраст)
│
├── src/
│   ├── models/               # модели: энкодеры, JEPA/LeJEPA, загрузчик чекпоинтов
│   │   ├── encoder.py
│   │   ├── jepa.py
│   │   ├── loader_model.py
│   │   └── test_models/
│   │
│   ├── data_utils/           # датасеты, трансформы, статистики, метрики графа
│   │   ├── datamodule.py
│   │   ├── transforms.py
│   │   ├── stats.py
│   │   ├── structural_stats.py
│   │   ├── check_dataset_stats.py
│   │   └── tests/
│   │
│   ├── cli/                  # точки входа (обучение, извлечение эмб., инференс)
│   │   ├── train/train_model.py
│   │   ├── embedding_pipeline.py
│   │   └── inference/
│   │       ├── 9009/evaluate_encoder_cv.py
│   │       ├── minnie65/{evaluate_encoder_cv.py, minnie65_get_class.py}
│   │       └── human_age/{evaluate_encoder_cv.py, stats_analysis.ipynb}
│   │
│   ├── representation/       # unsupervised-метрики качества пространства
│   │   └── estimators.py
│   │
│   ├── explainer/            # объяснимость (GNNExplainer, IG, counterfactual)
│   │   ├── models.py         # GraphExplainerWrapper
│   │   ├── utils.py          # setup_explainer_environment
│   │   ├── visuals.py        # 3D-визуализация на меше
│   │   ├── 9009/{explainer, global_explainer, integrated_grad, contr_fact}.py
│   │   └── tests/
│   │
│   └── experiment/           # скрипты экспериментов
│       └── train_val/multi_step_experiment.py
│
├── exp/                      # ноутбуки + сводки по экспериментам (alpha_enc, enc_layers, ...)
├── tests/                    # интеграционные тесты пайплайна данных
├── checkpoints/              # обученные веса (.ckpt) — много версий
├── data/                     # статистики нормализации + извлечённые эмбеддинги
├── datasets/                 # исходные графы .pt датасетов (не в git)
└── docs/                     # этот документ + images
```

---

## 3. Поток данных (data pipeline)

> **Без отдельного шага записи на диск.** Раньше был `preprocess_dataset`,
> материализовавший второй датасет на диск (оправдано дорогим `networkx` и
> структурными PE). Сейчас сложные энкодинги убраны, канонизация дешёвая —
> считается на лету в `GraphDataSet` и кэшируется в RAM (`save_cache`). Один
> исходный датасет, без `prepare`-копии. См. §8.

**Принцип единообразия:** датасеты неоднородны — h01 приходит без рёбер (только
`x`, `pos`), minnie65 полносвязный, где-то есть предпосчитанные дистанции, где-то
нет. Общий знаменатель — **`x` + `pos`**, поэтому структура графа **всегда
строится из `pos`** (а `edge_attr` = дистанции из `pos`), любой исходный
`edge_index`/`edge_attr` перетирается.

```
один исходный датасет .pt графов
   │
   ▼  GraphDataSet.get(idx)  — КАНОНИЗАЦИЯ (детерминированная, кэш в RAM):
   │     build_canonical_transform() (transforms.py)
   │     ├─ FeatureChoice           (опц., выбор подмножества признаков)
   │     ├─ NormNoEps               (z-нормализация x, игнор нулей)
   │     ├─ LocalPos                (нормализация pos на граф)
   │     ├─ BuildGeometricGraph(r)  (рёбра из pos по радиусу r + edge_attr = дистанции)
   │     └─ ThesisMacroMetrics      (опц., по флагу cfg.use_macro)
   │
   ▼  collate_fn (стохастически, на каждый батч, num_views раз на граф):
   │     ├─ augment(view)           (хук под аугментации, напр. поворот; пока выключен)
   │     └─ MaskData(mask_ratio) → (context_graph, target_graph)   независимо на каждый view
   │
   ▼  GraphDataModule  — train/val/test split по ratio, DataLoader'ы
```

Ключевая идея разделения:
- **Канонизация** (нормализация + построение графа) детерминирована → считается
  один раз и кэшируется в RAM (`save_cache=True`).
- **Аугментация + маскирование** стохастические → на каждом шаге в `collate_fn`.
  На каждый базовый граф генерируется `num_views` представлений, и для каждого
  **независимо** считается своя маска → пара (context, target) для JEPA.

### Трансформы (`src/data_utils/transforms.py`)

| Класс | Назначение |
|---|---|
| `NormNoEps` | z-нормализация признаков узлов, не трогает значения `|x|<eps` (нули-заглушки) |
| `LocalPos` | per-graph нормализация 3D-позиций |
| `BuildGeometricGraph(r)` | **единообразное** построение графа из `pos`: рёбра `radius_graph(r)` + `edge_attr` = евклидовы дистанции `[E,1]`; перетирает любой исходный формат |
| `ThesisMacroMetrics` | 7 макро-признаков графа (опц., за флагом `use_macro`; см. ниже) |
| `MaskData(ratio)` | возвращает **два** графа: context и target (JEPA) |
| `FeatureChoice` | выбор подмножества признаков по индексам |
| `FeatureShuffling(ratio)` | абляция: перемешивание признаков (ломает связь со структурой) |
| `GaussianNoiseAugmentation` / `GaussianPositionNoise` | абляции на устойчивость к шуму |

Сборка: `build_canonical_transform(cfg, mean_x, std_x)` собирает детерминированную
часть; `create_mask_collate_fn(transform, num_views, augment)` — стохастическую
(маскирование + хук аугментаций). Нормализация рёбер (`EdgeNorm`) и структурные PE
(`ConcatStructuralPE`, Laplacian/Centrality/RandomWalk) **удалены**: дистанции
теперь нормализуются per-graph в модели (`linear_edge_weighting`), а от PE
отказались.

**Макро-метрики** (`ThesisMacroMetrics`, тензор `[1,7]`): средний размер
компоненты связности, средняя внутрикластерная дистанция, modularity (Louvain),
clustering coefficient, число узлов, число рёбер, плотность. Считаются через
`networkx` — **дорого на CPU**, поэтому по умолчанию выключены (`use_macro: false`).

---

## 4. Модели (`src/models/`)

### 4.1 Энкодеры (`encoder.py`)

Два варианта графовых энкодеров, оба с **residual-связями + RMSNorm** для борьбы
с oversmoothing (идея из «Residual Connections and Normalization Can Provably
Prevent Oversmoothing in GNNs», 2025):

- **`GraphGcnEncoder`** — стек `GraphGCNResNorm` (GCNConv + residual + `BatchRmsNorm`).
- **`GraphGinEncoder`** — стек `GraphGINResNorm` (кастомный `WeightedGINConv` с
  поддержкой весов рёбер + обучаемый `alpha` на residual). **Используется по
  умолчанию** в `config.yaml`.

Особенности:
- `self.proj` — входная линейная проекция `in_channels → out_channels`,
  **заморожена** (`requires_grad_(False)`) — фактически случайная фиксированная
  проекция признаков.
- `BatchRmsNorm` — кастомная RMS-нормализация по последней оси с обучаемыми
  `gamma/beta`.
- `linear_edge_weighting(batch)` — min-max нормализация `edge_attr` per-graph,
  применяется в `JepaLight._apply_linear_weights` и в `GraphLatent`.

### 4.2 `GraphLatent` (encoder.py)

Обёртка для **инференса**: энкодер → pooling по графу → (опц.) конкатенация
нормализованных макро-метрик. Возвращает граф-уровневый эмбеддинг. Используется
в `embedding_pipeline.py`.

### 4.3 JEPA / LeJEPA (`jepa.py`)

- **`CrossAttentionPredictor`** — предиктор: query = позиции target-узлов
  (`mask_token + pos_embed`), key/value = эмбеддинги+позиции context-узлов.
  Cross-attention → residual → MLP → LayerNorm. Предсказывает латенты target из
  контекста. Позиции **абсолютные** (линейный `pos_embed`), а внимание считается
  по **всему батчу** (target формально аттендится ко всем context в батче, не
  только своего графа).

> ℹ️ **Возможная альтернатива предиктора (опробована, без улучшений).** Можно
> сделать предиктор на **относительных** позициях (`target_pos − context_pos`
> как аддитивный bias к вниманию) и ограничить внимание **внутри графа** (по
> `batch`-векторам, реализация по парам через `scatter`/`softmax`, без плотной
> `[Nt×Nc]` матрицы — масштабируется). Мотивация: инвариантность к системе
> координат + устранение межграфового внимания. **Первые эксперименты на h01
> улучшений не показали** (loss/downstream — на уровне или хуже абсолютной
> версии), поэтому в коде оставлена абсолютная версия. Вариант стоит держать как
> кандидата для будущих абляций (особенно вместе с устранением межграфового
> внимания как отдельным фактором).

- **`LeJEPA`** — основная модель (`config.yaml` → `network._target_`).
  Loss = `(1-λ)·MSE(pred, target_enc) + λ·SIGReg`, где **`sigreg`** —
  регуляризатор изотропии через sliced characteristic functions (из
  [LeJEPA, arXiv:2511.08544](https://arxiv.org/abs/2511.08544)), стабилизирует
  обучение и борется с **representation collapse**.

- **`JepaLight`** — LightningModule-обёртка: `_shared_step` применяет
  `linear_edge_weighting` к context/target, считает loss, логирует. Оптимайзер и
  scheduler инстанцируются из конфига через `getattr(optim, ...)`.

> **EMA не используется.** Context и target кодируются **одним общим** энкодером
> `self.encoder`; стабильность обеспечивает SIGReg, а не teacher-network. Старый
> код teacher-student / EMA удалён (метод `_ema`, вызов в `on_train_batch_end`,
> параметр `ema` в конфиге).

### 4.4 Загрузчик (`loader_model.py`)

`load_encoder_from_folder(folder)` — берёт последний `.ckpt` по mtime + `hparams.yaml`,
инстанцирует сеть, грузит веса в `JepaLight`, возвращает `.encoder` (или
`.student_encoder`, если есть). Используется во всех инференс-скриптах.

---

## 5. Конфигурация (Hydra + OmegaConf)

`configs/config.yaml` — корневой, через `defaults` подмешивает пути датасетов
(`9009`, `minnie65`, `human_descriptors`). Основные секции:

| Секция | Содержит |
|---|---|
| `network` | `_target_` модели (LeJEPA), энкодер, предиктор, `lambd`, `num_slices` |
| `training` | lr, weight_decay, max_epochs, optimizer, scheduler |
| `trainer` | параметры Lightning `Trainer` (gpu, devices, ...) |
| `datamodule` | путь датасета (`dataset.path`, один — без `prepare`), `stats_path`, `batch_size`, `num_workers`, `r` (радиус графа), `use_macro`, `num_views`, `mask_ratio`, `ratio` (split) |
| `classifier` | linear-probe: пути чекпоинтов, `num_classes`, CV (`n_splits`, `n_repeats` → 95% CI), pooling |

> ⚠️ Все пути в `*.yaml` — **абсолютные и захардкожены** под конкретную машину
> (`/home/eugen/...`). Для переносимости их стоит вынести в переменные
> окружения / относительные пути. См. §8.

> ℹ️ `network.encoder.in_channels` зависит от используемого набора дескрипторов:
> ~100 для сферических (spherical harmonics), ~31 для классических
> морфологических. Значение нужно выставлять под конкретный датасет. (Структурные
> PE убраны, так что в размерность они больше не входят.)

---

## 6. Пайплайны (точки входа)

### 6.1 Обучение JEPA

```bash
python -m src.cli.train.train_model
```
(README указывает `src.cli.train_model` — **устаревший путь**, актуальный —
`src.cli.train.train_model`.)

Поток: `get_datamodule` (канонизация на лету + RAM-кэш, без диск-шага) →
инстанцирование `network` → `JepaLight` → `Trainer.fit`. Чекпоинты + TensorBoard
логи в `lightning_logs/main_train/`. Выбор лучшего по `val_loss`.

### 6.2 Мульти-степ эксперимент

```bash
python -m src.experiment.train_val.multi_step_experiment
```
Обучает серию моделей с разным `max_epochs` (сейчас `_EPOCH_STEPS = [1, 200]`),
складывает чекпоинты в `src/experiment/train_val/checkpoints/epXXX/`.

### 6.3 Извлечение эмбеддингов

```bash
python -m src.cli.embedding_pipeline
```
`load_encoder_from_folder` → `GraphLatent` (encoder + pooling + макро) →
`EmbeddingExtractor.extract_from_graph_dataset` → опц. `pool_by_segment`
(агрегация шипиков в нейрон) → сохранение `EmbeddingSet` (`.pt`).
Уровни pooling: `graph` (на фрагмент) или `neuron` (по `segment_id`).

### 6.4 Оценка через linear probe (CV)

```bash
python -m src.cli.inference.d9009.evaluate_encoder_cv       # ab/wt мыши (2 класса)
python -m src.cli.inference.minnie65.evaluate_encoder_cv    # типы клеток (exc/inh)
python -m src.cli.inference.human_age.evaluate_encoder_cv   # человеческие шипики
```
Извлекает эмбеддинги замороженным энкодером → `train_cv` (**repeated** Stratified
K-fold: `n_repeats × n_splits` прогонов, фиксированные сиды → возможно paired-
сравнение) → `EmbeddingsLightModule` (голова через `classifier_factory`: линейная
по умолчанию / MLP для human_age) → метрики Accuracy/F1 (per-class + macro).
Сводка `summarize_cv`: mean ± std **+ bootstrap 95% CI**.

> Замечание: `train_cv`/`summarize_cv` теперь в одном месте
> (`embedding_pipeline.py`), дубликат из human_age удалён. Остаётся дублирование
> `LinearClassifier` в нескольких файлах с разными «головами» (линейная vs MLP) —
> частично снято через `classifier_factory`. См. §8.

### 6.5 Объяснимость (`src/explainer/9009/`)

Все скрипты через `setup_explainer_environment` собирают энкодер + классификатор +
макро-статистики, оборачивают в `GraphExplainerWrapper` (склеивает x с
broadcast'нутыми глобальными фичами, применяет exp-ядро к рёбрам с `sigma`):

- **`explainer.py`** — GNNExplainer на одном сэмпле → 3D-визуализация важности
  узлов/рёбер на меше (`DendriteVisualizer`, plotly HTML).
- **`global_explainer.py`** — агрегированная важность признаков по выборке.
- **`integrated_grad.py`** — Integrated Gradients по признакам.
- **`contr_fact.py`** — gradient saliency / sensitivity analysis.

> ℹ️ `integrated_grad.py`, `contr_fact.py`, `global_explainer.py` импортируют
> `from paper_plots import plot_feature_importance` — `paper_plots` это **личный
> пакет автора**, установленный в окружении (не часть репозитория). Его нужно
> установить отдельно, чтобы запускать эти скрипты. См. §8.

### 6.6 Unsupervised-метрики качества пространства (`src/representation/estimators.py`)

Набор эстиматоров над матрицей эмбеддингов (вход — dict с `embeddings`/`labels`):

| Эстиматор | Метрика | Нужны метки? |
|---|---|---|
| `RankMeEstimator` | RankMe (эффективный ранг), condition number, nuclear norm | нет |
| `IsotropyEstimator` | isotropy, effective_dim, explained variance (PCA) | нет |
| `UniformityEstimator` | uniformity на гиперсфере (Wang & Isola) | нет |
| `AlignmentEstimator` | alignment позитивных пар | да |
| `ClusterQualityEstimator` | silhouette, davies-bouldin | да |
| `RecallAtKEstimator` | Recall@K | да |
| `CompositeEstimator` | объединяет несколько | — |

Это **intrinsic-геометрия** латентного пространства без привязки к домену.
Полезно как дешёвый детектор **collapse** в абляциях (упал RankMe/effective_dim =
схлопывание), но **не** как «полезность латента» в смысле доменных знаний — для
этого нужен анализ кластеров/направлений в терминах биологии (типы шипиков,
патология). Модуль рассматривается к удалению, если роль детектора коллапса не
нужна.

---

## 7. Датасеты

| Датасет | Конфиг | Задача | get_class |
|---|---|---|---|
| **Minnie65** | `minnie65.yaml` | тип клетки (exc=0 / inh=1) по `cell_type` из CSV | `minnie65_get_class.py` |
| **9009** | `9009.yaml` | ab (0) / wt (1) — генотип мыши, по имени папки | `get_class_9009` |
| **human spines** | `human_descriptors.yaml` | возрастные группы человека | (в inference-скрипте) |

Сегментация мешей делалась через [NEURD](https://github.com/reimerlab/NEURD).
`segment_id` извлекается из имени файла / поля графа и используется для
neuron-уровневого pooling.

---

## 8. Известные проблемы / технический долг

Сгруппировано по приоритету для будущей чистки (см. также комментарии «⚠️» выше).

### Функциональные расхождения
1. ✅ **EMA-артефакты удалены** (метод `_ema`, вызов в `on_train_batch_end`,
   параметр `ema` в конфиге). README приведён в соответствие с кодом (LeJEPA, один
   энкодер, корректная команда запуска).
2. **Внешняя зависимость `paper_plots`** — три explainer-скрипта импортируют
   `from paper_plots import plot_feature_importance`. Это личный пакет автора,
   установленный в окружении (не в репозитории). Стоит зафиксировать его в файле
   зависимостей (пункт 7) или сделать импорт опциональным, чтобы проект
   запускался у других без него.

3. ✅ **Пайплайн данных упрощён** (эта сессия): убран диск-шаг `preprocess_dataset`
   (канонизация на лету + RAM-кэш), удалены структурные PE (`ConcatStructuralPE`)
   и `EdgeNorm`; добавлен `BuildGeometricGraph` (единообразие из `pos`); macro за
   флагом `use_macro`; collate с `num_views` и хуком аугментаций. Data-pipeline
   тесты приведены к новому API и сведены в одно дерево (`tests/`).

### Переносимость / гигиена
5. **Захардкоженные абсолютные пути** во всех конфигах и в `main()` ряда
   скриптов (`embedding_pipeline.py`, `explainer.py`). Вынести в env / Hydra
   overrides / относительные пути.
6. **Дублирование `LinearClassifier`** в нескольких файлах (линейная vs MLP
   голова). `train_cv`/`summarize_cv` уже сведены в `embedding_pipeline.py`,
   дубликат из human_age удалён; голова параметризуется `classifier_factory`.
   Остаётся свести определения самого `LinearClassifier`.
7. **Нет файла зависимостей** (`requirements.txt` / `environment.yml` /
   `pyproject.toml`). Окружение восстанавливается только из упоминания
   `conda activate torch_5060`. Стоит зафиксировать (torch, torch_geometric,
   torch_cluster, pytorch_lightning, hydra, omegaconf, networkx, sklearn, plotly,
   trimesh, torchmetrics, pandas, scipy).
8. **`GraphLatent.forward`** содержит «висячую» строку `self.encoder` (без
   эффекта) внутри `torch.no_grad()` — артефакт, можно убрать.

### Производительность
9. **`ThesisMacroMetrics` на networkx** (Louvain, shortest paths) — дорого на CPU.
   Теперь считается на лету (не на диск) и **по умолчанию выключено**
   (`use_macro: false`); при включении на больших датасетах — кандидат на
   векторизацию / параллелизм, особенно без RAM-кэша.

---

## 9. Тесты

Тесты есть на трёх уровнях (запуск: `pytest`):
- `tests/` — интеграция пайплайна данных (`test_data_pipeline.py`, `test_data/`).
- `src/data_utils/tests/` — трансформы, статистики, datamodule.
- `src/models/test_models/` — энкодеры, JEPA, загрузчик.
- `src/explainer/tests/` — wrapper, GNNExplainer, IG, counterfactual, визуалы.

> Стоит проверить, что все тесты зелёные на `main` (особенно explainer — из-за
> отсутствующего `paper_plots`).

---

## 10. Возможные направления развития

Не предписание, а варианты — для выбора «куда надёжнее двигаться»:

1. **Стабилизация фундамента (рекомендую первым).** Закрыть техдолг из §8:
   зафиксировать зависимости, починить пути, удалить мёртвый код, привести README
   в соответствие. Это сделает все эксперименты воспроизводимыми.
2. **EMA / teacher-student.** Реализовать полноценный I-JEPA-style EMA-таргет и
   сравнить с текущим SIGReg-подходом на метриках collapse (RankMe, isotropy).
3. **Сравнение с реконструктивными методами** (VGAE / Graph-MAE) — упомянуто в
   README как «under consideration». Дало бы baseline для статьи.
4. **Расширение оценки.** Систематизировать прогон `estimators.py` по чекпоинтам
   (RankMe / uniformity по эпохам) — связать с linear-probe accuracy.
5. **Препроцессинг в репозиторий.** Включить (хотя бы документально) шаг
   «меш → граф», чтобы проект был самодостаточным.
6. **Объяснимость → биология.** Довести explainer-пайплайн до выводов о том,
   какие геометрические признаки шипиков различают классы (ab/wt, типы клеток).

---

## 11. Связанные материалы

- LeJEPA: https://arxiv.org/abs/2511.08544
- NEURD (сегментация): https://github.com/reimerlab/NEURD
- Репозиторий: https://github.com/NEurosgen/GraphJepaSpines
- Ветки: `main` (стабильно, конфиг-обучение), `exp` (эксперименты + лог
  экспериментов в README ветки).
- **Черновик статьи** — [`NeuroGraph/main.tex`](../NeuroGraph/main.tex)
  (`references.bib`, рисунки рядом). Содержит постановку, метод (LeJEPA-SSL),
  результаты linear-probing vs PointNet++ / Spiking PointNet, разбор
  объяснимости и ограничений.

### Ключевые цифры из статьи (для справки)

Linear probing (замороженный энкодер + 1 линейный слой) против supervised-бейзлайнов:

| Модель | Minnie65 Binary Acc/F1 | Minnie65 Multi Acc/F1 | 9009 Acc/F1 |
|---|---|---|---|
| PointNet++ | 0.897 / 0.639 | 0.231 / 0.098 | 0.912 / 0.870 |
| Spiking PointNet | 0.872 / 0.466 | 0.256 / 0.123 | 0.867 / 0.820 |
| **Ours (LeJEPA)** | **0.986 / 0.947** | **0.425 / 0.531** | **0.912 / 0.912** |

Предобучение: `minnie65_public`, ~450 нейронов → ~90 000 графов, 200 эпох.
Датасет 9009: 61 меш / ~1200 шипиков (37 контроль + 24 патология, модель
болезни Альцгеймера).
```
