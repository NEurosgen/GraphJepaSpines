# NeuroGraph / GraphJEPA Spines: SSL for Dendritic Spine Representation

This repository contains an implementation of a **Self-Supervised Learning (SSL)**
method for learning informative latent representations of dendritic spines from
their geometry, **without relying on labeled data**.

A biologically-informed Graph Neural Network (GNN) encoder is pre-trained with an
SSL objective inspired by **LeJEPA**, then evaluated via *linear probing* on
downstream classification tasks. Linear evaluation of the frozen encoder
outperforms fully-supervised baselines trained from scratch (PointNet++, Spiking
PointNet).

> 📄 The accompanying paper draft lives in [`NeuroGraph/`](NeuroGraph/) (`main.tex`).
> A detailed architecture/code map is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Project Overview

The goal is to generate robust vector embeddings of dendritic spines and, via
pooling, of whole dendrites and neurons. **JEPA (Joint Embedding Predictive
Architecture)** is used as the core idea to avoid the expensive decoding step of
autoencoders (MAE, VGAE) and to focus on the semantics of the latent space —
critical given limited labeled data and high feature dimensionality.

Key contributions (see the paper for details):

* **Biologically-grounded SSL encoder** — a GCN/GIN architecture focusing on
  individual spine morphology while accounting for the spatial context of
  neighboring dendritic nodes.
* **Hierarchical representativeness** — pooling yields informative embeddings not
  only for individual spines but for macrostructures (dendrites, whole neurons).
* **Efficient transfer learning** — the unsupervised pre-trained encoder reaches
  high performance on downstream tasks with limited labeled data.
* **Interpretability** — the graph-convolutional architecture allows direct
  application of XAI methods (GNNExplainer, Integrated Gradients) to surface the
  biological patterns driving predictions.

## Data and Preprocessing

> **Note:** The pipeline for converting raw meshes into graph format (`.pt`) is
> **not included** in this repository. This project works with a pre-processed
> dataset.

### Data Source

* **Pre-training dataset:** `minnie65_public` (~450 neurons → ~90,000 dendritic graphs).
* **Segmentation:** performed with the [NEURD](https://github.com/reimerlab/NEURD) framework.
* **Spine descriptors:** rotation-invariant spherical-harmonic descriptors.
* **Object of study:** dendritic branches with pre-segmented spines.

### Graph Representation

Each input is a graph where:

* **Nodes** — individual spines, with a feature vector of geometric descriptors.
* **Edges** — spatial proximity; two spines are connected when their Euclidean
  distance is below a radius `r`. Edge attributes hold the distance.
* **Positions** (`data.pos`) — 3D coordinates, used by the predictor and for
  radius-graph construction.

![Example](docs/images/Screenshot%20from%202026-02-07%2011-50-08.png)

## Architecture and Methodology

This project applies the JEPA family of architectures to the graph modality. The
motivation for predictive (JEPA) over reconstructive (MAE, VGAE) methods is to
eliminate the decoder.

### LeJEPA (current architecture)

The implemented model (`src.models.jepa.LeJEPA`) follows the
[LeJEPA paper](https://arxiv.org/abs/2511.08544):

* **Encoder:** a 3-layer Graph Isomorphism Network (`GraphGinEncoder`) with
  skip-connections and RMS normalization (mitigates oversmoothing). A single
  shared encoder produces both context and target latents (**no EMA / teacher
  network** — earlier teacher-student/EMA code has been removed).
* **Masking:** random isolated target nodes are split from the context graph
  (`MaskData`). Random node masking provides an informative gradient and avoids
  the trivial averaged-vector solution observed with cluster masking.
* **Predictor:** a Cross-Attention module that predicts target latents from the
  context latents and the positional encoding of the masked nodes.
* **Loss:** `(1 - λ)·MSE(pred, target) + λ·SIGReg`, where **SIGReg** (sliced
  characteristic-function regularization) shapes the latent space toward an
  isotropic Gaussian and prevents **representation collapse**.

## Results (linear probing)

Frozen encoder + a single linear layer vs. fully-supervised baselines trained
from scratch (full 3D meshes/point clouds as input):

| Model | Minnie65 Binary Acc / F1 | Minnie65 Multiclass Acc / F1 | 9009 Acc / F1 |
|---|---|---|---|
| PointNet++ | 0.897 / 0.639 | 0.231 / 0.098 | 0.912 / 0.870 |
| Spiking PointNet | 0.872 / 0.466 | 0.256 / 0.123 | 0.867 / 0.820 |
| **Ours (LeJEPA)** | **0.986 / 0.947** | **0.425 / 0.531** | **0.912 / 0.912** |

Downstream tasks: binary inhibitory/excitatory and multiclass morphological-type
classification on `minnie65`; binary normal/pathological dendrite classification
on the in-house `9009` dataset (Alzheimer's modeling, 61 meshes / ~1,200 spines).

## Usage

Pre-train the encoder (Hydra config in [`configs/`](configs/)):

```bash
python -m src.cli.train.train_model
```

Evaluate a pre-trained encoder with linear probing (Stratified K-Fold CV):

```bash
python -m src.cli.inference.9009.evaluate_encoder_cv        # ab/wt mice (binary)
python -m src.cli.inference.minnie65.evaluate_encoder_cv    # cell types
python -m src.cli.inference.human_age.evaluate_encoder_cv   # human spines
```

Extract embeddings from a trained encoder:

```bash
python -m src.cli.embedding_pipeline
```

> Paths in `configs/*.yaml` are currently absolute and machine-specific — adjust
> them to your environment before running.

---

### Branching Policy

* **`main`**: Stable code, ready for configuration-based training.
* **`exp`**: Primary branch for tracking experiments; contains experiment-specific
  utilities, and its `README` logs all current experiments.
