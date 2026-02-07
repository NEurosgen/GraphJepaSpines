# GraphJEPA Spines: SSL for Dendritic Spine Representation

This repository contains an implementation of Self-Supervised Learning (SSL) methods designed to learn informative latent spaces based on the geometric data of dendritic spines.

## Project Overview

The goal of this project is to generate robust vector embeddings of dendritic spines without relying on labeled data. **JEPA (Joint Embedding Predictive Architecture)** was selected as the core architecture to avoid the computationally expensive decoding step typical of autoencoders and to focus on the semantics of the latent space.

## Data and Preprocessing

> **Note:** The pipeline for converting raw meshes into graph format (`.pt`) is not included in this repository. This project utilizes a pre-processed dataset.

### Data Source

* **Dataset:** Minnie65_public.
* **Segmentation:** Performed using the [NEURD](https://www.google.com/search?q=https://github.com/cajal/NEURD) framework.
* **Object of Study:** Dendritic branches with pre-segmented spines.

### Graph Representation

The input data consists of a fully connected graph , where:

* **Nodes ():** Each node corresponds to an individual spine and contains a vector of its geometric features.
* **Edges ():** Edges encode spatial relationships between spines. Edge attributes contain the Euclidean distance between corresponding nodes.

## Architecture and Methodology

This project explores the applicability of JEPA-family architectures to graph modalities. The primary motivation for choosing predictive methods (JEPA) over reconstructive methods (MAE, VGAE) is to eliminate the need for a decoder, which is critical given the limited training data and high dimensionality of features.

*Currently, a comparative analysis with VGAE architectures is also under consideration.*

### Implemented Approaches

The codebase includes two architectural variations:

#### 1. Graph JEPA (Baseline)

A direct adaptation of JEPA principles for graphs.

* **Mechanism:** Utilizes a Teacher-Student architecture. The input graph is masked (random subgraph sampling). The predictor's task is to reconstruct the latent representation of the masked target region based on the graph context and the positional encoding of the mask center.
* **Limitations:** Experiments revealed a tendency toward **representation collapse**, where the model converges to trivial constant solutions.

#### 2. LeJEPA (Logic-enhanced JEPA)

An enhanced version based on training stabilization methods.

* **Features:** Implements mechanisms proposed in the [LeJEPA paper](https://arxiv.org/abs/2511.08544).
* **Result:** This approach demonstrates significantly higher training stability and resistance to latent space collapse.

## Usage

To start training the model, use the following command:

```bash
python -m src.cli.train_model

```

---

### Branching Policy

* **`main`**: Contains stable code versions ready for configuration-based training.
* **`exp`**: The primary branch for tracking experiments. It contains code utilities specific to experimentation. Additionally, the `README` in this branch logs all currently existing experiments.