# Evaluating Graph Neural Networks

Code link for paper titled "Position: Metric Illusion-Restoring Contextual Utility to Graph Learning". The codebase is built upon the foundational work of the **tunedGNN paper** ([LUOyk1999/tunedGNN](https://github.com/LUOyk1999/tunedGNN)) and has been significantly enhanced with additional models, a unified experimental pipeline, and detailed result logging.

## 📜 Overview

This project is organized into two primary tasks:

* **Node Classification**: Scripts and models for predicting the labels of individual nodes within a graph. This includes standard GNNs like GCN and GAT, as well as more advanced models like FSGCN and Graph transformers.
* **Graph Classification**: Scripts and models for predicting the label of an entire graph. This includes standard GNNs with pooling layers and advanced Graph Transformer models like GPS and Subgraphormer.

The framework is designed to be modular, enabling straightforward experimentation with different models, hyperparameters, and datasets.

---
Position: Metric Illusion-Restoring Contextual Utility to Graph Learning

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* PyTorch Geometric
* OGB (Open Graph Benchmark)
* scikit-learn
* torcheval

<!-- ### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ncometa/eval_gnn.git
    cd eval_gnn
    ``` -->

1.  **Install dependencies:**
    It is highly recommended to use a virtual environment (e.g., `conda` or `venv`).

    ```bash
    # Install PyTorch and PyG (adjust for your CUDA version if necessary)
    pip install torch torchvision torchaudio
    pip install torch-geometric

    # Install other required packages
    pip install ogb scikit-learn torcheval gdown tqdm
    ```

---

## 🔬 How to Run Experiments

This repository uses shell scripts to automate hyperparameter tuning and evaluation.

### Node Classification

You can obtain the results by running the dedicated bash files:

```bash
bash node_classification/run_gnn.sh
```

### Graph Classification
You can obtain the results by running the dedicated bash files:

```bash
bash graph_classification/run_gnn.sh
```