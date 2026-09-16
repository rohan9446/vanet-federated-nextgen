# RSU-Aware Federated Learning for V2X Misbehavior Detection on VeReMi NextGen

This repository contains the code and experimental pipeline for our paper:

> **RSU-Aware Federated Learning for V2X Misbehavior Detection: An Evaluation on VeReMi NextGen**
> Bandaru Rohan Satya Balaji, Sreenivasa Chakravarthi Sangapu
> *Submitted to Journal of Data Science and Intelligent Systems (JDSIS)*

## Overview

We evaluate federated learning for misbehavior detection in VANETs using the [VeReMi NextGen](https://veremi-dataset.github.io/) dataset (Hermann et al., IEEE VNC 2026). The pipeline addresses the dataset's multi-receiver log structure, engineers 12 domain-specific features, and evaluates federated ensemble (XGBoost) and federated averaging (LSTM, BiLSTM) under RSU-aware spatial partitioning.

### Key Results

| Setting | Model | AUC | F1 |
|---------|-------|-----|-----|
| Centralized | XGBoost | 0.954 | 0.893 |
| Federated (k=4) | XGBoost Ensemble | 0.949 | 0.899 |
| Federated (k=4) | BiLSTM FedAvg | 0.924 | 0.779 |

Federated XGBoost retains **99.4%** of centralized AUC at k=4 partitions.

## Repository Structure

```
├── preprocessing/
│   └── preprocess.py          # Data pipeline: receiver aggregation → dedup → features
├── models/
│   ├── centralized/
│   │   └── train_baselines.py # RF, ET, XGBoost, LSTM, BiLSTM baselines
│   └── federated/
│       ├── run_federated.py   # FedAvg / FedProx for DL models
│       └── fed_xgb.py         # Flower FedXgbBagging for XGBoost
├── evaluation/
│   ├── per_attack_analysis.py # Per-attack binary detection (14 attacks)
│   └── analyze_weak_attacks.py# Feature-level analysis of weak attacks
├── generate_figures.py        # All paper figures (PDF + PNG)
├── repartition.py             # K-Means RSU partitioning (k=4,8,16)
├── configs/                   # Hyperparameter configs
├── results/                   # Experiment outputs (JSON + CSV)
├── figures/                   # Generated figures
└── README.md
```

## Data Pipeline

VeReMi NextGen records each BSM once per receiving vehicle (~14.8× inflation). Our pipeline:

1. **Receiver aggregation** — compute `mean_sender_receiver_dist`, `n_receivers`, `tx_delay` per unique message
2. **Deduplication** — collapse to one row per `(sender_id, messageID)`
3. **Sequential features** — `speed_consistency`, `accel_consistency`, `heading_trajectory_consistency`, etc.
4. **Downsampling** — sender-level removal to ~25% attack rate
5. **K-Means partitioning** — RSU-aware splits at k = 4, 8, 16

## Features (12 classification + 2 partition-only)

| Category | Features |
|----------|----------|
| Raw | `sender_spd`, `sender_acl` |
| Sequential | `time_delta`, `distance`, `heading_change`, `speed_consistency`, `accel_consistency`, `heading_trajectory_consistency` |
| Receiver | `tx_delay`, `mean_sender_receiver_dist`, `n_receivers` |
| Context | `sender_dist_to_road_edge` |

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/vanet-federated-nextgen.git
cd vanet-federated-nextgen

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn xgboost torch flwr matplotlib pyarrow
```

## Usage

### 1. Download VeReMi NextGen

Download the dataset from [veremi-dataset.github.io](https://veremi-dataset.github.io/) and place the per-attack parquet files under `data/per_attack/{train,val,test}/`.

### 2. Preprocess

```bash
python preprocessing/preprocess.py --data_dir data/per_attack --output_dir data/processed
```

### 3. Create RSU Partitions

```bash
python repartition.py --data_dir data/processed --clusters 4 8 16
```

### 4. Train Centralized Baselines

```bash
python models/centralized/train_baselines.py --data_dir data/processed --output_dir results/centralized
```

### 5. Run Federated Experiments

```bash
# DL models (FedAvg / FedProx)
python models/federated/run_federated.py --data_dir data/processed --output_dir results/federated

# XGBoost (Flower FedXgbBagging)
python models/federated/fed_xgb.py --data_dir data/processed --output_dir results/fl_xgb
```

### 6. Per-Attack Analysis

```bash
python evaluation/per_attack_analysis.py --data_dir data/processed --output_dir results/per_attack
```

### 7. Generate Figures

```bash
python generate_figures.py --data_dir data/processed --output_dir figures/
```

## Classification Schemes

- **Binary** — Normal vs Attack
- **Tri-class** — Normal / Single-parameter / Multi-parameter
- **7-class** — Normal / Position / Speed / Heading / Acceleration / Time / Multi

## Environment

- Python 3.11
- PyTorch 2.x
- XGBoost 2.x
- Flower (flwr) 1.31.0
- scikit-learn 1.x
- Tested on SDSU HPC: 1× L40 GPU, 24 CPUs, 32 GB RAM

## Citation

If you find this code useful, please cite this repository:

https://github.com/YOUR_USERNAME/vanet-federated-nextgen

Paper citation will be added upon publication.

## License

This project is for academic research purposes. Please contact the authors for commercial use.
