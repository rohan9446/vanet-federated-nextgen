# Adaptive Anomaly Detection in VANETs

Federated Learning-based anomaly detection for Vehicular Ad hoc Networks (VANETs) using the VeReMi Extension dataset.

## Overview

This repository implements centralized and federated machine learning models for detecting anomalies in VANET communications. The key contribution is **FedXGB** (Federated XGBoost with cyclic training), extended with **FedLSTM**, **FedBiLSTM**, and **FedCNN** using the Flower framework.

### Models

| Model | Type | Strategy |
|-------|------|----------|
| Random Forest | Centralized | — |
| Extra Trees | Centralized | — |
| XGBoost | Centralized | — |
| LSTM | Centralized | — |
| BiLSTM | Centralized | — |
| CNN | Centralized | — |
| FedXGB | Federated | Cyclic / Bagging |
| FedLSTM | Federated | Cyclic / FedAvg |
| FedBiLSTM | Federated | Cyclic / FedAvg |
| FedCNN | Federated | Cyclic / FedAvg |

## Project Structure

```
vanet-anomaly-detection/
├── configs/config.yaml            # All hyperparameters
├── preprocessing/
│   ├── feature_engineering.py     # Distance, speed, acceleration, heading
│   ├── feature_selection.py       # Training feature set
│   ├── rsu_partitioning.py        # K-Means and grid-based RSU splits
│   └── balancing.py               # SMOTETomek per partition
├── models/
│   ├── centralized/train_models.py
│   └── federated/
│       ├── fed_xgb.py             # FedXGB cyclic + bagging
│       └── fed_dl.py              # FedLSTM, FedBiLSTM, FedCNN
├── evaluation/metrics.py          # AUC, Precision, Recall, F1
├── run_pipeline.py                # Main entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone repo
git clone <repo-url>
cd vanet-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# For GPU (HPC cluster)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Dataset

Download the VeReMi Extension dataset (refined CSV version by Slama et al.) and place it at:
```
data/VeReMi_Extension.csv
```

Source: https://data.mendeley.com/datasets/k62n4z9gdz/1

## Usage

```bash
# Run everything (centralized + federated)
python run_pipeline.py --config configs/config.yaml --mode all

# Only centralized baselines
python run_pipeline.py --config configs/config.yaml --mode centralized

# Only federated models
python run_pipeline.py --config configs/config.yaml --mode federated

# Specific federated model
python run_pipeline.py --mode federated --fed-model fed_lstm
python run_pipeline.py --mode federated --fed-model fed_cnn
```

### HPC Cluster (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=vanet-fl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module load python/3.11 cuda/12.1
source venv/bin/activate
python run_pipeline.py --mode all
```

## Configuration

All hyperparameters are in `configs/config.yaml`. Key settings:

- `rsu_partitioning.method`: `"kmeans"` or `"grid"`
- `federated.strategy`: `"cyclic"` or `"fedavg"`
- `federated.num_rounds`: Number of FL rounds
- `models.centralized.xgboost.tree_method`: `"hist"` (CPU) or `"gpu_hist"` (GPU)

## Results

Results are saved to `results/metrics.json` after each run.

## Citation

```bibtex
@inproceedings{chakravarthi2025adaptive,
  title={Adaptive Anomaly Detection in VANETs: Leveraging Federated Learning for Privacy and Performance},
  author={Sangapu, Sreenivasa Chakravarthi and others},
  year={2025}
}
```

## License

MIT
