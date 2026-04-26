# 🔐 Federated Learning for Vulnerability Detection

**Master's Thesis Project — University of Messina, Data Science**

A noise-robust federated learning framework for software vulnerability detection using [Flower FL](https://flower.ai/), [CodeBERT](https://github.com/microsoft/CodeBERT), and the [DiverseVul](https://github.com/wagner-group/diversevul) dataset.

---

## 🧩 Overview

This project investigates whether federated learning can effectively train vulnerability detection models across distributed clients — even in the presence of **label noise** and **non-IID data distributions**.

| Component | Choice |
|---|---|
| FL Framework | [Flower (flwr)](https://flower.ai/) |
| Model | `microsoft/codebert-base` |
| Dataset | DiverseVul (~350K functions, 150 CWE types) |
| Noise Methods | FedCorr, FedLN, baseline FedAvg/FedProx |
| Languages | Python 3.10+ |

---

## 📁 Repository Structure

```
fl-vuln-detect/
├── src/
│   ├── client/           # Flower client logic, local training, dataset loading
│   ├── server/           # Flower server, aggregation strategies
│   ├── models/           # CodeBERT encoder + classification head
│   ├── noise/            # Noise injection, detection, and label correction
│   └── utils/            # Metrics, data utilities, partitioning, logging
├── configs/              # YAML configs for each experiment
├── experiments/
│   ├── baselines/        # FedAvg, FedProx, Centralized
│   └── noise_robust/     # FedCorr, FedLN, ablation studies
├── data/
│   ├── raw/              # Raw DiverseVul JSON (not committed)
│   ├── processed/        # Tokenized, cleaned samples
│   └── partitions/       # Per-client data splits (IID / non-IID)
├── scripts/              # Shell scripts for data prep and experiment runs
├── notebooks/            # EDA, noise analysis, result visualization
├── tests/                # Unit tests
└── results/              # Logs, checkpoints, plots (not committed)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/fl-vuln-detect.git
cd fl-vuln-detect
pip install -e ".[dev]"
```

### 2. Prepare Data

```bash
bash scripts/prepare_data.sh
python src/utils/partition.py --n_clients 10 --alpha 0.5
```

### 3. Run Baseline (FedAvg)

```bash
python experiments/baselines/run_fedavg.py --config configs/fedavg.yaml
```

### 4. Run Noise-Robust Experiment (FedCorr)

```bash
python experiments/noise_robust/run_fedcorr.py --config configs/noise_robust.yaml \
    --noise_rate 0.3 --noise_type symmetric
```

---

## 🔬 Experiments

### Baselines

| Experiment | Script | Description |
|---|---|---|
| Centralized | `experiments/baselines/run_centralized.py` | Single-node CodeBERT fine-tuning |
| FedAvg | `experiments/baselines/run_fedavg.py` | Standard FL baseline |
| FedProx | `experiments/baselines/run_fedprox.py` | Proximal regularization for heterogeneous clients |

### Noise-Robust Methods

| Experiment | Script | Reference |
|---|---|---|
| FedCorr | `experiments/noise_robust/run_fedcorr.py` | Xu et al., CVPR 2022 |
| FedLN | `experiments/noise_robust/run_fedln.py` | Tsouvalas et al., TIST 2024 |
| Ablation | `experiments/noise_robust/run_ablation.py` | Component-level analysis |

### Noise Settings

- **Noise types:** Symmetric, Asymmetric, Instance-dependent
- **Noise rates:** 0%, 10%, 20%, 30%, 40%
- **Noisy client fraction:** 20%, 50%, 80%
- **Data distribution:** IID, Dirichlet non-IID (α = 0.1, 0.5, 1.0)

---

## 📊 Evaluation Metrics

- **F1-score** (primary — handles class imbalance in DiverseVul)
- Precision, Recall, Accuracy
- Per-CWE category breakdown
- Communication rounds to convergence

---

## 🗂️ Key Papers

- **DiverseVul** — Chen et al., 2023
- **FedCorr** — Xu et al., CVPR 2022 — Multi-stage label noise correction in FL
- **FedLN** — Tsouvalas et al., ACM TIST 2024 — Energy/embedding-based noise detection in FL
- **FedProx** — Li et al., MLSys 2020
- **CodeBERT** — Feng et al., EMNLP 2020

---

## 📝 Citation

```bibtex
@mastersthesis{abdumutalliev2025flvuln,
  author  = {Begzod Abdumutalliev},
  title   = {Federated Learning for Vulnerability Detection with Noisy Labels},
  school  = {Università degli Studi di Messina},
  year    = {2025}
}
```

---

## 🤝 Acknowledgements

Supervised at the University of Messina, Department of Data Science.
