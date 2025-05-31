
# Verifiable Federated Learning for Climate Risk Forecasting

This repository simulates a Federated Learning (FL) pipeline aligned with the Nexus Observatory Protocol (OP). The goal is to train decentralized AI models for flood and heatwave prediction using climate-related sensor data, while supporting trustless, verifiable computation.

## 🔍 Project Overview

- **Local FL Training:** Simulated client-side model training using weather and sensor data.
- **Model Aggregation:** Server-side federated averaging (FedAvg) to build a global model.
- **Verifiability:** Hash-based simulation of zero-knowledge proofs for model auditability.
- **Alignment:** Fully aligned with Nexus OP's architecture and mission for DRR, DRI, and DRF.

## 📁 Repository Structure

```
nexus-fl-risk-prediction/
├── data/
│   └── sample_input.csv
├── models/
│   └── federated_model.py
├── scripts/
│   ├── train_local.py
│   └── aggregate.py
├── proofs/
│   └── zk_simulation.py
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 How to Run

1. Train the local model:
```bash
python scripts/train_local.py
```

2. Aggregate client models:
```bash
python scripts/aggregate.py
```

3. Generate a simulated ZK proof:
```bash
python proofs/zk_simulation.py
```

## 📦 Requirements

See `requirements.txt` for a list of Python dependencies.

## 📜 License

For pilot demonstration purposes with the Global Centre for Risk and Innovation (GCRI).
