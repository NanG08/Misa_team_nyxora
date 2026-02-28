# MISA × AMD — Hardware & Software Integration

> MISA's entire AI/ML pipeline is built to run on AMD hardware and software with no code changes required. This document covers what AMD products are used, why, and how to get started.

---

## Overview

MISA processes millions of animal GPS tracking records, trains deep learning models on decades of migration data, and runs real-time inference to serve route predictions and story cards to the public. Each of these workloads maps cleanly onto a specific AMD product.

---

## Hardware Map

| MISA Component | AMD Product | Why |
|---|---|---|
| LSTM Route Predictor (Model 1) | **Instinct MI300X / MI325X** | Memory-bandwidth-heavy sequential training. 6 TB/s HBM3e bandwidth keeps long GPS sequences in-flight |
| LSTM Autoencoder (Model 3) | **Instinct MI300X** | Batch autoencoder training across hundreds of migration seasons |
| RAG + LLM Story Layer (Model 4) | **Instinct MI355X** | 288GB HBM3E supports 520B+ parameter models on a single card for local LLM hosting |
| Habitat Scorer + Data Pipeline (Model 2) | **EPYC 9004 Series** | XGBoost and geospatial preprocessing are CPU-bound, large-memory workloads — EPYC's strength |

---

## Software Stack (ROCm)

MISA uses the full AMD ROCm open-source stack. Since ROCm exposes AMD GPUs as a CUDA-compatible interface, all PyTorch model code runs without modification.

| ROCm Library | Role in MISA |
|---|---|
| **MIOpen** | Accelerates LSTM and Autoencoder neural network operations (Models 1 & 3) |
| **rocBLAS** | Matrix operations for XGBoost gradient boosting (Model 2) |
| **AMD Quark** | Model quantisation (FP8/FP4) for faster inference with no accuracy loss |
| **RCCL** | Multi-GPU communication for scaling training across multiple Instinct cards |
| **AMD Infinity Hub** | Pre-built PyTorch + ROCm Docker containers — zero environment setup |

---

## Switching from CUDA to ROCm

The entire MISA codebase works on AMD with a single line change at install time. Nothing else changes.

```bash
# NVIDIA CUDA (default)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# AMD ROCm — all model code stays identical
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

Verify your AMD GPU is detected:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# → AMD Instinct MI300X
```

For XGBoost (Model 2), the ROCm backend is enabled with the same flag:

```python
# Works on both NVIDIA and AMD
params = {'device': 'cuda', 'tree_method': 'hist'}
```

---

## Docker (AMD Infinity Hub)

Pull a pre-built container with PyTorch and ROCm already configured:

```bash
docker pull rocm/pytorch:latest
docker run -it --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest
```

---

## Prospective Usage

As MISA scales, here is how the AMD stack grows with it.

**Near-term — Training at scale**
Training on the full Movebank dataset (850+ species, 30+ years of GPS records) will require sustained GPU throughput. The Instinct MI300X cluster handles this with its unified memory architecture, keeping entire species datasets in HBM3e across multi-GPU nodes without CPU offloading.

**Mid-term — Local LLM hosting**
Currently MISA's story generation layer calls the Anthropic Claude API externally. As the platform scales to real-time event processing, hosting an open-weight LLM locally on the Instinct MI355X eliminates API latency and per-call cost for high-volume story generation — a single MI355X can serve a 70B model continuously at production traffic.

**Long-term — Edge and citizen science**
MISA's roadmap includes a citizen science contribution layer where researchers in the field submit tracking data. AMD's Ryzen AI NPU in edge devices could run lightweight anomaly detection locally before syncing to the main pipeline, enabling offline-first field use in remote migration zones.

---

## Free Access — AMD Developer Cloud

Prototype and train on real Instinct hardware for free before any hardware purchase.

```
https://www.amd.com/en/developer/resources/developer-cloud.html
```

Sign up for free credits, spin up an MI300X instance, and run:

```bash
git clone https://github.com/your-username/misa.git
cd misa && pip install -r ml/requirements.txt
python ml/pipeline.py
```

---

## Summary

AMD's hardware and software stack covers every layer of MISA — from training deep learning models on Instinct GPUs, to running tabular ML on EPYC CPUs, to hosting LLMs locally at scale. The ROCm ecosystem means the codebase stays clean and portable, with no vendor lock-in and a clear path from free-tier prototyping on the AMD Developer Cloud to full production deployment on Instinct infrastructure.