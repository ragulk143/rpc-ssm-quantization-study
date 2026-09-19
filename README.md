# Quantization Sensitivity in Recurrent State-Space Model Dynamics

## Overview

This repository contains the code, experimental logs, and research notes for an empirical investigation into quantization behavior in recurrent State-Space Model (SSM) dynamics.

The project explores whether quantization sensitivity is uniformly distributed across model components or whether recurrent pathways exhibit higher sensitivity than transient projection pathways. Experiments were conducted using a Mamba-inspired recurrent topology and profiled on an NVIDIA RTX 3050 6GB GPU.

---

## Research Question

**Can quantization-induced state divergence in recurrent State-Space Models be reduced through topology-aware precision allocation while preserving the efficiency benefits of low-bit quantization?**

---

## Motivation

Most low-bit quantization techniques were developed and evaluated primarily on Transformer architectures.

State-Space Models such as Mamba introduce recurrent hidden-state dynamics, where numerical errors may propagate differently due to sequential state updates.

This project investigates whether recurrent components demonstrate greater sensitivity to quantization than transient projection pathways.

---

## Experimental Setup

### Simulated Mamba Configuration

| Parameter | Value |
|------------|---------|
| D_MODEL | 768 |
| D_STATE | 16 |
| D_INNER | 1536 |
| Context Length | 15,000 Tokens |
| Device | NVIDIA RTX 3050 6GB |
| Environment | WSL Ubuntu 22.04 |

### Evaluated Policies

#### 1. FP16 Reference

Full precision recurrent state evolution.

#### 2. Uniform 4-Bit Quantization

Uniform quantization applied across recurrent and projection pathways.

#### 3. Topology-Aware Precision Allocation

High precision retained in recurrent dynamics while transient projection pathways are quantized.

---

## Results

Hidden-state divergence was measured relative to the FP16 reference trajectory.

| Tokens | Uniform Quantization Drift | Topology-Aware Drift |
|---------|---------|---------|
| 2,500 | 18.47 | 17.29 |
| 5,000 | 28.70 | 26.14 |
| 7,500 | 21.94 | 19.91 |
| 10,000 | 21.31 | 19.12 |
| 12,500 | 16.70 | 13.10 |
| 15,000 | 23.18 | 21.95 |

Across the evaluated sequence lengths, topology-aware precision allocation consistently produced lower state divergence than uniform quantization.

---

## Hardware Profiling

Nsight Compute profiling on an RTX 3050 reported:

| Metric | Value |
|---------|---------|
| DRAM Throughput | 92.28% |
| Compute Throughput | 66.72% |
| Achieved Occupancy | 72.64% |

These measurements indicate a strongly memory-bound workload, suggesting that recurrent state movement may represent a primary execution bottleneck.

---

## Key Observation

The experiments suggest that quantization sensitivity may not be uniformly distributed across recurrent State-Space Model topologies.

Within the evaluated setup, preserving precision in recurrent pathways reduced hidden-state divergence relative to a uniformly quantized baseline.

---

## Limitations

This project should be interpreted as an exploratory empirical study.

Current limitations include:

- Mamba-inspired simulation rather than pretrained checkpoints.
- Synthetic input workloads.
- No language-model perplexity evaluation.
- No downstream task benchmarks.
- Single-GPU experimental environment.
- Results should not be interpreted as proof of a general property of all SSM architectures.

Further validation on pretrained Mamba, Jamba, and related architectures remains future work.

---

## Reproducibility

Environment:

- Python 3.10
- PyTorch
- CUDA
- NVIDIA Nsight Compute
- Ubuntu 22.04 (WSL)

All experiments can be reproduced by running:

```bash
python3 mamba_profiler.py
```
## NSight Compute
<img width="1382" height="830" alt="Screenshot 2026-09-20 022725" src="https://github.com/user-attachments/assets/bf112640-1275-4b7d-ad1f-36edac36f196" />
<img width="1386" height="905" alt="Screenshot 2026-09-20 022702" src="https://github.com/user-attachments/assets/e73e33f4-e42a-4173-a2ca-05cc33d827e5" />


---

## Citation

If this work is useful in your research, please cite the associated Zenodo record once released.

---

## Disclaimer

This repository presents exploratory research observations and measurements. Conclusions should be interpreted as empirical findings within the described experimental setup rather than definitive claims about all State-Space Model architectures.
