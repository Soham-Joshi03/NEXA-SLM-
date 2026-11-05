# NEXA-SLM
### A Scalable Hybrid Mamba-3, Sliding Window GQA and Mixture-of-Experts Small Language Model with a Learnable Virtual Scratchpad

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic_Research-green.svg)](README.md)
[![Tests](https://img.shields.io/badge/Tests-19%2F19%20Passing-brightgreen.svg)](tests/)

---

## 1. Project Overview

**NEXA-SLM** is a research-oriented Small Language Model (SLM) architecture developed from the ground up for efficient long-context comprehension and low-compute edge deployment (2–4 GB VRAM GPUs and CPU-only environments).

Large Language Models (LLMs) built on standard Transformers suffer from quadratic computational complexity $\mathcal{O}(n^2)$ and linear Key-Value (KV) cache memory growth $\mathcal{O}(n)$, making long sequences intractable on resource-constrained devices. NEXA-SLM solves this by combining five synergistic innovations:

1. **Mamba-3 Linear-Time Sequence Modeling:** Employs an exponential-trapezoidal selective scan recurrence that captures long-range dependencies with $\mathcal{O}(n)$ computational cost and constant $\mathcal{O}(1)$ recurrent state memory.
2. **Sliding Window Grouped Query Attention (SW-GQA):** Selectively applied in a 4:1 ratio to provide exact token-level retrieval while constraining KV cache to a fixed sliding window $W$.
3. **Learnable Virtual Scratchpad:** A persistent prefix of $m \in [128, 256]$ learnable vectors prepended to input representations, providing global memory accessible to attention without inflating the dynamic KV cache.
4. **Sparse Mixture of Experts (MoE):** 8 routed experts with Top-2 selection per token plus 1 always-on shared expert, decoupling total parameter capacity from per-token compute.
5. **Repeating SuperBlock Structure:** Stacked in the strict sequence `[M+E, M+E, A, M+E, M+E]`, stabilizing gradient propagation and giving an 80% linear-time / 20% local-attention computational balance.

---

## 2. Research Attribution

- **Project:** Final-Year B.Tech Project Stage-I / Stage-II (Course 417527)
- **Institution:** Department of Artificial Intelligence and Data Science, ABMSP's Anantrao Pawar College of Engineering & Research, Pune
- **Affiliation:** Savitribai Phule Pune University (SPPU), Academic Year 2026–27
- **Authors:**
  - Soham Joshi (`jsohamg@gmail.com`)
  - Srihari Mhaisekar (`sriharimhaisekar@gmail.com`)
  - Tanmayee Ozarkar (`tirtharya9381@gmail.com`)
  - Tanay Mehta (`tanaymehta090905@gmail.com`)
- **Project Mentor:** Dr. Jitendra Musale (`jitendra.musale@abmspcoerpune.org`)

---

## 3. Architecture Specification

```
Input Tokens (T_1 ... T_n) + Virtual Scratchpad (P_1 ... P_m)
                         │
                        RoPE
                         │
     ┌───────────────────┴───────────────────┐
     │      Repeat L SuperBlocks:            │
     │  Layer 0: M+E (Mamba-3 + MoE)         │
     │  Layer 1: M+E (Mamba-3 + MoE)         │
     │  Layer 2: A   (SW-GQA + Shared MLP)   │
     │  Layer 3: M+E (Mamba-3 + MoE)         │
     │  Layer 4: M+E (Mamba-3 + MoE)         │
     └───────────────────┬───────────────────┘
                         │
               Extract Sequence Tokens
                         │
                      RMSNorm
                         │
               Tied LM Projection Head
                         │
                Next-Token Prediction
```

### Mathematical Formulation

- **Input Representation:**
  $$Z_0 = \text{RoPE}([P ; T]), \quad P \in \mathbb{R}^{m \times d}, \quad T \in \mathbb{R}^{n \times d}$$
- **Pre-Norm Residual Sublayer:**
  $$Z_{l+1} = Z_l + F_l(\text{RMSNorm}(Z_l))$$
- **Mamba-3 Exponential-Trapezoidal Recurrence (Lahoti et al., 2026):**
  $$h_t = \alpha_t h_{t-1} + \beta_t (x_{t-1} \otimes B_{t-1}) + \gamma_t (x_t \otimes B_t), \quad y_t = C_t^\top h_t + D x_t$$
  $$\alpha_t = \exp(\Delta_t A), \quad \beta_t = (1 - \lambda_t) \Delta_t \exp(\Delta_t A), \quad \gamma_t = \lambda_t \Delta_t, \quad \lambda_t \in (0, 1)$$
- **SW-GQA Attention with Scratchpad Access:**
  $$O_h = \text{softmax}\left(\frac{Q_h K_{g(h)}^\top}{\sqrt{d_k}} + M\right) V_{g(h)}$$
  $$M_{ij} = \begin{cases} 0 & \text{if } j < m \text{ or } (i - W < j \le i) \\ -\infty & \text{otherwise} \end{cases}$$
- **Sparse MoE with Shared Expert:**
  $$\text{MoE}(x) = E_s(x) + \sum_{i \in \text{Top-2}(g)} \tilde{g}_i(x) E_i(x), \quad g(x) = \text{softmax}(W_r x)$$
  $$L_{\text{total}} = L_{\text{CE}} + \lambda_{\text{aux}} L_{\text{aux}} + \lambda_z L_z$$

---

## 4. Model Configurations & Sizing

| Parameter | Unit-Test Config | Dev Config | Base Config | Research POC Config |
|---|---|---|---|---|
| **Target Scale** | 3M | 42M | 149M | **397M (Paper Target)** |
| **Model Width ($d$)** | 64 | 256 | 512 | **512** |
| **SuperBlocks ($L$)** | 1 (5 layers) | 2 (10 layers) | 2 (10 layers) | **4 (20 layers)** |
| **MoE FFN Hidden Dim** | 128 | 512 | 1024 | **1536** |
| **Routed Experts** | 8 (2 active) | 8 (2 active) | 8 (2 active) | **8 (2 active)** |
| **Shared Expert** | 1 always active | 1 always active | 1 always active | **1 always active** |
| **Attention Heads (Q / KV)**| 2 / 1 | 4 / 2 | 8 / 2 | **8 / 2** |
| **Sliding Window ($W$)** | 16 | 128 | 256 | **256** |
| **Scratchpad Tokens ($m$)** | 16 | 64 | 256 | **256** |
| **Total Trainable Params** | **2.75 M** | **41.60 M** | **148.74 M** | **396.97 M** |
| **Active Params / Token** | **2.55 M** | **22.73 M** | **73.25 M** | **170.48 M** |
| **FP16 Checkpoint Size** | 5.25 MB | 79.35 MB | 283.70 MB | **757.17 MB (< 1 GB)** |
| **INT8 Checkpoint Size** | 2.62 MB | 39.67 MB | 141.85 MB | **378.58 MB (< 0.4 GB)** |

---

## 5. Analytical KV Cache & State Memory

At context length $n = 16,384$ tokens (FP16 precision):

$$\text{Dense Transformer (20 layers):} \quad 2 \times 20 \times 2 \times 64 \times 16384 \times 2 \text{ B} = \mathbf{160.0 \text{ MiB}}$$
$$\text{NEXA-SLM (4 SuperBlocks):} \quad 2 \times 4 \times 2 \times 64 \times (256 + 256) \times 2 \text{ B} = \mathbf{1.0 \text{ MiB}}$$

**NEXA-SLM achieves a 160× reduction in dynamic attention cache memory.**  
Additionally, the recurrent SSM state across all 16 Mamba-3 layers requires only **0.625 MiB**, yielding a total dynamic inference state of under **1.7 MiB** that remains strictly constant as context length grows to 32K+.

---

## 6. Repository Structure

```
NEXA-SLM/
├── nexa/                          # Core modular package
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Configurations (POC, Base, Dev, Test)
│   ├── mamba3.py                  # Trapezoidal selective scan & Mamba-3 layer
│   ├── attention.py               # SW-GQA chunked & dense attention + RoPE
│   ├── moe.py                     # Top-2 MoE + shared expert, aux & z-loss
│   ├── scratchpad.py              # Virtual scratchpad prefix manager
│   ├── model.py                   # SuperBlock, NexaSLM, parameter counter
│   ├── data.py                    # MemmapDataset, validators, synthetic generator
│   ├── inference.py               # Autoregressive generation & sampling
│   └── quantization.py            # INT8/INT4 weight quantization & profiler
├── scripts/
│   └── assemble_tokenizer.py      # Reconstructs jamba_bbpe_tokenizer.json
├── tokenizer/                     # Tokenizer split parts (part-01, part-02)
├── tests/                         # Full automated test suite (19 tests)
│   ├── test_components.py         # Mamba-3, SW-GQA, MoE, RoPE, Scratchpad
│   ├── test_training.py           # Optimizer, cosine schedule min_lr, checkpoints
│   ├── test_inference.py          # Autoregressive decoding, INT8 quantization
│   └── test_e2e.py                # Complete pipeline integration test
├── docs/                          # Comprehensive technical & academic documentation
│   ├── ARCHITECTURE.md            # Architectural specification & mathematical formulation
│   ├── METHODOLOGY.md             # Training, inference, and optimization methodology
│   ├── TESTING.md                 # Test suite documentation & empirical results
│   ├── RESULTS.md                 # Benchmark & ablation results (Table VII)
│   ├── VIVA_PREPARATION.md        # Comprehensive exam & viva defense guide
│   └── IMPLEMENTATION_STATUS.md   # Component verification status matrix
├── train.py                       # Master training entry point (CLI-ready)
├── inference.py                   # Master inference entry point (CLI-ready)
├── evaluate.py                    # Evaluation and Table VII ablation suite
├── requirements.txt               # Verified dependency specification
└── README.md                      # Project documentation
```

---

## 7. Quickstart Guide

### 7.1 Setup Environment

```bash
git clone https://github.com/Soham-Joshi03/NEXA-SLM-.git
cd NEXA-SLM-
pip install -r requirements.txt
```

### 7.2 Assemble Tokenizer

```bash
python scripts/assemble_tokenizer.py
```
*Reconstructs the 32,768-token ByteLevel BPE tokenizer with 32,501 merge rules and 11 special tokens.*

### 7.3 Run Test Suite

```bash
python -m pytest tests/ -v
```
*Executes all 19 automated tests covering component equivalence, gradient finiteness, checkpointing, and inference.*

### 7.4 Generate Development Dataset

```bash
python -m nexa.data --create-sample --train-tokens 150000 --val-tokens 30000
```

### 7.5 Train NEXA-SLM

```bash
# Smoke test on test config (CPU or GPU):
python train.py --config test --steps 20 --eval_every 5

# Local development training (42M model):
python train.py --config dev --steps 1000 --eval_every 50

# Research POC configuration (397M model, 20 layers):
python train.py --config poc --batch_size 2 --grad_accum 64 --mixed_precision fp16
```

### 7.6 Autoregressive Generation & Inference

```bash
# Run cached inference using a trained checkpoint:
python inference.py --config test --checkpoint checkpoints/best_model.pt --prompt "Small language models are"

# Run INT8 quantized low-compute inference with speed profiling:
python inference.py --config test --checkpoint checkpoints/best_model.pt --quantize_int8 --benchmark
```

### 7.7 Run Table VII Ablation Matrix

```bash
python evaluate.py --run_ablations
```

---

## 8. Empirical Test and Ablation Results

All 19 tests in the test suite pass with zero errors:

```text
tests/test_components.py::TestTokenizer::test_tokenizer_file_exists_and_loads PASSED
tests/test_components.py::TestTokenizer::test_special_tokens_roundtrip PASSED
tests/test_components.py::TestMamba3::test_mamba3_trapezoidal_vs_step_numerical_equivalence PASSED
tests/test_components.py::TestMamba3::test_mamba3_backward_gradients_finite PASSED
tests/test_components.py::TestSWGQA::test_swgqa_chunked_vs_dense_equivalence PASSED
tests/test_components.py::TestSWGQA::test_swgqa_sliding_kv_cache_bounded PASSED
tests/test_components.py::TestMoE::test_top2_routing_weights_normalized PASSED
tests/test_components.py::TestMoE::test_shared_expert_always_active PASSED
tests/test_components.py::TestVirtualScratchpad::test_scratchpad_prefix_and_extraction PASSED
tests/test_components.py::TestSuperBlockAndModel::test_superblock_sublayer_order PASSED
tests/test_components.py::TestSuperBlockAndModel::test_full_model_forward_and_loss PASSED
tests/test_e2e.py::test_full_end_to_end_pipeline PASSED
tests/test_inference.py::test_autoregressive_generation_length PASSED
tests/test_inference.py::test_deterministic_greedy_generation PASSED
tests/test_inference.py::test_eos_early_stopping PASSED
tests/test_inference.py::test_int8_quantization_stability PASSED
tests/test_training.py::test_cosine_schedule_with_min_lr_floor PASSED
tests/test_training.py::test_checkpoint_save_and_reload PASSED
tests/test_training.py::test_perplexity_computation PASSED

============================= 19 passed in 18.90s =============================
```

### Table VII: Research Paper Ablation Results

| Ablation Variant | Total Params (M) | Active Params (M) | Step Latency (ms) | KV Cache Size (KiB) |
|---|---|---|---|---|
| **Full Model (Baseline)** | **2.751** | **2.555** | **65.42** | **4.00** |
| w/o Virtual Scratchpad ($m=0$) | 2.750 | 2.554 | 43.93 | 2.00 (-50%) |
| w/o Mamba-3 (Mamba-1 Euler) | 2.747 | 2.550 | 49.48 | 4.00 |
| SW-GQA Dense Attention | 2.751 | 2.555 | 53.27 | 4.00 |
| **SW-GQA Chunked $\mathcal{O}(nW)$** | **2.751** | **2.555** | **50.71** | **4.00** |

---

## 9. Academic Project Readiness

| Milestone | Status | Assessment |
|---|---|---|
| **Architecture Implementation** | Verified | Faithful implementation of Mamba-3 (Eqs. 3–4), SW-GQA, MoE, Scratchpad, and SuperBlock. |
| **Mathematical Correctness** | Verified | Scan and cached step decoded outputs match numerically ($< 10^{-5}$). |
| **Low-Compute Optimization** | Verified | Constant-size KV cache, INT8 weight quantization, 160× memory saving at 16K context. |
| **Automated Testing** | Verified | 19 unit, training, inference, and end-to-end integration tests passing. |
| **Internal & External Viva** | **Ready** | Thorough documentation (`docs/VIVA_PREPARATION.md`, `ARCHITECTURE.md`, `METHODOLOGY.md`). |

---

## 10. References

1. Vaswani, A., et al. "Attention is all you need." *NeurIPS*, 2017.
2. Gu, A., & Dao, T. "Mamba: Linear-time sequence modeling with selective state spaces." *arXiv:2312.00752*, 2023.
3. Lahoti, A., et al. "Mamba-3: Improved sequence modeling using state space principles." *ICLR*, 2026.
4. Ainslie, J., et al. "GQA: Training generalized multi-query transformer models from multi-head checkpoints." *EMNLP*, 2023.
5. Jiang, A. Q., et al. "Mixtral of Experts." *arXiv:2401.04088*, 2024.
6. Lieber, O., et al. "Jamba: A hybrid Transformer-Mamba language model." *arXiv:2403.19887*, 2024.
