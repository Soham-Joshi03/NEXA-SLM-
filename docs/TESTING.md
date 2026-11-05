# NEXA-SLM Testing and Quality Assurance Documentation

This document describes the testing strategy, test suites, and empirical validation methods for the **NEXA-SLM** project.

---

## 1. Test Suite Architecture

The automated test suite is organized under `tests/` and built using `pytest`:

```
tests/
├── test_components.py   # Unit tests for core neural components
├── test_training.py     # Optimizer, schedule, and checkpoint resumption
├── test_inference.py    # Autoregressive decoding and INT8 quantization
└── test_e2e.py          # Complete pipeline integration test
```

---

## 2. Test Execution Commands

```bash
# Run all tests with verbose output:
python -m pytest tests/ -v

# Run component-level tests only:
python -m pytest tests/test_components.py -v

# Run training tests only:
python -m pytest tests/test_training.py -v

# Run inference tests only:
python -m pytest tests/test_inference.py -v

# Run end-to-end integration test only:
python -m pytest tests/test_e2e.py -v
```

---

## 3. Test Coverage Summary

### 3.1 Component Tests (`tests/test_components.py`)
- **Tokenizer:** Confirms 32,768 vocabulary entries, 32,501 merge rules, and special token preservation.
- **Mamba-3:** Proves numerical equivalence ($< 10^{-5}$) between vectorized full-sequence scan and cached step-by-step autoregressive decoding.
- **SW-GQA Attention:** Validates causal masking, sliding-window eviction, and equivalence between $\mathcal{O}(L \cdot (W+m))$ chunked attention and dense masked attention.
- **MoE:** Confirms Top-2 expert routing, weight normalization ($\sum \tilde{g}_i = 1.0$), always-on shared expert contribution, and finite load-balancing loss gradients.
- **Virtual Scratchpad:** Confirms prefix concatenation $[P; T]$ and sequence extraction $Z[:, m:, :]$ without loss contamination.
- **SuperBlock:** Confirms strict 5-layer sequencing `[M+E, M+E, A, M+E, M+E]`.

### 3.2 Training Tests (`tests/test_training.py`)
- **LR Schedule:** Validates linear warmup and cosine decay strictly respecting the $\eta_{\min} = 3 \times 10^{-5}$ floor.
- **Checkpointing:** Validates single-file `.pt` state dictionary serialization and reload without state corruption.
- **Perplexity:** Confirms exact mathematical computation of $PPL = \exp(L_{\text{CE}})$.

### 3.3 Inference Tests (`tests/test_inference.py`)
- **Autoregressive Generation:** Confirms exact length control and state propagation.
- **Greedy Determinism:** Validates that temperature=0.0 produces 100% deterministic outputs across repeated runs.
- **EOS Halting:** Validates that decoding stops immediately when end-of-sequence token ID 0 is encountered.
- **INT8 Quantization:** Confirms stability and valid output generation under dynamic 8-bit weight quantization.

### 3.4 Integration Test (`tests/test_e2e.py`)
- Executes the full lifecycle: Tokenizer → Synthetic Binary Dataset → Model → Accelerate Training Loop → Checkpoint Save → Checkpoint Reload → Cached Autoregressive Decoding.
