# NEXA-SLM Experimental Results & Validation

---

## 1. Parameter Sizing & Sparsity Analysis

Calculated using `nexa.model.count_parameters`:

| Configuration Preset | SuperBlocks | Total Parameters | Active Parameters / Token | Sparsity Ratio | FP16 Size | INT8 Size | INT4 Size |
|---|---|---|---|---|---|---|---|
| **Unit-Test** | 1 (5 layers) | 2.75 M | 2.55 M | 92.7% | 5.25 MB | 2.62 MB | 1.31 MB |
| **Development** | 2 (10 layers) | 41.60 M | 22.73 M | 54.6% | 79.35 MB | 39.67 MB | 19.84 MB |
| **Base** | 2 (10 layers) | 148.74 M | 73.25 M | 49.2% | 283.70 MB | 141.85 MB | 70.92 MB |
| **Research POC** | 4 (20 layers) | **396.97 M** | **170.48 M** | **42.9%** | **757.17 MB** | **378.58 MB** | **189.29 MB** |

> **Key Finding:** In the Research POC configuration, Top-2 sparse routing reduces active per-token compute by **57.1%** (from 397M to 170.5M active parameters), while retaining full 397M parameter capacity in memory.

---

## 2. Analytical Memory Scaling: KV Cache & SSM State

Calculated using Eq. 11: $M_{\text{KV}} = 2 \cdot L_{\text{blocks}} \cdot n_{\text{kv}} \cdot d_h \cdot (W + m) \cdot b$ (with $L_{\text{blocks}} = 4, n_{\text{kv}} = 2, d_h = 64, W = 256, m = 256$, FP16 $b = 2$):

| Context Length ($n$) | Standard Dense Transformer | NEXA-SLM Hybrid | Memory Reduction |
|---|---|---|---|
| 2,048 tokens | 20.0 MiB | 1.0 MiB | **20×** |
| 4,096 tokens | 40.0 MiB | 1.0 MiB | **40×** |
| 8,192 tokens | 80.0 MiB | 1.0 MiB | **80×** |
| 16,384 tokens | 160.0 MiB | 1.0 MiB | **160×** |
| 32,768 tokens | 320.0 MiB | 1.0 MiB | **320×** |

- **KV Cache Memory:** Exactly **1.00 MiB** (constant regardless of context length).
- **Mamba-3 Recurrent State:** Exactly **0.625 MiB** across all 16 Mamba layers.
- **Total Dynamic Inference State:** Under **1.63 MiB**.

---

## 3. Automated Test Suite Execution Results

Ran using `python -m pytest tests/ -v`:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0
collected 19 items

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

---

## 4. Table VII: Architectural Ablation Matrix Results

Executed on local test configuration via `evaluate.py --run_ablations`:

| Ablation Variant | Total Parameters (M) | Active Parameters (M) | Step Forward Latency (ms) | KV Cache Size (KiB) |
|---|---|---|---|---|
| **Full Model (Baseline)** | **2.751** | **2.555** | **65.42** | **4.00** |
| w/o Virtual Scratchpad ($m=0$) | 2.750 | 2.554 | 43.93 | 2.00 (-50%) |
| w/o Mamba-3 (Mamba-1 Euler) | 2.747 | 2.550 | 49.48 | 4.00 |
| SW-GQA Dense Attention | 2.751 | 2.555 | 53.27 | 4.00 |
| **SW-GQA Chunked $\mathcal{O}(nW)$** | **2.751** | **2.555** | **50.71** | **4.00** |

---

## 5. Inference Benchmarking Results

Measured on local workstation CPU via `inference.py --benchmark`:

| Precision | Throughput (tokens/sec) | Latency / Token (ms) | Checkpoint Size (Test Config) |
|---|---|---|---|
| **FP32** | 44.42 | 22.51 | 10.49 MB |
| **INT8 Quantized** | 39.45 | 25.35 | 2.62 MB (-75%) |
