# NEXA-SLM Research Methodology

---

## 1. Overview
This document details the engineering and mathematical methodology for the design, training, inference, and optimization of **NEXA-SLM: A Scalable Hybrid Mamba-Attention MoE Small Language Model**.

---

## 2. Tokenization and Data Preprocessing
- **Tokenizer:** ByteLevel Byte-Pair Encoding (BBPE) tokenizer with a vocabulary size of 32,768, 32,501 merge rules, and 11 reserved special tokens (`<|endoftext|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, `<|doc_start|>`, `<|doc_end|>`, `<think>`, `</think>`, `<|system|>`, `<|tool_call|>`, `<|tool_resp|>`).
- **Binary Format:** Memory-mapped (`np.memmap`) arrays of 16-bit unsigned integers (`np.uint16`). Since $32,768 \le 2^{16} - 1 = 65,535$, uint16 representation cuts dataset storage by 50% compared to standard int32, while allowing zero-copy operating system page caching.
- **Sequence Preparation:** Contiguous non-overlapping chunks of length $L + 1$ are sliced, producing input sequences $x = T_{0:L}$ and target sequences $y = T_{1:L+1}$ for causal next-token prediction.

---

## 3. Training Recipe & Loss Optimization
- **Loss Formulation:**
  $$L_{\text{total}} = L_{\text{CE}} + \lambda_{\text{aux}} L_{\text{aux}} + \lambda_z L_z$$
  - Cross-Entropy Loss: $L_{\text{CE}} = -\frac{1}{L} \sum_{t=1}^L \log p_\theta(y_t \mid x_{\le t})$
  - Auxiliary Load-Balancing Loss: $L_{\text{aux}} = N \sum_{i=1}^N f_i P_i$ ($\lambda_{\text{aux}} = 0.01$)
  - Router Z-Loss: $L_z = \frac{1}{B \cdot L} \sum (\log \sum \exp(z_i))^2$ ($\lambda_z = 0.001$)
- **Optimization Strategy:**
  - Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$, weight decay = 0.01).
  - Learning Rate Schedule: Linear warmup for 10% of total steps, followed by Cosine Decay down to a verified minimum learning rate ($\eta_{\min} = 3 \times 10^{-5}$).
  - Gradient Accumulation: Micro-batch size per device with gradient accumulation to achieve effective batch size of ~1M tokens per global step.
  - Gradient Clipping: Norm clipped at 1.0 to ensure gradient stability across hybrid state-space and attention layers.
  - Precision: Automatic mixed precision (FP16 / BF16 on GPU, FP32 on CPU).

---

## 4. Inference & Low-Compute Acceleration
- **Bounded KV Cache:** Bounded to $(W + m)$ tokens. Oldest sliding-window tokens beyond the window $W$ are evicted, keeping memory strictly $\mathcal{O}(1)$ with respect to sequence length.
- **Mamba-3 Recurrent State:** Hidden state $h_t \in \mathbb{R}^{d_{\text{inner}} \times d_{\text{state}}}$, rolling convolution buffer $x_{\text{conv}} \in \mathbb{R}^{d_{\text{inner}} \times (K_c - 1)}$, and previous driving vectors $u_{t-1}, B_{t-1}$ are preserved across autoregressive steps.
- **INT8 Quantization:** Symmetric per-channel dynamic quantization maps weights to 8-bit signed integers ($q \in [-128, 127]$), reducing model weight memory by 4× compared to FP32 without retraining.
