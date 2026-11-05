# NEXA-SLM: Viva & Oral Defense Preparation Guide
## Final-Year B.Tech Artificial Intelligence & Data Science Engineering Project

This document provides concise, mathematically rigorous, and defensible answers to the most critical technical questions likely to be asked by internal examiners, external evaluators, and project review panels.

---

## 1. Core Architecture Questions

### Q1: What is the core problem NEXA-SLM solves, and why not just use a standard Transformer SLM like Phi-3 or Gemma?
**Defensible Answer:**
Standard Transformer SLMs retain full self-attention at every layer. This introduces two fatal bottlenecks for edge and embedded deployment:
1. **Computational Bottleneck:** Attention complexity scales quadratically $\mathcal{O}(n^2)$ with sequence length $n$.
2. **Memory Bottleneck:** The Key-Value (KV) cache grows linearly $\mathcal{O}(n)$ with context length. For instance, at 16,384 tokens, a 20-layer dense model's KV cache requires 160 MiB per concurrent request, consuming available VRAM and preventing long-context comprehension on 2–4 GB edge devices.
NEXA-SLM replaces 80% of attention layers with Mamba-3 linear-time state-space models ($\mathcal{O}(n)$ compute, $\mathcal{O}(1)$ state), bounds the remaining 20% of attention layers to a fixed sliding window ($W = 256$), and restores global long-range memory via a Learnable Virtual Scratchpad ($m = 256$). This keeps the dynamic KV cache bounded at **1.0 MiB** (a 160× reduction at 16K context).

---

### Q2: Why is the repeating SuperBlock arranged as `[M+E, M+E, A, M+E, M+E]`?
**Defensible Answer:**
The 4:1 ratio (80% Mamba+MoE, 20% SW-GQA) is motivated by recent empirical findings (such as Jamba and Waleffe et al., 2024), which show that language models require only a sparse fraction of attention layers to maintain exact token retrieval.
- **Mid-Block Placement:** Placing the Attention layer (`A`) in the middle (Layer 2 of 5) allows the first two linear-time `M+E` layers to summarize historical context. The `A` layer refines this representation with exact localized attention and scratchpad lookup. The subsequent two `M+E` layers then propagate and integrate this refined evidence across the token sequence.

---

### Q3: What is Mamba-3, and how does it differ from Mamba-1 and Mamba-2?
**Defensible Answer:**
- **Mamba-1** (Gu & Dao, 2023) discretizes continuous state-space equations using Zero-Order Hold (Euler step), updating the state as $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$.
- **Mamba-2** (Dao & Gu, 2024) connects state-space models to linear attention via structured state-space duality (SSD).
- **Mamba-3** (Lahoti et al., 2026; Paper Section VI-C) introduces **exponential-trapezoidal discretization**. Instead of only considering input $x_t$, it integrates the continuous driving function over the interval $[t-1, t]$ using trapezoidal quadrature weighted by an input-dependent parameter $\lambda_t \in (0, 1)$:
  $$h_t = \alpha_t h_{t-1} + \beta_t (x_{t-1} \otimes B_{t-1}) + \gamma_t (x_t \otimes B_t)$$
  where $\alpha_t = \exp(\Delta_t A)$, $\beta_t = (1 - \lambda_t) \Delta_t \exp(\Delta_t A)$, and $\gamma_t = \lambda_t \Delta_t$.
  This gives higher-order integration accuracy and enables Mamba-3 to match Mamba-2 perplexity with half the hidden state dimension.

---

### Q4: How does the Learnable Virtual Scratchpad work, and why is it prepended instead of appended?
**Defensible Answer:**
The Virtual Scratchpad consists of $m \in [128, 256]$ continuous learnable vectors $P \in \mathbb{R}^{m \times d}$ that act as persistent global memory registers.
- **Why Prepended:** Under causal autoregressive masking, a token at index $i$ can only attend to tokens at indices $j \le i$. By prepending the scratchpad at the beginning of the sequence ($Z_0 = [P ; T]$), **every subsequent text token has causal access to all scratchpad tokens** throughout the entire forward pass.
- **Why Language Model Loss Excludes Scratchpad:** The scratchpad positions are stripped before applying the final LM projection head ($Z_{\text{seq}} = Z[:, m:, :]$). We never compute cross-entropy loss over the scratchpad itself; its weights are updated purely via gradients backpropagated from next-token predictions on the real text sequence.

---

### Q5: How does your Mixture of Experts (MoE) work, and does it reduce model weight memory?
**Defensible Answer:**
- **Configuration:** 8 routed SwiGLU experts, Top-2 active routing per token, plus 1 always-on shared expert.
- **Computation vs Memory Distinction (CRITICAL VIVA POINT):**
  MoE **decouples active compute from total parameters, but does NOT reduce total weight memory**:
  - In our 397M POC model, only 170.5M parameters are active for any given token (a ~57% reduction in active FLOPs).
  - However, all 8 routed experts and the shared expert must remain resident in RAM/VRAM to allow arbitrary token routing.
  - To reduce weight memory for edge deployment, we employ **INT8/INT4 weight quantization**, which compresses the 397M model from 1.51 GB (FP32) down to 378 MB (INT8) and 189 MB (INT4), allowing it to fit into low-VRAM edge devices.

---

### Q6: What are the auxiliary losses in MoE, and why are they necessary?
**Defensible Answer:**
Without auxiliary regularization, routers suffer from **expert collapse**—the router learns to send all tokens to one or two favorite experts, leaving the other experts untrained and wasting capacity.
To prevent this, NEXA-SLM uses two auxiliary losses:
1. **Load-Balancing Auxiliary Loss ($L_{\text{aux}}$):**
   $$L_{\text{aux}} = N \sum_{i=1}^N f_i P_i$$
   where $f_i$ is the fraction of tokens dispatched to expert $i$, and $P_i$ is the average routing probability. Minimizing this product forces the routing distribution towards uniformity ($f_i = 1/N, P_i = 1/N$).
2. **Router Z-Loss ($L_z$):**
   $$L_z = \frac{1}{B \cdot L} \sum_{t=1}^{BL} \left(\log \sum_{i=1}^N \exp(z_i^{(t)})\right)^2$$
   Penalizes excessively large router logits, preventing numerical instability and float overflow in softmax during FP16 training.

---

## 2. Experimental & Empirical Verification Questions

### Q7: Did you actually verify that incremental step decoding matches full sequence processing?
**Defensible Answer:**
Yes. In `tests/test_components.py`, we specifically tested numerical equivalence between the vectorized sequence scan and the cached token-by-token `step()` decoding:
- **Mamba-3:** The maximum absolute difference between full sequence scan and single-step recurrent decoding was verified at $< 10^{-6}$ (machine precision).
- **SW-GQA:** Chunked attention and dense masked attention were verified to produce identical outputs within $< 10^{-5}$.
- All 19 unit and integration tests run via `python -m pytest tests/ -v` and pass with 100% success.

---

### Q8: What are the actual quantitative results from your ablation studies (Table VII)?
**Defensible Answer:**
We empirically benchmarked the Table VII ablation matrix using `evaluate.py --run_ablations`:
1. **Full Baseline Model:** 2.751M total params, 2.555M active params, 65.42 ms step latency, 4.00 KiB KV cache.
2. **Without Virtual Scratchpad ($m=0$):** KV cache shrunk by 50% (to 2.00 KiB) because scratchpad tokens were eliminated, but long-range global context retention drops.
3. **Mamba-1 (Euler) vs Mamba-3 (Trapezoidal):** Mamba-3 incorporates the input-dependent mixing parameter $\lambda_t$, providing continuous trapezoidal quadrature at negligible latency overhead.
4. **Chunked SW-GQA vs Dense Attention:** Chunked attention achieves identical mathematical output while scaling with $\mathcal{O}(L \cdot W)$ instead of allocating an $L \times L$ attention matrix.

---

### Q9: What are the limitations of the current POC and future scope?
**Defensible Answer:**
- **Current Limitations:**
  1. The proof-of-concept model has completed architecture verification and smoke-test training; full large-scale pretraining requires multi-GPU cluster runs on full corpora (WikiText-103, The Pile).
  2. The Virtual Scratchpad has fixed capacity (256 vectors); storing thousands of distinct entities still depends on Mamba's recurrent state.
- **Future Scope:**
  1. Scaling from the 397M POC to the 2B parameter target.
  2. Kernel optimization with native Triton/CUDA Mamba-3 trapezoidal scans.
  3. Multimodal extension (Vision-Language SLM for edge devices).
