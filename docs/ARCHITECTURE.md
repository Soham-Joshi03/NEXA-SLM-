# NEXA-SLM Architecture Notes

This document records the proposed design in the research paper. It is a specification, not proof that the model has been implemented or benchmarked.

## 1. Input path
Let token embeddings be T ∈ R^(n×d) and learnable scratchpad vectors be P ∈ R^(m×d), with m ∈ [128, 256]. The paper defines the starting sequence as:

```text
Z₀ = RoPE([P ; T])
```

The scratchpad is prepended so text tokens can attend to the prefix under the described causal mask.

## 2. SuperBlock
Each SuperBlock repeats this order:

```text
[M+E, M+E, A, M+E, M+E]
```

- M+E: Mamba-3 sequence mixer plus sparse MoE.
- A: SW-GQA attention plus shared MLP.

Thus, four of five layers per SuperBlock use Mamba-3 + MoE, and one uses attention.

## 3. Residual and normalization
The paper gives the pre-norm residual expression:

```text
Zₗ₊₁ = Zₗ + Fₗ(RMSNorm(Zₗ))
```

The algorithm also describes normalization and residual operations around the Mamba-3 and MoE sublayers in M+E blocks, and around SW-GQA and the shared MLP in A blocks. Keep implementation details explicit and test them.

## 4. Mamba-3
Mamba-3 uses the trapezoidal selective-scan recurrence described by the paper, with input-dependent terms. It is intended for bulk sequence mixing and long-range dependencies with linear sequence cost.

## 5. SW-GQA
For H query heads and G key/value groups, query heads share KV groups. Each query attends within a local window W and to the scratchpad prefix of length m. One attention layer appears in each five-layer SuperBlock.

## 6. Mixture of Experts
The research paper specifies eight routed experts, top-2 routed experts per token, and one always-on shared expert. The mathematical section defines N=8 routed experts plus a separate shared expert. Auxiliary load-balancing loss and router z-loss are used to stabilize routing.

## 7. Virtual Scratchpad
The scratchpad comprises 128–256 learnable prefix tokens. It is intended to provide persistent global context to attention. The paper states the prefix keys/values are cached once.

## 8. Output
After the final SuperBlock, RMSNorm and the LM head produce next-token logits. Inference is autoregressive.

## 9. Configuration targets (not measured results)

| Parameter | Specification |
|---|---|
| POC size | 300M–400M |
| Scale target | Up to 2B |
| Context | 8K–16K+ tokens |
| Scratchpad | 128–256 tokens |
| Routed experts | 8; top-2 active |
| Shared expert | 1 always active |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Inference target | 2–4 GB VRAM GPU / CPU |

These targets must not be presented as verified capabilities until tested.