# NEXA-SLM
### A Scalable Hybrid Mamba–Attention Mixture-of-Experts Small Language Model

NEXA-SLM is a research project exploring a compact language model architecture for constrained hardware. Its proposed design combines **Mamba-3**, **Sliding-Window Grouped-Query Attention (SW-GQA)**, a sparsely routed **Mixture of Experts (MoE)**, and a **Learnable Virtual Scratchpad** for persistent global context.

> **Status:** Research / proof-of-concept implementation in progress. The paper specifies a 300M–400M parameter POC and a longer-term scale target of up to 2B parameters. This repository is being prepared for collaborative development. Implementation code, trained checkpoints, and benchmark results are not yet included in this initial documentation commit. Hardware targets below are goals, not verified results.

## Architecture at a glance

| Component | Research specification |
|---|---|
| Sequence model | Mamba-3, trapezoidal selective scan |
| Attention | Sliding-Window Grouped-Query Attention (SW-GQA) |
| MoE | 8 routed experts; top-2 active per token; 1 shared expert |
| Global memory | 128–256 learnable prefix / scratchpad tokens |
| SuperBlock | `[M+E, M+E, A, M+E, M+E]` |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| POC parameters | 300M–400M |
| Longer-term scale target | Up to 2B |
| Context target | 8K–16K+ tokens |
| Inference target | 2–4 GB VRAM GPU / CPU (not yet validated) |

`M+E` means Mamba-3 + MoE; `A` means SW-GQA + shared MLP.

## Conceptual data flow

```text
Input tokens + 128–256 learnable scratchpad prefix
                    |
                   RoPE
                    |
      Repeat L SuperBlocks in sequence:
      [M+E] -> [M+E] -> [A] -> [M+E] -> [M+E]
                    |
                 RMSNorm
                    |
           LM head / projection
                    |
           Next-token prediction
```

## Design rationale

- **Mamba-3** is intended to handle most sequence mixing through a linear-time state-space path.
- **SW-GQA** is used sparingly for precise local retrieval with grouped key/value heads and a bounded window.
- **Sparse MoE** increases model capacity while selecting only two routed experts per token, alongside an always-on shared expert.
- **Virtual Scratchpad** tokens are prepended to provide persistent global context to the attention layer.

The design and analytical motivation are described in the project's research paper. Actual quality, throughput, memory consumption, and CPU/low-VRAM feasibility require implementation and empirical validation.

## Training and evaluation plan

The paper proposes causal next-token prediction with cross-entropy, MoE load-balancing auxiliary loss, and router z-loss. Its training strategy specifies AdamW, warmup followed by cosine decay, dropout/weight decay, RMSNorm, and mixed precision.

The paper's evaluation plan names WikiText-2, The Pile, and OpenWebText for language modeling; RULER and position-sensitivity tests for long-context evaluation; SQuAD, CNN/DailyMail, and HumanEval for downstream evaluation; and ablations removing the scratchpad, MoE routing, and Mamba-3.

## Repository structure

```text
.github/                  Issue and pull-request templates
docs/                     Architecture, development, and experiment notes
research/                 Research paper information
CONTRIBUTING.md           Collaboration guidelines
.gitignore                Ignore rules for environments, data, and checkpoints
README.md                 Project overview
```

As implementation work begins, organize code into focused modules such as `nexa/`, `configs/`, `scripts/`, `tests/`, and `notebooks/`. Do not commit credentials, private data, virtual environments, datasets, or large checkpoints to ordinary Git history.

## Getting started

This initial repository state is a documentation scaffold; it does not yet provide runnable training or inference commands.

```bash
git clone https://github.com/Soham-Joshi03/NEXA-SLM-.git
cd NEXA-SLM-
git switch -c feature/your-change
```

Before claiming support for a device or precision, validate it and document the exact configuration and measurements.

## Current implementation files

- `inference.py`: token-by-token autoregressive generation with cached Mamba and SW-GQA states, sampling, and an execution example.
- `tokenizer/`: the supplied ByteLevel BPE tokenizer JSON, stored in two consecutive parts to preserve the original data.
- `scripts/assemble_tokenizer.py`: reconstructs and validates the root-level `jamba_bbpe_tokenizer.json` expected by the inference script.

Before running inference, execute `python scripts/assemble_tokenizer.py`. **The inference script also imports model definitions from `train.py`, which has not yet been uploaded; inference is not runnable from this repository alone until that module and a compatible checkpoint are added.** See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

## Collaboration

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Work on feature branches and submit pull requests to `main`; keep commits focused and include tests and reproducibility information.

## Research limitations

The paper identifies the POC as not yet empirically validated. Its analytical memory estimates exclude activations, SSM states, and runtime overhead. Sparse routing reduces active compute but does not eliminate resident expert-weight memory. Scratchpad memory does not guarantee exact recall of all distant facts. CPU/embedded kernels and INT8/INT4 behavior require separate validation.

## License status

No license has been selected yet. Because the research paper lists multiple authors, confirm the software and research-artifact licensing terms with all relevant rights holders before adding a license or redistributing the paper, code, or weights. Until then, do not assume this public repository grants general reuse rights; grant trusted collaborators repository access or have them submit fork-based pull requests.

## Acknowledgement

NEXA-SLM is a collaborative research project from the Department of AI & Data Science, ABMSP's Anantrao Pawar College of Engineering & Research, Pune, India.
