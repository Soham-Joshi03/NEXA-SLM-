# Project Assets and Artifacts

---

## 1. Google Drive Working Folder
Working project assets and draft presentations were initially coordinated in the shared Drive folder:
- [NEXA-SLM Working Drive Folder](https://drive.google.com/drive/folders/1-fNrwISVklwcCSns8pu_Oto-5fnY-fBT?usp=sharing)
*(Note: Drive access is controlled separately from GitHub repository permissions).*

---

## 2. In-Repository Source Assets
The complete codebase has now been implemented, verified, and structured into the repository:
1. **Core Package (`nexa/`):**
   - `config.py`: Architecture specifications and preset configurations (`poc`, `base`, `dev`, `test`).
   - `mamba3.py`: Exponential-trapezoidal selective scan (Eqs. 3–4) and cached recurrent decoding.
   - `attention.py`: True $\mathcal{O}(L \cdot (W+m))$ chunked sliding-window attention and RoPE.
   - `moe.py`: 8 routed experts, Top-2 selection, 1 shared expert, auxiliary and z-loss.
   - `scratchpad.py`: Persistent virtual scratchpad prefix memory.
   - `model.py`: SuperBlock `[M+E, M+E, A, M+E, M+E]`, NexaSLM, and parameter counter.
   - `data.py`: uint16 memmap dataset loader, validator, and synthetic dataset generator.
   - `inference.py`: Autoregressive generation engine with prefill and sampling.
   - `quantization.py`: INT8 dynamic weight quantization and latency profiler.
2. **Master Entry Points:**
   - `train.py`: Full training loop with cosine decay, min_lr floor, and single-file checkpointing.
   - `inference.py`: Token generation with temperature, top-k, top-p, and INT8 quantization.
   - `evaluate.py`: Validation evaluation and Table VII ablation matrix execution.
3. **Automated Testing Suite (`tests/`):**
   - 19 automated tests covering components, training, inference, and end-to-end integration.
4. **Tokenizer:**
   - Reconstructed 32,768 ByteLevel BPE tokenizer via `scripts/assemble_tokenizer.py`.
