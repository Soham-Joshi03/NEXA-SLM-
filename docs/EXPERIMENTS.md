# Experiments and Evaluation Plan

This is the evaluation protocol proposed in the research paper. It is a plan; results must only be added after experiments have been run.

## Language modeling
- Datasets named in the paper: WikiText-2, The Pile, and OpenWebText.
- Primary metric: perplexity, PPL = exp(L_CE).
- Record dataset versions, preprocessing, tokenizer, splits, context length, training budget, and decontamination checks.

## Long-context evaluation
- Target context: 8K–16K+ tokens.
- Proposed tests: RULER-style synthetic retrieval and position-sensitivity analysis.
- Report retrieval accuracy by context length and token position, with failure cases.

## Downstream evaluation
- Question answering: SQuAD.
- Summarization: CNN/DailyMail.
- Code generation: HumanEval.

Use accepted benchmark protocols and document any prompt or decoding changes.

## Ablations
1. Without Scratchpad: remove the 128–256 prefix tokens.
2. Without MoE: replace sparse routing with a dense MLP.
3. Without Mamba-3: remove the linear-time sequence mixer.

Keep other settings and training budgets comparable when practical.

## Reproducibility record
For every run, record experiment ID/date, Git commit SHA, configuration, seed, dataset versions/licenses/splits, preprocessing and contamination checks, hardware, driver/CUDA/Python/framework versions, parameter count, precision, batch size, sequence length, optimizer, schedule, training tokens/steps, peak memory, throughput, latency, loss, perplexity, downstream metrics, and checkpoint URI/checksum.

## Limitations
The paper states that the POC has no empirical results yet. Its analytical memory analysis excludes activations, SSM states, and runtime overhead. Sparse routing lowers active compute but not resident expert weight memory. CPU/embedded execution and quantized recurrent-state/routing behavior require separate validation.