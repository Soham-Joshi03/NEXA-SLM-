# Contributing to NEXA-SLM

Thanks for helping develop NEXA-SLM. Contributions should prioritize correctness, reproducibility, clear documentation, and evidence-backed claims.

## Before you start
- Read the root README and docs/ARCHITECTURE.md.
- Check open issues and pull requests to avoid duplicate work.
- Discuss major architecture changes before implementing them.
- Confirm dataset and artifact licenses before use.

## Workflow
1. Branch from the latest `main`.
2. Create a focused branch such as `feature/router-load-balance`, `fix/causal-mask`, or `docs/setup-guide`.
3. Make focused commits.
4. Add or update tests and documentation.
5. Run relevant tests and include exact commands/results in the PR.
6. Open a PR against `main`, describing motivation, implementation, validation, and limitations.

## Standards
- Keep code modular and documented.
- Prefer deterministic experiments with explicit configuration.
- Never present a target as an achieved result.
- Do not include secrets, private data, or unlicensed datasets.
- Do not commit large weights, generated data, or environment folders.
- Document tensor shapes, masking, routing semantics, and precision assumptions.

Maintainers review correctness, architecture compatibility, tests, reproducibility, documentation, and resource implications.