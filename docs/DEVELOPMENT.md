# Development Guide

## Collaboration model
1. Keep `main` reviewable.
2. Create a branch for each task: `feature/<name>`, `fix/<name>`, or `docs/<name>`.
3. Make focused commits.
4. Run relevant tests and record commands/results in the PR.
5. Open a pull request against `main` and request at least one teammate review.
6. Address review comments before merging.

## Clone and branch

```bash
git clone https://github.com/Soham-Joshi03/NEXA-SLM-.git
cd NEXA-SLM-
git switch -c feature/my-change
```

## Update from main

```bash
git switch main
git pull --ff-only origin main
git switch feature/my-change
git merge main
```

## Push and open a pull request

```bash
git status
git add <files-you-intend-to-commit>
git commit -m "feat: describe the change"
git push -u origin feature/my-change
```

Open a PR from your branch into `main`.

## Repository hygiene
- Never commit passwords, API keys, access tokens, or private data.
- Do not commit virtual environments, caches, temporary logs, raw datasets, or model checkpoints to ordinary Git history.
- Use Git LFS or approved artifact/model storage for large approved assets.
- Record dataset provenance, license, preprocessing, splits, and checksums.
- Pin dependencies once the implementation stack is settled.
- Add tests for tensor shapes, causal masking, routing, numerical stability, checkpoint loading, and generation as those modules are built.
- Record hardware, software versions, seeds, configuration, training steps/tokens, and evaluation commands.

Suggested commit prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `exp:`, `chore:`.