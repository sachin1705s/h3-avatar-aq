# AQ Integration

AQ was used as the experiment orchestration layer around the H3 avatar work:

- recipe tracking for training and evaluation runs
- run metadata capture
- checkpoint/eval bookkeeping
- lightweight reproducibility notes around Modal jobs

The local AQ checkout used during development lives at:

```text
/Users/sachin/aq
```

It is intentionally not vendored into this repo because it is its own project with its own web app, package files, generated docs, and dependencies. This repo keeps the H3-specific AQ-facing recipe and run files instead:

- `recipe.yaml`
- `configs/`
- `benchmarks/`
- `evals/`
- `scripts/`

## Expected Usage

```bash
aq run recipe.yaml
```

The recipe is the canonical handoff point between AQ and this H3 Avatar repo.

