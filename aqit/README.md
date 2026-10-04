# AQIT Integration

AQIT was used as the interpretability and checkpoint-inspection layer for the H3 avatar LoRA work.

The local AQIT checkout used during development lives at:

```text
/Users/sachin/AQIT
```

It is intentionally not vendored here because AQIT is its own toolchain. This repo keeps the H3-specific probes and result artifacts that make the AQIT usage reproducible:

- `aqit/probes/avatar_prompts.jsonl`
- `benchmarks/objective_checkpoint_report*.md`
- `benchmarks/weight_summary_ckpt100.md`
- `artifacts/h3/*` locally, with selected reports promoted into `benchmarks/`

## What AQIT Was Used For

- checkpoint weight summaries
- objective checkpoint reports
- identity and speaking-quality eval prompts
- regression tracking across training checkpoints
- deciding which LoRA checkpoint to lock for the final adapter

