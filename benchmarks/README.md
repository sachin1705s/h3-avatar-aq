# Benchmark Outputs

This directory is for generated benchmark summaries. Source definitions live in:

- `../evals/avatar_benchmarks.json`
- `../evals/progress_prompts.jsonl`
- `../configs/progress_checkpoints.json`

Use `scripts/record_checkpoint_eval.py` after each checkpoint review, then
`scripts/benchmark_report.py` to create a markdown progress table.
