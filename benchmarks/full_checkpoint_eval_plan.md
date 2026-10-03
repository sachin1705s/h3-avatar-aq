# Full LoRA Checkpoint Eval Plan

Run: `/runs/h3/ref2va-lora-full-8000`

Checkpoints: `1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000`

For each checkpoint, record:

| eval | source | pass signal |
| --- | --- | --- |
| validation video loss | trainer log | lower than prior checkpoint or plateau without visual degradation |
| ffprobe sanity | `scripts/eval_ref2va_batch.py` | generated MP4 has video, audio, expected resolution, expected duration |
| reference vs generated gallery | `scripts/render_eval_gallery.py` | identity and motion can be inspected side by side |
| base vs LoRA gallery | same refs, seeds, prompts | LoRA output is visibly different from base in the intended direction |
| manual scorecard | `benchmarks/*_scorecard.csv` | identity, talking motion, audio sync, prompt following, generalization |
| AQ gate row | `scripts/apply_eval_scorecard.py` | weighted checkpoint score is written to `artifacts/h3/avatar_gate.json` |
| AQIT weight summary | `scripts/checkpoint_weight_summary.py` / AQIT when available | checkpoint is non-empty and LoRA norms are not exploding |

Manual score weights:

| metric | weight |
| --- | ---: |
| identity | 0.35 |
| talking_motion | 0.30 |
| audio_sync | 0.05 |
| prompt_following | 0.15 |
| generalization | 0.15 |

Selection rule:

Pick the checkpoint with the best combination of validation loss and manual visual score. Do not automatically pick the final step if validation loss worsens or the A/B gallery shows no meaningful improvement over base H3.

Generation policy:

- Generate 3 held-out validation references per checkpoint first.
- Use the same refs, prompts, seeds, frame count, and inference steps across checkpoints.
- Also generate base H3 outputs for those same refs once, so each checkpoint has a base-vs-LoRA comparison.
- Treat generated audio as a separate failure-prone lane. For checkpoint selection, prefer visual identity/talking motion; generated audio should be replaced with external clean English TTS or re-evaluated with a dedicated audio path.
- Use `--no-include-reference-audio` for visual checks, so source Korean audio is not copied into the reference conditioning.
