# Ref2VA Eval Batch 001

Objective pass: 3/3

Gallery: `benchmarks/ref2va_eval_gallery.html`

| sample | seed | reference duration | generated duration | generated resolution | generated audio | objective status |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 206000 | 9.96s | 5.17s | 256x256 | yes | pass |
| 1 | 206001 | 9.96s | 5.17s | 256x256 | yes | pass |
| 2 | 206002 | 9.96s | 5.17s | 256x256 | yes | pass |

## Manual Scorecard

CSV: `benchmarks/ref2va_full_ckpt6000_diverse3_scorecard.csv`

Score each metric from 0.0 to 1.0 after watching the gallery:

- `identity`: generated subject resembles the reference character.
- `talking_motion`: face/body motion is plausible for talking.
- `audio_sync`: mouth motion and audio feel aligned.
- `prompt_following`: output follows the friendly talking prompt.
- `generalization`: held-out validation reference does not collapse to a memorized training character.

After manual scoring, record the checkpoint with `scripts/record_checkpoint_eval.py` and run AQ eval.
