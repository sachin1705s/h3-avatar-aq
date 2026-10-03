# H3 Avatar AQ

Locked MiniMax-H3 Ref2VA LoRA for talking-avatar generation with model-generated English audio.

## Locked Checkpoint

The current usable adapter is:

`models/locked/h3-avatar-audio-v2-step1000.pt`

Modal volume alias:

`/runs/h3/locked/h3-avatar-audio-v2-step1000.pt`

Original training checkpoint:

`/runs/h3/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/checkpoint-1000.pt`

## Key Stats

- Base: MiniMax-H3 Ref2VA
- Adapter: LoRA, rank 16
- Trainable params: 10,655,744
- Accepted transcript clips: 481
- Train split: 385 clips
- Val/test: 48 / 48
- Training: 1000 optimizer steps, about 2.6 passes over train split
- Cache: 384x384, 56 frames
- Main improvement: transcript-conditioned audio prompts from ASR.

## Run The Local UI

```bash
python3 ui/h3_avatar_ui.py
open http://127.0.0.1:8787
```

The UI defaults to the locked audio-v2 checkpoint.

## Modal Inference

The current generation path uses Modal H100 inference through `modal_h3_avatar.py`.

```bash
modal run modal_h3_avatar.py::generate_ref2va_sample \
  --checkpoint /runs/h3/locked/h3-avatar-audio-v2-step1000.pt \
  --reference-video /runs/samples/image_batch_10s_refs/ab_headshot_ref_512_10s.mp4 \
  --output-path /runs/generated/manual/generated.mp4 \
  --prompt '<Video 1> The same character says in English: "Hello from the avatar model." with natural voice, synchronized mouth movement, subtle head motion, and a friendly expression.' \
  --num-frames 124 \
  --height 512 \
  --width 512 \
  --num-inference-steps 30 \
  --seed 2100 \
  --no-include-reference-audio \
  --generate-audio
```

## Locked Recipe

See:

- `recipes/locked_h3_avatar_checkpoint.md`
- `recipes/h3_lora_audio_v2_transcripts.md`

---

# Original AQ Run Notes

This train folder wraps the H3 Ref2VA LoRA plan from `H3-Avatar.docx` in AQ and leaves the H3-specific training work to a real MiniMax-H3 fine-tuning repo.

## What Is Here

- `recipe.yaml`: AQ run identity for MiniMax H3 Ref2VA BF16 LoRA.
- `tools/h3_ref2va.py`: custom AQ method that calls the external H3 trainer.
- `scripts/prepare_ta20_subset.py`: identity-based TA2.0 split using `ref_id`.
- `scripts/write_avatar_gate.py`: writes the score consumed by `aq eval`.
- `scripts/make_checkpoint_eval_plan.py`: creates generation tasks for each progress checkpoint.
- `scripts/record_checkpoint_eval.py`: records benchmark scores per checkpoint.
- `scripts/benchmark_report.py`: renders a checkpoint progress report.
- `configs/progress_checkpoints.json`: five decision checkpoints at steps 20, 40, 60, 80, and 100.
- `evals/avatar_benchmarks.json`: benchmark suite definitions and score weights.
- `aqit/probes/avatar_prompts.jsonl`: starter AQIT probes for the inspect loop.

The recipe starts with `dry_run: true` so AQ can create artifacts before the dataset, H3 weights, and trainer are wired in.

## Smoke Flow

```bash
cd /Users/sachin/h3-avatar-aq
/Users/sachin/aq/aq/bin/aq train
/Users/sachin/aq/aq/bin/aq eval
cat artifacts/inspect.md
```

In `dry_run: true`, `aq train` should pass and write the run plan. `aq eval`
will report `avatar_gate=0` / fail because no real H3 samples have been
generated yet.

The dry run also creates placeholder slots for:

- `ckpt_020`
- `ckpt_040`
- `ckpt_060`
- `ckpt_080`
- `ckpt_100`

## Prepare TA2.0 Subset

```bash
cd /Users/sachin/h3-avatar-aq
python3 scripts/prepare_ta20_subset.py \
  --metadata /path/to/ta20_metadata.jsonl \
  --root /path/to/TA2.0_animation_dataset \
  --limit 300 \
  --out data/metadata
```

The split keeps every `ref_id` in only one of train, val, or test.

## Five Checkpoints

The default smoke run tracks five decision points:

| checkpoint | step | purpose |
| --- | ---: | --- |
| `ckpt_020` | 20 | pipeline sanity: decodes, visible reference, audio exists |
| `ckpt_040` | 40 | early identity on easy prompts |
| `ckpt_060` | 60 | static and active talking motion |
| `ckpt_080` | 80 | prompt control without identity loss |
| `ckpt_100` | 100 | smoke target for deciding whether to scale up |

The schedule is in `configs/progress_checkpoints.json` and mirrored in
`recipe.yaml` as `checkpoint_steps: 20,40,60,80,100`.

## Real H3 Training

```bash
cd /Users/sachin/h3-avatar-aq
source configs/h3_ref2va_lora_smoke.env.example
# edit recipe.yaml: dry_run: false
/Users/sachin/aq/aq/bin/aq train
```

The adapter expects trainer scripts compatible with:

```bash
prepare_cache.py --model <MiniMax-H3> --variant ref2va --manifest <train.jsonl> --cache <cache> --encode-audio
train.py --model <MiniMax-H3> --variant ref2va --cache <cache> --output <run> --max-steps <N> --trainable lora
```

If your chosen H3 trainer uses different flags, update `tools/h3_ref2va.py` or set `H3_PREPARE_CMD` / `H3_TRAIN_CMD` to wrapper scripts.

For checkpoint saving, the adapter defaults to passing:

```bash
--save-every 20 --checkpoint-dir artifacts/h3/runs/ref2va-lora-smoke/checkpoints
```

If your H3 trainer uses different checkpoint flags, set:

```bash
export H3_CHECKPOINT_ARGS="--your-save-flag 20 --your-output-flag /path/to/checkpoints"
```

## Benchmark Plan

Create a fixed generation/eval task list for all five checkpoints:

```bash
cd /Users/sachin/h3-avatar-aq
python3 scripts/make_checkpoint_eval_plan.py
```

This writes `artifacts/h3/checkpoint_eval_plan.jsonl`. Use those rows as the
sample list for generation so every checkpoint is judged on the same prompts.

The benchmark suite scores:

- `identity`: same-character visual consistency
- `talking_motion`: static and active talking motion quality
- `audio_sync`: speech presence and lip-sync plausibility
- `prompt_following`: behavior follows prompt without losing identity
- `generalization`: held-out `ref_id` characters still work

Record a checkpoint review:

```bash
python3 scripts/record_checkpoint_eval.py \
  --checkpoint ckpt_020 \
  --step 20 \
  --identity 0.40 \
  --talking-motion 0.35 \
  --audio-sync 0.35 \
  --prompt-following 0.30 \
  --generalization 0.25 \
  --n 8 \
  --notes "decodes, identity weak"
```

Then render the progress table:

```bash
python3 scripts/benchmark_report.py
open benchmarks/progress_report.md
```

`record_checkpoint_eval.py` also updates `artifacts/h3/avatar_gate.json`, so
`/Users/sachin/aq/aq/bin/aq eval` reflects the latest reviewed checkpoint.

## Eval And AQIT

After generating validation samples, write the AQ gate:

```bash
python3 scripts/write_avatar_gate.py \
  --identity 0.8 \
  --talking-motion 0.7 \
  --audio-sync 0.7 \
  --prompt-following 0.75 \
  --generalization 0.65 \
  --n 20

/Users/sachin/aq/aq/bin/aq eval
```

Use AQIT after the gate fails or drifts:

```bash
aqit help
cd /Users/sachin/AQIT
aqit eval custom --prompts /Users/sachin/h3-avatar-aq/aqit/probes/avatar_prompts.jsonl
aqit check weight --checkpoint /Users/sachin/h3-avatar-aq/artifacts/checkpoints/last.json
aqit diff weight --checkpoint /Users/sachin/h3-avatar-aq/artifacts/checkpoints/last.json
```

For H3 specifically, the most important gate is reference conditioning: outputs should keep identity when the reference image changes, and should not collapse to training characters.
