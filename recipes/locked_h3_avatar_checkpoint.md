# Locked H3 Avatar Checkpoint

Locked on: 2026-10-03

## Winner

Use this as the default usable LoRA:

`/runs/h3/locked/h3-avatar-audio-v2-step1000.pt`

Original training checkpoint:

`/runs/h3/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/checkpoint-1000.pt`

## Why This One

- User-approved as visually good and audio usable.
- Generated speech was clear enough on two different references.
- Lip sync improved versus audio-v1.
- Step 1000 was chosen before further joint video/audio training could drift the already-good visuals.

## Training Recipe That Worked

- Source manifest: `/runs/manifests/ta20_audio_v1_en_500/ta20_audio_v1_manifest.jsonl`
- ASR/transcript manifest: `/runs/manifests/ta20_audio_v2_transcripts_en_500_w40`
- Accepted clips: 481
- Train/val/test split: 385 / 48 / 48
- Cache: `/runs/cache/ta20_audio_v2_transcripts_en_500_w40_train_384_56`
- Cache shape: 384x384, 56 frames, encoded video/audio/text
- LoRA output: `/runs/h3/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16`
- Learning rate: `5e-5`
- Audio loss scale: `2.0`
- Checkpoints saved: 500, 1000

## Known Good Samples

- `/runs/generated/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/ab_headshot_audio_v2_ckpt1000_5s/generated.mp4`
- `/runs/generated/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/boy_closeup_audio_v2_ckpt1000_5s/generated.mp4`

Local copies:

- `artifacts/generated/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/ab_headshot_audio_v2_ckpt1000_5s/generated.mp4`
- `artifacts/generated/ref2va-lora-audio-v2-transcripts-en500-w40-384-56-r16/boy_closeup_audio_v2_ckpt1000_5s/generated.mp4`

## Do Not Overwrite

Do not keep training this same run as the main path. Future experiments should write to a new output directory and should preserve this checkpoint as the stable baseline.

Next experimental direction: patch trainer for audio-focused tuning/freeze visual blocks instead of repeating joint visual/audio LoRA training.
