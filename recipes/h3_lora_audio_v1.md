# H3 Ref2VA LoRA Audio V1

Goal: make the LoRA useful when H3 itself generates the audio, with clear English-ish speech, audible voice, and mouth motion that follows the generated sound. This run is not using external TTS as the final audio source.

## Dataset Order

1. Start from TA2.0 only.
2. Keep clips tagged `language=en` and `motion_mode=active`.
3. Require at least 124 frames so the model sees multi-second speech.
4. Prefer clean source audio with `snr >= 8`, `c50 >= 10`, and `vqa_score >= 0.45`.
5. Cap to 2 clips per reference identity so one character cannot dominate the LoRA.

Manifest target: `/runs/manifests/ta20_audio_v1_en_500`.

## Cache

Use longer cached samples than the visual smoke run:

- resolution: `384x384`
- frames: `56`
- train rows: all curated train rows
- cache: `/runs/cache/ta20_audio_v1_en_500_train_384_56`

## Training

Start conservative:

- output: `/runs/h3/ref2va-lora-audio-v1-en-384-56-r16`
- max steps: `1500`
- checkpoints: every `500`
- LoRA rank: existing trainer rank 16
- lr: `5e-5`
- audio loss scale: `2.0`

## Eval

At every checkpoint:

- Generate exactly 3 clips with model audio enabled.
- Use fixed references, seeds, prompts, and 30 inference steps.
- Track audio presence, duration, loudness/RMS, subjective English clarity, lip sync, identity retention, and artifacts.

Pass target for this run:

- voice is audible without external audio replacement
- generated speech sounds closer to English than the previous Korean/non-English outputs
- mouth opens/closes in plausible rhythm with generated voice
- identity is no worse than the visual-v2 1500 checkpoint

If audio is still non-English/noisy by checkpoint 1500, do not brute-force longer training. Next move is a stricter transcript/phoneme-ready dataset pass or H3 audio-head-only/dual-adapter experiment.
