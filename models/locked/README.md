# Locked Model Adapter

This directory contains the locked MiniMax-H3 Ref2VA LoRA adapter:

`h3-avatar-audio-v2-step1000.pt`

## Summary

- Base model: MiniMax-H3 Ref2VA
- Adapter type: LoRA
- Trainable parameters: 10,655,744
- Dataset: 385 train clips from 481 accepted English transcript-conditioned clips
- Training: 1000 optimizer steps, approximately 2.6 passes over the train split
- Resolution/cache: 384x384, 56 frames
- Audio loss scale: 2.0

## Why This Checkpoint

This checkpoint was selected after generated samples showed usable English model audio, improved lip sync, and stable visual identity. The key change was training with prompts that include the actual ASR transcript text, instead of only generic English speech captions.

See `recipes/locked_h3_avatar_checkpoint.md` for the full locked recipe and known-good sample paths.

