# H3 Avatar

Locked MiniMax-H3 Ref2VA LoRA for single-image talking-avatar generation with model-generated audio.

## What Is Included

- `models/locked/h3-avatar-audio-v2-step1000.pt`  
  Original locked training checkpoint, including optimizer state.
- `partner_bundle/adapter_model.safetensors`  
  Clean LoRA-only adapter export for hosting partners.
- `partner_bundle/adapter_config.json`  
  LoRA architecture and base-model compatibility notes.
- `partner_bundle/inference_config.json`  
  Default request/serving parameters.
- `modal_h3_avatar.py`  
  Current Modal reference serving implementation.
- `ui/h3_avatar_ui.py`  
  Local UI for reference image, dialogue, direction, and background.

## Locked Model

- Base model: MiniMax-H3 Ref2VA
- Adapter type: LoRA
- Rank: 16
- Alpha: 32
- Target modules: `to_qkv`, `to_out.0`, `linear_1`, `linear_2`
- Trainable params: 10,655,744
- Training data: 385 training clips from 481 accepted transcript-conditioned clips
- Training length: 1000 optimizer steps, about 2.6 train-set passes
- Training cache: 384x384, 56 frames

## Local UI

```bash
H3_WARM_ENDPOINT_URL=https://aquin--h3-avatar-warm-generate.modal.run \
python3 ui/h3_avatar_ui.py
open http://127.0.0.1:8787
```

The UI is intentionally simple:

- reference image
- dialogue
- direction
- background

Everything else is locked to the selected model recipe.

## Modal Reference Deployment

```bash
modal deploy modal_h3_avatar.py
```

The deployed endpoint currently expects the local UI to send a 512px reference video as base64 and returns the generated MP4 as base64.

Current deployed endpoint:

`https://aquin--h3-avatar-warm-generate.modal.run`

## Partner Hosting Notes

For Reactor/fal/another inference partner, use `partner_bundle/`.

The expected production setup is a dedicated hosted model variant, not arbitrary dynamic LoRA upload per request:

```text
base MiniMax-H3 Ref2VA + h3-avatar-audio-v2 adapter
```

Recommended product modes:

- Preview: 5 seconds, fewer steps if the host supports a fast path.
- Final: 10 seconds, 30 steps or host-optimized equivalent.

Runtime compatibility matters more than file size. The adapter is small; the heavy part is serving the H3 base model efficiently.
