# H3 LoRA Visual V2 Recipe

## End Goal

Build a reusable H3 Ref2VA LoRA that makes a reference character look alive on video.

The model should:

- preserve the input character identity, outfit, colors, and framing;
- create plausible face, head, and mouth motion for a silent/talking-avatar clip;
- keep temporal consistency with no melting, face swaps, or identity drift;
- generalize across at least three reference styles: anime portrait, western/cartoon character, and 3D mascot/object-like character;
- run with external audio added later. H3-generated audio is not part of the success target.

The model should not:

- be selected because training loss alone improved;
- be judged from low-quality `256x256` / `6` step previews;
- rely on generated H3 audio;
- overfit to one repeated character, one crop, or one dataset style.

## Product Acceptance Bar

A checkpoint is usable only if it passes all of these gates:

| gate | target |
| --- | --- |
| visual identity | at least 8/12 eval clips preserve the reference character clearly |
| motion | at least 8/12 eval clips show visible but stable mouth/head motion |
| temporal stability | no more than 2/12 clips have major melting, flicker, or face collapse |
| generalization | passes at least 2/3 style buckets: anime, illustrated/cartoon, 3D/mascot |
| base comparison | LoRA beats base H3 on at least 7/12 clips |
| validation loss | not worse than the best checkpoint by more than 5% unless visual score is clearly better |
| audio | ignored for checkpoint selection; generated audio can fail without rejecting the visual LoRA |

## Dataset V2

Do not train on the whole TA2.0 dump directly.

Create a curated set:

| bucket | target clips | rule |
| --- | ---: | --- |
| clean anime talking heads | 600-1000 | centered face, visible mouth, stable camera |
| western/cartoon references | 300-600 | non-anime shapes, broader face/body diversity |
| 3D/mascot/toy-like references | 200-400 | round/simple characters, strong identity cues |
| hard negatives / reject holdout | 100-200 | blurry, duplicate, occluded, bad-mouth clips for eval only |

Filtering rules:

- remove near-duplicates by perceptual hash and CLIP/image embedding clustering;
- reject clips with tiny faces, heavy occlusion, extreme crops, subtitles/watermarks, or unstable camera;
- keep clips where mouth changes over time;
- keep diverse hair/eyes/body/clothing, not just blue-eyed closeups;
- cap each source character cluster so one character cannot dominate.

Split by character/source cluster, not random clip:

- train: 80%
- validation: 10%
- test: 10%

No near-duplicate character should appear in both train and eval.

## Training

Important correction from upstream docs:

- The current `IAmIronMan42/MiniMax-H3-FineTuning` trainer says reference media are not yet encoded into the conditioning sequence. That means it is not a complete faithful Ref2VA fine-tuning path for our end goal unless we add reference-row conditioning.
- SimpleTuner has MiniMax H3 adapter support and explicitly supports visual-only target mode with `minimax_h3_target_mode: "video"`.
- SimpleTuner also recommends H3 drift distillation to keep the adapter close to the frozen base model instead of drifting away from the distilled H3 behavior.
- The MiniMax integration index notes that FL2VA generally has better raw output quality, while Ref2VA accepts richer reference media but has lower base quality. So if our goal is a polished user-facing demo, we should benchmark FL2VA/I2V and Ref2VA separately instead of assuming Ref2VA is always better.

Recommended first serious run:

| setting | value |
| --- | --- |
| base model | MiniMax-H3 Ref2VA only if reference conditioning is fixed; otherwise SimpleTuner H3 FL2VA/visual adapter |
| target | visual LoRA only (`minimax_h3_target_mode: "video"` equivalent) |
| audio loss | disabled or low-weighted unless target-audio conditioning is verified |
| cache resolution | 384 first, then 512 for final |
| frames | model-compatible 124-frame eval; training cache can stay trainer-compatible |
| LoRA rank | test `r=16` and `r=32` |
| LR | start lower than v1; prefer `5e-5` to `1e-4` sweep |
| checkpoints | every 500 or 1000 steps |
| stop rule | stop when visual score plateaus or validation worsens twice |
| drift control | keep a frozen-base drift/distillation term enabled if using SimpleTuner |

Run two short sweeps before a full train:

1. **Resolution sweep:** same curated subset, `256` vs `384`.
2. **Capacity sweep:** `r=16` vs `r=32`, same steps.

Only run the full training after one sweep clearly wins.

## Eval

Every checkpoint gets the same eval refs and seeds.

Primary visual eval:

- `512x512`
- `30` inference steps
- `--no-generate-audio`
- same 12 held-out references
- same prompts and seeds

Eval set:

| group | count |
| --- | ---: |
| in-domain anime | 4 |
| out-of-domain cartoon/illustration | 3 |
| 3D/mascot/object-like | 3 |
| user-provided references | 2 |

Manual scorecard:

| metric | weight |
| --- | ---: |
| identity | 0.35 |
| talking/head motion | 0.30 |
| temporal stability | 0.20 |
| prompt/usefulness | 0.10 |
| audio | 0.05 diagnostic only |

Hard reject any checkpoint with:

- repeated face collapse on mascot/cartoon refs;
- identity drift across frames;
- no visible motion in most clips;
- visual score worse than base H3.

## AQ / AQIT Usage

AQ tracks the experiment as a gate:

- dataset version;
- training config hash;
- checkpoint step;
- validation loss;
- visual score;
- base-vs-LoRA result;
- AQIT weight summary.

AQIT checks:

- LoRA file exists and loads;
- trainable parameter count matches expected rank;
- LoRA norms are not exploding;
- compare checkpoint deltas across steps;
- flag checkpoints where norms spike while validation worsens.

## Decision Rule

Pick the earliest checkpoint that clears the product acceptance bar.

If none clear it, do not continue training the same recipe. Fix the data or objective first.

For v1, the likely diagnosis is:

- too much noisy/broad data;
- low-resolution training/eval path;
- audio branch distracting evaluation;
- no high-quality base-vs-LoRA gate;
- out-of-domain refs like Minion are not represented in training.

## Next Run Plan

1. Decide trainer path:
   - Patch the current trainer to include Ref2VA reference conditioning rows; or
   - Switch to SimpleTuner MiniMax H3 adapter training for a visual-only LoRA.
2. Build a curated `visual_v2_1k` set.
3. Generate the 12-reference eval pack, including the Minion image/video as user-provided out-of-domain eval.
4. Train a `384px r=16` smoke for 1000-2000 steps.
5. Run `512px / 30-step / no-audio` eval every 500-1000 steps.
6. Compare against base H3, v1 checkpoints, and an FL2VA baseline.
7. Only then scale to `visual_v2_full`.
