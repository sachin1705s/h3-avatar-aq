# H3 Visual V2 Dataset Sources

Goal: improve the LoRA by training on cleaner visual motion and judging it on out-of-domain references like the Minion clip, without relying on H3-generated audio.

## Immediate Choice

Use a curated subset of `neosapience/TA2.0_animation_dataset` first.

Why:

- already downloaded on Modal;
- permissive license;
- directly matches animated talking-avatar references;
- metadata includes identity-ish grouping (`ref_id`), motion mode, language, VQA score, C50/SNR, duration/frame counts, and padding fields;
- big enough to filter aggressively instead of training on the whole noisy dump.

Risk:

- synthetic/anime-heavy;
- may not teach enough 3D mascot/cartoon behavior;
- quality scores are only proxies, so visual eval still decides.

## Candidate Datasets

| dataset | license/access | best use | decision |
| --- | --- | --- | --- |
| `neosapience/TA2.0_animation_dataset` | Apache-2.0/open | animated talking-avatar LoRA data | use now, curated only |
| `Shofo/shofo-talking-head-en` | gated/approval | real English talking-head motion and framing | request access; do not block v2 |
| `EngineeringAI-LAB/3DTalkingDataset` | Apache-2.0/open | realistic talking-head/lip/head-pose motion priors and eval references | sample later; too human-centric for primary character LoRA |
| `FreedomIntelligence/TalkVid` | CC-BY-NC-4.0/open | broad multilingual talking-head benchmark/eval | eval/research only unless usage is non-commercial |

## V2 Curation Rules

The new `ta20_visual_v2_1k` manifest does this:

- require local target video files;
- prefer active motion;
- prefer English rows where present;
- prefer higher `vqa_score`, `c50`, and `snr`;
- reject very short clips;
- penalize heavy padding;
- cap each `ref_id` to avoid one character dominating;
- split by `ref_id`, so near-same identity does not leak across train/val/test.

## Next Training Dataset Ladder

| stage | clips | purpose |
| --- | ---: | --- |
| `ta20_visual_v2_1k` | 1000 | quick v2 proof run, 384px, r16 |
| `ta20_visual_v2_2500` | 2500 | better full run if 1k beats base/v1 |
| `mixed_visual_v3` | 2500-5000 | add approved Shofo/3DTalking samples and mascot/cartoon augment refs |

## Acceptance Target

Before scaling beyond 1k clips, the checkpoint must beat the old 8000-step LoRA and base H3 on:

- identity on anime refs;
- motion without face collapse;
- temporal stability;
- out-of-domain Minion/mascot reference usefulness;
- validation loss not worsening across two checkpoints without visual improvement.

