# H3 Ref2VA LoRA Audio V2: Transcript Conditioning

Audio-v1 improved lip timing but did not produce reliable English because the dataset only had `language=en`, not the actual spoken words. The next audio run must train on prompts that include transcripts.

## Goal

The model should generate audible speech where:

- lip motion stays synchronized
- output sounds speech-like
- short English phrases are at least partially intelligible
- identity quality does not regress from visual-v2

## Required Dataset Upgrade

For each English TA2.0 clip:

1. Extract audio.
2. Run ASR to get transcript and confidence.
3. Keep clips with:
   - English transcript
   - transcript length between 3 and 20 words
   - ASR confidence above threshold
   - SNR/C50 filters from audio-v1
4. Put the actual transcript into the prompt:

`<Video 1> The same character says: "..." with natural voice and synchronized mouth movement.`

## Training Change

Use the same cache settings as audio-v1 initially:

- 384x384
- 56 frames
- LoRA rank 16
- audio loss scale 2.0

But choose checkpoints by generated transcript intelligibility, not video validation loss alone.

## UI Behavior

The local UI should expose:

- model audio mode: current H3 generated audio
- transcript-conditioned checkpoint once audio-v2 exists
- no-audio only as a diagnostic option, not the default

Until audio-v2 is trained, UI output audio is experimental and should be treated as placeholder speech, not production narration.
