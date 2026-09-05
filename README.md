Transformer — Rock Music Generation

A decoder-only Transformer, trained from scratch, that generates rock music as MIDI — built end-to-end: raw piano-roll data → custom tokenizer → 57.7M-parameter Transformer → generated MIDI → rendered audio, with an interactive Gradio GUI on top.

Trained on [LPD-5 Cleansed](https://salu133445.github.io/lpd/) (Lakh Pianoroll Dataset), a five-track (Drums, Piano, Guitar, Bass, Strings) symbolic music dataset.

> Semester project — AI Department, Latakia University.

## What's actually here

This repo holds the **code**, not the trained model or generated data — checkpoints, the tokenized dataset, and generated audio/MIDI are excluded (see [Data & checkpoints](#data--checkpoints) below for why and how to reproduce them).

```
midi_generator/
├── data/
│   ├── preprocess.py      # Data preprocessing (npz → tokens)
│   └── dataset.py         # PyTorch Dataset with group-boundary alignment
├── model/
│   ├── transformer.py     # GPT-style Transformer model
│   ├── train.py           # Training loop with checkpoint resume
│   └── generate.py        # Generation → MIDI → MP3
├── checkpoints/           # Saved model checkpoints
├── outputs/               # Generated MIDI/MP3 files
└── soundfonts/            # FluidR3_GM.sf2 (for MP3 conversion)
```

## How it works

**Tokenization.** Each note event becomes 5 tokens: `TIME_DELTA`, `PITCH`, `VELOCITY`, `INSTRUMENT`, `DURATION`. Time is encoded as the *delta* since the previous note (not absolute time) — early versions that used absolute time wrapped around on long songs and scrambled the ordering of far-apart events; delta-time avoids that regardless of song length.

**Model.** Decoder-only Transformer (GPT-style): 768 embedding dim, 12 attention heads, 8 layers, 512-token context, ~57.7M parameters. Uses RMSNorm, weight-tied embeddings, PyTorch's `scaled_dot_product_attention` (Flash Attention), and gradient checkpointing to fit on a single RTX 3050.

**Training.** Cross-entropy with label smoothing, AdamW, mixed precision (AMP), gradient accumulation (effective batch size 36). A cosine learning-rate decay (peak 3e-4 → floor 1.5e-5) was added partway through training after loss plateaued at a flat learning rate.

**Generation.** Autoregressive sampling (temperature + top-k) starting from a `SONG_START` token, decoded back into note events, written to MIDI via `miditoolkit`, then rendered to audio via FluidSynth + FFmpeg. The GUI additionally lets you choose which of the 5 instrument tracks to keep in the final output (e.g. drums + bass only) — filtering happens after generation, so it doesn't change what the model predicts, only what makes it into the file.

## Results

| Metric | Value |
|---|---|
| Parameters | 57.7M |
| Training songs | 13,215 (of 21,425 in LPD-5 Cleansed — the rest had unusable/empty tracks) |
| Training steps | 80,000 |
| Final train / val loss | 1.83 / 1.87 |
| Valid token-group rate | 18% (step 4K) → 100% (step 60K+) |

"Valid token-group rate" measures how often a generated 5-token group actually decodes into a real note — this climbed steadily alongside loss and is a more direct signal of the model learning its own tokenization format than loss alone.

## Known bugs found & fixed along the way

- **Absolute-time wraparound** — encoding time as `t % TIME_MAX` scrambled event order in songs longer than the modulus. Fixed by switching to delta-time encoding.
- **Unordered events** — notes were tokenized per-instrument/per-pitch instead of by actual time order, producing a non-sequential token stream. Fixed by sorting all events by `(time, instrument)` before encoding.
- **Phase drift at generation time** — since training windows could start mid-token-group, the model sometimes generated sequences shifted by 1-4 tokens from a true group boundary, causing decode validity to collapse. Fixed two ways: (1) training samples are now constrained to start only at true group boundaries, and (2) the decoder tries all 5 possible offsets and keeps whichever yields the most valid notes.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need, as system binaries (not pip-installable):
- **FluidSynth** — renders MIDI to audio
- **FFmpeg** — converts the rendered WAV to MP3
- A **General MIDI soundfont** (e.g. `FluidR3_GM.sf2`) — place it at `soundfonts/FluidR3_GM.sf2`

See the comments at the top of `requirements.txt` for install commands per OS.

## Data & checkpoints

Not included in this repo (kept to code + docs only):

- **Raw dataset** — download [LPD-5 Cleansed](https://salu133445.github.io/lpd/) and place the `.npz` files under `data/raw_midi/lpd_5_cleansed/lpd_5/lpd_5_cleansed/`.
- **Tokenized data** — generate it yourself:
  ```bash
  python data/preprocess.py
  ```
- **Trained checkpoints** — train from scratch:
  ```bash
  python model/train.py
  ```
  Checkpoints save to `checkpoints/` every 1,000 steps (both a rolling `checkpoint.pt` for resuming, and a permanent `checkpoint_step{N}.pt` per interval).

All of the above paths are relative to the repo root by default — see the top of `data/preprocess.py`, `model/train.py`, and `model/generate.py` if you want to point them elsewhere.

## Running it

**Command line**, once you have a checkpoint:
```bash
python model/generate.py --max_tokens 8192 --temperature 0.9 --top_k 100
python model/generate.py --instruments drums,bass     # keep only these tracks
python model/generate.py --all                        # render every saved checkpoint
```

**GUI** — run from the repo root:
```bash
python app.py
```
Opens a local Gradio interface (usually `http://127.0.0.1:7860`) with sliders for generation parameters, checkboxes for which instruments to keep, and a plot of the valid-token-group-rate training curve.

## Possible next steps

- Genre-filter the training data for a purer rock-specific model (LPD-5 Cleansed isn't rock-only)
- Train past 80K steps — the model hasn't completed a full epoch over the dataset yet
- Relative positional attention (à la Music Transformer) for better long-range structure (verse/chorus-level patterns)
- Objective evaluation against real recordings, beyond the loss/valid-rate metrics and manual listening used here

## References

- Huang et al., [Music Transformer: Generating Music with Long-Term Structure](https://arxiv.org/abs/1809.04281) (2018)
- Dong et al., [MuseGAN](https://salu133445.github.io/lpd/) / Lakh Pianoroll Dataset (2018)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
