import torch
import json
import subprocess
import argparse
from pathlib import Path
from transformer import GPTLanguageModel
from miditoolkit import MidiFile, Instrument, Note
import sys
sys.path.append(str(Path(__file__).parent.parent))


from data.preprocess import (
    VOCAB_SIZE,
    PAD,
    SONG_START,
    DELTA_TIME_MAX,
    TIME_OFFSET,
    PITCH_OFFSET,
    VELOCITY_OFFSET,
    INSTRUMENT_OFFSET,
    DURATION_OFFSET,
    NUM_INSTRUMENTS,
)


N_EMBD = 768
N_HEAD = 12
N_LAYER = 8
BLOCK_SIZE = 512
DROPOUT = 0.25


STEPS_PER_BEAT = 24
TICKS_PER_BEAT = 480
TICKS_PER_STEP = TICKS_PER_BEAT // STEPS_PER_BEAT  # e.g. 480/24 = 20

# Paths
CHECKPOINT_DIR = Path("../checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pt"  # default/latest checkpoint
OUTPUT_DIR = Path("../outputs")

SOUNDFONT_PATH = Path("../soundfonts/FluidR3_GM.sf2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# LPD-5 instrument order: Drums, Piano, Guitar, Bass, Strings
INSTRUMENT_PROGRAMS = {
    0: {"program": 0, "is_drum": True},    # Drums
    1: {"program": 0, "is_drum": False},   # Piano (Acoustic Grand)
    2: {"program": 27, "is_drum": False},  # Guitar
    3: {"program": 33, "is_drum": False},  # Bass (Acoustic/Electric)
    4: {"program": 29, "is_drum": False},  # Strings (Ensemble) — LPD-5's "Strings" slot
}

INSTRUMENT_NAMES = {
    0: "Drums",
    1: "Piano",
    2: "Guitar",
    3: "Bass",
    4: "Strings",
}
NAME_TO_INSTRUMENT = {v: k for k, v in INSTRUMENT_NAMES.items()}


# ============ LOAD MODEL ============
def load_model(checkpoint_path=CHECKPOINT_PATH):
    print(f"Loading model from {checkpoint_path.name}...")
    model = GPTLanguageModel(
        VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT
    )
    model = model.to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    step = checkpoint['step']
    print(f"✓ Model loaded from step {step}")
    return model, step


# ============ GENERATE TOKENS ============
@torch.no_grad()
def generate(model, tokens_out_path, prompt_tokens=None, max_new_tokens=4096, temperature=0.8, top_k=40):
    """Generate a token sequence from the model."""
    if prompt_tokens is None:
        prompt_tokens = [SONG_START]

    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)
    idx = model.generate(idx, max_new_tokens, temperature, top_k)

    tokens = idx[0].tolist()
    print(f"Generated {len(tokens)} tokens")

    with open(tokens_out_path, 'w') as f:
        json.dump(tokens, f)
    print(f"Tokens saved to {tokens_out_path}")

    return tokens


# ============ DECODE TOKENS ============
def decode_tokens(tokens):

    base_stream = tokens[1:] if tokens and tokens[0] == SONG_START else tokens[:]

    best_offset = 0
    best_valid_count = -1
    best_events = []

    for offset in range(5):
        stream = base_stream[offset:]
        n_groups = len(stream) // 5
        if n_groups == 0:
            continue

        events = []
        abs_time = 0
        valid_count = 0

        for g in range(n_groups):
            time_tok, pitch_tok, vel_tok, inst_tok, dur_tok = stream[g * 5: g * 5 + 5]

            delta = time_tok - TIME_OFFSET
            pitch = pitch_tok - PITCH_OFFSET
            velocity = vel_tok - VELOCITY_OFFSET
            instrument = inst_tok - INSTRUMENT_OFFSET
            duration = dur_tok - DURATION_OFFSET

            valid = (
                0 <= delta <= DELTA_TIME_MAX
                and 0 <= pitch <= 127
                and 0 <= velocity <= 127
                and 0 <= instrument < NUM_INSTRUMENTS
                and 0 <= duration <= 127
            )

            if not valid:
                continue

            valid_count += 1
            abs_time += delta
            events.append((abs_time, pitch, max(velocity, 1), instrument, max(duration, 1)))

        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_offset = offset
            best_events = events

    total_groups = len(base_stream) // 5
    dropped = total_groups - best_valid_count
    if best_offset != 0:
        print(f"  (note: generation was phase-shifted by {best_offset} token(s) — auto-corrected)")
    if dropped:
        print(f"  (skipped {dropped}/{total_groups} malformed token groups, offset={best_offset})")

    return best_events


# ============ CONVERT TOKENS TO MIDI ============
def tokens_to_midi(tokens, output_path, allowed_instruments=None):
    events = decode_tokens(tokens)

    if allowed_instruments is not None:
        allowed_instruments = set(allowed_instruments)
        events = [e for e in events if e[3] in allowed_instruments]

    if not events:
        print(" No valid note events decoded — nothing to write.")
        return False

    midi = MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    tracks = {}
    for inst_id, cfg in INSTRUMENT_PROGRAMS.items():
        if allowed_instruments is not None and inst_id not in allowed_instruments:
            continue
        tracks[inst_id] = Instrument(program=cfg["program"], is_drum=cfg["is_drum"])

    for abs_time, pitch, velocity, instrument, duration in events:
        start_tick = abs_time * TICKS_PER_STEP
        end_tick = start_tick + max(duration * TICKS_PER_STEP, TICKS_PER_STEP)

        note = Note(
            velocity=velocity,
            pitch=pitch,
            start=start_tick,
            end=end_tick,
        )
        tracks[instrument].notes.append(note)

    for inst_id, track in tracks.items():
        if track.notes:
            midi.instruments.append(track)
            print(f"  ✓ Instrument {inst_id}: {len(track.notes)} notes")

    if not midi.instruments:
        print("No notes generated for the selected instrument(s).")
        return False

    midi.dump(str(output_path))
    print(f"MIDI saved with {len(midi.instruments)} instrument tracks")
    return True


# ============ RENDER MIDI -> MP3 ============
def midi_to_mp3(midi_path, mp3_path, soundfont_path=SOUNDFONT_PATH, keep_wav=False):

    if not midi_path.exists():
        print(f" MIDI file not found at {midi_path} — skipping audio render.")
        return False

    if not soundfont_path.exists():
        print(f"Soundfont not found at {soundfont_path} — skipping audio render.")
        print("   Download a General MIDI .sf2 (e.g. FluidR3_GM.sf2) and update SOUNDFONT_PATH.")
        return False

    wav_path = mp3_path.with_suffix(".wav")
    wav_path.unlink(missing_ok=True)  # clear any stale file so we can trust its presence as a success signal

    fluidsynth_cmd = [
        "fluidsynth", "-ni", "-F", str(wav_path), "-r", "44100",
        str(soundfont_path), str(midi_path),
    ]

    try:
        result = subprocess.run(fluidsynth_cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(" 'fluidsynth' not found on PATH. Install with: conda install -c conda-forge fluidsynth")
        return False

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        print(" fluidsynth did not produce a wav file. Output:")
        print(result.stdout)
        print(result.stderr)
        return False

    mp3_path.unlink(missing_ok=True)
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(wav_path), str(mp3_path)]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(" 'ffmpeg' not found on PATH. Install with: conda install -c conda-forge ffmpeg")
        return False

    if not mp3_path.exists() or mp3_path.stat().st_size == 0:
        print(" ffmpeg did not produce an mp3 file. Output:")
        print(result.stderr)
        return False

    if not keep_wav:
        wav_path.unlink(missing_ok=True)

    print(f"Audio rendered to {mp3_path}")
    return True


def render_checkpoint(checkpoint_path, max_new_tokens=8192, temperature=0.9, top_k=60, soundfont_path=SOUNDFONT_PATH,
                      allowed_instruments=None, progress_callback=None):

    if progress_callback:
        progress_callback(0.1, "Loading model...")

    model, step = load_model(checkpoint_path)

    tag = f"step{step}"
    tokens_path = OUTPUT_DIR / f"tokens_{tag}.json"
    midi_path = OUTPUT_DIR / f"song_{tag}.mid"
    mp3_path = OUTPUT_DIR / f"song_{tag}.mp3"

    if progress_callback:
        progress_callback(0.3, "Generating tokens...")

    tokens = generate(model, tokens_path, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    if progress_callback:
        progress_callback(0.6, "Converting to MIDI...")

    if not tokens_to_midi(tokens, midi_path, allowed_instruments=allowed_instruments):
        return None

    if progress_callback:
        progress_callback(0.8, "Rendering audio...")

    audio_ok = midi_to_mp3(midi_path, mp3_path, soundfont_path=soundfont_path)

    if progress_callback:
        progress_callback(1.0, "Done!")

    return mp3_path if audio_ok else None

# ============ MAIN ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIDI/MP3 samples from trained checkpoints.")
    parser.add_argument(
        "--all", action="store_true",
        help="Render every checkpoint_step*.pt found in CHECKPOINT_DIR, not just the latest.",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument(
        "--instruments", type=str, default=None,
        help="Comma-separated list of instruments to keep, e.g. 'drums,bass' or '0,3'. "
             f"Names: {', '.join(INSTRUMENT_NAMES.values())}. Default: all instruments.",
    )
    args = parser.parse_args()

    allowed_instruments = None
    if args.instruments:
        allowed_instruments = []
        for tok in args.instruments.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.isdigit():
                allowed_instruments.append(int(tok))
            else:
                matched = NAME_TO_INSTRUMENT.get(tok.capitalize())
                if matched is None:
                    raise ValueError(
                        f"Unknown instrument '{tok}'. Use a number 0-4 or one of: "
                        f"{', '.join(INSTRUMENT_NAMES.values())}"
                    )
                allowed_instruments.append(matched)

    print("\n" + "=" * 50)
    print(" GENERATING ROCK MUSIC")
    print("=" * 50)
    if allowed_instruments is not None:
        names = [INSTRUMENT_NAMES[i] for i in allowed_instruments]
        print(f"   Instruments: {', '.join(names)}")

    if args.all:
        checkpoint_files = sorted(CHECKPOINT_DIR.glob("checkpoint*.pt"))
        if not checkpoint_files:
            print(f"No checkpoint files found in {CHECKPOINT_DIR}")
        for ckpt in checkpoint_files:
            print(f"\n--- Rendering {ckpt.name} ---")
            render_checkpoint(ckpt, args.max_tokens, args.temperature, args.top_k, allowed_instruments=allowed_instruments)
    else:
        mp3_path = render_checkpoint(
            CHECKPOINT_PATH, args.max_tokens, args.temperature, args.top_k,
            allowed_instruments=allowed_instruments,
        )
        if mp3_path:
            print("\n" + "=" * 50)
            print(" COMPLETE!")
            print("=" * 50)
            print(f" {mp3_path}")
        else:
            print("\n Audio render failed, but check outputs/ — the .mid file may still have been saved.")
            print("   See the fluidsynth/ffmpeg output above for the specific error.")