import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import gradio as gr

# Import from generate.py
from model.generate import (
    render_checkpoint,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    SOUNDFONT_PATH,
    INSTRUMENT_NAMES,
    NAME_TO_INSTRUMENT,
)

def discover_checkpoints():
    step_files = sorted(
        CHECKPOINT_DIR.glob("checkpoint_step*.pt"),
        key=lambda p: int(p.stem.replace("checkpoint_step", "")),
    )
    if step_files:
        return step_files[-1].name
    latest = CHECKPOINT_DIR / "checkpoint.pt"
    return latest.name if latest.exists() else None


DEFAULT_CHECKPOINT = discover_checkpoints()


def run_generation(max_new_tokens, temperature, top_k, selected_instruments, progress=gr.Progress()):
    if not DEFAULT_CHECKPOINT:
        return None, None, " No checkpoint found in checkpoints/. Train the model first, or place a checkpoint_step*.pt file there."

    checkpoint_path = CHECKPOINT_DIR / DEFAULT_CHECKPOINT
    if not checkpoint_path.exists():
        return None, None, f"Checkpoint not found on disk: {checkpoint_path}"

    allowed_instruments = None
    if selected_instruments:
        allowed_instruments = [NAME_TO_INSTRUMENT[name] for name in selected_instruments]

    inst_desc = ", ".join(selected_instruments) if selected_instruments else "all"
    log_lines = [
        f"$ generating from {DEFAULT_CHECKPOINT}  (temp={temperature:.2f}, top_k={int(top_k)}, "
        f"max_tokens={int(max_new_tokens)}, instruments={inst_desc})"
    ]

    start = time.time()

    try:
        progress(0.1, desc="Loading model & generating...")

        # Call the render function
        result_path = render_checkpoint(
            checkpoint_path,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            soundfont_path=SOUNDFONT_PATH,
            allowed_instruments=allowed_instruments,
        )

        log_lines.append(f" render_checkpoint returned: {result_path}")

    except Exception as e:
        import traceback
        log_lines.append(f" generation raised an exception: {e}")
        log_lines.append(traceback.format_exc())
        return None, None, "\n".join(log_lines)

    elapsed = time.time() - start
    progress(1.0, desc="Done!")

    # Get all files in output directory
    all_files = list(OUTPUT_DIR.glob("*"))
    log_lines.append(f" All files in {OUTPUT_DIR}: {[f.name for f in all_files]}")

    # Look for MIDI and MP3 files
    midi_files = [f for f in all_files if f.suffix == '.mid']
    mp3_files = [f for f in all_files if f.suffix == '.mp3']

    log_lines.append(f" Found MIDI files: {[f.name for f in midi_files]}")
    log_lines.append(f" Found MP3 files: {[f.name for f in mp3_files]}")

    # Get the most recent files
    if midi_files:
        midi_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        possible_midi = midi_files[0]
        log_lines.append(f"✓ Most recent MIDI: {possible_midi.name}")
    else:
        possible_midi = None
        log_lines.append(" No MIDI files found")

    if mp3_files:
        mp3_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        possible_mp3 = mp3_files[0]
        log_lines.append(f"Most recent MP3: {possible_mp3.name}")
    else:
        possible_mp3 = None
        log_lines.append(" No MP3 files found")

    log_lines.append(f"Generation completed in {elapsed:.1f}s")

    # Return the files if they exist
    mp3_to_return = str(possible_mp3) if possible_mp3 else None
    midi_to_return = str(possible_midi) if possible_midi else None

    if not mp3_to_return and not midi_to_return:
        log_lines.append(" No output files were found in the output directory!")
        return None, None, "\n".join(log_lines)

    return mp3_to_return, midi_to_return, "\n".join(log_lines)

CUSTOM_CSS = """
.gradio-container { background: #161414 !important; }
#title-md h1 { color: #EDE7DD; font-weight: 700; letter-spacing: 0.02em; }
#title-md p { color: #A69E8F; }
.gr-button-primary { background: #E8A33D !important; border: none !important; color: #161414 !important; font-weight: 700 !important; }
.gr-button-primary:hover { background: #d4912e !important; }
"""

with gr.Blocks(title="RIFF//TRANSFORMER") as demo:
    gr.Markdown(
        "# RIFF // TRANSFORMER\n"
        "Decoder-only Transformer trained from scratch on **LPD-5 Cleansed** to generate rock MIDI. "
        "Generate a new take — this calls the real `generate.py` pipeline directly.",
        elem_id="title-md",
    )

    with gr.Row():
        with gr.Column(scale=1):
            if DEFAULT_CHECKPOINT:
                gr.Markdown(f"**Using checkpoint:** `{DEFAULT_CHECKPOINT}`")
            else:
                gr.Markdown("**No checkpoint found!** Please train the model first.")

            max_tokens_sl = gr.Slider(512, 16384, value=4096, step=256, label="Max new tokens",
                                      info="More tokens = longer generation")
            temperature_sl = gr.Slider(0.5, 1.5, value=0.9, step=0.05, label="Temperature",
                                       info="Higher = more random, lower = more predictable")
            top_k_sl = gr.Slider(10, 200, value=100, step=5, label="Top-k", info="Number of top tokens to sample from")

            instruments_cbg = gr.CheckboxGroup(
                choices=list(INSTRUMENT_NAMES.values()),
                value=list(INSTRUMENT_NAMES.values()),  # all selected by default
                label="Instruments",
                info="Uncheck any you want silenced in the output — e.g. just Drums, or Drums + Bass. "
                     "The model still generates the full arrangement; this only controls what's kept.",
            )

            generate_btn = gr.Button("▶ Generate track", variant="primary", size="lg")

        with gr.Column(scale=1):
            audio_out = gr.Audio(label="Generated audio", type="filepath")
            midi_out = gr.File(label="MIDI file")
            log_out = gr.Textbox(label="Generation log", lines=15, interactive=False)

    generate_btn.click(
        fn=run_generation,
        inputs=[max_tokens_sl, temperature_sl, top_k_sl, instruments_cbg],
        outputs=[audio_out, midi_out, log_out],
    )

if __name__ == "__main__":
    # Add the output directory to allowed paths so Gradio can serve the files
    demo.launch(
        share=True,
        css=CUSTOM_CSS,
        allowed_paths=[str(OUTPUT_DIR)]
    )