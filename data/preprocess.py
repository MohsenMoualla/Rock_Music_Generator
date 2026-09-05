import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sparse


PAD = 0
SONG_START = 1

DELTA_TIME_MAX = 512
TIME_OFFSET = 2
PITCH_OFFSET = TIME_OFFSET + DELTA_TIME_MAX + 1
VELOCITY_OFFSET = PITCH_OFFSET + 128
INSTRUMENT_OFFSET = VELOCITY_OFFSET + 128
DURATION_OFFSET = INSTRUMENT_OFFSET + 5

VOCAB_SIZE = DURATION_OFFSET + 128

# LPD-5 program order: Drums, Piano, Guitar, Bass, Strings (index 0-4)
NUM_INSTRUMENTS = 5

def extract_note_events(npz_path):
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Error loading {npz_path.name}: {e}")
        return None

    events = []

    for inst in range(NUM_INSTRUMENTS):
        indices = data.get(f'pianoroll_{inst}_csc_indices')
        indptr = data.get(f'pianoroll_{inst}_csc_indptr')
        shape = data.get(f'pianoroll_{inst}_csc_shape')
        values = data.get(f'pianoroll_{inst}_csc_data')

        if indices is None or len(indices) == 0:
            continue

        shape = tuple(shape.tolist()) if isinstance(shape, np.ndarray) else tuple(shape)

        try:
            sparse_mat = sparse.csc_matrix((values, indices, indptr), shape=shape)
            dense = sparse_mat.toarray()
        except Exception as e:
            print(f"Error reconstructing matrix for inst {inst} in {npz_path.name}: {e}")
            continue

        max_val = dense.max() if dense.size else 0
        needs_scaling = 0 < max_val <= 1.0

        n_steps, n_pitches = dense.shape
        for p in range(n_pitches):
            col = dense[:, p]
            nonzero = col > 0
            if not nonzero.any():
                continue

            onsets = np.where(nonzero & ~np.concatenate(([False], nonzero[:-1])))[0]

            for t in onsets:
                v = col[t]
                dur = 0
                while t + dur < n_steps and col[t + dur] > 0:
                    dur += 1

                velocity_scaled = int(round(v * 127)) if needs_scaling else int(v)
                velocity_scaled = max(0, min(velocity_scaled, 127))

                events.append((
                    int(t),
                    int(p),
                    velocity_scaled,
                    inst,
                    min(dur, 127),
                ))

    if not events:
        return None

    events.sort(key=lambda e: (e[0], e[3], e[1]))
    return events


def events_to_tokens(events):
    tokens = [SONG_START]
    prev_t = 0

    for t, p, v, inst, dur in events:
        delta = t - prev_t
        prev_t = t


        delta_clamped = max(0, min(delta, DELTA_TIME_MAX))

        tokens.extend([
            delta_clamped + TIME_OFFSET,
            p + PITCH_OFFSET,
            v + VELOCITY_OFFSET,
            inst + INSTRUMENT_OFFSET,
            dur + DURATION_OFFSET,
        ])

    return tokens


def process_file(npz_path):
    events = extract_note_events(npz_path)
    if events is None:
        return None
    return events_to_tokens(events)


def main():
    # Paths
    base_path = Path("../data/raw_midi/lpd_5_cleansed/lpd_5/lpd_5_cleansed")
    output_path = Path("../data/processed_tokens")
    output_path.mkdir(parents=True, exist_ok=True)

    npz_files = list(base_path.glob("**/*.npz"))
    print(f"Found {len(npz_files)} .npz files")
    print(f"Vocab size needed: {VOCAB_SIZE}")

    success = 0
    failed = 0

    for npz_file in tqdm(npz_files, desc="Converting"):
        try:
            tokens = process_file(npz_file)
            if tokens:
                json_file = output_path / f"{npz_file.stem}.json"
                with open(json_file, 'w') as f:
                    json.dump(tokens, f)
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"Error with {npz_file.name}: {e}")

    print(f"\n Success: {success}")
    print(f" Failed: {failed}")

    saved_files = list(output_path.glob("*.json"))
    print(f"Total JSON files: {len(saved_files)}")


if __name__ == "__main__":
    main()
