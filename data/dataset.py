import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from tqdm import tqdm


class MIDITokenDataset(Dataset):
    def __init__(self, token_folders, block_size, split_ratio=0.9, seed=42):

        self.block_size = block_size

        # Handle both single folder and list of folders
        if isinstance(token_folders, (str, Path)):
            token_folders = [token_folders]

        # Collect all file paths first
        all_files = []
        print(f"Scanning {len(token_folders)} batch folders...")
        for folder in token_folders:
            folder_path = Path(folder)
            json_files = list(folder_path.glob("*.json"))
            print(f"  {folder_path.name}: {len(json_files)} files")
            all_files.extend(json_files)

        # Shuffle files
        rng = random.Random(seed)
        rng.shuffle(all_files)

        n_train_files = int(split_ratio * len(all_files))
        train_files = all_files[:n_train_files]
        val_files = all_files[n_train_files:]

        print(f"Songs: {len(all_files)} total -> {len(train_files)} train / {len(val_files)} val")

        self.train_data, self.train_song_starts = self._load_files(train_files, desc="Loading train")
        self.val_data, self.val_song_starts = self._load_files(val_files, desc="Loading val")

        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")

    @staticmethod
    def _load_files(file_paths, desc):

        data = []
        song_starts = []
        for file_path in tqdm(file_paths, desc=desc):
            try:
                with open(file_path, 'r') as f:
                    tokens = json.load(f)
                    song_starts.append(len(data))
                    data.extend(tokens)
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        return data, song_starts

    def _valid_start_indices(self, data, song_starts, block_size):
        song_starts_sorted = sorted(song_starts)
        valid = []
        n = len(data)
        for idx, start in enumerate(song_starts_sorted):
            end = song_starts_sorted[idx + 1] if idx + 1 < len(song_starts_sorted) else n
            # valid group-boundary positions within [start, end), respecting block_size
            i = start
            while i + block_size <= end and i + block_size <= n:
                valid.append(i)
                i += 5
        return valid

    def get_batch(self, split, batch_size, block_size, device):
        if split == 'train':
            data, song_starts = self.train_data, self.train_song_starts
            cache_attr = '_train_valid_starts'
        else:
            data, song_starts = self.val_data, self.val_song_starts
            cache_attr = '_val_valid_starts'

        cached = getattr(self, cache_attr, None)
        if cached is None or cached[1] != block_size:
            valid_starts = self._valid_start_indices(data, song_starts, block_size)
            setattr(self, cache_attr, (valid_starts, block_size))
        else:
            valid_starts = cached[0]

        chosen = [valid_starts[k] for k in torch.randint(len(valid_starts), (batch_size,))]
        x = torch.stack([torch.tensor(data[i:i + block_size], dtype=torch.long) for i in chosen])
        y = torch.stack([torch.tensor(data[i + 1:i + block_size + 1], dtype=torch.long) for i in chosen])
        x, y = x.to(device), y.to(device)
        return x, y
