import torch
from pathlib import Path
import json
import math
from transformer import GPTLanguageModel
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.dataset import MIDITokenDataset
from data.preprocess import VOCAB_SIZE, SONG_START


# parameters
# Model architecture
N_EMBD = 768
N_HEAD = 12
N_LAYER = 8
BLOCK_SIZE = 512
DROPOUT = 0.25

# Training
BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS = 3  # Effective batch = 36
LEARNING_RATE = 3e-4
MAX_ITERS = 100000
EVAL_INTERVAL = 500
SAVE_INTERVAL = 1000
EVAL_ITERS = 200


WARMUP_ITERS = 1000
LR_DECAY_ITERS = MAX_ITERS
MIN_LR = LEARNING_RATE / 20


def get_lr(iter_num):
    if iter_num < WARMUP_ITERS:
        return LEARNING_RATE * (iter_num + 1) / WARMUP_ITERS
    if iter_num >= LR_DECAY_ITERS:
        return MIN_LR
    progress = (iter_num - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


DATA_PATH = Path("../data/processed_tokens")
CHECKPOINT_DIR = Path("../checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pt"

GENERATED_DIR = Path("../generated")

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


@torch.no_grad()
def estimate_loss(model, dataset, eval_iters, batch_size, block_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x1, y1 = dataset.get_batch(split, batch_size, block_size, device)
            with torch.amp.autocast('cuda'):
                logits, loss = model(x1, y1)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# checkpoints function
def save_checkpoint(model, optimizer, scaler, step, loss, dataset=None):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'train_data': dataset.train_data[:100] if dataset is not None else None,  # small sample for verification
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f" Checkpoint saved at step {step}, loss {loss:.4f}")


def load_checkpoint(model, optimizer, scaler):
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f" Resumed from step {checkpoint['step']}, loss {checkpoint['loss']:.4f}")
        return checkpoint['step']
    return 0


# training loop
def train():
    # Create dataset
    print("Loading dataset...")
    dataset = MIDITokenDataset(DATA_PATH, BLOCK_SIZE)

    # Create model
    print("Creating model...")
    model = GPTLanguageModel(
        VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT
    )
    model = model.to(DEVICE)

    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.2f}M parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # Resume from checkpoint
    start_step = load_checkpoint(model, optimizer, scaler)

    resume_lr = get_lr(start_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = resume_lr
    print(f"Learning rate set to {resume_lr:.2e} for step {start_step}")

    # Training loop
    model.train()
    step = start_step
    print(f"\nStarting training from step {step} to {MAX_ITERS}")

    for iter_num in range(start_step, MAX_ITERS):
        # Get batch
        xb, yb = dataset.get_batch('train', BATCH_SIZE, BLOCK_SIZE, DEVICE)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            _, loss = model(xb, yb)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (iter_num + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            # Update learning rate for this optimizer step before applying it
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Print progress
            if iter_num % 50 == 0:
                step_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                print(f"step {iter_num}: loss {step_loss:.4f}  lr {lr:.2e}")

        # Evaluate loss
        if iter_num > 0 and iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, dataset, EVAL_ITERS, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Save checkpoint
        if iter_num > 0 and iter_num % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, scaler, iter_num, loss.item(), dataset)

    # Final save
    save_checkpoint(model, optimizer, scaler, MAX_ITERS, loss.item(), dataset)
    print("Training complete")


# generate function
@torch.no_grad()
def generate(prompt_tokens=None, max_new_tokens=500, temperature=0.8, top_k=40, out_name="generated_tokens.json"):
    # Load model
    model = GPTLanguageModel(
        VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT
    )
    model = model.to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create prompt
    if prompt_tokens is None:
        prompt_tokens = [SONG_START]
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)

    # Generate
    idx = model.generate(idx, max_new_tokens, temperature, top_k)
    tokens = idx[0].tolist()

    # Save tokens
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_DIR / out_name
    with open(output_path, 'w') as f:
        json.dump(tokens, f)

    print(f"Generated {len(tokens)} tokens -> saved to {output_path}")
    return tokens


# main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'generate'], default='train')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max_tokens', type=int, default=500)
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        generate(max_new_tokens=args.max_tokens, temperature=args.temperature)
