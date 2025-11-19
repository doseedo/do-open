import torch
from torch.utils.data import DataLoader
from audiocraft.models import MusicGen
from datasets import ChromaTokenDataset
from pathlib import Path
from tqdm import tqdm
import time

# === Config ===
INDEX_PATH = "/home/arlo/Data/train_index.jsonl"
BATCH_SIZE = 1
NUM_EPOCHS = 3
LR = 1e-4
SAVE_EVERY = 500
DEVICE = "cuda"

# === Load model ===
model = MusicGen.get_pretrained("facebook/musicgen-melody")
model.to(DEVICE)
model.train()

# === Dataset & Loader ===
dataset = ChromaTokenDataset(INDEX_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
steps_per_epoch = len(loader)

# === Optimizer ===
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# === Training ===
global_step = 0
for epoch in range(NUM_EPOCHS):
    print(f"\n🌍 Epoch {epoch + 1}/{NUM_EPOCHS} — {steps_per_epoch} steps")
    epoch_start = time.time()

    running_loss = 0
    loader_tqdm = tqdm(loader, total=steps_per_epoch)

    for batch in loader_tqdm:
        chroma = batch["chroma"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)

        optimizer.zero_grad()
        loss = model.forward_with_chroma_targets(chroma=chroma, target_codes=tokens)
        loss.backward()
        optimizer.step()

        global_step += 1
        running_loss = 0.98 * running_loss + 0.02 * loss.item() if global_step > 1 else loss.item()

        loader_tqdm.set_description(f"[Step {global_step}] Loss: {loss.item():.4f} (avg: {running_loss:.4f})")

        if global_step % SAVE_EVERY == 0:
            ckpt_path = f"musicgen_finetune_step{global_step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"\n💾 Saved checkpoint: {ckpt_path}")

    elapsed = time.time() - epoch_start
    print(f"⏱️  Epoch {epoch+1} finished in {elapsed/60:.2f} min")

print("✅ Training complete.")
