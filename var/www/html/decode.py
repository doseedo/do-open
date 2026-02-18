import os
import torch
import torchaudio
from pathlib import Path
from acestep.pipeline_ace_step import ACEStepPipeline

# === Config ===
LATENTS_DIR = Path("/mnt/msdd/dcae_latentsnew")
RECONSTRUCTED_DIR = Path("/mnt/msdd/reconstructed_wavs_overlap")
CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints"
USE_OVERLAP_DECODER = True  # ✅ Use better decoder by default
TARGET_SR = 48000  # Output sample rate
DOWNSAMPLE_FACTOR = 4096  # Should match your encode pipeline

# === Load Model ===
print("🔁 Loading model...")
pipeline = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR)
pipeline.load_checkpoint(CHECKPOINT_DIR)
model = pipeline.music_dcae.eval().to("cuda")

# === Utility ===
def reconstruct_wav_from_latent_file(pt_path: Path):
    try:
        data = torch.load(pt_path)
        latents = data["latents"].unsqueeze(0).to("cuda")  # [1, C, H, W]

        # Use more accurate length if stored
        if "latent_duration" in data:
            estimated_audio_len = int(data["latent_duration"] * TARGET_SR)
        else:
            latent_frames = data["length"]
            estimated_audio_len = int(latent_frames * DOWNSAMPLE_FACTOR * TARGET_SR / 44100)

        audio_len = torch.tensor([estimated_audio_len], device="cuda")

        # Decode audio
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            if USE_OVERLAP_DECODER:
                sr, wavs = model.decode_overlap(latents, audio_lengths=audio_len, sr=TARGET_SR)
            else:
                sr, wavs = model.decode(latents, audio_lengths=audio_len, sr=TARGET_SR)

        # Output path
        rel_path = pt_path.relative_to(LATENTS_DIR).with_suffix(".wav")
        out_path = (RECONSTRUCTED_DIR / rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(str(out_path), wavs[0].cpu(), sample_rate=sr)
        print(f"✅ {pt_path.name} → {out_path.name}")
    except Exception as e:
        print(f"❌ Failed to decode {pt_path.name}: {e}")

# === Entry Point ===
def main():
    latent_files = sorted(LATENTS_DIR.rglob("*.pt"))
    print(f"🎵 Found {len(latent_files)} latent files.")
    for pt in latent_files:
        reconstruct_wav_from_latent_file(pt)

if __name__ == "__main__":
    main()
