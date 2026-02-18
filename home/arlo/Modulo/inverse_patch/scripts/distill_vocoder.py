#!/usr/bin/env python3
"""
Vocoder Distillation: Train a structured DSP student to match the neural vocoder.

Teacher: DCAE neural vocoder (mel → audio, black box)
Student: Structured additive synth (sines + noise + interactions, interpretable)

Each piece of complexity that improves the student IS a discovered operation —
"bins 30-40 need this interaction term" is a discovery about frequency coupling.

Usage:
    python3 distill_vocoder.py --generate-data   # Step 1: create training pairs
    python3 distill_vocoder.py --train            # Step 2: train student
    python3 distill_vocoder.py --eval             # Step 3: evaluate + save .wav
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "vocoder_distill"
DATA_DIR = OUTPUT_DIR / "pairs"
MODEL_PATH = OUTPUT_DIR / "distilled_vocoder.pt"

SR = 44100
HOP_LENGTH = 512
N_MELS = 128


# ============================================================
# Mel bin frequencies (reused from sparse_vocoder.py)
# ============================================================

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def get_mel_frequencies(n_mels=128, f_min=40, f_max=16000):
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mels = np.linspace(mel_min, mel_max, n_mels)
    return mel_to_hz(mels)

MEL_FREQS = torch.tensor(get_mel_frequencies(128, 40, 16000), dtype=torch.float32)


# ============================================================
# Step 1: Generate training pairs
# ============================================================

def generate_training_data(max_samples=500, crop_frames=32):
    """Load latents, decode through DCAE+vocoder, save (mel, audio) pairs."""
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
    import orjson

    device = 'cuda'

    print("Loading DCAE + Vocoder...")
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH)
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    print(f"  Mel range: [{dcae.min_mel_value}, {dcae.max_mel_value}]")
    print(f"  Scale: {dcae.scale_factor}, Shift: {dcae.shift_factor}")

    # Collect latent paths from SMS manifest (already indexed, avoids slow GCS rglob)
    import orjson
    manifest_path = Path('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json')
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())
    latent_paths = [Path(e['latent_path']) for e in manifest['entries'][:max_samples]]

    print(f"Found {len(latent_paths)} latent files")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, lpath in enumerate(latent_paths):
        try:
            z = torch.load(lpath, weights_only=True, map_location='cpu')
            if isinstance(z, dict):
                z = z.get('latents', z.get('z_real', None))
                if z is None:
                    continue
            if z.dim() == 3:
                z = z.unsqueeze(0)  # [1, 8, 16, T]
            if z.dim() != 4 or z.shape[1] != 8 or z.shape[2] != 16:
                continue

            # Crop to fixed length
            T = z.shape[3]
            if T < crop_frames:
                continue
            # Random crop
            start = torch.randint(0, T - crop_frames, (1,)).item() if T > crop_frames else 0
            z = z[:, :, :, start:start + crop_frames].to(device)

            with torch.no_grad():
                # z → decoder → mel (raw, in [-1, 1])
                z_denorm = z / dcae.scale_factor + dcae.shift_factor
                mel_raw = dcae.dcae.decoder(z_denorm).mean(dim=1)  # [1, 128, T_mel]

                # Scale mel for vocoder
                mel_scaled = mel_raw * 0.5 + 0.5
                mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value

                # Vocoder → audio
                audio = dcae.vocoder.decode(mel_scaled).squeeze(1)  # [1, samples]

            # Save pair
            torch.save({
                'mel': mel_scaled.cpu(),     # [1, 128, T_mel] — vocoder input range
                'audio': audio.cpu(),         # [1, samples]
            }, DATA_DIR / f'pair_{saved:04d}.pt')

            saved += 1
            if saved % 50 == 0:
                print(f"  Saved {saved}/{max_samples}")

        except Exception as e:
            continue

    print(f"\nDone: {saved} pairs saved to {DATA_DIR}")
    return saved


# ============================================================
# Step 2: Structured Student Vocoder
# ============================================================

class StructuredVocoder(nn.Module):
    """
    Structured DSP vocoder: sines + interactions + noise.

    Progressive complexity levels:
      Level 0: Independent sines with learned amplitude per bin
      Level 1: Nonlinear amplitude mapping (MLP per band)
      Level 2: Inter-bin interactions (Conv1d across bins)
      Level 3: Noise component (filtered noise from mel envelope)

    All levels are trained jointly.
    """

    def __init__(self, n_mels=128, n_bands=8, sample_rate=44100, hop_length=512):
        super().__init__()
        self.n_mels = n_mels
        self.n_bands = n_bands
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.bins_per_band = n_mels // n_bands  # 16

        self.register_buffer('mel_freqs', MEL_FREQS.clone())

        # Level 0: Per-bin linear amplitude mapping
        self.amp_scale = nn.Parameter(torch.ones(n_mels) * 0.5)
        self.amp_bias = nn.Parameter(torch.zeros(n_mels))

        # Level 1: Nonlinear amplitude — small MLP per band (shared within band)
        # Input: mel values for bins in this band [B, T, bins_per_band]
        # Output: amplitude adjustments [B, T, bins_per_band]
        self.band_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bins_per_band, 32),
                nn.GELU(),
                nn.Linear(32, self.bins_per_band),
            )
            for _ in range(n_bands)
        ])

        # Level 2: Inter-bin interaction (residual Conv1d across frequency)
        self.interaction = nn.Sequential(
            nn.Conv1d(n_mels, n_mels, kernel_size=9, padding=4, groups=8),
            nn.GELU(),
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
        )
        self.interaction_gate = nn.Parameter(torch.tensor(0.0))  # starts at 0 (no effect)

        # Level 3: Noise shaping — predict noise filter from mel envelope
        self.noise_filter = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, n_mels, kernel_size=3, padding=1),
        )
        self.noise_gate = nn.Parameter(torch.tensor(0.0))  # starts at 0

        # Learned frequency correction per bin (small adjustments, ±10%)
        self.freq_correction = nn.Parameter(torch.zeros(n_mels))

    def forward(self, mel):
        """
        mel: [B, 128, T_mel] in vocoder range (e.g. [-11, 3])
        Returns: audio [B, n_samples]
        """
        B, C, T_mel = mel.shape
        device = mel.device
        n_samples = T_mel * self.hop_length

        # --- Level 2: Inter-bin interaction ---
        mel_interacted = mel + torch.sigmoid(self.interaction_gate) * self.interaction(mel)

        # --- Level 0 + 1: Amplitude mapping ---
        # Per-bin linear (Level 0)
        amp_linear = torch.sigmoid(
            self.amp_scale.view(1, -1, 1) * mel_interacted + self.amp_bias.view(1, -1, 1)
        )  # [B, 128, T_mel]

        # Per-band nonlinear adjustment (Level 1)
        amp_nonlinear = torch.zeros_like(amp_linear)
        mel_t = mel_interacted.permute(0, 2, 1)  # [B, T_mel, 128]
        for band_idx in range(self.n_bands):
            start = band_idx * self.bins_per_band
            end = start + self.bins_per_band
            band_mel = mel_t[:, :, start:end]  # [B, T_mel, 16]
            band_adj = self.band_mlps[band_idx](band_mel)  # [B, T_mel, 16]
            amp_nonlinear[:, start:end, :] = band_adj.permute(0, 2, 1)

        amp = amp_linear + amp_nonlinear  # combined amplitude
        amp = amp.clamp(min=0)  # ensure non-negative

        # --- Synthesis: vectorized additive ---
        # Frequencies with learned correction
        freq_hz = self.mel_freqs.to(device) * (1 + 0.1 * torch.tanh(self.freq_correction))

        # Interpolate amplitude to sample rate [B, 128, n_samples]
        amp_interp = F.interpolate(amp, size=n_samples, mode='linear', align_corners=False)

        # Phase: 2π * f * t
        t = torch.arange(n_samples, device=device, dtype=torch.float32) / self.sample_rate
        phase = 2 * np.pi * freq_hz.unsqueeze(-1) * t.unsqueeze(0)  # [128, n_samples]

        # Sum sines: [B, 128, n_samples] * sin([128, n_samples])
        audio = (amp_interp * torch.sin(phase.unsqueeze(0))).sum(dim=1)  # [B, n_samples]

        # --- Level 3: Noise component ---
        noise_gain = torch.sigmoid(self.noise_gate)
        if noise_gain > 1e-4:
            # Predict per-bin noise envelope
            noise_env = torch.sigmoid(self.noise_filter(mel))  # [B, 128, T_mel]
            noise_env_interp = F.interpolate(noise_env, size=n_samples, mode='linear', align_corners=False)

            # Generate white noise and shape it
            white_noise = torch.randn(B, n_samples, device=device) * 0.1
            # Apply frequency-shaped envelope (approximate bandpass per bin)
            # Simple: sum noise * envelope across bins
            shaped_noise = (noise_env_interp * white_noise.unsqueeze(1)).sum(dim=1)
            audio = audio + noise_gain * shaped_noise

        # Normalize
        peak = audio.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        audio = audio / peak * 0.9

        return audio


# ============================================================
# Step 3: Training
# ============================================================

def load_training_data():
    """Load all (mel, audio) pairs from disk."""
    pairs = sorted(DATA_DIR.glob('pair_*.pt'))
    print(f"Loading {len(pairs)} training pairs...")

    mels = []
    audios = []
    for p in pairs:
        data = torch.load(p, weights_only=False, map_location='cpu')
        mels.append(data['mel'].squeeze(0))   # [128, T_mel]
        audios.append(data['audio'].squeeze(0))  # [samples]

    # Find common lengths (crop to shortest)
    min_mel_T = min(m.shape[-1] for m in mels)
    min_audio_len = min(a.shape[-1] for a in audios)
    # Ensure audio length matches mel * hop
    target_audio_len = min(min_audio_len, min_mel_T * HOP_LENGTH)
    target_mel_T = target_audio_len // HOP_LENGTH

    mels = torch.stack([m[:, :target_mel_T] for m in mels])          # [N, 128, T]
    audios = torch.stack([a[:target_audio_len] for a in audios])      # [N, samples]

    print(f"  mel: {mels.shape}, audio: {audios.shape}")
    return mels, audios


def train(epochs=50, batch_size=8, lr=3e-4):
    sys.path.insert(0, str(SCRIPT_DIR.parent / 'training'))
    from losses import MultiResolutionSTFTLoss

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mels, audios = load_training_data()
    N = len(mels)
    n_val = max(1, N // 10)
    n_train = N - n_val

    # Split
    perm = torch.randperm(N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_mels = mels[train_idx]
    train_audios = audios[train_idx]
    val_mels = mels[val_idx]
    val_audios = audios[val_idx]

    print(f"  Train: {n_train}, Val: {n_val}")

    model = StructuredVocoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MultiResolutionSTFTLoss().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"Training: {epochs} epochs, batch={batch_size}, lr={lr}")
    print(f"Device: {device}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0
        n_batches = 0

        for batch_start in range(0, n_train, batch_size):
            idx = perm[batch_start:batch_start + batch_size]
            mel_batch = train_mels[idx].to(device)
            audio_target = train_audios[idx].to(device)

            audio_pred = model(mel_batch)

            # Match lengths
            min_len = min(audio_pred.shape[-1], audio_target.shape[-1])
            audio_pred = audio_pred[:, :min_len]
            audio_target = audio_target[:, :min_len]

            loss = criterion(audio_pred, audio_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_mel = val_mels.to(device)
            val_audio_target = val_audios.to(device)
            val_audio_pred = model(val_mel)
            min_len = min(val_audio_pred.shape[-1], val_audio_target.shape[-1])
            val_loss = criterion(val_audio_pred[:, :min_len], val_audio_target[:, :min_len]).item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            ig = torch.sigmoid(model.interaction_gate).item()
            ng = torch.sigmoid(model.noise_gate).item()
            print(f"  Epoch {epoch+1:3d}/{epochs}: train={avg_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  interact={ig:.3f}  noise={ng:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'n_params': n_params,
            }, MODEL_PATH)

    print(f"\n  Best val loss: {best_loss:.4f}")
    print(f"  Model saved to {MODEL_PATH}")


# ============================================================
# Step 4: Evaluation
# ============================================================

def evaluate():
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

    device = 'cuda'

    # Load student
    print("Loading distilled vocoder...")
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location='cpu')
    model = StructuredVocoder()
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    print(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['best_loss']:.4f}")

    # Load teacher (neural vocoder)
    print("Loading neural vocoder...")
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH)
    dcae.vocoder.to(device).eval()

    # Load a few test pairs
    pairs = sorted(DATA_DIR.glob('pair_*.pt'))[-5:]  # last 5 as test
    eval_dir = OUTPUT_DIR / "eval_audio"
    eval_dir.mkdir(parents=True, exist_ok=True)

    window = torch.hann_window(2048, device=device)

    print(f"\nEvaluating {len(pairs)} samples → {eval_dir}/")
    print()

    for i, ppath in enumerate(pairs):
        data = torch.load(ppath, weights_only=False, map_location='cpu')
        mel = data['mel'].to(device)          # [1, 128, T]
        teacher_audio = data['audio'].to(device)  # [1, samples]

        with torch.no_grad():
            student_audio = model(mel)

        # Match lengths
        min_len = min(teacher_audio.shape[-1], student_audio.shape[-1])
        teacher_audio = teacher_audio[:, :min_len].squeeze()
        student_audio = student_audio[:, :min_len].squeeze()

        # Spectral convergence
        spec_teacher = torch.stft(teacher_audio, 2048, 512, window=window, return_complex=True).abs()
        spec_student = torch.stft(student_audio, 2048, 512, window=window, return_complex=True).abs()
        sc = (torch.norm(spec_teacher - spec_student, p='fro') /
              (torch.norm(spec_teacher, p='fro') + 1e-8)).item()

        # Log mag MSE
        log_mse = F.mse_loss(
            torch.log(spec_student + 1e-8),
            torch.log(spec_teacher + 1e-8)
        ).item()

        print(f"  Sample {i}: spectral_conv={sc:.4f}, log_mag_MSE={log_mse:.4f}")

        # Save audio
        t_norm = teacher_audio / (teacher_audio.abs().max() + 1e-8) * 0.9
        s_norm = student_audio / (student_audio.abs().max() + 1e-8) * 0.9

        torchaudio.save(str(eval_dir / f'sample_{i:02d}_teacher.wav'),
                        t_norm.unsqueeze(0).cpu(), SR)
        torchaudio.save(str(eval_dir / f'sample_{i:02d}_student.wav'),
                        s_norm.unsqueeze(0).cpu(), SR)

        # A/B comparison
        silence = torch.zeros(SR // 2)
        combined = torch.cat([t_norm.cpu(), silence, s_norm.cpu()])
        torchaudio.save(str(eval_dir / f'sample_{i:02d}_AB.wav'),
                        combined.unsqueeze(0), SR)

    # Report learned gates
    print()
    print("Learned component gates:")
    print(f"  Interaction gate: {torch.sigmoid(model.interaction_gate).item():.4f}")
    print(f"  Noise gate: {torch.sigmoid(model.noise_gate).item():.4f}")

    # Report frequency corrections
    freq_corr = 0.1 * torch.tanh(model.freq_correction).abs().mean().item()
    print(f"  Avg freq correction: {freq_corr*100:.2f}%")

    print(f"\nAudio saved to {eval_dir}/")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vocoder Distillation')
    parser.add_argument('--generate-data', action='store_true', help='Generate training pairs')
    parser.add_argument('--train', action='store_true', help='Train student vocoder')
    parser.add_argument('--eval', action='store_true', help='Evaluate and save audio')
    parser.add_argument('--max-samples', type=int, default=500, help='Max training samples')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    args = parser.parse_args()

    if not any([args.generate_data, args.train, args.eval]):
        parser.print_help()
        sys.exit(0)

    print("=" * 60)
    print("VOCODER DISTILLATION")
    print("=" * 60)

    if args.generate_data:
        print("\n--- Step 1: Generate Training Data ---")
        generate_training_data(max_samples=args.max_samples)

    if args.train:
        print("\n--- Step 2: Train Student Vocoder ---")
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    if args.eval:
        print("\n--- Step 3: Evaluate ---")
        evaluate()
