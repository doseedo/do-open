# ~/Data/decoder_finetune.py
# Finetunes ONLY the DCAE decoder/postnet on predicted latents from your frozen Transformer.
# Apache 2.0

import os, json, math, argparse, inspect, sys
from datetime import datetime
from typing import List, Optional
from pathlib import Path

# Add ACE-Step to Python path
sys.path.append('/home/arlo/Data/ACE-Step')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from acestep.pipeline_ace_step import ACEStepTrainComponents
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from dataloader import PerformerAIDataset, collate_latent_cond

# ---- Grid constants (match your training) ----
DCAE_SR, DCAE_HOP = 44100, 4096   # slow grid that DCAE expects
SLOW_HZ = DCAE_SR / DCAE_HOP

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")


# ---------------- Custom callback for saving decoder-only artifacts ----------------
class SaveDecoderOnly(pl.Callback):
    def __init__(self, dirpath):
        self.dir = Path(dirpath)

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        state = {
            "decoder": pl_module.dcae.dcae.decoder.state_dict(),
            "meta": {
                "min_mel_value": pl_module.dcae.min_mel_value,
                "max_mel_value": pl_module.dcae.max_mel_value,
                "scale_factor": pl_module.dcae.scale_factor,
                "shift_factor": pl_module.dcae.shift_factor,
            },
        }
        out = self.dir / f"decoder_only-step{trainer.global_step}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, out)
        print(f"[decoder-ft] Saved decoder-only artifact to: {out}")

        # Also save a "latest" symlink for easy access
        latest = self.dir / "decoder_only-latest.pt"
        if latest.exists():
            latest.unlink()
        latest.symlink_to(out.name)


# ---------------- MR-STFT with transient & highband weighting ----------------
class MRSTFTLoss(nn.Module):
    def __init__(self,
                 fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512),
                 win_sizes=(512, 1024, 2048),
                 sr=48000,
                 highband_boost_hz=4000.0,
                 sc_factor=0.5,  # spectral convergence weight
                 mag_factor=0.5  # log-mag L1 weight
                 ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.sr = sr
        self.sc_factor = sc_factor
        self.mag_factor = mag_factor
        self.highband_boost_hz = highband_boost_hz
        self.register_buffer("eps", torch.tensor(1e-7), persistent=False)

        self.windows = nn.ParameterList(
            [nn.Parameter(torch.hann_window(w, periodic=True), requires_grad=False) for w in win_sizes]
        )

    def _get_stft_frames(self, audio_length: int, n_fft: int, hop_length: int) -> int:
        """Calculate exact STFT frame count matching torch.stft with center=True"""
        # torch.stft with center=True adds padding of n_fft//2 on each side
        padded_length = audio_length + n_fft
        return max(1, (padded_length - n_fft) // hop_length + 1)

    def _stft_mag(self, x, n_fft, hop, win_length, window_tensor):
        # x: [B, 1, T]
        # Ensure window tensor size matches win_length
        if window_tensor.size(0) != win_length:
            # Resize window if needed
            if window_tensor.size(0) > win_length:
                window_tensor = window_tensor[:win_length]
            else:
                # Pad with zeros
                pad_amount = win_length - window_tensor.size(0)
                window_tensor = torch.nn.functional.pad(window_tensor, (0, pad_amount), mode='constant', value=0.0)

        X = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win_length,
                       window=window_tensor.to(x.device), center=True, return_complex=True)
        mag = torch.abs(X)  # [B, F, T]
        return mag

    def _freq_weight(self, n_fft, device):
        # [F] weight vector, 1x below threshold, 3x above
        nyq = self.sr / 2.0
        freqs = torch.linspace(0, nyq, n_fft // 2 + 1, device=device)
        w = torch.ones_like(freqs)
        w = torch.where(freqs >= self.highband_boost_hz, w * 3.0, w)
        return w  # [F]

    def forward(self, x_pred, x_ref, time_weights: Optional[List[torch.Tensor]] = None):
        """
        x_pred, x_ref: [B, 1, T] waveform
        time_weights: list per scale, each [B, T_spec] aligned to hop of that scale (optional)
        """
        loss = x_pred.new_zeros(())
        audio_length = x_pred.shape[-1]

        for i, (n, h, w) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_sizes)):
            window_tensor = self.windows[i]
            mag_p = self._stft_mag(x_pred, n, h, w, window_tensor) + self.eps  # [B, F, T]
            mag_t = self._stft_mag(x_ref,  n, h, w, window_tensor) + self.eps
            fw = self._freq_weight(n, device=x_pred.device).view(1, -1, 1)  # [1,F,1]

            # Validate time weights against actual STFT output
            expected_frames = self._get_stft_frames(audio_length, n, h)
            actual_frames = mag_p.shape[2]

            if expected_frames != actual_frames:
                print(f"Warning: Frame count mismatch for scale {i}: expected {expected_frames}, got {actual_frames}")
                # This helps debug remaining issues

            # spectral convergence and log-mag L1
            sc = ( (mag_p - mag_t).abs() * fw ).sum(dim=(1,2)) / (mag_t * fw).sum(dim=(1,2)).clamp_min(1e-6)
            l1 = ( (mag_p.log() - mag_t.log()).abs() * fw ).mean(dim=1)  # [B, T]

            if time_weights is not None and time_weights[i] is not None:
                tw = time_weights[i].to(x_pred.device)                  # [B, T]
                # normalize per-sample so weighting changes emphasis, not scale
                tw = tw / (tw.mean(dim=1, keepdim=True).clamp_min(1e-6))

                # Handle size mismatch between l1 and tw
                l1_time_dim = l1.shape[1]
                tw_time_dim = tw.shape[1]

                if l1_time_dim != tw_time_dim:
                    # Resize time weights to match l1 dimensions
                    if tw_time_dim > l1_time_dim:
                        # Truncate time weights
                        tw = tw[:, :l1_time_dim]
                    else:
                        # Pad time weights
                        pad_amount = l1_time_dim - tw_time_dim
                        tw = torch.nn.functional.pad(tw, (0, pad_amount), mode='constant', value=1.0)

                l1 = (l1 * tw).mean(dim=1)
            else:
                l1 = l1.mean(dim=1)

            loss = loss + self.sc_factor * sc.mean() + self.mag_factor * l1.mean()
        return loss


def onset_time_weights_from_pr(pr_128_tlat: torch.Tensor,
                               T_audio: int,
                               specs: List[tuple],
                               sr_audio: int) -> List[torch.Tensor]:
    """
    pr_128_tlat: [B, 128, T_lat] (0/1 piano roll)
    Returns list of [B, T_spec] weight tracks per STFT scale,
    emphasizing the first ~30 ms after onsets.
    """
    B, _, T_lat = pr_128_tlat.shape
    # onset map in latent time
    d = (pr_128_tlat[..., 1:] - pr_128_tlat[..., :-1]).clamp_min(0.0)  # [B,128,T_lat-1]
    onset_lat = torch.zeros(B, T_lat, device=pr_128_tlat.device, dtype=pr_128_tlat.dtype)
    onset_lat[..., 1:] = d.amax(dim=1)  # [B, T_lat]

    # Upsample to audio samples
    onset_audio = F.interpolate(onset_lat.unsqueeze(1), size=T_audio, mode="linear", align_corners=False).squeeze(1)  # [B,T_audio]

    # Smear ~30 ms
    smear = max(1, int(round(0.03 * sr_audio)))  # 30 ms
    if smear > 1:
        k = torch.ones(1, 1, smear, device=onset_audio.device) / smear
        onset_audio = F.conv1d(onset_audio.unsqueeze(1), k, padding=smear//2).squeeze(1)

    # Build per-scale weights aligned to STFT time frames
    tw_all = []
    for (n_fft, hop, win) in specs:
        # Calculate exact STFT frame count to match torch.stft behavior
        # torch.stft with center=True adds padding of n_fft//2 on each side
        padded_length = T_audio + n_fft
        T_spec = (padded_length - n_fft) // hop + 1
        T_spec = max(1, T_spec)  # Ensure at least 1 frame

        tw = F.interpolate(onset_audio.unsqueeze(1), size=T_spec, mode="linear", align_corners=False).squeeze(1)  # [B,T_spec]
        # base weight 1.0 + 2.0*onset_emphasis → up to 3× at onsets
        tw = 1.0 + 2.0 * tw
        tw_all.append(tw)
    return tw_all


# ------------------------------ Lightning module ------------------------------
class DecoderOnlyFinetune(LightningModule):
    def __init__(self,
                 checkpoint_dir: str,
                 manifest_json: str,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 1e-4,
                 num_workers: int = 8,
                 batch_size: int = 2,
                 max_steps: int = 3000,
                 warmup_steps: int = 100,
                 window_slow: int = 1024,
                 sr_out: int = 48000,
                 preview_steps: int = 40,
                 every_plot_step: int = 500,
                 scheduler_T: int = 1000,
                 scheduler_shift: float = 3.0,
                 use_cpu_offloading: bool = True,
                 ):
        super().__init__()
        self.save_hyperparameters()

        # Build ACEStep components
        self.comps = ACEStepTrainComponents(checkpoint_dir=checkpoint_dir, dtype="float32")

        # Store if we should use CPU offloading (only for single GPU training)
        self.use_cpu_offloading = use_cpu_offloading
        self._xf_pinned_to_gpu = False

        # Always load transformer on CPU first to avoid OOM - handle DDP differently
        print(f"[Memory Optimization] Loading transformer on CPU (CPU offloading: {'enabled' if self.use_cpu_offloading else 'for multi-GPU init'})...")
        original_device = self.comps.device
        self.comps.device = "cpu"
        self.transformer = self.comps.build_transformer_pretrained()   # frozen, on CPU
        self.transformer.eval().requires_grad_(False)
        self.comps.device = original_device

        if self.use_cpu_offloading:
            print("[Memory Optimization] Transformer will stay on CPU and move to GPU only during inference")
        else:
            print("[Memory Optimization] Transformer will be kept on CPU for multi-GPU memory efficiency")

        # DCAE on GPU, train mode; we will unfreeze ONLY decoder/postnet
        self.dcae = self.comps.load_dcae()
        self.dcae.train()

        # keep the vocoder frozen & in eval (no BN/Dropout updates)
        self.dcae.vocoder.eval()
        for n, p in self.dcae.vocoder.named_parameters():
            p.requires_grad = False


        

        # Ensure the entire DCAE is on the correct device and in train mode
        print(f"[decoder-ft] DCAE device: {next(self.dcae.parameters()).device}")
        print(f"[decoder-ft] DCAE training mode: {self.dcae.training}")

        # Freeze everything in DCAE first, then selectively unfreeze decoder/postnet
        for p in self.dcae.parameters():
            p.requires_grad = False

        # Unfreeze ONLY the AutoencoderDC decoder path
        trainable = []
        for name, module in self.dcae.named_modules():
            # Names look like: "dcae.encoder.*", "dcae.decoder.*", "vocoder.*"
            if name.startswith("dcae.decoder"):
                for p in module.parameters():
                    p.requires_grad = True
                trainable.append(name)

        print(f"[decoder-ft] trainable DCAE submodules: {trainable if trainable else '(none)'}")

        # Safety: assert nothing in vocoder is trainable
        voc_trainable = sum(p.numel() for n, p in self.dcae.named_parameters()
                            if p.requires_grad and n.startswith("vocoder."))
        print(f"[decoder-ft] trainable vocoder params: {voc_trainable}")
        assert voc_trainable == 0, "Vocoder should be frozen for decoder-only FT"

        # Verify we have trainable parameters
        trainable_count = sum(p.numel() for p in self.dcae.parameters() if p.requires_grad)
        print(f"[decoder-ft] Total trainable DCAE parameters: {trainable_count:,}")
        if trainable_count == 0:
            raise RuntimeError("No trainable DCAE parameters found after decoder detection")

        # Detailed trainable parameter breakdown
        trainable_params = [(n, p.numel()) for n, p in self.dcae.named_parameters() if p.requires_grad]
        print(f"[decoder-ft] trainable params: {sum(n for _, n in trainable_params):,}")
        print("examples:", [n for n, _ in trainable_params[:10]])
        voc_t = sum(p.numel() for n, p in self.dcae.named_parameters() if p.requires_grad and "vocoder" in n.lower())
        print(f"[decoder-ft] trainable vocoder params: {voc_t:,}")

        # Scheduler for RF objective (used only to nudge latents off-manifold)
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=scheduler_T, shift=scheduler_shift)

        # Loss (transient-weighted MR-STFT)
        self.mrstft = MRSTFTLoss(
            fft_sizes=(512, 1024, 2048),
            hop_sizes=(128, 256, 512),
            win_sizes=(512, 1024, 2048),
            sr=sr_out,
            highband_boost_hz=4000.0,
            sc_factor=0.5,
            mag_factor=0.5
        )

        self.manifest_json = manifest_json
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.window_slow = window_slow
        self.sr_out = sr_out
        self.preview_steps = preview_steps
        self.every_plot_step = every_plot_step

        # tiny fixed preview batch
        self._preview_batch = None
        self._val_dataset = None

    # ---------------- Data ----------------
    def setup(self, stage=None):
        # Same dataset, but with *stable* conditioning & fixed gain vibes (no extreme augments)
        self.ds = PerformerAIDataset(
            json_path=self.manifest_json,
            conditioning_dropout={"piano_roll": 0.0, "amp": 0.0, "rbend": 0.0, "rframe": 0.0},
            use_trim=True,
            pre_roll_seconds=0.25,        # give some pre-context
            post_roll_seconds=0.0,
            keep_untrimmed_prob=0.0,
            amp_activity_thr=0.06,
            require_all_core=True,
            collapse_sparse_subgroups_to_any=False,
            static_window=False,
            window_slow=self.window_slow,
            seed=123
        )
        # Deterministic single preview item
        if len(self.ds) == 0:
            print("Warning: Dataset is empty - preview will be disabled")
            self._preview_batch = None
        else:
            prev_idx = 0 % len(self.ds)
            self._preview_batch = collate_latent_cond([self.ds[prev_idx]])

            # Create small validation set from end of dataset
            val_size = min(50, max(1, len(self.ds) // 20))  # 5% or 50 samples max
            val_indices = list(range(len(self.ds) - val_size, len(self.ds)))
            self._val_dataset = torch.utils.data.Subset(self.ds, val_indices)
            print(f"[validation] Created validation set with {len(self._val_dataset)} samples")

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        def _seed_worker(worker_id):
            import numpy as np, random, torch
            seed = torch.initial_seed() % 2**32
            np.random.seed(seed); random.seed(seed)
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_latent_cond,
            worker_init_fn=_seed_worker
        )

    def val_dataloader(self):
        if self._val_dataset is None:
            return None
        from torch.utils.data import DataLoader
        return DataLoader(
            self._val_dataset,
            batch_size=min(self.batch_size, 4),  # Smaller val batch
            shuffle=False,
            num_workers=min(self.num_workers, 4),
            pin_memory=True,
            collate_fn=collate_latent_cond
        )

    # ---------------- Helpers ----------------
    def _call_transformer_no_xattn(self, latents, t_idx):
        """
        Frozen transformer forward without cross-attn (dummy context),
        with memory-efficient GPU transfer for both single and multi-GPU training.
        """
        # Optimize transformer GPU placement based on offloading strategy
        on_cpu = next(self.transformer.parameters()).device.type == 'cpu'

        if on_cpu:
            self.transformer = self.transformer.to(self.device)
            if not self.use_cpu_offloading:   # multi-GPU or no offload → keep it on GPU
                self._xf_pinned_to_gpu = True

        sig = inspect.signature(self.transformer.forward).parameters
        kwargs = {}
        if "hidden_states" in sig: kwargs["hidden_states"] = latents
        elif "x" in sig:           kwargs["x"] = latents

        for k in ("timestep", "timesteps", "t", "sigma", "noise_sigma"):
            if k in sig:
                kwargs[k] = t_idx
                break

        # dummy context to satisfy interfaces
        B = latents.size(0)
        d_text = getattr(self.transformer.config, "text_embedding_dim", 768)
        dummy_ctx = torch.zeros(B, 1, d_text, device=latents.device, dtype=latents.dtype)
        dummy_mask = torch.zeros(B, 1, device=latents.device, dtype=torch.float32)  # Masks always float32

        for k in ("encoder_text_hidden_states", "encoder_hidden_states", "context", "text_embeds"):
            if k in sig:
                kwargs[k] = dummy_ctx
        for k in ("text_attention_mask", "encoder_hidden_mask", "encoder_attention_mask", "context_mask"):
            if k in sig:
                kwargs[k] = dummy_mask

        if "speaker_embeds" in sig:
            kwargs["speaker_embeds"] = torch.zeros(B, 512, device=latents.device, dtype=latents.dtype)
        if "lyric_token_idx" in sig:
            kwargs["lyric_token_idx"] = torch.zeros(B, 1, device=latents.device, dtype=torch.long)
        if "lyric_mask" in sig:
            kwargs["lyric_mask"] = torch.zeros(B, 1, device=latents.device, dtype=torch.float32)

        # Add attention_mask if required
        if "attention_mask" in sig:
            # Create attention mask matching your original trainer's _make_attn_mask
            # latents: [B, C, H, T_slow] -> mask [B, T_slow]
            _, _, _, T_slow = latents.shape
            kwargs["attention_mask"] = torch.ones(B, T_slow, device=latents.device, dtype=torch.float32)

        out = self.transformer(**kwargs)

        # Only move back to CPU if we're actually offloading
        if self.use_cpu_offloading and on_cpu:
            self.transformer = self.transformer.cpu()
            torch.cuda.empty_cache()

        return out.sample if hasattr(out, "sample") else out

    def _match_mod_dtype(self, x, module):
        """Match tensor device/dtype to module parameters"""
        p = next(module.parameters(), None)
        return x if p is None else x.to(device=p.device, dtype=p.dtype)

    def _decode_to_wave(self, x_latent):
        """
        x_latent: [B, C, H, T_slow] in DCAE grid (ACE slow grid).
        Returns:  [B, 1, T_out] @ self.sr_out (differentiable)
        """
        B, _, _, T_slow = x_latent.shape
        # 1) Undo DCAE normalization
        latents_norm = x_latent / self.dcae.scale_factor + self.dcae.shift_factor
        latents_norm = self._match_mod_dtype(latents_norm, self.dcae)  # device/dtype match

        # 2) DCAE decoder → mels (keep grads)
        #    Expect mels: [B, 2, 128, Tm]  (2 = stereo mels)
        mels = self.dcae.dcae.decoder(latents_norm)
        # denormalize to real mel scale
        mels = mels * 0.5 + 0.5
        mels = mels * (self.dcae.max_mel_value - self.dcae.min_mel_value) + self.dcae.min_mel_value

        # 3) Vocoder.forward on each channel, batched
        #    ADaMoSHiFiGANV1.forward expects [B, 128, Tm]
        B, C, F_mel, Tm = mels.shape  # C should be 2, F_mel should be 128
        mels_bc = mels.view(B * C, F_mel, Tm)                      # [B*C, 128, Tm]
        voc = self.dcae.vocoder
        voc.eval()                                             # keep it eval; grads still flow through to the decoder upstream
        wav_bc = voc.forward(mels_bc)                          # [B*C, 1, T_441]
        # reshape back to [B, C, T]
        wav = wav_bc.view(B, C, 1, -1).squeeze(2)              # [B, 2, T_441]

        # 4) (Optional) downmix to mono before loss
        wav = wav.mean(dim=1, keepdim=True)                    # [B, 1, T_441]

        # 5) Differentiable resample if needed (avoid torchaudio here)
        sr_native = getattr(voc, "sampling_rate", 44100)
        if self.sr_out != sr_native:
            T_in = wav.shape[-1]
            T_out = int(round(T_in * (self.sr_out / float(sr_native))))
            # F.interpolate expects [N,C,L], mode='linear' works for 1D
            wav = F.interpolate(wav, size=T_out, mode="linear", align_corners=False)

        return wav.to(dtype=torch.float32)

    # ---------------- Lightning core ----------------
    def training_step(self, batch, batch_idx):
        # Move tensors
        def to_dev(x):
            if isinstance(x, torch.Tensor): return x.to(self.device)
            if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
            return x
        batch = to_dev(batch)

        x0 = batch["latents"]  # [B,C,H,T_slow]
        B = x0.size(0)

        # RF-style one-step latent prediction to obtain off-manifold x0_hat
        T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        tau = torch.rand(B, device=self.device).clamp(1e-4, 1 - 1e-4)              # [B] - use non-inplace clamp
        t_idx = (tau * (T_train - 1)).long().clamp(0, T_train - 1)                  # [B]
        sigma = tau.to(x0.dtype).view(B, *([1] * (x0.ndim - 1)))                    # [B,1,1,1]
        z = torch.randn_like(x0)

        # Create noisy version (need gradients through this path)
        x_noisy = (1.0 - sigma) * x0 + sigma * z

        # Get v_pred from frozen transformer (no gradients needed through transformer)
        with torch.no_grad():
            v_pred = self._call_transformer_no_xattn(latents=x_noisy, t_idx=t_idx)

        # keep decoder grads only (inputs needn't require_grad for param grads to flow)
        x0_hat = x_noisy.detach() - sigma * v_pred.detach()

        # Decode both to waveform (grad flows ONLY through decoder for x0_hat)
        with torch.no_grad():
            wav_ref = self._decode_to_wave(x0)               # no grad for reference
        wav_pred = self._decode_to_wave(x0_hat)              # grad to decoder

        # Ensure both waveforms have the same length
        min_len = min(wav_ref.shape[-1], wav_pred.shape[-1])
        wav_ref = wav_ref[..., :min_len]
        wav_pred = wav_pred[..., :min_len]

        # Transient weights from PR (if present)
        specs = list(zip(self.mrstft.fft_sizes, self.mrstft.hop_sizes, self.mrstft.win_sizes))
        T_audio = wav_pred.shape[-1]
        pr = batch["conds"].get("piano_roll", None)
        time_weights = None
        if isinstance(pr, torch.Tensor) and pr.ndim == 3:
            # PR is [B,128,T_lat]; make sure it's float
            pr = pr.to(dtype=torch.float32)
            time_weights = onset_time_weights_from_pr(pr, T_audio, specs, self.sr_out)

        loss = self.mrstft(wav_pred, wav_ref, time_weights=time_weights)

        # Verify gradients are flowing properly
        if not loss.requires_grad:
            print(f"ERROR: Loss tensor does not require grad!")
            print(f"wav_pred requires_grad: {wav_pred.requires_grad}")
            print(f"wav_ref requires_grad: {wav_ref.requires_grad}")
            print(f"x0_hat requires_grad: {x0_hat.requires_grad}")

        self.log("train/mrstft", loss, on_step=True, prog_bar=True)

        # Occasionally dump short previews
        if (self.global_step in (0,1,2,5,10)) or ((self.global_step + 1) % max(1, self.hparams.every_plot_step) == 0):
            self._write_preview(wav_pred, wav_ref)

        return loss

    def validation_step(self, batch, batch_idx):
        # Move tensors
        def to_dev(x):
            if isinstance(x, torch.Tensor): return x.to(self.device)
            if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
            return x
        batch = to_dev(batch)

        x0 = batch["latents"]  # [B,C,H,T_slow]
        B = x0.size(0)

        # Use fixed timestep for validation consistency
        T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        tau = torch.full((B,), 0.5, device=self.device)  # Fixed tau=0.5 for validation
        t_idx = (tau * (T_train - 1)).long().clamp(0, T_train - 1)
        sigma = tau.to(x0.dtype).view(B, *([1] * (x0.ndim - 1)))
        z = torch.randn_like(x0)
        x_noisy = (1.0 - sigma) * x0 + sigma * z

        # Get prediction from frozen transformer
        with torch.no_grad():
            v_pred = self._call_transformer_no_xattn(latents=x_noisy, t_idx=t_idx)
        x0_hat = x_noisy - sigma * v_pred.detach()

        # Decode both to waveform
        with torch.no_grad():
            wav_ref = self._decode_to_wave(x0)
            wav_pred = self._decode_to_wave(x0_hat)  # Validation doesn't need gradients

        # Ensure both waveforms have the same length
        min_len = min(wav_ref.shape[-1], wav_pred.shape[-1])
        wav_ref = wav_ref[..., :min_len]
        wav_pred = wav_pred[..., :min_len]

        # Calculate validation loss (without time weights for simplicity)
        val_loss = self.mrstft(wav_pred, wav_ref, time_weights=None)
        self.log("val/mrstft", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return val_loss

    @torch.no_grad()
    def _write_preview(self, wav_pred, wav_ref, seconds=6.0):
        try:
            out_dir = f"{getattr(self.logger, 'log_dir', './logs')}/eval_results/step_{int(self.global_step)}"
            os.makedirs(out_dir, exist_ok=True)
            L = min(wav_pred.shape[-1], int(self.sr_out * seconds))

            # Move to CPU and clean up GPU memory
            pr = wav_pred[0:1, :, :L].detach().cpu()
            gt = wav_ref[0:1, :, :L].detach().cpu()

            # Clear GPU tensors explicitly
            del wav_pred, wav_ref
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            torchaudio.save(f"{out_dir}/pred.wav", pr[0], self.sr_out)
            torchaudio.save(f"{out_dir}/ref.wav", gt[0], self.sr_out)
        except Exception as e:
            print(f"Warning: Preview save failed: {e}")

    def configure_optimizers(self):
        # Optimize ONLY decoder/postnet params
        dec_params = [p for p in self.dcae.parameters() if p.requires_grad]
        if not dec_params:
            raise RuntimeError("No trainable DCAE decoder params found.")
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.PagedAdamW8bit(dec_params, lr=self.hparams.learning_rate,
                                           betas=(0.9, 0.999), weight_decay=self.hparams.weight_decay, eps=1e-8)
        except Exception:
            opt = torch.optim.AdamW(dec_params, lr=self.hparams.learning_rate,
                                    betas=(0.9, 0.999), weight_decay=self.hparams.weight_decay, eps=1e-8, foreach=False)

        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return float(step) / max(1, self.hparams.warmup_steps)
            progress = float(step - self.hparams.warmup_steps) / max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            return max(0.0, 1.0 - progress)

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=-1)
        return [opt], [{"scheduler": sch, "interval": "step"}]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--manifest_json",  required=True)
    ap.add_argument("--logger_dir",     required=True)
    ap.add_argument("--exp_name",       required=True)

    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--num_nodes", type=int, default=1)
    ap.add_argument("--precision", type=str, default="bf16-mixed")

    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--batch_size",    type=int,   default=1)
    ap.add_argument("--num_workers",   type=int,   default=8)
    ap.add_argument("--max_steps",     type=int,   default=3000)
    ap.add_argument("--warmup_steps",  type=int,   default=100)
    ap.add_argument("--window_slow",   type=int,   default=1024)
    ap.add_argument("--sr_out",        type=int,   default=48000)
    ap.add_argument("--preview_steps", type=int,   default=300)
    ap.add_argument("--every_plot_step", type=int, default=5000)

    # Optional: resume a full model ckpt (we load only weights we need; Transformer stays frozen)
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--weights_only", action="store_true")

    args = ap.parse_args()

    # Only use CPU offloading for single GPU training to avoid DDP issues
    use_cpu_offloading = args.devices == 1 and args.num_nodes == 1
    print(f"[Memory] CPU offloading: {'enabled' if use_cpu_offloading else 'disabled'} (devices={args.devices}, num_nodes={args.num_nodes})")

    # Further reduce memory usage for multi-GPU training
    if not use_cpu_offloading:
        args.batch_size = min(args.batch_size, 1)  # Force batch size to 1 for multi-GPU
        args.window_slow = min(args.window_slow, 256)  # Reduce window size for multi-GPU
        args.num_workers = min(args.num_workers, 2)  # Reduce workers for multi-GPU
        print(f"[Memory] Multi-GPU adjustments: batch_size={args.batch_size}, window_slow={args.window_slow}, num_workers={args.num_workers}")

    model = DecoderOnlyFinetune(
        checkpoint_dir=args.checkpoint_dir,
        manifest_json=args.manifest_json,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        window_slow=args.window_slow,
        sr_out=args.sr_out,
        preview_steps=args.preview_steps,
        every_plot_step=args.every_plot_step,
        use_cpu_offloading=use_cpu_offloading
    )

    # If a previous training ckpt is provided, load encoder/transformer/dcae weights (Transformer remains frozen)
    if args.ckpt_path and args.weights_only:
        try:
            print(f"[resume] Loading weights from: {args.ckpt_path}")
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)

            # Keep only transformer.* and dcae.* keys
            keep = {k: v for k, v in sd.items() if k.startswith("transformer.") or k.startswith("dcae.")}
            if not keep:
                print("[resume] WARNING: No transformer or dcae weights found in checkpoint")

            missing, unexpected = model.load_state_dict(keep, strict=False)
            print(f"[resume] Loaded base weights; missing={len(missing)}, unexpected={len(unexpected)}")

            # Log critical missing weights
            critical_missing = [k for k in missing if 'dcae' in k and any(x in k.lower() for x in ['decoder', 'decode', 'post'])]
            if critical_missing:
                print(f"[resume] WARNING: Missing critical decoder weights: {critical_missing[:5]}...")

            # Prevent PL from trying to restore optimizer/scheduler
            args.ckpt_path = None
        except Exception as e:
            print(f"[resume] ERROR: Failed to load checkpoint: {e}")
            print("[resume] Continuing with random initialization...")
            args.ckpt_path = None

    # Checkpoint callback (use weights-only to avoid 17GB files)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"/mnt/msdd/checkpoints/decoder_ft/{args.exp_name}",
        filename="step-{step}",
        monitor="val/mrstft",   # you log this already
        mode="min",
        save_top_k=1,           # keep the single best ckpt
        save_last=True,         # plus a resume anchor
        save_weights_only=True  # Keep file size manageable (~2GB vs ~17GB)
    )

    # Custom callback for tiny decoder-only artifacts
    small_dir = f"/mnt/msdd/checkpoints/decoder_ft/{args.exp_name}/decoder_only"
    decoder_saver = SaveDecoderOnly(small_dir)

    # Logger placement
    logger_cb = TensorBoardLogger(
        save_dir="/mnt/msdd/logs/decoder_ft",
        name=args.exp_name,
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        strategy="ddp_find_unused_parameters_true" if int(args.devices) > 1 else "auto",
        max_steps=args.max_steps,
        max_epochs=-1,
        logger=logger_cb,
        callbacks=[ckpt_cb, decoder_saver],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        limit_val_batches=20,  # Run validation on 20 batches
        val_check_interval=1000,  # Validate every 1000 steps (less frequent saves)
        num_sanity_val_steps=2,  # Run 2 sanity validation steps
    )
    trainer.fit(model, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
