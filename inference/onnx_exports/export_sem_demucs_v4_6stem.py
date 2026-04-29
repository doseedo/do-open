"""Export SemDemucs v4 6-stem ("v4-small 6") to ONNX for WebGPU.

Produces: sem_demucs_v4_6s_packed.onnx (+ .onnx.data for weights >2GB free)

Browser runs this BEFORE latent demucs on every upload:
  waveform [1, 2, N] → stft_masks [1, 6, F, T_stft]   mix-vs-solo detection + instant stem waveforms
                    → rms         [1, 6, T', 2]       per-stem amplitude envelopes
                    → embedding   [1, 6, 128]          (unused for now — keep for future cond)

STFT masks sum to 1 across the 6 stems at each (f, t) bin (softmax),
so by integrating mask energy per stem we get a mix-vs-solo classifier:
  - energy concentrated in drums + bass + vocals + other  → full mix
  - energy concentrated in one stem (e.g., vocals only)   → solo stem

rms is used DIRECTLY as the per-stem waveform display (6 stems) so the
DAW shows 6 stem rows the instant upload completes, with no wait for
the 325 MB latent demucs.

Stem order (matches DistillDataset6 STEMS_6):
  0 drums, 1 bass, 2 other, 3 vocals, 4 guitar, 5 piano
  (Earlier doc here had vocals/other swapped; training code is authoritative.)

STFT decomposition (2026-04-14):
  The model uses torch.stft inside forward() to compute the mix spectrogram.
  torch.onnx.export turns that into the native ONNX STFT op (opset 17+),
  which ORT-Web 1.22 mishandles — WASM returns NaN, WebGPU returns zero,
  causing every stem mask to come back NaN/zero in the browser. The mix-detect
  classifier and the instant 6-stem waveform painter both depend on these
  masks. We monkeypatch the forward() to compute |STFT|² via a fixed
  Conv1d filterbank (sin/cos basis windowed by Hann), which exports as
  plain Conv/Mul/Sqrt and works in ORT-Web. Output is mathematically
  equivalent to torch.stft(...).abs() (offline parity verified < 1.2e-3
  max abs diff on white noise). Adds ~17 MB to the weights file (the
  basis filterbank), which is fine — ONNX is still ~26 MB total.
"""
from __future__ import annotations
import argparse, math, os, sys, types
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latent_demucs_student"))


class STFTMagnitude(nn.Module):
    """|STFT(mono)| via Conv1d. Drop-in replacement for torch.stft(...).abs()
    that exports cleanly to ORT-Web (avoids the broken native STFT op)."""
    def __init__(self, n_fft: int, hop: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        win = torch.hann_window(n_fft)
        n_freqs = n_fft // 2 + 1
        n = torch.arange(n_fft).float()
        k = torch.arange(n_freqs).float().unsqueeze(1)
        angle = 2.0 * math.pi * k * n / n_fft
        cos = (torch.cos(angle) * win).unsqueeze(1).float()  # [n_freqs, 1, n_fft]
        sin = (-torch.sin(angle) * win).unsqueeze(1).float()
        self.register_buffer("cos_basis", cos)
        self.register_buffer("sin_basis", sin)

    def forward(self, mono: torch.Tensor) -> torch.Tensor:
        pad = self.n_fft // 2
        x = mono.unsqueeze(1)                                  # [B, 1, N]
        x = F.pad(x, (pad, pad), mode="reflect")               # match torch.stft(center=True)
        re = F.conv1d(x, self.cos_basis, stride=self.hop)
        im = F.conv1d(x, self.sin_basis, stride=self.hop)
        return torch.sqrt(re * re + im * im + 1e-12)           # [B, n_freqs, T_stft]


def _patch_stft(model):
    """Replace model.forward()'s torch.stft call with Conv1d STFTMagnitude."""
    stft_mag = STFTMagnitude(model.n_fft, model.n_fft // 4).to(next(model.parameters()).device)
    model._stft_mag = stft_mag

    def patched_forward(self, waveform):
        h = self.encoder(waveform)
        B, C, T = h.shape
        h_seq = h.transpose(1, 2)
        mono = waveform.mean(dim=1)
        spec = self._stft_mag(mono)                            # was: torch.stft(...).abs()
        mix_spec = (spec + 1e-8).log().unsqueeze(1)
        h_stems = self.separator(h_seq)
        mask_logits = self.mask_head(h_stems, mix_spec)
        stft_masks = torch.softmax(mask_logits, dim=1)
        pitch_logits = self.pitch_head(h_stems)
        rms = self.rms_head(h_stems)
        embeddings = []
        for i in range(self.n_stems):
            q = self.pool_query[i].expand(B, -1, -1)
            h_stem = h_stems[:, i]
            attn = torch.bmm(q, h_stem.transpose(1, 2)).softmax(dim=-1)
            pooled = torch.bmm(attn, h_stem).squeeze(1)
            embeddings.append(self.embed_head(pooled))
        embeddings = torch.stack(embeddings, dim=1)
        vocal = self.vocal_head(embeddings).squeeze(-1)
        return {
            "stft_masks": stft_masks, "mask_logits": mask_logits,
            "embedding": embeddings, "pitch_logits": pitch_logits,
            "rms": rms, "vocal": vocal,
        }

    model.forward = types.MethodType(patched_forward, model)


class V4SmallExport(torch.nn.Module):
    """Returns the subset the browser needs: stft_masks, masks_envelope, rms, embedding.

    `masks_envelope = stft_masks.sum(dim=2)` is a per-(stem, time-frame) integration
    of the softmax mask across the freq axis. It's the signal the DAW actually
    needs for both classification (sum over T → per-stem energy fraction) and the
    instant 6-stem placeholder waveforms (per-stem envelope vs time). Tiny output
    (~17K floats for a 30 s clip vs. 17 M for the full 4D mask), so it survives
    ORT-Web's GPU→CPU copy of large buffers (which silently zeroes the full mask).

    `stft_masks` is kept because latentDemucsV4 takes it as conditioning input.
    The 17M-float buffer still gets computed inside the graph (intermediate of
    softmax) — that compute path is fine; only the output readback breaks, and
    the envelope provides an independent small readback that does work.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        out = self.model(waveform)
        masks = out["stft_masks"]                # [B, S, F, T_stft]
        masks_envelope = masks.sum(dim=2)        # [B, S, T_stft]   per-stem energy/time
        # Drop mask_logits (raw; browser uses softmax-ed stft_masks).
        # Drop pitch_logits, vocal (not needed for mix detection / viz).
        return masks, masks_envelope, out["rms"], out["embedding"]


def export(ckpt_path, out_path, n_stems=6, channels=64, opset=17):
    from sem_demucs import SemDemucs

    print(f"[v4-small-{n_stems} export] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SemDemucs(n_stems=n_stems, channels=channels)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    if missing: print(f"  missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected: print(f"  unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    model.eval()

    # Swap torch.stft for a Conv1d magnitude STFT so the export doesn't
    # emit the native ONNX STFT op (which ORT-Web 1.22 silently breaks).
    _patch_stft(model)

    wrapper = V4SmallExport(model).eval().float()

    # 4 seconds of stereo at 48kHz for tracing — dynamic axis on the time dim.
    # White-noise (not zeros) so the spectrogram path actually has signal — keeps
    # parity check honest if anyone re-runs export with a different opset.
    dummy = torch.randn(1, 2, 48000 * 4) * 0.3

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[v4-small-{n_stems} export] running torch.onnx.export → {out_path}")
    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["waveform"],
        output_names=["stft_masks", "masks_envelope", "rms", "embedding"],
        dynamic_axes={
            "waveform":       {2: "n_samples"},
            "stft_masks":     {3: "t_stft"},
            "masks_envelope": {2: "t_stft"},
            "rms":            {2: "t_latent"},
            # embedding is per-stem global (shape [1, S, 128]) — no dynamic axis
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    sz = os.path.getsize(out_path) / 1e6
    data_path = out_path + ".data"
    data_sz = os.path.getsize(data_path) / 1e6 if os.path.exists(data_path) else 0
    print(f"[v4-small-{n_stems} export] DONE — graph {sz:.1f} MB, weights {data_sz:.1f} MB")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/latent_demucs_student/sem_demucs_v4_6stem_ckpts/sem_demucs_v4_6s_final.pt")
    ap.add_argument("--out", default="/tmp/sem_demucs_v4_6s_packed.onnx")
    ap.add_argument("--n-stems", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--opset", type=int, default=17)  # 17 is enough — STFT op (18) no longer used.
    args = ap.parse_args()
    export(args.ckpt, args.out, n_stems=args.n_stems, channels=args.channels, opset=args.opset)
