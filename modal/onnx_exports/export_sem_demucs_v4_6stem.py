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
  0 drums, 1 bass, 2 vocals, 3 other, 4 guitar, 5 piano
"""
from __future__ import annotations
import argparse, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latent_demucs_student"))


class V4SmallExport(torch.nn.Module):
    """Returns the subset the browser needs: stft_masks, rms, embedding."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        out = self.model(waveform)
        # Drop mask_logits (raw; browser uses softmax-ed stft_masks).
        # Drop pitch_logits, vocal (not needed for mix detection / viz).
        return out["stft_masks"], out["rms"], out["embedding"]


def export(ckpt_path, out_path, n_stems=6, channels=64, opset=17):
    from sem_demucs import SemDemucs

    print(f"[v4-small-{n_stems} export] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SemDemucs(n_stems=n_stems, channels=channels)
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    if missing: print(f"  missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected: print(f"  unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    model.eval()

    wrapper = V4SmallExport(model).eval().float()

    # 4 seconds of stereo at 48kHz for tracing — dynamic axis on the time dim.
    dummy = torch.zeros(1, 2, 48000 * 4)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[v4-small-{n_stems} export] running torch.onnx.export → {out_path}")
    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["waveform"],
        output_names=["stft_masks", "rms", "embedding"],
        dynamic_axes={
            "waveform":    {2: "n_samples"},
            "stft_masks":  {3: "t_stft"},
            "rms":         {2: "t_latent"},
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
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    export(args.ckpt, args.out, n_stems=args.n_stems, channels=args.channels, opset=args.opset)
