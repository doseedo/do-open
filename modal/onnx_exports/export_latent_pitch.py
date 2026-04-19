"""Export the LatentPitch student (latent → MIDI) to ONNX.

The student (`LatentBasicPitchStudent`) takes [B, T, 64] VAE latents at
25 Hz and returns four per-frame heads: onset_logits, frame_logits,
velocity, onset_offset — all [B, T, 128].

PyTorch forward returns a dict, which the ONNX exporter can't represent
as-is; we wrap the module so forward returns a fixed-order tuple.

Output:
    /scratch/onnx/latent_pitch_packed.onnx  (single packed file)

Usage:
    python -m onnx_exports.export_latent_pitch \
        --ckpt /scratch/latent_pitch_ckpts/pitch_054000.pt \
        --out  /scratch/onnx/latent_pitch_packed.onnx
"""
from __future__ import annotations
import argparse, os, sys
import torch
import torch.nn as nn

# Student source lives next to its checkpoints on the training VM. When
# exporting locally, pass --src to point at a directory containing the
# `latent_pitch/` package.
_DEFAULT_SRC = "/scratch/Do/home/arlo/do2"

DEFAULT_CKPT = "/scratch/latent_pitch_ckpts/pitch_054000.pt"
DEFAULT_OUT  = "/scratch/onnx/latent_pitch_packed.onnx"


class _PitchExportWrapper(nn.Module):
    """Tuple-output wrapper for ONNX (no dict outputs)."""
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, latent: torch.Tensor):
        o = self.inner(latent)
        return (
            o["onset_logits"],
            o["frame_logits"],
            o["velocity"],
            o["onset_offset"],
        )


def _infer_arch(sd):
    """Derive d_model / n_conv_blocks / n_tr_layers / max_len from state dict
    so v1 (256/4/4) and v4 (384/6/8) both load without a manifest."""
    pos = sd.get("pos")
    max_len = pos.shape[1] if pos is not None else 2048
    d_model = pos.shape[2] if pos is not None else 256
    n_conv_blocks = sum(
        1 for k in sd if k.startswith("conv_stack.") and k.endswith(".conv1.weight")
    )
    n_tr_layers = sum(
        1 for k in sd if k.startswith("tr.layers.") and k.endswith(".self_attn.out_proj.weight")
    )
    return dict(
        d_model=d_model,
        n_conv_blocks=n_conv_blocks or 4,
        n_tr_layers=n_tr_layers or 4,
        max_len=max_len,
    )


def export(ckpt_path: str,
           out_path: str,
           src_dir: str = _DEFAULT_SRC,
           opset: int = 17,
           dummy_frames: int = 256,
           device: str = "cpu"):
    sys.path.insert(0, src_dir)
    from latent_pitch.model import LatentBasicPitchStudent  # type: ignore

    print(f"[init] loading ckpt {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    arch = _infer_arch(sd)
    print(f"[init] arch {arch}")

    inner = LatentBasicPitchStudent(**arch).to(device).float().eval()
    inner.load_state_dict(sd)
    model = _PitchExportWrapper(inner).to(device).eval()

    # Dummy input: [1, T, 64] — T chosen ≤ max_len. Dynamic axis covers
    # any length up to max_len at inference.
    dummy_frames = min(dummy_frames, arch["max_len"])
    x = torch.randn(1, dummy_frames, 64, device=device).float()
    print(f"[trace] dummy input shape={tuple(x.shape)} ({dummy_frames / 25.0:.2f}s @ 25 Hz)")

    with torch.no_grad():
        ys = model(x)
    print(f"[trace] outputs:")
    print(f"  onset_logits  {tuple(ys[0].shape)}")
    print(f"  frame_logits  {tuple(ys[1].shape)}")
    print(f"  velocity      {tuple(ys[2].shape)}")
    print(f"  onset_offset  {tuple(ys[3].shape)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw_path = out_path.replace("_packed.onnx", ".onnx")
    print(f"[export] writing {raw_path}")
    torch.onnx.export(
        model,
        (x,),
        raw_path,
        input_names=["latent"],
        output_names=["onset_logits", "frame_logits", "velocity", "onset_offset"],
        dynamic_axes={
            "latent":        {0: "batch", 1: "frames"},
            "onset_logits":  {0: "batch", 1: "frames"},
            "frame_logits":  {0: "batch", 1: "frames"},
            "velocity":      {0: "batch", 1: "frames"},
            "onset_offset":  {0: "batch", 1: "frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )

    # Pack weights inline so the browser only has to fetch one file.
    try:
        import onnx
        m = onnx.load(raw_path, load_external_data=True)
        onnx.save_model(m, out_path, save_as_external_data=False)
        ext = raw_path + ".data"
        if os.path.exists(ext):
            os.unlink(ext)
        if os.path.exists(raw_path) and raw_path != out_path:
            os.unlink(raw_path)
    except Exception as e:
        print(f"[pack] skipped: {e}")
        if raw_path != out_path and os.path.exists(raw_path):
            os.rename(raw_path, out_path)

    size_mb = os.path.getsize(out_path) / (1 << 20)
    print(f"[done] {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out",  default=DEFAULT_OUT)
    ap.add_argument("--src",  default=_DEFAULT_SRC,
                    help="Path to a directory containing the `latent_pitch/` package.")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dummy-frames", type=int, default=256,
                    help="Frames in the traced dummy input. The transformer's "
                         "attention reshape gets baked to this value during "
                         "export, so runtime calls must pass exactly this many "
                         "frames (client pads the tail chunk).")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    export(args.ckpt, args.out, src_dir=args.src,
           opset=args.opset, dummy_frames=args.dummy_frames, device=args.device)
