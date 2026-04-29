"""Export SemDemucs (sem-only, no masks) to ONNX for WebGPU.

Produces: sem_demucs_packed.onnx (~9 MB for 2.2M params)

Browser runs:
  waveform [1, 2, N] → sem_demucs → rms [1, 4, T, 2] + embedding [1, 4, 128]

RMS is used for instant waveform visualization.
Embedding is passed to the latent demucs as conditioning.
"""
from __future__ import annotations
import argparse, os, sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latent_demucs_student"))


class SemDemucsExport(torch.nn.Module):
    """Wrapper that returns only rms + embedding (skip mask/pitch/vocal)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        out = self.model(waveform)
        # rms: [B, 4, T, 2], embedding: [B, 4, 128]
        return out["rms"], out["embedding"]


def export(ckpt_path, out_path, opset=17):
    from sem_demucs import SemDemucs

    print(f"[sem-demucs export] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SemDemucs(channels=64)
    model.load_state_dict(sd["model"], strict=False)
    model.eval()

    wrapper = SemDemucsExport(model).eval().float()
    n = sum(p.numel() for p in wrapper.parameters()) / 1e6
    print(f"[sem-demucs export] {n:.1f}M params")

    # Dummy: 4 seconds stereo
    dummy = torch.randn(1, 2, 48000 * 4)
    with torch.no_grad():
        rms, emb = wrapper(dummy)
    print(f"[sem-demucs export] waveform {tuple(dummy.shape)} → rms {tuple(rms.shape)}, emb {tuple(emb.shape)}")

    raw_path = out_path.replace("_packed.onnx", ".onnx")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper, (dummy,), raw_path,
        input_names=["waveform"],
        output_names=["rms", "embedding"],
        dynamic_axes={
            "waveform": {2: "samples"},
            "rms": {2: "frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    try:
        import onnx
        m = onnx.load(raw_path, load_external_data=True)
        onnx.save_model(m, out_path, save_as_external_data=False)
        ext = raw_path + ".data"
        if os.path.exists(ext): os.unlink(ext)
        if os.path.exists(raw_path) and raw_path != out_path: os.unlink(raw_path)
    except Exception as e:
        print(f"[pack] skipped: {e}")

    size_mb = os.path.getsize(out_path) / (1 << 20)
    print(f"[sem-demucs export] {out_path} ({size_mb:.1f} MB)")

    # Validate
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        for dur in [2, 4, 10]:
            N = 48000 * dur
            N = (N // 1920) * 1920
            x = np.random.randn(1, 2, N).astype(np.float32)
            r, e = sess.run(None, {"waveform": x})
            print(f"  {dur}s: rms {r.shape}, emb {e.shape}")
    except ImportError:
        print("[validate] onnxruntime not installed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="/scratch/onnx/sem_demucs_packed.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    export(args.ckpt, args.out, args.opset)
