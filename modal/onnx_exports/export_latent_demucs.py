"""Export the LatentDemucs student to ONNX for WebGPU inference.

The student is `WaveformToFourStemLatents` from
/scratch/latent_demucs/code/distill_model.py. It takes [1, 2, samples]
@ 48 kHz stereo and returns [1, 4, 64, T] (drums, bass, vocals, other).

Output:
    /scratch/onnx/latent_demucs_student_packed.onnx (single packed file)

Usage:
    python -m onnx_export.export_latent_demucs

Then the browser fetches it via the existing /api/onnx/<file> endpoint
and runs the entire stem separation pipeline locally — no backend GPU
involved at all once the model is cached.
"""
from __future__ import annotations
import argparse, os, sys, hashlib
import torch

# Student source lives next to its checkpoint on this VM.
sys.path.insert(0, "/scratch/latent_demucs/code")

SR = 48000
SAMPLES_PER_FRAME = 1920
DEFAULT_CKPT = "/scratch/latent_demucs/distill_final.pt"
DEFAULT_OUT  = "/scratch/onnx/latent_demucs_student_packed.onnx"


def vae_version_hash(vae_dir: str = "/scratch/ACE-Step-1.5/checkpoints/vae") -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(vae_dir)):
        p = os.path.join(vae_dir, fn)
        if not os.path.isfile(p):
            continue
        h.update(fn.encode())
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b: break
                h.update(b)
    return h.hexdigest()[:12]


def export(ckpt_path: str = DEFAULT_CKPT,
           out_path: str = DEFAULT_OUT,
           opset: int = 17,
           dummy_seconds: float = 4.0):
    from distill_model import WaveformToFourStemLatents  # type: ignore

    print(f"[init] loading student from {ckpt_path}")
    model = WaveformToFourStemLatents().to("cuda").float().eval()
    sd = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(state)

    # Sanity dummy input — must be a multiple of SAMPLES_PER_FRAME so
    # the encoder backbone produces a clean integer number of latent
    # frames. ONNX dynamic axis covers any length later.
    dummy_samples = int(round(dummy_seconds * SR))
    dummy_samples = (dummy_samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    x = torch.randn(1, 2, dummy_samples, device="cuda").float()
    print(f"[trace] dummy input shape={tuple(x.shape)} "
          f"({dummy_samples / SR:.2f}s, {dummy_samples // SAMPLES_PER_FRAME} frames)")

    with torch.no_grad():
        y = model(x)
    print(f"[trace] output shape={tuple(y.shape)} "
          f"(expected [1, n_stems=4, 64, T={dummy_samples // SAMPLES_PER_FRAME}])")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw_path = out_path.replace("_packed.onnx", ".onnx")
    print(f"[export] writing {raw_path}")
    torch.onnx.export(
        model,
        (x,),
        raw_path,
        input_names=["audio"],
        output_names=["stem_latents"],
        dynamic_axes={
            "audio": {0: "batch", 2: "samples"},
            "stem_latents": {0: "batch", 3: "frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )

    # Pack weights inline (the new dynamo exporter splits to .onnx + .onnx.data)
    try:
        import onnx
        m = onnx.load(raw_path, load_external_data=True)
        onnx.save_model(m, out_path, save_as_external_data=False)
        print(f"[pack] {out_path} ({os.path.getsize(out_path)/(1<<20):.1f} MB)")
        ext = raw_path + ".data"
        if os.path.exists(ext):
            os.unlink(ext)
        if os.path.exists(raw_path):
            os.unlink(raw_path)
    except Exception as e:
        print(f"[pack] skipped: {e}")

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1 << 20)
        print(f"[done] {out_path} ({size_mb:.1f} MB)")
        print(f"[vae_version] {vae_version_hash()}")
        print(f"  → the student is coupled to this VAE checkpoint; latents")
        print(f"    it produces are only valid against the matching decoder.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dummy-seconds", type=float, default=4.0)
    args = ap.parse_args()
    export(args.ckpt, args.out, opset=args.opset, dummy_seconds=args.dummy_seconds)
