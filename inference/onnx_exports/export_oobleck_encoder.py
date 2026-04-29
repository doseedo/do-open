"""Export the Oobleck VAE encoder to ONNX for WebGPU inference.

Mirrors the existing decoder export. Output is a single .onnx file
the browser can load via ONNX Runtime Web with executionProviders: ['webgpu'].

Usage:
    python -m onnx_export.export_oobleck_encoder \\
        --vae /scratch/ACE-Step-1.5/checkpoints/vae \\
        --out /scratch/onnx/oobleck_encoder.onnx \\
        --opset 17

The encoder takes [B, 2, S] @ 48kHz stereo float32 and returns
[B, 64, T] latent (T = S / 1920 frames). Browser will use this to
encode user-uploaded audio without the backend.

The model uses a wrapper that returns the .latent_dist.mode() (NOT
.sample()) so the export is deterministic — sampling adds noise that
isn't reproducible from a deterministic ONNX graph.
"""
from __future__ import annotations
import argparse, os, hashlib
import torch
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck

SR = 48000
SAMPLES_PER_FRAME = 1920


class EncoderWrapper(torch.nn.Module):
    """Thin wrapper exposing only the encode path. Returns the
    distribution's MODE (deterministic — equivalent to .sample() at
    zero noise) so the export is reproducible. The mode is used by
    every downstream consumer that doesn't need stochastic sampling
    (cover-mode latents, soundfont caching, etc.)."""

    def __init__(self, vae: AutoencoderOobleck):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, S] stereo @ 48k
        dist = self.vae.encode(x).latent_dist
        # mode = distribution mean — deterministic
        return dist.mode()  # [B, 64, T]


def vae_version_hash(vae_dir: str) -> str:
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


def export(vae_dir: str, out_path: str, opset: int = 17,
           dummy_seconds: float = 2.0):
    print(f"[init] loading vae from {vae_dir}")
    vae = AutoencoderOobleck.from_pretrained(vae_dir).to("cuda").float().eval()
    wrapper = EncoderWrapper(vae).eval()

    # Build a dummy input long enough that ONNX captures the right
    # operator shape inference. We trace at 2s = 96000 samples = 50
    # frames. Dynamic axis on time so the browser can pass any length.
    dummy_samples = int(round(dummy_seconds * SR))
    # Round to a multiple of SAMPLES_PER_FRAME so the encoder doesn't
    # truncate the test input.
    dummy_samples = (dummy_samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    x = torch.randn(1, 2, dummy_samples, device="cuda").float()
    print(f"[trace] dummy input shape={tuple(x.shape)} "
          f"({dummy_samples / SR:.2f}s, {dummy_samples // SAMPLES_PER_FRAME} frames)")

    # Sanity-check forward
    with torch.no_grad():
        y = wrapper(x)
    print(f"[trace] output shape={tuple(y.shape)} "
          f"(expected channels=64, frames={dummy_samples // SAMPLES_PER_FRAME})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[export] writing {out_path}")
    torch.onnx.export(
        wrapper,
        (x,),
        out_path,
        input_names=["audio"],
        output_names=["latent"],
        dynamic_axes={
            "audio": {0: "batch", 2: "samples"},
            "latent": {0: "batch", 2: "frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )
    # Pack weights inline so the browser can fetch ONE file instead of
    # needing both .onnx and .onnx.data side by side.
    try:
        import onnx
        m = onnx.load(out_path, load_external_data=True)
        packed_path = out_path.replace(".onnx", "_packed.onnx")
        onnx.save_model(m, packed_path, save_as_external_data=False)
        print(f"[pack] {packed_path} ({os.path.getsize(packed_path)/(1<<20):.1f} MB)")
        # Remove the loose external data file once the packed copy exists.
        ext = out_path + ".data"
        if os.path.exists(ext):
            os.unlink(ext)
            os.unlink(out_path)
            print(f"[clean] removed {ext} and unpacked {out_path}")
    except Exception as e:
        print(f"[pack] skipped: {e}")
    size_mb = os.path.getsize(out_path) / (1 << 20) if os.path.exists(out_path) else 0
    print(f"[done] {out_path} ({size_mb:.1f} MB)")
    print(f"[vae_version] {vae_version_hash(vae_dir)}")
    print(f"  → embed this hash in your browser code so latents from")
    print(f"    a future VAE can't be silently mis-decoded.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="/scratch/ACE-Step-1.5/checkpoints/vae")
    ap.add_argument("--out", default="/scratch/onnx/oobleck_encoder.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dummy-seconds", type=float, default=2.0)
    args = ap.parse_args()
    export(args.vae, args.out, opset=args.opset, dummy_seconds=args.dummy_seconds)
