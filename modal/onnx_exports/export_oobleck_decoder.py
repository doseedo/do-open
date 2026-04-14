"""Export the Oobleck VAE decoder to ONNX for WebGPU inference.
Mirrors export_oobleck_encoder.py. Output: a single packed .onnx file
the browser fetches via /api/onnx/oobleck_decoder_packed.onnx."""
from __future__ import annotations
import argparse, os, hashlib
import torch
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck

SR = 48000
SAMPLES_PER_FRAME = 1920


class DecoderWrapper(torch.nn.Module):
    """Thin wrapper exposing only the decode path. Returns the decoded
    waveform [B, 2, S] from a [B, 64, T] latent."""

    def __init__(self, vae: AutoencoderOobleck):
        super().__init__()
        self.vae = vae

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent).sample


def vae_version_hash(vae_dir: str) -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(vae_dir)):
        p = os.path.join(vae_dir, fn)
        if not os.path.isfile(p): continue
        h.update(fn.encode())
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b: break
                h.update(b)
    return h.hexdigest()[:12]


def export(vae_dir: str, out_path: str, opset: int = 17, dummy_frames: int = 64):
    print(f"[init] loading vae from {vae_dir}")
    vae = AutoencoderOobleck.from_pretrained(vae_dir).to("cuda").float().eval()
    wrapper = DecoderWrapper(vae).eval()

    # Dummy [B=1, 64, T] latent — dynamic axis on T so any length works
    x = torch.randn(1, 64, dummy_frames, device="cuda").float()
    print(f"[trace] dummy input shape={tuple(x.shape)} ({dummy_frames} frames = {dummy_frames*SAMPLES_PER_FRAME/SR:.2f}s)")
    with torch.no_grad():
        y = wrapper(x)
    print(f"[trace] output shape={tuple(y.shape)} (expected channels=2)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw_path = out_path.replace("_packed.onnx", ".onnx")
    print(f"[export] writing {raw_path}")
    torch.onnx.export(
        wrapper, (x,), raw_path,
        input_names=["latent"], output_names=["audio"],
        dynamic_axes={"latent": {0: "batch", 2: "frames"},
                      "audio":  {0: "batch", 2: "samples"}},
        opset_version=opset, do_constant_folding=True, export_params=True,
    )

    try:
        import onnx
        m = onnx.load(raw_path, load_external_data=True)
        onnx.save_model(m, out_path, save_as_external_data=False)
        print(f"[pack] {out_path} ({os.path.getsize(out_path)/(1<<20):.1f} MB)")
        ext = raw_path + ".data"
        if os.path.exists(ext): os.unlink(ext)
        if os.path.exists(raw_path): os.unlink(raw_path)
    except Exception as e:
        print(f"[pack] skipped: {e}")

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1 << 20)
        print(f"[done] {out_path} ({size_mb:.1f} MB)")
        print(f"[vae_version] {vae_version_hash(vae_dir)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="/scratch/ACE-Step-1.5/checkpoints/vae")
    ap.add_argument("--out", default="/scratch/onnx/oobleck_decoder_packed.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dummy-frames", type=int, default=64)
    args = ap.parse_args()
    export(args.vae, args.out, opset=args.opset, dummy_frames=args.dummy_frames)
