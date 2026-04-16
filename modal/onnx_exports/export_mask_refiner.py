"""Export LatentMaskRefiner to ONNX fp16 for WebGPU.

Runs per-stem as the final step in the upload pipeline:
  v4-small (noisy masks) → v4cond-pred (clean stem latents) →
  THIS MODEL → refined masks → maskPlayback AudioWorklet

Inputs:
  latent      [B, 64, T]          per-stem clean latent from v4cond-pred
  noisy_mask  [B, 1025, T_stft]   per-stem noisy softmax mask from v4-small

Output:
  refined_mask [B, 1025, T_stft]  sigmoid-normalized refined mask

~1.5M params. Tiny.

Shipped checkpoint: refiner_step20000.pt (final, 20K/20K, +25.2% SI-SDR
over the v4-small input masks).
"""
from __future__ import annotations
import argparse, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latent_demucs_student"))

DEFAULT_CKPT = "/scratch/latent_demucs_student/latent_mask_refiner_ckpts/refiner_step20000.pt"
DEFAULT_OUT = "/mnt/work/tmp/mask_refiner_fp16.onnx"


def export(ckpt_path, out_path, dummy_seconds=4.0, opset=17, fp16=True):
    from latent_mask_refiner import LatentMaskRefiner

    print(f"[mask-refiner export] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd

    model = LatentMaskRefiner(latent_dim=64, n_freqs=1025).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  missing={len(missing)}: {missing[:4]}")
    if unexpected: print(f"  unexpected={len(unexpected)}: {unexpected[:4]}")
    model = model.float()

    SR = 48000
    SAMPLES_PER_FRAME = 1920
    samples = int(round(dummy_seconds * SR))
    samples = (samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    T_latent = samples // SAMPLES_PER_FRAME
    t_stft = samples // 512 + 1

    latent     = torch.randn(1, 64, T_latent) * 0.1
    noisy_mask = torch.rand(1, 1025, t_stft)  # values in [0, 1]

    with torch.no_grad():
        y = model(latent, noisy_mask)
    print(f"[mask-refiner export] trace out shape={tuple(y.shape)} "
          f"(expected [1, 1025, {t_stft}])")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    raw_path = out_path.replace(".onnx", ".fp32.onnx")
    torch.onnx.export(
        model, (latent, noisy_mask), raw_path,
        input_names=["latent", "noisy_mask"],
        output_names=["refined_mask"],
        dynamic_axes={
            "latent":       {0: "batch", 2: "t_latent"},
            "noisy_mask":   {0: "batch", 2: "t_stft"},
            "refined_mask": {0: "batch", 2: "t_stft"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

    import onnx
    m = onnx.load(raw_path, load_external_data=True)

    if fp16:
        print("[mask-refiner export] converting weights to fp16")
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(
            m, keep_io_types=True, disable_shape_infer=False,
            op_block_list=["Range", "Shape", "NonZero"],
        )

    # Refiner is tiny — pack inline (no external .data file).
    onnx.save_model(m, out_path, save_as_external_data=False)
    if os.path.exists(raw_path): os.unlink(raw_path)
    if os.path.exists(raw_path + ".data"): os.unlink(raw_path + ".data")

    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"[mask-refiner export] DONE — {sz_mb:.1f} MB ({'fp16' if fp16 else 'fp32'})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dummy-seconds", type=float, default=4.0)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()
    export(args.ckpt, args.out, opset=args.opset,
           dummy_seconds=args.dummy_seconds, fp16=not args.no_fp16)
