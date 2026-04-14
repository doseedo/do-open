"""Export SemMaskTempCondOobleckStudent to ONNX fp16 for WebGPU.

This replaces sem_decoder_packed.onnx in the frontend. New input signature:
  latent    [B, 64, T]
  sem_emb   [B, 128]
  stft_mask [B, 1025, T_m]   per-stem STFT magnitude mask from v4-small
→
  audio     [B, 2, T*1920]

Called from decoderWorker.js: one forward per stem, passing that stem's
sem_emb + mask from the v4-small analyze() result that's already sitting
in memory from the upload-flow classifier + conditioning pass.

Training run as of export: temp warmstart @ step 17500, wav=0.024,
meaningfully below the deployed sem_v1 baseline (~0.027).
"""
from __future__ import annotations
import argparse, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latent_oobleck_student"))

DEFAULT_CKPT = "/scratch/latent_oobleck_student/ckpts_sem_mask_temp/temp_ws_step17000.pt"
DEFAULT_OUT = "/mnt/work/tmp/sem_mask_temp_decoder_fp16.onnx"


def export(ckpt_path, out_path, dummy_seconds=4.0, opset=17, fp16=True):
    from sem_mask_temp_model import SemMaskTempCondOobleckStudent

    print(f"[sem-mask-temp export] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd

    model = SemMaskTempCondOobleckStudent().eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  missing={len(missing)}: {missing[:4]}")
    if unexpected: print(f"  unexpected={len(unexpected)}: {unexpected[:4]}")
    model = model.float()

    # Remove weight_norm parametrizations so ONNX sees plain Conv1d.
    # (weight_norm is a parametrization that doesn't export cleanly on
    # the legacy torch.onnx path.)
    def _strip_weight_norm(m):
        for child in m.children():
            _strip_weight_norm(child)
            # torch 2.x uses register_parametrization; check presence
            if hasattr(child, "parametrizations") and "weight" in getattr(child, "parametrizations", {}):
                torch.nn.utils.parametrize.remove_parametrizations(child, "weight", leave_parametrized=True)
            elif hasattr(child, "weight_g") and hasattr(child, "weight_v"):
                torch.nn.utils.remove_weight_norm(child)
    _strip_weight_norm(model)

    SR = 48000
    SAMPLES_PER_FRAME = 1920
    samples = int(round(dummy_seconds * SR))
    samples = (samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    T_latent = samples // SAMPLES_PER_FRAME          # oobleck frame rate
    t_stft = samples // 512 + 1                      # v4-small mask time axis

    latent    = torch.zeros(1, 64, T_latent)
    sem_emb   = torch.zeros(1, 128)
    stft_mask = torch.zeros(1, 1025, t_stft)

    with torch.no_grad():
        y = model(latent, sem_emb, stft_mask)
    print(f"[sem-mask-temp export] trace out shape={tuple(y.shape)} "
          f"(expected [1, 2, {T_latent * SAMPLES_PER_FRAME}])")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    raw_path = out_path.replace(".onnx", ".fp32.onnx")
    print(f"[sem-mask-temp export] writing raw fp32 → {raw_path}")
    torch.onnx.export(
        model, (latent, sem_emb, stft_mask), raw_path,
        input_names=["latent", "sem_emb", "stft_mask"],
        output_names=["audio"],
        dynamic_axes={
            "latent":    {0: "batch", 2: "t_latent"},
            "sem_emb":   {0: "batch"},
            "stft_mask": {0: "batch", 2: "t_stft"},
            "audio":     {0: "batch", 2: "n_samples"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

    import onnx
    m = onnx.load(raw_path, load_external_data=True)

    if fp16:
        print("[sem-mask-temp export] converting weights to fp16")
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(
            m, keep_io_types=True, disable_shape_infer=False,
            op_block_list=["Range", "Shape", "NonZero"],
        )

    onnx.save_model(
        m, out_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(out_path) + ".data",
        size_threshold=1024,
    )
    for p in [raw_path, raw_path + ".data"]:
        if os.path.exists(p): os.unlink(p)

    graph_sz = os.path.getsize(out_path) / 1e6
    data_sz = os.path.getsize(out_path + ".data") / 1e6 if os.path.exists(out_path + ".data") else 0
    print(f"[sem-mask-temp export] DONE — graph {graph_sz:.1f} MB, weights {data_sz:.1f} MB "
          f"({'fp16' if fp16 else 'fp32'})")
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
