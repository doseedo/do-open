"""Export v4cond-pred 6-stem student to ONNX (fp16) for WebGPU.

SmallAdditiveDemucsV4 — student trained with PREDICTED conditioning
from frozen v4-small. This is the strict successor to distill_demucs:
same output shape [1, S, 64, T_enc], but S=6 and takes the v4-small
outputs as conditioning. The frontend chains semDemucsV4.analyze
(already wired) → this model → per-stem DOAE upload.

Inputs (three, because the model is conditioned on v4-small):
  waveform     [1, 2, N]           stereo 48 kHz
  sem_emb      [1, 6, 128]         from semDemucsV4.embedding
  stft_masks   [1, 6, 1025, T_stft] from semDemucsV4.stft_masks

Output:
  stem_latents [1, 6, 64, T_enc]   feeds uploadLatent + VAE decoder

Size target: ~165 MB weights in fp16 (half of the fp32 baseline).
Converting post-export via onnxconverter_common to avoid tracing
instability under torch.autocast.

Stem order (from DistillDataset6.STEMS_6):
  0 drums  1 bass  2 vocals  3 other  4 guitar  5 piano
"""
from __future__ import annotations
import argparse, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latent_demucs_student"))

SR = 48000
DEFAULT_CKPT = "/scratch/latent_demucs_student/v4cond_predicted_6stem_ckpts/v4cond_pred_final.pt"
DEFAULT_OUT = "/tmp/v4cond_pred_6s_fp16.onnx"


def export(ckpt_path, out_path, n_stems=6, hidden=96, dummy_seconds=4.0, opset=17, fp16=True):
    from train_distill_small_v4cond import SmallAdditiveDemucsV4

    print(f"[v4cond-pred export] loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model", sd) if isinstance(sd, dict) else sd

    # Build model at the same size as training. The train script defaults:
    #   n_stems=6, hidden=96, mults=(1,2,4,8,16)
    model = SmallAdditiveDemucsV4(n_stems=n_stems, hidden=hidden).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  missing={len(missing)}: {missing[:4]}")
    if unexpected: print(f"  unexpected={len(unexpected)}: {unexpected[:4]}")

    model = model.float()

    # Dummy inputs at traced shapes — dynamic axis covers variable length.
    # samples must hit a multiple of 1920 so the oobleck encoder gives a
    # clean integer T_enc.
    SAMPLES_PER_FRAME = 1920
    samples = int(round(dummy_seconds * SR))
    samples = (samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    # v4-small stft uses n_fft=2048, hop=512 → T_stft = samples/512 + 1
    t_stft = samples // 512 + 1

    waveform   = torch.zeros(1, 2, samples)
    sem_emb    = torch.zeros(1, n_stems, 128)
    stft_masks = torch.zeros(1, n_stems, 1025, t_stft)

    with torch.no_grad():
        y = model(waveform, sem_emb, stft_masks)
    print(f"[v4cond-pred export] trace out shape={tuple(y.shape)} "
          f"(expected [1, {n_stems}, 64, T_enc])")

    # ── 1. Raw fp32 ONNX export ──────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    raw_path = out_path.replace(".onnx", ".fp32.onnx")
    print(f"[v4cond-pred export] writing raw fp32 → {raw_path}")
    # dynamo=False pins the legacy torch.onnx exporter — the new
    # dynamo-based path chokes on dynamic_axes that don't cover every
    # input (sem_emb is [1, 6, 128], all fixed). Legacy export is
    # stable for this architecture and still produces valid opset-17
    # graphs that ORT-Web reads.
    torch.onnx.export(
        model, (waveform, sem_emb, stft_masks), raw_path,
        input_names=["waveform", "sem_emb", "stft_masks"],
        output_names=["stem_latents"],
        dynamic_axes={
            "waveform":    {2: "n_samples"},
            "stft_masks":  {3: "t_stft"},
            "stem_latents": {3: "t_enc"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

    # ── 2. Pack external data inline so we have a clean baseline ────
    import onnx
    m = onnx.load(raw_path, load_external_data=True)

    if fp16:
        # ── 3. Convert weights to fp16 — ~2× smaller, WebGPU-native ──
        #
        # keep_io_types=True so the network still takes fp32 inputs and
        # emits fp32 outputs (browser Float32Array stays the same on
        # the boundary; cast happens inside the graph). Skip ops that
        # don't have stable fp16 support in ORT-Web 1.22.
        print("[v4cond-pred export] converting weights to fp16")
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(
            m,
            keep_io_types=True,
            disable_shape_infer=False,
            op_block_list=[
                # STFT/IFFT ops are fp32-only in most runtimes.
                # Keep Range/Shape/Gather/Cast in fp32 to avoid corner
                # cases on small int-ish tensors.
                "Range", "Shape", "NonZero",
            ],
        )

    onnx.save_model(
        m, out_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(out_path) + ".data",
        size_threshold=1024,
    )

    # Cleanup raw
    for p in [raw_path, raw_path + ".data"]:
        if os.path.exists(p): os.unlink(p)

    graph_sz = os.path.getsize(out_path) / 1e6
    data_path = out_path + ".data"
    data_sz = os.path.getsize(data_path) / 1e6 if os.path.exists(data_path) else 0
    print(f"[v4cond-pred export] DONE — graph {graph_sz:.1f} MB, weights {data_sz:.1f} MB "
          f"({'fp16' if fp16 else 'fp32'})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-stems", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dummy-seconds", type=float, default=4.0)
    ap.add_argument("--no-fp16", action="store_true", help="export as fp32 (~2× larger)")
    args = ap.parse_args()
    export(args.ckpt, args.out, n_stems=args.n_stems, hidden=args.hidden,
           opset=args.opset, dummy_seconds=args.dummy_seconds, fp16=not args.no_fp16)
