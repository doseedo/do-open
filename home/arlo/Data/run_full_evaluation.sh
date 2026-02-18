#!/bin/bash
# run_full_evaluation.sh — Run all evaluation modes for the Performer-AI paper.
#
# Usage:  bash /home/arlo/Data/run_full_evaluation.sh [n_samples] [steps]
#   n_samples: number of test samples per mode (default: 50)
#   steps:     Euler denoising steps (default: 30)

set -e

N=${1:-50}
STEPS=${2:-30}
OUT_BASE="/home/arlo/Data/eval_output"
CKPT="/mnt/models/epoch=102-step=60000.ckpt"
TEST="/home/arlo/Data/test_manifest.json"
CKPT_DIR="/home/arlo/Data/ACE-Step/checkpoints"

cd /home/arlo/Data
eval "$(conda shell.bash hook)" && conda activate ace_step

echo "============================================"
echo "Performer-AI Full Evaluation"
echo "  Samples per mode: $N"
echo "  Euler steps: $STEPS"
echo "  Checkpoint: $CKPT"
echo "============================================"

for MODE in full no_pianoroll no_pitchbend no_ctrlbranch unconditional; do
    mkdir -p "${OUT_BASE}/${MODE}"
    echo ""
    echo ">>> Running mode: $MODE"
    echo "============================================"
    python3 evaluate_performer.py \
        --ckpt "$CKPT" \
        --test_manifest "$TEST" \
        --checkpoint_dir "$CKPT_DIR" \
        --out_dir "${OUT_BASE}/${MODE}" \
        --n_samples "$N" \
        --steps "$STEPS" \
        --mode "$MODE" \
        2>&1 | tee "${OUT_BASE}/${MODE}/log.txt"
done

echo ""
echo "============================================"
echo "All modes complete. Generating comparison table..."
echo "============================================"

python3 - << 'PYEOF'
import json, os
from pathlib import Path

base = Path("/home/arlo/Data/eval_output")
modes = ["full", "no_pianoroll", "no_pitchbend", "no_ctrlbranch", "unconditional"]

rows = []
for mode in modes:
    rpath = base / mode / f"results_{mode}.json"
    if not rpath.exists():
        print(f"  [skip] {mode}: no results")
        continue
    with open(rpath) as f:
        r = json.load(f)
    s = r.get("summary", {})
    fad = r.get("fad", {}).get("FAD_vggish", "N/A")
    rows.append({
        "mode": mode,
        "pitch_acc": s.get("pitch_acc", {}).get("mean", 0),
        "chroma_acc": s.get("chroma_acc", {}).get("mean", 0),
        "f0_rmse": s.get("f0_rmse_cents", {}).get("mean", 0),
        "onset_f1": s.get("onset_f1", {}).get("mean", 0),
        "dyn_corr": s.get("dynamics_pearson", {}).get("mean", 0),
        "timbre": s.get("timbre_cosine", {}).get("mean", 0),
        "group_acc": s.get("group_correct", {}).get("mean", 0),
        "fad": fad,
    })

print("\n" + "=" * 100)
print("COMPARISON TABLE — Performer-AI Ablation Study")
print("=" * 100)
header = f"{'Mode':20s} {'Pitch↑':>8s} {'Chroma↑':>8s} {'F0 RMSE↓':>9s} {'Onset F1↑':>10s} {'Dyn ρ↑':>8s} {'Timbre↑':>8s} {'Grp Acc↑':>9s} {'FAD↓':>8s}"
print(header)
print("-" * 100)
for r in rows:
    fad_str = f"{r['fad']:.2f}" if isinstance(r['fad'], (int,float)) else str(r['fad'])
    print(f"{r['mode']:20s} {r['pitch_acc']:8.3f} {r['chroma_acc']:8.3f} {r['f0_rmse']:9.1f} {r['onset_f1']:10.3f} {r['dyn_corr']:8.3f} {r['timbre']:8.3f} {r['group_acc']:9.3f} {fad_str:>8s}")

# LaTeX table
print("\n--- LaTeX ---")
print(r"\begin{tabular}{lcccccccc}")
print(r"\toprule")
print(r"Configuration & Pitch Acc $\uparrow$ & Chroma Acc $\uparrow$ & F0 RMSE $\downarrow$ & Onset F1 $\uparrow$ & Dyn. $\rho$ $\uparrow$ & Timbre Sim $\uparrow$ & Group Acc $\uparrow$ & FAD $\downarrow$ \\")
print(r"\midrule")
for r in rows:
    name = r['mode'].replace("_", " ").title()
    fad_str = f"{r['fad']:.2f}" if isinstance(r['fad'], (int,float)) else str(r['fad'])
    print(f"{name} & {r['pitch_acc']:.3f} & {r['chroma_acc']:.3f} & {r['f0_rmse']:.1f} & {r['onset_f1']:.3f} & {r['dyn_corr']:.3f} & {r['timbre']:.3f} & {r['group_acc']:.3f} & {fad_str} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")

# Save combined results
combined = {"modes": rows}
with open(base / "combined_results.json", "w") as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nCombined results saved to {base / 'combined_results.json'}")
PYEOF
