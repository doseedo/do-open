# test_loader_full.py
import os, random, math, sys
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import PerformerAIDataset, collate_latent_cond

JSON = "/home/arlo/Data/final_training_manifest_final.json"

# Sample sizes
N_ITEM_CHECK = 200          # per-item (no collate) deep checks
BATCH = 4                   # batch size to test collate/padding
NUM_WORKERS = 4

# Tolerances copied from your validator logic
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR, ENC_HOP = 24000, 320
SLOW_HZ = DCAE_SR / DCAE_HOP
FAST_PER_SLOW = (ENC_SR / ENC_HOP) / SLOW_HZ  # ~6.96

ENC_ABS_FLOOR_FRAMES = 24
ENC_REL_FRACTION = 0.015  # 1.5%

AMP_ACTIVITY_THR = 0.06   # must match dataloader init if you changed it

def expected_fast_len(t_slow: int) -> int:
    return int(round(t_slow * FAST_PER_SLOW))

def enc_within_tol(actual: int, ref_slow: int) -> bool:
    exp = expected_fast_len(ref_slow)
    tol = max(ENC_ABS_FLOOR_FRAMES, int(ENC_REL_FRACTION * exp))
    return abs(actual - exp) <= tol

def active_notes_at_t(pr: torch.Tensor, t: int) -> List[int]:
    # pr: [128, T]
    col = pr[:, t]
    return torch.nonzero(col > 0, as_tuple=False).view(-1).tolist()

def is_5th_or_octave(n1: int, n2: int, tol: int = 1) -> bool:
    d = abs(int(n1) - int(n2))
    for target in (12, 7):
        if abs(d - target) <= tol:
            return True
    return False

def first_activity_frame(piano_roll: torch.Tensor, rframe: torch.Tensor, amp: torch.Tensor) -> int:
    """
    Estimate first activity after trim to sanity check pre-roll:
    We find earliest t where (any PR) or (rframe>0) or (amp>thr).
    """
    T = piano_roll.shape[-1]
    idxs = []
    if piano_roll is not None and piano_roll.numel() > 0:
        any_pr = (piano_roll > 0).any(dim=0)
        if any_pr.any():
            idxs.append(int(torch.nonzero(any_pr, as_tuple=False)[0].item()))
    if rframe is not None and (rframe > 0).any():
        idxs.append(int(torch.nonzero(rframe > 0, as_tuple=False)[0].item()))
    if amp is not None and (amp > AMP_ACTIVITY_THR).any():
        idxs.append(int(torch.nonzero(amp > AMP_ACTIVITY_THR, as_tuple=False)[0].item()))
    if len(idxs) == 0:
        return 0
    return min(idxs)

def pretty_pct(x, n):
    return f"{x}/{n} ({(100.0*x/max(1,n)):.1f}%)"

def main():
    ds = PerformerAIDataset(
        json_path=JSON,
        use_trim=True,
        pre_roll_seconds=1.0,
        keep_untrimmed_prob=0.1,
        conditioning_dropout={"piano_roll":0.2, "amp":0.1, "rbend":0.1, "rframe":0.05},
        require_all_core=False
    )
    print(f"Dataset size: {len(ds)} items")

    # ---------- Per-item checks ----------
    indices = random.sample(range(len(ds)), k=min(N_ITEM_CHECK, len(ds)))

    # Stats/violations we’ll collect
    slow_align_ok = 0
    enc_align_ok  = 0

    rbend_mask_consistent_ok = 0
    rframe_gating_ok = 0
    family_rule_ok = 0
    mono_rule_ok = 0

    # dropout: count how many streams are exactly-all-zero (proxy for dropout or natural silence)
    drop_counts = {k:0 for k in ["piano_roll","amp","rbend","rframe"]}
    drop_total = 0

    # trimming sanity: first activity should not be way after start (since we leave ~1s pre-roll)
    trim_ok = 0
    trim_checked = 0

    # store first few violations
    violations = {
        "slow_align": [],
        "enc_align": [],
        "rbend_mask": [],
        "rframe_gate": [],
        "family": [],
        "mono_rule": [],
        "trim": [],
    }

    for i in indices:
        item = ds[i]  # dict single item
        lat = item["latents"]                 # [8,16,T_slow]
        enc = item["encodec_tokens"]          # [C,T_fast]
        conds = item["conds"]
        pr = conds["piano_roll"]              # [128,T_slow]
        amp = conds["amp"]                    # [T_slow]
        rb  = conds["rbend"]                  # [T_slow]
        rf  = conds["rframe"]                 # [T_slow]
        rbm = conds["rbend_mask"]             # [T_slow]
        group_id = item["instrument"]["group_id"].item()
        subgroup_id = item["instrument"]["subgroup_id"].item()
        group_name = item["instrument"]["group"]  # provided by dataloader
        subgroup_name = item["instrument"]["subgroup"]

        Tslow = lat.shape[-1]
        # 1) Slow-grid alignment
        if pr.shape[-1]==Tslow and amp.shape[-1]==Tslow and rb.shape[-1]==Tslow and rf.shape[-1]==Tslow and rbm.shape[-1]==Tslow:
            slow_align_ok += 1
        else:
            violations["slow_align"].append(i)

        # 2) Encodec proportionality
        if enc_within_tol(enc.shape[-1], Tslow):
            enc_align_ok += 1
        else:
            violations["enc_align"].append((i, enc.shape[-1], expected_fast_len(Tslow)))

        # 3) rbend mask consistency: rb is nonzero only where rbm==1
        nonzero_rb = (rb.abs() > 1e-8)
        bad_mask = (nonzero_rb & (rbm <= 0.5))
        if not bad_mask.any():
            rbend_mask_consistent_ok += 1
        else:
            violations["rbend_mask"].append(i)

        # 4) rframe gating: when rframe==0, rbend should be ~0 (mask uses rf too)
        bad_rf = (rb.abs() > 1e-8) & (rf <= 0.5)
        if not bad_rf.any():
            rframe_gating_ok += 1
        else:
            violations["rframe_gate"].append(i)

        # 5) family rules: piano → rbend==0 ; guitar/plucked → rbend==0
        fam_ok = True
        if group_name == "piano":
            if (rb.abs() > 1e-8).any():
                fam_ok = False
        if (group_name == "guitar" and subgroup_name == "plucked"):
            if (rb.abs() > 1e-8).any():
                fam_ok = False
        if fam_ok:
            family_rule_ok += 1
        else:
            violations["family"].append((i, group_name, subgroup_name))

        # 6) mono rule proxy:
        # Where rbend_mask==1 (true “allowed” spots), we expect PR to have <=2 active notes;
        # and if ==2, they should be 5th/octave (±1 semitone).
        # (Skip if PR is all zeros.)
        mono_ok = True
        if (pr > 0).any() and (rbm > 0.5).any():
            t_idxs = torch.nonzero(rbm > 0.5, as_tuple=False).view(-1)
            # sample up to 32 time points for speed
            if len(t_idxs) > 32:
                t_idxs = t_idxs[torch.randperm(len(t_idxs))[:32]]
            for t in t_idxs:
                act = active_notes_at_t(pr, int(t))
                if len(act) > 2:
                    mono_ok = False
                    break
                if len(act) == 2 and not is_5th_or_octave(act[0], act[1], tol=1):
                    mono_ok = False
                    break
        # Also: if PR is empty but rbm>0 somewhere (unlikely), we treat as inconclusive (don’t fail).
        if mono_ok:
            mono_rule_ok += 1
        else:
            violations["mono_rule"].append(i)

        # 7) dropout proxy: count exact-all-zero vectors (per stream)
        drop_total += 1
        for k in ["piano_roll","amp","rbend","rframe"]:
            x = conds[k]
            if x.numel() == 0:
                drop_counts[k] += 1
            else:
                if x.dim() == 2:  # PR
                    if (x.abs().sum() < 1e-9):
                        drop_counts[k] += 1
                else:
                    if (x.abs().sum() < 1e-9):
                        drop_counts[k] += 1

        # 8) trim sanity: first activity index should be reasonably near the start
        # We left ~1s pre-roll. After trimming, first activity should be not way out (>2s).
        # (If everything is silent, we skip check.)
        fa = first_activity_frame(pr, rf, amp)
        if fa == 0 and not (pr>0).any() and not (rf>0.5).any() and not (amp>AMP_ACTIVITY_THR).any():
            pass  # truly silent item → ignore
        else:
            trim_checked += 1
            # allow up to ~2s before first activity (pre_roll + wiggle)
            max_ok = int(round(2.0 * SLOW_HZ))
            if fa <= max_ok:
                trim_ok += 1
            else:
                violations["trim"].append((i, fa))

    # ---------- Collate / padding checks ----------
    # Build a small batch and assert padded tails are zero
    batch_indices = random.sample(range(len(ds)), k=min(BATCH, len(ds)))
    items = [ds[i] for i in batch_indices]
    # keep originals' lengths to validate padding
    item_Tslow = [it["latents"].shape[-1] for it in items]
    item_Tfast = [it["encodec_tokens"].shape[-1] for it in items]
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                    collate_fn=collate_latent_cond, pin_memory=True)
    # But DataLoader will start at 0; instead just apply collate directly to our sampled items:
    batch = collate_latent_cond(items)
    blat = batch["latents"]           # [B, 8, 16, T_slow_max]
    benc = batch["encodec_tokens"]    # [B, C, T_fast_max]
    bconds = batch["conds"]

    Tslow_max = blat.shape[-1]
    Tfast_max = benc.shape[-1]

    pad_fail = []
    for bi, (Tsi, Tfi) in enumerate(zip(item_Tslow, item_Tfast)):
        # latent tail zeros
        if (blat[bi, :, :, Tsi:] != 0).any():
            pad_fail.append(("latents", bi, Tsi, Tslow_max))
        # encodec tail zeros
        if (benc[bi, :, Tfi:] != 0).any():
            pad_fail.append(("encodec", bi, Tfi, Tfast_max))
        # cond tails zeros
        for ck, cv in bconds.items():
            if (cv[bi, ..., Tsi:] != 0).any():
                pad_fail.append((ck, bi, Tsi, Tslow_max))

    # ---------- Report ----------
    print("\n--- DATASET RULES CHECK ---")
    print(f"Checked items: {len(indices)}")
    print(f"Slow-grid alignment:     {pretty_pct(slow_align_ok, len(indices))}")
    print(f"Encodec ~ 6.96x ratio:   {pretty_pct(enc_align_ok, len(indices))}")
    print(f"rbend masked correctly:  {pretty_pct(rbend_mask_consistent_ok, len(indices))}")
    print(f"rframe gating ok:        {pretty_pct(rframe_gating_ok, len(indices))}")
    print(f"Family rules ok:         {pretty_pct(family_rule_ok, len(indices))}")
    print(f"Mono-rule proxy ok:      {pretty_pct(mono_rule_ok, len(indices))}")
    if trim_checked > 0:
        print(f"Trim pre-roll sanity:    {pretty_pct(trim_ok, trim_checked)}")
    else:
        print("Trim pre-roll sanity:    (no active items to check)")

    # Dropout proxies
    print("\nApprox. stream zero-rate (dropout+silence):")
    for k,v in drop_counts.items():
        print(f"  {k:11s}: {pretty_pct(v, drop_total)}")

    if pad_fail:
        print("\nPadding FAILURES (tail not zero):")
        for r in pad_fail[:10]:
            print("  ", r)
    else:
        print("\nPadding tails zero:      OK")

    # Show a few violations (if any) for debugging
    def show_some(tag, N=5):
        arr = violations[tag]
        if not arr: return
        print(f"\nExample {tag} violations:")
        for e in arr[:N]:
            print("  ", e)

    show_some("slow_align")
    show_some("enc_align")
    show_some("rbend_mask")
    show_some("rframe_gate")
    show_some("family")
    show_some("mono_rule")
    show_some("trim")

    # Exit non-zero if any hard failures
    hard_fail = (
        slow_align_ok < len(indices) or
        enc_align_ok  < len(indices) or
        rbend_mask_consistent_ok < len(indices) or
        rframe_gating_ok < len(indices) or
        family_rule_ok < len(indices) or
        len(pad_fail) > 0
    )
    if hard_fail:
        print("\n❌ One or more hard checks failed. See details above.")
        sys.exit(1)
    else:
        print("\n✅ All hard checks passed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
