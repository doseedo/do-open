# dataloader.py
import json, random
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ==== Grid rates (nominal) ====
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000,  320
SLOW_HZ = DCAE_SR / DCAE_HOP                # ~10.77 Hz
FAST_PER_SLOW = (ENC_SR / ENC_HOP) / SLOW_HZ  # ~6.96

# ==== Vocab ====
APPROVED_GROUPS = ["piano","guitar","bass","strings","brass","winds"]
APPROVED_SUBGROUPS = {
    "piano":   ["acoustic_piano","keys","undefined"],
    "guitar":  ["acoustic_guitar","electric_guitar","plucked","undefined"],
    "bass":    ["electric_bass","upright_bass","undefined"],
    "strings": ["violin","viola","cello","undefined"],
    "brass":   ["trumpet","trombone","french_horn","tuba","undefined"],
    "winds":   ["bassoon","clarinet","flute","oboe","sax"],
}

# -------- helpers --------
def _safe_np_load(path: Optional[str]) -> Optional[np.ndarray]:
    try:
        if not path: return None
        p = Path(path)
        if not (p.exists() and p.is_file()): return None
        try:
            return np.load(p)
        except ValueError:
            return np.load(p, allow_pickle=True)
    except Exception:
        return None

def _safe_pt_load(path: Optional[str]) -> Optional[torch.Tensor]:
    try:
        if not path: return None
        p = Path(path)
        if not (p.exists() and p.is_file()): return None
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            for k in ("latents","codes","tokens","encodec","audio_tokens","data"):
                if k in obj and isinstance(obj[k], torch.Tensor):
                    return obj[k]
        def first_tensor(x):
            if isinstance(x, torch.Tensor): return x
            if isinstance(x, (list,tuple)):
                for it in x:
                    t = first_tensor(it)
                    if t is not None: return t
            if isinstance(x, dict):
                for v in x.values():
                    t = first_tensor(v)
                    if t is not None: return t
            return None
        return first_tensor(obj)
    except Exception:
        return None

def _pad_last(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
    cur = x.shape[-1]
    if cur == target_len: return x
    if cur < target_len:
        if pad_value == 0.0:
            return F.pad(x, (0, target_len - cur))
        else:
            return F.pad(x, (0, target_len - cur), value=pad_value)
    return x[..., :target_len]

def _pad_dim(x: torch.Tensor, target_len: int, dim: int) -> torch.Tensor:
    cur = x.shape[dim]
    if cur == target_len: return x
    if cur > target_len:
        slc = [slice(None)] * x.dim()
        slc[dim] = slice(0, target_len)
        return x[tuple(slc)]
    pad_shape = list(x.shape)
    pad_shape[dim] = target_len - cur
    pad = x.new_zeros(*pad_shape)
    return torch.cat([x, pad], dim=dim)

def _bool_smooth(mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    if k <= 1: return mask
    pad = k // 2
    x = mask.float().view(1,1,-1)
    filt = torch.ones(1,1,k, device=x.device) / k
    y = F.conv1d(F.pad(x, (pad,pad), mode="reflect"), filt)
    return (y.view(-1) > 0.5)

def _interval_is_oct_or_fifth(n1: int, n2: int, tol: int = 1) -> bool:
    d = abs(int(n1) - int(n2))
    for target in (12,7):
        if abs(d - target) <= tol:
            return True
    return False

def compute_polyphony_mask_from_pr(piano_roll: Optional[torch.Tensor],
                                   smear_tol_frames: int = 2,
                                   allow_power_intervals: bool = True) -> Optional[torch.Tensor]:
    if piano_roll is None or piano_roll.numel() == 0:
        return None
    pr = (piano_roll > 0).to(torch.bool)   # [128,T]
    active = pr.sum(dim=0)                 # [T]
    T = pr.shape[1]
    mono = torch.zeros(T, dtype=torch.bool, device=pr.device)
    mono |= (active <= 1)
    two_mask = (active == 2)
    if two_mask.any() and allow_power_intervals:
        for t in two_mask.nonzero(as_tuple=False).view(-1):
            notes = pr[:, t].nonzero(as_tuple=False).view(-1).tolist()
            if len(notes) == 2 and _interval_is_oct_or_fifth(notes[0], notes[1]):
                mono[t] = True
    mono = _bool_smooth(mono, k=2*smear_tol_frames+1)
    return mono

def compute_rbend_mask(group, subgroup, piano_roll, rframe, amp,
                       amp_thr=0.06, smooth_k=5):
    T = int(rframe.shape[0])
    dev = rframe.device

    # --- unify amp length to T ---
    if isinstance(amp, torch.Tensor):
        amp = amp.view(-1)
        if amp.shape[-1] != T:
            amp = F.interpolate(amp.view(1,1,-1).float(), size=T, mode="linear", align_corners=False).view(-1).to(dtype=amp.dtype, device=dev)
    else:
        x  = np.linspace(0, 1, num=len(amp), endpoint=True)
        xi = np.linspace(0, 1, num=T,        endpoint=True)
        amp = torch.from_numpy(np.interp(xi, x, amp)).to(device=dev, dtype=torch.float32)

    # --- unify rframe length to T ---
    rframe = rframe.view(-1).to(dev)
    if rframe.shape[-1] != T:
        rframe = F.interpolate(rframe.view(1,1,-1).float(), size=T, mode="nearest").view(-1)

    # --- unify piano_roll time to T (nearest to keep it boolean) ---
    if piano_roll is not None:
        pr = piano_roll
        if pr.dim() == 3 and pr.shape[0] == 1:
            pr = pr.squeeze(0)
        if pr.shape[-1] != T:
            pr = F.interpolate(pr.unsqueeze(0), size=T, mode="nearest").squeeze(0)
        mono_mask = compute_polyphony_mask_from_pr(pr)
    else:
        mono_mask = None

    # --- build mask ---
    mask = torch.ones(T, dtype=torch.bool, device=dev)
    if mono_mask is not None:
        if mono_mask.shape[-1] != T:
            mono_mask = F.interpolate(mono_mask.view(1,1,-1).float(), size=T, mode="nearest").view(-1).to(torch.bool)
        mask &= mono_mask

    mask &= (rframe > 0)
    bleed = (amp > amp_thr) & (rframe == 0)
    mask &= (~bleed)
    mask &= (amp > 0.01)

    if smooth_k and smooth_k > 1:
        mask = _bool_smooth(mask, k=smooth_k)
    return mask

def _crop_fast_from_slow(enc_fast: torch.Tensor,
                         start_slow: int,
                         end_slow: int,
                         r: float) -> torch.Tensor:
    """
    enc_fast: [C_fast, T_fast]
    start_slow, end_slow: slow-grid indices (exclusive end)
    r: FAST_PER_SLOW (use per-item r_local)
    """
    expect_fast = int(round((end_slow - start_slow) * r))
    start_f = int(round(start_slow * r))
    end_f = start_f + expect_fast
    x = enc_fast[..., start_f:min(end_f, enc_fast.shape[-1])]
    C = enc_fast.shape[0]
    if x.shape[-1] < expect_fast:
        pad = enc_fast.new_zeros(C, expect_fast - x.shape[-1])
        x = torch.cat([x, pad], dim=-1)
    elif x.shape[-1] > expect_fast:
        x = x[..., :expect_fast]
    return x

def _first_active_t(pr: Optional[torch.Tensor], rframe: Optional[torch.Tensor],
                    amp: Optional[torch.Tensor], thr: float, pre_roll_frames: int) -> int:
    cands = []
    if pr is not None and pr.numel() > 0:
        any_pr = pr.any(dim=0)
        if any_pr.any(): cands.append(int(any_pr.float().argmax().item()))
    if rframe is not None and (rframe > 0).any():
        cands.append(int((rframe > 0).float().argmax().item()))
    if amp is not None:
        idxs = (amp > thr).nonzero(as_tuple=False)
        if len(idxs) > 0: cands.append(int(idxs[0].item()))
    if not cands: return 0
    return max(0, min(cands) - pre_roll_frames)

def _last_active_t(pr: Optional[torch.Tensor], rframe: Optional[torch.Tensor],
                   amp: Optional[torch.Tensor], thr: float, post_roll_frames: int,
                   T_slow: int) -> int:
    lasts = []
    if pr is not None and pr.numel() > 0:
        any_pr = pr.any(dim=0)
        if any_pr.any():
            lasts.append(int(any_pr.nonzero(as_tuple=False)[-1].item()))
    if rframe is not None and (rframe > 0).any():
        lasts.append(int((rframe > 0).nonzero(as_tuple=False)[-1].item()))
    if amp is not None and (amp > thr).any():
        lasts.append(int((amp > thr).nonzero(as_tuple=False)[-1].item()))
    if not lasts:
        return T_slow
    t_last = max(lasts) + 1 + post_roll_frames
    return min(T_slow, t_last)

class PerformerAIDataset(Dataset):
    def __init__(self,
                 json_path: str,
                 conditioning_dropout: Dict[str, float] = None,
                 use_trim: bool = True,
                 pre_roll_seconds: float = 1.0,
                 post_roll_seconds: float = 0.25,
                 keep_untrimmed_prob: float = 0.1,
                 amp_activity_thr: float = 0.06,
                 require_all_core: bool = True,
                 static_window: bool = False,
                 window_slow: int = 2048,
                 seed: Optional[int] = None,
                 collapse_sparse_subgroups_to_any: bool = False):
        super().__init__()
        self.manifest: List[Dict[str, Any]] = json.loads(Path(json_path).read_text())
        self.use_trim = use_trim
        self.pre_roll = int(round(pre_roll_seconds * SLOW_HZ))
        self.post_roll = int(round(post_roll_seconds * SLOW_HZ))
        self.keep_untrimmed_prob = keep_untrimmed_prob
        self.amp_activity_thr = amp_activity_thr
        self.require_all_core = require_all_core
        self.collapse_sparse_subgroups_to_any = collapse_sparse_subgroups_to_any
        self.static_window = static_window
        self.window_slow = int(window_slow)
        self._static_start = None

        self.allowed_cond = ["piano_roll","amp","rbend","rframe"]
        self.drop_p = conditioning_dropout or {"piano_roll":0.20, "amp":0.10, "rbend":0.10, "rframe":0.05}

        self.group2id = {g:i for i,g in enumerate(APPROVED_GROUPS)}
        all_subs = sorted({sg for g,subs in APPROVED_SUBGROUPS.items() for sg in subs})
        self.sub2id = {sg:i for i,sg in enumerate(all_subs)}

        self.rng = np.random.default_rng(seed)
        
        # Fix: Seed Python's random module for consistent dropout across workers
        if seed is not None:
            random.seed(seed)

        if self.require_all_core:
            filtered = []
            for it in self.manifest:
                lat_ok = _safe_pt_load(it.get("latent_path"))   is not None
                enc_ok = _safe_pt_load(it.get("encodec_path"))  is not None
                pr_meta = it.get("piano_roll_path")
                pr_ok = True if pr_meta is None else (_safe_np_load(pr_meta) is not None)
                if lat_ok and enc_ok and pr_ok:
                    filtered.append(it)
            self.manifest = filtered

    def __len__(self): return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.manifest[idx]
        group = (meta.get("group") or "undefined").lower()
        subgroup = (meta.get("sub_group") or "undefined").lower()
        group = group if group in self.group2id else "guitar"
        if self.collapse_sparse_subgroups_to_any and subgroup not in APPROVED_SUBGROUPS.get(group, []):
            subgroup = "undefined"
        if subgroup not in self.sub2id: subgroup = "undefined"

        # ---- load streams ----
        latents = _safe_pt_load(meta.get("latent_path"))   # [8,16,T_slow_all]
        encodec = _safe_pt_load(meta.get("encodec_path"))  # [C_fast,T_fast] or variants
        if encodec is not None and encodec.dim() == 3 and encodec.shape[0] == 1:
            encodec = encodec.squeeze(0)
        if encodec is not None:
            if encodec.dim() == 2 and encodec.shape[0] != 8 and encodec.shape[1] == 8:
                encodec = encodec.transpose(0, 1)
            C = encodec.shape[0]
            if C > 8:   encodec = encodec[:8]
            elif C < 8:
                pad = torch.zeros(8 - C, encodec.shape[1], dtype=encodec.dtype)
                encodec = torch.cat([encodec, pad], dim=0)
            if not torch.is_floating_point(encodec):
                encodec = encodec.long()

        pr_np = _safe_np_load(meta.get("piano_roll_path"))
        piano_roll = torch.from_numpy(pr_np).float() if pr_np is not None else None

        assert latents is not None and latents.dim()==3, f"bad latents at {meta.get('latent_path')}"
        T_slow_all = latents.shape[2]
        T_fast_all = encodec.shape[-1] if encodec is not None else None

        # measure per-item ratio to be *exactly* aligned with this file
        r_local = (T_fast_all / T_slow_all) if (encodec is not None and T_slow_all > 0) else FAST_PER_SLOW

        # 1-D conds
        cond_paths = meta.get("conditioning_paths") or {}
        def get1d(key):
            arr = _safe_np_load(cond_paths.get(key))
            if arr is None: return None
            t = torch.from_numpy(arr).float()
            return _pad_last(t.view(-1), T_slow_all)
        amp    = get1d("amp")
        rbend  = get1d("rbend")
        rframe = get1d("rframe")
        if piano_roll is not None:
            piano_roll = _pad_last(piano_roll, T_slow_all)

        # ---- optional trim on slow streams only ----
        t0_abs, t1_abs = 0, T_slow_all
        if self.use_trim and random.random() > self.keep_untrimmed_prob:
            t0 = _first_active_t(piano_roll, rframe, amp, self.amp_activity_thr, self.pre_roll)
            t1 = _last_active_t(piano_roll, rframe, amp, self.amp_activity_thr, self.post_roll, T_slow_all)
            if t0 >= t1: t0, t1 = 0, T_slow_all
            latents = latents[..., t0:t1]
            if piano_roll is not None: piano_roll = piano_roll[..., t0:t1]
            if amp    is not None: amp    = amp[t0:t1]
            if rbend  is not None: rbend  = rbend[t0:t1]
            if rframe is not None: rframe = rframe[t0:t1]
            t0_abs, t1_abs = t0, t1

        # ---- fixed window on trimmed slow streams ----
        T_slow = latents.shape[2]
        W = min(int(getattr(self, "window_slow", T_slow)), T_slow)
        max_start = max(0, T_slow - W)
        if self.static_window:
            if self._static_start is None:
                self._static_start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
            start = min(self._static_start, max_start)  # clamp per-sample
        else:
            start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end = start + W

        latents = latents[..., start:end]
        if piano_roll is not None: piano_roll = piano_roll[..., start:end]
        if amp    is not None: amp    = amp[start:end]
        if rbend  is not None: rbend  = rbend[start:end]
        if rframe is not None: rframe = rframe[start:end]

        # ---- crop EnCodec once using absolute slow indices & r_local ----
        if encodec is not None:
            start_abs = t0_abs + start
            end_abs   = t0_abs + end
            encodec   = _crop_fast_from_slow(encodec, start_abs, end_abs, r_local)

        # exact expected fast len from *final* slow len
        T_slow_final = latents.shape[2]
        expect_fast  = int(round(T_slow_final * r_local))
        if encodec is None:
            encodec = torch.zeros(8, expect_fast, dtype=torch.long)
        else:
            # last-mile clamp/pad for any residual mismatch
            if encodec.shape[-1] != expect_fast:
                if encodec.shape[-1] < expect_fast:
                    pad = encodec.new_zeros(encodec.shape[0], expect_fast - encodec.shape[-1])
                    encodec = torch.cat([encodec, pad], dim=-1)
                else:
                    encodec = encodec[..., :expect_fast]
        assert encodec.shape[-1] == expect_fast, (encodec.shape, expect_fast)

        # ---- per-stream dropout / defaults ----
        def maybe_zero(x, key, T):
            drop = (x is None) or (random.random() < self.drop_p.get(key, 0.0))
            return torch.zeros(T, dtype=torch.float32) if drop else x
        if piano_roll is None:
            piano_roll = torch.zeros(128, T_slow_final)
        elif random.random() < self.drop_p.get("piano_roll", 0.0):
            piano_roll = torch.zeros_like(piano_roll)
        amp    = maybe_zero(amp,   "amp",   T_slow_final)
        rframe = maybe_zero(rframe,"rframe",T_slow_final)
        rbend  = maybe_zero(rbend, "rbend", T_slow_final)

        # ---- rbend gating ----
        rb_mask = compute_rbend_mask(group, subgroup, piano_roll, rframe, amp, amp_thr=self.amp_activity_thr, smooth_k=5)
        if rb_mask.shape[0] != rbend.shape[0]:
            rb_mask = F.interpolate(rb_mask.view(1,1,-1).float(), size=rbend.shape[0], mode="nearest").view(-1)
        rbend = rbend * rb_mask.float()

        # sanity
        assert latents.shape[-1] == T_slow_final
        assert amp.shape[-1] == T_slow_final
        assert rbend.shape[-1] == T_slow_final
        assert rframe.shape[-1] == T_slow_final
        assert piano_roll.shape[-1] == T_slow_final

        return {
            "latents": latents,                 # [8,16,T_slow]
            "encodec_tokens": encodec,          # [8,T_fast]
            "conds": {
                "piano_roll": piano_roll,       # [128,T_slow]
                "amp": amp,                     # [T_slow]
                "rbend": rbend,                 # [T_slow]
                "rframe": rframe,               # [T_slow]
                "rbend_mask": rb_mask.float(),  # [T_slow]
            },
            "instrument": {
                "group": group, "subgroup": subgroup,
                "group_id": torch.tensor(self.group2id.get(group,0), dtype=torch.long),
                "subgroup_id": torch.tensor(self.sub2id.get(subgroup,0), dtype=torch.long),
            },
            "meta": {
                "audio_path": meta.get("audio_path",""),
                "latent_path": meta.get("latent_path",""),
                "encodec_path": meta.get("encodec_path",""),
                "piano_roll_path": meta.get("piano_roll_path",""),
            }
        }

def collate_latent_cond(batch: List[Dict[str,Any]]) -> Dict[str,Any]:
    maxT_slow = max(it["latents"].shape[2] for it in batch)
    maxT_fast = max(it["encodec_tokens"].shape[1] for it in batch)

    lat_list, enc_list = [], []
    cond_keys = list(batch[0]["conds"].keys())
    cond_lists: Dict[str,List[torch.Tensor]] = {k: [] for k in cond_keys}
    group_ids, subgroup_ids, metas = [], [], []

    for it in batch:
        lat_list.append(_pad_dim(it["latents"], maxT_slow, dim=2))
        enc_list.append(_pad_last(it["encodec_tokens"], maxT_fast))
        for k in cond_keys:
            cond_lists[k].append(_pad_last(it["conds"][k], maxT_slow))
        group_ids.append(it["instrument"]["group_id"])
        subgroup_ids.append(it["instrument"]["subgroup_id"])
        metas.append(it["meta"])

    return {
        "latents": torch.stack(lat_list, 0),
        "encodec_tokens": torch.stack(enc_list, 0),
        "conds": {k: torch.stack(v, 0) for k,v in cond_lists.items()},
        "instrument": {
            "group_id": torch.stack(group_ids, 0),
            "subgroup_id": torch.stack(subgroup_ids, 0),
        },
        "meta": metas,
    }
