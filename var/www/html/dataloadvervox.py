# dataloadvervox.py - Vocal-aware dataloader with lyric conditioning
import json, random
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Import base dataloader utilities
from dataloader import (
    DCAE_SR, DCAE_HOP, ENC_SR, ENC_HOP, SLOW_HZ, FAST_PER_SLOW,
    APPROVED_GROUPS, APPROVED_SUBGROUPS,
    _safe_np_load, _safe_pt_load, _pad_last, _pad_dim, _bool_smooth,
    _interval_is_oct_or_fifth, compute_polyphony_mask_from_pr,
    compute_rbend_mask, _crop_fast_from_slow,
    _first_active_t, _last_active_t
)

# Extended vocab for vocals
APPROVED_GROUPS_VOX = APPROVED_GROUPS + ["vocal"]
APPROVED_SUBGROUPS_VOX = {
    **APPROVED_SUBGROUPS,
    "vocal": ["lead_vocal", "backing_vocal", "choir", "undefined"]
}


class PerformerAIVocalDataset(Dataset):
    """
    Extended dataset that loads vocal conditioning data (lyrics, syllable boundaries)
    in addition to standard instrument conditioning.
    """
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
                 collapse_sparse_subgroups_to_any: bool = False,
                 # NEW: Vocal-specific parameters
                 vocal_conditioning_dropout: float = 0.05,
                 require_vocal_conditioning: bool = False,
                 # NEW: Voice reference parameters
                 voice_reference_dropout: float = 0.20,
                 voice_amp_threshold: float = 0.06):
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

        # NEW: Vocal conditioning
        self.vocal_conditioning_dropout = vocal_conditioning_dropout
        self.require_vocal_conditioning = require_vocal_conditioning

        # NEW: Voice reference
        self.voice_reference_dropout = voice_reference_dropout
        self.voice_amp_threshold = voice_amp_threshold

        self.allowed_cond = ["piano_roll","amp","rbend","rframe"]
        self.drop_p = conditioning_dropout or {"piano_roll":0.20, "amp":0.10, "rbend":0.10, "rframe":0.05}

        # Extended vocab for vocals
        self.group2id = {g:i for i,g in enumerate(APPROVED_GROUPS_VOX)}
        all_subs = sorted({sg for g,subs in APPROVED_SUBGROUPS_VOX.items() for sg in subs})
        self.sub2id = {sg:i for i,sg in enumerate(all_subs)}

        self.rng = np.random.default_rng(seed)

        if seed is not None:
            random.seed(seed)

        # Speaker embeddings are now loaded from preprocessed files
        # No need to initialize Resemblyzer speaker encoder

        if self.require_all_core:
            filtered = []
            for it in self.manifest:
                lat_ok = _safe_pt_load(it.get("latent_path"))   is not None
                enc_ok = _safe_pt_load(it.get("encodec_path"))  is not None
                pr_meta = it.get("piano_roll_path")
                pr_ok = True if pr_meta is None else (_safe_np_load(pr_meta) is not None)

                # NEW: Check vocal conditioning if required
                vocal_ok = True
                if self.require_vocal_conditioning:
                    vcp = it.get("vocal_conditioning_paths") or {}
                    if vcp:
                        lyrics_data_ok = Path(vcp.get("lyrics_data", "")).exists() if vcp.get("lyrics_data") else False
                        lyrics_tensors_ok = Path(vcp.get("lyrics_tensors", "")).exists() if vcp.get("lyrics_tensors") else False
                        syllable_ok = Path(vcp.get("syllable_boundaries", "")).exists() if vcp.get("syllable_boundaries") else False
                        vocal_ok = lyrics_data_ok and lyrics_tensors_ok and syllable_ok
                    else:
                        vocal_ok = False

                if lat_ok and enc_ok and pr_ok and vocal_ok:
                    filtered.append(it)
            self.manifest = filtered

    def __len__(self): return len(self.manifest)

    def _load_vocal_conditioning(self, vocal_cond_paths: Dict[str, str], T_slow: int) -> Optional[Dict[str, Any]]:
        """
        Load and process vocal conditioning data.

        Returns:
            Dict with keys:
                - lyrics_data: parsed JSON with syllable timings
                - lyrics_tensors: dict of {lyrics_indices, lyrics_embeddings, phoneme_embeddings}
                - syllable_boundaries: [T_slow] tensor
        """
        if not vocal_cond_paths:
            return None

        try:
            # Load lyrics JSON
            lyrics_data = None
            lyrics_data_path = vocal_cond_paths.get("lyrics_data")
            if lyrics_data_path and Path(lyrics_data_path).exists():
                with open(lyrics_data_path, 'r') as f:
                    lyrics_data = json.load(f)

            # Load lyrics tensors
            lyrics_tensors = None
            lyrics_tensors_path = vocal_cond_paths.get("lyrics_tensors")
            if lyrics_tensors_path and Path(lyrics_tensors_path).exists():
                lyrics_tensors = torch.load(lyrics_tensors_path, map_location="cpu")

            # Load syllable boundaries
            syllable_boundaries = None
            syllable_path = vocal_cond_paths.get("syllable_boundaries")
            if syllable_path:
                syll_np = _safe_np_load(syllable_path)
                if syll_np is not None:
                    syllable_boundaries = torch.from_numpy(syll_np).float()
                    # Pad/crop to T_slow
                    syllable_boundaries = _pad_last(syllable_boundaries.view(-1), T_slow)

            # Only return if we have all three components
            if lyrics_data and lyrics_tensors and syllable_boundaries is not None:
                return {
                    "lyrics_data": lyrics_data,
                    "lyrics_tensors": lyrics_tensors,
                    "syllable_boundaries": syllable_boundaries,
                }

        except Exception as e:
            print(f"⚠ Failed to load vocal conditioning: {e}")

        return None

    def _load_voice_reference(self, current_meta: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Load preprocessed speaker embedding from current item (same take).

        Args:
            current_meta: Current item's metadata dict

        Returns:
            speaker_embedding: [256] tensor or None
        """
        # Load preprocessed speaker embedding from current item
        spk_emb_path = current_meta.get("speaker_emb_path")

        if not spk_emb_path or spk_emb_path == "null":
            return None

        try:
            # Load preprocessed speaker embedding from disk
            speaker_emb = torch.load(spk_emb_path, map_location='cpu')  # [256]

            if not isinstance(speaker_emb, torch.Tensor):
                speaker_emb = torch.tensor(speaker_emb, dtype=torch.float32)

            return speaker_emb.float()

        except Exception as e:
            # Silently fail - this is expected for some entries
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.manifest[idx]
        # Everything is vocals - no group/subgroup needed
        group = "vocal"
        subgroup = "lead_vocal"

        # ---- load streams ----
        latents = _safe_pt_load(meta.get("latent_path"))   # [8,16,T_slow_all]
        # Handle null encodec path - treat as None (dropout)
        encodec_path = meta.get("encodec_path")
        encodec = _safe_pt_load(encodec_path) if encodec_path is not None else None  # [C_fast,T_fast] or variants
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

        # Handle null piano roll path - treat as None (dropout)
        pr_path = meta.get("piano_roll_path")
        pr_np = _safe_np_load(pr_path) if pr_path is not None else None
        piano_roll = torch.from_numpy(pr_np).float() if pr_np is not None else None

        assert latents is not None and latents.dim()==3, f"bad latents at {meta.get('latent_path')}"
        T_slow_all = latents.shape[2]
        T_fast_all = encodec.shape[-1] if encodec is not None else None

        # measure per-item ratio to be *exactly* aligned with this file
        r_local = (T_fast_all / T_slow_all) if (encodec is not None and T_slow_all > 0) else FAST_PER_SLOW

        # 1-D conds - handle null paths as None (will be treated as dropout)
        cond_paths = meta.get("conditioning_paths") or {}
        def get1d(key):
            path = cond_paths.get(key)
            # Treat null/None paths as missing (dropout)
            if path is None:
                return None
            arr = _safe_np_load(path)
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
            start = min(self._static_start, max_start)
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

        # ---- rbend gating for vocals ----
        # For vocals, rbend is gated by rframe (voiced regions only)
        rb_mask = (rframe > 0.5).float()
        rbend = rbend * rb_mask

        # sanity
        assert latents.shape[-1] == T_slow_final
        assert amp.shape[-1] == T_slow_final
        assert rbend.shape[-1] == T_slow_final
        assert rframe.shape[-1] == T_slow_final
        assert piano_roll.shape[-1] == T_slow_final

        # ---- NEW: Load vocal conditioning ----
        vocal_conditioning = None
        vocal_cond_paths = meta.get("vocal_conditioning_paths") or {}
        if vocal_cond_paths and random.random() > self.vocal_conditioning_dropout:
            # Load before windowing, then crop syllable boundaries to match window
            vocal_conditioning_full = self._load_vocal_conditioning(vocal_cond_paths, T_slow_all)
            if vocal_conditioning_full:
                # Crop syllable boundaries to match the windowed segment
                syllable_boundaries_full = vocal_conditioning_full["syllable_boundaries"]
                # Apply same trim and window as other conditioning
                if t0_abs != 0 or t1_abs != T_slow_all:
                    syllable_boundaries_full = syllable_boundaries_full[t0_abs:t1_abs]
                syllable_boundaries_windowed = syllable_boundaries_full[start:end]

                vocal_conditioning = {
                    "lyrics_data": vocal_conditioning_full["lyrics_data"],
                    "lyrics_tensors": vocal_conditioning_full["lyrics_tensors"],
                    "syllable_boundaries": syllable_boundaries_windowed,  # [T_slow_final]
                }

        # ---- NEW: Load voice reference from same take ----
        reference_latent = None
        if random.random() > self.voice_reference_dropout:
            reference_latent = self._load_voice_reference(meta)

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
            # No instrument info needed - everything is vocals
            "group_id": torch.tensor(0, dtype=torch.long),  # Dummy for compatibility
            "subgroup_id": torch.tensor(0, dtype=torch.long),  # Dummy for compatibility
            "vocal_conditioning": vocal_conditioning,  # NEW: None or dict
            "reference_latent": reference_latent,  # NEW: [256] speaker embedding or None
            "meta": {
                "audio_path": meta.get("audio_path",""),
                "latent_path": meta.get("latent_path",""),
                "encodec_path": meta.get("encodec_path",""),
                "piano_roll_path": meta.get("piano_roll_path",""),
            }
        }


def collate_latent_cond_vocal(batch: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    Collate function that handles vocal conditioning data.
    """
    maxT_slow = max(it["latents"].shape[2] for it in batch)
    maxT_fast = max(it["encodec_tokens"].shape[1] for it in batch)

    lat_list, enc_list = [], []
    cond_keys = list(batch[0]["conds"].keys())
    cond_lists: Dict[str,List[torch.Tensor]] = {k: [] for k in cond_keys}
    group_ids, subgroup_ids, metas = [], [], []
    vocal_cond_list = []
    reference_latent_list = []

    for it in batch:
        lat_list.append(_pad_dim(it["latents"], maxT_slow, dim=2))
        enc_list.append(_pad_last(it["encodec_tokens"], maxT_fast))
        for k in cond_keys:
            cond_lists[k].append(_pad_last(it["conds"][k], maxT_slow))
        group_ids.append(it["group_id"])
        subgroup_ids.append(it["subgroup_id"])
        metas.append(it["meta"])

        # NEW: Collect vocal conditioning
        vc = it["vocal_conditioning"]
        if vc is not None:
            # Pad syllable boundaries to maxT_slow
            vc_padded = {
                "lyrics_data": vc["lyrics_data"],
                "lyrics_tensors": vc["lyrics_tensors"],
                "syllable_boundaries": _pad_last(vc["syllable_boundaries"], maxT_slow),
            }
            vocal_cond_list.append(vc_padded)
        else:
            vocal_cond_list.append(None)

        # NEW: Collect voice reference
        ref_lat = it.get("reference_latent")
        reference_latent_list.append(ref_lat)

    # Stack speaker embeddings (handle None values)
    reference_latent_batch = None
    if any(ref is not None for ref in reference_latent_list):
        # Create batch tensor, fill with zeros where None
        reference_latent_batch = torch.stack([
            ref if ref is not None else torch.zeros(256)
            for ref in reference_latent_list
        ], dim=0)  # [B, 256]

    return {
        "latents": torch.stack(lat_list, 0),
        "encodec_tokens": torch.stack(enc_list, 0),
        "conds": {k: torch.stack(v, 0) for k,v in cond_lists.items()},
        "group_id": torch.stack(group_ids, 0),
        "subgroup_id": torch.stack(subgroup_ids, 0),
        "vocal_conditioning": vocal_cond_list,  # NEW: List of dicts or Nones
        "reference_latent": reference_latent_batch,  # NEW: [B, 256] speaker embeddings or None
        "meta": metas,
    }
