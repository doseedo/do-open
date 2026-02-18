# dataloadvervox_fast.py - Fast dataloader using precomputed speaker embeddings
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


class PerformerAIVocalDatasetFast(Dataset):
    """
    Fast dataset that loads precomputed speaker embeddings instead of extracting on-the-fly.
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
                 # NEW: Voice reference parameters (precomputed)
                 voice_reference_dropout: float = 0.20,
                 use_precomputed_embeddings: bool = True):
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

        # NEW: Voice reference (precomputed embeddings)
        self.voice_reference_dropout = voice_reference_dropout
        self.use_precomputed_embeddings = use_precomputed_embeddings

        self.allowed_cond = ["piano_roll","amp","rbend","rframe"]
        self.drop_p = conditioning_dropout or {"piano_roll":0.20, "amp":0.10, "rbend":0.10, "rframe":0.05}

        # Extended vocab for vocals
        self.group2id = {g:i for i,g in enumerate(APPROVED_GROUPS_VOX)}
        all_subs = sorted({sg for g,subs in APPROVED_SUBGROUPS_VOX.items() for sg in subs})
        self.sub2id = {sg:i for i,sg in enumerate(all_subs)}

        self.rng = np.random.default_rng(seed)

        if seed is not None:
            random.seed(seed)

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
        """Load and process vocal conditioning data."""
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
                    syllable_boundaries = _pad_last(syllable_boundaries.view(-1), T_slow)

            if lyrics_data and lyrics_tensors and syllable_boundaries is not None:
                return {
                    "lyrics_data": lyrics_data,
                    "lyrics_tensors": lyrics_tensors,
                    "syllable_boundaries": syllable_boundaries,
                }

        except Exception as e:
            print(f"⚠ Failed to load vocal conditioning: {e}")

        return None

    def _load_voice_reference_fast(self, alternate_takes: List[Dict[str, Any]], exclude_path: str) -> Optional[torch.Tensor]:
        """
        Load precomputed speaker embedding from alternate take.

        Args:
            alternate_takes: List of dicts with keys: manifest_index, audio_path, speaker_embedding_path
            exclude_path: Current item's path to exclude

        Returns:
            speaker_embedding: [256] tensor or None
        """
        if not alternate_takes:
            return None

        # Select random alternate
        ref_alt = random.choice(alternate_takes)

        # Try to load precomputed embedding first
        if self.use_precomputed_embeddings:
            emb_path = ref_alt.get("speaker_embedding_path")
            if emb_path and Path(emb_path).exists():
                try:
                    speaker_emb = torch.load(emb_path, map_location="cpu")
                    return speaker_emb.float()
                except Exception as e:
                    print(f"⚠ Failed to load precomputed embedding from {emb_path}: {e}")

        # Fallback: extract on-the-fly (slower)
        ref_idx = ref_alt.get("manifest_index")
        if ref_idx is None or ref_idx >= len(self.manifest):
            return None

        ref_meta = self.manifest[ref_idx]
        ref_audio_path = ref_meta.get("audio_path")

        if not ref_audio_path or not Path(ref_audio_path).exists():
            return None

        try:
            # On-the-fly extraction (fallback - will be slow)
            import torchaudio
            from resemblyzer import VoiceEncoder

            audio, sr = torchaudio.load(ref_audio_path)

            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)

            # Initialize encoder on CPU (to avoid CUDA context issues in dataloader workers)
            if not hasattr(self, '_cpu_encoder'):
                self._cpu_encoder = VoiceEncoder(device='cpu')
                self._cpu_encoder.eval()

            audio_np = audio.squeeze().numpy()
            speaker_emb = self._cpu_encoder.embed_utterance(audio_np)

            return torch.from_numpy(speaker_emb).float()

        except Exception as e:
            print(f"⚠ Failed to extract speaker embedding on-the-fly: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.manifest[idx]
        group = (meta.get("group") or "undefined").lower()
        subgroup = (meta.get("sub_group") or "undefined").lower()
        group = group if group in self.group2id else "guitar"
        if self.collapse_sparse_subgroups_to_any and subgroup not in APPROVED_SUBGROUPS_VOX.get(group, []):
            subgroup = "undefined"
        if subgroup not in self.sub2id: subgroup = "undefined"

        # ---- load streams ----
        latents = _safe_pt_load(meta.get("latent_path"))
        encodec = _safe_pt_load(meta.get("encodec_path"))
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

        # ---- optional trim ----
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

        # ---- fixed window ----
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

        # ---- crop EnCodec ----
        if encodec is not None:
            start_abs = t0_abs + start
            end_abs   = t0_abs + end
            encodec   = _crop_fast_from_slow(encodec, start_abs, end_abs, r_local)

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

        # ---- conditioning dropout ----
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

        # ---- NEW: Load vocal conditioning ----
        vocal_conditioning = None
        vocal_cond_paths = meta.get("vocal_conditioning_paths") or {}
        if vocal_cond_paths and random.random() > self.vocal_conditioning_dropout:
            vocal_conditioning_full = self._load_vocal_conditioning(vocal_cond_paths, T_slow_all)
            if vocal_conditioning_full:
                syllable_boundaries_full = vocal_conditioning_full["syllable_boundaries"]
                if t0_abs != 0 or t1_abs != T_slow_all:
                    syllable_boundaries_full = syllable_boundaries_full[t0_abs:t1_abs]
                syllable_boundaries_windowed = syllable_boundaries_full[start:end]

                vocal_conditioning = {
                    "lyrics_data": vocal_conditioning_full["lyrics_data"],
                    "lyrics_tensors": vocal_conditioning_full["lyrics_tensors"],
                    "syllable_boundaries": syllable_boundaries_windowed,
                }

        # ---- NEW: Load voice reference (precomputed) ----
        reference_latent = None
        alternate_takes = meta.get("alternate_takes") or []
        if alternate_takes and random.random() > self.voice_reference_dropout:
            reference_latent = self._load_voice_reference_fast(alternate_takes, meta.get("audio_path", ""))

        return {
            "latents": latents,
            "encodec_tokens": encodec,
            "conds": {
                "piano_roll": piano_roll,
                "amp": amp,
                "rbend": rbend,
                "rframe": rframe,
                "rbend_mask": rb_mask.float(),
            },
            "instrument": {
                "group": group, "subgroup": subgroup,
                "group_id": torch.tensor(self.group2id.get(group,0), dtype=torch.long),
                "subgroup_id": torch.tensor(self.sub2id.get(subgroup,0), dtype=torch.long),
            },
            "vocal_conditioning": vocal_conditioning,
            "reference_latent": reference_latent,  # [256] precomputed or on-the-fly
            "meta": {
                "audio_path": meta.get("audio_path",""),
                "latent_path": meta.get("latent_path",""),
                "encodec_path": meta.get("encodec_path",""),
                "piano_roll_path": meta.get("piano_roll_path",""),
            }
        }


# Use same collate function
from dataloadvervox import collate_latent_cond_vocal
