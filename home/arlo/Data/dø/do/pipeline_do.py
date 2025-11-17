# ~/Data/dø/do/pipeline_do.py
# Apache 2.0

import os, json
import torch
from loguru import logger

from do.music_dcae.music_dcae_pipeline import MusicDCAE
# from do.models.ace_step_transformer import PerformerTransformer as ACEStepTransformer2DModel
from do.models.ace_step_transformer import ACEStepTransformer2DModel

def _ensure_dir(p):
    p = str(p)
    os.makedirs(p, exist_ok=True)
    return p


def _pick_root_with_assets(checkpoint_dir: str):
    """
    Accept either:
      - a root dir that already contains the 4 subdirs, or
      - a HF 'snapshots/<hash>' dir that contains them, or
      - the top-level HF cache with a single 'snapshots/<hash>' -> we pick the first.
    Returns the directory that contains:
      ace_step_transformer/, music_dcae_f8c8/, music_vocoder/, umt5-base/  (umt5-base not used)
    """
    must = {"ace_step_transformer", "music_dcae_f8c8", "music_vocoder"}

    def has_all(d):
        if not d or not os.path.isdir(d):
            return False
        present = {n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))}
        return must.issubset(present)

    # 1) direct
    if has_all(checkpoint_dir):
        return checkpoint_dir

    # 2) snapshots/<hash>
    sn = os.path.join(checkpoint_dir, "snapshots")
    if os.path.isdir(sn):
        for h in sorted(os.listdir(sn)):
            cand = os.path.join(sn, h)
            if has_all(cand):
                return cand

    # 3) one level above (user passed ".../models--..../")
    try:
        for root, dirs, _ in os.walk(checkpoint_dir):
            if has_all(root):
                return root
    except Exception:
        pass

    raise FileNotFoundError(
        f"Did not find required subdirs under '{checkpoint_dir}'. "
        "Expected {ace_step_transformer, music_dcae_f8c8, music_vocoder}."
    )


class DoTrainComponents:
    """
    Minimal, training-first loader (dø custom model):
      - MusicDCAE (+vocoder) for encode/decode of latents (frozen).
      - NEW, randomly-initialized ACE transformer, built from config.json (no weights).
    """

    def __init__(self, checkpoint_dir: str, device_id: int = 0, dtype: str = "float32"):
        if checkpoint_dir is None:
            raise ValueError("--checkpoint_dir is required (for DCAE + transformer config).")
        self.root = _pick_root_with_assets(checkpoint_dir)

        self.device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        logger.info(f"[DoTrainComponents] root = {self.root}")
        logger.info(f"[DoTrainComponents] device = {self.device}, dtype = {self.dtype}")

        self.music_dcae = None
        self.transformer = None

    # ---------- DCAE ----------
    def load_dcae(self):
        if self.music_dcae is not None:
            return self.music_dcae

        dcae_dir = os.path.join(self.root, "music_dcae_f8c8")
        voc_dir  = os.path.join(self.root, "music_vocoder")
        if not (os.path.isdir(dcae_dir) and os.path.isdir(voc_dir)):
            raise FileNotFoundError("music_dcae_f8c8/ and music_vocoder/ not found next to ace_step_transformer/")

        m = MusicDCAE(dcae_checkpoint_path=dcae_dir, vocoder_checkpoint_path=voc_dir)
        m = m.to(self.device).eval()
        m.requires_grad_(False)
        self.music_dcae = m
        logger.info("[DoTrainComponents] MusicDCAE loaded (frozen).")
        return self.music_dcae
   
    def _get_transformer_class(self, from_scratch=False):
        if from_scratch:
            from do.models.ace_step_transformer_CUSTOM import PerformerTransformer as CustomTransformer
            return CustomTransformer
        else:
            from do.models.ace_step_transformer import ACEStepTransformer2DModel as OfficialTransformer
            return OfficialTransformer

    # do/pipeline_do.py
    def build_transformer_pretrained(self):
        if self.transformer is not None:
            return self.transformer
        
        OfficialTransformer = self._get_transformer_class(from_scratch=False)
        model_path = os.path.join(self.root, "ace_step_transformer")
        model = OfficialTransformer.from_pretrained(model_path)

        model = model.to(self.device).to(self.dtype).train()
        self.transformer = model
        logger.info("[DoTrainComponents] Transformer loaded WITH PRE-TRAINED weights.")
        return self.transformer   # <-- add this


    def build_transformer_random(self):
        if self.transformer is not None:
            return self.transformer

        CustomTransformer = self._get_transformer_class(from_scratch=True)
        cfg = {
            "num_layers": 12, "num_attention_heads": 12, "attention_head_dim": 64,
            "mlp_ratio": 2.0, "patch_size": [16, 16], "text_embedding_dim": 768,
            "in_channels": 8, "out_channels": 8, "max_width": 4096,
        }
        model = CustomTransformer.from_config(cfg)
        logger.info("[DoTrainComponents] Built custom transformer via from_config().")

        model = model.to(self.device).to(self.dtype).train()
        self.transformer = model
        logger.info("[DoTrainComponents] Transformer initialized RANDOMLY (no weights loaded).")
        return self.transformer