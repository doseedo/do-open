#!/usr/bin/env python3
"""
DO1 Training Script — ACE-Step 1.5 VAE Format

Fine-tunes ACE-Step v1.5 base DiT for DO1's 7-task multi-task training.

VAE format: [B, 64, T] at 25Hz, 48kHz stereo (AutoencoderOobleck)
DiT input:  [x_noisy(64) | x_cond(64) | mask(1)] = 129 channels → Conv1d patchify

Usage:
    # Single GPU debug
    python train.py --debug --latents_dir /data/latents_v15

    # Single GPU full
    python train.py --latents_dir /data/latents_v15 --fx_pairs_dir /data/fx_pairs

    # Multi-GPU
    torchrun --nproc_per_node=4 train.py \\
        --latents_dir /data/latents_v15 \\
        --fx_pairs_dir /data/fx_pairs \\
        --mix_pairs_dir /data/mix_pairs \\
        --acestep_base /checkpoints/acestep-v15-base

    # Resume
    python train.py --resume output/latest/checkpoints/last.ckpt --latents_dir /data/latents_v15
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, str(Path(__file__).parent.parent))

from do1.training import DO1Pipeline


# =============================================================================
# ACE-Step 1.5 VAE constants
# =============================================================================
VAE_DIM = 64          # latent channels
VAE_HZ = 25.0         # temporal resolution
VAE_SR = 48000        # audio sample rate
PATCH_STRIDE = 2      # ACE-Step 1.5 patchifies at stride 2 → 12.5Hz tokens


# =============================================================================
# Default configs for 1.5 VAE format
# =============================================================================

def get_model_config_2b():
    """~2B param config matching ACE-Step 1.5 DiT backbone."""
    return {
        # 1.5 VAE latent format
        "latent_dim": VAE_DIM,                    # 64
        "in_channels_noisy": VAE_DIM,             # 64
        "in_channels_cond": VAE_DIM,              # 64
        "in_channels_mask": 1,                     # 1
        "in_channels_ref": VAE_DIM,               # 64
        "out_channels": VAE_DIM,                   # 64
        # Patchify (1D, stride 2 → 12.5Hz tokens)
        "patch_stride": PATCH_STRIDE,
        # Transformer (match ACE-Step 1.5 DiT)
        "model_dim": 2048,
        "num_attention_heads": 32,
        "attention_head_dim": 64,
        "num_layers": 24,                          # ACE-Step 1.5 has 24 blocks
        "mlp_ratio": 4.0,
        # Position encoding
        "max_position": 32768,
        "rope_theta": 1000000.0,
        # Normalization
        "qk_norm": True,
    }


def get_model_config_small():
    """Small config for debugging."""
    return {
        "latent_dim": VAE_DIM,
        "in_channels_noisy": VAE_DIM,
        "in_channels_cond": VAE_DIM,
        "in_channels_mask": 1,
        "in_channels_ref": VAE_DIM,
        "out_channels": VAE_DIM,
        "patch_stride": PATCH_STRIDE,
        "model_dim": 512,
        "num_attention_heads": 8,
        "attention_head_dim": 64,
        "num_layers": 4,
        "mlp_ratio": 4.0,
        "max_position": 8192,
        "rope_theta": 1000000.0,
        "qk_norm": True,
    }


def get_training_config():
    """Default training config."""
    return {
        # Optimizer
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 5000,
        "max_steps": 500000,
        # Batch
        "batch_size": 4,
        "gradient_accumulation": 8,
        "grad_clip": 1.0,
        # Precision
        "precision": "bf16-mixed",
        # Data
        "num_workers": 8,
        "max_time_frames": 2500,     # ~100s at 25Hz
        "samples_per_epoch": 100000,
        # CFG
        "cfg_dropout": 0.3,
        # Flow matching
        "timestep_distribution": "logit_normal",
        "logit_mean": 0.0,
        "logit_std": 1.0,
        # Logging
        "log_every": 100,
        # Checkpointing
        "checkpoint_every": 5000,
        "save_top_k": 3,
        # Freeze schedule (step thresholds)
        "unfreeze_last_n_at_start": 4,     # unfreeze last 4 blocks at step 0
        "unfreeze_last_n_at_step1": 8,     # unfreeze last 8 blocks at step
        "unfreeze_step1": 2000,
        "unfreeze_all_step": 6000,         # unfreeze all 24 blocks
        # LR groups
        "head_lr_multiplier": 2.0,         # proj_in, final_layer, ref_encoder at 2x LR
    }


# =============================================================================
# Progressive Unfreeze Callback
# =============================================================================

class ProgressiveUnfreezeCallback(Callback):
    """
    Progressive unfreezing schedule matching CN2/old DO1 pattern.

    Step 0:     Freeze everything. Unfreeze: proj_in, final_layer,
                timestep_embedder, t_block, ref_encoder, last N blocks.
    Step T1:    Unfreeze last M blocks.
    Step T2:    Unfreeze all blocks.

    Lyrics/speaker/genre embedders stay frozen (unused).
    """

    def __init__(
        self,
        unfreeze_last_n_at_start: int = 4,
        unfreeze_last_n_at_step1: int = 8,
        unfreeze_step1: int = 2000,
        unfreeze_all_step: int = 6000,
    ):
        self.n_start = unfreeze_last_n_at_start
        self.n_step1 = unfreeze_last_n_at_step1
        self.step1 = unfreeze_step1
        self.all_step = unfreeze_all_step
        self._phase = -1

    def _freeze_all(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def _unfreeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = True

    def _unfreeze_last_n_blocks(self, model, n):
        blocks = model.transformer_blocks
        total = len(blocks)
        for i in range(max(0, total - n), total):
            self._unfreeze_module(blocks[i])

    def _apply_phase(self, model, phase):
        if phase == self._phase:
            return
        self._phase = phase

        if phase == 0:
            # Freeze everything first
            self._freeze_all(model)
            # Unfreeze heads
            self._unfreeze_module(model.patch_embed)       # expanded proj_in
            self._unfreeze_module(model.final_layer)
            self._unfreeze_module(model.timestep_embedder)
            self._unfreeze_module(model.t_block)
            self._unfreeze_module(model.reference_encoder)
            # Unfreeze last N blocks (including their cross-attention)
            self._unfreeze_last_n_blocks(model, self.n_start)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"[Unfreeze Phase 0] Last {self.n_start} blocks + heads: "
                  f"{trainable:,}/{total:,} params trainable")

        elif phase == 1:
            self._unfreeze_last_n_blocks(model, self.n_step1)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Unfreeze Phase 1] Last {self.n_step1} blocks: {trainable:,} params trainable")

        elif phase == 2:
            # Unfreeze everything except unused embedders
            for name, param in model.named_parameters():
                # Keep lyrics/speaker/genre frozen if they exist
                if any(skip in name for skip in ["lyric", "speaker", "genre", "text_encoder"]):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Unfreeze Phase 2] All blocks: {trainable:,} params trainable")

    def on_train_start(self, trainer, pl_module):
        self._apply_phase(pl_module.model, 0)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = trainer.global_step
        if step >= self.all_step:
            self._apply_phase(pl_module.model, 2)
        elif step >= self.step1:
            self._apply_phase(pl_module.model, 1)


# =============================================================================
# ACE-Step Base Model Loading
# =============================================================================

def load_acestep_base(model, acestep_path: str):
    """
    Load ACE-Step v1.5 base weights into DO1 model.

    Handles:
    - Expanding proj_in from original channels to 129 (64+64+1)
    - Zero-initializing new channels so model starts as original ACE-Step
    - Cloning proj_in weights to ref_encoder
    - Remapping cross-attention keys from lyrics/speaker to x_ref
    """
    print(f"Loading ACE-Step base from: {acestep_path}")

    # Load ACE-Step state dict
    acestep_state = torch.load(
        acestep_path,
        map_location="cpu",
        weights_only=True,
    )
    if "state_dict" in acestep_state:
        acestep_state = acestep_state["state_dict"]
    elif "model" in acestep_state:
        acestep_state = acestep_state["model"]

    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    expanded = 0

    for key, param in acestep_state.items():
        # Skip unused embedders
        if any(skip in key for skip in ["lyric", "speaker", "genre", "text_encoder", "lm_head"]):
            skipped += 1
            continue

        # Handle proj_in expansion (original channels → 129)
        if "patch_embed" in key and "early_conv" in key and param.shape != model_state.get(key, param).shape:
            if key in model_state:
                target_shape = model_state[key].shape
                if len(param.shape) >= 2 and len(target_shape) >= 2:
                    # Conv weight: [out, in, ...]
                    # Original in_channels → new in_channels (129)
                    if param.shape[1] != target_shape[1]:
                        # Zero-initialize expanded tensor
                        new_param = torch.zeros_like(model_state[key])
                        # Copy original weights into first channels
                        in_orig = param.shape[1]
                        new_param[:, :in_orig] = param
                        model_state[key] = new_param
                        expanded += 1
                        continue

        # Direct copy for matching shapes
        if key in model_state and param.shape == model_state[key].shape:
            model_state[key] = param
            loaded += 1

    model.load_state_dict(model_state, strict=False)

    # Clone proj_in weights to ref_encoder (first 64 channels)
    # So ref_encoder starts in same embedding space
    if hasattr(model, "reference_encoder") and hasattr(model, "patch_embed"):
        try:
            src_weight = model.patch_embed.early_conv_layers[0].weight
            ref_conv = model.reference_encoder.patch_embed[0]
            if hasattr(ref_conv, "weight"):
                # Copy first 64 channels from expanded proj_in
                in_ref = ref_conv.weight.shape[1]  # 64
                ref_conv.weight.data.copy_(src_weight.data[:, :in_ref])
                if ref_conv.bias is not None and model.patch_embed.early_conv_layers[0].bias is not None:
                    ref_conv.bias.data.copy_(model.patch_embed.early_conv_layers[0].bias.data)
                print(f"Cloned proj_in weights to ref_encoder")
        except Exception as e:
            print(f"Warning: could not clone ref_encoder weights: {e}")

    print(f"ACE-Step loading: {loaded} loaded, {expanded} expanded, {skipped} skipped")
    return model


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DO1 — Universal Latent Audio Operator (ACE-Step 1.5 VAE format)"
    )

    # Config files (optional, overrides defaults)
    parser.add_argument("--model_config", type=str, default=None,
                        help="Model config YAML (default: built-in 2B config)")
    parser.add_argument("--training_config", type=str, default=None,
                        help="Training config YAML (default: built-in config)")

    # ACE-Step base model
    parser.add_argument("--acestep_base", type=str, default=None,
                        help="Path to ACE-Step v1.5 base checkpoint for fine-tuning")

    # Data paths
    parser.add_argument("--latents_dir", type=str, required=True,
                        help="Directory with VAE-encoded session latents (.pt files, [64, T] format)")
    parser.add_argument("--fx_pairs_dir", type=str, default=None,
                        help="Directory with FX dry/wet latent pairs")
    parser.add_argument("--mix_pairs_dir", type=str, default=None,
                        help="Directory with precomputed audio-domain mix latents")
    parser.add_argument("--demucs_pairs_dir", type=str, default=None,
                        help="Directory with Demucs-separated mix/stem latent pairs")
    parser.add_argument("--vst_synths_dir", type=str, default=None,
                        help="Directory with VST synth latent pairs")
    parser.add_argument("--labels_path", type=str, default=None,
                        help="Path to instrument labels JSON")
    parser.add_argument("--val_latents_dir", type=str, default=None,
                        help="Directory with validation latents")

    # Output
    parser.add_argument("--output_dir", "-o", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (default: timestamp)")

    # Training overrides
    parser.add_argument("--batch_size", "-b", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", "--lr", type=float, default=None)
    parser.add_argument("--devices", type=int, default=None,
                        help="Number of GPUs (default: all available)")
    parser.add_argument("--gradient_accumulation", type=int, default=None)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: small model, 100 steps, batch_size=2")

    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # ---- Load or build configs ----
    if args.model_config:
        model_config = load_yaml(args.model_config)
    else:
        model_config = get_model_config_2b()

    if args.training_config:
        training_config = load_yaml(args.training_config)
    else:
        training_config = get_training_config()

    # Debug overrides
    if args.debug:
        model_config = get_model_config_small()
        training_config["max_steps"] = 100
        training_config["batch_size"] = 2
        training_config["gradient_accumulation"] = 1
        training_config["num_workers"] = 0
        training_config["checkpoint_every"] = 50
        training_config["log_every"] = 10
        print("=== DEBUG MODE ===")

    # CLI overrides
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    if args.max_steps is not None:
        training_config["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    if args.gradient_accumulation is not None:
        training_config["gradient_accumulation"] = args.gradient_accumulation

    # Data config
    data_config = {
        "latents_dir": args.latents_dir,
        "fx_pairs_dir": args.fx_pairs_dir,
        "mix_pairs_dir": args.mix_pairs_dir,
        "demucs_pairs_dir": args.demucs_pairs_dir,
        "vst_synths_dir": args.vst_synths_dir,
        "labels_path": args.labels_path,
        "val_latents_dir": args.val_latents_dir,
        "max_time_frames": training_config.get("max_time_frames", 2500),
        "samples_per_epoch": training_config.get("samples_per_epoch", 100000),
    }

    # ---- Experiment directory ----
    exp_name = args.exp_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configs for reproducibility
    with open(output_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)
    with open(output_dir / "training_config.yaml", "w") as f:
        yaml.dump(training_config, f)
    with open(output_dir / "data_config.yaml", "w") as f:
        yaml.dump(data_config, f)

    print(f"Experiment: {exp_name}")
    print(f"Output: {output_dir}")
    print(f"Latent format: [{VAE_DIM}, T] at {VAE_HZ}Hz")
    print(f"Model dim: {model_config['model_dim']}, layers: {model_config['num_layers']}")

    # ---- Create pipeline ----
    pipeline = DO1Pipeline(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
    )

    # ---- Load ACE-Step base weights ----
    if args.acestep_base:
        pipeline.model = load_acestep_base(pipeline.model, args.acestep_base)
    else:
        print("WARNING: No --acestep_base provided. Training from scratch.")
        print("         This is fine for debugging but NOT for real training.")

    # Print param counts
    total = sum(p.numel() for p in pipeline.model.parameters())
    trainable = sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    print(f"Model size: {total * 2 / 1e9:.2f} GB (bf16)")

    # ---- Callbacks ----
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="do1-{step:07d}",
            every_n_train_steps=training_config.get("checkpoint_every", 5000),
            save_top_k=training_config.get("save_top_k", 3),
            monitor="train/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        ProgressiveUnfreezeCallback(
            unfreeze_last_n_at_start=training_config.get("unfreeze_last_n_at_start", 4),
            unfreeze_last_n_at_step1=training_config.get("unfreeze_last_n_at_step1", 8),
            unfreeze_step1=training_config.get("unfreeze_step1", 2000),
            unfreeze_all_step=training_config.get("unfreeze_all_step", 6000),
        ),
    ]

    # ---- Logger ----
    logger = TensorBoardLogger(save_dir=output_dir, name="logs", version="")

    # ---- Devices & strategy ----
    devices = args.devices or torch.cuda.device_count() or 1

    if devices > 1:
        from pytorch_lightning.strategies import FSDPStrategy
        from torch.distributed.fsdp import MixedPrecision

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Wrap at transformer block level
        # TODO: import correct block class once model is updated
        try:
            from do1.models.layers import DO1TransformerBlock
            wrap_policy = {DO1TransformerBlock}
        except ImportError:
            wrap_policy = None

        strategy = FSDPStrategy(
            auto_wrap_policy=wrap_policy,
            activation_checkpointing_policy=wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy="SHARD_GRAD_OP",
        )
    else:
        strategy = "auto"

    # ---- Trainer ----
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy=strategy,
        precision=training_config.get("precision", "bf16-mixed"),
        max_steps=training_config["max_steps"],
        accumulate_grad_batches=training_config.get("gradient_accumulation", 8),
        gradient_clip_val=training_config.get("grad_clip", 1.0),
        log_every_n_steps=training_config.get("log_every", 100),
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ---- Train ----
    print(f"\nStarting training: {training_config['max_steps']} steps")
    print(f"Effective batch size: {training_config['batch_size']} × "
          f"{training_config.get('gradient_accumulation', 8)} × {devices} = "
          f"{training_config['batch_size'] * training_config.get('gradient_accumulation', 8) * devices}")

    trainer.fit(pipeline, ckpt_path=args.resume)

    print(f"\nTraining complete!")
    print(f"Checkpoints: {output_dir / 'checkpoints'}")
    print(f"Logs: {output_dir / 'logs'}")


if __name__ == "__main__":
    main()