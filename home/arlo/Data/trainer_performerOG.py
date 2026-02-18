# ~/Data/trainer_performer.py
# Apache 2.0

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse, os, json, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS
from typing import Optional, Union
import os
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor


import inspect
import torch.nn.functional as F
import torch.nn.functional as F

from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers import DDPMScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor


from acestep.pipeline_ace_step import ACEStepTrainComponents
from dataloader import PerformerAIDataset, collate_latent_cond
from conditioning_encoder import PerformanceConditionEncoder
import torchaudio



# Match your grids
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000,  320
SLOW_HZ = DCAE_SR / DCAE_HOP
FAST_PER_SLOW = (ENC_SR/ENC_HOP) / SLOW_HZ  # ~6.96

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

try:
    from loguru import logger
except Exception:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s"
    )
    logger = logging.getLogger("ace_step")

class Pipeline(LightningModule):
    def __init__(
        self,
        checkpoint_dir: str,
        manifest_json: str,
        learning_rate: float = 1e-4,
        num_workers: int = 8,
        T: int = 1000,
        weight_decay: float = 1e-2,
        every_plot_step: int = 2000,
        shift: float = 3.0,
        cond_cfg_drop_prob: float = 0.15,
        max_steps: int = 200000,
        warmup_steps: int = 10,
        window_slow: int = 256,
        # NEW:
        override_group: Optional[Union[str, int]] = None,
        override_subgroup: Optional[Union[str, int]] = None,
        preview_steps: int = 50,
        batch_size: int = 1,
        preview_index: int = 0,
        train_from_scratch: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters() 

        # --- 1) Scheduler (ACE FlowMatch Euler)
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=T, shift=shift)

        # self.scheduler = DDPMScheduler(num_train_timesteps=T, prediction_type="epsilon")

        self._wrote_gt = False
        self._did_preview_once = False



#TRAIN FROM SCRATCH
        # --- 2) Build components (NO pretrained weights for transformer)
        # comps = ACEStepTrainComponents(checkpoint_dir=checkpoint_dir, dtype="float32")
        # self.dcae = comps.load_dcae()                            # frozen
        # self.transformers = comps.build_transformer_random()     # RANDOM INIT
        # self.transformers.enable_gradient_checkpointing()
        # self.transformers.train()
        # self._preview_batch = None



        # --- REPLACE THE MODEL LOADING BLOCK ---
        comps = ACEStepTrainComponents(checkpoint_dir=self.hparams.checkpoint_dir, dtype="float32")
        self.dcae = comps.load_dcae()  # frozen

        if self.hparams.train_from_scratch:
            logger.info("Initializing a new Transformer model to train FROM SCRATCH.")
            self.transformers = comps.build_transformer_random() # Your stable, from-scratch version
        else:
            logger.info("Loading official ACE-Step weights to FINE-TUNE.")
            self.transformers = comps.build_transformer_pretrained() # The official pre-trained model

        self.transformers.enable_gradient_checkpointing()
        self.transformers.train()
        self._preview_batch = None

        # Determine D for the conditioning encoder
        d_text = getattr(self.transformers.config, "text_embedding_dim", None)
        if d_text is None:
            # config may be dict-like
            d_text = self.transformers.config.get("text_embedding_dim", 768)

        # with torch.no_grad():
        #     if hasattr(self.transformers, "cond_gain"):
        #         self.transformers.cond_gain.data.fill_(1.0) 
        
        # x = x + cond_patched


        # --- 3) Conditioning encoder (your controls → tokens)
        self.ctrl_enc = PerformanceConditionEncoder(
            d_text=d_text,
            enc_channels=8,
            fast_per_slow=FAST_PER_SLOW,
            group_vocab=6,
            subgroup_vocab=self._infer_subgroup_vocab(manifest_json),
        )
        self.ctrl_enc.requires_grad_(False)
        self.ctrl_enc.eval()

        # Build lookup tables from your approved lists
        
        if isinstance(APPROVED_GROUPS, dict):
            group_names = list(APPROVED_GROUPS.keys())
        else:
            group_names = list(APPROVED_GROUPS)
        sub_names = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})

        self.group2id    = {n: i for i, n in enumerate(group_names)}
        self.subgroup2id = {n: i for i, n in enumerate(sub_names)}

        def _to_id(s, table):
            if s is None: return None
            if isinstance(s, int): return s
            if isinstance(s, str) and s.isdigit(): return int(s)
            return table.get(s, None)

        # USE THE RAW CTOR ARGS (not self.hparams.*) so it also works even if you don’t save hparams
        self.override_gid  = _to_id(override_group, self.group2id)
        self.override_sgid = _to_id(override_subgroup, self.subgroup2id)

        self.manifest_json = manifest_json
        self.cond_cfg_drop_prob = cond_cfg_drop_prob
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.every_plot_step = every_plot_step

        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=T, shift=shift)


        if hasattr(self.scheduler, "config"):
            self.scheduler.config.prediction_type = "v_prediction"
        # self.scheduler = DDPMScheduler(num_train_timesteps=T, prediction_type="epsilon")
      
        # if hasattr(self.scheduler, "config") and hasattr(self.scheduler.config, "prediction_type"):
        #     self.scheduler.config.prediction_type = "epsilon"

    # ----- Data -----
    def _infer_subgroup_vocab(self, manifest_json):
        try:
            from pathlib import Path
            data = json.loads(Path(manifest_json).read_text())
            subs = { (it.get("sub_group") or "undefined").lower() for it in data }
            return max(16, len(subs))
        except Exception:
            return 16

    def on_fit_start(self):
        # one-item, deterministic preview set
        ds_prev = PerformerAIDataset(
            json_path=self.manifest_json,
            conditioning_dropout={"piano_roll":0.0, "amp":0.0, "rbend":0.0, "rframe":0.0},
            use_trim=True,
            pre_roll_seconds=1.0,
            post_roll_seconds=0.0,
            keep_untrimmed_prob=0.0,         # <- deterministic
            amp_activity_thr=0.06,
            require_all_core=True,
            collapse_sparse_subgroups_to_any=False,
            static_window=True,               # <- fixed window
            window_slow=getattr(self.hparams, "window_slow", 256),
            seed=0
        )
        idx = int(self.hparams.preview_index) % len(ds_prev)
        item = ds_prev[idx]
        self._preview_batch = collate_latent_cond([item])  # stays on CPU; preview fns .to(self.device)
        self._wrote_gt = False  # ensure we write GT for this fixed clip


    def _save_gt_once(self, batch, sr_out=48000):
        if self._wrote_gt:
            return
        x0 = batch["latents"][:1].to(self.device)
        T_slow = x0.shape[-1]
        audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
        audio_lengths = torch.full((1,), audio_len, device=self.device, dtype=torch.long)
        sr_gt, wav_gt = self.dcae.decode(x0, audio_lengths=audio_lengths, sr=sr_out)
        out_dir = f"{getattr(self.logger, 'log_dir', './logs')}/eval_results/step_{self.global_step}"
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(f"{out_dir}/gt.wav", wav_gt[0].float().cpu(), sr_gt)
        self._wrote_gt = True
        

    def _scale_in(self, x, t):
        # Diffusers-style scaling for sigma-param schedulers
        if hasattr(self.scheduler, "scale_model_input"):
            return self.scheduler.scale_model_input(x, t)
        return x
        


    # helper (drop anywhere inside Pipeline, e.g., under other helpers)
    def _preview_sigma0(self):
        if hasattr(self.scheduler, "sigmas"):
            # take the largest sigma in the schedule
            return float(self.scheduler.sigmas.max().item())
        return 1.0


    def _sample_ts_and_sigmas(self, bsz, ref_tensor=None):
        # pick device from Lightning
        device = self.device

        # [B] integer timesteps on the training device
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,),
            device=device
        )

        # move the whole sigma schedule to that device *before* indexing
        sigmas_all = self.scheduler.sigmas.to(device=device, dtype=torch.float32)

        # [B] grab the per-sample sigma
        sigmas = sigmas_all.index_select(0, timesteps)  # equivalent to sigmas_all[timesteps]

        # optional: broadcast to match a reference tensor's shape
        if ref_tensor is not None:
            sigmas = sigmas.view((bsz,) + (1,) * (ref_tensor.ndim - 1))

        return timesteps, sigmas


    # In trainer_performer.py

    def train_dataloader(self):
        # For the overfitting test, we create a dummy dataloader that
        # yields our single, pre-loaded preview batch.


#TEST DATALOADER:

        # if self._preview_batch is not None:

        #     dummy_dataset = [self._preview_batch]
            

        #     return DataLoader(
        #         dummy_dataset, 
        #         batch_size=1, 
        #         collate_fn=lambda x: x[0]
        #     )

        # This is the original code for full dataset training.
        # It will only be used when you are NOT overfitting a single file.
        ds = PerformerAIDataset(
            json_path=self.manifest_json,
            conditioning_dropout={"piano_roll":0.15, "amp":0.10, "rbend":0.10, "rframe":0.05},
            use_trim=True,
            pre_roll_seconds=1.0,
            post_roll_seconds=0.25,
            keep_untrimmed_prob=0.1,
            amp_activity_thr=0.06,
            require_all_core=True,
            collapse_sparse_subgroups_to_any=False,
            static_window=False,
            window_slow=getattr(self.hparams, "window_slow", 256),
            seed=0
        )
        return DataLoader(
            ds,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_latent_cond,
            batch_size=self.hparams.batch_size,
        )

    @torch.no_grad()
    def _preview_x0_direct_rf(self, batch, t_scalar=0.5, sr_out=48000):
        x0 = batch["latents"][:1].to(self.device)
        B,_,_,T_slow = x0.shape
        tokens, mask = self.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"].to(self.device),
            amp=batch["conds"]["amp"].to(self.device),
            rframe=batch["conds"]["rframe"].to(self.device),
            rbend=batch["conds"]["rbend"].to(self.device),
            rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
            encodec_tokens=batch["encodec_tokens"].to(self.device),
            group_id=batch["instrument"]["group_id"].to(self.device),
            subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
        )
        t = torch.full((B,), float(t_scalar), device=self.device)
        z = torch.randn_like(x0)
        x_t = (1.0 - t.view(B,1,1,1)) * x0 + t.view(B,1,1,1) * z
        t_embed = t * (int(getattr(self.scheduler.config, "num_train_timesteps", 1000)) - 1)
        v_pred = self.transformers(
            hidden_states=x_t, timestep=t_embed,
            encoder_hidden_states=tokens, encoder_attention_mask=mask,
        ).sample
        x0_hat = x_t - t.view(B,1,1,1) * v_pred

        K = min(T_slow, int(6.0 * SLOW_HZ))   # ~6 seconds
        x_slice = x0_hat[..., :K]
        audio_len = int(round(K * DCAE_HOP * (sr_out / DCAE_SR)))
        audio_lengths = torch.full((1,), audio_len, device=self.device, dtype=torch.long)
        sr_pred, wav_pred = self.dcae.decode(x_slice[:1], audio_lengths=audio_lengths, sr=32000)
        out_dir = f"{self.logger.log_dir}/eval_results/step_{self.global_step}"
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(f"{out_dir}/x0_direct_rf_{self.global_step}.wav", wav_pred[0].float().cpu(), sr_pred)


    @torch.no_grad()
    def _preview_from_noisy_gt(self, batch, t0=0.8, steps=30, sr_out=48000):
        self.transformers.eval()
        x0 = batch["latents"][:1].to(self.device)
        B, _, _, T_slow = x0.shape

        # controls
        tokens, mask = self.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"].to(self.device),
            amp=batch["conds"]["amp"].to(self.device),
            rframe=batch["conds"]["rframe"].to(self.device),
            rbend=batch["conds"]["rbend"].to(self.device),
            rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
            encodec_tokens=batch["encodec_tokens"].to(self.device),
            group_id=batch["instrument"]["group_id"].to(self.device),
            subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
        )

        # start near x0 instead of pure noise
        torch.manual_seed(0)
        z = torch.randn_like(x0)
        x = (1.0 - t0) * x0 + t0 * z

        T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        dt = t0 / float(steps)

        for i in range(steps, 0, -1):   # integrate t: t0 -> 0
            t_cont = torch.full((1,), i * dt, device=self.device)
            t_embed = t_cont * (T_train - 1)
            v_pred = self.transformers(
                hidden_states=x,
                timestep=t_embed,
                encoder_hidden_states=tokens[:1],
                encoder_attention_mask=mask[:1],
            ).sample
            x = x - dt * v_pred

        audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
        audio_lengths = torch.full((1,), audio_len, device=self.device, dtype=torch.long)
        sr_pred, wav_pred = self.dcae.decode(x[:1], audio_lengths=audio_lengths, sr=sr_out)
        out_dir = f"{self.logger.log_dir}/eval_results/step_{self.global_step}"
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(f"{out_dir}/preview_from_gt_{self.global_step}.wav", wav_pred[0].float().cpu(), sr_pred)
        self.transformers.train()


    @torch.no_grad()
    def _preview(self, batch, steps=30, sr_out=48000):
        self.transformers.eval()
        x0 = batch["latents"][:1].to(self.device)
        B, _, _, T_slow = x0.shape

        # decode and save GT once (so you have a reference)
        if not self._wrote_gt:
            audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
            audio_lengths = torch.full((1,), audio_len, device=self.device, dtype=torch.long)
            sr_gt, wav_gt = self.dcae.decode(x0[:1], audio_lengths=audio_lengths, sr=sr_out)
            out_dir = f"{self.logger.log_dir}/eval_results/step_{self.global_step}"
            os.makedirs(out_dir, exist_ok=True)
            torchaudio.save(f"{out_dir}/gt.wav", wav_gt[0].float().cpu(), sr_gt)
            self._wrote_gt = True

        # controls
        tokens, mask = self.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"].to(self.device),
            amp=batch["conds"]["amp"].to(self.device),
            rframe=batch["conds"]["rframe"].to(self.device),
            rbend=batch["conds"]["rbend"].to(self.device),
            rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
            encodec_tokens=batch["encodec_tokens"].to(self.device),
            group_id=batch["instrument"]["group_id"].to(self.device),
            subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
        )
        tokens, mask = tokens[:B], mask[:B]

        # RF sampler: start from standard normal (NOT sigma_max * noise)
        torch.manual_seed(0)
        x = torch.randn_like(x0)
        T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        dt = 1.0 / float(steps)

        for i in range(steps, 0, -1):     # t: 1.0 -> 0.0
            t_cont = torch.full((B,), i * dt, device=self.device)
            t_embed = t_cont * (T_train - 1)
            v_pred = self.transformers(
                hidden_states=x,
                timestep=t_embed,
                encoder_hidden_states=tokens,
                encoder_attention_mask=mask,
            ).sample
            x_bad = x + dt * v_pred 
            x = x - dt * v_pred
            

        # decode
        audio_len_out = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
        audio_lengths = torch.full((1,), audio_len_out, device=self.device, dtype=torch.long)
        sr_pred, wav_pred = self.dcae.decode(x[:1], audio_lengths=audio_lengths, sr=sr_out)
        out_dir = f"{self.logger.log_dir}/eval_results/step_{self.global_step}"
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(f"{out_dir}/preview_{self.global_step}.wav", wav_pred[0].float().cpu(), sr_pred)
        self.transformers.train()

    def on_train_batch_start(self, batch, batch_idx):
        # save GT once for the fixed preview clip
        if self._preview_batch is not None:
            self._save_gt_once(self._preview_batch)


    def on_train_batch_end(self, outputs, batch, batch_idx):
        fixed = self._preview_batch if self._preview_batch is not None else batch

        preview_now = (self.global_step in (0,1,2,5,10,20)) or (
            ((self.global_step + 1) % max(1, getattr(self.hparams, "every_plot_step", 999999))) == 0
        )
        if preview_now:
            try:
                self._preview_x0_direct_rf(fixed, t_scalar=0.05, sr_out=32000)
                self._preview_from_noisy_gt(fixed, t0=0.2, steps=min(20, self.hparams.preview_steps), sr_out=32000)
                if self.global_step in (0, 5, 10, 20, 50, 100):
                    self._preview(batch, steps=min(10, self.hparams.preview_steps), sr_out=32000)
            except Exception as e:
                print(f"[preview] skipped due to error: {e}")

        if (self.global_step + 1) % max(1, getattr(self.hparams, "every_plot_step", 999999)) == 0:
            try:
                self._preview(fixed, steps=self.hparams.preview_steps, sr_out=48000)
            except Exception as e:
                print(f"[preview] skipped due to error: {e}")

        # optional: full-from-noise preview less frequently
        if (self.global_step + 1) % max(1, getattr(self.hparams, "every_plot_step", 999999)) == 0:
            try:
                self._preview(batch, steps=self.hparams.preview_steps, sr_out=48000)
            except Exception as e:
                print(f"[preview] skipped due to error: {e}")

            

        
    def _resample_curve(self, x_bt, K):
        K = int(K)
        if K <= 0:
            K = x_bt.shape[-1]
        x = x_bt.unsqueeze(1)
        y = F.interpolate(x, size=K, mode="linear", align_corners=False)
        return y.squeeze(1)


    def _ensure_timbre_cond(self, encodec_tokens, Kc, num_embedders):
        """
        encodec_tokens: [B, C_fast, T_fast]
        Returns: [B, num_embedders, Kc] (trim/pad channels, resize time)
        """
        import torch, torch.nn.functional as F
        B, C_fast, T_fast = encodec_tokens.shape
        x = encodec_tokens
        if C_fast > num_embedders:
            x = x[:, :num_embedders, :]
        elif C_fast < num_embedders:
            pad = torch.zeros(B, num_embedders - C_fast, T_fast, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        x = F.interpolate(x, size=Kc, mode="linear", align_corners=False)  # [B,num_embedders,Kc]
        return x

    def _pad_to_len(self, x, L: int):
        # x: [B, N] -> [B, L]
        if x.shape[-1] < L:
            return F.pad(x, (0, L - x.shape[-1]))
        elif x.shape[-1] > L:
            return x[..., :L]
        return x

    def _gather_time_nearest(self, x_bct, K: int):
        """
        x_bct: [B, C, T] of integer codes
        Returns [B, C, K] by nearest-neighbor index selection (no float interpolate).
        """
        B, C, T = x_bct.shape
        # indices 0..T-1 sampled to length K
        idx = torch.linspace(0, T - 1, steps=K, device=x_bct.device)
        idx = idx.round().long().clamp_(0, T - 1)               # [K]
        # gather along time
        return x_bct.index_select(dim=2, index=idx)             # [B, C, K]




    def _resample_channels_time(self, x_ct, K):
        # x_ct: [B,C,T]
        B, C, T = x_ct.shape
        if K % C == 0:
            Tprime = K // C
        else:
            Tprime = max(1, round(K / C))
        y = F.interpolate(x_ct, size=Tprime, mode="linear", align_corners=False)  # [B,C,T']
        vec = y.reshape(B, C*Tprime)
        if vec.shape[-1] < K:
            vec = F.pad(vec, (0, K - vec.shape[-1]))
        elif vec.shape[-1] > K:
            vec = vec[..., :K]
        return vec  # [B,K]

        # ----- Optim -----
  

    def configure_optimizers(self):
        trainable = [p for p in self.transformers.parameters() if p.requires_grad] + \
                    [p for p in self.ctrl_enc.parameters() if p.requires_grad]
        
        # Use more stable, standard hyperparameters
        lr = 1e-4 
        weight_decay = 1e-2
        betas = (0.9, 0.999)

        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(
                trainable, lr=lr, weight_decay=weight_decay, betas=betas
            )
        except Exception:
            opt = torch.optim.AdamW(
                trainable, lr=lr, weight_decay=weight_decay, betas=betas, eps=1e-8, foreach=False
            )

        def lr_lambda(cur):
            if cur < self.hparams.warmup_steps:
                return float(cur) / max(1, self.hparams.warmup_steps)
            progress = float(cur - self.hparams.warmup_steps) / max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            return max(0.0, 1.0 - progress)

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=-1)
        return [opt], [{"scheduler": sch, "interval": "step"}]
    # ----- Helpers -----
    def get_timestep(self, bsz, device):
        # Logit-normal pick like ACE (defaults)
        u = torch.normal(mean=0.0, std=1.0, size=(bsz,), device="cpu")
        u = torch.sigmoid(u)
        idx = (u * self.scheduler.config.num_train_timesteps).long().clamp(0, self.scheduler.config.num_train_timesteps - 1)
        return self.scheduler.timesteps[idx].to(device)

    def get_sd3_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)              # [T]
        schedule_ts = self.scheduler.timesteps.to(device=device, dtype=timesteps.dtype)  # [T]

        # Find nearest index in schedule for each sampled t
        # result: step_indices shape [B]
        diff = (schedule_ts[None, :] - timesteps[:, None]).abs()
        step_indices = diff.argmin(dim=1)

        sigma = sigmas[step_indices]  # [B]
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)  # -> [B,1,1,1] for 4D latents
        return sigma



    # def _prepare_explicit_conds(self, batch, T_slow):
    #     pr   = batch["conds"]["piano_roll"].to(self.device)              # [B,128,T]
    #     amp  = batch["conds"]["amp"].to(self.device)                     # [B,T]
    #     rb   = (batch["conds"]["rbend"].to(self.device) *
    #             batch["conds"]["rbend_mask"].to(self.device))            # [B,T]
    #     enc  = batch["encodec_tokens"].to(self.device)                   # [B,C_fast,Tfast]

    #     B = pr.shape[0]
    #     Dtext = getattr(self.transformers.config, "text_embedding_dim", 768)
    #     prompt = torch.zeros(B, 1, Dtext, device=self.device, dtype=pr.dtype)

    #     # --- fixed conditioning lengths from the model buffers ---


    #     # Piano roll → [B, 128*PR_T]




    #     # Timbre per-channel → [B, C, K] (indices if Embedding)
    #     num_embedders = len(self.transformers.encodec_embedders)
    #     is_emb = isinstance(self.transformers.encodec_embedders[0], torch.nn.Embedding)

    #     if is_emb:
    #         x = enc.round().long()
    #         C_fast = x.shape[1]
    #         if C_fast > num_embedders: x = x[:, :num_embedders, :]
    #         elif C_fast < num_embedders:
    #             pad = torch.zeros(B, num_embedders - C_fast, x.shape[-1], device=x.device, dtype=x.dtype)
    #             x = torch.cat([x, pad], dim=1)
    #         Kc = x.shape[-1]
    #         timb = self._gather_time_nearest(x, Kc)                                  # [B,C,Kc]
    #         vocab = self.transformers.encodec_embedders[0].num_embeddings
    #         timb = timb.clamp_(0, vocab - 1)
    #     else:
    #         Kc = getattr(self.transformers.encodec_embedders[0], "in_features", T_slow)
    #         x = enc.to(torch.float32)
    #         C_fast = x.shape[1]
    #         if C_fast > num_embedders: x = x[:, :num_embedders, :]
    #         elif C_fast < num_embedders:
    #             pad = torch.zeros(B, num_embedders - C_fast, x.shape[-1], device=x.device, dtype=x.dtype)
    #             x = torch.cat([x, pad], dim=1)
    #         timb = F.interpolate(x, size=Kc, mode="linear", align_corners=False)     # [B,C,Kc]

    #     return {
    #         "prompt_embeds":   prompt,
    #         "piano_roll_cond": pr_vec,
    #         "pitch_bend_cond": pb,
    #         "dynamics_cond":   dyn,
    #         "timbre_cond":     timb,
    #     }



    def _call_transformer(self, latents, tokens, token_mask, timesteps, attn_mask, batch=None):
        """
        Robustly call the transformer no matter which API this build exposes:
        - explicit conditioning API: (prompt_embeds, piano_roll_cond, pitch_bend_cond, dynamics_cond, timbre_cond)
        - cross-attention API: (encoder_text_hidden_states + masks)
        """
        sig = inspect.signature(self.transformers.forward).parameters
        def has(*names): return all(n in sig for n in names)

        # Branch 1: explicit cond API
        if has("prompt_embeds","piano_roll_cond","pitch_bend_cond","dynamics_cond","timbre_cond"):
            assert batch is not None, "batch is required for explicit cond API"
            B, _, _, T_slow = latents.shape
            conds = self._prepare_explicit_conds(batch, T_slow)

            def try_call(pr, timb):
                return self.transformers(
                    hidden_states=latents,
                    timestep=timesteps,
                    prompt_embeds=conds["prompt_embeds"],
                    piano_roll_cond=pr,
                    pitch_bend_cond=conds["pitch_bend_cond"],
                    dynamics_cond=conds["dynamics_cond"],
                    timbre_cond=timb,
                )

            # try canonical layout first; if shape mismatch, flip time/channel dims
            try:
                out = try_call(conds["piano_roll_cond"], conds["timbre_cond"])
            except RuntimeError:
                pr = conds["piano_roll_cond"].transpose(1,2) if conds["piano_roll_cond"].dim()==3 else conds["piano_roll_cond"]
                timb = conds["timbre_cond"].transpose(1,2)   if conds["timbre_cond"].dim()==3 else conds["timbre_cond"]
                out = try_call(pr, timb)

            return out.sample if hasattr(out, "sample") else out

        # Branch 2: cross-attention API (your original control-token path)
        kwargs = {}
        if "hidden_states" in sig:  kwargs["hidden_states"] = latents
        elif "x" in sig:            kwargs["x"] = latents

        for k in ("timestep","timesteps","t","sigma","noise_sigma"):
            if k in sig:
                kwargs[k] = timesteps
                break

        for k in ("encoder_text_hidden_states","encoder_hidden_states","context","text_embeds"):
            if k in sig:
                kwargs[k] = tokens
                break

        for k in ("text_attention_mask","encoder_hidden_mask","encoder_attention_mask","context_mask"):
            if k in sig:
                kwargs[k] = token_mask
                break

        for k in ("attention_mask","attn_mask","mask"):
            if k in sig:
                kwargs[k] = attn_mask
                break

        if "speaker_embeds" in sig:
            kwargs["speaker_embeds"] = torch.zeros(tokens.shape[0], 512, device=latents.device, dtype=latents.dtype)
        if "lyric_token_idx" in sig:
            kwargs["lyric_token_idx"] = torch.zeros(tokens.shape[0], 1, device=latents.device, dtype=torch.long)
        if "lyric_mask" in sig:
            kwargs["lyric_mask"] = torch.zeros(tokens.shape[0], 1, device=latents.device, dtype=torch.long)

        out = self.transformers(**kwargs)
        return out.sample if hasattr(out, "sample") else out


    # ----- Training core -----

    
    # def training_step(self, batch, batch_idx):
    #     x0 = batch["latents"].to(self.device)
    #     B  = x0.shape[0]

    #     tokens, mask = self.ctrl_enc(
    #         piano_roll=batch["conds"]["piano_roll"].to(self.device),
    #         amp=batch["conds"]["amp"].to(self.device),
    #         rframe=batch["conds"]["rframe"].to(self.device),
    #         rbend=batch["conds"]["rbend"].to(self.device),
    #         rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
    #         encodec_tokens=batch["encodec_tokens"].to(self.device),
    #         group_id=batch["instrument"]["group_id"].to(self.device),
    #         subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
    #     )

    #     t, sigma = self._sample_ts_and_sigmas(B, ref_tensor=x0)       # [B], [B]
    #     while sigma.ndim < x0.ndim:                       # -> [B,1,1,1]
    #         sigma = sigma.unsqueeze(-1)

    #     eps  = torch.randn_like(x0)
    #     x_t  = x0 + sigma * eps
    #     x_in = self._scale_in(x_t, t)

    #     eps_pred = self.transformers(
    #         hidden_states=x_in,
    #         timestep=t,                      # keep as integer
    #         encoder_hidden_states=tokens,
    #         encoder_attention_mask=mask,
    #     ).sample


    #     loss = F.mse_loss(eps_pred, eps)
    #     self.log("train/loss", loss, on_step=True, prog_bar=True)
    #     self.log("train/lr", self.lr_schedulers().get_last_lr()[0], on_step=True)
    #     return loss



# OG ACE STEP DIFFUSION

    def training_step(self, batch, batch_idx):
    # The dataloader is now fast, so we can use the batch directly.
    # We just need to move its contents to the GPU.
        def to_device(data, device):
            if isinstance(data, torch.Tensor):
                return data.to(device)
            if isinstance(data, dict):
                return {k: to_device(v, device) for k, v in data.items()}
            if isinstance(data, list):
                return [to_device(i, device) for i in data]
            return data
        batch = to_device(batch, self.device)

        # --- The rest of your training step logic ---
        x0 = batch["latents"]
        B  = x0.shape[0]

        tokens, mask = self.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"],
            amp=batch["conds"]["amp"],
            rframe=batch["conds"]["rframe"],
            rbend=batch["conds"]["rbend"],
            rbend_mask=batch["conds"]["rbend_mask"],
            encodec_tokens=batch["encodec_tokens"],
            group_id=batch["instrument"]["group_id"],
            subgroup_id=batch["instrument"]["subgroup_id"],
        )

        # ----- x0-reconstruction rectified-flow objective -----
        tau = torch.rand(B, device=x0.device, dtype=x0.dtype)
        T_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        
        t_int = (tau * (T_train - 1)).long()
# Move the scheduler's timesteps to the same device as the indices
        t_for_embed = self.scheduler.timesteps.to(t_int.device)[t_int]

        sigma = tau.view(B, *([1] * (x0.ndim - 1))).to(dtype=x0.dtype)
        z     = torch.randn_like(x0)
        x_noisy = (1.0 - sigma) * x0 + sigma * z

        v_pred = self.transformers(
            hidden_states=x_noisy,
            timestep=t_for_embed,
            encoder_hidden_states=tokens,
            encoder_attention_mask=mask,
        ).sample

        x0_hat = x_noisy - sigma * v_pred
        loss = F.mse_loss(x0_hat, x0)

        # ---- diagnostics ----
        with torch.no_grad():
            v_target = z - x0
            cos = F.cosine_similarity(v_pred.flatten(1), v_target.flatten(1), dim=1).mean()
            self.log("dbg/cos_vpred_vtgt", cos, on_step=True)
            self.log("dbg/|v_pred|", v_pred.float().pow(2).mean().sqrt(), on_step=True)
            self.log("dbg/|v_tgt|",  v_target.float().pow(2).mean().sqrt(), on_step=True)

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0], on_step=True)
        return loss


# NEW METHOD

    # def training_step(self, batch, batch_idx):
    #     x0 = batch["latents"].to(self.device)
    #     B = x0.shape[0]

    #     # --- build cond tokens ---
    #     tokens, mask = self.ctrl_enc(
    #         piano_roll=batch["conds"]["piano_roll"].to(self.device),
    #         amp=batch["conds"]["amp"].to(self.device),
    #         rframe=batch["conds"]["rframe"].to(self.device),
    #         rbend=batch["conds"]["rbend"].to(self.device),
    #         rbend_mask=batch["conds"]["rbend_mask"].to(self.device),
    #         encodec_tokens=batch["encodec_tokens"].to(self.device),
    #         group_id=batch["instrument"]["group_id"].to(self.device),
    #         subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),
    #     )

    #     # --- sample t and σ ON THE SAME DEVICE ---
    #     timesteps_all = self.scheduler.timesteps.to(self.device)
    #     sigmas_all    = self.scheduler.sigmas.to(self.device)

    #     idx = torch.randint(0, timesteps_all.numel(), (B,), device=self.device)
    #     t   = timesteps_all[idx]         # [B]
    #     σ   = sigmas_all[idx]            # [B]

    #     # broadcast σ to x0 shape
    #     σ = σ.view(B, *([1] * (x0.ndim - 1)))

    #     # --- epsilon training objective ---
    #     noise   = torch.randn_like(x0)
    #     x_noisy = x0 + σ * noise

    #     eps_hat = self.transformers(
    #         hidden_states=x_noisy,
    #         timestep=t,
    #         encoder_hidden_states=tokens,
    #         encoder_attention_mask=mask,
    #     ).sample

    #     loss = F.mse_loss(eps_hat, noise)
    #     self.log("train/loss", loss, on_step=True, prog_bar=True)
    #     self.log("train/lr", self.lr_schedulers().get_last_lr()[0], on_step=True)
    #     return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Use the same batch as training for overfitting test
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=True)
        return loss




    @torch.no_grad()
    def _log_preview(self, batch, tokens, token_mask, infer_steps=30, guidance_scale=0.0, omega_scale=10.0):
        """Quick sample to hear progress. Uses your control tokens (no text)."""
        target_latents = batch["latents"].to(self.device)
        bsz, _, _, T_slow = target_latents.shape
        dtype = target_latents.dtype

        # in _log_preview(...)
        scheduler = self.scheduler
        timesteps = self.scheduler.set_timesteps(infer_steps, device=self.device) or self.scheduler.timesteps
        sigma0 = float(self.scheduler.sigmas[0].item())
        lat = randn_tensor((bsz, 8, 16, T_slow), device=self.device, dtype=dtype) * sigma0

        for t in timesteps:
            pred = self._call_transformer(
                latents=self._scale_in(lat, t),
                tokens=tokens, token_mask=token_mask,
                timesteps=t.expand(bsz), attn_mask=None, batch=batch
            )
            lat = self.scheduler.step(model_output=pred, timestep=t, sample=lat, return_dict=False)[0]

        # Decode a single example
        audio_len_48k = int(round(T_slow * DCAE_HOP * (48000 / DCAE_SR)))
        audio_lengths = torch.full((1,), audio_len_48k, device=self.device, dtype=torch.long)
        sr_pred, wav_pred = self.dcae.decode(lat[:1], audio_lengths=audio_lengths, sr=48000)

        log_dir = self.logger.log_dir if hasattr(self.logger, "log_dir") else "./logs"
        save_dir = f"{log_dir}/eval_results/step_{self.global_step}"
        os.makedirs(save_dir, exist_ok=True)
        torchaudio.save(f"{save_dir}/pred_{self.global_step}.wav", wav_pred[0].float().cpu(), sr_pred)


    def plot_loss_from_tfevents(tfevents_path, out_png_path):
        """Reads a TensorBoard tfevents file and plots the train/loss scalar."""
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing import event_accumulator

        print(f"Reading tfevents file: {tfevents_path}")
        if not os.path.exists(tfevents_path):
            print(f"Error: File not found at {tfevents_path}")
            return

        ea = event_accumulator.EventAccumulator(
            tfevents_path,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()

        if 'train/loss' not in ea.Tags()['scalars']:
            print("Error: 'train/loss' tag not found in the tfevents file.")
            print(f"Available tags: {ea.Tags()['scalars']}")
            return

        loss_events = ea.Scalars('train/loss')
        steps = [e.step for e in loss_events]
        values = [e.value for e in loss_events]

        plt.figure(figsize=(12, 6))
        plt.plot(steps, values, label='Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(out_png_path)
        print(f"Loss curve saved to: {out_png_path}")




def main(args):
    # parse overfit here so it's in-scope
    def _parse_overfit(v):
        if v is None:
            return None
        try:
            f = float(v)   # accepts "1", "1.0", 1, 0.1, etc.
        except ValueError:
            raise ValueError(f"--overfit_batches must be a number (got {v!r})")
        return int(f) if f >= 1 else f

    ov = _parse_overfit(args.overfit_batches)

    model = Pipeline(
        checkpoint_dir=args.checkpoint_dir,
        manifest_json=args.manifest_json,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        shift=args.shift,
        max_steps=args.max_steps,
        every_plot_step=args.every_plot_step,
        cond_cfg_drop_prob=args.cond_cfg_drop_prob,
        warmup_steps=args.warmup_steps,
        override_group=args.override_group,
        override_subgroup=args.override_subgroup,
        preview_steps=args.preview_steps,
        batch_size=args.batch_size,
        preview_index=args.preview_index,
        train_from_scratch=args.train_from_scratch
    )

    ckpt_cb = ModelCheckpoint(monitor=None, every_n_train_steps=args.every_n_train_steps, save_top_k=-1)
    logger_cb = TensorBoardLogger(version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.exp_name,
                                  save_dir=args.logger_dir)

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="auto" if args.devices == 1 else "ddp_find_unused_parameters_true",
        max_epochs=-1,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_cb,
        callbacks=[ckpt_cb],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        val_check_interval=None,     # ← use this instead of overfit mode
        limit_val_batches=0,        # ← no val loop
        num_sanity_val_steps=0
    )

    trainer.fit(model, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_nodes", type=int, default=1)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")



    ap.add_argument("--plot_only", type=str, default=None, 
                    help="Path to a tfevents file to plot. If provided, skips training.")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=-1)
    ap.add_argument("--max_steps", type=int, default=2000000)
    ap.add_argument("--every_n_train_steps", type=int, default=2000)
    ap.add_argument("--manifest_json", type=str, default="./final_training_manifest.json")
    ap.add_argument("--exp_name", type=str, default="scratch_perf_transformer")
    ap.add_argument("--precision", type=str, default="32")
    ap.add_argument("--accumulate_grad_batches", type=int, default=4)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--logger_dir", type=str, default="./exps/logs/")
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--checkpoint_dir", type=str, required=True, help="Dir that contains ace_step_transformer/config.json + music_dcae_f8c8 + music_vocoder (can be a snapshots/<hash> dir)")
    ap.add_argument("--gradient_clip_val", type=float, default=1.0)
    ap.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    ap.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    ap.add_argument("--every_plot_step", type=int, default=2000)
    ap.add_argument("--val_check_interval", type=int, default=None)
    ap.add_argument("--cond_cfg_drop_prob", type=float, default=0.15)

    ap.add_argument("--override_group", type=str, default=None,
                help="Instrument group name or int id (e.g., 'strings' or 2)")
    ap.add_argument("--override_subgroup", type=str, default=None,
                    help="Instrument subgroup name or int id (e.g., 'violin' or 7)")

    ap.add_argument("--overfit_batches", type=str, default=None)
    ap.add_argument("--window_slow", type=int, default=2048)


    ap.add_argument("--preview_steps", type=int, default=50,
                help="Denoising steps for preview sampling")

    ap.add_argument("--preview_index", type=int, default=0,
            help="The index of the audio file in the manifest to use for previews.")

    ap.add_argument("--train_from_scratch", action="store_true",
                help="If set, trains a new model from scratch. Otherwise, fine-tunes the pre-trained ACE-Step model.")

 # for this test only



    args = ap.parse_args()

    if args.plot_only:
        output_path = os.path.join(os.path.dirname(args.plot_only), "loss_curve.png")
        plot_loss_from_tfevents(args.plot_only, output_path)
    else:
        main(args)
    
    # main(args)
