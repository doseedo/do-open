"""
Production inference: latent → MIDI (drop-in replacement for BasicPitch).

Usage:
    rt = LatentPitchRuntime("/scratch/latent_pitch_ckpts/pitch_final.pt")
    pm = rt.transcribe(latent)   # latent: [T, 64] tensor
    pm.write("out.mid")
"""
from __future__ import annotations
import torch
import numpy as np
import pretty_midi

from .model import LatentBasicPitchStudent
from .dataset import VAE_HZ, N_PITCH


class LatentPitchRuntime:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        onset_thresh: float = 0.7,   # tuned for Gaussian-smoothed onset training
        frame_thresh: float = 0.5,
        min_note_frames: int = 2,
    ):
        self.device = device
        self.onset_thresh = onset_thresh
        self.frame_thresh = frame_thresh
        self.min_note_frames = min_note_frames
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model"]
        pos = sd.get("pos")
        max_len = pos.shape[1] if pos is not None else 2048
        self.max_len = max_len
        # infer architecture sizes from the ckpt so v1 (256/4/4) and
        # v4 (384/6/8) both load
        d_model = pos.shape[2] if pos is not None else 256
        n_conv_blocks = sum(1 for k in sd if k.startswith("conv_stack.")
                            and k.endswith(".conv1.weight"))
        n_tr_layers = sum(1 for k in sd if k.startswith("tr.layers.")
                          and k.endswith(".self_attn.out_proj.weight"))
        self.model = LatentBasicPitchStudent(
            d_model=d_model, n_conv_blocks=n_conv_blocks or 4,
            n_tr_layers=n_tr_layers or 4, max_len=max_len,
        ).to(device)
        self.model.load_state_dict(sd)
        self.model.eval()

    @torch.no_grad()
    def predict(self, latent: torch.Tensor):
        """Returns dict of [T,128] numpy arrays: onset_prob, frame_prob, velocity.

        Supports inputs longer than the trained max_len by processing in
        non-overlapping chunks and concatenating the posteriors along
        time. Non-overlap is OK because each frame's posterior only
        depends on ~local context (short-range conv + transformer over
        a bounded window).
        """
        L = latent.to(self.device)
        if L.dim() == 2:
            L = L.unsqueeze(0)
        B, T, D = L.shape
        if T <= self.max_len:
            out = self.model(L)
        else:
            # Chunked inference: split [B, T, 64] into windows of max_len
            # frames, forward each, concat outputs along the time axis.
            parts = {"onset_logits": [], "frame_logits": [],
                     "velocity": [], "onset_offset": []}
            for s in range(0, T, self.max_len):
                e = min(s + self.max_len, T)
                chunk = L[:, s:e, :]
                oc = self.model(chunk)
                for k in parts:
                    parts[k].append(oc[k])
            out = {k: torch.cat(v, dim=1) for k, v in parts.items()}
        return {
            "onset_prob":   torch.sigmoid(out["onset_logits"])[0].cpu().numpy(),
            "frame_prob":   torch.sigmoid(out["frame_logits"])[0].cpu().numpy(),
            "velocity":     out["velocity"][0].cpu().numpy(),
            "onset_offset": out["onset_offset"][0].cpu().numpy(),  # in [0,1)
        }

    def transcribe(self, latent: torch.Tensor, program: int = 0) -> pretty_midi.PrettyMIDI:
        """Decode posteriors into a PrettyMIDI object using simple onset/frame
        gating (BasicPitch-style)."""
        p = self.predict(latent)
        onset  = p["onset_prob"]      # [T,128]
        frame  = p["frame_prob"]
        vel    = p["velocity"]
        offset = p["onset_offset"]    # [T,128] in [0,1) within frame
        T = onset.shape[0]
        dt = 1.0 / VAE_HZ

        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=program)

        # NMS-based onset picking: a frame is an onset only if it's a
        # local max of the onset probability within ±nms_radius frames
        # AND above threshold. This is the right post-processing when the
        # model was trained on Gaussian-smoothed onset targets — without
        # NMS, every frame in the smoothed neighborhood passes threshold
        # and the "first one" heuristic locks onto the leading edge,
        # which is ~2 frames before the true onset.
        nms_radius = 2
        for pitch in range(N_PITCH):
            on_p = onset[:, pitch]
            fr_mask = frame[:, pitch] > self.frame_thresh
            for t in range(T):
                if on_p[t] <= self.onset_thresh:
                    continue
                lo = max(0, t - nms_radius)
                hi = min(T, t + nms_radius + 1)
                if on_p[t] < on_p[lo:hi].max() - 1e-9:
                    continue  # not a local max
                # found a peak — extend through frame_mask
                end = t + 1
                while end < T and fr_mask[end]:
                    end += 1
                if end - t >= self.min_note_frames:
                    v = vel[t:end, pitch].mean()
                    sub = float(offset[t, pitch])
                    inst.notes.append(pretty_midi.Note(
                        velocity=int(np.clip(v * 127, 1, 127)),
                        pitch=pitch,
                        start=(t + sub) * dt,
                        end=end * dt,
                    ))

        pm.instruments.append(inst)
        return pm
