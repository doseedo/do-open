"""
Inference helper for the boundary-repair latent editor.

Usage from production code:

    from latent_editor.infer import LatentEditorRuntime
    rt = LatentEditorRuntime("/scratch/latent_editor_ckpts/editor_final.pt")
    L_out = rt.edit(L_a, L_b, cut_sample)        # sample-accurate concat
    # L_a, L_b: [T, 64] tensors  (matches /Latents2 layout after transpose)
    # cut_sample: int, sample-accurate cut position into L_a (samples @ 48kHz)

The model only repairs frames in a small band around the cut; everything
else is exact pass-through (residual is masked).
"""
from __future__ import annotations
import torch
from .model import LatentEditor
from .dataset import SAMPLES_PER_FRAME


class LatentEditorRuntime:
    def __init__(self, ckpt_path: str, device: str = "cuda", boundary_radius: int = 4):
        self.device = device
        self.r = boundary_radius
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # rebuild model with the exact max_len used at training time so the
        # positional embedding matches the checkpoint
        pos_shape = ckpt["model"].get("pos")
        max_len = pos_shape.shape[1] if pos_shape is not None else ckpt.get("args", {}).get("win_frames", 64)
        self.win = max_len
        self.model = LatentEditor(max_len=max_len).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def edit(
        self,
        L_a: torch.Tensor,    # [Ta, 64]
        L_b: torch.Tensor,    # [Tb, 64]
        cut_sample: int,      # sample-accurate position (samples @ 48kHz) into L_a
    ) -> torch.Tensor:
        """Sample-accurate concat: keep L_a up to cut_sample, then L_b after."""
        # frame-aligned cut + sub-frame phase
        cut_frame = cut_sample // SAMPLES_PER_FRAME
        sub = cut_sample - cut_frame * SAMPLES_PER_FRAME
        phase = sub / SAMPLES_PER_FRAME

        # naive splice in latent space
        L_naive_full = torch.cat([L_a[:cut_frame], L_b[cut_frame:]], dim=0)  # [T,64]

        # Center a fixed-size window around the cut so that the model sees
        # the same positional layout it was trained with (cut at win//2).
        win = self.win
        half = win // 2
        T = L_naive_full.shape[0]
        # left/right halves taken from L_a and L_b respectively, then padded
        left = L_a[max(0, cut_frame - half) : cut_frame]
        right = L_b[cut_frame : cut_frame + (win - half)]
        if left.shape[0] < half:
            left = torch.cat([torch.zeros(half - left.shape[0], 64), left], 0)
        if right.shape[0] < (win - half):
            right = torch.cat(
                [right, torch.zeros((win - half) - right.shape[0], 64)], 0
            )
        local = torch.cat([left, right], 0).to(self.device)  # [win, 64]

        mask = torch.zeros(win, device=self.device)
        mask[max(0, half - self.r) : min(win, half + self.r + 1)] = 1.0

        repaired = self.model(
            local.unsqueeze(0),
            mask.unsqueeze(0),
            torch.tensor([phase], device=self.device, dtype=torch.float32),
        )[0].cpu()

        # Splice repaired window back into the full naive sequence at the
        # boundary band only.
        out = L_naive_full.clone()
        # absolute positions of the boundary band in L_naive_full
        band_lo_local = max(0, half - self.r)
        band_hi_local = min(win, half + self.r + 1)
        abs_lo = cut_frame - half + band_lo_local
        abs_hi = cut_frame - half + band_hi_local
        clip_lo = max(0, abs_lo)
        clip_hi = min(T, abs_hi)
        rep_lo = band_lo_local + (clip_lo - abs_lo)
        rep_hi = band_hi_local - (abs_hi - clip_hi)
        out[clip_lo:clip_hi] = repaired[rep_lo:rep_hi].cpu()
        return out

    @torch.no_grad()
    def edit_many_into_track(
        self,
        track: torch.Tensor,            # [T, 64] — running output
        pastes: list,                   # list of (note_lat [N,64], start_frame, dur_frames)
    ) -> torch.Tensor:
        """Apply many splices into one track in a SINGLE batched forward
        pass. Each entry in `pastes` is a (note_lat, start_frame,
        dur_frames) tuple. The function:
          1. Builds the post-splice naive track by overwriting each
             paste region.
          2. Builds N model windows centered at each cut frame.
          3. Stacks them and runs ONE forward pass.
          4. Scatters the repaired boundary bands back into the track.

        ~Nx faster than calling .edit() in a loop because the dominant
        cost is the per-call model forward pass."""
        if not pastes:
            return track
        win = self.win
        half = win // 2
        T = track.shape[0]

        # Build naive post-splice track + capture L_a (pre-paste) snapshots.
        # Process pastes in time order so each window's "left" context
        # reflects the state just before this note's cut.
        ordered = sorted(pastes, key=lambda p: p[1])
        windows = []
        cut_frames = []
        bands = []   # (band_lo_local, band_hi_local, abs_lo, abs_hi) per paste

        # Snapshot of the track BEFORE each paste — we use the running
        # state so editor sees previous splices, but we DON'T re-clone T
        # tensors. Instead apply pastes in order to a single mutable
        # track and capture (left,right) windows just before each write.
        for note_lat, start_frame, dur_frames in ordered:
            if note_lat.shape[0] == 0 or dur_frames <= 0:
                continue
            use = note_lat[:dur_frames]
            paste_n = min(track.shape[0] - start_frame, use.shape[0])
            if paste_n <= 0:
                continue
            cut_frame = start_frame
            cut_sample = cut_frame * SAMPLES_PER_FRAME
            # phase is 0 for frame-aligned starts (our case)
            # Build the local window: half from current track (L_a) on
            # the left, half from the about-to-paste content on the right.
            left = track[max(0, cut_frame - half): cut_frame]
            if left.shape[0] < half:
                left = torch.cat([torch.zeros(half - left.shape[0], 64), left], 0)
            right_src = use[: (win - half)]
            if right_src.shape[0] < (win - half):
                right_src = torch.cat(
                    [right_src, torch.zeros((win - half) - right_src.shape[0], 64)], 0
                )
            local = torch.cat([left, right_src], 0)
            windows.append(local)
            cut_frames.append(cut_frame)

            band_lo_local = max(0, half - self.r)
            band_hi_local = min(win, half + self.r + 1)
            abs_lo = cut_frame - half + band_lo_local
            abs_hi = cut_frame - half + band_hi_local
            bands.append((band_lo_local, band_hi_local, abs_lo, abs_hi))

            # Apply the paste to the track now (so the next window's
            # left context reflects this note when notes overlap).
            track[start_frame:start_frame + paste_n] = use[:paste_n]

        if not windows:
            return track

        # One batched forward pass
        batch = torch.stack(windows, dim=0).to(self.device)            # [N, win, 64]
        masks = torch.zeros(batch.shape[0], win, device=self.device)
        masks[:, max(0, half - self.r): min(win, half + self.r + 1)] = 1.0
        phases = torch.zeros(batch.shape[0], device=self.device, dtype=torch.float32)
        repaired = self.model(batch, masks, phases).cpu()              # [N, win, 64]

        # Scatter repaired boundary bands back into the track
        for i, (band_lo_local, band_hi_local, abs_lo, abs_hi) in enumerate(bands):
            clip_lo = max(0, abs_lo)
            clip_hi = min(T, abs_hi)
            rep_lo = band_lo_local + (clip_lo - abs_lo)
            rep_hi = band_hi_local - (abs_hi - clip_hi)
            if clip_hi > clip_lo:
                track[clip_lo:clip_hi] = repaired[i, rep_lo:rep_hi]
        return track
