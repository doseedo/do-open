"""
Pitch-Conditioned Register Translator

Key differences from MuteTranslator:
1. Pitch embedding conditioning (learns what each pitch should sound like)
2. Shift amount conditioning (knows how much correction to apply)
3. Dual-path architecture (content preservation + timbre correction)
4. FiLM layers for feature-wise conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation for conditioning.

    FiLM applies an affine transformation to features based on conditioning:
        y = gamma(condition) * x + beta(condition)

    This allows the network to adaptively scale and shift features
    based on pitch and shift conditioning.
    """

    def __init__(self, num_features: int, condition_dim: int):
        super().__init__()
        self.scale = nn.Linear(condition_dim, num_features)
        self.shift = nn.Linear(condition_dim, num_features)

        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.ones_(self.scale.weight.data[:, :condition_dim // 2])
        nn.init.zeros_(self.scale.weight.data[:, condition_dim // 2:])
        nn.init.zeros_(self.scale.bias.data)
        nn.init.zeros_(self.shift.weight.data)
        nn.init.zeros_(self.shift.bias.data)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            x: [B, C, ...] features
            condition: [B, D] conditioning vector

        Returns:
            Modulated features [B, C, ...]
        """
        scale = self.scale(condition)  # [B, C]
        shift = self.shift(condition)  # [B, C]

        # Reshape for broadcasting
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)

        return x * (1 + scale) + shift


class ResidualBlockConditioned(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

        self.film1 = FiLMLayer(channels, condition_dim)
        self.film2 = FiLMLayer(channels, condition_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.film1(x, condition)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film2(x, condition)

        return self.act(x + residual)


class RegisterAwareTranslator(nn.Module):
    """
    Register-aware pitch shift corrector.

    Key features:
    1. Pitch embeddings: Learn what each semitone sounds like for this instrument
    2. Shift embeddings: Learn what each shift amount's artifacts look like
    3. FiLM conditioning: Modulate features based on target pitch/shift
    4. Residual architecture: Preserve content while correcting timbre

    Input: [B, C, H, T] degraded latent + target_pitch + shift_amount
    Output: [B, C, H, T] corrected latent

    The pitch embedding is the key innovation - it learns the characteristic
    formant structure and timbre at each pitch, allowing the model to know
    "what should C4 sound like on this instrument" vs "what should G5 sound like".
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        num_pitches: int = 128,  # Full MIDI range
        num_shifts: int = 25,    # -12 to +12 semitones
        pitch_embed_dim: int = 64,
        shift_embed_dim: int = 32,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels

        # Learnable embeddings for pitch (what each pitch SHOULD sound like)
        # This is the key to register-aware translation
        self.pitch_embeddings = nn.Embedding(num_pitches, pitch_embed_dim)

        # Learnable embeddings for shift amount (what artifacts look like)
        self.shift_embeddings = nn.Embedding(num_shifts, shift_embed_dim)

        # Combined condition dimension
        condition_dim = pitch_embed_dim + shift_embed_dim
        self.condition_dim = condition_dim

        # Input projection
        self.input_proj = nn.Conv1d(latent_channels, hidden_channels, 1)

        # Conditioned residual blocks with increasing dilation
        self.blocks = nn.ModuleList([
            ResidualBlockConditioned(
                hidden_channels,
                condition_dim,
                kernel_size,
                dilation=2 ** i
            )
            for i in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, latent_channels, 1),
        )

        # Initialize output small for residual learning
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Learnable residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        degraded_latent: torch.Tensor,
        target_pitch: torch.Tensor,
        shift_amount: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply register-aware pitch shift correction.

        Args:
            degraded_latent: [B, C, H, T] pitch-shifted latent with artifacts
            target_pitch: [B] target MIDI pitch (0-127)
            shift_amount: [B] shift in semitones (-12 to +12)

        Returns:
            corrected_latent: [B, C, H, T] corrected latent with proper timbre
        """
        B, C, H, T = degraded_latent.shape

        # Get embeddings
        pitch_emb = self.pitch_embeddings(target_pitch.long().clamp(0, 127))
        shift_idx = (shift_amount + 12).long().clamp(0, 24)
        shift_emb = self.shift_embeddings(shift_idx)

        # Combined conditioning vector
        condition = torch.cat([pitch_emb, shift_emb], dim=-1)  # [B, condition_dim]

        # Flatten H into batch for 1D processing: [B*H, C, T]
        x = degraded_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Expand condition for each H slice
        condition_expanded = condition.unsqueeze(1).expand(-1, H, -1).reshape(B * H, -1)

        # Process through network
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, condition_expanded)
        residual = self.output_proj(x)

        # Reshape back: [B, H, C, T] -> [B, C, H, T]
        residual = residual.reshape(B, H, C, T).permute(0, 2, 1, 3)

        # Residual connection
        corrected = degraded_latent + self.residual_scale * residual

        return corrected

    def get_pitch_embedding(self, pitch: int) -> torch.Tensor:
        """Get the learned embedding for a specific pitch (for visualization)."""
        return self.pitch_embeddings.weight[pitch]

    def interpolate_pitch_embedding(self, pitch_float: float) -> torch.Tensor:
        """Interpolate between pitch embeddings for microtonal adjustment."""
        low_pitch = int(pitch_float)
        high_pitch = low_pitch + 1
        alpha = pitch_float - low_pitch

        low_emb = self.pitch_embeddings.weight[low_pitch]
        high_emb = self.pitch_embeddings.weight[high_pitch]

        return (1 - alpha) * low_emb + alpha * high_emb


class RegisterAwareTranslatorDirect(nn.Module):
    """
    Non-residual version: output = f(input, condition), not input + f(input, condition).

    Better for large shifts where the timbre needs to change substantially.
    The direct architecture allows complete transformation of the input.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 7,
        num_pitches: int = 128,
        num_shifts: int = 25,
        pitch_embed_dim: int = 64,
        shift_embed_dim: int = 32,
    ):
        super().__init__()

        self.pitch_embeddings = nn.Embedding(num_pitches, pitch_embed_dim)
        self.shift_embeddings = nn.Embedding(num_shifts, shift_embed_dim)
        condition_dim = pitch_embed_dim + shift_embed_dim

        self.input_proj = nn.Sequential(
            nn.Conv1d(latent_channels, hidden_channels, 1),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            ResidualBlockConditioned(hidden_channels, condition_dim,
                                     kernel_size, dilation=2 ** i)
            for i in range(num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, latent_channels, 1),
        )

        # Small init (not zero - we want to learn transformation)
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)

    def forward(
        self,
        degraded_latent: torch.Tensor,
        target_pitch: torch.Tensor,
        shift_amount: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, T = degraded_latent.shape

        pitch_emb = self.pitch_embeddings(target_pitch.long().clamp(0, 127))
        shift_idx = (shift_amount + 12).long().clamp(0, 24)
        shift_emb = self.shift_embeddings(shift_idx)
        condition = torch.cat([pitch_emb, shift_emb], dim=-1)

        x = degraded_latent.permute(0, 2, 1, 3).reshape(B * H, C, T)
        condition_expanded = condition.unsqueeze(1).expand(-1, H, -1).reshape(B * H, -1)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, condition_expanded)
        output = self.output_proj(x)

        output = output.reshape(B, H, C, T).permute(0, 2, 1, 3)
        return output


class RegisterAwareTranslatorLarge(nn.Module):
    """
    Larger 2D version with full spatial attention.

    Uses 2D convolutions to better capture cross-frequency relationships
    important for formant correction.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 32,
        num_blocks: int = 8,
        num_pitches: int = 128,
        num_shifts: int = 25,
        pitch_embed_dim: int = 64,
        shift_embed_dim: int = 32,
    ):
        super().__init__()

        self.pitch_embeddings = nn.Embedding(num_pitches, pitch_embed_dim)
        self.shift_embeddings = nn.Embedding(num_shifts, shift_embed_dim)
        condition_dim = pitch_embed_dim + shift_embed_dim

        # Project condition to spatial modulation
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )

        self.input_proj = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
            ))

        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, 3, padding=1),
        )

        # Zero init for residual
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        degraded_latent: torch.Tensor,
        target_pitch: torch.Tensor,
        shift_amount: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, T = degraded_latent.shape

        # Get conditioning
        pitch_emb = self.pitch_embeddings(target_pitch.long().clamp(0, 127))
        shift_idx = (shift_amount + 12).long().clamp(0, 24)
        shift_emb = self.shift_embeddings(shift_idx)
        condition = torch.cat([pitch_emb, shift_emb], dim=-1)

        # Project condition to gamma, beta
        cond_proj = self.condition_proj(condition)
        gamma, beta = cond_proj.chunk(2, dim=-1)
        gamma = gamma.view(B, -1, 1, 1)
        beta = beta.view(B, -1, 1, 1)

        # Process
        x = self.input_proj(degraded_latent)

        # Apply conditioning
        x = x * (1 + gamma) + beta

        for block in self.blocks:
            x = x + block(x)

        residual = self.output_proj(x)
        return degraded_latent + self.residual_scale * residual


# ============================================================================
# Loss Functions
# ============================================================================


class TimbreLoss(nn.Module):
    """
    Multi-scale timbre matching in latent space.

    Uses multiple techniques to capture timbre:
    1. Global spectral envelope (mean over time)
    2. Spectral variance (dynamics)
    3. Multi-scale temporal pooling
    4. Gram matrix (frequency band correlations = timbre signature)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare timbre at multiple time scales.

        Args:
            output: [B, C, H, T] model output
            reference: [B, C, H, T] reference at target pitch
        """
        losses = []

        # 1. Global spectral envelope
        out_env = output.mean(dim=-1)
        ref_env = reference.mean(dim=-1)
        losses.append(F.l1_loss(out_env, ref_env))

        # 2. Spectral envelope variance (dynamics)
        out_var = output.var(dim=-1)
        ref_var = reference.var(dim=-1)
        losses.append(0.5 * F.l1_loss(out_var, ref_var))

        # 3. Multi-scale: compare at different temporal resolutions
        for pool_size in [4, 8, 16]:
            if output.shape[-1] >= pool_size:
                out_pooled = F.avg_pool1d(output.flatten(1, 2), pool_size)
                ref_pooled = F.avg_pool1d(reference.flatten(1, 2), pool_size)
                losses.append(0.3 * F.l1_loss(out_pooled, ref_pooled))

        # 4. Gram matrix (captures frequency band correlations = timbre signature)
        out_gram = self._gram_matrix(output)
        ref_gram = self._gram_matrix(reference)
        losses.append(0.5 * F.l1_loss(out_gram, ref_gram))

        return sum(losses) / len(losses)

    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Gram matrix captures style/timbre correlations."""
        B, C, H, T = x.shape
        features = x.reshape(B, C * H, T)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * T)


class ContentLoss(nn.Module):
    """
    Loss for preserving content (pitch, timing, dynamics).

    Compares in a way that's invariant to timbre changes - focuses on
    the temporal envelope and relative energy patterns.
    """

    def __init__(self):
        super().__init__()

    def temporal_envelope(self, latent: torch.Tensor) -> torch.Tensor:
        """Extract temporal envelope (energy over time)."""
        return (latent ** 2).mean(dim=(1, 2))  # [B, T]

    def forward(
        self,
        output: torch.Tensor,
        input_shifted: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare content/timing characteristics.

        Args:
            output: [B, C, H, T] model output
            input_shifted: [B, C, H, T] input (shifted, has artifacts but correct timing)
        """
        out_envelope = self.temporal_envelope(output)
        in_envelope = self.temporal_envelope(input_shifted)

        # Normalize envelopes
        out_envelope = out_envelope / (out_envelope.max(dim=-1, keepdim=True)[0] + 1e-8)
        in_envelope = in_envelope / (in_envelope.max(dim=-1, keepdim=True)[0] + 1e-8)

        return F.l1_loss(out_envelope, in_envelope)


class CombinedLoss(nn.Module):
    """
    Combined loss for register-aware pitch shift correction.

    Two modes:
    1. Artifact removal (loss_type=0): Full reconstruction loss
       - We have ground truth (original before double-shift)
       - Goal: remove artifacts, preserve exact content

    2. Register transfer (loss_type=1): Content + timbre loss
       - No exact ground truth (pitch actually changed)
       - Goal: preserve timing/dynamics, match timbre of target register
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        timbre_weight: float = 0.5,
        content_weight: float = 0.3,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.timbre_weight = timbre_weight
        self.content_weight = content_weight

        self.timbre_loss = TimbreLoss()
        self.content_loss = ContentLoss()

    def forward(
        self,
        output: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            output: [B, C, H, T] model output
            batch: Dictionary with 'target_latent', 'reference_latent',
                   'input_latent', 'loss_type', 'valid'

        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual losses for logging
        """
        loss_type = batch['loss_type']
        valid = batch['valid'].float()

        total_loss = torch.tensor(0.0, device=output.device)
        loss_dict = {}

        # Masks for different loss types
        artifact_mask = (loss_type == 0).float() * valid
        transfer_mask = (loss_type == 1).float() * valid

        # Artifact removal: full reconstruction
        if artifact_mask.sum() > 0:
            target = batch['target_latent']
            recon_loss = F.l1_loss(output, target, reduction='none')
            recon_loss = (recon_loss.mean(dim=(1, 2, 3)) * artifact_mask).sum()
            recon_loss = recon_loss / (artifact_mask.sum() + 1e-8)
            total_loss = total_loss + self.reconstruction_weight * recon_loss
            loss_dict['reconstruction'] = recon_loss.item()

        # Register transfer: content + timbre
        if transfer_mask.sum() > 0:
            input_latent = batch['input_latent']
            reference = batch.get('reference_latent', input_latent)

            # Content preservation (timing/dynamics)
            content = self.content_loss(output, input_latent)
            # Weight by mask proportion
            content_weighted = content * (transfer_mask.sum() / (len(transfer_mask) + 1e-8))
            total_loss = total_loss + self.content_weight * content_weighted
            loss_dict['content'] = content.item()

            # Timbre matching to reference
            timbre = self.timbre_loss(output, reference)
            timbre_weighted = timbre * (transfer_mask.sum() / (len(transfer_mask) + 1e-8))
            total_loss = total_loss + self.timbre_weight * timbre_weighted
            loss_dict['timbre'] = timbre.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# ============================================================================
# Student Model (for VST export)
# ============================================================================


class PitchShiftStudentModel(nn.Module):
    """
    Lightweight student model for VST deployment.

    Distilled from the teacher (RegisterAwareTranslator + DCAE).
    Operates directly on mel spectrograms for efficiency.
    """

    def __init__(
        self,
        n_mels: int = 128,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        num_pitches: int = 128,
        num_shifts: int = 25,
    ):
        super().__init__()

        # Lightweight pitch/shift conditioning
        self.pitch_embed = nn.Embedding(num_pitches, 32)
        self.shift_embed = nn.Embedding(num_shifts, 16)

        # Efficient 2D UNet-style architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.SiLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim + 48, hidden_dim, 3, padding=1),  # +48 for conditioning
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

    def forward(
        self,
        mel: torch.Tensor,
        target_pitch: torch.Tensor,
        shift_amount: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mel: [B, 1, n_mels, T] input mel spectrogram
            target_pitch: [B] target MIDI pitch
            shift_amount: [B] shift in semitones

        Returns:
            corrected mel: [B, 1, n_mels, T]
        """
        B, _, H, W = mel.shape

        # Get conditioning
        pitch_emb = self.pitch_embed(target_pitch.long().clamp(0, 127))
        shift_idx = (shift_amount + 12).long().clamp(0, 24)
        shift_emb = self.shift_embed(shift_idx)
        cond = torch.cat([pitch_emb, shift_emb], dim=-1)  # [B, 48]

        # Encode
        x = self.encoder(mel)  # [B, hidden, H//2, W//2]

        # Add conditioning
        cond_spatial = cond.view(B, 48, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, cond_spatial], dim=1)

        # Process
        x = self.bottleneck(x)

        # Decode
        x = self.decoder(x)

        # Crop to match input size
        x = x[:, :, :H, :W]

        return mel + x  # Residual


if __name__ == "__main__":
    # Test models
    print("Testing RegisterAwareTranslator...")
    batch = torch.randn(2, 8, 16, 128)  # [B, C, H, T]
    target_pitch = torch.tensor([60, 72])  # C4, C5
    shift_amount = torch.tensor([5.0, -3.0])  # Shift amounts

    model = RegisterAwareTranslator()
    out = model(batch, target_pitch, shift_amount)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting RegisterAwareTranslatorDirect...")
    model_direct = RegisterAwareTranslatorDirect()
    out = model_direct(batch, target_pitch, shift_amount)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_direct.parameters()):,}")

    print("\nTesting RegisterAwareTranslatorLarge...")
    model_large = RegisterAwareTranslatorLarge()
    out = model_large(batch, target_pitch, shift_amount)
    print(f"  Input: {batch.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_large.parameters()):,}")

    print("\nTesting CombinedLoss...")
    loss_fn = CombinedLoss()
    fake_batch = {
        'target_latent': torch.randn(2, 8, 16, 128),
        'reference_latent': torch.randn(2, 8, 16, 128),
        'input_latent': torch.randn(2, 8, 16, 128),
        'loss_type': torch.tensor([0, 1]),  # Artifact, Register
        'valid': torch.tensor([True, True]),
    }
    loss, loss_dict = loss_fn(out, fake_batch)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Components: {loss_dict}")

    print("\nTesting PitchShiftStudentModel...")
    mel = torch.randn(2, 1, 128, 256)  # [B, 1, n_mels, T]
    student = PitchShiftStudentModel()
    out = student(mel, target_pitch, shift_amount)
    print(f"  Input: {mel.shape}, Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in student.parameters()):,}")
