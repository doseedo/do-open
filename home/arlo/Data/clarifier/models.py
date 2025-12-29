# /home/arlo/Data/clarifier/models.py
# Apache 2.0
# InstrumentClarifier: Post-generation latent correction model

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import GroupConfig


class ResidualBlock1D(nn.Module):
    """1D dilated residual block similar to WaveNet/FormantCorrector style."""
    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual


class InstrumentClarifier(nn.Module):
    """
    Post-generation latent clarifier.

    Takes DCAE latents from the performer model (which may have artifacts,
    wrong timbre, or mixed instrument characteristics) and outputs clarified
    latents that sound like the target instrument.

    Input:
        - generated_latent [B, 8, 16, T]
        - group_id [B]
        - subgroup_id [B]
    Output:
        - clarified_latent [B, 8, 16, T]
    """

    def __init__(
        self,
        group_vocab: int = 6,
        subgroup_vocab: int = 13,
        latent_channels: int = 8,
        latent_height: int = 16,
        hidden_dim: int = 256,
        inst_emb_dim: int = 64,
        num_blocks: int = 6,
        dilations: tuple = (1, 2, 4, 8, 16, 32),
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.flat_dim = latent_channels * latent_height  # 128

        # Instrument conditioning embeddings
        self.group_emb = nn.Embedding(group_vocab, inst_emb_dim)
        self.subgroup_emb = nn.Embedding(subgroup_vocab, inst_emb_dim)
        self.inst_proj = nn.Linear(inst_emb_dim * 2, inst_emb_dim)  # fuse group + subgroup

        # Input projection: flattened latent + instrument conditioning
        self.input_proj = nn.Conv1d(self.flat_dim + inst_emb_dim, hidden_dim, kernel_size=1)

        # Main processing blocks with dilations
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, dilation=d)
            for d in dilations[:num_blocks]
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, self.flat_dim, kernel_size=1)

        # Residual connection with learnable scales
        # Start with strong input preservation, weak correction
        self.input_scale = nn.Parameter(torch.tensor(0.8))
        self.output_scale = nn.Parameter(torch.tensor(0.2))

        # Initialize output projection to near-zero for stable training start
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        latent: torch.Tensor,
        group_id: torch.Tensor,
        subgroup_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, 8, 16, T] DCAE latent from performer model
            group_id: [B] instrument group indices
            subgroup_id: [B] instrument subgroup indices

        Returns:
            clarified: [B, 8, 16, T] clarified latent
        """
        B, C, H, T = latent.shape
        assert C == self.latent_channels and H == self.latent_height, \
            f"Expected latent shape [B, {self.latent_channels}, {self.latent_height}, T], got {latent.shape}"

        # Flatten spatial dims: [B, 8, 16, T] -> [B, 128, T]
        x = latent.view(B, C * H, T)

        # Instrument conditioning
        g_emb = self.group_emb(group_id)      # [B, inst_emb_dim]
        sg_emb = self.subgroup_emb(subgroup_id)  # [B, inst_emb_dim]
        inst = self.inst_proj(torch.cat([g_emb, sg_emb], dim=-1))  # [B, inst_emb_dim]
        inst_expand = inst.unsqueeze(-1).expand(-1, -1, T)  # [B, inst_emb_dim, T]

        # Concat input with conditioning and project
        x_cond = torch.cat([x, inst_expand], dim=1)  # [B, 128+inst_emb_dim, T]
        h = self.input_proj(x_cond)  # [B, hidden_dim, T]

        # Process through residual blocks
        for block in self.blocks:
            h = block(h)

        # Output projection
        out = self.output_proj(h)  # [B, 128, T]
        out = out.view(B, C, H, T)  # [B, 8, 16, T]

        # Residual output with learnable scaling
        clarified = self.input_scale * latent + self.output_scale * out

        return clarified


class InstrumentClarifierLarge(InstrumentClarifier):
    """Larger version with more capacity for harder corrections."""

    def __init__(
        self,
        group_vocab: int = 6,
        subgroup_vocab: int = 13,
        latent_channels: int = 8,
        latent_height: int = 16,
    ):
        super().__init__(
            group_vocab=group_vocab,
            subgroup_vocab=subgroup_vocab,
            latent_channels=latent_channels,
            latent_height=latent_height,
            hidden_dim=512,
            inst_emb_dim=128,
            num_blocks=8,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        )


class SimpleInstrumentClassifier(nn.Module):
    """
    Auxiliary classifier for instrument identification from latents.
    Used as a pretrain target and aux loss for clarifier training.
    """

    def __init__(
        self,
        group_vocab: int = 6,
        subgroup_vocab: int = 13,
        latent_channels: int = 8,
        latent_height: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.flat_dim = latent_channels * latent_height

        # Temporal convs for local pattern extraction
        self.convs = nn.Sequential(
            nn.Conv1d(self.flat_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # Global pooling + classifiers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.group_head = nn.Linear(hidden_dim, group_vocab)
        self.subgroup_head = nn.Linear(hidden_dim, subgroup_vocab)

    def forward(self, latent: torch.Tensor):
        """
        Args:
            latent: [B, 8, 16, T]

        Returns:
            group_logits: [B, group_vocab]
            subgroup_logits: [B, subgroup_vocab]
        """
        B, C, H, T = latent.shape
        x = latent.view(B, C * H, T)  # [B, 128, T]

        x = self.convs(x)  # [B, hidden_dim, T]
        x = self.pool(x).squeeze(-1)  # [B, hidden_dim]

        group_logits = self.group_head(x)
        subgroup_logits = self.subgroup_head(x)

        return group_logits, subgroup_logits


class GroupClarifier(nn.Module):
    """
    Clarifier for a SINGLE instrument group (e.g., brass only).

    This is a simpler model that only distinguishes between subgroups
    within a single group. More efficient for group-specific training.

    Input:
        - generated_latent [B, 8, 16, T]
        - subgroup_id [B] (local to this group, e.g., 0=trumpet, 1=trombone, etc.)
    Output:
        - clarified_latent [B, 8, 16, T]
    """

    def __init__(
        self,
        subgroup_vocab: int = 5,  # Number of subgroups in this group
        latent_channels: int = 8,
        latent_height: int = 16,
        hidden_dim: int = 256,
        inst_emb_dim: int = 64,
        num_blocks: int = 6,
        dilations: tuple = (1, 2, 4, 8, 16, 32),
        group_name: str = "unknown",  # For metadata
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.flat_dim = latent_channels * latent_height  # 128
        self.group_name = group_name
        self.subgroup_vocab = subgroup_vocab

        # Only subgroup embedding (no group embedding needed)
        self.subgroup_emb = nn.Embedding(subgroup_vocab, inst_emb_dim)

        # Input projection: flattened latent + instrument conditioning
        self.input_proj = nn.Conv1d(self.flat_dim + inst_emb_dim, hidden_dim, kernel_size=1)

        # Main processing blocks with dilations
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, dilation=d)
            for d in dilations[:num_blocks]
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, self.flat_dim, kernel_size=1)

        # Residual connection with learnable scales
        self.input_scale = nn.Parameter(torch.tensor(0.8))
        self.output_scale = nn.Parameter(torch.tensor(0.2))

        # Initialize output projection to near-zero for stable training start
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        latent: torch.Tensor,
        subgroup_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, 8, 16, T] DCAE latent from performer model
            subgroup_id: [B] local subgroup indices within this group

        Returns:
            clarified: [B, 8, 16, T] clarified latent
        """
        B, C, H, T = latent.shape
        assert C == self.latent_channels and H == self.latent_height

        # Flatten spatial dims
        x = latent.view(B, C * H, T)

        # Subgroup conditioning
        sg_emb = self.subgroup_emb(subgroup_id)  # [B, inst_emb_dim]
        inst_expand = sg_emb.unsqueeze(-1).expand(-1, -1, T)  # [B, inst_emb_dim, T]

        # Concat and process
        x_cond = torch.cat([x, inst_expand], dim=1)
        h = self.input_proj(x_cond)

        for block in self.blocks:
            h = block(h)

        out = self.output_proj(h)
        out = out.view(B, C, H, T)

        clarified = self.input_scale * latent + self.output_scale * out
        return clarified


class GroupSubgroupClassifier(nn.Module):
    """
    Classifier for subgroups within a single group.
    Used as auxiliary loss for group-specific training.
    """

    def __init__(
        self,
        subgroup_vocab: int = 5,
        latent_channels: int = 8,
        latent_height: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.flat_dim = latent_channels * latent_height

        self.convs = nn.Sequential(
            nn.Conv1d(self.flat_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, subgroup_vocab)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, 8, 16, T]

        Returns:
            logits: [B, subgroup_vocab]
        """
        B, C, H, T = latent.shape
        x = latent.view(B, C * H, T)
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# Factory functions for creating models from configs

def create_group_clarifier(
    group_config: "GroupConfig",
    hidden_dim: Optional[int] = None,
    num_blocks: Optional[int] = None,
    inst_emb_dim: Optional[int] = None,
) -> GroupClarifier:
    """
    Create a GroupClarifier from a GroupConfig.

    Args:
        group_config: Configuration object for the group
        hidden_dim: Override hidden dimension (default from config)
        num_blocks: Override number of blocks (default 6)
        inst_emb_dim: Override embedding dimension (default from config)

    Returns:
        Configured GroupClarifier model
    """
    return GroupClarifier(
        subgroup_vocab=group_config.num_subgroups,
        hidden_dim=hidden_dim or group_config.hidden_dim,
        num_blocks=num_blocks or group_config.num_blocks,
        inst_emb_dim=inst_emb_dim or group_config.inst_emb_dim,
        group_name=group_config.name,
    )


def create_group_classifier(
    group_config: "GroupConfig",
    hidden_dim: int = 256,
) -> GroupSubgroupClassifier:
    """Create a subgroup classifier for a specific group."""
    return GroupSubgroupClassifier(
        subgroup_vocab=group_config.num_subgroups,
        hidden_dim=hidden_dim,
    )


if __name__ == "__main__":
    # Quick test
    print("=== Testing InstrumentClarifier (all groups) ===")
    model = InstrumentClarifier()

    B, T = 2, 256
    latent = torch.randn(B, 8, 16, T)
    group_id = torch.randint(0, 6, (B,))
    subgroup_id = torch.randint(0, 13, (B,))

    out = model(latent, group_id, subgroup_id)
    print(f"Input shape: {latent.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Test classifier
    classifier = SimpleInstrumentClassifier()
    g_logits, sg_logits = classifier(latent)
    print(f"Group logits: {g_logits.shape}, Subgroup logits: {sg_logits.shape}")

    # Test group-specific model
    print("\n=== Testing GroupClarifier (brass only) ===")
    from config import BRASS_CONFIG

    brass_model = create_group_clarifier(BRASS_CONFIG)
    brass_subgroup_id = torch.randint(0, BRASS_CONFIG.num_subgroups, (B,))

    out = brass_model(latent, brass_subgroup_id)
    print(f"Brass clarifier output: {out.shape}")
    print(f"Brass clarifier params: {sum(p.numel() for p in brass_model.parameters()):,}")
    print(f"Subgroups: {BRASS_CONFIG.subgroups}")

    # Test group classifier
    brass_classifier = create_group_classifier(BRASS_CONFIG)
    logits = brass_classifier(latent)
    print(f"Brass classifier logits: {logits.shape}")
