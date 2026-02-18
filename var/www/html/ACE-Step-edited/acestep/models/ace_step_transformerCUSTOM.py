# =============================================================================
# FILE: ace_step_transformer.py
# This is your modified version of acestep_transformer.py
# =============================================================================
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin

# --- INTERNAL TRANSFORMER COMPONENTS (UNCHANGED) ---

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def _apply_rotary_emb(x, rotary_pos_emb):
    """
    Apply rotary positional embedding to input tensor.
    x: [B, num_heads, seq_len, head_dim]
    rotary_pos_emb: tuple of (cos, sin) tensors
    """
    cos, sin = rotary_pos_emb
    head_dim = x.shape[-1]              # Hd
    half = head_dim // 2                # 32

    # Ensure cos/sin have the right shape for broadcasting
    # cos/sin should be broadcastable to [B, num_heads, seq_len, half]
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    
    # Make cos/sin match the half-dimension
    if cos.shape[-1] != half:
        cos = cos[..., :half]
        sin = sin[..., :half]

    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ADD THIS NEW, STABLE ATTENTION CLASS
class StandardAttention(nn.Module):
    def __init__(self, dim, num_attention_heads=8, attention_head_dim=64):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        inner_dim = self.head_dim * self.num_heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, hidden_states, encoder_hidden_states=None, rotary_freqs_cis=None, encoder_attention_mask=None):
        q = self.to_q(hidden_states)
        
        # Use encoder_hidden_states for k and v if provided (for cross-attention)
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.to_k(kv_src)
        v = self.to_v(kv_src)

        # Reshape for multi-head attention
        B, N, _ = q.shape
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = k.shape[1]
        k = k.view(B, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings only for self-attention
        if rotary_freqs_cis is not None and encoder_hidden_states is None:
            # Make rotary embedding broadcast-safe
            cos, sin = rotary_freqs_cis
            if cos.ndim == 2:  # [seq_len, head_dim//2]
                cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
                sin = sin.unsqueeze(0).unsqueeze(0)
            elif cos.ndim == 3:  # [B, seq_len, head_dim//2]
                cos = cos.unsqueeze(1)  # [B, 1, seq_len, head_dim//2]
                sin = sin.unsqueeze(1)
            # Apply rotary to q and k
            q = _apply_rotary_emb(q, (cos, sin))
            k = _apply_rotary_emb(k, (cos, sin))
        
        # Handle attention mask for cross-attention
        attn_mask = None
        if encoder_attention_mask is not None and encoder_hidden_states is not None:
            # Normalize mask to boolean (True=keep, False=mask out)
            m = encoder_attention_mask
            if m.dtype != torch.bool:
                m = m > 0.5
                
            # encoder_attention_mask: [B, encoder_seq_len] -> need [B, num_heads, query_seq_len, encoder_seq_len]
            attn_mask = m.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, encoder_seq_len]
            attn_mask = attn_mask.expand(B, self.num_heads, N, kv_seq_len)  # [B, num_heads, N, kv_seq_len]
            # Use finite negative value instead of -inf for bf16 stability
            NEG = -1e4
            attn_mask = torch.where(attn_mask, torch.tensor(0.0, device=m.device), torch.tensor(NEG, device=m.device))
            
        # Use PyTorch's optimized attention function with FP32 for stability
        out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=attn_mask)
        out = out.to(q.dtype)  # cast back to original precision
        
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.to_out(out)

class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act_fn = nn.SiLU()
        self.conv3 = nn.Conv1d(hidden_features, out_features, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act_fn(x)
        x = self.conv3(x)
        return x.permute(0, 2, 1)

class LinearTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0, add_cross_attention=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = StandardAttention(dim, num_attention_heads, attention_head_dim) # <-- CHANGE THIS
        self.add_cross_attention = add_cross_attention
        if add_cross_attention:
            self.norm2 = nn.LayerNorm(dim)
            self.attn2 = StandardAttention(dim, num_attention_heads, attention_head_dim) # <-- AND THIS
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, int(dim * mlp_ratio))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
    
    # --- REPLACE THE BROKEN forward METHOD WITH THIS CORRECT ONE ---
    def forward(self, hidden_states, temb, rotary_freqs_cis, encoder_hidden_states=None, encoder_attention_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(temb).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1), gate_msa.unsqueeze(1)
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)

        # self-attn (rotary)
        x = t2i_modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        attn_out = self.attn1(x, rotary_freqs_cis=rotary_freqs_cis)
        hidden_states = hidden_states + gate_msa * attn_out

        # cross-attn (this part will be skipped if add_cross_attention=False)
        if self.add_cross_attention and encoder_hidden_states is not None:
            x2 = self.norm2(hidden_states)
            attn_out = self.attn2(
                x2,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            # Full cross-attention residual connection (un-capped for CA pivot)
            hidden_states = hidden_states + attn_out

        # FFN
        x = t2i_modulate(self.norm3(hidden_states), shift_mlp, scale_mlp)
        ffn_out = self.ffn(x)
        hidden_states = hidden_states + gate_mlp * ffn_out
        return hidden_states




class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))

class T2IFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size=[16, 1], out_channels=8):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def unpatchfy(self, hidden_states: torch.Tensor, out_channels, patch_size):
        batch, seq_len, _ = hidden_states.shape
        h, w = 1, seq_len
        ph, pw = patch_size
        x = hidden_states.reshape(batch, h, w, ph, pw, out_channels)
        x = torch.einsum('bhwpqc->bchpwq', x)
        return x.reshape(batch, out_channels, h * ph, w * pw)

    def forward(self, x, t, output_length, out_channels, patch_size):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        y = self.unpatchfy(x, out_channels, patch_size)
        if y.shape[-1] > output_length:
            y = y[..., :output_length]
        elif y.shape[-1] < output_length:
            y = F.pad(y, (0, output_length - y.shape[-1]))
        return y

class PatchEmbed(nn.Module):
    def __init__(self, height=16, width=4096, patch_size=(16, 1), in_channels=8, embed_dim=1536, bias=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, latent):
        B, C, H, W = latent.shape
        pw = self.proj.kernel_size[1]
        if (W % pw) != 0:
            pad_right = pw - (W % pw)
            latent = F.pad(latent, (0, pad_right, 0, 0))
        return self.proj(latent).flatten(2).transpose(1, 2)

@dataclass
class PerformerTransformerOutput(BaseOutput):
    sample: torch.FloatTensor

# --- PerformerTransformer MODEL ---

class PerformerTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        num_layers: int = 28,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        mlp_ratio: float = 4.0,
        patch_size: List[int] = [16, 8],
        max_width: int = 4096,
        # The text_embedding_dim is now the dimension of our unified conditioning signal
        text_embedding_dim: int = 768, 
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        
        self.proj_in = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=self.inner_dim)
        self.final_layer = T2IFinalLayer(self.inner_dim, patch_size=patch_size, out_channels=out_channels)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)
        self.rotary_emb = Qwen2RotaryEmbedding(dim=self.attention_head_dim, max_position_embeddings=max_width // patch_size[1])
        
        self.cond_ln   = nn.LayerNorm(self.inner_dim)
        self.cond_gain = nn.Parameter(torch.tensor(0.1))

        self.cond_proj = (
            nn.Linear(self.config.text_embedding_dim, self.inner_dim)
            if self.config.text_embedding_dim != self.inner_dim else nn.Identity()
        )

        self.transformer_blocks = nn.ModuleList([
            LinearTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                add_cross_attention=True,   
            ) for _ in range(num_layers)
        ])
        self.gradient_checkpointing = True

    def _get_t_emb(self, timestep):
        t_emb = self.time_proj(timestep).to(dtype=self.dtype)
        return self.timestep_embedder(t_emb)


    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        output_length = hidden_states.shape[-1]

        # 1. Patchify input latents
        x = self.proj_in(hidden_states)
        B, N, D = x.shape

        # 2. Prepare conditioning for Cross-Attention
        enc_tokens = encoder_hidden_states
        global_cond = enc_tokens[:, :2, :].mean(dim=1, keepdim=True)
        frame_cond = enc_tokens[:, 2:, :]

        # The conditioning context is now the full sequence of global + frame tokens
        # It does NOT need to be the same length as the latents
        cond_ctx = self.cond_proj(global_cond.expand(-1, frame_cond.shape[1], -1) + frame_cond)
        cond_ctx = self.cond_ln(cond_ctx)
        # Apply learnable conditioning gate
        cond_ctx = cond_ctx * torch.sigmoid(self.cond_gain)
        
        # Fix mask-context mismatch: adjust mask to match cond_ctx length
        mask_ctx = encoder_attention_mask
        if mask_ctx is not None:
            # mask_ctx: [B, 2+T] -> [B, T] to match frame_cond/cond_ctx
            mask_ctx = mask_ctx[:, 2:2 + frame_cond.shape[1]]
            if mask_ctx.dtype != torch.bool:
                mask_ctx = mask_ctx > 0.5

        # 3. Prepare timestep and rotary embeddings
        t_emb = self._get_t_emb(timestep)
        rotary_embeds = self.rotary_emb(x, seq_len=N)

        # 4. Pass through Transformer blocks using Cross-Attention
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing during training for memory efficiency
                x = checkpoint(
                    block,
                    x,
                    t_emb,
                    rotary_embeds,
                    cond_ctx,
                    mask_ctx,
                    use_reentrant=False
                )
            else:
                x = block(
                    hidden_states=x,
                    temb=t_emb,
                    rotary_freqs_cis=rotary_embeds,
                    encoder_hidden_states=cond_ctx, # <-- Pass conditioning here
                    encoder_attention_mask=mask_ctx,  # <-- Pass fixed mask
                )

        # 5. Final layer projects back to the latent space shape
        y = self.final_layer(
            x,
            t=t_emb,
            output_length=output_length,
            out_channels=self.config.out_channels,
            patch_size=self.config.patch_size,
        )

        return PerformerTransformerOutput(sample=y)
