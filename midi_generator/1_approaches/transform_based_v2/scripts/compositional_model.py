#!/usr/bin/env python3
"""
Transformer Model for Compositional Pattern Generation
=======================================================

A small GPT-style decoder-only transformer for generating
interval+COMPOSE sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CompositionalTransformer(nn.Module):
    """
    Decoder-only transformer for compositional pattern generation.

    Architecture:
    - Token embedding + positional encoding
    - N transformer decoder layers
    - Output projection to vocabulary

    Input: sequence of token IDs (intervals, COMPOSE, BOS, EOS)
    Output: logits over vocabulary for next token prediction
    """

    def __init__(
        self,
        vocab_size: int = 242,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
        pad_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,}")

    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] 1 for real tokens, 0 for padding
            labels: [batch, seq_len] target token IDs (-100 = ignore)

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar if labels provided
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.pos_encoding(x)

        # Causal mask
        causal_mask = self._generate_square_subsequent_mask(seq_len, device)

        # Padding mask (True = ignore)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        # Decoder (self-attention only, no encoder)
        # Using memory=x as a trick for decoder-only: we just need causal self-attention
        x = self.decoder(
            tgt=x,
            memory=torch.zeros(batch_size, 1, self.d_model, device=device),  # Dummy memory
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            prompt: [1, seq_len] initial token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling threshold
            eos_id: end-of-sequence token ID

        Returns:
            generated: [1, seq_len + generated_len] full sequence
        """
        self.eval()
        device = prompt.device

        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Get logits for last position
            logits, _ = self.forward(generated)
            logits = logits[:, -1, :] / temperature  # [1, vocab_size]

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS
            if next_token.item() == eos_id:
                break

        return generated


def create_model(vocab_size: int = 242, **kwargs) -> CompositionalTransformer:
    """Create model with default settings."""
    return CompositionalTransformer(
        vocab_size=vocab_size,
        d_model=kwargs.get('d_model', 256),
        n_heads=kwargs.get('n_heads', 8),
        n_layers=kwargs.get('n_layers', 6),
        d_ff=kwargs.get('d_ff', 1024),
        dropout=kwargs.get('dropout', 0.1),
        max_len=kwargs.get('max_len', 2048),
        pad_id=kwargs.get('pad_id', 0),
    )


if __name__ == '__main__':
    # Test the model
    model = create_model(vocab_size=242)

    # Test forward pass
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 242, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 242, (batch_size, seq_len))

    logits, loss = model(input_ids, attention_mask, labels)
    print(f"\nForward pass:")
    print(f"  Input: {input_ids.shape}")
    print(f"  Output: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test generation
    prompt = torch.tensor([[1]])  # BOS token
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\nGeneration:")
    print(f"  Prompt: {prompt.shape}")
    print(f"  Generated: {generated.shape}")
    print(f"  Tokens: {generated[0].tolist()}")
