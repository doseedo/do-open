"""Latent Demucs runtime — single forward pass produces 4 stem latents
(drums, bass, vocals, other) directly from waveform, skipping
htdemucs + per-stem vae.encode entirely.

~5× faster than the legacy htdemucs+encode chain (107 ms vs 514 ms for
10 s of stereo @ 48k on A100), and the output is already in the same
VAE latent space the rest of the studio pipeline expects."""
from .runtime import LatentDemucsRuntime, latent_demucs_separate, STEM_NAMES
