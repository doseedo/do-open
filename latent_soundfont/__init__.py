"""Latent soundfont — pre-encoded note latents per soundfont so MIDI can
be synthesized directly into Oobleck VAE space, no fluidsynth+encode at
inference time."""
from .build import build_latent_soundfont, build_all
from .synth import latent_synthesize_midi, load_latent_soundfont
