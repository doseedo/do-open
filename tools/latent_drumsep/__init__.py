"""Latent-domain drum sub-separation.

Goal: take a drum-only latent [T, 64] (e.g. the `drums` output of
LatentDemucsRuntime) and split it into 6 sub-stem latents
(kick / snare / toms / hh / ride / crash) — entirely in latent space,
no VAE round-trip.

Trained as a small student against teacher data generated offline by
running MDX23C-DrumSep on real drum audio and VAE-encoding both the
input mix and the per-stem outputs.
"""
STEMS = ("kick", "snare", "toms", "hh", "ride", "crash")
