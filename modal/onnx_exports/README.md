# ONNX model exports for Modal/frontend

Models in this directory are produced by the Modal backend and served
from the frontend GCS bucket. The binary \`.onnx\` + \`.onnx.data\`
files are gitignored — canonical locations:

  gs://doseedo-production/stemphonic/onnx/   (source of truth)
  gs://doseedo-frontend-static/static/models/   (served to browser)

## oobleck_encoder
Stereo 48 kHz waveform → VAE latents.
  audio   [1, 2, N]
  latent  [1, 64, N / 1920]

Used by src/services/latentEncoder.js (client-side replacement for
/api/encode-audio-latent). Exported from stemphonic handler.vae.encode.

