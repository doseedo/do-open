#!/usr/bin/env python3
"""
Sparse Synthesizer API - FastAPI backend for controllable audio synthesis.

Maps sparse interpretable parameters to DCAE latents:
  sparse_params → z latent → mel → audio

Based on discovered latent structure:
- Dims 48-63: Strong spectral control (brightness, energy bands)
- Dims 64-127: Temporal/amplitude control (attack, sustain, release)
- Dims 0-47: Fine detail/residual
"""

import torch
import torch.nn.functional as F
import numpy as np
import io
import base64
import sys
import os
from typing import Dict, Optional
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

# ============================================================================
# DISCOVERED LATENT MAPPINGS
# ============================================================================

# Key dimension ranges from atom discovery
DIM_SPECTRAL_BRIGHT = list(range(59, 64))   # Dims 59-63: brightness/centroid
DIM_SPECTRAL_LOW = list(range(48, 52))      # Dims 48-51: low frequencies
DIM_SPECTRAL_MID = list(range(52, 56))      # Dims 52-55: mid frequencies
DIM_SPECTRAL_HIGH = list(range(56, 59))     # Dims 56-58: high frequencies
DIM_TEMPORAL = list(range(64, 80))          # Dims 64-79: temporal shape
DIM_STEREO = list(range(80, 96))            # Dims 80-95: stereo/spatial
DIM_DETAIL = list(range(0, 48))             # Dims 0-47: fine detail

# Default scales (from quadratic fitting analysis)
DEFAULT_SCALE = 0.5
DEFAULT_OFFSET = 0.0


class SparseParams(BaseModel):
    """Sparse controllable parameters for synthesis."""
    brightness: float = 0.5        # 0-1, spectral centroid
    low_energy: float = 0.5        # 0-1, bass content
    mid_energy: float = 0.5        # 0-1, mid frequencies
    high_energy: float = 0.5       # 0-1, treble content
    attack: float = 0.3            # 0-1, onset sharpness
    sustain: float = 0.7           # 0-1, body duration
    release: float = 0.3           # 0-1, decay length
    stereo_width: float = 0.5      # 0-1, spatial spread
    detail: float = 0.5            # 0-1, fine texture amount
    duration_frames: int = 32      # Number of z frames (~3 seconds at 32)


class SynthRequest(BaseModel):
    """Request for audio synthesis."""
    params: SparseParams
    base_z: Optional[str] = None   # Base64 encoded base latent (optional)
    output_format: str = "wav"     # wav or mp3


class SynthResponse(BaseModel):
    """Response with generated audio."""
    audio_b64: str                 # Base64 encoded audio
    sample_rate: int = 44100
    z_b64: str                     # Base64 encoded z latent (for editing)


# ============================================================================
# MODEL LOADING
# ============================================================================

class SparseSynthesizer:
    """Main synthesizer class managing DCAE model and parameter mapping."""

    def __init__(self, device='cuda'):
        self.device = device
        self.dcae = None
        self.scale_factor = 0.1786
        self.shift_factor = -1.9091

    def load_models(self):
        """Load DCAE model."""
        print("Loading DCAE...")
        DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
        VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

        self.dcae = MusicDCAE(
            dcae_checkpoint_path=DCAE_PATH,
            vocoder_checkpoint_path=VOCODER_PATH,
        )
        self.dcae.dcae.to(self.device).eval()
        self.dcae.vocoder.to(self.device).eval()
        print("Models loaded!")

    def sparse_to_z(self, params: SparseParams, base_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert sparse parameters to z latent.

        Based on discovered quadratic relationships:
        feature = a*z² + b*z + c

        We invert this to: z = f(desired_feature)
        """
        T = params.duration_frames

        # Start with base z or zeros
        if base_z is not None:
            z = base_z.clone()
            # Adjust T if needed
            if z.shape[-1] != T:
                z = F.interpolate(z, size=T, mode='linear', align_corners=False)
        else:
            # Initialize with small random values (natural distribution)
            z = torch.randn(1, 8, 16, T, device=self.device) * 0.05

        # Flatten to [B, 128, T] for easier manipulation
        z_flat = z.reshape(1, 128, T)

        # Apply sparse parameter mappings
        # Each param maps to specific dimension ranges with learned scales

        # Brightness → dims 59-63 (strongest influence on centroid)
        brightness_value = (params.brightness - 0.5) * 2.0  # Scale to [-1, 1]
        for dim in DIM_SPECTRAL_BRIGHT:
            z_flat[:, dim, :] += brightness_value * 0.8

        # Low energy → dims 48-51
        low_value = (params.low_energy - 0.5) * 2.0
        for dim in DIM_SPECTRAL_LOW:
            z_flat[:, dim, :] += low_value * 0.6

        # Mid energy → dims 52-55
        mid_value = (params.mid_energy - 0.5) * 2.0
        for dim in DIM_SPECTRAL_MID:
            z_flat[:, dim, :] += mid_value * 0.6

        # High energy → dims 56-58
        high_value = (params.high_energy - 0.5) * 2.0
        for dim in DIM_SPECTRAL_HIGH:
            z_flat[:, dim, :] += high_value * 0.5

        # Temporal envelope (attack/sustain/release) → dims 64-79
        # Create envelope shape
        t_axis = torch.linspace(0, 1, T, device=self.device)

        # Simple ADSR-like envelope
        attack_point = params.attack * 0.3  # Where attack ends (0-30% of duration)
        release_point = 1.0 - params.release * 0.3  # Where release starts

        envelope = torch.ones(T, device=self.device)
        for t in range(T):
            t_norm = t / T
            if t_norm < attack_point:
                envelope[t] = t_norm / (attack_point + 1e-6)
            elif t_norm > release_point:
                envelope[t] = (1.0 - t_norm) / (1.0 - release_point + 1e-6)
            else:
                envelope[t] = params.sustain

        # Apply envelope to temporal dimensions
        for dim in DIM_TEMPORAL:
            z_flat[:, dim, :] *= envelope.unsqueeze(0)

        # Stereo width → dims 80-95
        stereo_value = (params.stereo_width - 0.5) * 2.0
        for dim in DIM_STEREO:
            z_flat[:, dim, :] += stereo_value * 0.3

        # Detail → dims 0-47 (fine texture)
        detail_scale = params.detail
        z_flat[:, :48, :] *= detail_scale

        # Reshape back to [B, 8, 16, T]
        z = z_flat.reshape(1, 8, 16, T)

        return z

    def z_to_audio(self, z: torch.Tensor) -> np.ndarray:
        """Decode z latent to audio using DCAE + vocoder."""
        with torch.no_grad():
            # Denormalize
            z_denorm = z / self.scale_factor + self.shift_factor

            # Decode to mel
            mel = self.dcae.dcae.decoder(z_denorm)
            mel = mel.mean(dim=1)  # [B, 128, T_mel]

            # Scale mel for vocoder
            mel_scaled = mel * 0.5 + 0.5
            mel_scaled = mel_scaled * (self.dcae.max_mel_value - self.dcae.min_mel_value) + self.dcae.min_mel_value

            # Vocoder to audio
            audio = self.dcae.vocoder.decode(mel_scaled).squeeze()

            # Normalize
            audio = audio / (audio.abs().max() + 1e-8) * 0.9

            return audio.cpu().numpy()

    def synthesize(self, params: SparseParams, base_z: Optional[torch.Tensor] = None) -> tuple:
        """Full synthesis pipeline: params → z → audio."""
        z = self.sparse_to_z(params, base_z)
        audio = self.z_to_audio(z)
        return audio, z


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Sparse Synthesizer API",
    description="Controllable audio synthesis with interpretable parameters",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global synthesizer instance
synth: Optional[SparseSynthesizer] = None


@app.on_event("startup")
async def startup():
    global synth
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    synth = SparseSynthesizer(device=device)
    synth.load_models()


@app.get("/")
async def root():
    return {"status": "ok", "message": "Sparse Synthesizer API"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": synth is not None}


@app.post("/synthesize", response_model=SynthResponse)
async def synthesize(request: SynthRequest):
    """Generate audio from sparse parameters."""
    if synth is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base_z if provided
        base_z = None
        if request.base_z:
            z_bytes = base64.b64decode(request.base_z)
            base_z = torch.from_numpy(
                np.frombuffer(z_bytes, dtype=np.float32).reshape(1, 8, 16, -1)
            ).to(synth.device)

        # Synthesize
        audio, z = synth.synthesize(request.params, base_z)

        # Encode audio to wav bytes
        import soundfile as sf
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, 44100, format='WAV')
        audio_b64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')

        # Encode z latent
        z_bytes = z.cpu().numpy().astype(np.float32).tobytes()
        z_b64 = base64.b64encode(z_bytes).decode('utf-8')

        return SynthResponse(
            audio_b64=audio_b64,
            sample_rate=44100,
            z_b64=z_b64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/params/defaults")
async def get_default_params():
    """Get default parameter values and ranges."""
    return {
        "brightness": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "Spectral brightness/centroid"},
        "low_energy": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "Bass content"},
        "mid_energy": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "Mid frequencies"},
        "high_energy": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "Treble content"},
        "attack": {"default": 0.3, "min": 0.0, "max": 1.0, "description": "Onset sharpness"},
        "sustain": {"default": 0.7, "min": 0.0, "max": 1.0, "description": "Body level"},
        "release": {"default": 0.3, "min": 0.0, "max": 1.0, "description": "Decay length"},
        "stereo_width": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "Spatial spread"},
        "detail": {"default": 0.5, "min": 0.0, "max": 1.0, "description": "Fine texture amount"},
        "duration_frames": {"default": 32, "min": 8, "max": 64, "description": "Duration in z frames"},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8097)
