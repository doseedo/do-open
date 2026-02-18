#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom ACE-Step Pipeline Extension with Noise Level Support

This extends the original ACEStepPipeline to support GT latent mixing,
similar to the approach in genfromweb5.py.

Key addition:
- noise_level parameter: controls mixing of ground truth latents with noise
  - 0.0 = pure GT latents (perfect reconstruction)
  - 1.0 = pure noise (creative generation)
  - 0.8 = 20% GT + 80% noise (controlled variation)
"""

import sys
import torch
from diffusers.utils.torch_utils import randn_tensor

# Add ACE-Step to path
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from acestep.pipeline_ace_step import ACEStepPipeline


class ACEStepPipelineWithNoise(ACEStepPipeline):
    """
    Extended ACE-Step pipeline with noise_level control for GT latent mixing.

    Usage:
        pipeline = ACEStepPipelineWithNoise(device_id=0, dtype="bfloat16")
        pipeline(
            prompt="male vocals",
            lyrics="extracted lyrics text",
            ref_audio_input="/path/to/vocals.wav",
            audio2audio_enable=True,
            noise_level=0.8,  # 80% noise, 20% GT latents
            save_path="output.wav"
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("✅ Initialized ACEStepPipelineWithNoise (supports noise_level mixing)")

    def __call__(
        self,
        noise_level=1.0,  # NEW: noise level for GT latent mixing
        **kwargs
    ):
        """
        Extended __call__ with noise_level parameter.

        Args:
            noise_level (float): Controls GT latent mixing
                - 0.0 = pure GT latents (reconstruction)
                - 1.0 = pure noise (creative generation)
                - 0.0-1.0 = linear mix
            **kwargs: All other ACEStepPipeline parameters
        """
        # Store noise_level for use in latent generation
        self._custom_noise_level = float(noise_level)

        # Call parent __call__ which will trigger our custom generate method
        return super().__call__(**kwargs)

    def _generate_initial_latents(
        self,
        shape,
        generator,
        ref_latents=None,
    ):
        """
        Generate initial latents with optional GT latent mixing.

        This is where we implement the noise_level control similar to genfromweb5.py:
        x = (1.0 - noise_level) * gt_latents + noise_level * noise
        """
        noise_level = getattr(self, '_custom_noise_level', 1.0)

        # Generate pure noise
        pure_noise = randn_tensor(
            shape=shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # If noise_level is 1.0 or no ref_latents, return pure noise
        if noise_level >= 1.0 or ref_latents is None:
            if noise_level < 1.0 and ref_latents is None:
                print(f"⚠️  noise_level={noise_level} but no ref_latents available, using pure noise")
            return pure_noise

        # Mix GT latents with noise based on noise_level
        # Ensure ref_latents matches target shape
        if ref_latents.shape != shape:
            print(f"⚠️  ref_latents shape {ref_latents.shape} != target {shape}, using pure noise")
            return pure_noise

        if noise_level <= 0.0:
            # Pure GT latents (perfect reconstruction)
            print(f"✅ Using pure GT latents (noise_level=0.0)")
            return ref_latents
        else:
            # Mix GT latents with noise
            mixed_latents = (1.0 - noise_level) * ref_latents + noise_level * pure_noise
            print(f"✅ Mixed GT latents: {(1.0-noise_level)*100:.1f}% GT + {noise_level*100:.1f}% noise")
            return mixed_latents


# Convenience wrapper function
def create_pipeline_with_noise(device_id=0, dtype="bfloat16", **kwargs):
    """
    Create an ACE-Step pipeline with noise level support.

    Args:
        device_id: GPU device ID
        dtype: Model dtype ("bfloat16" or "float32")
        **kwargs: Additional pipeline initialization args

    Returns:
        ACEStepPipelineWithNoise instance
    """
    return ACEStepPipelineWithNoise(
        device_id=device_id,
        dtype=dtype,
        **kwargs
    )


if __name__ == "__main__":
    print("""
ACE-Step Custom Pipeline with Noise Level Support

This module provides ACEStepPipelineWithNoise, which extends the original
ACE-Step pipeline to support mixing ground truth latents with noise.

Example usage:

    from ace_step_pipeline_custom import ACEStepPipelineWithNoise

    pipeline = ACEStepPipelineWithNoise(device_id=0, dtype="bfloat16")

    # Generate with 80% noise, 20% GT latents
    pipeline(
        prompt="male vocals",
        lyrics="song lyrics here",
        ref_audio_input="/path/to/reference.wav",
        audio2audio_enable=True,
        noise_level=0.8,  # Key parameter!
        save_path="output.wav"
    )

Note: The noise_level mixing happens internally when ref_latents are extracted.
For full functionality, the parent ACEStepPipeline needs to pass ref_latents
to our _generate_initial_latents method.
""")
