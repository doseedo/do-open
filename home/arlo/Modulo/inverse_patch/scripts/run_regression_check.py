#!/usr/bin/env python3
"""Quick regression check: original 8 patches only."""
import sys
sys.path.insert(0, 'scripts')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from test_audio_domain import (
    optimize_patch_audio_domain, generate_target_audio, TARGET_PATCHES,
)

ORIGINAL_PATCHES = ['pluck_saw220', 'acid_bass', 'warm_pad', 'bright_lead',
                    'dark_bass', 'square_stab', 'triangle_key', 'saw_high']

print("=" * 60)
print("Regression Check: Original 8 Patches")
print("=" * 60)

results = []
for pname in ORIGINAL_PATCHES:
    pdef = TARGET_PATCHES[pname]
    print(f"\n  {pname}: {pdef['waveform']}@{pdef['pitch']}Hz")
    audio = generate_target_audio(pdef)
    result = optimize_patch_audio_domain(audio, pdef['waveform'], pdef['pitch'], verbose=False)
    print(f"    Spec={result['spectral_sim']:.4f} Corr={result['time_corr']:.4f} "
          f"Time={result['time_s']:.1f}s")
    results.append(result)

avg_spec = np.mean([r['spectral_sim'] for r in results])
avg_corr = np.mean([r['time_corr'] for r in results])
print(f"\n  AVERAGE: Spec={avg_spec:.4f} Corr={avg_corr:.4f}")
print(f"  Previous: Spec=0.9409 Corr=0.9752")
print(f"  {'NO REGRESSION' if avg_spec >= 0.93 else 'REGRESSION!'}")
