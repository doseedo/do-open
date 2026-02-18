#!/usr/bin/env python3
"""Quick runner for Test 5 (audio-domain roundtrip) only."""
import sys
sys.path.insert(0, 'scripts')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path

# Import just what we need from test_inverse_synth (avoid DCAE import)
# We need: TARGET_PATCHES, detect_pitch_yin, classify_waveform, _save_wav
# But test_inverse_synth imports DCAE at module level, so we can't avoid it
# Instead, run test_audio_domain_roundtrip directly with minimal deps

from test_audio_domain import (
    optimize_patch_audio_domain, optimize_patch_auto_waveform,
    generate_target_audio, TARGET_PATCHES,
)
from fast_dsp import spectral_similarity, time_correlation

OUTPUT_DIR = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/inverse_synth")

# We need pitch/waveform detection from test_inverse_synth
# But that imports DCAE. Let's just test the optimizer without the detection
# since we already tested that in test_audio_domain.py

print("=" * 60)
print("Test 5: Audio-Domain Round-Trip (standalone)")
print("=" * 60)

test_dir = OUTPUT_DIR / "audio_domain_roundtrip"
test_dir.mkdir(parents=True, exist_ok=True)

results = []
for pname, pdef in TARGET_PATCHES.items():
    print(f"\n  Target: {pname}")
    print(f"    True: {pdef['waveform']}@{pdef['pitch']}Hz, "
          f"cutoff={pdef['filter_base_hz']}-{pdef['filter_peak_hz']}Hz, "
          f"res={pdef['resonance']}")

    audio_original = generate_target_audio(pdef)

    result = optimize_patch_audio_domain(
        audio_original, pdef['waveform'], pdef['pitch'], verbose=True
    )

    print(f"    Recovered: cutoff={result['filter_base_hz']:.0f}-{result['filter_peak_hz']:.0f}Hz, "
          f"res={result['resonance']:.2f}")
    print(f"    Spec={result['spectral_sim']:.4f} Corr={result['time_corr']:.4f} "
          f"Time={result['time_s']:.1f}s")

    results.append({
        'name': pname, 'spec': result['spectral_sim'],
        'corr': result['time_corr'], 'time': result['time_s'],
    })

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for r in results:
    print(f"  {r['name']:<18s} Spec={r['spec']:.4f} Corr={r['corr']:.4f} Time={r['time']:.1f}s")

avg_spec = np.mean([r['spec'] for r in results])
avg_corr = np.mean([r['corr'] for r in results])
print(f"\n  AVERAGE: Spec={avg_spec:.4f} Corr={avg_corr:.4f}")
