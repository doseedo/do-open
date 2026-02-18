#!/usr/bin/env python3
"""Modulo synth patch schema — JSON format for inverse synthesis results.

Converts optimizer result dicts to a clean, loadable patch format.
The patch JSON can be loaded directly by the Modulo synth engine.

Schema:
{
    "version": "1.0",
    "synth_type": "subtractive" | "fm",
    "pitch": 220.0,

    # Subtractive params
    "waveform": "saw",
    "filter": {
        "type": "lowpass",
        "base_hz": 300.0,
        "peak_hz": 4000.0,
        "resonance": 0.3,
        "envelope": {"attack": 0.01, "decay": 0.2, "sustain": 0.1, "release": 0.3, "noteoff": 1.2}
    },
    "amp": {
        "envelope": {"attack": 0.005, "decay": 0.15, "sustain": 0.2, "release": 0.3, "noteoff": 1.2}
    },
    "lfo": {
        "rate": 0.0,
        "depth": 0.0,
        "target": "filter"
    },

    # FM params (when synth_type == "fm")
    "fm": {
        "mod_ratio": 1.0,
        "index_peak": 3.0,
        "envelope": {"attack": 0.001, "decay": 0.3, "sustain": 0.2, "release": 0.3, "noteoff": 1.5}
    },

    # Effects chain
    "effects": [
        {"type": "distortion", "drive": 5.0, "tone_hz": 4000},
        {"type": "delay", "time_s": 0.2, "feedback": 0.4, "mix": 0.3},
        ...
    ],

    # Quality metrics from optimizer
    "quality": {
        "spectral_similarity": 0.95,
        "time_correlation": 0.88,
        "pipeline": "subtractive"
    }
}
"""

import numpy as np
import orjson
from pathlib import Path


def _to_native(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def optimizer_result_to_patch(result, pitch=None):
    """Convert an optimizer result dict to a clean patch dict.

    Args:
        result: dict from optimize_patch_full() or similar
        pitch: override pitch (Hz), or use from result

    Returns:
        patch dict matching the schema above
    """
    synth_type = result.get('synth_type', 'subtractive')
    if result.get('pipeline') == 'fm' or 'mod_ratio' in result:
        synth_type = 'fm'

    p = pitch or result.get('pitch', 220.0)

    patch = {
        'version': '1.0',
        'synth_type': synth_type,
        'pitch': float(p),
    }

    if synth_type == 'fm':
        # FM synthesis patch
        fm_adsr = result.get('fm_adsr', result.get('fm_envelope', (0.001, 0.3, 0.2, 0.3, 1.0)))
        amp_adsr = result.get('amp_adsr', (0.001, 0.2, 0.5, 0.4, 1.5))

        patch['fm'] = {
            'mod_ratio': float(result.get('mod_ratio', 1.0)),
            'index_peak': float(result.get('fm_index_peak', 3.0)),
            'envelope': _adsr_dict(fm_adsr),
        }
        patch['amp'] = {
            'envelope': _adsr_dict(amp_adsr),
        }
    else:
        # Subtractive synthesis patch
        patch['waveform'] = result.get('waveform', 'saw')

        filter_adsr = result.get('filter_adsr', (0.01, 0.2, 0.5, 0.3, 1.0))
        amp_adsr = result.get('amp_adsr', (0.01, 0.15, 0.5, 0.3, 1.0))

        patch['filter'] = {
            'type': result.get('filter_type', 'lowpass'),
            'base_hz': float(result.get('filter_base_hz', 300)),
            'peak_hz': float(result.get('filter_peak_hz', 4000)),
            'resonance': float(result.get('resonance', 0.0)),
            'envelope': _adsr_dict(filter_adsr),
        }
        patch['amp'] = {
            'envelope': _adsr_dict(amp_adsr),
        }

        lfo_rate = float(result.get('lfo_rate', 0.0))
        lfo_depth = float(result.get('lfo_depth', 0.0))
        patch['lfo'] = {
            'rate': lfo_rate,
            'depth': lfo_depth,
            'target': 'filter',
        }

    # Effects
    effects_list = result.get('effects', [])
    effect_params = result.get('effect_params', {})
    patch['effects'] = []
    for fx_name in effects_list:
        fx_entry = {'type': fx_name}
        if fx_name in effect_params:
            fx_entry.update(effect_params[fx_name])
        patch['effects'].append(fx_entry)

    # Quality metrics
    patch['quality'] = {
        'spectral_similarity': float(result.get('spectral_sim', result.get('spectral_similarity', 0.0))),
        'time_correlation': float(result.get('time_corr', result.get('time_correlation', 0.0))),
        'pipeline': result.get('pipeline', synth_type),
    }
    if 'time_s' in result:
        patch['quality']['optimization_time_s'] = float(result['time_s'])

    return patch


def _adsr_dict(adsr_tuple):
    """Convert (a, d, s, r, noteoff) tuple to named dict."""
    if isinstance(adsr_tuple, dict):
        return adsr_tuple
    a = list(adsr_tuple) if not isinstance(adsr_tuple, list) else adsr_tuple
    return {
        'attack': float(a[0]) if len(a) > 0 else 0.01,
        'decay': float(a[1]) if len(a) > 1 else 0.2,
        'sustain': float(a[2]) if len(a) > 2 else 0.5,
        'release': float(a[3]) if len(a) > 3 else 0.3,
        'noteoff': float(a[4]) if len(a) > 4 else 1.0,
    }


def patch_to_render_params(patch):
    """Convert patch dict back to fast_dsp render params.

    Returns (params_array, waveform_name, filter_type, pitch) for subtractive,
    or (fm_params_array, 'fm', None, pitch) for FM.
    params_array is a numpy float64 array ready for fast_dsp.
    """
    import numpy as np
    pitch = patch.get('pitch', 220.0)

    if patch['synth_type'] == 'fm':
        fm = patch['fm']
        amp = patch['amp']['envelope']
        fm_env = fm['envelope']
        params = np.array([
            fm['mod_ratio'], fm['index_peak'],
            fm_env['attack'], fm_env['decay'], fm_env['sustain'],
            fm_env['release'], fm_env['noteoff'],
            amp['attack'], amp['decay'], amp['sustain'],
            amp['release'], amp['noteoff'],
        ], dtype=np.float64)
        return params, 'fm', None, pitch

    # Subtractive
    f = patch['filter']
    f_env = f['envelope']
    a_env = patch['amp']['envelope']

    params_list = [
        f['base_hz'], f['peak_hz'], f['resonance'],
        f_env['attack'], f_env['decay'], f_env['sustain'],
        f_env['release'], f_env['noteoff'],
        a_env['attack'], a_env['decay'], a_env['sustain'],
        a_env['release'], a_env['noteoff'],
    ]

    # LFO
    lfo = patch.get('lfo', {})
    if lfo.get('rate', 0) > 0 and lfo.get('depth', 0) > 0:
        params_list.extend([lfo['rate'], lfo['depth']])

    return np.array(params_list, dtype=np.float64), patch.get('waveform', 'saw'), f.get('type', 'lowpass'), pitch


def save_patch(patch, path):
    """Save patch to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    native_patch = _to_native(patch)
    with open(path, 'wb') as f:
        f.write(orjson.dumps(native_patch, option=orjson.OPT_INDENT_2))


def load_patch(path):
    """Load patch from JSON file."""
    with open(path, 'rb') as f:
        return orjson.loads(f.read())


def patch_summary(patch):
    """One-line human-readable summary of a patch."""
    st = patch['synth_type']
    pitch = patch.get('pitch', 0)
    quality = patch.get('quality', {})
    spec = quality.get('spectral_similarity', 0)

    if st == 'fm':
        fm = patch.get('fm', {})
        return (f"FM pitch={pitch:.0f}Hz ratio={fm.get('mod_ratio', '?')} "
                f"index={fm.get('index_peak', '?')} spec={spec:.3f}")
    else:
        wf = patch.get('waveform', '?')
        filt = patch.get('filter', {})
        ft = filt.get('type', 'lpf')
        base = filt.get('base_hz', 0)
        peak = filt.get('peak_hz', 0)
        res = filt.get('resonance', 0)
        lfo = patch.get('lfo', {})
        lfo_str = f" LFO={lfo['rate']:.1f}Hz" if lfo.get('rate', 0) > 0 else ""
        fx = patch.get('effects', [])
        fx_str = f" fx=[{','.join(e['type'] for e in fx)}]" if fx else ""
        return (f"{wf}@{pitch:.0f}Hz {ft} {base:.0f}-{peak:.0f}Hz "
                f"res={res:.2f}{lfo_str}{fx_str} spec={spec:.3f}")
