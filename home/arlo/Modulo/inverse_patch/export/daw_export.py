"""
DAW Export functionality.

Converts estimated effect chains to DAW-readable formats.
"""

import json
import torch
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass


# Plugin mapping database
PLUGIN_MAPPINGS = {
    'eq': {
        'suggestions': [
            'FabFilter Pro-Q 3',
            'ReaEQ',
            'TDR Nova',
            'Waves F6',
            'iZotope Ozone EQ',
        ],
        'param_mapping': {
            'low_shelf_gain_db': 'Low Shelf Gain',
            'low_shelf_cutoff_freq': 'Low Shelf Frequency',
            'low_shelf_q_factor': 'Low Shelf Q',
            'band0_gain_db': 'Band 1 Gain',
            'band0_cutoff_freq': 'Band 1 Frequency',
            'band0_q_factor': 'Band 1 Q',
            'band1_gain_db': 'Band 2 Gain',
            'band1_cutoff_freq': 'Band 2 Frequency',
            'band1_q_factor': 'Band 2 Q',
            'band2_gain_db': 'Band 3 Gain',
            'band2_cutoff_freq': 'Band 3 Frequency',
            'band2_q_factor': 'Band 3 Q',
            'high_shelf_gain_db': 'High Shelf Gain',
            'high_shelf_cutoff_freq': 'High Shelf Frequency',
            'high_shelf_q_factor': 'High Shelf Q',
        }
    },
    'compressor': {
        'suggestions': [
            'FabFilter Pro-C 2',
            'ReaComp',
            'TDR Kotelnikov',
            'Waves CLA-76',
            'Waves SSL G-Master',
        ],
        'param_mapping': {
            'threshold_db': 'Threshold',
            'ratio': 'Ratio',
            'attack_ms': 'Attack',
            'release_ms': 'Release',
            'knee_db': 'Knee',
            'makeup_db': 'Makeup Gain',
        }
    },
    'reverb': {
        'suggestions': [
            'Valhalla Room',
            'FabFilter Pro-R',
            'ReaVerb',
            'Waves H-Reverb',
            'Eventide Blackhole',
        ],
        'param_mapping': {
            'decay_time': 'Decay Time',
            'pre_delay_ms': 'Pre-Delay',
            'wet_mix': 'Wet/Dry Mix',
            'damping': 'Damping',
        }
    },
    'distortion': {
        'suggestions': [
            'FabFilter Saturn 2',
            'Soundtoys Decapitator',
            'iZotope Trash 2',
            'Waves Abbey Road Saturator',
        ],
        'param_mapping': {
            'drive': 'Drive',
            'tone': 'Tone',
            'mix': 'Mix',
            'output_gain_db': 'Output',
        }
    },
    'chorus': {
        'suggestions': [
            'Valhalla SpaceModulator',
            'Waves Reel ADT',
            'TAL-Chorus-LX',
            'Soundtoys MicroShift',
        ],
        'param_mapping': {
            'rate': 'Rate',
            'depth': 'Depth',
            'mix': 'Mix',
            'feedback': 'Feedback',
        }
    },
    'delay': {
        'suggestions': [
            'Valhalla Delay',
            'Waves H-Delay',
            'ReaDelay',
            'Soundtoys EchoBoy',
        ],
        'param_mapping': {
            'delay_ms': 'Delay Time',
            'feedback': 'Feedback',
            'mix': 'Mix',
        }
    },
}


# Parameter ranges for denormalization
PARAM_RANGES = {
    'eq': {
        'low_shelf_gain_db': (-12.0, 12.0),
        'low_shelf_cutoff_freq': (20.0, 2000.0),
        'low_shelf_q_factor': (0.1, 10.0),
        'band0_gain_db': (-12.0, 12.0),
        'band0_cutoff_freq': (20.0, 200.0),
        'band0_q_factor': (0.1, 10.0),
        'band1_gain_db': (-12.0, 12.0),
        'band1_cutoff_freq': (200.0, 2000.0),
        'band1_q_factor': (0.1, 10.0),
        'band2_gain_db': (-12.0, 12.0),
        'band2_cutoff_freq': (2000.0, 12000.0),
        'band2_q_factor': (0.1, 10.0),
        'high_shelf_gain_db': (-12.0, 12.0),
        'high_shelf_cutoff_freq': (4000.0, 16000.0),
        'high_shelf_q_factor': (0.1, 10.0),
    },
    'compressor': {
        'threshold_db': (-60.0, 0.0),
        'ratio': (1.0, 20.0),
        'attack_ms': (0.1, 100.0),
        'release_ms': (10.0, 1000.0),
        'knee_db': (0.0, 12.0),
        'makeup_db': (0.0, 24.0),
    },
    'reverb': {
        'decay_time': (0.1, 10.0),
        'pre_delay_ms': (0.0, 100.0),
        'wet_mix': (0.0, 1.0),
        'damping': (0.0, 1.0),
    },
    'distortion': {
        'drive': (0.0, 1.0),
        'tone': (0.0, 1.0),
        'mix': (0.0, 1.0),
        'output_gain_db': (-12.0, 12.0),
    },
    'chorus': {
        'rate': (0.1, 10.0),
        'depth': (0.0, 1.0),
        'mix': (0.0, 1.0),
        'feedback': (0.0, 0.9),
    },
    'delay': {
        'delay_ms': (1.0, 2000.0),
        'feedback': (0.0, 0.95),
        'mix': (0.0, 1.0),
    },
}


def denormalize_params(
    effect_type: str,
    normalized_params: torch.Tensor,
) -> Dict[str, float]:
    """
    Convert normalized parameters to actual values.

    Args:
        effect_type: Type of effect
        normalized_params: Normalized parameters [N]

    Returns:
        Dictionary of denormalized parameters
    """
    if effect_type not in PARAM_RANGES:
        return {}

    ranges = PARAM_RANGES[effect_type]
    param_names = list(ranges.keys())

    result = {}
    for i, name in enumerate(param_names):
        if i < len(normalized_params):
            min_val, max_val = ranges[name]
            norm_val = normalized_params[i].item() if isinstance(normalized_params[i], torch.Tensor) else normalized_params[i]
            result[name] = min_val + norm_val * (max_val - min_val)

    return result


def map_to_plugin(effect_type: str) -> List[str]:
    """
    Map internal effect type to common plugin names.

    Args:
        effect_type: Internal effect type

    Returns:
        List of suggested plugin names
    """
    if effect_type in PLUGIN_MAPPINGS:
        return PLUGIN_MAPPINGS[effect_type]['suggestions']
    return []


def export_json(
    chain_spec: List[Tuple[str, torch.Tensor]],
    confidence_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Export chain specification to universal JSON format.

    Args:
        chain_spec: List of (effect_type, params) tuples
        confidence_scores: Optional confidence scores per effect

    Returns:
        JSON-serializable dictionary
    """
    result = {
        'version': '1.0',
        'format': 'inverse_afx',
        'chain': [],
        'metadata': {
            'num_effects': len(chain_spec),
            'effect_types': [fx for fx, _ in chain_spec],
        }
    }

    for i, (effect_type, params) in enumerate(chain_spec):
        # Denormalize parameters
        denorm_params = denormalize_params(effect_type, params)

        # Get human-readable parameter names
        if effect_type in PLUGIN_MAPPINGS:
            readable_params = {}
            param_mapping = PLUGIN_MAPPINGS[effect_type]['param_mapping']
            for internal_name, value in denorm_params.items():
                readable_name = param_mapping.get(internal_name, internal_name)
                readable_params[readable_name] = round(value, 4)
        else:
            readable_params = {k: round(v, 4) for k, v in denorm_params.items()}

        effect_entry = {
            'index': i,
            'effect_type': effect_type,
            'plugin_suggestions': map_to_plugin(effect_type),
            'parameters': readable_params,
            'raw_parameters': {
                k: round(v, 4) for k, v in denorm_params.items()
            },
        }

        if confidence_scores is not None and i < len(confidence_scores):
            effect_entry['confidence'] = round(confidence_scores[i], 4)

        result['chain'].append(effect_entry)

    return result


def export_reaper_fx_chain(
    chain_spec: List[Tuple[str, torch.Tensor]],
    output_path: Optional[str] = None,
) -> str:
    """
    Export chain as REAPER FX chain file (.RfxChain).

    Args:
        chain_spec: List of (effect_type, params) tuples
        output_path: Optional output file path

    Returns:
        REAPER FX chain content as string
    """
    lines = ['<FXCHAIN']

    for effect_type, params in chain_spec:
        denorm_params = denormalize_params(effect_type, params)
        plugin_name = map_to_plugin(effect_type)[0] if map_to_plugin(effect_type) else effect_type

        # Add comment with effect info
        lines.append(f'  # {effect_type.upper()}')
        lines.append(f'  # Suggested plugin: {plugin_name}')
        lines.append(f'  # Parameters:')

        for param_name, value in denorm_params.items():
            lines.append(f'  #   {param_name}: {value:.4f}')

        # Add placeholder VST entry
        lines.append(f'  <VST "VST: {plugin_name}" ""')
        lines.append('    BYPASS 0 0 0')
        lines.append('  >')
        lines.append('')

    lines.append('>')

    content = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(content)

    return content


def chain_to_daw_preset(
    chain_spec: List[Tuple[str, torch.Tensor]],
    format: str = 'json',
    output_path: Optional[str] = None,
    confidence_scores: Optional[List[float]] = None,
) -> Any:
    """
    Convert estimated chain to DAW-readable format.

    Args:
        chain_spec: List of (effect_type, params) tuples
        format: Output format ('json', 'reaper', 'ableton')
        output_path: Optional output file path
        confidence_scores: Optional confidence scores

    Returns:
        Exported preset in specified format
    """
    if format == 'json':
        result = export_json(chain_spec, confidence_scores)
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
        return result

    elif format == 'reaper':
        return export_reaper_fx_chain(chain_spec, output_path)

    elif format == 'ableton':
        # Ableton Live Set format is complex XML
        # Return JSON with Ableton-specific metadata for now
        result = export_json(chain_spec, confidence_scores)
        result['metadata']['daw'] = 'ableton'
        result['metadata']['note'] = 'Import manually using suggested plugins'

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
        return result

    else:
        raise ValueError(f"Unknown format: {format}")


def generate_processing_report(
    chain_spec: List[Tuple[str, torch.Tensor]],
    confidence_scores: Optional[List[float]] = None,
    dry_audio_path: Optional[str] = None,
) -> str:
    """
    Generate a human-readable processing report.

    Args:
        chain_spec: Estimated effect chain
        confidence_scores: Confidence scores per effect
        dry_audio_path: Path to recovered dry audio

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "INVERSE AUDIO EFFECTS - PROCESSING REPORT",
        "=" * 60,
        "",
        f"Detected {len(chain_spec)} effect(s) in chain:",
        "",
    ]

    for i, (effect_type, params) in enumerate(chain_spec):
        denorm_params = denormalize_params(effect_type, params)
        plugins = map_to_plugin(effect_type)

        lines.append(f"Effect {i+1}: {effect_type.upper()}")

        if confidence_scores and i < len(confidence_scores):
            lines.append(f"  Confidence: {confidence_scores[i]*100:.1f}%")

        lines.append("  Parameters:")
        for name, value in denorm_params.items():
            lines.append(f"    - {name}: {value:.4f}")

        lines.append("  Suggested plugins:")
        for plugin in plugins[:3]:
            lines.append(f"    - {plugin}")

        lines.append("")

    if dry_audio_path:
        lines.append(f"Recovered dry audio saved to: {dry_audio_path}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("To recreate this effect chain:")
    lines.append("1. Load your dry audio in your DAW")
    lines.append("2. Add the effects in order listed above")
    lines.append("3. Adjust parameters to match the values shown")
    lines.append("=" * 60)

    return "\n".join(lines)
