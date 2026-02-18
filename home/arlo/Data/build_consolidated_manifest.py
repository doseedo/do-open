#!/usr/bin/env python3
"""
Build a consolidated manifest combining all known data about every audio file.

Sources (priority order for group labels):
  1. corrections.json — manual overrides (highest priority)
  2. unified_manifest.json — filename-derived labels
  3. latent_classifier/predictions.json — ML predictions for undefined files

Also merges: subgroup predictions, multilabel classifier, stems_classified, master_manifest.

Output: /home/arlo/gcs-bucket/Manifests/consolidated_manifest.json
"""

import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import orjson

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)

# Paths
MANIFESTS_DIR = Path("/home/arlo/gcs-bucket/Manifests")
UNIFIED_MANIFEST = MANIFESTS_DIR / "unified_manifest.json"
CORRECTIONS = MANIFESTS_DIR / "corrections.json"
MASTER_MANIFEST = MANIFESTS_DIR / "master_manifest.json"

LATENT_CLASSIFIER_PREDS = Path("/home/arlo/Data/latent_classifier/predictions.json")
MULTILABEL_PREDS = Path("/home/arlo/Data/multilabel_classifier/predictions.json")
STEMS_CLASSIFIED = Path("/home/arlo/Data/mix_classifier/stems_classified.json")

SUBGROUP_PREDICTION_FILES = [
    MANIFESTS_DIR / "bass_subgroup_predictions_manifest.json",
    MANIFESTS_DIR / "brass_subgroup_predictions_manifest.json",
    MANIFESTS_DIR / "guitar_subgroup_predictions_manifest.json",
    MANIFESTS_DIR / "piano_subgroup_predictions_manifest.json",
    MANIFESTS_DIR / "strings_subgroup_predictions_manifest.json",
]

OUTPUT_PATH = MANIFESTS_DIR / "consolidated_manifest.json"

MIX_FILENAME_KEYWORDS = ['_mix', '/mix', 'mix.', 'mix_', 'master_', 'bounce', 'full_track']

EXCLUDED_GROUPS = {'undefined', 'room', 'fx', 'click', 'silent', 'junk', 'review_vocals', 'full-track'}


def load_json(path: Path) -> dict:
    """Load a JSON file using orjson."""
    logging.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    with open(path, 'rb') as f:
        return orjson.loads(f.read())


def load_all_sources():
    """Load all data sources into memory."""

    # 1. Unified manifest (base)
    unified = load_json(UNIFIED_MANIFEST)
    logging.info(f"  Unified manifest: {unified['total_entries']} entries")

    # 2. Corrections
    corrections = {}
    if CORRECTIONS.exists():
        corrections = load_json(CORRECTIONS)
        logging.info(f"  Corrections: {len(corrections)} entries")

    # 3. Latent classifier predictions → dict by path
    lc_by_path = {}
    if LATENT_CLASSIFIER_PREDS.exists():
        lc_data = load_json(LATENT_CLASSIFIER_PREDS)
        for pred in lc_data.get('predictions', []):
            lc_by_path[pred['path']] = pred
        logging.info(f"  Latent classifier: {len(lc_by_path)} predictions")

    # 4. Subgroup predictions → dict by path
    subgroup_by_path = {}
    for sg_file in SUBGROUP_PREDICTION_FILES:
        if sg_file.exists():
            sg_data = load_json(sg_file)
            for path, entry in sg_data.items():
                if isinstance(entry, dict):
                    subgroup_by_path[path] = entry
            logging.info(f"  {sg_file.name}: {len(sg_data)} entries")
    logging.info(f"  Total subgroup predictions: {len(subgroup_by_path)}")

    # 5. Multilabel classifier predictions → dict by path
    ml_by_path = {}
    if MULTILABEL_PREDS.exists():
        ml_data = load_json(MULTILABEL_PREDS)
        for pred in ml_data.get('results', []):
            ml_by_path[pred['path']] = pred
        logging.info(f"  Multilabel classifier: {len(ml_by_path)} predictions")

    # 6. Stems classified → dict by original_path
    stems_by_path = {}
    if STEMS_CLASSIFIED.exists():
        stems_data = load_json(STEMS_CLASSIFIED)
        for result in stems_data.get('results', []):
            stems_by_path[result['original_path']] = result
        logging.info(f"  Stems classified: {len(stems_by_path)} mix files")

    # 7. Master manifest → dict by path (entries are keyed by audio_path)
    master_by_path = {}
    if MASTER_MANIFEST.exists():
        master_data = load_json(MASTER_MANIFEST)
        master_by_path = master_data.get('entries', {})
        logging.info(f"  Master manifest: {len(master_by_path)} entries")

    return unified, corrections, lc_by_path, subgroup_by_path, ml_by_path, stems_by_path, master_by_path


def is_mix_by_filename(path: str) -> bool:
    """Check if filename suggests this is a mix file."""
    lower = path.lower()
    return any(kw in lower for kw in MIX_FILENAME_KEYWORDS)


def build_consolidated():
    """Build the consolidated manifest."""
    (unified, corrections, lc_by_path, subgroup_by_path,
     ml_by_path, stems_by_path, master_by_path) = load_all_sources()

    entries = []
    stats = Counter()
    group_dist = Counter()
    group_source_dist = Counter()
    subgroup_source_dist = Counter()

    for base_entry in unified['entries']:
        audio_path = base_entry.get('audio_path', '')
        if not audio_path:
            continue

        corr = corrections.get(audio_path)
        lc_pred = lc_by_path.get(audio_path)
        sg_pred = subgroup_by_path.get(audio_path)
        ml_pred = ml_by_path.get(audio_path)
        stems = stems_by_path.get(audio_path)
        master = master_by_path.get(audio_path, {})

        # --- GROUP ---
        group = None
        group_source = None
        group_confidence = None

        # Priority 1: Correction
        if corr and corr.get('group') and (corr['group'] not in EXCLUDED_GROUPS or corr['group'] in ('full-track', 'ensemble', 'mix')):
            group = corr['group']
            group_source = 'correction'

        # Priority 2: Unified manifest (filename-derived)
        if group is None:
            um_group = base_entry.get('group', '')
            if um_group and um_group not in ('undefined', 'unknown', ''):
                group = um_group
                group_source = 'filename'

        # Priority 3: Latent classifier prediction
        if group is None and lc_pred:
            group = lc_pred['predicted_group']
            group_source = 'classifier'
            group_confidence = round(lc_pred['confidence'], 4)

        # Still undefined
        if group is None:
            group = 'undefined'
            group_source = 'none'

        # Handle _roomy suffix
        if group.endswith('_roomy'):
            group = group.replace('_roomy', '')

        # Map ensemble/mix/full-track to unified 'mix' group
        if group in ('ensemble', 'mix', 'full-track'):
            group = 'mix'
            if group_source is None:
                group_source = 'correction'

        # --- SUBGROUP ---
        subgroup = ''
        subgroup_source = None
        subgroup_confidence = None

        # Priority 1: Correction
        if corr and corr.get('subgroup'):
            subgroup = corr['subgroup']
            subgroup_source = 'correction'

        # Priority 2: Subgroup classifier
        if not subgroup and sg_pred:
            subgroup = sg_pred.get('subgroup', '')
            subgroup_source = 'subgroup_classifier'
            subgroup_confidence = round(sg_pred.get('confidence', 0), 4)

        # Priority 3: Unified manifest
        if not subgroup:
            subgroup = base_entry.get('subgroup', '')
            if subgroup:
                subgroup_source = 'filename'

        # --- MIX DETECTION ---
        is_mix = False

        # From master manifest
        if master.get('is_mix'):
            is_mix = True

        # From corrections (ensemble/mix/full-track group)
        if corr and corr.get('group') in ('ensemble', 'mix', 'full-track'):
            is_mix = True

        # From multilabel classifier
        if ml_pred and ml_pred.get('is_multilabel'):
            is_mix = True

        # From stems classified
        if stems:
            is_mix = True

        # From filename
        if is_mix_by_filename(audio_path):
            is_mix = True

        # --- CONSOLIDATE MIX GROUP ---
        # Any file detected as mix should live in the 'mix' group, not in instrument groups
        if is_mix and group not in ('mix',):
            # Preserve original group as detected instrument info
            if group not in EXCLUDED_GROUPS and group != 'undefined':
                pass  # detected_instruments will capture it below
            group = 'mix'
            group_source = group_source or 'mix_detection'

        # Undefined files that are mix by filename should also be 'mix'
        if group == 'undefined' and is_mix_by_filename(audio_path):
            group = 'mix'
            is_mix = True
            group_source = 'filename'

        # --- DETECTED INSTRUMENTS ---
        detected_instruments = []
        multilabel_probabilities = {}

        if ml_pred:
            detected_instruments = ml_pred.get('predicted_labels', [])
            multilabel_probabilities = ml_pred.get('all_probabilities', {})

        # Merge stems detected instruments
        if stems:
            stems_instruments = stems.get('detected_instruments', [])
            for inst in stems_instruments:
                if inst not in detected_instruments:
                    detected_instruments.append(inst)

        # If not mix and no multilabel, single instrument = group
        if not is_mix and not detected_instruments and group not in EXCLUDED_GROUPS and group != 'undefined':
            detected_instruments = [group]

        # --- BLEED / ROOMY ---
        bleed_instruments = []
        has_bleed = False
        roomy = False

        if master:
            bleed_instruments = master.get('bleed_instruments', [])
            roomy = master.get('roomy', False)

        if corr and 'bleed_instruments' in corr:
            bleed_instruments = corr['bleed_instruments']

        has_bleed = len(bleed_instruments) > 0

        # --- TECHNIQUE ---
        technique = base_entry.get('technique')

        # --- FLAGGING ---
        flagged = False
        flag_reason = None

        # If filename says one group and classifier says another
        um_group = base_entry.get('group', '')
        if (um_group and um_group not in ('undefined', 'unknown', '')
                and lc_pred and lc_pred['predicted_group'] != um_group
                and um_group not in EXCLUDED_GROUPS):
            flagged = True
            flag_reason = f"classifier={lc_pred['predicted_group']} vs filename={um_group}"

        # --- BUILD ENTRY ---
        entry = {
            'audio_path': audio_path,
            'latent_path': base_entry.get('latent_path', ''),
            'has_latent': base_entry.get('has_latent', False),
            'group': group,
            'subgroup': subgroup,
            'group_source': group_source,
            'subgroup_source': subgroup_source,
            'is_mix': is_mix,
            'detected_instruments': detected_instruments,
            'has_bleed': has_bleed,
            'bleed_instruments': bleed_instruments,
            'roomy': roomy,
            'technique': technique,
            'has_correction': corr is not None,
            'flagged': flagged,
            'source': base_entry.get('source', ''),
        }

        # Only include non-None confidence values
        if group_confidence is not None:
            entry['group_confidence'] = group_confidence
        if subgroup_confidence is not None:
            entry['subgroup_confidence'] = subgroup_confidence
        if multilabel_probabilities:
            entry['multilabel_probabilities'] = multilabel_probabilities
        if flag_reason:
            entry['flag_reason'] = flag_reason

        entries.append(entry)

        # --- STATS ---
        stats['total'] += 1
        group_dist[group] += 1
        group_source_dist[group_source] += 1
        if subgroup_source:
            subgroup_source_dist[subgroup_source] += 1
        if is_mix:
            stats['mix'] += 1
        if has_bleed:
            stats['has_bleed'] += 1
        if flagged:
            stats['flagged'] += 1
        if corr:
            stats['corrected'] += 1
        if group == 'undefined':
            stats['still_undefined'] += 1
        if base_entry.get('has_latent'):
            stats['has_latent'] += 1

    # Build output
    output = {
        'generated_at': datetime.now().isoformat(),
        'total_entries': len(entries),
        'stats': {
            'total': stats['total'],
            'has_latent': stats['has_latent'],
            'mix_files': stats['mix'],
            'has_bleed': stats['has_bleed'],
            'flagged': stats['flagged'],
            'corrected': stats['corrected'],
            'still_undefined': stats['still_undefined'],
        },
        'group_distribution': dict(sorted(group_dist.items())),
        'group_source_distribution': dict(sorted(group_source_dist.items())),
        'subgroup_source_distribution': dict(sorted(subgroup_source_dist.items())),
        'entries': entries,
    }

    logging.info(f"\n{'='*50}")
    logging.info(f"CONSOLIDATED MANIFEST SUMMARY")
    logging.info(f"{'='*50}")
    logging.info(f"Total entries: {len(entries)}")
    logging.info(f"Still undefined: {stats['still_undefined']}")
    logging.info(f"Mix files: {stats['mix']}")
    logging.info(f"Has bleed: {stats['has_bleed']}")
    logging.info(f"Flagged: {stats['flagged']}")
    logging.info(f"Corrected: {stats['corrected']}")
    logging.info(f"Has latent: {stats['has_latent']}")
    logging.info(f"\nGroup distribution:")
    for g, c in sorted(group_dist.items(), key=lambda x: -x[1]):
        logging.info(f"  {g}: {c}")
    logging.info(f"\nGroup source distribution:")
    for s, c in sorted(group_source_dist.items(), key=lambda x: -x[1]):
        logging.info(f"  {s}: {c}")
    logging.info(f"\nSubgroup source distribution:")
    for s, c in sorted(subgroup_source_dist.items(), key=lambda x: -x[1]):
        logging.info(f"  {s}: {c}")

    # Write output
    logging.info(f"\nWriting to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(orjson.dumps(output, option=orjson.OPT_INDENT_2))
    logging.info(f"Done! {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB written")


if __name__ == '__main__':
    build_consolidated()
