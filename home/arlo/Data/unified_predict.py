#!/usr/bin/env python3
"""
Generate predictions.json from the unified classifier for review in doseedo.com/label.

Two-phase inference:
  Phase 1: Whole-file pooled inference on all files → group, subgroup, mix detection
  Phase 2: Sliding window temporal inference on detected mix files → timeline

Output:
  - Isolated files: single group + subgroup prediction (NO multilabel clutter)
  - Mix files: timeline of instruments per time window + detected_instruments summary

Output format matches what monitor_service.py expects from CLASSIFIER_BASE_DIRS.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
import numpy as np
import orjson

# Import from unified classifier
sys.path.insert(0, str(Path(__file__).parent))
from unified_self_improving_classifier import (
    UnifiedClassifier, UnifiedTrainer,
    GROUP_CLASSES, SUBGROUP_MAP, HIERARCHY, INPUT_DIM,
    MANIFEST_PATH, OUTPUT_DIR,
    load_latent, pool_latent, audio_path_to_latent_path,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)

# Temporal inference constants
LATENT_FPS = 44100 / 512  # ~86.13 frames per second
WINDOW_SECONDS = 4.5
WINDOW_FRAMES = int(WINDOW_SECONDS * LATENT_FPS)  # ~388 frames
STRIDE_SECONDS = 2.25  # 50% overlap
STRIDE_FRAMES = int(STRIDE_SECONDS * LATENT_FPS)  # ~194 frames
MIN_INSTRUMENT_CONFIDENCE = 0.35  # threshold for including instrument in timeline


def load_model(checkpoint_path: Path, device: str = 'cuda'):
    """Load unified classifier from checkpoint."""
    model = UnifiedClassifier(
        group_classes=GROUP_CLASSES,
        subgroup_map=SUBGROUP_MAP,
        hierarchy=HIERARCHY,
    )
    trainer = UnifiedTrainer(model, device=device)
    trainer.load_checkpoint(checkpoint_path)
    trainer.model.eval()
    return trainer


def load_all_entries(manifest_path: Path):
    """Load manifest and return all entries with their current labels."""
    logging.info(f"Loading manifest from {manifest_path}")
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    entries = []
    is_unified = 'entries' in manifest and isinstance(manifest.get('entries'), list)

    if is_unified:
        for entry in manifest['entries']:
            entries.append({
                'path': entry.get('audio_path', ''),
                'manifest_group': entry.get('group', 'undefined'),
                'manifest_subgroup': entry.get('subgroup', ''),
                'has_latent': entry.get('has_latent', False),
                'latent_path': entry.get('latent_path', ''),
                'is_mix': entry.get('is_mix', False),
            })
    else:
        for path, meta in manifest.items():
            if not isinstance(meta, dict):
                continue
            entries.append({
                'path': path,
                'manifest_group': meta.get('group', 'undefined'),
                'manifest_subgroup': meta.get('subgroup', ''),
                'has_latent': bool(meta.get('has_latent')),
                'latent_path': meta.get('latent_path', ''),
                'is_mix': meta.get('is_mix', False),
            })

    logging.info(f"Loaded {len(entries)} manifest entries")
    return entries


def _load_one_pooled(entry):
    """Load a single entry's pooled features. Returns (path, features) or None."""
    path = entry['path']
    latent = None
    if entry.get('latent_path'):
        latent = load_latent(Path(entry['latent_path']))
    if latent is None:
        latent_path = audio_path_to_latent_path(path)
        if latent_path:
            latent = load_latent(latent_path)
    if latent is None:
        return None
    return (path, pool_latent(latent))


def load_features_parallel(entries, num_workers=32):
    """Load all pooled latent features in parallel using a thread pool."""
    total = len(entries)
    loaded = []
    skipped = 0

    logging.info(f"Loading features with {num_workers} workers for {total} entries...")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_load_one_pooled, e): i for i, e in enumerate(entries)}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 25000 == 0:
                logging.info(f"  Loaded: {done}/{total} | valid: {len(loaded)} | skipped: {skipped}")
            result = future.result()
            if result is None:
                skipped += 1
            else:
                loaded.append(result)

    logging.info(f"Feature loading done: {len(loaded)} valid, {skipped} skipped (no latent)")
    return loaded


def _load_raw_latent(path):
    """Load raw (unpooled) latent for a path. Returns (path, latent_tensor) or None."""
    latent_path = audio_path_to_latent_path(path)
    if latent_path is None:
        return None
    latent = load_latent(latent_path)
    if latent is None:
        return None
    return (path, latent)


@torch.no_grad()
def predict_batch_wholefile(trainer, features_batch, paths_batch):
    """Run whole-file inference on a batch. Returns per-sample predictions."""
    features = torch.stack(features_batch).to(trainer.device)
    features = trainer.normalize(features)

    outputs = trainer.model(features)

    # Group predictions (softmax)
    group_probs = F.softmax(outputs['group'], dim=-1)
    group_confs, group_preds = group_probs.max(dim=-1)

    # Mix detector
    mix_probs = F.softmax(outputs['is_mix'], dim=-1)

    # Bleed detection (sigmoid, multi-label)
    bleed_probs = torch.sigmoid(outputs['bleed'])

    # Dialogue detection (sigmoid, binary)
    dialogue_probs = torch.sigmoid(outputs['dialogue']).squeeze(-1)

    results = []
    for i in range(len(paths_batch)):
        group_idx = group_preds[i].item()
        group_name = GROUP_CLASSES[group_idx]
        group_conf = group_confs[i].item()

        # All group probabilities
        all_probs = {GROUP_CLASSES[j]: round(group_probs[i, j].item(), 6)
                     for j in range(len(GROUP_CLASSES))}

        # Mix probability
        is_mix_prob = mix_probs[i, 1].item()
        is_mix = is_mix_prob > 0.5

        # Subgroup prediction for the top group
        predicted_subgroup = ""
        subgroup_conf = 0.0
        subgroup_key = f'subgroup_{group_name}'
        if subgroup_key in outputs and group_name in SUBGROUP_MAP:
            sub_logits = outputs[subgroup_key][i]
            sub_probs = F.softmax(sub_logits, dim=-1)
            sub_conf, sub_idx = sub_probs.max(dim=-1)
            subgroup_list = SUBGROUP_MAP[group_name]
            if sub_idx.item() < len(subgroup_list):
                predicted_subgroup = subgroup_list[sub_idx.item()]
                subgroup_conf = sub_conf.item()

        # Bleed predictions (threshold 0.3 for high recall)
        bleed_instruments = []
        for j in range(len(GROUP_CLASSES)):
            bp = bleed_probs[i, j].item()
            if bp > 0.3:
                bleed_instruments.append(GROUP_CLASSES[j])
        has_bleed = len(bleed_instruments) > 0

        # Dialogue prediction (threshold 0.5)
        dialogue_prob = dialogue_probs[i].item()
        is_dialogue = dialogue_prob > 0.5

        results.append({
            'path': paths_batch[i],
            'predicted_group': group_name,
            'confidence': round(group_conf, 6),
            'all_probabilities': all_probs,
            'is_mix': is_mix,
            'multi_probability': round(is_mix_prob, 6),
            'predicted_subgroup': predicted_subgroup,
            'subgroup_confidence': round(subgroup_conf, 6),
            'parent_group': group_name,
            'has_bleed': has_bleed,
            'bleed_instruments': bleed_instruments,
            'is_dialogue': is_dialogue,
            'dialogue_probability': round(dialogue_prob, 6),
        })

    return results


@torch.no_grad()
def run_phase1_predictions(trainer, features_list, batch_size=512):
    """Phase 1: Whole-file pooled inference on all files."""
    predictions = []
    total = len(features_list)
    logging.info(f"Phase 1: Whole-file inference on {total} samples (batch_size={batch_size})...")

    for start in range(0, total, batch_size):
        batch = features_list[start:start + batch_size]
        paths_batch = [p for p, _ in batch]
        feats_batch = [f for _, f in batch]
        results = predict_batch_wholefile(trainer, feats_batch, paths_batch)
        predictions.extend(results)

        if (start // batch_size) % 20 == 0:
            logging.info(f"  Predicted: {len(predictions)}/{total}")

    logging.info(f"Phase 1 done: {len(predictions)} predictions")
    return predictions


@torch.no_grad()
def run_temporal_inference(trainer, raw_latent, min_conf=MIN_INSTRUMENT_CONFIDENCE):
    """
    Run sliding window temporal inference on a single raw latent.

    Uses forward_temporal() (bidirectional GRU) for cross-window context smoothing.

    Returns timeline: list of {start, end, instruments, has_dialogue, dialogue_confidence}
    and detected_instruments: list of unique instruments found.
    """
    # Ensure [8, 16, T] shape
    if raw_latent.dim() == 2 and raw_latent.shape[0] == 128:
        raw_latent = raw_latent.view(8, 16, -1)

    T = raw_latent.shape[-1]
    if T < 10:
        return [], []

    # Generate windows
    windows = []
    window_times = []

    if T <= WINDOW_FRAMES:
        # File shorter than one window - single window
        windows.append(pool_latent(raw_latent))
        window_times.append((0.0, T / LATENT_FPS))
    else:
        start = 0
        while start < T:
            end = min(start + WINDOW_FRAMES, T)
            # Skip very short trailing windows
            if end - start < WINDOW_FRAMES // 4 and start > 0:
                break
            window_latent = raw_latent[:, :, start:end] if raw_latent.dim() == 3 else raw_latent[:, start:end]
            windows.append(pool_latent(window_latent))
            window_times.append((start / LATENT_FPS, end / LATENT_FPS))
            start += STRIDE_FRAMES

    if not windows:
        return [], []

    # Stack features: [num_windows, input_dim]
    features = torch.stack(windows).to(trainer.device)
    features = trainer.normalize(features)

    # Use forward_temporal (GRU) for cross-window context: [1, num_windows, input_dim]
    features_seq = features.unsqueeze(0)  # [1, T, D]
    outputs = trainer.model.forward_temporal(features_seq)

    # Squeeze batch dim: outputs are [1, T, C] → [T, C]
    ml_probs = torch.sigmoid(outputs['multilabel'][0])
    group_probs = F.softmax(outputs['group'][0], dim=-1)
    dialogue_probs = torch.sigmoid(outputs['dialogue'][0]).squeeze(-1)

    timeline = []
    all_instruments = set()

    for i in range(len(windows)):
        start_sec = round(window_times[i][0], 2)
        end_sec = round(window_times[i][1], 2)

        # Get instruments from multilabel head
        instruments = []
        for j in range(len(GROUP_CLASSES)):
            prob = ml_probs[i, j].item()
            if prob > min_conf:
                instruments.append({
                    'instrument': GROUP_CLASSES[j],
                    'confidence': round(prob, 4),
                })
                all_instruments.add(GROUP_CLASSES[j])

        # If multilabel head finds nothing above threshold, use group head top prediction
        if not instruments:
            top_idx = group_probs[i].argmax().item()
            top_conf = group_probs[i, top_idx].item()
            instruments.append({
                'instrument': GROUP_CLASSES[top_idx],
                'confidence': round(top_conf, 4),
            })
            all_instruments.add(GROUP_CLASSES[top_idx])

        # Sort by confidence descending
        instruments.sort(key=lambda x: x['confidence'], reverse=True)

        # Dialogue detection per window
        dlg_conf = dialogue_probs[i].item()

        timeline.append({
            'start': start_sec,
            'end': end_sec,
            'instruments': instruments,
            'has_dialogue': dlg_conf > 0.5,
            'dialogue_confidence': round(dlg_conf, 4),
        })

    detected_instruments = sorted(all_instruments)
    return timeline, detected_instruments


def _merge_dialogue_regions(timeline):
    """Merge adjacent/overlapping timeline windows with has_dialogue=True into contiguous regions."""
    regions = []
    current_start = None
    current_end = None

    for entry in timeline:
        if entry.get('has_dialogue', False):
            if current_start is None:
                current_start = entry['start']
                current_end = entry['end']
            elif entry['start'] <= current_end:
                # Overlapping or adjacent - extend
                current_end = max(current_end, entry['end'])
            else:
                # Gap - save previous region, start new one
                regions.append({'start': current_start, 'end': current_end})
                current_start = entry['start']
                current_end = entry['end']
        else:
            if current_start is not None:
                regions.append({'start': current_start, 'end': current_end})
                current_start = None
                current_end = None

    # Don't forget last region
    if current_start is not None:
        regions.append({'start': current_start, 'end': current_end})

    return regions


@torch.no_grad()
def run_phase2_temporal(trainer, mix_paths, num_workers=16, batch_size=64):
    """
    Phase 2: Load raw latents for mix files and run temporal sliding window inference.
    Returns dict: path -> {timeline, detected_instruments}
    """
    logging.info(f"Phase 2: Temporal inference on {len(mix_paths)} mix files...")

    # Load raw latents in parallel
    raw_latents = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_load_raw_latent, p): p for p in mix_paths}
        loaded = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                path, latent = result
                raw_latents[path] = latent
                loaded += 1
            if loaded % 500 == 0 and loaded > 0:
                logging.info(f"  Raw latents loaded: {loaded}/{len(mix_paths)}")

    logging.info(f"  Loaded {len(raw_latents)} raw latents for temporal analysis")

    # Run temporal inference on each
    temporal_results = {}
    done = 0
    for path, latent in raw_latents.items():
        timeline, detected = run_temporal_inference(trainer, latent)

        # Merge adjacent dialogue windows into dialogue_regions
        dialogue_regions = _merge_dialogue_regions(timeline)

        temporal_results[path] = {
            'timeline': timeline,
            'detected_instruments': detected,
            'dialogue_regions': dialogue_regions,
        }
        done += 1
        if done % 500 == 0:
            logging.info(f"  Temporal inference: {done}/{len(raw_latents)}")

    logging.info(f"Phase 2 done: {len(temporal_results)} files with temporal predictions")
    return temporal_results


def build_output(predictions, entries, ckpt_name):
    """Build the output JSON matching monitor_service.py format."""
    # Build lookup for manifest info
    manifest_lookup = {e['path']: e for e in entries}

    # Categorize predictions
    label_dist = Counter()
    multilabel_count = 0
    flagged_count = 0
    mix_count = 0
    bleed_count = 0
    dialogue_count = 0
    dialogue_region_count = 0
    source_counts = {'unlabeled': 0, 'labeled': 0, 'mix': 0}

    for pred in predictions:
        path = pred['path']
        manifest_info = manifest_lookup.get(path, {})
        manifest_group = manifest_info.get('manifest_group', 'undefined')

        # Annotate with manifest info
        pred['manifest_group'] = manifest_group
        pred['matches_manifest'] = (
            pred['predicted_group'] == manifest_group
            or manifest_group in ('undefined', 'unknown', '')
        )
        pred['filename'] = os.path.basename(path)

        # Flag mismatches on labeled entries
        pred['flagged'] = (
            manifest_group not in ('undefined', 'unknown', '')
            and pred['predicted_group'] != manifest_group
        )

        # Category
        if manifest_group in ('undefined', 'unknown', ''):
            pred['source_type'] = 'unlabeled'
            source_counts['unlabeled'] += 1
        elif pred.get('is_mix', False):
            pred['source_type'] = 'mix'
            source_counts['mix'] += 1
        else:
            pred['source_type'] = 'labeled'
            source_counts['labeled'] += 1

        label_dist[pred['predicted_group']] += 1

        # Count mix and multilabel
        if pred.get('is_mix', False):
            mix_count += 1
            # For mix files, multilabel = has 2+ detected instruments
            detected = pred.get('detected_instruments', [])
            if len(detected) >= 2:
                multilabel_count += 1
                pred['is_multilabel'] = True
                pred['predicted_labels'] = detected
                pred['num_labels'] = len(detected)
            else:
                pred['is_multilabel'] = False
                pred['predicted_labels'] = detected or [pred['predicted_group']]
                pred['num_labels'] = len(pred['predicted_labels'])
        else:
            # Isolated: single instrument only
            pred['is_multilabel'] = False
            pred['predicted_labels'] = [pred['predicted_group']]
            pred['num_labels'] = 1

        if pred['flagged']:
            flagged_count += 1

        # Count bleed and dialogue
        if pred.get('has_bleed', False):
            bleed_count += 1
        if pred.get('is_dialogue', False):
            dialogue_count += 1
        dialogue_region_count += len(pred.get('dialogue_regions', []))

    output = {
        'model': 'unified_classifier',
        'checkpoint': ckpt_name,
        'generated_at': datetime.now().isoformat(),
        'total': len(predictions),
        'multilabel_count': multilabel_count,
        'flagged_count': flagged_count,
        'mix_count': mix_count,
        'bleed_count': bleed_count,
        'dialogue_count': dialogue_count,
        'dialogue_region_count': dialogue_region_count,
        'label_distribution': dict(sorted(label_dist.items())),
        'source_counts': source_counts,
        'summary': {
            'total': len(predictions),
            'high_confidence': sum(1 for p in predictions if p['confidence'] >= 0.85),
            'medium_confidence': sum(1 for p in predictions if 0.65 <= p['confidence'] < 0.85),
            'low_confidence': sum(1 for p in predictions if p['confidence'] < 0.65),
            'flagged': flagged_count,
            'multilabel': multilabel_count,
            'by_class': dict(sorted(label_dist.items())),
            'mix_files': mix_count,
            'isolated_files': len(predictions) - mix_count,
            'bleed_detected': bleed_count,
            'dialogue_detected': dialogue_count,
            'dialogue_regions': dialogue_region_count,
        },
        'predictions': predictions,
    }

    return output


def main():
    parser = argparse.ArgumentParser(description='Generate unified classifier predictions for review')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (default: unified_model_refined_best.pt or unified_model_best.pt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output predictions.json path (default: OUTPUT_DIR/predictions.json)')
    parser.add_argument('--manifest', type=str, default=str(MANIFEST_PATH),
                        help='Manifest path')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of threads for parallel latent loading')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mix-threshold', type=float, default=0.5,
                        help='Mix detection threshold (default: 0.5)')
    parser.add_argument('--skip-temporal', action='store_true',
                        help='Skip Phase 2 temporal inference (faster, no timeline)')
    args = parser.parse_args()

    # Find best checkpoint
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        candidates = [
            OUTPUT_DIR / 'unified_model_refined_best.pt',
            OUTPUT_DIR / 'unified_model_best.pt',
            OUTPUT_DIR / 'unified_model_final.pt',
        ]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            logging.error("No checkpoint found!")
            sys.exit(1)

    logging.info(f"Using checkpoint: {ckpt}")
    logging.info(f"Manifest: {args.manifest}")
    logging.info(f"Workers: {args.workers}")
    logging.info(f"Mix threshold: {args.mix_threshold}")

    # Load model
    trainer = load_model(ckpt, device=args.device)

    # Load manifest entries
    entries = load_all_entries(Path(args.manifest))

    # Phase 1: Parallel latent loading (I/O bound - threads) + whole-file inference
    features_list = load_features_parallel(entries, num_workers=args.workers)
    predictions = run_phase1_predictions(trainer, features_list, batch_size=args.batch_size)

    # Identify mix files from Phase 1 results
    # Use both model mix_detector AND manifest is_mix flag
    entry_lookup = {e['path']: e for e in entries}
    mix_paths = []
    for pred in predictions:
        path = pred['path']
        entry = entry_lookup.get(path, {})
        # File is mix if: model says mix OR manifest says mix OR filename suggests mix
        fname_lower = path.lower()
        manifest_is_mix = entry.get('is_mix', False)
        model_is_mix = pred['multi_probability'] > args.mix_threshold
        filename_is_mix = any(kw in fname_lower for kw in [
            '_mix', '/mix', 'mix.', 'mix_', 'master_', 'bounce', 'full_track',
        ])

        if model_is_mix or manifest_is_mix or filename_is_mix:
            pred['is_mix'] = True
            mix_paths.append(path)
        else:
            pred['is_mix'] = False

    logging.info(f"Detected {len(mix_paths)} mix files for temporal analysis")

    # Phase 2: Temporal sliding window on mix files
    if not args.skip_temporal and mix_paths:
        temporal_results = run_phase2_temporal(
            trainer, mix_paths,
            num_workers=args.workers,
            batch_size=args.batch_size,
        )

        # Merge temporal results into predictions
        for pred in predictions:
            path = pred['path']
            if path in temporal_results:
                tr = temporal_results[path]
                pred['timeline'] = tr['timeline']
                pred['detected_instruments'] = tr['detected_instruments']
                pred['dialogue_regions'] = tr['dialogue_regions']
    else:
        if args.skip_temporal:
            logging.info("Skipping Phase 2 temporal inference (--skip-temporal)")
        # No temporal data for mix files - just use group prediction
        for pred in predictions:
            if pred.get('is_mix'):
                pred['detected_instruments'] = [pred['predicted_group']]

    # Build output
    output = build_output(predictions, entries, ckpt.name)

    # Save
    output_path = Path(args.output) if args.output else OUTPUT_DIR / 'predictions.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(orjson.dumps(output, option=orjson.OPT_INDENT_2))

    logging.info(f"Saved {len(predictions)} predictions to {output_path}")
    logging.info(f"  High confidence (>=0.85): {output['summary']['high_confidence']}")
    logging.info(f"  Medium (0.65-0.85): {output['summary']['medium_confidence']}")
    logging.info(f"  Low (<0.65): {output['summary']['low_confidence']}")
    logging.info(f"  Flagged mismatches: {output['flagged_count']}")
    logging.info(f"  Mix files: {output['mix_count']}")
    logging.info(f"  Multi-instrument: {output['multilabel_count']}")
    logging.info(f"  Bleed detected: {output['bleed_count']}")
    logging.info(f"  Dialogue detected: {output['dialogue_count']}")
    logging.info(f"  Dialogue regions: {output['dialogue_region_count']}")
    logging.info(f"  Sources: {output['source_counts']}")
    logging.info(f"  Label distribution: {output['summary']['by_class']}")


if __name__ == '__main__':
    main()
