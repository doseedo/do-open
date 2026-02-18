#!/usr/bin/env python3
"""
Iterative Disagreement Mining — Self-correcting manifest purification.

Each iteration:
  1. Apply accumulated corrections to master_manifest.json
  2. Retrain classifiers on corrected labels (calls train_classifiers.py)
  3. Run inference on ALL labeled data (features extracted once, reused)
  4. Find high-confidence disagreements (group + mix classifiers must agree)
  5. Add new corrections to accumulator
  6. Stop when corrections stabilize (<5% of previous iteration)

Features are extracted ONCE and cached — only models change between iterations.
Manual corrections (source='manual') are never overridden.

Output: disagreements.json with full audit trail for review.

Usage:
  python3 disagreement_mining.py --device cuda --workers 8
  python3 disagreement_mining.py --device cuda --iterations 4 --min-confidence 0.9
"""

import argparse
import logging
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import orjson
import torch
import torch.nn as nn

# ===================== PATHS =====================

MASTER_MANIFEST = Path("/home/arlo/gcs-bucket/Manifests/master_manifest.json")
MASTER_BACKUP = Path("/home/arlo/gcs-bucket/Manifests/master_manifest.pre_mining.json")
FORMAT_MANIFEST = Path("/home/arlo/gcs-bucket/Manifests/format_manifest_clean.json")
LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
AUDIO_ROOT = Path("/home/arlo/gcs-bucket")

MIX_MODEL_PATH = Path("/home/arlo/Data/mix_classifier/mix_classifier.pt")
GROUP_MODEL_PATH = Path("/home/arlo/Data/latent_classifier/model.pt")

TRAIN_SCRIPT = Path("/home/arlo/Data/train_classifiers.py")
OUTPUT_PATH = Path("/home/arlo/gcs-bucket/Manifests/disagreements.json")

# ===================== CONFIG =====================

LATENT_SILENT_THRESHOLD = 0.01

# Groups to skip (not trainable instrument classes)
SKIP_GROUPS = {'undefined', 'mix', 'junk', 'silent', 'dialogue', 'ensemble'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')


# ===================== MODELS (match train_classifiers.py) =====================

class GroupClassifier(nn.Module):
    def __init__(self, input_dim=384, num_classes=10, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MixClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=8):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, input_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.transform(x))


# ===================== FEATURE EXTRACTION =====================

def audio_path_to_latent_path(audio_path: str):
    audio_path = Path(audio_path)
    try:
        rel_path = audio_path.relative_to(AUDIO_ROOT)
    except ValueError:
        parts = audio_path.parts
        for prefix in ('protools', 'protoolsA'):
            if prefix in parts:
                idx = parts.index(prefix)
                rel_path = Path(*parts[idx:])
                break
        else:
            if 'gcs-bucket' in parts:
                idx = parts.index('gcs-bucket')
                rel_path = Path(*parts[idx + 1:])
            else:
                rel_path = audio_path

    stem = rel_path.with_suffix('')
    dcae_path = LATENTS_ROOT / f"{stem}.dcae.pt"
    pt_path = LATENTS_ROOT / f"{stem}.pt"

    if dcae_path.exists():
        return dcae_path
    elif pt_path.exists():
        return pt_path
    return None


def load_and_pool(audio_path: str):
    latent_path = audio_path_to_latent_path(audio_path)
    if latent_path is None:
        return None
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            latent = data.get('latents', data.get('latent'))
        else:
            latent = data

        if latent is None or latent.numel() == 0:
            return None

        if latent.shape[-1] > 1:
            energy = torch.sqrt((latent ** 2).mean(dim=(0, 1)))
            mask = energy > LATENT_SILENT_THRESHOLD
            if mask.sum() > 0:
                latent = latent[:, :, mask]

        pools = [
            latent.mean(dim=-1),
            latent.std(dim=-1) if latent.shape[-1] > 1 else torch.zeros_like(latent.mean(dim=-1)),
            latent.max(dim=-1)[0],
        ]
        return torch.stack(pools, dim=-1).flatten()
    except Exception:
        return None


# ===================== CORE FUNCTIONS =====================

def load_models(device):
    """Load mix and group classifier checkpoints."""
    ckpt = torch.load(MIX_MODEL_PATH, map_location='cpu', weights_only=False)
    mix_model = MixClassifier(ckpt['input_dim'], ckpt.get('hidden_dim', 256), ckpt['num_classes'])
    mix_model.load_state_dict(ckpt['model_state'])
    mix_model.to(device).eval()
    mix_info = {
        'model': mix_model,
        'mean': ckpt['X_mean'].to(device),
        'std': ckpt['X_std'].to(device),
        'classes': ckpt['target_classes'],
    }

    ckpt = torch.load(GROUP_MODEL_PATH, map_location='cpu', weights_only=False)
    group_model = GroupClassifier(ckpt['input_dim'], ckpt['num_classes'], ckpt.get('hidden_dim', 256))
    group_model.load_state_dict(ckpt['model_state'])
    group_model.to(device).eval()
    group_info = {
        'model': group_model,
        'mean': ckpt['mean'].to(device),
        'std': ckpt['std'].to(device),
        'classes': list(ckpt['label_encoder_classes']),
    }

    return mix_info, group_info


def run_inference(features, valid_paths, current_labels, mix_info, group_info,
                  device, batch_size, min_confidence, min_mix_cross):
    """Run both classifiers and find disagreements."""
    n = len(features)

    # Group classifier
    X_group = (features.to(device) - group_info['mean']) / group_info['std']
    group_probs_all = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_group[i:i + batch_size]
            logits = group_info['model'](batch)
            probs = torch.softmax(logits, dim=1).cpu()
            group_probs_all.append(probs)
    group_probs_all = torch.cat(group_probs_all)

    # Mix classifier
    X_mix = (features.to(device) - mix_info['mean']) / mix_info['std']
    mix_probs_all = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_mix[i:i + batch_size]
            logits = mix_info['model'](batch)
            probs = torch.sigmoid(logits).cpu()
            mix_probs_all.append(probs)
    mix_probs_all = torch.cat(mix_probs_all)

    group_classes = group_info['classes']
    mix_classes = mix_info['classes']

    disagreements = []
    stats = Counter()
    confusion = defaultdict(Counter)

    for i in range(n):
        path = valid_paths[i]
        current_group = current_labels[path]

        # Group classifier prediction
        group_probs = group_probs_all[i]
        top_idx = group_probs.argmax().item()
        classifier_group = group_classes[top_idx]
        classifier_conf = group_probs[top_idx].item()

        # Runner-up
        sorted_idx = group_probs.argsort(descending=True)
        runner_up = group_classes[sorted_idx[1].item()]
        runner_up_conf = group_probs[sorted_idx[1].item()].item()

        # Mix classifier scores
        mix_probs = mix_probs_all[i]
        mix_scores = {cls: round(mix_probs[j].item(), 4)
                      for j, cls in enumerate(mix_classes)}

        stats['total'] += 1

        if classifier_group == current_group:
            stats['agree'] += 1
            continue

        stats['disagree'] += 1
        confusion[current_group][classifier_group] += 1

        if classifier_conf < min_confidence:
            stats['low_confidence_skip'] += 1
            continue

        # Dual agreement check
        mix_supports = (classifier_group in mix_scores and
                        mix_scores[classifier_group] >= min_mix_cross)
        mix_contradicts = (current_group in mix_scores and
                           mix_scores[current_group] < 0.2)
        dual_agree = mix_supports or mix_contradicts

        if not dual_agree:
            stats['no_dual_agreement'] += 1
            continue

        stats['flagged'] += 1
        disagreements.append({
            'path': path,
            'filename': os.path.basename(path),
            'from_group': current_group,
            'to_group': classifier_group,
            'classifier_confidence': round(classifier_conf, 4),
            'runner_up': runner_up,
            'runner_up_confidence': round(runner_up_conf, 4),
            'mix_scores': mix_scores,
            'mix_supports': mix_supports,
            'mix_contradicts': mix_contradicts,
        })

    return disagreements, stats, confusion


def apply_corrections_to_manifest(corrections, manifest_data):
    """Apply corrections dict to manifest entries in-place."""
    entries = manifest_data['entries']
    applied = 0
    for path, new_group in corrections.items():
        if path in entries:
            entry = entries[path]
            # Never override manual corrections
            if entry.get('source') == 'manual':
                continue
            if entry['group'] != new_group:
                entry['group'] = new_group
                entry['source'] = 'mining'
                applied += 1
    return applied


def retrain_classifiers(device, workers, epochs=20):
    """Call train_classifiers.py as subprocess."""
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        '--stage', 'all',
        '--epochs', str(epochs),
        '--device', device,
        '--workers', str(workers),
    ]
    logging.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"  Training failed:\n{result.stderr[-2000:]}")
        return False

    # Extract summary from output
    for line in result.stdout.split('\n'):
        if any(k in line for k in ['Mix:', 'Group:', 'Subgroup/']):
            logging.info(f"    {line.strip()}")
    return True


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description='Iterative disagreement mining')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs per iteration (lower = faster)')
    parser.add_argument('--min-confidence', type=float, default=0.85)
    parser.add_argument('--min-mix-cross', type=float, default=0.6)
    parser.add_argument('--convergence-ratio', type=float, default=0.05,
                        help='Stop if new corrections < this ratio of iteration 1')
    parser.add_argument('--output', type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    logging.info(f"Max iterations: {args.iterations}")
    logging.info(f"Min confidence: {args.min_confidence}")
    logging.info(f"Min mix cross-check: {args.min_mix_cross}")
    logging.info(f"Convergence ratio: {args.convergence_ratio}")

    # ── Backup original manifest ──
    if not MASTER_BACKUP.exists():
        logging.info(f"Backing up master manifest to {MASTER_BACKUP}")
        with open(MASTER_MANIFEST, 'rb') as f:
            backup_data = f.read()
        with open(MASTER_BACKUP, 'wb') as f:
            f.write(backup_data)
        del backup_data
    else:
        logging.info(f"Backup already exists: {MASTER_BACKUP}")

    # ── Load manifest ──
    logging.info("Loading master manifest...")
    with open(MASTER_MANIFEST, 'rb') as f:
        manifest_data = orjson.loads(f.read())

    # ── Build list of labeled entries with latents ──
    logging.info("Loading format manifest for latent availability...")
    with open(FORMAT_MANIFEST, 'rb') as f:
        fm = orjson.loads(f.read())
    latent_rel_paths = {e['path'] for e in fm['entries'] if e.get('has_latent')}
    del fm

    audio_root_str = str(AUDIO_ROOT) + '/'
    labeled_paths = []
    for path, entry in manifest_data['entries'].items():
        if entry['group'] in SKIP_GROUPS:
            continue
        # Skip manual corrections — they're ground truth
        if entry.get('source') == 'manual':
            continue
        rel = path[len(audio_root_str):] if path.startswith(audio_root_str) else path
        if rel in latent_rel_paths:
            labeled_paths.append(path)
    del latent_rel_paths

    logging.info(f"  Labeled entries with latents (non-manual): {len(labeled_paths):,}")

    # ── Extract features ONCE ──
    logging.info(f"\nExtracting features for {len(labeled_paths):,} files (one-time)...")
    idx_map = {p: i for i, p in enumerate(labeled_paths)}
    features_list = []
    valid_indices = []
    processed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(load_and_pool, p): i
                   for i, p in enumerate(labeled_paths)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                feat = future.result()
                if feat is not None:
                    features_list.append((idx, feat))
                else:
                    failed += 1
            except Exception:
                failed += 1
            processed += 1
            if processed % 10000 == 0:
                logging.info(f"  Features: {processed:,}/{len(labeled_paths):,} "
                             f"({len(features_list):,} valid, {failed:,} failed)")

    logging.info(f"  Features done: {len(features_list):,} valid, {failed:,} failed")

    # Sort and build tensors
    features_list.sort(key=lambda x: x[0])
    valid_paths = [labeled_paths[idx] for idx, _ in features_list]
    features = torch.stack([f for _, f in features_list])
    del features_list
    logging.info(f"  Feature tensor: {features.shape}")

    # ── Iterative loop ──
    all_corrections = {}  # path -> new_group
    iteration_log = []
    first_iter_count = None

    for iteration in range(1, args.iterations + 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"ITERATION {iteration}/{args.iterations}")
        logging.info(f"{'='*60}")

        # 1. Retrain classifiers on current manifest
        logging.info("\n[1/3] Retraining classifiers...")
        success = retrain_classifiers(device, args.workers, args.epochs)
        if not success:
            logging.error("Training failed, stopping.")
            break

        # 2. Load fresh models
        logging.info("\n[2/3] Loading retrained models...")
        mix_info, group_info = load_models(device)
        logging.info(f"  Group classes: {group_info['classes']}")
        logging.info(f"  Mix classes: {mix_info['classes']}")

        # 3. Build current labels dict (manifest labels + accumulated corrections)
        current_labels = {}
        for path in valid_paths:
            entry = manifest_data['entries'].get(path, {})
            current_labels[path] = entry.get('group', 'undefined')

        # 4. Run inference and find disagreements
        logging.info("\n[3/3] Running inference and finding disagreements...")
        disagreements, stats, confusion = run_inference(
            features, valid_paths, current_labels,
            mix_info, group_info, device, args.batch_size,
            args.min_confidence, args.min_mix_cross,
        )

        # Apply new corrections
        new_corrections = {}
        for d in disagreements:
            path = d['path']
            # Don't flip something we already corrected back
            if path in all_corrections and all_corrections[path] == d['from_group']:
                continue
            new_corrections[path] = d['to_group']

        logging.info(f"\n  Total analyzed:    {stats['total']:,}")
        logging.info(f"  Agree:             {stats['agree']:,} ({stats['agree']/max(stats['total'],1):.1%})")
        logging.info(f"  Disagree:          {stats['disagree']:,}")
        logging.info(f"  New corrections:   {len(new_corrections):,}")

        # Top flows
        flows = []
        for fg, targets in confusion.items():
            for cg, count in targets.items():
                flows.append((count, fg, cg))
        flows.sort(reverse=True)
        logging.info(f"\n  Top correction flows:")
        for count, fg, cg in flows[:10]:
            logging.info(f"    {fg:12s} → {cg:12s}: {count:,}")

        # Track iteration
        iter_info = {
            'iteration': iteration,
            'stats': dict(stats),
            'new_corrections': len(new_corrections),
            'total_corrections': len(all_corrections) + len(new_corrections),
            'top_flows': [{'from': fg, 'to': cg, 'count': c} for c, fg, cg in flows[:20]],
            'sample_corrections': [
                {'path': d['path'], 'filename': d['filename'],
                 'from': d['from_group'], 'to': d['to_group'],
                 'confidence': d['classifier_confidence']}
                for d in sorted(disagreements, key=lambda x: -x['classifier_confidence'])[:20]
            ],
        }
        iteration_log.append(iter_info)

        if not new_corrections:
            logging.info("\n  No new corrections — converged!")
            break

        # Convergence check
        if first_iter_count is None:
            first_iter_count = len(new_corrections)
        elif len(new_corrections) < first_iter_count * args.convergence_ratio:
            logging.info(f"\n  Converged: {len(new_corrections)} < "
                         f"{first_iter_count * args.convergence_ratio:.0f} "
                         f"({args.convergence_ratio:.0%} of iteration 1)")
            # Still apply these final corrections
            all_corrections.update(new_corrections)
            applied = apply_corrections_to_manifest(new_corrections, manifest_data)
            logging.info(f"  Applied {applied} final corrections to manifest")
            # Write manifest
            with open(MASTER_MANIFEST, 'wb') as f:
                f.write(orjson.dumps(manifest_data, option=orjson.OPT_INDENT_2))
            break

        # Apply corrections
        all_corrections.update(new_corrections)
        applied = apply_corrections_to_manifest(new_corrections, manifest_data)
        logging.info(f"\n  Applied {applied} corrections to manifest")

        # Write updated manifest for next training iteration
        manifest_data['stats'] = manifest_data.get('stats', {})
        manifest_data['stats']['mining_iteration'] = iteration
        manifest_data['stats']['mining_corrections'] = len(all_corrections)
        with open(MASTER_MANIFEST, 'wb') as f:
            f.write(orjson.dumps(manifest_data, option=orjson.OPT_INDENT_2))
        logging.info(f"  Manifest updated ({MASTER_MANIFEST})")

    # ── Final summary ──
    logging.info(f"\n{'='*60}")
    logging.info("MINING COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"Iterations run: {len(iteration_log)}")
    logging.info(f"Total corrections: {len(all_corrections):,}")

    # Per-iteration summary
    for it in iteration_log:
        logging.info(f"  Iter {it['iteration']}: {it['new_corrections']:,} new, "
                     f"{it['total_corrections']:,} total, "
                     f"agree={it['stats']['agree']/max(it['stats']['total'],1):.1%}")

    # Correction flow summary
    flow_totals = defaultdict(Counter)
    for path, new_group in all_corrections.items():
        entry = manifest_data['entries'].get(path, {})
        orig = entry.get('original_group', entry.get('group', '?'))
        flow_totals[orig][new_group] += 1

    logging.info(f"\nAll correction flows (original → mined):")
    all_flows = []
    for fg, targets in flow_totals.items():
        for cg, count in targets.items():
            all_flows.append((count, fg, cg))
    all_flows.sort(reverse=True)
    for count, fg, cg in all_flows[:25]:
        logging.info(f"  {fg:15s} → {cg:15s}: {count:,}")

    # ── Save output ──
    output = {
        'generated_at': datetime.now().isoformat(),
        'config': {
            'iterations': args.iterations,
            'min_confidence': args.min_confidence,
            'min_mix_cross': args.min_mix_cross,
            'convergence_ratio': args.convergence_ratio,
            'epochs_per_iter': args.epochs,
        },
        'summary': {
            'iterations_run': len(iteration_log),
            'total_corrections': len(all_corrections),
        },
        'iteration_log': iteration_log,
        'all_corrections': [
            {
                'path': path,
                'filename': os.path.basename(path),
                'original_group': manifest_data['entries'].get(path, {}).get('original_group', '?'),
                'corrected_to': new_group,
            }
            for path, new_group in sorted(all_corrections.items())
        ],
    }

    output_path = Path(args.output)
    with open(output_path, 'wb') as f:
        f.write(orjson.dumps(output, option=orjson.OPT_INDENT_2))

    logging.info(f"\nSaved {len(all_corrections):,} corrections to {output_path}")
    logging.info(f"Original manifest backed up at: {MASTER_BACKUP}")
    logging.info("Done! Review disagreements.json before finalizing.")


if __name__ == "__main__":
    main()
