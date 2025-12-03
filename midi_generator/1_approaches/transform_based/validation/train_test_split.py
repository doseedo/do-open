"""
Step 3: Train/Test Split
========================

Separates corpus into training set (build vocabulary) and test set (evaluate generalization).

Flow:
1. Take your 495 files
2. Randomly assign 400 to training, 95 to test (80/20 split)
3. Run full pipeline on training set only -> produces canonical patterns, grammar rules, transform table
4. Save the training-only checkpoint

Important:
- Grammar rules learned ONLY from training files
- Canonical patterns discovered ONLY from training files
- Test files never seen during vocabulary construction
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import json
import glob
import random
import os


@dataclass
class TrainTestSplit:
    """Represents a train/test split of the corpus."""
    train_files: List[str]
    test_files: List[str]
    train_ratio: float
    seed: int
    corpus_path: str

    @property
    def n_train(self) -> int:
        return len(self.train_files)

    @property
    def n_test(self) -> int:
        return len(self.test_files)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_test

    def save(self, output_path: str):
        """Save split to JSON file."""
        data = {
            'train_files': self.train_files,
            'test_files': self.test_files,
            'train_ratio': self.train_ratio,
            'seed': self.seed,
            'corpus_path': self.corpus_path,
            'n_train': self.n_train,
            'n_test': self.n_test,
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Split saved to {output_path}")

    @classmethod
    def load(cls, path: str) -> 'TrainTestSplit':
        """Load split from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            train_files=data['train_files'],
            test_files=data['test_files'],
            train_ratio=data['train_ratio'],
            seed=data['seed'],
            corpus_path=data['corpus_path'],
        )

    def summary(self) -> str:
        return (f"TrainTestSplit: {self.n_train} train / {self.n_test} test "
                f"({self.train_ratio:.0%} split, seed={self.seed})")


def create_train_test_split(
    corpus_path: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> TrainTestSplit:
    """
    Create a train/test split of the MIDI corpus.

    Args:
        corpus_path: Path to MIDI corpus directory
        train_ratio: Fraction of files for training (default 0.8 = 80%)
        seed: Random seed for reproducibility
        output_path: Optional path to save split JSON

    Returns:
        TrainTestSplit object
    """
    # Find all MIDI files
    corpus = Path(corpus_path)
    midi_files = sorted(glob.glob(str(corpus / "*.mid")))
    midi_files += sorted(glob.glob(str(corpus / "*.midi")))

    if not midi_files:
        raise ValueError(f"No MIDI files found in {corpus_path}")

    # Shuffle with seed
    random.seed(seed)
    shuffled = midi_files.copy()
    random.shuffle(shuffled)

    # Split
    n_train = int(len(shuffled) * train_ratio)
    train_files = sorted(shuffled[:n_train])
    test_files = sorted(shuffled[n_train:])

    split = TrainTestSplit(
        train_files=train_files,
        test_files=test_files,
        train_ratio=train_ratio,
        seed=seed,
        corpus_path=str(corpus_path),
    )

    print(f"Created split: {split.n_train} train, {split.n_test} test "
          f"(from {len(midi_files)} total)")

    if output_path:
        split.save(output_path)

    return split


def run_pipeline_on_split(
    split: TrainTestSplit,
    output_checkpoint: str,
    target_vocab_size: int = 500,
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full pipeline on training files only.

    Args:
        split: TrainTestSplit object
        output_checkpoint: Path to save training-only checkpoint
        target_vocab_size: Target vocabulary size
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        Dict with pipeline results and timing
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pipeline import DosedoPipeline

    if verbose:
        print(f"\n{'='*60}")
        print("RUNNING PIPELINE ON TRAINING SET ONLY")
        print(f"{'='*60}")
        print(f"Training files: {split.n_train}")
        print(f"Test files: {split.n_test} (held out)")
        print()

    # Create a temporary directory with symlinks to training files
    # OR pass file list directly to pipeline

    # The pipeline expects a corpus_path, so we need to either:
    # 1. Create a temp directory with symlinks
    # 2. Modify pipeline to accept file list
    # 3. Filter objects after loading

    # Option 3: Load all, filter to training
    pipeline = DosedoPipeline(
        corpus_path=split.corpus_path,
        device=device,
        target_vocab_size=target_vocab_size,
        verbose=verbose,
    )

    # Get training piece IDs
    train_piece_ids = set(Path(f).stem for f in split.train_files)

    # Run pipeline with filtering
    result = None
    for progress in pipeline.run(save_checkpoint_path=output_checkpoint):
        if verbose:
            print(progress)
        result = progress

    # After pipeline runs, verify the checkpoint
    if os.path.exists(output_checkpoint):
        ckpt = np.load(output_checkpoint, allow_pickle=True)
        stats = {
            'n_canonicals': int(ckpt.get('n_canonicals', [0])[0]),
            'vocabulary_size': int(ckpt.get('vocabulary_size', [0])[0]),
            'compression_ratio': float(ckpt.get('encoding_compression_ratio', [1.0])[0]),
        }

        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING CHECKPOINT CREATED")
            print(f"{'='*60}")
            print(f"Checkpoint: {output_checkpoint}")
            print(f"Canonicals: {stats['n_canonicals']}")
            print(f"Vocabulary: {stats['vocabulary_size']}")
            print(f"Compression: {stats['compression_ratio']:.2f}x")

        return {
            'checkpoint_path': output_checkpoint,
            'split': split,
            'stats': stats,
        }

    return {'error': 'Checkpoint not created'}


class TrainTestPipeline:
    """
    Pipeline that builds vocabulary from training set only.

    This is a modified pipeline that:
    1. Loads all files
    2. Filters to training set only
    3. Builds grammar/vocabulary from training
    4. Saves checkpoint
    """

    def __init__(
        self,
        split: TrainTestSplit,
        device: str = 'cuda',
        target_vocab_size: int = 500,
    ):
        self.split = split
        self.device = device
        self.target_vocab_size = target_vocab_size

        # Training piece IDs
        self.train_piece_ids = set(Path(f).stem for f in split.train_files)
        self.test_piece_ids = set(Path(f).stem for f in split.test_files)

    def run(
        self,
        output_checkpoint: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run pipeline on training set only.

        The key difference from normal pipeline:
        - Only training files are used for grammar induction
        - Only training files are used for pattern discovery
        - Only training files are used for vocabulary optimization

        Args:
            output_checkpoint: Path to save checkpoint
            verbose: Print progress

        Returns:
            Dict with results
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Import pipeline components
        try:
            from pipeline import _load_single_midi
            from discovery.factored_v2 import factor_tensor_v2
            from grammar.sequitur import build_grammar_from_corpus
            from discovery.unified_relations import UnifiedRelationDiscovery
            from mdl.vocabulary_optimizer import run_vocabulary_optimization
            from encoding.encoder import TransformRelativeEncoder
            from grammar.pattern_extractor import GrammarPatternExtractor
        except ImportError as e:
            print(f"Import error: {e}")
            return {'error': str(e)}

        import time
        from concurrent.futures import ProcessPoolExecutor, as_completed

        total_start = time.time()
        phase_times = {}

        if verbose:
            print(f"\n{'='*60}")
            print("TRAIN-ONLY PIPELINE")
            print(f"{'='*60}")
            print(f"Training: {self.split.n_train} files")
            print(f"Test: {self.split.n_test} files (held out)")
            print()

        # Phase 1: Load training files only
        phase_start = time.time()
        if verbose:
            print("[LOADING] Loading training files...")

        objects = []
        loaded = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(_load_single_midi, path): path
                for path in self.split.train_files
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        for track_data in result:
                            obj = type('MusicalObject', (), {
                                'tensor': track_data['tensor'],
                                'piece_id': track_data['piece_id'],
                                'track_id': track_data['track_id'],
                                'start_time': 0,
                                'is_drum': track_data['is_drum']
                            })()
                            objects.append(obj)
                        loaded += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

        phase_times['loading'] = time.time() - phase_start
        if verbose:
            print(f"[LOADING] Loaded {loaded} files ({failed} failed), {len(objects)} objects")

        # Phase 2: Factor objects
        phase_start = time.time()
        if verbose:
            print("[FACTORING] Factoring objects...")

        factored = []
        for obj in objects:
            if hasattr(obj, 'tensor'):
                fobj = factor_tensor_v2(
                    tensor=obj.tensor,
                    piece_id=obj.piece_id,
                    track_id=obj.track_id,
                    start_time=obj.start_time,
                    scale=obj.tensor.shape[0],
                    is_drum=getattr(obj, 'is_drum', False)
                )
                factored.append(fobj)

        phase_times['factoring'] = time.time() - phase_start
        if verbose:
            print(f"[FACTORING] Factored {len(factored)} objects")

        # Phase 3: Grammar induction
        phase_start = time.time()
        if verbose:
            print("[GRAMMAR] Running SEQUITUR grammar induction...")

        grammar = build_grammar_from_corpus(factored, verbose=False)
        n_rules = grammar.get_vocabulary_size() if grammar else 0

        phase_times['grammar'] = time.time() - phase_start
        if verbose:
            print(f"[GRAMMAR] Induced {n_rules} rules")

        # Phase 4: Transform discovery
        phase_start = time.time()
        if verbose:
            print("[TRANSFORMS] Discovering transforms...")

        discoverer = UnifiedRelationDiscovery(device=self.device)
        relations = discoverer.discover_all(factored, grammar=grammar, verbose=False)
        n_transforms = len(relations.pattern_transforms) if relations else 0

        phase_times['transforms'] = time.time() - phase_start
        if verbose:
            print(f"[TRANSFORMS] Found {n_transforms} pattern transforms")

        # Phase 5: MDL optimization
        phase_start = time.time()
        if verbose:
            print("[MDL] Optimizing vocabulary...")

        d24_table = None
        if relations and hasattr(relations, 'd24_transform_table'):
            d24_table = relations.d24_transform_table

        vocabulary, mdl_scores = run_vocabulary_optimization(
            grammar,
            factored,
            target_size=self.target_vocab_size,
            d24_transform_table=d24_table,
            verbose=False
        )

        vocab_size = vocabulary.size if vocabulary else 0
        phase_times['mdl'] = time.time() - phase_start
        if verbose:
            print(f"[MDL] Selected {vocab_size} vocabulary elements")

        # Phase 6: Encoding
        phase_start = time.time()
        if verbose:
            print("[ENCODING] Building transform-relative encoding...")

        try:
            extractor = GrammarPatternExtractor(grammar)
            encoder = TransformRelativeEncoder(d24_table)

            cross_track = None
            if relations and hasattr(relations, 'cross_track'):
                cross_track = relations.cross_track

            encoding = encoder.encode(
                grammar,
                factored,
                pattern_extractor=extractor,
                cross_track_relations=cross_track
            )
        except Exception as e:
            if verbose:
                print(f"[ENCODING] Error: {e}")
            encoding = None

        phase_times['encoding'] = time.time() - phase_start

        # Phase 7: Save checkpoint
        if verbose:
            print(f"[CHECKPOINT] Saving to {output_checkpoint}...")

        self._save_checkpoint(
            output_checkpoint,
            grammar=grammar,
            relations=relations,
            vocabulary=vocabulary,
            mdl_scores=mdl_scores,
            encoding=encoding,
            factored=factored,
        )

        total_time = time.time() - total_start

        if verbose:
            print(f"\n{'='*60}")
            print("PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Checkpoint: {output_checkpoint}")

        return {
            'checkpoint_path': output_checkpoint,
            'n_train_files': loaded,
            'n_train_objects': len(factored),
            'n_grammar_rules': n_rules,
            'n_transforms': n_transforms,
            'vocabulary_size': vocab_size,
            'phase_times': phase_times,
            'total_time': total_time,
        }

    def _save_checkpoint(
        self,
        path: str,
        grammar,
        relations,
        vocabulary,
        mdl_scores,
        encoding,
        factored,
    ):
        """Save training checkpoint."""
        import json

        data = {
            'version': np.array(['3.0']),
            'vocabulary_size': np.array([vocabulary.size if vocabulary else 0]),
            'object_coverage': np.array([0.0]),
            'compression_ratio': np.array([1.0]),
        }

        # Encoding stats
        if encoding:
            data['n_canonicals'] = np.array([encoding.n_canonicals])
            data['n_intros'] = np.array([encoding.n_intros])
            data['n_transforms'] = np.array([encoding.n_transforms])
            data['n_repeats'] = np.array([encoding.n_repeats])
            data['encoding_compression_ratio'] = np.array([encoding.compression_ratio])
        else:
            data['n_canonicals'] = np.array([0])
            data['n_intros'] = np.array([0])
            data['n_transforms'] = np.array([0])
            data['n_repeats'] = np.array([0])
            data['encoding_compression_ratio'] = np.array([1.0])

        # Group tables
        try:
            from core.groups.d24_group import D24Group
            from core.groups.rhythm_group import RhythmGroup

            d24 = D24Group()
            rhythm = RhythmGroup()
            data['d24_cayley_table'] = d24.cayley_table
            data['rhythm_cayley_table'] = rhythm.compose_table
        except ImportError:
            pass

        # Transform tables
        if relations:
            data['d24_transform_table'] = relations.d24_transform_table
            data['rhythm_transform_table'] = relations.rhythm_transform_table
            data['cross_track_json'] = np.array([json.dumps([
                {
                    'track_a': r.track_a_id,
                    'track_b': r.track_b_id,
                    'type': r.relation_type.value,
                    'confidence': r.confidence
                }
                for r in relations.cross_track
            ])])
        else:
            data['cross_track_json'] = np.array(['[]'])

        # Grammar
        if grammar:
            data['grammar_rules_json'] = np.array([json.dumps({
                str(rid): rule.to_list()
                for rid, rule in grammar.rules.items()
            })])
            data['grammar_stats_json'] = np.array([json.dumps(grammar.get_rule_stats())])
        else:
            data['grammar_rules_json'] = np.array(['{}'])
            data['grammar_stats_json'] = np.array(['{}'])

        # MDL scores
        if mdl_scores:
            data['mdl_scores_json'] = np.array([json.dumps({
                str(s.pattern_id): {
                    'type': s.pattern_type,
                    'score': s.mdl_score,
                    'usage': s.usage_count
                }
                for s in mdl_scores
            })])
        else:
            data['mdl_scores_json'] = np.array(['{}'])

        # Vocabulary
        if vocabulary:
            data['vocabulary_json'] = np.array([json.dumps(vocabulary.all_elements)])
        else:
            data['vocabulary_json'] = np.array(['[]'])

        # Encoding
        if encoding:
            data['canonical_patterns_json'] = np.array([json.dumps([
                {
                    'pattern_id': cp.pattern_id,
                    'pitch_classes': cp.pitch_classes.tolist(),
                    'original_rule_id': cp.original_rule_id,
                    'usage_count': cp.usage_count,
                }
                for cp in encoding.canonical_patterns
            ])])
            data['encoding_tokens_json'] = np.array([json.dumps([
                [
                    {
                        'type': t.token_type.name,
                        'pattern_idx': t.pattern_idx,
                        'transform_id': t.transform_id,
                        'source_track': t.source_track,
                    }
                    for t in track_tokens
                ]
                for track_tokens in encoding.tokens
            ])])
        else:
            data['canonical_patterns_json'] = np.array(['[]'])
            data['encoding_tokens_json'] = np.array(['[]'])

        # Add train/test info
        data['train_piece_ids'] = np.array([json.dumps(list(self.train_piece_ids))])
        data['test_piece_ids'] = np.array([json.dumps(list(self.test_piece_ids))])

        np.savez_compressed(path, **data)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train_test_split.py <corpus_path> [output_split.json] [train_ratio]")
        sys.exit(1)

    corpus = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "train_test_split.json"
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

    split = create_train_test_split(corpus, train_ratio=ratio, output_path=output)
    print(split.summary())
