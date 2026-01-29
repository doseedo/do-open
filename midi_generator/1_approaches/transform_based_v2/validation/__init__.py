"""
Validation module for transform-based MIDI encoding.

Implements 6 validation steps:
1. Decoder - reconstruct notes from transform-relative encoding
2. Round-trip test - verify encode(decode(x)) == x
3. Train/test split - separate corpus for generalization testing
4. Generalization evaluation - test vocabulary on unseen files
5. Pattern inspection - analyze musical meaningfulness
6. Cross-piece analysis - analyze pattern sharing across pieces
"""

from .decoder import ValidationDecoder, decode_encoding, DecodedNote
from .round_trip import RoundTripTest, run_round_trip_test
from .train_test_split import create_train_test_split, TrainTestSplit
from .generalization import GeneralizationEvaluator, evaluate_generalization
from .pattern_inspection import PatternInspector, inspect_patterns
from .cross_piece import CrossPieceAnalyzer, analyze_cross_piece_sharing

__all__ = [
    'ValidationDecoder',
    'decode_encoding',
    'DecodedNote',
    'RoundTripTest',
    'run_round_trip_test',
    'create_train_test_split',
    'TrainTestSplit',
    'GeneralizationEvaluator',
    'evaluate_generalization',
    'PatternInspector',
    'inspect_patterns',
    'CrossPieceAnalyzer',
    'analyze_cross_piece_sharing',
]
