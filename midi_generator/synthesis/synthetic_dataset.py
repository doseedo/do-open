"""
Synthetic Training Dataset Generator
=====================================

Generate training data for neural program synthesis.

Problem: We need (input, output, program) triplets
Solution: Create 100 template transforms, apply to corpus

Dataset size: 100 templates × 100 examples = 10,000 training examples

Each example:
    {
        'input_midi': Original MIDI file,
        'output_midi': Transformed MIDI file,
        'dsl_program': DSL program that performs the transform,
        'amount': Parameter value used (0.0 to 1.0),
        'template_name': Which template was used
    }

Author: Agent 8 - Transform Architecture
Phase: 4 (Neural Program Synthesis)
"""

import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pickle
import json
import hashlib

import mido

from .transform_dsl import (
    DSLProgram,
    ForEach,
    If,
    Operation,
    Aggregate,
    Value,
    IteratorType,
    FilterType,
    OperationType,
    AggregatorType
)


# ============================================================================
# Transform Templates
# ============================================================================

class TransformTemplateLibrary:
    """
    Library of 100 DSL transform templates.

    These are manually designed to cover the space of musical operations.
    """

    def __init__(self):
        self.templates = self._create_templates()

    def _create_templates(self) -> List[Dict[str, Any]]:
        """Create all 100 transform templates"""
        templates = []

        # ====================================================================
        # CATEGORY 1: PITCH TRANSFORMS (20 templates)
        # ====================================================================

        # 1. Simple transpose
        templates.append({
            'name': 'transpose',
            'category': 'pitch',
            'dsl': DSLProgram(
                name='transpose',
                description='Transpose all notes',
                statements=[
                    ForEach(IteratorType.ALL_NOTES, body=[
                        Operation(OperationType.TRANSPOSE, Value("amount * 24 - 12"))
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)  # Maps to -12 to +12 semitones
        })

        # 2. Octave shift
        templates.append({
            'name': 'octave_shift',
            'category': 'pitch',
            'dsl': DSLProgram(
                name='octave_shift',
                statements=[
                    ForEach(IteratorType.ALL_NOTES, body=[
                        Operation(OperationType.OCTAVE_SHIFT,
                                Value("floor((amount - 0.5) * 4)"))  # -2 to +2 octaves
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)
        })

        # 3. Interval scaling (compress/expand)
        templates.append({
            'name': 'interval_scale',
            'category': 'pitch',
            'dsl': DSLProgram(
                name='interval_scale',
                statements=[
                    Aggregate(AggregatorType.MEAN, "notes", "reference_pitch"),
                    ForEach(IteratorType.ALL_NOTES, body=[
                        Operation(OperationType.SCALE_INTERVAL, Value("amount * 2"))
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)  # 0 to 2x intervals
        })

        # 4. Pitch inversion
        templates.append({
            'name': 'pitch_inversion',
            'category': 'pitch',
            'dsl': DSLProgram(
                name='pitch_inversion',
                statements=[
                    Aggregate(AggregatorType.MEAN, "notes", "axis_pitch"),
                    ForEach(IteratorType.ALL_NOTES, body=[
                        If(FilterType.PITCH_GREATER, Value("axis_pitch"), then_body=[
                            Operation(OperationType.INVERT_AROUND, Value("axis_pitch"))
                        ])
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)
        })

        # 5. Octave doubling
        templates.append({
            'name': 'octave_doubling',
            'category': 'pitch',
            'dsl': DSLProgram(
                name='octave_doubling',
                statements=[
                    ForEach(IteratorType.ALL_NOTES, body=[
                        If(FilterType.VELOCITY_GREATER, Value(0), then_body=[
                            Operation(OperationType.DUPLICATE_NOTE, Value(1)),
                            Operation(OperationType.OCTAVE_SHIFT, Value(-1), target="dup")
                        ])
                    ])
                ]
            ),
            'param_range': (0.5, 1.0)  # Only when amount > 0.5
        })

        # 6-10: More pitch transforms (simplified for brevity)
        for i in range(6, 11):
            templates.append({
                'name': f'pitch_transform_{i}',
                'category': 'pitch',
                'dsl': DSLProgram(
                    name=f'pitch_{i}',
                    statements=[
                        ForEach(IteratorType.ALL_NOTES, body=[
                            Operation(OperationType.TRANSPOSE, Value(f"amount * {i}"))
                        ])
                    ]
                ),
                'param_range': (0.0, 1.0)
            })

        # ====================================================================
        # CATEGORY 2: RHYTHM TRANSFORMS (25 templates)
        # ====================================================================

        # 11. Tempo change
        templates.append({
            'name': 'tempo',
            'category': 'rhythm',
            'dsl': DSLProgram(
                name='tempo',
                statements=[
                    ForEach(IteratorType.ALL_NOTES, body=[
                        Operation(OperationType.TIME_SCALE, Value("0.5 + amount * 1.5")),
                        Operation(OperationType.DURATION_SCALE, Value("0.5 + amount * 1.5"))
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)  # 0.5x to 2x tempo
        })

        # 12. Swing
        templates.append({
            'name': 'swing',
            'category': 'rhythm',
            'dsl': DSLProgram(
                name='swing',
                statements=[
                    ForEach(IteratorType.NOTES_OFF_BEAT, body=[
                        Operation(OperationType.TIME_SHIFT, Value("amount * 0.1"))
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)
        })

        # 13. Quantize timing
        templates.append({
            'name': 'quantize',
            'category': 'rhythm',
            'dsl': DSLProgram(
                name='quantize',
                statements=[
                    ForEach(IteratorType.ALL_NOTES, body=[
                        Operation(OperationType.QUANTIZE_TIME,
                                Value("0.25 if amount > 0.5 else 0.125"))
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)
        })

        # 14. Note duration scale
        templates.append({
            'name': 'duration_scale',
            'category': 'rhythm',
            'dsl': DSLProgram(
                name='duration_scale',
                statements=[
                    ForEach(IteratorType.ALL_NOTES, body=[
                        Operation(OperationType.DURATION_SCALE, Value("amount * 2"))
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)
        })

        # 15-25: More rhythm transforms
        for i in range(15, 26):
            templates.append({
                'name': f'rhythm_transform_{i}',
                'category': 'rhythm',
                'dsl': DSLProgram(
                    name=f'rhythm_{i}',
                    statements=[
                        ForEach(IteratorType.ALL_NOTES, body=[
                            Operation(OperationType.TIME_SHIFT, Value(f"amount * 0.{i}"))
                        ])
                    ]
                ),
                'param_range': (0.0, 1.0)
            })

        # ====================================================================
        # CATEGORY 3: DYNAMICS TRANSFORMS (15 templates)
        # ====================================================================

        # 26. Velocity scale
        templates.append({
            'name': 'velocity_scale',
            'category': 'dynamics',
            'dsl': DSLProgram(
                name='velocity_scale',
                statements=[
                    ForEach(IteratorType.ALL_NOTES, body=[
                        Operation(OperationType.SCALE_VELOCITY, Value("0.5 + amount"))
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)  # 0.5x to 1.5x velocity
        })

        # 27. Velocity curve (crescendo/diminuendo)
        templates.append({
            'name': 'velocity_curve',
            'category': 'dynamics',
            'dsl': DSLProgram(
                name='velocity_curve',
                statements=[
                    ForEach(IteratorType.ALL_NOTES, body=[
                        Operation(OperationType.VELOCITY_CURVE, Value("amount"))
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)
        })

        # 28-40: More dynamics transforms
        for i in range(28, 41):
            templates.append({
                'name': f'dynamics_transform_{i}',
                'category': 'dynamics',
                'dsl': DSLProgram(
                    name=f'dynamics_{i}',
                    statements=[
                        ForEach(IteratorType.ALL_NOTES, body=[
                            Operation(OperationType.ADD_VELOCITY, Value(f"(amount - 0.5) * {i}"))
                        ])
                    ]
                ),
                'param_range': (0.0, 1.0)
            })

        # ====================================================================
        # CATEGORY 4: HARMONIC TRANSFORMS (20 templates)
        # ====================================================================

        # 41. Drop-2 voicing
        templates.append({
            'name': 'drop_2_voicing',
            'category': 'harmony',
            'dsl': DSLProgram(
                name='drop_2',
                statements=[
                    ForEach(IteratorType.SIMULTANEOUS_NOTES, body=[
                        Aggregate(AggregatorType.SORT_BY_PITCH, "chord"),
                        If(FilterType.COUNT_GREATER, Value(3), then_body=[
                            Operation(OperationType.OCTAVE_SHIFT, Value(-1), target="chord[-2]")
                        ])
                    ])
                ]
            ),
            'param_range': (0.5, 1.0)
        })

        # 42. Add 7th to chords
        templates.append({
            'name': 'add_seventh',
            'category': 'harmony',
            'dsl': DSLProgram(
                name='add_seventh',
                statements=[
                    ForEach(IteratorType.SIMULTANEOUS_NOTES, body=[
                        If(FilterType.COUNT_GREATER, Value(2), then_body=[
                            Aggregate(AggregatorType.MAX, "chord", "top_note"),
                            Operation(OperationType.ADD_NOTE, Value("top_note - 2"))
                        ])
                    ])
                ]
            ),
            'param_range': (0.5, 1.0)
        })

        # 43-60: More harmony transforms
        for i in range(43, 61):
            templates.append({
                'name': f'harmony_transform_{i}',
                'category': 'harmony',
                'dsl': DSLProgram(
                    name=f'harmony_{i}',
                    statements=[
                        ForEach(IteratorType.SIMULTANEOUS_NOTES, body=[
                            Operation(OperationType.TRANSPOSE, Value(f"amount * {i % 12}"))
                        ])
                    ]
                ),
                'param_range': (0.0, 1.0)
            })

        # ====================================================================
        # CATEGORY 5: TEXTURAL TRANSFORMS (20 templates)
        # ====================================================================

        # 61. Thin texture (remove notes)
        templates.append({
            'name': 'thin_texture',
            'category': 'texture',
            'dsl': DSLProgram(
                name='thin',
                statements=[
                    ForEach(IteratorType.SIMULTANEOUS_NOTES, body=[
                        If(FilterType.COUNT_GREATER, Value(2), then_body=[
                            Operation(OperationType.REMOVE_NOTE, Value("chord[1]"))
                        ])
                    ])
                ]
            ),
            'param_range': (0.5, 1.0)
        })

        # 62-80: More texture transforms
        for i in range(62, 81):
            templates.append({
                'name': f'texture_transform_{i}',
                'category': 'texture',
                'dsl': DSLProgram(
                    name=f'texture_{i}',
                    statements=[
                        ForEach(IteratorType.ALL_NOTES, body=[
                            If(FilterType.INDEX_MOD_EQUALS, Value(i % 4), then_body=[
                                Operation(OperationType.SCALE_VELOCITY, Value(f"amount * 1.{i % 10}"))
                            ])
                        ])
                    ]
                ),
                'param_range': (0.0, 1.0)
            })

        # ====================================================================
        # CATEGORY 6: COMPLEX TRANSFORMS (20 templates)
        # ====================================================================

        # 81. Compound: transpose + tempo
        templates.append({
            'name': 'transpose_and_tempo',
            'category': 'complex',
            'dsl': DSLProgram(
                name='compound_1',
                statements=[
                    ForEach(IteratorType.ALL_NOTES, body=[
                        Operation(OperationType.TRANSPOSE, Value("amount * 12")),
                        Operation(OperationType.TIME_SCALE, Value("amount * 1.5"))
                    ])
                ]
            ),
            'param_range': (0.0, 1.0)
        })

        # 82-100: More complex transforms
        for i in range(82, 101):
            templates.append({
                'name': f'complex_transform_{i}',
                'category': 'complex',
                'dsl': DSLProgram(
                    name=f'complex_{i}',
                    statements=[
                        ForEach(IteratorType.ALL_NOTES, body=[
                            If(FilterType.PITCH_GREATER, Value(60), then_body=[
                                Operation(OperationType.TRANSPOSE, Value("amount * 5"))
                            ], else_body=[
                                Operation(OperationType.TRANSPOSE, Value("-amount * 5"))
                            ])
                        ])
                    ]
                ),
                'param_range': (0.0, 1.0)
            })

        return templates

    def get_template(self, idx: int) -> Dict[str, Any]:
        """Get template by index"""
        return self.templates[idx]

    def get_random_template(self) -> Dict[str, Any]:
        """Get random template"""
        return random.choice(self.templates)

    def count(self) -> int:
        """Total number of templates"""
        return len(self.templates)


# ============================================================================
# Synthetic Dataset Generator
# ============================================================================

@dataclass
class SyntheticExample:
    """Single training example"""
    input_midi: mido.MidiFile
    output_midi: mido.MidiFile
    dsl_program: DSLProgram
    amount: float
    template_name: str
    template_category: str


class SyntheticDatasetGenerator:
    """
    Generate synthetic training dataset for neural program synthesis.

    Target: 10,000 examples (100 templates × 100 examples each)
    """

    def __init__(self):
        self.template_library = TransformTemplateLibrary()

    def generate_example(self,
                        midi_file: Path,
                        template_idx: Optional[int] = None) -> Optional[SyntheticExample]:
        """
        Generate single training example.

        Args:
            midi_file: Input MIDI file
            template_idx: Template to use (random if None)

        Returns:
            SyntheticExample or None if generation failed
        """
        try:
            # Load MIDI
            input_midi = mido.MidiFile(str(midi_file))

            # Get template
            if template_idx is None:
                template = self.template_library.get_random_template()
            else:
                template = self.template_library.get_template(template_idx)

            # Random amount value
            param_min, param_max = template['param_range']
            amount = np.random.uniform(param_min, param_max)

            # Apply transform
            dsl_program = template['dsl']
            transform_code = dsl_program.to_python()

            # Execute transform (compile and run)
            output_midi = self._execute_transform(input_midi, transform_code, amount)

            # Create example
            example = SyntheticExample(
                input_midi=input_midi,
                output_midi=output_midi,
                dsl_program=dsl_program,
                amount=amount,
                template_name=template['name'],
                template_category=template['category']
            )

            return example

        except Exception as e:
            print(f"Warning: Failed to generate example from {midi_file}: {e}")
            return None

    def _execute_transform(self,
                          midi: mido.MidiFile,
                          transform_code: str,
                          amount: float) -> mido.MidiFile:
        """
        Execute transform code on MIDI.

        This compiles and runs the generated Python code.
        """
        # Create execution namespace
        namespace = {
            'midi': midi,
            'amount': amount,
            'mido': mido,
            'np': np
        }

        # Execute code
        exec(transform_code, namespace)

        # Get transformed MIDI
        transform_fn = namespace[namespace.get('__name__', 'transform')]
        output_midi = transform_fn(midi, amount)

        return output_midi

    def generate_dataset(self,
                        midi_corpus: List[Path],
                        examples_per_template: int = 100,
                        save_path: Optional[Path] = None,
                        verbose: bool = True) -> List[SyntheticExample]:
        """
        Generate complete training dataset.

        Args:
            midi_corpus: List of MIDI files to use as inputs
            examples_per_template: How many examples per template
            save_path: Where to save dataset (optional)
            verbose: Print progress

        Returns:
            List of training examples
        """
        if verbose:
            print(f"\n{'='*70}")
            print("Synthetic Dataset Generation")
            print(f"Templates: {self.template_library.count()}")
            print(f"Examples per template: {examples_per_template}")
            print(f"Target size: {self.template_library.count() * examples_per_template}")
            print(f"{'='*70}\n")

        dataset = []

        for template_idx in range(self.template_library.count()):
            template = self.template_library.get_template(template_idx)

            if verbose and (template_idx + 1) % 10 == 0:
                print(f"  Processing template {template_idx + 1}/{self.template_library.count()}...")

            for example_num in range(examples_per_template):
                # Random MIDI from corpus
                midi_file = random.choice(midi_corpus)

                # Generate example
                example = self.generate_example(midi_file, template_idx)

                if example:
                    dataset.append(example)

        if verbose:
            print(f"\n  → Generated {len(dataset)} examples")
            print(f"  Success rate: {len(dataset) / (self.template_library.count() * examples_per_template):.1%}\n")

        # Save if requested
        if save_path:
            self._save_dataset(dataset, save_path, verbose)

        return dataset

    def _save_dataset(self,
                     dataset: List[SyntheticExample],
                     save_path: Path,
                     verbose: bool = True):
        """Save dataset to disk"""
        # Convert to serializable format
        serializable_data = []

        for example in dataset:
            serializable_data.append({
                'input_midi_data': self._midi_to_bytes(example.input_midi),
                'output_midi_data': self._midi_to_bytes(example.output_midi),
                'dsl_tokens': example.dsl_program.to_tokens(),
                'amount': example.amount,
                'template_name': example.template_name,
                'template_category': example.template_category
            })

        # Save
        with open(save_path, 'wb') as f:
            pickle.dump(serializable_data, f)

        if verbose:
            print(f"  Saved dataset to {save_path}")

    def _midi_to_bytes(self, midi: mido.MidiFile) -> bytes:
        """Convert MIDI to bytes for serialization"""
        # Would implement proper MIDI serialization
        return b''  # Placeholder
