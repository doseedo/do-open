#!/usr/bin/env python3
"""
EMI-Style Musical Program Synthesis
====================================
Discovers musical "genes" (DSL programs) from MIDI data.
Fully whitebox - every transformation is inspectable and editable.

Pipeline:
1. Extract features from MIDI (1000D)
2. Discover which DSL programs produce similar features
3. Store DSL programs as editable "genome"
4. Recombine programs to generate new music
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import random

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

import numpy as np

# Import our components
from synthesis.deep_feature_extractor import DeepFeatureExtractor, FeatureVector
from synthesis.transform_dsl import (
    DSLProgram, ForEach, If, Operation, Value,
    IteratorType, FilterType, OperationType,
    DSLVocabulary
)


@dataclass
class MusicalGene:
    """A discovered musical transformation with its feature signature."""
    name: str
    program: DSLProgram
    feature_signature: Dict[str, float]  # Key features this gene affects
    source_pieces: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'program': self.program.to_string() if hasattr(self.program, 'to_string') else str(self.program),
            'feature_signature': self.feature_signature,
            'source_pieces': self.source_pieces
        }


class EMIPipeline:
    """
    EMI-style discovery and generation pipeline.

    Discovers musical patterns as DSL programs (not opaque vectors).
    Each "gene" is an interpretable, editable transformation.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.feature_extractor = DeepFeatureExtractor()
        self.vocabulary = DSLVocabulary()
        self.discovered_genes: Dict[str, MusicalGene] = {}

        if verbose:
            print("EMI Pipeline initialized")
            print(f"  Feature dimensions: 1000")
            print(f"  DSL vocabulary: {len(list(IteratorType))} iterators, "
                  f"{len(list(OperationType))} operations, "
                  f"{len(list(FilterType))} filters")

    def extract_features(self, midi_path: str) -> FeatureVector:
        """Extract 1000D feature vector from MIDI."""
        return self.feature_extractor.extract(midi_path)

    def analyze_piece(self, midi_path: str) -> Dict[str, Any]:
        """
        Analyze a MIDI file and extract its musical characteristics.
        Returns interpretable analysis, not just vectors.
        """
        features = self.extract_features(midi_path)
        feature_dict = features.to_dict()

        # Group features by category
        analysis = {
            'path': midi_path,
            'statistical': {},
            'harmonic': {},
            'melodic': {},
            'rhythmic': {},
            'structural': {}
        }

        for name, value in feature_dict.items():
            if name.startswith('stat.'):
                analysis['statistical'][name] = value
            elif name.startswith('harm.'):
                analysis['harmonic'][name] = value
            elif name.startswith('mel.'):
                analysis['melodic'][name] = value
            elif name.startswith('rhy.'):
                analysis['rhythmic'][name] = value
            elif name.startswith('struct.'):
                analysis['structural'][name] = value

        return analysis

    def discover_gene_from_difference(
        self,
        before_path: str,
        after_path: str,
        gene_name: str
    ) -> Optional[MusicalGene]:
        """
        Discover a musical gene by comparing two MIDI files.
        The gene represents the transformation from 'before' to 'after'.
        """
        before_features = self.extract_features(before_path).to_dict()
        after_features = self.extract_features(after_path).to_dict()

        # Find which features changed significantly
        changed_features = {}
        for name in before_features:
            if name in after_features:
                diff = after_features[name] - before_features[name]
                if abs(diff) > 0.1:  # Significant change
                    changed_features[name] = diff

        if not changed_features:
            if self.verbose:
                print(f"No significant differences found between files")
            return None

        # Infer DSL program from feature changes
        program = self._infer_program_from_features(changed_features)

        gene = MusicalGene(
            name=gene_name,
            program=program,
            feature_signature=changed_features,
            source_pieces=[before_path, after_path]
        )

        self.discovered_genes[gene_name] = gene

        if self.verbose:
            print(f"Discovered gene '{gene_name}':")
            print(f"  Changed features: {len(changed_features)}")
            print(f"  Top changes: {list(changed_features.items())[:3]}")

        return gene

    def _infer_program_from_features(self, changed_features: Dict[str, float]) -> DSLProgram:
        """
        Infer a DSL program from observed feature changes.
        This is the core EMI insight: features → interpretable programs.
        """
        statements = []

        # Analyze what changed
        pitch_changes = {k: v for k, v in changed_features.items() if 'pitch' in k.lower()}
        rhythm_changes = {k: v for k, v in changed_features.items() if 'rhythm' in k.lower() or 'duration' in k.lower()}
        velocity_changes = {k: v for k, v in changed_features.items() if 'velocity' in k.lower() or 'dynamic' in k.lower()}

        # Build program based on what changed
        if pitch_changes:
            avg_pitch_change = np.mean(list(pitch_changes.values()))
            if abs(avg_pitch_change) > 0.5:
                # Significant pitch shift - create transpose operation
                transpose_amount = int(avg_pitch_change * 12)  # Scale to semitones
                statements.append(
                    ForEach(
                        iterator=IteratorType.ALL_NOTES,
                        body=[Operation(
                            op_type=OperationType.TRANSPOSE,
                            value=Value(value=transpose_amount)
                        )]
                    )
                )

        if rhythm_changes:
            avg_rhythm_change = np.mean(list(rhythm_changes.values()))
            if abs(avg_rhythm_change) > 0.3:
                # Rhythm scaling
                scale_factor = 1.0 + avg_rhythm_change
                statements.append(
                    ForEach(
                        iterator=IteratorType.ALL_NOTES,
                        body=[Operation(
                            op_type=OperationType.TIME_SCALE,
                            value=Value(value=scale_factor)
                        )]
                    )
                )

        if velocity_changes:
            avg_vel_change = np.mean(list(velocity_changes.values()))
            if abs(avg_vel_change) > 0.2:
                # Velocity scaling
                scale_factor = 1.0 + avg_vel_change
                statements.append(
                    ForEach(
                        iterator=IteratorType.ALL_NOTES,
                        body=[Operation(
                            op_type=OperationType.SCALE_VELOCITY,
                            value=Value(value=scale_factor)
                        )]
                    )
                )

        # If nothing specific, create identity program
        if not statements:
            statements.append(
                ForEach(
                    iterator=IteratorType.ALL_NOTES,
                    body=[]  # No-op
                )
            )

        return DSLProgram(statements=statements)

    def create_gene_library_from_corpus(
        self,
        midi_paths: List[str],
        min_genes: int = 10
    ) -> Dict[str, MusicalGene]:
        """
        Discover genes by analyzing patterns across a corpus.
        Compares all pairs of MIDI files to find common transformations.
        """
        if self.verbose:
            print(f"\nAnalyzing corpus of {len(midi_paths)} files...")

        # Extract features from all files
        all_features = {}
        for path in midi_paths:
            try:
                all_features[path] = self.extract_features(path).to_dict()
            except Exception as e:
                if self.verbose:
                    print(f"  Skipping {path}: {e}")

        if len(all_features) < 2:
            print("Need at least 2 valid MIDI files")
            return {}

        # Find common transformation patterns
        # Group files by similarity
        paths = list(all_features.keys())

        for i, path1 in enumerate(paths[:min(10, len(paths))]):
            for j, path2 in enumerate(paths[i+1:min(10, len(paths))]):
                gene_name = f"gene_{i}_{j}"

                # Calculate feature difference
                diff = {}
                for k in all_features[path1]:
                    if k in all_features[path2]:
                        d = all_features[path2][k] - all_features[path1][k]
                        if abs(d) > 0.1:
                            diff[k] = d

                if diff:
                    program = self._infer_program_from_features(diff)
                    gene = MusicalGene(
                        name=gene_name,
                        program=program,
                        feature_signature=diff,
                        source_pieces=[path1, path2]
                    )
                    self.discovered_genes[gene_name] = gene

        if self.verbose:
            print(f"Discovered {len(self.discovered_genes)} genes")

        return self.discovered_genes

    def get_gene(self, name: str) -> Optional[MusicalGene]:
        """Get a discovered gene by name."""
        return self.discovered_genes.get(name)

    def list_genes(self) -> List[str]:
        """List all discovered gene names."""
        return list(self.discovered_genes.keys())

    def edit_gene(self, name: str, new_program: DSLProgram) -> bool:
        """
        Edit a gene's program directly.
        This is the key to whitebox control - programs are editable.
        """
        if name not in self.discovered_genes:
            return False

        self.discovered_genes[name].program = new_program
        return True

    def combine_genes(self, gene_names: List[str], new_name: str) -> Optional[MusicalGene]:
        """
        Combine multiple genes into a new composite gene.
        This is EMI-style recombination.
        """
        programs = []
        combined_signature = {}
        sources = []

        for name in gene_names:
            gene = self.discovered_genes.get(name)
            if gene:
                programs.append(gene.program)
                combined_signature.update(gene.feature_signature)
                sources.extend(gene.source_pieces)

        if not programs:
            return None

        # Combine all statements
        combined_statements = []
        for prog in programs:
            combined_statements.extend(prog.statements)

        combined_program = DSLProgram(statements=combined_statements)

        new_gene = MusicalGene(
            name=new_name,
            program=combined_program,
            feature_signature=combined_signature,
            source_pieces=list(set(sources))
        )

        self.discovered_genes[new_name] = new_gene
        return new_gene

    def save_genes(self, path: str):
        """Save discovered genes to JSON."""
        data = {name: gene.to_dict() for name, gene in self.discovered_genes.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        if self.verbose:
            print(f"Saved {len(data)} genes to {path}")

    def print_gene(self, name: str):
        """Print a gene's details in human-readable form."""
        gene = self.discovered_genes.get(name)
        if not gene:
            print(f"Gene '{name}' not found")
            return

        print(f"\n{'='*50}")
        print(f"Gene: {gene.name}")
        print(f"{'='*50}")
        print(f"Program:")
        for stmt in gene.program.statements:
            print(f"  {stmt}")
        print(f"\nFeature signature ({len(gene.feature_signature)} features):")
        for feat, val in list(gene.feature_signature.items())[:5]:
            print(f"  {feat}: {val:+.3f}")
        if len(gene.feature_signature) > 5:
            print(f"  ... and {len(gene.feature_signature)-5} more")
        print(f"\nSource pieces: {gene.source_pieces}")


# Demo / Test
if __name__ == '__main__':
    print("="*60)
    print("EMI-Style Musical Program Synthesis")
    print("="*60)

    pipeline = EMIPipeline(verbose=True)

    # Test with available MIDI files
    test_files = [
        'basie_test.mid',
        'swing_final.mid',
        'swing_fixed.mid',
        'swing_improved.mid'
    ]

    existing_files = [f for f in test_files if os.path.exists(f)]

    if len(existing_files) >= 2:
        print(f"\nFound {len(existing_files)} test files")

        # Analyze first file
        print(f"\nAnalyzing: {existing_files[0]}")
        analysis = pipeline.analyze_piece(existing_files[0])
        print(f"  Statistical features: {len(analysis['statistical'])}")
        print(f"  Harmonic features: {len(analysis['harmonic'])}")

        # Discover genes from corpus
        print("\nDiscovering genes from corpus...")
        genes = pipeline.create_gene_library_from_corpus(existing_files)

        # Print discovered genes
        for name in pipeline.list_genes()[:3]:
            pipeline.print_gene(name)

        # Save genes
        pipeline.save_genes('discovered_genes.json')

        print("\n" + "="*60)
        print("Pipeline complete!")
        print("="*60)
        print("\nKey insight: Each gene is an editable DSL program, not an opaque vector.")
        print("You can:")
        print("  - pipeline.edit_gene('name', new_program)  # Edit transformation")
        print("  - pipeline.combine_genes(['g1', 'g2'], 'new')  # Recombine")
        print("  - pipeline.print_gene('name')  # Inspect")
    else:
        print(f"\nNeed at least 2 MIDI files. Found: {existing_files}")
