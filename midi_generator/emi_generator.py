#!/usr/bin/env python3
"""
EMI-Style Generator with DSL Execution
========================================
Discovers genes from corpus, executes DSL transformations, generates MIDI.

Full pipeline:
1. Analyze corpus → discover genes (DSL programs)
2. Select/combine genes for generation
3. Execute DSL programs on seed MIDI
4. Output transformed MIDI
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import random
import copy

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

import numpy as np

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido not available. pip install mido")

from synthesis.deep_feature_extractor import DeepFeatureExtractor, FeatureVector
from synthesis.transform_dsl import (
    DSLProgram, ForEach, If, Operation, Value,
    IteratorType, FilterType, OperationType,
)


@dataclass
class Note:
    """Simple note representation for DSL execution."""
    pitch: int
    start_time: float  # In beats
    duration: float    # In beats
    velocity: int
    channel: int = 0
    track: int = 0


@dataclass
class MusicalGene:
    """A discovered musical transformation."""
    name: str
    program: DSLProgram
    feature_signature: Dict[str, float]
    source_pieces: List[str] = field(default_factory=list)


class DSLExecutor:
    """Executes DSL programs on note sequences."""

    def __init__(self):
        self.operations = {
            OperationType.TRANSPOSE: self._transpose,
            OperationType.SET_PITCH: self._set_pitch,
            OperationType.OCTAVE_SHIFT: self._octave_shift,
            OperationType.TIME_SCALE: self._time_scale,
            OperationType.TIME_SHIFT: self._time_shift,
            OperationType.DURATION_SCALE: self._duration_scale,
            OperationType.SET_VELOCITY: self._set_velocity,
            OperationType.SCALE_VELOCITY: self._scale_velocity,
            OperationType.ADD_VELOCITY: self._add_velocity,
            OperationType.QUANTIZE_TIME: self._quantize_time,
            OperationType.SWING: self._swing,
        }

    def execute(self, program: DSLProgram, notes: List[Note]) -> List[Note]:
        """Execute a DSL program on a list of notes."""
        result = [copy.deepcopy(n) for n in notes]

        for statement in program.statements:
            result = self._execute_statement(statement, result)

        return result

    def _execute_statement(self, stmt, notes: List[Note]) -> List[Note]:
        """Execute a single statement."""
        if isinstance(stmt, ForEach):
            return self._execute_foreach(stmt, notes)
        elif isinstance(stmt, If):
            return self._execute_if(stmt, notes)
        elif isinstance(stmt, Operation):
            return self._execute_operation(stmt, notes)
        return notes

    def _execute_foreach(self, stmt: ForEach, notes: List[Note]) -> List[Note]:
        """Execute foreach statement."""
        iterator = stmt.iterator

        if iterator == IteratorType.ALL_NOTES:
            selected = notes
        elif iterator == IteratorType.NOTES_ON_BEAT:
            selected = [n for n in notes if n.start_time % 1.0 < 0.1]
        elif iterator == IteratorType.NOTES_OFF_BEAT:
            selected = [n for n in notes if n.start_time % 1.0 >= 0.1]
        elif iterator == IteratorType.SIMULTANEOUS_NOTES:
            # Group by start time
            by_time = {}
            for n in notes:
                t = round(n.start_time, 2)
                if t not in by_time:
                    by_time[t] = []
                by_time[t].append(n)
            selected = [n for group in by_time.values() if len(group) > 1 for n in group]
        elif iterator == IteratorType.SEQUENTIAL_NOTES:
            by_time = {}
            for n in notes:
                t = round(n.start_time, 2)
                if t not in by_time:
                    by_time[t] = []
                by_time[t].append(n)
            selected = [n for group in by_time.values() if len(group) == 1 for n in group]
        else:
            selected = notes

        # Execute body operations on selected notes
        for op in stmt.body:
            if isinstance(op, Operation):
                for note in selected:
                    self._apply_operation(op, note)

        return notes

    def _execute_if(self, stmt: If, notes: List[Note]) -> List[Note]:
        """Execute conditional statement."""
        # Filter notes based on condition
        filtered = self._apply_filter(stmt.filter_type, notes, stmt.threshold)

        # Execute body on filtered notes
        for op in stmt.body:
            if isinstance(op, Operation):
                for note in filtered:
                    self._apply_operation(op, note)

        return notes

    def _execute_operation(self, op: Operation, notes: List[Note]) -> List[Note]:
        """Execute operation on all notes."""
        for note in notes:
            self._apply_operation(op, note)
        return notes

    def _apply_operation(self, op: Operation, note: Note):
        """Apply a single operation to a note."""
        value = self._get_value(op.value)
        op_func = self.operations.get(op.op_type)
        if op_func:
            op_func(note, value)

    def _get_value(self, value: Value) -> float:
        """Extract numeric value from Value object."""
        if isinstance(value.value, (int, float, np.number)):
            return float(value.value)
        return 0.0

    def _apply_filter(self, filter_type: FilterType, notes: List[Note], threshold: float) -> List[Note]:
        """Filter notes based on condition."""
        if filter_type == FilterType.PITCH_GREATER:
            return [n for n in notes if n.pitch > threshold]
        elif filter_type == FilterType.PITCH_LESS:
            return [n for n in notes if n.pitch < threshold]
        elif filter_type == FilterType.VELOCITY_GREATER:
            return [n for n in notes if n.velocity > threshold]
        elif filter_type == FilterType.VELOCITY_LESS:
            return [n for n in notes if n.velocity < threshold]
        elif filter_type == FilterType.DURATION_GREATER:
            return [n for n in notes if n.duration > threshold]
        elif filter_type == FilterType.DURATION_LESS:
            return [n for n in notes if n.duration < threshold]
        return notes

    # Operation implementations
    def _transpose(self, note: Note, value: float):
        note.pitch = max(0, min(127, note.pitch + int(value)))

    def _set_pitch(self, note: Note, value: float):
        note.pitch = max(0, min(127, int(value)))

    def _octave_shift(self, note: Note, value: float):
        note.pitch = max(0, min(127, note.pitch + int(value) * 12))

    def _time_scale(self, note: Note, value: float):
        if value > 0:
            note.start_time *= value
            note.duration *= value

    def _time_shift(self, note: Note, value: float):
        note.start_time = max(0, note.start_time + value)

    def _duration_scale(self, note: Note, value: float):
        if value > 0:
            note.duration *= value

    def _set_velocity(self, note: Note, value: float):
        note.velocity = max(1, min(127, int(value)))

    def _scale_velocity(self, note: Note, value: float):
        if value > 0:
            note.velocity = max(1, min(127, int(note.velocity * value)))

    def _add_velocity(self, note: Note, value: float):
        note.velocity = max(1, min(127, note.velocity + int(value)))

    def _quantize_time(self, note: Note, value: float):
        if value > 0:
            note.start_time = round(note.start_time / value) * value

    def _swing(self, note: Note, value: float):
        # Apply swing to off-beats
        beat_pos = note.start_time % 1.0
        if 0.4 < beat_pos < 0.6:  # Off-beat
            swing_amount = value * 0.1
            note.start_time += swing_amount


class EMIGenerator:
    """Full EMI-style generator with corpus learning and MIDI output."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.feature_extractor = DeepFeatureExtractor()
        self.executor = DSLExecutor()
        self.genes: Dict[str, MusicalGene] = {}
        self.corpus_features: Dict[str, Dict] = {}

        if verbose:
            print("EMI Generator initialized")

    def load_corpus(self, midi_dir: str, max_files: int = 50) -> int:
        """Load and analyze a corpus of MIDI files."""
        midi_path = Path(midi_dir)
        if not midi_path.exists():
            print(f"Directory not found: {midi_dir}")
            return 0

        midi_files = list(midi_path.glob("*.mid")) + list(midi_path.glob("*.MID"))
        midi_files = midi_files[:max_files]

        if self.verbose:
            print(f"Loading corpus from {midi_dir}...")
            print(f"Found {len(midi_files)} MIDI files")

        loaded = 0
        for midi_file in midi_files:
            try:
                features = self.feature_extractor.extract(str(midi_file))
                self.corpus_features[str(midi_file)] = features.to_dict()
                loaded += 1
                if self.verbose and loaded % 10 == 0:
                    print(f"  Loaded {loaded}/{len(midi_files)}")
            except Exception as e:
                if self.verbose:
                    print(f"  Skip {midi_file.name}: {e}")

        if self.verbose:
            print(f"Successfully loaded {loaded} files")

        return loaded

    def discover_genes(self, min_feature_diff: float = 0.1) -> int:
        """Discover genes by comparing pieces in corpus."""
        if len(self.corpus_features) < 2:
            print("Need at least 2 files in corpus")
            return 0

        paths = list(self.corpus_features.keys())
        gene_count = 0

        for i in range(min(len(paths), 20)):
            for j in range(i + 1, min(len(paths), 20)):
                path1, path2 = paths[i], paths[j]
                f1, f2 = self.corpus_features[path1], self.corpus_features[path2]

                # Find significant differences
                diff = {}
                for k in f1:
                    if k in f2:
                        d = f2[k] - f1[k]
                        if abs(d) > min_feature_diff:
                            diff[k] = d

                if len(diff) > 5:  # Meaningful difference
                    program = self._infer_program(diff)
                    gene_name = f"gene_{gene_count}"
                    self.genes[gene_name] = MusicalGene(
                        name=gene_name,
                        program=program,
                        feature_signature=diff,
                        source_pieces=[path1, path2]
                    )
                    gene_count += 1

        if self.verbose:
            print(f"Discovered {gene_count} genes")

        return gene_count

    def _infer_program(self, feature_diff: Dict[str, float]) -> DSLProgram:
        """Infer DSL program from feature differences."""
        statements = []

        # Analyze changes
        pitch_changes = [v for k, v in feature_diff.items() if 'pitch' in k.lower()]
        rhythm_changes = [v for k, v in feature_diff.items() if 'rhythm' in k.lower() or 'duration' in k.lower()]
        velocity_changes = [v for k, v in feature_diff.items() if 'velocity' in k.lower()]

        if pitch_changes:
            avg = np.mean(pitch_changes)
            if abs(avg) > 0.5:
                transpose = int(avg * 12)
                statements.append(ForEach(
                    iterator=IteratorType.ALL_NOTES,
                    body=[Operation(op_type=OperationType.TRANSPOSE, value=Value(value=transpose))]
                ))

        if rhythm_changes:
            avg = np.mean(rhythm_changes)
            if abs(avg) > 0.3:
                scale = max(0.5, min(2.0, 1.0 + avg * 0.5))
                statements.append(ForEach(
                    iterator=IteratorType.ALL_NOTES,
                    body=[Operation(op_type=OperationType.TIME_SCALE, value=Value(value=scale))]
                ))

        if velocity_changes:
            avg = np.mean(velocity_changes)
            if abs(avg) > 0.2:
                scale = max(0.5, min(1.5, 1.0 + avg * 0.3))
                statements.append(ForEach(
                    iterator=IteratorType.ALL_NOTES,
                    body=[Operation(op_type=OperationType.SCALE_VELOCITY, value=Value(value=scale))]
                ))

        if not statements:
            # Identity operation
            statements.append(ForEach(iterator=IteratorType.ALL_NOTES, body=[]))

        return DSLProgram(statements=statements)

    def load_midi(self, midi_path: str) -> List[Note]:
        """Load MIDI file into Note list."""
        if not MIDO_AVAILABLE:
            print("mido not available")
            return []

        mid = MidiFile(midi_path)
        notes = []
        ticks_per_beat = mid.ticks_per_beat

        for track_idx, track in enumerate(mid.tracks):
            current_time = 0
            active_notes = {}  # pitch -> (start_time, velocity, channel)

            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (current_time, msg.velocity, msg.channel)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_ticks, velocity, channel = active_notes.pop(msg.note)
                        notes.append(Note(
                            pitch=msg.note,
                            start_time=start_ticks / ticks_per_beat,
                            duration=(current_time - start_ticks) / ticks_per_beat,
                            velocity=velocity,
                            channel=channel,
                            track=track_idx
                        ))

        return notes

    def save_midi(self, notes: List[Note], output_path: str, tempo: int = 120):
        """Save Note list to MIDI file."""
        if not MIDO_AVAILABLE:
            print("mido not available")
            return

        mid = MidiFile(ticks_per_beat=480)
        ticks_per_beat = 480

        # Group by track
        by_track = {}
        for note in notes:
            if note.track not in by_track:
                by_track[note.track] = []
            by_track[note.track].append(note)

        for track_idx in sorted(by_track.keys()):
            track = MidiTrack()
            mid.tracks.append(track)

            if track_idx == 0:
                # Add tempo
                track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))

            track_notes = sorted(by_track[track_idx], key=lambda n: n.start_time)

            events = []
            for note in track_notes:
                start_tick = int(note.start_time * ticks_per_beat)
                end_tick = int((note.start_time + note.duration) * ticks_per_beat)
                events.append((start_tick, 'on', note.pitch, note.velocity, note.channel))
                events.append((end_tick, 'off', note.pitch, 0, note.channel))

            events.sort(key=lambda x: (x[0], x[1] == 'on'))

            last_tick = 0
            for tick, event_type, pitch, vel, ch in events:
                delta = tick - last_tick
                if event_type == 'on':
                    track.append(Message('note_on', note=pitch, velocity=vel, channel=ch, time=delta))
                else:
                    track.append(Message('note_off', note=pitch, velocity=0, channel=ch, time=delta))
                last_tick = tick

        mid.save(output_path)
        if self.verbose:
            print(f"Saved: {output_path}")

    def generate(self, seed_midi: str, gene_names: List[str], output_path: str):
        """Generate new MIDI by applying genes to seed."""
        # Load seed
        notes = self.load_midi(seed_midi)
        if not notes:
            print(f"Could not load seed: {seed_midi}")
            return

        if self.verbose:
            print(f"Loaded seed: {len(notes)} notes")

        # Apply genes
        for gene_name in gene_names:
            if gene_name not in self.genes:
                print(f"Gene not found: {gene_name}")
                continue

            gene = self.genes[gene_name]
            notes = self.executor.execute(gene.program, notes)
            if self.verbose:
                print(f"Applied gene: {gene_name}")

        # Save output
        self.save_midi(notes, output_path)

    def generate_variation(self, seed_midi: str, output_path: str, variation_strength: float = 0.5):
        """Generate a variation of seed MIDI using random genes."""
        notes = self.load_midi(seed_midi)
        if not notes:
            return

        if self.verbose:
            print(f"Generating variation (strength={variation_strength})")

        # Select random genes
        num_genes = max(1, int(len(self.genes) * variation_strength * 0.3))
        selected = random.sample(list(self.genes.keys()), min(num_genes, len(self.genes)))

        for gene_name in selected:
            gene = self.genes[gene_name]
            notes = self.executor.execute(gene.program, notes)

        self.save_midi(notes, output_path)

    def print_gene(self, name: str):
        """Print gene details."""
        gene = self.genes.get(name)
        if not gene:
            print(f"Gene not found: {name}")
            return

        print(f"\n{'='*50}")
        print(f"Gene: {gene.name}")
        print(f"{'='*50}")
        print("Program:")
        for stmt in gene.program.statements:
            print(f"  {stmt}")
        print(f"\nFeature changes: {len(gene.feature_signature)}")
        for k, v in list(gene.feature_signature.items())[:5]:
            print(f"  {k}: {v:+.3f}")
        print(f"Sources: {[Path(p).name for p in gene.source_pieces]}")

    def save_genes(self, path: str):
        """Save discovered genes."""
        data = {}
        for name, gene in self.genes.items():
            data[name] = {
                'name': gene.name,
                'program': str(gene.program),
                'feature_signature': gene.feature_signature,
                'source_pieces': gene.source_pieces
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved {len(data)} genes to {path}")


def main():
    print("=" * 60)
    print("EMI Generator - Big Band Test")
    print("=" * 60)

    gen = EMIGenerator(verbose=True)

    # Load big band corpus
    corpus_path = "midi_corpus/big_band"
    loaded = gen.load_corpus(corpus_path, max_files=30)

    if loaded < 2:
        print("Not enough files loaded")
        return

    # Discover genes
    gen.discover_genes()

    # Print some genes
    for name in list(gen.genes.keys())[:3]:
        gen.print_gene(name)

    # Save genes
    gen.save_genes("bigband_genes.json")

    # Generate variations
    # Find a seed file
    seed_files = list(Path(corpus_path).glob("*.mid"))[:1]
    if seed_files:
        seed = str(seed_files[0])
        print(f"\nGenerating from seed: {seed_files[0].name}")

        # Generate with specific genes
        if gen.genes:
            gene_names = list(gen.genes.keys())[:2]
            gen.generate(seed, gene_names, "emi_output_genes.mid")

        # Generate random variation
        gen.generate_variation(seed, "emi_output_variation.mid", variation_strength=0.3)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
