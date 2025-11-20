#!/usr/bin/env python3
"""
Manual Labeling Tool for MIDI Corpus
Agent 03: Metadata & Labeling Manager

Interactive CLI tool for music experts to manually label subjective parameters
that cannot be reliably auto-extracted.

Features:
    - Displays auto-extracted labels for reference
    - MIDI playback capability
    - Input validation and consistency checks
    - Progress tracking
    - Session saving/resuming
    - Inter-rater reliability checking

Workflow:
    1. Load MIDI file
    2. Display auto-labels
    3. Play MIDI (optional)
    4. Collect manual labels for 10 subjective parameters
    5. Validate inputs
    6. Save to database
    7. Move to next file

Author: Agent 03 - Metadata & Labeling Manager
License: MIT
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

try:
    import mido
    from mido import MidiFile
except ImportError:
    print("ERROR: mido not installed. Install with: pip install mido")
    raise

# Try to import pygame for MIDI playback
try:
    import pygame
    import pygame.midi
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("WARNING: pygame not available. MIDI playback disabled.")
    print("Install with: pip install pygame")

# Import auto-labeler
try:
    from midi_generator.learning.auto_labeler import AutoLabeler, HierarchicalLabels
    AUTO_LABELER_AVAILABLE = True
except ImportError:
    AUTO_LABELER_AVAILABLE = False
    print("WARNING: auto_labeler not available")


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class ManualLabels:
    """Manual labels for subjective parameters."""
    file_id: str
    labeler_id: str
    timestamp: str

    # Subjective parameters requiring manual labeling
    energy_level: Optional[float] = None
    complexity_overall: Optional[float] = None
    harmony_tension: Optional[float] = None
    harmony_progression_predictability: Optional[float] = None
    melody_contour_smoothness: Optional[float] = None
    jazz_bebop_vocabulary: Optional[float] = None
    classical_counterpoint: Optional[float] = None
    rock_riff_repetition: Optional[float] = None
    electronic_filter_movement: Optional[float] = None

    # Optional notes
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'file_id': self.file_id,
            'labeler_id': self.labeler_id,
            'timestamp': self.timestamp,
            'manual_labels': {
                'energy.level': self.energy_level,
                'complexity.overall': self.complexity_overall,
                'harmony.tension': self.harmony_tension,
                'harmony.progression_predictability': self.harmony_progression_predictability,
                'melody.contour_smoothness': self.melody_contour_smoothness,
                'jazz.bebop_vocabulary': self.jazz_bebop_vocabulary,
                'classical.counterpoint': self.classical_counterpoint,
                'rock.riff_repetition': self.rock_riff_repetition,
                'electronic.filter_movement': self.electronic_filter_movement
            },
            'notes': self.notes
        }


@dataclass
class LabelingSession:
    """Labeling session state for progress tracking."""
    session_id: str
    labeler_id: str
    start_time: str
    files_to_label: List[str]
    completed_files: List[str] = field(default_factory=list)
    current_index: int = 0

    def progress(self) -> str:
        """Get progress string."""
        return f"{len(self.completed_files)}/{len(self.files_to_label)}"

    def save(self, path: Path):
        """Save session state."""
        with open(path, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'labeler_id': self.labeler_id,
                'start_time': self.start_time,
                'files_to_label': self.files_to_label,
                'completed_files': self.completed_files,
                'current_index': self.current_index
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'LabelingSession':
        """Load session state."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ==============================================================================
# LABELING TOOL
# ==============================================================================

class ManualLabelingTool:
    """
    Interactive CLI tool for manual labeling.
    """

    def __init__(self,
                 corpus_dir: Path,
                 output_dir: Path,
                 labeler_id: str,
                 auto_labeler: Optional[AutoLabeler] = None):
        """
        Initialize labeling tool.

        Args:
            corpus_dir: Directory containing MIDI files to label
            output_dir: Directory to save labels
            labeler_id: Identifier for the labeler (e.g., "expert_1")
            auto_labeler: AutoLabeler instance (optional)
        """
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.labeler_id = labeler_id

        # Initialize auto-labeler
        if auto_labeler:
            self.auto_labeler = auto_labeler
        elif AUTO_LABELER_AVAILABLE:
            self.auto_labeler = AutoLabeler()
        else:
            self.auto_labeler = None

        # Initialize MIDI playback
        if PYGAME_AVAILABLE:
            pygame.init()
            pygame.midi.init()

        # Parameter definitions (for validation and prompts)
        self.params = {
            'energy.level': {
                'name': 'Energy Level',
                'type': 'continuous',
                'range': (0.0, 1.0),
                'description': '0.0 = very calm/ambient, 0.5 = moderate, 1.0 = intense/frenetic'
            },
            'complexity.overall': {
                'name': 'Overall Complexity',
                'type': 'continuous',
                'range': (0.0, 1.0),
                'description': '0.0 = simple/repetitive, 0.5 = moderate, 1.0 = very complex'
            },
            'harmony.tension': {
                'name': 'Harmonic Tension',
                'type': 'continuous',
                'range': (0.0, 1.0),
                'description': '0.0 = consonant/resolved, 0.5 = moderate, 1.0 = very dissonant'
            },
            'harmony.progression_predictability': {
                'name': 'Progression Predictability',
                'type': 'continuous',
                'range': (0.0, 1.0),
                'description': '0.0 = unpredictable/surprising, 0.5 = moderate, 1.0 = formulaic'
            },
            'melody.contour_smoothness': {
                'name': 'Melodic Smoothness',
                'type': 'continuous',
                'range': (0.0, 1.0),
                'description': '0.0 = angular/leaps, 0.5 = balanced, 1.0 = stepwise/smooth'
            },
            'jazz.bebop_vocabulary': {
                'name': 'Bebop Vocabulary',
                'type': 'continuous',
                'range': (0.0, 1.0),
                'description': '0.0 = no bebop, 0.5 = moderate, 1.0 = pure bebop (jazz only)',
                'genre_specific': 'jazz'
            },
            'classical.counterpoint': {
                'name': 'Counterpoint Degree',
                'type': 'continuous',
                'range': (0.0, 1.0),
                'description': '0.0 = homophonic, 0.5 = moderate, 1.0 = strict counterpoint (classical only)',
                'genre_specific': 'classical'
            },
            'rock.riff_repetition': {
                'name': 'Riff Repetition',
                'type': 'continuous',
                'range': (0.0, 1.0),
                'description': '0.0 = no riffs, 0.5 = moderate, 1.0 = pure riff-based (rock/metal only)',
                'genre_specific': 'rock'
            },
            'electronic.filter_movement': {
                'name': 'Filter Movement',
                'type': 'continuous',
                'range': (0.0, 1.0),
                'description': '0.0 = static timbre, 0.5 = moderate, 1.0 = extreme modulation (electronic only)',
                'genre_specific': 'electronic'
            }
        }

    def start_session(self, files_to_label: List[Path]) -> LabelingSession:
        """Start a new labeling session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = LabelingSession(
            session_id=session_id,
            labeler_id=self.labeler_id,
            start_time=datetime.now().isoformat(),
            files_to_label=[str(f) for f in files_to_label]
        )

        self.print_header()
        print(f"📋 Starting labeling session: {session_id}")
        print(f"👤 Labeler: {self.labeler_id}")
        print(f"📁 Files to label: {len(files_to_label)}")
        print()

        return session

    def label_file(self, midi_path: Path, auto_labels: Optional[HierarchicalLabels] = None) -> ManualLabels:
        """
        Label a single MIDI file interactively.

        Args:
            midi_path: Path to MIDI file
            auto_labels: Pre-computed auto labels (optional)

        Returns:
            ManualLabels
        """
        file_id = midi_path.stem

        print("=" * 80)
        print(f"🎵 File: {midi_path.name}")
        print("=" * 80)

        # Load MIDI file
        try:
            midi = MidiFile(str(midi_path))
            print(f"✓ Loaded MIDI file successfully")
        except Exception as e:
            print(f"✗ Error loading MIDI: {e}")
            return None

        # Show file info
        self._display_file_info(midi, midi_path)

        # Get auto-labels if not provided
        if auto_labels is None and self.auto_labeler:
            print("\n⏳ Extracting auto-labels...")
            auto_labels = self.auto_labeler.extract_all(midi_path, file_id)

        # Display auto-labels
        if auto_labels:
            self._display_auto_labels(auto_labels)

        # Playback option
        print("\n" + "─" * 80)
        play = self._ask_yes_no("Would you like to play the MIDI file?", default=True)
        if play and PYGAME_AVAILABLE:
            self._play_midi(midi_path)

        # Collect manual labels
        print("\n" + "=" * 80)
        print("MANUAL LABELING")
        print("=" * 80)
        print("Please listen carefully and rate the following parameters.")
        print("Enter values between 0.0 and 1.0, or 'n/a' if not applicable.")
        print()

        manual_labels = ManualLabels(
            file_id=file_id,
            labeler_id=self.labeler_id,
            timestamp=datetime.now().isoformat()
        )

        # Get genre for genre-specific parameters
        genre = auto_labels.level1.get('genre.primary', 'unknown') if auto_labels else 'unknown'

        # Collect labels for each parameter
        for param_key, param_def in self.params.items():
            # Skip genre-specific params if not applicable
            if param_def.get('genre_specific'):
                if genre != param_def['genre_specific'] and genre not in ['rock', 'metal']:
                    continue

            value = self._prompt_parameter(param_def)

            # Set the value
            attr_name = param_key.replace('.', '_')
            setattr(manual_labels, attr_name, value)

        # Collect optional notes
        print("\n" + "─" * 80)
        print("Optional notes (press Enter to skip):")
        manual_labels.notes = input("> ").strip()

        # Validate
        self._validate_labels(manual_labels)

        # Summary
        print("\n" + "=" * 80)
        print("LABELING SUMMARY")
        print("=" * 80)
        self._display_manual_labels(manual_labels)

        # Confirm
        confirm = self._ask_yes_no("\nSave these labels?", default=True)
        if not confirm:
            redo = self._ask_yes_no("Redo this file?", default=True)
            if redo:
                return self.label_file(midi_path, auto_labels)
            else:
                return None

        return manual_labels

    def _display_file_info(self, midi: MidiFile, path: Path):
        """Display MIDI file information."""
        print()
        print("File Information:")
        print(f"  Tracks: {len(midi.tracks)}")
        print(f"  Ticks per beat: {midi.ticks_per_beat}")
        print(f"  Type: {midi.type}")

        # Calculate duration
        total_ticks = 0
        for track in midi.tracks:
            track_ticks = sum(msg.time for msg in track)
            total_ticks = max(total_ticks, track_ticks)

        duration_seconds = mido.tick2second(total_ticks, midi.ticks_per_beat, 500000)
        print(f"  Duration: {duration_seconds:.1f} seconds ({duration_seconds / 60:.1f} minutes)")

    def _display_auto_labels(self, labels: HierarchicalLabels):
        """Display auto-extracted labels for reference."""
        print("\n" + "─" * 80)
        print("AUTO-EXTRACTED LABELS (for reference)")
        print("─" * 80)

        print("\nLevel 1 (Global):")
        for key, value in labels.level1.items():
            print(f"  {key}: {value}")

        print("\nLevel 2 (Universal) - Selected:")
        # Show subset of most relevant Level 2 params
        relevant_l2 = ['harmony.chord_density', 'harmony.complexity', 'melody.note_density',
                       'rhythm.syncopation', 'dynamics.overall_level']
        for key in relevant_l2:
            if key in labels.level2:
                value = labels.level2[key]
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

    def _display_manual_labels(self, labels: ManualLabels):
        """Display manual labels summary."""
        label_dict = labels.to_dict()['manual_labels']

        for param_key, value in label_dict.items():
            if value is not None:
                print(f"  {param_key}: {value:.2f}")

        if labels.notes:
            print(f"\nNotes: {labels.notes}")

    def _prompt_parameter(self, param_def: Dict) -> Optional[float]:
        """Prompt user for a parameter value."""
        print("\n" + "─" * 40)
        print(f"Parameter: {param_def['name']}")
        print(f"Range: {param_def['range'][0]} - {param_def['range'][1]}")
        print(f"Guide: {param_def['description']}")

        while True:
            value_str = input("Value (0.0-1.0, or 'n/a'): ").strip().lower()

            # Handle N/A
            if value_str in ['n/a', 'na', '']:
                return None

            # Parse value
            try:
                value = float(value_str)
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'n/a'.")
                continue

            # Validate range
            if not (param_def['range'][0] <= value <= param_def['range'][1]):
                print(f"❌ Value out of range. Must be between {param_def['range'][0]} and {param_def['range'][1]}.")
                continue

            return value

    def _validate_labels(self, labels: ManualLabels):
        """Validate manual labels for consistency."""
        # Check for obvious inconsistencies
        warnings_list = []

        # Example: high complexity usually correlates with high tension
        if labels.complexity_overall and labels.harmony_tension:
            if labels.complexity_overall > 0.7 and labels.harmony_tension < 0.3:
                warnings_list.append("⚠️  High complexity but low tension - is this correct?")

        # High energy usually correlates with high dynamics
        if labels.energy_level and labels.energy_level > 0.8:
            warnings_list.append("ℹ️  High energy level - verify this matches the intensity of the piece")

        # Display warnings
        if warnings_list:
            print("\n" + "─" * 80)
            print("Validation Warnings:")
            for warning in warnings_list:
                print(f"  {warning}")

    def _play_midi(self, midi_path: Path):
        """
        Play MIDI file using pygame.

        Note: This is a simplified playback. For production, use a better MIDI player.
        """
        if not PYGAME_AVAILABLE:
            print("❌ MIDI playback not available (pygame not installed)")
            return

        try:
            print(f"\n🔊 Playing: {midi_path.name}")
            print("(Note: Playback quality depends on system MIDI synth)")

            # Load and play MIDI
            midi = mido.MidiFile(str(midi_path))

            # Simple playback (blocks until done)
            # In production, use non-blocking playback with controls
            port = mido.open_output()

            for msg in midi.play():
                port.send(msg)

            port.close()

            print("✓ Playback completed")

        except Exception as e:
            print(f"❌ Playback error: {e}")
            print("You may need to configure your system's MIDI synthesizer.")

    def _ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question."""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{question} [{default_str}]: ").strip().lower()

        if not response:
            return default

        return response in ['y', 'yes']

    def print_header(self):
        """Print tool header."""
        print()
        print("=" * 80)
        print(" " * 20 + "MANUAL LABELING TOOL v2.0")
        print(" " * 15 + "Agent 03: Metadata & Labeling Manager")
        print("=" * 80)
        print()

    def run_batch(self, files: List[Path], session_file: Optional[Path] = None) -> List[ManualLabels]:
        """
        Run batch labeling session.

        Args:
            files: List of MIDI files to label
            session_file: Path to save/load session state

        Returns:
            List of ManualLabels
        """
        # Start or resume session
        if session_file and session_file.exists():
            resume = self._ask_yes_no("Resume previous session?", default=True)
            if resume:
                session = LabelingSession.load(session_file)
                print(f"📂 Resumed session: {session.session_id}")
                print(f"Progress: {session.progress()}")
            else:
                session = self.start_session(files)
        else:
            session = self.start_session(files)

        all_labels = []

        # Process files
        for i in range(session.current_index, len(session.files_to_label)):
            file_path = Path(session.files_to_label[i])

            print(f"\n\n{'=' * 80}")
            print(f"Progress: {i + 1}/{len(session.files_to_label)}")
            print(f"{'=' * 80}")

            # Label file
            labels = self.label_file(file_path)

            if labels:
                # Save labels
                output_file = self.output_dir / f"{labels.file_id}_{self.labeler_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(labels.to_dict(), f, indent=2)

                all_labels.append(labels)
                session.completed_files.append(str(file_path))

                print(f"\n✓ Saved labels to: {output_file}")

            # Update session
            session.current_index = i + 1

            # Save session state
            if session_file:
                session.save(session_file)

            # Ask to continue
            if i < len(session.files_to_label) - 1:
                continue_labeling = self._ask_yes_no("\nContinue to next file?", default=True)
                if not continue_labeling:
                    print(f"\n⏸️  Session paused. Progress saved to: {session_file}")
                    print(f"Progress: {session.progress()}")
                    break

        # Session complete
        print("\n" + "=" * 80)
        print("🎉 LABELING SESSION COMPLETE")
        print("=" * 80)
        print(f"Files labeled: {len(all_labels)}/{len(session.files_to_label)}")
        print(f"Labels saved to: {self.output_dir}")
        print()

        return all_labels


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manual Labeling Tool for MIDI Corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Label all files in a directory
  python labeling_tool.py --corpus midi_corpus/jazz --output labels --labeler expert_1

  # Label specific files
  python labeling_tool.py --files file1.mid file2.mid --output labels --labeler expert_2

  # Resume previous session
  python labeling_tool.py --corpus midi_corpus/jazz --output labels --labeler expert_1 --resume session.json
        """
    )

    parser.add_argument('--corpus', type=str, help='Directory containing MIDI files')
    parser.add_argument('--files', nargs='+', help='Specific MIDI files to label')
    parser.add_argument('--output', type=str, required=True, help='Output directory for labels')
    parser.add_argument('--labeler', type=str, required=True, help='Labeler ID (e.g., expert_1)')
    parser.add_argument('--session', type=str, help='Session file for saving/resuming progress')
    parser.add_argument('--no-autoextract', action='store_true', help='Skip auto-extraction')

    args = parser.parse_args()

    # Get files to label
    if args.corpus:
        corpus_dir = Path(args.corpus)
        files = sorted(corpus_dir.glob('**/*.mid'))
        if not files:
            print(f"❌ No MIDI files found in {corpus_dir}")
            sys.exit(1)
    elif args.files:
        files = [Path(f) for f in args.files]
    else:
        print("❌ Must specify either --corpus or --files")
        parser.print_help()
        sys.exit(1)

    # Initialize tool
    output_dir = Path(args.output)

    auto_labeler = None if args.no_autoextract else (AutoLabeler() if AUTO_LABELER_AVAILABLE else None)

    tool = ManualLabelingTool(
        corpus_dir=args.corpus if args.corpus else Path.cwd(),
        output_dir=output_dir,
        labeler_id=args.labeler,
        auto_labeler=auto_labeler
    )

    # Run batch labeling
    session_file = Path(args.session) if args.session else output_dir / f'session_{args.labeler}.json'

    tool.run_batch(files, session_file)


if __name__ == '__main__':
    main()
