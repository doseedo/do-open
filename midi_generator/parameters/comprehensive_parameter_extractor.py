"""
Comprehensive Parameter Extractor - Agent 3
============================================

Extracts 300+ comprehensive parameters from MIDI files:
- 50 Hierarchical parameters (Level 1, 2, 3)
- 120 Modular semantic parameters (Harmony, Rhythm, Form, Orchestration, Texture, Cross-dimensional)
- 130 Rich data extensions (Per-track, Temporal, Genre-specific)

This is the main extraction pipeline that combines all parameter extractors.

Author: Agent 3 - Comprehensive Parameter Extraction Specialist
Date: November 21, 2025
"""

import json
import numpy as np
import mido
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
import multiprocessing as mp
from tqdm import tqdm


# Import hierarchical extractor
from midi_generator.parameters.hierarchical_extractor_v2 import (
    HierarchicalParameterExtractorV2,
    MIDIAnalysis
)

# Import modular semantic extractors
from midi_generator.parameters.modular_semantic_extractors import (
    MIDIAnalysisData,
    HarmonyParameterExtractor,
    RhythmParameterExtractor
)

# Import form/texture extractors
from midi_generator.parameters.form_texture_extractors import (
    FormParameterExtractor,
    OrchestrationParameterExtractor,
    TextureParameterExtractor,
    CrossDimensionalExtractor
)

# Import rich data extractors
from midi_generator.parameters.rich_data_extractors import (
    PerTrackParameterExtractor,
    TemporalEvolutionExtractor,
    GenreSpecificExtractor
)


class ComprehensiveParameterExtractor:
    """
    Extracts all 300+ parameters from MIDI files.

    Usage:
        extractor = ComprehensiveParameterExtractor()
        params = extractor.extract("path/to/file.mid")
        # Returns dict with 300+ parameters
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Initialize hierarchical extractor
        self.hierarchical_extractor = HierarchicalParameterExtractorV2(verbose=False)

        # Initialize modular semantic extractors
        self.harmony_extractor = HarmonyParameterExtractor()
        self.rhythm_extractor = RhythmParameterExtractor()
        self.form_extractor = FormParameterExtractor()
        self.orchestration_extractor = OrchestrationParameterExtractor()
        self.texture_extractor = TextureParameterExtractor()
        self.cross_dim_extractor = CrossDimensionalExtractor()

        # Initialize rich data extractors
        self.per_track_extractor = PerTrackParameterExtractor()
        self.temporal_extractor = TemporalEvolutionExtractor()
        self.genre_specific_extractor = GenreSpecificExtractor()

        if self.verbose:
            print("ComprehensiveParameterExtractor initialized")
            print("  - Hierarchical parameters: 50")
            print("  - Modular semantic parameters: 120")
            print("  - Rich data extensions: 130")
            print("  - Total: 300+")

    def extract(self, midi_path: str) -> Dict[str, Any]:
        """
        Extract all 300+ parameters from a MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dictionary with all parameters organized by category
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Extracting from: {Path(midi_path).name}")
            print(f"{'='*70}")

        try:
            # Step 1: Extract hierarchical parameters (50 params)
            hierarchical_data = self.hierarchical_extractor.extract_complete(str(midi_path))

            # Extract the parameters (without features for now)
            hierarchical_params = {
                'level1': hierarchical_data['parameters']['level1_global'],
                'level2': hierarchical_data['parameters']['level2_universal'],
                'level3': hierarchical_data['parameters']['level3_genre_specific']
            }

            # Get genre for genre-specific extraction
            genre = hierarchical_params['level1'].get('genre.primary', 'unknown')

            # Step 2: Analyze MIDI file for modular/rich extractors
            analysis_data = self._analyze_midi_comprehensive(midi_path)

            # Step 3: Extract modular semantic parameters (120 params)
            if self.verbose:
                print("Extracting modular semantic parameters...")

            harmony_params = self.harmony_extractor.extract(analysis_data)
            rhythm_params = self.rhythm_extractor.extract(analysis_data)
            form_params = self.form_extractor.extract(analysis_data)
            orchestration_params = self.orchestration_extractor.extract(analysis_data)
            texture_params = self.texture_extractor.extract(analysis_data)

            # Cross-dimensional requires other params as input
            cross_dim_params = self.cross_dim_extractor.extract(
                analysis_data, harmony_params, rhythm_params,
                form_params, orchestration_params, texture_params
            )

            modular_semantic_params = {
                'harmony': harmony_params,      # 30 params
                'rhythm': rhythm_params,        # 20 params
                'form': form_params,            # 15 params
                'orchestration': orchestration_params,  # 25 params
                'texture': texture_params,      # 20 params
                'cross_dimensional': cross_dim_params  # 10 params
            }

            # Step 4: Extract rich data extensions (130 params)
            if self.verbose:
                print("Extracting rich data extensions...")

            per_track_params = self.per_track_extractor.extract(analysis_data)  # 80 params
            temporal_params = self.temporal_extractor.extract(analysis_data)    # 40 params
            genre_specific_params = self.genre_specific_extractor.extract(      # 10 params
                analysis_data, genre
            )

            rich_extensions = {
                'per_track': per_track_params,
                'temporal': temporal_params,
                'genre_specific': genre_specific_params
            }

            # Step 5: Combine all parameters
            comprehensive_params = {
                'file_id': Path(midi_path).stem,
                'midi_path': str(midi_path),
                'genre': genre,

                # 50 hierarchical parameters
                'hierarchical': hierarchical_params,

                # 120 modular semantic parameters
                'modular_semantic': modular_semantic_params,

                # 130 rich data extensions
                'rich_extensions': rich_extensions,

                # Metadata
                'metadata': {
                    'total_notes': len(analysis_data.all_notes),
                    'duration_seconds': analysis_data.duration_seconds,
                    'tempo_bpm': analysis_data.tempo_bpm,
                    'time_signature': analysis_data.time_signature,
                    'instrument_count': len(analysis_data.instrument_programs),
                    'extraction_version': '3.0.0'
                }
            }

            # Count total parameters
            param_count = self._count_parameters(comprehensive_params)

            if self.verbose:
                print(f"\n✅ Extraction complete!")
                print(f"   Total parameters extracted: {param_count}")
                print(f"   Genre: {genre}")
                print(f"   Duration: {analysis_data.duration_seconds:.1f}s")

            return comprehensive_params

        except Exception as e:
            warnings.warn(f"Error extracting from {midi_path}: {e}")
            # Return fallback parameters
            return self._get_fallback_parameters(midi_path)

    def _analyze_midi_comprehensive(self, midi_path: str) -> MIDIAnalysisData:
        """Analyze MIDI file and create MIDIAnalysisData"""
        midi_file = mido.MidiFile(midi_path)

        # Initialize data structures
        all_notes = []
        tracks_by_instrument = defaultdict(list)
        instrument_programs = []
        tempo_bpm = 120.0
        time_signature = "4/4"

        # Extract tempo and time signature
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo_bpm = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':
                    time_signature = f"{msg.numerator}/{msg.denominator}"

        duration_seconds = midi_file.length

        # Extract all notes
        for track_idx, track in enumerate(midi_file.tracks):
            current_time = 0.0
            instrument_program = 0
            active_notes = {}

            for msg in track:
                current_time += mido.tick2second(
                    msg.time, midi_file.ticks_per_beat,
                    mido.bpm2tempo(tempo_bpm)
                )

                if msg.type == 'program_change':
                    instrument_program = msg.program
                    if instrument_program not in instrument_programs:
                        instrument_programs.append(instrument_program)

                elif msg.type == 'note_on' and msg.velocity > 0:
                    key = (track_idx, msg.channel, msg.note)
                    active_notes[key] = (current_time, msg.velocity)

                elif msg.type in ['note_off', 'note_on']:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        continue

                    key = (track_idx, msg.channel, msg.note)
                    if key in active_notes:
                        onset_time, velocity = active_notes[key]
                        duration = current_time - onset_time

                        note_data = {
                            'pitch': msg.note,
                            'velocity': velocity,
                            'onset': onset_time,
                            'duration': duration,
                            'track': track_idx,
                            'channel': msg.channel,
                            'program': instrument_program
                        }

                        all_notes.append(note_data)
                        tracks_by_instrument[instrument_program].append(note_data)

                        del active_notes[key]

        # Detect chords (simultaneous notes)
        chord_notes = self._detect_chords(all_notes)

        # Identify melody track
        melody_notes = []
        melody_pitches = []
        if tracks_by_instrument:
            # Highest average pitch is likely melody
            melody_program = max(
                tracks_by_instrument.keys(),
                key=lambda p: np.mean([n['pitch'] for n in tracks_by_instrument[p]])
                if tracks_by_instrument[p] else 0
            )
            melody_notes = tracks_by_instrument[melody_program]
            melody_pitches = [n['pitch'] for n in melody_notes]

        # Create MIDIAnalysisData
        return MIDIAnalysisData(
            all_notes=all_notes,
            melody_notes=melody_notes,
            chord_notes=chord_notes,
            tracks_by_instrument=dict(tracks_by_instrument),
            all_pitches=[n['pitch'] for n in all_notes],
            melody_pitches=melody_pitches,
            all_velocities=[n['velocity'] for n in all_notes],
            note_onsets=[n['onset'] for n in all_notes],
            note_durations=[n['duration'] for n in all_notes],
            tempo_bpm=tempo_bpm,
            duration_seconds=duration_seconds,
            time_signature=time_signature,
            instrument_programs=instrument_programs
        )

    def _detect_chords(self, notes: List[Dict], time_window: float = 0.05) -> List[List[Dict]]:
        """Detect simultaneous notes (chords)"""
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n['onset'])
        chords = []
        current_chord = [sorted_notes[0]]

        for note in sorted_notes[1:]:
            if abs(note['onset'] - current_chord[0]['onset']) < time_window:
                current_chord.append(note)
            else:
                if len(current_chord) >= 2:
                    chords.append(current_chord)
                current_chord = [note]

        if len(current_chord) >= 2:
            chords.append(current_chord)

        return chords

    def _count_parameters(self, params: Dict) -> int:
        """Count total number of parameters"""
        count = 0

        # Hierarchical
        if 'hierarchical' in params:
            for level_data in params['hierarchical'].values():
                if isinstance(level_data, dict):
                    for value in level_data.values():
                        if isinstance(value, dict):
                            count += len(value)
                        else:
                            count += 1

        # Modular semantic
        if 'modular_semantic' in params:
            for dim_params in params['modular_semantic'].values():
                if isinstance(dim_params, dict):
                    count += len(dim_params)

        # Rich extensions
        if 'rich_extensions' in params:
            # Per-track (list of dicts)
            if 'per_track' in params['rich_extensions']:
                for track_params in params['rich_extensions']['per_track']:
                    if isinstance(track_params, dict):
                        count += len(track_params)

            # Temporal (list of dicts)
            if 'temporal' in params['rich_extensions']:
                for section_params in params['rich_extensions']['temporal']:
                    if isinstance(section_params, dict):
                        count += len(section_params)

            # Genre-specific (dict)
            if 'genre_specific' in params['rich_extensions']:
                count += len(params['rich_extensions']['genre_specific'])

        return count

    def _get_fallback_parameters(self, midi_path: str) -> Dict[str, Any]:
        """Return fallback parameters for failed extraction"""
        return {
            'file_id': Path(midi_path).stem,
            'midi_path': str(midi_path),
            'genre': 'unknown',
            'hierarchical': {},
            'modular_semantic': {},
            'rich_extensions': {},
            'metadata': {
                'extraction_failed': True,
                'extraction_version': '3.0.0'
            }
        }

    def extract_batch(self, midi_paths: List[str],
                     output_path: Optional[str] = None,
                     num_workers: int = 16,
                     checkpoint_frequency: int = 100) -> List[Dict[str, Any]]:
        """
        Extract parameters from multiple MIDI files in parallel.

        Args:
            midi_paths: List of paths to MIDI files
            output_path: Optional path to save results
            num_workers: Number of parallel workers
            checkpoint_frequency: Save checkpoint every N files

        Returns:
            List of parameter dictionaries
        """
        print(f"\n{'='*70}")
        print(f"BATCH PARAMETER EXTRACTION")
        print(f"{'='*70}")
        print(f"Files to process: {len(midi_paths)}")
        print(f"Parallel workers: {num_workers}")
        print(f"Checkpoint frequency: {checkpoint_frequency}")
        print(f"{'='*70}\n")

        results = []

        # Process in parallel with progress bar
        with mp.Pool(processes=num_workers) as pool:
            for i, result in enumerate(tqdm(
                pool.imap(self._extract_single, midi_paths),
                total=len(midi_paths),
                desc="Extracting parameters"
            )):
                results.append(result)

                # Checkpoint
                if output_path and (i + 1) % checkpoint_frequency == 0:
                    self._save_checkpoint(results, output_path, i + 1)

        # Final save
        if output_path:
            self._save_results(results, output_path)

        print(f"\n✅ Batch extraction complete!")
        print(f"   Total files processed: {len(results)}")
        print(f"   Successful: {sum(1 for r in results if not r['metadata'].get('extraction_failed'))}")
        print(f"   Failed: {sum(1 for r in results if r['metadata'].get('extraction_failed'))}")

        return results

    def _extract_single(self, midi_path: str) -> Dict[str, Any]:
        """Extract parameters from single file (for multiprocessing)"""
        try:
            return self.extract(midi_path)
        except Exception as e:
            warnings.warn(f"Failed to extract {midi_path}: {e}")
            return self._get_fallback_parameters(midi_path)

    def _save_checkpoint(self, results: List[Dict], output_path: str, count: int):
        """Save checkpoint"""
        checkpoint_path = f"{output_path}.checkpoint_{count}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")

    def _save_results(self, results: List[Dict], output_path: str):
        """Save final results"""
        output_data = {
            'metadata': {
                'total_samples': len(results),
                'parameter_count': 300,
                'extraction_version': '3.0.0'
            },
            'samples': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved: {output_path}")


def main():
    """Demo/test the comprehensive extractor"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python comprehensive_parameter_extractor.py <midi_file.mid>")
        print("   or: python comprehensive_parameter_extractor.py <directory>")
        return

    path = Path(sys.argv[1])
    extractor = ComprehensiveParameterExtractor(verbose=True)

    if path.is_file():
        # Single file
        params = extractor.extract(str(path))

        # Save results
        output_path = path.with_suffix('.params.json')
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"\n💾 Parameters saved to: {output_path}")

        # Print summary
        print("\n" + "="*70)
        print("PARAMETER SUMMARY")
        print("="*70)
        print(f"Total parameters: {extractor._count_parameters(params)}")
        print(f"\nHierarchical (50):")
        print(f"  - Level 1: {len(params['hierarchical']['level1'])} params")
        print(f"  - Level 2: {sum(len(v) for v in params['hierarchical']['level2'].values())} params")
        print(f"  - Level 3: {sum(len(v) if isinstance(v, dict) else 1 for v in params['hierarchical']['level3'].values())} params")
        print(f"\nModular Semantic (120):")
        for dim, dim_params in params['modular_semantic'].items():
            print(f"  - {dim}: {len(dim_params)} params")
        print(f"\nRich Extensions (130):")
        print(f"  - Per-track: {len(params['rich_extensions']['per_track'])} tracks × 10 = {len(params['rich_extensions']['per_track']) * 10} params")
        print(f"  - Temporal: {len(params['rich_extensions']['temporal'])} sections × 10 = {len(params['rich_extensions']['temporal']) * 10} params")
        print(f"  - Genre-specific: {len(params['rich_extensions']['genre_specific'])} params")

    elif path.is_dir():
        # Directory of MIDI files
        midi_files = list(path.glob("**/*.mid")) + list(path.glob("**/*.midi"))
        print(f"Found {len(midi_files)} MIDI files")

        if midi_files:
            output_path = path / "labeled_dataset_comprehensive.json"
            results = extractor.extract_batch(
                [str(f) for f in midi_files],
                output_path=str(output_path),
                num_workers=16
            )
    else:
        print(f"Error: {path} is neither a file nor a directory")


if __name__ == "__main__":
    main()
