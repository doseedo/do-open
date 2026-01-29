"""
Hierarchical MIDI corpus representation.

Preserves track structure for meta-pattern discovery.
Each piece has separate tracks that can be analyzed for derivation relationships.

Architecture:
  corpus[piece_id] = {
      'tracks': {track_name: Tensor(T, F), ...},
      'segments': {track_name: [Tensor(seg_len, F), ...], ...},
      'metadata': {...}
  }

Author: Agent - Meta-Pattern Discovery
"""

import numpy as np
from typing import Dict, List, Tuple
import mido
from collections import defaultdict


class HierarchicalMIDICorpus:
    """
    MIDI corpus with preserved track structure for meta-pattern discovery.

    Unlike flattened representation (B, T, F), this keeps tracks separate
    so we can discover patterns like:
        sax_1 = Transpose(+7)(melody)
        sax_2 = Transpose(+3)(melody)
        sax_3 = Transpose(0)(melody)

    And abstract to meta-pattern:
        sax_voicing = λ(X): Transpose(X)(melody)
    """

    def __init__(self, max_time_steps: int = 2000, segment_bars: int = 4):
        """
        Args:
            max_time_steps: Max time steps per track
            segment_bars: Bars per segment for segment-level analysis
        """
        self.max_time_steps = max_time_steps
        self.segment_bars = segment_bars
        self.segment_steps = segment_bars * 16  # 16th note resolution

    def load_corpus_hierarchical(
        self,
        midi_files: List[mido.MidiFile],
        verbose: bool = True
    ) -> Dict:
        """
        Load MIDI files into hierarchical structure.

        Args:
            midi_files: List of mido.MidiFile objects
            verbose: Print progress

        Returns:
            corpus: {
                piece_id: {
                    'tracks': {track_name: np.ndarray(T, F)},
                    'segments': {track_name: [np.ndarray(seg_len, F)]},
                    'track_metadata': {track_name: {'program': int, 'channel': int}},
                    'global_metadata': {'tempo': int, 'time_signature': tuple}
                }
            }
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        corpus = {}
        n_workers = min(12, len(midi_files))
        lock = threading.Lock()
        processed_count = [0]  # Use list for closure mutability

        def process_one(args):
            i, midi_file = args
            try:
                piece_data = self._process_midi_file(midi_file)
                return (i, piece_data, None)
            except Exception as e:
                return (i, None, str(e)[:50])

        if verbose:
            print(f"  Using {n_workers} parallel workers...")

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_one, (i, mf)): i for i, mf in enumerate(midi_files)}
            for future in as_completed(futures):
                i, piece_data, error = future.result()
                if piece_data:
                    corpus[f'piece_{i}'] = piece_data
                elif error and verbose:
                    print(f"  [!] Failed to process piece {i}: {error}")

                with lock:
                    processed_count[0] += 1
                    if verbose and processed_count[0] % 100 == 0:
                        print(f"  Processed {processed_count[0]}/{len(midi_files)} pieces...")

        if verbose:
            print(f"\n✓ Loaded {len(corpus)} pieces hierarchically")
            print(f"  Total tracks: {sum(len(p['tracks']) for p in corpus.values())}")

        return corpus

    def _process_midi_file(self, midi_file: mido.MidiFile) -> Dict:
        """
        Process single MIDI file into hierarchical structure.

        Returns:
            {
                'tracks': {track_name: np.ndarray},
                'segments': {track_name: List[np.ndarray]},
                'track_metadata': {...},
                'global_metadata': {...}
            }
        """
        from core.space_level_transforms import extract_notes_from_midi

        # Extract notes with track information
        notes = extract_notes_from_midi(midi_file)

        if not notes:
            return None

        # Group notes by track
        notes_by_track = defaultdict(list)
        for note in notes:
            track_id = note.get('track', 0)
            program = note.get('program', 0)
            channel = note.get('channel', 0)

            # Create track identifier (program determines instrument)
            track_name = self._get_track_name(program, channel)
            notes_by_track[track_name].append(note)

        # Convert each track to tensor
        tracks = {}
        track_metadata = {}

        for track_name, track_notes in notes_by_track.items():
            if not track_notes:
                continue

            # Convert to tensor
            track_tensor = self._notes_to_tensor(track_notes, midi_file)
            tracks[track_name] = track_tensor

            # Store metadata
            track_metadata[track_name] = {
                'program': track_notes[0].get('program', 0),
                'channel': track_notes[0].get('channel', 0),
                'is_drum': track_notes[0].get('is_drum', False)
            }

        # Segment each track
        segments = {
            name: self._segment_track(tensor)
            for name, tensor in tracks.items()
        }

        # Extract global metadata
        global_metadata = {
            'tempo': 120,  # Default, could extract from MIDI
            'time_signature': (4, 4)
        }

        return {
            'tracks': tracks,
            'segments': segments,
            'track_metadata': track_metadata,
            'global_metadata': global_metadata
        }

    def _get_track_name(self, program: int, channel: int) -> str:
        """
        Get semantic track name from MIDI program number.

        General MIDI instrument mapping:
          0-7: Piano family
          24-31: Guitar family
          32-39: Bass family
          40-47: Strings family
          56-63: Brass family (trumpet, trombone, etc.)
          64-71: Reed family (sax, clarinet, etc.)
          Channel 9 = drums
        """
        if channel == 9:
            return 'drums'

        # Simplified mapping for common instruments
        if 0 <= program <= 7:
            return f'piano_{program}'
        elif 24 <= program <= 31:
            return f'guitar_{program-24}'
        elif 32 <= program <= 39:
            return f'bass_{program-32}'
        elif 40 <= program <= 47:
            return f'strings_{program-40}'
        elif 56 <= program <= 63:
            return f'brass_{program-56}'
        elif 64 <= program <= 71:
            return f'reed_{program-64}'
        else:
            return f'inst_{program}'

    def _notes_to_tensor(
        self,
        notes: List[Dict],
        midi_file: mido.MidiFile
    ) -> np.ndarray:
        """
        Convert notes to tensor representation.

        Returns:
            tensor: (T, 133) where T = max_time_steps
                Features: [pitch(128), velocity(1), program(1), channel(1), is_drum(1), track_id(1)]
        """
        # Create time grid (16th note resolution)
        ticks_per_16th = midi_file.ticks_per_beat // 4

        # Default tempo (120 BPM = 500000 microseconds per beat)
        # We need to convert seconds back to ticks
        default_tempo = 500000

        # Initialize tensor (T, 133)
        tensor = np.zeros((self.max_time_steps, 133))

        # Fill in notes
        for note in notes:
            # CRITICAL FIX: note['start_time'] and note['duration'] are in SECONDS
            # but we need to convert to timesteps (16th notes)
            # Convert seconds -> ticks -> steps
            start_ticks = mido.second2tick(note['start_time'], midi_file.ticks_per_beat, default_tempo)
            duration_ticks = mido.second2tick(note['duration'], midi_file.ticks_per_beat, default_tempo)

            start_step = int(start_ticks / ticks_per_16th)
            duration_steps = max(1, int(duration_ticks / ticks_per_16th))
            end_step = min(start_step + duration_steps, self.max_time_steps)

            if start_step >= self.max_time_steps:
                continue

            for step in range(start_step, end_step):
                # Pitch (one-hot)
                pitch = np.clip(note['pitch'], 0, 127)
                tensor[step, pitch] = 1.0

                # Velocity
                tensor[step, 128] = note['velocity'] / 127.0

                # Program/Instrument
                tensor[step, 129] = note.get('program', 0) / 127.0

                # Channel
                tensor[step, 130] = note.get('channel', 0) / 15.0

                # Is drum
                tensor[step, 131] = 1.0 if note.get('is_drum', False) else 0.0

                # Track ID
                tensor[step, 132] = np.clip(note.get('track', 0), 0, 19) / 20.0

        return tensor

    def _segment_track(self, track_tensor: np.ndarray) -> List[np.ndarray]:
        """
        Segment track into fixed-length chunks.

        Args:
            track_tensor: (T, F) track data

        Returns:
            segments: List of (segment_steps, F) arrays
        """
        T, F = track_tensor.shape
        num_segments = T // self.segment_steps

        segments = []
        for i in range(num_segments):
            start = i * self.segment_steps
            end = start + self.segment_steps
            segments.append(track_tensor[start:end])

        return segments

    def get_track_pairs(self, corpus: Dict) -> List[Tuple]:
        """
        Get all track pairs within each piece for transform discovery.

        Returns:
            pairs: [(piece_id, source_track_name, target_track_name, source_tensor, target_tensor)]
        """
        pairs = []

        for piece_id, piece_data in corpus.items():
            tracks = piece_data['tracks']

            for source_name, source_tensor in tracks.items():
                for target_name, target_tensor in tracks.items():
                    if source_name == target_name:
                        continue

                    pairs.append((
                        piece_id,
                        source_name,
                        target_name,
                        source_tensor,
                        target_tensor
                    ))

        return pairs
