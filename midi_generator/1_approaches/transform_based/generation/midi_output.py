"""MIDI file output for generated arrangements."""

from typing import Dict, List, Tuple
from .orchestrator import Track, TrackSegment
from .timing import TimingAssigner

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


class MIDIWriter:
    """Write tracks to MIDI file."""

    def __init__(self, ticks_per_beat: int = 480, tempo: int = 120):
        if not MIDO_AVAILABLE:
            raise ImportError("mido library required. Install with: pip install mido")
        self.ticks_per_beat = ticks_per_beat
        self.tempo = tempo
        self.default_velocity = 80
        self.note_duration_fraction = 0.9

    def segments_to_notes(
        self,
        segments: List[TrackSegment],
        base_pitch: int = 60
    ) -> List[Tuple[int, int, int, int]]:
        """Convert segments to notes (onset, pitch, duration, velocity).

        Args:
            segments: List of track segments
            base_pitch: Starting pitch (e.g., 60 = middle C)

        Returns:
            List of (onset_time, pitch, duration, velocity) tuples
        """
        notes = []

        for segment in segments:
            onset = segment.onset_time
            ioi = TimingAssigner().bucket_to_ioi(segment.rhythm_bucket)
            velocity = self._bucket_to_velocity(segment.velocity_bucket)

            pitch = base_pitch + segment.pitch_offset
            note_dur = int(ioi * self.note_duration_fraction)
            notes.append((onset, pitch, note_dur, velocity))
            onset += ioi

            for interval in segment.pitch_intervals:
                pitch += interval
                pitch = max(0, min(127, pitch))
                notes.append((onset, pitch, note_dur, velocity))
                onset += ioi

        return notes

    def _bucket_to_velocity(self, velocity_bucket: int) -> int:
        """Convert velocity bucket to MIDI velocity."""
        velocity_map = {
            0: 40,   # pp
            1: 55,   # p
            2: 70,   # mp
            3: 85,   # mf
            4: 100,  # f
            5: 115,  # ff
        }
        return velocity_map.get(velocity_bucket, 80)

    def track_to_midi_track(
        self,
        track: Track,
        base_pitch: int = 60
    ) -> 'mido.MidiTrack':
        """Convert a Track to a mido MidiTrack."""
        midi_track = mido.MidiTrack()

        midi_track.append(mido.Message('program_change', program=track.gm_program, time=0))

        notes = self.segments_to_notes(track.segments, base_pitch)

        events = []
        for onset, pitch, duration, velocity in notes:
            if track.is_drum:
                events.append(('note_on', onset, pitch, velocity, 9))
                events.append(('note_off', onset + duration, pitch, 0, 9))
            else:
                events.append(('note_on', onset, pitch, velocity, 0))
                events.append(('note_off', onset + duration, pitch, 0, 0))

        events.sort(key=lambda x: (x[1], x[0] != 'note_on'))

        last_time = 0
        for event_type, time, pitch, velocity, channel in events:
            delta = time - last_time
            if event_type == 'note_on':
                midi_track.append(mido.Message(
                    'note_on', note=pitch, velocity=velocity, time=delta, channel=channel
                ))
            else:
                midi_track.append(mido.Message(
                    'note_off', note=pitch, velocity=0, time=delta, channel=channel
                ))
            last_time = time

        return midi_track

    def tracks_to_midi(
        self,
        tracks: Dict[str, Track],
        base_pitch: int = 60
    ) -> 'mido.MidiFile':
        """Convert multiple tracks to a MIDI file."""
        mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)

        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.tempo)))
        tempo_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))

        for name, track in tracks.items():
            midi_track = self.track_to_midi_track(track, base_pitch)
            midi_track.insert(0, mido.MetaMessage('track_name', name=name, time=0))
            mid.tracks.append(midi_track)

        return mid

    def save(
        self,
        tracks: Dict[str, Track],
        output_path: str,
        base_pitch: int = 60
    ):
        """Save tracks to a MIDI file."""
        mid = self.tracks_to_midi(tracks, base_pitch)
        mid.save(output_path)
        return output_path
