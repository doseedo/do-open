"""
Simple drum sampler for API integration
Matches the logic from the original Gradio app
"""
import os
import glob
import mido
import numpy as np
import soundfile as sf
from typing import List, Tuple, Optional, Dict
import json
import random
from scipy import signal as sp_signal

# Configuration
STANDARD_SAMPLE_RATE = 44100
DEFAULT_TEMPO_BPM = 120
SHARED_TEMPO_MICROSECONDS = int(60000000 / DEFAULT_TEMPO_BPM)
DRUM_CROPS_DIR = "/home/arlo/free-midi-chords/DrumCrops"
DRUM_SUSTAIN_SECONDS = 4.0  # How long each drum sample lasts

# DRUM_SAMPLE_NAMES from original app - defines the mapping
DRUM_SAMPLE_NAMES = [
    "KICK", "KICK", "SIDESTICK", "SNARE", "CLAP", "SNARE", "KICK", "HAT",
    "TOM", "HATCLOSE", "TOM", "OPENHAT", "TOM", "TOM", "CRASH", "TOM",
    "RIDE", "CHINA", "RIDEBELL", "TAMB", "CRASH SHORT", "COWBELL", "CRASH", "PERCSTICK",
    "RIDE", "PERCDRUM", "PERCDRUM", "PERCDRUM", "PERCDRUM", "PERCDRUM", "PERCDRUM", "PERCDRUM",
    "PERCCYMBAL", "PERCCYMBAL", "PERCSHAKER", "PERKSHAKER", "PERC WHISTLE", "PERCWHISTLE", "PERC STICK", "PERCSTICK",
    "PERCSTICK", "PERCSTICK", "PERCSTICK", "PERCVOX", "PERCVOX", "PERCBELL", "PERCBELL", "PERCSHAKER",
    "PERCBELL", "PERCBELL", "PERCSTICK", "SNARE", "TOM", "SNARE", "FX", "RIDE",
    "PERCSTICK", "FX", "FX", "FX", "CRASHFX", "PERCSTICK", "PERCDRUM", "PERCBELL"
]

def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if original_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / original_sr)
    return sp_signal.resample(audio, num_samples).astype(audio.dtype)

class SimpleDrumSampler:
    """Simplified drum sampler matching original app logic"""

    def __init__(self, crops_dir: str = DRUM_CROPS_DIR):
        self.crops_dir = crops_dir
        self.drum_presets = {}  # Kit name -> dict of {midi_note: sample_data}
        self.sample_rate = STANDARD_SAMPLE_RATE
        self.current_preset = None
        self.load_drum_kits()

    def load_drum_kits(self):
        """Load drum kits from DrumCrops directory, splitting into 24 samples each"""
        print(f"Loading drum kits from {self.crops_dir}")

        # Get all .aif/.wav files (skip ._ resource fork files and perc/fx files)
        crop_files = []
        for ext in ["*.aif", "*.wav", "*.aiff"]:
            found = glob.glob(os.path.join(self.crops_dir, ext))
            crop_files.extend(found)

        # Filter out ._* files and perc/fx files
        crop_files = [f for f in crop_files
                      if not os.path.basename(f).startswith("._")
                      and 'perc' not in os.path.basename(f).lower()
                      and 'fx' not in os.path.basename(f).lower()]

        crop_files.sort()
        print(f"Found {len(crop_files)} drum kit files")

        for crop_file in crop_files:
            try:
                kit_name = os.path.basename(crop_file).replace('.aif', '').replace('.wav', '').replace('.aiff', '')
                self.load_drum_kit(crop_file, kit_name)
            except Exception as e:
                print(f"Error loading {crop_file}: {e}")

        # Set first kit as current
        if self.drum_presets:
            self.current_preset = list(self.drum_presets.keys())[0]
            print(f"Loaded {len(self.drum_presets)} drum kits. Current: {self.current_preset}")

    def load_drum_kit(self, crop_file: str, kit_name: str):
        """Load a single drum kit and split into 24 individual samples"""
        # Load audio
        audio, sr = sf.read(crop_file)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            audio = resample_audio(audio, sr, self.sample_rate)

        # Calculate sample boundaries
        # Original uses 5.03 seconds per sample or onset detection
        # For simplicity, use fixed 5.03s timing (can improve with onset detection later)
        seconds_per_sample = 5.03
        sustain_samples = int(DRUM_SUSTAIN_SECONDS * self.sample_rate)

        drum_samples = {}

        # Extract 24 samples (MIDI notes 28-51)
        for i in range(min(24, len(DRUM_SAMPLE_NAMES))):
            sample_name = DRUM_SAMPLE_NAMES[i]

            # Skip PERC* and *FX samples
            if sample_name.startswith("PERC") or sample_name.endswith("FX"):
                continue

            midi_note = 28 + i
            start_time = i * seconds_per_sample
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + sustain_samples

            # Extract sample if within bounds
            if start_sample < len(audio) and end_sample <= len(audio):
                sample_audio = audio[start_sample:end_sample].astype(np.float32)

                drum_samples[midi_note] = {
                    'audio': sample_audio,
                    'name': sample_name,
                    'file': crop_file
                }

        self.drum_presets[kit_name] = drum_samples
        print(f"  Loaded kit '{kit_name}': {len(drum_samples)} samples")

    def get_random_kit(self) -> str:
        """Get a random drum kit name"""
        if not self.drum_presets:
            return None
        return random.choice(list(self.drum_presets.keys()))

    def get_sample_for_note(self, midi_note: int, kit_name: str = None) -> Optional[Dict]:
        """Get drum sample for a MIDI note from specified kit or current kit"""
        if kit_name is None:
            kit_name = self.current_preset

        if not kit_name or kit_name not in self.drum_presets:
            return None

        return self.drum_presets[kit_name].get(midi_note)

    def render_midi(self, midi_path: str, output_path: str, kit_name: str = None, bpm: int = 120) -> Tuple[int, np.ndarray]:
        """
        Render a MIDI file with drum samples

        Args:
            midi_path: Path to MIDI file
            output_path: Path to save output WAV
            kit_name: Drum kit to use (if None, uses a random kit)
            bpm: Target BPM for rendering (default: 120)

        Returns:
            (sample_rate, audio_array, kit_used)
        """
        # Select kit
        if kit_name is None:
            kit_name = self.get_random_kit()

        if not kit_name:
            raise Exception("No drum kits available")

        print(f"Rendering with drum kit: {kit_name} at {bpm} BPM")

        # Load MIDI file
        mid = mido.MidiFile(midi_path)

        # Calculate MIDI duration using specified BPM (IGNORE MIDI's embedded tempo)
        tempo_microseconds = int(60000000 / bpm)  # Convert BPM to microseconds per beat
        tempo = tempo_microseconds
        total_time = 0.0

        for msg in mido.merge_tracks(mid.tracks):
            dt = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            total_time += dt
            # REMOVED: Don't use MIDI's embedded tempo - always use the BPM parameter
            # if msg.type == 'set_tempo':
            #     tempo = msg.tempo

        # Create output buffer (add 2 seconds for drum tails)
        output_duration = total_time + 2.0
        output_samples = int(output_duration * self.sample_rate)
        output_audio = np.zeros(output_samples, dtype=np.float32)

        # Render MIDI messages using specified BPM (IGNORE embedded tempo)
        tempo = tempo_microseconds
        t_sec = 0.0

        for msg in mido.merge_tracks(mid.tracks):
            dt = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            t_sec += dt

            # REMOVED: Don't use MIDI's embedded tempo - always use the BPM parameter
            # if msg.type == 'set_tempo':
            #     tempo = msg.tempo

            if msg.type == 'note_on' and msg.velocity > 0:
                # Only process drum range (28-91)
                if 28 <= msg.note <= 91:
                    sample_data = self.get_sample_for_note(msg.note, kit_name)

                    if sample_data:
                        sample_audio = sample_data['audio'].copy()

                        # Apply velocity scaling
                        velocity_scale = msg.velocity / 127.0
                        sample_audio = sample_audio * velocity_scale

                        # Apply sample-specific level adjustments (from original)
                        sample_index = msg.note - 28
                        if sample_index < len(DRUM_SAMPLE_NAMES):
                            sample_name = DRUM_SAMPLE_NAMES[sample_index]

                            # Boost snare by 27.5%
                            if sample_name == "SNARE":
                                sample_audio = sample_audio * 1.275
                            # Lower crash by 37.25%
                            elif sample_name in ["CRASH", "CRASH SHORT"]:
                                sample_audio = sample_audio * 0.6375
                            # Lower clap by 25%
                            elif sample_name == "CLAP":
                                sample_audio = sample_audio * 0.75

                        # Apply fade in/out to avoid clicks
                        fade_samples = int(0.01 * self.sample_rate)
                        if len(sample_audio) > 2 * fade_samples:
                            fade_in = np.linspace(0, 1, fade_samples)
                            sample_audio[:fade_samples] *= fade_in

                            fade_out = np.linspace(1, 0, fade_samples)
                            sample_audio[-fade_samples:] *= fade_out

                        # Mix into output buffer
                        start_sample = int(t_sec * self.sample_rate)
                        end_sample = min(start_sample + len(sample_audio), len(output_audio))
                        sample_len = end_sample - start_sample

                        if sample_len > 0:
                            output_audio[start_sample:end_sample] += sample_audio[:sample_len]

        # Normalize
        max_val = np.max(np.abs(output_audio))
        if max_val > 0:
            output_audio = output_audio / max_val * 0.8

        # Save to file
        sf.write(output_path, output_audio, self.sample_rate)

        return self.sample_rate, output_audio, kit_name


def render_drums_from_midi(midi_path: str, output_path: str, kit_name: str = None) -> dict:
    """
    API function to render drums from MIDI

    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output WAV file
        kit_name: Optional specific drum kit to use

    Returns:
        dict with status, output path, and kit used
    """
    try:
        sampler = SimpleDrumSampler()
        sr, audio, kit_used = sampler.render_midi(midi_path, output_path, kit_name)

        return {
            "status": "success",
            "output_path": output_path,
            "sample_rate": sr,
            "duration": len(audio) / sr,
            "kit_used": kit_used
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        midi_file = sys.argv[1]
        output = "/tmp/test_drums.wav"
        result = render_drums_from_midi(midi_file, output)
        print(json.dumps(result, indent=2))
