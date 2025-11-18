"""
MIDI Constants and Definitions

Comprehensive MIDI constants including:
- General MIDI (GM) drum mappings
- MIDI CC (Control Change) numbers
- Standard PPQN (Pulses Per Quarter Note) resolutions
- MIDI channel definitions
- Note number ranges for instruments
"""

from typing import Dict, List

# ============================================================================
# PPQN (Pulses Per Quarter Note) Resolutions
# ============================================================================

# Standard MIDI resolution
PPQN_STANDARD = 480

# High-precision for sub-tick timing (groove, humanization)
PPQN_HIGH_RES = 960

# Ultra-high precision for advanced rhythm algorithms
PPQN_ULTRA_HIGH = 1920

# Default PPQN for the library
DEFAULT_PPQN = PPQN_HIGH_RES


# ============================================================================
# General MIDI Drum Map (Channel 10)
# ============================================================================

# Standard GM drum map (MIDI note numbers)
GM_DRUM_MAP: Dict[str, int] = {
    # Kick Drums
    'ACOUSTIC_BASS_DRUM': 35,
    'BASS_DRUM_1': 36,

    # Snare Drums
    'SIDE_STICK': 37,
    'ACOUSTIC_SNARE': 38,
    'HAND_CLAP': 39,
    'ELECTRIC_SNARE': 40,

    # Toms
    'LOW_FLOOR_TOM': 41,
    'CLOSED_HI_HAT': 42,
    'HIGH_FLOOR_TOM': 43,
    'PEDAL_HI_HAT': 44,
    'LOW_TOM': 45,
    'OPEN_HI_HAT': 46,
    'LOW_MID_TOM': 47,
    'HI_MID_TOM': 48,
    'CRASH_CYMBAL_1': 49,
    'HIGH_TOM': 50,
    'RIDE_CYMBAL_1': 51,
    'CHINESE_CYMBAL': 52,
    'RIDE_BELL': 53,
    'TAMBOURINE': 54,
    'SPLASH_CYMBAL': 55,
    'COWBELL': 56,
    'CRASH_CYMBAL_2': 57,
    'VIBRASLAP': 58,
    'RIDE_CYMBAL_2': 59,

    # Percussion
    'HI_BONGO': 60,
    'LOW_BONGO': 61,
    'MUTE_HI_CONGA': 62,
    'OPEN_HI_CONGA': 63,
    'LOW_CONGA': 64,
    'HIGH_TIMBALE': 65,
    'LOW_TIMBALE': 66,
    'HIGH_AGOGO': 67,
    'LOW_AGOGO': 68,
    'CABASA': 69,
    'MARACAS': 70,
    'SHORT_WHISTLE': 71,
    'LONG_WHISTLE': 72,
    'SHORT_GUIRO': 73,
    'LONG_GUIRO': 74,
    'CLAVES': 75,
    'HI_WOOD_BLOCK': 76,
    'LOW_WOOD_BLOCK': 77,
    'MUTE_CUICA': 78,
    'OPEN_CUICA': 79,
    'MUTE_TRIANGLE': 80,
    'OPEN_TRIANGLE': 81,
}

# Reverse mapping: note number -> drum name
GM_DRUM_NAME: Dict[int, str] = {v: k for k, v in GM_DRUM_MAP.items()}

# Simplified drum categories
KICK_NOTES = [35, 36]
SNARE_NOTES = [37, 38, 40]
HIHAT_NOTES = [42, 44, 46]
TOM_NOTES = [41, 43, 45, 47, 48, 50]
CYMBAL_NOTES = [49, 51, 52, 53, 55, 57, 59]
PERCUSSION_NOTES = list(range(60, 82))


# ============================================================================
# MIDI CC (Control Change) Numbers
# ============================================================================

MIDI_CC: Dict[str, int] = {
    # Standard Controllers
    'BANK_SELECT': 0,
    'MODULATION': 1,
    'BREATH_CONTROLLER': 2,
    'FOOT_CONTROLLER': 4,
    'PORTAMENTO_TIME': 5,
    'DATA_ENTRY': 6,
    'VOLUME': 7,
    'BALANCE': 8,
    'PAN': 10,
    'EXPRESSION': 11,
    'EFFECT_CONTROL_1': 12,
    'EFFECT_CONTROL_2': 13,

    # General Purpose Controllers
    'GP_CONTROLLER_1': 16,
    'GP_CONTROLLER_2': 17,
    'GP_CONTROLLER_3': 18,
    'GP_CONTROLLER_4': 19,

    # Sustain and Expression
    'DAMPER_PEDAL': 64,  # Sustain
    'PORTAMENTO': 65,
    'SOSTENUTO': 66,
    'SOFT_PEDAL': 67,
    'LEGATO_FOOTSWITCH': 68,
    'HOLD_2': 69,

    # Sound Controllers
    'SOUND_CONTROLLER_1': 70,  # Sound Variation
    'SOUND_CONTROLLER_2': 71,  # Timbre/Harmonic Intensity
    'SOUND_CONTROLLER_3': 72,  # Release Time
    'SOUND_CONTROLLER_4': 73,  # Attack Time
    'SOUND_CONTROLLER_5': 74,  # Brightness
    'SOUND_CONTROLLER_6': 75,
    'SOUND_CONTROLLER_7': 76,
    'SOUND_CONTROLLER_8': 77,
    'SOUND_CONTROLLER_9': 78,
    'SOUND_CONTROLLER_10': 79,

    # Effects Depth
    'REVERB_DEPTH': 91,
    'TREMOLO_DEPTH': 92,
    'CHORUS_DEPTH': 93,
    'DETUNE_DEPTH': 94,
    'PHASER_DEPTH': 95,

    # All Notes Off
    'ALL_SOUND_OFF': 120,
    'RESET_ALL_CONTROLLERS': 121,
    'ALL_NOTES_OFF': 123,
}


# ============================================================================
# MIDI Channels
# ============================================================================

DRUM_CHANNEL = 9  # Channel 10 in 1-based indexing (9 in 0-based)
DEFAULT_CHANNEL = 0


# ============================================================================
# Note Ranges for Instruments
# ============================================================================

INSTRUMENT_RANGES: Dict[str, tuple] = {
    # Strings
    'violin': (55, 103),      # G3-G7
    'viola': (48, 91),        # C3-G6
    'cello': (36, 76),        # C2-E5
    'double_bass': (28, 67),  # E1-G4

    # Woodwinds
    'flute': (60, 96),        # C4-C7
    'oboe': (58, 91),         # Bb3-G6
    'clarinet': (50, 94),     # D3-Bb6
    'bassoon': (34, 75),      # Bb1-Eb5

    # Brass
    'trumpet': (55, 82),      # G3-Bb5
    'horn': (41, 77),         # F2-F5
    'trombone': (40, 72),     # E2-C5
    'tuba': (28, 58),         # E1-Bb3

    # Piano
    'piano': (21, 108),       # A0-C8

    # Guitar
    'guitar': (40, 88),       # E2-E6
    'bass_guitar': (28, 55),  # E1-G3
}


# ============================================================================
# Timing Constants
# ============================================================================

# Microseconds per beat at different tempos
def tempo_to_microseconds(bpm: float) -> int:
    """Convert BPM to microseconds per beat"""
    return int(60_000_000 / bpm)

# Common time signatures (numerator, denominator)
TIME_SIGNATURES = {
    '4/4': (4, 4),
    '3/4': (3, 4),
    '6/8': (6, 8),
    '5/4': (5, 4),
    '7/8': (7, 8),
    '2/4': (2, 4),
    '12/8': (12, 8),
    '5/8': (5, 8),
    '7/4': (7, 4),
    '9/8': (9, 8),
}


# ============================================================================
# Velocity Constants
# ============================================================================

# Standard dynamic levels
DYNAMICS = {
    'pppp': 8,
    'ppp': 20,
    'pp': 31,
    'p': 42,
    'mp': 53,
    'mf': 64,
    'f': 80,
    'ff': 96,
    'fff': 112,
    'ffff': 127,
}

# Velocity ranges
VELOCITY_MIN = 1
VELOCITY_MAX = 127
VELOCITY_DEFAULT = 64


# ============================================================================
# General MIDI Program Numbers
# ============================================================================

GM_PROGRAMS: Dict[str, int] = {
    # Piano
    'acoustic_grand_piano': 0,
    'bright_acoustic_piano': 1,
    'electric_grand_piano': 2,
    'honky_tonk_piano': 3,
    'electric_piano_1': 4,
    'electric_piano_2': 5,
    'harpsichord': 6,
    'clavinet': 7,

    # Chromatic Percussion
    'celesta': 8,
    'glockenspiel': 9,
    'music_box': 10,
    'vibraphone': 11,
    'marimba': 12,
    'xylophone': 13,
    'tubular_bells': 14,
    'dulcimer': 15,

    # Organ
    'drawbar_organ': 16,
    'percussive_organ': 17,
    'rock_organ': 18,
    'church_organ': 19,
    'reed_organ': 20,
    'accordion': 21,
    'harmonica': 22,
    'tango_accordion': 23,

    # Guitar
    'acoustic_guitar_nylon': 24,
    'acoustic_guitar_steel': 25,
    'electric_guitar_jazz': 26,
    'electric_guitar_clean': 27,
    'electric_guitar_muted': 28,
    'overdriven_guitar': 29,
    'distortion_guitar': 30,
    'guitar_harmonics': 31,

    # Bass
    'acoustic_bass': 32,
    'electric_bass_finger': 33,
    'electric_bass_pick': 34,
    'fretless_bass': 35,
    'slap_bass_1': 36,
    'slap_bass_2': 37,
    'synth_bass_1': 38,
    'synth_bass_2': 39,

    # Strings
    'violin': 40,
    'viola': 41,
    'cello': 42,
    'contrabass': 43,
    'tremolo_strings': 44,
    'pizzicato_strings': 45,
    'orchestral_harp': 46,
    'timpani': 47,

    # Ensemble
    'string_ensemble_1': 48,
    'string_ensemble_2': 49,
    'synth_strings_1': 50,
    'synth_strings_2': 51,
    'choir_aahs': 52,
    'voice_oohs': 53,
    'synth_voice': 54,
    'orchestra_hit': 55,

    # Brass
    'trumpet': 56,
    'trombone': 57,
    'tuba': 58,
    'muted_trumpet': 59,
    'french_horn': 60,
    'brass_section': 61,
    'synth_brass_1': 62,
    'synth_brass_2': 63,

    # Reed
    'soprano_sax': 64,
    'alto_sax': 65,
    'tenor_sax': 66,
    'baritone_sax': 67,
    'oboe': 68,
    'english_horn': 69,
    'bassoon': 70,
    'clarinet': 71,

    # Pipe
    'piccolo': 72,
    'flute': 73,
    'recorder': 74,
    'pan_flute': 75,
    'blown_bottle': 76,
    'shakuhachi': 77,
    'whistle': 78,
    'ocarina': 79,
}
