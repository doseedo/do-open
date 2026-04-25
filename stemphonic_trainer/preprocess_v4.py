"""
Preprocess v4 — Build text embeddings using real Pro Tools session metadata.

Each stem gets a prompt like:
  "electric bass, fingerstyle bass. Track: Bass DI"
  with metas: bpm=90, timesignature=4/4, duration=32

Cached by (group, subgroup, bpm_bucket, time_sig) key.
"""

from typing import List, Optional, Dict

# Subgroup → human-readable description
SUBGROUP_DESCRIPTIONS = {
    # Bass
    "electric_bass": "electric bass guitar",
    "upright_bass": "upright acoustic bass, double bass",
    # Guitar
    "electric_guitar": "electric guitar",
    "acoustic_guitar": "acoustic guitar, steel string",
    # Piano/Keys
    "acoustic_piano": "acoustic piano, grand piano",
    "electric_piano": "electric piano, Rhodes, Wurlitzer",
    "keys": "keyboards, synthesizer keys",
    # Strings
    "violin": "violin",
    "viola": "viola",
    "cello": "cello",
    # Brass
    "trumpet": "trumpet",
    "trombone": "trombone",
    "french_horn": "french horn",
    # Winds
    "sax": "saxophone",
    "flute": "flute",
    "clarinet": "clarinet",
    # Voice
    "voice": "vocals, singing voice",
    # Drums
    "drums": "drum kit",
    # Percussion
    "percussion": "percussion, hand drums",
    # Other
    "organ": "organ, Hammond organ",
    "synth": "synthesizer",
}

GROUP_DESCRIPTIONS = {
    "drums": "drums, drum kit, percussion rhythm section",
    "bass": "bass guitar",
    "guitar": "guitar",
    "piano": "piano, keyboard",
    "keys": "keyboards, synthesizer",
    "organ": "organ",
    "strings": "strings, string section",
    "brass": "brass",
    "winds": "woodwinds",
    "synth": "synthesizer, electronic",
    "percussion": "percussion",
    "voice": "vocals, singing",
    "mallets": "mallets, vibraphone, marimba",
    "plucked": "plucked strings, banjo, mandolin",
    "ensemble": "ensemble, orchestra",
    "undefined": "musical instrument",
    "fx": "sound effects, production elements",
}

SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

DEFAULT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"


def build_caption(group: str, subgroup: str = "", track_name: str = "") -> str:
    """Build a caption from metadata. Uses subgroup for detail."""
    # Use subgroup description if available and valid
    if subgroup and subgroup != "undefined" and subgroup in SUBGROUP_DESCRIPTIONS:
        desc = SUBGROUP_DESCRIPTIONS[subgroup]
    elif subgroup and subgroup != "undefined" and subgroup != group:
        desc = f"{group}, {subgroup}"
    else:
        desc = GROUP_DESCRIPTIONS.get(group, group)

    # Add track name hint if available (cleans up Pro Tools naming)
    if track_name:
        import re
        # Clean track name: remove take numbers, channel suffixes
        clean = re.sub(r'\.\d+$', '', track_name)
        clean = re.sub(r'_\d+$', '', clean)
        clean = clean.strip()
        if clean and clean.lower() not in desc.lower():
            desc = f"{desc}. Track: {clean}"

    return desc


def build_metas(bpm: str = "", time_sig: str = "", duration_sec: float = 32.0) -> str:
    """Build metas string matching ACE-Step SFT format."""
    bpm_str = bpm if bpm else "N/A"
    ts_str = time_sig if time_sig else "N/A"
    return (f"- bpm: {bpm_str}\n"
            f"- timesignature: {ts_str}\n"
            f"- keyscale: N/A\n"
            f"- duration: {int(duration_sec)} seconds")


def format_sft_prompt(caption: str, bpm: str = "", time_sig: str = "",
                      duration_sec: float = 32.0) -> str:
    """Format into ACE-Step's SFT training template with real metadata."""
    metas = build_metas(bpm, time_sig, duration_sec)
    return SFT_GEN_PROMPT.format(DEFAULT_INSTRUCTION, caption, metas)


def format_lyrics(lyrics: str = "[Instrumental]", language: str = "en") -> str:
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


def bpm_bucket(bpm_str: str) -> str:
    """Bucket BPM into ranges."""
    if not bpm_str:
        return ""
    try:
        b = int(float(bpm_str))
        return f"{(b // 20) * 20}-{(b // 20) * 20 + 19}"
    except ValueError:
        return ""


def make_cache_key(group: str, subgroup: str = "", bpm: str = "",
                   time_sig: str = "") -> str:
    """Build a cache key for text embedding lookup."""
    bucket = bpm_bucket(bpm)
    sg = subgroup if subgroup and subgroup != "undefined" else ""
    return f"{group}|{sg}|{bucket}|{time_sig}"
