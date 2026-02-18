#!/usr/bin/env python3
"""
Filename Group Verification & Classification

Contains the canonical INSTRUMENT_PATTERNS and classify_filename() used by:
  - create_master_manifest.py (to validate/correct unified manifest labels)
  - This script (to verify consolidated manifest entries)

Writes filename_group / filename_verified fields to consolidated manifest.
"""

import logging
import re
import os
from collections import Counter
from pathlib import Path

import orjson

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)

MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/consolidated_manifest.json")

# ===================== INSTRUMENT PATTERNS =====================
#
# Rules:
# - Tokens are matched after splitting on [ _\-\.]+
# - Multi-word patterns (with spaces) are checked as substrings first
# - Order within each group doesn't matter (groups are checked per-token)
# - PRIORITY_PATTERNS are checked before INSTRUMENT_PATTERNS to resolve ambiguity
# - EXCLUSIONS prevent false matches (e.g. "bassoon" should NOT match "bass")

# Multi-token patterns checked FIRST (highest priority, most specific)
# Note: Use (?<![a-z]) instead of \b for start-of-pattern because \b treats
# underscores as word chars, failing on "Cosmo_SopranoSax" or CamelCase.
_B = r'(?<![a-z])'  # "boundary" that works across _ and CamelCase

PRIORITY_PATTERNS = [
    # Winds - must beat voice for "tenor/alto/bari" when followed by sax
    ("winds", re.compile(_B + r'tenor[\s_\-\.]*sax', re.I)),
    ("winds", re.compile(_B + r'alto[\s_\-\.]*sax', re.I)),
    ("winds", re.compile(_B + r'bari(?:tone)?[\s_\-\.]*sax', re.I)),
    ("winds", re.compile(_B + r'soprano[\s_\-\.]*sax', re.I)),
    # Winds - bassoon must beat bass
    ("winds", re.compile(_B + r'bassoon', re.I)),
    # Guitar harmonics should not be voice
    ("guitar", re.compile(_B + r'gtr[\s_\-\.]*(?:hole[\s_\-\.]*)?harmonics?', re.I)),
    ("guitar", re.compile(_B + r'guitar[\s_\-\.]*harmonics?', re.I)),
    # Percussion - tambourine must not become room
    ("percussion", re.compile(_B + r'tamb(?:ourine)?(?![a-z])', re.I)),
    # Voice - "lead vocal" must beat synth "lead"
    ("voice", re.compile(_B + r'lead[\s_\-\.]*vox(?:al)?(?![a-z])', re.I)),
    ("voice", re.compile(_B + r'lead[\s_\-\.]*vocal(?![a-z])', re.I)),
    # Drums - "kick in/out" multi-word
    ("drums", re.compile(_B + r'kick[\s_\-\.]*(?:in|out)(?![a-z])', re.I)),
    ("drums", re.compile(_B + r'snare[\s_\-\.]*(?:top|bot|bottom)(?![a-z])', re.I)),
    # Piano - "board mix" is a mix, not a keyboard
    ("mix", re.compile(_B + r'board[\s_\-\.]*mix(?![a-z])', re.I)),
    # Room - explicit room mic patterns
    ("room", re.compile(_B + r'room[\s_\-\.]*mic(?![a-z])', re.I)),
]

INSTRUMENT_PATTERNS = {
    "drums": [
        "kick", "kik", "bd", "bdin", "bdout", "snare", "sn", "snr", "snrtop",
        "snrbottom", "hihat", "hh", "hat", "chh", "ohh", "closedhat", "openhat",
        "tom", "racktom", "floortom", "rtom", "ftom", "overhead", "oh", "ohl",
        "ohr", "cymbal", "cym", "crash", "ride", "china", "splash",
        "stack", "floor", "kkin", "kkout",
        "rack", "drums", "drum", "drumkit", "kit", "rimshot", "djembe",
        "ovh", "ovhl", "ovhr",
    ],
    "bass": [
        "bass", "bss", "bassamp", "bassdi", "subbass", "bsamp",
        "upright_bass", "electric_bass", "dbass", "ebass",
    ],
    "guitar": [
        "guitar", "gtr", "git", "guit", "elecgtr", "acgtr", "egtr",
        "acoustic_guitar", "electric_guitar", "strat", "tele", "lespaul",
        "reamp",
    ],
    "piano": [
        "piano", "keys", "rhodes", "wurlitzer", "wurl", "wurli", "nord", "pno",
        "steinway", "keyboard", "epiano", "electric_piano",
    ],
    "synth": [
        "synth", "synthesizer", "moog", "juno", "prophet",
        "lead_synth", "analog_synth",
    ],
    "organ": ["organ", "b3", "hammond", "leslie", "b3organ"],
    "voice": [
        "vox", "vocal", "vocals", "voice", "ldvox", "bgvox", "choir", "bgv",
        "dubs", "dbls", "adlibs", "adlib",
        "lead_vocal", "background_vocal", "soprano", "baritone",
        "double_vocal",
    ],
    "strings": [
        "violin", "viola", "cello", "string", "strings", "vln", "vla",
        "fiddle", "contrabass", "string_ensemble", "string_section",
    ],
    "brass": [
        "trumpet", "tpt", "trombone", "bone", "horn", "flugel", "tuba",
        "trmpt", "tromb", "hrn", "brass", "brass_section", "horn_section",
        "flugelhorn", "cornet", "french_horn",
    ],
    "winds": [
        "sax", "saxophone", "tenor_sax", "alto_sax", "bari_sax", "soprano_sax",
        "flute", "clarinet", "clari", "oboe", "bassoon", "piccolo",
        "woodwind", "recorder", "panflute", "shakuhachi",
    ],
    "mallets": [
        "glock", "glockenspiel", "marimba", "xylo", "xylophone", "vibes",
        "vibraphone", "celesta", "tubular_bells",
    ],
    "plucked": ["banjo", "mandolin", "ukelele", "ukulele", "harp", "sitar",
                "shamisen"],
    "percussion": [
        "perc", "tambourine", "clap", "shaker", "conga", "bongo",
        "cabasa", "cowbell", "triangle", "windchime", "timp", "timpani",
        "belltree", "cajon", "guasa", "guasas",
    ],
    "fx": ["fx", "sfx", "sweep", "impact", "boom", "whoosh", "glitch",
           "riser", "reverb"],
    "click": ["click", "clk", "metronome", "2pop", "click_track"],
    "room": ["room", "rooml", "roomr", "room_mic", "ambience"],
}

# Short tokens: only match if nothing else matched (too ambiguous alone)
AMBIGUOUS_SHORT = {
    "oh", "ep", "gt", "vo", "st", "rt", "sb", "rm", "tp", "tb",
    "fh", "vc", "str", "key", "lead", "pad", "bell",
}

# Exclusion rules: if token matches an exclusion, skip that group.
# Format: group -> set of tokens that should NOT match that group.
EXCLUSIONS = {
    "bass": {"bassoon", "bassoons"},        # bassoon → winds not bass
    "voice": {"tenor_sax", "alto_sax"},     # sax parts → winds not voice
    "drums": {"drumstick"},                 # edge case
    "room": {"harmony", "harmonies", "harm", "harmonic", "harmonics"},  # vocal harmonies
}

# Groups to merge into canonical groups
GROUP_MERGES = {
    "review_vocals": "voice",
    "vocals": "voice",
    "e-drums": "drums",
    "full-track": "mix",
    "noise_hiss": "junk",
    "violin": "strings",  # standalone "violin" group → strings
}


# Token-level disqualifiers: if the filename contains these tokens,
# the group assignment is suspect and should be set to 'undefined'
# Format: group -> set of tokens/substrings that disqualify that group
DISQUALIFIERS = {
    "room": {"harm", "harmony", "harmonies", "harmonic", "harmonics"},
    "click": {"countryman", "country"},  # Countryman is a mic brand
    "voice": {"sax", "saxophone"},  # sax should be winds
}


def tokenize(name):
    """Split filename into tokens on separators."""
    return re.split(r'[\s_\-\.]+', name.lower())


def classify_filename(audio_path):
    """Classify an audio file's group from its filename.

    Returns (group, matched_keyword) or ('undefined', None).
    """
    fname = os.path.basename(audio_path)
    fname_lower = fname.lower()
    fname_stem = os.path.splitext(fname_lower)[0]

    # 0. Priority patterns (multi-token regex, highest specificity)
    for group, regex in PRIORITY_PATTERNS:
        if regex.search(fname_stem):
            return group, regex.pattern

    # 1. Check multi-word patterns (with spaces) against full filename
    for group, patterns in INSTRUMENT_PATTERNS.items():
        for pat in patterns:
            if ' ' in pat and pat in fname_lower:
                return group, pat

    # 2. Token-based matching on filename
    tokens = tokenize(fname_stem)

    for token in tokens:
        if not token or len(token) < 2:
            continue

        for group, patterns in INSTRUMENT_PATTERNS.items():
            if token in patterns:
                # Skip ambiguous short tokens
                if token in AMBIGUOUS_SHORT:
                    continue
                # Check exclusions
                excl = EXCLUSIONS.get(group, set())
                if token in excl:
                    continue
                # Check if any other token triggers an exclusion for this group
                if excl and any(t in excl for t in tokens):
                    continue
                return group, token

    # 3. Check ambiguous short tokens if nothing else matched
    for token in tokens:
        if token in AMBIGUOUS_SHORT:
            for group, patterns in INSTRUMENT_PATTERNS.items():
                if token in patterns:
                    return group, token

    # 4. Substring match for longer patterns (>=6 chars) to catch compound names
    for group, patterns in INSTRUMENT_PATTERNS.items():
        for pat in patterns:
            if len(pat) >= 6 and pat in fname_lower:
                # Check exclusions
                excl = EXCLUSIONS.get(group, set())
                if pat in excl:
                    continue
                return group, pat

    # 5. Check parent directory for group hints (only strong patterns)
    parent = os.path.basename(os.path.dirname(audio_path)).lower()
    parent_tokens = tokenize(parent)
    for token in parent_tokens:
        if not token or len(token) < 4:
            continue
        for group, patterns in INSTRUMENT_PATTERNS.items():
            if token in patterns and token not in AMBIGUOUS_SHORT:
                return group, f"dir:{token}"

    return "undefined", None


def disqualify_group(audio_path, group):
    """Check if a group assignment should be disqualified based on filename.

    Returns True if the filename contains substrings that contradict the given group.
    Used when classify_filename returns 'undefined' but the existing label is suspect.
    Checks both exact token matches AND substring matches for compound tokens.
    """
    if group not in DISQUALIFIERS:
        return False
    fname = os.path.basename(audio_path).lower()
    fname_stem = os.path.splitext(fname)[0]
    disq = DISQUALIFIERS[group]
    # Check substring matches (handles compound tokens like "ReeseHarm1")
    return any(d in fname_stem for d in disq)


def main():
    logging.info(f"Loading {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    entries = manifest['entries']
    logging.info(f"Processing {len(entries)} entries...")

    stats = Counter()
    verify_by_group = Counter()
    total_by_group = Counter()
    mismatch_examples = {}

    for entry in entries:
        audio_path = entry.get('audio_path', '')
        assigned_group = entry.get('group', 'undefined')

        fn_group, matched_kw = classify_filename(audio_path)

        if fn_group == 'undefined':
            verified = False
            entry['filename_verified'] = False
            entry['filename_group'] = None
            stats['no_filename_match'] += 1
        elif fn_group == assigned_group:
            verified = True
            entry['filename_verified'] = True
            entry['filename_group'] = fn_group
            stats['verified'] += 1
        else:
            verified = False
            entry['filename_verified'] = False
            entry['filename_group'] = fn_group
            stats['mismatch'] += 1

            if assigned_group not in mismatch_examples:
                mismatch_examples[assigned_group] = []
            if len(mismatch_examples[assigned_group]) < 5:
                mismatch_examples[assigned_group].append({
                    'file': os.path.basename(audio_path),
                    'assigned': assigned_group,
                    'filename_says': fn_group,
                    'keyword': matched_kw,
                })

        total_by_group[assigned_group] += 1
        if verified:
            verify_by_group[assigned_group] += 1

        stats['total'] += 1

    # Print summary
    logging.info(f"\n{'='*60}")
    logging.info("FILENAME VERIFICATION SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Total entries:      {stats['total']:,}")
    logging.info(f"Filename verified:  {stats['verified']:,} ({stats['verified']/stats['total']:.1%})")
    logging.info(f"No filename match:  {stats['no_filename_match']:,} ({stats['no_filename_match']/stats['total']:.1%})")
    logging.info(f"Mismatch:           {stats['mismatch']:,} ({stats['mismatch']/stats['total']:.1%})")

    logging.info(f"\nVerification rate by group:")
    for group in sorted(total_by_group.keys()):
        total = total_by_group[group]
        verified = verify_by_group.get(group, 0)
        unverified = total - verified
        pct = verified / total * 100 if total > 0 else 0
        logging.info(f"  {group:15s}: {verified:>7,} / {total:>7,} verified ({pct:5.1f}%) | {unverified:>7,} unverified")

    if mismatch_examples:
        logging.info(f"\nMismatch examples:")
        for group, examples in sorted(mismatch_examples.items()):
            for ex in examples[:3]:
                logging.info(f"  [{ex['assigned']}] {ex['file']} -> filename says '{ex['filename_says']}' (keyword: {ex['keyword']})")

    manifest['filename_verification'] = {
        'total': stats['total'],
        'verified': stats['verified'],
        'no_match': stats['no_filename_match'],
        'mismatch': stats['mismatch'],
        'rate_by_group': {
            g: {
                'verified': verify_by_group.get(g, 0),
                'total': total_by_group[g],
                'rate': round(verify_by_group.get(g, 0) / total_by_group[g], 4) if total_by_group[g] > 0 else 0,
            }
            for g in sorted(total_by_group.keys())
        },
    }

    logging.info(f"\nWriting updated manifest to {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, 'wb') as f:
        f.write(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    logging.info(f"Done! {MANIFEST_PATH.stat().st_size / 1e6:.1f} MB written")


if __name__ == '__main__':
    main()
