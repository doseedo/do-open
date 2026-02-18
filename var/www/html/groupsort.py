from pathlib import Path
import re
import json
import subprocess
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import shutil
import os

# === CONFIG ===
AUDIO_PATH_LIST = Path("/home/arlo/Data/all_audio_paths6.txt")
UNDEFINED_LOG = Path("/home/arlo/Data/undefined_instruments.txt")
CATEGORIZED_LISTS_DIR = Path("/home/arlo/Data/categorized_inst_more") # Directory to save path lists
OLLAMA_MODEL = "phi3"

# === INSTRUMENT CATEGORY MATCHES (Expanded) ===
INSTRUMENT_PATTERNS = {
    "drums": ["kick", "kik", "bd", "bdin", "bdout", "snare", "sn", "snr", "snrtop", "snrbottom", "hihat", "hh", "hat", "chh", "ohh", "closedhat", "openhat", "tom", "racktom", "floortom", "rtom", "ftom", "overhead", "oh", "ohl", "ohr", "cymbal", "cym", "cymbal", "crash", "ride", "china", "splash", "bell", "stack", "k in", "k out", "floor", "kkin", "kk in", "kkout", "kk out", "rack", "drums", "drum", "rt", "sb", "st", "rt", "ko_", "ki_", "flt", "ovh", "ovhl", "ovhr", "kit", "rimshot", "djembe"],
    "room": ["room", "rooml", "roomr", "rml", "rmr", "rm"],
    "bass": ["bass", "bss", "bassamp", "bassdi", "subbass", "bsamp", "bs amp"],
    "guitar": ["guitar", "gtr", "git", "gt", "guit", "elecgtr", "acgtr", "egtr"],
    "piano": ["piano", "key", "keys", "rhodes", "ep", "upright", "grnd", "wurlitzer", "wurl", "wurli", "nord", "pno"],
    "synth": ["synth", "moog", "juno", "prophet", "lead", "bassynth"],
    "organ": ["organ", "b3", "hammond"],
    "voice": ["vox", "vocal", "voice", "ldvox", "vox1", "vox2", "bgvox", "choir", "vo", "bgv", "double", "harm", "print", "laud", "dubs", "dbls", "adlibs", "libs"],
    "pad": ["pad", "ambient", "pad1", "pad2"],
    "strings": ["violin", "viola", "cello", "string", "str", "ensemble", "vln", "vla", "vc", "vlin"],
    "brass": ["trumpet", "tpt", "trombone", "bone", "horn", "flugel", "tuba", "trmpt", "tromb", 'TB', 'TPT', 'FH', "TP", "hrn", "tp", "tb", "fh", "tpt", "brass"],
    "winds": ["sax", "tenor", "bari", "flute", "clari", "oboe", "alto"],
    "mallets": ["glock", "marimba", "xylo", "vibes", "vibraphone"],
    "plucked": ["banjo", "mandolin", "ukelele", "harp", "sitar"],
    "percussion": ["perc", "tamb", "clap", "shaker", "conga", "bongo", "cabasa", "cowbell", "triangle", "windchime", "timp", "timpani", "belltree"],
    "fx": ["fx", "sfx", "sweep", "impact", "boom", "whoosh", "glitch", "echo", "reverb"],
    "click": ["click", "clk", "metronome", "tempo", "count"]
}

def normalize_name(name):
    """Normalizes a filename for pattern matching."""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def tokenize(name):
    """Splits a filename into tokens for keyword matching."""
    return re.split(r'[\s_\-.]+', name.lower())

def categorize_instrument(filename):
    """Categorizes a filename based on predefined patterns."""

    lower_filename = filename.lower()
    for category, patterns in INSTRUMENT_PATTERNS.items():
        for pat in patterns:
            if ' ' in pat and pat in lower_filename: # Check if pattern has a space and is in the filename
                return category


    tokens = tokenize(filename)
    for token in tokens:
        for category, patterns in INSTRUMENT_PATTERNS.items():
            if token in patterns:
                return category
    norm = normalize_name(filename)
    for category, patterns in INSTRUMENT_PATTERNS.items():
        if any(pat in norm for pat in patterns):
            return category
    return "undefined"

def classify_with_ollama(filenames):
    """Uses Ollama to classify a batch of filenames."""
    prompt = (
        "You are classifying audio filenames into instrument categories.\n"
        "Available categories: "
        + ", ".join(INSTRUMENT_PATTERNS.keys()) +
        ".\nClassify each filename as accurately as possible using filename cues, abbreviations, and context.\n\n"
        "Examples:\n"
        "- 'print.07_08.wav' → voice\n"
        "- 'laud double_02.L.wav' → voice\n"
        "- 'la harm 1_02.wav' → voice\n"
        "- 'outro whistle_01.wav' → percussion\n"
        "- 'nick U87.07-FZ.wav' → voice\n"
        "- 'chiara bgv 7_1.wav' → voice\n"
        "- 'VO.06_55.wav' → voice\n"
        "- 'WURLI.1.wav' → piano\n"
        "- 'nord_01.L.wav' → piano\n"
        "- 'e VLN Amp_05.wav' → strings\n"
        "- 'Vln Close.16_12.wav' → strings\n"
        "- 'VintageDrums.wav' → drums\n"
        "- 'alto 4_11.wav' → winds\n"
        "- 'Windchimes_1.wav' → percussion\n"
        "- 'Timpani 3.03_08.wav' → percussion\n"
        "- 'Tuba.03_22.wav' → brass\n"
        "- 'Tenor.07_93.wav' → winds\n"
        "- 'TRMPT_Storm Chasing.06_02.wav' → brass\n"
        "- 'FH_02.wav' → brass\n"
        "- 'Fiddle' → strings\n"
        "- 'FL' → winds\n"
        "- 'PNO' → piano\n"
        "- 'K_In.wav' → drums\n"
        "- 'SN_top.wav' → drums\n"
        "- 'TROMB.04_07.wav' → brass\n"
        "- 'Vc Vla.STM4.05_05.wav' → strings\n"
        "- 'Solo_V_05_05.wav' → strings\n"
        "- 'BARI 1_1.wav' → winds\n"
        "- 'CLAR 1_1.wav' → winds\n"
        "- 'ob1.wav' → winds\n"
        "- 'obo1.wav' → winds\n"
        "- 'Banjo.21_23.wav' → plucked\n"
        "- 'Chorus.1.wav' → voice\n"
        "- 'Nancy-Soprano.1.wav' → voice\n"
        "- 'Flute.1.wav' → winds\n"    
        "- 'Main-verse.1.wav' → voice\n"
        "- 'glock3.wav' → mallets\n"
        "- 'xylo.wav' → mallets\n"
        "- 'mandolin.wav' → plucked\n"
        "- 'uke.wav' → plucked\n"
        "- 'uke.wav' → plucked\n"
        "- 'harp.wav' → plucked\n"
        "- 'Electric1.wav' → guitar\n"
        "- 'Riser1.wav' → fx\n"
        "- 'Echo.wav' → fx\n\n"

        "Now classify:\n"
    )
    prompt += "\n".join(filenames)
    result = subprocess.run(["ollama", "run", OLLAMA_MODEL],
                            input=prompt.encode(),
                            capture_output=True)
    response = result.stdout.decode().strip()
    lines = [line.strip().split("→")[-1].strip().lower() for line in response.splitlines() if "→" in line]
    return dict(zip(filenames, lines))

def analyze_audio_paths(audio_list_path):
    """Reads audio paths, categorizes them, and returns analysis data."""
    inst_counts = Counter()
    session_counts = Counter()
    undefined_files = []

    with open(audio_list_path, 'r') as f:
        paths = [line.strip() for line in f if "/Prev/" not in line]

    categorized = {}
    print(f"🔍 Starting categorization for {len(paths)} files...")
    for path in paths:
        filename = Path(path).name
        category = categorize_instrument(filename)
        if category == "undefined":
            undefined_files.append((filename, path))
        else:
            categorized[path] = category

        match = re.search(r'protools[AB]?/(\d{4}-\d{2}-\d{2})/', path)
        if match:
            session_counts[match.group(1)] += 1

    if undefined_files:
        print(f"🤖 Found {len(undefined_files)} undefined files. Sending to Ollama for classification...")
        batch = list(set(f[0] for f in undefined_files))
        classified = classify_with_ollama(batch)
        for fname, full_path in undefined_files:
            inferred = classified.get(fname, "undefined")
            categorized[full_path] = inferred

    for _, cat in categorized.items():
        inst_counts[cat] += 1

    final_undefined = [f[0] for f in undefined_files if categorized.get(f[1]) == "undefined"]
    
    print("✅ Analysis complete.")
    return inst_counts, session_counts, final_undefined, categorized

def save_categorized_paths(categorized_data, output_dir):
    """Saves lists of file paths sorted by their category."""
    print(f"💾 Saving categorized path lists to {output_dir}...")
    
    # Create a clean directory for the output lists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Group paths by their category
    paths_by_category = defaultdict(list)
    for path, category in categorized_data.items():
        paths_by_category[category].append(path)

    # Write each category's paths to a separate file
    for category, paths in paths_by_category.items():
        output_file = output_dir / f"{category}.txt"
        with open(output_file, 'w') as f:
            # Sort the paths for consistent output
            f.write('\n'.join(sorted(paths)))
    
    print("✅ Path lists saved.")

# === RUN ===
instrument_data, session_data, final_undefined, all_categorized_data = analyze_audio_paths(AUDIO_PATH_LIST)

# Save the categorized path lists to the specified directory
save_categorized_paths(all_categorized_data, CATEGORIZED_LISTS_DIR)

# Log undefined
print(f"📝 Logging {len(final_undefined)} finally undefined files to {UNDEFINED_LOG}")
with open(UNDEFINED_LOG, "w") as f:
    for name in sorted(set(final_undefined)):
        f.write(name + "\n")

# Plot 1: Sessions by Date
print("📊 Generating plot: Sessions by Date")
plt.figure(figsize=(12, 6))
sorted_dates = sorted(session_data.items())
x, y = zip(*sorted_dates)
plt.plot(x, y, marker='o')
plt.xticks(rotation=45, ha='right')
plt.title("Sessions per Day")
plt.xlabel("Date")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig("/tmp/sessions_by_date.png")
plt.close()

# Plot 2: Audio Files by Instrument
print("📊 Generating plot: Audio Files by Instrument")
plt.figure(figsize=(12, 6))
labels, values = zip(*instrument_data.most_common())
plt.bar(labels, values)
plt.xticks(rotation=45, ha='right')
plt.title("Audio Files by Instrument Type")
plt.xlabel("Instrument")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("/tmp/audio_by_instrument.png")
plt.close()

# Copy output
print("🚚 Copying output files...")
shutil.copy("/tmp/sessions_by_date.png", "/home/arlo/Data/sessions_by_date.png")
shutil.copy("/tmp/audio_by_instrument.png", "/home/arlo/Data/audio_by_instrument.png")

print("🎉 All tasks finished successfully!")
