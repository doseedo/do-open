import re
import csv
import subprocess
from pathlib import Path

# === CONFIG ===
ENCODEC_DIR = Path("/home/arlo/Data/encodec_tokens")
CHROMA_DIR = Path("/mnt/msdd/audio_features")
OUTPUT_CSV = Path("/home/arlo/Data/paired_training_data.csv")
OLLAMA_MODEL = "phi3"
UNDEFINED_LOG = Path("/home/arlo/Data/undefined_instruments.txt")

# === INSTRUMENT CATEGORY MATCHES ===
INSTRUMENT_PATTERNS = {
    "drums": ["kick", "kik", "bd", "bdin", "bdout", "snare", "sn", "snr", "snrtop", "snrbottom", "hihat", "hh", "hat", "chh", "ohh", "closedhat", "openhat", "tom", "racktom", "floortom", "rtom", "ftom", "overhead", "oh", "ohl", "ohr", "cymbal", "cym", "crash", "ride", "china", "splash", "bell", "stack"],
    "room": ["room", "rooml", "roomr", "rml", "rmr"],
    "bass": ["bass", "bss", "bassamp", "bassdi", "subbass"],
    "guitar": ["guitar", "gtr", "git", "gt", "guit", "elecgtr", "acgtr", "egtr"],
    "piano": ["piano", "key", "keys", "rhodes", "ep", "upright", "grnd", "wurlitzer", "wurl", "wurli", "nord", "pno"],
    "synth": ["synth", "moog", "juno", "prophet", "lead", "bassynth"],
    "organ": ["organ", "b3", "hammond"],
    "voice": ["vox", "vocal", "voice", "ldvox", "vox1", "vox2", "bgvox", "choir", "vo", "bgv", "double", "harm", "print", "laud", "dubs", "dbls"],
    "pad": ["pad", "ambient", "pad1", "pad2"],
    "strings": ["violin", "viola", "cello", "string", "str", "ensemble", "vln", "vla", "vc"],
    "brass": ["trumpet", "tpt", "trombone", "bone", "horn", "flugel", "tuba", "trmpt", "tromb", 'tb', 'tpt', 'fh', "tp"],
    "winds": ["sax", "tenor", "bari", "flute", "clari", "oboe", "alto"],
    "mallets": ["glock", "marimba", "xylo", "vibes", "vibraphone"],
    "plucked": ["banjo", "mandolin", "ukelele", "harp", "sitar"],
    "percussion": ["perc", "tamb", "clap", "shaker", "conga", "bongo", "cabasa", "cowbell", "triangle", "windchime", "timp", "timpani", "belltree"],
    "fx": ["fx", "sweep", "impact", "boom", "whoosh", "glitch", "echo", "reverb"],
    "click": ["click", "clk", "metronome", "tempo", "count"]
}



def normalize_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())

def tokenize(name):
    return re.split(r'[\s_\-.]+', name.lower())

def categorize_instrument(filename):
    tokens = tokenize(filename)
    for token in tokens:
        for category, patterns in INSTRUMENT_PATTERNS.items():
            if token in patterns:
                return category
    norm = normalize_name(filename)
    for category, patterns in INSTRUMENT_PATTERNS.items():
        if any(pat in norm for pat in patterns):
            return category
    return None

def classify_with_ollama(filename: str) -> str:
    system_prompt = (
        "You are classifying audio filenames into instrument categories.\n"
        "Available categories: " + ", ".join(INSTRUMENT_PATTERNS.keys()) + ".\n"
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
        "Classify this filename as accurately as possible based on instrument name and context:\n"
        f"Filename: {filename}\nCategory:"
    )
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=system_prompt.encode(),
            capture_output=True,
            timeout=10
        )
        response = result.stdout.decode().strip().lower()
        response = re.sub(r'[^a-z]', '', response)
        return response if response in INSTRUMENT_PATTERNS else "undefined"
    except Exception as e:
        return "undefined"


# === 1. Index files ===
print("📦 Indexing encodec and chroma paths...")
encodec_index = {}
chroma_index = {}

for pt_file in ENCODEC_DIR.rglob("*.pt"):
    encodec_index[pt_file.stem.lower()] = pt_file
for npy_file in CHROMA_DIR.rglob("*.chroma.npy"):
    stem = npy_file.stem.lower().replace(".chroma", "")
    chroma_index[stem] = npy_file

print(f"✅ Indexed {len(encodec_index)} encodec and {len(chroma_index)} chroma files.")

# === 2. Process and log ===
with open(OUTPUT_CSV, "w", newline="") as out_csv, open(UNDEFINED_LOG, "w") as log:
    writer = csv.DictWriter(out_csv, fieldnames=["encodec_path", "chroma_path", "instrument"])
    writer.writeheader()

    match_count = 0
    for stem, pt_path in encodec_index.items():
        if stem not in chroma_index:
            print(f"❌ No chroma match for: {pt_path.name}")
            continue

        chroma_path = chroma_index[stem]
        instrument = categorize_instrument(pt_path.name)

        if instrument is None:
            instrument = classify_with_ollama(pt_path.name)
            print(f"🧠 Ollama: {pt_path.name} → {instrument}")
            if instrument == "undefined":
                log.write(f"{pt_path.name}\n")

        writer.writerow({
            "encodec_path": str(pt_path),
            "chroma_path": str(chroma_path),
            "instrument": instrument
        })

        match_count += 1
        print(f"✅ {match_count}: {pt_path.name} matched as {instrument}")

        if match_count % 1000 == 0:
            print(f"🔢 Processed {match_count} pairs...")

print(f"🏁 Done. Total matched pairs: {match_count}")
print(f"📄 Saved to: {OUTPUT_CSV}")