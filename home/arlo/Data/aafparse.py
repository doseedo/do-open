import os
from pathlib import Path
from aaf2 import open as aaf_open

INPUT_DIR = Path("/home/arlo/Data/sessionmetadata/Test")

def extract_tempo_map(aaf_path):
    print(f"🎼 Parsing tempo from: {aaf_path.name}")
    try:
        with aaf_open(str(aaf_path)) as f:
            for mob in f.content.mobs:
                if mob.mob_type != "CompositionMob":
                    continue

                for slot in mob.slots:
                    if "tempo" in slot.name.lower():
                        print(f"🎚️ Tempo Slot: {slot.name}")
                        segment = slot.segment
                        if hasattr(segment, "components"):
                            for comp in segment.components:
                                if hasattr(comp, "attributes"):
                                    for k, v in comp.attributes.items():
                                        print(f"  🔸 {k}: {v}")
                        else:
                            print(f"  ⚠️ Segment has no components")

        print("✅ Done parsing (if any tempo info existed)")

    except Exception as e:
        print(f"❌ Failed: {e}")

def run():
    aaf_files = list(INPUT_DIR.glob("*.aaf")) + list(INPUT_DIR.glob("*/*.aaf"))
    if not aaf_files:
        print("❌ No AAF files found.")
        return
    print(f"📦 Found {len(aaf_files)} AAF file(s)")
    for aaf in aaf_files:
        extract_tempo_map(aaf)

if __name__ == "__main__":
    run()
