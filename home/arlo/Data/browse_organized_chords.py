#!/usr/bin/env python3
"""
Simple browser for organized chord results
"""

import os
from pathlib import Path

def browse_chord_results(chord_name):
    """Browse organized results for a specific chord"""
    base_dir = Path("/home/arlo/Data/chord_organized")
    chord_dir = base_dir / chord_name.replace(' ', '_').replace('#', 'sharp').replace('b', 'flat')

    if not chord_dir.exists():
        print(f"❌ No organized results found for chord '{chord_name}'")
        print(f"Expected directory: {chord_dir}")
        return

    print(f"🎵 Browsing results for chord: {chord_name}")
    print(f"📁 Directory: {chord_dir}")
    print("=" * 60)

    # Read index file if it exists
    index_file = chord_dir / "INDEX.txt"
    if index_file.exists():
        print("\n📄 INDEX FILE:")
        print("-" * 40)
        with open(index_file, 'r') as f:
            print(f.read())

    # List all session directories
    session_dirs = [d for d in chord_dir.iterdir() if d.is_dir() and d.name.startswith('session_')]

    if session_dirs:
        print(f"\n📂 SESSIONS FOUND: {len(session_dirs)}")
        print("-" * 40)

        for session_dir in sorted(session_dirs):
            session_name = session_dir.name.replace('session_', '')
            print(f"\n🎤 Session: {session_name}")

            # List MIDI files
            midi_files = list(session_dir.glob("*.mid"))
            summary_files = list(session_dir.glob("*_summary.txt"))

            print(f"   📄 Files: {len(midi_files)} MIDI, {len(summary_files)} summaries")

            for midi_file in sorted(midi_files):
                summary_file = session_dir / f"{midi_file.stem}_summary.txt"
                info_file = session_dir / f"{midi_file.stem}_info.json"

                print(f"   🎹 {midi_file.name}")
                print(f"      📊 Summary: {summary_file.name if summary_file.exists() else 'Not found'}")
                print(f"      📋 Info: {info_file.name if info_file.exists() else 'Not found'}")

                # Show quick summary info
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if 'Duration:' in line or 'Chord occurrences:' in line:
                                    print(f"      {line.strip()}")
                    except Exception as e:
                        print(f"      ⚠️ Error reading summary: {e}")

    print(f"\n💡 TIP: To listen to files:")
    print(f"   cd {chord_dir}")
    print(f"   fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 session_*/filename.mid")
    print(f"\n💡 To see detailed chord analysis:")
    print(f"   cat session_*/*_summary.txt")

def list_all_organized_chords():
    """List all available organized chord results"""
    base_dir = Path("/home/arlo/Data/chord_organized")

    if not base_dir.exists():
        print(f"❌ No organized chord results directory found at {base_dir}")
        return

    chord_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    if not chord_dirs:
        print(f"❌ No organized chord results found in {base_dir}")
        return

    print(f"🎵 Available organized chord results:")
    print("=" * 50)

    for chord_dir in sorted(chord_dirs):
        chord_name = chord_dir.name
        index_file = chord_dir / "INDEX.txt"

        # Count files
        session_count = len([d for d in chord_dir.iterdir() if d.is_dir() and d.name.startswith('session_')])

        print(f"📁 {chord_name}")
        print(f"   Sessions: {session_count}")

        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    for line in f:
                        if 'Total Files:' in line:
                            print(f"   {line.strip()}")
                            break
            except:
                pass
        print()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python browse_organized_chords.py <chord_name>  - Browse specific chord")
        print("  python browse_organized_chords.py --list       - List all organized chords")
        print()
        print("Examples:")
        print("  python browse_organized_chords.py major")
        print("  python browse_organized_chords.py Cmajor")
        print("  python browse_organized_chords.py --list")
        sys.exit(1)

    if sys.argv[1] == "--list":
        list_all_organized_chords()
    else:
        browse_chord_results(sys.argv[1])