#!/usr/bin/env python3
"""
Launch script for MIDI Note Search Interface
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🎵 MIDI Note Search Interface Launcher")
    print("=" * 50)

    # Check if analysis file exists
    analysis_path = "/home/arlo/Data/midi_analysis/chord_summary.csv"
    if not Path(analysis_path).exists():
        print(f"❌ {analysis_path} not found!")
        print("Please run: python /home/arlo/Data/midianal.py")
        return

    print("✅ MIDI analysis data found")
    print("🚀 Launching Note Search Interface...")
    print("🎵 This searches for actual notes in MIDI files (not chord labels)")
    print("📱 Interface will open at: http://localhost:7863")
    print("Press Ctrl+C to stop")
    print()

    try:
        subprocess.run([sys.executable, "/home/arlo/Data/note_search_interface.py"])
    except KeyboardInterrupt:
        print("\n👋 Interface stopped")
    except FileNotFoundError:
        print("❌ /home/arlo/Data/note_search_interface.py not found!")

if __name__ == "__main__":
    main()