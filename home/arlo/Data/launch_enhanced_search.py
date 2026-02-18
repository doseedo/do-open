#!/usr/bin/env python3
"""
Launch Enhanced MIDI Note Search Interface
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🎵 Enhanced MIDI Note Search Interface Launcher")
    print("=" * 60)

    # Check requirements
    analysis_path = "/home/arlo/Data/midi_analysis/chord_summary.csv"
    manifest_path = "/home/arlo/Data/final_training_manifest_final.json"

    if not Path(analysis_path).exists():
        print(f"❌ {analysis_path} not found!")
        print("Please run: python /home/arlo/Data/midianal.py")
        return

    if not Path(manifest_path).exists():
        print(f"⚠️ {manifest_path} not found!")
        print("Original audio preview will not be available")

    print("✅ MIDI analysis data found")
    print("🚀 Launching Enhanced Note Search Interface...")
    print()
    print("🔊 Features:")
    print("  • Original audio preview from training manifest")
    print("  • MIDI renders of detected note sections")
    print("  • Instrument grouping and sorting")
    print("  • Precise timing information")
    print()
    print("📱 Interface will open at: http://localhost:7864")
    print("Press Ctrl+C to stop")
    print()

    try:
        subprocess.run([sys.executable, "/home/arlo/Data/enhanced_note_search_interface.py"])
    except KeyboardInterrupt:
        print("\n👋 Interface stopped")
    except FileNotFoundError:
        print("❌ enhanced_note_search_interface.py not found!")

if __name__ == "__main__":
    main()