#!/usr/bin/env python3
"""
Launch script for MIDI Audio Search Interface
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🎹 MIDI Audio Search Interface Launcher")
    print("=" * 50)

    # Check if analysis file exists
    analysis_path = "/home/arlo/Data/midi_analysis/chord_summary.csv"
    if not Path(analysis_path).exists():
        print(f"❌ {analysis_path} not found!")
        print("Please run: python /home/arlo/Data/midianal.py")
        return

    # Check required packages
    required_packages = ['gradio', 'pandas', 'pretty_midi', 'soundfile']
    missing = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return

    print("✅ All dependencies found")
    print("🚀 Launching MIDI Audio Search Interface...")
    print("🔊 This interface creates audio snippets for chord searches")
    print("📱 Interface will open at: http://localhost:7862")
    print("Press Ctrl+C to stop")
    print()

    try:
        subprocess.run([sys.executable, "/home/arlo/Data/midi_audio_search_interface.py"])
    except KeyboardInterrupt:
        print("\n👋 Interface stopped")
    except FileNotFoundError:
        print("❌ /home/arlo/Data/midi_audio_search_interface.py not found!")

if __name__ == "__main__":
    main()