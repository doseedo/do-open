#!/usr/bin/env python3
"""
MIDI Search Interface Launcher

Quick launcher for the MIDI analysis search interface.
"""

import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['gradio', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'pretty_midi']
    missing = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False

    return True

def main():
    print("🎹 MIDI Search Interface Launcher")
    print("=" * 40)

    # Check if analysis file exists
    analysis_path = "/home/arlo/Data/midi_analysis.json"
    if not Path(analysis_path).exists():
        print(f"❌ {analysis_path} not found!")
        print("Please run: python /home/arlo/Data/midianal.py")
        print()

        # Offer to run midianal.py
        response = input("Run midianal.py now? (y/n): ").strip().lower()
        if response == 'y':
            print("🔄 Running midianal.py...")
            try:
                subprocess.run([sys.executable, "/home/arlo/Data/midianal.py"], check=True, cwd="/home/arlo/Data")
                print("✅ Analysis complete!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error running midianal.py: {e}")
                return
            except FileNotFoundError:
                print("❌ /home/arlo/Data/midianal.py not found!")
                return
        else:
            return

    # Check requirements
    if not check_requirements():
        return

    # Launch interface
    print("🚀 Launching MIDI Search Interface...")
    print("📱 Interface will open at: http://localhost:7860")
    print("Press Ctrl+C to stop")
    print()

    try:
        subprocess.run([sys.executable, "/home/arlo/Data/midi_search_interface.py"])
    except KeyboardInterrupt:
        print("\n👋 Interface stopped")
    except FileNotFoundError:
        print("❌ /home/arlo/Data/midi_search_interface.py not found!")

if __name__ == "__main__":
    main()