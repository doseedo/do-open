#!/usr/bin/env python3
"""
Windows VST Bridge for Linux
This script helps identify and solve Windows VST compatibility issues.
"""

import os
import sys
import subprocess

def check_wine_version():
    """Check installed Wine version."""
    try:
        result = subprocess.run(['wine', '--version'], capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"✓ Wine installed: {version}")

        # Extract major version
        if 'wine-' in version:
            major_ver = int(version.split('wine-')[1].split('.')[0])
            if major_ver < 8:
                print(f"  ⚠ Warning: yabridge requires Wine 8.0+, you have {version}")
                return False
            else:
                print(f"  ✓ Wine version is compatible with yabridge")
                return True
    except FileNotFoundError:
        print("✗ Wine is not installed")
        return False

def check_yabridge():
    """Check if yabridge is installed."""
    yabridge_path = os.path.expanduser("~/.local/share/yabridge/yabridgectl")
    if os.path.exists(yabridge_path):
        print(f"✓ Yabridge installed at {yabridge_path}")
        return True
    else:
        print("✗ Yabridge not found")
        return False

def list_vst_files():
    """List available VST files."""
    vst_dir = os.path.expanduser("~/vst-windows")
    if not os.path.exists(vst_dir):
        print(f"✗ VST directory not found: {vst_dir}")
        return

    print(f"\n📦 VST files in {vst_dir}:")
    for file in os.listdir(vst_dir):
        filepath = os.path.join(vst_dir, file)
        size = os.path.getsize(filepath) / (1024*1024)  # MB
        print(f"  - {file} ({size:.1f} MB)")

def print_solutions():
    """Print available solutions."""
    print("\n" + "="*60)
    print("SOLUTIONS FOR WINDOWS VST ON LINUX")
    print("="*60)

    print("\n🔧 Option 1: Use Linux-Native VST Plugins (Easiest)")
    print("   Download Linux versions of your favorite plugins:")
    print("   - Look for .vst3 files with .so extensions inside")
    print("   - Popular free Linux reverbs:")
    print("     • Dragonfly Reverb: https://github.com/michaelwillis/dragonfly-reverb")
    print("     • OrilRiver: http://www.kvraudio.com/product/orilriver-by-denis-tihanov")
    print("     • CloudReverb: https://github.com/xunil-cloud/CloudReverb")

    print("\n🔧 Option 2: Upgrade Wine + Use Yabridge")
    print("   Required steps:")
    print("   1. Remove old Wine: sudo apt remove wine wine64")
    print("   2. Follow WineHQ Debian installation:")
    print("      https://wiki.winehq.org/Debian")
    print("   3. Install yabridge (already extracted at ~/.local/share/yabridge)")
    print("   4. Configure:")
    print("      yabridgectl add ~/vst-windows")
    print("      yabridgectl sync")

    print("\n🔧 Option 3: Use Built-in Effects")
    print("   Your vst_processor.py can use built-in Pedalboard effects:")
    print("   - Reverb, Delay, Chorus, Distortion, Compressor, etc.")
    print("   - Example script available at: demo_builtin_effects.py")

    print("\n" + "="*60)

def main():
    print("="*60)
    print("WINDOWS VST COMPATIBILITY CHECK")
    print("="*60)
    print()

    # Run checks
    wine_ok = check_wine_version()
    yabridge_ok = check_yabridge()
    list_vst_files()

    # Print status
    print("\n" + "="*60)
    print("STATUS")
    print("="*60)
    if wine_ok and yabridge_ok:
        print("✓ System is ready to use Windows VSTs with yabridge")
        print("  Run: yabridgectl sync")
    else:
        print("✗ System cannot use Windows VSTs yet")
        print_solutions()

    print("\n" + "="*60)
    print("For detailed setup instructions, see:")
    print("  /home/arlo/Data/windows_vst_setup.md")
    print("="*60)

if __name__ == '__main__':
    main()
