#!/usr/bin/env python3
"""
VST Plugin Audio Processor
Loads VST plugins and processes audio files through them.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from pedalboard import Pedalboard, load_plugin
from pedalboard.io import AudioFile


def list_plugin_parameters(plugin):
    """List all parameters available in the plugin."""
    print(f"\n{'='*60}")
    print(f"Plugin: {plugin}")
    print(f"{'='*60}")

    try:
        # Get parameter info
        params = {}
        for param_name in dir(plugin):
            if not param_name.startswith('_'):
                try:
                    value = getattr(plugin, param_name)
                    if isinstance(value, (int, float, bool)):
                        params[param_name] = value
                except:
                    pass

        if params:
            print("\nAvailable Parameters:")
            for name, value in params.items():
                print(f"  {name}: {value}")
        else:
            print("\nNo adjustable parameters found (or they require special access)")

    except Exception as e:
        print(f"Error listing parameters: {e}")

    print(f"{'='*60}\n")


def process_audio(input_file, output_file, plugin_path, sample_rate=44100,
                  show_params=False, **plugin_params):
    """
    Process an audio file through a VST plugin.

    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        plugin_path: Path to VST plugin (.so or .vst3)
        sample_rate: Sample rate for processing
        show_params: If True, list available parameters and exit
        **plugin_params: Plugin parameters to set (e.g., mix=0.5, decay=2.0)
    """

    print(f"Loading plugin: {plugin_path}")

    # Load the VST plugin
    try:
        plugin = load_plugin(plugin_path, plugin_name=None)
        print(f"✓ Plugin loaded successfully: {plugin}")
    except Exception as e:
        print(f"✗ Error loading plugin: {e}")
        print("\nNote: Make sure you're using a Linux-compatible VST plugin (.so or .vst3)")
        print("macOS plugins (.component, .vst) will not work on Linux.")
        return False

    # If user wants to see parameters, show them and exit
    if show_params:
        list_plugin_parameters(plugin)
        return True

    # Set plugin parameters if provided
    if plugin_params:
        print("\nSetting plugin parameters:")
        for param_name, param_value in plugin_params.items():
            try:
                setattr(plugin, param_name, param_value)
                print(f"  {param_name} = {param_value}")
            except AttributeError:
                print(f"  Warning: Parameter '{param_name}' not found on plugin")
            except Exception as e:
                print(f"  Error setting {param_name}: {e}")

    # Create pedalboard with the plugin
    board = Pedalboard([plugin])

    # Read input audio file
    print(f"\nReading input file: {input_file}")
    with AudioFile(input_file) as f:
        audio = f.read(f.frames)
        input_sample_rate = f.samplerate
        num_channels = f.num_channels

    print(f"  Channels: {num_channels}")
    print(f"  Sample rate: {input_sample_rate} Hz")
    print(f"  Duration: {len(audio[0])/input_sample_rate:.2f} seconds")

    # Process the audio
    print(f"\nProcessing audio through plugin...")
    processed = board(audio, input_sample_rate)

    # Write output audio file
    print(f"Writing output file: {output_file}")
    with AudioFile(output_file, 'w', input_sample_rate, num_channels) as f:
        f.write(processed)

    print(f"\n✓ Processing complete!")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")

    return True


def find_vst_plugins(search_paths=None):
    """Search for VST plugins in common locations."""
    if search_paths is None:
        search_paths = [
            "/usr/lib/vst",
            "/usr/lib/vst3",
            "/usr/local/lib/vst",
            "/usr/local/lib/vst3",
            os.path.expanduser("~/.vst"),
            os.path.expanduser("~/.vst3"),
            "/home/arlo/Data/plugin"
        ]

    plugins = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.so'):
                        plugins.append(os.path.join(root, file))
                for dir in dirs:
                    if dir.endswith('.vst3'):
                        plugins.append(os.path.join(root, dir))

    return plugins


def main():
    parser = argparse.ArgumentParser(
        description='Process audio through VST plugins',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available parameters
  python vst_processor.py -p /path/to/plugin.so --show-params

  # Process audio with default settings
  python vst_processor.py -i input.wav -o output.wav -p /path/to/plugin.so

  # Process audio with custom parameters
  python vst_processor.py -i input.wav -o output.wav -p /path/to/reverb.so --param mix=0.5 --param decay=2.0

  # Search for installed VST plugins
  python vst_processor.py --find-plugins
        """
    )

    parser.add_argument('-i', '--input', help='Input audio file')
    parser.add_argument('-o', '--output', help='Output audio file')
    parser.add_argument('-p', '--plugin', help='Path to VST plugin (.so or .vst3)')
    parser.add_argument('-r', '--sample-rate', type=int, default=44100,
                        help='Sample rate (default: 44100)')
    parser.add_argument('--show-params', action='store_true',
                        help='Show plugin parameters and exit')
    parser.add_argument('--param', action='append', nargs=1, metavar='KEY=VALUE',
                        help='Set plugin parameter (can be used multiple times)')
    parser.add_argument('--find-plugins', action='store_true',
                        help='Search for VST plugins on the system')

    args = parser.parse_args()

    # Find plugins mode
    if args.find_plugins:
        print("Searching for VST plugins...")
        plugins = find_vst_plugins()
        if plugins:
            print(f"\nFound {len(plugins)} plugin(s):")
            for plugin in plugins:
                print(f"  {plugin}")
        else:
            print("\nNo VST plugins found in common locations.")
            print("You may need to install Linux VST plugins or specify a custom search path.")
        return

    # Normal processing mode
    if not args.plugin:
        parser.error("Plugin path is required (use -p/--plugin)")

    if not args.show_params:
        if not args.input:
            parser.error("Input file is required (use -i/--input)")
        if not args.output:
            parser.error("Output file is required (use -o/--output)")

    # Parse plugin parameters
    plugin_params = {}
    if args.param:
        for param in args.param:
            param_str = param[0]
            if '=' not in param_str:
                print(f"Warning: Invalid parameter format '{param_str}', expected KEY=VALUE")
                continue
            key, value = param_str.split('=', 1)
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    plugin_params[key] = float(value)
                else:
                    plugin_params[key] = int(value)
            except ValueError:
                if value.lower() in ('true', 'false'):
                    plugin_params[key] = value.lower() == 'true'
                else:
                    plugin_params[key] = value

    # Process the audio
    success = process_audio(
        args.input,
        args.output,
        args.plugin,
        args.sample_rate,
        args.show_params,
        **plugin_params
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
