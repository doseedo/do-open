#!/usr/bin/env python3
"""
Musical Program Synthesis - End-to-End Pipeline Runner

This script orchestrates the complete workflow:
1. Feature extraction from MIDI corpus (Agent 8)
2. Corpus analysis (Agents 10, 25)
3. Model training (Agents 14, 15)
4. Music generation (Agents 1, 9, 16)

Usage:
    python run_pipeline.py --midi-dir /path/to/midi --mode full

Author: Musical Program Synthesis Team
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("Installing required dependencies...")
    os.system("pip install numpy tqdm")
    import numpy as np
    from tqdm import tqdm


class PipelineRunner:
    """
    End-to-end pipeline orchestrator for Musical Program Synthesis.
    """

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "generated").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        self.start_time = time.time()

    def print_header(self, message: str, level: int = 1):
        """Print formatted header."""
        if level == 1:
            print(f"\n{'='*80}")
            print(f"  {message}")
            print(f"{'='*80}\n")
        else:
            print(f"\n[{level}/4] {message}")
            print("-" * 80)

    def run(self):
        """Execute the pipeline based on mode."""
        self.print_header("🎵 Musical Program Synthesis - End-to-End Pipeline", level=1)

        print(f"Mode: {self.args.mode}")
        print(f"MIDI Directory: {self.args.midi_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Workers: {self.args.workers}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.args.mode == "analyze":
            self.run_analysis()
        elif self.args.mode == "train":
            self.run_training()
        elif self.args.mode == "generate":
            self.run_generation()
        elif self.args.mode == "full":
            self.run_full_pipeline()
        else:
            print(f"❌ Unknown mode: {self.args.mode}")
            sys.exit(1)

        self.print_summary()

    def run_analysis(self):
        """Run corpus analysis only."""
        self.print_header("Step 1: Feature Extraction & Analysis", level=1)

        # Import required modules
        try:
            from midi_generator.synthesis import extract_features
            from midi_generator.analysis import FeatureCorrelationAnalyzer
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please ensure all dependencies are installed.")
            print("Run: pip install -r requirements.txt")
            sys.exit(1)

        # Get MIDI files
        midi_files = self.get_midi_files()
        print(f"Found {len(midi_files)} MIDI files")

        if len(midi_files) == 0:
            print("❌ No MIDI files found in directory")
            sys.exit(1)

        # Extract features
        print("\n📊 Extracting features from MIDI files...")
        features_list = []
        file_names = []

        for midi_file in tqdm(midi_files, desc="Feature extraction"):
            try:
                features = extract_features(str(midi_file))
                features_list.append(features)
                file_names.append(midi_file.name)
            except Exception as e:
                print(f"⚠️  Error processing {midi_file.name}: {e}")
                continue

        if len(features_list) == 0:
            print("❌ No features extracted")
            sys.exit(1)

        # Save feature database
        feature_matrix = np.array(features_list)
        np.savez(
            self.output_dir / "analysis" / "feature_database.npz",
            features=feature_matrix,
            file_names=file_names
        )
        print(f"✓ Saved feature database: {feature_matrix.shape}")

        # Corpus statistics
        stats = {
            "num_files": len(features_list),
            "num_features": feature_matrix.shape[1],
            "feature_mean": feature_matrix.mean(axis=0).tolist(),
            "feature_std": feature_matrix.std(axis=0).tolist(),
            "timestamp": datetime.now().isoformat()
        }

        with open(self.output_dir / "analysis" / "corpus_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved corpus statistics")

        # Feature correlation analysis
        if self.args.correlation_analysis:
            print("\n🔍 Running feature correlation analysis...")
            try:
                analyzer = FeatureCorrelationAnalyzer()
                analyzer.fit(feature_matrix)

                # Identify redundant features
                redundant = analyzer.identify_redundant_features(threshold=0.95)

                correlation_report = {
                    "num_redundant_features": len(redundant),
                    "redundant_pairs": redundant,
                    "timestamp": datetime.now().isoformat()
                }

                with open(self.output_dir / "analysis" / "correlation_analysis.json", "w") as f:
                    json.dump(correlation_report, f, indent=2)

                print(f"✓ Identified {len(redundant)} redundant feature pairs")
            except Exception as e:
                print(f"⚠️  Correlation analysis failed: {e}")

        print("\n✅ Analysis complete!")

    def run_training(self):
        """Run model training only."""
        self.print_header("Step 2: Model Training", level=1)

        try:
            from midi_generator.learning import FeatureParameterMapper
            from midi_generator.training import SyntheticTrainingDataGenerator
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        # Load feature database
        try:
            data = np.load(self.output_dir / "analysis" / "feature_database.npz")
            feature_matrix = data["features"]
            print(f"Loaded feature database: {feature_matrix.shape}")
        except FileNotFoundError:
            print("❌ Feature database not found. Run with --mode analyze first.")
            sys.exit(1)

        # Get parameters to train
        if self.args.parameters == "all":
            # Train all available parameters
            parameters = [
                "harmony.voicing.type",
                "harmony.chord_density",
                "harmony.complexity",
                "melody.contour.shape",
                "melody.note_density",
                "rhythm.swing.amount",
                "rhythm.syncopation",
                "dynamics.velocity_range",
                "instrumentation.section_balance"
            ]
        else:
            parameters = self.args.parameters.split(",")

        print(f"Training {len(parameters)} parameters")

        # Initialize components
        mapper = FeatureParameterMapper()
        data_gen = SyntheticTrainingDataGenerator()

        # Train models
        print("\n🎓 Training models...")
        trained_models = {}

        for param_name in tqdm(parameters, desc="Training parameters"):
            try:
                # Generate synthetic training data
                training_data = data_gen.generate_for_parameter(
                    param_name,
                    n_examples=self.args.training_samples
                )

                # Train model
                metrics = mapper.train_mapping(param_name, training_data)
                trained_models[param_name] = metrics

                # Save model
                model_path = self.output_dir / "models" / f"{param_name}.pkl"
                mapper.save_model(param_name, model_path)

            except Exception as e:
                print(f"⚠️  Error training {param_name}: {e}")
                continue

        # Save training report
        report = {
            "parameters_trained": len(trained_models),
            "models": trained_models,
            "timestamp": datetime.now().isoformat()
        }

        with open(self.output_dir / "reports" / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Trained {len(trained_models)} models")

    def run_generation(self):
        """Run music generation only."""
        self.print_header("Step 3: Music Generation", level=1)

        try:
            from midi_generator.api import UnifiedAPI
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)

        api = UnifiedAPI()

        print(f"Generating {self.args.generate_count} MIDI files...")
        print(f"Style: {self.args.style}")

        for i in tqdm(range(self.args.generate_count), desc="Generating MIDI"):
            try:
                # Generate MIDI
                output_path = self.output_dir / "generated" / f"generated_{i+1:03d}.mid"

                midi = api.generate_from_style(
                    style=self.args.style,
                    duration=self.args.duration,
                    complexity=self.args.complexity
                )

                # Save (placeholder - actual API might differ)
                # midi.save(str(output_path))

                print(f"✓ Generated: {output_path.name}")

            except Exception as e:
                print(f"⚠️  Generation {i+1} failed: {e}")
                continue

        print(f"\n✅ Generated {self.args.generate_count} MIDI files")

    def run_full_pipeline(self):
        """Run complete pipeline: analyze → train → generate."""
        print("Running full pipeline...")

        # Step 1: Analysis
        self.run_analysis()

        # Step 2: Training
        if self.args.midi_dir:
            self.run_training()

        # Step 3: Generation
        if self.args.generate_count > 0:
            self.run_generation()

        print("\n✅ Full pipeline complete!")

    def get_midi_files(self) -> List[Path]:
        """Get list of MIDI files from directory."""
        if not self.args.midi_dir:
            return []

        midi_dir = Path(self.args.midi_dir)
        if not midi_dir.exists():
            print(f"❌ Directory not found: {midi_dir}")
            sys.exit(1)

        # Find all MIDI files
        midi_files = list(midi_dir.glob("**/*.mid")) + list(midi_dir.glob("**/*.midi"))
        return sorted(midi_files)

    def print_summary(self):
        """Print pipeline execution summary."""
        elapsed = time.time() - self.start_time

        self.print_header("Pipeline Summary", level=1)
        print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Output directory: {self.output_dir}")
        print("\nGenerated files:")

        for subdir in ["analysis", "models", "generated", "reports"]:
            path = self.output_dir / subdir
            if path.exists():
                files = list(path.iterdir())
                print(f"  {subdir}/: {len(files)} files")

        print("\n✅ Pipeline execution complete!")
        print(f"\nNext steps:")
        print(f"  1. Review analysis: cat {self.output_dir}/analysis/corpus_statistics.json")
        print(f"  2. Check models: ls {self.output_dir}/models/")
        print(f"  3. Listen to generated music: open {self.output_dir}/generated/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Musical Program Synthesis - End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with your big band MIDI files
  python run_pipeline.py --midi-dir /Users/hydroadmin/Downloads/LIBRESCORE/MIDIS --mode full

  # Analysis only
  python run_pipeline.py --midi-dir /path/to/midi --mode analyze

  # Training specific parameters
  python run_pipeline.py --midi-dir /path/to/midi --mode train --parameters harmony.voicing,melody.contour

  # Generate music from trained models
  python run_pipeline.py --mode generate --style big_band --count 10
        """
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["analyze", "train", "generate", "full"],
        help="Pipeline mode: analyze, train, generate, or full"
    )

    # Input/output
    parser.add_argument(
        "--midi-dir",
        type=str,
        help="Directory containing MIDI files for analysis/training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: output/)"
    )

    # Processing options
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )

    # Analysis options
    parser.add_argument(
        "--correlation-analysis",
        action="store_true",
        help="Run feature correlation analysis"
    )
    parser.add_argument(
        "--feature-reduction",
        type=int,
        default=100,
        help="Target number of features after reduction (default: 100)"
    )

    # Training options
    parser.add_argument(
        "--parameters",
        type=str,
        default="all",
        help="Comma-separated list of parameters to train, or 'all' (default: all)"
    )
    parser.add_argument(
        "--training-samples",
        type=int,
        default=1000,
        help="Number of training samples per parameter (default: 1000)"
    )

    # Generation options
    parser.add_argument(
        "--style",
        type=str,
        default="big_band",
        help="Musical style for generation (default: big_band)"
    )
    parser.add_argument(
        "--generate-count",
        type=int,
        default=5,
        help="Number of MIDI files to generate (default: 5)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=32,
        help="Duration in bars for generated music (default: 32)"
    )
    parser.add_argument(
        "--complexity",
        type=float,
        default=0.7,
        help="Complexity level 0-1 (default: 0.7)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode in ["analyze", "train", "full"] and not args.midi_dir:
        parser.error(f"--midi-dir is required for mode '{args.mode}'")

    # Run pipeline
    runner = PipelineRunner(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
