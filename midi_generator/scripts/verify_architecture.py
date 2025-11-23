#!/usr/bin/env python3
"""
Verify Discovery Pipeline Architecture

Checks that all components are in place and ready for discovery.
Does NOT require PyTorch/GPU - just verifies file structure and imports.

Usage:
    python scripts/verify_architecture.py

Author: Agent 8
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "1_approaches/transform_based"))

def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and report."""
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    if exists:
        lines = len(path.read_text().splitlines())
        print(f"     ({lines:,} lines)")
    return exists

def check_imports(module_path: str, description: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_path)
        print(f"  ✓ {description}")
        return True
    except ImportError as e:
        print(f"  ✗ {description}")
        print(f"     Error: {e}")
        return False
    except Exception as e:
        print(f"  ⚠ {description} (exists but has runtime error)")
        print(f"     Error: {e}")
        return True  # File exists, just needs dependencies

def main():
    print("="*70)
    print("DISCOVERY PIPELINE ARCHITECTURE VERIFICATION")
    print("="*70)
    print()

    all_good = True

    # Check core components
    print("Core Components:")
    core_dir = project_root / "1_approaches/transform_based/core"
    all_good &= check_file_exists(core_dir / "minimal_theoretical_base.py",
                                   "17 irreducible primitives (theoretical foundation)")
    all_good &= check_file_exists(core_dir / "tensor_representation.py",
                                   "MIDI ↔ tensor conversion (GPU-ready)")
    all_good &= check_file_exists(core_dir / "tensor_transforms.py",
                                   "GPU transform operations (instrument-aware)")
    all_good &= check_file_exists(core_dir / "gpu_memory_manager.py",
                                   "A100 memory optimization")
    all_good &= check_file_exists(core_dir / "space_level_transforms.py",
                                   "MIDI extraction (program number support)")
    print()

    # Check discovery components
    print("Discovery Components:")
    discovery_dir = project_root / "1_approaches/transform_based/discovery"
    all_good &= check_file_exists(discovery_dir / "gpu_sparse_coding.py",
                                   "FISTA sparse coding (GPU-accelerated)")
    all_good &= check_file_exists(discovery_dir / "gpu_discovery_pipeline.py",
                                   "End-to-end discovery pipeline")
    all_good &= check_file_exists(discovery_dir / "abstraction_layer.py",
                                   "V2 hierarchical abstraction")
    print()

    # Check scripts
    print("Startup Scripts:")
    scripts_dir = project_root / "scripts"
    transform_scripts_dir = project_root / "1_approaches/transform_based/scripts"
    all_good &= check_file_exists(scripts_dir / "start_discovery.py",
                                   "Discovery startup script")
    all_good &= check_file_exists(transform_scripts_dir / "benchmark_gpu.py",
                                   "GPU vs CPU benchmarking")
    print()

    # Check documentation
    print("Documentation:")
    all_good &= check_file_exists(project_root / "DISCOVERY_QUICKSTART.md",
                                   "Quickstart guide")
    all_good &= check_file_exists(project_root / "docs/GPU_TENSORIZATION_GUIDE.md",
                                   "GPU tensorization guide")
    print()

    # Check corpus
    print("MIDI Corpus:")
    corpus_path = Path("/home/user/Do/midi_generator/midi_corpus/big_band")
    if corpus_path.exists():
        midi_files = list(corpus_path.glob("*.mid"))
        print(f"  ✓ Corpus directory found")
        print(f"     {len(midi_files):,} MIDI files")
    else:
        print(f"  ✗ Corpus directory not found: {corpus_path}")
        all_good = False
    print()

    # Check Python dependencies (optional - will fail if not installed)
    print("Python Dependencies (optional - install before running):")
    deps_ok = True
    deps_ok &= check_imports("torch", "PyTorch (required for GPU)")
    deps_ok &= check_imports("numpy", "NumPy")
    deps_ok &= check_imports("mido", "Mido (MIDI library)")
    print()

    if not deps_ok:
        print("NOTE: Dependencies not installed yet. Install with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  pip install mido numpy tqdm")
        print()

    # Summary
    print("="*70)
    if all_good and deps_ok:
        print("✓ ARCHITECTURE COMPLETE - READY FOR DISCOVERY")
        print("="*70)
        print()
        print("To start discovery:")
        print("  python scripts/start_discovery.py")
        print()
        return 0
    elif all_good:
        print("✓ ARCHITECTURE COMPLETE - INSTALL DEPENDENCIES")
        print("="*70)
        print()
        print("Install dependencies:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  pip install mido numpy tqdm")
        print()
        print("Then start discovery:")
        print("  python scripts/start_discovery.py")
        print()
        return 0
    else:
        print("✗ ARCHITECTURE INCOMPLETE")
        print("="*70)
        print()
        print("Some components are missing. Review the checklist above.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
