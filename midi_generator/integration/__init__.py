"""
MIDI Generator Integration Module - Agent 09
=============================================

Integration layer for hierarchical multi-task learning models with HarmonyModule.

This module provides:
- Parameter prediction from MIDI files
- MIDI generation from parameters
- Bidirectional workflows
- Style transfer capabilities
- Real-time generation pipeline

Components:
-----------
- HierarchicalMTLWrapper: Wrapper for trained neural network models
- ParameterPredictionAPI: High-level API for parameter prediction
- BidirectionalWorkflow: Coordinator for MIDI ↔ Parameters workflows
- EnhancedHarmonyModuleAPI: ML-enhanced HarmonyModule API

Quick Start:
------------
    >>> from midi_generator.integration import EnhancedHarmonyModuleAPI
    >>>
    >>> # Create API
    >>> api = EnhancedHarmonyModuleAPI()
    >>>
    >>> # Analyze MIDI
    >>> params = api.analyze_and_extract("song.mid")
    >>> print(f"Genre: {params['genre.primary']}")
    >>>
    >>> # Generate with style
    >>> api.generate_with_style("song.mid", "output.mid")
    >>>
    >>> # Style transfer
    >>> api.transfer_style_intelligent("source.mid", "target.mid", "output.mid")

Author: Agent 09 - HarmonyModule Integration Lead
Date: 2025-11-20
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Agent 09 - HarmonyModule Integration Lead"
__all__ = [
    # Main API
    'EnhancedHarmonyModuleAPI',
    'create_enhanced_api',

    # Core Components
    'HierarchicalMTLWrapper',
    'ParameterPredictionAPI',
    'BidirectionalWorkflow',

    # Data Structures
    'HierarchicalPrediction',
    'ParameterAnalysisResult',
    'StyleTransferResult',
    'ParameterBlendResult',
    'ParameterComparisonResult',

    # Convenience Functions
    'analyze_midi_file',
    'analyze_midi_directory',
    'transfer_style',
    'blend_styles',
]

# Import main components
try:
    from midi_generator.integration.hierarchical_model_wrapper import (
        HierarchicalMTLWrapper,
        HierarchicalPrediction,
        create_wrapper
    )
except ImportError as e:
    print(f"Warning: Could not import HierarchicalMTLWrapper: {e}")
    HierarchicalMTLWrapper = None
    HierarchicalPrediction = None
    create_wrapper = None

try:
    from midi_generator.integration.parameter_prediction_api import (
        ParameterPredictionAPI,
        ParameterAnalysisResult,
        analyze_midi_file,
        analyze_midi_directory
    )
except ImportError as e:
    print(f"Warning: Could not import ParameterPredictionAPI: {e}")
    ParameterPredictionAPI = None
    ParameterAnalysisResult = None
    analyze_midi_file = None
    analyze_midi_directory = None

try:
    from midi_generator.integration.bidirectional_workflow import (
        BidirectionalWorkflow,
        StyleTransferResult,
        ParameterBlendResult,
        ParameterComparisonResult,
        transfer_style,
        blend_styles
    )
except ImportError as e:
    print(f"Warning: Could not import BidirectionalWorkflow: {e}")
    BidirectionalWorkflow = None
    StyleTransferResult = None
    ParameterBlendResult = None
    ParameterComparisonResult = None
    transfer_style = None
    blend_styles = None

try:
    from midi_generator.integration.harmony_module_integration import (
        EnhancedHarmonyModuleAPI,
        create_enhanced_api
    )
except ImportError as e:
    print(f"Warning: Could not import EnhancedHarmonyModuleAPI: {e}")
    EnhancedHarmonyModuleAPI = None
    create_enhanced_api = None


def get_version():
    """Get module version"""
    return __version__


def check_dependencies():
    """Check if all dependencies are available"""
    dependencies = {
        'HierarchicalMTLWrapper': HierarchicalMTLWrapper is not None,
        'ParameterPredictionAPI': ParameterPredictionAPI is not None,
        'BidirectionalWorkflow': BidirectionalWorkflow is not None,
        'EnhancedHarmonyModuleAPI': EnhancedHarmonyModuleAPI is not None,
    }

    print("\nDependency Check:")
    print("=" * 50)
    for name, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    print("=" * 50)

    all_available = all(dependencies.values())
    if all_available:
        print("✓ All components available")
    else:
        print("⚠ Some components unavailable")

    return all_available


if __name__ == "__main__":
    print(f"MIDI Generator Integration Module v{__version__}")
    print(f"Author: {__author__}")
    print()
    check_dependencies()
