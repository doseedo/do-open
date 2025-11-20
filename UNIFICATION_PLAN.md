# Library Unification Plan

## Executive Summary

Merge `harmonymodule/midi_generator` and standalone `midi_generator` into one unified, comprehensive music generation library.

## Current State Analysis

### Directory Structure
```
/home/user/Do/
├── home/arlo/harmonymodule/
│   ├── midi_generator/ (92 Python files) - Advanced version
│   ├── advanced_modules/ (27 files) - Specialized modules
│   ├── inference/api/ - Production API
│   └── scripts/ - Production tools
└── midi_generator/ (77 Python files) - Standalone version
```

### File Breakdown
- **27 files unique to harmonymodule**: Advanced features (component system, style fusion, genre detection)
- **12 files unique to standalone**: Big band generators (9 versions), demos
- **65 files in common**: Core functionality (likely different versions)

### Size Comparison
- harmonymodule total: 4.8MB, 180 files, 112,617 lines
- standalone midi_generator: 2.2MB, 77 files, 43,352 lines
- Overlap: ~70% shared functionality

## Unification Strategy

### Phase 1: Analysis & Planning ✓
- [x] Compare file structures
- [x] Identify unique files in each
- [x] Identify common files (need version comparison)
- [x] Document current state

### Phase 2: Structure Design

**Proposed Unified Structure:**
```
midi_generator/  (new unified location)
├── core/                      # Music theory foundations
│   ├── instrument_library.py
│   ├── modal_harmony.py
│   ├── neo_riemannian.py
│   ├── microtonality.py
│   └── component_system.py    ← From harmonymodule (NEW)
│
├── algorithms/                # Composition algorithms
│   ├── rhythm_engine.py
│   ├── groove_library.py
│   ├── advanced_rhythm.py     ← From harmonymodule (NEW)
│   ├── drum_patterns.py       ← From harmonymodule (NEW)
│   ├── lsystem.py
│   ├── cellular_automata.py
│   └── constraint_solver.py
│
├── generators/                # Content generators
│   ├── form_generator.py
│   ├── orchestrator.py
│   ├── transition_engine.py
│   ├── development_engine.py
│   ├── texture_generator.py
│   ├── advanced_harmony_generator.py
│   ├── context_aware_generator.py  ← From harmonymodule (NEW)
│   ├── granular_control.py         ← From harmonymodule (NEW)
│   └── style_fusion.py             ← From harmonymodule (NEW)
│
├── genres/                    # Genre implementations
│   ├── jazz.py
│   ├── blues.py
│   ├── funk_soul.py
│   ├── electronic.py
│   ├── metal.py               ← From harmonymodule (NEW)
│   ├── rnb_neosoul.py
│   └── world/
│       ├── arabic.py
│       ├── indian.py
│       └── expanded.py
│
├── analysis/                  # MIDI analysis
│   ├── midi_analyzer.py
│   ├── pattern_extractor.py
│   ├── corpus_learner.py
│   └── genre_detector.py      ← From harmonymodule (NEW)
│
├── transformation/            # Style transfer
│   ├── style_transfer.py
│   ├── arrangement_engine.py
│   ├── inpainting_engine.py   ← From harmonymodule (NEW)
│   ├── tempo_converter.py     ← From harmonymodule (NEW)
│   └── meter_converter.py     ← From harmonymodule (NEW)
│
├── midi/                      # MIDI utilities
│   ├── midi_constants.py
│   ├── articulation_engine.py
│   ├── cc_automation.py
│   └── mpe_support.py
│
├── learning/                  # ML & pattern discovery
│   ├── pattern_extractor.py
│   ├── corpus_learner.py
│   ├── motif_library.py
│   └── fitness_learning.py
│
├── api/                       # High-level API
│   ├── __init__.py
│   ├── unified_api.py         ← From harmonymodule (NEW)
│   └── multi_genre_arranger.py ← From harmonymodule (NEW)
│
├── tools/                     # Production tools (NEW)
│   ├── big_band/              # Big band generators
│   │   ├── generate_final.py           # Recommended version
│   │   ├── generate_comprehensive.py   # With advanced harmony
│   │   └── README.md                   # Usage guide
│   └── examples/
│       ├── classic_rock_demo.py
│       ├── context_aware_examples.py
│       ├── style_fusion_demo.py
│       └── ... (all examples)
│
├── docs/                      # Unified documentation
│   ├── README.md              # Main documentation
│   ├── API_REFERENCE.md       # API documentation
│   ├── HARMONY_GUIDE.md       # Harmony system guide
│   ├── GENRE_GUIDE.md         # Genre capabilities
│   ├── ADVANCED_FEATURES.md   # Advanced features guide
│   └── MIGRATION_GUIDE.md     # For existing users
│
├── tests/                     # Test suite
│   ├── test_core/
│   ├── test_algorithms/
│   ├── test_generators/
│   └── test_integration/
│
└── README.md                  # Root README
```

### Phase 3: Version Comparison Strategy

For the 65 common files, determine which version to keep:

**Decision Criteria:**
1. **File size** - Larger usually means more features
2. **Last modified** - Newer usually has bug fixes
3. **Documentation** - Better documented version
4. **Dependencies** - Fewer external dependencies preferred
5. **Code quality** - Cleaner code structure

**Comparison Process:**
```bash
# For each common file:
1. Compare file sizes
2. Compare modification dates
3. diff to identify significant differences
4. Choose best version OR merge features
5. Document decision
```

### Phase 4: Migration Execution Plan

#### Step 1: Backup
```bash
# Create backups
cp -r midi_generator midi_generator.backup
cp -r home/arlo/harmonymodule home/arlo/harmonymodule.backup
```

#### Step 2: Create Unified Base
```bash
# Start with harmonymodule version (more complete)
cp -r home/arlo/harmonymodule/midi_generator midi_generator_unified
```

#### Step 3: Add Unique Standalone Features
```bash
# Create tools/big_band directory
mkdir -p midi_generator_unified/tools/big_band

# Copy big band generators
cp midi_generator/generate_big_band*.py midi_generator_unified/tools/big_band/

# Copy unique examples
cp midi_generator/examples/classic_rock_demo.py midi_generator_unified/tools/examples/
```

#### Step 4: Merge Common Files
For each of the 65 common files:
1. Compare versions
2. Keep better version OR merge manually
3. Document decision

#### Step 5: Add Advanced Modules Integration
```bash
# Copy advanced modules to unified location
cp -r home/arlo/harmonymodule/advanced_modules midi_generator_unified/advanced/
```

#### Step 6: Update All Imports
```python
# Update import paths throughout codebase
# Old: from midi_generator.core import X
# New: from midi_generator.core import X (unchanged)
# But ensure all cross-references work
```

#### Step 7: Create Unified API
```python
# midi_generator/__init__.py
"""
Unified Music Generation Library
Combines all capabilities into one comprehensive system
"""

# Export primary classes for easy access
from .api.unified_api import UnifiedMusicGenerator
from .generators.advanced_harmony_generator import AdvancedHarmonyGenerator
from .core.modal_harmony import Mode, ModalProgressionGenerator
from .core.neo_riemannian import Triad, NeoRiemannianTransformations
# ... etc

__version__ = "2.0.0"  # Major version for unification
```

#### Step 8: Create Comprehensive Documentation
- Merge docs from both systems
- Create migration guide for existing users
- Document all new features from harmonymodule
- Create quick start guide
- API reference

#### Step 9: Testing
```bash
# Run all tests
python -m pytest midi_generator_unified/tests/

# Test big band generators
python midi_generator_unified/tools/big_band/generate_final.py

# Test API
python -c "from midi_generator import UnifiedMusicGenerator; print('Success')"
```

#### Step 10: Replace Old Versions
```bash
# After testing passes:
mv midi_generator midi_generator_old
mv midi_generator_unified midi_generator

# Delete redundant harmonymodule version
# (Keep advanced_modules and inference layers separate)
```

### Phase 5: Cleanup & Organization

**Files to Keep from Both:**
- ✅ All 27 unique harmonymodule files (advanced features)
- ✅ All 12 unique standalone files (big band generators)
- ✅ Best version of 65 common files

**Files to Consolidate:**
- Big band generators → `tools/big_band/`
- All examples → `tools/examples/`
- Advanced modules → `advanced/` (new top-level directory)

**Files to Archive:**
- Old big band versions → `tools/big_band/archive/`
- Experimental scripts → `tools/experimental/`

### Phase 6: Documentation Updates

**New Documentation Structure:**
```
docs/
├── README.md                    # Main overview
├── GETTING_STARTED.md          # Quick start
├── API_REFERENCE.md            # Complete API docs
├── HARMONY_SYSTEM.md           # Harmony capabilities (31+ progressions)
├── GENRE_GUIDE.md              # All 35+ genres
├── ADVANCED_FEATURES.md        # Component system, fusion, etc.
├── BIG_BAND_GUIDE.md          # Big band generator guide
├── MIGRATION_GUIDE.md         # For existing users
└── CONTRIBUTING.md            # Development guide
```

## Benefits of Unification

### For Users
1. **One import path** - No confusion about which library to use
2. **All features available** - Component system + big band generators
3. **Consistent API** - Unified interface across all features
4. **Better documentation** - Comprehensive, unified docs
5. **Easier learning** - One library to learn, not two

### For Development
1. **No code duplication** - Maintain one version of shared files
2. **Easier testing** - One test suite
3. **Clearer architecture** - Organized structure
4. **Faster iteration** - Changes apply everywhere
5. **Better integration** - Features work together seamlessly

### Technical Benefits
1. **Reduced disk space** - Eliminate ~70% redundancy
2. **Faster imports** - No duplicate loading
3. **Better performance** - Optimized for single library
4. **Easier deployment** - One package to distribute
5. **Version control** - Cleaner git history

## Implementation Timeline

### Week 1: Analysis & Planning
- [x] Day 1-2: Complete file analysis
- [ ] Day 3-4: Version comparison of 65 common files
- [ ] Day 5: Finalize structure design
- [ ] Day 6-7: Create migration scripts

### Week 2: Execution
- [ ] Day 1: Create backups
- [ ] Day 2: Build unified base structure
- [ ] Day 3-4: Merge files and resolve conflicts
- [ ] Day 5: Update imports and references
- [ ] Day 6-7: Testing and bug fixes

### Week 3: Documentation & Release
- [ ] Day 1-2: Write unified documentation
- [ ] Day 3-4: Create migration guide
- [ ] Day 5: Final testing
- [ ] Day 6: Release unified library v2.0
- [ ] Day 7: Deprecate old libraries

## Migration Guide for Users

**If you were using standalone midi_generator:**
```python
# OLD
from midi_generator.generators.advanced_harmony_generator import AdvancedHarmonyGenerator

# NEW (same!)
from midi_generator.generators.advanced_harmony_generator import AdvancedHarmonyGenerator

# NEW FEATURES NOW AVAILABLE:
from midi_generator.api import UnifiedMusicGenerator  # High-level API
from midi_generator.generators import StyleFusion      # Genre blending
from midi_generator.analysis import GenreDetector      # Genre detection
```

**If you were using harmonymodule:**
```python
# OLD
from harmonymodule.midi_generator.core import modal_harmony

# NEW
from midi_generator.core import modal_harmony

# Big band generators now available:
from midi_generator.tools.big_band import generate_final
```

## Rollback Plan

If unification fails:
1. Restore from backups (`midi_generator.backup`, `harmonymodule.backup`)
2. Document issues encountered
3. Revise plan and retry

## Success Criteria

- [ ] All tests pass in unified library
- [ ] No functionality lost from either library
- [ ] All 27 harmonymodule unique files integrated
- [ ] All 12 standalone unique files integrated
- [ ] Imports work correctly throughout
- [ ] Documentation is comprehensive
- [ ] Big band generators work as before
- [ ] API provides easy access to all features
- [ ] Performance is equal or better
- [ ] Users can migrate with minimal code changes

## Next Steps

1. **Immediate**: Complete version comparison of 65 common files
2. **Next**: Create migration scripts
3. **Then**: Execute unification
4. **Finally**: Test, document, and release v2.0

---

**Status**: Phase 1 Complete ✓ | Phase 2 In Progress
**Owner**: Claude
**Timeline**: 3 weeks
**Priority**: High
