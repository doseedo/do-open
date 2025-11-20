# Agent Coordination - Musical Program Synthesis System

## Active Agents & Module Assignments

Last Updated: 2025-11-20

### Agent 1: Parameter Auditor & Refactorer (THIS AGENT)
**Status**: ✅ Active - Full Refactoring Mode

**Assigned Modules**:
- ✅ `audit/` - Parameter auditor (COMPLETE)
- ✅ `parameters/` - Universal registry (FOUNDATION COMPLETE)
- 🔄 `core/` - All core modules (IN PROGRESS)
  - `core/modal_harmony.py`
  - `core/neo_riemannian.py`
  - `core/microtonality.py`
  - `core/instrument_library.py`
  - `core/ensemble_registry.py`
  - `core/component_system.py`
  - `core/multi_genre_arranger.py`
- 🔄 `generators/` - Foundational generators (IN PROGRESS)
  - `generators/advanced_harmony_generator.py`
  - `generators/context_aware_generator.py`
  - `generators/development_engine.py`
  - `generators/form_generator.py`
  - `generators/texture_generator.py`
  - `generators/transition_engine.py`
  - `generators/orchestrator.py`
  - `generators/granular_control.py`
  - `generators/style_fusion.py`
  - `generators/reharmonization_engine.py`
  - `generators/intro_outro_generator.py`
  - `generators/harmonic_rhythm.py`
- 🔄 `transformation/` - All transformation modules
- 🔄 `analysis/` - All analysis modules
- 🔄 `algorithms/` - Core algorithms
- 🔄 `api/` - API layer modules

**Parameter Registry Sections Owned**:
- `harmony.*` - All harmony parameters
- `melody.*` - All melody parameters (if not claimed)
- `rhythm.*` - All rhythm parameters (if not claimed)
- `structure.*` - Structural parameters
- `transformation.*` - Transformation parameters
- `analysis.*` - Analysis parameters

**Current Progress**:
- Foundation: 28 parameters registered
- Target: 1,000+ parameters from assigned modules
- Estimated completion: Next few hours

---

### Agent 2: Parameter Coverage Validator
**Status**: ⏸️ Waiting for Agent 1 foundation

**Assigned Work**:
- Build validation framework
- Create test MIDI corpus (100+ files)
- Gap analysis tools

**Dependencies**: Needs parameter registry to be substantially populated

---

### Agent 3: Parameter Registry Builder
**Status**: 🤝 Collaborating with Agent 1

**Assigned Work**:
- Expand registry metadata
- Generate comprehensive documentation
- Build dependency graphs

**Coordination**: Shares `parameters/` folder with Agent 1

---

### Other Agents (If Active)
**Please claim folders below and update this file**

#### Available Folders for Assignment:
- 📁 `genres/` - 40+ genre modules (~1,400 parameters)
  - `genres/classic_rock.py` - 51 high-severity findings
  - `genres/singer_songwriter.py` - 32 findings
  - `genres/world/` - World music modules
  - Others...

- 📁 `learning/` - Learning modules (~200 parameters)
  - `learning/pattern_extractor.py` - 31 findings
  - `learning/motif_library.py` - 23 findings
  - `learning/pattern_recognition.py` - 26 findings
  - `learning/corpus_learner.py`

- 📁 `midi/` - MIDI handling (~150 parameters)
  - `midi/articulation_engine.py` - 4 findings
  - `midi/cc_automation.py` - 10 findings
  - `midi/mpe_support.py` - 4 findings

- 📁 `styles/` - Style profiles
  - `styles/basie_profile.py`
  - `styles/ellington_profile.py`
  - `styles/thad_jones_profile.py`

- 📁 `tools/` - Utility tools
  - `tools/big_band/` - Big band specific tools

---

## Parameter Registry Namespace Coordination

### Claimed Namespaces (Avoid Conflicts!)

#### Agent 1 Owns:
- `harmony.*` - All harmony-related parameters
- `melody.*` - Melodic parameters
- `rhythm.general.*` - General rhythm (not genre-specific)
- `structure.*` - Musical structure
- `transformation.*` - Transformations
- `analysis.*` - Analysis parameters
- `core.*` - Core system parameters

#### Available Namespaces:
- `genre.*` - Genre-specific parameters
  - `genre.rock.*`
  - `genre.jazz.*`
  - `genre.classical.*`
  - `genre.world.*`
  - etc.
- `instrument.*` - Instrument-specific
  - `instrument.guitar.*`
  - `instrument.piano.*`
  - `instrument.drums.*`
  - etc.
- `learning.*` - Learning system parameters
- `midi.*` - MIDI-specific parameters
- `style.*` - Style profile parameters

### Registry Conflict Resolution

If two agents attempt to register the same parameter path:

1. **Check this file first** before adding parameters
2. **Use more specific paths** if conflict exists
   - Instead of: `harmony.voicing.type`
   - Use: `harmony.jazz.voicing.type` or `harmony.classical.voicing.type`
3. **Document in commit message** which parameters you're adding
4. **Pull before pushing** to catch conflicts early

---

## Communication Protocol

### When Starting Work on a Module:
1. Update this file with your agent number and module
2. Commit and push this file
3. Begin refactoring

### When Adding Parameters to Registry:
1. Check no conflicts with existing parameters
2. Use your claimed namespace
3. Document in commit message: "Added 50 parameters to genre.rock.*"

### When Committing:
1. Clear commit message with parameter count
2. Reference this coordination file
3. Push to your branch

---

## Progress Tracking

### Total Parameters Target: 2,000+

#### By Agent:
- Agent 1: 28 → Target 1,000 (core, generators, harmony, melody)
- Agent 2: 0 → Target 0 (validation only)
- Agent 3: 0 → Target 100 (registry expansion)
- [Other Agents]: 0 → Target 900 (genres, instruments, etc.)

#### By Category:
- ✅ Harmony: 10 registered → Target 300
- ✅ Melody: 5 registered → Target 200
- ✅ Rhythm: 3 registered → Target 200
- ✅ Genre: 4 registered → Target 800
- ✅ Dynamics: 2 registered → Target 100
- ✅ Articulation: 1 registered → Target 100
- ⏸️ Instruments: 0 → Target 200
- ⏸️ Learning: 0 → Target 100

---

## Current Status Summary

**Active Agents**: 1 (Agent 1)
**Modules Being Refactored**: core/, generators/
**Parameters Registered**: 28 / 2000+ (1.4%)
**Next Milestone**: 500 parameters (25% of target)

**Last Update**: Agent 1 - 2025-11-20 - Starting full refactoring

---

## Notes

- All agents should use the `ParameterDefinition` class from `parameters/universal_registry.py`
- Follow refactoring patterns in `parameters/REFACTORING_GUIDE.md`
- Maintain 100% backward compatibility (defaults = original hardcoded values)
- Add unit tests for refactored modules
- Document all parameters with musical impact and genre relevance
