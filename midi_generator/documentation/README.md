# Documentation Generator for Musical Parameter System

**Agent 27: Documentation Generator** - Part of the 35-agent self-expanding inverse music generation system.

## Overview

The Documentation Generator automatically creates comprehensive documentation for musical parameters in the system. When new parameters are proposed by the LLM-guided expansion pipeline, this system generates:

- **Reference Documentation**: Comprehensive markdown docs with musical context, usage examples, and genre-specific values
- **Usage Examples**: Python code examples demonstrating 5+ usage scenarios
- **Changelog Entries**: Formatted changelog entries for version control
- **API Documentation**: Machine-readable JSON API specifications
- **Quick Reference Cards**: Terminal-friendly quick reference guides
- **Integration Guides**: Developer guides for integrating new parameters

## Features

### Comprehensive Documentation Types

1. **Reference Documentation** (Markdown)
   - Parameter overview (type, range, default, category)
   - Musical context and theory
   - Code examples
   - Genre-specific value recommendations
   - Related parameters
   - Integration information

2. **Usage Examples** (Python)
   - Direct parameter control
   - Genre preset usage
   - Interactive parameter exploration
   - Batch generation with variations
   - Feature extractor integration

3. **Changelog Entries**
   - Formatted for CHANGELOG.md
   - Version-tracked parameter additions
   - Integration notes

4. **API Documentation** (JSON)
   - Machine-readable specifications
   - Validation rules
   - Integration points
   - Usage patterns

5. **Quick Reference Cards**
   - Terminal-friendly ASCII art
   - At-a-glance parameter information
   - Genre value examples

6. **Integration Guides**
   - Developer instructions
   - Code templates for HarmonyModule
   - Feature extraction templates
   - XGBoost model training guides
   - Validation templates
   - Testing examples

### Intelligent Features

- **Auto-enrichment**: Automatically adds missing context and metadata
- **Related Parameter Detection**: Builds parameter relationship graph
- **Genre Value Extraction**: Maps parameters to genre profiles
- **Musical Context Generation**: Creates theory-based explanations
- **Validation**: Comprehensive quality checks
- **Batch Processing**: Generate docs for multiple parameters

## Installation

The Documentation Generator is part of the main system. No additional installation required.

```bash
# Located at:
# /home/user/Do/midi_generator/documentation/
```

## Usage

### Basic Usage: Single Parameter

```python
from documentation.doc_generator import DocumentationGenerator

# Initialize generator
doc_gen = DocumentationGenerator()

# Define parameter proposal
proposal = {
    'name': 'harmony.tritone_substitution',
    'type': 'float',
    'range': {'min': 0.0, 'max': 1.0},
    'default': 0.5,
    'description': 'Controls frequency of tritone substitutions in chord progressions',
    'musical_context': 'Tritone substitution replaces dominant chords with chords '
                      'a tritone away, creating smooth voice leading and sophisticated '
                      'harmonic movement. Common in bebop and modern jazz.',
    'category': 'harmony',
    'example_values': {
        'bebop': 0.8,
        'cool_jazz': 0.5,
        'fusion': 0.7
    },
    'related_parameters': [
        'harmony.chromaticism',
        'harmony.bebop_changes'
    ]
}

# Generate all documentation
docs = doc_gen.generate_parameter_docs(proposal)

# Access different documentation types
print(docs['reference_doc'])      # Markdown reference
print(docs['usage_examples'])     # Python examples
print(docs['changelog_entry'])    # Changelog entry
print(docs['api_docs'])           # JSON API spec
print(docs['quick_reference'])    # Quick reference card
print(docs['integration_guide'])  # Integration guide
```

### Batch Documentation Generation

```python
# Multiple parameters
proposals = [
    {
        'name': 'harmony.modal_interchange',
        'type': 'float',
        'range': {'min': 0.0, 'max': 1.0},
        'default': 0.4,
        'description': 'Probability of borrowing chords from parallel modes',
        'category': 'harmony'
    },
    {
        'name': 'rhythm.metric_modulation',
        'type': 'float',
        'range': {'min': 0.0, 'max': 1.0},
        'default': 0.2,
        'description': 'Frequency of metric modulation events',
        'category': 'rhythm'
    }
]

# Generate batch documentation
batch_docs = doc_gen.generate_batch_docs(proposals)

# Access aggregated documentation
print(batch_docs['index'])                  # Documentation index
print(batch_docs['combined_changelog'])     # Combined changelog
print(batch_docs['category_summaries'])     # Category summaries
```

### Save Documentation to Files

```python
# Generate documentation
docs = doc_gen.generate_parameter_docs(proposal)

# Save all documentation types
saved_files = doc_gen.save_documentation(
    docs,
    param_name='harmony.tritone_substitution'
)

# Files saved:
# - {param_name}_reference.md      (Reference documentation)
# - {param_name}_examples.py       (Usage examples)
# - {param_name}_changelog.md      (Changelog entry)
# - {param_name}_api.json          (API documentation)
# - {param_name}_quickref.txt      (Quick reference)
# - {param_name}_integration.md    (Integration guide)
```

### Validate Documentation Quality

```python
from documentation.doc_generator import DocumentationValidator

validator = DocumentationValidator()

# Validate parameter proposal before generation
is_valid, issues = validator.validate_proposal(proposal)

if not is_valid:
    print("Proposal issues:")
    for issue in issues:
        print(f"  - {issue}")

# Validate generated documentation
is_valid, issues = validator.validate_documentation(docs)

if is_valid:
    print("✓ Documentation quality passed")
else:
    print("Documentation issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### Update Master Documentation

```python
# Update main PARAMETERS.md file
doc_gen.update_master_documentation(
    param_proposals=proposals,
    master_doc_path="/home/user/Do/midi_generator/parameters/PARAMETERS.md"
)
```

## Parameter Proposal Format

A parameter proposal requires these fields:

### Required Fields

- **`name`**: Parameter name in format `category.parameter_name` (e.g., `harmony.tritone_substitution`)
- **`type`**: Data type (`float`, `int`, `bool`, `str`)
- **`range`**: Valid range (dict with `min`/`max` or list of valid values)
- **`default`**: Default value
- **`description`**: Short description (minimum 20 characters)

### Optional Fields

- **`musical_context`**: Detailed musical theory explanation (recommended 50+ chars)
- **`category`**: Parameter category (extracted from name if not provided)
- **`example_values`**: Dict mapping genre names to typical values
- **`related_parameters`**: List of related parameter names (auto-detected if not provided)

### Example Proposal

```python
proposal = {
    # Required
    'name': 'harmony.voice_leading_strictness',
    'type': 'float',
    'range': {'min': 0.0, 'max': 1.0},
    'default': 0.7,
    'description': 'Controls strictness of voice leading rules',

    # Optional (but recommended)
    'musical_context': 'Voice leading determines how individual notes move from '
                      'chord to chord. Stricter voice leading follows classical '
                      'rules (stepwise motion, minimal leaps), while looser voice '
                      'leading allows more freedom.',
    'category': 'harmony',
    'example_values': {
        'bebop': 0.5,
        'classical': 0.9,
        'fusion': 0.4
    },
    'related_parameters': [
        'harmony.chromaticism',
        'voicing.guide_tones'
    ]
}
```

## Documentation Output

### 1. Reference Documentation (Markdown)

```markdown
# harmony.tritone_substitution

## Overview

**Category:** `harmony`
**Type:** `float`
**Range:** `[0.0, 1.0]`
**Default:** `0.5`

Controls frequency of tritone substitutions in chord progressions

## Musical Context

Tritone substitution replaces dominant chords with chords a tritone away,
creating smooth voice leading and sophisticated harmonic movement.

## Usage Examples

[Python code examples...]

## Genre-Specific Values

| Genre | Typical Value | Description |
|-------|---------------|-------------|
| Bebop | `0.8` | Fast, complex harmonic movement |
| Cool Jazz | `0.5` | Subtle, relaxed harmonies |

## Related Parameters

- **`harmony.chromaticism`**: Controls chromatic movement
- **`harmony.bebop_changes`**: Enables bebop chord substitutions
```

### 2. Usage Examples (Python)

```python
"""
Usage Examples for harmony.tritone_substitution
"""

## Scenario 1: Direct Parameter Control
# ...code...

## Scenario 2: Genre Preset
# ...code...

## Scenario 3: Interactive Parameter Exploration
# ...code...

## Scenario 4: Batch Generation with Variations
# ...code...

## Scenario 5: Integration with Feature Extractor
# ...code...
```

### 3. Changelog Entry

```markdown
## [2025-11-20] Parameter Addition: `harmony.tritone_substitution`

### Added
- **`harmony.tritone_substitution`** (float): Controls frequency of tritone substitutions
  - Range: `[0.0, 1.0]`
  - Default: `0.5`
  - Category: `harmony`
  - Integrated with DeepFeatureExtractor
  - Integrated with XGBoostSynthesizer
  - Integrated with HarmonyModule
```

### 4. API Documentation (JSON)

```json
{
  "parameter": {
    "name": "harmony.tritone_substitution",
    "type": "float",
    "range": {"min": 0.0, "max": 1.0},
    "default": 0.5,
    "category": "harmony",
    "description": "...",
    "version": "1.0.0"
  },
  "usage": {
    "example_values": {...},
    "related_parameters": [...],
    "genre_applications": [...]
  },
  "validation": {...},
  "integration": {...}
}
```

### 5. Quick Reference Card

```
╔══════════════════════════════════════════════════════════════╗
║ Parameter Quick Reference                                    ║
╠══════════════════════════════════════════════════════════════╣
║ Name:     harmony.tritone_substitution                       ║
║ Category: harmony                                            ║
║ Type:     float                                              ║
║ Range:    [0.0, 1.0]                                         ║
║ Default:  0.5                                                ║
╠══════════════════════════════════════════════════════════════╣
║ Description:                                                  ║
║ Controls frequency of tritone substitutions                  ║
╠══════════════════════════════════════════════════════════════╣
║ Example Values:                                               ║
  bebop: 0.8
  cool_jazz: 0.5
╚══════════════════════════════════════════════════════════════╝
```

### 6. Integration Guide

Provides templates for:
- Adding to HarmonyModule generator
- Adding to DeepFeatureExtractor
- Training XGBoost model
- Validation logic
- Testing

## Testing

Comprehensive test suite included:

```bash
# Run all tests
cd /home/user/Do/midi_generator
python documentation/test_doc_generator.py

# Tests include:
# - Single parameter documentation
# - Batch documentation generation
# - Validation
# - Template generation
# - Genre value formatting
# - File saving
# - Related parameter detection
```

## Architecture

### Classes

#### `DocumentationGenerator`

Main class for generating documentation.

**Methods:**
- `generate_parameter_docs(proposal)`: Generate all documentation types
- `generate_batch_docs(proposals)`: Generate docs for multiple parameters
- `save_documentation(docs, param_name)`: Save docs to files
- `update_master_documentation(proposals)`: Update PARAMETERS.md

**Private Methods:**
- `_generate_reference_doc()`: Generate markdown reference
- `_generate_usage_examples()`: Generate Python examples
- `_generate_changelog_entry()`: Generate changelog
- `_generate_api_docs()`: Generate JSON API docs
- `_generate_quick_reference()`: Generate quick ref card
- `_generate_integration_guide()`: Generate integration guide
- `_enrich_proposal()`: Add missing context/metadata
- `_find_related_parameters()`: Auto-detect related params
- `_find_genre_values()`: Extract genre-specific values
- `_format_genre_values()`: Format genre table
- `_format_related_params()`: Format related params list

#### `DocumentationTemplates`

Template library for consistent formatting.

**Methods:**
- `get_reference_template()`: Markdown template
- `get_usage_template()`: Python examples template
- `get_changelog_template()`: Changelog template
- `get_quick_reference_template()`: Quick ref template
- `get_integration_template()`: Integration guide template

#### `DocumentationValidator`

Validation for documentation quality.

**Methods:**
- `validate_documentation(docs)`: Validate generated docs
- `validate_proposal(proposal)`: Validate parameter proposal

## Integration with Self-Expanding System

The Documentation Generator integrates with the broader self-expanding system:

```
┌─────────────────────────────────────────────────────────────┐
│                  Self-Expanding Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. MIDI Analysis (DeepFeatureExtractor)                     │
│         ↓                                                     │
│  2. Reconstruction Failure Detection (Gap Analysis)          │
│         ↓                                                     │
│  3. LLM Parameter Proposal (Gap Proposer)                    │
│         ↓                                                     │
│  4. DOCUMENTATION GENERATION ← YOU ARE HERE                  │
│         ↓                                                     │
│  5. Code Generation (Generator updates)                      │
│         ↓                                                     │
│  6. Feature Extraction (Add new features)                    │
│         ↓                                                     │
│  7. Model Training (XGBoost for new parameter)               │
│         ↓                                                     │
│  8. Validation & Testing                                     │
│         ↓                                                     │
│  9. System Expansion Complete ✓                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
documentation/
├── README.md                    # This file
├── doc_generator.py             # Main generator (1,500+ lines)
├── test_doc_generator.py        # Test suite (600+ lines)
└── generated/                   # Generated documentation
    ├── {param}_reference.md     # Reference docs
    ├── {param}_examples.py      # Usage examples
    ├── {param}_changelog.md     # Changelog entries
    ├── {param}_api.json         # API specs
    ├── {param}_quickref.txt     # Quick reference
    └── {param}_integration.md   # Integration guides
```

## Configuration

The generator automatically detects file paths but can be configured:

```python
doc_gen = DocumentationGenerator(
    registry_path="/path/to/registry.json",
    output_dir="/path/to/output"
)
```

## Quality Standards

The validator enforces these standards:

- **Minimum description length**: 20 characters
- **Minimum musical context**: 50 characters (if provided)
- **Minimum code examples**: 3 scenarios
- **Valid parameter naming**: `category.parameter_name` format
- **Valid JSON**: API documentation must be valid JSON
- **Complete fields**: All required fields must be present

## Genre Profiles

The system includes pre-configured genre profiles:

- **Bebop**: Fast, complex harmonic movement
- **Modal Jazz**: Emphasis on modes and scales
- **Stride Piano**: Left-hand stride accompaniment
- **Latin Jazz**: Latin rhythmic patterns
- **Cool Jazz**: Subtle, relaxed harmonies
- **Fusion**: Modern, extended harmonies
- **Gospel**: Rich, soulful harmonies
- **Contemporary**: Modern reharmonization

Genre values are used to provide context-specific parameter recommendations.

## Statistics

- **Main module**: 1,381 lines
- **Test suite**: 600+ lines
- **Total system**: 1,981+ lines
- **Documentation types**: 6
- **Test cases**: 7 comprehensive tests
- **Template types**: 5 specialized templates

## Future Enhancements

Planned improvements:

1. **HTML Documentation**: Generate styled HTML docs
2. **Interactive Docs**: Web-based interactive parameter explorer
3. **Audio Examples**: Generate audio examples for each parameter
4. **Version Comparison**: Compare parameter versions across releases
5. **Search Index**: Full-text search across all parameter docs
6. **Localization**: Multi-language documentation support
7. **Diff Generation**: Document parameter changes between versions

## Support

For issues or questions:

1. Check existing documentation in `/documentation/generated/`
2. Run test suite: `python documentation/test_doc_generator.py`
3. Review parameter registry: `/parameters/registry.json`
4. Check main PARAMETERS.md: `/parameters/PARAMETERS.md`

## License

Part of the Do music generation system.

---

**Generated by Agent 27: Documentation Generator**
*Part of the 35-agent self-expanding inverse music generation system*
