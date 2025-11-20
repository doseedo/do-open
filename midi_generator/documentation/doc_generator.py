"""
Documentation Generator for Musical Parameter System
=====================================================

Auto-generates comprehensive documentation for new parameters including:
- Parameter reference documentation (markdown)
- Musical context and examples
- Usage examples (Python code)
- Genre-specific value mappings
- Related parameters
- Changelog updates
- API documentation

Part of the self-expanding inverse music generation system.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re
from collections import defaultdict


class DocumentationGenerator:
    """
    Generate comprehensive documentation for musical parameters.

    Supports multiple documentation formats:
    - Markdown reference documentation
    - Python usage examples
    - Changelog entries
    - API documentation (JSON, HTML)
    - Integration guides
    """

    def __init__(self, registry_path: str = None, output_dir: str = None):
        """
        Initialize DocumentationGenerator.

        Args:
            registry_path: Path to parameter registry JSON
            output_dir: Directory for generated documentation
        """
        self.registry_path = registry_path or self._find_registry()
        self.output_dir = output_dir or self._find_output_dir()

        # Load parameter registry
        self.registry = self._load_registry()

        # Documentation templates
        self.templates = DocumentationTemplates()

        # Genre mappings (populated from genre profiles)
        self.genre_profiles = self._load_genre_profiles()

        # Related parameter graph
        self.param_graph = self._build_parameter_graph()

    def _find_registry(self) -> str:
        """Find parameter registry file"""
        possible_paths = [
            "/home/user/Do/midi_generator/parameters/registry.json",
            "../parameters/registry.json",
            "parameters/registry.json"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return "parameters/registry.json"

    def _find_output_dir(self) -> str:
        """Find output directory for documentation"""
        base_dir = "/home/user/Do/midi_generator"

        # Create documentation directory if it doesn't exist
        doc_dir = os.path.join(base_dir, "documentation", "generated")
        os.makedirs(doc_dir, exist_ok=True)

        return doc_dir

    def _load_registry(self) -> Dict:
        """Load parameter registry"""
        if not os.path.exists(self.registry_path):
            return {"parameters": {}}

        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load registry: {e}")
            return {"parameters": {}}

    def _load_genre_profiles(self) -> Dict:
        """Load genre profiles to extract parameter usage patterns"""
        genre_dir = "/home/user/Do/midi_generator/genres"
        profiles = {}

        if not os.path.exists(genre_dir):
            return profiles

        # Map common genres to their parameter preferences
        # This would ideally parse actual genre profile files
        profiles = {
            'bebop': {
                'harmony.bebop_changes': 0.8,
                'harmony.chromaticism': 0.7,
                'rhythm.swing_ratio': 0.67,
                'melody.chromatic_approach': 0.8
            },
            'modal_jazz': {
                'harmony.modal_harmony': 0.9,
                'harmony.sus_chords': 0.6,
                'melody.modal_interchange': 0.7,
                'rhythm.polyrhythm': 0.5
            },
            'stride_piano': {
                'rhythm.stride_pattern': 0.9,
                'harmony.rootless_voicings': 0.4,
                'voicing.tenths': 0.7,
                'rhythm.swing_ratio': 0.60
            },
            'latin_jazz': {
                'rhythm.clave_pattern': 0.8,
                'rhythm.polyrhythm': 0.7,
                'harmony.montuno_pattern': 0.6,
                'percussion.congas': 0.8
            },
            'cool_jazz': {
                'harmony.quartal_harmony': 0.7,
                'voicing.drop2': 0.8,
                'melody.long_lines': 0.6,
                'dynamics.subtle_variation': 0.7
            },
            'fusion': {
                'harmony.extended_chords': 0.9,
                'rhythm.odd_meter': 0.6,
                'melody.synth_lead': 0.7,
                'harmony.upper_structures': 0.8
            },
            'gospel': {
                'harmony.gospel_turnaround': 0.9,
                'voicing.gospel_voicing': 0.8,
                'rhythm.gospel_feel': 0.7,
                'harmony.secondary_dominants': 0.8
            },
            'contemporary': {
                'harmony.reharmonization': 0.7,
                'voicing.cluster_voicing': 0.6,
                'rhythm.metric_modulation': 0.5,
                'harmony.polychords': 0.6
            }
        }

        return profiles

    def _build_parameter_graph(self) -> Dict[str, List[str]]:
        """
        Build graph of related parameters based on category and musical function.

        Returns:
            Dict mapping parameter names to lists of related parameters
        """
        graph = defaultdict(list)

        if 'parameters' not in self.registry:
            return dict(graph)

        params = self.registry['parameters']

        # Group by category
        categories = defaultdict(list)
        for param_name, param_info in params.items():
            category = param_info.get('category', 'general')
            categories[category].append(param_name)

        # Parameters in same category are related
        for param_name, param_info in params.items():
            category = param_info.get('category', 'general')
            related = [p for p in categories[category] if p != param_name]
            graph[param_name].extend(related)

        # Add specific musical relationships
        musical_relationships = {
            'harmony.bebop_changes': ['melody.chromatic_approach', 'harmony.chromaticism'],
            'harmony.modal_harmony': ['melody.modal_interchange', 'harmony.sus_chords'],
            'rhythm.swing_ratio': ['rhythm.humanize', 'articulation.accent_pattern'],
            'voicing.drop2': ['voicing.spread', 'harmony.rootless_voicings'],
            'harmony.upper_structures': ['harmony.extended_chords', 'voicing.tension_voicing']
        }

        for param, related in musical_relationships.items():
            if param in graph:
                graph[param].extend([r for r in related if r not in graph[param]])

        return dict(graph)

    def generate_parameter_docs(self, param_proposal: Dict) -> Dict[str, str]:
        """
        Generate all documentation for a parameter.

        Args:
            param_proposal: Parameter proposal dict with keys:
                - name: Parameter name
                - type: Data type
                - range: Valid range
                - default: Default value
                - description: Short description
                - musical_context: Musical explanation
                - example_values: Dict of example values
                - related_parameters: List of related params
                - category: Parameter category

        Returns:
            Dict with keys:
                - reference_doc: Markdown reference documentation
                - usage_examples: Python code examples
                - changelog_entry: CHANGELOG.md entry
                - api_docs: API documentation (JSON)
                - quick_reference: Quick reference card
                - integration_guide: Integration guide
        """
        # Validate proposal
        self._validate_proposal(param_proposal)

        # Enrich proposal with additional context
        enriched_proposal = self._enrich_proposal(param_proposal)

        # Generate all documentation types
        reference = self._generate_reference_doc(enriched_proposal)
        examples = self._generate_usage_examples(enriched_proposal)
        changelog = self._generate_changelog_entry(enriched_proposal)
        api_docs = self._generate_api_docs(enriched_proposal)
        quick_ref = self._generate_quick_reference(enriched_proposal)
        integration = self._generate_integration_guide(enriched_proposal)

        return {
            'reference_doc': reference,
            'usage_examples': examples,
            'changelog_entry': changelog,
            'api_docs': api_docs,
            'quick_reference': quick_ref,
            'integration_guide': integration
        }

    def _validate_proposal(self, proposal: Dict) -> None:
        """Validate parameter proposal has required fields"""
        required_fields = ['name', 'type', 'range', 'default', 'description']

        for field in required_fields:
            if field not in proposal:
                raise ValueError(f"Missing required field: {field}")

        # Validate parameter name format
        if not re.match(r'^[a-z_]+\.[a-z_]+$', proposal['name']):
            raise ValueError(
                f"Invalid parameter name format: {proposal['name']}. "
                "Should be 'category.parameter_name'"
            )

    def _enrich_proposal(self, proposal: Dict) -> Dict:
        """Enrich proposal with additional context and metadata"""
        enriched = proposal.copy()

        # Add timestamp
        enriched['generated_at'] = datetime.now().isoformat()

        # Add version info
        enriched['version'] = '1.0.0'

        # Find related parameters if not provided
        if 'related_parameters' not in enriched:
            enriched['related_parameters'] = self._find_related_parameters(
                enriched['name']
            )

        # Add genre-specific values if not provided
        if 'example_values' not in enriched:
            enriched['example_values'] = self._find_genre_values(enriched['name'])

        # Extract category from name
        if 'category' not in enriched:
            enriched['category'] = enriched['name'].split('.')[0]

        # Add musical context if missing
        if 'musical_context' not in enriched:
            enriched['musical_context'] = self._generate_musical_context(enriched)

        return enriched

    def _find_related_parameters(self, param_name: str) -> List[str]:
        """Find related parameters using the parameter graph"""
        if param_name in self.param_graph:
            return self.param_graph[param_name][:5]  # Top 5 related

        # Fallback: find by same category
        category = param_name.split('.')[0]
        related = []

        if 'parameters' in self.registry:
            for name, info in self.registry['parameters'].items():
                if name != param_name and name.startswith(category + '.'):
                    related.append(name)
                if len(related) >= 5:
                    break

        return related

    def _find_genre_values(self, param_name: str) -> Dict[str, float]:
        """Find typical parameter values across genres"""
        genre_values = {}

        for genre, params in self.genre_profiles.items():
            if param_name in params:
                genre_values[genre] = params[param_name]

        return genre_values

    def _generate_musical_context(self, proposal: Dict) -> str:
        """Generate musical context explanation"""
        category = proposal['category']

        context_templates = {
            'harmony': 'This parameter controls harmonic content and chord selection. '
                      'It influences the sophistication and color of the harmonic palette.',
            'melody': 'This parameter shapes melodic contour and note selection. '
                     'It affects the character and movement of melodic lines.',
            'rhythm': 'This parameter governs rhythmic patterns and timing. '
                     'It determines the groove and feel of the generated music.',
            'voicing': 'This parameter controls chord voicing and spacing. '
                      'It affects the texture and register distribution of harmonies.',
            'dynamics': 'This parameter manages volume and intensity variation. '
                       'It shapes the expressive dynamic contour of the music.',
            'articulation': 'This parameter determines note articulation and phrasing. '
                          'It controls the attack, sustain, and connection between notes.'
        }

        base_context = context_templates.get(
            category,
            'This parameter influences the musical generation process.'
        )

        return f"{base_context} {proposal.get('description', '')}"

    def _generate_reference_doc(self, proposal: Dict) -> str:
        """Generate comprehensive markdown reference documentation"""
        template = self.templates.get_reference_template()

        # Format related parameters
        related_params_md = self._format_related_params(
            proposal.get('related_parameters', [])
        )

        # Format genre values
        genre_values_md = self._format_genre_values(
            proposal.get('example_values', {})
        )

        # Format range information
        range_info = self._format_range_info(proposal)

        # Generate code examples
        code_examples = self._generate_code_examples(proposal)

        # Format the template
        doc = template.format(
            name=proposal['name'],
            type=proposal['type'],
            range=range_info,
            default=proposal['default'],
            description=proposal['description'],
            musical_context=proposal.get('musical_context', ''),
            code_examples=code_examples,
            genre_values=genre_values_md,
            related_parameters=related_params_md,
            category=proposal['category'],
            version=proposal.get('version', '1.0.0'),
            generated_at=proposal.get('generated_at', datetime.now().isoformat())
        )

        return doc

    def _format_range_info(self, proposal: Dict) -> str:
        """Format range information with validation rules"""
        range_val = proposal['range']
        type_val = proposal['type']

        if isinstance(range_val, dict):
            if 'min' in range_val and 'max' in range_val:
                return f"`[{range_val['min']}, {range_val['max']}]`"
            elif 'values' in range_val:
                return f"`{range_val['values']}`"
        elif isinstance(range_val, str):
            return f"`{range_val}`"

        return f"`{range_val}`"

    def _generate_code_examples(self, proposal: Dict) -> str:
        """Generate comprehensive code examples"""
        param_name = proposal['name']
        default_val = proposal['default']

        examples = []

        # Example 1: Basic usage
        examples.append(f'''
# Example 1: Basic Usage
from core.HarmonyModule import HarmonyModule

generator = HarmonyModule()

params = {{
    "{param_name}": {default_val},
    "general.tempo": 120,
    "general.key": "C"
}}

midi = generator.generate(params)
midi.write('output.mid')
''')

        # Example 2: Genre-specific usage
        genre_values = proposal.get('example_values', {})
        if genre_values:
            first_genre = list(genre_values.keys())[0]
            first_value = genre_values[first_genre]

            examples.append(f'''
# Example 2: {first_genre.title()} Style
params = {{
    "{param_name}": {first_value},
    "general.genre": "{first_genre}",
    "general.complexity": 0.7
}}

midi = generator.generate(params)
''')

        # Example 3: Parameter sweep
        examples.append(f'''
# Example 3: Parameter Sweep (exploring range)
import numpy as np

for value in np.linspace(0.0, 1.0, 5):
    params = {{
        "{param_name}": float(value),
        "general.variation_seed": int(value * 1000)
    }}

    midi = generator.generate(params)
    midi.write(f'output_{{value:.2f}}.mid')
''')

        # Example 4: Integration with other parameters
        related = proposal.get('related_parameters', [])
        if related:
            examples.append(f'''
# Example 4: Combined with Related Parameters
params = {{
    "{param_name}": {default_val},
    "{related[0]}": 0.6,  # Related parameter
    "general.bars": 32
}}

midi = generator.generate(params)
''')

        return '\n'.join(examples)

    def _format_genre_values(self, genre_values: Dict[str, float]) -> str:
        """Format genre-specific values as markdown table"""
        if not genre_values:
            return "*No genre-specific examples available yet.*"

        lines = [
            "| Genre | Typical Value | Description |",
            "|-------|---------------|-------------|"
        ]

        descriptions = {
            'bebop': 'Fast, complex harmonic movement',
            'modal_jazz': 'Emphasis on modes and scales',
            'stride_piano': 'Left-hand stride accompaniment',
            'latin_jazz': 'Latin rhythmic patterns',
            'cool_jazz': 'Subtle, relaxed harmonies',
            'fusion': 'Modern, extended harmonies',
            'gospel': 'Rich, soulful harmonies',
            'contemporary': 'Modern reharmonization'
        }

        for genre, value in sorted(genre_values.items()):
            desc = descriptions.get(genre, 'Genre-specific setting')
            lines.append(f"| {genre.replace('_', ' ').title()} | `{value}` | {desc} |")

        return '\n'.join(lines)

    def _format_related_params(self, related_params: List[str]) -> str:
        """Format related parameters as markdown list with descriptions"""
        if not related_params:
            return "*No directly related parameters identified.*"

        lines = []

        for param in related_params:
            # Try to get description from registry
            desc = "Related parameter"

            if 'parameters' in self.registry:
                if param in self.registry['parameters']:
                    desc = self.registry['parameters'][param].get(
                        'description',
                        'Related parameter'
                    )

            lines.append(f"- **`{param}`**: {desc}")

        return '\n'.join(lines)

    def _generate_usage_examples(self, proposal: Dict) -> str:
        """Generate comprehensive Python usage examples"""
        template = self.templates.get_usage_template()

        param_name = proposal['name']
        default_val = proposal['default']

        # Generate different usage scenarios
        scenarios = []

        # Scenario 1: Direct parameter setting
        scenarios.append(self._generate_direct_usage(proposal))

        # Scenario 2: Preset-based usage
        scenarios.append(self._generate_preset_usage(proposal))

        # Scenario 3: Interactive exploration
        scenarios.append(self._generate_interactive_usage(proposal))

        # Scenario 4: Batch generation
        scenarios.append(self._generate_batch_usage(proposal))

        # Scenario 5: Integration with feature extractor
        scenarios.append(self._generate_extraction_usage(proposal))

        doc = template.format(
            param_name=param_name,
            scenarios='\n\n'.join(scenarios)
        )

        return doc

    def _generate_direct_usage(self, proposal: Dict) -> str:
        """Generate direct usage example"""
        return f'''
## Scenario 1: Direct Parameter Control

```python
from core.HarmonyModule import HarmonyModule

# Initialize generator
generator = HarmonyModule()

# Set parameter directly
params = {{
    "{proposal['name']}": {proposal['default']},
    "general.tempo": 120,
    "general.key": "C",
    "general.bars": 16
}}

# Generate MIDI
midi = generator.generate(params)
midi.write('output.mid')

print(f"Generated MIDI with {proposal['name']}={proposal['default']}")
```
'''

    def _generate_preset_usage(self, proposal: Dict) -> str:
        """Generate preset-based usage example"""
        genre_values = proposal.get('example_values', {})

        if genre_values:
            first_genre = list(genre_values.keys())[0]
            first_value = genre_values[first_genre]
        else:
            first_genre = 'bebop'
            first_value = 0.7

        return f'''
## Scenario 2: Genre Preset

```python
from core.HarmonyModule import HarmonyModule
from parameters.universal_registry import UniversalParameterRegistry

# Load registry
registry = UniversalParameterRegistry()

# Get genre preset
preset = registry.get_genre_preset('{first_genre}')

# Override specific parameter
preset["{proposal['name']}"] = {first_value}

# Generate
generator = HarmonyModule()
midi = generator.generate(preset)
midi.write('{first_genre}_output.mid')
```
'''

    def _generate_interactive_usage(self, proposal: Dict) -> str:
        """Generate interactive exploration example"""
        return f'''
## Scenario 3: Interactive Parameter Exploration

```python
from core.HarmonyModule import HarmonyModule
import numpy as np

generator = HarmonyModule()

# Explore parameter range
values = np.linspace(0.0, 1.0, 10)

for i, val in enumerate(values):
    params = {{
        "{proposal['name']}": float(val),
        "general.variation_seed": i,
        "general.bars": 8
    }}

    midi = generator.generate(params)
    midi.write(f'exploration_{proposal["name"].replace(".", "_")}_{{i:02d}}.mid')

    print(f"Generated variation {{i+1}}/10: {proposal['name']}={{val:.2f}}")
```
'''

    def _generate_batch_usage(self, proposal: Dict) -> str:
        """Generate batch generation example"""
        return f'''
## Scenario 4: Batch Generation with Variations

```python
from core.HarmonyModule import HarmonyModule
from concurrent.futures import ProcessPoolExecutor
import itertools

def generate_variation(args):
    param_value, seed = args

    generator = HarmonyModule()
    params = {{
        "{proposal['name']}": param_value,
        "general.variation_seed": seed,
        "general.bars": 16
    }}

    midi = generator.generate(params)
    filename = f'batch_{{seed}}_{proposal["name"].replace(".", "_")}_{{param_value:.2f}}.mid'
    midi.write(filename)

    return filename

# Generate 20 variations
param_values = [0.3, 0.5, 0.7, 0.9]
seeds = range(5)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(
        generate_variation,
        itertools.product(param_values, seeds)
    )

print(f"Generated {{len(list(results))}} variations")
```
'''

    def _generate_extraction_usage(self, proposal: Dict) -> str:
        """Generate feature extraction integration example"""
        return f'''
## Scenario 5: Integration with Feature Extractor

```python
from features.deep_feature_extractor import DeepFeatureExtractor
from synthesizers.xgboost_synthesizer import XGBoostSynthesizer
from core.HarmonyModule import HarmonyModule

# Extract features from reference MIDI
extractor = DeepFeatureExtractor()
features = extractor.extract('reference.mid')

# Predict parameter value
synthesizer = XGBoostSynthesizer()
predicted_params = synthesizer.predict(features)

print(f"Predicted {proposal['name']}: {{predicted_params['{proposal['name']}']}}")

# Generate new MIDI with predicted parameters
generator = HarmonyModule()
midi = generator.generate(predicted_params)
midi.write('reconstructed.mid')

# Compare original parameter to prediction
if '{proposal['name']}' in predicted_params:
    print(f"Parameter value: {{predicted_params['{proposal['name']}']}}")
```
'''

    def _generate_changelog_entry(self, proposal: Dict) -> str:
        """Generate CHANGELOG.md entry"""
        template = self.templates.get_changelog_template()

        today = datetime.now().strftime('%Y-%m-%d')

        entry = template.format(
            date=today,
            param_name=proposal['name'],
            description=proposal['description'],
            type=proposal['type'],
            range=self._format_range_info(proposal),
            default=proposal['default'],
            category=proposal['category']
        )

        return entry

    def _generate_api_docs(self, proposal: Dict) -> str:
        """Generate API documentation in JSON format"""
        api_doc = {
            "parameter": {
                "name": proposal['name'],
                "type": proposal['type'],
                "range": proposal['range'],
                "default": proposal['default'],
                "category": proposal['category'],
                "description": proposal['description'],
                "musical_context": proposal.get('musical_context', ''),
                "version": proposal.get('version', '1.0.0'),
                "generated_at": proposal.get('generated_at', datetime.now().isoformat())
            },
            "usage": {
                "example_values": proposal.get('example_values', {}),
                "related_parameters": proposal.get('related_parameters', []),
                "genre_applications": list(proposal.get('example_values', {}).keys())
            },
            "validation": {
                "required": False,
                "validates_range": True,
                "type_checking": True
            },
            "integration": {
                "generator": "HarmonyModule",
                "feature_extraction": True,
                "xgboost_model": True,
                "registry": "UniversalParameterRegistry"
            }
        }

        return json.dumps(api_doc, indent=2)

    def _generate_quick_reference(self, proposal: Dict) -> str:
        """Generate quick reference card"""
        template = self.templates.get_quick_reference_template()

        genre_values = proposal.get('example_values', {})
        top_genres = list(genre_values.items())[:3] if genre_values else []

        genre_examples = '\n'.join([
            f"  {genre}: {value}"
            for genre, value in top_genres
        ]) if top_genres else "  (See full documentation)"

        quick_ref = template.format(
            param_name=proposal['name'],
            type=proposal['type'],
            range=self._format_range_info(proposal),
            default=proposal['default'],
            description=proposal['description'],
            category=proposal['category'],
            genre_examples=genre_examples
        )

        return quick_ref

    def _generate_integration_guide(self, proposal: Dict) -> str:
        """Generate integration guide for developers"""
        template = self.templates.get_integration_template()

        guide = template.format(
            param_name=proposal['name'],
            category=proposal['category'],
            type=proposal['type'],
            description=proposal['description']
        )

        return guide

    def generate_batch_docs(self, param_proposals: List[Dict]) -> Dict[str, Any]:
        """
        Generate documentation for multiple parameters.

        Args:
            param_proposals: List of parameter proposal dicts

        Returns:
            Dict with aggregated documentation
        """
        all_docs = []
        changelog_entries = []
        api_docs = []

        for proposal in param_proposals:
            docs = self.generate_parameter_docs(proposal)
            all_docs.append(docs)
            changelog_entries.append(docs['changelog_entry'])
            api_docs.append(json.loads(docs['api_docs']))

        # Generate index
        index = self._generate_index(param_proposals)

        # Generate category summaries
        category_summaries = self._generate_category_summaries(param_proposals)

        return {
            'individual_docs': all_docs,
            'combined_changelog': '\n\n'.join(changelog_entries),
            'combined_api_docs': json.dumps(api_docs, indent=2),
            'index': index,
            'category_summaries': category_summaries,
            'total_parameters': len(param_proposals)
        }

    def _generate_index(self, proposals: List[Dict]) -> str:
        """Generate documentation index"""
        lines = [
            "# Parameter Documentation Index",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Parameters: {len(proposals)}",
            "",
            "## Parameters by Category",
            ""
        ]

        # Group by category
        by_category = defaultdict(list)
        for proposal in proposals:
            category = proposal.get('category', 'general')
            by_category[category].append(proposal)

        # Generate category sections
        for category in sorted(by_category.keys()):
            lines.append(f"### {category.title()}")
            lines.append("")

            for proposal in sorted(by_category[category], key=lambda p: p['name']):
                lines.append(
                    f"- **`{proposal['name']}`**: {proposal['description']}"
                )

            lines.append("")

        return '\n'.join(lines)

    def _generate_category_summaries(self, proposals: List[Dict]) -> Dict[str, str]:
        """Generate summary documentation for each category"""
        summaries = {}

        # Group by category
        by_category = defaultdict(list)
        for proposal in proposals:
            category = proposal.get('category', 'general')
            by_category[category].append(proposal)

        # Generate summary for each category
        for category, params in by_category.items():
            summary_lines = [
                f"# {category.title()} Parameters",
                "",
                f"This category contains {len(params)} parameters controlling "
                f"{category}-related aspects of music generation.",
                "",
                "## Parameters",
                ""
            ]

            for param in sorted(params, key=lambda p: p['name']):
                summary_lines.extend([
                    f"### `{param['name']}`",
                    "",
                    param['description'],
                    "",
                    f"- **Type**: `{param['type']}`",
                    f"- **Range**: {self._format_range_info(param)}",
                    f"- **Default**: `{param['default']}`",
                    ""
                ])

            summaries[category] = '\n'.join(summary_lines)

        return summaries

    def save_documentation(
        self,
        docs: Dict[str, str],
        param_name: str,
        output_dir: str = None
    ) -> Dict[str, str]:
        """
        Save generated documentation to files.

        Args:
            docs: Documentation dict from generate_parameter_docs()
            param_name: Parameter name
            output_dir: Output directory (uses self.output_dir if None)

        Returns:
            Dict mapping doc type to file path
        """
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Sanitize parameter name for filename
        safe_name = param_name.replace('.', '_')

        saved_files = {}

        # Save reference documentation
        ref_path = os.path.join(output_dir, f"{safe_name}_reference.md")
        with open(ref_path, 'w') as f:
            f.write(docs['reference_doc'])
        saved_files['reference'] = ref_path

        # Save usage examples
        usage_path = os.path.join(output_dir, f"{safe_name}_examples.py")
        with open(usage_path, 'w') as f:
            f.write(docs['usage_examples'])
        saved_files['examples'] = usage_path

        # Save changelog entry
        changelog_path = os.path.join(output_dir, f"{safe_name}_changelog.md")
        with open(changelog_path, 'w') as f:
            f.write(docs['changelog_entry'])
        saved_files['changelog'] = changelog_path

        # Save API docs
        api_path = os.path.join(output_dir, f"{safe_name}_api.json")
        with open(api_path, 'w') as f:
            f.write(docs['api_docs'])
        saved_files['api'] = api_path

        # Save quick reference
        quick_ref_path = os.path.join(output_dir, f"{safe_name}_quickref.txt")
        with open(quick_ref_path, 'w') as f:
            f.write(docs['quick_reference'])
        saved_files['quick_reference'] = quick_ref_path

        # Save integration guide
        integration_path = os.path.join(output_dir, f"{safe_name}_integration.md")
        with open(integration_path, 'w') as f:
            f.write(docs['integration_guide'])
        saved_files['integration'] = integration_path

        return saved_files

    def update_master_documentation(
        self,
        param_proposals: List[Dict],
        master_doc_path: str = None
    ) -> str:
        """
        Update master PARAMETERS.md file with new parameters.

        Args:
            param_proposals: List of parameter proposals
            master_doc_path: Path to master PARAMETERS.md

        Returns:
            Path to updated master documentation
        """
        if master_doc_path is None:
            master_doc_path = "/home/user/Do/midi_generator/parameters/PARAMETERS.md"

        # Read existing documentation
        if os.path.exists(master_doc_path):
            with open(master_doc_path, 'r') as f:
                existing_content = f.read()
        else:
            existing_content = "# Musical Parameter Reference\n\n"

        # Generate new sections
        new_sections = []

        for proposal in param_proposals:
            docs = self.generate_parameter_docs(proposal)
            new_sections.append(docs['reference_doc'])

        # Append to existing content
        updated_content = existing_content + "\n\n---\n\n" + "\n\n---\n\n".join(new_sections)

        # Write updated documentation
        with open(master_doc_path, 'w') as f:
            f.write(updated_content)

        return master_doc_path


class DocumentationTemplates:
    """
    Template library for documentation generation.

    Provides consistent formatting across all documentation types.
    """

    def get_reference_template(self) -> str:
        """Get markdown reference documentation template"""
        return '''# {name}

## Overview

**Category:** `{category}`
**Type:** `{type}`
**Range:** {range}
**Default:** `{default}`
**Version:** `{version}`

{description}

## Musical Context

{musical_context}

## Usage Examples

```python
{code_examples}
```

## Genre-Specific Values

{genre_values}

## Related Parameters

{related_parameters}

## Integration

This parameter integrates with:
- **HarmonyModule**: Core generation engine
- **DeepFeatureExtractor**: Feature extraction for inverse synthesis
- **XGBoostSynthesizer**: Parameter prediction from MIDI
- **UniversalParameterRegistry**: Parameter validation and presets

## Validation

The parameter value is validated against the specified range. Invalid values will raise a `ValueError`.

## See Also

- [Parameter Guide](../PARAMETERS.md)
- [Musical Concepts](../docs/MUSICAL_CONCEPTS.md)
- [Category Documentation]({category}_parameters.md)

---

*Generated: {generated_at}*
'''

    def get_usage_template(self) -> str:
        """Get usage examples template"""
        return '''"""
Usage Examples for {param_name}
================================

Comprehensive examples demonstrating different use cases.
"""

{scenarios}

# Additional Resources
# --------------------
# - Parameter Reference: documentation/generated/{param_name}_reference.md
# - API Documentation: documentation/generated/{param_name}_api.json
# - Integration Guide: documentation/generated/{param_name}_integration.md
'''

    def get_changelog_template(self) -> str:
        """Get changelog entry template"""
        return '''## [{date}] Parameter Addition: `{param_name}`

### Added
- **`{param_name}`** ({type}): {description}
  - Range: {range}
  - Default: `{default}`
  - Category: `{category}`
  - Enables new musical capabilities in parameter prediction and generation
  - Integrated with DeepFeatureExtractor (feature extraction)
  - Integrated with XGBoostSynthesizer (parameter prediction)
  - Integrated with HarmonyModule (MIDI generation)

### Documentation
- Added comprehensive reference documentation
- Added usage examples with 5 scenarios
- Added API documentation (JSON)
- Added integration guide for developers
'''

    def get_quick_reference_template(self) -> str:
        """Get quick reference card template"""
        return '''╔══════════════════════════════════════════════════════════════╗
║ Parameter Quick Reference                                    ║
╠══════════════════════════════════════════════════════════════╣
║ Name:     {param_name:<50} ║
║ Category: {category:<50} ║
║ Type:     {type:<50} ║
║ Range:    {range:<50} ║
║ Default:  {default:<50} ║
╠══════════════════════════════════════════════════════════════╣
║ Description:                                                  ║
║ {description:<58} ║
╠══════════════════════════════════════════════════════════════╣
║ Example Values:                                               ║
{genre_examples}
╚══════════════════════════════════════════════════════════════╝
'''

    def get_integration_template(self) -> str:
        """Get integration guide template"""
        return '''# Integration Guide: `{param_name}`

## Adding to HarmonyModule Generator

```python
# In core/HarmonyModule.py or relevant engine

class HarmonyEngine:
    def apply_{param_name}_logic(self, value: {type}) -> None:
        """
        Apply {param_name} parameter to generation.

        Args:
            value: Parameter value ({type})
        """
        # TODO: Implement parameter logic
        # This parameter controls: {description}

        pass
```

## Adding to DeepFeatureExtractor

```python
# In features/deep_feature_extractor.py

class DeepFeatureExtractor:
    def extract_{param_name}_features(self, midi_data) -> List[float]:
        """
        Extract features related to {param_name}.

        Returns:
            List of feature values
        """
        features = []

        # TODO: Implement feature extraction
        # Extract features that correlate with {description}

        return features
```

## Adding XGBoost Model

```python
# In synthesizers/xgboost_synthesizer.py

class XGBoostSynthesizer:
    def train_{param_name}_model(self, X_train, y_train):
        """
        Train XGBoost model for {param_name} prediction.

        Args:
            X_train: Feature matrix
            y_train: Target values for {param_name}
        """
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1
        )

        model.fit(X_train, y_train)

        # Save model
        self.models['{param_name}'] = model
```

## Validation

```python
# In parameters/universal_registry.py

def validate_{param_name}(value: {type}) -> bool:
    """Validate {param_name} parameter value"""
    # TODO: Implement validation logic
    return True
```

## Testing

```python
# In tests/test_{category}_parameters.py

def test_{param_name}():
    """Test {param_name} parameter"""
    generator = HarmonyModule()

    params = {{
        '{param_name}': <test_value>,
        'general.bars': 8
    }}

    midi = generator.generate(params)

    assert midi is not None
    # TODO: Add specific assertions
```
'''


class DocumentationValidator:
    """
    Validate generated documentation for quality and completeness.
    """

    def __init__(self):
        self.min_description_length = 20
        self.min_musical_context_length = 50
        self.min_code_examples = 3

    def validate_documentation(self, docs: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate generated documentation.

        Args:
            docs: Documentation dict from generate_parameter_docs()

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check reference doc
        if len(docs.get('reference_doc', '')) < 200:
            issues.append("Reference documentation too short")

        # Check usage examples
        usage = docs.get('usage_examples', '')
        if usage.count('```python') < self.min_code_examples:
            issues.append(
                f"Insufficient code examples (minimum {self.min_code_examples})"
            )

        # Check changelog
        if 'Parameter Addition' not in docs.get('changelog_entry', ''):
            issues.append("Invalid changelog format")

        # Check API docs
        try:
            api = json.loads(docs.get('api_docs', '{}'))
            if 'parameter' not in api:
                issues.append("Missing parameter section in API docs")
        except json.JSONDecodeError:
            issues.append("Invalid JSON in API docs")

        # Check quick reference
        if '╔' not in docs.get('quick_reference', ''):
            issues.append("Quick reference formatting issues")

        return len(issues) == 0, issues

    def validate_proposal(self, proposal: Dict) -> Tuple[bool, List[str]]:
        """
        Validate parameter proposal before documentation generation.

        Args:
            proposal: Parameter proposal dict

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check required fields
        required = ['name', 'type', 'range', 'default', 'description']
        for field in required:
            if field not in proposal:
                issues.append(f"Missing required field: {field}")

        # Check description length
        if len(proposal.get('description', '')) < self.min_description_length:
            issues.append("Description too short")

        # Check musical context
        if 'musical_context' in proposal:
            if len(proposal['musical_context']) < self.min_musical_context_length:
                issues.append("Musical context too short")

        # Check parameter name format
        if 'name' in proposal:
            if not re.match(r'^[a-z_]+\.[a-z_]+$', proposal['name']):
                issues.append("Invalid parameter name format (should be category.name)")

        return len(issues) == 0, issues


# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    doc_gen = DocumentationGenerator()

    # Example parameter proposal
    example_proposal = {
        'name': 'harmony.tritone_substitution',
        'type': 'float',
        'range': {'min': 0.0, 'max': 1.0},
        'default': 0.5,
        'description': 'Controls frequency of tritone substitutions in chord progressions',
        'musical_context': 'Tritone substitution replaces dominant chords with chords a tritone away, '
                          'creating smooth voice leading and sophisticated harmonic movement. '
                          'Common in bebop and modern jazz.',
        'category': 'harmony',
        'example_values': {
            'bebop': 0.8,
            'cool_jazz': 0.5,
            'fusion': 0.7,
            'contemporary': 0.6
        },
        'related_parameters': [
            'harmony.chromaticism',
            'harmony.bebop_changes',
            'voicing.guide_tones'
        ]
    }

    # Generate documentation
    print("Generating documentation for:", example_proposal['name'])
    docs = doc_gen.generate_parameter_docs(example_proposal)

    # Validate documentation
    validator = DocumentationValidator()
    is_valid, issues = validator.validate_documentation(docs)

    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")

    # Save documentation
    print("\nSaving documentation...")
    saved_files = doc_gen.save_documentation(
        docs,
        example_proposal['name']
    )

    print("\nGenerated files:")
    for doc_type, path in saved_files.items():
        print(f"  {doc_type}: {path}")

    print("\n✓ Documentation generation complete!")
    newline = '\n'
    print(f"  Total lines: {sum(len(doc.split(newline)) for doc in docs.values())}")
