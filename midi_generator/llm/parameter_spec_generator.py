"""
Agent 12: Parameter Specification Generator (REFACTORED)

OLD: Generated Python code (DANGEROUS!)
NEW: Generates validated .params YAML specifications (SAFE!)

This agent uses LLMs to propose new parameter specifications in a declarative
format, not executable code. All specs are validated before deployment.

Author: Musical Program Synthesis Team
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import yaml
import os

try:
    from anthropic import Anthropic
except ImportError:
    print("Warning: Anthropic SDK not installed. LLM features disabled.")
    Anthropic = None


class ParameterSpecificationGenerator:
    """
    Generates parameter specifications in .params format using LLM.

    NO CODE GENERATION - only validated YAML specifications.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize generator with optional API key."""
        if Anthropic is None:
            self.client = None
            print("Warning: LLM features disabled (anthropic not installed)")
        else:
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key) if api_key else None

    def generate_parameter_spec(
        self,
        gap_analysis: Dict[str, Any],
        domain: str = "harmony"
    ) -> Dict[str, Any]:
        """
        Generate a parameter specification from gap analysis.

        Args:
            gap_analysis: Analysis of missing capability
            domain: Musical domain (harmony, melody, rhythm, etc.)

        Returns:
            Parameter specification as dict (ready to save as YAML)
        """
        if not self.client:
            # Fallback: return template spec
            return self._generate_template_spec(gap_analysis, domain)

        prompt = self._create_specification_prompt(gap_analysis, domain)

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse YAML response
            yaml_text = response.content[0].text

            # Extract YAML from markdown code blocks if present
            if "```yaml" in yaml_text:
                yaml_text = yaml_text.split("```yaml")[1].split("```")[0].strip()
            elif "```" in yaml_text:
                yaml_text = yaml_text.split("```")[1].split("```")[0].strip()

            spec = yaml.safe_load(yaml_text)

            # Add metadata
            spec['created_by'] = 'llm_agent'
            spec['created_at'] = datetime.now().isoformat()
            spec['version'] = '1.0.0'

            return spec

        except Exception as e:
            print(f"Error generating spec with LLM: {e}")
            return self._generate_template_spec(gap_analysis, domain)

    def _create_specification_prompt(
        self,
        gap_analysis: Dict[str, Any],
        domain: str
    ) -> str:
        """Create prompt for LLM to generate parameter specification."""
        return f"""
Generate a parameter specification in YAML format for a musical program synthesis system.

Gap Analysis:
{yaml.dump(gap_analysis, default_flow_style=False)}

Domain: {domain}

Required Format (YAML):
```yaml
name: "domain.subdomain.parameter_name"
type: "categorical" | "float" | "int" | "boolean" | "probability"
domain: "{domain}"
subdomain: "specific_subdomain"
default: <valid_default_value>
description: >
  Clear description using musical terminology (minimum 20 characters).
  Explain what this parameter controls and its musical effect.

# For categorical type:
values:
  - "value1"
  - "value2"
  - "value3"

# For numeric types (float, int, probability):
range:
  min: <number>
  max: <number>

# Examples with explanations
examples:
  value1: "Explanation of what this value does musically"
  value2: "Another explanation"

# Parameter constraints
constraints:
  requires:
    - "other.parameter.name"  # This parameter needs these others
  conflicts_with:
    - "conflicting.parameter"  # Cannot be used with these

# Feature mappings from Agent 8 (feature extractor)
feature_mappings:
  - "feature_name_1"
  - "feature_name_2"
  - "feature_name_3"

# Training hints for model training
training_hints:
  importance: "high"  # critical, high, medium, low
  training_samples: 2000
  feature_selection_method: "mutual_info"

# Music theory reference
theory_reference:
  source: "Book or paper name"
  page: 47
  section: "Relevant section"
```

Requirements:
1. Name MUST follow pattern: domain.subdomain.param_name (all lowercase, underscores)
2. Type MUST be one of: categorical, float, int, boolean, probability
3. Default MUST be a valid value (in values list for categorical, in range for numeric)
4. Description MUST be at least 20 characters and use musical terminology
5. Examples MUST explain the musical effect, not just repeat the value name
6. Feature mappings should list actual features that would correlate with this parameter
7. DO NOT include code, functions, or classes
8. Return ONLY valid YAML, no explanatory text

Generate the specification now:
"""

    def _generate_template_spec(
        self,
        gap_analysis: Dict[str, Any],
        domain: str
    ) -> Dict[str, Any]:
        """Generate a template spec when LLM is unavailable."""
        gap_name = gap_analysis.get('gap_name', 'unknown_gap')

        # Convert gap name to parameter name format
        param_name = gap_name.lower().replace(' ', '_').replace('-', '_')

        return {
            'name': f"{domain}.expansion.{param_name}",
            'type': 'categorical',
            'domain': domain,
            'subdomain': 'expansion',
            'default': 'default_value',
            'values': ['default_value', 'option_1', 'option_2'],
            'description': f"Parameter for {gap_name} - automatically generated template",
            'examples': {
                'default_value': 'Standard application',
                'option_1': 'Alternative option 1',
                'option_2': 'Alternative option 2'
            },
            'feature_mappings': [
                f"{domain}_complexity",
                f"{domain}_density",
                f"{domain}_variation"
            ],
            'training_hints': {
                'importance': 'medium',
                'training_samples': 1000,
                'feature_selection_method': 'mutual_info'
            },
            'created_by': 'llm_agent',
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }

    def save_specification(
        self,
        spec: Dict[str, Any],
        output_dir: Path,
        status: str = "proposed"
    ) -> Path:
        """
        Save parameter specification to .params file.

        Args:
            spec: Parameter specification
            output_dir: Directory to save file
            status: "proposed", "validated", or "active"

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from parameter name
        param_name = spec['name']
        filename = f"{param_name.replace('.', '_')}_{status}.params"
        filepath = output_dir / filename

        # Save as YAML
        with open(filepath, 'w') as f:
            yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

        return filepath

    def generate_multiple_specs(
        self,
        gap_analyses: List[Dict[str, Any]],
        domain: str = "harmony"
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple parameter specifications.

        Args:
            gap_analyses: List of gap analyses
            domain: Musical domain

        Returns:
            List of parameter specifications
        """
        specs = []
        for gap in gap_analyses:
            spec = self.generate_parameter_spec(gap, domain)
            specs.append(spec)

        return specs


def generate_spec_from_description(
    description: str,
    domain: str = "harmony",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate spec from text description.

    Example:
        >>> spec = generate_spec_from_description(
        ...     "We need a parameter for jazz walking bass patterns",
        ...     domain="melody"
        ... )
    """
    generator = ParameterSpecificationGenerator(api_key=api_key)

    gap_analysis = {
        'gap_name': 'user_requested_parameter',
        'description': description,
        'domain': domain
    }

    return generator.generate_parameter_spec(gap_analysis, domain)


# Example usage
if __name__ == "__main__":
    # Test spec generation
    generator = ParameterSpecificationGenerator()

    gap = {
        'gap_name': 'Extended Jazz Voicings',
        'description': 'Missing parameter for drop 2, drop 3, and spread voicings in big band arranging',
        'domain': 'harmony',
        'evidence': 'Corpus analysis shows 75% of big band arrangements use these voicings'
    }

    spec = generator.generate_parameter_spec(gap, domain='harmony')

    print("Generated Parameter Specification:")
    print(yaml.dump(spec, default_flow_style=False))

    # Save to file
    output_path = generator.save_specification(
        spec,
        output_dir=Path("parameters/proposed"),
        status="proposed"
    )

    print(f"\nSaved to: {output_path}")
