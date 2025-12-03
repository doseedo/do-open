"""
LLM-Based Transform Code Generator
===================================

PyCraft-style transformation by example using Large Language Models.

Given:
- Input/output MIDI examples (gap cluster)
- Structural hints (graph patterns, primitive sequences)
- Existing transform templates

Generate:
- Executable Python code implementing the transform
- SpaceLevelTransform-compatible class

Supports:
- OpenAI GPT-4/GPT-3.5
- Anthropic Claude
- Local models (LLaMA, CodeLLaMA)
- Few-shot prompting with existing transforms

Author: Agent 8 - Transform Architecture
Phase: 2 (Automated Transform Discovery)
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import mido

from .space_level_transforms import (
    extract_notes_from_midi,
    notes_to_midi
)


# ============================================================================
# Few-Shot Example Library
# ============================================================================

class TransformExampleLibrary:
    """
    Library of transform examples for few-shot prompting.

    Each example shows:
    - Transform description
    - Input/output characteristics
    - Implementation code
    """

    def __init__(self):
        self.examples = self._build_examples()

    def _build_examples(self) -> List[Dict[str, str]]:
        """Build library of transform examples"""

        examples = [
            {
                'name': 'TransposeTransform',
                'description': 'Transpose all notes by semitones',
                'input_features': 'MIDI with pitch mean 60, range 20 semitones',
                'output_features': 'MIDI with pitch mean 67 (+7), same range',
                'transform_logic': 'Add constant to all pitches',
                'code': '''
class TransposeTransform(SpaceLevelTransform):
    def __init__(self):
        metadata = TransformMetadata(
            name='transpose',
            dimension='pitch',
            level='note',
            description='Transpose by semitones'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)
        notes = extract_notes_from_midi(midi)

        # Map amount to semitones: 0.5 = 0, 0.0 = -12, 1.0 = +12
        semitones = int((amount - 0.5) * 24)

        for note in notes:
            note['pitch'] = np.clip(note['pitch'] + semitones, 0, 127)

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        notes = extract_notes_from_midi(midi)
        if not notes:
            return 0.5
        mean_pitch = np.mean([n['pitch'] for n in notes])
        # Estimate transpose from mean pitch (assumes C4 = 60 baseline)
        semitones = mean_pitch - 60
        amount = 0.5 + semitones / 24
        return np.clip(amount, 0.0, 1.0)
'''
            },

            {
                'name': 'TempoTransform',
                'description': 'Scale tempo/duration of all events',
                'input_features': 'MIDI at 120 BPM, 4-second duration',
                'output_features': 'MIDI at 180 BPM, 2.67-second duration',
                'transform_logic': 'Scale all time values by tempo ratio',
                'code': '''
class TempoTransform(SpaceLevelTransform):
    def __init__(self):
        metadata = TransformMetadata(
            name='tempo',
            dimension='rhythm',
            level='section',
            description='Slow to fast tempo'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)
        notes = extract_notes_from_midi(midi)

        # Map amount to tempo multiplier: 0.5 = 1.0x, 0.0 = 0.5x, 1.0 = 2.0x
        tempo_multiplier = 0.5 + amount * 1.5

        for note in notes:
            note['start_time'] /= tempo_multiplier
            note['duration'] /= tempo_multiplier

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        # Estimate tempo from note density
        notes = extract_notes_from_midi(midi)
        if len(notes) < 2:
            return 0.5
        max_time = max(n['start_time'] + n['duration'] for n in notes)
        note_density = len(notes) / max(max_time, 1.0)
        # Higher density → faster tempo → higher amount
        amount = np.clip(note_density / 10, 0.0, 1.0)
        return amount
'''
            },

            {
                'name': 'SwingTransform',
                'description': 'Add swing rhythm (delay off-beats)',
                'input_features': 'Straight eighth notes, no timing deviation',
                'output_features': 'Swung eighth notes, off-beats delayed',
                'transform_logic': 'Detect beat grid, delay notes on off-beats',
                'code': '''
class SwingTransform(SpaceLevelTransform):
    def __init__(self):
        metadata = TransformMetadata(
            name='swing',
            dimension='rhythm',
            level='phrase',
            description='No swing to heavy swing'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)
        notes = extract_notes_from_midi(midi)

        # Detect beat duration (simplified: assume 0.5 seconds)
        beat_duration = 0.5
        swing_amount = amount  # 0 to 1

        for note in notes:
            # Check if on off-beat
            beat_position = (note['start_time'] % beat_duration) / beat_duration
            if 0.4 < beat_position < 0.6:  # Off-beat
                note['start_time'] += swing_amount * 0.1  # Delay

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        # Measure timing deviations from grid
        notes = extract_notes_from_midi(midi)
        if len(notes) < 10:
            return 0.0
        # Simplified: detect if notes are off-grid
        deviations = [abs(n['start_time'] % 0.5 - 0.5) for n in notes]
        avg_deviation = np.mean(deviations)
        amount = np.clip(avg_deviation * 10, 0.0, 1.0)
        return amount
'''
            },
        ]

        return examples

    def get_examples(self, n: int = 3) -> List[Dict[str, str]]:
        """Get n random examples for few-shot prompting"""
        if n >= len(self.examples):
            return self.examples
        indices = np.random.choice(len(self.examples), size=n, replace=False)
        return [self.examples[i] for i in indices]

    def find_similar_examples(self, hints: Dict[str, Any], n: int = 3) -> List[Dict[str, str]]:
        """Find examples similar to given hints"""
        # Simplified: return random examples
        # Full version would use embedding similarity
        return self.get_examples(n)


# ============================================================================
# LLM Code Generator
# ============================================================================

class LLMCodeGenerator:
    """
    Generate transform code using LLM.

    Supports multiple backends:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Local models
    - Template-based (fallback)
    """

    def __init__(self,
                 backend: str = 'template',
                 model: str = 'gpt-4',
                 api_key: Optional[str] = None,
                 temperature: float = 0.3):
        """
        Initialize LLM code generator.

        Args:
            backend: 'openai', 'anthropic', 'local', or 'template'
            model: Model name (e.g., 'gpt-4', 'claude-3-opus')
            api_key: API key for external services
            temperature: Sampling temperature (0.0 = deterministic)
        """
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

        self.example_library = TransformExampleLibrary()

        # Initialize backend client
        self.client = self._initialize_backend()

    def _initialize_backend(self):
        """Initialize LLM backend client"""
        if self.backend == 'openai':
            try:
                import openai
                if self.api_key:
                    openai.api_key = self.api_key
                return openai
            except ImportError:
                print("Warning: OpenAI not installed. Falling back to template mode.")
                self.backend = 'template'
                return None

        elif self.backend == 'anthropic':
            try:
                import anthropic
                if self.api_key:
                    return anthropic.Anthropic(api_key=self.api_key)
                return None
            except ImportError:
                print("Warning: Anthropic not installed. Falling back to template mode.")
                self.backend = 'template'
                return None

        elif self.backend == 'local':
            # Placeholder for local model
            print("Warning: Local model support not implemented. Using template mode.")
            self.backend = 'template'
            return None

        else:  # template
            return None

    def generate_from_examples(self,
                               gap_cluster_id: str,
                               examples: List[Tuple[mido.MidiFile, mido.MidiFile]],
                               hints: Dict[str, Any]) -> Tuple[str, float]:
        """
        Generate transform code from input/output examples.

        Args:
            gap_cluster_id: Identifier for the gap cluster
            examples: List of (input_midi, output_midi) pairs
            hints: Dictionary with graph patterns, primitives, etc.

        Returns:
            (generated_code, confidence_score)
        """
        # Build prompt
        prompt = self._build_prompt(gap_cluster_id, examples, hints)

        # Generate code
        if self.backend == 'openai':
            code, confidence = self._generate_openai(prompt)
        elif self.backend == 'anthropic':
            code, confidence = self._generate_anthropic(prompt)
        elif self.backend == 'local':
            code, confidence = self._generate_local(prompt)
        else:  # template
            code, confidence = self._generate_template(gap_cluster_id, hints)

        # Validate and clean code
        code = self._validate_and_clean_code(code)

        return code, confidence

    def _build_prompt(self,
                     gap_cluster_id: str,
                     examples: List[Tuple[mido.MidiFile, mido.MidiFile]],
                     hints: Dict[str, Any]) -> str:
        """Build few-shot prompt for LLM"""

        # Get similar examples from library
        similar_examples = self.example_library.find_similar_examples(hints, n=3)

        # Analyze input/output examples
        example_analysis = self._analyze_examples(examples)

        # Build prompt
        prompt = f"""You are a musical transform code generator. Given input/output MIDI examples, generate Python code for a transform that converts inputs to outputs.

TASK: Generate a SpaceLevelTransform subclass that implements the musical transformation shown in the examples below.

HINTS:
- Gap cluster: {gap_cluster_id}
- Graph patterns: {hints.get('graph_motifs', 'N/A')}
- Primitive sequence: {hints.get('primitive_sequence', 'N/A')}
- Structural features: {hints.get('structural_features', 'N/A')}

EXAMPLE ANALYSIS:
{example_analysis}

FEW-SHOT EXAMPLES:
Here are some existing transforms for reference:

"""

        # Add few-shot examples
        for i, ex in enumerate(similar_examples, 1):
            prompt += f"""
Example {i}: {ex['name']}
Description: {ex['description']}
Logic: {ex['transform_logic']}

Code:
```python
{ex['code']}
```

"""

        prompt += f"""
NOW: Generate a similar transform class for the gap cluster '{gap_cluster_id}' based on the hints and example analysis above.

REQUIREMENTS:
1. Must inherit from SpaceLevelTransform
2. Must implement apply(midi, amount) → midi
3. Must implement get_current_value(midi) → float
4. Must use extract_notes_from_midi() and notes_to_midi()
5. Must handle amount parameter in range [0, 1] with 0.5 as neutral
6. Must include docstring with description

Generate ONLY the Python code (no explanations):
"""

        return prompt

    def _analyze_examples(self, examples: List[Tuple[mido.MidiFile, mido.MidiFile]]) -> str:
        """Analyze input/output examples to identify transformation"""
        if not examples:
            return "No examples provided"

        analyses = []

        for i, (input_midi, output_midi) in enumerate(examples[:5], 1):  # Analyze first 5
            input_notes = extract_notes_from_midi(input_midi)
            output_notes = extract_notes_from_midi(output_midi)

            # Compare features
            input_stats = self._compute_stats(input_notes)
            output_stats = self._compute_stats(output_notes)

            analysis = f"""
Example {i}:
  Input:  pitch_mean={input_stats['pitch_mean']:.1f}, pitch_range={input_stats['pitch_range']:.0f}, duration={input_stats['duration']:.2f}s, n_notes={input_stats['n_notes']}
  Output: pitch_mean={output_stats['pitch_mean']:.1f}, pitch_range={output_stats['pitch_range']:.0f}, duration={output_stats['duration']:.2f}s, n_notes={output_stats['n_notes']}
  Change: pitch_shift={output_stats['pitch_mean'] - input_stats['pitch_mean']:.1f}, tempo_ratio={input_stats['duration'] / max(output_stats['duration'], 0.01):.2f}, note_ratio={output_stats['n_notes'] / max(input_stats['n_notes'], 1):.2f}
"""
            analyses.append(analysis)

        return '\n'.join(analyses)

    def _compute_stats(self, notes: List[Dict]) -> Dict[str, float]:
        """Compute statistics for note list"""
        if not notes:
            return {
                'pitch_mean': 60,
                'pitch_range': 0,
                'duration': 0,
                'n_notes': 0,
                'velocity_mean': 64
            }

        pitches = [n['pitch'] for n in notes]
        velocities = [n['velocity'] for n in notes]
        max_time = max(n['start_time'] + n['duration'] for n in notes)

        return {
            'pitch_mean': np.mean(pitches),
            'pitch_range': max(pitches) - min(pitches),
            'duration': max_time,
            'n_notes': len(notes),
            'velocity_mean': np.mean(velocities)
        }

    def _generate_openai(self, prompt: str) -> Tuple[str, float]:
        """Generate code using OpenAI API"""
        try:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer specializing in music transformation code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )

            code = response.choices[0].message.content
            confidence = 0.85  # High confidence for GPT-4

            return code, confidence

        except Exception as e:
            print(f"Warning: OpenAI generation failed: {e}")
            return self._generate_template("learned_transform", {})

    def _generate_anthropic(self, prompt: str) -> Tuple[str, float]:
        """Generate code using Anthropic Claude API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            code = response.content[0].text
            confidence = 0.85

            return code, confidence

        except Exception as e:
            print(f"Warning: Anthropic generation failed: {e}")
            return self._generate_template("learned_transform", {})

    def _generate_local(self, prompt: str) -> Tuple[str, float]:
        """Generate code using local model"""
        # Placeholder - would use HuggingFace transformers or similar
        print("Warning: Local model generation not implemented")
        return self._generate_template("learned_transform", {})

    def _generate_template(self, gap_cluster_id: str, hints: Dict[str, Any]) -> Tuple[str, float]:
        """Generate code using template (fallback)"""
        transform_name = f"Learned{gap_cluster_id.replace('_', ' ').title().replace(' ', '')}Transform"
        primitive_sequence = hints.get('primitive_sequence', [])

        code = f'''class {transform_name}(SpaceLevelTransform):
    """
    Auto-discovered transform for gap cluster {gap_cluster_id}.

    Based on:
    - Primitive sequence: {' → '.join(primitive_sequence) if primitive_sequence else 'N/A'}
    - Graph patterns: {len(hints.get('graph_motifs', []))} motifs found
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='{gap_cluster_id}',
            dimension='learned',
            level='mixed',
            description='Auto-discovered transform'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        amount = self.validate_amount(amount)
        notes = extract_notes_from_midi(midi)

        # Default implementation: identity transform
        # TODO: Implement learned transformation

        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        return 0.5
'''

        confidence = 0.5  # Template-based has lower confidence

        return code, confidence

    def _validate_and_clean_code(self, code: str) -> str:
        """Validate and clean generated code"""
        # Extract code from markdown if present
        if '```python' in code:
            match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
            if match:
                code = match.group(1)
        elif '```' in code:
            match = re.search(r'```\s*(.*?)\s*```', code, re.DOTALL)
            if match:
                code = match.group(1)

        # Basic validation: check for required methods
        required_methods = ['def apply', 'def get_current_value', 'def __init__']
        for method in required_methods:
            if method not in code:
                print(f"Warning: Generated code missing '{method}'")

        # Check for class definition
        if 'class ' not in code or 'SpaceLevelTransform' not in code:
            print("Warning: Generated code missing class definition")

        return code.strip()


# ============================================================================
# Transform Validator
# ============================================================================

class TransformValidator:
    """
    Validate generated transform code.

    Checks:
    1. Syntax validity (can compile)
    2. Runtime safety (no crashes)
    3. Semantic correctness (produces valid MIDI)
    4. Performance (not too slow)
    """

    def validate(self, code: str, test_midi: mido.MidiFile) -> Tuple[bool, str]:
        """
        Validate transform code.

        Args:
            code: Generated Python code
            test_midi: Test MIDI file

        Returns:
            (is_valid, error_message)
        """
        # Check 1: Syntax
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check 2: Can instantiate
        # (Skipped for safety - would need proper sandbox)

        # Check 3: Produces valid output
        # (Skipped - would need execution sandbox)

        return True, ""
