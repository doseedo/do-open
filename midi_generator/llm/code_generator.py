"""
AGENT 12: LLM Code Generation Agent
====================================

Generates production-ready Python code to implement new parameters in the
85,989-line HarmonyModule generator using Claude API.

This agent:
1. Identifies relevant files for parameter implementation
2. Builds code context for LLM
3. Generates complete implementation code
4. Validates syntax and patterns
5. Ensures backward compatibility

Author: Agent 12 - Code Generation Specialist
License: MIT
"""

import ast
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import os

try:
    import anthropic
except ImportError:
    anthropic = None
    print("WARNING: anthropic package not installed. Code generation will use mock mode.")


@dataclass
class CodeSection:
    """Represents a section of generated code"""
    section_type: str  # 'registry', 'modification', 'new_method', 'test', 'integration'
    file_path: str
    code: str
    description: str
    line_number: Optional[int] = None
    class_name: Optional[str] = None
    method_name: Optional[str] = None


@dataclass
class GeneratedImplementation:
    """Complete implementation package for a new parameter"""
    parameter_name: str
    registry_update: str
    generator_modifications: Dict[str, str]  # file_path -> code
    new_methods: Dict[str, str]  # method_signature -> code
    test_code: str
    integration_points: List[str]
    documentation: str
    migration_notes: str
    code_sections: List[CodeSection] = field(default_factory=list)
    generation_time: float = 0.0
    tokens_used: int = 0


class CodebaseIndex:
    """Index of codebase for fast code retrieval"""

    def __init__(self, root_dir: Path = Path('midi_generator')):
        self.root_dir = Path(root_dir)
        self.file_index: Dict[str, Path] = {}
        self.class_index: Dict[str, List[Path]] = {}
        self.method_index: Dict[str, List[Tuple[Path, str]]] = {}
        self._build_index()

    def _build_index(self):
        """Build index of all Python files, classes, and methods"""
        if not self.root_dir.exists():
            return

        for py_file in self.root_dir.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue

            # Index file
            relative_path = py_file.relative_to(self.root_dir.parent)
            self.file_index[str(relative_path)] = py_file

            # Parse and index classes/methods
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    tree = ast.parse(f.read(), filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if node.name not in self.class_index:
                            self.class_index[node.name] = []
                        self.class_index[node.name].append(py_file)

                    elif isinstance(node, ast.FunctionDef):
                        if node.name not in self.method_index:
                            self.method_index[node.name] = []
                        parent_class = self._get_parent_class(tree, node)
                        self.method_index[node.name].append((py_file, parent_class))

            except Exception as e:
                # Skip files that can't be parsed
                pass

    def _get_parent_class(self, tree: ast.AST, func_node: ast.FunctionDef) -> Optional[str]:
        """Get the parent class name for a function node"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item == func_node:
                        return node.name
        return None

    def find_file(self, partial_path: str) -> Optional[Path]:
        """Find file by partial path"""
        for file_path, full_path in self.file_index.items():
            if partial_path in file_path:
                return full_path
        return None

    def find_class(self, class_name: str) -> List[Path]:
        """Find files containing a class"""
        return self.class_index.get(class_name, [])

    def find_method(self, method_name: str) -> List[Tuple[Path, str]]:
        """Find files containing a method and its parent class"""
        return self.method_index.get(method_name, [])


class CodePatternLibrary:
    """Library of code patterns for parameter implementation"""

    @staticmethod
    def get_probability_pattern() -> str:
        return '''
def _apply_{param_name}(self, params: dict, context: dict) -> Any:
    """
    Apply {param_description}

    Args:
        params: Parameter dictionary
        context: Current musical context

    Returns:
        Result based on probability
    """
    # Get parameter with default
    probability = params.get('{full_param_name}', {default_value})

    # Validate range
    probability = max(0.0, min(1.0, probability))

    # Apply probability
    import random
    if random.random() < probability:
        return self._do_{action}_variant(context)
    else:
        return self._do_{action}_normal(context)
'''

    @staticmethod
    def get_categorical_pattern() -> str:
        return '''
def _apply_{param_name}(self, params: dict, context: dict) -> Any:
    """
    Apply {param_description}

    Args:
        params: Parameter dictionary
        context: Current musical context

    Returns:
        Result based on categorical choice
    """
    # Get parameter with default
    choice = params.get('{full_param_name}', '{default_value}')

    # Apply based on category
    {category_cases}

    # Default fallback
    return self._default_{action}(context)
'''

    @staticmethod
    def get_continuous_pattern() -> str:
        return '''
def _apply_{param_name}(self, params: dict, context: dict) -> Any:
    """
    Apply {param_description}

    Args:
        params: Parameter dictionary
        context: Current musical context

    Returns:
        Result scaled by continuous value
    """
    # Get parameter with default
    value = params.get('{full_param_name}', {default_value})

    # Validate range
    value = max({min_value}, min({max_value}, value))

    # Apply scaling
    {application_logic}

    return result
'''

    @staticmethod
    def get_registry_pattern() -> str:
        return '''
    # {param_description}
    "{full_param_name}": ParameterDefinition(
        name="{param_short_name}",
        full_path="{full_param_name}",
        description="{param_description}",
        param_type=ParameterType.{param_type},
        default_value={default_value},
        min_value={min_value},
        max_value={max_value},
        options={options},
        category=ParameterCategory.{category},
        musical_impact=MusicalImpact.{impact},
        genre_relevance={genre_list},
        learnable=True,
        constraint_description="{constraints}"
    ),
'''

    @staticmethod
    def get_test_pattern() -> str:
        return '''
def test_{param_name}_{test_case}(self):
    """Test {param_description} with {test_case}"""
    # Setup
    params = {{
        '{full_param_name}': {test_value}
    }}

    # Execute
    result = self.generator.generate(params)

    # Verify
    assert result is not None, "Generation should not fail"
    {additional_assertions}
'''


class LLMCodeGenerationAgent:
    """
    Main agent for LLM-powered code generation

    This agent uses Claude API to generate production-ready code for
    implementing new parameters in the music generation system.
    """

    def __init__(self, api_key: Optional[str] = None, mock_mode: bool = False):
        """
        Initialize code generation agent

        Args:
            api_key: Anthropic API key (or None to use env var)
            mock_mode: If True, use mock responses instead of real API
        """
        self.mock_mode = mock_mode or anthropic is None

        if not self.mock_mode:
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.llm = anthropic.Anthropic(api_key=api_key)
            else:
                print("WARNING: No API key found, using mock mode")
                self.mock_mode = True

        self.codebase_index = CodebaseIndex()
        self.code_patterns = CodePatternLibrary()
        self.generation_history: List[GeneratedImplementation] = []

    def generate_implementation(self, parameter_proposal: dict) -> GeneratedImplementation:
        """
        Generate complete implementation code for new parameter

        Args:
            parameter_proposal: Parameter proposal from Agent 11 containing:
                - name: Full parameter name (e.g., 'harmony.voicing.quartal_probability')
                - type: Parameter type
                - range: Valid range/options
                - default: Default value
                - description: What the parameter does
                - musical_context: When/why it's used
                - implementation_strategy: How to implement it
                - test_cases: List of test scenarios

        Returns:
            GeneratedImplementation with all code components
        """
        start_time = time.time()

        print(f"\n{'='*80}")
        print(f"GENERATING CODE FOR: {parameter_proposal['name']}")
        print(f"{'='*80}\n")

        # 1. Identify relevant files
        print("Step 1: Identifying relevant files...")
        relevant_files = self._identify_relevant_files(parameter_proposal)
        print(f"  Found {len(relevant_files)} files to modify")
        for f in relevant_files:
            print(f"    - {f}")

        # 2. Build code context
        print("\nStep 2: Building code context...")
        code_context = self._build_code_context(relevant_files)
        print(f"  Loaded context from {len(code_context)} files")

        # 3. Generate implementation via LLM or mock
        print("\nStep 3: Generating implementation code...")
        if self.mock_mode:
            implementation_dict = self._mock_generate_code(parameter_proposal, code_context)
        else:
            implementation_dict = self._llm_generate_code(parameter_proposal, code_context)

        # 4. Validate generated code
        print("\nStep 4: Validating generated code...")
        validation_result = self._validate_code(implementation_dict)
        if not validation_result['valid']:
            print(f"  WARNING: Validation issues detected:")
            for issue in validation_result['issues']:
                print(f"    - {issue}")
        else:
            print(f"  ✅ Code validation passed")

        # 5. Build GeneratedImplementation object
        generation_time = time.time() - start_time

        implementation = GeneratedImplementation(
            parameter_name=parameter_proposal['name'],
            registry_update=implementation_dict.get('registry_update', ''),
            generator_modifications=implementation_dict.get('generator_modifications', {}),
            new_methods=implementation_dict.get('new_methods', {}),
            test_code=implementation_dict.get('test_code', ''),
            integration_points=implementation_dict.get('integration_points', []),
            documentation=implementation_dict.get('documentation', ''),
            migration_notes=implementation_dict.get('migration_notes', ''),
            generation_time=generation_time,
            tokens_used=implementation_dict.get('tokens_used', 0)
        )

        # Record in history
        self.generation_history.append(implementation)

        print(f"\n✅ Code generation completed in {generation_time:.2f}s")
        print(f"   Registry updates: {len(implementation.registry_update)} chars")
        print(f"   Modified files: {len(implementation.generator_modifications)}")
        print(f"   New methods: {len(implementation.new_methods)}")
        print(f"   Test code: {len(implementation.test_code)} chars")

        return implementation

    def _identify_relevant_files(self, proposal: dict) -> List[str]:
        """
        Identify which generator files need modification

        Args:
            proposal: Parameter proposal

        Returns:
            List of file paths that need modification
        """
        param_name = proposal['name']
        domain = param_name.split('.')[0] if '.' in param_name else 'harmony'

        # Map domain to generator files
        domain_to_files = {
            'harmony': [
                'midi_generator/generators/advanced_harmony_generator.py',
                'midi_generator/generators/harmonic_rhythm.py',
                'midi_generator/generators/reharmonization_engine.py',
                'midi_generator/core/modal_harmony.py',
                'midi_generator/core/neo_riemannian.py'
            ],
            'melody': [
                'midi_generator/generators/development_engine.py',
                'midi_generator/generators/texture_generator.py'
            ],
            'rhythm': [
                'midi_generator/algorithms/rhythm_engine.py',
                'midi_generator/algorithms/advanced_rhythm.py',
                'midi_generator/algorithms/groove_library.py'
            ],
            'instrumentation': [
                'midi_generator/generators/orchestrator.py',
                'midi_generator/core/instrument_library.py',
                'midi_generator/core/ensemble_registry.py'
            ],
            'dynamics': [
                'midi_generator/generators/texture_generator.py'
            ],
            'structure': [
                'midi_generator/generators/form_generator.py',
                'midi_generator/generators/development_engine.py'
            ],
            'bass': [
                'midi_generator/generators/orchestrator.py'
            ],
            'drums': [
                'midi_generator/algorithms/drum_patterns.py',
                'midi_generator/algorithms/rhythm_engine.py'
            ],
            'voicing': [
                'midi_generator/generators/advanced_harmony_generator.py'
            ],
            'articulation': [
                'midi_generator/generators/texture_generator.py'
            ],
            'expression': [
                'midi_generator/generators/texture_generator.py'
            ]
        }

        # Get base files for domain
        files = domain_to_files.get(domain, ['midi_generator/generators/advanced_harmony_generator.py'])

        # Always add parameter registry
        files.append('midi_generator/parameters/universal_registry.py')

        # Add files from integration points if specified
        if 'generator_integration_points' in proposal:
            for point in proposal['generator_integration_points']:
                if '::' in point:
                    file_path = point.split('::')[0]
                    if file_path not in files:
                        files.append(file_path)

        # Filter to only existing files
        existing_files = []
        for file_path in files:
            full_path = Path(file_path)
            if full_path.exists():
                existing_files.append(file_path)

        return existing_files

    def _build_code_context(self, file_paths: List[str]) -> Dict[str, dict]:
        """
        Load relevant code excerpts for LLM context

        Args:
            file_paths: List of files to load

        Returns:
            Dictionary mapping file_path to context dict
        """
        context = {}

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Extract relevant classes and methods
                context[file_path] = {
                    'full_content': content[:50000],  # Limit to 50k chars
                    'classes': self._extract_classes(content),
                    'key_methods': self._extract_key_methods(content),
                    'imports': self._extract_imports(content),
                    'line_count': len(content.split('\n'))
                }

            except FileNotFoundError:
                context[file_path] = {
                    'exists': False,
                    'error': f'File not found: {file_path}'
                }

            except Exception as e:
                context[file_path] = {
                    'exists': False,
                    'error': f'Error reading file: {e}'
                }

        return context

    def _extract_classes(self, code: str) -> Dict[str, str]:
        """Extract class definitions from code"""
        classes = {}

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    try:
                        class_source = ast.get_source_segment(code, node)
                        if class_source and len(class_source) < 10000:  # Limit size
                            classes[node.name] = class_source[:5000]
                    except:
                        pass
        except:
            pass

        return classes

    def _extract_key_methods(self, code: str) -> Dict[str, str]:
        """Extract important method definitions"""
        methods = {}

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract methods matching common patterns
                    if any(pattern in node.name for pattern in [
                        'generate', 'create', 'build', 'apply', 'process',
                        'calculate', 'compute', 'determine', 'select'
                    ]):
                        try:
                            method_source = ast.get_source_segment(code, node)
                            if method_source and len(method_source) < 5000:
                                methods[node.name] = method_source[:2000]
                        except:
                            pass
        except:
            pass

        return methods

    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements"""
        imports = []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        names = ', '.join([alias.name for alias in node.names])
                        imports.append(f"from {node.module} import {names}")
        except:
            pass

        return imports

    def _llm_generate_code(self, proposal: dict, context: dict) -> dict:
        """
        Use Claude API to generate implementation code

        Args:
            proposal: Parameter proposal
            context: Code context from existing files

        Returns:
            Dictionary with code sections
        """
        system_prompt = self._get_code_generation_system_prompt()
        user_prompt = self._build_code_generation_prompt(proposal, context)

        try:
            response = self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.2
            )

            response_text = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # Parse response into structured code
            result = self._parse_code_response(response_text)
            result['tokens_used'] = tokens_used

            return result

        except Exception as e:
            print(f"ERROR calling Claude API: {e}")
            print("Falling back to mock generation")
            return self._mock_generate_code(proposal, context)

    def _mock_generate_code(self, proposal: dict, context: dict) -> dict:
        """
        Generate mock implementation (for testing without API)

        Args:
            proposal: Parameter proposal
            context: Code context

        Returns:
            Dictionary with mock code sections
        """
        param_name = proposal['name']
        param_type = proposal.get('type', 'CONTINUOUS')
        param_short = param_name.split('.')[-1]

        # Generate registry update
        registry_update = self._generate_mock_registry(proposal)

        # Generate modification
        generator_mods = self._generate_mock_modifications(proposal)

        # Generate new methods
        new_methods = self._generate_mock_methods(proposal)

        # Generate tests
        test_code = self._generate_mock_tests(proposal)

        return {
            'registry_update': registry_update,
            'generator_modifications': generator_mods,
            'new_methods': new_methods,
            'test_code': test_code,
            'integration_points': [
                f"generators/advanced_harmony_generator.py::HarmonyGenerator.generate()"
            ],
            'documentation': f"Implements {param_name} parameter",
            'migration_notes': "No breaking changes - backward compatible",
            'tokens_used': 0
        }

    def _generate_mock_registry(self, proposal: dict) -> str:
        """Generate mock registry code"""
        param_name = proposal['name']
        param_type = proposal.get('type', 'CONTINUOUS')
        default = proposal.get('default', 0.5)
        description = proposal.get('description', 'No description')

        return f'''
# Auto-generated parameter: {param_name}
PARAMETER_REGISTRY["{param_name}"] = ParameterDefinition(
    name="{param_name.split('.')[-1]}",
    full_path="{param_name}",
    description="{description}",
    param_type=ParameterType.{param_type},
    default_value={default},
    learnable=True
)
'''

    def _generate_mock_modifications(self, proposal: dict) -> Dict[str, str]:
        """Generate mock generator modifications"""
        param_name = proposal['name']
        param_short = param_name.split('.')[-1]
        default = proposal.get('default', 0.5)

        code = f'''
    def _apply_{param_short}(self, params: dict) -> float:
        """Apply {param_name} parameter"""
        value = params.get('{param_name}', {default})
        value = max(0.0, min(1.0, value))
        return value
'''

        return {
            'midi_generator/generators/advanced_harmony_generator.py': code
        }

    def _generate_mock_methods(self, proposal: dict) -> Dict[str, str]:
        """Generate mock new methods"""
        return {}

    def _generate_mock_tests(self, proposal: dict) -> str:
        """Generate mock test code"""
        param_name = proposal['name']
        param_short = param_name.split('.')[-1]

        return f'''
def test_{param_short}_basic():
    """Test {param_name} parameter"""
    params = {{'{param_name}': 0.7}}
    generator = HarmonyGenerator()
    result = generator.generate(params)
    assert result is not None

def test_{param_short}_edge_cases():
    """Test edge cases for {param_name}"""
    params = {{'{param_name}': 0.0}}
    generator = HarmonyGenerator()
    result = generator.generate(params)
    assert result is not None

    params = {{'{param_name}': 1.0}}
    result = generator.generate(params)
    assert result is not None
'''

    def _get_code_generation_system_prompt(self) -> str:
        """Get system prompt for Claude code generation"""
        return """
You are an expert Python developer working on a large-scale music generation system.

Your task is to generate production-ready code to implement a new parameter in the existing generator.

CODE QUALITY REQUIREMENTS:

1. **Backward Compatibility**: CRITICAL
   - ALL existing parameters must continue to work
   - Use .get() with defaults for ALL parameter access
   - Never break existing functionality

2. **Code Style**:
   - Follow existing naming conventions
   - Match indentation and formatting
   - Use type hints
   - Add comprehensive docstrings

3. **Error Handling**:
   - Validate parameter values
   - Handle edge cases (0.0, 1.0, None, etc.)
   - Graceful degradation if parameter missing

4. **Testing**:
   - Provide unit tests
   - Test edge cases
   - Test integration with other parameters

5. **Documentation**:
   - Clear docstrings
   - Inline comments for complex logic
   - Parameter usage examples

OUTPUT FORMAT:

Provide code in structured sections with clear markers:
```python
# === SECTION 1: REGISTRY UPDATE ===
# File: parameters/universal_registry.py

[registry code here]

# === SECTION 2: GENERATOR MODIFICATION ===
# File: generators/example_generator.py
# Method: ExampleGenerator.generate_something()
# Action: MODIFY EXISTING METHOD

[modified method code here]

# === SECTION 3: NEW METHOD ===
# File: generators/example_generator.py
# Class: ExampleGenerator

[new method code here]

# === SECTION 4: TESTS ===
# File: tests/test_new_parameter.py

[test code here]

# === SECTION 5: INTEGRATION NOTES ===

[notes on where parameter is used, how it integrates, etc.]
```

CRITICAL:
- Generate COMPLETE, RUNNABLE code
- NO placeholders like "# TODO" or "# implement this"
- ALL methods must be fully implemented
- Include ALL necessary imports
"""

    def _build_code_generation_prompt(self, proposal: dict, context: dict) -> str:
        """Build specific prompt for this parameter"""

        prompt = f"""
PARAMETER TO IMPLEMENT:

Name: {proposal['name']}
Type: {proposal.get('type', 'CONTINUOUS')}
Range: {proposal.get('range', '(0.0, 1.0)')}
Default: {proposal.get('default', 0.5)}

Description: {proposal.get('description', 'No description provided')}

Musical Context: {proposal.get('musical_context', 'No context provided')}

Implementation Strategy: {proposal.get('implementation_strategy', 'No strategy provided')}

Test Cases:
{json.dumps(proposal.get('test_cases', []), indent=2)}

Example Values:
{json.dumps(proposal.get('example_values', {}), indent=2)}

EXISTING CODEBASE CONTEXT:
"""

        # Add relevant code context (limit to avoid token overflow)
        context_count = 0
        max_context_files = 3

        for file_path, file_context in list(context.items())[:max_context_files]:
            context_count += 1

            if not file_context.get('exists', True):
                prompt += f"\n{file_path}: [NEW FILE - does not exist yet]\n"
                continue

            prompt += f"\n\n# {file_path}\n"
            prompt += f"Lines: {file_context.get('line_count', 'unknown')}\n"

            # Add key methods
            if 'key_methods' in file_context and file_context['key_methods']:
                prompt += "\nKey methods:\n"
                for method_name, method_code in list(file_context['key_methods'].items())[:3]:
                    prompt += f"\n```python\n{method_code[:1000]}\n```\n"

        prompt += """

YOUR TASK:

Generate COMPLETE implementation code for this parameter following these steps:

1. **Registry Update**: Add parameter to parameters/universal_registry.py

2. **Generator Modifications**: Modify existing generator methods to use the parameter
   - Identify the right place to check the parameter
   - Use .get() with default value
   - Implement the musical logic described

3. **New Methods**: Create any helper methods needed
   - Full implementation, no placeholders
   - Docstrings with examples
   - Error handling

4. **Tests**: Write comprehensive unit tests
   - Test all values in test_cases
   - Test edge cases (0.0, 1.0, None)
   - Test integration with other parameters

5. **Integration Notes**: Document where and how parameter is used

Generate code following the OUTPUT FORMAT specified in your system prompt.

CRITICAL REQUIREMENTS:
- BACKWARD COMPATIBLE: All existing parameters must still work
- COMPLETE: No TODOs or placeholders
- TESTED: Include working unit tests
- DOCUMENTED: Clear docstrings and comments
"""

        return prompt

    def _parse_code_response(self, response_text: str) -> dict:
        """
        Parse LLM response into structured code

        Args:
            response_text: Raw text from LLM

        Returns:
            Dictionary with code sections
        """
        sections = {
            'registry_update': '',
            'generator_modifications': {},
            'new_methods': {},
            'test_code': '',
            'integration_points': [],
            'documentation': '',
            'migration_notes': ''
        }

        # Split response by section markers
        current_section = None
        current_file = None
        current_buffer = []

        lines = response_text.split('\n')

        for line in lines:
            # Check for section markers
            if '=== SECTION 1: REGISTRY UPDATE ===' in line or '=== REGISTRY UPDATE ===' in line:
                self._flush_buffer(sections, current_section, current_file, current_buffer)
                current_section = 'registry'
                current_buffer = []

            elif '=== SECTION 2: GENERATOR MODIFICATION ===' in line or '=== GENERATOR MODIFICATION ===' in line:
                self._flush_buffer(sections, current_section, current_file, current_buffer)
                current_section = 'modification'
                current_buffer = []

            elif '=== SECTION 3: NEW METHOD ===' in line or '=== NEW METHOD ===' in line:
                self._flush_buffer(sections, current_section, current_file, current_buffer)
                current_section = 'new_method'
                current_buffer = []

            elif '=== SECTION 4: TESTS ===' in line or '=== TESTS ===' in line:
                self._flush_buffer(sections, current_section, current_file, current_buffer)
                current_section = 'tests'
                current_buffer = []

            elif '=== SECTION 5: INTEGRATION NOTES ===' in line or '=== INTEGRATION NOTES ===' in line:
                self._flush_buffer(sections, current_section, current_file, current_buffer)
                current_section = 'notes'
                current_buffer = []

            # Check for file markers
            elif line.startswith('# File: '):
                if current_buffer and current_section == 'modification':
                    if current_file:
                        sections['generator_modifications'][current_file] = '\n'.join(current_buffer)
                    current_buffer = []
                current_file = line.replace('# File: ', '').strip()

            # Accumulate code
            else:
                current_buffer.append(line)

        # Flush final section
        self._flush_buffer(sections, current_section, current_file, current_buffer)

        return sections

    def _flush_buffer(self, sections: dict, current_section: Optional[str],
                      current_file: Optional[str], current_buffer: List[str]):
        """Flush current buffer to sections"""
        if not current_section or not current_buffer:
            return

        content = '\n'.join(current_buffer)

        if current_section == 'registry':
            sections['registry_update'] = content
        elif current_section == 'modification' and current_file:
            sections['generator_modifications'][current_file] = content
        elif current_section == 'new_method':
            method_name = self._extract_method_name(content)
            if method_name:
                sections['new_methods'][method_name] = content
        elif current_section == 'tests':
            sections['test_code'] = content
        elif current_section == 'notes':
            sections['integration_points'] = self._extract_integration_points(content)
            sections['documentation'] = content

    def _extract_method_name(self, code: str) -> Optional[str]:
        """Extract method name from code"""
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
        return None

    def _extract_integration_points(self, text: str) -> List[str]:
        """Extract integration points from notes"""
        points = []
        for line in text.split('\n'):
            if '::' in line or 'line' in line.lower():
                points.append(line.strip())
        return points

    def _validate_code(self, implementation: dict) -> dict:
        """
        Validate generated code for syntax and patterns

        Args:
            implementation: Dictionary with code sections

        Returns:
            Dictionary with validation results
        """
        issues = []

        # 1. Syntax check
        for file_path, code in implementation.get('generator_modifications', {}).items():
            if code and code.strip():
                try:
                    compile(code, file_path, 'exec')
                except SyntaxError as e:
                    issues.append(f"Syntax error in {file_path}: {e}")

        # 2. Check for required patterns
        registry_code = implementation.get('registry_update', '')
        if registry_code and 'Parameter' not in registry_code:
            issues.append("Registry update missing Parameter definition")

        # 3. Check for backward compatibility (.get usage)
        for code in implementation.get('generator_modifications', {}).values():
            if code and 'params[' in code and 'params.get(' not in code:
                issues.append("Warning: Direct dict access (params[]) instead of .get() - may break compatibility")

        # 4. Check for tests
        if not implementation.get('test_code'):
            issues.append("No test code provided")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    def save_implementation(self, implementation: GeneratedImplementation, output_dir: Path):
        """
        Save generated implementation to files

        Args:
            implementation: Generated implementation
            output_dir: Directory to save to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        impl_dict = {
            'parameter_name': implementation.parameter_name,
            'registry_update': implementation.registry_update,
            'generator_modifications': implementation.generator_modifications,
            'new_methods': implementation.new_methods,
            'test_code': implementation.test_code,
            'integration_points': implementation.integration_points,
            'documentation': implementation.documentation,
            'migration_notes': implementation.migration_notes,
            'generation_time': implementation.generation_time,
            'tokens_used': implementation.tokens_used
        }

        output_file = output_dir / f"{implementation.parameter_name.replace('.', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(impl_dict, f, indent=2)

        print(f"Saved implementation to: {output_file}")

    def load_implementation(self, file_path: Path) -> GeneratedImplementation:
        """
        Load implementation from JSON file

        Args:
            file_path: Path to JSON file

        Returns:
            GeneratedImplementation object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        return GeneratedImplementation(
            parameter_name=data['parameter_name'],
            registry_update=data['registry_update'],
            generator_modifications=data['generator_modifications'],
            new_methods=data['new_methods'],
            test_code=data['test_code'],
            integration_points=data['integration_points'],
            documentation=data['documentation'],
            migration_notes=data['migration_notes'],
            generation_time=data.get('generation_time', 0.0),
            tokens_used=data.get('tokens_used', 0)
        )


# Utility functions for code analysis and manipulation

def extract_function_signature(code: str, function_name: str) -> Optional[str]:
    """Extract function signature from code"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get signature
                args = []
                for arg in node.args.args:
                    args.append(arg.arg)
                return f"def {function_name}({', '.join(args)})"
    except:
        pass
    return None


def get_class_methods(code: str, class_name: str) -> List[str]:
    """Get all method names in a class"""
    methods = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
    except:
        pass
    return methods


def insert_code_after_marker(original_code: str, marker: str, new_code: str) -> str:
    """Insert code after a marker in original code"""
    lines = original_code.split('\n')
    for i, line in enumerate(lines):
        if marker in line:
            lines.insert(i + 1, new_code)
            return '\n'.join(lines)
    return original_code


def replace_method_in_class(original_code: str, class_name: str,
                           method_name: str, new_method_code: str) -> str:
    """Replace a method in a class"""
    try:
        tree = ast.parse(original_code)
        # This is a simplified version - real implementation would need AST manipulation
        # For now, just append
        return original_code + '\n' + new_method_code
    except:
        return original_code


# Example usage
if __name__ == '__main__':
    # Example parameter proposal
    example_proposal = {
        'name': 'harmony.voicing.quartal_probability',
        'type': 'CONTINUOUS',
        'range': (0.0, 1.0),
        'default': 0.3,
        'description': 'Probability of using quartal (fourth-based) voicings instead of tertian (third-based)',
        'musical_context': 'Quartal harmony is common in modal jazz and modern jazz. Higher values create more open, modern sounds.',
        'implementation_strategy': 'In voicing generation, check this probability before creating chord voicings. If triggered, build voicings in fourths instead of thirds.',
        'test_cases': [
            {'value': 0.0, 'expected': 'No quartal voicings'},
            {'value': 0.5, 'expected': 'Mix of quartal and tertian'},
            {'value': 1.0, 'expected': 'All quartal voicings'}
        ],
        'example_values': {
            'modal_jazz': 0.7,
            'bebop': 0.1,
            'fusion': 0.6
        }
    }

    # Create agent in mock mode
    agent = LLMCodeGenerationAgent(mock_mode=True)

    # Generate implementation
    implementation = agent.generate_implementation(example_proposal)

    # Print results
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\nParameter: {implementation.parameter_name}")
    print(f"Generation time: {implementation.generation_time:.2f}s")
    print(f"\nRegistry update ({len(implementation.registry_update)} chars):")
    print(implementation.registry_update[:500])
    print(f"\nGenerated {len(implementation.generator_modifications)} file modifications")
    print(f"Generated {len(implementation.new_methods)} new methods")
    print(f"Generated {len(implementation.test_code)} chars of test code")
