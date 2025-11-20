"""
LLM Code Generation Agent - Agent 12
====================================

Generates production-ready Python code to implement new parameters in the
85,989-line HarmonyModule generator using Claude AI.

This agent:
1. Analyzes parameter proposals from Agent 11
2. Identifies relevant generator files to modify
3. Extracts code context from the codebase
4. Uses Claude to generate implementation code
5. Validates and parses generated code
6. Ensures backward compatibility

Author: Agent 12 - Code Generation Agent
Date: 2025-11-20
License: MIT
"""

import ast
import json
import logging
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic library not available. Install with: pip install anthropic")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeSection(Enum):
    """Sections of generated code"""
    REGISTRY_UPDATE = "registry_update"
    GENERATOR_MODIFICATION = "generator_modification"
    NEW_METHOD = "new_method"
    TEST_CODE = "test_code"
    INTEGRATION_NOTES = "integration_notes"


@dataclass
class GeneratedCode:
    """Container for generated code"""
    registry_update: str = ""
    generator_modifications: Dict[str, str] = field(default_factory=dict)
    new_methods: Dict[str, str] = field(default_factory=dict)
    test_code: str = ""
    integration_notes: str = ""
    integration_points: List[str] = field(default_factory=list)
    documentation: str = ""
    migration_notes: str = ""


@dataclass
class CodeContext:
    """Context information about existing code"""
    file_path: str
    exists: bool = True
    full_content: str = ""
    classes: Dict[str, str] = field(default_factory=dict)
    key_methods: Dict[str, str] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    line_count: int = 0


@dataclass
class ValidationResult:
    """Result of code validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class CodebaseIndex:
    """
    Indexes the codebase for efficient searching and context extraction.
    """

    def __init__(self, root_path: str = None):
        """
        Initialize codebase index.

        Args:
            root_path: Root path of the codebase (defaults to midi_generator/)
        """
        if root_path is None:
            # Auto-detect root path
            current = Path(__file__).parent
            while current.name != 'midi_generator' and current.parent != current:
                current = current.parent
            if current.name == 'midi_generator':
                root_path = str(current)
            else:
                root_path = '/home/user/Do/midi_generator'

        self.root_path = Path(root_path)
        self.file_index: Dict[str, Path] = {}
        self.class_index: Dict[str, List[Path]] = {}
        self.method_index: Dict[str, List[Path]] = {}

        logger.info(f"Initializing codebase index at: {self.root_path}")
        self._build_index()

    def _build_index(self):
        """Build index of all Python files in the codebase"""
        logger.info("Building codebase index...")

        # Find all Python files
        for py_file in self.root_path.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue

            relative_path = py_file.relative_to(self.root_path)
            self.file_index[str(relative_path)] = py_file

            # Index classes and methods
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if node.name not in self.class_index:
                                self.class_index[node.name] = []
                            self.class_index[node.name].append(py_file)

                        elif isinstance(node, ast.FunctionDef):
                            if node.name not in self.method_index:
                                self.method_index[node.name] = []
                            self.method_index[node.name].append(py_file)

            except Exception as e:
                logger.warning(f"Failed to index {py_file}: {e}")

        logger.info(f"Indexed {len(self.file_index)} files, "
                   f"{len(self.class_index)} classes, "
                   f"{len(self.method_index)} methods")

    def find_file(self, pattern: str) -> List[Path]:
        """
        Find files matching a pattern.

        Args:
            pattern: File name pattern (e.g., "*_generator.py")

        Returns:
            List of matching file paths
        """
        matches = []
        for file_path in self.file_index.values():
            if file_path.match(pattern):
                matches.append(file_path)
        return matches

    def find_class(self, class_name: str) -> List[Path]:
        """Find files containing a class"""
        return self.class_index.get(class_name, [])

    def find_method(self, method_name: str) -> List[Path]:
        """Find files containing a method"""
        return self.method_index.get(method_name, [])

    def get_file_path(self, relative_path: str) -> Optional[Path]:
        """Get absolute path for a relative path"""
        return self.file_index.get(relative_path)


class CodeExtractor:
    """
    Extracts code elements from Python source files using AST.
    """

    @staticmethod
    def extract_classes(code: str) -> Dict[str, str]:
        """
        Extract class definitions from code.

        Args:
            code: Python source code

        Returns:
            Dictionary mapping class names to their source code
        """
        classes = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    try:
                        class_source = ast.get_source_segment(code, node)
                        if class_source:
                            classes[node.name] = class_source
                    except Exception as e:
                        logger.warning(f"Failed to extract class {node.name}: {e}")
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")

        return classes

    @staticmethod
    def extract_methods(code: str, filter_patterns: List[str] = None) -> Dict[str, str]:
        """
        Extract method definitions from code.

        Args:
            code: Python source code
            filter_patterns: List of patterns to match method names against

        Returns:
            Dictionary mapping method names to their source code
        """
        if filter_patterns is None:
            filter_patterns = ['generate', 'create', '_build', '_apply',
                             '_process', '_handle', '_render']

        methods = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if method matches any filter pattern
                    if any(pattern in node.name for pattern in filter_patterns):
                        try:
                            method_source = ast.get_source_segment(code, node)
                            if method_source:
                                methods[node.name] = method_source
                        except Exception as e:
                            logger.warning(f"Failed to extract method {node.name}: {e}")
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")

        return methods

    @staticmethod
    def extract_imports(code: str) -> List[str]:
        """
        Extract import statements from code.

        Args:
            code: Python source code

        Returns:
            List of import statements
        """
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_str = f"import {alias.name}"
                        if alias.asname:
                            import_str += f" as {alias.asname}"
                        imports.append(import_str)

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ', '.join([
                        f"{alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                        for alias in node.names
                    ])
                    imports.append(f"from {module} import {names}")

        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")

        return imports

    @staticmethod
    def extract_method_name(code: str) -> Optional[str]:
        """
        Extract method/function name from code snippet.

        Args:
            code: Python code containing a function/method definition

        Returns:
            Method name or None if not found
        """
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else None

    @staticmethod
    def extract_docstring(code: str) -> Optional[str]:
        """
        Extract docstring from code.

        Args:
            code: Python source code

        Returns:
            Docstring or None if not found
        """
        try:
            tree = ast.parse(code)
            if isinstance(tree.body[0], (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(tree.body[0])
                return docstring
        except Exception:
            pass
        return None


class CodeValidator:
    """
    Validates generated code for correctness and compatibility.
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []

    def validate(self, generated_code: GeneratedCode) -> ValidationResult:
        """
        Validate all generated code.

        Args:
            generated_code: Generated code to validate

        Returns:
            ValidationResult with errors, warnings, suggestions
        """
        self.errors = []
        self.warnings = []
        self.suggestions = []

        # 1. Validate registry update
        self._validate_registry_update(generated_code.registry_update)

        # 2. Validate generator modifications
        for file_path, code in generated_code.generator_modifications.items():
            self._validate_generator_code(file_path, code)

        # 3. Validate new methods
        for method_name, code in generated_code.new_methods.items():
            self._validate_method_code(method_name, code)

        # 4. Validate test code
        self._validate_test_code(generated_code.test_code)

        # 5. Check backward compatibility
        self._check_backward_compatibility(generated_code)

        return ValidationResult(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            suggestions=self.suggestions
        )

    def _validate_registry_update(self, code: str):
        """Validate parameter registry update"""
        if not code.strip():
            self.errors.append("Registry update is empty")
            return

        # Check for Parameter definition
        if 'Parameter(' not in code and 'ParameterDefinition(' not in code:
            self.errors.append("Registry update missing Parameter() or ParameterDefinition()")

        # Check syntax
        try:
            compile(code, '<registry>', 'exec')
        except SyntaxError as e:
            self.errors.append(f"Syntax error in registry update: {e}")

    def _validate_generator_code(self, file_path: str, code: str):
        """Validate generator modification code"""
        if not code.strip():
            self.warnings.append(f"Empty code for {file_path}")
            return

        # Check syntax
        try:
            compile(code, file_path, 'exec')
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {file_path}: {e}")

        # Check for .get() usage (backward compatibility)
        if '.get(' not in code and 'params[' in code:
            self.warnings.append(
                f"{file_path}: Direct dict access found. Use .get() for backward compatibility"
            )

        # Check for validation
        if 'max(' not in code and 'min(' not in code:
            self.suggestions.append(
                f"{file_path}: Consider adding value validation (min/max)"
            )

    def _validate_method_code(self, method_name: str, code: str):
        """Validate new method code"""
        if not code.strip():
            self.errors.append(f"Empty code for method {method_name}")
            return

        # Check syntax
        try:
            compile(code, f'<method:{method_name}>', 'exec')
        except SyntaxError as e:
            self.errors.append(f"Syntax error in method {method_name}: {e}")

        # Check for docstring
        if '"""' not in code and "'''" not in code:
            self.warnings.append(f"Method {method_name} missing docstring")

        # Check for type hints
        if '->' not in code:
            self.suggestions.append(f"Method {method_name} could benefit from type hints")

    def _validate_test_code(self, code: str):
        """Validate test code"""
        if not code.strip():
            self.errors.append("No test code provided")
            return

        # Check syntax
        try:
            compile(code, '<tests>', 'exec')
        except SyntaxError as e:
            self.errors.append(f"Syntax error in test code: {e}")

        # Check for test functions
        if 'def test_' not in code:
            self.warnings.append("Test code missing test functions (def test_*)")

        # Check for assertions
        if 'assert' not in code:
            self.warnings.append("Test code missing assertions")

    def _check_backward_compatibility(self, generated_code: GeneratedCode):
        """Check for backward compatibility issues"""
        all_code = (
            generated_code.registry_update + "\n" +
            "\n".join(generated_code.generator_modifications.values()) +
            "\n".join(generated_code.new_methods.values())
        )

        # Check for required imports
        if 'from parameters' in all_code and 'import' not in all_code:
            self.warnings.append("Missing import statements")

        # Check for default values in .get() calls
        get_pattern = r'\.get\([\'"]([^\'"]+)[\'"](?:,\s*([^)]+))?\)'
        matches = re.findall(get_pattern, all_code)

        for param_name, default in matches:
            if not default:
                self.warnings.append(
                    f"Parameter '{param_name}' accessed without default value"
                )


class LLMCodeGenerationAgent:
    """
    Main agent for LLM-powered code generation.

    Uses Claude to generate production-ready Python code for implementing
    new parameters in the music generation system.
    """

    def __init__(self, api_key: str = None, root_path: str = None):
        """
        Initialize the code generation agent.

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            root_path: Root path of the codebase
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

        # Initialize Claude client
        self.llm = anthropic.Anthropic(api_key=api_key)

        # Initialize codebase index
        self.codebase_index = CodebaseIndex(root_path)

        # Initialize extractor and validator
        self.extractor = CodeExtractor()
        self.validator = CodeValidator()

        # Load code patterns
        self.code_patterns = self._load_code_patterns()

        logger.info("LLMCodeGenerationAgent initialized")

    def generate_implementation(self, parameter_proposal: dict) -> GeneratedCode:
        """
        Generate complete implementation code for new parameter.

        Args:
            parameter_proposal: Parameter proposal from Agent 11

        Returns:
            GeneratedCode object with all generated code sections

        Example:
            >>> agent = LLMCodeGenerationAgent()
            >>> proposal = {
            ...     'name': 'harmony.voicing.quartal_probability',
            ...     'type': 'continuous',
            ...     'range': (0.0, 1.0),
            ...     'default': 0.15,
            ...     'description': 'Probability of using quartal voicings',
            ...     'musical_context': 'Jazz voicings built in fourths',
            ...     'implementation_strategy': '...',
            ...     'affected_features': [...],
            ...     'test_cases': [...]
            ... }
            >>> code = agent.generate_implementation(proposal)
        """
        logger.info(f"Generating implementation for parameter: {parameter_proposal.get('name')}")

        # 1. Identify relevant files
        relevant_files = self._identify_relevant_files(parameter_proposal)
        logger.info(f"Identified {len(relevant_files)} relevant files")

        # 2. Build code context
        code_context = self._build_code_context(relevant_files)
        logger.info(f"Built code context for {len(code_context)} files")

        # 3. Generate implementation via LLM
        generated_code = self._llm_generate_code(parameter_proposal, code_context)
        logger.info("Generated code via LLM")

        # 4. Validate generated code
        validation_result = self.validator.validate(generated_code)

        if not validation_result.valid:
            logger.error(f"Generated code validation failed: {validation_result.errors}")
            raise ValueError(f"Code validation failed: {validation_result.errors}")

        if validation_result.warnings:
            logger.warning(f"Code validation warnings: {validation_result.warnings}")

        if validation_result.suggestions:
            logger.info(f"Code suggestions: {validation_result.suggestions}")

        logger.info("Code generation completed successfully")
        return generated_code

    def _identify_relevant_files(self, proposal: dict) -> List[str]:
        """
        Identify which generator files need modification.

        Args:
            proposal: Parameter proposal

        Returns:
            List of relative file paths
        """
        param_name = proposal['name']
        domain = param_name.split('.')[0]

        # Map domain to generator files
        domain_to_files = {
            'harmony': [
                'generators/advanced_harmony_generator.py',
                'generators/harmonic_rhythm.py',
                'generators/reharmonization_engine.py',
                'core/modal_harmony.py',
                'core/neo_riemannian.py'
            ],
            'melody': [
                'generators/context_aware_generator.py',
                'generators/granular_control.py'
            ],
            'rhythm': [
                'algorithms/rhythm_engine.py',
                'algorithms/advanced_rhythm.py',
                'algorithms/groove_library.py',
                'algorithms/drum_patterns.py'
            ],
            'instrumentation': [
                'generators/orchestrator.py',
                'generators/texture_generator.py',
                'core/instrument_library.py',
                'core/ensemble_registry.py'
            ],
            'dynamics': [
                'generators/granular_control.py'
            ],
            'structure': [
                'generators/form_generator.py',
                'generators/development_engine.py',
                'generators/transition_engine.py'
            ],
            'bass': [
                'generators/context_aware_generator.py'
            ],
            'drums': [
                'algorithms/drum_patterns.py'
            ],
            'articulation': [
                'midi/articulation_engine.py'
            ],
            'timbre': [
                'generators/texture_generator.py',
                'core/instrument_library.py'
            ],
            'style': [
                'generators/style_fusion.py',
                'core/multi_genre_arranger.py'
            ],
            'genre': [
                'genres/blues.py',
                'core/multi_genre_arranger.py'
            ]
        }

        # Get base files for domain
        files = domain_to_files.get(domain, [])

        # Always include parameter registry
        files.append('parameters/universal_registry.py')

        # Check integration points from proposal
        if 'generator_integration_points' in proposal:
            for point in proposal['generator_integration_points']:
                file_path = point.split('::')[0]
                if file_path not in files:
                    files.append(file_path)

        # Remove duplicates and filter existing files
        unique_files = []
        for file_path in files:
            if file_path not in unique_files:
                # Check if file exists
                full_path = self.codebase_index.root_path / file_path
                if full_path.exists():
                    unique_files.append(file_path)
                else:
                    logger.warning(f"File does not exist: {file_path}")

        return unique_files

    def _build_code_context(self, file_paths: List[str]) -> Dict[str, CodeContext]:
        """
        Load relevant code excerpts for LLM context.

        Args:
            file_paths: List of relative file paths

        Returns:
            Dictionary mapping file paths to CodeContext objects
        """
        context = {}

        for file_path in file_paths:
            full_path = self.codebase_index.root_path / file_path

            if not full_path.exists():
                context[file_path] = CodeContext(
                    file_path=file_path,
                    exists=False
                )
                continue

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract code elements
                classes = self.extractor.extract_classes(content)
                methods = self.extractor.extract_methods(content)
                imports = self.extractor.extract_imports(content)

                context[file_path] = CodeContext(
                    file_path=file_path,
                    exists=True,
                    full_content=content,
                    classes=classes,
                    key_methods=methods,
                    imports=imports,
                    line_count=len(content.split('\n'))
                )

                logger.debug(f"Extracted context from {file_path}: "
                           f"{len(classes)} classes, {len(methods)} methods")

            except Exception as e:
                logger.error(f"Failed to load context from {file_path}: {e}")
                context[file_path] = CodeContext(
                    file_path=file_path,
                    exists=False
                )

        return context

    def _llm_generate_code(
        self,
        proposal: dict,
        context: Dict[str, CodeContext]
    ) -> GeneratedCode:
        """
        Use Claude to generate implementation code.

        Args:
            proposal: Parameter proposal
            context: Code context from existing files

        Returns:
            GeneratedCode object
        """
        logger.info("Generating code via Claude API...")

        # Build prompts
        system_prompt = self._get_code_generation_system_prompt()
        user_prompt = self._build_code_generation_prompt(proposal, context)

        try:
            # Call Claude API
            response = self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.2
            )

            # Extract response text
            response_text = response.content[0].text

            # Parse response into structured code
            generated_code = self._parse_code_response(response_text)

            logger.info("Successfully generated code via Claude")
            return generated_code

        except Exception as e:
            logger.error(f"Failed to generate code via Claude: {e}")
            raise

    def _get_code_generation_system_prompt(self) -> str:
        """
        Get system prompt for code generation.

        Returns:
            System prompt string
        """
        return """
You are an expert Python developer working on a large-scale music generation system.

Your task is to generate production-ready code to implement a new parameter in the existing generator.

CODE QUALITY REQUIREMENTS:

1. Backward Compatibility: CRITICAL
   - ALL existing parameters must continue to work
   - Use .get() with defaults for ALL parameter access
   - Never break existing functionality

2. Code Style:
   - Follow existing naming conventions
   - Match indentation and formatting
   - Use type hints
   - Add comprehensive docstrings

3. Error Handling:
   - Validate parameter values
   - Handle edge cases (0.0, 1.0, None, etc.)
   - Graceful degradation if parameter missing

4. Testing:
   - Provide unit tests
   - Test edge cases
   - Test integration with other parameters

5. Documentation:
   - Clear docstrings
   - Inline comments for complex logic
   - Parameter usage examples

IMPLEMENTATION PATTERNS:

# PATTERN 1: Probability-based parameter
def generate_something(self, params: Dict[str, Any]):
    # Get parameter with default
    prob = params.get('domain.module.param_prob', 0.3)

    # Validate range
    prob = max(0.0, min(1.0, prob))

    # Use probability
    if random.random() < prob:
        return self._do_special_thing()
    else:
        return self._do_normal_thing()

# PATTERN 2: Categorical parameter
def generate_something(self, params: Dict[str, Any]):
    choice = params.get('domain.module.param_type', 'default')

    if choice == 'option1':
        return self._method1()
    elif choice == 'option2':
        return self._method2()
    else:
        return self._default_method()

# PATTERN 3: Continuous value parameter
def generate_something(self, params: Dict[str, Any]):
    density = params.get('domain.module.density', 0.5)
    density = max(0.0, min(1.0, density))

    # Use value in calculation
    num_notes = int(density * max_notes)
    return self._generate_notes(num_notes)

REGISTRY UPDATE PATTERN:

# parameters/universal_registry.py

DOMAIN_PARAMETERS = {
    # ...existing parameters...

    "domain.module.new_parameter": ParameterDefinition(
        name="new_parameter",
        full_path="domain.module.new_parameter",
        description="Clear description of what parameter does",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "fusion"]
    ),
}

OUTPUT FORMAT:

Provide code in structured sections with clear markers:

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

CRITICAL:
- Generate COMPLETE, RUNNABLE code
- NO placeholders like "# TODO" or "# implement this"
- ALL methods must be fully implemented
- Include ALL necessary imports
"""

    def _build_code_generation_prompt(
        self,
        proposal: dict,
        context: Dict[str, CodeContext]
    ) -> str:
        """
        Build specific prompt for this parameter.

        Args:
            proposal: Parameter proposal
            context: Code context

        Returns:
            User prompt string
        """
        prompt = f"""
PARAMETER TO IMPLEMENT:

Name: {proposal['name']}
Type: {proposal['type']}
Range: {proposal.get('range', 'N/A')}
Default: {proposal['default']}

Description: {proposal['description']}

Musical Context: {proposal.get('musical_context', 'N/A')}

Implementation Strategy: {proposal.get('implementation_strategy', 'N/A')}

Affected Features: {json.dumps(proposal.get('affected_features', []), indent=2)}

Test Cases:
{json.dumps(proposal.get('test_cases', []), indent=2)}

Example Values:
{json.dumps(proposal.get('example_values', {}), indent=2)}

EXISTING CODEBASE CONTEXT:
"""

        # Add relevant code context (limit to avoid token overflow)
        max_context_per_file = 2000  # characters

        for file_path, file_context in context.items():
            if not file_context.exists:
                prompt += f"\n{file_path}: [NEW FILE - does not exist yet]\n"
                continue

            prompt += f"\n\n# {file_path}\n"
            prompt += f"Lines: {file_context.line_count}\n"

            # Add imports
            if file_context.imports:
                prompt += "\nImports:\n"
                for imp in file_context.imports[:10]:  # Limit imports
                    prompt += f"  {imp}\n"

            # Add key methods (truncated)
            if file_context.key_methods:
                prompt += "\nKey methods:\n"
                for method_name, method_code in list(file_context.key_methods.items())[:3]:
                    # Truncate long methods
                    if len(method_code) > max_context_per_file:
                        method_code = method_code[:max_context_per_file] + "\n    # ... [truncated]"
                    prompt += f"\n```python\n{method_code}\n```\n"

        prompt += """

YOUR TASK:

Generate COMPLETE implementation code for this parameter following these steps:

1. Registry Update: Add parameter to parameters/universal_registry.py

2. Generator Modifications: Modify existing generator methods to use the parameter
   - Identify the right place to check the parameter
   - Use .get() with default value
   - Implement the musical logic described

3. New Methods: Create any helper methods needed
   - Full implementation, no placeholders
   - Docstrings with examples
   - Error handling

4. Tests: Write comprehensive unit tests
   - Test all values in test_cases
   - Test edge cases (0.0, 1.0, None)
   - Test integration with other parameters

5. Integration Notes: Document where and how parameter is used

Generate code following the OUTPUT FORMAT specified in your system prompt.

CRITICAL REQUIREMENTS:
- BACKWARD COMPATIBLE: All existing parameters must still work
- COMPLETE: No TODOs or placeholders
- TESTED: Include working unit tests
- DOCUMENTED: Clear docstrings and comments
"""

        return prompt

    def _parse_code_response(self, response_text: str) -> GeneratedCode:
        """
        Parse LLM response into structured code.

        Args:
            response_text: Response from Claude

        Returns:
            GeneratedCode object
        """
        generated_code = GeneratedCode()

        # Split response by section markers
        current_section = None
        current_file = None
        current_buffer = []

        lines = response_text.split('\n')

        for line in lines:
            # Check for section markers
            if '=== SECTION 1: REGISTRY UPDATE ===' in line:
                # Save previous section
                self._save_section(
                    generated_code, current_section, current_file, current_buffer
                )
                current_section = CodeSection.REGISTRY_UPDATE
                current_file = None
                current_buffer = []

            elif '=== SECTION 2: GENERATOR MODIFICATION ===' in line:
                self._save_section(
                    generated_code, current_section, current_file, current_buffer
                )
                current_section = CodeSection.GENERATOR_MODIFICATION
                current_file = None
                current_buffer = []

            elif '=== SECTION 3: NEW METHOD ===' in line:
                self._save_section(
                    generated_code, current_section, current_file, current_buffer
                )
                current_section = CodeSection.NEW_METHOD
                current_file = None
                current_buffer = []

            elif '=== SECTION 4: TESTS ===' in line:
                self._save_section(
                    generated_code, current_section, current_file, current_buffer
                )
                current_section = CodeSection.TEST_CODE
                current_file = None
                current_buffer = []

            elif '=== SECTION 5: INTEGRATION NOTES ===' in line:
                self._save_section(
                    generated_code, current_section, current_file, current_buffer
                )
                current_section = CodeSection.INTEGRATION_NOTES
                current_file = None
                current_buffer = []

            # Check for file markers
            elif line.startswith('# File: '):
                # Save previous file
                if current_section == CodeSection.GENERATOR_MODIFICATION and current_file:
                    generated_code.generator_modifications[current_file] = '\n'.join(current_buffer)
                    current_buffer = []

                current_file = line.replace('# File: ', '').strip()

            # Accumulate code
            else:
                current_buffer.append(line)

        # Save final section
        self._save_section(
            generated_code, current_section, current_file, current_buffer
        )

        return generated_code

    def _save_section(
        self,
        generated_code: GeneratedCode,
        section: Optional[CodeSection],
        file_path: Optional[str],
        buffer: List[str]
    ):
        """
        Save accumulated buffer to appropriate section.

        Args:
            generated_code: GeneratedCode object to update
            section: Current section type
            file_path: Current file path (for modifications)
            buffer: Accumulated lines
        """
        if not section or not buffer:
            return

        content = '\n'.join(buffer).strip()

        if section == CodeSection.REGISTRY_UPDATE:
            generated_code.registry_update = content

        elif section == CodeSection.GENERATOR_MODIFICATION:
            if file_path:
                generated_code.generator_modifications[file_path] = content

        elif section == CodeSection.NEW_METHOD:
            # Extract method name
            method_name = self.extractor.extract_method_name(content)
            if method_name:
                generated_code.new_methods[method_name] = content

        elif section == CodeSection.TEST_CODE:
            generated_code.test_code = content

        elif section == CodeSection.INTEGRATION_NOTES:
            generated_code.integration_notes = content

    def _load_code_patterns(self) -> Dict[str, str]:
        """
        Load common code patterns for reference.

        Returns:
            Dictionary of pattern name to code template
        """
        return {
            'probability': '''
def generate_{name}(self, params: Dict[str, Any]) -> Any:
    """Generate {description}"""
    prob = params.get('{param_path}', {default})
    prob = max(0.0, min(1.0, prob))

    if random.random() < prob:
        return self._generate_variant()
    else:
        return self._generate_default()
''',
            'categorical': '''
def generate_{name}(self, params: Dict[str, Any]) -> Any:
    """Generate {description}"""
    choice = params.get('{param_path}', '{default}')

    if choice == 'option1':
        return self._method1()
    elif choice == 'option2':
        return self._method2()
    else:
        return self._default_method()
''',
            'continuous': '''
def generate_{name}(self, params: Dict[str, Any]) -> Any:
    """Generate {description}"""
    value = params.get('{param_path}', {default})
    value = max({min_val}, min({max_val}, value))

    # Use value in calculation
    result = self._calculate_with_value(value)
    return result
'''
        }

    def save_generated_code(
        self,
        generated_code: GeneratedCode,
        output_dir: str = None
    ) -> Dict[str, str]:
        """
        Save generated code to files.

        Args:
            generated_code: GeneratedCode object
            output_dir: Output directory (defaults to codebase root)

        Returns:
            Dictionary mapping file paths to written content
        """
        if output_dir is None:
            output_dir = str(self.codebase_index.root_path)

        output_dir = Path(output_dir)
        written_files = {}

        # Save registry update
        if generated_code.registry_update:
            registry_path = output_dir / 'parameters' / 'universal_registry.py'
            logger.info(f"Updating registry: {registry_path}")
            # Note: This would append to the file in practice
            written_files[str(registry_path)] = generated_code.registry_update

        # Save generator modifications
        for file_path, code in generated_code.generator_modifications.items():
            full_path = output_dir / file_path
            logger.info(f"Modifying generator: {full_path}")
            written_files[str(full_path)] = code

        # Save test code
        if generated_code.test_code:
            test_path = output_dir / 'tests' / 'test_generated_parameter.py'
            logger.info(f"Writing tests: {test_path}")
            written_files[str(test_path)] = generated_code.test_code

        logger.info(f"Prepared {len(written_files)} files for writing")
        return written_files


# Example usage
if __name__ == "__main__":
    # Example parameter proposal
    example_proposal = {
        'name': 'harmony.voicing.quartal_probability',
        'type': 'continuous',
        'range': (0.0, 1.0),
        'default': 0.15,
        'description': 'Probability of using quartal voicings (fourths) instead of tertian',
        'musical_context': 'Jazz voicings built in fourths instead of thirds. Common in modern jazz and fusion.',
        'implementation_strategy': 'Add probability check in voicing generation. When triggered, build chords in stacks of fourths.',
        'affected_features': [
            'chord_density',
            'voicing_openness',
            'harmonic_color'
        ],
        'test_cases': [
            {'value': 0.0, 'expected': 'No quartal voicings'},
            {'value': 0.5, 'expected': '50% quartal voicings'},
            {'value': 1.0, 'expected': 'All quartal voicings'}
        ],
        'example_values': {
            'traditional_jazz': 0.1,
            'modern_jazz': 0.3,
            'fusion': 0.5,
            'avant_garde': 0.8
        },
        'generator_integration_points': [
            'generators/advanced_harmony_generator.py::generate_voicing()',
            'core/modal_harmony.py::create_chord_voicing()'
        ]
    }

    print("LLM Code Generation Agent - Example")
    print("=" * 60)
    print("\nThis module generates code for new parameters.")
    print("\nTo use:")
    print("1. Set ANTHROPIC_API_KEY environment variable")
    print("2. Create parameter proposal (see example above)")
    print("3. Call agent.generate_implementation(proposal)")
    print("\nExample proposal:")
    print(json.dumps(example_proposal, indent=2))
