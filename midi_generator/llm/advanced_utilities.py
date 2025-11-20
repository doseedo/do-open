"""
Advanced Utilities for Code Generation - Agent 12
=================================================

Advanced utilities for code generation, analysis, and integration:
- Dependency analysis
- Code merging
- Documentation generation
- Migration planning
- Impact analysis

Author: Agent 12 - Code Generation Agent
Date: 2025-11-20
License: MIT
"""

import ast
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Dependency:
    """Represents a dependency between parameters or modules"""
    source: str
    target: str
    dependency_type: str  # 'requires', 'conflicts', 'enhances', 'modifies'
    strength: float = 1.0  # 0.0 to 1.0
    description: str = ""


@dataclass
class ImpactAnalysis:
    """Analysis of parameter impact on the system"""
    parameter_name: str
    affected_files: List[str] = field(default_factory=list)
    affected_parameters: List[str] = field(default_factory=list)
    affected_features: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    risk_level: str = "low"  # low, medium, high
    testing_requirements: List[str] = field(default_factory=list)
    migration_steps: List[str] = field(default_factory=list)


@dataclass
class CodeMergeResult:
    """Result of merging code into existing files"""
    success: bool
    merged_files: Dict[str, str] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DependencyAnalyzer:
    """
    Analyzes dependencies between parameters and modules.
    """

    def __init__(self):
        self.dependency_graph: Dict[str, List[Dependency]] = defaultdict(list)
        self.parameter_to_files: Dict[str, Set[str]] = defaultdict(set)

    def add_dependency(self, dependency: Dependency):
        """Add a dependency to the graph"""
        self.dependency_graph[dependency.source].append(dependency)

    def analyze_parameter_dependencies(
        self,
        parameter_name: str,
        parameter_registry: Dict[str, Any]
    ) -> List[Dependency]:
        """
        Analyze dependencies for a parameter.

        Args:
            parameter_name: Parameter to analyze
            parameter_registry: Full parameter registry

        Returns:
            List of dependencies
        """
        dependencies = []

        if parameter_name not in parameter_registry:
            return dependencies

        param_def = parameter_registry[parameter_name]

        # Check explicit dependencies
        if hasattr(param_def, 'depends_on'):
            for dep in param_def.depends_on:
                dependencies.append(Dependency(
                    source=parameter_name,
                    target=dep,
                    dependency_type='requires',
                    strength=1.0,
                    description=f"{parameter_name} requires {dep}"
                ))

        # Check mutual exclusions
        if hasattr(param_def, 'mutually_exclusive_with'):
            for exclusive in param_def.mutually_exclusive_with:
                dependencies.append(Dependency(
                    source=parameter_name,
                    target=exclusive,
                    dependency_type='conflicts',
                    strength=1.0,
                    description=f"{parameter_name} conflicts with {exclusive}"
                ))

        # Infer domain-level dependencies
        domain = parameter_name.split('.')[0]
        for other_param in parameter_registry:
            if other_param == parameter_name:
                continue

            other_domain = other_param.split('.')[0]

            # Same domain = potential enhancement
            if other_domain == domain:
                # Check for semantic relationships
                if self._are_related(parameter_name, other_param):
                    dependencies.append(Dependency(
                        source=parameter_name,
                        target=other_param,
                        dependency_type='enhances',
                        strength=0.5,
                        description=f"{parameter_name} may enhance {other_param}"
                    ))

        return dependencies

    def _are_related(self, param1: str, param2: str) -> bool:
        """Check if two parameters are semantically related"""
        # Extract keywords from parameter names
        keywords1 = set(re.findall(r'\w+', param1.lower()))
        keywords2 = set(re.findall(r'\w+', param2.lower()))

        # Check for common keywords (excluding domain)
        common = keywords1 & keywords2
        common.discard(param1.split('.')[0])  # Remove domain

        return len(common) > 0

    def find_dependency_chain(
        self,
        source: str,
        target: str
    ) -> Optional[List[str]]:
        """
        Find dependency chain from source to target.

        Args:
            source: Source parameter
            target: Target parameter

        Returns:
            List of parameters in chain, or None if no chain exists
        """
        # BFS to find shortest path
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if current == target:
                return path

            for dep in self.dependency_graph.get(current, []):
                if dep.target not in visited:
                    visited.add(dep.target)
                    queue.append((dep.target, path + [dep.target]))

        return None

    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the graph.

        Returns:
            List of circular dependency chains
        """
        cycles = []

        def dfs(node: str, path: List[str], visited: Set[str]):
            if node in visited:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                cycles.append(cycle)
                return

            visited.add(node)
            path.append(node)

            for dep in self.dependency_graph.get(node, []):
                dfs(dep.target, path[:], visited.copy())

        for node in self.dependency_graph:
            dfs(node, [], set())

        return cycles

    def compute_dependency_depth(self, parameter: str) -> int:
        """
        Compute depth of dependency tree for a parameter.

        Args:
            parameter: Parameter to analyze

        Returns:
            Maximum depth of dependency tree
        """
        def depth(node: str, visited: Set[str]) -> int:
            if node in visited:
                return 0

            visited.add(node)

            max_child_depth = 0
            for dep in self.dependency_graph.get(node, []):
                child_depth = depth(dep.target, visited.copy())
                max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth + 1

        return depth(parameter, set())


class CodeMerger:
    """
    Intelligently merges generated code into existing files.
    """

    def __init__(self):
        self.conflicts: List[str] = []
        self.warnings: List[str] = []

    def merge_into_file(
        self,
        file_path: str,
        new_code: str,
        merge_strategy: str = 'append'
    ) -> CodeMergeResult:
        """
        Merge new code into an existing file.

        Args:
            file_path: Path to existing file
            new_code: Code to merge
            merge_strategy: 'append', 'replace', 'smart'

        Returns:
            CodeMergeResult
        """
        self.conflicts = []
        self.warnings = []

        path = Path(file_path)

        # Read existing file
        if not path.exists():
            # New file - just write
            return CodeMergeResult(
                success=True,
                merged_files={file_path: new_code}
            )

        try:
            with open(path, 'r', encoding='utf-8') as f:
                existing_code = f.read()
        except Exception as e:
            self.conflicts.append(f"Failed to read {file_path}: {e}")
            return CodeMergeResult(success=False, conflicts=self.conflicts)

        # Perform merge based on strategy
        if merge_strategy == 'append':
            merged = self._merge_append(existing_code, new_code)
        elif merge_strategy == 'replace':
            merged = self._merge_replace(existing_code, new_code, file_path)
        elif merge_strategy == 'smart':
            merged = self._merge_smart(existing_code, new_code, file_path)
        else:
            self.conflicts.append(f"Unknown merge strategy: {merge_strategy}")
            return CodeMergeResult(success=False, conflicts=self.conflicts)

        if merged is None:
            return CodeMergeResult(
                success=False,
                conflicts=self.conflicts,
                warnings=self.warnings
            )

        return CodeMergeResult(
            success=True,
            merged_files={file_path: merged},
            warnings=self.warnings
        )

    def _merge_append(self, existing: str, new: str) -> str:
        """Append new code to existing file"""
        # Add separator
        separator = "\n\n# " + "=" * 70 + "\n"
        separator += "# Auto-generated code - Agent 12\n"
        separator += "# " + "=" * 70 + "\n\n"

        return existing + separator + new

    def _merge_replace(self, existing: str, new: str, file_path: str) -> Optional[str]:
        """Replace specific sections in existing file"""
        # Try to identify what to replace
        # Look for method definitions in new code

        try:
            new_tree = ast.parse(new)
        except SyntaxError:
            self.conflicts.append(f"Syntax error in new code for {file_path}")
            return None

        merged = existing

        # Extract methods from new code
        for node in ast.walk(new_tree):
            if isinstance(node, ast.FunctionDef):
                # Try to replace this method in existing code
                method_name = node.name
                merged = self._replace_method(merged, method_name, new)

        return merged

    def _merge_smart(self, existing: str, new: str, file_path: str) -> Optional[str]:
        """Smart merge - analyze and merge intelligently"""
        try:
            existing_tree = ast.parse(existing)
            new_tree = ast.parse(new)
        except SyntaxError as e:
            self.conflicts.append(f"Syntax error: {e}")
            return None

        # Extract elements from both
        existing_methods = self._extract_method_names(existing_tree)
        new_methods = self._extract_method_names(new_tree)

        # Check for conflicts
        conflicts = existing_methods & new_methods
        if conflicts:
            self.warnings.append(
                f"Method name conflicts in {file_path}: {conflicts}"
            )
            # Could attempt to merge or rename

        # For now, append non-conflicting methods
        merged = existing

        for node in ast.walk(new_tree):
            if isinstance(node, ast.FunctionDef):
                if node.name not in existing_methods:
                    # Safe to add
                    method_code = ast.get_source_segment(new, node)
                    if method_code:
                        merged += f"\n\n{method_code}"

        return merged

    def _replace_method(self, code: str, method_name: str, new_method_code: str) -> str:
        """Replace a method in code"""
        # Simple regex-based replacement
        # This is simplistic - a full AST-based approach would be better

        pattern = rf'def {method_name}\([^)]*\):.*?(?=\n(?:def |class |\Z))'
        match = re.search(pattern, code, re.DOTALL)

        if match:
            # Replace the method
            return code[:match.start()] + new_method_code + code[match.end():]
        else:
            # Method not found - append
            return code + f"\n\n{new_method_code}"

    def _extract_method_names(self, tree: ast.AST) -> Set[str]:
        """Extract all method names from AST"""
        methods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                methods.add(node.name)
        return methods


class DocumentationGenerator:
    """
    Generates documentation for new parameters and code.
    """

    def generate_parameter_doc(
        self,
        parameter_name: str,
        parameter_def: Any,
        code_locations: List[str]
    ) -> str:
        """
        Generate markdown documentation for a parameter.

        Args:
            parameter_name: Parameter name
            parameter_def: Parameter definition
            code_locations: List of code locations where used

        Returns:
            Markdown documentation string
        """
        doc = f"# Parameter: `{parameter_name}`\n\n"

        # Description
        if hasattr(parameter_def, 'description'):
            doc += f"## Description\n\n{parameter_def.description}\n\n"

        # Type and range
        doc += "## Specification\n\n"
        if hasattr(parameter_def, 'param_type'):
            doc += f"- **Type**: {parameter_def.param_type.value}\n"
        if hasattr(parameter_def, 'default_value'):
            doc += f"- **Default**: {parameter_def.default_value}\n"
        if hasattr(parameter_def, 'min_value') and parameter_def.min_value is not None:
            doc += f"- **Range**: [{parameter_def.min_value}, {parameter_def.max_value}]\n"
        if hasattr(parameter_def, 'options') and parameter_def.options:
            doc += f"- **Options**: {', '.join(map(str, parameter_def.options))}\n"

        doc += "\n"

        # Musical context
        if hasattr(parameter_def, 'musical_impact'):
            doc += "## Musical Context\n\n"
            doc += f"- **Impact Level**: {parameter_def.musical_impact.value}\n"

        if hasattr(parameter_def, 'genre_relevance') and parameter_def.genre_relevance:
            doc += f"- **Relevant Genres**: {', '.join(parameter_def.genre_relevance)}\n"

        doc += "\n"

        # Usage locations
        if code_locations:
            doc += "## Implementation\n\n"
            doc += "This parameter is used in:\n\n"
            for location in code_locations:
                doc += f"- `{location}`\n"
            doc += "\n"

        # Dependencies
        if hasattr(parameter_def, 'depends_on') and parameter_def.depends_on:
            doc += "## Dependencies\n\n"
            doc += "This parameter depends on:\n\n"
            for dep in parameter_def.depends_on:
                doc += f"- `{dep}`\n"
            doc += "\n"

        # Examples
        doc += "## Example Usage\n\n"
        doc += "```python\n"
        doc += "params = {\n"
        doc += f"    '{parameter_name}': {parameter_def.default_value},\n"
        doc += "    # ... other parameters\n"
        doc += "}\n"
        doc += "generator.generate(params)\n"
        doc += "```\n\n"

        return doc

    def generate_api_doc(
        self,
        class_name: str,
        methods: Dict[str, str]
    ) -> str:
        """
        Generate API documentation for a class.

        Args:
            class_name: Class name
            methods: Dictionary of method names to code

        Returns:
            Markdown documentation
        """
        doc = f"# API: {class_name}\n\n"

        doc += "## Methods\n\n"

        for method_name, method_code in methods.items():
            # Extract docstring
            try:
                tree = ast.parse(method_code)
                docstring = ast.get_docstring(tree.body[0])
            except:
                docstring = None

            doc += f"### `{method_name}()`\n\n"

            if docstring:
                doc += f"{docstring}\n\n"

            # Extract signature
            match = re.search(r'def\s+\w+\s*\([^)]*\)', method_code)
            if match:
                doc += f"**Signature**: `{match.group(0)}`\n\n"

        return doc

    def generate_changelog_entry(
        self,
        parameter_name: str,
        version: str,
        impact: ImpactAnalysis
    ) -> str:
        """
        Generate a changelog entry for a new parameter.

        Args:
            parameter_name: Parameter name
            version: Version number
            impact: Impact analysis

        Returns:
            Changelog entry
        """
        entry = f"## [{version}] - New Parameter: {parameter_name}\n\n"

        entry += "### Added\n\n"
        entry += f"- New parameter: `{parameter_name}`\n"

        if impact.affected_features:
            entry += f"- Affects features: {', '.join(impact.affected_features)}\n"

        entry += "\n### Changed\n\n"

        if impact.affected_files:
            entry += "Modified files:\n"
            for file in impact.affected_files:
                entry += f"- `{file}`\n"

        entry += "\n### Testing\n\n"

        if impact.testing_requirements:
            for req in impact.testing_requirements:
                entry += f"- {req}\n"

        if impact.migration_steps:
            entry += "\n### Migration\n\n"
            for step in impact.migration_steps:
                entry += f"{step}\n"

        entry += "\n"

        return entry


class ImpactAnalyzer:
    """
    Analyzes the impact of adding a new parameter.
    """

    def __init__(self, codebase_index):
        self.codebase_index = codebase_index

    def analyze_impact(
        self,
        parameter_name: str,
        affected_files: List[str],
        parameter_def: Any = None
    ) -> ImpactAnalysis:
        """
        Analyze the impact of adding a parameter.

        Args:
            parameter_name: Parameter name
            affected_files: List of files that will be modified
            parameter_def: Parameter definition (optional)

        Returns:
            ImpactAnalysis object
        """
        analysis = ImpactAnalysis(parameter_name=parameter_name)

        # Files
        analysis.affected_files = affected_files

        # Complexity score (0-100)
        complexity = 0

        # Base complexity from number of files
        complexity += len(affected_files) * 10

        # Additional complexity from file sizes
        for file_path in affected_files:
            full_path = self.codebase_index.get_file_path(file_path)
            if full_path and full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        lines = len(f.readlines())
                    # Large files = higher complexity
                    if lines > 1000:
                        complexity += 5
                except:
                    pass

        # Complexity from parameter type
        if parameter_def:
            if hasattr(parameter_def, 'param_type'):
                param_type = str(parameter_def.param_type.value)
                if param_type == 'array_int' or param_type == 'array_float':
                    complexity += 15  # Arrays are complex
                elif param_type == 'categorical':
                    complexity += 10  # Multiple options

            # Dependencies add complexity
            if hasattr(parameter_def, 'depends_on'):
                complexity += len(parameter_def.depends_on) * 5

        analysis.complexity_score = min(100, complexity)

        # Risk level
        if complexity < 30:
            analysis.risk_level = 'low'
        elif complexity < 60:
            analysis.risk_level = 'medium'
        else:
            analysis.risk_level = 'high'

        # Testing requirements
        analysis.testing_requirements = self._generate_testing_requirements(
            parameter_name, parameter_def, analysis.complexity_score
        )

        # Migration steps
        if analysis.risk_level != 'low':
            analysis.migration_steps = self._generate_migration_steps(
                parameter_name, affected_files
            )

        return analysis

    def _generate_testing_requirements(
        self,
        parameter_name: str,
        parameter_def: Any,
        complexity: float
    ) -> List[str]:
        """Generate testing requirements"""
        requirements = [
            f"Unit test for parameter validation",
            f"Integration test with existing parameters",
            f"Test edge cases (min, max, default values)"
        ]

        if complexity > 50:
            requirements.append("Full system regression test")
            requirements.append("Performance impact assessment")

        if parameter_def and hasattr(parameter_def, 'genre_relevance'):
            for genre in parameter_def.genre_relevance:
                requirements.append(f"Test with {genre} genre profile")

        return requirements

    def _generate_migration_steps(
        self,
        parameter_name: str,
        affected_files: List[str]
    ) -> List[str]:
        """Generate migration steps"""
        steps = [
            "1. Review all affected files for compatibility",
            "2. Update parameter registry with new parameter",
            "3. Run full test suite to ensure backward compatibility",
            f"4. Update documentation for {parameter_name}",
            "5. Deploy to staging environment for testing",
            "6. Monitor for any issues in production"
        ]

        if len(affected_files) > 5:
            steps.insert(3, "4. Conduct code review with team")

        return steps


# Example usage
if __name__ == "__main__":
    print("Advanced Utilities for Code Generation")
    print("=" * 60)

    # Dependency analysis example
    analyzer = DependencyAnalyzer()

    dep1 = Dependency(
        source="harmony.voicing.quartal_probability",
        target="harmony.voicing.density",
        dependency_type="enhances",
        strength=0.7,
        description="Quartal voicings work well with higher density"
    )
    analyzer.add_dependency(dep1)

    print("\nDependency Analysis:")
    print(f"  Dependencies for {dep1.source}:")
    print(f"    - {dep1.dependency_type} {dep1.target} (strength: {dep1.strength})")

    # Documentation generation example
    doc_gen = DocumentationGenerator()

    print("\nDocumentation Generation:")
    print("  Generates markdown docs for parameters and APIs")

    # Impact analysis example
    print("\nImpact Analysis:")
    print("  Analyzes complexity and risk of parameter additions")
    print("  Generates testing requirements and migration steps")
