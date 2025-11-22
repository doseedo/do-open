"""
Integration Helpers for Code Generation - Agent 12
==================================================

Helper utilities for integrating generated code into the system:
- File modification helpers
- Registry updaters
- Test execution
- Validation runners
- Deployment helpers

Author: Agent 12 - Code Generation Agent
Date: 2025-11-20
License: MIT
"""

import ast
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .code_generator import GeneratedCode, CodeContext
from .advanced_utilities import (
    CodeMerger,
    DocumentationGenerator,
    ImpactAnalyzer,
    ImpactAnalysis
)

logger = logging.getLogger(__name__)


class RegistryIntegrator:
    """
    Integrates new parameters into the universal registry.
    """

    def __init__(self, registry_path: str = None):
        """
        Initialize registry integrator.

        Args:
            registry_path: Path to universal_registry.py
        """
        if registry_path is None:
            registry_path = "/home/user/Do/midi_generator/parameters/universal_registry.py"

        self.registry_path = Path(registry_path)

        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")

        logger.info(f"Registry integrator initialized: {self.registry_path}")

    def add_parameter(self, parameter_code: str) -> bool:
        """
        Add a parameter to the registry.

        Args:
            parameter_code: Parameter definition code

        Returns:
            True if successful
        """
        try:
            # Read existing registry
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                registry_content = f.read()

            # Find the DOMAIN_PARAMETERS dict
            # We'll insert before the closing brace

            # Parse to find insertion point
            tree = ast.parse(registry_content)

            # Find DOMAIN_PARAMETERS or similar dict
            insertion_point = None
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if 'PARAMETER' in target.id.upper():
                                # Found a parameter dict
                                # Get its end position
                                if isinstance(node.value, ast.Dict):
                                    insertion_point = node.value.end_lineno

            if insertion_point is None:
                logger.error("Could not find parameter dictionary in registry")
                return False

            # Split content by lines
            lines = registry_content.split('\n')

            # Find the actual closing brace by scanning backwards from insertion point
            closing_brace_line = None
            for i in range(insertion_point - 1, -1, -1):
                if '}' in lines[i]:
                    closing_brace_line = i
                    break

            if closing_brace_line is None:
                logger.error("Could not find closing brace in parameter dict")
                return False

            # Insert new parameter before closing brace
            indent = "    "  # Match existing indentation

            # Add comma to previous line if needed
            if not lines[closing_brace_line - 1].rstrip().endswith(','):
                lines[closing_brace_line - 1] += ','

            # Insert new parameter
            parameter_lines = parameter_code.strip().split('\n')
            indented_lines = [indent + line if line.strip() else line
                            for line in parameter_lines]

            # Insert before closing brace
            for line in reversed(indented_lines):
                lines.insert(closing_brace_line, line)

            # Add trailing comma
            if not lines[closing_brace_line + len(indented_lines) - 1].rstrip().endswith(','):
                lines[closing_brace_line + len(indented_lines) - 1] += ','

            # Write back
            updated_content = '\n'.join(lines)

            # Validate syntax
            try:
                compile(updated_content, self.registry_path, 'exec')
            except SyntaxError as e:
                logger.error(f"Syntax error in updated registry: {e}")
                return False

            # Write to file
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            logger.info("Successfully added parameter to registry")
            return True

        except Exception as e:
            logger.error(f"Failed to add parameter to registry: {e}")
            return False

    def validate_registry(self) -> Tuple[bool, List[str]]:
        """
        Validate the registry file.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check syntax
            try:
                compile(content, self.registry_path, 'exec')
            except SyntaxError as e:
                errors.append(f"Syntax error: {e}")
                return False, errors

            # Try to execute and check structure
            namespace = {}
            try:
                exec(content, namespace)
            except Exception as e:
                errors.append(f"Execution error: {e}")
                return False, errors

            # Check for required variables
            required_vars = ['ParameterDefinition', 'ParameterType', 'ParameterCategory']
            for var in required_vars:
                if var not in namespace:
                    errors.append(f"Missing required definition: {var}")

            if errors:
                return False, errors

            logger.info("Registry validation successful")
            return True, []

        except Exception as e:
            errors.append(f"Failed to read registry: {e}")
            return False, errors


class CodeIntegrator:
    """
    Integrates generated code into existing generator files.
    """

    def __init__(self, root_path: str = None):
        """
        Initialize code integrator.

        Args:
            root_path: Root path of codebase
        """
        if root_path is None:
            root_path = "/home/user/Do/midi_generator"

        self.root_path = Path(root_path)
        self.merger = CodeMerger()

        logger.info(f"Code integrator initialized: {self.root_path}")

    def integrate_generated_code(
        self,
        generated_code: GeneratedCode,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Integrate all generated code into the codebase.

        Args:
            generated_code: GeneratedCode object
            dry_run: If True, don't actually modify files

        Returns:
            Dictionary with integration results
        """
        results = {
            'success': False,
            'files_modified': [],
            'files_created': [],
            'errors': [],
            'warnings': []
        }

        # 1. Integrate registry update
        if generated_code.registry_update:
            registry_path = self.root_path / 'parameters' / 'universal_registry.py'
            integrator = RegistryIntegrator(str(registry_path))

            if not dry_run:
                success = integrator.add_parameter(generated_code.registry_update)
                if success:
                    results['files_modified'].append(str(registry_path))
                else:
                    results['errors'].append("Failed to update registry")
            else:
                logger.info(f"[DRY RUN] Would update registry: {registry_path}")
                results['files_modified'].append(f"[DRY RUN] {registry_path}")

        # 2. Integrate generator modifications
        for file_path, code in generated_code.generator_modifications.items():
            full_path = self.root_path / file_path

            if not full_path.exists():
                results['errors'].append(f"File not found: {file_path}")
                continue

            merge_result = self.merger.merge_into_file(
                str(full_path),
                code,
                merge_strategy='smart'
            )

            if merge_result.success:
                if not dry_run:
                    # Write merged code
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(merge_result.merged_files[str(full_path)])

                    results['files_modified'].append(str(full_path))
                    logger.info(f"Modified file: {full_path}")
                else:
                    logger.info(f"[DRY RUN] Would modify: {full_path}")
                    results['files_modified'].append(f"[DRY RUN] {full_path}")

                results['warnings'].extend(merge_result.warnings)
            else:
                results['errors'].extend(merge_result.conflicts)

        # 3. Create test files
        if generated_code.test_code:
            test_path = self.root_path / 'tests' / 'test_generated_parameter.py'

            if not dry_run:
                test_path.parent.mkdir(parents=True, exist_ok=True)
                with open(test_path, 'w', encoding='utf-8') as f:
                    f.write(generated_code.test_code)

                results['files_created'].append(str(test_path))
                logger.info(f"Created test file: {test_path}")
            else:
                logger.info(f"[DRY RUN] Would create test: {test_path}")
                results['files_created'].append(f"[DRY RUN] {test_path}")

        # Overall success
        results['success'] = len(results['errors']) == 0

        return results


class TestRunner:
    """
    Runs tests for generated code.
    """

    def __init__(self, root_path: str = None):
        """
        Initialize test runner.

        Args:
            root_path: Root path of codebase
        """
        if root_path is None:
            root_path = "/home/user/Do/midi_generator"

        self.root_path = Path(root_path)

        logger.info(f"Test runner initialized: {self.root_path}")

    def run_tests(
        self,
        test_file: str = None,
        test_pattern: str = "test_*.py"
    ) -> Dict[str, Any]:
        """
        Run tests.

        Args:
            test_file: Specific test file to run (optional)
            test_pattern: Pattern for test discovery

        Returns:
            Dictionary with test results
        """
        results = {
            'success': False,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'output': '',
            'errors': []
        }

        try:
            # Build pytest command
            cmd = ['python', '-m', 'pytest']

            if test_file:
                test_path = self.root_path / test_file
                if not test_path.exists():
                    results['errors'].append(f"Test file not found: {test_file}")
                    return results
                cmd.append(str(test_path))
            else:
                # Run all tests matching pattern
                test_dir = self.root_path / 'tests'
                if test_dir.exists():
                    cmd.extend([str(test_dir), '-k', test_pattern])
                else:
                    results['errors'].append("Tests directory not found")
                    return results

            # Add options
            cmd.extend(['-v', '--tb=short'])

            logger.info(f"Running tests: {' '.join(cmd)}")

            # Run tests
            process = subprocess.run(
                cmd,
                cwd=str(self.root_path),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            results['output'] = process.stdout + process.stderr

            # Parse output for results
            # Look for pytest summary
            summary_match = re.search(
                r'(\d+) passed|(\d+) failed|(\d+) error',
                results['output']
            )

            if summary_match:
                passed = summary_match.group(1)
                if passed:
                    results['tests_passed'] = int(passed)

            results['success'] = process.returncode == 0

            logger.info(f"Tests completed: {results['tests_passed']} passed")

        except subprocess.TimeoutExpired:
            results['errors'].append("Tests timed out after 5 minutes")
            logger.error("Test execution timed out")

        except Exception as e:
            results['errors'].append(f"Test execution failed: {e}")
            logger.error(f"Failed to run tests: {e}")

        return results

    def run_validation_tests(self, parameter_name: str) -> Dict[str, Any]:
        """
        Run validation tests for a specific parameter.

        Args:
            parameter_name: Parameter to test

        Returns:
            Test results
        """
        # Create a simple validation test
        test_code = f"""
import pytest

def test_{parameter_name.replace('.', '_')}_validation():
    \"\"\"Test validation for {parameter_name}\"\"\"
    from parameters.universal_registry import DOMAIN_PARAMETERS

    # Check parameter exists
    assert '{parameter_name}' in DOMAIN_PARAMETERS

    param_def = DOMAIN_PARAMETERS['{parameter_name}']

    # Check has required fields
    assert hasattr(param_def, 'default_value')
    assert hasattr(param_def, 'param_type')

    print(f"Parameter {parameter_name} validated successfully")

if __name__ == '__main__':
    test_{parameter_name.replace('.', '_')}_validation()
"""

        # Write temporary test file
        test_file = self.root_path / 'tests' / f'test_validation_{parameter_name.replace(".", "_")}.py'

        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_code)

            # Run the test
            results = self.run_tests(test_file=str(test_file.relative_to(self.root_path)))

            # Clean up temp file
            if test_file.exists():
                test_file.unlink()

            return results

        except Exception as e:
            logger.error(f"Failed to run validation tests: {e}")
            return {
                'success': False,
                'errors': [str(e)]
            }


class DeploymentHelper:
    """
    Helps with deploying generated code changes.
    """

    def __init__(self, root_path: str = None):
        """
        Initialize deployment helper.

        Args:
            root_path: Root path of codebase
        """
        if root_path is None:
            root_path = "/home/user/Do/midi_generator"

        self.root_path = Path(root_path)

        logger.info(f"Deployment helper initialized: {self.root_path}")

    def create_deployment_checklist(
        self,
        parameter_name: str,
        impact: ImpactAnalysis,
        files_modified: List[str]
    ) -> str:
        """
        Create a deployment checklist.

        Args:
            parameter_name: Parameter name
            impact: Impact analysis
            files_modified: List of modified files

        Returns:
            Markdown checklist
        """
        checklist = f"# Deployment Checklist: {parameter_name}\n\n"

        checklist += "## Pre-Deployment\n\n"
        checklist += "- [ ] Code review completed\n"
        checklist += "- [ ] All tests passing\n"
        checklist += "- [ ] Documentation updated\n"
        checklist += "- [ ] Impact analysis reviewed\n"
        checklist += f"- [ ] Risk level: **{impact.risk_level}**\n"
        checklist += "\n"

        checklist += "## Files Modified\n\n"
        for file in files_modified:
            checklist += f"- [ ] Reviewed: `{file}`\n"
        checklist += "\n"

        checklist += "## Testing\n\n"
        for req in impact.testing_requirements:
            checklist += f"- [ ] {req}\n"
        checklist += "\n"

        checklist += "## Deployment Steps\n\n"
        if impact.migration_steps:
            for step in impact.migration_steps:
                checklist += f"- [ ] {step}\n"
        else:
            checklist += "- [ ] Deploy to staging\n"
            checklist += "- [ ] Run smoke tests\n"
            checklist += "- [ ] Deploy to production\n"
            checklist += "- [ ] Monitor for issues\n"

        checklist += "\n"

        checklist += "## Post-Deployment\n\n"
        checklist += "- [ ] Verify parameter is working\n"
        checklist += "- [ ] Check logs for errors\n"
        checklist += "- [ ] Update changelog\n"
        checklist += "- [ ] Notify team\n"

        return checklist

    def generate_git_commit_message(
        self,
        parameter_name: str,
        files_modified: List[str],
        impact: ImpactAnalysis
    ) -> str:
        """
        Generate a git commit message.

        Args:
            parameter_name: Parameter name
            files_modified: List of modified files
            impact: Impact analysis

        Returns:
            Commit message
        """
        message = f"feat: Add parameter {parameter_name}\n\n"

        # Summary
        message += f"Adds new parameter '{parameter_name}' to the music generation system.\n\n"

        # Impact
        message += f"Impact: {impact.complexity_score:.0f}/100 complexity, "
        message += f"{impact.risk_level} risk\n\n"

        # Files changed
        message += "Files modified:\n"
        for file in files_modified[:10]:  # Limit to 10
            message += f"- {file}\n"

        if len(files_modified) > 10:
            message += f"- ... and {len(files_modified) - 10} more\n"

        message += "\n"

        # Testing
        message += "Testing:\n"
        for req in impact.testing_requirements[:5]:  # Limit to 5
            message += f"- {req}\n"

        return message

    def create_rollback_plan(
        self,
        parameter_name: str,
        files_modified: List[str]
    ) -> str:
        """
        Create a rollback plan.

        Args:
            parameter_name: Parameter name
            files_modified: List of modified files

        Returns:
            Markdown rollback plan
        """
        plan = f"# Rollback Plan: {parameter_name}\n\n"

        plan += "## Quick Rollback (Git)\n\n"
        plan += "```bash\n"
        plan += "# Revert the commit\n"
        plan += "git revert HEAD\n"
        plan += "git push\n"
        plan += "```\n\n"

        plan += "## Manual Rollback\n\n"
        plan += "If git revert is not possible:\n\n"

        plan += "1. Remove parameter from registry:\n"
        plan += f"   - Edit `parameters/universal_registry.py`\n"
        plan += f"   - Remove entry for `{parameter_name}`\n\n"

        plan += "2. Revert file changes:\n"
        for file in files_modified:
            plan += f"   - Restore `{file}` from backup\n"

        plan += "\n3. Run tests:\n"
        plan += "   ```bash\n"
        plan += "   python -m pytest tests/\n"
        plan += "   ```\n\n"

        plan += "4. Restart services:\n"
        plan += "   - Restart generator service\n"
        plan += "   - Clear any caches\n"

        return plan


# Example usage
if __name__ == "__main__":
    print("Integration Helpers for Code Generation")
    print("=" * 60)

    # Registry integration
    print("\n1. Registry Integration:")
    print("   - Adds parameters to universal_registry.py")
    print("   - Validates registry syntax")

    # Code integration
    print("\n2. Code Integration:")
    print("   - Merges generated code into existing files")
    print("   - Handles conflicts intelligently")

    # Test running
    print("\n3. Test Execution:")
    print("   - Runs pytest on generated tests")
    print("   - Validates parameter integration")

    # Deployment
    print("\n4. Deployment:")
    print("   - Creates deployment checklists")
    print("   - Generates commit messages")
    print("   - Plans rollback procedures")
