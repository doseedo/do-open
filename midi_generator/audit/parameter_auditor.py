"""
Parameter Auditor - Agent 1
Scans all modules for hardcoded values that should be parameterized
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class HardcodedValue:
    """Represents a hardcoded value found in the code"""
    file_path: str
    line_number: int
    value: any
    context: str  # The line of code containing the value
    category: str  # 'magic_number', 'fixed_pattern', 'string_choice', etc.
    severity: str  # 'high', 'medium', 'low' - how critical to parameterize
    module_type: str  # 'harmony', 'melody', 'rhythm', 'genre', 'other'


@dataclass
class AuditReport:
    """Complete audit report for the codebase"""
    total_files_scanned: int = 0
    total_lines_scanned: int = 0
    findings: List[HardcodedValue] = field(default_factory=list)
    summary_by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    summary_by_module: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    summary_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class ParameterAuditor:
    """
    Comprehensive auditor for finding hardcoded values that should be parameters
    """

    # Common musical magic numbers that appear in code
    MUSICAL_MAGIC_NUMBERS = {
        0.5, 0.67, 0.75, 0.8, 0.9,  # Common probabilities
        2, 3, 4, 5, 7, 9, 11, 13,  # Musical intervals and extensions
        12, 24, 48, 96, 192,  # MIDI ticks and divisions
        60, 64, 67, 72,  # Common MIDI notes (C4, E4, G4, C5)
        127,  # Max MIDI velocity
        0.3, 0.2, 0.1,  # More probabilities
    }

    # Keywords that suggest musical decisions
    DECISION_KEYWORDS = [
        'voicing', 'substitution', 'extension', 'alteration',
        'swing', 'groove', 'syncopation', 'rhythm',
        'chord', 'progression', 'harmony', 'melody',
        'interval', 'contour', 'ornament', 'phrase'
    ]

    # Paths to exclude from audit
    EXCLUDE_PATTERNS = [
        '__pycache__',
        '.git',
        'tests/',
        'examples/',
        'docs/',
    ]

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.report = AuditReport()

    def should_audit_file(self, file_path: Path) -> bool:
        """Determine if a file should be audited"""
        if not file_path.suffix == '.py':
            return False

        path_str = str(file_path)
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern in path_str:
                return False

        return True

    def categorize_module(self, file_path: Path) -> str:
        """Determine the module type from path"""
        path_str = str(file_path).lower()

        if 'harmony' in path_str or 'chord' in path_str or 'voicing' in path_str:
            return 'harmony'
        elif 'melody' in path_str or 'melodic' in path_str:
            return 'melody'
        elif 'rhythm' in path_str or 'groove' in path_str or 'drum' in path_str:
            return 'rhythm'
        elif 'genre' in path_str or 'style' in path_str:
            return 'genre'
        elif 'bass' in path_str:
            return 'bass'
        elif 'voice' in path_str:
            return 'voice'
        else:
            return 'other'

    def assess_severity(self, value, context: str, module_type: str) -> str:
        """Assess how critical it is to parameterize this value"""
        context_lower = context.lower()

        # High severity: Direct musical decisions
        for keyword in self.DECISION_KEYWORDS:
            if keyword in context_lower:
                return 'high'

        # High severity: Probabilities and random thresholds
        if 'random' in context_lower or 'probability' in context_lower:
            return 'high'

        # Medium severity: Configuration values
        if module_type in ['harmony', 'melody', 'rhythm', 'genre']:
            return 'medium'

        # Low severity: Everything else
        return 'low'

    def audit_file(self, file_path: Path):
        """Audit a single Python file for hardcoded values"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            self.report.total_files_scanned += 1
            self.report.total_lines_scanned += len(lines)

            module_type = self.categorize_module(file_path)

            # Parse with AST
            try:
                tree = ast.parse(content)
                self._audit_ast(tree, file_path, lines, module_type)
            except SyntaxError:
                print(f"⚠️  Syntax error in {file_path}, skipping AST analysis")

            # Also do regex-based scanning for patterns AST might miss
            self._audit_with_regex(file_path, lines, module_type)

        except Exception as e:
            print(f"❌ Error auditing {file_path}: {e}")

    def _audit_ast(self, tree: ast.AST, file_path: Path, lines: List[str], module_type: str):
        """Audit using AST parsing"""

        class HardcodedValueVisitor(ast.NodeVisitor):
            def __init__(self, auditor, file_path, lines, module_type):
                self.auditor = auditor
                self.file_path = file_path
                self.lines = lines
                self.module_type = module_type

            def visit_Num(self, node):
                """Visit numeric constants"""
                if hasattr(node, 'lineno'):
                    line_num = node.lineno
                    context = self.lines[line_num - 1] if line_num <= len(self.lines) else ""

                    # Skip obvious non-musical constants
                    if self._should_check_number(node.n, context):
                        severity = self.auditor.assess_severity(node.n, context, self.module_type)

                        finding = HardcodedValue(
                            file_path=str(self.file_path),
                            line_number=line_num,
                            value=node.n,
                            context=context.strip(),
                            category='magic_number',
                            severity=severity,
                            module_type=self.module_type
                        )
                        self.auditor.report.findings.append(finding)
                        self.auditor.report.summary_by_category['magic_number'] += 1
                        self.auditor.report.summary_by_severity[severity] += 1

                self.generic_visit(node)

            def visit_Constant(self, node):
                """Visit constant values (Python 3.8+)"""
                if hasattr(node, 'lineno'):
                    line_num = node.lineno
                    context = self.lines[line_num - 1] if line_num <= len(self.lines) else ""

                    if isinstance(node.value, (int, float)):
                        if self._should_check_number(node.value, context):
                            severity = self.auditor.assess_severity(node.value, context, self.module_type)

                            finding = HardcodedValue(
                                file_path=str(self.file_path),
                                line_number=line_num,
                                value=node.value,
                                context=context.strip(),
                                category='magic_number',
                                severity=severity,
                                module_type=self.module_type
                            )
                            self.auditor.report.findings.append(finding)
                            self.auditor.report.summary_by_category['magic_number'] += 1
                            self.auditor.report.summary_by_severity[severity] += 1

                    elif isinstance(node.value, str):
                        # Check for musical string choices
                        if self._is_musical_choice(node.value, context):
                            severity = self.auditor.assess_severity(node.value, context, self.module_type)

                            finding = HardcodedValue(
                                file_path=str(self.file_path),
                                line_number=line_num,
                                value=node.value,
                                context=context.strip(),
                                category='string_choice',
                                severity=severity,
                                module_type=self.module_type
                            )
                            self.auditor.report.findings.append(finding)
                            self.auditor.report.summary_by_category['string_choice'] += 1
                            self.auditor.report.summary_by_severity[severity] += 1

                self.generic_visit(node)

            def visit_List(self, node):
                """Visit list literals (potential fixed patterns)"""
                if hasattr(node, 'lineno') and len(node.elts) > 0:
                    line_num = node.lineno
                    context = self.lines[line_num - 1] if line_num <= len(self.lines) else ""

                    # Check if this looks like a musical pattern
                    if self._is_musical_pattern(node, context):
                        severity = self.auditor.assess_severity(node.elts, context, self.module_type)

                        finding = HardcodedValue(
                            file_path=str(self.file_path),
                            line_number=line_num,
                            value=f"List with {len(node.elts)} elements",
                            context=context.strip(),
                            category='fixed_pattern',
                            severity=severity,
                            module_type=self.module_type
                        )
                        self.auditor.report.findings.append(finding)
                        self.auditor.report.summary_by_category['fixed_pattern'] += 1
                        self.auditor.report.summary_by_severity[severity] += 1

                self.generic_visit(node)

            def visit_Compare(self, node):
                """Visit comparison operations (potential random thresholds)"""
                if hasattr(node, 'lineno'):
                    line_num = node.lineno
                    context = self.lines[line_num - 1] if line_num <= len(self.lines) else ""

                    # Check for random() comparisons
                    if 'random' in context.lower():
                        severity = 'high'  # Random thresholds are always high priority

                        finding = HardcodedValue(
                            file_path=str(self.file_path),
                            line_number=line_num,
                            value="Random threshold",
                            context=context.strip(),
                            category='random_threshold',
                            severity=severity,
                            module_type=self.module_type
                        )
                        self.auditor.report.findings.append(finding)
                        self.auditor.report.summary_by_category['random_threshold'] += 1
                        self.auditor.report.summary_by_severity[severity] += 1

                self.generic_visit(node)

            def _should_check_number(self, num, context: str) -> bool:
                """Determine if a number should be flagged"""
                # Skip obvious non-musical constants
                if num in [0, 1, -1, 2, 10, 100, 1000]:
                    # But include them if in musical context
                    context_lower = context.lower()
                    has_musical_keyword = any(kw in context_lower for kw in self.auditor.DECISION_KEYWORDS)
                    if not has_musical_keyword:
                        return False

                # Skip array indices
                if '[' in context and ']' in context:
                    return False

                # Skip range() calls
                if 'range(' in context:
                    return False

                return True

            def _is_musical_choice(self, value: str, context: str) -> bool:
                """Check if string looks like a musical choice"""
                musical_terms = [
                    'major', 'minor', 'dorian', 'lydian', 'mixolydian',
                    'rootless', 'quartal', 'close', 'spread', 'drop',
                    'jazz', 'blues', 'funk', 'latin', 'swing',
                    'smooth', 'angular', 'chromatic', 'diatonic'
                ]

                value_lower = value.lower()
                return any(term in value_lower for term in musical_terms)

            def _is_musical_pattern(self, node: ast.List, context: str) -> bool:
                """Check if list looks like a musical pattern"""
                if len(node.elts) < 2:
                    return False

                # Check context for musical keywords
                context_lower = context.lower()
                pattern_keywords = ['pattern', 'rhythm', 'chord', 'scale', 'notes', 'intervals']
                return any(kw in context_lower for kw in pattern_keywords)

        visitor = HardcodedValueVisitor(self, file_path, lines, module_type)
        visitor.visit(tree)

    def _audit_with_regex(self, file_path: Path, lines: List[str], module_type: str):
        """Additional regex-based auditing for patterns AST might miss"""

        # Pattern: if style == "something"
        style_pattern = re.compile(r'if\s+\w+\s*==\s*["\'](\w+)["\']')

        for i, line in enumerate(lines, 1):
            matches = style_pattern.findall(line)
            if matches:
                severity = self.assess_severity(matches[0], line, module_type)

                finding = HardcodedValue(
                    file_path=str(file_path),
                    line_number=i,
                    value=matches[0],
                    context=line.strip(),
                    category='conditional_branch',
                    severity=severity,
                    module_type=module_type
                )
                self.report.findings.append(finding)
                self.report.summary_by_category['conditional_branch'] += 1
                self.report.summary_by_severity[severity] += 1

    def audit_directory(self):
        """Audit all Python files in the directory"""
        print(f"🔍 Starting audit of {self.root_path}")
        print("=" * 80)

        python_files = list(self.root_path.rglob("*.py"))
        files_to_audit = [f for f in python_files if self.should_audit_file(f)]

        print(f"📁 Found {len(files_to_audit)} Python files to audit\n")

        for i, file_path in enumerate(files_to_audit, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(files_to_audit)} files audited...")
            self.audit_file(file_path)

        print(f"\n✅ Audit complete!")
        print(f"   Files scanned: {self.report.total_files_scanned}")
        print(f"   Lines scanned: {self.report.total_lines_scanned}")
        print(f"   Findings: {len(self.report.findings)}")

    def generate_report(self, output_path: str = None):
        """Generate a comprehensive audit report"""

        # Sort findings by severity and module
        high_severity = [f for f in self.report.findings if f.severity == 'high']
        medium_severity = [f for f in self.report.findings if f.severity == 'medium']
        low_severity = [f for f in self.report.findings if f.severity == 'low']

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PARAMETER AUDIT REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\n📊 SUMMARY")
        report_lines.append(f"   Total Files Scanned: {self.report.total_files_scanned}")
        report_lines.append(f"   Total Lines Scanned: {self.report.total_lines_scanned:,}")
        report_lines.append(f"   Total Findings: {len(self.report.findings)}")
        report_lines.append(f"\n🎯 BY SEVERITY")
        report_lines.append(f"   High:   {len(high_severity):4d} findings")
        report_lines.append(f"   Medium: {len(medium_severity):4d} findings")
        report_lines.append(f"   Low:    {len(low_severity):4d} findings")

        report_lines.append(f"\n📁 BY CATEGORY")
        for category, count in sorted(self.report.summary_by_category.items(),
                                     key=lambda x: x[1], reverse=True):
            report_lines.append(f"   {category:20s}: {count:4d}")

        report_lines.append(f"\n🎵 BY MODULE TYPE")
        module_counts = defaultdict(int)
        for finding in self.report.findings:
            module_counts[finding.module_type] += 1

        for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"   {module:15s}: {count:4d}")

        # High severity findings detail
        report_lines.append(f"\n\n{'=' * 80}")
        report_lines.append(f"🚨 HIGH SEVERITY FINDINGS ({len(high_severity)} total)")
        report_lines.append(f"{'=' * 80}")

        # Group by file
        findings_by_file = defaultdict(list)
        for finding in high_severity[:200]:  # Limit to first 200 for readability
            findings_by_file[finding.file_path].append(finding)

        for file_path, findings in sorted(findings_by_file.items())[:20]:  # Top 20 files
            report_lines.append(f"\n📄 {file_path}")
            report_lines.append(f"   {len(findings)} high-severity findings")
            for finding in findings[:5]:  # Top 5 per file
                report_lines.append(f"   Line {finding.line_number:4d} [{finding.category}]: {finding.context[:70]}")

        report_text = "\n".join(report_lines)

        # Print to console
        print("\n" + report_text)

        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\n💾 Report saved to {output_path}")

        # Also save detailed JSON
        json_path = output_path.replace('.txt', '.json') if output_path else 'audit_report.json'
        self.save_json_report(json_path)

    def save_json_report(self, output_path: str):
        """Save detailed findings as JSON"""
        data = {
            'summary': {
                'total_files': self.report.total_files_scanned,
                'total_lines': self.report.total_lines_scanned,
                'total_findings': len(self.report.findings),
                'by_severity': dict(self.report.summary_by_severity),
                'by_category': dict(self.report.summary_by_category),
            },
            'findings': [
                {
                    'file': f.file_path,
                    'line': f.line_number,
                    'value': str(f.value),
                    'context': f.context,
                    'category': f.category,
                    'severity': f.severity,
                    'module_type': f.module_type
                }
                for f in self.report.findings
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Detailed JSON report saved to {output_path}")


def main():
    """Run the audit"""
    auditor = ParameterAuditor("/home/user/Do/midi_generator")
    auditor.audit_directory()
    auditor.generate_report("/home/user/Do/midi_generator/audit/audit_report.txt")


if __name__ == "__main__":
    main()
