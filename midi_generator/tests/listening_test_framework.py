#!/usr/bin/env python3
"""
AGENT 17: Listening Test Framework for A/B Comparisons
=======================================================

Framework for conducting perceptual listening tests to evaluate
generated big band arrangements against professional recordings.

This module provides:
1. A/B test pair generation
2. Blind listening test protocols
3. Perceptual evaluation metrics (musicality, authenticity, style)
4. Statistical analysis of listening test results
5. Export to survey formats for human evaluators

Based on research from:
- ISMIR papers on perceptual evaluation of music generation
- Turing test protocols for music (Ariza, 2009)
- MUSHRA (Multiple Stimuli with Hidden Reference and Anchor) methodology
- Subjective listening test protocols (ITU-R BS.1116)

Author: Agent 17 - Quality Validation & Testing Engineer
Date: 2025
License: MIT
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import statistics


# ============================================================================
# LISTENING TEST DATA STRUCTURES
# ============================================================================

class TestType(Enum):
    """Type of listening test."""
    AB = "ab"                      # A/B comparison (which is better?)
    ABX = "abx"                    # ABX (which matches X: A or B?)
    TURING = "turing"              # Turing test (which is human-made?)
    MUSHRA = "mushra"              # Multiple stimuli rating
    LIKERT = "likert"              # Likert scale rating (1-5)
    PAIRWISE = "pairwise"          # Pairwise preference


class EvaluationCriterion(Enum):
    """Criteria for evaluation."""
    MUSICALITY = "musicality"                    # Overall musical quality
    AUTHENTICITY = "authenticity"                # Sounds like professional recording
    STYLE_ACCURACY = "style_accuracy"            # Matches expected style (Basie, Ellington)
    VOICE_LEADING = "voice_leading"              # Smooth voice leading
    ARRANGEMENT_QUALITY = "arrangement_quality"  # Quality of orchestration
    SWING_FEEL = "swing_feel"                    # Authentic swing rhythm
    DYNAMICS = "dynamics"                        # Use of dynamics
    ARTICULATION = "articulation"                # Realistic articulations


@dataclass
class ListeningTestItem:
    """Single item in a listening test."""
    id: str
    sample_a_path: str              # Path to first audio/MIDI file
    sample_b_path: str              # Path to second audio/MIDI file
    sample_a_label: str = "A"       # Display label
    sample_b_label: str = "B"
    sample_a_is_generated: bool = True
    sample_b_is_generated: bool = False
    criterion: EvaluationCriterion = EvaluationCriterion.MUSICALITY
    instructions: str = "Which sample sounds more musical?"
    randomize_order: bool = True    # Randomize A/B presentation order

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            'id': self.id,
            'sample_a_path': self.sample_a_path,
            'sample_b_path': self.sample_b_path,
            'criterion': self.criterion.value,
            'instructions': self.instructions,
        }


@dataclass
class ListeningTestResponse:
    """Response to a listening test item."""
    item_id: str
    selected_sample: str           # "A" or "B"
    confidence: int = 3            # 1 (low) to 5 (high)
    comments: str = ""
    response_time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'selected_sample': self.selected_sample,
            'confidence': self.confidence,
            'comments': self.comments,
            'response_time': self.response_time_seconds
        }


@dataclass
class ListeningTestResults:
    """Results from a listening test session."""
    test_name: str
    test_type: TestType
    num_items: int
    responses: List[ListeningTestResponse]
    accuracy: float = 0.0           # Proportion correct (if known ground truth)
    preference_for_generated: float = 0.0  # Proportion preferring generated
    average_confidence: float = 0.0
    turing_pass_rate: float = 0.0  # For Turing tests

    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'num_items': self.num_items,
            'responses': [r.to_dict() for r in self.responses],
            'accuracy': self.accuracy,
            'preference_for_generated': self.preference_for_generated,
            'average_confidence': self.average_confidence,
            'turing_pass_rate': self.turing_pass_rate
        }


# ============================================================================
# LISTENING TEST GENERATOR
# ============================================================================

class ListeningTestGenerator:
    """
    Generate listening test pairs and protocols.
    """

    def __init__(self, test_name: str = "Big Band Evaluation"):
        """
        Initialize test generator.

        Args:
            test_name: Name of the listening test
        """
        self.test_name = test_name
        self.items: List[ListeningTestItem] = []

    def add_ab_pair(self,
                    generated_path: str,
                    reference_path: str,
                    criterion: EvaluationCriterion = EvaluationCriterion.MUSICALITY,
                    item_id: Optional[str] = None) -> None:
        """
        Add A/B comparison pair.

        Args:
            generated_path: Path to generated arrangement
            reference_path: Path to professional reference
            criterion: What to evaluate
            item_id: Optional ID (auto-generated if not provided)
        """
        if item_id is None:
            item_id = f"item_{len(self.items) + 1:03d}"

        instructions_map = {
            EvaluationCriterion.MUSICALITY: "Which sample sounds more musical?",
            EvaluationCriterion.AUTHENTICITY: "Which sample sounds more like a professional recording?",
            EvaluationCriterion.STYLE_ACCURACY: "Which sample better captures the intended style?",
            EvaluationCriterion.SWING_FEEL: "Which sample has a better swing feel?",
            EvaluationCriterion.VOICE_LEADING: "Which sample has smoother voice leading?",
        }

        item = ListeningTestItem(
            id=item_id,
            sample_a_path=generated_path,
            sample_b_path=reference_path,
            sample_a_is_generated=True,
            sample_b_is_generated=False,
            criterion=criterion,
            instructions=instructions_map.get(criterion, "Which sample is better?"),
            randomize_order=True
        )

        self.items.append(item)

    def add_turing_test_pair(self,
                            sample1_path: str,
                            sample2_path: str,
                            sample1_is_generated: bool,
                            item_id: Optional[str] = None) -> None:
        """
        Add Turing test pair (which is human-made?).

        Args:
            sample1_path: Path to first sample
            sample2_path: Path to second sample
            sample1_is_generated: Is sample1 generated (not human)?
            item_id: Optional ID
        """
        if item_id is None:
            item_id = f"turing_{len(self.items) + 1:03d}"

        item = ListeningTestItem(
            id=item_id,
            sample_a_path=sample1_path,
            sample_b_path=sample2_path,
            sample_a_is_generated=sample1_is_generated,
            sample_b_is_generated=not sample1_is_generated,
            criterion=EvaluationCriterion.AUTHENTICITY,
            instructions="Which sample was arranged by a human arranger?",
            randomize_order=True
        )

        self.items.append(item)

    def generate_test_protocol(self,
                              output_format: str = "json",
                              output_path: Optional[str] = None) -> str:
        """
        Generate test protocol for human evaluators.

        Args:
            output_format: Format (json, markdown, html)
            output_path: Optional path to save protocol

        Returns:
            Formatted test protocol
        """
        if output_format == "json":
            protocol = self._generate_json_protocol()
        elif output_format == "markdown":
            protocol = self._generate_markdown_protocol()
        elif output_format == "html":
            protocol = self._generate_html_protocol()
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        if output_path:
            with open(output_path, 'w') as f:
                f.write(protocol)

        return protocol

    def _generate_json_protocol(self) -> str:
        """Generate JSON format protocol."""
        protocol = {
            'test_name': self.test_name,
            'instructions': (
                "Listen to each pair of samples and answer the questions. "
                "Rate your confidence in each answer."
            ),
            'items': [item.to_dict() for item in self.items],
            'response_options': {
                'selection': ['A', 'B', 'Equal'],
                'confidence': [1, 2, 3, 4, 5]
            }
        }

        return json.dumps(protocol, indent=2)

    def _generate_markdown_protocol(self) -> str:
        """Generate Markdown format protocol."""
        lines = []
        lines.append(f"# {self.test_name}")
        lines.append("")
        lines.append("## Instructions")
        lines.append("")
        lines.append("Listen to each pair of samples (A and B) and answer the question.")
        lines.append("Rate your confidence: 1 (guessing) to 5 (completely sure).")
        lines.append("")
        lines.append("---")
        lines.append("")

        for i, item in enumerate(self.items, 1):
            lines.append(f"## Item {i}: {item.id}")
            lines.append("")
            lines.append(f"**Question:** {item.instructions}")
            lines.append("")
            lines.append(f"- Sample A: `{item.sample_a_path}`")
            lines.append(f"- Sample B: `{item.sample_b_path}`")
            lines.append("")
            lines.append("**Your answer:**")
            lines.append("- Selected: [ ] A  [ ] B  [ ] Equal")
            lines.append("- Confidence: [ ] 1  [ ] 2  [ ] 3  [ ] 4  [ ] 5")
            lines.append("- Comments: _____________________________________")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _generate_html_protocol(self) -> str:
        """Generate HTML format protocol (for web-based testing)."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.test_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .item {{ border: 1px solid #ccc; padding: 20px; margin: 20px 0; }}
        .audio-player {{ margin: 10px 0; }}
        .response {{ margin: 15px 0; }}
    </style>
</head>
<body>
    <h1>{self.test_name}</h1>
    <p><strong>Instructions:</strong> Listen to each pair and answer the question.
       Rate your confidence from 1 (guessing) to 5 (certain).</p>
"""

        for i, item in enumerate(self.items, 1):
            html += f"""
    <div class="item">
        <h2>Item {i}: {item.id}</h2>
        <p><strong>{item.instructions}</strong></p>

        <div class="audio-player">
            <label>Sample A:</label><br>
            <audio controls src="{item.sample_a_path}"></audio>
        </div>

        <div class="audio-player">
            <label>Sample B:</label><br>
            <audio controls src="{item.sample_b_path}"></audio>
        </div>

        <div class="response">
            <label>Your answer:</label><br>
            <input type="radio" name="item_{item.id}" value="A"> A
            <input type="radio" name="item_{item.id}" value="B"> B
            <input type="radio" name="item_{item.id}" value="Equal"> Equal
        </div>

        <div class="response">
            <label>Confidence:</label><br>
            <input type="radio" name="conf_{item.id}" value="1"> 1
            <input type="radio" name="conf_{item.id}" value="2"> 2
            <input type="radio" name="conf_{item.id}" value="3"> 3
            <input type="radio" name="conf_{item.id}" value="4"> 4
            <input type="radio" name="conf_{item.id}" value="5"> 5
        </div>

        <div class="response">
            <label>Comments:</label><br>
            <textarea name="comments_{item.id}" rows="3" cols="60"></textarea>
        </div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def shuffle_items(self) -> None:
        """Randomize item order to avoid order effects."""
        random.shuffle(self.items)


# ============================================================================
# LISTENING TEST ANALYZER
# ============================================================================

class ListeningTestAnalyzer:
    """
    Analyze results from listening tests.
    """

    @staticmethod
    def analyze_responses(items: List[ListeningTestItem],
                         responses: List[ListeningTestResponse]) -> ListeningTestResults:
        """
        Analyze listening test responses.

        Args:
            items: Test items
            responses: Participant responses

        Returns:
            Analyzed results
        """
        # Match responses to items
        item_map = {item.id: item for item in items}

        correct_count = 0
        prefer_generated = 0
        confidences = []
        turing_passes = 0  # Times generated was mistaken for human

        for response in responses:
            item = item_map.get(response.item_id)
            if not item:
                continue

            confidences.append(response.confidence)

            # Check if correct (assuming "better" = reference)
            if item.sample_b_is_generated:
                # B is generated, A is reference
                correct_answer = "A"
            else:
                # A is generated, B is reference
                correct_answer = "B"

            if response.selected_sample == correct_answer:
                correct_count += 1
            else:
                prefer_generated += 1

            # For Turing test: check if participant chose generated as "human"
            if item.criterion == EvaluationCriterion.AUTHENTICITY:
                # Question is "which is human?"
                selected_is_generated = (
                    (response.selected_sample == "A" and item.sample_a_is_generated) or
                    (response.selected_sample == "B" and item.sample_b_is_generated)
                )

                if selected_is_generated:
                    turing_passes += 1

        # Calculate metrics
        num_responses = len(responses)
        accuracy = correct_count / num_responses if num_responses > 0 else 0.0
        pref_generated = prefer_generated / num_responses if num_responses > 0 else 0.0
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        turing_rate = turing_passes / num_responses if num_responses > 0 else 0.0

        return ListeningTestResults(
            test_name="Big Band Listening Test",
            test_type=TestType.AB,
            num_items=len(items),
            responses=responses,
            accuracy=accuracy,
            preference_for_generated=pref_generated,
            average_confidence=avg_confidence,
            turing_pass_rate=turing_rate
        )

    @staticmethod
    def generate_results_report(results: ListeningTestResults,
                               output_path: Optional[str] = None) -> str:
        """
        Generate human-readable results report.

        Args:
            results: Test results
            output_path: Optional path to save report

        Returns:
            Formatted report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("LISTENING TEST RESULTS")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Test Name: {results.test_name}")
        lines.append(f"Test Type: {results.test_type.value}")
        lines.append(f"Number of Items: {results.num_items}")
        lines.append(f"Number of Responses: {len(results.responses)}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("OVERALL METRICS")
        lines.append("-" * 80)
        lines.append(f"Accuracy (chose reference over generated): {results.accuracy:.1%}")
        lines.append(f"Preference for generated: {results.preference_for_generated:.1%}")
        lines.append(f"Average confidence: {results.average_confidence:.2f} / 5.0")
        lines.append(f"Turing pass rate (generated mistaken for human): {results.turing_pass_rate:.1%}")
        lines.append("")

        # Interpretation
        lines.append("-" * 80)
        lines.append("INTERPRETATION")
        lines.append("-" * 80)

        if results.preference_for_generated > 0.4:
            lines.append("✓ Strong result: Generated arrangements are competitive with references")
        elif results.preference_for_generated > 0.3:
            lines.append("○ Moderate result: Generated arrangements show promise but need improvement")
        else:
            lines.append("✗ Weak result: Generated arrangements significantly behind references")

        if results.turing_pass_rate > 0.4:
            lines.append("✓ Turing test: Generated music frequently mistaken for human-made")
        elif results.turing_pass_rate > 0.25:
            lines.append("○ Turing test: Generated music occasionally mistaken for human-made")
        else:
            lines.append("✗ Turing test: Generated music easily distinguished from human-made")

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report


# ============================================================================
# EXAMPLE USAGE & UTILITIES
# ============================================================================

def create_example_listening_test() -> ListeningTestGenerator:
    """
    Create example listening test for big band arrangements.

    Returns:
        Configured listening test generator
    """
    test = ListeningTestGenerator("Big Band Arrangement Evaluation")

    # Add A/B pairs comparing generated to professional
    test.add_ab_pair(
        generated_path="output/generated_basie_style.mid",
        reference_path="references/basie_one_oclock_jump.mid",
        criterion=EvaluationCriterion.STYLE_ACCURACY,
        item_id="basie_comparison"
    )

    test.add_ab_pair(
        generated_path="output/generated_ellington_style.mid",
        reference_path="references/ellington_caravan.mid",
        criterion=EvaluationCriterion.ARTICULATION,
        item_id="ellington_comparison"
    )

    test.add_ab_pair(
        generated_path="output/generated_bebop_melody.mid",
        reference_path="references/parker_confirmation.mid",
        criterion=EvaluationCriterion.MUSICALITY,
        item_id="bebop_melody"
    )

    # Add Turing test pairs
    test.add_turing_test_pair(
        sample1_path="output/generated_swing.mid",
        sample2_path="references/basie_april_in_paris.mid",
        sample1_is_generated=True,
        item_id="turing_swing"
    )

    test.shuffle_items()

    return test


# ============================================================================
# MAIN - Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Agent 17: Listening Test Framework")
    print("=" * 60)

    # Create example test
    test = create_example_listening_test()

    # Generate protocols in different formats
    print("\nGenerating test protocols...")

    json_protocol = test.generate_test_protocol(
        output_format="json",
        output_path="/tmp/listening_test.json"
    )
    print("✓ JSON protocol saved to /tmp/listening_test.json")

    markdown_protocol = test.generate_test_protocol(
        output_format="markdown",
        output_path="/tmp/listening_test.md"
    )
    print("✓ Markdown protocol saved to /tmp/listening_test.md")

    html_protocol = test.generate_test_protocol(
        output_format="html",
        output_path="/tmp/listening_test.html"
    )
    print("✓ HTML protocol saved to /tmp/listening_test.html")

    print("\nExample usage:")
    print("""
    # 1. Create listening test
    test = ListeningTestGenerator("My Test")

    # 2. Add comparison pairs
    test.add_ab_pair(
        generated_path="my_arrangement.mid",
        reference_path="professional.mid",
        criterion=EvaluationCriterion.MUSICALITY
    )

    # 3. Generate protocol
    test.generate_test_protocol(output_format="html", output_path="test.html")

    # 4. Collect responses (external process - surveys, web forms, etc.)
    responses = [...]  # Load from survey results

    # 5. Analyze results
    analyzer = ListeningTestAnalyzer()
    results = analyzer.analyze_responses(test.items, responses)

    # 6. Generate report
    report = analyzer.generate_results_report(results, "results.txt")
    print(report)
    """)
