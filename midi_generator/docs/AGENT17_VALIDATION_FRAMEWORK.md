# AGENT 17: Quality Validation & Testing Framework

## Mission Statement

Create comprehensive test suite and validation framework to ensure generated big band arrangements meet professional standards, with quantitative metrics, authenticity measurement, and continuous quality improvement capabilities.

---

## Overview

Agent 17 delivers a complete quality assurance system for the big band generator, providing:

1. **Automated Validation** - Test voice leading, harmony, and form automatically
2. **Authenticity Measurement** - Compare to professional recordings using statistical metrics
3. **Regression Testing** - Catch quality regressions when updating modules
4. **Listening Tests** - Framework for human perceptual evaluation
5. **Quality Reports** - Comprehensive reports with actionable feedback

---

## Deliverables

### 1. Automated Test Suite (`validation_tests.py`)

Core validation module with professional music theory rules checking.

#### ArrangementValidator Class

**Methods:**
- `validate_voice_leading()` - Check voice leading quality
- `validate_harmony()` - Verify harmonic correctness and style appropriateness
- `validate_form()` - Ensure form structure adheres to expectations
- `measure_authenticity()` - Compare to professional recordings

**Voice Leading Validation:**
```python
from tests.validation_tests import ArrangementValidator

validator = ArrangementValidator()

# Validate arrangement
result = validator.validate_voice_leading(arrangement)

print(f"Passed: {result.passed}")
print(f"Score: {result.score:.3f}")
print(f"Max voice leap: {result.metrics['max_voice_leap']} semitones")
print(f"Avg movement: {result.metrics['avg_voice_movement']:.2f}")
print(f"Parallel fifths: {result.metrics['parallel_fifths']}")

# Errors and warnings
for error in result.errors:
    print(f"Error: {error}")

for warning in result.warnings:
    print(f"Warning: {warning}")
```

**Checks performed:**
- ✓ No extreme voice leaps (>12 semitones)
- ✓ Average voice movement < 5 semitones
- ✓ All voices within instrument range
- ✓ Parallel fifths detection (warnings in jazz, errors in classical)
- ✓ Voice crossing detection

**Harmony Validation:**
```python
result = validator.validate_harmony(chord_progression, style='bebop')

print(f"Resolution errors: {result.metrics['resolution_errors']}")
print(f"Awkward movements: {result.metrics['awkward_movements']}")
print(f"Inappropriate chords: {result.metrics['inappropriate_chords']}")
```

**Checks performed:**
- ✓ Chord types appropriate for style (bebop, swing, modal, etc.)
- ✓ V7 → I resolutions correct
- ✓ Tritone substitutions resolve properly
- ✓ No awkward root movements (tritone leaps)
- ✓ Harmonic rhythm is musical

**Form Validation:**
```python
result = validator.validate_form(
    arrangement,
    expected_form='aaba',
    expected_bars=32
)

print(f"Estimated bars: {result.metrics['estimated_bars']}")
print(f"Bridge differentiated: {result.metrics['bridge_differentiated']}")
print(f"Shout chorus present: {result.metrics['shout_chorus_present']}")
```

**Checks performed:**
- ✓ Correct bar count (within 2-bar margin)
- ✓ Intro/outro present if expected
- ✓ Bridge differentiated from A sections (dynamics/texture)
- ✓ Shout chorus louder than other sections
- ✓ Section boundaries clear

**Authenticity Measurement:**
```python
metrics = validator.measure_authenticity(arrangement)

print(f"Overall authenticity: {metrics.overall_authenticity:.3f}")
print(f"Interval similarity: {metrics.interval_similarity:.3f}")
print(f"Rhythm similarity: {metrics.rhythm_similarity:.3f}")
print(f"Voicing match: {metrics.voicing_match:.3f}")
```

**Metrics computed:**
- **Interval distribution** - Compare to bebop reference (Charlie Parker analysis)
- **Rhythm complexity** - Note duration variety
- **Voice spacing** - Average spacing in harmonic sections (target: 3-5 semitones)
- **Overall authenticity** - Weighted combination (target: >0.85)

---

### 2. Comprehensive Validator (`ComprehensiveValidator`)

Run complete validation suite and generate quality reports.

```python
from tests.validation_tests import ComprehensiveValidator

validator = ComprehensiveValidator()

# Run all validations
report = validator.validate_arrangement(
    arrangement=my_arrangement,
    progression=my_chords,
    expected_form='aaba',
    expected_bars=32,
    style='bebop'
)

# Generate human-readable report
text_report = validator.generate_quality_report(
    report,
    output_file='quality_report.txt'
)

print(text_report)
```

**Output Example:**
```
================================================================================
BIG BAND ARRANGEMENT QUALITY REPORT
================================================================================

Overall Status: ✓ PASSED
Overall Score: 0.875 / 1.000

--------------------------------------------------------------------------------
VOICE LEADING VALIDATION
--------------------------------------------------------------------------------
Status: ✓ PASSED
Score: 0.923
  Max voice leap: 7 semitones
  Avg voice movement: 2.34 semitones
  Parallel fifths: 2
  Warnings:
    - Found 2 parallel fifths (acceptable in jazz)

--------------------------------------------------------------------------------
HARMONY VALIDATION
--------------------------------------------------------------------------------
Status: ✓ PASSED
Score: 0.900
  Resolution errors: 0
  Awkward movements: 0

--------------------------------------------------------------------------------
FORM VALIDATION
--------------------------------------------------------------------------------
Status: ✓ PASSED
Score: 0.950
  Estimated bars: 32
  Expected bars: 32
  Bridge differentiated: True
  Shout chorus present: True

--------------------------------------------------------------------------------
AUTHENTICITY METRICS
--------------------------------------------------------------------------------
Overall Authenticity: 0.872
  Interval similarity: 0.891
  Rhythm similarity: 0.823
  Voicing match: 0.902

================================================================================
```

---

### 3. Regression Test Suite (`test_bigband_validation.py`)

Comprehensive unittest suite for continuous quality assurance.

**Test Classes:**

1. **TestVoiceLeading** - Voice leading validation tests
2. **TestHarmonyValidation** - Harmony correctness tests
3. **TestFormValidation** - Form structure tests
4. **TestAuthenticityMeasurement** - Authenticity metrics tests
5. **TestCompleteArrangement** - End-to-end integration tests
6. **TestBebopMelodyQuality** - Bebop melody specific tests (Agent 1)
7. **TestSaxVoicingSpacing** - Sax soli voicing tests (Agent 2)

**Running Tests:**
```bash
# Run all tests
python midi_generator/tests/test_bigband_validation.py

# Run with pytest (verbose)
pytest midi_generator/tests/test_bigband_validation.py -v

# Run specific test class
pytest midi_generator/tests/test_bigband_validation.py::TestVoiceLeading -v
```

**Example Tests:**

```python
def test_good_voice_leading(self):
    """Test that smooth voice leading passes validation."""
    arrangement = create_smooth_arrangement()
    result = self.validator.validate_voice_leading(arrangement)

    self.assertTrue(result.passed)
    self.assertGreater(result.score, 0.7)
    self.assertLess(result.metrics['avg_voice_movement'], 5.0)

def test_32_bar_aaba_form(self):
    """Test validation of standard 32-bar AABA form."""
    arrangement = create_32_bar_arrangement()
    result = self.validator.validate_form(arrangement, 'aaba', 32)

    self.assertLessEqual(abs(result.metrics['estimated_bars'] - 32), 2)
```

**Success Criteria:**
- All tests must pass ✓
- Voice leading score > 0.7
- Average voice movement < 5 semitones
- Authenticity score > 0.85

---

### 4. Listening Test Framework (`listening_test_framework.py`)

Framework for conducting perceptual listening tests with human evaluators.

**Test Types Supported:**
- **A/B Comparison** - Which sample is better?
- **ABX Test** - Which sample (A or B) matches X?
- **Turing Test** - Which was arranged by a human?
- **MUSHRA** - Multiple stimuli rating
- **Pairwise** - Pairwise preference

**Creating a Listening Test:**

```python
from tests.listening_test_framework import (
    ListeningTestGenerator,
    EvaluationCriterion
)

# Create test
test = ListeningTestGenerator("Big Band Evaluation")

# Add A/B pairs
test.add_ab_pair(
    generated_path="output/basie_style.mid",
    reference_path="references/basie_one_oclock_jump.mid",
    criterion=EvaluationCriterion.STYLE_ACCURACY,
    item_id="basie_comparison"
)

test.add_ab_pair(
    generated_path="output/bebop_melody.mid",
    reference_path="references/parker_confirmation.mid",
    criterion=EvaluationCriterion.MUSICALITY,
    item_id="bebop_melody"
)

# Add Turing test
test.add_turing_test_pair(
    sample1_path="output/generated.mid",
    sample2_path="references/professional.mid",
    sample1_is_generated=True,
    item_id="turing_test"
)

# Shuffle to avoid order effects
test.shuffle_items()

# Generate protocols for evaluators
test.generate_test_protocol(output_format="html", output_path="test.html")
test.generate_test_protocol(output_format="json", output_path="test.json")
test.generate_test_protocol(output_format="markdown", output_path="test.md")
```

**Analyzing Results:**

```python
from tests.listening_test_framework import ListeningTestAnalyzer

analyzer = ListeningTestAnalyzer()

# Analyze responses
results = analyzer.analyze_responses(test.items, responses)

print(f"Accuracy: {results.accuracy:.1%}")
print(f"Preference for generated: {results.preference_for_generated:.1%}")
print(f"Turing pass rate: {results.turing_pass_rate:.1%}")

# Generate report
report = analyzer.generate_results_report(
    results,
    output_path="listening_test_results.txt"
)
```

**Evaluation Criteria:**
- **Musicality** - Overall musical quality
- **Authenticity** - Sounds like professional recording
- **Style Accuracy** - Matches expected style (Basie, Ellington, etc.)
- **Voice Leading** - Smooth voice leading
- **Swing Feel** - Authentic swing rhythm
- **Dynamics** - Use of dynamics
- **Articulation** - Realistic articulations

---

## Research Foundation

### Music Information Retrieval (MIR) Metrics
- ISMIR 2023-2024 papers on music generation evaluation
- Computational analysis of jazz datasets (PiJAMA, Weimar)
- Statistical pattern recognition for style matching

### Music Theory Validation
- Mark Levine: "The Jazz Theory Book" - Voice leading and harmony rules
- Ted Pease & Ken Pullig: "Modern Jazz Voicings" - Voicing standards
- Matthew Keating (2023): LSTM voice-leading distance minimization

### Perceptual Testing
- MUSHRA methodology (ITU-R BS.1116)
- Turing test protocols for music (Ariza, 2009)
- A/B testing best practices from psychoacoustics research

---

## Integration with Other Agents

### Agent 1 (Bebop Melody):
```python
# Test bebop melody quality
def test_bebop_melody_vocabulary():
    melody = BebopMelodyGenerator().generate_phrase(...)
    result = validator.validate_voice_leading({'lead': melody})
    assert result.score > 0.8
```

### Agent 2 (Sax Voicing):
```python
# Test sax voicing spacing
def test_drop2_spacing():
    voicing = SaxSoliVoicing.voice_melody(...)
    spacing = validator._analyze_voice_spacing({'saxes': voicing})
    assert spacing > 3.0  # Professional standard
```

### Agent 17 validates ALL agents:
- Agents 1-12: Core modules (melody, harmony, voicing, rhythm)
- Agents 13-15: Style analyzers (Ellington, Basie, Modern)
- Agent 18: Integration architecture
- Agent 19: Genre scalability

---

## Metrics & Thresholds

### Professional Standards

| Metric | Target | Acceptable | Poor |
|--------|--------|-----------|------|
| Voice leading score | >0.90 | >0.70 | <0.70 |
| Avg voice movement | <3 semitones | <5 | >5 |
| Max voice leap | <7 semitones | <12 | >12 |
| Authenticity score | >0.90 | >0.85 | <0.85 |
| Interval similarity | >0.85 | >0.75 | <0.75 |
| Voice spacing (bass) | 3-5 semitones | 2-6 | <2 or >6 |

### Validation Checklist

**Before Release:**
- [ ] All unit tests pass
- [ ] Voice leading score > 0.70
- [ ] Authenticity score > 0.85
- [ ] No unresolved dominant chords
- [ ] Form structure correct
- [ ] Shout chorus present and louder
- [ ] Bridge differentiated
- [ ] Voice spacing appropriate for register

**Quality Tiers:**
- **Excellent** (>0.90): Publication-ready, indistinguishable from professional
- **Good** (0.80-0.90): Professional quality with minor issues
- **Acceptable** (0.70-0.80): Usable but needs improvement
- **Poor** (<0.70): Not ready for use

---

## Usage Examples

### Example 1: Validate Complete Arrangement

```python
from tests.validation_tests import ComprehensiveValidator

# Create validator
validator = ComprehensiveValidator()

# Load or generate arrangement
arrangement = generate_big_band_arrangement(style='basie')

# Validate
report = validator.validate_arrangement(
    arrangement=arrangement,
    progression=chord_progression,
    expected_form='aaba',
    expected_bars=32,
    style='bebop'
)

# Check results
if report['overall_passed']:
    print("✓ Arrangement passed all validations!")
    print(f"Overall score: {report['overall_score']:.3f}")
else:
    print("✗ Arrangement has issues:")
    for validation, result in report['validations'].items():
        if not result.get('passed', True):
            print(f"  - {validation}: FAILED")
```

### Example 2: Continuous Integration Testing

```bash
#!/bin/bash
# CI test script

# Run validation tests
python midi_generator/tests/test_bigband_validation.py

if [ $? -eq 0 ]; then
    echo "✓ All validation tests passed"
else
    echo "✗ Validation tests failed"
    exit 1
fi

# Generate test arrangements and validate
python scripts/generate_and_validate.py

# Check quality threshold
if [ $(cat quality_score.txt) -lt 85 ]; then
    echo "✗ Quality score below threshold"
    exit 1
fi

echo "✓ All quality checks passed"
```

### Example 3: Listening Test Setup

```python
from tests.listening_test_framework import (
    ListeningTestGenerator,
    EvaluationCriterion
)

# Generate arrangements to test
arrangements = [
    ("basie", generate_basie_style()),
    ("ellington", generate_ellington_style()),
    ("thad_jones", generate_modern_style())
]

# Create listening test
test = ListeningTestGenerator("Big Band Style Comparison")

for style, arrangement in arrangements:
    # Export to MIDI/audio
    export_to_midi(arrangement, f"output/{style}.mid")

    # Add to test
    test.add_ab_pair(
        generated_path=f"output/{style}.mid",
        reference_path=f"references/{style}_reference.mid",
        criterion=EvaluationCriterion.STYLE_ACCURACY,
        item_id=f"{style}_comparison"
    )

# Generate web-based test
test.generate_test_protocol(output_format="html", output_path="listening_test.html")

print("Listening test ready at listening_test.html")
print("Distribute to evaluators and collect responses")
```

---

## Future Enhancements

### Planned Additions
1. **Dataset Comparison** - Direct comparison to PiJAMA, Weimar datasets
2. **Style-Specific Metrics** - Bebop vs. swing vs. modal validation
3. **Audio-Based Validation** - Analyze rendered audio, not just MIDI
4. **Machine Learning Evaluation** - Train discriminator to distinguish generated vs. professional
5. **Real-Time Validation** - Validate during generation, not after

### Integration with Agent 16 (Dataset Analysis)
- Use extracted patterns from PiJAMA for authenticity baselines
- Compare generated statistics to real dataset distributions
- Validate swing ratios against measured values from recordings

---

## Validation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Generate Arrangement                      │
│              (BigBandArranger or other modules)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Automated Validation Suite                      │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Voice Leading Validation                          │   │
│  │    - Check leaps, movement, parallel motion          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. Harmony Validation                                │   │
│  │    - Check resolutions, style appropriateness        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 3. Form Validation                                   │   │
│  │    - Check bar count, shout chorus, bridge           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 4. Authenticity Measurement                          │   │
│  │    - Compare intervals, rhythms, voicing to dataset  │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Quality Report Generated                    │
│                                                               │
│  Overall Score: 0.875                                        │
│  Passed: YES                                                 │
│                                                               │
│  Issues to Address:                                          │
│  - 2 large voice leaps in brass                             │
│  - Bridge could be more differentiated                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  Pass?       │
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
           YES                    NO
              │                     │
              ▼                     ▼
    ┌─────────────────┐   ┌───────────────────┐
    │ Export to MIDI  │   │ Revise & Re-test  │
    └─────────────────┘   └───────────────────┘
```

---

## Conclusion

Agent 17 provides a comprehensive quality assurance framework that:

✓ **Validates** arrangements against professional music theory standards
✓ **Measures** authenticity compared to real jazz recordings
✓ **Tests** continuously to catch regressions
✓ **Evaluates** perceptually through listening tests
✓ **Reports** quality metrics with actionable feedback

This ensures that the big band generator produces arrangements that meet the master prompt's goal:

> **"Generate arrangements indistinguishable from professional human arrangers"**

---

## References

1. Richard Cohn (1997) - "Neo-Riemannian Operations, Parsimonious Trichords"
2. Matthew Keating (2023) - "An Algorithmic Approach to Jazz Guitar Voice-Leading"
3. Mark Levine - "The Jazz Theory Book"
4. PiJAMA Dataset - 200+ hours jazz piano analysis
5. ISMIR 2023-2024 - Music generation evaluation papers
6. ITU-R BS.1116 - Subjective listening test methodology

---

**Agent 17 Status:** ✓ COMPLETE

All deliverables implemented:
- [x] Automated validation suite
- [x] Voice leading, harmony, form validation
- [x] Authenticity measurement
- [x] Regression test suite
- [x] Listening test framework
- [x] Quality report generation
- [x] Documentation

**Integration:** Ready for use by all agents and final system testing
