# Parameterization Refactoring Guide

## Agent 1: Parameter Auditor & Refactorer

This guide demonstrates how to transform hardcoded values into learnable parameters across the 85,989-line codebase.

## Audit Results Summary

- **Total Files Scanned**: 119
- **Total Lines**: 83,973
- **Total Hardcoded Values Found**: 13,432
  - High Severity: 3,533
  - Medium Severity: 5,555
  - Low Severity: 4,344

### By Category
- Magic Numbers: 10,273
- String Choices: 1,948
- Conditional Branches: 548
- Fixed Patterns: 504
- Random Thresholds: 159

### By Module Type
- Genre Modules: 6,302 findings
- Other: 5,501 findings
- Rhythm: 962 findings
- Harmony: 478 findings
- Voice: 105 findings
- Bass: 84 findings

## Refactoring Patterns

### Pattern 1: Random Thresholds → Probability Parameters

**BEFORE (Hardcoded)**:
```python
if random.random() < 0.3:  # 30% chance of bend
    bend = random.choice([0.25, 0.5, 1.0])
```

**AFTER (Parameterized)**:
```python
def generate_lick(self,
                  bend_probability: float = 0.3,
                  bend_amounts: List[float] = [0.25, 0.5, 1.0]):
    if random.random() < bend_probability:
        bend = random.choice(bend_amounts)
```

**Registry Entry**:
```python
ParameterDefinition(
    name="bend_probability",
    full_path="genre.rock.guitar.bend_probability",
    description="Probability of guitar string bends",
    param_type=ParameterType.PROBABILITY,
    default_value=0.3,
    category=ParameterCategory.GENRE,
    musical_impact=MusicalImpact.HIGH
)
```

### Pattern 2: Magic Numbers → Continuous Parameters

**BEFORE (Hardcoded)**:
```python
velocity = random.randint(80, 110)
```

**AFTER (Parameterized)**:
```python
def generate_note(self,
                  velocity_min: int = 80,
                  velocity_max: int = 110):
    velocity = random.randint(velocity_min, velocity_max)
```

**Registry Entry**:
```python
ParameterDefinition(
    name="velocity_min",
    full_path="genre.rock.guitar.velocity_min",
    param_type=ParameterType.VELOCITY,
    default_value=80
)
```

### Pattern 3: Fixed Patterns → Array Parameters

**BEFORE (Hardcoded)**:
```python
note_durations = [0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5]
```

**AFTER (Parameterized)**:
```python
def generate_rhythm(self,
                    note_durations: List[float] = None):
    if note_durations is None:
        note_durations = [0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5]
```

**Registry Entry**:
```python
ParameterDefinition(
    name="rock_lick_durations",
    full_path="genre.rock.guitar.lick_durations",
    param_type=ParameterType.ARRAY_FLOAT,
    default_value=[0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5]
)
```

### Pattern 4: String Choices → Categorical Parameters

**BEFORE (Hardcoded)**:
```python
voicing = "rootless"
```

**AFTER (Parameterized)**:
```python
def generate_chord(self,
                   voicing_type: str = "rootless"):
    if voicing_type == "rootless":
        ...
    elif voicing_type == "close":
        ...
```

**Registry Entry**:
```python
ParameterDefinition(
    name="voicing_type",
    full_path="harmony.jazz.voicing_type",
    param_type=ParameterType.CATEGORICAL,
    options=["rootless", "close", "spread", "drop2"],
    default_value="rootless"
)
```

### Pattern 5: Conditional Branches → Parameter-Driven Logic

**BEFORE (Hardcoded)**:
```python
if style == "jazz":
    use_extensions = True
    swing = 0.67
elif style == "rock":
    use_extensions = False
    swing = 0.5
```

**AFTER (Parameterized)**:
```python
def generate(self,
             use_extensions: bool = True,
             swing_amount: float = 0.67):
    # Logic now driven by parameters
    if use_extensions:
        ...
```

### Pattern 6: Complex Progressions → Parameterized Data

**BEFORE (Hardcoded)**:
```python
progressions = {
    'i_iv_v': [
        (0, 'maj', 4.0),
        (5, 'maj', 4.0),
        (7, 'maj', 4.0),
        (0, 'maj', 4.0),
    ]
}
```

**AFTER (Parameterized)**:
```python
@dataclass
class ProgressionPattern:
    degrees: List[int] = field(default_factory=lambda: [0, 5, 7, 0])
    qualities: List[str] = field(default_factory=lambda: ['maj', 'maj', 'maj', 'maj'])
    durations: List[float] = field(default_factory=lambda: [4.0, 4.0, 4.0, 4.0])

def get_progression(self, pattern: ProgressionPattern = None):
    if pattern is None:
        pattern = ProgressionPattern()
```

## Class-Level Refactoring Strategy

### Original Class Structure
```python
class RockGenerator:
    def generate_lick(self, root, length_beats):
        # All values hardcoded inside methods
        bend_prob = 0.3
        velocity_min = 80
        velocity_max = 110
        ...
```

### Refactored Class Structure
```python
@dataclass
class RockGeneratorConfig:
    """Configuration for rock music generation"""
    # Guitar technique parameters
    bend_probability: float = 0.3
    bend_amounts: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    vibrato_probability: float = 0.4
    vibrato_depth_range: Tuple[float, float] = (0.0, 30.0)

    # Velocity parameters
    velocity_min: int = 80
    velocity_max: int = 110
    palm_mute_velocity_min: int = 70
    palm_mute_velocity_max: int = 90

    # Rhythm parameters
    lick_durations: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5])
    note_duration_ratio: float = 0.9  # Gap between notes

    # Power chord parameters
    power_chord_voicing: List[int] = field(default_factory=lambda: [0, 7, 12])  # root, fifth, octave
    power_chord_duration_ratio: float = 0.95
    palm_mute_duration_ratio: float = 0.85

class RockGenerator:
    def __init__(self, config: RockGeneratorConfig = None):
        self.config = config or RockGeneratorConfig()

    def generate_lick(self, root, length_beats):
        # Use self.config parameters instead of hardcoded values
        if random.random() < self.config.bend_probability:
            bend = random.choice(self.config.bend_amounts)
```

## Benefits of This Approach

1. **Backward Compatibility**: Default values match original behavior exactly
2. **Learnability**: All parameters can be learned by XGBoost
3. **Flexibility**: Users can override any parameter
4. **Discoverability**: All parameters documented in registry
5. **Validation**: Type checking and range validation built-in

## Comprehensive Example: Classic Rock Module

See `genres/classic_rock_parameterized.py` for a complete refactored example with:
- 50+ exposed parameters (up from 0)
- Config dataclass for organization
- Full backward compatibility
- Registry integration
- Documentation

## Target Metrics

By completion of Phase 1 refactoring:
- **2,000+ total parameters** exposed across all modules
- **100% backward compatibility** (all defaults match original behavior)
- **90%+ test coverage** for parameterized modules
- **Complete registry** with validation and metadata

## Next Steps for Other Agents

Once Agent 1 completes parameter exposure:
- **Agent 2**: Validates parameter coverage can recreate any MIDI
- **Agent 4**: Extracts features from MIDI for learning
- **Agent 5**: Trains XGBoost models to predict these parameters
- **Agent 6**: Compiles predictions into executable code
