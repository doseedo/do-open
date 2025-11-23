"""
MIDI Transform Domain-Specific Language (DSL)
==============================================

A constrained language for expressing musical transformations.

Why a DSL?
- Reduces search space from infinite Python to finite grammar
- Ensures generated programs are safe and correct
- Enables neural program synthesis
- Guarantees universal applicability

Grammar:
    Program    ::= Statement+
    Statement  ::= ForEach | If | Operation | Aggregate
    ForEach    ::= "foreach" Iterator ":" Statement+
    If         ::= "if" Filter Threshold ":" Statement+
    Operation  ::= OpType Value
    Aggregate  ::= AggType Target

    Iterator   ::= ALL_NOTES | SIMULTANEOUS_NOTES | SEQUENTIAL_NOTES | ...
    Filter     ::= PITCH_GREATER | DURATION_LESS | IS_CHORD_TONE | ...
    OpType     ::= TRANSPOSE | TIME_SCALE | SET_VELOCITY | ...
    AggType    ::= MEAN | MEDIAN | MIN | MAX | SORT_BY_PITCH | ...

    Value      ::= Float | "amount" | Expression
    Expression ::= Value ("+" | "-" | "*") Value

Example Programs:
    1. Transpose:
        foreach ALL_NOTES:
            TRANSPOSE amount * 24 - 12

    2. Drop-2 voicing:
        foreach SIMULTANEOUS_NOTES:
            SORT_BY_PITCH notes
            if len(notes) >= 4:
                notes[-2] OCTAVE_SHIFT -1

Author: Agent 8 - Transform Architecture
Phase: 4 (Neural Program Synthesis)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict, Any, Callable
import ast
import numpy as np


# ============================================================================
# DSL Token Types (Vocabulary)
# ============================================================================

class IteratorType(Enum):
    """How to iterate over notes"""
    ALL_NOTES = "all_notes"
    NOTES_IN_TRACK = "notes_in_track"
    SIMULTANEOUS_NOTES = "simultaneous_notes"  # Chords
    SEQUENTIAL_NOTES = "sequential_notes"       # Melodies
    NOTES_IN_PITCH_RANGE = "notes_in_pitch_range"
    NOTES_IN_TIME_WINDOW = "notes_in_time_window"
    EVERY_NTH_NOTE = "every_nth_note"
    NOTES_ON_BEAT = "notes_on_beat"
    NOTES_OFF_BEAT = "notes_off_beat"


class FilterType(Enum):
    """Conditional filters on notes"""
    # Pitch filters
    PITCH_GREATER = "pitch_greater"
    PITCH_LESS = "pitch_less"
    PITCH_EQUALS = "pitch_equals"
    PITCH_IN_RANGE = "pitch_in_range"
    PITCH_IS_EVEN = "pitch_is_even"

    # Time filters
    ONSET_GREATER = "onset_greater"
    ONSET_LESS = "onset_less"
    DURATION_GREATER = "duration_greater"
    DURATION_LESS = "duration_less"

    # Velocity filters
    VELOCITY_GREATER = "velocity_greater"
    VELOCITY_LESS = "velocity_less"

    # Musical filters
    IS_CHORD_TONE = "is_chord_tone"
    IS_PASSING_TONE = "is_passing_tone"
    IS_HIGHEST_IN_CHORD = "is_highest_in_chord"
    IS_LOWEST_IN_CHORD = "is_lowest_in_chord"

    # Position filters
    IS_FIRST = "is_first"
    IS_LAST = "is_last"
    INDEX_EQUALS = "index_equals"
    INDEX_MOD_EQUALS = "index_mod_equals"

    # Collection filters
    COUNT_GREATER = "count_greater"
    COUNT_LESS = "count_less"
    COUNT_EQUALS = "count_equals"


class OperationType(Enum):
    """Operations that modify notes"""
    # Pitch operations
    TRANSPOSE = "transpose"                    # pitch += value
    SET_PITCH = "set_pitch"                   # pitch = value
    SCALE_INTERVAL = "scale_interval"          # pitch = root + (pitch - root) * factor
    OCTAVE_SHIFT = "octave_shift"             # pitch += 12 * value
    INVERT_AROUND = "invert_around"           # pitch = axis - (pitch - axis)
    QUANTIZE_PITCH = "quantize_pitch"         # Snap to scale

    # Timing operations
    TIME_SHIFT = "time_shift"                 # onset += value
    TIME_SCALE = "time_scale"                 # onset *= factor
    QUANTIZE_TIME = "quantize_time"           # onset = round(onset / grid) * grid
    SWING = "swing"                           # Apply swing to off-beats
    DURATION_SCALE = "duration_scale"         # duration *= factor
    DURATION_SET = "duration_set"             # duration = value

    # Velocity operations
    SET_VELOCITY = "set_velocity"             # velocity = value
    SCALE_VELOCITY = "scale_velocity"         # velocity *= factor
    ADD_VELOCITY = "add_velocity"             # velocity += value
    VELOCITY_CURVE = "velocity_curve"         # Apply curve based on position

    # Structural operations
    ADD_NOTE = "add_note"                     # notes.append(new_note)
    REMOVE_NOTE = "remove_note"               # notes.remove(note)
    DUPLICATE_NOTE = "duplicate_note"         # notes.append(note.copy())
    SPLIT_NOTE = "split_note"                 # Split into multiple notes


class AggregatorType(Enum):
    """Aggregate operations on note collections"""
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    FIRST = "first"
    LAST = "last"
    COUNT = "count"
    SORT_BY_PITCH = "sort_by_pitch"
    SORT_BY_TIME = "sort_by_time"
    SORT_BY_VELOCITY = "sort_by_velocity"
    RANGE = "range"
    STD = "std"


# ============================================================================
# DSL Abstract Syntax Tree (AST)
# ============================================================================

@dataclass
class Statement:
    """Base class for DSL statements"""

    def to_python(self, indent: int = 0) -> str:
        """Compile to Python code"""
        raise NotImplementedError

    def to_tokens(self) -> List[str]:
        """Convert to token sequence for neural model"""
        raise NotImplementedError

    def validate(self) -> bool:
        """Check if statement is valid"""
        return True


@dataclass
class Value:
    """Value in DSL (float, variable, or expression)"""
    value: Union[float, str]  # e.g., 12.0, "amount", "amount * 24"

    def to_python(self) -> str:
        if isinstance(self.value, (int, float)):
            return str(self.value)
        return str(self.value)

    def to_tokens(self) -> List[str]:
        if isinstance(self.value, (int, float)):
            return [str(self.value)]
        # Parse expression
        tokens = str(self.value).split()
        return tokens


@dataclass
class ForEach(Statement):
    """Iterate over notes"""
    iterator: IteratorType
    iterator_params: Optional[Dict[str, Any]] = None
    body: List[Statement] = field(default_factory=list)

    def to_python(self, indent: int = 0) -> str:
        spaces = "    " * indent
        code = []

        # Generate iterator
        if self.iterator == IteratorType.ALL_NOTES:
            code.append(f"{spaces}for note in notes:")

        elif self.iterator == IteratorType.SIMULTANEOUS_NOTES:
            code.append(f"{spaces}for chord in get_chords(notes):")
            code.append(f"{spaces}    for note in chord:")

        elif self.iterator == IteratorType.SEQUENTIAL_NOTES:
            code.append(f"{spaces}sorted_notes = sorted(notes, key=lambda n: n['start_time'])")
            code.append(f"{spaces}for i in range(len(sorted_notes) - 1):")
            code.append(f"{spaces}    note = sorted_notes[i]")
            code.append(f"{spaces}    next_note = sorted_notes[i + 1]")

        elif self.iterator == IteratorType.NOTES_ON_BEAT:
            code.append(f"{spaces}for note in [n for n in notes if is_on_beat(n)]:")

        elif self.iterator == IteratorType.NOTES_OFF_BEAT:
            code.append(f"{spaces}for note in [n for n in notes if not is_on_beat(n)]:")

        else:
            code.append(f"{spaces}for note in iterate_{self.iterator.value}(notes):")

        # Generate body
        for stmt in self.body:
            code.append(stmt.to_python(indent + 1))

        return "\n".join(code)

    def to_tokens(self) -> List[str]:
        tokens = ["FOREACH", self.iterator.value]
        for stmt in self.body:
            tokens.extend(stmt.to_tokens())
        tokens.append("END_FOREACH")
        return tokens


@dataclass
class If(Statement):
    """Conditional statement"""
    filter: FilterType
    threshold: Value
    then_body: List[Statement] = field(default_factory=list)
    else_body: List[Statement] = field(default_factory=list)

    def to_python(self, indent: int = 0) -> str:
        spaces = "    " * indent
        code = []

        # Generate condition
        if self.filter == FilterType.PITCH_GREATER:
            condition = f"note['pitch'] > {self.threshold.to_python()}"
        elif self.filter == FilterType.PITCH_LESS:
            condition = f"note['pitch'] < {self.threshold.to_python()}"
        elif self.filter == FilterType.DURATION_GREATER:
            condition = f"note['duration'] > {self.threshold.to_python()}"
        elif self.filter == FilterType.VELOCITY_GREATER:
            condition = f"note['velocity'] > {self.threshold.to_python()}"
        elif self.filter == FilterType.IS_CHORD_TONE:
            condition = "is_chord_tone(note, current_key)"
        elif self.filter == FilterType.IS_HIGHEST_IN_CHORD:
            condition = "note['pitch'] == max(n['pitch'] for n in chord)"
        elif self.filter == FilterType.COUNT_GREATER:
            condition = f"len(notes) > {self.threshold.to_python()}"
        else:
            condition = f"check_{self.filter.value}(note, {self.threshold.to_python()})"

        code.append(f"{spaces}if {condition}:")

        # Then body
        if self.then_body:
            for stmt in self.then_body:
                code.append(stmt.to_python(indent + 1))
        else:
            code.append(f"{spaces}    pass")

        # Else body
        if self.else_body:
            code.append(f"{spaces}else:")
            for stmt in self.else_body:
                code.append(stmt.to_python(indent + 1))

        return "\n".join(code)

    def to_tokens(self) -> List[str]:
        tokens = ["IF", self.filter.value]
        tokens.extend(self.threshold.to_tokens())
        tokens.append("THEN")
        for stmt in self.then_body:
            tokens.extend(stmt.to_tokens())
        if self.else_body:
            tokens.append("ELSE")
            for stmt in self.else_body:
                tokens.extend(stmt.to_tokens())
        tokens.append("END_IF")
        return tokens


@dataclass
class Operation(Statement):
    """Execute operation on note"""
    op_type: OperationType
    value: Value
    target: str = "note"  # Which variable to operate on

    def to_python(self, indent: int = 0) -> str:
        spaces = "    " * indent
        val = self.value.to_python()

        if self.op_type == OperationType.TRANSPOSE:
            return f"{spaces}{self.target}['pitch'] = int(np.clip({self.target}['pitch'] + {val}, 0, 127))"

        elif self.op_type == OperationType.OCTAVE_SHIFT:
            return f"{spaces}{self.target}['pitch'] = int(np.clip({self.target}['pitch'] + 12 * ({val}), 0, 127))"

        elif self.op_type == OperationType.SET_PITCH:
            return f"{spaces}{self.target}['pitch'] = int(np.clip({val}, 0, 127))"

        elif self.op_type == OperationType.SCALE_INTERVAL:
            return f"{spaces}{self.target}['pitch'] = int(reference_pitch + ({self.target}['pitch'] - reference_pitch) * {val})"

        elif self.op_type == OperationType.INVERT_AROUND:
            return f"{spaces}{self.target}['pitch'] = int({val} - ({self.target}['pitch'] - {val}))"

        elif self.op_type == OperationType.TIME_SHIFT:
            return f"{spaces}{self.target}['start_time'] = max(0, {self.target}['start_time'] + {val})"

        elif self.op_type == OperationType.TIME_SCALE:
            return f"{spaces}{self.target}['start_time'] *= {val}"

        elif self.op_type == OperationType.DURATION_SCALE:
            return f"{spaces}{self.target}['duration'] *= {val}"

        elif self.op_type == OperationType.SET_VELOCITY:
            return f"{spaces}{self.target}['velocity'] = int(np.clip({val}, 1, 127))"

        elif self.op_type == OperationType.SCALE_VELOCITY:
            return f"{spaces}{self.target}['velocity'] = int(np.clip({self.target}['velocity'] * {val}, 1, 127))"

        elif self.op_type == OperationType.QUANTIZE_TIME:
            grid = val
            return f"{spaces}{self.target}['start_time'] = round({self.target}['start_time'] / {grid}) * {grid}"

        else:
            return f"{spaces}apply_{self.op_type.value}({self.target}, {val})"

    def to_tokens(self) -> List[str]:
        tokens = [self.op_type.value]
        tokens.extend(self.value.to_tokens())
        return tokens


@dataclass
class Aggregate(Statement):
    """Compute aggregate value"""
    aggregator: AggregatorType
    source: str = "notes"           # What to aggregate
    target: str = "aggregate_value"  # Where to store result

    def to_python(self, indent: int = 0) -> str:
        spaces = "    " * indent

        if self.aggregator == AggregatorType.MEAN:
            return f"{spaces}{self.target} = np.mean([n['pitch'] for n in {self.source}])"

        elif self.aggregator == AggregatorType.MEDIAN:
            return f"{spaces}{self.target} = np.median([n['pitch'] for n in {self.source}])"

        elif self.aggregator == AggregatorType.MIN:
            return f"{spaces}{self.target} = min(n['pitch'] for n in {self.source})"

        elif self.aggregator == AggregatorType.MAX:
            return f"{spaces}{self.target} = max(n['pitch'] for n in {self.source})"

        elif self.aggregator == AggregatorType.SORT_BY_PITCH:
            return f"{spaces}{self.source} = sorted({self.source}, key=lambda n: n['pitch'])"

        elif self.aggregator == AggregatorType.SORT_BY_TIME:
            return f"{spaces}{self.source} = sorted({self.source}, key=lambda n: n['start_time'])"

        elif self.aggregator == AggregatorType.COUNT:
            return f"{spaces}{self.target} = len({self.source})"

        else:
            return f"{spaces}{self.target} = {self.aggregator.value}({self.source})"

    def to_tokens(self) -> List[str]:
        return [self.aggregator.value, self.source, "->", self.target]


# ============================================================================
# Complete DSL Program
# ============================================================================

@dataclass
class DSLProgram:
    """Complete transform program in DSL"""
    statements: List[Statement]
    name: str = "transform"
    description: str = ""

    def to_python(self) -> str:
        """Compile to executable Python function"""
        code = []

        # Function signature
        code.append(f"def {self.name}(midi, amount):")
        if self.description:
            code.append(f'    """{self.description}"""')

        # Import and setup
        code.append("    import numpy as np")
        code.append("    from midi_generator.transforms.space_level_transforms import extract_notes_from_midi, notes_to_midi")
        code.append("")
        code.append("    # Extract notes")
        code.append("    notes = extract_notes_from_midi(midi)")
        code.append("    if not notes:")
        code.append("        return midi")
        code.append("")

        # Helper functions
        code.append("    # Helper functions")
        code.append("    def is_on_beat(note, beat_duration=0.5):")
        code.append("        return abs(note['start_time'] % beat_duration) < 0.1")
        code.append("")
        code.append("    def get_chords(notes, threshold=0.05):")
        code.append("        sorted_notes = sorted(notes, key=lambda n: n['start_time'])")
        code.append("        chords = []")
        code.append("        current_chord = []")
        code.append("        current_time = -1")
        code.append("        for note in sorted_notes:")
        code.append("            if current_time < 0 or abs(note['start_time'] - current_time) < threshold:")
        code.append("                current_chord.append(note)")
        code.append("                current_time = note['start_time']")
        code.append("            else:")
        code.append("                if current_chord:")
        code.append("                    chords.append(current_chord)")
        code.append("                current_chord = [note]")
        code.append("                current_time = note['start_time']")
        code.append("        if current_chord:")
        code.append("            chords.append(current_chord)")
        code.append("        return chords")
        code.append("")

        # Main transform logic
        code.append("    # Transform logic")
        code.append("    reference_pitch = np.mean([n['pitch'] for n in notes])")
        code.append("")

        # Generate code for each statement
        for stmt in self.statements:
            code.append(stmt.to_python(indent=1))

        # Return
        code.append("")
        code.append("    # Convert back to MIDI")
        code.append("    return notes_to_midi(notes, midi.ticks_per_beat)")

        return "\n".join(code)

    def to_tokens(self) -> List[str]:
        """Convert to token sequence for neural model"""
        tokens = ["<SOS>"]
        for stmt in self.statements:
            tokens.extend(stmt.to_tokens())
        tokens.append("<EOS>")
        return tokens

    def validate(self) -> bool:
        """Validate program syntax"""
        for stmt in self.statements:
            if not stmt.validate():
                return False
        return True

    def __len__(self) -> int:
        """Number of tokens in program"""
        return len(self.to_tokens())


# ============================================================================
# DSL Vocabulary Builder
# ============================================================================

class DSLVocabulary:
    """
    Build token vocabulary for neural model
    """

    def __init__(self):
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build complete vocabulary from DSL"""
        vocab = []

        # Special tokens
        vocab.extend(['<PAD>', '<SOS>', '<EOS>', '<UNK>'])

        # Structural tokens
        vocab.extend(['FOREACH', 'IF', 'THEN', 'ELSE', 'END_FOREACH', 'END_IF'])
        vocab.extend(['(', ')', '->', 'note', 'notes', 'chord'])

        # Iterators
        for it in IteratorType:
            vocab.append(it.value)

        # Filters
        for filt in FilterType:
            vocab.append(filt.value)

        # Operations
        for op in OperationType:
            vocab.append(op.value)

        # Aggregators
        for agg in AggregatorType:
            vocab.append(agg.value)

        # Keywords
        vocab.extend(['amount', 'reference_pitch', 'aggregate_value'])

        # Operators
        vocab.extend(['+', '-', '*', '/', '//', '%'])

        # Common numeric values
        for i in range(-24, 25):
            vocab.append(str(i))
        for i in range(0, 21):
            vocab.append(str(i / 10))

        # Build mappings
        for idx, token in enumerate(vocab):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [self.token_to_idx.get(t, self.token_to_idx['<UNK>']) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens"""
        return [self.idx_to_token.get(i, '<UNK>') for i in indices]


# ============================================================================
# Example DSL Programs
# ============================================================================

def create_example_programs() -> List[DSLProgram]:
    """Create example DSL programs for testing"""

    examples = []

    # Example 1: Simple transpose
    examples.append(DSLProgram(
        name="transpose",
        description="Transpose all notes by semitones",
        statements=[
            ForEach(
                iterator=IteratorType.ALL_NOTES,
                body=[
                    Operation(
                        op_type=OperationType.TRANSPOSE,
                        value=Value("amount * 24 - 12")
                    )
                ]
            )
        ]
    ))

    # Example 2: Octave doubling
    examples.append(DSLProgram(
        name="octave_doubling",
        description="Add octave below each note",
        statements=[
            ForEach(
                iterator=IteratorType.ALL_NOTES,
                body=[
                    Operation(
                        op_type=OperationType.DUPLICATE_NOTE,
                        value=Value(1)
                    ),
                    Operation(
                        op_type=OperationType.OCTAVE_SHIFT,
                        value=Value(-1),
                        target="duplicated_note"
                    )
                ]
            )
        ]
    ))

    # Example 3: Swing rhythm
    examples.append(DSLProgram(
        name="swing",
        description="Apply swing to off-beat notes",
        statements=[
            ForEach(
                iterator=IteratorType.NOTES_OFF_BEAT,
                body=[
                    Operation(
                        op_type=OperationType.TIME_SHIFT,
                        value=Value("amount * 0.1")
                    )
                ]
            )
        ]
    ))

    # Example 4: Drop-2 voicing
    examples.append(DSLProgram(
        name="drop_2_voicing",
        description="Drop second-highest note by octave in chords",
        statements=[
            ForEach(
                iterator=IteratorType.SIMULTANEOUS_NOTES,
                body=[
                    Aggregate(
                        aggregator=AggregatorType.SORT_BY_PITCH,
                        source="chord"
                    ),
                    If(
                        filter=FilterType.COUNT_GREATER,
                        threshold=Value(3),
                        then_body=[
                            Operation(
                                op_type=OperationType.OCTAVE_SHIFT,
                                value=Value(-1),
                                target="chord[-2]"
                            )
                        ]
                    )
                ]
            )
        ]
    ))

    return examples
