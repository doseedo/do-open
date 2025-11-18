#!/usr/bin/env python3
"""
L-System (Lindenmayer System) for Musical Composition

Implements formal grammar-based melody generation using L-Systems, a parallel rewriting
system originally developed for modeling plant growth. Applied to music, L-Systems can
generate complex, self-similar melodic patterns with hierarchical structure.

Features:
- Context-free L-Systems (basic production rules)
- Context-sensitive L-Systems (rules depend on neighbors)
- Parametric L-Systems (control pitch, rhythm, dynamics)
- Stochastic L-Systems (probabilistic rule selection)
- Pre-built musical grammars (Bach, minimalist, jazz)

References:
- Aristid Lindenmayer (1968) - "Mathematical models for cellular interactions"
- Przemyslaw Prusinkiewicz (1986) - "Score generation with L-systems"
- Gary William Flake (1998) - "The Computational Beauty of Nature"
- Mason & Saffle (1994) - "L-systems, melodies and musical structure"

Author: MIDI Generator Library - Agent 2
"""

from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from abc import ABC, abstractmethod

# Import from our music theory module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.music_theory import Scale, ScaleType, note_number_to_name, MIDDLE_C


# =============================================================================
# MUSICAL SYMBOL DEFINITIONS
# =============================================================================

class MusicalSymbol(Enum):
    """
    Symbols for musical L-Systems.

    Based on turtle graphics interpretation:
    - Pitch: F (forward/up), B (backward/down), U (up octave), D (down octave)
    - Rhythm: L (longer), S (shorter), R (rest)
    - Dynamics: + (louder), - (softer)
    - Structure: [ (push state), ] (pop state)
    - Notes: N (note), C (chord)
    """
    # Pitch movement
    FORWARD = 'F'        # Move up one scale degree
    BACKWARD = 'B'       # Move down one scale degree
    OCTAVE_UP = 'U'      # Jump up one octave
    OCTAVE_DOWN = 'D'    # Jump down one octave
    JUMP_THIRD = 'T'     # Jump up a third
    JUMP_FIFTH = 'Q'     # Jump up a fifth

    # Rhythm
    LENGTHEN = 'L'       # Double note duration
    SHORTEN = 'S'        # Halve note duration
    REST = 'R'           # Insert rest

    # Dynamics
    LOUDER = '+'         # Increase velocity
    SOFTER = '-'         # Decrease velocity

    # Structure
    PUSH = '['           # Save current state
    POP = ']'            # Restore saved state

    # Note generation
    NOTE = 'N'           # Generate a note
    CHORD = 'C'          # Generate a chord

    # Non-terminals (for production rules)
    AXIOM = 'A'          # Start symbol
    MOTIF = 'M'          # Melodic motif
    PHRASE = 'P'         # Musical phrase
    VARIATION = 'V'      # Variation
    CLIMAX = 'X'         # Climax point


# =============================================================================
# L-SYSTEM DATA STRUCTURES
# =============================================================================

@dataclass
class ProductionRule:
    """
    A production rule for L-System rewriting.

    Format: predecessor → successor
    Example: 'A' → 'FNFN[+FL][-FR]'
    """
    predecessor: str
    successor: Union[str, List[str]]  # Can be string or list of alternatives
    probability: float = 1.0  # For stochastic L-systems
    condition: Optional[Callable] = None  # For parametric L-systems

    def __repr__(self):
        if isinstance(self.successor, list):
            return f"{self.predecessor} → {self.successor} (p={self.probability})"
        return f"{self.predecessor} → {self.successor}"


@dataclass
class MusicalState:
    """
    Current state of the musical turtle during L-System interpretation.
    """
    pitch: int = MIDDLE_C                # Current MIDI pitch
    duration: float = 1.0                # Current note duration (in beats)
    velocity: int = 80                   # Current velocity (0-127)
    position: float = 0.0                # Current time position
    scale: Optional[Scale] = None        # Current scale context

    def copy(self) -> 'MusicalState':
        """Create a deep copy of the current state"""
        return MusicalState(
            pitch=self.pitch,
            duration=self.duration,
            velocity=self.velocity,
            position=self.position,
            scale=self.scale
        )


@dataclass
class Note:
    """A musical note with timing and dynamics"""
    pitch: int          # MIDI note number
    start: float        # Start time in beats
    duration: float     # Duration in beats
    velocity: int       # Velocity (0-127)

    def __repr__(self):
        return f"Note({note_number_to_name(self.pitch)}, t={self.start:.2f}, dur={self.duration:.2f}, vel={self.velocity})"


# =============================================================================
# L-SYSTEM ENGINE
# =============================================================================

class LSystem:
    """
    Base L-System implementation.

    An L-System consists of:
    - Axiom: Starting symbol/string
    - Rules: Production rules for rewriting
    - Iterations: Number of derivation steps
    """

    def __init__(self, axiom: str, rules: List[ProductionRule], iterations: int = 4):
        """
        Initialize L-System.

        Args:
            axiom: Starting string (e.g., 'A')
            rules: List of production rules
            iterations: Number of derivation iterations
        """
        self.axiom = axiom
        self.rules = rules
        self.iterations = iterations

        # Build rule lookup for efficiency
        self._rule_dict: Dict[str, List[ProductionRule]] = {}
        for rule in rules:
            if rule.predecessor not in self._rule_dict:
                self._rule_dict[rule.predecessor] = []
            self._rule_dict[rule.predecessor].append(rule)

    def derive(self, iterations: Optional[int] = None) -> str:
        """
        Derive the L-System for specified iterations.

        Args:
            iterations: Number of derivations (uses self.iterations if None)

        Returns:
            Derived string after all iterations
        """
        if iterations is None:
            iterations = self.iterations

        current = self.axiom

        for i in range(iterations):
            current = self._derive_step(current)

        return current

    def _derive_step(self, string: str) -> str:
        """
        Perform one derivation step (parallel rewriting).

        Args:
            string: Current string

        Returns:
            String after one derivation
        """
        result = []

        for symbol in string:
            replacement = self._apply_rules(symbol)
            result.append(replacement)

        return ''.join(result)

    def _apply_rules(self, symbol: str) -> str:
        """
        Apply production rules to a symbol.

        Args:
            symbol: Symbol to rewrite

        Returns:
            Replacement string
        """
        if symbol not in self._rule_dict:
            # No rule found, return symbol unchanged
            return symbol

        rules = self._rule_dict[symbol]

        # Stochastic selection if multiple rules
        if len(rules) == 1:
            rule = rules[0]
        else:
            # Weighted random selection based on probabilities
            total_prob = sum(r.probability for r in rules)
            rand = random.uniform(0, total_prob)
            cumulative = 0.0
            rule = rules[0]

            for r in rules:
                cumulative += r.probability
                if rand <= cumulative:
                    rule = r
                    break

        # Return successor
        if isinstance(rule.successor, list):
            return random.choice(rule.successor)
        return rule.successor

    def interpret_musical(self,
                         string: str,
                         scale: Scale,
                         initial_pitch: int = MIDDLE_C,
                         base_duration: float = 0.5) -> List[Note]:
        """
        Interpret L-System string as musical notes.

        Args:
            string: Derived L-System string
            scale: Musical scale for pitch context
            initial_pitch: Starting pitch
            base_duration: Base note duration

        Returns:
            List of Note objects
        """
        state = MusicalState(
            pitch=initial_pitch,
            duration=base_duration,
            velocity=80,
            position=0.0,
            scale=scale
        )

        state_stack: List[MusicalState] = []
        notes: List[Note] = []

        for symbol in string:
            if symbol == 'F':  # Forward (up one scale degree)
                state.pitch = self._next_scale_degree(state.pitch, scale, 1)

            elif symbol == 'B':  # Backward (down one scale degree)
                state.pitch = self._next_scale_degree(state.pitch, scale, -1)

            elif symbol == 'U':  # Octave up
                state.pitch = min(127, state.pitch + 12)

            elif symbol == 'D':  # Octave down
                state.pitch = max(0, state.pitch - 12)

            elif symbol == 'T':  # Jump third
                state.pitch = self._next_scale_degree(state.pitch, scale, 2)

            elif symbol == 'Q':  # Jump fifth
                state.pitch = self._next_scale_degree(state.pitch, scale, 4)

            elif symbol == 'L':  # Lengthen
                state.duration *= 2.0

            elif symbol == 'S':  # Shorten
                state.duration *= 0.5

            elif symbol == 'R':  # Rest
                state.position += state.duration

            elif symbol == '+':  # Louder
                state.velocity = min(127, state.velocity + 10)

            elif symbol == '-':  # Softer
                state.velocity = max(1, state.velocity - 10)

            elif symbol == '[':  # Push state
                state_stack.append(state.copy())

            elif symbol == ']':  # Pop state
                if state_stack:
                    state = state_stack.pop()

            elif symbol == 'N':  # Generate note
                note = Note(
                    pitch=state.pitch,
                    start=state.position,
                    duration=state.duration,
                    velocity=state.velocity
                )
                notes.append(note)
                state.position += state.duration

        return notes

    def _next_scale_degree(self, current_pitch: int, scale: Scale, steps: int) -> int:
        """
        Move to next scale degree.

        Args:
            current_pitch: Current MIDI pitch
            scale: Scale context
            steps: Number of scale degrees to move (positive or negative)

        Returns:
            New MIDI pitch
        """
        # Get all scale notes in a wide range
        scale_notes = scale.get_notes(octaves=10, ascending=True)

        # Find closest scale note to current pitch
        closest_idx = min(range(len(scale_notes)),
                         key=lambda i: abs(scale_notes[i] - current_pitch))

        # Move by steps
        new_idx = closest_idx + steps
        new_idx = max(0, min(len(scale_notes) - 1, new_idx))

        return scale_notes[new_idx]


# =============================================================================
# CONTEXT-SENSITIVE L-SYSTEM
# =============================================================================

@dataclass
class ContextSensitiveRule(ProductionRule):
    """
    Context-sensitive production rule.

    Format: left < predecessor > right → successor
    Example: 'F' < 'N' > 'F' → 'FNF'  (only applies when N is between two Fs)
    """
    left_context: Optional[str] = None
    right_context: Optional[str] = None


class ContextSensitiveLSystem(LSystem):
    """
    Context-sensitive L-System where rules can depend on neighboring symbols.
    """

    def _derive_step(self, string: str) -> str:
        """Override to handle context-sensitive rules"""
        result = []

        for i, symbol in enumerate(string):
            left = string[i-1] if i > 0 else None
            right = string[i+1] if i < len(string) - 1 else None

            replacement = self._apply_rules_contextual(symbol, left, right)
            result.append(replacement)

        return ''.join(result)

    def _apply_rules_contextual(self, symbol: str, left: Optional[str], right: Optional[str]) -> str:
        """
        Apply context-sensitive rules.

        Args:
            symbol: Symbol to rewrite
            left: Left neighbor (or None)
            right: Right neighbor (or None)

        Returns:
            Replacement string
        """
        if symbol not in self._rule_dict:
            return symbol

        # Filter rules by context
        applicable_rules = []
        for rule in self._rule_dict[symbol]:
            if isinstance(rule, ContextSensitiveRule):
                if rule.left_context and left != rule.left_context:
                    continue
                if rule.right_context and right != rule.right_context:
                    continue
            applicable_rules.append(rule)

        if not applicable_rules:
            return symbol

        # Select rule (stochastically if multiple)
        if len(applicable_rules) == 1:
            rule = applicable_rules[0]
        else:
            total_prob = sum(r.probability for r in applicable_rules)
            rand = random.uniform(0, total_prob)
            cumulative = 0.0
            rule = applicable_rules[0]

            for r in applicable_rules:
                cumulative += r.probability
                if rand <= cumulative:
                    rule = r
                    break

        if isinstance(rule.successor, list):
            return random.choice(rule.successor)
        return rule.successor


# =============================================================================
# PRE-BUILT MUSICAL GRAMMARS
# =============================================================================

class MusicalGrammar:
    """Collection of pre-built musical L-System grammars"""

    @staticmethod
    def bach_chorale() -> LSystem:
        """
        Bach-inspired chorale melody generator.

        Characteristics:
        - Stepwise motion predominates
        - Occasional leaps (thirds, fifths)
        - Balanced ascending/descending motion
        - Phrase structure with cadences
        """
        rules = [
            ProductionRule('A', 'PNPNPNC'),  # Axiom -> 3 phrases + cadence
            ProductionRule('P', 'MNMNV'),     # Phrase -> 2 motifs + variation
            ProductionRule('M', 'FNFNFB'),    # Motif -> stepwise motion
            ProductionRule('V', 'FTFB'),      # Variation -> with leap
            ProductionRule('C', 'FBFBFN'),    # Cadence -> descending resolution
        ]
        return LSystem('A', rules, iterations=2)

    @staticmethod
    def minimalist() -> LSystem:
        """
        Minimalist pattern generator (Reich/Glass style).

        Characteristics:
        - Repetitive patterns
        - Gradual transformation
        - Phase shifting effect
        - Additive process
        """
        rules = [
            ProductionRule('A', 'MNM'),
            ProductionRule('M', 'FNFN[+FN][-FN]'),  # Pattern with variations
        ]
        return LSystem('A', rules, iterations=3)

    @staticmethod
    def jazz_bebop() -> LSystem:
        """
        Bebop-style melody generator.

        Characteristics:
        - Scalar runs
        - Arpeggios (thirds, fifths)
        - Rhythmic variety
        - Chromatic passing tones
        """
        rules = [
            ProductionRule('A', 'MNVNMN'),
            ProductionRule('M', 'SFNFNFNFN'),      # Fast scalar run
            ProductionRule('V', 'TNTQTN'),          # Arpeggio pattern
            ProductionRule('N', 'N', 0.7),          # Note
            ProductionRule('N', 'R', 0.3),          # Or rest (stochastic)
        ]
        return LSystem('A', rules, iterations=2)

    @staticmethod
    def fractal_melody() -> LSystem:
        """
        Self-similar fractal melody.

        Characteristics:
        - Hierarchical structure
        - Self-similarity at different scales
        - Branching phrases
        """
        rules = [
            ProductionRule('A', 'FN[+A][-A]FN'),  # Recursive branching
        ]
        return LSystem('A', rules, iterations=4)

    @staticmethod
    def romantic_expression() -> LSystem:
        """
        Romantic-era expressive melody.

        Characteristics:
        - Wide leaps
        - Dynamic contrast
        - Rubato-like rhythm variation
        - Climax points
        """
        rules = [
            ProductionRule('A', 'P+XP'),          # Phrase -> Climax -> Phrase
            ProductionRule('P', 'MNMN'),
            ProductionRule('M', 'FNFTFN'),        # Stepwise + leap
            ProductionRule('X', 'U+TNTN-D'),      # Climax: high, loud, then resolve
        ]
        return LSystem('A', rules, iterations=2)

    @staticmethod
    def pentatonic_folk() -> LSystem:
        """
        Pentatonic folk melody (use with pentatonic scale).

        Characteristics:
        - Pentatonic vocabulary
        - Simple rhythms
        - Repetitive motifs
        - Question-answer structure
        """
        rules = [
            ProductionRule('A', 'QNANQNANPN'),    # Question-answer
            ProductionRule('Q', 'FNFNFT'),         # Question phrase
            ProductionRule('P', 'FBFBFN'),         # Answer phrase
        ]
        return LSystem('A', rules, iterations=2)

    @staticmethod
    def twelve_tone() -> LSystem:
        """
        Twelve-tone serial technique inspired.

        Characteristics:
        - Wide intervallic leaps
        - Avoid repetition
        - Angular contour
        - Atonal aesthetic
        """
        rules = [
            ProductionRule('A', 'RNRNRNRN'),
            ProductionRule('R', ['QNTN', 'UNDN', 'BNFN']),  # Random intervals
        ]
        return LSystem('A', rules, iterations=3)

    @staticmethod
    def renaissance_polyphony() -> LSystem:
        """
        Renaissance-style melodic line.

        Characteristics:
        - Smooth voice leading
        - Modal harmony
        - Balanced contour
        - Cadential patterns
        """
        rules = [
            ProductionRule('A', 'MNMNC'),
            ProductionRule('M', 'FNFNFBFB'),      # Balanced motion
            ProductionRule('C', 'FBFNFB'),         # Cadence
        ]
        return LSystem('A', rules, iterations=2)


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("L-SYSTEM MUSICAL COMPOSITION EXAMPLES")
    print("=" * 80)

    # Create a C major scale
    c_major = Scale(MIDDLE_C, ScaleType.MAJOR)

    # Example 1: Bach Chorale
    print("\n1. BACH CHORALE STYLE")
    print("-" * 80)
    bach = MusicalGrammar.bach_chorale()
    derived = bach.derive()
    print(f"Derived string: {derived[:100]}...")
    notes = bach.interpret_musical(derived, c_major)
    print(f"Generated {len(notes)} notes")
    print(f"First 5 notes: {notes[:5]}")

    # Example 2: Minimalist
    print("\n2. MINIMALIST PATTERN")
    print("-" * 80)
    minimalist = MusicalGrammar.minimalist()
    derived = minimalist.derive()
    print(f"Derived string: {derived[:100]}...")
    notes = minimalist.interpret_musical(derived, c_major)
    print(f"Generated {len(notes)} notes")
    print(f"Pattern structure visible in: {notes[:8]}")

    # Example 3: Jazz Bebop
    print("\n3. JAZZ BEBOP")
    print("-" * 80)
    bebop = MusicalGrammar.jazz_bebop()
    derived = bebop.derive()
    print(f"Derived string: {derived[:100]}...")
    notes = bebop.interpret_musical(derived, c_major, base_duration=0.25)
    print(f"Generated {len(notes)} notes")
    print(f"Fast notes: {notes[:6]}")

    # Example 4: Custom L-System
    print("\n4. CUSTOM L-SYSTEM")
    print("-" * 80)
    custom_rules = [
        ProductionRule('A', 'FNFN[+UFNFN][-DFNFN]'),
        ProductionRule('F', 'FN'),
    ]
    custom = LSystem('A', custom_rules, iterations=2)
    derived = custom.derive()
    print(f"Derived string: {derived}")
    notes = custom.interpret_musical(derived, c_major)
    print(f"Generated {len(notes)} notes")
    print(f"All notes: {notes}")

    # Example 5: Fractal Self-Similar
    print("\n5. FRACTAL SELF-SIMILAR MELODY")
    print("-" * 80)
    fractal = MusicalGrammar.fractal_melody()
    derived = fractal.derive(iterations=3)  # Don't go too deep!
    print(f"Derived string length: {len(derived)}")
    print(f"First 50 symbols: {derived[:50]}...")
    notes = fractal.interpret_musical(derived, c_major)
    print(f"Generated {len(notes)} notes with hierarchical structure")

    print("\n" + "=" * 80)
    print("L-System implementation complete!")
    print("=" * 80)
