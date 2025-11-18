#!/usr/bin/env python3
"""
Cellular Automata for Musical Composition

Implements 1D and 2D cellular automata (CA) for algorithmic music generation.
Cellular automata are discrete models where cells evolve based on local rules,
creating emergent patterns from simple rules - perfect for generating musical structures.

Features:
- Elementary Cellular Automata (1D, Wolfram rules)
- Conway's Game of Life (2D)
- Custom musical CA rules
- Multiple interpretation modes (pitch, rhythm, dynamics)
- Pattern evolution and emergence detection

Applications:
- Generative melodies from evolving patterns
- Rhythmic sequences from CA evolution
- Harmonic progression from 2D CA
- Ambient/experimental texture generation

References:
- Stephen Wolfram (1983) - "Statistical mechanics of cellular automata"
- John Conway (1970) - "Game of Life"
- Eduardo Miranda (2001) - "Composing Music with Computers"
- Bill Manaris et al. (2005) - "Investigating Esperanto's Linguistic Properties"

Author: MIDI Generator Library - Agent 2
"""

from typing import List, Tuple, Optional, Callable, Dict, Set
from dataclasses import dataclass
from enum import Enum
import random

# Import from our music theory module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.music_theory import Scale, ScaleType, MIDDLE_C, note_number_to_name


# =============================================================================
# CELLULAR AUTOMATON DATA STRUCTURES
# =============================================================================

class CAState(Enum):
    """Cell states for binary CA"""
    DEAD = 0
    ALIVE = 1


@dataclass
class MusicEvent:
    """Musical event generated from CA"""
    pitch: int           # MIDI pitch
    start: float         # Start time (beats)
    duration: float      # Duration (beats)
    velocity: int        # Velocity (0-127)
    generation: int      # Which CA generation produced this


# =============================================================================
# 1D CELLULAR AUTOMATON (ELEMENTARY CA)
# =============================================================================

class ElementaryCA:
    """
    Elementary Cellular Automaton (1D).

    Uses Wolfram's elementary CA rules (256 possible rules numbered 0-255).
    Each cell's next state depends on itself and its two neighbors.

    Famous rules:
    - Rule 30: Chaotic, used in Mathematica's random number generator
    - Rule 110: Turing complete
    - Rule 90: Sierpiński triangle fractal
    - Rule 184: Traffic flow model
    """

    def __init__(self, width: int = 64, rule: int = 30):
        """
        Initialize Elementary CA.

        Args:
            width: Number of cells
            rule: Wolfram rule number (0-255)
        """
        self.width = width
        self.rule = rule
        self.rule_table = self._generate_rule_table(rule)
        self.state = [0] * width
        self.history: List[List[int]] = []

    def _generate_rule_table(self, rule: int) -> Dict[Tuple[int, int, int], int]:
        """
        Generate lookup table for rule.

        Args:
            rule: Rule number (0-255)

        Returns:
            Dictionary mapping (left, center, right) -> next state
        """
        binary = format(rule, '08b')[::-1]  # Reverse for correct ordering
        neighborhoods = [
            (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
            (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)
        ]
        return {neighborhood: int(binary[i]) for i, neighborhood in enumerate(neighborhoods)}

    def set_initial_state(self, state: Optional[List[int]] = None, mode: str = 'center'):
        """
        Set initial state of CA.

        Args:
            state: Custom initial state (list of 0s and 1s)
            mode: 'center', 'random', 'edges', or 'custom'
        """
        if state is not None:
            self.state = state[:self.width]
        elif mode == 'center':
            self.state = [0] * self.width
            self.state[self.width // 2] = 1
        elif mode == 'random':
            self.state = [random.randint(0, 1) for _ in range(self.width)]
        elif mode == 'edges':
            self.state = [0] * self.width
            self.state[0] = 1
            self.state[-1] = 1

        self.history = [self.state.copy()]

    def step(self):
        """Evolve CA one generation"""
        new_state = [0] * self.width

        for i in range(self.width):
            left = self.state[(i - 1) % self.width]
            center = self.state[i]
            right = self.state[(i + 1) % self.width]

            new_state[i] = self.rule_table[(left, center, right)]

        self.state = new_state
        self.history.append(self.state.copy())

    def evolve(self, generations: int):
        """
        Evolve CA for multiple generations.

        Args:
            generations: Number of generations to evolve
        """
        for _ in range(generations):
            self.step()

    def to_melody(self,
                  scale: Scale,
                  mode: str = 'horizontal',
                  base_duration: float = 0.25) -> List[MusicEvent]:
        """
        Convert CA evolution to melody.

        Args:
            scale: Musical scale for pitch mapping
            mode: 'horizontal' (time->x, pitch->y) or 'vertical' (time->y, pitch->x)
            base_duration: Base note duration

        Returns:
            List of musical events
        """
        notes = []
        scale_notes = scale.get_notes(octaves=6)

        if mode == 'horizontal':
            # Each generation is a time step, active cells are pitches
            for gen_idx, generation in enumerate(self.history):
                time = gen_idx * base_duration
                for cell_idx, cell_value in enumerate(generation):
                    if cell_value == 1:
                        # Map cell position to pitch
                        pitch_idx = cell_idx % len(scale_notes)
                        pitch = scale_notes[pitch_idx]

                        notes.append(MusicEvent(
                            pitch=pitch,
                            start=time,
                            duration=base_duration,
                            velocity=80,
                            generation=gen_idx
                        ))

        elif mode == 'vertical':
            # Each cell position is a time step, read generations as pitches
            for cell_idx in range(self.width):
                time = cell_idx * base_duration
                # Count active cells in this column across all generations
                active_count = sum(gen[cell_idx] for gen in self.history)

                if active_count > 0:
                    # Map activity to pitch
                    pitch_idx = active_count % len(scale_notes)
                    pitch = scale_notes[pitch_idx]

                    notes.append(MusicEvent(
                        pitch=pitch,
                        start=time,
                        duration=base_duration,
                        velocity=min(127, 40 + active_count * 10),
                        generation=0
                    ))

        return notes

    def to_rhythm(self, base_duration: float = 0.25) -> List[Tuple[float, float]]:
        """
        Convert CA to rhythmic pattern.

        Args:
            base_duration: Duration of each cell

        Returns:
            List of (onset_time, duration) tuples
        """
        rhythm = []

        for gen_idx, generation in enumerate(self.history):
            time = gen_idx * base_duration
            for cell_idx, cell_value in enumerate(generation):
                if cell_value == 1:
                    rhythm.append((time, base_duration))

        return rhythm

    def __repr__(self):
        """Visual representation of current state"""
        return ''.join(['█' if cell else '·' for cell in self.state])


# =============================================================================
# 2D CELLULAR AUTOMATON (GAME OF LIFE)
# =============================================================================

class GameOfLife:
    """
    Conway's Game of Life (2D CA).

    Rules:
    1. Any live cell with 2-3 live neighbors survives
    2. Any dead cell with exactly 3 live neighbors becomes alive
    3. All other cells die or stay dead

    Produces fascinating emergent patterns:
    - Still lifes (stable)
    - Oscillators (periodic)
    - Spaceships (moving patterns)
    - Chaotic regions
    """

    def __init__(self, width: int = 32, height: int = 32):
        """
        Initialize Game of Life.

        Args:
            width: Grid width
            height: Grid height
        """
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.history: List[List[List[int]]] = []

    def set_initial_state(self, mode: str = 'random', density: float = 0.3):
        """
        Set initial state.

        Args:
            mode: 'random', 'glider', 'pulsar', 'gosper_gun'
            density: For random mode, probability of live cell
        """
        if mode == 'random':
            self.grid = [[1 if random.random() < density else 0 for _ in range(self.width)]
                        for _ in range(self.height)]

        elif mode == 'glider':
            # Classic glider pattern
            self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
            center_y, center_x = self.height // 2, self.width // 2
            glider = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
            for dy, dx in glider:
                y, x = center_y + dy, center_x + dx
                if 0 <= y < self.height and 0 <= x < self.width:
                    self.grid[y][x] = 1

        elif mode == 'pulsar':
            # Pulsar oscillator (period 3)
            self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
            center_y, center_x = self.height // 2, self.width // 2
            # Simplified pulsar pattern
            offsets = [
                (-3, -1), (-3, 1), (-1, -3), (-1, 3),
                (1, -3), (1, 3), (3, -1), (3, 1)
            ]
            for dy, dx in offsets:
                y, x = center_y + dy, center_x + dx
                if 0 <= y < self.height and 0 <= x < self.width:
                    self.grid[y][x] = 1

        elif mode == 'blinker':
            # Simple oscillator
            self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
            center_y, center_x = self.height // 2, self.width // 2
            for x in range(max(0, center_x-1), min(self.width, center_x+2)):
                self.grid[center_y][x] = 1

        self.history = [[row[:] for row in self.grid]]

    def count_neighbors(self, y: int, x: int) -> int:
        """
        Count live neighbors of cell at (y, x).

        Args:
            y: Row index
            x: Column index

        Returns:
            Number of live neighbors (0-8)
        """
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny = (y + dy) % self.height
                nx = (x + dx) % self.width
                count += self.grid[ny][nx]
        return count

    def step(self):
        """Evolve one generation"""
        new_grid = [[0 for _ in range(self.width)] for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.count_neighbors(y, x)
                current = self.grid[y][x]

                # Apply Game of Life rules
                if current == 1:
                    # Live cell
                    if neighbors in [2, 3]:
                        new_grid[y][x] = 1  # Survive
                    else:
                        new_grid[y][x] = 0  # Die
                else:
                    # Dead cell
                    if neighbors == 3:
                        new_grid[y][x] = 1  # Birth

        self.grid = new_grid
        self.history.append([row[:] for row in self.grid])

    def evolve(self, generations: int):
        """Evolve for multiple generations"""
        for _ in range(generations):
            self.step()

    def to_melody(self, scale: Scale, base_duration: float = 0.25) -> List[MusicEvent]:
        """
        Convert 2D CA to melody.

        Maps:
        - X axis -> Time
        - Y axis -> Pitch
        - Cell state -> Note on/off

        Args:
            scale: Musical scale
            base_duration: Duration per cell

        Returns:
            List of musical events
        """
        notes = []
        scale_notes = scale.get_notes(octaves=6)

        for gen_idx, grid in enumerate(self.history):
            for y in range(self.height):
                for x in range(self.width):
                    if grid[y][x] == 1:
                        time = x * base_duration + (gen_idx * self.width * base_duration)

                        # Map Y position to pitch
                        pitch_idx = y % len(scale_notes)
                        pitch = scale_notes[pitch_idx]

                        # Velocity based on neighborhood density
                        neighbors_sum = 0
                        for ny in range(max(0, y-1), min(self.height, y+2)):
                            for nx in range(max(0, x-1), min(self.width, x+2)):
                                neighbors_sum += grid[ny][nx]
                        velocity = min(127, 40 + neighbors_sum * 10)

                        notes.append(MusicEvent(
                            pitch=pitch,
                            start=time,
                            duration=base_duration,
                            velocity=velocity,
                            generation=gen_idx
                        ))

        return notes

    def to_harmony(self, scale: Scale, slice_duration: float = 2.0) -> List[List[MusicEvent]]:
        """
        Convert CA to harmonic progression.

        Each generation becomes a vertical slice (chord).

        Args:
            scale: Musical scale
            slice_duration: Duration of each chord

        Returns:
            List of chords (each chord is list of MusicEvents)
        """
        chords = []
        scale_notes = scale.get_notes(octaves=4)

        for gen_idx, grid in enumerate(self.history):
            chord = []
            time = gen_idx * slice_duration

            # Count active cells in each row
            row_activity = [sum(grid[y]) for y in range(self.height)]

            # Generate chord from active rows
            for y, activity in enumerate(row_activity):
                if activity > 0:
                    pitch_idx = y % len(scale_notes)
                    pitch = scale_notes[pitch_idx]

                    chord.append(MusicEvent(
                        pitch=pitch,
                        start=time,
                        duration=slice_duration,
                        velocity=min(127, 40 + int(activity) * 5),
                        generation=gen_idx
                    ))

            if chord:
                chords.append(chord)

        return chords


# =============================================================================
# MUSICAL CA RULES (CUSTOM)
# =============================================================================

class MusicalCA:
    """
    Custom CA with musically-designed rules.

    Unlike elementary CA or Game of Life, these rules are specifically
    designed to produce musically interesting patterns.
    """

    def __init__(self, width: int = 32):
        """Initialize Musical CA"""
        self.width = width
        self.state = [0] * width
        self.history: List[List[int]] = []

    def set_initial_state(self, mode: str = 'seed'):
        """Set initial state"""
        if mode == 'seed':
            self.state = [0] * self.width
            self.state[self.width // 2] = 4  # Start with medium value
        elif mode == 'random':
            self.state = [random.randint(0, 7) for _ in range(self.width)]

        self.history = [self.state.copy()]

    def step_melodic(self):
        """
        Melodic CA rule: Creates stepwise motion.

        Rule: Each cell becomes the average of neighbors, creating smooth transitions.
        """
        new_state = [0] * self.width

        for i in range(self.width):
            left = self.state[(i - 1) % self.width]
            center = self.state[i]
            right = self.state[(i + 1) % self.width]

            # Average of neighbors (smooth motion)
            avg = (left + center + right) // 3

            # Add slight randomness for variation
            variation = random.randint(-1, 1)
            new_state[i] = max(0, min(7, avg + variation))

        self.state = new_state
        self.history.append(self.state.copy())

    def step_rhythmic(self):
        """
        Rhythmic CA rule: Creates rhythmic patterns.

        Rule: Active cells spread rhythmically based on binary patterns.
        """
        new_state = [0] * self.width

        for i in range(self.width):
            left = self.state[(i - 1) % self.width]
            center = self.state[i]
            right = self.state[(i + 1) % self.width]

            # Rhythm emerges from modulo patterns
            if (left + center + right) % 3 == 0:
                new_state[i] = (center + 1) % 8
            else:
                new_state[i] = max(0, center - 1)

        self.state = new_state
        self.history.append(self.state.copy())

    def evolve(self, generations: int, rule_type: str = 'melodic'):
        """
        Evolve CA.

        Args:
            generations: Number of generations
            rule_type: 'melodic' or 'rhythmic'
        """
        for _ in range(generations):
            if rule_type == 'melodic':
                self.step_melodic()
            else:
                self.step_rhythmic()


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CELLULAR AUTOMATA MUSICAL COMPOSITION EXAMPLES")
    print("=" * 80)

    # Create a C major scale
    c_major = Scale(MIDDLE_C, ScaleType.MAJOR)

    # Example 1: Elementary CA (Rule 30 - Chaotic)
    print("\n1. ELEMENTARY CA - RULE 30 (CHAOTIC)")
    print("-" * 80)
    ca30 = ElementaryCA(width=32, rule=30)
    ca30.set_initial_state(mode='center')
    ca30.evolve(16)
    print("CA Evolution (first 8 generations):")
    for gen in ca30.history[:8]:
        print(''.join(['█' if cell else '·' for cell in gen]))

    melody30 = ca30.to_melody(c_major, mode='horizontal')
    print(f"\nGenerated {len(melody30)} notes from Rule 30")
    print(f"Sample notes: {melody30[:5]}")

    # Example 2: Elementary CA (Rule 110 - Complex)
    print("\n2. ELEMENTARY CA - RULE 110 (TURING COMPLETE)")
    print("-" * 80)
    ca110 = ElementaryCA(width=32, rule=110)
    ca110.set_initial_state(mode='random')
    ca110.evolve(16)
    melody110 = ca110.to_melody(c_major, mode='horizontal')
    print(f"Generated {len(melody110)} notes from Rule 110")

    # Example 3: Game of Life - Glider
    print("\n3. GAME OF LIFE - GLIDER PATTERN")
    print("-" * 80)
    gol = GameOfLife(width=16, height=16)
    gol.set_initial_state(mode='glider')
    gol.evolve(10)
    print(f"Evolved {len(gol.history)} generations")

    melody_gol = gol.to_melody(c_major, base_duration=0.5)
    print(f"Generated {len(melody_gol)} notes from Game of Life")

    harmony_gol = gol.to_harmony(c_major, slice_duration=1.0)
    print(f"Generated {len(harmony_gol)} chords")
    print(f"First chord: {harmony_gol[0][:3] if harmony_gol else 'None'}")

    # Example 4: Musical CA
    print("\n4. MUSICAL CA - MELODIC RULE")
    print("-" * 80)
    mca = MusicalCA(width=24)
    mca.set_initial_state(mode='seed')
    mca.evolve(12, rule_type='melodic')

    print("Musical CA Evolution (first 6 generations):")
    for gen in mca.history[:6]:
        print(' '.join([str(cell) for cell in gen]))

    print("\n" + "=" * 80)
    print("Cellular Automata implementation complete!")
    print("=" * 80)
