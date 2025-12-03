"""
SEQUITUR Grammar Induction Algorithm

SEQUITUR discovers a context-free grammar from a sequence in O(n) time and space.
It enforces two key constraints:
1. Digram Uniqueness: No pair of symbols appears more than once in the grammar
2. Rule Utility: Every rule is used more than once

This produces a grammar that guarantees 100% lossless reconstruction while
discovering hierarchical patterns (repeated subsequences become rules).

References:
- Nevill-Manning & Witten (1997): "Identifying Hierarchical Structure in Sequences"
- Original algorithm: https://sequitur.info/

Author: SEQUITUR Implementation for Musical Pattern Discovery
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Iterator, Union
from collections import defaultdict
import numpy as np


@dataclass
class Symbol:
    """
    A symbol in the grammar - either a terminal (input token) or non-terminal (rule reference).

    Symbols form a doubly-linked list within rules for O(1) insertion/deletion.
    """
    value: Union[int, 'Rule']  # Terminal (int) or non-terminal (Rule)

    # Doubly-linked list pointers
    prev: Optional['Symbol'] = None
    next: Optional['Symbol'] = None

    # The rule this symbol belongs to (for utility counting)
    rule: Optional['Rule'] = None

    @property
    def is_terminal(self) -> bool:
        # Handle both Python int and numpy integer types
        return isinstance(self.value, (int, np.integer))

    @property
    def is_non_terminal(self) -> bool:
        return isinstance(self.value, Rule)

    def clone(self) -> 'Symbol':
        """Create a copy of this symbol (without list pointers)."""
        return Symbol(value=self.value)

    def __hash__(self):
        if self.is_terminal:
            return hash(('T', self.value))
        else:
            return hash(('NT', id(self.value)))

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        if self.is_terminal != other.is_terminal:
            return False
        if self.is_terminal:
            return self.value == other.value
        else:
            return self.value is other.value

    def __repr__(self):
        if self.is_terminal:
            return f"T({self.value})"
        else:
            return f"R{self.value.id}"


@dataclass
class Rule:
    """
    A grammar rule: left-hand side -> right-hand side.

    The right-hand side is a doubly-linked list of symbols for O(1) operations.
    """
    id: int

    # Doubly-linked list: guard node that points to first/last symbols
    guard: Symbol = field(default_factory=lambda: Symbol(value=None))

    # Usage count: how many times this rule is referenced
    usage_count: int = 0

    def __post_init__(self):
        # Initialize guard to point to itself (empty rule)
        self.guard.prev = self.guard
        self.guard.next = self.guard
        self.guard.rule = self

    @property
    def is_empty(self) -> bool:
        return self.guard.next is self.guard

    @property
    def first(self) -> Optional[Symbol]:
        """First symbol in the rule, or None if empty."""
        if self.is_empty:
            return None
        return self.guard.next

    @property
    def last(self) -> Optional[Symbol]:
        """Last symbol in the rule, or None if empty."""
        if self.is_empty:
            return None
        return self.guard.prev

    @property
    def length(self) -> int:
        """Number of symbols in this rule."""
        count = 0
        sym = self.guard.next
        while sym is not self.guard:
            count += 1
            sym = sym.next
        return count

    def symbols(self) -> Iterator[Symbol]:
        """Iterate over all symbols in the rule."""
        sym = self.guard.next
        while sym is not self.guard:
            yield sym
            sym = sym.next

    def append(self, sym: Symbol) -> Symbol:
        """Append a symbol to the end of the rule."""
        sym.rule = self
        sym.prev = self.guard.prev
        sym.next = self.guard
        self.guard.prev.next = sym
        self.guard.prev = sym
        return sym

    def insert_after(self, after: Symbol, sym: Symbol) -> Symbol:
        """Insert sym after the given symbol."""
        sym.rule = self
        sym.prev = after
        sym.next = after.next
        after.next.prev = sym
        after.next = sym
        return sym

    def remove(self, sym: Symbol) -> Symbol:
        """Remove a symbol from the rule."""
        sym.prev.next = sym.next
        sym.next.prev = sym.prev
        sym.rule = None
        return sym

    def to_list(self) -> List[Union[int, int]]:
        """
        Convert to list representation.

        Returns list of (type, value) where type is 0 for terminal, 1 for non-terminal.
        """
        result = []
        for sym in self.symbols():
            if sym.is_terminal:
                result.append(sym.value)
            else:
                result.append(('R', sym.value.id))
        return result

    def expand(self) -> List[int]:
        """Fully expand this rule to terminal sequence."""
        result = []
        for sym in self.symbols():
            if sym.is_terminal:
                result.append(sym.value)
            else:
                result.extend(sym.value.expand())
        return result

    def __repr__(self):
        symbols_str = ' '.join(repr(s) for s in self.symbols())
        return f"R{self.id} -> {symbols_str}"


def _digram_key(s1: Symbol, s2: Symbol) -> Tuple:
    """Create a hashable key for a digram (pair of symbols)."""
    def sym_key(s):
        if s.is_terminal:
            return ('T', s.value)
        else:
            return ('NT', id(s.value))
    return (sym_key(s1), sym_key(s2))


class SequiturGrammar:
    """
    SEQUITUR Grammar Induction with O(n) time and space.

    Maintains two invariants:
    1. Digram Uniqueness: No digram appears more than once in entire grammar
    2. Rule Utility: Every rule is used at least twice

    Usage:
        grammar = SequiturGrammar()
        grammar.ingest([1, 2, 3, 1, 2, 3, 4, 5, 4, 5])

        # Grammar will contain rules for repeated patterns
        print(grammar.rules)

        # Encode new sequence using grammar
        encoded = grammar.encode([1, 2, 3, 1, 2, 3])

        # Decode back to original
        decoded = grammar.decode(encoded)
    """

    def __init__(self):
        """Initialize empty grammar with start rule S."""
        self._next_rule_id = 0
        self.start_rule = self._new_rule()  # Rule 0 is the start symbol S

        # Digram index: key -> Symbol (first symbol of the digram)
        # If digram appears, this points to its first occurrence
        self._digram_index: Dict[Tuple, Symbol] = {}

        # All rules in the grammar
        self.rules: Dict[int, Rule] = {self.start_rule.id: self.start_rule}

    def _new_rule(self) -> Rule:
        """Create a new rule with unique ID."""
        rule = Rule(id=self._next_rule_id)
        self._next_rule_id += 1
        return rule

    def _get_digram(self, sym: Symbol) -> Optional[Tuple]:
        """Get the digram key starting at sym (sym, sym.next), or None if at end."""
        if sym.rule is None or sym.next is None or sym.next is sym.rule.guard:
            return None
        return _digram_key(sym, sym.next)

    def _register_digram(self, sym: Symbol):
        """Register a digram in the index."""
        digram = self._get_digram(sym)
        if digram is not None:
            self._digram_index[digram] = sym

    def _unregister_digram(self, sym: Symbol):
        """Unregister a digram from the index."""
        digram = self._get_digram(sym)
        if digram is not None and digram in self._digram_index:
            if self._digram_index[digram] is sym:
                del self._digram_index[digram]

    def _check_digram(self, sym: Symbol):
        """
        Check if the digram starting at sym violates uniqueness.
        If so, either create a new rule or reuse an existing one.
        """
        # Safety check - sym must be in a rule
        if sym.rule is None:
            return

        digram = self._get_digram(sym)
        if digram is None:
            return

        # Check if this digram already exists elsewhere
        if digram in self._digram_index:
            existing = self._digram_index[digram]

            # Safety check - existing must still be valid
            if existing.rule is None:
                # Stale entry - remove and register new
                del self._digram_index[digram]
                self._digram_index[digram] = sym
                return

            # Don't match overlapping digrams (same symbol)
            if existing is sym or existing.next is sym:
                return

            # Don't match if existing is no longer valid
            if existing.next is None or existing.next.rule is None:
                del self._digram_index[digram]
                self._digram_index[digram] = sym
                return

            # Found duplicate digram - need to enforce uniqueness
            self._handle_duplicate_digram(sym, existing)
        else:
            # New digram - register it
            self._digram_index[digram] = sym

    def _handle_duplicate_digram(self, new_sym: Symbol, existing_sym: Symbol):
        """
        Handle a duplicate digram by creating or reusing a rule.

        Cases:
        1. Existing digram is entire RHS of a rule -> reuse that rule
        2. Otherwise -> create new rule for the digram
        """
        # Check if existing is a complete rule (rule has exactly 2 symbols)
        existing_rule = existing_sym.rule
        if (existing_rule is not self.start_rule and
            existing_rule.length == 2 and
            existing_sym is existing_rule.first):
            # Reuse existing rule
            self._substitute_digram(new_sym, existing_rule)
        else:
            # Create new rule for this digram
            self._create_rule_for_digram(new_sym, existing_sym)

    def _substitute_digram(self, sym: Symbol, rule: Rule):
        """
        Replace digram (sym, sym.next) with reference to rule.
        """
        # Safety check
        if sym.rule is None or sym.next is None:
            return

        parent_rule = sym.rule

        # Unregister affected digrams
        if sym.prev is not None and sym.prev is not parent_rule.guard:
            self._unregister_digram(sym.prev)
        self._unregister_digram(sym)
        if sym.next.next is not None and sym.next.next is not parent_rule.guard:
            self._unregister_digram(sym.next)

        # Remove second symbol of digram
        second = sym.next
        parent_rule.remove(second)

        # If second was a non-terminal, decrement its rule's usage
        if second.is_non_terminal:
            second.value.usage_count -= 1
            self._check_rule_utility(second.value)

        # Replace first symbol with rule reference
        sym.value = rule
        rule.usage_count += 1

        # Re-register digrams
        if sym.rule is not None:
            if sym.prev is not None and sym.prev is not sym.rule.guard:
                self._check_digram(sym.prev)
            self._check_digram(sym)

    def _create_rule_for_digram(self, new_sym: Symbol, existing_sym: Symbol):
        """
        Create a new rule for a repeated digram.

        Both occurrences get replaced with the new rule reference.
        """
        # Create new rule
        rule = self._new_rule()
        self.rules[rule.id] = rule

        # Copy digram symbols to new rule
        first_copy = new_sym.clone()
        second_copy = new_sym.next.clone()

        rule.append(first_copy)
        rule.append(second_copy)

        # Register the digram in the new rule
        self._digram_index[_digram_key(first_copy, second_copy)] = first_copy

        # Update usage counts for non-terminals in the new rule
        if first_copy.is_non_terminal:
            first_copy.value.usage_count += 1
        if second_copy.is_non_terminal:
            second_copy.value.usage_count += 1

        # Replace both occurrences with rule reference
        # Do existing first (order matters for digram handling)
        self._replace_digram_with_rule(existing_sym, rule)
        self._replace_digram_with_rule(new_sym, rule)

    def _replace_digram_with_rule(self, sym: Symbol, rule: Rule):
        """Replace digram starting at sym with a reference to rule."""
        # Safety check
        if sym.rule is None or sym.next is None:
            return

        # Get the containing rule
        parent_rule = sym.rule

        # Unregister affected digrams
        if sym.prev is not None and sym.prev is not parent_rule.guard:
            self._unregister_digram(sym.prev)
        self._unregister_digram(sym)
        if sym.next.next is not None and sym.next.next is not parent_rule.guard:
            self._unregister_digram(sym.next)

        # Remove second symbol
        second = sym.next
        parent_rule.remove(second)

        # Decrement usage if non-terminal
        if second.is_non_terminal:
            second.value.usage_count -= 1
            self._check_rule_utility(second.value)

        # Decrement usage for first if non-terminal
        if sym.is_non_terminal:
            sym.value.usage_count -= 1
            old_rule = sym.value
        else:
            old_rule = None

        # Replace first symbol with rule reference
        sym.value = rule
        rule.usage_count += 1

        # Check utility of old rule
        if old_rule is not None:
            self._check_rule_utility(old_rule)

        # Re-register digrams
        if sym.rule is not None:
            if sym.prev is not None and sym.prev is not sym.rule.guard:
                self._check_digram(sym.prev)
            if sym.next is not None and sym.next is not sym.rule.guard:
                self._check_digram(sym)

    def _check_rule_utility(self, rule: Rule):
        """
        Check if a rule satisfies the utility constraint (used more than once).
        If used only once, inline it and remove the rule.
        """
        if rule is self.start_rule:
            return

        if rule.usage_count <= 1:
            self._inline_rule(rule)

    def _inline_rule(self, rule: Rule):
        """
        Inline a rule that's only used once (or zero times).
        Replace the single reference with the rule's contents.
        """
        # Find where this rule is referenced
        reference = None
        for r in self.rules.values():
            for sym in r.symbols():
                if sym.is_non_terminal and sym.value is rule:
                    reference = sym
                    break
            if reference:
                break

        if reference is None:
            # Rule not referenced - just delete it
            del self.rules[rule.id]
            return

        # Unregister digrams around the reference
        parent_rule = reference.rule
        if parent_rule is None:
            # Reference is no longer in a rule - just delete the rule
            if rule.id in self.rules:
                del self.rules[rule.id]
            return

        if reference.prev is not None and reference.prev is not parent_rule.guard:
            self._unregister_digram(reference.prev)
        self._unregister_digram(reference)

        # Get rule contents
        symbols_to_insert = list(rule.symbols())

        # Unregister internal digrams of the rule being inlined
        for sym in symbols_to_insert[:-1]:
            self._unregister_digram(sym)

        if not symbols_to_insert:
            # Empty rule - just remove the reference
            parent_rule.remove(reference)
        else:
            # Replace reference with rule contents
            # First symbol replaces the reference
            first_sym = symbols_to_insert[0]
            reference.value = first_sym.value
            if first_sym.is_non_terminal:
                first_sym.value.usage_count += 1

            # Insert remaining symbols after
            current = reference
            for sym in symbols_to_insert[1:]:
                new_sym = sym.clone()
                parent_rule.insert_after(current, new_sym)
                if new_sym.is_non_terminal:
                    new_sym.value.usage_count += 1
                current = new_sym

        # Delete the inlined rule
        del self.rules[rule.id]

        # Check digrams at insertion points
        if reference.prev is not parent_rule.guard:
            self._check_digram(reference.prev)

        # Check all new digrams
        current = reference
        while current is not parent_rule.guard and current.next is not parent_rule.guard:
            self._check_digram(current)
            current = current.next

    def ingest(self, sequence: List[int]):
        """
        Process a sequence of terminals, building the grammar.

        This is the main entry point. Call with your tokenized sequence.
        O(n) time and space for sequence of length n.

        Args:
            sequence: List of integer tokens (terminals)
        """
        for token in sequence:
            # Create new terminal symbol
            sym = Symbol(value=token)

            # Append to start rule
            self.start_rule.append(sym)

            # Check digram uniqueness for the new digram (prev, sym)
            if sym.prev is not self.start_rule.guard:
                self._check_digram(sym.prev)

    def ingest_batch(self, sequences: List[List[int]], separator: int = -1):
        """
        Process multiple sequences, separated by a special token.

        Args:
            sequences: List of token sequences
            separator: Token to insert between sequences (default -1)
        """
        for i, seq in enumerate(sequences):
            self.ingest(seq)
            if i < len(sequences) - 1 and separator is not None:
                self.ingest([separator])

    def encode(self, sequence: List[int]) -> List[Union[int, Tuple[str, int]]]:
        """
        Encode a sequence using the learned grammar.

        Returns a list where each element is either:
        - An int (terminal that doesn't match any rule)
        - ('R', rule_id) for a rule application

        Note: This is a greedy left-to-right encoding, not necessarily optimal.

        Args:
            sequence: Token sequence to encode

        Returns:
            Encoded sequence using grammar rules
        """
        if not sequence:
            return []

        # Build rule lookup: expansion -> rule_id
        # Only include rules with length >= 2
        rule_expansions = {}
        for rule_id, rule in self.rules.items():
            if rule_id == 0:  # Skip start rule
                continue
            expansion = tuple(rule.expand())
            if len(expansion) >= 2:
                rule_expansions[expansion] = rule_id

        # Sort by length descending (prefer longer matches)
        sorted_rules = sorted(rule_expansions.items(), key=lambda x: len(x[0]), reverse=True)

        result = []
        i = 0
        while i < len(sequence):
            matched = False

            # Try to match rules (longest first)
            for expansion, rule_id in sorted_rules:
                exp_len = len(expansion)
                if i + exp_len <= len(sequence):
                    if tuple(sequence[i:i+exp_len]) == expansion:
                        result.append(('R', rule_id))
                        i += exp_len
                        matched = True
                        break

            if not matched:
                result.append(sequence[i])
                i += 1

        return result

    def decode(self, encoded: List[Union[int, Tuple[str, int]]]) -> List[int]:
        """
        Decode an encoded sequence back to terminals.

        Guarantees lossless reconstruction.

        Args:
            encoded: Encoded sequence from encode()

        Returns:
            Original terminal sequence
        """
        result = []
        for item in encoded:
            if isinstance(item, tuple) and item[0] == 'R':
                rule_id = item[1]
                if rule_id in self.rules:
                    result.extend(self.rules[rule_id].expand())
                else:
                    raise ValueError(f"Unknown rule: R{rule_id}")
            else:
                result.append(item)
        return result

    def get_vocabulary_size(self) -> int:
        """Return total number of rules (target: 300-400)."""
        return len(self.rules)

    def get_rule_stats(self) -> Dict[int, Dict]:
        """
        Get statistics for each rule.

        Returns:
            Dict mapping rule_id -> {
                'length': number of symbols in rule,
                'usage_count': times rule is referenced,
                'expansion_length': length when fully expanded,
                'depth': nesting depth (1 = only terminals)
            }
        """
        def get_depth(rule: Rule, cache: Dict[int, int]) -> int:
            if rule.id in cache:
                return cache[rule.id]

            max_child_depth = 0
            for sym in rule.symbols():
                if sym.is_non_terminal:
                    child_depth = get_depth(sym.value, cache)
                    max_child_depth = max(max_child_depth, child_depth)

            depth = max_child_depth + 1
            cache[rule.id] = depth
            return depth

        depth_cache = {}
        stats = {}

        for rule_id, rule in self.rules.items():
            expansion = rule.expand()
            stats[rule_id] = {
                'length': rule.length,
                'usage_count': rule.usage_count,
                'expansion_length': len(expansion),
                'depth': get_depth(rule, depth_cache)
            }

        return stats

    def get_grammar_representation(self) -> str:
        """Get human-readable grammar representation."""
        lines = []
        for rule_id in sorted(self.rules.keys()):
            rule = self.rules[rule_id]
            lines.append(repr(rule))
        return '\n'.join(lines)

    def __repr__(self):
        return f"SequiturGrammar(rules={len(self.rules)}, start={repr(self.start_rule)})"


# =============================================================================
# MUSICAL TOKEN SERIALIZATION
# =============================================================================

def serialize_factored_object(obj,
                              rhythm_vocab_size: int = 1000,
                              pitch_class_vocab_size: int = 12,
                              octave_vocab_size: int = 10,
                              duration_vocab_size: int = 100,
                              velocity_levels: int = 8) -> List[int]:
    """
    Serialize a FactoredObject into a token sequence for SEQUITUR.

    Token structure (offsets):
    - [0, rhythm_vocab_size): Rhythm pattern tokens
    - [rhythm_vocab_size, rhythm_vocab_size + 12*octave_vocab_size): Pitch tokens (pitch_class + octave*12)
    - [+duration_vocab_size): Duration tokens
    - [+velocity_levels): Velocity tokens

    Works with both old format (pitches attribute) and new format (pitch_class + octave).

    Args:
        obj: FactoredObject with rhythm, pitches/pitch_class+octave, durations, velocities
        rhythm_vocab_size: Max rhythm pattern ID
        pitch_class_vocab_size: 12 (always)
        octave_vocab_size: Number of octave levels
        duration_vocab_size: Max duration token
        velocity_levels: Number of velocity quantization levels

    Returns:
        List of integer tokens
    """
    tokens = []

    # Rhythm pattern (hash-based ID)
    rhythm_id = hash(obj.rhythm.tobytes()) % rhythm_vocab_size
    tokens.append(rhythm_id)

    # For each note: pitch_class, octave, duration, velocity
    pitch_offset = rhythm_vocab_size
    duration_offset = pitch_offset + pitch_class_vocab_size * octave_vocab_size
    velocity_offset = duration_offset + duration_vocab_size

    # Handle both new format (pitch_class, octave) and old format (pitches property)
    if hasattr(obj, 'pitch_class') and hasattr(obj, 'octave'):
        # New format
        for i in range(obj.num_notes):
            pc = int(obj.pitch_class[i])
            oct = min(int(obj.octave[i]), octave_vocab_size - 1)
            duration = min(int(obj.durations[i]), duration_vocab_size - 1)

            # Quantize velocity to levels
            # New format has velocities as 0-7, old format has 0.0-1.0
            vel = obj.velocities[i]
            if vel <= velocity_levels:  # Already quantized
                vel_level = min(int(vel), velocity_levels - 1)
            else:
                vel_level = min(int(float(vel) * velocity_levels / 127), velocity_levels - 1)

            # Add tokens (all Python int)
            tokens.append(int(pitch_offset + oct * 12 + pc))
            tokens.append(int(duration_offset + duration))
            tokens.append(int(velocity_offset + vel_level))
    else:
        # Old format - pitches is direct attribute
        pitches_array = obj.pitches if hasattr(obj, 'pitches') else getattr(obj, '_pitches', [])
        for i in range(obj.num_notes):
            pitch = int(pitches_array[i])  # Ensure Python int
            pitch_class = pitch % 12
            octave = min(pitch // 12, octave_vocab_size - 1)

            duration = min(int(obj.durations[i]), duration_vocab_size - 1)

            # Quantize velocity to levels (assume 0.0-1.0 range)
            vel = float(obj.velocities[i])
            vel_level = min(int(vel * velocity_levels), velocity_levels - 1)

            # Add tokens (all Python int)
            tokens.append(int(pitch_offset + octave * 12 + pitch_class))
            tokens.append(int(duration_offset + duration))
            tokens.append(int(velocity_offset + vel_level))

    return tokens


def serialize_track(objects: List, **kwargs) -> List[int]:
    """
    Serialize a list of FactoredObjects (a track) into tokens.

    Objects are serialized in order with a separator token.
    """
    tokens = []
    separator = -1  # Will be handled specially

    for i, obj in enumerate(objects):
        obj_tokens = serialize_factored_object(obj, **kwargs)
        tokens.extend(obj_tokens)
        if i < len(objects) - 1:
            tokens.append(separator)

    return tokens


def serialize_corpus(tracks: List[List], track_separator: int = -2, **kwargs) -> List[int]:
    """
    Serialize entire corpus (multiple tracks) into token sequence.

    Args:
        tracks: List of lists of FactoredObjects
        track_separator: Token to separate tracks
        **kwargs: Arguments passed to serialize_factored_object

    Returns:
        Single token sequence for the entire corpus
    """
    tokens = []

    for i, track in enumerate(tracks):
        track_tokens = serialize_track(track, **kwargs)
        tokens.extend(track_tokens)
        if i < len(tracks) - 1:
            tokens.append(track_separator)

    return tokens


# =============================================================================
# GPU-ACCELERATED TOKENIZATION
# =============================================================================

def tokenize_batch_gpu(objects: List, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated batch tokenization of FactoredObjects.

    Extracts pitch_class and octave in parallel on GPU.

    Args:
        objects: List of FactoredObjects
        device: PyTorch device

    Returns:
        (pitch_class_array, octave_array) - both shape (N, max_notes)
    """
    import torch

    # Find max notes across all objects
    max_notes = max(obj.num_notes for obj in objects) if objects else 0
    if max_notes == 0:
        return np.array([]), np.array([])

    N = len(objects)

    # Stack pitches with padding
    pitch_padded = np.zeros((N, max_notes), dtype=np.int32)
    mask = np.zeros((N, max_notes), dtype=bool)

    for i, obj in enumerate(objects):
        n = obj.num_notes
        if n > 0:
            pitch_padded[i, :n] = obj.pitches
            mask[i, :n] = True

    # Move to GPU
    pitch_tensor = torch.tensor(pitch_padded, device=device, dtype=torch.int32)

    # Compute pitch_class and octave in parallel
    pitch_class = (pitch_tensor % 12).cpu().numpy()
    octave = (pitch_tensor // 12).cpu().numpy()

    return pitch_class, octave, mask


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_grammar_from_corpus(objects: List,
                               verbose: bool = True,
                               **serialize_kwargs) -> SequiturGrammar:
    """
    Build a SEQUITUR grammar from a corpus of FactoredObjects.

    Args:
        objects: List of FactoredObjects
        verbose: Print progress
        **serialize_kwargs: Arguments for serialization

    Returns:
        SequiturGrammar with induced rules
    """
    if verbose:
        print(f"\n{'='*70}")
        print("SEQUITUR GRAMMAR INDUCTION")
        print(f"{'='*70}")
        print(f"  Objects: {len(objects)}")

    # Serialize corpus
    tokens = []
    for i, obj in enumerate(objects):
        obj_tokens = serialize_factored_object(obj, **serialize_kwargs)
        tokens.extend(obj_tokens)
        if i < len(objects) - 1:
            tokens.append(-1)  # Object separator

    if verbose:
        print(f"  Total tokens: {len(tokens):,}")

    # Build grammar
    grammar = SequiturGrammar()
    grammar.ingest(tokens)

    if verbose:
        print(f"  Rules discovered: {grammar.get_vocabulary_size()}")

        # Show rule statistics
        stats = grammar.get_rule_stats()
        depths = [s['depth'] for s in stats.values()]
        usages = [s['usage_count'] for s in stats.values()]

        print(f"  Max depth: {max(depths) if depths else 0}")
        print(f"  Avg usage: {np.mean(usages):.1f}" if usages else "  Avg usage: N/A")

        # Show top rules by usage
        top_rules = sorted(stats.items(), key=lambda x: x[1]['usage_count'], reverse=True)[:10]
        print(f"\n  Top 10 rules by usage:")
        for rule_id, rule_stats in top_rules:
            print(f"    R{rule_id}: usage={rule_stats['usage_count']}, "
                  f"length={rule_stats['length']}, depth={rule_stats['depth']}")

    return grammar
