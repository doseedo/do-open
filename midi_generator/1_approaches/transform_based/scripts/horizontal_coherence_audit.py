#!/usr/bin/env python3
"""
Horizontal Coherence Audit - Compare corpus vs generated MIDI.

Diagnoses why generated output has "signs of musicality but isn't coherent."
Vertical coordination (harmony) works. What's broken is horizontal structure.

Metrics:
1. Transition entropy - how predictable are note-to-note transitions?
2. Phrase repetition - do 4/8/16 beat patterns repeat?
3. Melodic contour - do melodies have shape vs random walk?
4. Rhythmic consistency - do rhythm patterns repeat?
5. Chord progression structure - are chord changes patterned?
"""

import orjson
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import mido
from mido import MidiFile
import math


class HorizontalCoherenceAudit:
    """Diagnose horizontal coherence by comparing corpus to generated."""

    def __init__(self, patterns: dict, verbose: bool = True):
        self.patterns = patterns
        self.verbose = verbose
        self.corpus_metrics = {}
        self.generated_metrics = {}

    def extract_pitch_sequences_from_corpus(self) -> Dict[int, List[List[int]]]:
        """Extract pitch sequences per GM from corpus occurrences.

        Returns: {gm: [[pitch_seq_from_piece1], [pitch_seq_from_piece2], ...]}
        """
        # Group patterns by piece and beat
        piece_gm_beats = defaultdict(lambda: defaultdict(list))

        for pid, p in self.patterns.items():
            gm = p.get('gm_program', 0)
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                pitch = occ.get('first_pitch', 60)
                beat = onset // 480
                piece_gm_beats[piece][gm].append((beat, pitch))

        # Convert to sequences
        result = defaultdict(list)
        for piece, gm_data in piece_gm_beats.items():
            for gm, beat_pitches in gm_data.items():
                if len(beat_pitches) >= 4:  # Need enough notes for analysis
                    sorted_bp = sorted(beat_pitches, key=lambda x: x[0])
                    pitch_seq = [p for _, p in sorted_bp]
                    result[gm].append(pitch_seq)

        return dict(result)

    def extract_pitch_sequences_from_midi(self, midi_path: str) -> Dict[int, List[int]]:
        """Extract pitch sequences per track from MIDI file.

        Returns: {gm: [pitch_sequence]}
        """
        mid = MidiFile(midi_path)
        result = {}

        for track in mid.tracks:
            current_gm = 0
            pitches = []
            current_time = 0
            events = []

            for msg in track:
                current_time += msg.time
                if msg.type == 'program_change':
                    current_gm = msg.program
                elif msg.type == 'note_on' and msg.velocity > 0:
                    events.append((current_time, msg.note))

            if events:
                events.sort(key=lambda x: x[0])
                pitches = [p for _, p in events]
                if pitches:
                    result[current_gm] = pitches

        return result

    def compute_bigram_entropy(self, sequences: List[List[int]]) -> float:
        """Compute average bigram entropy over sequences.

        Lower entropy = more predictable transitions.
        """
        # Count bigrams
        bigram_counts = Counter()
        unigram_counts = Counter()

        for seq in sequences:
            for i in range(len(seq) - 1):
                # Use intervals instead of absolute pitch for better generalization
                interval = seq[i+1] - seq[i]
                interval = max(-12, min(12, interval))  # Clip to octave
                bigram_counts[(seq[i] % 12, interval)] += 1
                unigram_counts[seq[i] % 12] += 1

        if not bigram_counts:
            return 0.0

        # Compute conditional entropy H(next|current)
        entropy = 0.0
        total = sum(unigram_counts.values())

        # Group by first element
        grouped = defaultdict(Counter)
        for (current, interval), count in bigram_counts.items():
            grouped[current][interval] += count

        for current, interval_counts in grouped.items():
            p_current = unigram_counts[current] / total
            interval_total = sum(interval_counts.values())

            for interval, count in interval_counts.items():
                p_interval_given_current = count / interval_total
                if p_interval_given_current > 0:
                    entropy -= p_current * p_interval_given_current * math.log2(p_interval_given_current)

        return entropy

    def compute_phrase_repetition(self, sequences: List[List[int]], chunk_size: int = 4) -> float:
        """Compute phrase repetition rate.

        How often do 4-beat chunks repeat within a piece?
        """
        total_chunks = 0
        repeated_chunks = 0

        for seq in sequences:
            # Convert to interval sequence (more invariant to transposition)
            intervals = []
            for i in range(len(seq) - 1):
                intervals.append(seq[i+1] - seq[i])

            # Extract chunks
            chunks = []
            for i in range(0, len(intervals) - chunk_size + 1, chunk_size):
                chunk = tuple(intervals[i:i+chunk_size])
                chunks.append(chunk)

            if not chunks:
                continue

            # Count repetitions
            chunk_counts = Counter(chunks)
            for chunk, count in chunk_counts.items():
                total_chunks += count
                if count > 1:
                    repeated_chunks += count

        if total_chunks == 0:
            return 0.0

        return repeated_chunks / total_chunks

    def compute_melodic_autocorr(self, sequences: List[List[int]], lag: int = 8) -> float:
        """Compute average autocorrelation at given lag.

        High autocorrelation = melodic shape persists over time.
        """
        correlations = []

        for seq in sequences:
            if len(seq) <= lag:
                continue

            # Normalize to intervals
            intervals = np.diff(seq)
            if len(intervals) <= lag:
                continue

            # Compute autocorrelation
            x = intervals - np.mean(intervals)
            if np.std(x) == 0:
                continue

            x = x / np.std(x)
            n = len(x)

            autocorr = np.correlate(x, x, mode='full')[n-1:]
            autocorr = autocorr / n

            if lag < len(autocorr):
                correlations.append(autocorr[lag])

        if not correlations:
            return 0.0

        return np.mean(correlations)

    def compute_rhythm_repetition(self, midi_path: str) -> float:
        """Compute IOI pattern repetition from MIDI."""
        mid = MidiFile(midi_path)

        all_iois = []
        for track in mid.tracks:
            current_time = 0
            note_times = []

            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_times.append(current_time)

            if len(note_times) >= 2:
                iois = np.diff(note_times)
                # Quantize to 16th notes (120 ticks at 480 ppq)
                iois = [int(round(ioi / 120)) * 120 for ioi in iois]
                all_iois.extend(iois)

        if len(all_iois) < 8:
            return 0.0

        # Extract 4-IOI chunks
        chunks = []
        for i in range(0, len(all_iois) - 3, 4):
            chunk = tuple(all_iois[i:i+4])
            chunks.append(chunk)

        if not chunks:
            return 0.0

        chunk_counts = Counter(chunks)
        total = len(chunks)
        repeated = sum(c for c in chunk_counts.values() if c > 1)

        return repeated / total

    def compute_corpus_rhythm_repetition(self) -> float:
        """Compute rhythm pattern repetition from corpus patterns."""
        # Use rhythm_ratios from patterns
        all_chunks = []

        for pid, p in self.patterns.items():
            ratios = p.get('rhythm_ratios', [])
            if len(ratios) >= 4:
                # Quantize ratios to 0.25 increments
                quantized = tuple(round(r * 4) / 4 for r in ratios[:4])
                all_chunks.append(quantized)

        if not all_chunks:
            return 0.0

        chunk_counts = Counter(all_chunks)
        total = len(all_chunks)
        repeated = sum(c for c in chunk_counts.values() if c > 1)

        return repeated / total

    def compute_chord_transition_coverage(
        self,
        corpus_pc_sets: List[Tuple[int, ...]],
        generated_pc_sets: List[Tuple[int, ...]]
    ) -> Tuple[float, float]:
        """Check how much of generated chord progressions match corpus.

        Returns: (corpus_top10_coverage, generated_match_rate)
        """
        # Compute transitions
        def get_transitions(pc_sets):
            transitions = []
            for i in range(len(pc_sets) - 1):
                transitions.append((pc_sets[i], pc_sets[i+1]))
            return transitions

        corpus_trans = get_transitions(corpus_pc_sets)
        gen_trans = get_transitions(generated_pc_sets)

        corpus_counts = Counter(corpus_trans)

        # Get top 10 corpus transitions
        top10 = set(t for t, _ in corpus_counts.most_common(10))

        if not gen_trans:
            return 0.0, 0.0

        # How many generated transitions are in top 10?
        matches = sum(1 for t in gen_trans if t in top10)

        return len(top10), matches / len(gen_trans)

    def analyze_chord_transitions_detailed(
        self,
        corpus_pc_sets: List[Tuple[int, ...]],
        generated_pc_sets: List[Tuple[int, ...]]
    ) -> dict:
        """Detailed analysis of chord transitions - what's different?"""
        pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        def pc_set_name(pc_tuple):
            if not pc_tuple:
                return "{}"
            return "{" + ",".join(pc_names[p] for p in pc_tuple) + "}"

        def get_transitions(pc_sets):
            transitions = []
            for i in range(len(pc_sets) - 1):
                transitions.append((pc_sets[i], pc_sets[i+1]))
            return transitions

        corpus_trans = get_transitions(corpus_pc_sets)
        gen_trans = get_transitions(generated_pc_sets)

        corpus_counts = Counter(corpus_trans)
        gen_counts = Counter(gen_trans)

        # Top 10 corpus transitions
        print("\n--- TOP 10 CORPUS CHORD TRANSITIONS ---")
        corpus_top10 = corpus_counts.most_common(10)
        for (from_pc, to_pc), count in corpus_top10:
            pct = 100 * count / len(corpus_trans) if corpus_trans else 0
            print(f"  {pc_set_name(from_pc):20s} → {pc_set_name(to_pc):20s}  {count:5d} ({pct:.2f}%)")

        # Top 10 generated transitions
        print("\n--- TOP 10 GENERATED CHORD TRANSITIONS ---")
        gen_top10 = gen_counts.most_common(10)
        for (from_pc, to_pc), count in gen_top10:
            pct = 100 * count / len(gen_trans) if gen_trans else 0
            in_corpus = "✓" if (from_pc, to_pc) in corpus_counts else "✗"
            corpus_rank = "---"
            if (from_pc, to_pc) in corpus_counts:
                # Find rank
                for i, (t, _) in enumerate(corpus_counts.most_common()):
                    if t == (from_pc, to_pc):
                        corpus_rank = f"#{i+1}"
                        break
            print(f"  {pc_set_name(from_pc):20s} → {pc_set_name(to_pc):20s}  {count:5d} ({pct:.2f}%) {in_corpus} corpus:{corpus_rank}")

        # Overlap analysis
        corpus_top10_set = set(t for t, _ in corpus_top10)
        gen_top10_set = set(t for t, _ in gen_top10)
        overlap = corpus_top10_set & gen_top10_set

        print(f"\n--- OVERLAP ---")
        print(f"  Corpus unique PC sets:    {len(set(corpus_pc_sets))}")
        print(f"  Generated unique PC sets: {len(set(generated_pc_sets))}")
        print(f"  Top 10 overlap:           {len(overlap)} / 10")

        # What's in generated that's rare in corpus?
        print("\n--- GENERATED TRANSITIONS RARE IN CORPUS ---")
        rare_count = 0
        for (from_pc, to_pc), count in gen_top10:
            corpus_count = corpus_counts.get((from_pc, to_pc), 0)
            if corpus_count < 10:  # Rare or absent
                print(f"  {pc_set_name(from_pc):20s} → {pc_set_name(to_pc):20s}  gen:{count} corpus:{corpus_count}")
                rare_count += 1

        # Coverage: what fraction of generated transitions exist in corpus at all?
        gen_in_corpus = sum(1 for t in gen_trans if t in corpus_counts)
        coverage = gen_in_corpus / len(gen_trans) if gen_trans else 0

        print(f"\n--- OVERALL COVERAGE ---")
        print(f"  Generated transitions in corpus: {gen_in_corpus} / {len(gen_trans)} ({coverage*100:.1f}%)")

        return {
            'corpus_top10': corpus_top10,
            'gen_top10': gen_top10,
            'overlap': len(overlap),
            'coverage': coverage,
            'rare_in_corpus': rare_count
        }

    def extract_pc_sets_from_corpus(self) -> List[Tuple[int, ...]]:
        """Extract PC sets per beat from corpus."""
        beat_pcs = defaultdict(set)

        for pid, p in self.patterns.items():
            for occ in p.get('occurrences', []):
                piece = occ.get('piece_id', 'unknown')
                onset = occ.get('onset_time', 0)
                pitch = occ.get('first_pitch', 60)
                beat = onset // 480
                beat_pcs[(piece, beat)].add(pitch % 12)

        # Sort by piece and beat
        sorted_keys = sorted(beat_pcs.keys())
        return [tuple(sorted(beat_pcs[k])) for k in sorted_keys]

    def extract_pc_sets_from_midi(self, midi_path: str) -> List[Tuple[int, ...]]:
        """Extract PC sets per beat from MIDI."""
        mid = MidiFile(midi_path)

        beat_pcs = defaultdict(set)

        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    beat = current_time // 480
                    beat_pcs[beat].add(msg.note % 12)

        sorted_beats = sorted(beat_pcs.keys())
        return [tuple(sorted(beat_pcs[b])) for b in sorted_beats]

    def run_audit(self, generated_midi_path: str) -> dict:
        """Run full horizontal coherence audit."""
        print("=" * 60)
        print("HORIZONTAL COHERENCE AUDIT")
        print("=" * 60)

        # Extract corpus data
        print("\nExtracting corpus data...")
        corpus_seqs = self.extract_pitch_sequences_from_corpus()
        corpus_pc_sets = self.extract_pc_sets_from_corpus()

        total_seqs = sum(len(s) for s in corpus_seqs.values())
        print(f"  Corpus sequences: {total_seqs} across {len(corpus_seqs)} instruments")
        print(f"  Corpus PC sets: {len(corpus_pc_sets)} beats")

        # Extract generated data
        print(f"\nExtracting generated data from: {generated_midi_path}")
        try:
            gen_seqs = self.extract_pitch_sequences_from_midi(generated_midi_path)
            gen_pc_sets = self.extract_pc_sets_from_midi(generated_midi_path)
            print(f"  Generated tracks: {len(gen_seqs)}")
            print(f"  Generated PC sets: {len(gen_pc_sets)} beats")
        except Exception as e:
            print(f"  ERROR loading MIDI: {e}")
            return {"error": str(e)}

        # Flatten corpus sequences for comparison
        corpus_all_seqs = []
        for gm, seqs in corpus_seqs.items():
            corpus_all_seqs.extend(seqs)

        gen_all_seqs = [[p for p in seq] for seq in gen_seqs.values()]

        results = {}

        # 1. Pitch bigram entropy
        print("\n--- DIAGNOSTIC 1: Transition Entropy ---")
        corpus_entropy = self.compute_bigram_entropy(corpus_all_seqs)
        gen_entropy = self.compute_bigram_entropy(gen_all_seqs)
        gap = ((gen_entropy - corpus_entropy) / corpus_entropy * 100) if corpus_entropy > 0 else 0

        print(f"  Corpus:    {corpus_entropy:.2f} bits")
        print(f"  Generated: {gen_entropy:.2f} bits")
        print(f"  Gap:       {gap:+.0f}%")

        results['entropy'] = {
            'corpus': corpus_entropy,
            'generated': gen_entropy,
            'gap_pct': gap
        }

        # 2. Phrase repetition (4-beat)
        print("\n--- DIAGNOSTIC 2: Phrase Repetition (4-beat chunks) ---")
        corpus_rep = self.compute_phrase_repetition(corpus_all_seqs, 4)
        gen_rep = self.compute_phrase_repetition(gen_all_seqs, 4)
        gap = ((gen_rep - corpus_rep) / corpus_rep * 100) if corpus_rep > 0 else 0

        print(f"  Corpus:    {corpus_rep*100:.1f}%")
        print(f"  Generated: {gen_rep*100:.1f}%")
        print(f"  Gap:       {gap:+.0f}%")

        results['phrase_repetition'] = {
            'corpus': corpus_rep,
            'generated': gen_rep,
            'gap_pct': gap
        }

        # 3. Melodic autocorrelation
        print("\n--- DIAGNOSTIC 3: Melodic Autocorrelation (lag=8) ---")
        corpus_autocorr = self.compute_melodic_autocorr(corpus_all_seqs, 8)
        gen_autocorr = self.compute_melodic_autocorr(gen_all_seqs, 8)
        gap = ((gen_autocorr - corpus_autocorr) / abs(corpus_autocorr) * 100) if corpus_autocorr != 0 else 0

        print(f"  Corpus:    {corpus_autocorr:.3f}")
        print(f"  Generated: {gen_autocorr:.3f}")
        print(f"  Gap:       {gap:+.0f}%")

        results['melodic_autocorr'] = {
            'corpus': corpus_autocorr,
            'generated': gen_autocorr,
            'gap_pct': gap
        }

        # 4. Rhythm repetition
        print("\n--- DIAGNOSTIC 4: Rhythm Pattern Repetition ---")
        corpus_rhythm = self.compute_corpus_rhythm_repetition()
        gen_rhythm = self.compute_rhythm_repetition(generated_midi_path)
        gap = ((gen_rhythm - corpus_rhythm) / corpus_rhythm * 100) if corpus_rhythm > 0 else 0

        print(f"  Corpus:    {corpus_rhythm*100:.1f}%")
        print(f"  Generated: {gen_rhythm*100:.1f}%")
        print(f"  Gap:       {gap:+.0f}%")

        results['rhythm_repetition'] = {
            'corpus': corpus_rhythm,
            'generated': gen_rhythm,
            'gap_pct': gap
        }

        # 5. Chord progression coverage
        print("\n--- DIAGNOSTIC 5: Chord Transition Coverage ---")
        top10_count, gen_match = self.compute_chord_transition_coverage(corpus_pc_sets, gen_pc_sets)

        print(f"  Top corpus transitions: {top10_count}")
        print(f"  Generated match rate:   {gen_match*100:.1f}%")

        results['chord_coverage'] = {
            'top_transitions': top10_count,
            'generated_match': gen_match
        }

        # 6. Detailed chord transition analysis
        chord_details = self.analyze_chord_transitions_detailed(corpus_pc_sets, gen_pc_sets)
        results['chord_details'] = chord_details

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<25} {'Corpus':>10} {'Generated':>10} {'Gap':>10}")
        print("-" * 60)
        print(f"{'Pitch bigram entropy':<25} {corpus_entropy:>10.2f} {gen_entropy:>10.2f} {results['entropy']['gap_pct']:>+9.0f}%")
        print(f"{'4-beat repetition':<25} {corpus_rep*100:>9.1f}% {gen_rep*100:>9.1f}% {results['phrase_repetition']['gap_pct']:>+9.0f}%")
        print(f"{'Melodic autocorr(8)':<25} {corpus_autocorr:>10.3f} {gen_autocorr:>10.3f} {results['melodic_autocorr']['gap_pct']:>+9.0f}%")
        print(f"{'Rhythm repetition':<25} {corpus_rhythm*100:>9.1f}% {gen_rhythm*100:>9.1f}% {results['rhythm_repetition']['gap_pct']:>+9.0f}%")
        print(f"{'Chord coverage':<25} {'':>10} {gen_match*100:>9.1f}%")

        # Diagnosis
        print("\n" + "=" * 60)
        print("DIAGNOSIS")
        print("=" * 60)

        issues = []

        if results['entropy']['gap_pct'] > 30:
            issues.append("HIGH ENTROPY GAP: Transitions are too random. Need longer Markov context (2nd/3rd order) or PPM*.")

        if results['phrase_repetition']['gap_pct'] < -50:
            issues.append("LOW PHRASE REPETITION: No musical structure. Need phrase-level patterns or templates.")

        if results['melodic_autocorr']['gap_pct'] < -50:
            issues.append("NO MELODIC CONTOUR: Melodies lack shape. Need contour constraints or melodic pattern memory.")

        if results['rhythm_repetition']['gap_pct'] < -50:
            issues.append("RANDOM RHYTHM: No groove. Need rhythmic pattern persistence across beats.")

        if gen_match < 0.1:
            issues.append("CHORD CHANGES DON'T MATCH CORPUS: Harmonic progressions are atypical.")

        if not issues:
            issues.append("No major horizontal issues detected. Check for other problems.")

        for issue in issues:
            print(f"  → {issue}")

        results['diagnosis'] = issues

        return results


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python horizontal_coherence_audit.py <checkpoint.npz> <generated.mid>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    generated_path = sys.argv[2]

    # Load patterns
    print(f"Loading checkpoint: {checkpoint_path}")
    data = np.load(checkpoint_path, allow_pickle=True)
    patterns_file = str(data['patterns_json_file'][0])

    import os
    base_dir = os.path.dirname(checkpoint_path)
    json_path = os.path.join(base_dir, patterns_file) if base_dir else patterns_file

    print(f"Loading patterns from: {json_path}")
    with open(json_path, 'rb') as f:
        patterns = orjson.loads(f.read())

    print(f"Loaded {len(patterns)} patterns")

    # Run audit
    auditor = HorizontalCoherenceAudit(patterns)
    results = auditor.run_audit(generated_path)

    return results


if __name__ == '__main__':
    main()
