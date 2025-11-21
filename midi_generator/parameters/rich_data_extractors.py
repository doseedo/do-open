"""
Rich Data Extensions Extractors - Agent 3
==========================================

Extracts 130 rich data extension parameters:
- Per-track parameters (80): 8 tracks × 10 params each
- Temporal evolution (40): 4 sections × 10 params
- Genre-specific detail (10 per genre)

These provide fine-grained detail beyond the hierarchical and modular semantic parameters.

Author: Agent 3 - Comprehensive Parameter Extraction Specialist
Date: November 21, 2025
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
from scipy.stats import entropy


# Import shared dataclass
from midi_generator.parameters.modular_semantic_extractors import MIDIAnalysisData


class PerTrackParameterExtractor:
    """Extract per-track parameters (80 total: 8 tracks × 10 params)"""

    def extract(self, analysis: MIDIAnalysisData) -> List[Dict[str, float]]:
        """Extract parameters for up to 8 tracks"""
        # Organize notes by track
        track_data = self._organize_by_track(analysis)

        # Extract parameters for each track (up to 8)
        track_params = []
        for i in range(8):
            if i < len(track_data):
                params = self._extract_track_params(track_data[i], analysis)
            else:
                # Padding for missing tracks
                params = self._get_empty_track_params()
            track_params.append(params)

        return track_params

    def _organize_by_track(self, analysis: MIDIAnalysisData) -> List[Dict]:
        """Organize notes by track/instrument"""
        tracks = []

        # If we have instrument-based tracks
        if analysis.tracks_by_instrument:
            for program, notes in sorted(analysis.tracks_by_instrument.items())[:8]:
                tracks.append({
                    'program': program,
                    'notes': notes,
                    'pitches': [n['pitch'] for n in notes],
                    'velocities': [n['velocity'] for n in notes],
                    'durations': [n['duration'] for n in notes],
                    'onsets': [n['onset'] for n in notes]
                })
        else:
            # Fallback: treat all notes as one track
            if analysis.all_notes:
                tracks.append({
                    'program': 0,
                    'notes': analysis.all_notes,
                    'pitches': analysis.all_pitches,
                    'velocities': analysis.all_velocities,
                    'durations': analysis.note_durations,
                    'onsets': analysis.note_onsets
                })

        return tracks

    def _extract_track_params(self, track: Dict, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Extract 10 parameters for a single track"""
        params = {}

        # 1. Track role (0=bass, 0.5=harmony, 1=melody)
        params['role'] = self._identify_role(track, analysis)

        # 2. Note density (notes per second)
        if analysis.duration_seconds > 0:
            params['density'] = float(len(track['notes']) / analysis.duration_seconds)
        else:
            params['density'] = 0.0

        # 3. Average register (normalized pitch)
        if track['pitches']:
            params['register'] = float(np.mean(track['pitches']) / 127.0)
        else:
            params['register'] = 0.5

        # 4. Register range
        if track['pitches']:
            pitch_range = max(track['pitches']) - min(track['pitches'])
            params['range'] = float(min(pitch_range / 48.0, 1.0))
        else:
            params['range'] = 0.0

        # 5. Rhythmic activity (note density variance)
        params['rhythmic_activity'] = self._compute_rhythmic_activity(track)

        # 6. Melodic contour smoothness
        params['contour_smoothness'] = self._compute_contour_smoothness(track)

        # 7. Articulation (legato ratio)
        params['articulation'] = self._compute_articulation(track)

        # 8. Dynamic level (average velocity)
        if track['velocities']:
            params['dynamic_level'] = float(np.mean(track['velocities']) / 127.0)
        else:
            params['dynamic_level'] = 0.5

        # 9. Dynamic range (velocity variance)
        if track['velocities']:
            params['dynamic_range'] = float(np.std(track['velocities']) / 127.0)
        else:
            params['dynamic_range'] = 0.0

        # 10. Importance (relative activity vs other tracks)
        params['importance'] = self._compute_importance(track, analysis)

        return params

    def _identify_role(self, track: Dict, analysis: MIDIAnalysisData) -> float:
        """Identify track role: 0=bass, 0.5=harmony, 1=melody"""
        if not track['pitches']:
            return 0.5

        avg_pitch = np.mean(track['pitches'])

        # Check if bass (low register)
        if avg_pitch < 48:  # Below middle C
            return 0.0

        # Check if melody (highest notes, more activity)
        if track['pitches'] == analysis.melody_pitches:
            return 1.0

        # Check register and activity
        if avg_pitch > 60:  # Above middle C
            return 0.8  # Likely melody
        else:
            return 0.5  # Likely harmony/accompaniment

    def _compute_rhythmic_activity(self, track: Dict) -> float:
        """Rhythmic activity level"""
        if not track['durations'] or len(track['durations']) < 2:
            return 0.0
        # Entropy of durations
        dur_buckets = [round(d * 4) / 4 for d in track['durations']]
        dur_counts = Counter(dur_buckets)
        probs = np.array(list(dur_counts.values())) / len(track['durations'])
        dur_entropy = entropy(probs)
        max_entropy = np.log(min(len(dur_counts), 8))
        return float(dur_entropy / max_entropy) if max_entropy > 0 else 0.0

    def _compute_contour_smoothness(self, track: Dict) -> float:
        """Melodic contour smoothness"""
        if not track['pitches'] or len(track['pitches']) < 2:
            return 0.5
        intervals = [abs(track['pitches'][i+1] - track['pitches'][i])
                    for i in range(len(track['pitches'])-1)]
        if not intervals:
            return 0.5
        stepwise_ratio = sum(1 for i in intervals if i <= 2) / len(intervals)
        return float(stepwise_ratio)

    def _compute_articulation(self, track: Dict) -> float:
        """Articulation (legato ratio)"""
        if not track['notes'] or len(track['notes']) < 2:
            return 0.5
        sorted_notes = sorted(track['notes'], key=lambda n: n['onset'])
        legato_count = 0
        for i in range(len(sorted_notes) - 1):
            duration = sorted_notes[i]['duration']
            ioi = sorted_notes[i+1]['onset'] - sorted_notes[i]['onset']
            if ioi > 0 and duration / ioi > 0.9:  # 90% legato threshold
                legato_count += 1
        return float(legato_count / (len(sorted_notes) - 1))

    def _compute_importance(self, track: Dict, analysis: MIDIAnalysisData) -> float:
        """Track importance relative to overall piece"""
        if not analysis.all_notes:
            return 0.5
        track_note_count = len(track['notes'])
        total_note_count = len(analysis.all_notes)
        return float(min(track_note_count / (total_note_count + 1), 1.0))

    def _get_empty_track_params(self) -> Dict[str, float]:
        """Empty track parameters (for padding)"""
        return {
            'role': 0.5,
            'density': 0.0,
            'register': 0.5,
            'range': 0.0,
            'rhythmic_activity': 0.0,
            'contour_smoothness': 0.5,
            'articulation': 0.5,
            'dynamic_level': 0.5,
            'dynamic_range': 0.0,
            'importance': 0.0
        }


class TemporalEvolutionExtractor:
    """Extract temporal evolution parameters (40 total: 4 sections × 10 params)"""

    def extract(self, analysis: MIDIAnalysisData) -> List[Dict[str, float]]:
        """Extract parameters for 4 temporal sections"""
        # Divide piece into 4 sections
        sections = self._divide_into_sections(analysis, num_sections=4)

        # Extract parameters for each section
        section_params = []
        for section in sections:
            params = self._extract_section_params(section, analysis)
            section_params.append(params)

        return section_params

    def _divide_into_sections(self, analysis: MIDIAnalysisData,
                               num_sections: int = 4) -> List[Dict]:
        """Divide piece into temporal sections"""
        section_duration = analysis.duration_seconds / num_sections
        sections = []

        for i in range(num_sections):
            start_time = i * section_duration
            end_time = (i + 1) * section_duration

            section_notes = [n for n in analysis.all_notes
                            if start_time <= n['onset'] < end_time]

            sections.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': section_duration,
                'notes': section_notes,
                'pitches': [n['pitch'] for n in section_notes],
                'velocities': [n['velocity'] for n in section_notes],
                'durations': [n['duration'] for n in section_notes],
                'onsets': [n['onset'] for n in section_notes]
            })

        return sections

    def _extract_section_params(self, section: Dict,
                                analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Extract 10 parameters for a temporal section"""
        params = {}

        # 1. Energy level
        params['energy'] = self._compute_energy(section)

        # 2. Density
        if section['duration'] > 0:
            params['density'] = float(len(section['notes']) / section['duration'])
        else:
            params['density'] = 0.0

        # 3. Complexity (harmonic + rhythmic)
        params['complexity'] = self._compute_complexity(section)

        # 4. Tension
        params['tension'] = self._compute_tension(section)

        # 5. Dynamic level
        if section['velocities']:
            params['dynamics'] = float(np.mean(section['velocities']) / 127.0)
        else:
            params['dynamics'] = 0.5

        # 6. Register (average pitch)
        if section['pitches']:
            params['register'] = float(np.mean(section['pitches']) / 127.0)
        else:
            params['register'] = 0.5

        # 7. Polyphony
        params['polyphony'] = self._compute_polyphony(section, analysis)

        # 8. Rhythmic intensity
        params['rhythmic_intensity'] = self._compute_rhythmic_intensity(section)

        # 9. Harmonic stability
        params['harmonic_stability'] = self._compute_harmonic_stability(section)

        # 10. Textural density
        params['textural_density'] = self._compute_textural_density(section)

        return params

    def _compute_energy(self, section: Dict) -> float:
        """Energy level (velocity × density)"""
        if not section['velocities'] or section['duration'] == 0:
            return 0.0
        avg_velocity = np.mean(section['velocities']) / 127.0
        density = len(section['notes']) / section['duration']
        energy = avg_velocity * min(density / 5.0, 1.0)
        return float(energy)

    def _compute_complexity(self, section: Dict) -> float:
        """Musical complexity"""
        if not section['pitches'] or not section['durations']:
            return 0.0

        # Pitch complexity (unique pitches)
        pitch_complexity = len(set(section['pitches'])) / 12.0

        # Rhythmic complexity (duration variety)
        if len(section['durations']) > 1:
            dur_buckets = [round(d * 4) / 4 for d in section['durations']]
            rhythm_complexity = len(set(dur_buckets)) / 8.0
        else:
            rhythm_complexity = 0.0

        return float((pitch_complexity + rhythm_complexity) / 2.0)

    def _compute_tension(self, section: Dict) -> float:
        """Harmonic/melodic tension"""
        if not section['pitches'] or len(section['pitches']) < 2:
            return 0.5

        # Dissonance measure
        pitch_classes = [p % 12 for p in section['pitches']]
        pc_counts = [pitch_classes.count(i) for i in range(12)]
        total = sum(pc_counts)
        if total == 0:
            return 0.5

        # Entropy-based tension
        probs = [c / total for c in pc_counts if c > 0]
        pc_entropy = entropy(probs)
        max_entropy = np.log(12)
        return float(pc_entropy / max_entropy)

    def _compute_polyphony(self, section: Dict, analysis: MIDIAnalysisData) -> float:
        """Average polyphony in section"""
        if not section['notes'] or section['duration'] == 0:
            return 0.0

        time_points = np.linspace(section['start_time'], section['end_time'], 10)
        poly_values = []
        for t in time_points:
            active = sum(1 for n in section['notes']
                        if n['onset'] <= t <= n['onset'] + n['duration'])
            poly_values.append(active)

        return float(min(np.mean(poly_values) / 10.0, 1.0))

    def _compute_rhythmic_intensity(self, section: Dict) -> float:
        """Rhythmic intensity"""
        if section['duration'] == 0:
            return 0.0
        note_density = len(section['notes']) / section['duration']
        return float(min(note_density / 10.0, 1.0))

    def _compute_harmonic_stability(self, section: Dict) -> float:
        """Harmonic stability (inverse of tension)"""
        return 1.0 - self._compute_tension(section)

    def _compute_textural_density(self, section: Dict) -> float:
        """Textural density"""
        if not section['notes'] or section['duration'] == 0:
            return 0.0
        # Notes per second
        density = len(section['notes']) / section['duration']
        return float(min(density / 10.0, 1.0))


class GenreSpecificExtractor:
    """Extract genre-specific parameters (10 per genre)"""

    def extract(self, analysis: MIDIAnalysisData, genre: str) -> Dict[str, float]:
        """Extract genre-specific parameters based on detected genre"""
        genre = genre.lower()

        if genre == 'jazz':
            return self._extract_jazz(analysis)
        elif genre == 'classical':
            return self._extract_classical(analysis)
        elif genre == 'rock':
            return self._extract_rock(analysis)
        elif genre == 'electronic':
            return self._extract_electronic(analysis)
        elif genre == 'pop':
            return self._extract_pop(analysis)
        elif genre == 'hiphop':
            return self._extract_hiphop(analysis)
        elif genre == 'latin':
            return self._extract_latin(analysis)
        elif genre == 'world':
            return self._extract_world(analysis)
        else:
            return self._extract_generic(analysis)

    def _extract_jazz(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Jazz-specific parameters"""
        return {
            'swing_ratio': self._compute_swing(analysis),
            'walking_bass_density': self._compute_walking_bass(analysis),
            'comping_pattern': self._compute_comping(analysis),
            'bebop_articulation': self._compute_bebop_articulation(analysis),
            'improvisation_markers': 0.3,  # Placeholder
            'blue_note_usage': self._compute_blue_notes(analysis),
            'ride_cymbal_pattern': 0.5,  # Placeholder
            'soli_section_ratio': 0.2,  # Placeholder
            'turnaround_frequency': 0.3,  # Placeholder
            'substitution_ratio': 0.2  # Placeholder
        }

    def _extract_classical(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Classical-specific parameters"""
        return {
            'counterpoint_complexity': self._compute_counterpoint(analysis),
            'voice_leading_quality': 0.7,  # Placeholder
            'development_density': 0.5,  # Placeholder
            'thematic_transformation': 0.4,  # Placeholder
            'orchestral_balance': 0.6,  # Placeholder
            'dynamic_contrast': self._compute_dynamic_contrast(analysis),
            'phrase_structure': 0.5,  # Placeholder
            'cadence_strength': 0.6,  # Placeholder
            'modulation_complexity': 0.4,  # Placeholder
            'textural_variety': 0.5  # Placeholder
        }

    def _extract_rock(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Rock-specific parameters"""
        return {
            'power_chord_ratio': self._compute_power_chords(analysis),
            'distortion_markers': 0.7,  # Placeholder
            'riff_repetition': 0.8,  # Placeholder
            'backbeat_strength': self._compute_backbeat(analysis),
            'guitar_solo_presence': 0.3,  # Placeholder
            'verse_chorus_contrast': 0.6,  # Placeholder
            'groove_consistency': self._compute_groove(analysis),
            'palm_mute_ratio': 0.4,  # Placeholder
            'bend_usage': 0.3,  # Placeholder
            'power_stance_energy': 0.7  # Placeholder
        }

    def _extract_electronic(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Electronic-specific parameters"""
        return {
            'quantization_level': self._compute_quantization(analysis),
            'filter_sweep_markers': 0.5,  # Placeholder
            'arpeggio_density': self._compute_arpeggios(analysis),
            'sidechain_pumping': 0.6,  # Placeholder
            'build_drop_structure': 0.7,  # Placeholder
            'layering_complexity': 0.8,  # Placeholder
            'automation_intensity': 0.5,  # Placeholder
            'sub_bass_presence': 0.6,  # Placeholder
            'white_noise_usage': 0.4,  # Placeholder
            'glitch_elements': 0.3  # Placeholder
        }

    def _extract_pop(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Pop-specific parameters"""
        return {
            'hook_strength': 0.7,  # Placeholder
            'verse_chorus_ratio': 0.6,  # Placeholder
            'production_polish': self._compute_quantization(analysis),
            'melodic_catchiness': 0.8,  # Placeholder
            'rhythmic_simplicity': 0.7,  # Placeholder
            'dynamic_compression': 0.8,  # Placeholder
            'layering_density': 0.6,  # Placeholder
            'vocal_prominence': 0.7,  # Placeholder
            'bridge_contrast': 0.5,  # Placeholder
            'earworm_factor': 0.8  # Placeholder
        }

    def _extract_hiphop(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Hip-hop-specific parameters"""
        return {
            'boom_bap_feel': 0.7,  # Placeholder
            'sample_loop_ratio': 0.8,  # Placeholder
            'drum_break_complexity': 0.6,  # Placeholder
            'bass_weight': self._compute_bass_weight(analysis),
            'scratch_markers': 0.3,  # Placeholder
            'flow_syncopation': 0.7,  # Placeholder
            'trap_hi_hat_pattern': 0.5,  # Placeholder
            '808_presence': 0.8,  # Placeholder
            'swing_percentage': self._compute_swing(analysis),
            'lyrical_density': 0.6  # Placeholder
        }

    def _extract_latin(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Latin-specific parameters"""
        return {
            'clave_adherence': 0.7,  # Placeholder
            'montuno_complexity': 0.6,  # Placeholder
            'tumbao_pattern': 0.7,  # Placeholder
            'syncopation_level': self._compute_syncopation(analysis),
            'polyrhythm_density': 0.6,  # Placeholder
            'percussion_layering': 0.8,  # Placeholder
            'call_response_ratio': 0.5,  # Placeholder
            'guajeo_presence': 0.6,  # Placeholder
            'cascara_pattern': 0.7,  # Placeholder
            'anticipation_ratio': 0.6  # Placeholder
        }

    def _extract_world(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """World music-specific parameters"""
        return {
            'microtonal_markers': 0.3,  # Placeholder
            'modal_characteristics': 0.6,  # Placeholder
            'drone_presence': 0.4,  # Placeholder
            'ornament_density': 0.5,  # Placeholder
            'rhythmic_cycle_length': 0.6,  # Placeholder
            'pentatonic_usage': self._compute_pentatonic(analysis),
            'melisma_ratio': 0.4,  # Placeholder
            'heterophony': 0.5,  # Placeholder
            'metric_complexity': 0.6,  # Placeholder
            'instrumental_timbre': 0.5  # Placeholder
        }

    def _extract_generic(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Generic parameters for unknown genres"""
        return {
            'style_marker_1': 0.5,
            'style_marker_2': 0.5,
            'style_marker_3': 0.5,
            'style_marker_4': 0.5,
            'style_marker_5': 0.5,
            'style_marker_6': 0.5,
            'style_marker_7': 0.5,
            'style_marker_8': 0.5,
            'style_marker_9': 0.5,
            'style_marker_10': 0.5
        }

    # Helper methods
    def _compute_swing(self, analysis: MIDIAnalysisData) -> float:
        """Compute swing ratio"""
        # Simplified swing detection
        return 0.5  # Placeholder

    def _compute_walking_bass(self, analysis: MIDIAnalysisData) -> float:
        """Walking bass detection"""
        bass_notes = [n for n in analysis.all_notes if n['pitch'] < 48]
        if not bass_notes or analysis.duration_seconds == 0:
            return 0.0
        bass_density = len(bass_notes) / analysis.duration_seconds
        return float(min(bass_density / 4.0, 1.0))  # 4 notes per second = walking

    def _compute_comping(self, analysis: MIDIAnalysisData) -> float:
        """Comping pattern detection"""
        return 0.5  # Placeholder

    def _compute_bebop_articulation(self, analysis: MIDIAnalysisData) -> float:
        """Bebop articulation markers"""
        return 0.5  # Placeholder

    def _compute_blue_notes(self, analysis: MIDIAnalysisData) -> float:
        """Blue note usage"""
        # Detect flatted 3rd, 5th, 7th
        return 0.3  # Placeholder

    def _compute_counterpoint(self, analysis: MIDIAnalysisData) -> float:
        """Contrapuntal complexity"""
        if len(analysis.tracks_by_instrument) < 2:
            return 0.0
        return 0.5  # Placeholder

    def _compute_dynamic_contrast(self, analysis: MIDIAnalysisData) -> float:
        """Dynamic contrast"""
        if not analysis.all_velocities:
            return 0.0
        vel_range = max(analysis.all_velocities) - min(analysis.all_velocities)
        return float(vel_range / 127.0)

    def _compute_power_chords(self, analysis: MIDIAnalysisData) -> float:
        """Power chord detection"""
        if not analysis.chord_notes:
            return 0.0
        power_count = 0
        for chord in analysis.chord_notes:
            if len(chord) == 2:
                pitches = sorted([n['pitch'] for n in chord])
                interval = (pitches[1] - pitches[0]) % 12
                if interval == 7:  # Perfect fifth
                    power_count += 1
        return float(power_count / len(analysis.chord_notes))

    def _compute_backbeat(self, analysis: MIDIAnalysisData) -> float:
        """Backbeat strength"""
        return 0.7  # Placeholder

    def _compute_groove(self, analysis: MIDIAnalysisData) -> float:
        """Groove consistency"""
        if len(analysis.note_onsets) < 2:
            return 1.0
        iois = [analysis.note_onsets[i+1] - analysis.note_onsets[i]
                for i in range(len(analysis.note_onsets)-1)]
        if not iois:
            return 1.0
        cv = np.std(iois) / (np.mean(iois) + 0.001)
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def _compute_quantization(self, analysis: MIDIAnalysisData) -> float:
        """Quantization level"""
        if not analysis.note_onsets:
            return 0.5
        beat_duration = 60.0 / analysis.tempo_bpm
        deviations = []
        for onset in analysis.note_onsets:
            grid_pos = round(onset / beat_duration) * beat_duration
            deviation = abs(onset - grid_pos)
            deviations.append(deviation)
        avg_deviation = np.mean(deviations) / beat_duration
        return float(np.clip(1.0 - avg_deviation, 0.0, 1.0))

    def _compute_arpeggios(self, analysis: MIDIAnalysisData) -> float:
        """Arpeggio density"""
        if not analysis.melody_pitches or len(analysis.melody_pitches) < 3:
            return 0.0
        arp_count = 0
        for i in range(len(analysis.melody_pitches) - 2):
            diff1 = analysis.melody_pitches[i+1] - analysis.melody_pitches[i]
            diff2 = analysis.melody_pitches[i+2] - analysis.melody_pitches[i+1]
            if (diff1 > 0 and diff2 > 0) or (diff1 < 0 and diff2 < 0):
                if 3 <= abs(diff1) <= 7 and 3 <= abs(diff2) <= 7:
                    arp_count += 1
        return float(arp_count / (len(analysis.melody_pitches) - 2))

    def _compute_bass_weight(self, analysis: MIDIAnalysisData) -> float:
        """Bass presence/weight"""
        if not analysis.all_notes:
            return 0.0
        bass_notes = [n for n in analysis.all_notes if n['pitch'] < 48]
        if not bass_notes:
            return 0.0
        bass_velocity = np.mean([n['velocity'] for n in bass_notes])
        all_velocity = np.mean([n['velocity'] for n in analysis.all_notes])
        return float(bass_velocity / all_velocity) if all_velocity > 0 else 0.5

    def _compute_syncopation(self, analysis: MIDIAnalysisData) -> float:
        """Syncopation level"""
        if not analysis.note_onsets:
            return 0.0
        beat_duration = 60.0 / analysis.tempo_bpm
        off_beat = 0
        for onset in analysis.note_onsets:
            beat_pos = (onset % beat_duration) / beat_duration
            if 0.2 < beat_pos < 0.8:
                off_beat += 1
        return float(off_beat / len(analysis.note_onsets))

    def _compute_pentatonic(self, analysis: MIDIAnalysisData) -> float:
        """Pentatonic scale usage"""
        if not analysis.all_pitches:
            return 0.0
        # Major pentatonic: 0, 2, 4, 7, 9
        # Minor pentatonic: 0, 3, 5, 7, 10
        pitch_classes = [p % 12 for p in analysis.all_pitches]
        pc_set = set(pitch_classes)

        major_pent = {0, 2, 4, 7, 9}
        minor_pent = {0, 3, 5, 7, 10}

        major_overlap = len(pc_set & major_pent) / 5.0
        minor_overlap = len(pc_set & minor_pent) / 5.0

        return float(max(major_overlap, minor_overlap))
