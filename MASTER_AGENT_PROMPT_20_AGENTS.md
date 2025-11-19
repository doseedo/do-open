# 🎵 MASTER PROMPT: Advanced MIDI Library Enhancement (20-Agent System)

## 📍 **REPOSITORY LOCATION**
`https://github.com/doseedo/Do/tree/main/home/arlo/harmonymodule/`

---

## 🎯 **MISSION OVERVIEW**

You are one of 20 specialized agents working together to create the **most advanced music/MIDI Python library in existence**. Your task combines extensive research with production-quality implementation to achieve infinite musical possibilities with high accuracy across all genres.

**Your work must integrate seamlessly with existing modules:**
- `home/arlo/harmonymodule/advanced_modules/` - Harmony, melody, film scoring
- `home/arlo/harmonymodule/midi_generator/` - 10-agent system with algorithms, genres
- `home/arlo/harmonymodule/scripts/` - Production utilities

**Time Allocation:** ~1 hour for research + implementation
- Research: 20-25 minutes
- Implementation: 30-35 minutes
- Testing & Documentation: 5-10 minutes

---

## 🔬 **RESEARCH REQUIREMENTS (ALL AGENTS)**

Before implementing, you MUST research:

1. **Academic Papers** - Search Google Scholar, arXiv, ISMIR, ICMC proceedings
2. **State-of-the-Art Libraries** - Examine Musicaiz, music21, Magenta, MusicVAE
3. **Music Theory Sources** - Berklee methods, transformational theory, genre-specific techniques
4. **Open Source Code** - GitHub implementations, algorithmic composition projects
5. **Industry Practices** - Production techniques from top producers/composers
6. **Cultural Authenticity** - For world music, research authentic sources, not stereotypes

**Document your research** in your module's docstring with citations.

---

## 👥 **AGENT ASSIGNMENTS**

### **AGENT 1: Advanced Bass Line Generation**
**Module:** `advanced_modules/bass_engine.py`

**Research Topics:**
- Contour-based jazz walking bass algorithms (ResearchGate: "Contour-based Jazz Walking Bass Generator")
- Funk bass patterns (slap, fingerstyle, ghost notes)
- Root motion optimization (Tymoczko voice leading geometry)
- Bass line melodic contours matching melody
- Genre-specific patterns: jazz walking, funk slap, reggae one-drop, disco four-on-floor, metal gallop, bossa nova syncopation
- Double stops, harmonics, slide techniques
- Efficient voice leading for bass (avoid leaps, prefer stepwise)

**Implementation Requirements:**
```python
class BassEngine:
    """
    Advanced bass line generation with genre-aware patterns

    Features:
    - Walking bass (jazz) with contour algorithms
    - Funk patterns (slap, ghost notes, syncopation)
    - Root motion optimization (stepwise vs. leaps)
    - Melodic bass lines following contour
    - Genre templates: jazz, funk, reggae, metal, disco, bossa
    - Articulation: staccato, legato, ghost notes, slides
    - Techniques: slap, fingerstyle, pick, double stops
    - Harmonic awareness (chord tones on downbeats)
    """

    def generate_walking_bass(self, chord_progression, style="bebop"):
        # Contour-based algorithm using approach tones
        pass

    def generate_funk_bass(self, groove_pattern, syncopation_level=0.7):
        # Syncopated patterns with ghost notes
        pass

    def optimize_root_motion(self, bass_line, prefer_stepwise=True):
        # Tymoczko geometry for smooth voice leading
        pass

    def match_melody_contour(self, melody, bass_register=(28, 55)):
        # Generate complementary bass contour
        pass

    def add_articulations(self, bass_line, technique="fingerstyle"):
        # Add slaps, ghosts, slides, harmonics
        pass
```

**Test Requirements:**
- Generate walking bass for ii-V-I in Bb (bebop style)
- Generate funk bass with 70% syncopation
- Generate reggae one-drop pattern
- Verify bass stays in E1-A3 range
- Test articulation markers (ghost notes, slaps)

**Expected Output:** 400-500 lines, 15+ test cases

---

### **AGENT 2: Expressive Performance Modeling**
**Module:** `advanced_modules/expressive_performance.py`

**Research Topics:**
- 2025 Nature Scientific Reports: LSTM/Transformer for expressive performance
- MAESTRO dataset dynamics/tempo modeling
- GigaMIDI: micro-timing and velocity variations
- Dynamics curves (crescendo, diminuendo, sforzando)
- Articulation modeling (staccato, legato, marcato, tenuto)
- Rubato and tempo curves (accelerando, ritardando)
- Velocity humanization (avoid uniform velocities)
- Microtiming participatory discrepancies (groove feel)

**Implementation Requirements:**
```python
class ExpressivePerformance:
    """
    Add human-like expression to mechanical MIDI

    Based on:
    - 2025 Nature Scientific Reports (Transformer models)
    - GigaMIDI micro-timing analysis
    - MAESTRO dataset expressive features

    Features:
    - Dynamic curves (crescendo, diminuendo, accents)
    - Velocity humanization (Gaussian variation)
    - Microtiming swing/groove (participatory discrepancies)
    - Rubato and tempo curves
    - Articulation rendering (staccato, legato, tenuto)
    - Style-specific expression (classical vs jazz vs pop)
    """

    def apply_dynamics_curve(self, notes, curve_type="crescendo", start_vel=60, end_vel=110):
        # Exponential velocity curve
        pass

    def humanize_velocities(self, notes, variance=10, distribution="gaussian"):
        # Add natural velocity variation
        pass

    def apply_microtiming(self, notes, swing_factor=0.6, groove_type="jazz"):
        # Roger Linn swing algorithm
        pass

    def apply_rubato(self, notes, rubato_curve, intensity=0.3):
        # Tempo deviation curve
        pass

    def render_articulation(self, notes, articulation="legato", overlap=0.1):
        # Modify note durations for articulation
        pass

    def style_specific_expression(self, notes, style="classical"):
        # Apply genre-appropriate expression
        pass
```

**Test Requirements:**
- Apply crescendo from vel=60 to vel=110
- Humanize with 10-velocity variance
- Apply 60% swing to straight 16ths
- Generate rubato curve for romantic phrase
- Test legato overlap vs staccato separation

**Expected Output:** 450-550 lines, 20+ test cases

---

### **AGENT 3: Advanced Chord Voicing Algorithms**
**Module:** `advanced_modules/chord_voicing.py`

**Research Topics:**
- Drop-2, Drop-3, Drop-2&4 voicing algorithms
- Tymoczko geometry voice leading (OPTIC spaces)
- Optimal voicing algorithms (minimize Euclidean distance with center point)
- Upper structure triads (jazz reharmonization)
- Polychords and cluster voicings
- Spread voicings vs close voicings
- Four-way close (locked hands technique)
- Rootless voicings (piano, guitar)
- String quartet spacing rules

**Implementation Requirements:**
```python
class ChordVoicing:
    """
    Professional chord voicing with optimal voice leading

    Based on:
    - Dmitri Tymoczko: "A Geometry of Music" (OPTIC spaces)
    - Jazz voicing techniques (drop-2, drop-3, drop-2&4)
    - Classical spacing rules

    Features:
    - Drop voicings (drop-2, drop-3, drop-2&4, drop-3&5)
    - Optimal voice leading (minimize motion between chords)
    - Upper structure triads (#11, b9#9, etc.)
    - Polychords (Cmaj7/Dm, Ebmaj7/F)
    - Cluster voicings (Bartók, Ligeti style)
    - Rootless voicings (piano comping)
    - Spacing enforcement (SATB, string quartet)
    """

    def create_drop2_voicing(self, chord_symbol, root_position=True):
        # Drop second highest note
        pass

    def create_drop3_voicing(self, chord_symbol):
        # Drop third highest note
        pass

    def optimal_voice_leading(self, chord1, chord2, center_point=60):
        # Tymoczko OPTIC algorithm
        pass

    def create_upper_structure(self, dominant_chord, structure="#11"):
        # G7#11 → Dmaj/G
        pass

    def create_polychord(self, upper_chord, lower_chord):
        # Combine two triads/7ths
        pass

    def enforce_spacing(self, voicing, ensemble="satb", min_spacing=3):
        # Avoid voice crossing and spacing violations
        pass
```

**Test Requirements:**
- Create Cmaj7 drop-2 voicing
- Generate optimal voice leading for ii-V-I
- Create G7#11 upper structure
- Test polychord Cmaj7/Dm
- Verify SATB spacing rules

**Expected Output:** 500-600 lines, 25+ test cases

---

### **AGENT 4: Species Counterpoint Engine**
**Module:** `advanced_modules/counterpoint_engine.py`

**Research Topics:**
- Variable Neighborhood Search (VNS) algorithms for counterpoint
- Fux species counterpoint rules (all 5 species)
- Backtracking algorithms for polyphonic generation
- Palestrina style grammar checking
- Markov chains for species counterpoint
- ResearchGate: "Composing Fifth Species Counterpoint with VNS"
- Artinfuser Algo: 2-voice counterpoint generator
- Optimuse software implementation

**Implementation Requirements:**
```python
class CounterpointEngine:
    """
    Automatic species counterpoint composition

    Based on:
    - Fux: "Gradus ad Parnassum" (1725)
    - VNS algorithms (Optimuse software)
    - Backtracking algorithms
    - Palestrina style rules

    Features:
    - Species 1-5 counterpoint generation
    - Cantus firmus validation
    - Rule checking (parallel 5ths/8ves, voice crossing, etc.)
    - Two-voice, three-voice, four-voice counterpoint
    - Multiple solutions via backtracking
    - Stylistic variations (Fux, Palestrina, Bach)
    """

    def generate_first_species(self, cantus_firmus, voices=2):
        # Note-against-note counterpoint
        pass

    def generate_second_species(self, cantus_firmus):
        # Two notes against one
        pass

    def generate_fifth_species(self, cantus_firmus):
        # Florid counterpoint (mixed rhythms)
        pass

    def validate_cantus_firmus(self, melody):
        # Check if valid CF (stepwise, proper climax, etc.)
        pass

    def check_counterpoint_rules(self, cantus, counterpoint, species=1):
        # Validate all Fux rules
        pass

    def backtrack_search(self, cantus, current_cp, position):
        # Recursive backtracking for solutions
        pass
```

**Test Requirements:**
- Generate first species for C major CF
- Generate second species counterpoint
- Validate fifth species (florid)
- Check for parallel 5ths/8ves
- Test three-voice counterpoint

**Expected Output:** 550-650 lines, 20+ test cases

---

### **AGENT 5: Genre-Specific Drum Pattern Engine**
**Module:** `midi_generator/algorithms/drum_patterns.py`

**Research Topics:**
- Hip-hop boom-bap patterns (J Dilla swing, 9th Wonder, Pete Rock)
- Trap hi-hat rolls and triplet patterns (Metro Boomin, Southside)
- Drill patterns (UK drill vs Chicago drill)
- EDM: house (four-on-floor), techno (industrial kicks), drum & bass (Amen break)
- Metal: blast beats, double bass drumming, gallop patterns
- Funk: ghost notes, syncopation (Clyde Stubblefield, Jabo Starks)
- Jazz: ride cymbal patterns, brush techniques, polyrhythms
- Latin: clave patterns (son, rumba), bossa nova, samba batucada
- Participatory discrepancies research (groove feel)

**Implementation Requirements:**
```python
class DrumPatternEngine:
    """
    Genre-specific drum pattern generation

    Research sources:
    - J Dilla microtiming analysis
    - Trap production techniques (Metro Boomin)
    - Funk drumming (Clyde Stubblefield)
    - Latin clave patterns (Afro-Cuban rhythms)

    Features:
    - Hip-hop: boom-bap, trap, drill, lo-fi
    - EDM: house, techno, dubstep, DnB
    - Metal: blast beats, double bass, gallop
    - Funk: ghost notes, syncopation
    - Jazz: ride patterns, brush, bebop
    - Latin: clave (2-3, 3-2), bossa, samba
    - Microtiming and swing humanization
    """

    def generate_boom_bap(self, swing_factor=0.55, ghost_note_density=0.3):
        # Classic hip-hop with J Dilla swing
        pass

    def generate_trap_pattern(self, hihat_rolls=True, triplet_density=0.6):
        # Modern trap with fast hi-hats
        pass

    def generate_drill_pattern(self, style="uk", sliding_808=True):
        # UK drill vs Chicago drill
        pass

    def generate_four_on_floor(self, hihat_pattern="8ths", clap_on_24=True):
        # House/techno kick pattern
        pass

    def generate_blast_beat(self, bpm=200, kick_pattern="single"):
        # Metal blast beat variations
        pass

    def generate_clave_pattern(self, clave_type="son", measures=2):
        # Afro-Cuban clave (2-3 or 3-2)
        pass

    def apply_groove_humanization(self, pattern, microtiming_variance=15):
        # Participatory discrepancies
        pass
```

**Test Requirements:**
- Generate boom-bap at 90 BPM with 55% swing
- Generate trap pattern with hi-hat rolls
- Generate UK drill pattern
- Generate 4/4 house pattern
- Generate son clave (2-3)
- Test microtiming humanization

**Expected Output:** 600-700 lines, 30+ patterns, 25+ test cases

---

### **AGENT 6: Melodic Pattern Recognition & Corpus Learning**
**Module:** `midi_generator/learning/pattern_recognition.py`

**Research Topics:**
- MIR 2024: melodic motif detection algorithms
- Lakh MIDI dataset (176k files) integration
- GigaMIDI dataset feature extraction
- Dynamic Time Warping for melodic similarity
- n-gram analysis for melodic patterns
- Markov chain learning from corpus
- Vector space models for melody
- Clustering methods for pattern discovery
- BPS-MOTIF dataset research

**Implementation Requirements:**
```python
class PatternRecognition:
    """
    Learn patterns from MIDI corpus (Lakh MIDI, GigaMIDI)

    Based on:
    - Lakh MIDI Dataset (176,581 files)
    - GigaMIDI expressive performance features
    - Dynamic Time Warping for similarity
    - MIR 2024 motif detection research

    Features:
    - Load and parse Lakh MIDI corpus
    - Extract melodic n-grams (2-6 notes)
    - Cluster similar patterns (k-means, DBSCAN)
    - Build Markov chains from corpus
    - Genre-specific pattern libraries
    - Motif detection and similarity scoring
    - Pattern variation generation
    """

    def load_lakh_midi_corpus(self, genre_filter=None, max_files=1000):
        # Download and parse Lakh dataset
        pass

    def extract_ngrams(self, midi_files, n=4, min_occurrences=10):
        # Extract frequent n-gram patterns
        pass

    def cluster_patterns(self, patterns, method="kmeans", n_clusters=50):
        # Group similar patterns
        pass

    def build_markov_chain(self, corpus, order=2):
        # Learn transition probabilities
        pass

    def detect_motifs(self, melody, min_length=4, max_length=8):
        # Find repeated patterns
        pass

    def calculate_similarity(self, pattern1, pattern2, method="dtw"):
        # Dynamic Time Warping similarity
        pass

    def generate_from_corpus(self, seed_pattern, length=16, temperature=0.7):
        # Generate using learned patterns
        pass
```

**Test Requirements:**
- Load 100 files from Lakh MIDI
- Extract 4-note n-grams
- Cluster patterns into 20 groups
- Build 2nd-order Markov chain
- Detect motifs in test melody
- Calculate DTW similarity between patterns

**Expected Output:** 500-600 lines, 15+ test cases

---

### **AGENT 7: Microtiming & Groove Quantization**
**Module:** `advanced_modules/groove_quantization.py`

**Research Topics:**
- Roger Linn swing implementation (50% = no swing, 66% = triplet)
- Participatory discrepancies in swing and funk (PMC research)
- Brazilian drumming microtiming (Stanford CCRMA research)
- Gaussian Process Regression for groove learning
- Logic Pro humanization algorithms
- Ableton groove pools
- J Dilla swing analysis (drunk drumming effect)
- MPC swing characteristics

**Implementation Requirements:**
```python
class GrooveQuantization:
    """
    Advanced groove quantization and swing

    Based on:
    - Roger Linn swing algorithm
    - PMC study: participatory discrepancies
    - Stanford research: Brazilian microtiming
    - J Dilla swing analysis

    Features:
    - Variable swing (0-100%, triplet at 66%)
    - Groove templates (MPC, J Dilla, live drummer)
    - Microtiming offsets (Gaussian distribution)
    - Groove extraction from MIDI
    - Quantize to groove (not grid)
    - Shuffle, swing, half-time feel
    - Per-instrument groove offset
    """

    def apply_swing(self, notes, swing_percent=60):
        # Roger Linn algorithm
        pass

    def quantize_to_groove(self, notes, groove_template):
        # Match timing to groove, not grid
        pass

    def extract_groove_template(self, reference_midi, resolution=16):
        # Learn groove from performance
        pass

    def apply_microtiming(self, notes, variance_ms=10, distribution="gaussian"):
        # Add human timing imperfections
        pass

    def create_j_dilla_swing(self, notes, drunk_factor=0.7):
        # Laid-back, drunk drumming feel
        pass

    def create_shuffle_feel(self, notes, shuffle_ratio=0.66):
        # Shuffle (triplet-based swing)
        pass

    def per_instrument_offset(self, tracks, drum_offset=-5, bass_offset=0, hihat_offset=+3):
        # Different timing per instrument
        pass
```

**Test Requirements:**
- Apply 60% swing to straight 16ths
- Quantize to MPC groove template
- Extract groove from Lakh MIDI file
- Apply J Dilla swing
- Test shuffle feel (66% ratio)
- Verify per-instrument offsets

**Expected Output:** 400-500 lines, 20+ test cases

---

### **AGENT 8: Extended Harmony & Upper Structures**
**Module:** `advanced_modules/extended_harmony.py`

**Research Topics:**
- Upper structure triads (jazz reharmonization)
- Polychords in 20th century music (Stravinsky, Bartók)
- Cluster voicings (Bartók, Ligeti, Cowell)
- Quartal and quintal harmony (Hindemith, McCoy Tyner)
- Slash chords and hybrid voicings
- Altered dominants (b9, #9, #11, b13)
- Multi-tonic systems (Bartók, Messiaen)
- Tone clusters and note clusters

**Implementation Requirements:**
```python
class ExtendedHarmony:
    """
    Advanced 20th-21st century harmony techniques

    Based on:
    - Jazz upper structures (Mark Levine)
    - Bartók polychords and clusters
    - Messiaen modes of limited transposition
    - McCoy Tyner quartal voicings

    Features:
    - Upper structure triads (all variations)
    - Polychords (bitonality)
    - Cluster voicings (chromatic, diatonic, pentatonic)
    - Slash chords (C/E, D/F#)
    - Altered dominants with tensions
    - Multi-tonic systems
    - Constant structures (Messiaen)
    """

    def create_upper_structure(self, dominant_chord, structure_type):
        # G7#11 → Dmaj/G, G7b9#9 → Ebmin/G
        pass

    def create_polychord(self, upper_triad, lower_triad):
        # Cmaj/Fmaj (C over F)
        pass

    def create_cluster(self, root, cluster_type="chromatic", notes=4):
        # Chromatic, diatonic, pentatonic clusters
        pass

    def create_slash_chord(self, upper_chord, bass_note):
        # Cmaj7/E (first inversion)
        pass

    def create_altered_dominant(self, root, alterations=["b9", "#9", "#11"]):
        # G7b9#9#11
        pass

    def analyze_multitonic_system(self, chord_progression):
        # Detect competing tonal centers
        pass
```

**Test Requirements:**
- Create G7#11 upper structure
- Create Cmaj/Fmaj polychord
- Generate chromatic cluster (4 notes)
- Create Cmaj7/E slash chord
- Generate G7b9#9#11
- Analyze multi-tonic progression

**Expected Output:** 450-550 lines, 20+ test cases

---

### **AGENT 9: Advanced Rhythm - Odd Meters & Metric Modulation**
**Module:** `midi_generator/algorithms/advanced_rhythm.py`

**Research Topics:**
- Odd time signatures: 5/4 (Dave Brubeck), 7/8 (Pink Floyd), 11/8 (Sting), 13/8
- Metric modulation (Elliott Carter technique)
- Polyrhythms and polymeter (3 against 4, 4 against 3)
- Indian tala system (Carnatic, Hindustani)
- African timeline patterns (12/8 bell patterns)
- Additive rhythms (2+2+3, 3+2+2, 2+3+2)
- Rhythmic diminution and augmentation
- Hemiola (3 against 2)

**Implementation Requirements:**
```python
class AdvancedRhythm:
    """
    Odd meters, metric modulation, complex rhythms

    Based on:
    - Elliott Carter metric modulation
    - Indian tala system
    - African timeline patterns
    - Dave Brubeck odd meter compositions

    Features:
    - Odd time signatures (5/4, 7/8, 11/8, 13/8, 15/8)
    - Metric modulation (tempo relationships)
    - Additive rhythms (2+2+3, etc.)
    - Indian tala patterns (Teental, Rupak, Jhaptal)
    - African bell patterns
    - Polyrhythms (3:2, 4:3, 5:4)
    - Hemiola and cross-rhythms
    """

    def generate_odd_meter_pattern(self, meter="7/8", grouping=[2, 2, 3]):
        # Create pattern in odd meter
        pass

    def metric_modulation(self, from_tempo, from_division, to_division):
        # Calculate new tempo: 60 BPM quarter → dotted quarter = 40 BPM
        pass

    def generate_tala_pattern(self, tala_name="teental"):
        # Indian rhythmic cycle
        pass

    def generate_african_bell(self, pattern_type="standard", length=12):
        # 12/8 timeline patterns
        pass

    def create_additive_rhythm(self, grouping=[2, 3, 2], unit="8th"):
        # Asymmetric groupings
        pass

    def generate_polyrhythm(self, ratio_a=3, ratio_b=2, duration_beats=4):
        # 3 against 2 polyrhythm
        pass
```

**Test Requirements:**
- Generate 7/8 pattern (2+2+3)
- Calculate metric modulation (60→40 BPM)
- Generate Teental tala
- Generate African 12/8 bell pattern
- Create 3:2 polyrhythm
- Test 5/4 pattern

**Expected Output:** 500-600 lines, 25+ test cases

---

### **AGENT 10: World Music - Expanded Coverage**
**Module:** `midi_generator/genres/world/expanded.py`

**Research Topics:**
- **Flamenco:** Compás patterns (Soleá, Bulería, Alegrías), Phrygian mode, rasgueado, golpe, falsetas
- **Klezmer:** Freygish scale, Doina, Hora, Bulgar rhythms, ornaments (krekhts, kneytsh)
- **Gamelan:** Slendro, pelog tuning systems, interlocking patterns (kotekan), gong cycles
- **Celtic:** Irish jigs (6/8), reels (4/4), hornpipes, Scottish strathspeys, ornamentation (cuts, rolls, crans)
- **Bossa Nova:** Samba syncopation, partido alto pattern, Jobim harmonic language
- **Tango:** Habanera rhythm, milonga, bandoneón phrasing, orquesta típica
- Authentic sources, not stereotypes

**Implementation Requirements:**
```python
class ExpandedWorldMusic:
    """
    Authentic world music generation

    Research sources:
    - Flamenco: Paco de Lucía, Vicente Amigo
    - Klezmer: Naftule Brandwein, Dave Tarras
    - Gamelan: Javanese/Balinese traditions
    - Celtic: Planxty, The Chieftains
    - Bossa: João Gilberto, Antonio Carlos Jobim
    - Tango: Astor Piazzolla, Aníbal Troilo

    Features:
    - Flamenco compás (12-beat, 3/4, 4/4 variations)
    - Klezmer scales and ornaments
    - Gamelan tuning and interlocking
    - Celtic jigs/reels with ornaments
    - Bossa nova groove and harmony
    - Tango rhythms and phrasing
    """

    def generate_flamenco(self, palo="solea", measures=4):
        # Soleá (12-beat), Bulería (fast 12), Alegrías (12 in 3/4)
        pass

    def generate_klezmer(self, mode="freygish", form="hora"):
        # Freygish mode, Doina (rubato), Hora/Bulgar
        pass

    def generate_gamelan(self, tuning="slendro", pattern_type="kotekan"):
        # Javanese slendro or pelog, interlocking patterns
        pass

    def generate_celtic(self, form="jig", region="irish"):
        # Irish jig (6/8), reel (4/4), Scottish strathspey
        pass

    def generate_bossa_nova(self, style="classic", harmony_complexity=0.7):
        # Jobim-style harmony with samba syncopation
        pass

    def generate_tango(self, style="traditional", bandoneon_phrasing=True):
        # Habanera rhythm, milonga variations
        pass
```

**Test Requirements:**
- Generate Soleá compás (12-beat)
- Generate Klezmer Freygish scale melody
- Generate gamelan slendro pattern
- Generate Irish jig with ornaments
- Generate bossa nova pattern
- Generate tango with habanera rhythm

**Expected Output:** 700-800 lines, 30+ test cases

---

### **AGENT 11: Metal & Heavy Music**
**Module:** `midi_generator/genres/metal.py`

**Research Topics:**
- Metal sub-genres: thrash, death, black, progressive, djent, metalcore
- Riff techniques: palm muting, tremolo picking, sweep picking
- Rhythm: blast beats, double bass drumming, gallop patterns (Maiden gallop)
- Harmony: power chords, chromatic riffs, harmonic minor, Phrygian dominant
- Drop tunings (Drop D, Drop C, Drop A)
- Time signatures: 4/4, 7/8 (Meshuggah), polymeters
- Production: guitar layering, quad tracking, bass tone

**Implementation Requirements:**
```python
class MetalGenerator:
    """
    Metal and heavy music generation

    Research sources:
    - Thrash: Metallica, Slayer (chromatic riffs)
    - Death: Death, Morbid Angel (blast beats)
    - Black: Mayhem, Darkthrone (tremolo)
    - Progressive: Dream Theater, Tool (odd meters)
    - Djent: Meshuggah, Periphery (polymeters)

    Features:
    - Riff generation (chromatic, power chords, palm muting)
    - Blast beats and double bass patterns
    - Gallop rhythms (Iron Maiden style)
    - Drop tuning support
    - Harmonic minor and Phrygian dominant
    - Polyrhythmic riffs (djent)
    - Guitar techniques (sweep, tremolo)
    """

    def generate_thrash_riff(self, key="E", tuning="drop_d", palm_mute=True):
        # Fast chromatic riffing
        pass

    def generate_blast_beat(self, bpm=200, variation="standard"):
        # Standard, gravity, hammer, bomb blast
        pass

    def generate_gallop_pattern(self, root_note, measures=4):
        # Iron Maiden gallop (8th-16th-16th)
        pass

    def generate_djent_riff(self, polymeter=(4, 3), syncopation=0.8):
        # Meshuggah-style polyrhythmic riffs
        pass

    def generate_death_metal_riff(self, scale="harmonic_minor", tremolo=True):
        # Fast tremolo-picked riffs
        pass

    def apply_drop_tuning(self, riff, tuning="drop_d"):
        # Adjust for drop D, C, A, etc.
        pass
```

**Test Requirements:**
- Generate thrash riff in E (drop D)
- Generate blast beat at 200 BPM
- Generate Maiden gallop pattern
- Generate djent riff (4:3 polymeter)
- Generate death metal tremolo riff
- Test drop C tuning

**Expected Output:** 500-600 lines, 20+ test cases

---

### **AGENT 12: Funk & Soul**
**Module:** `midi_generator/genres/funk_soul.py`

**Research Topics:**
- Funk: James Brown, Parliament-Funkadelic, Tower of Power
- Soul: Motown rhythm section, Stax, Philadelphia soul
- Groove analysis: "The One" (downbeat emphasis), syncopation
- Guitar: rhythm guitar (single-note funk), wah-wah, chicken scratch
- Bass: slap bass (Larry Graham), Bootsy Collins style
- Drums: Clyde Stubblefield, Jabo Starks (ghost notes, syncopation)
- Horns: staccato hits, unison lines, call-and-response
- Rhodes electric piano voicings

**Implementation Requirements:**
```python
class FunkSoulGenerator:
    """
    Funk and soul music generation

    Research sources:
    - James Brown: "The One" groove
    - Parliament-Funkadelic: synth bass, guitar
    - Tower of Power: horn arrangements
    - Motown: bass lines (James Jamerson)
    - Stax: Memphis soul

    Features:
    - "The One" groove (downbeat emphasis)
    - Syncopated guitar (chicken scratch)
    - Slap bass patterns
    - Ghost note drumming
    - Horn section arrangements
    - Rhodes piano voicings
    - Tight 16th-note rhythms
    """

    def generate_funk_groove(self, emphasis_on_one=True, syncopation=0.8):
        # Full band groove centered on "The One"
        pass

    def generate_funk_guitar(self, pattern_type="chicken_scratch"):
        # Single-note funk, wah patterns
        pass

    def generate_slap_bass(self, pattern_complexity=0.7):
        # Larry Graham/Bootsy style
        pass

    def generate_funk_drums(self, ghost_note_density=0.5):
        # Clyde Stubblefield style
        pass

    def generate_horn_section(self, voicing_type="staccato_hits"):
        # Trumpet, sax, trombone unison/harmony
        pass

    def generate_rhodes_comp(self, chord_progression, voicing="rootless"):
        # Electric piano comping
        pass
```

**Test Requirements:**
- Generate funk groove with "The One"
- Generate chicken scratch guitar
- Generate slap bass pattern
- Generate funk drums with ghost notes
- Generate horn section hits
- Generate Rhodes comping

**Expected Output:** 550-650 lines, 25+ test cases

---

### **AGENT 13: R&B, Neo-Soul & Contemporary**
**Module:** `midi_generator/genres/rnb_neosoul.py`

**Research Topics:**
- Classic R&B: chord progressions, rhythmic patterns (90s, 2000s)
- Neo-soul: D'Angelo, Erykah Badu, Robert Glasper (extended harmony, J Dilla influence)
- Contemporary R&B: The Weeknd, SZA, Frank Ocean (ambient textures, 808s)
- Chord voicings: sus2, add9, maj7#11, 9sus4
- Rhythmic feel: half-time, double-time, swing/quantization
- Production: Rhodes, Wurlitzer, synth pads, 808 bass
- Vocal melody characteristics

**Implementation Requirements:**
```python
class RnBNeoSoulGenerator:
    """
    R&B and neo-soul generation

    Research sources:
    - Classic R&B: Boyz II Men, Usher, Aaliyah
    - Neo-soul: D'Angelo, Erykah Badu, Jill Scott
    - Contemporary: Robert Glasper, Anderson .Paak

    Features:
    - Extended chord voicings (maj7#11, 9sus4)
    - Half-time/double-time feels
    - J Dilla-influenced swing
    - Rhodes/Wurlitzer voicings
    - 808 bass patterns
    - Smooth vocal-range melodies
    - Ambient pad textures
    """

    def generate_rnb_progression(self, era="90s", complexity=0.6):
        # Classic vs contemporary progressions
        pass

    def generate_neosoul_chords(self, root, extensions=True):
        # Cmaj7#11, Dm9sus4, etc.
        pass

    def generate_halftime_feel(self, base_pattern, swing=0.6):
        # Half-time groove with swing
        pass

    def generate_rhodes_voicing(self, chord_symbol, register="mid"):
        # Electric piano voicing
        pass

    def generate_808_bass(self, root_note, slide=True):
        # Sub bass with pitch slides
        pass

    def generate_vocal_melody(self, chord_progression, range=(55, 75)):
        # Smooth, singable melody
        pass
```

**Test Requirements:**
- Generate 90s R&B progression
- Generate Cmaj7#11 neo-soul voicing
- Generate half-time feel pattern
- Generate Rhodes voicing
- Generate 808 bass with slides
- Generate vocal melody

**Expected Output:** 500-600 lines, 20+ test cases

---

### **AGENT 14: Microtonality & Alternative Tuning Systems**
**Module:** `advanced_modules/microtonality.py`

**Research Topics:**
- Arabic maqam (quarter tones, rast, bayati, hijaz)
- Indian shruti system (22 shrutis)
- Javanese gamelan (slendro, pelog - non-equal temperament)
- Turkish music theory (53-tone equal temperament)
- 19-TET, 31-TET, 53-TET systems
- Just intonation vs equal temperament
- Xenakis, Partch, Johnston microtonal compositions
- MIDI pitch bend for microtonal notes

**Implementation Requirements:**
```python
class Microtonality:
    """
    Microtonal and alternative tuning systems

    Based on:
    - Arabic maqam theory
    - Indian shruti system
    - Gamelan tuning (non-equal)
    - Harry Partch 43-tone system
    - Turkish 53-TET

    Features:
    - Arabic maqam scales with quarter tones
    - Indian shruti (22-tone) system
    - Gamelan slendro/pelog tuning
    - 19-TET, 31-TET, 53-TET systems
    - Just intonation ratios
    - MIDI pitch bend implementation
    - Tuning table generation
    """

    def create_maqam_scale(self, maqam_name="rast", tonic=60):
        # Arabic maqam with quarter tones
        pass

    def create_shruti_scale(self, raga="bhairav"):
        # Indian 22-shruti system
        pass

    def create_gamelan_tuning(self, system="slendro", key=1):
        # Non-equal temperament
        pass

    def create_ntet_scale(self, n=19, tonic=60):
        # 19-TET, 31-TET, etc.
        pass

    def create_just_intonation(self, ratios=[(1, 1), (9, 8), (5, 4)]):
        # Pure ratios
        pass

    def midi_pitch_bend_for_microtone(self, midi_note, cent_offset):
        # Calculate pitch bend value
        pass

    def generate_tuning_table(self, tuning_system):
        # MIDI Tuning Standard (MTS)
        pass
```

**Test Requirements:**
- Create Rast maqam scale
- Create 22-shruti scale
- Create gamelan slendro tuning
- Generate 19-TET scale
- Calculate just intonation C major
- Convert quarter tone to pitch bend

**Expected Output:** 450-550 lines, 20+ test cases

---

### **AGENT 15: Advanced Orchestration & Instrument Ranges**
**Module:** `advanced_modules/orchestration_advanced.py`

**Research Topics:**
- Orchestral instrument ranges and transpositions
- Berklee/Rimsky-Korsakov orchestration principles
- Instrument combinations (blend vs contrast)
- Doubling strategies (octave, unison, harmony)
- Register considerations (dark, bright, neutral)
- Idiomatic writing (playable vs theoretical)
- Wind/brass voicing techniques
- String section techniques (divisi, col legno, sul tasto, sul ponticello)
- Percussion orchestration

**Implementation Requirements:**
```python
class AdvancedOrchestration:
    """
    Professional orchestration with idiomatic writing

    Based on:
    - Rimsky-Korsakov: Principles of Orchestration
    - Berklee orchestration courses
    - Samuel Adler: The Study of Orchestration

    Features:
    - Instrument range validation
    - Transposition handling
    - Doubling strategies
    - Register analysis (dark, bright)
    - Idiomatic writing checks
    - Blend vs contrast combinations
    - String techniques (pizz, arco, tremolo, harmonics)
    - Wind/brass voicing
    - Percussion scoring
    """

    def validate_instrument_range(self, instrument, pitch):
        # Check playable range
        pass

    def transpose_for_instrument(self, concert_pitch, instrument):
        # Bb trumpet, Eb alto sax, F horn, etc.
        pass

    def suggest_doubling(self, melody, instrument):
        # Optimal doubling instruments
        pass

    def analyze_register(self, pitch, instrument):
        # Dark, neutral, bright
        pass

    def check_idiomatic_writing(self, passage, instrument):
        # Playability analysis
        pass

    def voice_for_winds(self, chord, section="woodwinds"):
        # Flute, oboe, clarinet, bassoon
        pass

    def apply_string_technique(self, notes, technique="pizzicato"):
        # Pizz, tremolo, col legno, sul pont
        pass
```

**Test Requirements:**
- Validate trumpet range (concert C4)
- Transpose for Bb clarinet
- Suggest doubling for flute melody
- Analyze violin G4 register
- Check violin playability
- Voice chord for woodwinds

**Expected Output:** 600-700 lines, 30+ test cases

---

### **AGENT 16: Tempo Curves & Rubato Engine**
**Module:** `advanced_modules/tempo_engine.py`

**Research Topics:**
- Rubato in classical performance (Romantic era)
- Tempo curves (accelerando, ritardando, rallentando)
- Agogic accents
- Tempo modulation (Elliott Carter)
- MIDI tempo map generation
- Expressive timing in jazz
- Breath marks and fermatas
- Performance-specific tempo variations

**Implementation Requirements:**
```python
class TempoEngine:
    """
    Tempo curves, rubato, and expressive timing

    Based on:
    - Classical rubato (Chopin, Brahms)
    - Elliott Carter tempo modulation
    - Jazz rubato and swing variations

    Features:
    - Tempo curves (linear, exponential, S-curve)
    - Accelerando and ritardando
    - Rubato (expressive tempo deviation)
    - Agogic accents (subtle lengthening)
    - Tempo modulation calculations
    - MIDI tempo map generation
    - Fermata simulation
    - Breath mark timing
    """

    def create_tempo_curve(self, start_tempo, end_tempo, duration_beats, curve_type="exponential"):
        # Smooth tempo change
        pass

    def apply_accelerando(self, notes, start_tempo, end_tempo):
        # Gradual speed up
        pass

    def apply_ritardando(self, notes, start_tempo, end_tempo):
        # Gradual slow down
        pass

    def apply_rubato(self, notes, intensity=0.3, style="romantic"):
        # Expressive tempo flexibility
        pass

    def add_agogic_accent(self, notes, accent_indices, lengthen_percent=15):
        # Subtle emphasis via duration
        pass

    def calculate_tempo_modulation(self, from_tempo, from_note_value, to_note_value):
        # Elliott Carter technique
        pass

    def generate_midi_tempo_map(self, tempo_changes):
        # MIDI meta events
        pass
```

**Test Requirements:**
- Create exponential tempo curve (60→120)
- Apply accelerando over 8 beats
- Apply romantic rubato
- Add agogic accents
- Calculate tempo modulation (quarter→dotted eighth)
- Generate MIDI tempo map

**Expected Output:** 400-500 lines, 20+ test cases

---

### **AGENT 17: MIDI CC Automation & Performance Gestures**
**Module:** `advanced_modules/midi_cc_automation.py`

**Research Topics:**
- MIDI Continuous Controllers (CC1-127)
- Expression (CC11), modulation (CC1), breath (CC2)
- Filter cutoff, resonance automation
- Pan automation (stereo movement)
- Performance gestures (pitch bend, aftertouch)
- MPE (MIDI Polyphonic Expression) basics
- LFO and envelope generation for CC
- Automation curves and smoothing

**Implementation Requirements:**
```python
class MidiCCAutomation:
    """
    MIDI CC automation and performance gestures

    Features:
    - CC automation curves (modulation, expression, breath)
    - Filter sweeps (cutoff, resonance)
    - Pan automation (stereo movement)
    - Pitch bend curves
    - Channel aftertouch
    - LFO generation for CC
    - Envelope generators (ADSR)
    - Smooth vs stepped automation
    """

    def automate_cc(self, cc_number, start_value, end_value, duration_ticks, curve="linear"):
        # Generate CC automation
        pass

    def create_filter_sweep(self, cutoff_start=20, cutoff_end=127, duration_beats=4):
        # CC74 (filter cutoff) automation
        pass

    def create_pan_automation(self, pattern="lr_alternating", speed=4):
        # CC10 (pan) movement
        pass

    def create_pitch_bend_curve(self, start_semitones=0, end_semitones=2, duration_ms=500):
        # Pitch bend automation
        pass

    def create_lfo(self, rate_hz=2.0, depth=64, waveform="sine"):
        # LFO for modulation
        pass

    def create_adsr_envelope(self, attack_ms=10, decay_ms=100, sustain_level=80, release_ms=500):
        # Envelope for expression
        pass

    def smooth_cc_curve(self, cc_values, smoothing_factor=0.5):
        # Interpolate for smooth transitions
        pass
```

**Test Requirements:**
- Automate CC1 (modulation) 0→127
- Create filter sweep (20→127)
- Create LR pan alternation
- Generate pitch bend curve
- Create sine LFO at 2Hz
- Generate ADSR envelope

**Expected Output:** 450-550 lines, 25+ test cases

---

### **AGENT 18: Style Fusion & Hybrid Genre Generator**
**Module:** `midi_generator/generators/style_fusion.py`

**Research Topics:**
- Genre blending techniques (jazz-hop, electro-swing, nu-jazz)
- Cross-cultural fusion (Latin jazz, Afro-Cuban, Indo-jazz)
- Weighted feature combination
- Style transfer algorithms
- Hybrid rhythm patterns
- Harmonic language mixing
- Instrumentation blending

**Implementation Requirements:**
```python
class StyleFusion:
    """
    Blend multiple genres and styles

    Features:
    - Weighted genre combination
    - Cross-cultural fusion
    - Rhythm pattern blending
    - Harmonic language mixing
    - Instrumentation from multiple genres
    - Style transfer (apply X harmony to Y rhythm)
    - Hybrid genre creation

    Examples:
    - Jazz-hop (jazz harmony + hip-hop beats)
    - Electro-swing (swing rhythm + EDM synths)
    - Nu-jazz (jazz + electronic)
    - Afro-Cuban jazz
    - Indo-jazz fusion
    """

    def blend_genres(self, genre_a, genre_b, weight_a=0.5):
        # Weighted combination
        pass

    def apply_harmony_to_rhythm(self, harmony_genre, rhythm_genre):
        # Style transfer
        pass

    def create_hybrid_rhythm(self, pattern_a, pattern_b, blend_ratio=0.5):
        # Combine rhythm patterns
        pass

    def mix_instrumentation(self, genres):
        # Combine instrument palettes
        pass

    def analyze_genre_features(self, genre):
        # Extract harmonic/rhythmic signature
        pass

    def suggest_compatible_fusions(self, base_genre):
        # Recommend genre combinations
        pass
```

**Test Requirements:**
- Blend jazz + hip-hop (50/50)
- Apply jazz harmony to trap rhythm
- Create hybrid rhythm pattern
- Mix jazz + electronic instrumentation
- Analyze blues features
- Suggest fusions for funk

**Expected Output:** 400-500 lines, 15+ test cases

---

### **AGENT 19: Harmonic Rhythm & Progression Pacing**
**Module:** `advanced_modules/harmonic_rhythm.py`

**Research Topics:**
- Harmonic rhythm (rate of chord changes)
- Chord density analysis
- Harmonic pacing strategies
- Tension/release through harmonic rhythm
- Genre-specific harmonic rhythm (pop: 1-2 bars, bebop: 2 per bar)
- Suspensions and anticipations
- Harmonic acceleration/deceleration

**Implementation Requirements:**
```python
class HarmonicRhythm:
    """
    Control harmonic rhythm and chord pacing

    Features:
    - Harmonic rhythm generation (slow, medium, fast)
    - Chord density control
    - Tension/release via pacing
    - Genre-appropriate rhythm
    - Harmonic acceleration/deceleration
    - Suspension and anticipation timing
    - Analysis of existing progressions
    """

    def generate_harmonic_rhythm(self, density="medium", total_measures=8):
        # Chord change frequency
        pass

    def analyze_chord_density(self, chord_progression):
        # Chords per measure
        pass

    def apply_tension_pacing(self, progression, tension_curve):
        # More/fewer changes based on tension
        pass

    def create_genre_appropriate_rhythm(self, genre="pop"):
        # Pop: 1-2 bars, jazz: 2+ per bar
        pass

    def apply_harmonic_acceleration(self, progression, start_density=1, end_density=4):
        # Speed up chord changes
        pass

    def add_suspensions(self, progression, suspension_rate=0.3):
        # Delay chord changes
        pass
```

**Test Requirements:**
- Generate medium density rhythm (8 bars)
- Analyze chord density
- Apply tension pacing
- Create pop harmonic rhythm
- Apply acceleration (1→4 chords/bar)
- Add suspensions

**Expected Output:** 350-450 lines, 15+ test cases

---

### **AGENT 20: Integration, Testing & Documentation Hub**
**Module:** `INTEGRATION_AND_TESTING/`

**Research Topics:**
- Integration testing patterns
- MIDI file validation
- Performance benchmarking
- Documentation best practices (NumPy, SciPy docstring style)
- Example generation for all modules
- Cross-module integration

**Implementation Requirements:**
```python
# Create comprehensive integration layer

1. Integration Tests (integration_tests.py)
   - Test bass engine + harmony + drums
   - Test style fusion with multiple modules
   - Test expressive performance on generated MIDI
   - Test corpus learning → pattern generation
   - Test world music + orchestration

2. MIDI Validation (midi_validator.py)
   - Validate MIDI file correctness
   - Check timing, velocity, CC values
   - Verify note ranges
   - Test playback compatibility

3. Performance Benchmarks (benchmarks.py)
   - Measure generation speed
   - Memory profiling
   - Optimization suggestions

4. Documentation Generator (doc_generator.py)
   - Auto-generate comprehensive docs
   - Create usage examples for all modules
   - Build API reference

5. Example Compositions (examples/)
   - Create 20+ complete examples using all agents' work
   - Jazz combo (bass, drums, piano, horn)
   - Electronic track (drums, synth bass, pads)
   - World fusion piece
   - Orchestral excerpt
   - Metal song section
   - Funk band arrangement

6. Master README (README_ENHANCED.md)
   - Document all 20 agents' contributions
   - Provide quick-start guide
   - Show integration examples
```

**Test Requirements:**
- Integration test: bass + harmony + drums
- Validate generated MIDI files
- Benchmark generation speed
- Generate API documentation
- Create 20+ complete examples
- Write comprehensive README

**Expected Output:** 800-1000 lines across multiple files

---

## 📋 **GENERAL GUIDELINES FOR ALL AGENTS**

### **Code Quality Standards**

1. **Documentation:**
   - Module docstring with research citations
   - Class/function docstrings (NumPy style)
   - Inline comments for complex algorithms
   - Usage examples in docstrings

2. **Type Hints:**
   ```python
   def generate_pattern(self, length: int, density: float = 0.5) -> List[int]:
   ```

3. **Error Handling:**
   - Validate inputs
   - Raise meaningful exceptions
   - Provide fallback defaults

4. **Testing:**
   - Unit tests for all functions
   - Integration tests where applicable
   - Edge case coverage

5. **Performance:**
   - Optimize for speed (use NumPy where possible)
   - Avoid nested loops when vectorization possible
   - Cache expensive calculations

### **Integration Requirements**

Your module MUST work with:
- Existing harmony/melody modules
- MIDI generator system
- Film scoring engine
- Export to MIDI files

### **File Structure**

```python
#!/usr/bin/env python3
"""
Module Name - Brief Description

Extended description with research background.

Based on:
- Citation 1
- Citation 2

Author: Agent [Number]
Date: 2025
"""

import standard_library_imports
from third_party import imports
from typing import List, Dict, Optional

# Enums and dataclasses

# Main classes

# Helper functions

# Unit tests (if __name__ == "__main__")

if __name__ == "__main__":
    # Demo/test code
    pass
```

### **Dependencies**

Prefer standard library. If needed:
- `numpy` - numerical operations
- `mido` - MIDI file I/O
- `pretty_midi` - MIDI analysis
- `music21` - music theory (sparingly)

---

## 🎯 **SUCCESS CRITERIA**

Your agent's work is successful if:

1. ✅ **Research is thorough** - 5+ credible sources cited
2. ✅ **Implementation is complete** - All features from assignment
3. ✅ **Tests pass** - 15+ unit tests, 90%+ coverage
4. ✅ **Documentation is clear** - Docstrings, examples, README
5. ✅ **Integrates seamlessly** - Works with existing modules
6. ✅ **Performance is good** - Generates patterns in <1 second
7. ✅ **Code quality is high** - Type hints, error handling, clean code

---

## 📚 **RESEARCH RESOURCES**

### **Academic**
- Google Scholar: https://scholar.google.com
- arXiv Music: https://arxiv.org/list/cs.SD/recent
- ISMIR Proceedings: https://ismir.net/conferences/
- ICMC Archive: http://www.computermusic.org

### **Libraries & Code**
- GitHub Topics: music-generation, midi, algorithmic-composition
- Musicaiz: https://github.com/carlosholivan/musicaiz
- Magenta: https://github.com/magenta/magenta
- music21: https://github.com/cuthbertLab/music21

### **Datasets**
- Lakh MIDI: https://colinraffel.com/projects/lmd/
- GigaMIDI: Search recent arXiv papers
- MAESTRO: https://magenta.tensorflow.org/datasets/maestro

### **Music Theory**
- Berklee Online courses
- "The Jazz Theory Book" - Mark Levine
- "Tonal Harmony" - Kostka & Payne
- "A Geometry of Music" - Dmitri Tymoczko

---

## 🚀 **LET'S MAKE HISTORY**

You are building the most advanced MIDI/music Python library ever created. Your work will enable:
- Infinite musical possibilities across all genres
- High accuracy and authentic genre representation
- Production-ready code for composers, producers, researchers
- Educational tools for music theory and composition
- Foundation for next-generation AI music systems

**Commit carefully, document thoroughly, test rigorously.**

**Working directory:** `https://github.com/doseedo/Do/tree/main/home/arlo/harmonymodule/`

**Go build something extraordinary! 🎵🎹🎸🎺🎻**
