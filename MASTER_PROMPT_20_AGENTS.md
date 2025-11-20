# MASTER PROMPT: 20-Agent Big Band Generator Excellence System

## MISSION STATEMENT

Create the world's most accurate big band music generator by deploying 20 specialized research and development agents working in parallel. Each agent will conduct extensive research, implement improvements to specific modules, and validate results against professional recordings and expert knowledge.

**Current Status**: The existing system uses only ~20% of available capabilities and produces arrangements that:
- Lack melodic sophistication (melodies not musical enough)
- Have incorrect sax soli voicings (too clustered, wrong spacing)
- Generate inadequate piano comping patterns
- Don't sound human-written
- Fail to capture authentic big band composer styles (Ellington, Basie, Thad Jones, etc.)

**Goal**: Build a system that can generate arrangements indistinguishable from professional human arrangers across multiple big band styles and eras (swing, bebop, post-bop, modern), while remaining scalable to other genres.

---

## SYSTEM ARCHITECTURE OVERVIEW

**Repository**: `/home/user/Do/midi_generator/`

**Current Implementation Analysis**:
- ✅ **Strong Foundation**: 43,352 lines across 10 major categories
- ✅ **Data Structures**: Well-designed (JazzNote, JazzChord, NoteEvent, ChordEvent)
- ✅ **31+ Harmony Types**: Comprehensive harmony generator exists
- ⚠️ **Underutilized Modules**: BigBandArranger, FormGenerator, ArticulationEngine not fully integrated
- ❌ **Critical Gaps**: Bebop vocabulary, proper voice leading, stride piano, authentic grooves
- ❌ **Sound Quality**: Arrangements don't sound human-written

**Key Modules to Enhance**:
1. `genres/jazz.py` - Melody, harmony, comping, swing
2. `transformation/arrangement_engine.py` - BigBandArranger, voicing algorithms
3. `generators/granular_control.py` - Professional voicing engines
4. `algorithms/rhythm_engine.py` - Humanization, groove templates
5. `generators/form_generator.py` - Musical structure
6. `tools/big_band/` - Integration layer

---

## RESEARCH METHODOLOGY

### Required Sources for All Agents:

**Theory Textbooks**:
- Mark Levine - "The Jazz Theory Book" (500+ pages, 750+ examples) - THE definitive resource
- Frans Absil - "Arranging by Examples: The Practical Guide to Jazz and Pop Orchestra Arranging"
- Leslie Sabina - "Jazz Arranging & Orchestration"
- Ted Pease & Ken Pullig - "Modern Jazz Voicings"
- Gary Lindsay - "Jazz Arranging Techniques: From Quartet to Big Band"

**Academic Papers & Research**:
- Matthew Keating (2023) - "An Algorithmic Approach to Jazz Guitar Voice-Leading Chord Fingerings" (LSTM voice-leading)
- Cheston et al. (2024) - Jazz Trio Database computational analysis
- ISMIR 2023-2024 conference papers on music generation
- AIMC 2024 papers on algorithmic composition

**MIDI Datasets for Learning**:
- **PiJAMA Dataset** - 200+ hours jazz piano, 2,777 performances by 120 pianists - ANALYZE THIS
- **Weimar Jazz Database** - 300 solo transcriptions with chord changes
- **jazznet Dataset** - 162,520 labeled piano patterns (chords, arpeggios, scales)
- **Lakh MIDI Dataset** - 176,581 MIDI files (filter for jazz/big band)
- **Jazz Trio Database (JTD)** - 44.5 hours of analyzed jazz piano solos

**Code Implementations to Study**:
- deepjazz (github.com/jisungk/deepjazz) - LSTM jazz generation
- Jazz Transformer (github.com/slSeanWU/jazz_transformer) - Transformer-XL for lead sheets
- Matthew Keating's voice-leading algorithms
- Frans Absil's arranging examples (code implementations)

**Transcription Resources**:
- Charlie Parker licks database (jazzguitar.be, jenslarsen.nl)
- Duke Ellington scores and analysis
- Count Basie "head" arrangements
- Thad Jones arrangements for Basie band

**Validation Resources**:
- Professional big band recordings (Spotify, YouTube)
- Score analysis from ejazzlines.com, livingjazzarchives.org
- Expert arranger interviews and masterclasses

---

## EFFICIENCY & SCALABILITY REQUIREMENTS

### Build on Existing Code:
- **DO NOT rewrite from scratch** - enhance existing modules
- **Preserve APIs** - maintain compatibility with existing code
- **Add, don't replace** - keep working features, improve weak ones
- **Document changes** - explain what was changed and why

### Think Multi-Genre:
- Big band is ONE of countless genres that need this level of detail
- Design solutions that can scale to: orchestral, chamber, electronic, world music
- Use configurable parameters, not hardcoded big band-specific values
- Create reusable components (voice leading optimizer can work for strings, brass, vocals, etc.)

### Proven Results Required:
- **Cite your sources** - list papers, books, datasets used
- **Show examples** - provide specific musical examples that validate your approach
- **Quantitative metrics** - measure improvement (voice leading distance, melodic contour, etc.)
- **Qualitative validation** - "does it sound good?" tested against real recordings

---

## 20 SPECIALIZED AGENTS

### AGENT 1: BEBOP MELODY ARCHITECT

**Objective**: Transform the BebopMelodyGenerator from basic scale-based note generation into a sophisticated melody engine that produces musically compelling bebop lines with authentic vocabulary, phrasing, and contour.

**Current Limitations** (`genres/jazz.py:393-471`):
- No phrase shaping or motivic development
- No rhythmic variation patterns
- Fixed register (C4-C6), no harmonic context adaptation
- Missing bebop vocabulary patterns (II-V licks, enclosures, turns)
- No call-and-response phrasing
- No dynamic contour (crescendo/diminuendo)

**Research Sources**:
1. **Charlie Parker transcriptions**:
   - Analyze licks from jazzguitar.be/blog/charlie-parker
   - Study 18 bebop licks from Richie Zellon's collection
   - Extract patterns from Jens Larsen's bebop vocabulary lessons

2. **Bebop theory**:
   - Mark Levine "Jazz Theory Book" - Chapter on bebop scales (pages on chromatic approach)
   - Study "minorization" technique (staying on ii instead of V)
   - Analyze dim7 arpeggio from b9 technique
   - Chromatic enclosure patterns (half-step above, two chromatic below, resolve)

3. **PiJAMA Dataset analysis**:
   - Extract melodic contour patterns from 200+ hours of jazz piano
   - Measure phrase lengths, rest patterns, registral distribution
   - Analyze how professionals shape melodies over chord changes

4. **Academic research**:
   - Max Stehr - "Bird's words and Lennie's lessons: Using or avoiding patterns in bebop"
   - Study which patterns are overused vs. authentically varied

**Deliverables**:
1. **Bebop Vocabulary Library** (`genres/jazz_vocabulary.py`):
   ```python
   class BebopVocabulary:
       @staticmethod
       def get_ii_V_I_licks(key: int, difficulty: str) -> List[List[JazzNote]]

       @staticmethod
       def get_chromatic_enclosure(target_note: int, approach_style: str) -> List[JazzNote]

       @staticmethod
       def get_turnaround_lick(key: int, style: str) -> List[JazzNote]
   ```
   - Minimum 50 authentic II-V-I licks in all keys
   - 25+ chromatic enclosure patterns
   - 20+ turnaround vocabulary patterns
   - Categorized by difficulty, style era (swing, bebop, post-bop)

2. **Enhanced BebopMelodyGenerator** (`genres/jazz.py`):
   ```python
   def generate_phrase(self, chord: JazzChord, length_beats: int = 4,
                       use_vocabulary: bool = True,
                       phrase_shape: str = "arch",  # arch, ascending, descending, peak_early
                       rhythmic_density_curve: List[float] = None,
                       target_register: Tuple[int, int] = None) -> List[JazzNote]
   ```
   - Phrase shaping with configurable contour
   - Vocabulary integration (use licks 30-50% of the time)
   - Rhythmic variation (not just uniform subdivision)
   - Register adaptation based on harmonic context
   - Rest patterns for natural phrasing

3. **Validation**:
   - Generate 10 bebop melodies over II-V-I progressions
   - Compare melodic intervals, rhythm complexity, and phrase shapes to Charlie Parker transcriptions
   - Metric: Voice leading distance, phrase length variance, rest distribution should match Parker within 15%

**Integration Points**:
- Used by BigBandArranger._create_lead()
- Used by generate_professional.py._generate_melody()
- Scalable to: melodic minor jazz, modal jazz, contemporary jazz, ANY melodic instrument

---

### AGENT 2: SAX SOLI VOICING MASTER

**Objective**: Fix the sax soli voicing algorithm to use proper drop-2, drop-3, and spread voicings instead of only clustered close position, with voice leading optimization between chords.

**Current Limitations** (`transformation/arrangement_engine.py:142-163`):
- Only close position voicings (all voices within octave) - sounds muddy
- No drop voicings (drop-2, drop-3 are ESSENTIAL for big band)
- No voice leading optimization - large leaps between chords
- No register-specific spacing (wider in bass, closer in treble)
- Voices strictly ascending (no crossing for smooth voice leading)

**Research Sources**:
1. **Big band voicing theory**:
   - Evan Rogers "Big Band Arranging | Voicings" (evanrogersmusic.com)
   - Study drop-2 voicing: drop 2nd voice from top down an octave
   - Study drop-3 voicing: drop 3rd voice from top down an octave
   - Study drop-2-4 voicing: drop 2nd AND 4th voices
   - Frans Absil "Arranging by Examples" - sax soli section

2. **Voice leading principles**:
   - Mark Levine "Jazz Theory Book" - voice leading chapter
   - Matthew Keating (2023) LSTM voice-leading paper - study distance minimization
   - Neo-Riemannian voice leading (already in `core/neo_riemannian.py`) - adapt for jazz

3. **Professional score analysis**:
   - Analyze 20+ Thad Jones sax soli arrangements (livingjazzarchives.org)
   - Study Count Basie sax section voicings (ejazzlines.com transcriptions)
   - Measure: voice spacing, voice movement, doubling rules

4. **MIDI dataset analysis**:
   - Extract sax voicings from Lakh MIDI big band files
   - Measure spacing between voices at different registers
   - Analyze when drop-2 vs drop-3 vs close is used

**Deliverables**:
1. **Professional Sax Voicing Engine** (`transformation/sax_voicing.py`):
   ```python
   class SaxSoliVoicing:
       @staticmethod
       def voice_melody(melody: List[NoteEvent],
                       chords: List[ChordEvent],
                       voicing_style: str = "drop_2",  # close, drop_2, drop_3, drop_2_4, spread
                       optimize_voice_leading: bool = True,
                       section: List[str] = ["alto1", "alto2", "tenor1", "tenor2", "bari"]) -> Dict[str, List[NoteEvent]]
   ```

2. **Voice Leading Optimizer** (`transformation/voice_leading_optimizer.py`):
   ```python
   class VoiceLeadingOptimizer:
       @staticmethod
       def minimize_motion(chord_sequence: List[List[int]]) -> List[List[int]]:
           """Find inversions that minimize total voice movement"""

       @staticmethod
       def calculate_voice_leading_distance(voicing1, voicing2) -> float:
           """Measure total semitone movement between voicings"""

       @staticmethod
       def apply_smoothing(voicings: List[List[int]],
                          max_leap: int = 7) -> List[List[int]]:
           """Re-voice to avoid large leaps"""
   ```

3. **Voicing Style Implementations**:
   - **Close**: All voices within octave (current implementation - keep as option)
   - **Drop-2**: Drop 2nd voice from top down octave (MOST COMMON big band voicing)
   - **Drop-3**: Drop 3rd voice from top down octave
   - **Drop-2-4**: Drop 2nd and 4th voices (open, powerful sound)
   - **Spread**: Wide spacing throughout (modern sound)

4. **Register-Specific Spacing**:
   - Below C4 (middle C): minimum 4-semitone spacing (avoid mud)
   - C4-C5: 3-4 semitone spacing
   - Above C5: 2-3 semitone spacing (close is OK in high register)

5. **Validation**:
   - Generate sax soli over 32-bar AABA form
   - Compare voice spacing and voice movement to Thad Jones "The Deacon"
   - Metrics:
     - Average voice movement per chord change: < 3 semitones (professional standard)
     - Voice spacing in bass register: > 3 semitones
     - Use of drop-2 voicing: > 60% of chords (industry standard)

**Integration Points**:
- Replace BigBandArranger._harmonize_saxes()
- Scalable to: brass soli, string sections, vocal harmony, woodwind ensembles

---

### AGENT 3: PIANO COMPING VIRTUOSO

**Objective**: Implement authentic jazz piano comping with stride patterns, rootless voicings, rhythmic variation, and style-specific patterns (Bill Evans, McCoy Tyner, Red Garland).

**Current Limitations** (`genres/jazz.py:477-535`):
- No stride piano (mentioned in enum but NOT IMPLEMENTED)
- No comping rhythm patterns (only provides pitches, no timing variations)
- No upper structure tensions (9, 11, 13)
- No two-handed voicings
- Limited to 4 static styles (shell, rootless, quartal, block)

**Research Sources**:
1. **Stride piano study**:
   - Study James P. Johnson, Fats Waller, Art Tatum recordings
   - Pattern: Bass note on 1 & 3, chord on 2 & 4 (oom-pah, oom-pah)
   - Extract from PiJAMA dataset (200+ hours) - how often is stride used? What tempos?

2. **Bill Evans rootless voicings**:
   - Mark Levine "Jazz Piano Book" - rootless voicing chapter
   - Type A: 3-5-7-9 (C7 = E-G-Bb-D)
   - Type B: 7-9-3-5 (C7 = Bb-D-E-G)
   - Study when to use A vs B (voice leading determines choice)

3. **McCoy Tyner quartal voicings**:
   - Stacked perfect 4ths: C-F-Bb-Eb-Ab
   - Used over modal vamps, Coltrane changes
   - Extract from "A Love Supreme" transcriptions

4. **Comping rhythms**:
   - Study Red Garland, Wynton Kelly comping patterns
   - Charleston rhythm (syncopated) vs. on-beat comping
   - Extract rhythm patterns from PiJAMA dataset - quantize and categorize

5. **Upper structures**:
   - Mark Levine - upper structure triads chapter
   - C7alt = Db major triad over C bass (gives b9, #9, #11)
   - Catalog 20+ upper structure combinations

**Deliverables**:
1. **Stride Piano Generator** (`genres/stride_piano.py`):
   ```python
   class StridePianoGenerator:
       @staticmethod
       def generate_stride_pattern(chord: JazzChord,
                                  tempo: int,
                                  bars: int = 4,
                                  left_hand_pattern: str = "alternating_bass",  # alternating, walking, tenths
                                  right_hand_density: float = 0.6) -> List[JazzNote]:
           """
           Generate authentic stride piano
           Left hand: Bass note (1&3) + chord (2&4)
           Right hand: Melody, fills, runs
           """
   ```

2. **Enhanced PianoComping** (`genres/jazz.py`):
   ```python
   class CompingStyle(Enum):
       SHELL = "shell"
       ROOTLESS_A = "rootless_a"  # NEW: 3-5-7-9
       ROOTLESS_B = "rootless_b"  # NEW: 7-9-3-5
       QUARTAL = "quartal"
       BLOCK = "block"
       STRIDE = "stride"  # NEW: Actually implemented
       UPPER_STRUCTURES = "upper_structures"  # NEW

   def comp_pattern(self,
                   chords: List[JazzChord],
                   rhythm_pattern: str = "charleston",  # charleston, on_beat, sparse, dense
                   voicing_style: CompingStyle = ROOTLESS_A,
                   use_voice_leading: bool = True) -> List[JazzNote]:
       """Generate comping with RHYTHM + VOICING"""
   ```

3. **Comping Rhythm Library** (`genres/comping_rhythms.py`):
   ```python
   CHARLESTON_PATTERN = [0.25, 0.75, 1.25, 1.75, ...]  # Offbeats
   MONTUNO_PATTERN = [0, 0.5, 1.5, 2, 2.5, ...]  # Latin
   SPARSE_PATTERN = [0.75, 2.75, ...]  # Minimal
   DENSE_PATTERN = [0, 0.5, 1, 1.5, ...]  # Busy

   # Extract 20+ authentic patterns from PiJAMA dataset
   ```

4. **Upper Structure Catalog**:
   - Document 20+ upper structure triads for altered dominants
   - Implement automatic upper structure selection for V7alt chords

5. **Validation**:
   - Generate stride piano at 120 BPM over 12-bar blues
   - Compare left-hand pattern timing to James P. Johnson recordings
   - Generate Bill Evans-style comping over "Autumn Leaves"
   - Metric: Rhythm pattern accuracy within 50ms of authentic recordings

**Integration Points**:
- Replace BigBandArranger._create_piano_comping()
- Scalable to: solo piano, piano trio, any keyboard instrument (organ, electric piano)

---

### AGENT 4: HARMONIC PROGRESSION DESIGNER

**Objective**: Create genre-specific harmonic progression generators for bebop, post-bop, modal, and blues styles with authentic harmonic rhythm, reharmonization, and substitute chord algorithms.

**Current State**:
- ComprehensiveHarmonyGenerator has 31+ progression types (GOOD)
- But progressions are STATIC - no reharmonization or variation
- No harmonic rhythm control (one chord per bar is limiting)
- No diatonic/tritone substitution engine
- No tension/release analysis

**Research Sources**:
1. **Mark Levine "Jazz Theory Book"**:
   - Reharmonization techniques chapter
   - Tritone substitution (bII7 for V7)
   - Diatonic substitution (iii for I, vi for I)
   - Approach chords (ii-V before any target chord)
   - Pedal points, sus chords, modal interchange

2. **Bebop harmonic practices**:
   - Study Charlie Parker "Ko-Ko" changes (Rhythm changes with chromatic approaches)
   - Analyze Dizzy Gillespie "Night in Tunisia" (exotic modulations)
   - Extract from Weimar Jazz Database (300 transcriptions with chord changes)

3. **Post-bop harmony**:
   - John Coltrane "Giant Steps" analysis (major 3rd cycles)
   - Wayne Shorter "Speak No Evil" (ambiguous tonality)
   - Herbie Hancock "Dolphin Dance" (modal/tonal mixture)

4. **Modal progressions**:
   - Miles Davis "So What" (Dorian vamp)
   - McCoy Tyner "Passion Dance" (Dorian with pedal)
   - Study when to use static harmony vs. traditional changes

**Deliverables**:
1. **Reharmonization Engine** (`generators/reharmonization_engine.py`):
   ```python
   class ReharmonizationEngine:
       @staticmethod
       def apply_tritone_subs(progression: List[JazzChord],
                             probability: float = 0.3) -> List[JazzChord]:
           """Replace V7 chords with bII7 substitutes"""

       @staticmethod
       def add_approach_chords(progression: List[JazzChord],
                              approach_type: str = "ii_V") -> List[JazzChord]:
           """Add ii-V before target chords"""

       @staticmethod
       def apply_modal_interchange(progression: List[JazzChord],
                                  mode: str = "dorian") -> List[JazzChord]:
           """Borrow chords from parallel modes"""

       @staticmethod
       def generate_coltrane_substitution(target_chord: JazzChord) -> List[JazzChord]:
           """Generate descending major 3rd cycle (B→G→Eb→target)"""
   ```

2. **Harmonic Rhythm Controller** (`generators/harmonic_rhythm.py`):
   ```python
   class HarmonicRhythmEngine:
       @staticmethod
       def expand_progression(base_progression: List[JazzChord],
                             bars: int,
                             chords_per_bar: float = 1.0,  # Can be 0.5, 1, 2, 4
                             rhythm_pattern: str = "standard") -> List[ChordEvent]:
           """
           Create chord events with specific timing
           rhythm_pattern options:
           - standard: one chord per bar
           - fast: 2 chords per bar (ii-V, I-IV, etc.)
           - slow: one chord per 2 bars (ballads)
           - mixed: varying (bebop style)
           """
   ```

3. **Style-Specific Generators** - Enhance ComprehensiveHarmonyGenerator:
   ```python
   def generate_bebop_progression(self, form: str = "rhythm_changes",
                                 reharmonization_level: float = 0.5) -> List[JazzChord]:
       """
       Bebop: Heavy ii-V usage, chromatic approaches, tritone subs
       reharmonization_level: 0.0 (basic) to 1.0 (Bird-level complexity)
       """

   def generate_postbop_progression(self, style: str = "coltrane") -> List[JazzChord]:
       """
       Post-bop: Coltrane changes, Wayne Shorter ambiguity, modal sections
       """

   def generate_modal_progression(self, mode: str = "dorian",
                                 pedal_point: bool = True) -> List[JazzChord]:
       """
       Modal: Static or slow-moving harmony, pedal tones
       """
   ```

4. **Validation**:
   - Generate bebop progression over 32-bar AABA
   - Compare chord function distribution to Charlie Parker "Confirmation" changes
   - Metric: ii-V frequency, tritone sub usage, approach chord density should match bebop standards
   - Test: Can a jazz musician follow the changes? (playability test)

**Integration Points**:
- Used by generate_professional.py
- Feeds into all arranging modules
- Scalable to: any harmonic music (classical, film, pop with jazz influence)

---

### AGENT 5: BRASS SECTION ARRANGER

**Objective**: Transform brass writing from basic stabs to sophisticated section writing with sustained pads, calls-and-response, shout chorus, and authentic articulations.

**Current Limitations** (`transformation/arrangement_engine.py:166-190`):
- Only short stabs (0.25 beats) - no sustained brass
- No call-and-response with saxes
- No shout chorus (climactic arranged section)
- No articulation variety (accents, falls, doits)

**Research Sources**:
1. **Duke Ellington brass writing**:
   - Study "Ko-Ko" - plunger mutes, growls
   - "Caravan" - sustained brass pads
   - Analyze from livingjazzarchives.org scores
   - Extract: When does Ellington use unison vs harmony? What registers?

2. **Count Basie brass**:
   - "One O'Clock Jump" shout chorus analysis
   - Brass riffs behind sax solos (backgrounds)
   - Simple, powerful, rhythmic figures
   - Study from ejazzlines.com transcriptions

3. **Thad Jones brass**:
   - "A Child is Born" - lush brass harmony
   - "Three and One" - angular brass lines
   - Modern voicings, wider intervals

4. **Brass technique resources**:
   - Study brass ranges: Trumpet (written C4-C6 comfortable), Trombone (E2-Bb4)
   - Articulation types: accents, staccato, legato, falls (down), doits (up), shakes
   - Mute types: straight, cup, harmon, plunger (Ellington signature)

**Deliverables**:
1. **Brass Arranger Module** (`transformation/brass_arranger.py`):
   ```python
   class BrassArranger:
       @staticmethod
       def create_sustained_pad(chords: List[ChordEvent],
                               voicing_type: str = "drop_2",
                               dynamic_shape: str = "crescendo") -> List[NoteEvent]:
           """Long tones, background pad"""

       @staticmethod
       def create_shout_chorus(melody: List[NoteEvent],
                              chords: List[ChordEvent],
                              intensity: float = 0.9) -> Dict[str, List[NoteEvent]]:
           """
           Climactic section: full band in unison or block harmony
           Typically final A section in AABA form
           """

       @staticmethod
       def create_brass_riff(chord: JazzChord,
                            pattern_style: str = "basie_riff",
                            bars: int = 4) -> List[NoteEvent]:
           """
           Short rhythmic figures (backgrounds behind solos)
           Styles: basie_riff, ellington_call, thad_modern
           """

       @staticmethod
       def create_call_response(sax_phrase: List[NoteEvent],
                               response_delay: float = 4.0) -> List[NoteEvent]:
           """Brass responds to sax phrase (antiphonal writing)"""
   ```

2. **Brass Voicing Engine** - Enhance existing `generators/granular_control.py:529-607`:
   ```python
   class BrassVoicingEngine:
       # Already has: drop_2, unison, octaves, close
       # ADD:
       @staticmethod
       def spread_voicing(chord: JazzChord,
                         section: List[str] = ["trumpet1", "trumpet2", "trumpet3", "trumpet4",
                                              "trombone1", "trombone2", "trombone3", "trombone4"]) -> List[int]:
           """Wide spacing for powerful sound"""

       @staticmethod
       def section_blend(trumpets: List[NoteEvent],
                        trombones: List[NoteEvent],
                        blend_ratio: float = 0.5) -> Dict[str, List[NoteEvent]]:
           """Balance trumpets (bright) vs trombones (dark)"""
   ```

3. **Shout Chorus Builder**:
   - Detect AABA form final A section
   - Arrange in: unison (Basie), block harmony (Ellington), or spread voicing
   - Increase velocity 20% above rest of arrangement
   - Add accent articulations on strong beats

4. **Validation**:
   - Generate shout chorus for 32-bar AABA
   - Compare to Count Basie "April in Paris" shout chorus (famous ending)
   - Metrics:
     - Dynamic increase in final section: +20 velocity points
     - Voicing type matches style (unison for Basie, harmony for Ellington)
     - Call-response timing: 4-bar phrases alternate sax/brass

**Integration Points**:
- Replace BigBandArranger._create_brass_figures()
- Integrates with FormGenerator (detect shout chorus location)
- Scalable to: any brass ensemble (quintet, orchestra, marching band)

---

### AGENT 6: WALKING BASS ARCHITECT

**Objective**: Enhance walking bass from basic quarter-note root-fifth patterns to sophisticated bass lines with chromatic approaches, enclosures, and voice leading to chord tones.

**Current Implementation** (`transformation/arrangement_engine.py:216-241`):
- Basic pattern: Root (beat 1), 5th (beat 3), chromatic approach (beat 4)
- No voice leading to next chord's root
- No encircle patterns
- No scalar runs
- Limited to single octave (E1-G2)

**Research Sources**:
1. **Ray Brown walking bass study**:
   - Transcribe "Honeysuckle Rose" bass line
   - Analyze approach note choices (chromatic vs diatonic)
   - Study when Ray uses chord tones vs passing tones

2. **Paul Chambers bass lines**:
   - "So What" - modal walking
   - "Giant Steps" - handling fast changes
   - Extract from PiJAMA dataset (bass parts)

3. **Walking bass theory**:
   - Mark Levine "Jazz Theory Book" - walking bass chapter
   - Rules:
     1. Beat 1: Almost always chord root
     2. Beat 3: Usually 3rd or 5th
     3. Beat 2, 4: Approach tones to next strong beat
   - Approach types: chromatic (half-step below), scalar, encircle

4. **MIDI dataset analysis**:
   - Extract bass lines from Lakh MIDI jazz files
   - Measure: chord tone vs passing tone ratio, approach note frequency, octave range

**Deliverables**:
1. **Walking Bass Generator** (`transformation/walking_bass_generator.py`):
   ```python
   class WalkingBassGenerator:
       @staticmethod
       def generate_walking_line(chords: List[ChordEvent],
                                swing_feel: bool = True,
                                approach_style: str = "mixed",  # chromatic, diatonic, mixed
                                voice_leading: bool = True) -> List[NoteEvent]:
           """
           Generate professional walking bass

           Beat 1: Chord root (or 3rd/5th if voice-leading from previous bar)
           Beat 2: Approach to beat 3
           Beat 3: Chord tone (3rd, 5th, or 7th)
           Beat 4: Approach to next bar's root

           approach_style:
           - chromatic: half-step approaches
           - diatonic: scale-tone approaches
           - mixed: combination (most authentic)

           voice_leading: If True, optimize for smooth connection between chords
           """

       @staticmethod
       def generate_chromatic_approach(target_note: int,
                                      from_below: bool = None) -> int:
           """Generate chromatic approach (half-step above or below)"""

       @staticmethod
       def generate_encircle(target_note: int,
                            beats: int = 2) -> List[int]:
           """Encircle target: step above, step below, target"""

       @staticmethod
       def generate_scalar_run(start_note: int,
                              end_note: int,
                              scale: List[int],
                              beats: int) -> List[int]:
           """Scalar passage connecting two chord tones"""
   ```

2. **Voice Leading Optimization**:
   ```python
   @staticmethod
   def optimize_voice_leading_between_chords(chord1: ChordEvent,
                                            chord2: ChordEvent,
                                            last_note: int) -> int:
       """
       Choose best chord tone for chord2 based on proximity to last_note
       Example: If last_note = E2, and chord2 = Cmaj7
       Options: C1 (distance=4), E2 (distance=0), G2 (distance=3), B2 (distance=7)
       Choose E2 (nearest)
       """
   ```

3. **Octave Management**:
   - Expand range to E1-C3 (standard upright bass)
   - Stay in comfortable range (avoid extreme highs during fast changes)
   - Use higher register for climax sections, lower for intros

4. **Validation**:
   - Generate walking bass over ii-V-I in 12 keys
   - Compare to Ray Brown transcriptions
   - Metrics:
     - Beat 1 chord root frequency: >80% (professional standard)
     - Chromatic approach usage: 40-60% of beat 4 (authentic)
     - Smooth voice leading: average interval < 4 semitones per chord change

**Integration Points**:
- Replace BigBandArranger._create_walking_bass()
- Works with any chord progression
- Scalable to: upright bass, electric bass, tuba (low brass), any bass instrument

---

### AGENT 7: DRUM PATTERN & GROOVE SPECIALIST

**Objective**: Create authentic big band drum patterns across styles (swing, bebop, Latin) with fills, dynamic variation, and integration with groove template system.

**Current Limitations** (`transformation/arrangement_engine.py:268-311`):
- Only ride cymbal with swing 8ths and hi-hat on 2&4
- No kick drum, no fills
- No variation between sections (intro, verse, shout chorus, ending)
- No Latin patterns (Afro-Cuban, samba, bossa nova)
- Groove template system exists but is UNUSED

**Research Sources**:
1. **Big band drumming study**:
   - Buddy Rich "West Side Story" - bebop drum approach
   - Louie Bellson with Duke Ellington - classic swing
   - Mel Lewis with Thad Jones - modern jazz orchestral

2. **Pattern extraction**:
   - Use groove_library.py GrooveTemplateEngine to extract from recordings
   - Analyze "Purdie Shuffle", "Motown Backbeat" patterns already in library
   - Add big band swing patterns (feathered bass drum, brushes)

3. **Latin jazz patterns**:
   - Study "Manteca" (Dizzy Gillespie) - Afro-Cuban
   - "Samba de Uma Nota So" - Brazilian samba
   - Extract clave patterns (3-2 vs 2-3 son clave)

4. **Form-based dynamics**:
   - Study how drums change from: intro → A section → bridge → shout chorus → ending
   - Extract fill patterns at phrase endings (4-bar, 8-bar)

**Deliverables**:
1. **Big Band Drum Pattern Library** (`transformation/bigband_drums.py`):
   ```python
   class BigBandDrumPatterns:
       # Ride cymbal patterns
       SWING_RIDE = [0, 0.33, 0.66, 1, 1.33, 1.66, ...]  # Swing 8ths
       BEBOP_RIDE = [0, 0.25, 0.66, 1, 1.25, 1.66, ...]  # More syncopated

       # Hi-hat patterns
       HIHAT_2_4 = [1, 3]  # Backbeat only (classic)
       HIHAT_ALL_BEATS = [0, 1, 2, 3]  # Four on floor (modern)

       # Bass drum patterns
       FEATHERED_KICK = [0, 1, 2, 3]  # All four beats, soft (Count Basie style)
       BEBOP_KICK = [0, 1.5, 2.75, ...]  # Syncopated bombs

       # Latin patterns
       AFRO_CUBAN_BELL = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]  # Cowbell pattern
       SAMBA_SURDO = [0, 1.5, 2.5]  # Bass drum samba
       BOSSA_RIDE = [0, 0.66, 1.33, 2, 2.66, 3.33]  # Bossa nova ride

       @staticmethod
       def generate_fill(length_beats: int = 2,
                        intensity: float = 0.7,
                        target_instrument: str = "snare") -> List[NoteEvent]:
           """Generate drum fill at phrase ending"""
   ```

2. **Dynamic Drum Arranger** (`transformation/drum_arranger.py`):
   ```python
   class DrumArranger:
       @staticmethod
       def arrange_drums_for_form(form: MusicalForm,
                                 style: str = "swing",  # swing, bebop, latin_afro, latin_bossa
                                 dynamic_map: Dict[str, float] = None) -> List[NoteEvent]:
           """
           Generate drums with variation per section

           Example dynamic_map for AABA:
           {
               "intro": 0.3,      # Soft, sparse
               "A1": 0.5,         # Medium
               "A2": 0.5,
               "B": 0.7,          # Build on bridge
               "A3": 0.9,         # Shout chorus - loud!
               "ending": 0.4      # Soft ending
           }
           """

       @staticmethod
       def add_fills_at_phrase_endings(drums: List[NoteEvent],
                                       phrase_length_bars: int = 4) -> List[NoteEvent]:
           """Insert fills at bar 4, 8, 12, etc."""

       @staticmethod
       def apply_groove_template(drums: List[NoteEvent],
                                template: GrooveTemplate) -> List[NoteEvent]:
           """Apply authentic groove timing/velocity from template"""
   ```

3. **Integrate Existing Groove Library** - Use `algorithms/groove_library.py`:
   - Extract big band swing groove from Count Basie recordings
   - Store as GrooveTemplate
   - Apply to generated drum patterns for authentic feel

4. **Validation**:
   - Generate swing drums over 32-bar AABA
   - Compare ride cymbal timing to Mel Lewis recordings (measure swing ratio)
   - Add fills at 8-bar phrases
   - Metrics:
     - Swing ratio on ride: 0.62-0.67 (authentic big band)
     - Fill placement: every 4 or 8 bars
     - Dynamic variation between sections: minimum 20 velocity points difference

**Integration Points**:
- Replace BigBandArranger._create_swing_drums()
- Integrates with FormGenerator (section-based variation)
- Scalable to: any drum style (rock, funk, electronic), any percussion ensemble

---

### AGENT 8: ARTICULATION & EXPRESSION ENGINE

**Objective**: Implement realistic brass and woodwind articulations (falls, rips, growls, shakes, scoops, doits) and integrate into arrangement engine.

**Current State**:
- ArticulationEngine exists in codebase but is NOT INTEGRATED
- JazzNote has `articulation` field but only uses: "normal", "staccato", "accent", "ghost", "legato"
- Missing: falls (pitch bend down), doits (pitch up), rips (fast ascending gliss), shakes (trill)

**Research Sources**:
1. **Duke Ellington articulation study**:
   - "Concerto for Cootie" - plunger mute, growls, wa-wa effects
   - Transcribe pitch bend amounts (falls typically -200 to -400 cents)
   - Study Bubber Miley, Cootie Williams trumpet techniques

2. **MIDI pitch bend implementation**:
   - Study how to encode pitch bends in MIDI (pitch bend messages)
   - Rip: rapid pitch bend from -1200 cents to 0 over 200-400ms
   - Fall: pitch bend from 0 to -200 cents over 300-600ms
   - Shake: rapid alternation ±100 cents at 6-8 Hz

3. **Big band articulation conventions**:
   - Falls at phrase endings (especially final note)
   - Doits at strong beats before syncopated figures
   - Shakes on sustained notes (whole notes, half notes)
   - Rips into climactic notes (shout chorus entrances)

**Deliverables**:
1. **Enhanced Articulation System** (`transformation/articulation_engine.py`):
   ```python
   class ArticulationType(Enum):
       NORMAL = "normal"
       STACCATO = "staccato"
       ACCENT = "accent"
       LEGATO = "legato"
       FALL_SHORT = "fall_short"       # NEW: -200 cents, 300ms
       FALL_LONG = "fall_long"         # NEW: -400 cents, 600ms
       DOIT = "doit"                   # NEW: +200 cents, 200ms
       RIP = "rip"                     # NEW: -1200→0 cents, 400ms
       SHAKE = "shake"                 # NEW: ±100 cents @ 6Hz
       GROWL = "growl"                 # NEW: (distortion effect)
       SCOOP = "scoop"                 # NEW: -100→0 cents, 150ms

   class ArticulationEngine:
       @staticmethod
       def apply_articulation(note: NoteEvent,
                             articulation: ArticulationType) -> List[MidiMessage]:
           """
           Convert NoteEvent with articulation to MIDI messages

           Returns: [note_on, pitch_bend*, note_off]
           (* pitch_bend messages for falls, rips, etc.)
           """

       @staticmethod
       def suggest_articulations(notes: List[NoteEvent],
                                section: str = "brass",
                                style: str = "ellington") -> List[ArticulationType]:
           """
           Automatically suggest articulations based on note context

           Rules:
           - Phrase ending notes: FALL_SHORT (70% probability)
           - Whole notes: SHAKE (50% probability)
           - Shout chorus entrances: RIP (80% probability)
           - Short notes: STACCATO
           - Ellington style: More growls, plunger effects
           """
   ```

2. **MIDI Encoding for Pitch Bends**:
   ```python
   @staticmethod
   def encode_fall(note_on_tick: int,
                  duration_ticks: int,
                  fall_cents: int = -200,
                  fall_duration_ms: int = 300,
                  tempo_bpm: int = 120) -> List[Message]:
       """
       Generate MIDI pitch bend messages for fall

       Returns: [PitchBend(tick=note_on_tick, pitch=0),
                 PitchBend(tick=note_on_tick+X, pitch=-200),
                 PitchBend(tick=note_off_tick, pitch=0)]
       """
   ```

3. **Style-Specific Articulation Profiles**:
   ```python
   ELLINGTON_PROFILE = {
       "fall_probability": 0.6,
       "growl_probability": 0.4,
       "shake_probability": 0.3,
       "preferred_articulations": ["fall_long", "growl", "shake"]
   }

   BASIE_PROFILE = {
       "fall_probability": 0.3,
       "staccato_probability": 0.7,
       "shake_probability": 0.1,
       "preferred_articulations": ["staccato", "accent", "fall_short"]
   }
   ```

4. **Integration with Export**:
   - Modify MIDI export in `generate_professional.py` to handle pitch bend messages
   - Add pitch bend sensitivity setting (usually ±2 semitones)

5. **Validation**:
   - Generate brass section with articulations
   - Listen: Do falls sound natural? (smooth pitch descent)
   - Measure: Pitch bend amount matches theory (-200 cents for short fall)
   - Style test: Ellington arrangement has more growls/falls than Basie

**Integration Points**:
- Used by BrassArranger, SaxSoliVoicing
- Integrates into MIDI export pipeline
- Scalable to: any wind instrument (flute, clarinet, saxophone, trumpet, trombone, etc.)

---

### AGENT 9: DYNAMIC SHAPING & PHRASING MASTER

**Objective**: Add musical phrasing with crescendo, diminuendo, accent patterns, and breath marks to make arrangements sound human and musical.

**Current State**:
- No dynamic shaping whatsoever - all notes have static velocity
- No crescendo/diminuendo over phrases
- No accent patterns (strong-weak-medium-weak)
- No breath marks or phrase boundaries

**Research Sources**:
1. **Musical phrasing theory**:
   - Study classical and jazz phrasing principles
   - 4-bar phrase typically: strong start, build to bar 3, release in bar 4
   - 8-bar phrase: two 4-bar phrases, second louder than first
   - Analyze professional recordings with waveform visualization (see dynamic contour)

2. **Big band phrasing conventions**:
   - Shout chorus: loudest section (fff)
   - Bridge: often softer for contrast (mp)
   - Endings: diminuendo or sudden cutoff
   - Study from big band scores with dynamic markings

3. **MIDI velocity mapping**:
   - ppp (pianississimo): velocity 20-30
   - pp: 30-45
   - p: 45-60
   - mp: 60-75
   - mf: 75-90
   - f: 90-105
   - ff: 105-115
   - fff: 115-127

**Deliverables**:
1. **Dynamic Shaping Engine** (`transformation/dynamic_shaping.py`):
   ```python
   class DynamicShaping:
       @staticmethod
       def apply_phrase_contour(notes: List[NoteEvent],
                               phrase_length_bars: int = 4,
                               contour: str = "arch") -> List[NoteEvent]:
           """
           Apply dynamic contour to phrase

           Contour types:
           - arch: Start mp, crescendo to bar 3, diminuendo bar 4
           - build: Crescendo throughout
           - decay: Diminuendo throughout
           - terrace: Sudden dynamic shifts (Baroque style)

           Returns: Notes with modified velocity
           """

       @staticmethod
       def apply_crescendo(notes: List[NoteEvent],
                          start_velocity: int = 60,
                          end_velocity: int = 100) -> List[NoteEvent]:
           """Linear crescendo from start to end velocity"""

       @staticmethod
       def apply_accent_pattern(notes: List[NoteEvent],
                               pattern: str = "strong_weak") -> List[NoteEvent]:
           """
           Apply accent pattern
           - strong_weak: beats 1,3 louder than 2,4
           - syncopated: offbeats accented
           - none: even velocity
           """

       @staticmethod
       def mark_breath_points(notes: List[NoteEvent],
                             phrase_length_bars: int = 4) -> List[NoteEvent]:
           """
           Add gaps for breath marks
           Insert 0.1-0.2 beat rest at phrase boundaries
           """
   ```

2. **Form-Based Dynamic Map**:
   ```python
   def generate_dynamic_map_for_form(form: MusicalForm) -> Dict[str, int]:
       """
       Generate dynamic levels for each section

       Example AABA:
       {
           "A1": 75,    # mf
           "A2": 80,    # slightly louder
           "B": 70,     # bridge softer for contrast
           "A3": 100,   # shout chorus - f/ff
       }
       """
   ```

3. **Section-Level Dynamics**:
   - Intro: p to mp (soft start)
   - First A: mf (medium)
   - Second A: mf to f (build)
   - Bridge: mp (contrast)
   - Final A (shout): ff to fff (climax)
   - Ending: diminuendo to pp or sudden cutoff

4. **Validation**:
   - Generate arrangement and measure velocity variation
   - Check: Shout chorus velocity > other sections by minimum 20 points
   - Check: Phrases have arch contour (mid-point louder than start/end)
   - Listen: Does it sound musical and expressive?

**Integration Points**:
- Applied to all sections (lead, saxes, brass, rhythm)
- Integrates with FormGenerator (know where sections are)
- Scalable to: any musical arrangement, any genre

---

### AGENT 10: FORM STRUCTURE INTEGRATOR

**Objective**: Fully integrate FormGenerator with arrangement engine to create proper musical structures (intros, outros, modulations, key changes, form-specific arranging).

**Current State**:
- FormGenerator exists with 10+ form types (GOOD)
- But NOT integrated with BigBandArranger
- No intro/outro generation
- No modulation between sections
- No form-specific arranging (bridge should sound different from A section)

**Research Sources**:
1. **Big band forms**:
   - Study 100 jazz standards from Real Book
   - Catalog forms: AABA (70%), ABAC (15%), Blues (10%), Verse-Chorus (5%)
   - Measure section lengths: A = 8 bars (most common), 16 bars, 32 bars

2. **Intro/ending conventions**:
   - Common intros: last 4 bars of progression, vamp on I, Count Basie "button" (short punchy intro)
   - Common endings: tag (repeat last 4 bars), ritardando, fermata on final chord, Basie "button" (short punchy ending)

3. **Modulation techniques**:
   - Key change up half-step for final chorus (excitement)
   - Bridge modulation to IV or bVI (common)
   - Study modulation in "Mack the Knife" (up half-step), "I Wish" (up whole-step)

**Deliverables**:
1. **Intro/Outro Generator** (`generators/intro_outro_generator.py`):
   ```python
   class IntroOutroGenerator:
       @staticmethod
       def generate_intro(progression: List[JazzChord],
                         style: str = "vamp",  # vamp, last_4, button, rubato
                         length_bars: int = 4) -> Dict:
           """
           Generate intro section

           Styles:
           - vamp: Repeat I chord with rhythm section figures
           - last_4: Last 4 bars of progression
           - button: Short punchy hit (Basie style)
           - rubato: Free tempo piano intro

           Returns: {
               'intro_progression': List[JazzChord],
               'intro_arrangement': Dict[str, List[NoteEvent]]
           }
           """

       @staticmethod
       def generate_ending(progression: List[JazzChord],
                          style: str = "tag",  # tag, fermata, ritardando, button
                          length_bars: int = 4) -> Dict:
           """
           Generate ending section

           Styles:
           - tag: Repeat last 4 bars, often with ritardando
           - fermata: Sustain final chord
           - ritardando: Slow down gradually
           - button: Short punchy ending (Basie style)
           """
   ```

2. **Form-Aware Arranger** - Enhance BigBandArranger:
   ```python
   @staticmethod
   def arrange_with_form(melody: List[NoteEvent],
                        chords: List[ChordEvent],
                        form: MusicalForm) -> Dict[str, List[NoteEvent]]:
       """
       Arrange with form awareness

       - A sections: Full arrangement (melody + harmony)
       - Bridge: Different texture (e.g., brass only, softer dynamics)
       - Shout chorus: Unison or block harmony, louder
       - Apply intro/outro
       - Apply section-specific dynamics
       """
   ```

3. **Bridge Differentiation**:
   ```python
   @staticmethod
   def arrange_bridge_section(melody: List[NoteEvent],
                             chords: List[ChordEvent],
                             contrast_style: str = "brass_only") -> Dict:
       """
       Make bridge sound different from A sections

       Contrast techniques:
       - brass_only: Brass plays melody, saxes rest
       - sax_only: Reverse
       - softer: Reduce velocity 20%
       - different_voicing: Use spread instead of close
       - modulation: Bridge in different key
       """
   ```

4. **Modulation Implementation**:
   ```python
   @staticmethod
   def apply_modulation(progression: List[JazzChord],
                       from_key: int,
                       to_key: int,
                       modulation_bar: int) -> List[JazzChord]:
       """
       Insert modulation (key change)

       Common modulation points:
       - Before final chorus (up half-step or whole-step)
       - At bridge (to IV, bVI, or relative minor)
       """
   ```

5. **Validation**:
   - Generate 32-bar AABA arrangement with intro and ending
   - Check: Intro is 4 bars, uses vamp or last-4 style
   - Check: Bridge has different texture than A sections
   - Check: Ending has tag or ritardando
   - Listen: Does form sound complete and professional?

**Integration Points**:
- Core integration with BigBandArranger
- Uses IntroOutroGenerator, DynamicShaping
- Scalable to: any musical form, any genre

---

### AGENT 11: VOICE LEADING OPTIMIZATION ENGINE

**Objective**: Create a universal voice leading optimizer that minimizes voice movement between chords while respecting range constraints - usable across sax, brass, strings, vocals, etc.

**Current State**:
- Neo-Riemannian VoiceLeadingAnalyzer exists (calculates distance)
- But no optimizer that finds BEST inversions/voicings
- Voicings have large jumps between chords (sounds unmusical)

**Research Sources**:
1. **Matthew Keating (2023) paper**:
   - "An Algorithmic Approach to Jazz Guitar Voice-Leading Chord Fingerings"
   - LSTM encoder-decoder for sequence-to-sequence generation
   - Study distance minimization algorithm

2. **Classical voice leading rules**:
   - Common tone retention (keep common notes in same voice)
   - Contrary motion (outer voices move in opposite directions)
   - Avoid parallel 5ths and octaves
   - Minimize total voice movement

3. **Jazz voice leading** (Mark Levine):
   - Similar to classical but more relaxed
   - Parallel 5ths acceptable in jazz
   - Goal: smooth, singable voice lines

4. **Optimization algorithms**:
   - Study dynamic programming for voice leading (find optimal path through voicing space)
   - Study k-nearest neighbors for finding closest voicing

**Deliverables**:
1. **Universal Voice Leading Optimizer** (`transformation/voice_leading_optimizer.py`):
   ```python
   class VoiceLeadingOptimizer:
       @staticmethod
       def optimize_chord_sequence(chord_sequence: List[JazzChord],
                                  num_voices: int = 4,
                                  voice_ranges: List[Tuple[int, int]] = None,
                                  voicing_types: List[str] = ["close", "drop_2"],
                                  minimize: str = "total_motion") -> List[List[int]]:
           """
           Find optimal voicings for chord sequence

           Algorithm:
           1. Generate all possible voicings for each chord (within ranges)
           2. Build graph: nodes = voicings, edges = voice leading distance
           3. Find shortest path through graph (dynamic programming)

           minimize options:
           - total_motion: minimize sum of all voice movements
           - max_leap: minimize largest single voice leap
           - weighted: prefer small movements in outer voices

           Returns: List of voicings (each voicing is list of MIDI pitches)
           """

       @staticmethod
       def calculate_voice_leading_distance(voicing1: List[int],
                                           voicing2: List[int],
                                           weights: List[float] = None) -> float:
           """
           Measure distance between two voicings

           Weighted distance: sum(weight[i] * abs(v1[i] - v2[i]))
           Default weights: [1.0, 1.0, 1.0, 1.0] (equal)
           Outer voice emphasis: [1.5, 1.0, 1.0, 1.5] (penalize bass/soprano leaps)
           """

       @staticmethod
       def generate_all_voicings(chord: JazzChord,
                                num_voices: int,
                                voice_ranges: List[Tuple[int, int]],
                                voicing_type: str = "close") -> List[List[int]]:
           """
           Generate all valid voicings for chord within range constraints

           Returns: List of voicings (may be 100+ options per chord)
           """

       @staticmethod
       def apply_common_tone_retention(voicing1: List[int],
                                      voicing2_options: List[List[int]]) -> List[int]:
           """
           From voicing2_options, choose one that maximizes common tones with voicing1
           Common tone retention: if note appears in both chords, keep it in same voice
           """
   ```

2. **Dynamic Programming Implementation**:
   ```python
   @staticmethod
   def find_optimal_path_dp(chord_sequence: List[JazzChord],
                           voicing_options: List[List[List[int]]]) -> List[List[int]]:
       """
       Dynamic programming to find optimal voice leading path

       DP table: dp[chord_idx][voicing_idx] = min cost to reach this voicing
       Backtrack to reconstruct optimal path

       Time complexity: O(n * m^2) where n=chords, m=voicings per chord
       """
   ```

3. **Integration with Existing Voicers**:
   - Use in SaxSoliVoicing
   - Use in BrassVoicing
   - Use in PianoComping
   - Universal algorithm works for any number of voices

4. **Validation**:
   - Generate voicing sequence for ii-V-I progression
   - Measure total voice movement
   - Compare to manual voicings from Mark Levine textbook
   - Metric: Total movement should be < 50% of unoptimized voicings
   - Test: All voices stay within specified ranges

**Integration Points**:
- Used by ALL voicing modules (sax, brass, piano, strings)
- Core algorithm for smooth harmony
- Scalable to: any multi-voice harmony (SATB choir, string quartet, brass quintet, etc.)

---

### AGENT 12: SWING FEEL CALIBRATION SPECIALIST

**Objective**: Enhance swing timing beyond simple ratio to include 16th-note swing, microtime variation, laid-back/rushing feel, and groove template integration.

**Current State** (`genres/jazz.py:541-591`):
- SwingTiming applies 0.62 ratio to 8th notes (GOOD)
- But only 8th notes - no 16th swing
- No microtiming variation (all offbeats delayed exactly the same)
- No groove template application
- No laid-back vs rushing control

**Research Sources**:
1. **Swing ratio research**:
   - Study various swing ratios: 0.56 (light), 0.62 (medium), 0.67 (heavy triplet feel)
   - Measure swing ratio from PiJAMA dataset (200+ hours jazz piano)
   - Does ratio change with tempo? (hypothesis: lighter swing at fast tempos)

2. **16th-note swing**:
   - Modern jazz uses swing on 16th notes, not just 8ths
   - Study Robert Glasper, Brad Mehldau recordings
   - Extract 16th-note timing from MIDI dataset

3. **Microtiming studies**:
   - Research on "groove" - slight timing deviations create feel
   - Study Roger Linn MPC swing algorithm (already mentioned in code)
   - Extract timing variance from real recordings (standard deviation of offbeat placement)

4. **Laid-back vs rushing**:
   - Laid-back: notes slightly late (cool, relaxed feel)
   - Rushing: notes slightly early (energetic, intense feel)
   - Measure timing in Miles Davis (laid-back) vs Buddy Rich (rushing)

**Deliverables**:
1. **Enhanced SwingTiming** (`genres/jazz.py`):
   ```python
   class SwingTiming:
       @staticmethod
       def apply_swing(notes: List[JazzNote],
                      swing_ratio: float = 0.62,
                      intensity: float = 1.0,
                      subdivision: str = "8th",  # NEW: 8th, 16th, mixed
                      microtiming_variance: float = 0.02,  # NEW: timing variation
                      feel: str = "neutral") -> List[JazzNote]:  # NEW: neutral, laid_back, rushing
           """
           Enhanced swing with:
           - 8th and 16th note swing
           - Microtiming variation (slight randomness)
           - Laid-back/rushing feel

           microtiming_variance: Standard deviation of timing offset (in beats)
           - 0.0: perfect quantization
           - 0.02: subtle variation (human)
           - 0.05: noticeable variation (drunk)

           feel:
           - neutral: swing ratio applied as-is
           - laid_back: additional +0.01 to +0.03 beat delay
           - rushing: -0.01 to -0.03 beat early
           """

       @staticmethod
       def apply_16th_swing(notes: List[JazzNote],
                           swing_ratio: float = 0.58) -> List[JazzNote]:
           """
           Apply swing to 16th notes
           Find notes at beat positions: 0.25, 0.75, 1.25, 1.75 (2nd and 4th 16ths)
           Delay them by swing ratio
           """

       @staticmethod
       def apply_groove_template(notes: List[JazzNote],
                                template: GrooveTemplate) -> List[JazzNote]:
           """
           Apply authentic groove timing from template
           Uses groove_library.py GrooveTemplate
           """
   ```

2. **Tempo-Adaptive Swing**:
   ```python
   @staticmethod
   def calculate_adaptive_swing_ratio(tempo: int) -> float:
       """
       Adjust swing ratio based on tempo

       Research-based formula:
       - Slow tempos (60-100 BPM): 0.65-0.67 (heavier swing)
       - Medium tempos (100-160 BPM): 0.62-0.64 (medium swing)
       - Fast tempos (160-300 BPM): 0.56-0.60 (lighter swing)

       Rationale: At fast tempos, heavy swing is too sluggish
       """
   ```

3. **Groove Template Integration** - Use existing `algorithms/groove_library.py`:
   - Extract swing groove from Count Basie recording
   - Apply to generated drums, bass, piano
   - Measure timing offsets at grid positions

4. **Validation**:
   - Generate swing pattern at 140 BPM with 0.62 ratio
   - Measure actual offbeat timing (should be ~0.62 ± 0.02 with variance)
   - Generate 16th-note swing pattern
   - Compare to Brad Mehldau recording (modern jazz pianist)
   - Metrics:
     - Swing ratio accuracy: ±0.02 of target
     - Microtiming variance: matches human recordings (σ ≈ 0.02 beats)

**Integration Points**:
- Used by all modules (melody, bass, drums, piano)
- Integrates with GrooveTemplateEngine
- Scalable to: any swing-based music (shuffle, swing, New Orleans 2nd line, etc.)

---

### AGENT 13: DUKE ELLINGTON STYLE ANALYZER

**Objective**: Analyze Duke Ellington's arranging style and create a style profile that can generate Ellington-esque arrangements.

**Research Sources**:
1. **Ellington scores**:
   - Study "Ko-Ko", "Caravan", "Mood Indigo", "Concerto for Cootie"
   - Available from livingjazzarchives.org, ejazzlines.com
   - Analyze: voicing choices, orchestration, form, harmony

2. **Ellington techniques**:
   - **Plunger mute brass** - signature sound (Bubber Miley, Cootie Williams)
   - **Exotic harmonies** - whole tone, diminished, bitonal
   - **Voice as instrument** - unusual doublings (clarinet + muted trombone)
   - **Jungle sounds** - growls, wa-wa effects
   - **Rich harmony** - extended chords (9ths, 11ths, 13ths)

3. **Academic analysis**:
   - Mark Tucker "The Duke Ellington Reader"
   - Gunther Schuller "The Swing Era" - chapter on Ellington
   - Study how Ellington orchestrated vs. Basie (complex vs. simple)

**Deliverables**:
1. **Ellington Style Profile** (`styles/ellington_profile.py`):
   ```python
   ELLINGTON_STYLE = {
       # Orchestration
       "voicing_preference": "close_with_doublings",  # Unique doublings
       "voicing_spacing": "varied",  # Not consistent spacing
       "use_plunger_mutes": 0.6,  # 60% of brass notes
       "use_growls": 0.4,

       # Harmony
       "harmony_complexity": 0.9,  # Very complex
       "use_whole_tone": 0.3,  # Moderate
       "use_diminished": 0.4,
       "use_bitonal": 0.2,  # Occasional
       "chord_extensions": [9, 11, 13],  # Rich harmony

       # Articulations
       "articulation_variety": 0.8,  # High variety
       "fall_probability": 0.6,
       "shake_probability": 0.3,

       # Dynamics
       "dynamic_range": "wide",  # ppp to fff
       "use_crescendo": 0.7,

       # Form
       "intro_style": "rubato",  # Often free intro
       "ending_style": "fermata",  # Sustained ending

       # Texture
       "texture_density": 0.8,  # Rich, full
       "unusual_doublings": True,  # Signature
   }
   ```

2. **Ellington Arranger** (`styles/ellington_arranger.py`):
   ```python
   class EllingtonArranger:
       @staticmethod
       def arrange_in_ellington_style(melody: List[NoteEvent],
                                      chords: List[ChordEvent],
                                      form: MusicalForm) -> Dict[str, List[NoteEvent]]:
           """
           Arrange using Ellington techniques

           - Apply exotic harmonies (reharmonize with whole tone, diminished)
           - Use plunger mutes and growls
           - Create unusual doublings (clarinet + muted trombone)
           - Rich voicings with extensions
           - Wide dynamic range
           """
   ```

3. **Validation**:
   - Generate arrangement in Ellington style
   - Compare to "Caravan" score
   - Check: Plunger mutes used (articulation markers)
   - Check: Rich harmony with 9ths, 11ths, 13ths
   - Listening test: Does it evoke Ellington sound?

**Integration Points**:
- Used as style option in generate_professional.py
- `--style ellington` flag applies this profile
- Demonstrates style-based generation framework

---

### AGENT 14: COUNT BASIE STYLE ANALYZER

**Objective**: Analyze Count Basie's arranging style and create a style profile for generating Basie-esque arrangements.

**Research Sources**:
1. **Basie scores**:
   - Study "One O'Clock Jump", "April in Paris", "Li'l Darlin'", "Corner Pocket"
   - From ejazzlines.com transcriptions
   - Analyze: simplicity, riff-based, rhythm section interaction

2. **Basie techniques**:
   - **"Head" arrangements** - simple, loosely organized, easy to customize
   - **Riff-based** - short repeated figures
   - **Sparse piano** - minimalist comping (opposite of stride)
   - **Powerful rhythm section** - Freddie Green guitar (4-to-the-bar), Page/Brown bass
   - **Section hits** - punchy brass/sax stabs
   - **Shout chorus** - famous climactic sections
   - **"Button" endings** - short, punchy

3. **Basie vs. Ellington contrast**:
   - Basie: Simple, riff-based, rhythm section driven
   - Ellington: Complex, orchestral colors, exotic harmony
   - Study how arrangement density differs

**Deliverables**:
1. **Basie Style Profile** (`styles/basie_profile.py`):
   ```python
   BASIE_STYLE = {
       # Orchestration
       "voicing_preference": "unison_and_octaves",  # Simple
       "voicing_spacing": "open",  # Spread voicings
       "use_section_hits": 0.9,  # Very high - signature
       "use_riffs": 0.8,

       # Harmony
       "harmony_complexity": 0.3,  # Simple, functional
       "use_blues": 0.7,  # Blues-based
       "chord_extensions": [7],  # Basic 7th chords, not 9/11/13

       # Piano
       "piano_style": "sparse",  # Minimalist comping
       "piano_density": 0.2,  # Very sparse

       # Rhythm Section
       "emphasis_on_rhythm": 0.9,  # Rhythm section is star
       "feathered_kick": True,  # All four beats, soft
       "freddie_green_guitar": True,  # 4-to-the-bar

       # Articulations
       "articulation_variety": 0.4,  # Lower than Ellington
       "staccato_probability": 0.7,  # Punchy, crisp

       # Dynamics
       "dynamic_range": "medium",  # Less extreme than Ellington
       "shout_chorus_intensity": 1.0,  # Famous shout choruses

       # Form
       "intro_style": "vamp",  # Simple vamp
       "ending_style": "button",  # Short punchy ending

       # Texture
       "texture_density": 0.5,  # Sparser than Ellington
       "riff_based": True,  # Signature
   }
   ```

2. **Basie Arranger** (`styles/basie_arranger.py`):
   ```python
   class BasieArranger:
       @staticmethod
       def arrange_in_basie_style(melody: List[NoteEvent],
                                 chords: List[ChordEvent],
                                 form: MusicalForm) -> Dict[str, List[NoteEvent]]:
           """
           Arrange using Basie techniques

           - Simple, riff-based backgrounds
           - Punchy section hits
           - Sparse piano comping
           - Emphasis on rhythm section groove
           - Powerful shout chorus
           - Button intro/ending
           """

       @staticmethod
       def generate_basie_riff(chord: JazzChord,
                              bars: int = 2) -> List[NoteEvent]:
           """
           Generate simple, rhythmic riff
           Typically 1-2 bar repeated figure
           """
   ```

3. **Validation**:
   - Generate arrangement in Basie style
   - Compare to "One O'Clock Jump" score
   - Check: Simple harmony (7th chords, no 9/11/13)
   - Check: Section hits prominent
   - Check: Piano sparse (< 30% density)
   - Listening test: Does it have Basie swing and simplicity?

**Integration Points**:
- Used as style option: `--style basie`
- Contrasts with Ellington style
- Demonstrates range of big band aesthetics

---

### AGENT 15: MODERN BIG BAND STYLE ANALYZER (Thad Jones, Maria Schneider)

**Objective**: Analyze modern big band styles (Thad Jones, Maria Schneider, Gordon Goodwin) and create contemporary arranging profiles.

**Research Sources**:
1. **Thad Jones scores**:
   - "A Child is Born" - lush harmony
   - "Three and One" - angular, modern
   - Study from livingjazzarchives.org Thad Jones archive
   - Analyze: wider intervals, quartal harmony, rhythmic complexity

2. **Maria Schneider**:
   - "Concert in the Garden" - orchestral colors
   - Study orchestration techniques (woodwind doublings, unique timbres)
   - Cinematic, impressionistic approach

3. **Gordon Goodwin**:
   - "Hunting Wabbits" - contemporary swing
   - Fast tempos, complex rhythms, high energy
   - Modern studio production techniques

4. **Modern techniques**:
   - Quartal/quintal harmony (stacked 4ths/5ths)
   - Metric modulation
   - Odd meters (5/4, 7/4)
   - Contemporary jazz harmony (altered scales, symmetrical scales)

**Deliverables**:
1. **Modern Style Profiles** (`styles/modern_profiles.py`):
   ```python
   THAD_JONES_STYLE = {
       "voicing_preference": "quartal_and_clusters",
       "voicing_spacing": "wide_intervals",
       "harmony_complexity": 0.8,
       "use_quartal": 0.6,
       "rhythmic_complexity": 0.8,
       "angular_melodies": True,
       "use_odd_meters": 0.3,
   }

   MARIA_SCHNEIDER_STYLE = {
       "voicing_preference": "orchestral_colors",
       "woodwind_doublings": True,  # Flute, clarinet doubles
       "harmony_complexity": 0.9,
       "use_pedal_tones": 0.7,
       "impressionistic": True,
       "dynamic_range": "very_wide",
       "texture_density": "varied",  # From sparse to dense
   }

   GORDON_GOODWIN_STYLE = {
       "tempo_range": "fast",  # 180-240 BPM
       "rhythmic_complexity": 0.9,
       "energy_level": 1.0,
       "use_latin_grooves": 0.5,
       "modern_harmony": True,
       "shout_chorus_intensity": 1.0,
   }
   ```

2. **Modern Arrangers**:
   ```python
   class ModernBigBandArranger:
       @staticmethod
       def arrange_in_modern_style(melody: List[NoteEvent],
                                  chords: List[ChordEvent],
                                  style: str = "thad_jones") -> Dict:
           """
           Apply modern arranging techniques

           - Quartal voicings (McCoy Tyner style)
           - Wider interval voicings
           - More complex rhythms
           - Odd meters
           - Contemporary harmony
           """
   ```

3. **Validation**:
   - Generate in Thad Jones style
   - Check: Quartal voicings used (>50%)
   - Check: Wide interval spacing (average > 5 semitones between adjacent voices)
   - Compare to "A Child is Born" harmonic density
   - Listening test: Does it sound contemporary vs. swing era?

**Integration Points**:
- Expands style options: `--style thad_jones`, `--style schneider`, `--style goodwin`
- Demonstrates evolution of big band across eras
- Shows system flexibility

---

### AGENT 16: MIDI DATASET ANALYSIS ENGINE

**Objective**: Build tools to analyze MIDI datasets (PiJAMA, Weimar, Lakh) and extract statistical patterns for validation and improvement.

**Research Sources**:
1. **Datasets to analyze**:
   - PiJAMA: 200+ hours jazz piano (2,777 performances)
   - Weimar: 300 jazz solos with chord changes
   - Lakh MIDI: 176,581 files (filter for jazz/big band)

2. **Analysis techniques**:
   - Chord progression frequency analysis
   - Melodic interval distribution
   - Rhythmic pattern extraction
   - Voice leading distance measurement
   - Swing ratio measurement

**Deliverables**:
1. **MIDI Analysis Toolkit** (`analysis/dataset_analyzer.py`):
   ```python
   class DatasetAnalyzer:
       @staticmethod
       def analyze_chord_progressions(midi_files: List[str]) -> Dict:
           """
           Extract and count chord progressions

           Returns:
           {
               "ii-V-I": 1523,  # Frequency
               "I-vi-ii-V": 892,
               ...
           }
           """

       @staticmethod
       def analyze_melodic_intervals(midi_files: List[str]) -> Dict:
           """
           Measure interval distribution

           Returns:
           {
               "unison": 0.05,
               "minor_2nd": 0.15,
               "major_2nd": 0.25,
               "minor_3rd": 0.18,
               ...
           }
           """

       @staticmethod
       def measure_swing_ratios(midi_files: List[str]) -> Dict:
           """
           Extract swing ratios from real recordings

           Algorithm:
           1. Quantize to 8th note grid
           2. Measure deviation of offbeats
           3. Calculate ratio

           Returns:
           {
               "mean_swing_ratio": 0.623,
               "std_dev": 0.038,
               "tempo_correlation": -0.42  # Negative: lighter swing at fast tempos
           }
           """

       @staticmethod
       def extract_comping_rhythms(midi_files: List[str],
                                  track_type: str = "piano") -> List[List[float]]:
           """
           Extract rhythmic patterns from piano comping

           Returns: List of rhythm patterns (beat positions)
           [
               [0.25, 0.75, 1.25, 1.75],  # Charleston pattern
               [0, 0.5, 1.5, 2, 2.5],  # Montuno
               ...
           ]
           """
   ```

2. **Pattern Extraction**:
   - Extract bebop vocabulary from Charlie Parker MIDI transcriptions
   - Extract walking bass patterns from Paul Chambers
   - Extract comping patterns from Bill Evans
   - Store in libraries for use by generators

3. **Validation Metrics**:
   ```python
   @staticmethod
   def compare_generated_to_dataset(generated: List[NoteEvent],
                                   dataset: str = "PiJAMA") -> Dict:
       """
       Compare generated music to real dataset

       Metrics:
       - Interval distribution similarity (KL divergence)
       - Rhythm complexity similarity
       - Harmonic rhythm match
       - Swing ratio accuracy

       Returns:
       {
           "interval_similarity": 0.87,  # 0-1, higher is better
           "rhythm_similarity": 0.82,
           "swing_accuracy": 0.94,
           "overall_authenticity": 0.88
       }
       """
   ```

4. **Usage**:
   - Run analysis on PiJAMA dataset
   - Extract swing ratios, comping rhythms, melodic intervals
   - Use as validation baseline for generated music
   - Store statistics for future reference

**Integration Points**:
- Provides validation for ALL other agents
- Extracts patterns used by melody, comping, bass generators
- Quantifies improvement over baseline

---

### AGENT 17: QUALITY VALIDATION & TESTING ENGINEER

**Objective**: Create comprehensive test suite and validation framework to ensure generated arrangements meet professional standards.

**Research Sources**:
1. **Music information retrieval (MIR) metrics**:
   - Study ISMIR papers on music generation evaluation
   - Metrics: pitch class distribution, rhythmic complexity, harmonic tension

2. **Perceptual testing**:
   - Design listening tests (A/B comparison)
   - Metrics: musicality, authenticity, style accuracy

3. **Music theory validation**:
   - Voice leading rules (parallel 5ths, voice range violations)
   - Harmonic function correctness
   - Form structure adherence

**Deliverables**:
1. **Automated Test Suite** (`tests/validation_tests.py`):
   ```python
   class ArrangementValidator:
       @staticmethod
       def validate_voice_leading(arrangement: Dict) -> Dict:
           """
           Check voice leading rules

           Tests:
           - No parallel 5ths/octaves (warn if found, jazz is flexible)
           - All voices within range
           - No voice crossing (unless intentional)
           - Maximum leap size respected (<12 semitones)

           Returns:
           {
               "parallel_fifths": 2,  # Count
               "range_violations": 0,
               "max_leap": 9,  # Semitones
               "passed": True
           }
           """

       @staticmethod
       def validate_harmony(progression: List[JazzChord],
                           style: str = "bebop") -> Dict:
           """
           Check harmonic correctness

           Tests:
           - Appropriate chord types for style
           - Resolution of V7 to I
           - Tritone subs resolve correctly
           - No awkward root movements

           Returns:
           {
               "resolution_errors": 0,
               "inappropriate_chords": [],
               "passed": True
           }
           """

       @staticmethod
       def validate_form(arrangement: Dict,
                        expected_form: MusicalForm) -> Dict:
           """
           Check form structure

           Tests:
           - Correct number of bars
           - Intro/ending present if expected
           - Bridge differentiated from A sections
           - Shout chorus in correct location

           Returns:
           {
               "bar_count_correct": True,
               "intro_present": True,
               "bridge_differentiated": True,
               "passed": True
           }
           """

       @staticmethod
       def measure_authenticity(generated: Dict,
                               reference_dataset: str = "PiJAMA") -> Dict:
           """
           Compare generated arrangement to real recordings

           Metrics:
           - Interval distribution KL divergence
           - Rhythm complexity similarity
           - Harmonic rhythm match
           - Swing ratio accuracy
           - Voice spacing distribution

           Returns:
           {
               "authenticity_score": 0.88,  # 0-1
               "interval_match": 0.91,
               "rhythm_match": 0.85,
               "swing_match": 0.92,
               "voicing_match": 0.84
           }
           """
   ```

2. **Regression Test Suite**:
   ```python
   def test_bebop_melody_quality():
       """Test that bebop melodies use vocabulary and have good contour"""
       melody = BebopMelodyGenerator().generate_phrase(...)
       assert vocabulary_usage(melody) > 0.3  # 30% vocabulary licks
       assert phrase_contour(melody) == "arch"  # Natural shape

   def test_sax_voicing_spacing():
       """Test that sax voicings use drop-2 and have correct spacing"""
       voicing = SaxSoliVoicing.voice_melody(...)
       assert voicing_type(voicing) == "drop_2"
       assert average_voice_spacing(voicing) > 3  # >3 semitones in bass

   def test_complete_arrangement_32bar_aaba():
       """Integration test: full 32-bar AABA arrangement"""
       result = generate_professional_arrangement(form="aaba", style="basie")
       assert result['form'].total_bars == 32
       assert "intro" in result
       assert "ending" in result
       assert shout_chorus_louder(result['arrangement'])
   ```

3. **Listening Test Framework**:
   ```python
   def generate_ab_test_pairs():
       """Generate pairs for listening tests"""
       # A: Generated arrangement
       # B: Professional recording
       # Question: Which is more musical? Which is real?
   ```

4. **Validation**:
   - Run full test suite on generated arrangements
   - All tests must pass before considering system complete
   - Authenticity score must be > 0.85 vs. real recordings

**Integration Points**:
- Tests ALL modules
- Catches regressions
- Quantifies improvement
- Ensures professional quality

---

### AGENT 18: INTEGRATION ARCHITECTURE DESIGNER

**Objective**: Design clean integration layer that connects all 20 agents' modules into a cohesive system with simple API.

**Current State**:
- Modules exist but poorly integrated
- generate_professional.py is a start but incomplete
- Need clean API for users

**Deliverables**:
1. **Unified API** (`api/big_band_api.py`):
   ```python
   class BigBandGenerator:
       def __init__(self, style: str = "basie",  # basie, ellington, thad_jones, schneider
                    tempo: int = 140,
                    key: int = 0,
                    form: str = "aaba"):  # aaba, blues, abac
           self.style = style
           self.config = self._load_style_config(style)
           self.tempo = tempo
           self.key = key
           self.form_type = form

       def generate(self) -> MidiFile:
           """
           One method to generate complete arrangement

           Returns: MIDI file ready to export
           """
           # 1. Generate form structure
           form = FormGenerator.generate(self.form_type, self.key, self.tempo)

           # 2. Generate harmony
           progression = HarmonyGenerator.generate(style=self.style, form=form)

           # 3. Generate melody
           melody = MelodyGenerator.generate(progression, style=self.style)

           # 4. Arrange for big band
           arrangement = BigBandArranger.arrange_with_style(
               melody, progression, form, self.config
           )

           # 5. Apply articulations
           arrangement = ArticulationEngine.apply(arrangement, self.config)

           # 6. Apply dynamics
           arrangement = DynamicShaping.apply(arrangement, form)

           # 7. Export to MIDI
           return self._export_midi(arrangement)
   ```

2. **Style Configuration System**:
   ```python
   def _load_style_config(style: str) -> StyleConfig:
       """Load style profile (Ellington, Basie, etc.)"""
       if style == "ellington":
           return ELLINGTON_STYLE
       elif style == "basie":
           return BASIE_STYLE
       elif style == "thad_jones":
           return THAD_JONES_STYLE
       ...
   ```

3. **Command-Line Interface**:
   ```bash
   # Simple usage
   python generate_big_band.py --style basie --tempo 140 --form aaba --output my_tune.mid

   # Advanced usage
   python generate_big_band.py \
     --style ellington \
     --tempo 120 \
     --key Eb \
     --form aaba \
     --progression coltrane_changes \
     --intro vamp \
     --ending tag \
     --shout-chorus yes \
     --output giant_steps_big_band.mid
   ```

4. **Integration Tests**:
   ```python
   def test_full_pipeline():
       """Test that all modules work together"""
       generator = BigBandGenerator(style="basie", tempo=140)
       midi = generator.generate()
       assert midi is not None
       assert len(midi.tracks) == 9  # Lead, saxes, brass, rhythm
   ```

**Integration Points**:
- Central hub for all modules
- Simple user-facing API
- Extensible for future styles/features

---

### AGENT 19: GENRE SCALABILITY ARCHITECT

**Objective**: Design system architecture to scale beyond big band to other genres (orchestral, chamber, electronic, world music) without rewriting code.

**Key Insight**: Big band is ONE genre. Same voice leading, phrasing, harmony principles apply to:
- Orchestra (strings, woodwinds, brass, percussion)
- Chamber ensembles (string quartet, brass quintet)
- Vocal harmony (SATB choir, jazz vocals)
- World music (gamelan, raga, maqam)
- Electronic (synth pads, layers)

**Deliverables**:
1. **Generic Components** - Identify what's reusable:
   ```python
   # UNIVERSAL (works for any genre):
   - VoiceLeadingOptimizer  ✓
   - DynamicShaping  ✓
   - FormGenerator  ✓
   - HumanizationEngine  ✓
   - ArticulationEngine (extend for strings: pizz, arco, tremolo)

   # GENRE-SPECIFIC (need versions per genre):
   - Melody generators (bebop, classical, raga, etc.)
   - Harmony generators (jazz, classical, gamelan, etc.)
   - Voicing engines (big band, orchestra, choir, etc.)
   - Style profiles (composers/styles per genre)
   ```

2. **Abstraction Layers**:
   ```python
   class GenericArranger:
       """Base class for all arrangers"""
       def arrange(self, melody, harmony, ensemble):
           # 1. Optimize voice leading (UNIVERSAL)
           # 2. Apply dynamics (UNIVERSAL)
           # 3. Apply articulations (genre-specific)
           # 4. Export

   class BigBandArranger(GenericArranger):
       """Specialization for big band"""

   class OrchestraArranger(GenericArranger):
       """Future: orchestral arranging"""

   class ChoirArranger(GenericArranger):
       """Future: choral arranging"""
   ```

3. **Ensemble Definitions**:
   ```python
   BIG_BAND_ENSEMBLE = {
       "sections": {
           "saxes": ["alto1", "alto2", "tenor1", "tenor2", "bari"],
           "brass": ["trumpet1-4", "trombone1-4"],
           "rhythm": ["piano", "bass", "drums", "guitar"]
       },
       "ranges": {
           "alto": (52, 81),  # E3-A5
           "tenor": (47, 76),  # B2-E5
           ...
       }
   }

   ORCHESTRA_ENSEMBLE = {
       "sections": {
           "strings": ["violin1", "violin2", "viola", "cello", "bass"],
           "woodwinds": ["flute", "oboe", "clarinet", "bassoon"],
           "brass": ["horn", "trumpet", "trombone", "tuba"],
           "percussion": ["timpani", "snare", "cymbals"]
       },
       "ranges": { ... }
   }
   ```

4. **Documentation**:
   - Write guide: "How to Add a New Genre"
   - Example: Adding string quartet arranger
   - Show what to reuse vs. what to create

**Integration Points**:
- Foundation for future expansion
- Ensures current work is not wasted
- Demonstrates long-term vision

---

### AGENT 20: MASTER TESTING & BENCHMARKING LEAD

**Objective**: Coordinate final testing, benchmark against professional recordings, and produce quality report.

**Deliverables**:
1. **Benchmark Suite**:
   ```python
   BENCHMARK_TESTS = [
       {
           "name": "Basie Swing Test",
           "reference": "One O'Clock Jump (Count Basie)",
           "generate": {"style": "basie", "form": "aaba", "tempo": 180},
           "metrics": ["swing_accuracy", "riff_usage", "shout_chorus_intensity"],
           "target_score": 0.85
       },
       {
           "name": "Ellington Exotic Test",
           "reference": "Caravan (Duke Ellington)",
           "generate": {"style": "ellington", "form": "aaba", "tempo": 120},
           "metrics": ["harmony_complexity", "articulation_variety", "plunger_usage"],
           "target_score": 0.85
       },
       {
           "name": "Modern Jazz Test",
           "reference": "A Child is Born (Thad Jones)",
           "generate": {"style": "thad_jones", "form": "aaba", "tempo": 80},
           "metrics": ["quartal_voicing", "wide_spacing", "modern_harmony"],
           "target_score": 0.80
       },
   ]
   ```

2. **Quality Report**:
   ```markdown
   # Big Band Generator Quality Report

   ## Benchmark Results
   - Basie Swing Test: 0.87 ✓ (target: 0.85)
   - Ellington Exotic Test: 0.84 ✓ (target: 0.85)
   - Modern Jazz Test: 0.82 ✓ (target: 0.80)

   ## Module Performance
   - Bebop Melody: 0.89 authenticity
   - Sax Voicing: 0.91 (avg voice movement: 2.3 semitones)
   - Piano Comping: 0.86
   - Walking Bass: 0.88
   - Drums: 0.92 (swing ratio: 0.623 ± 0.019)

   ## Known Limitations
   - Articulations not yet exported to MIDI pitch bends
   - Limited composer styles (3 implemented, 10+ possible)
   - No solo section generation yet

   ## Recommendations
   - Add more composer styles (Gil Evans, Woody Herman, etc.)
   - Implement solo section framework
   - Export articulations to audio (not just MIDI)
   ```

3. **Final Validation**:
   - Generate 10 arrangements in different styles
   - Run through all validation tests
   - Produce audio exports
   - Listening test with real musicians (if possible)

**Integration Points**:
- Final checkpoint for all agents
- Produces deliverable quality report
- Identifies next priorities

---

## CROSS-AGENT COORDINATION

### Parallel Execution Strategy:
- **Phase 1 (Can run in parallel)**: Agents 1-12 (Core modules)
- **Phase 2 (Can run in parallel)**: Agents 13-15 (Style analyzers)
- **Phase 3 (Sequential)**: Agent 16 (Dataset analysis) → Agent 17 (Validation)
- **Phase 4 (Sequential)**: Agent 18 (Integration) → Agent 19 (Scalability) → Agent 20 (Testing)

### Communication Protocol:
- Each agent creates module in designated location
- Each agent documents API in module docstring
- Each agent provides usage examples
- Integration agent (18) connects all modules

### Success Criteria (OVERALL):
1. **Quantitative**:
   - Authenticity score > 0.85 vs. PiJAMA dataset
   - Voice leading distance < 3 semitones average
   - Swing ratio accuracy within 0.02 of target
   - All validation tests pass

2. **Qualitative**:
   - Arrangements sound human-written
   - Musicians can perform the arrangements
   - Style differences are audible (Basie vs. Ellington clearly different)
   - Melodies are singable and memorable

3. **Technical**:
   - Clean, documented code
   - Comprehensive test coverage
   - Scalable to other genres
   - Simple user API

---

## FINAL NOTES

**Remember**:
1. **Research extensively** - cite sources, learn from the best
2. **Build on existing code** - enhance, don't replace
3. **Think multi-genre** - solutions should scale
4. **Validate rigorously** - compare to professional recordings
5. **Document thoroughly** - future developers will thank you

**This is not just a big band generator** - it's a foundation for algorithmic music composition across all genres. The principles of voice leading, phrasing, harmony, and orchestration transcend big band and apply to all music.

**Make it the best in existence.**
