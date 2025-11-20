# Agent Master Prompts for Training Readiness
# Comprehensive Task Lists for 15 Specialized Agents

**System:** Dø MIDI Generator v2.0 (159,683 LOC)
**Goal:** Prepare for multi-genre MIDI corpus training
**Timeline:** 6-8 weeks
**Date:** November 20, 2025

---

## Agent 01: Parameter Consolidation Architect

### Mission
Consolidate the existing 165+ parameters into a coherent 50-parameter hierarchical system that works across all genres.

### Current State Analysis Required
- **Input:** `midi_generator/parameters/registry.json` (165 parameters)
- **Input:** `midi_generator/parameters/PARAMETERS.md`
- **Input:** Existing parameter expansion files

### Deliverables
1. **Hierarchical Parameter Specification** (`hierarchical_parameters.json`)
2. **Migration Mapping Document** (`parameter_migration_map.json`)
3. **Backward Compatibility Layer** (`legacy_parameter_adapter.py`)
4. **Validation Report** (`consolidation_validation_report.md`)

### Detailed Task List (35 tasks)

#### Phase 1: Analysis & Design (Tasks 1-10)
1. ☐ **Analyze existing 165 parameters**
   - Read `registry.json` completely
   - Categorize by: harmony, melody, rhythm, dynamics, structure, orchestration, genre-specific
   - Identify redundancies and overlaps
   - Document current usage in codebase (grep for references)

2. ☐ **Map parameters to validation report recommendations**
   - Cross-reference with 5 validation reports
   - Identify parameters marked as "critical" vs "low impact"
   - Calculate current feature correlation scores (use `feature_correlation_analyzer.py`)

3. ☐ **Define Level 1: Global Context (8 parameters)**
   - `genre.primary`: Extract from existing genre detection logic
   - `tempo.bpm`: Already exists, validate range (40-200)
   - `time_signature`: Consolidate from rhythm parameters
   - `key.tonic`, `key.mode`: Already exist in harmony section
   - `energy.level`: NEW - design scale based on existing dynamics + complexity metrics
   - `complexity.overall`: NEW - aggregate from existing harmony/melody complexity
   - `structure.form`: Use existing `structure_expansion.py` forms

4. ☐ **Define Level 2: Universal Dimensions (20 parameters)**
   - **Harmony (6 params):**
     - `harmony.chord_density`: Merge `harmony.chord_changes_per_measure` + voicing density
     - `harmony.complexity`: Aggregate extensions usage (9ths, 11ths, 13ths)
     - `harmony.chromaticism`: Merge tritone_sub + modal_interchange probabilities
     - `harmony.tension`: NEW - based on dissonance analysis
     - `harmony.voicing_spread`: Already exists as `harmony.voicing.spread`
     - `harmony.progression_predictability`: NEW - from progression analysis

   - **Melody (5 params):**
     - `melody.note_density`: From `melody.intervals` data
     - `melody.range_semitones`: Calculate from min/max intervals
     - `melody.contour_smoothness`: Inverse of `max_leap` + stepwise_probability
     - `melody.rhythmic_complexity`: From syncopation + subdivision data
     - `melody.repetition`: NEW - analyze motif repetition

   - **Rhythm (5 params):**
     - `rhythm.subdivision`: Categorical from existing rhythm engine
     - `rhythm.syncopation`: Already exists as `rhythm.syncopation.probability`
     - `rhythm.groove_consistency`: NEW - from groove library analysis
     - `rhythm.polyrhythm`: NEW - detect cross-rhythms
     - `rhythm.swing_amount`: Already exists as `rhythm.swing.amount`

   - **Dynamics (2 params):**
     - `dynamics.overall_level`: From `dynamics.velocity.base`
     - `dynamics.range`: From `dynamics.velocity.variation`

   - **Texture (2 params):**
     - `texture.polyphony`: NEW - simultaneous voice count
     - `texture.density`: Aggregate of note density + polyphony

5. ☐ **Define Level 3: Genre-Specific Details (22 parameters)**
   - **Universal (5 params):**
     - `orchestration.instrument_count`
     - `orchestration.register_balance`
     - `articulation.legato_ratio`: From `articulation.duration.ratio`
     - `structure.section_contrast`: NEW
     - `structure.repetition_level`: NEW

   - **Jazz (4 params):**
     - `jazz.swing_feel`: Categorized swing_amount (straight/light/medium/hard)
     - `jazz.walking_bass`: From `bass.style.walking_probability`
     - `jazz.improvisation_ratio`: NEW - from pattern analysis
     - `jazz.bebop_vocabulary`: NEW - detect bebop licks

   - **Classical (3 params):**
     - `classical.counterpoint`: NEW - contrapuntal analysis
     - `classical.development_density`: NEW - thematic development
     - `classical.voice_leading_quality`: Use existing voice leading optimizer

   - **Rock/Metal (3 params):**
     - `rock.power_chord_ratio`: From `genre.rock.power_chord_probability`
     - `rock.riff_repetition`: NEW
     - `rock.distortion_level`: NEW - from articulation intensity

   - **Electronic (3 params):**
     - `electronic.quantization`: NEW - grid alignment measure
     - `electronic.filter_movement`: NEW - spectral variation
     - `electronic.arpeggio_density`: NEW

   - **Hip-Hop (2 params):**
     - `hiphop.sample_based`: NEW - repetition analysis
     - `hiphop.boom_bap_feel`: NEW - drum pattern analysis

   - **Latin (2 params):**
     - `latin.clave_pattern`: NEW - categorical clave detection
     - `latin.montuno_complexity`: NEW

6. ☐ **Create migration mapping**
   - Build `OLD_PARAM → NEW_PARAM` dictionary
   - Document merged parameters (N→1)
   - Document dropped parameters (with justification)
   - Document new parameters (with extraction methodology)

7. ☐ **Design hierarchical dependencies**
   - Define which Level 2 params condition on which Level 1 params
   - Define which Level 3 params condition on genre.primary
   - Create dependency graph visualization

8. ☐ **Validate parameter coverage**
   - Ensure all critical musical dimensions covered
   - Cross-check against existing `musical_validator.py`
   - Verify genre-specific parameters are comprehensive

9. ☐ **Design backward compatibility layer**
   - Create adapter that maps old API calls → new hierarchical params
   - Ensure existing examples still work
   - Add deprecation warnings

10. ☐ **Document parameter semantics**
   - For each of 50 parameters:
     - Name, type, range/options
     - Musical meaning and effect
     - How to extract from MIDI
     - Relationship to existing parameters
     - Example values for different genres

#### Phase 2: Implementation (Tasks 11-25)

11. ☐ **Create `hierarchical_parameters.json` schema**
    ```json
    {
      "version": "2.0",
      "total_parameters": 50,
      "levels": {
        "level1_global": {...},
        "level2_universal": {...},
        "level3_genre_specific": {...}
      }
    }
    ```

12. ☐ **Implement Level 1 parameters**
    - Write extraction functions for each param
    - Validate against existing data
    - Unit tests for each parameter

13. ☐ **Implement Level 2 parameters**
    - Harmony extraction functions (6 params)
    - Melody extraction functions (5 params)
    - Rhythm extraction functions (5 params)
    - Dynamics extraction functions (2 params)
    - Texture extraction functions (2 params)

14. ☐ **Implement Level 3 parameters**
    - Universal orchestration params
    - Jazz-specific detection
    - Classical-specific detection
    - Rock/Metal-specific detection
    - Electronic-specific detection
    - Hip-Hop-specific detection
    - Latin-specific detection

15. ☐ **Create parameter extraction pipeline**
    ```python
    class HierarchicalParameterExtractor:
        def extract_from_midi(self, midi_file):
            # Level 1
            # Level 2 (conditioned on Level 1)
            # Level 3 (conditioned on genre)
    ```

16. ☐ **Implement migration adapter**
    ```python
    class LegacyParameterAdapter:
        def old_to_new(self, old_params_dict):
            # Map 165 old params → 50 new params

        def new_to_old(self, new_params_dict):
            # Backward compatibility
    ```

17. ☐ **Create parameter validator**
    ```python
    class HierarchicalParameterValidator:
        def validate_level1(self, params)
        def validate_level2(self, params, level1_params)
        def validate_level3(self, params, genre)
        def validate_all(self, params)
    ```

18. ☐ **Integrate with existing registry system**
    - Update `universal_registry.py`
    - Maintain old registry for compatibility
    - Add version migration logic

19. ☐ **Update parameter expansion modules**
    - `harmony_deep_expansion.py` → use hierarchical params
    - `melody_rhythm_expansion.py` → use hierarchical params
    - `dynamics_articulation_expansion.py` → use hierarchical params

20. ☐ **Create parameter visualization tool**
    - Hierarchical tree view of parameters
    - Dependency graph
    - Value distribution plots per genre

21. ☐ **Write comprehensive unit tests**
    - Test extraction from MIDI for all 50 params
    - Test migration mapping (old→new, new→old)
    - Test validation logic
    - Test edge cases and boundary values

22. ☐ **Test with existing examples**
    - Run all examples in `examples/` directory
    - Verify backward compatibility
    - Update examples to use new parameters
    - Ensure output quality maintained

23. ☐ **Performance optimization**
    - Profile parameter extraction speed
    - Optimize slow extractors
    - Add caching where appropriate
    - Target: < 100ms per MIDI file

24. ☐ **Documentation**
    - API documentation for all classes
    - Parameter reference guide (50 params)
    - Migration guide for existing users
    - Examples for each parameter

25. ☐ **Integration testing**
    - Test with HarmonyModule API
    - Test with ContextAwareGenerator
    - Test with all genre modules
    - Test batch processing

#### Phase 3: Validation & Deployment (Tasks 26-35)

26. ☐ **Validate against real MIDI corpus**
    - Extract parameters from 100 diverse MIDI files
    - Check value distributions are reasonable
    - Verify genre-specific params activate correctly

27. ☐ **Cross-validation with music theory experts**
    - Present parameter design to musicians
    - Get feedback on musical meaningfulness
    - Adjust ranges and semantics based on feedback

28. ☐ **Statistical validation**
    - Compute parameter correlations
    - Ensure hierarchical structure reduces redundancy
    - Verify genre discrimination power

29. ☐ **Generate consolidation report**
    - 165 → 50 mapping summary
    - Coverage analysis (what's preserved, what's dropped)
    - Performance comparison (old vs new)
    - Recommendations for future expansion

30. ☐ **Create parameter usage guidelines**
    - When to use Level 1 vs Level 2 vs Level 3
    - How to choose parameter values
    - Common pitfalls and best practices

31. ☐ **Update all dependent modules**
    - Update training pipeline to use 50 params
    - Update feature extractors
    - Update generators
    - Update validators

32. ☐ **Version control and migration path**
    - Tag current system as v1.5
    - Create v2.0 branch with hierarchical params
    - Provide migration scripts

33. ☐ **Deployment preparation**
    - Update configuration files
    - Update deployment scripts
    - Update API endpoints

34. ☐ **Final validation suite**
    - Run comprehensive test suite
    - Test all 50 parameters end-to-end
    - Verify musical quality maintained
    - Performance benchmarks

35. ☐ **Handoff documentation**
    - Complete parameter reference
    - Migration guide
    - Code documentation
    - Known issues and limitations

### Success Criteria
- ✅ All 165 existing parameters mapped or justified as dropped
- ✅ 50 hierarchical parameters fully implemented and tested
- ✅ Backward compatibility maintained (existing code works)
- ✅ Extraction speed < 100ms per MIDI file
- ✅ All unit tests passing (>95% coverage)
- ✅ Documentation complete and reviewed
- ✅ Validation report showing improved efficiency vs old system

### Dependencies
- None (foundational agent)

### Estimated Effort
- **Time:** 10-12 days
- **LOC:** ~5,000 lines
- **Complexity:** HIGH (requires deep musical + system knowledge)

---

## Agent 02: Corpus Acquisition Specialist

### Mission
Acquire and organize 750+ high-quality MIDI files across 5+ genres, ensuring diversity, quality, and proper copyright compliance.

### Deliverables
1. **Organized MIDI Corpus** (750+ files in `midi_corpus/`)
2. **Corpus Metadata Database** (`corpus_metadata.json`)
3. **Quality Assessment Report** (`corpus_quality_report.md`)
4. **Acquisition Documentation** (`corpus_sources.md`)

### Detailed Task List (30 tasks)

#### Phase 1: Source Identification (Tasks 1-8)

1. ☐ **Research legal MIDI sources**
   - Public domain / Creative Commons MIDI databases
   - Academic MIDI datasets (e.g., Lakh MIDI Dataset)
   - Open-source music repositories
   - User-contributed content (with licensing)

2. ☐ **Identify genre-specific sources**
   - **Jazz:** Real Book MIDI transcriptions, jazz standards databases
   - **Classical:** MuseScore, IMSLP MIDI files, classical MIDI archive
   - **Rock:** User transcriptions, guitar tab sites with MIDI export
   - **Electronic:** Electronic music production forums, synthesizer communities
   - **Pop:** Billboard chart transcriptions, pop standards
   - **Hip-Hop:** Beat maker communities, sample packs
   - **Latin:** Latin jazz resources, salsa/bossa repositories

3. ☐ **Document licensing and attribution**
   - Create spreadsheet of sources
   - Track license type (CC-BY, CC0, Public Domain, etc.)
   - Note attribution requirements
   - Flag any questionable sources

4. ☐ **Evaluate source quality**
   - Check transcription accuracy
   - Verify complete instrumentation
   - Assess timing/quantization quality
   - Test MIDI file compatibility

5. ☐ **Prioritize high-quality sources**
   - Professionally transcribed > amateur transcriptions
   - Multi-track > single-track
   - Expressive (velocity, CC) > flat
   - Genre-representative pieces

6. ☐ **Set genre-specific targets**
   - Jazz: 150 files (bebop 30, swing 40, modal 30, fusion 50)
   - Classical: 200 files (baroque 50, classical 50, romantic 50, contemporary 50)
   - Rock: 100 files (classic rock 30, progressive 30, metal 40)
   - Electronic: 120 files (ambient 30, techno 30, IDM 30, dnb 30)
   - Pop: 180 files (diverse decades and substyles)
   - **Total: 750 files**

7. ☐ **Create diversity criteria**
   - Tempo diversity: slow (40-80), medium (80-140), fast (140-200+)
   - Key diversity: all 12 keys represented
   - Time signature diversity: 4/4, 3/4, 6/8, 5/4, 7/8, etc.
   - Instrumentation diversity: solo, small ensemble, full orchestra, etc.
   - Complexity diversity: simple, medium, complex

8. ☐ **Plan acquisition workflow**
   - Automated download scripts where possible
   - Manual curation checklist
   - Quality control checkpoints
   - Naming convention standard

#### Phase 2: Acquisition (Tasks 9-20)

9. ☐ **Set up corpus directory structure**
   ```
   midi_corpus/
   ├── jazz/
   │   ├── bebop/
   │   ├── swing/
   │   ├── modal/
   │   └── fusion/
   ├── classical/
   │   ├── baroque/
   │   ├── classical_period/
   │   ├── romantic/
   │   └── contemporary/
   ├── rock/
   ├── electronic/
   ├── pop/
   ├── latin/
   ├── sources.md
   └── metadata.json
   ```

10. ☐ **Acquire Jazz corpus (150 files)**
    - Bebop: "Donna Lee", "Ornithology", "Anthropology", etc. (30 files)
    - Swing: Duke Ellington, Count Basie classics (40 files)
    - Modal: "So What", "Impressions", Bill Evans (30 files)
    - Fusion: Weather Report, Return to Forever (50 files)

11. ☐ **Acquire Classical corpus (200 files)**
    - Baroque: Bach fugues, Vivaldi, Handel (50 files)
    - Classical period: Mozart, Haydn sonatas (50 files)
    - Romantic: Chopin, Liszt, Brahms (50 files)
    - Contemporary: Debussy, Stravinsky, Bartók (50 files)

12. ☐ **Acquire Rock corpus (100 files)**
    - Classic rock: Beatles, Led Zeppelin, Rolling Stones (30 files)
    - Progressive: Pink Floyd, Yes, Genesis (30 files)
    - Metal: Black Sabbath, Metallica, Iron Maiden (40 files)

13. ☐ **Acquire Electronic corpus (120 files)**
    - Ambient: Brian Eno, Aphex Twin ambient (30 files)
    - Techno: Detroit techno, Berlin techno (30 files)
    - IDM: Autechre, Squarepusher (30 files)
    - Drum & Bass: LTJ Bukem, Goldie (30 files)

14. ☐ **Acquire Pop corpus (180 files)**
    - 60s-70s pop standards (40 files)
    - 80s pop/new wave (40 files)
    - 90s pop (40 files)
    - 2000s+ pop (60 files)

15. ☐ **Acquire Latin corpus (optional, 50+ files)**
    - Salsa, bossa nova, samba
    - Latin jazz
    - Regional styles

16. ☐ **Standardize file naming**
    Format: `{genre}_{subgenre}_{artist}_{title}_{id}.mid`
    Example: `jazz_bebop_charlie_parker_donna_lee_001.mid`

17. ☐ **Quality check each file**
    - Opens without errors in MIDI player
    - Contains note data (not empty)
    - Has reasonable tempo/duration
    - Instrumentation makes sense
    - No obvious transcription errors

18. ☐ **Track sources and attribution**
    - Create `sources.md` with:
      - File name
      - Source URL
      - License
      - Transcriber credit
      - Original composer/artist

19. ☐ **Batch download automation**
    ```python
    # Script to download from approved sources
    class CorpusDownloader:
        def download_from_source(self, source_url, genre)
        def validate_download(self, midi_file)
        def organize_by_genre(self, midi_file)
    ```

20. ☐ **Manual curation session**
    - Listen to each file
    - Remove low-quality transcriptions
    - Flag files needing metadata correction
    - Note any copyright concerns

#### Phase 3: Metadata & Organization (Tasks 21-30)

21. ☐ **Design metadata schema**
    ```json
    {
      "file_id": "unique_id",
      "file_path": "midi_corpus/jazz/bebop/...",
      "genre": {
        "primary": "jazz",
        "subgenre": "bebop"
      },
      "musical_properties": {
        "tempo_bpm": 240,
        "key_tonic": "Ab",
        ...
      },
      "source": {...},
      "quality": {...}
    }
    ```

22. ☐ **Implement automated metadata extraction**
    ```python
    class MIDIMetadataExtractor:
        def extract_basic_info(self, midi_file):
            # Duration, tempo, time sig, key
        def analyze_instrumentation(self, midi_file):
            # Track count, instruments used
        def compute_quality_metrics(self, midi_file):
            # Note density, velocity variation, etc.
    ```

23. ☐ **Generate metadata for all 750 files**
    - Run automated extraction
    - Save to `corpus_metadata.json`
    - Flag files with missing/invalid metadata

24. ☐ **Manual metadata enrichment**
    - For each file:
      - Verify genre classification
      - Add subgenre tags
      - Note special characteristics
      - Mark as "needs manual labels" if subjective params needed

25. ☐ **Create corpus statistics**
    - Files per genre
    - Tempo distribution
    - Key distribution
    - Duration statistics
    - Instrumentation breakdown

26. ☐ **Diversity analysis**
    - Check coverage of musical dimensions
    - Identify gaps (e.g., missing keys, tempos)
    - Plan targeted acquisition to fill gaps

27. ☐ **Quality assurance report**
    - Quality score distribution
    - Transcription accuracy assessment
    - Flagged issues summary
    - Recommendations for improvement

28. ☐ **Create corpus browser tool**
    ```python
    # Simple CLI/web tool to browse corpus
    class CorpusBrowser:
        def search_by_genre(self, genre)
        def filter_by_tempo(self, min_bpm, max_bpm)
        def filter_by_key(self, key)
        def show_statistics()
    ```

29. ☐ **Documentation**
    - Corpus overview (`README.md`)
    - Acquisition methodology
    - Quality criteria
    - Usage guidelines
    - Known limitations

30. ☐ **Validation and handoff**
    - Verify 750+ file count
    - Check metadata completeness
    - Test corpus browser
    - Deliver to Agent 03 for labeling

### Success Criteria
- ✅ 750+ high-quality MIDI files acquired
- ✅ All files properly licensed and attributed
- ✅ Organized directory structure
- ✅ Complete metadata for all files
- ✅ Diversity across genres, tempos, keys, etc.
- ✅ Quality score > 7/10 average
- ✅ Documentation complete

### Dependencies
- None

### Estimated Effort
- **Time:** 7-10 days
- **LOC:** ~2,000 lines (scripts + tools)
- **Complexity:** MEDIUM (time-intensive, not technically complex)

---

## Agent 03: Metadata & Labeling Manager

### Mission
Organize manual labeling of 50 MIDI files, automate labeling of remaining 700 files, and create comprehensive training dataset with all 50 hierarchical parameter labels.

### Deliverables
1. **Manual Labeling Interface** (`labeling_tool.py`)
2. **Auto-Labeling Pipeline** (`auto_labeler.py`)
3. **Complete Labeled Dataset** (`labeled_dataset.json`)
4. **Labeling Quality Report** (`labeling_quality_report.md`)

### Detailed Task List (40 tasks)

#### Phase 1: Manual Labeling Setup (Tasks 1-12)

1. ☐ **Design manual labeling interface**
   - CLI or simple web UI
   - Display MIDI playback
   - Show existing auto-labels
   - Input fields for 10 subjective parameters
   - Save progress functionality

2. ☐ **Define subjective parameters needing manual labels**
   ```python
   MANUAL_PARAMS = [
       'genre.primary',  # May need human verification
       'energy.level',  # Subjective 0.0-1.0
       'complexity.overall',  # Subjective 0.0-1.0
       'harmony.tension',  # Requires music theory
       'melody.contour_smoothness',  # Subjective judgment
       'harmony.progression_predictability',  # Music theory
       # Genre-specific (if applicable):
       'jazz.bebop_vocabulary',  # Expert judgment
       'classical.counterpoint',  # Expert judgment
       'rock.riff_repetition',  # Pattern recognition
       'electronic.filter_movement',  # Production knowledge
   ]
   ```

3. ☐ **Recruit 2 music experts**
   - Requirements:
     - Strong music theory background
     - Familiarity with multiple genres
     - Understanding of MIDI
     - 6-9 hours availability
   - Training on labeling criteria
   - Inter-rater reliability check

4. ☐ **Create labeling guidelines document**
   - For each subjective parameter:
     - Definition and musical meaning
     - Scale interpretation (what is 0.0 vs 1.0?)
     - Genre-specific considerations
     - Examples from reference recordings

5. ☐ **Select 50 files for manual labeling**
   - 10 files per genre (jazz, classical, rock, electronic, pop)
   - Maximize diversity within each genre
   - Include edge cases and clear examples
   - Ensure coverage of parameter space

6. ☐ **Implement labeling tool**
   ```python
   class ManualLabelingTool:
       def load_midi_file(self, path)
       def display_auto_labels(self, file_id)
       def play_audio(self)  # Convert MIDI → audio
       def collect_manual_labels(self, parameters)
       def save_labels(self, file_id, labels)
       def track_progress(self)
   ```

7. ☐ **Add labeling quality checks**
   - Range validation (e.g., 0.0-1.0)
   - Consistency checks (e.g., high complexity → high tension)
   - Outlier detection
   - Required field validation

8. ☐ **Test labeling tool**
   - Test with 5 sample files
   - Get feedback from experts
   - Fix usability issues
   - Optimize workflow

9. ☐ **Create labeling schedule**
   - Expert 1: Files 1-25 (6-7 hours, 15-18 min/file)
   - Expert 2: Files 26-50 (6-7 hours)
   - Parallel work possible
   - Deadlines and milestones

10. ☐ **Inter-rater reliability test**
    - Have both experts label same 5 files
    - Compute agreement scores
    - Resolve disagreements
    - Calibrate labeling criteria if needed

11. ☐ **Execute manual labeling sessions**
    - Supervise first few files
    - Check for consistency
    - Answer questions
    - Monitor progress

12. ☐ **Validate manual labels**
    - Check completeness (all 50 files labeled)
    - Check quality (no obvious errors)
    - Compute inter-rater agreement
    - Flag files needing re-labeling

#### Phase 2: Auto-Labeling Pipeline (Tasks 13-28)

13. ☐ **Design auto-labeling architecture**
    ```python
    class AutoLabeler:
        def extract_deterministic_labels(self, midi_file)
        def extract_level1_labels(self, midi_file)
        def extract_level2_labels(self, midi_file)
        def extract_level3_labels(self, midi_file, genre)
    ```

14. ☐ **Implement Level 1 auto-extractors**
    - `tempo.bpm`: Use MIDI tempo events or beat tracking
    - `time_signature`: From MIDI meta events
    - `key.tonic`, `key.mode`: Key detection algorithm (music21 or custom)
    - `structure.form`: Pattern-based form detection

15. ☐ **Implement Level 2 Harmony auto-extractors**
    - `harmony.chord_density`: Chord detection + count per measure
    - `harmony.complexity`: Analyze chord extensions (9ths, 11ths, 13ths)
    - `harmony.chromaticism`: Chromatic note ratio
    - `harmony.voicing_spread`: Analyze pitch range in chords

16. ☐ **Implement Level 2 Melody auto-extractors**
    - `melody.note_density`: Notes per measure in melody track
    - `melody.range_semitones`: max(pitch) - min(pitch)
    - `melody.rhythmic_complexity`: Rhythm entropy or variation
    - `melody.repetition`: Detect repeated motifs

17. ☐ **Implement Level 2 Rhythm auto-extractors**
    - `rhythm.subdivision`: Detect smallest note duration
    - `rhythm.syncopation`: Off-beat note ratio
    - `rhythm.groove_consistency`: Timing deviation analysis
    - `rhythm.polyrhythm`: Detect conflicting rhythms
    - `rhythm.swing_amount`: Swing ratio calculation

18. ☐ **Implement Level 2 Dynamics auto-extractors**
    - `dynamics.overall_level`: Mean MIDI velocity
    - `dynamics.range`: Velocity std dev or max-min

19. ☐ **Implement Level 2 Texture auto-extractors**
    - `texture.polyphony`: Max simultaneous notes
    - `texture.density`: Notes per second across all tracks

20. ☐ **Implement Level 3 Universal auto-extractors**
    - `orchestration.instrument_count`: Count distinct MIDI programs
    - `orchestration.register_balance`: Low vs high pitch distribution
    - `articulation.legato_ratio`: Note duration / inter-onset interval

21. ☐ **Implement genre-specific auto-extractors**
    - **Jazz:** `jazz.walking_bass` (detect walking bass pattern)
    - **Classical:** `classical.voice_leading_quality` (voice leading cost)
    - **Rock:** `rock.power_chord_ratio` (detect power chords)
    - **Electronic:** `electronic.quantization` (grid alignment)
    - **Hip-Hop:** `hiphop.sample_based` (detect loops)

22. ☐ **Integrate existing analysis modules**
    - Use `midi_generator.analysis.midi_analyzer.py`
    - Use `midi_generator.learning.pattern_extractor.py`
    - Use existing genre detection
    - Use harmony analysis from generators

23. ☐ **Test auto-labeling on sample files**
    - Run on 50 manually labeled files
    - Compare auto-labels vs manual labels (where overlap)
    - Validate deterministic labels are correct
    - Fix extraction bugs

24. ☐ **Optimize extraction speed**
    - Profile slow extractors
    - Parallelize where possible
    - Add caching
    - Target: < 2 seconds per file

25. ☐ **Handle edge cases**
    - Empty tracks
    - Missing tempo/time signature
    - Unusual instruments
    - Corrupt MIDI files
    - Graceful degradation

26. ☐ **Batch process remaining 700 files**
    ```python
    for midi_file in remaining_700_files:
        auto_labels = auto_labeler.extract_all(midi_file)
        save_labels(midi_file, auto_labels)
        progress_bar.update()
    ```

27. ☐ **Quality check auto-labels**
    - Check for NaN/invalid values
    - Check value distributions (outliers?)
    - Verify genre-specific params only set for correct genres
    - Flag suspicious files for manual review

28. ☐ **Manual review of flagged files**
    - Review ~50-100 files with unusual labels
    - Correct errors
    - Improve extraction algorithms if patterns emerge

#### Phase 3: Dataset Assembly & Validation (Tasks 29-40)

29. ☐ **Merge manual + auto labels**
    ```python
    for file_id in all_750_files:
        labels = {}
        # Auto-labels for all files
        labels.update(auto_labels[file_id])
        # Overwrite with manual labels if available
        if file_id in manually_labeled_files:
            labels.update(manual_labels[file_id])
        complete_dataset[file_id] = labels
    ```

30. ☐ **Create labeled dataset format**
    ```json
    {
      "file_id": "jazz_bebop_001",
      "file_path": "midi_corpus/jazz/bebop/...",
      "labels": {
        "level1": {
          "genre.primary": "jazz",
          "tempo.bpm": 240,
          ...
        },
        "level2": {
          "harmony.chord_density": 4.5,
          ...
        },
        "level3": {
          "jazz.swing_feel": "light",
          "jazz.walking_bass": 0.9,
          ...
        }
      },
      "labeling_metadata": {
        "auto_labeled": true/false,
        "manually_labeled": true/false,
        "quality_score": 0.9,
        "notes": "..."
      }
    }
    ```

31. ☐ **Statistical validation**
    - Compute label distributions per genre
    - Check for label imbalance
    - Verify diversity
    - Correlations between parameters

32. ☐ **Train/val/test split**
    - 70% train (525 files)
    - 15% val (112 files)
    - 15% test (113 files)
    - Stratify by genre
    - Save split metadata

33. ☐ **Create data loaders**
    ```python
    class LabeledMIDIDataset(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            midi_file = self.files[idx]
            features = extract_features(midi_file)
            labels = load_labels(midi_file)
            return features, labels
    ```

34. ☐ **Generate labeling quality report**
    - Manual labeling statistics
    - Inter-rater agreement scores
    - Auto-labeling validation results
    - Coverage analysis (all parameters labeled?)
    - Known issues and limitations

35. ☐ **Create dataset documentation**
    - Dataset overview
    - Labeling methodology
    - Parameter definitions
    - Usage examples
    - Citation/attribution

36. ☐ **Version control for dataset**
    - Save dataset with version number
    - Track changes over time
    - Allow rollback if needed

37. ☐ **Create dataset statistics dashboard**
    - Visualizations of label distributions
    - Genre breakdowns
    - Parameter correlations
    - Missing data summary

38. ☐ **Export dataset in multiple formats**
    - JSON (primary)
    - CSV (for analysis)
    - HDF5 (for ML frameworks)
    - Pickle (Python)

39. ☐ **Final validation**
    - Load dataset successfully
    - All 750 files present
    - All 50 parameters labeled (or marked N/A)
    - No corrupt/missing data
    - Reasonable value distributions

40. ☐ **Handoff to Agent 04 (Feature Selection)**
    - Provide labeled dataset
    - Provide labeling quality report
    - Provide documentation
    - Answer questions

### Success Criteria
- ✅ 50 files manually labeled by experts
- ✅ 700 files auto-labeled with high confidence
- ✅ Complete dataset with all 50 parameters
- ✅ Inter-rater agreement > 0.8 (for manual labels)
- ✅ Auto-label validation accuracy > 0.9 (deterministic params)
- ✅ Train/val/test split properly stratified
- ✅ Comprehensive documentation

### Dependencies
- Agent 01 (hierarchical parameters defined)
- Agent 02 (corpus acquired)

### Estimated Effort
- **Time:** 12-15 days (includes 12-17 hours of expert labeling)
- **LOC:** ~8,000 lines
- **Complexity:** HIGH (requires coordination, tools, validation)

---

## Agent 04: Feature Selection Optimizer

### Mission
Reduce the feature space from 1000+ features to the most predictive 200 features for efficient training.

### Deliverables
1. **Feature Selection Pipeline** (`feature_selector.py`)
2. **Selected Features List** (`selected_features_200.json`)
3. **Feature Importance Analysis** (`feature_importance_report.md`)
4. **Optimized Feature Extractor** (`optimized_feature_extractor.py`)

### Detailed Task List (32 tasks)

#### Phase 1: Baseline Feature Analysis (Tasks 1-10)

1. ☐ **Audit existing feature extraction**
   - Review `midi_generator/learning/feature_parameter_mapper.py`
   - Review `midi_generator/analysis/intelligent_gap_detector.py`
   - Count total features currently extracted
   - Categorize features by type

2. ☐ **Extract features from labeled dataset**
   ```python
   for file in labeled_dataset:
       features_1000D = extract_all_features(file)
       save_features(file_id, features_1000D)
   ```

3. ☐ **Feature categorization**
   - Spectral features (chroma, MFCC, spectral centroids, etc.)
   - Harmonic features (chord analysis, progressions, etc.)
   - Melodic features (intervals, contours, motifs)
   - Rhythmic features (onset density, syncopation, groove)
   - Structural features (form, repetition, development)
   - Timbral features (instrumentation, register, articulation)

4. ☐ **Compute feature statistics**
   - Mean, std dev, min, max for each feature
   - Value distributions
   - Missing value rates
   - Outliers

5. ☐ **Correlation analysis**
   - Use existing `feature_correlation_analyzer.py`
   - Compute pairwise feature correlations
   - Identify highly correlated feature groups (r > 0.9)
   - Flag redundant features

6. ☐ **Variance analysis**
   - Identify low-variance features (constant or near-constant)
   - Flag for removal (no predictive power)

7. ☐ **Initial feature filtering**
   - Remove features with > 50% missing values
   - Remove zero-variance features
   - Remove features with > 0.95 correlation to another feature
   - Reduces 1000+ → ~600 features

8. ☐ **Per-parameter correlation analysis**
   - For each of 50 parameters:
     - Compute correlation with all features
     - Rank features by correlation strength
     - Identify top 20 features per parameter

9. ☐ **Feature-parameter mapping**
   - Create matrix: [50 parameters × 600 features]
   - Identify features used by multiple parameters (shared)
   - Identify parameter-specific features

10. ☐ **Baseline feature set**
    - Aggregate top features across all parameters
    - Aim for ~400 features after initial filtering

#### Phase 2: Feature Selection Methods (Tasks 11-22)

11. ☐ **Method 1: Filter-based selection (Correlation)**
    - Select top N features by correlation with each parameter
    - Union of all top features
    - Fast, but may miss interactions

12. ☐ **Method 2: Univariate statistical tests**
    - For continuous parameters: F-statistic, mutual information
    - For categorical parameters: Chi-squared, ANOVA
    - Select features with p-value < 0.05

13. ☐ **Method 3: Tree-based feature importance**
    ```python
    # Train XGBoost on each parameter
    for param in parameters:
        model = XGBoost()
        model.fit(features_400D, labels[param])
        importance[param] = model.feature_importances_

    # Aggregate importance across parameters
    global_importance = aggregate(importance)
    top_features = select_top_N(global_importance, N=200)
    ```

14. ☐ **Method 4: L1 regularization (Lasso)**
    - Train Lasso regression for continuous parameters
    - Features with non-zero coefficients are selected

15. ☐ **Method 5: Recursive Feature Elimination (RFE)**
    - Iteratively remove least important features
    - Expensive but effective
    - Use on reduced set (400 → 200)

16. ☐ **Method 6: Principal Component Analysis (PCA)**
    - Transform features to principal components
    - Select components explaining 95% variance
    - Note: loses interpretability

17. ☐ **Method 7: Domain knowledge curation**
    - Music theory experts identify must-have features
    - Ensure critical musical dimensions covered
    - Override statistical selection if needed

18. ☐ **Ensemble feature selection**
    - Combine results from multiple methods
    - Features selected by >= 3 methods are kept
    - Increases robustness

19. ☐ **Hierarchical feature selection**
    - Level 1 parameters: select 50 features
    - Level 2 parameters: select 100 features (includes L1 features)
    - Level 3 parameters: select 50 additional genre-specific features
    - Total: 200 features

20. ☐ **Feature selection by category**
    - Ensure representation from all categories:
      - Spectral: 30 features
      - Harmonic: 40 features
      - Melodic: 35 features
      - Rhythmic: 40 features
      - Structural: 30 features
      - Timbral: 25 features

21. ☐ **Validation with holdout set**
    - Train models with 200-feature set
    - Compare performance vs 1000-feature set
    - Ensure minimal performance loss (<5%)

22. ☐ **Finalize feature selection**
    - Create `selected_features_200.json`
    - Document selection methodology
    - Provide justification for each feature

#### Phase 3: Implementation & Optimization (Tasks 23-32)

23. ☐ **Implement optimized feature extractor**
    ```python
    class OptimizedFeatureExtractor:
        def __init__(self, selected_features_list):
            self.features = selected_features_list

        def extract(self, midi_file):
            # Only extract selected 200 features
            # Skip unused features for speed
            return features_200D
    ```

24. ☐ **Optimize extraction performance**
    - Profile extraction time
    - Parallelize independent feature computations
    - Cache intermediate results
    - Target: < 1 second per file

25. ☐ **Feature normalization pipeline**
    ```python
    class FeatureNormalizer:
        def fit(self, features):
            # Compute mean, std from training set

        def transform(self, features):
            # Standardize: (x - mean) / std
    ```

26. ☐ **Handle missing features gracefully**
    - If feature can't be extracted → use mean value
    - Flag files with many missing features

27. ☐ **Create feature extraction batch processor**
    ```python
    def batch_extract_features(midi_files, output_dir):
        for file in tqdm(midi_files):
            features = extract_optimized(file)
            save(output_dir / f"{file_id}.npy", features)
    ```

28. ☐ **Validate extracted features**
    - Check shapes (all 200D vectors)
    - Check for NaN/Inf values
    - Check value ranges are reasonable

29. ☐ **Create feature importance visualization**
    - Bar charts of top features per parameter
    - Heatmap of feature-parameter correlations
    - Feature category breakdown

30. ☐ **Documentation**
    - Feature selection methodology
    - List of 200 selected features with descriptions
    - Usage guide for optimized extractor
    - Performance benchmarks

31. ☐ **Integration testing**
    - Test with training pipeline (Agent 06)
    - Verify compatibility with existing modules
    - Ensure backward compatibility option

32. ☐ **Final validation and handoff**
    - Extract features for all 750 files
    - Generate feature importance report
    - Deliver to Agent 05 (Hierarchical MTL)

### Success Criteria
- ✅ Feature space reduced from 1000+ to 200
- ✅ Performance loss < 5% vs full feature set
- ✅ All feature categories represented
- ✅ Extraction speed < 1 second per file
- ✅ Comprehensive documentation
- ✅ Validation passing on all 750 files

### Dependencies
- Agent 01 (hierarchical parameters)
- Agent 03 (labeled dataset)

### Estimated Effort
- **Time:** 8-10 days
- **LOC:** ~4,000 lines
- **Complexity:** HIGH (statistical analysis + domain knowledge)

---

## Agents 05-15: Continued...

(Due to length constraints, I'll provide condensed summaries for the remaining agents. Each would follow the same detailed format.)

---

## Agent 05: Hierarchical MTL Architect
**Mission:** Implement the hierarchical multi-task neural network architecture.
**Key Tasks:**
- Design 3-level neural architecture (shared encoder + hierarchical heads)
- Implement conditional parameter prediction
- Create multi-task loss function with hierarchical weighting
- Integration with PyTorch/TensorFlow
- Comprehensive testing
**LOC:** ~12,000

---

## Agent 06: Training Pipeline Engineer
**Mission:** Build end-to-end training infrastructure.
**Key Tasks:**
- Data loaders and preprocessing
- Training loop with early stopping
- Learning rate scheduling
- Checkpointing and model saving
- Distributed training support (optional)
- Integration with wandb/mlflow
**LOC:** ~8,000

---

## Agent 07: Multi-Genre Data Specialist
**Mission:** Handle genre-specific data requirements and balancing.
**Key Tasks:**
- Genre stratification strategies
- Data augmentation per genre
- Genre-specific validation splits
- Handling genre imbalance
- Cross-genre transfer learning
**LOC:** ~6,000

---

## Agent 08: Validation Framework Builder
**Mission:** Create comprehensive testing and validation suite.
**Key Tasks:**
- Per-parameter metric validation
- Musical quality validation (intervals, harmony, rhythm)
- Genre-specific validation functions
- Cross-genre generalization testing
- Regression testing
- Human evaluation framework
**LOC:** ~7,000

---

## Agent 09: HarmonyModule Integration Lead
**Mission:** Connect trained models to the existing HarmonyModule generator.
**Key Tasks:**
- Parameter prediction API
- Integration with existing `HarmonyModuleAPI`
- Bidirectional workflow (MIDI → params, params → MIDI)
- Real-time generation pipeline
- Performance optimization
**LOC:** ~5,000

---

## Agent 10: Performance Optimization Specialist
**Mission:** Optimize inference and training speed.
**Key Tasks:**
- Model quantization and pruning
- Batch prediction optimization
- GPU acceleration
- Caching strategies
- Profiling and bottleneck identification
**LOC:** ~4,000

---

## Agent 11: Monitoring & Logging Engineer
**Mission:** Implement comprehensive logging and monitoring.
**Key Tasks:**
- Training progress tracking
- Model performance dashboards
- Error logging and alerting
- Data quality monitoring
- Resource usage tracking
**LOC:** ~3,000

---

## Agent 12: Documentation & API Designer
**Mission:** Create user-facing documentation and APIs.
**Key Tasks:**
- API design for parameter prediction
- User guides and tutorials
- Code documentation (docstrings)
- Example notebooks
- Video tutorials
**LOC:** ~2,000

---

## Agent 13: Experiment Management Lead
**Mission:** Set up experiment tracking infrastructure.
**Key Tasks:**
- MLflow/Wandb integration
- Hyperparameter search management
- Experiment versioning
- Result comparison tools
**LOC:** ~3,000

---

## Agent 14: Error Analysis Specialist
**Mission:** Analyze model failures and improve robustness.
**Key Tasks:**
- Failure case analysis
- Prediction error analysis
- Data quality issues identification
- Improvement recommendations
**LOC:** ~4,000

---

## Agent 15: Production Deployment Engineer
**Mission:** Deploy system to production.
**Key Tasks:**
- Docker containerization
- API serving (FastAPI/Flask)
- CI/CD pipeline
- Load balancing
- Monitoring and alerts
**LOC:** ~3,000

---

## Agent Execution Plan

### Sequential Dependencies
```
Week 1-2:
├─ Agent 01 (Parameters) [PARALLEL START]
└─ Agent 02 (Corpus)     [PARALLEL START]

Week 2-3:
└─ Agent 03 (Labeling) [depends on 01, 02]

Week 3:
├─ Agent 04 (Features) [depends on 01, 03]
└─ Agent 05 (MTL Arch) [depends on 01, 04]

Week 4:
├─ Agent 06 (Training) [depends on 04, 05]
└─ Agent 07 (Multi-Genre) [depends on 03, 05]

Week 5:
├─ Agent 08 (Validation) [depends on 05, 06]
└─ Agent 09 (Integration) [depends on 06]

Week 6:
├─ Agent 10 (Optimization) [depends on 09]
├─ Agent 11 (Monitoring) [PARALLEL]
└─ Agent 12 (Documentation) [PARALLEL]

Week 7:
├─ Agent 13 (Experiments) [PARALLEL]
├─ Agent 14 (Error Analysis) [depends on 08]
└─ Agent 15 (Deployment) [depends on 09, 10]

Week 8:
└─ Final Integration & Testing
```

### Parallelization Opportunities
- Agents 01 & 02 can start simultaneously
- Agents 11, 12, 13 can run in parallel during Weeks 6-7
- Feature extraction (Agent 04) can overlap with architecture design (Agent 05)

---

## Total Estimated Impact

| Aspect | Estimate |
|--------|----------|
| **Total New LOC** | ~76,000 lines |
| **Total Effort** | 95-115 person-days |
| **Timeline** | 6-8 weeks (with parallelization) |
| **Success Probability** | 75-85% |
| **Final System Size** | 235K LOC (159K existing + 76K new) |

---

## Critical Success Factors

1. **Agent 01** must complete first (foundational)
2. **Agent 03** manual labeling is critical (12-17 hours of expert time)
3. **Agent 05** architecture must handle hierarchical conditioning
4. **Agent 08** validation ensures quality
5. **Continuous integration** as agents complete

---

**END OF AGENT MASTER PROMPTS**
