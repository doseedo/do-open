# Manual Labeling Guidelines for MIDI Corpus
**Agent 03: Metadata & Labeling Manager**
**Version:** 2.0
**Date:** November 20, 2025

---

## Overview

This document provides comprehensive guidelines for manually labeling 50 MIDI files with subjective musical parameters that cannot be reliably auto-extracted.

### Labeling Objectives
- Achieve **high inter-rater reliability** (agreement > 0.8)
- Provide consistent, musically meaningful labels
- Train machine learning models to predict these parameters for remaining 700 files

### Time Requirement
- **15-18 minutes per file** (estimated)
- **50 files total** = 12-15 hours per expert
- **2 experts** for reliability checking

---

## Subjective Parameters Requiring Manual Labels

### 1. **energy.level** (Level 1 - Global)
**Type:** Continuous [0.0 - 1.0]
**Musical Definition:** Overall perceived energy/intensity of the piece

#### Scale Interpretation
| Value | Description | Musical Characteristics |
|-------|-------------|------------------------|
| 0.0 - 0.2 | Very low energy | Ambient, calm, meditative, slow, quiet, sparse |
| 0.3 - 0.4 | Low energy | Relaxed, gentle, laid-back, moderate tempo |
| 0.5 - 0.6 | Moderate energy | Balanced, steady, comfortable pace |
| 0.7 - 0.8 | High energy | Driving, exciting, up-tempo, dynamic |
| 0.9 - 1.0 | Very high energy | Intense, aggressive, frenetic, very fast/loud |

#### Factors to Consider
- **Tempo:** Faster = higher energy
- **Dynamics:** Louder = higher energy
- **Density:** More notes/events = higher energy
- **Timbre:** Aggressive timbres (distortion, brass) = higher energy
- **Rhythm:** Driving rhythms = higher energy

#### Genre-Specific Examples
- **Jazz:** Bebop @ 240 BPM = 0.8-0.9, Ballad = 0.2-0.3
- **Classical:** Beethoven 5th Symphony (1st mvt) = 0.8, Debussy Clair de Lune = 0.3
- **Rock:** AC/DC "Thunderstruck" = 0.9, Pink Floyd "Breathe" = 0.4
- **Electronic:** Drum & Bass = 0.8-0.9, Ambient Techno = 0.3-0.4

---

### 2. **complexity.overall** (Level 1 - Global)
**Type:** Continuous [0.0 - 1.0]
**Musical Definition:** Overall musical complexity across harmony, melody, and rhythm

#### Scale Interpretation
| Value | Description | Musical Characteristics |
|-------|-------------|------------------------|
| 0.0 - 0.2 | Very simple | Repetitive, few chords, simple melody, basic rhythm |
| 0.3 - 0.4 | Simple | Clear patterns, diatonic harmony, singable melody |
| 0.5 - 0.6 | Moderate | Some variation, extended chords, developed themes |
| 0.7 - 0.8 | Complex | Rich harmony, intricate melody, polyrhythms |
| 0.9 - 1.0 | Very complex | Dense, chromatic, atonal, highly developed |

#### Factors to Consider
- **Harmonic Complexity:** Extensions, alterations, modulations
- **Melodic Complexity:** Range, leaps, ornamentation, development
- **Rhythmic Complexity:** Syncopation, polyrhythm, metric modulation
- **Structural Complexity:** Thematic development, form complexity
- **Density:** Information density per measure

#### Genre-Specific Examples
- **Jazz:** Simple blues = 0.3, Coltrane "Giant Steps" = 0.9
- **Classical:** Mozart simple sonata = 0.5, Schoenberg 12-tone = 0.95
- **Rock:** Three-chord punk = 0.2, Rush "La Villa Strangiato" = 0.8
- **Electronic:** Minimal techno = 0.3, IDM (Autechre) = 0.85

---

### 3. **harmony.tension** (Level 2 - Harmony)
**Type:** Continuous [0.0 - 1.0]
**Musical Definition:** Degree of harmonic tension and dissonance

#### Scale Interpretation
| Value | Description | Dissonance Level |
|-------|-------------|-----------------|
| 0.0 - 0.2 | Very consonant | Triads, resolved, stable |
| 0.3 - 0.4 | Mostly consonant | 7th chords, gentle dissonance |
| 0.5 - 0.6 | Balanced | Mix of tension and resolution |
| 0.7 - 0.8 | Tense | Heavy alterations, suspended, unresolved |
| 0.9 - 1.0 | Very tense | Clusters, atonality, extreme dissonance |

#### Factors to Consider
- **Chord Quality:** Major/minor (low) vs diminished/augmented (high)
- **Extensions:** 9ths, 11ths, 13ths add moderate tension
- **Alterations:** #11, b9, #9 add high tension
- **Resolution:** Frequent resolution (low) vs sustained tension (high)
- **Voice Leading:** Smooth (low) vs parallel dissonances (high)

---

### 4. **harmony.progression_predictability** (Level 2 - Harmony)
**Type:** Continuous [0.0 - 1.0]
**Musical Definition:** How predictable/formulaic the chord progressions are

#### Scale Interpretation
| Value | Description | Characteristics |
|-------|-------------|----------------|
| 0.0 - 0.2 | Unpredictable | Surprising, unusual, atonal |
| 0.3 - 0.4 | Somewhat unpredictable | Some unexpected changes |
| 0.5 - 0.6 | Moderately predictable | Mix of common and uncommon |
| 0.7 - 0.8 | Predictable | Common progressions (I-IV-V-I) |
| 0.9 - 1.0 | Very predictable | Formulaic, clichéd |

#### Common Predictable Progressions
- **Pop:** I - V - vi - IV (very predictable = 0.9)
- **Jazz:** ii-V-I (predictable in context = 0.7)
- **Classical:** Authentic cadences, circle of fifths

#### Unpredictable Progressions
- Modal interchange, tritone substitutions
- Chromatic mediants, unexpected modulations
- Non-functional harmony

---

### 5. **melody.contour_smoothness** (Level 2 - Melody)
**Type:** Continuous [0.0 - 1.0]
**Musical Definition:** How smooth (stepwise) vs angular (leaps) the melody is

#### Scale Interpretation
| Value | Description | Intervallic Content |
|-------|-------------|---------------------|
| 0.0 - 0.2 | Very angular | Mostly large leaps (6ths, 7ths, octaves+) |
| 0.3 - 0.4 | Angular | Frequent leaps mixed with steps |
| 0.5 - 0.6 | Balanced | Equal mix of steps and leaps |
| 0.7 - 0.8 | Smooth | Mostly stepwise with occasional leaps |
| 0.9 - 1.0 | Very smooth | Almost entirely stepwise motion |

#### Analysis Tips
- Count stepwise intervals (2nds) vs leaps (3rds+)
- Stepwise ratio > 70% = smooth (0.7-0.9)
- Stepwise ratio < 40% = angular (0.2-0.4)

---

### 6. **jazz.bebop_vocabulary** (Level 3 - Jazz-Specific)
**Type:** Continuous [0.0 - 1.0]
**Musical Definition:** Presence of bebop-specific melodic patterns and language

#### Scale Interpretation
| Value | Description | Bebop Characteristics |
|-------|-------------|---------------------|
| 0.0 - 0.2 | No bebop vocabulary | Swing/trad jazz, or other genre |
| 0.3 - 0.4 | Minimal bebop | A few bebop-ish phrases |
| 0.5 - 0.6 | Moderate bebop | Clear bebop influence |
| 0.7 - 0.8 | Strong bebop | Heavy use of bebop language |
| 0.9 - 1.0 | Pure bebop | Classic Charlie Parker style |

#### Bebop Indicators
- **Chromatic passing tones** on off-beats
- **Enclosures** (chromatic approach from above and below)
- **Altered scales** on dominant chords
- **Fast scalar runs** with chromatic alterations
- **Arpeggiation** of complex chords (9ths, 11ths, 13ths)
- **Angular melodies** with large leaps
- **Very fast tempos** (180-300 BPM)

#### Reference Recordings
- **0.9-1.0:** Charlie Parker "Donna Lee", "Anthropology"
- **0.7-0.8:** Clifford Brown, Dizzy Gillespie
- **0.3-0.4:** Some modern jazz with bebop influences
- **0.0-0.2:** Modal jazz, cool jazz, fusion, smooth jazz

---

### 7. **classical.counterpoint** (Level 3 - Classical-Specific)
**Type:** Continuous [0.0 - 1.0]
**Musical Definition:** Degree of contrapuntal writing (independent melodic lines)

#### Scale Interpretation
| Value | Description | Texture Type |
|-------|-------------|-------------|
| 0.0 - 0.2 | Homophonic | Melody + chordal accompaniment |
| 0.3 - 0.4 | Slightly contrapuntal | Some voice independence |
| 0.5 - 0.6 | Moderately contrapuntal | Clear multiple voices |
| 0.7 - 0.8 | Highly contrapuntal | Fugato, invention-style |
| 0.9 - 1.0 | Strict counterpoint | Fugue, canon, strict species |

#### Indicators of Counterpoint
- **Voice Independence:** Each voice has own melodic integrity
- **Imitation:** Themes passed between voices
- **Inversion:** Melodic lines inverted
- **Contrary Motion:** Voices move in opposite directions
- **Rhythmic Independence:** Different rhythms in each voice

#### Reference Examples
- **0.9-1.0:** Bach fugues, canons
- **0.7-0.8:** Bach inventions, Baroque trio sonatas
- **0.4-0.5:** Mozart string quartets (some movements)
- **0.0-0.2:** Romantic piano pieces (Chopin nocturnes)

---

### 8. **rock.riff_repetition** (Level 3 - Rock-Specific)
**Type:** Continuous [0.0 - 1.0]
**Musical Definition:** Degree to which composition is based on repeating riffs

#### Scale Interpretation
| Value | Description | Riff Usage |
|-------|-------------|-----------|
| 0.0 - 0.2 | No riffs | Through-composed, non-repetitive |
| 0.3 - 0.4 | Some riffs | Riff in intro/verse, but varied |
| 0.5 - 0.6 | Moderate riff-based | Central riff with variations |
| 0.7 - 0.8 | Heavily riff-based | Entire song built on riff(s) |
| 0.9 - 1.0 | Pure riff | Single riff repeated throughout |

#### Riff Characteristics
- **Short melodic/rhythmic pattern** (1-4 bars)
- **Repeated extensively** throughout piece
- **Often guitar-based** (but can be keyboard, bass)
- **Defines the song identity**

#### Reference Examples
- **0.9-1.0:** AC/DC "Back in Black", Black Sabbath "Iron Man"
- **0.7-0.8:** Led Zeppelin "Whole Lotta Love", Deep Purple "Smoke on the Water"
- **0.4-0.5:** Progressive rock with riff sections
- **0.0-0.2:** Ballads, through-composed prog

---

### 9. **electronic.filter_movement** (Level 3 - Electronic-Specific)
**Type:** Continuous [0.0 - 1.0]
**Musical Definition:** Amount of timbral/spectral variation (filter sweeps, modulation)

**Note:** This is challenging to judge from MIDI alone. Focus on **melodic/harmonic changes that imply filter movement**.

#### Scale Interpretation
| Value | Description | Timbral Characteristics |
|-------|-------------|----------------------|
| 0.0 - 0.2 | Static | Unchanging timbre, no sweeps |
| 0.3 - 0.4 | Minimal movement | Subtle variations |
| 0.5 - 0.6 | Moderate movement | Clear filter automation |
| 0.7 - 0.8 | Heavy movement | Frequent sweeps/modulation |
| 0.9 - 1.0 | Extreme movement | Constant timbral evolution |

#### Indicators in MIDI
- **Velocity variations** (implies brightness changes)
- **CC modulation** (if present)
- **Rhythmic patterns** that suggest filter envelopes
- **Genre knowledge:** Techno/trance = high, ambient = low-medium

---

## Labeling Workflow

### Preparation
1. **Read this entire document carefully**
2. **Listen to reference examples** for each parameter
3. **Calibrate** with 5 practice files
4. **Ask questions** before starting official labeling

### For Each File

1. **Listen Multiple Times**
   - First listen: Overall impression
   - Second listen: Focus on specific parameters
   - Third listen (if needed): Difficult parameters

2. **Use the Labeling Tool**
   - Tool will display auto-extracted labels for reference
   - Input your judgments for each subjective parameter
   - Use dropdown menus for categorical parameters
   - Use sliders or text input for continuous parameters

3. **Quality Checks**
   - Tool validates ranges automatically
   - Review for consistency (e.g., high complexity → likely higher tension)
   - Flag files if uncertain

4. **Notes Field**
   - Use notes field for:
     - Uncertainty about a parameter
     - Unusual characteristics
     - Genre ambiguity
     - Technical issues (corrupted MIDI, etc.)

### Consistency Tips

- **Take breaks** every 1-2 hours to maintain consistency
- **Re-listen to anchors** periodically (files you've already labeled)
- **Compare similar files** when uncertain
- **Use the middle of the range** (0.4-0.6) when truly uncertain
- **Avoid extremes** (0.0, 1.0) unless clearly justified

---

## Inter-Rater Reliability Protocol

### Initial Calibration
1. **Both experts label same 5 files**
2. **Compare results**
3. **Discuss disagreements** (differences > 0.2)
4. **Refine understanding** of scale interpretations
5. **Re-calibrate if needed**

### Target Agreement
- **Continuous parameters:** Mean absolute difference < 0.15
- **Categorical parameters:** Exact agreement > 80%

### Ongoing Checks
- **Every 10 files:** Experts swap and re-label 1 file
- **Compare results** to ensure no drift
- **Adjust if consistency degrading**

---

## Common Pitfalls to Avoid

### 1. **Genre Bias**
- Don't assume jazz = complex, pop = simple
- Judge each piece on its own merits

### 2. **Halo Effect**
- Don't let overall impression color all parameters
- Judge each parameter independently

### 3. **Range Restriction**
- Use the **full range** [0.0 - 1.0], not just [0.3 - 0.7]
- Extremes (0.0, 1.0) are valid for clear cases

### 4. **Personal Preference**
- Be **objective**, not subjective
- Dislike of a genre shouldn't affect labels

### 5. **Overthinking**
- Trust your musical expertise
- If struggling for >2 minutes, move on and return later

---

## Special Cases

### Highly Variable Pieces
- **Rate the predominant character**
- Note variability in comments

### Multi-Genre Pieces
- **Choose primary genre**
- Label genre-specific params for primary only

### Corrupted/Problematic Files
- **Flag for removal**
- Don't waste time on broken files

---

## Contact & Support

If you encounter issues or have questions during labeling:
1. **Email:** agent03@midigen.ai
2. **Slack:** #labeling-support
3. **Office Hours:** Mon/Wed/Fri 2-4pm

---

## Appendix: Quick Reference Card

| Parameter | Type | Range | Key Question |
|-----------|------|-------|--------------|
| energy.level | Continuous | 0-1 | How intense/energetic? |
| complexity.overall | Continuous | 0-1 | How musically complex? |
| harmony.tension | Continuous | 0-1 | How dissonant? |
| harmony.progression_predictability | Continuous | 0-1 | How formulaic? |
| melody.contour_smoothness | Continuous | 0-1 | How stepwise vs leaps? |
| jazz.bebop_vocabulary | Continuous | 0-1 | How much bebop language? |
| classical.counterpoint | Continuous | 0-1 | How contrapuntal? |
| rock.riff_repetition | Continuous | 0-1 | How riff-based? |
| electronic.filter_movement | Continuous | 0-1 | How much timbral variation? |

**Remember:** When in doubt, use 0.5 and add a note explaining your uncertainty.

---

**Thank you for your expertise and careful attention to detail!**
