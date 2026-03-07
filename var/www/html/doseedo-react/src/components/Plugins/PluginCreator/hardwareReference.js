/**
 * Hardware Reference Database — comprehensive profiles of classic audio hardware
 * and plugin category patterns for the Design Director AI.
 *
 * Used by expertPrompts.js to guide accurate hardware emulation and plugin generation.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CLASSIC HARDWARE COMPRESSOR PROFILES
// ═══════════════════════════════════════════════════════════════════════════════

export const COMPRESSOR_PROFILES = `
### Teletronix LA-2A
- faceplateColor: #C8C0B4 (brushed aluminum silver-gray)
- bgColor: #B8B0A4, accentColor: #8B0000, textColor: #2A2520
- layout: horizontal 2:1 (~704x330)
- meterType: vu-needle, large centered ~300x155, VISUAL HERO
- meterLabel: GAIN REDUCTION
- controls: knob "PEAK REDUCTION" (left, large), knob "GAIN" (right, large), switch "LIMIT/COMPRESS" (toggle)
- knobStyle: flux "vintage 1960s dark brown bakelite compressor knob with flat pointer tab indicator, worn aged patina, top-down view, black background"
- buttonStyle: vintage-toggle
- distinctiveFeatures: Only 2 knobs + 1 switch. Large VU meter dominates. Brushed aluminum faceplate. No attack/release controls. Most minimalist compressor ever.
- dsp: compressor (optical), threshold -60..0dB, ratio 3:1(compress)/inf:1(limit), attack ~10ms fixed, release 60ms-5s program-dependent

### UREI LA-3A
- faceplateColor: #C0C0C0 (brighter/cooler silver than LA-2A)
- bgColor: #B0B0B0, accentColor: #1A1A8B (dark blue), textColor: #1A1A1A
- layout: horizontal 2:1 (~704x330)
- meterType: vu-needle, large centered
- controls: knob "PEAK REDUCTION" (left, large), knob "GAIN" (right, large), switch "LIMIT/COMPRESS"
- knobStyle: flux "1970s silver-topped black plastic knob with clear plastic collar ring and white pointer line, solid-state hi-fi style, top-down view, black background"
- buttonStyle: vintage-toggle
- distinctiveFeatures: Cooler silver faceplate vs LA-2A. Black plastic knobs with silver caps (not bakelite). Solid-state, no tubes. Same T4B opto cell. Faster/more aggressive than LA-2A.
- dsp: compressor (optical), attack ~0.5ms (faster than LA-2A), release 60ms-5s

### UREI 1176LN Rev D (Blackface)
- faceplateColor: #1A1A1A (black anodized aluminum)
- bgColor: #111111, accentColor: #C0C0C0 (silver), textColor: #E0E0E0
- layout: horizontal 2.5:1 (~800x330)
- meterType: vu-needle, large upper-center, white face, UREI logo above
- controls: knob "INPUT" (large), knob "OUTPUT" (large), knob "ATTACK" (smaller, 20us-800us), knob "RELEASE" (smaller, 50ms-1100ms), button "20:1", button "12:1", button "8:1", button "4:1" (ratio row), button "OFF"/"GR"/"+4"/"+8" (meter row)
- knobStyle: flux "black plastic audio knob with silver metal top cap and clear plastic collar ring, white pointer line, 1970s studio hardware, top-down view, black background"
- buttonStyle: vintage-push (black rectangular pushbuttons in horizontal rows)
- distinctiveFeatures: Iconic black faceplate. 4 ratio pushbuttons (all-buttons-in trick). 2 large + 2 small knobs. 8 pushbuttons below meter. Attack/release counterintuitive (CCW=faster).
- dsp: compressor (FET), ratio 4:1/8:1/12:1/20:1/all-buttons, attack 20us-800us, release 50ms-1100ms

### UREI 1176 Rev A (Blue Stripe)
- faceplateColor: #C8C8C8 (brushed silver, NOT black)
- bgColor: #B0B0B0, accentColor: #2B5DAA (blue stripe around meter), textColor: #1A1A1A
- layout: horizontal 2.5:1 (~800x330)
- meterType: vu-needle, large upper-center, Weston meter with blue surround
- controls: same as blackface but with red power LED indicator
- knobStyle: flux "1960s black plastic knob with silver top cap and clear collar ring, white pointer line, early recording hardware, top-down view, black background"
- distinctiveFeatures: SILVER faceplate. Blue stripe painted around VU meter. Weston meter (older style). Red power LED. No "LN" suffix. Warmer/more colored than blackface.

### Fairchild 660
- faceplateColor: #1C1C1C (dark gray-black painted steel)
- bgColor: #181818, accentColor: #D4A040 (warm gold), textColor: #E8E0D0 (cream)
- layout: vertical 1:1.3 (~500x650), TALL standalone chassis, NOT standard rackmount
- meterType: vu-needle, large upper-center, cream face
- meterLabel: GAIN REDUCTION
- controls: knob "INPUT GAIN" (large bakelite), knob "THRESHOLD" (large bakelite), switch "TIME CONSTANT" (6-pos rotary: 0.2ms/0.3s, 0.2ms/0.8s, 0.4ms/2s, 0.4ms/5s, auto, auto-long)
- knobStyle: flux "1950s dark brown bakelite pointer knob with brass shaft, General Radio style, large diameter, flat pointer tab, vintage lab equipment, top-down view, black background"
- distinctiveFeatures: Single-channel. Dark painted steel. Large General Radio bakelite knobs. 6-position TIME CONSTANT (no separate attack/release). 14 vacuum tubes. Variable-mu topology.

### Fairchild 670
- faceplateColor: #1C1C1C, bgColor: #181818, accentColor: #D4A040, textColor: #E8E0D0
- layout: horizontal 2.5:1 (~900x400), massive 6U rackmount
- meterType: vu-needle, TWO large VU meters side by side (L/R), cream faces
- controls: per-channel: knob "INPUT GAIN", knob "THRESHOLD", switch "TIME CONSTANT" (6-pos), plus switch "AGC" (lateral/vertical M/S mode)
- distinctiveFeatures: DUAL-CHANNEL with lateral/vertical M/S mode. 20 tubes, 65 lbs. Dual VU meters as centerpiece. Symmetrical L/R controls. Holy grail compressor.

### SSL G-Series Bus Compressor
- faceplateColor: #2A2A2A (dark charcoal), bgColor: #222222, accentColor: #4CAF50 (SSL green), textColor: #FFFFFF
- layout: vertical 1:2 (~350x550), narrow console module strip
- meterType: led-bar, vertical LED GR meter (green/yellow/red), narrow ~30x120
- controls: knob "THRESHOLD" (-15..+20dBu), knob "MAKE-UP" (-5..+15dB), switch "RATIO" (2:1/4:1/10:1), switch "ATTACK" (6-pos: 0.1/0.3/1/3/10/30ms), switch "RELEASE" (5-pos: 0.1/0.3/0.6/1.2s/Auto), button "IN" (green LED)
- knobStyle: flux "small dark gray SSL console knob with white pointer, black cap, thin profile, top-down view, black background"
- buttonStyle: ssl-illuminated (small rectangular backlit pushbutton, green when engaged)
- distinctiveFeatures: Vertical strip module. Stepped attack/release. Only 3 ratio choices. Green "IN" button = SSL signature. VCA topology = transparent/punchy. THE mix bus standard.

### dbx 160 (Original)
- faceplateColor: #F0F0F0 (white/off-white), bgColor: #E8E8E8, accentColor: #8B4513 (wood cheeks), textColor: #1A1A1A
- layout: half-rack ~1:1.2 (~400x480), wood side panels
- meterType: vu-needle, large upper-half, white face
- controls: knob "COMPRESSION" (ratio 1:1 to infinity:1 continuous!), knob "THRESHOLD", knob "OUTPUT GAIN" (-20..+20dB), led "ABOVE"/"BELOW" (threshold indicators)
- knobStyle: flux "1970s large silver aluminum knob with black pointer line, machined metal, hi-fi style, top-down view, white background"
- distinctiveFeatures: WHITE face + wood cheeks. Continuously variable ratio 1:1→inf:1 (unique). NO attack/release controls. ABOVE/BELOW threshold LEDs. Half-rack. Three large knobs below large VU meter.

### API 2500
- faceplateColor: #1A1A1A (matte black), bgColor: #111111, accentColor: #2B5DAA (API blue), textColor: #FFFFFF
- layout: horizontal 3:1 (~900x300), 1U rackmount — wide and slim
- meterType: vu-needle, TWO VU meters side by side (L/R), white faces
- controls: knob "THRESHOLD", knob "ATTACK", knob "RELEASE" (+auto), knob "RATIO" (variable), knob "OUTPUT", switch "KNEE" (Hard/Medium/Soft), switch "THRUST" (sidechain tilt EQ), switch "TYPE" (Old feedback/New feedforward), switch "L/R LINK" (IND/50/60/70/80/100%)
- distinctiveFeatures: Matte black + blue side panels. Dual VU meters. THRUST (unique sidechain tilt EQ). TYPE switch toggles feedback/feedforward. Variable stereo link. Dense 1U layout. API blue LED.

### Empirical Labs Distressor (EL8)
- faceplateColor: #2C2C2C (dark charcoal), bgColor: #1E1E1E, accentColor: #FFD700 (gold), textColor: #FFFFFF
- layout: horizontal 2.5:1 (~800x320), 1U rackmount
- meterType: led-bar, horizontal 16-segment LED GR meter (green/yellow/red), center-upper
- controls: knob "INPUT" (large white, 0-10.5), knob "ATTACK" (large white), knob "RELEASE" (large white), knob "OUTPUT" (large white), button "RATIO" (cycles 1:1/2:1/3:1/4:1/6:1/10:1/20:1/NUKE), button "DIST 2" (tape-like), button "DIST 3" (tube-like)
- knobStyle: flux "large white plastic knob with black numbering to 10.5, oversized, Distressor style, top-down view, dark gray background"
- distinctiveFeatures: 4 LARGE white knobs on black. Numbering goes to 10.5 (Spinal Tap ref). Horizontal LED GR meter. 8 ratios including NUKE. DIST 2/3 harmonic distortion modes. Most versatile compressor ever.

### Manley Variable Mu
- faceplateColor: #4A4A5E (pewter-gray with blue tint), bgColor: #3E3E50, accentColor: #C8A050 (warm gold), textColor: #C0C0C0
- layout: horizontal 3:1 (~900x330), 2U rackmount
- meterType: vu-needle, TWO large illuminated Sifam VU meters (L/R), cream faces — visual heroes
- controls: per-channel stepped: knob "INPUT" (5-pos), knob "THRESHOLD", knob "ATTACK" (11-pos 15-90ms), knob "RECOVERY" (5-pos: 0.2/0.4/0.6/4/8s), knob "OUTPUT" (24-pos half-dB steps), switch "COMPRESS/LIMIT", switch "LINK", switch "HP SC"
- knobStyle: flux "precision stepped detented knob, dark gray metal with silver pointer arrow, CNC machined, mastering equipment, top-down view, dark gray background"
- distinctiveFeatures: CNC 1/4" thick steel faceplate. TWO illuminated Sifam VU meters. ALL stepped/detented for mastering recall. Pewter-blue faceplate. Variable-mu tube topology.

### Neve 33609
- faceplateColor: #5B6B7A (Neve "RAF blue-gray"), bgColor: #4A5A68, accentColor: #2A3A48, textColor: #FFFFFF
- layout: horizontal 3:1 (~900x300), 1U or 2U rackmount
- meterType: vu-needle, single center meter, white face
- controls (COMPRESSOR): knob "THRESHOLD" (stepped), switch "RATIO" (1.5:1/2:1/3:1/4:1/6:1), knob "RECOVERY", knob "MAKE UP"
- controls (LIMITER): knob "THRESHOLD", knob "RECOVERY"
- distinctiveFeatures: RAF blue-gray = unmistakably Neve. SEPARATE compressor + limiter sections. Diode bridge topology (unique Neve crunch). All stepped/detented.

### Tube-Tech CL 1B
- faceplateColor: #6BAED6 (sky/powder blue — signature Tube-Tech), bgColor: #5A9BC5, accentColor: #FFFFFF, textColor: #1A1A2E
- layout: horizontal 2.5:1 (~800x330), 2U rackmount
- meterType: vu-needle, single center-right, cream face
- controls: knob "THRESHOLD", knob "RATIO" (2:1-10:1 continuous), knob "ATTACK" (0.5-300ms), knob "RELEASE" (50ms-10s), knob "GAIN" (0..+30dB), switch "ATTACK MODE" (Fixed/Manual/Fixed+Manual), switch "RELEASE MODE" (Fixed/Manual/Fixed+Manual)
- knobStyle: flux "large dark navy blue knob with white pointer, smooth turned metal, Scandinavian precision audio, clean elegant, top-down view, light blue background"
- distinctiveFeatures: LIGHT BLUE faceplate — unique in pro audio. Scandinavian design. More controls than LA-2A. Unique ATTACK/RELEASE MODE switches (blend opto behavior with manual timing). Optical + tube.
`;

// ═══════════════════════════════════════════════════════════════════════════════
// CLASSIC HARDWARE EQ PROFILES
// ═══════════════════════════════════════════════════════════════════════════════

export const EQ_PROFILES = `
### Pultec EQP-1A
- faceplateColor: #4A6A8A (steel blue-grey, "Pultec blue")
- bgColor: #1A1A2E, accentColor: #C9B458 (warm gold), textColor: #F5F0E1 (cream)
- layout: horizontal 6:1 (~800x240), wide 3U rackmount
- meterType: none
- controls: knob "LOW FREQ BOOST" (0-10), switch "LOW FREQ" (20/30/60/100 Hz), knob "LOW FREQ ATTEN" (0-10), knob "HIGH FREQ BOOST" (0-10), switch "HIGH FREQ" (3/4/5/8/10/12/16 kHz), knob "HIGH FREQ BANDWIDTH", knob "HIGH FREQ ATTEN" (0-10), switch "ATTEN FREQ SEL" (5/10/20 kHz), toggle "IN/OUT"
- knobStyle: flux "large cream ivory bakelite chicken-head pointer knob with black indicator line, heavy weighted, vintage tube EQ, top-down view, dark blue background"
- distinctiveFeatures: Simultaneous boost AND attenuate on low band ("Pultec trick"). Wide spacious panel. Cream-on-blue. Three sections: low shelf / high peak / high cut. Tube passive EQ.
- dsp: lowshelf boost + lowshelf cut + highpeak + highshelf cut, tube saturation

### Pultec MEQ-5
- faceplateColor: #4A6A8A, bgColor: #1A1A2E, accentColor: #C9B458, textColor: #F5F0E1
- layout: horizontal 6:1 (~800x200), 2U rack
- controls: knob "LOW MID PEAK" (0-10), switch "LOW MID FREQ" (200-1000 Hz), knob "MID DIP" (0-10), switch "MID DIP FREQ" (200-5000 Hz), knob "HIGH MID PEAK" (0-10), switch "HIGH MID FREQ" (1.5-5 kHz)
- distinctiveFeatures: Midrange-only companion to EQP-1A. Three mid sections: two boost flanking one cut. Same Pultec blue aesthetic.

### Neve 1073
- faceplateColor: #2C405C (Neve 1073 dark blue-grey)
- bgColor: #0D1117, accentColor: #D4A843 (warm gold), textColor: #FFFFFF
- layout: vertical channel strip 1:4 (~200x700)
- meterType: none
- controls: switch "MIC GAIN" (stepped -80 to -20dB, maroon knob cap), knob "HF SHELF" (+/-16dB at 12kHz), switch "MID FREQ" (0.36-7.2 kHz stepped), knob "MID GAIN" (+/-18dB), switch "LF FREQ" (35/60/110/220 Hz), knob "LF GAIN" (+/-16dB shelf), switch "HPF" (OFF/50/80/160/300 Hz)
- knobStyle: flux "grey fluted plastic Neve 80-series knob with white pointer line, concentric for freq/gain. Maroon/red cap on mic gain knob, top-down view, dark blue background"
- distinctiveFeatures: Iconic dark blue-grey vertical strip. Only 3 EQ bands but transformer-coupled Class-A. Maroon mic gain knob instantly recognizable. White legends on battleship grey-blue paint.

### Neve 1081
- faceplateColor: #2C405C, bgColor: #0D1117, accentColor: #D4A843, textColor: #FFFFFF
- layout: vertical channel strip 1:5 (~200x800), taller than 1073
- controls: 4 dual-concentric band controls (outer=freq, inner=gain, +/-18dB each), plus HPF and LPF cut filters, Hi-Q switches on mid bands, shelf/bell switches on HF/LF
- distinctiveFeatures: 4-band with dual-concentric controls. More versatile than 1073. Hi-Q switches for surgical mids. Both HPF and LPF. Same Neve blue-grey livery.

### SSL E-Series Channel EQ (4000E)
- faceplateColor: #D2C9B8 (warm grey-cream console surface)
- bgColor: #1C1C1C, accentColor: #E8A030 (SSL amber), textColor: #1A1A1A
- layout: vertical channel strip 1:3 (~250x600)
- controls: knob "HF GAIN" + knob "HF FREQ" (1.5-16kHz), button "HF BELL", knob "HMF GAIN" + "HMF FREQ" (600Hz-7kHz) + "HMF Q", knob "LMF GAIN" + "LMF FREQ" (200Hz-2.5kHz) + "LMF Q", knob "LF GAIN" + "LF FREQ" (30-450Hz), button "LF BELL", switch "HPF"/"LPF", button "EQ IN"
- knobStyle: flux "small black SSL pointer knob with white indicator line, closely spaced vertical column, console aesthetic, top-down view, grey background"
- distinctiveFeatures: Black/Brown EQ variants define the SSL sound. 4-band with two fully parametric mids. Dense vertical column of small knobs. Console-integrated appearance.

### API 550A
- faceplateColor: #1A1A1A (black anodized), bgColor: #0A0A0A, accentColor: #3A7BD5 (API blue), textColor: #3A7BD5
- layout: vertical 500-series 1:3.5 (~150x500)
- controls: stepped "HF SELECT" (7 freqs), stepped "HF GAIN" (+/-12dB 2dB steps), stepped "MF SELECT" (7 freqs), stepped "MF GAIN", stepped "LF SELECT" (7 freqs), stepped "LF GAIN", toggle "HF PEAK/SHELF", toggle "LF PEAK/SHELF"
- knobStyle: "API aluminum pointer knobs with colored plastic cap inserts — red, blue, green caps on three bands"
- distinctiveFeatures: Proportional Q (bandwidth narrows with more boost). Colored knob caps on black with blue legends = iconic API. 500-series. Reciprocal EQ.

### API 560
- faceplateColor: #1A1A1A, accentColor: #3A7BD5, textColor: #3A7BD5
- layout: vertical 500-series 1:3.5 (~150x500)
- controls: 10 vertical sliders (+/-12dB each): 31/63/125/250/500/1k/2k/4k/8k/16k Hz
- distinctiveFeatures: 10 sliders = visible EQ curve shape. Proportional Q per band. Blue-on-black API. 500-series graphic EQ.

### Maag EQ4
- faceplateColor: #1E3A5F (Maag dark blue), bgColor: #0D1B2A, accentColor: #5BC0EB, textColor: #FFFFFF
- layout: vertical 500-series 1:3.5
- controls: knob "SUB 10Hz", knob "40Hz", knob "160Hz", knob "650Hz", knob "2.5kHz" (all +/-15dB), switch "AIR BAND FREQ" (OFF/2.5/5/10/15/20/40 kHz), knob "AIR GAIN" (0..+20dB boost only)
- distinctiveFeatures: AIR BAND is the signature — HF shelf up to 40kHz. 6 fixed-frequency bands. Modern blue with white text.
`;

// ═══════════════════════════════════════════════════════════════════════════════
// CLASSIC HARDWARE EFFECTS PROFILES
// ═══════════════════════════════════════════════════════════════════════════════

export const EFFECTS_PROFILES = `
### Lexicon 480L
- faceplateColor: #1A1A1A (black rack), bgColor: #0A0A0A, accentColor: #CC2222 (red LED), textColor: #CC2222
- layout: horizontal 4:1 (~800x200) for LARC remote controller
- meterType: led-bar (dual 16-segment LED headroom meters)
- controls: 6 vertical sliders (assignable params), numeric button row 0-9, buttons BANK/PROG/VAR/REG/PAGE/MUTE/STO, 2-line red LED display, 6 slider-label LED displays
- distinctiveFeatures: LARC wedge-shaped black controller with 6 sliders + red LED displays. Red-on-black LED is the visual signature. Algorithms: Reverb/Ambience/Random/Twin Delays.
- dsp: digital reverb with pre-delay, size, diffusion, HF damping, bass multiply

### EMT 140
- faceplateColor: #3A3A3A (dark grey metal remote), bgColor: #1A1A1A, accentColor: #4A8C3F (vintage green), textColor: #E0E0E0
- meterType: vu-needle (reverb time in seconds)
- controls: button "DECAY +"/"-" (drives damper motor), meter "REVERB TIME" (0.5-5.5s), knob "INPUT LEVEL", knob "OUTPUT LEVEL", switch "BASS CUT" (OFF/-4/-10/-16dB at 80Hz), knob "PRE-DELAY", knob "MIX"
- distinctiveFeatures: Small remote panel controls a massive hidden 2m x 1m steel plate. VU meter shows decay time in seconds. Industrial German broadcast aesthetic. "Controlling a massive hidden machine" concept.
- dsp: plate reverb, decay 0.5-5.5s, pre-delay 0-100ms, bass cut filter

### Roland RE-201 Space Echo
- faceplateColor: #2F4F2F (dark olive green metallic)
- bgColor: #1A2A1A, accentColor: #C0C0C0 (chrome), textColor: #E8E4D0 (cream)
- layout: horizontal tabletop 3:1 (~700x250)
- controls: knob "REPEAT RATE" (tape speed/delay time), knob "INTENSITY" (feedback, self-oscillation at max), knob "ECHO VOLUME", knob "REVERB VOLUME", switch "MODE SELECTOR" (large central 12-position chrome rotary: modes 1-12 combining 3 tape heads + spring reverb), knob "BASS"/"TREBLE" (echo EQ)
- distinctiveFeatures: Large central MODE SELECTOR (12 positions). Olive-green panel with chrome hardware. Tape echo + spring reverb combo. Warm degrading tape tone.
- dsp: tape delay (3 heads, variable speed) + spring reverb, feedback loop

### Eventide H3000
- faceplateColor: #2A3A5A (dark steel blue), bgColor: #0D1117, accentColor: #D4A843 (amber LCD), textColor: #D4A843
- layout: horizontal 5:1 (~900x200), 2U rack
- meterType: led-bar (input level LED meters)
- controls: knob "INPUT", encoder "JOG WHEEL" (large data entry), keypad 0-9, buttons PROGRAM/PARAMETER/SOFT KEY 1-4/BYPASS/COMPARE/STORE, LCD display
- distinctiveFeatures: Large LCD + jog wheel dominate. Known for pitch shifting, harmonizing, micro-pitch detune. 1980s digital rack aesthetic. Grey-on-blue or yellow-on-black variants.

### Ibanez Tube Screamer TS808
- faceplateColor: #2D8C3C (iconic green)
- bgColor: #1A3A1A, accentColor: #F5F5F5, textColor: #F5F5F5
- layout: stomp pedal 1.3:1 (~300x350)
- controls: knob "OVERDRIVE", knob "TONE", knob "LEVEL", footswitch "BYPASS", led green
- knobStyle: "small black plastic knobs in triangular pattern on green enclosure"
- distinctiveFeatures: Unmistakable green color = instant recognition. JRC4558D warm clipping. Three-knob simplicity. Mid-hump frequency response.
- dsp: overdrive (soft-clipping), drive + tone (RC low-pass) + output level

### Klon Centaur
- faceplateColor: #C4993B (gold/bronze metallic)
- bgColor: #2A1F0F, accentColor: #8B1A1A (oxblood), textColor: #1A1A1A
- layout: stomp pedal 1.5:1 (~350x300)
- controls: knob "GAIN", knob "TREBLE", knob "OUTPUT", footswitch, led
- knobStyle: "chrome-topped metal knobs with knurled shafts, horizontal row, chrome on gold is signature contrast"
- distinctiveFeatures: Gold enclosure + hand-drawn centaur graphic. Clean-blend overdrive (transparent at low gain). Only ~8000 made. Legendary collector value.
- dsp: overdrive (germanium diode hard-clipping with clean blend), gain-controlled mix of clean + clipped

### ProCo RAT
- faceplateColor: #1A1A1A (flat black), bgColor: #0A0A0A, accentColor: #B8FF00 (glow-in-dark yellow-green), textColor: #B8FF00
- layout: stomp pedal with sloped face 1.5:1 (~350x300)
- controls: knob "DISTORTION", knob "FILTER" (NOTE: works BACKWARDS — CW=darker), knob "VOLUME", footswitch
- distinctiveFeatures: Glow-in-dark graphics on flat black. FILTER reversed (CW=dark). Sloped enclosure. LM308 op-amp. Can do overdrive/distortion/fuzz from one circuit.
- dsp: distortion (hard-clipping), LM308 gain + diodes to ground + reverse low-pass filter

### Electro-Harmonix Small Stone
- faceplateColor: #E87830 (bright orange)
- bgColor: #1A1A1A, accentColor: #F5D020 (yellow), textColor: #1A1A1A
- layout: stomp pedal 2:1 (~350x200)
- controls: knob "RATE", switch "COLOR" (normal/feedback), footswitch
- distinctiveFeatures: Minimalist 2-control. COLOR switch adds resonant feedback. Bright orange + psychedelic graphics. EHX lightning bolt logo.
- dsp: phaser (4-stage OTA), rate + optional feedback

### MXR Flanger
- faceplateColor: #5A5A5A (metallic grey), bgColor: #1C1C1C, accentColor: #CC2222 (red LED), textColor: #FFFFFF
- layout: stomp pedal 1.5:1 (~350x280)
- controls: knob "MANUAL" (center freq), knob "WIDTH" (sweep range), knob "SPEED" (LFO rate), knob "REGEN" (feedback), footswitch
- distinctiveFeatures: All-analog BBD. MANUAL knob at zero = static comb-filter. Metallic grey + white graphics. 18V for headroom.
- dsp: flanger (BBD), manual offset + LFO width/speed + regeneration feedback

### Boss CE-1 Chorus Ensemble
- faceplateColor: #B8B8B8 (silver-grey brushed metal)
- bgColor: #2A2A2A, accentColor: #2A5CAA (Boss blue), textColor: #1A1A1A
- layout: large floor pedal 2.5:1 (~500x220)
- controls: knob "LEVEL", knob "CHORUS INTENSITY", knob "VIBRATO DEPTH", knob "VIBRATO RATE", footswitch "NORMAL/EFFECT", footswitch "CHORUS/VIBRATO"
- distinctiveFeatures: Larger than typical pedals. Silver brushed metal. Two footswitches. First standalone chorus pedal (from Jazz Chorus amp circuit).
- dsp: chorus/vibrato (BBD), intensity + depth/rate, stereo output
`;

// ═══════════════════════════════════════════════════════════════════════════════
// COMMON PLUGIN CATEGORY PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════

export const PLUGIN_CATEGORY_PATTERNS = `
## Plugin Category UI Patterns

When designing plugins in these categories, follow these conventions:

### DYNAMICS

**Compressor (generic)**
- requiredControls: knob Threshold, knob Ratio, knob Attack, knob Release, knob Makeup/Output, meter GR (led-bar or vu-needle)
- optionalControls: knob Mix (parallel), knob Knee, button Sidechain HPF, button Auto-Release, button Bypass
- layout: horizontal, meter on right or top. 400-700px wide, 300-400px tall
- paramRanges: threshold -60..0dB, ratio 1:1..20:1, attack 0.01..100ms, release 10..1000ms, makeup 0..30dB, mix 0..100%
- antiPatterns: never omit GR meter, never make attack/release linear (use skew/log)

**Limiter / Brickwall**
- requiredControls: knob Input/Gain, knob Ceiling/Output, knob Release, meter GR (led-bar), meter Output Level
- optionalControls: knob Lookahead, button True Peak, button Auto-Release, button Link
- layout: horizontal, dual meters (GR + output) prominent. Simple, few controls.
- paramRanges: input -12..+24dB, ceiling -6..0dB, release 0.01..500ms
- antiPatterns: too many controls (limiters should be simple), missing output meter

**Gate / Noise Gate**
- requiredControls: knob Threshold, knob Attack, knob Hold, knob Release, knob Range/Floor, led GR indicator
- optionalControls: knob Sidechain HPF/LPF, button Duck mode, button Sidechain Listen
- layout: horizontal, compact. 400-550px wide.
- paramRanges: threshold -80..0dB, attack 0.01..50ms, hold 0..500ms, release 5..2000ms, range -80..0dB

**De-esser**
- requiredControls: knob Threshold, knob Frequency (2-16kHz), knob Range/Amount, button Split/Wideband mode, meter GR
- optionalControls: knob Listen (solo sidechain band), button Male/Female preset
- layout: compact horizontal. Very few controls. 350-500px wide.

**Multiband Compressor**
- requiredControls: per-band (3-4 bands): knob Threshold, knob Ratio, knob Attack, knob Release, knob Gain, slider Crossover freq. Plus master output knob.
- meterType: per-band led-bar GR meters side by side
- layout: horizontal with vertical band sections side by side, OR tabbed per band. 700-900px wide, 400-500px tall.
- paramRanges: crossover 20Hz-20kHz (log), per-band same as compressor

**Transient Shaper**
- requiredControls: knob Attack (+/-), knob Sustain (+/-), knob Output, meter Output Level
- optionalControls: knob Mix, button Listen (to transient/sustain), knob Sensitivity
- layout: very compact. 350-450px wide. Few controls = simple design.

### EQ & FILTERING

**Parametric EQ**
- requiredControls: per-band (4-6 bands): knob Frequency, knob Gain (+/-dB), knob Q/Width, button Band On/Off, button Shelf/Bell mode (for top/bottom bands). Plus spectrum analyzer display.
- meterType: spectrum-analyzer display (waveform or eq-curve component)
- layout: horizontal with spectrum display spanning top, band controls below. 600-900px wide, 400-500px tall.
- paramRanges: freq 20Hz-20kHz (log), gain +/-18dB, Q 0.1-10

**Graphic EQ**
- requiredControls: vertical sliders for each band (7-31 bands), button Bypass
- layout: horizontal, all sliders side by side. Width depends on band count. 600-1000px wide.
- distinctiveConvention: sliders create visible frequency curve shape

**Channel Strip (EQ + Dynamics + Gain)**
- requiredControls: HPF, 3-4 band EQ, compressor section (threshold/ratio/attack/release), gate section (threshold), slider Output Fader, meter Output Level
- layout: vertical like a console strip, OR horizontal sections. 300-500px wide vertical, or 700-900px horizontal.
- distinctiveConvention: signal flow top-to-bottom (vertical) or left-to-right (horizontal)

### TIME-BASED

**Reverb (Algorithmic)**
- requiredControls: knob Size/Room, knob Decay/Time, knob Pre-Delay, knob Damping/HF, knob Mix/Wet, knob Width
- optionalControls: knob Early Reflections, knob Diffusion, knob Modulation, button Freeze, dropdown Algorithm type (Hall/Room/Plate/Chamber)
- layout: horizontal, 500-700px wide. Display showing reverb shape optional.

**Reverb (Convolution)**
- requiredControls: dropdown IR Select, knob Decay, knob Pre-Delay, knob Mix, knob Size/Stretch
- optionalControls: knob LP Filter, knob HP Filter, waveform IR display
- layout: horizontal with waveform display. 500-700px wide.

**Delay (Digital)**
- requiredControls: knob Time (ms or tempo-synced), knob Feedback, knob Mix, knob Tone/Filter
- optionalControls: knob Stereo Width, button Ping-Pong, button Tempo Sync, knob Modulation
- layout: horizontal, compact. 400-600px wide.
- paramRanges: time 1-2000ms, feedback 0-95%, mix 0-100%

**Delay (Tape/Analog)**
- requiredControls: knob Time/Rate, knob Feedback/Intensity, knob Mix, knob Tone, knob Wow/Flutter, knob Saturation/Drive
- optionalControls: knob Age/Condition, button Tape Speed, waveform display
- layout: horizontal, warm vintage aesthetic. 500-700px wide. Use vintage-analog styles.

### MODULATION

**Chorus**
- requiredControls: knob Rate, knob Depth, knob Mix, knob Width (stereo)
- optionalControls: knob Delay, knob Feedback, button Mono/Stereo, dropdown Voices
- layout: compact horizontal. 350-500px wide.

**Phaser**
- requiredControls: knob Rate, knob Depth, knob Feedback/Resonance, knob Mix
- optionalControls: knob Stages (4/6/8/12), knob Stereo
- layout: compact, very few controls. 300-450px wide.

**Flanger**
- requiredControls: knob Rate, knob Depth, knob Feedback/Regen, knob Manual (center freq), knob Mix
- layout: compact horizontal. 350-500px wide.

**Tremolo / Auto-Pan**
- requiredControls: knob Rate, knob Depth, dropdown Shape (sine/triangle/square), knob Stereo/Pan
- layout: very compact. 300-400px wide. Waveform display showing shape optional.

### DISTORTION & SATURATION

**Overdrive**
- requiredControls: knob Drive/Gain, knob Tone, knob Level/Output
- layout: stomp pedal style — compact, ~300-400px wide, 3-4 knobs max
- colorConventions: warm colors (green for TS808-style, gold for Klon-style)

**Distortion**
- requiredControls: knob Distortion/Gain, knob Tone/Filter, knob Volume/Level
- layout: stomp pedal style, compact. Dark/aggressive colors.

**Tape Saturation**
- requiredControls: knob Input/Drive, knob Saturation, knob Output, knob Bias, knob Wow/Flutter
- optionalControls: dropdown Tape Speed (7.5/15/30 ips), knob Hiss, button Noise
- layout: horizontal, warm vintage aesthetic. 450-600px wide.
- colorConventions: warm browns, cream, aged metal

**Bitcrusher / Lo-Fi**
- requiredControls: knob Bit Depth (1-24), knob Sample Rate (100Hz-44.1kHz), knob Mix
- optionalControls: knob Jitter, knob Noise, knob Filter
- layout: compact, digital/retro aesthetic.

### UTILITY

**Stereo Imager / Widener**
- requiredControls: knob Width (mono→wide), meter Correlation/Phase, meter L/R level
- optionalControls: knob Bass Mono (below freq), knob Mid/Side balance
- layout: compact with prominent metering. 350-500px wide.

**Metering (LUFS / Peak / RMS)**
- requiredControls: meter Loudness (large, LUFS), meter True Peak (L/R), meter Short-term, meter Integrated, label Loudness Range
- layout: vertical or horizontal, meters dominate. 300-500px wide, 400-600px tall.
- distinctiveConvention: meters are the ENTIRE UI, minimal controls

**Tuner**
- requiredControls: meter Pitch display (large), label Note Name (large), label Cents offset, led Sharp/Flat indicators
- layout: compact, single large display. 300x250 approx.
`;

// ═══════════════════════════════════════════════════════════════════════════════
// BUILD COMPLETE REFERENCE STRING FOR PROMPT INJECTION
// ═══════════════════════════════════════════════════════════════════════════════

export function buildHardwareReference() {
  return `
## HARDWARE REFERENCE DATABASE

When the user requests a plugin based on specific hardware, use these EXACT profiles for visual accuracy.
Match the faceplate color, control layout, meter type, knob style, and distinctive features precisely.

### COMPRESSORS / DYNAMICS
${COMPRESSOR_PROFILES}

### EQUALIZERS
${EQ_PROFILES}

### EFFECTS (Reverb, Delay, Modulation, Distortion)
${EFFECTS_PROFILES}

${PLUGIN_CATEGORY_PATTERNS}
`;
}
