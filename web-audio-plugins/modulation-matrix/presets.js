/**
 * Factory Presets for Modulation Matrix Plugins
 *
 * @author Agent 17 (Modulation Matrix & Advanced LFOs)
 * @version 1.0.0
 */

export const AdvancedLFOPresets = [
  {
    name: "Slow Sine Wave",
    category: "Basic",
    description: "Gentle sine wave modulation",
    parameters: {
      frequency: 0.5,
      depth: 0.8,
      phase: 0
    },
    waveform: "sine",
    bpmSync: false
  },
  {
    name: "Fast Tremolo",
    category: "Rhythmic",
    description: "Fast tremolo effect synced to tempo",
    parameters: {
      frequency: 8,
      depth: 0.6,
      phase: 0
    },
    waveform: "sine",
    bpmSync: true,
    bpm: 120,
    syncDivision: "1/8"
  },
  {
    name: "Sample & Hold Chaos",
    category: "Experimental",
    description: "Random stepped modulation",
    parameters: {
      frequency: 4,
      depth: 1.0,
      phase: 0
    },
    waveform: "samplehold",
    sampleHoldRate: 8
  },
  {
    name: "Pulsing Square",
    category: "Rhythmic",
    description: "On/off pulsing modulation",
    parameters: {
      frequency: 2,
      depth: 1.0,
      phase: 0
    },
    waveform: "square",
    bpmSync: true,
    syncDivision: "1/4"
  },
  {
    name: "Rising Sawtooth",
    category: "Basic",
    description: "Continuously rising modulation",
    parameters: {
      frequency: 0.25,
      depth: 0.7,
      phase: 0
    },
    waveform: "sawtooth"
  },
  {
    name: "Triplet Swing",
    category: "Rhythmic",
    description: "Triplet-based modulation",
    parameters: {
      frequency: 1,
      depth: 0.5,
      phase: 0
    },
    waveform: "triangle",
    bpmSync: true,
    syncDivision: "1/8T"
  },
  {
    name: "Step Sequencer",
    category: "Creative",
    description: "8-step melodic sequence",
    parameters: {
      frequency: 2,
      depth: 0.9,
      phase: 0
    },
    waveform: "step",
    stepSequence: [0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
  },
  {
    name: "Slow Wobble",
    category: "Creative",
    description: "Gentle random wobble",
    parameters: {
      frequency: 0.1,
      depth: 0.6,
      phase: 0
    },
    waveform: "random"
  }
];

export const EnvelopeGeneratorPresets = [
  {
    name: "Pluck",
    category: "Percussive",
    description: "Fast attack, quick decay",
    parameters: {
      attack: 0.001,
      decay: 0.3,
      sustain: 0,
      release: 0.1
    },
    envelopeType: "adsr",
    attackCurve: "linear",
    decayCurve: "exponential",
    releaseCurve: "exponential"
  },
  {
    name: "Pad",
    category: "Sustained",
    description: "Slow attack, long sustain",
    parameters: {
      attack: 1.5,
      decay: 0.5,
      sustain: 0.7,
      release: 2.0
    },
    envelopeType: "adsr",
    attackCurve: "logarithmic",
    decayCurve: "linear",
    releaseCurve: "exponential"
  },
  {
    name: "Piano",
    category: "Percussive",
    description: "Natural piano envelope",
    parameters: {
      attack: 0.01,
      decay: 0.8,
      sustain: 0.3,
      release: 0.5
    },
    envelopeType: "adsr",
    attackCurve: "linear",
    decayCurve: "exponential",
    releaseCurve: "exponential"
  },
  {
    name: "Organ",
    category: "Sustained",
    description: "Instant on/off like organ",
    parameters: {
      attack: 0.001,
      decay: 0.01,
      sustain: 1.0,
      release: 0.01
    },
    envelopeType: "adsr",
    attackCurve: "linear",
    decayCurve: "linear",
    releaseCurve: "linear"
  },
  {
    name: "Brass Swell",
    category: "Sustained",
    description: "Brass-like slow attack",
    parameters: {
      attack: 0.8,
      hold: 0.2,
      decay: 0.3,
      sustain: 0.8,
      release: 0.6
    },
    envelopeType: "ahdsr",
    attackCurve: "logarithmic",
    decayCurve: "linear",
    releaseCurve: "exponential"
  },
  {
    name: "Gate Pulse",
    category: "Rhythmic",
    description: "Short rhythmic pulse",
    parameters: {
      attack: 0.01,
      decay: 0.2,
      sustain: 0,
      release: 0.01
    },
    envelopeType: "adsr",
    triggerMode: "trigger"
  },
  {
    name: "Reverse Swell",
    category: "Creative",
    description: "Fade in effect",
    parameters: {
      attack: 2.0,
      decay: 0.1,
      sustain: 1.0,
      release: 0.5
    },
    envelopeType: "adsr",
    attackCurve: "exponential"
  },
  {
    name: "Punchy Bass",
    category: "Percussive",
    description: "Punchy with quick attack",
    parameters: {
      attack: 0.005,
      decay: 0.15,
      sustain: 0.4,
      release: 0.2
    },
    envelopeType: "adsr",
    velocitySensitivity: 0.7
  }
];

export const MacroControlsPresets = [
  {
    name: "Default Layout",
    category: "Utility",
    description: "Standard 8 macro configuration",
    macros: [
      { label: "Filter Cutoff", value: 0.5, midiCC: 74 },
      { label: "Resonance", value: 0.3, midiCC: 71 },
      { label: "Attack", value: 0.2, midiCC: 73 },
      { label: "Release", value: 0.5, midiCC: 72 },
      { label: "Reverb Mix", value: 0.3, midiCC: 91 },
      { label: "Delay Time", value: 0.4, midiCC: 92 },
      { label: "Drive", value: 0.0, midiCC: 93 },
      { label: "Master", value: 0.7, midiCC: 7 }
    ]
  },
  {
    name: "Synth Control",
    category: "Synthesis",
    description: "Synthesizer macro mapping",
    macros: [
      { label: "OSC Mix", value: 0.5, midiCC: 12 },
      { label: "PWM", value: 0.5, midiCC: 13 },
      { label: "Filter", value: 0.7, midiCC: 74 },
      { label: "Env Amount", value: 0.6, midiCC: 76 },
      { label: "LFO Rate", value: 0.4, midiCC: 77 },
      { label: "LFO Depth", value: 0.3, midiCC: 78 },
      { label: "Unison", value: 0.0, midiCC: 79 },
      { label: "Detune", value: 0.1, midiCC: 94 }
    ]
  },
  {
    name: "FX Chain",
    category: "Effects",
    description: "Effects processing macros",
    macros: [
      { label: "Distortion", value: 0.0, midiCC: 14 },
      { label: "Filter Freq", value: 0.5, midiCC: 74 },
      { label: "Phaser Rate", value: 0.3, midiCC: 15 },
      { label: "Chorus Depth", value: 0.4, midiCC: 16 },
      { label: "Delay FB", value: 0.5, midiCC: 17 },
      { label: "Reverb Size", value: 0.6, midiCC: 18 },
      { label: "Dry/Wet", value: 0.5, midiCC: 19 },
      { label: "Output", value: 0.8, midiCC: 7 }
    ]
  },
  {
    name: "Performance",
    category: "Live",
    description: "Live performance macros",
    macros: [
      { label: "Energy", value: 0.5, midiCC: 20 },
      { label: "Texture", value: 0.5, midiCC: 21 },
      { label: "Space", value: 0.3, midiCC: 22 },
      { label: "Movement", value: 0.4, midiCC: 23 },
      { label: "Chaos", value: 0.0, midiCC: 24 },
      { label: "Warmth", value: 0.6, midiCC: 25 },
      { label: "Brightness", value: 0.5, midiCC: 26 },
      { label: "Presence", value: 0.7, midiCC: 27 }
    ]
  },
  {
    name: "Minimal Setup",
    category: "Utility",
    description: "Essential controls only",
    macros: [
      { label: "Filter", value: 0.5, midiCC: 74 },
      { label: "Resonance", value: 0.3, midiCC: 71 },
      { label: "Envelope", value: 0.5, midiCC: 73 },
      { label: "FX Send", value: 0.3, midiCC: 91 },
      { label: "Unused 5", value: 0.0, midiCC: -1 },
      { label: "Unused 6", value: 0.0, midiCC: -1 },
      { label: "Unused 7", value: 0.0, midiCC: -1 },
      { label: "Volume", value: 0.7, midiCC: 7 }
    ]
  }
];

export const ModulationMatrixPresets = [
  {
    name: "Simple Vibrato",
    category: "Basic",
    description: "LFO to pitch modulation",
    routings: [
      {
        sourceId: "lfo1",
        destId: "oscillator_pitch",
        depth: 0.3,
        enabled: true
      }
    ]
  },
  {
    name: "Filter Sweep",
    category: "Basic",
    description: "Envelope to filter cutoff",
    routings: [
      {
        sourceId: "env1",
        destId: "filter_cutoff",
        depth: 0.8,
        enabled: true
      }
    ]
  },
  {
    name: "Rhythmic Modulation",
    category: "Complex",
    description: "Multiple LFOs for rhythm",
    routings: [
      {
        sourceId: "lfo1",
        destId: "filter_cutoff",
        depth: 0.6,
        enabled: true
      },
      {
        sourceId: "lfo2",
        destId: "amp_level",
        depth: 0.4,
        enabled: true
      },
      {
        sourceId: "lfo3",
        destId: "pan",
        depth: 0.5,
        enabled: true
      }
    ]
  },
  {
    name: "Meta-Modulation Example",
    category: "Advanced",
    description: "LFO modulating LFO depth",
    routings: [
      {
        sourceId: "lfo1",
        destId: "filter_cutoff",
        depth: 0.7,
        enabled: true
      }
    ],
    metaModulations: [
      {
        routingId: "routing_0",
        sourceId: "lfo2",
        depth: 0.8
      }
    ]
  },
  {
    name: "Chaotic System",
    category: "Experimental",
    description: "Complex modulation web",
    routings: [
      {
        sourceId: "lfo1",
        destId: "filter_cutoff",
        depth: 0.5,
        enabled: true
      },
      {
        sourceId: "lfo1",
        destId: "resonance",
        depth: 0.3,
        enabled: true
      },
      {
        sourceId: "lfo2",
        destId: "lfo1_rate",
        depth: 0.4,
        enabled: true
      },
      {
        sourceId: "env1",
        destId: "lfo2_depth",
        depth: 0.6,
        enabled: true
      }
    ]
  }
];

export const AllPresets = {
  AdvancedLFO: AdvancedLFOPresets,
  EnvelopeGenerator: EnvelopeGeneratorPresets,
  MacroControls: MacroControlsPresets,
  ModulationMatrix: ModulationMatrixPresets
};

export default AllPresets;
