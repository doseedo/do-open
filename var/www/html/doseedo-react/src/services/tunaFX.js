import Tuna from 'tunajs';

/**
 * Tuna Audio Effects Service
 * Manages the FX chain: tracks → FX bus → effects → master
 */
class TunaFXService {
  constructor() {
    this.tuna = null;
    this.audioContext = null;
    this.fxBusInput = null;
    this.fxBusOutput = null;

    // Effect instances
    this.effects = {
      reverb: null,
      delay: null,
      chorus: null,
      compressor: null,
      filter: null,
      phaser: null
    };

    this.initialized = false;
  }

  /**
   * Initialize Tuna with an audio context
   */
  initialize(audioContext) {
    if (this.initialized) {
      console.log('🎛️ Tuna FX already initialized');
      return;
    }

    this.audioContext = audioContext;
    this.tuna = new Tuna(audioContext);

    // Create FX bus input/output nodes
    this.fxBusInput = audioContext.createGain();
    this.fxBusOutput = audioContext.createGain();

    // Initialize all effects
    this.initializeEffects();

    // Connect effects in series
    this.connectFXChain();

    this.initialized = true;
    console.log('🎛️ Tuna FX initialized with signal chain');
  }

  /**
   * Initialize all Tuna effects with default parameters
   */
  initializeEffects() {
    // Reverb - using a simple gain node instead of Convolver to avoid impulse response 404
    // This is a placeholder until we implement a proper reverb
    this.effects.reverb = this.audioContext.createGain();
    this.effects.reverb.gain.value = 1.0;
    this.effects.reverb.connect = this.effects.reverb.connect.bind(this.effects.reverb);
    this.effects.reverb.input = this.effects.reverb;
    this.effects.reverb.bypass = true;

    // Delay
    this.effects.delay = new this.tuna.Delay({
      delayTime: 100,    // ms
      feedback: 0.45,    // 0-0.9
      wetLevel: 0,       // Dry only by default
      dryLevel: 1.0,
      cutoff: 20000,     // Hz
      bypass: true
    });

    // Chorus
    this.effects.chorus = new this.tuna.Chorus({
      rate: 1.5,         // Hz
      feedback: 0.4,     // 0-0.95
      delay: 0.0045,     // seconds
      depth: 0.7,        // 0-1
      bypass: true
    });

    // Compressor
    this.effects.compressor = new this.tuna.Compressor({
      threshold: -20,    // dB
      makeupGain: 1,
      attack: 0.003,     // seconds
      release: 0.25,     // seconds
      ratio: 4,          // 1-20
      knee: 5,
      automakeup: false,
      bypass: true
    });

    // Filter
    this.effects.filter = new this.tuna.Filter({
      frequency: 800,    // Hz
      Q: 1,              // resonance
      gain: 0,           // dB (-40 to 40)
      filterType: 'lowpass', // lowpass, highpass, bandpass, lowshelf, highshelf, peaking, notch, allpass
      bypass: true
    });

    // Phaser
    this.effects.phaser = new this.tuna.Phaser({
      rate: 0.1,         // Hz
      depth: 0.6,        // 0-1
      feedback: 0.7,     // 0-1
      stereoPhase: 30,   // degrees
      baseModulationFrequency: 700, // Hz
      bypass: true
    });

    console.log('✅ All Tuna effects initialized');
  }

  /**
   * Connect effects in series: input → reverb → delay → chorus → compressor → filter → phaser → output
   */
  connectFXChain() {
    if (!this.initialized) return;

    // Connect FX chain
    this.fxBusInput.connect(this.effects.reverb.input);
    this.effects.reverb.connect(this.effects.delay.input);
    this.effects.delay.connect(this.effects.chorus.input);
    this.effects.chorus.connect(this.effects.compressor.input);
    this.effects.compressor.connect(this.effects.filter.input);
    this.effects.filter.connect(this.effects.phaser.input);
    this.effects.phaser.connect(this.fxBusOutput);

    console.log('🔗 FX chain connected: Input → Reverb → Delay → Chorus → Compressor → Filter → Phaser → Output');
  }

  /**
   * Get the FX bus input node (connect tracks here)
   */
  getFXBusInput() {
    return this.fxBusInput;
  }

  /**
   * Get the FX bus output node (connect to master here)
   */
  getFXBusOutput() {
    return this.fxBusOutput;
  }

  /**
   * Update reverb parameters
   */
  updateReverb(params) {
    if (!this.effects.reverb) return;

    // Using simple gain node for now, just log parameter changes
    if (params.decay !== undefined) {
      console.log('🎛️ Reverb decay:', params.decay);
    }
    if (params.preDelay !== undefined) {
      console.log('🎛️ Reverb pre-delay:', params.preDelay);
    }
    if (params.roomSize !== undefined) {
      console.log('🎛️ Reverb room size:', params.roomSize);
    }
    if (params.damping !== undefined) {
      console.log('🎛️ Reverb damping:', params.damping);
    }
    if (params.mix !== undefined) {
      // Simple gain node - just pass through for now
      console.log('🎛️ Reverb mix:', params.mix);
    }
  }

  /**
   * Update delay parameters
   */
  updateDelay(params) {
    if (!this.effects.delay) return;

    if (params.time !== undefined) {
      this.effects.delay.delayTime = params.time;
    }
    if (params.feedback !== undefined) {
      this.effects.delay.feedback = params.feedback;
    }
    if (params.cutoff !== undefined) {
      this.effects.delay.cutoff = params.cutoff;
    }
    if (params.wet !== undefined) {
      this.effects.delay.wetLevel = params.wet;
    }
    if (params.dry !== undefined) {
      this.effects.delay.dryLevel = params.dry;
    }
  }

  /**
   * Update chorus parameters
   */
  updateChorus(params) {
    if (!this.effects.chorus) return;

    if (params.rate !== undefined) {
      this.effects.chorus.rate = params.rate;
    }
    if (params.depth !== undefined) {
      this.effects.chorus.depth = params.depth;
    }
    if (params.feedback !== undefined) {
      this.effects.chorus.feedback = params.feedback;
    }
  }

  /**
   * Update compressor parameters
   */
  updateCompressor(params) {
    if (!this.effects.compressor) return;

    if (params.threshold !== undefined) {
      this.effects.compressor.threshold = params.threshold;
    }
    if (params.ratio !== undefined) {
      this.effects.compressor.ratio = params.ratio;
    }
    if (params.attack !== undefined) {
      this.effects.compressor.attack = params.attack;
    }
    if (params.release !== undefined) {
      this.effects.compressor.release = params.release;
    }
  }

  /**
   * Update filter parameters
   */
  updateFilter(params) {
    if (!this.effects.filter) return;

    if (params.frequency !== undefined) {
      this.effects.filter.frequency = params.frequency;
    }
    if (params.resonance !== undefined) {
      this.effects.filter.Q = params.resonance;
    }
    if (params.gain !== undefined) {
      this.effects.filter.gain = params.gain;
    }
    if (params.type !== undefined) {
      this.effects.filter.filterType = params.type;
    }
  }

  /**
   * Update phaser parameters
   */
  updatePhaser(params) {
    if (!this.effects.phaser) return;

    if (params.rate !== undefined) {
      this.effects.phaser.rate = params.rate;
    }
    if (params.depth !== undefined) {
      this.effects.phaser.depth = params.depth;
    }
    if (params.feedback !== undefined) {
      this.effects.phaser.feedback = params.feedback;
    }
  }

  /**
   * Bypass/enable specific effect
   */
  bypassEffect(effectName, bypass) {
    if (this.effects[effectName]) {
      this.effects[effectName].bypass = bypass;
      console.log(`🎛️ ${effectName} ${bypass ? 'bypassed' : 'enabled'}`);
    }
  }

  /**
   * Clean up resources
   */
  destroy() {
    if (this.fxBusInput) {
      this.fxBusInput.disconnect();
    }
    if (this.fxBusOutput) {
      this.fxBusOutput.disconnect();
    }

    // Disconnect all effects
    Object.values(this.effects).forEach(effect => {
      if (effect) {
        effect.disconnect();
      }
    });

    this.initialized = false;
    console.log('🗑️ Tuna FX destroyed');
  }
}

// Export singleton instance
export const tunaFX = new TunaFXService();
export default tunaFX;
