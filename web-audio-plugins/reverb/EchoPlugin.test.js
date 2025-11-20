/**
 * EchoPlugin Test Suite
 *
 * Tests for AudioWorklet-based Echo plugin
 *
 * @author Agent 6 - Reverb Plugins
 * @version 1.0.0
 */

import { EchoPlugin } from './EchoPlugin.js';

/**
 * Helper function to create an offline audio context
 */
function createOfflineContext(duration = 1.0, sampleRate = 48000) {
  const numSamples = Math.floor(duration * sampleRate);
  return new OfflineAudioContext(2, numSamples, sampleRate);
}

/**
 * Helper function to create a test audio buffer with impulse
 */
function createImpulseBuffer(context, amplitude = 1.0) {
  const buffer = context.createBuffer(2, context.length, context.sampleRate);

  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const data = buffer.getChannelData(channel);
    data[0] = amplitude;
  }

  return buffer;
}

/**
 * Helper function to create a test audio buffer with sine wave
 */
function createSineBuffer(context, frequency = 440, amplitude = 0.5) {
  const buffer = context.createBuffer(2, context.length, context.sampleRate);

  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const data = buffer.getChannelData(channel);

    for (let i = 0; i < data.length; i++) {
      data[i] = amplitude * Math.sin(2 * Math.PI * frequency * i / context.sampleRate);
    }
  }

  return buffer;
}

/**
 * Test Suite
 */
describe('EchoPlugin', () => {
  let audioContext;
  let plugin;

  beforeEach(() => {
    audioContext = new AudioContext();
    plugin = new EchoPlugin(audioContext);
  });

  afterEach(async () => {
    if (plugin) {
      plugin.dispose();
    }
    if (audioContext) {
      await audioContext.close();
    }
  });

  describe('Initialization', () => {
    test('should create plugin instance', () => {
      expect(plugin).toBeDefined();
      expect(plugin.name).toBe('Echo');
      expect(plugin.category).toBe('reverb');
    });

    test('should initialize AudioWorklet', async () => {
      await plugin.initialize();
      expect(plugin.initialized).toBe(true);
      expect(plugin.workletNode).toBeDefined();
    });

    test('should use AudioWorklet', () => {
      expect(plugin.usesAudioWorklet()).toBe(true);
    });

    test('should have default parameters', () => {
      expect(plugin.defaults).toEqual({
        delayTimeL: 250,
        delayTimeR: 375,
        feedback: 40,
        numTaps: 4,
        tapDecay: 0.7,
        highpass: 20,
        lowpass: 20000,
        stereoOffset: 0,
        mix: 30
      });
    });
  });

  describe('Parameters', () => {
    beforeEach(async () => {
      await plugin.initialize();
    });

    test('should register all parameters', () => {
      const params = ['delayTimeL', 'delayTimeR', 'feedback', 'numTaps',
                      'tapDecay', 'highpass', 'lowpass', 'stereoOffset', 'mix'];

      params.forEach(param => {
        expect(plugin.getParameter(param)).toBeDefined();
      });
    });

    test('should set and get delayTimeL parameter', () => {
      plugin.setParameter('delayTimeL', 500);
      const value = plugin.getParameter('delayTimeL').param.value;
      expect(value).toBeCloseTo(500, 1);
    });

    test('should set and get delayTimeR parameter', () => {
      plugin.setParameter('delayTimeR', 750);
      const value = plugin.getParameter('delayTimeR').param.value;
      expect(value).toBeCloseTo(750, 1);
    });

    test('should set and get feedback parameter', () => {
      plugin.setParameter('feedback', 60);
      const value = plugin.getParameter('feedback').param.value;
      expect(value).toBeCloseTo(60, 1);
    });

    test('should set and get numTaps parameter', () => {
      plugin.setParameter('numTaps', 6);
      const value = plugin.getParameter('numTaps').param.value;
      expect(value).toBe(6);
    });

    test('should set and get tapDecay parameter', () => {
      plugin.setParameter('tapDecay', 0.8);
      const value = plugin.getParameter('tapDecay').param.value;
      expect(value).toBeCloseTo(0.8, 2);
    });
  });

  describe('Audio Processing', () => {
    test('should process audio offline (impulse)', async () => {
      const context = createOfflineContext(2.0);
      const inputBuffer = createImpulseBuffer(context);

      const outputBuffer = await plugin.processOffline(inputBuffer);

      expect(outputBuffer).toBeDefined();
      expect(outputBuffer.length).toBe(inputBuffer.length);
      expect(outputBuffer.numberOfChannels).toBe(inputBuffer.numberOfChannels);

      // Check for echo taps
      const channel0 = outputBuffer.getChannelData(0);
      let hasEchos = false;

      // Look for peaks at delay intervals
      const delayTime = 250; // ms
      const delaySamples = Math.floor((delayTime / 1000) * context.sampleRate);

      for (let tap = 1; tap <= 3; tap++) {
        const tapPosition = tap * delaySamples;
        if (tapPosition < channel0.length) {
          const energy = Math.abs(channel0[tapPosition]);
          if (energy > 0.01) {
            hasEchos = true;
            break;
          }
        }
      }

      expect(hasEchos).toBe(true);
    });

    test('should process audio offline (sine wave)', async () => {
      const context = createOfflineContext(1.0);
      const inputBuffer = createSineBuffer(context, 440, 0.5);

      const outputBuffer = await plugin.processOffline(inputBuffer);

      expect(outputBuffer).toBeDefined();

      // Check that output has signal
      const channel0 = outputBuffer.getChannelData(0);
      let rms = 0;
      for (let i = 0; i < channel0.length; i++) {
        rms += channel0[i] * channel0[i];
      }
      rms = Math.sqrt(rms / channel0.length);

      expect(rms).toBeGreaterThan(0);
    });

    test('should create stereo echo with different L/R times', async () => {
      const context = createOfflineContext(1.0);
      const inputBuffer = createImpulseBuffer(context);

      plugin.setParameter('delayTimeL', 200);
      plugin.setParameter('delayTimeR', 400);
      plugin.setParameter('mix', 100);

      const outputBuffer = await plugin.processOffline(inputBuffer);

      const channelL = outputBuffer.getChannelData(0);
      const channelR = outputBuffer.getChannelData(1);

      // Find peak positions
      let peakL = 0, peakR = 0;
      let maxL = 0, maxR = 0;

      for (let i = 0; i < outputBuffer.length; i++) {
        if (Math.abs(channelL[i]) > maxL) {
          maxL = Math.abs(channelL[i]);
          peakL = i;
        }
        if (Math.abs(channelR[i]) > maxR) {
          maxR = Math.abs(channelR[i]);
          peakR = i;
        }
      }

      // Right channel peak should be later than left
      expect(peakR).toBeGreaterThan(peakL);
    });

    test('should apply feedback correctly', async () => {
      const context1 = createOfflineContext(1.0);
      const context2 = createOfflineContext(1.0);
      const inputBuffer1 = createImpulseBuffer(context1);
      const inputBuffer2 = createImpulseBuffer(context2);

      // Low feedback
      plugin.setParameter('feedback', 10);
      plugin.setParameter('mix', 100);
      const output1 = await plugin.processOffline(inputBuffer1);

      // High feedback
      plugin.setParameter('feedback', 80);
      const output2 = await plugin.processOffline(inputBuffer2);

      // Measure energy in tail (after 0.5s)
      const measureStart = Math.floor(context1.sampleRate * 0.5);
      const channel1 = output1.getChannelData(0);
      const channel2 = output2.getChannelData(0);

      let energy1 = 0, energy2 = 0;
      for (let i = measureStart; i < channel1.length; i++) {
        energy1 += Math.abs(channel1[i]);
        energy2 += Math.abs(channel2[i]);
      }

      // Higher feedback should have more tail energy
      expect(energy2).toBeGreaterThan(energy1);
    });
  });

  describe('Performance', () => {
    test('should meet 20x real-time performance target', async () => {
      const duration = 10.0;
      const context = createOfflineContext(duration);
      const inputBuffer = createSineBuffer(context, 440, 0.5);

      const startTime = performance.now();
      await plugin.processOffline(inputBuffer);
      const endTime = performance.now();

      const processingTime = (endTime - startTime) / 1000;
      const realtimeRatio = duration / processingTime;

      console.log(`Performance: ${realtimeRatio.toFixed(1)}x real-time`);
      expect(realtimeRatio).toBeGreaterThan(20);
    });
  });

  describe('Disposal', () => {
    test('should dispose cleanly', async () => {
      await plugin.initialize();
      expect(plugin.workletNode).toBeDefined();

      plugin.dispose();

      expect(plugin.workletNode).toBeNull();
    });
  });
});
