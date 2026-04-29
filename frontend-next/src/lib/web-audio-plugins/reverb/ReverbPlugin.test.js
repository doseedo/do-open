/**
 * ReverbPlugin Test Suite
 *
 * Tests for AudioWorklet-based Reverb plugin
 *
 * @author Agent 6 - Reverb Plugins
 * @version 1.0.0
 */

import { ReverbPlugin } from './ReverbPlugin.js';

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
    data[0] = amplitude; // Impulse at start
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
describe('ReverbPlugin', () => {
  let audioContext;
  let plugin;

  beforeEach(() => {
    audioContext = new AudioContext();
    plugin = new ReverbPlugin(audioContext);
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
      expect(plugin.name).toBe('Reverb');
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
        preDelay: 0,
        decayTime: 2.0,
        size: 50,
        diffusion: 70,
        damping: 50,
        mix: 30
      });
    });
  });

  describe('Parameters', () => {
    beforeEach(async () => {
      await plugin.initialize();
    });

    test('should register all parameters', () => {
      const params = ['preDelay', 'decayTime', 'size', 'diffusion', 'damping', 'mix'];

      params.forEach(param => {
        expect(plugin.getParameter(param)).toBeDefined();
      });
    });

    test('should set and get preDelay parameter', () => {
      plugin.setParameter('preDelay', 50);
      const value = plugin.getParameter('preDelay').param.value;
      expect(value).toBeCloseTo(50, 1);
    });

    test('should set and get decayTime parameter', () => {
      plugin.setParameter('decayTime', 5.0);
      const value = plugin.getParameter('decayTime').param.value;
      expect(value).toBeCloseTo(5.0, 1);
    });

    test('should set and get size parameter', () => {
      plugin.setParameter('size', 75);
      const value = plugin.getParameter('size').param.value;
      expect(value).toBeCloseTo(75, 1);
    });

    test('should set and get diffusion parameter', () => {
      plugin.setParameter('diffusion', 85);
      const value = plugin.getParameter('diffusion').param.value;
      expect(value).toBeCloseTo(85, 1);
    });

    test('should set and get damping parameter', () => {
      plugin.setParameter('damping', 70);
      const value = plugin.getParameter('damping').param.value;
      expect(value).toBeCloseTo(70, 1);
    });

    test('should set and get mix parameter', () => {
      plugin.setParameter('mix', 50);
      const value = plugin.getParameter('mix').param.value;
      expect(value).toBeCloseTo(50, 1);
    });
  });

  describe('Audio Processing', () => {
    test('should process audio offline (impulse response)', async () => {
      const context = createOfflineContext(1.0);
      const inputBuffer = createImpulseBuffer(context);

      const outputBuffer = await plugin.processOffline(inputBuffer);

      expect(outputBuffer).toBeDefined();
      expect(outputBuffer.length).toBe(inputBuffer.length);
      expect(outputBuffer.numberOfChannels).toBe(inputBuffer.numberOfChannels);

      // Check that reverb tail exists (output has energy after impulse)
      const channel0 = outputBuffer.getChannelData(0);
      const tailStart = Math.floor(context.sampleRate * 0.1); // 100ms after start
      const tailEnd = Math.floor(context.sampleRate * 0.5); // 500ms after start

      let tailEnergy = 0;
      for (let i = tailStart; i < tailEnd; i++) {
        tailEnergy += Math.abs(channel0[i]);
      }

      expect(tailEnergy).toBeGreaterThan(0);
    });

    test('should process audio offline (sine wave)', async () => {
      const context = createOfflineContext(0.5);
      const inputBuffer = createSineBuffer(context, 440, 0.5);

      const outputBuffer = await plugin.processOffline(inputBuffer);

      expect(outputBuffer).toBeDefined();
      expect(outputBuffer.length).toBe(inputBuffer.length);

      // Check that output has signal
      const channel0 = outputBuffer.getChannelData(0);
      let rms = 0;
      for (let i = 0; i < channel0.length; i++) {
        rms += channel0[i] * channel0[i];
      }
      rms = Math.sqrt(rms / channel0.length);

      expect(rms).toBeGreaterThan(0);
    });

    test('should process with different decay times', async () => {
      const context1 = createOfflineContext(2.0);
      const context2 = createOfflineContext(2.0);
      const inputBuffer1 = createImpulseBuffer(context1);
      const inputBuffer2 = createImpulseBuffer(context2);

      // Short decay
      plugin.setParameter('decayTime', 0.5);
      const output1 = await plugin.processOffline(inputBuffer1);

      // Long decay
      plugin.setParameter('decayTime', 10.0);
      const output2 = await plugin.processOffline(inputBuffer2);

      // Measure tail energy at 1 second
      const measurePoint = Math.floor(48000);
      const channel1 = output1.getChannelData(0);
      const channel2 = output2.getChannelData(0);

      const energy1 = Math.abs(channel1[measurePoint]);
      const energy2 = Math.abs(channel2[measurePoint]);

      // Longer decay should have more energy at 1 second
      expect(energy2).toBeGreaterThan(energy1);
    });

    test('should respect mix parameter (dry/wet)', async () => {
      const context = createOfflineContext(0.1);
      const inputBuffer = createSineBuffer(context, 1000, 0.5);

      // 0% wet (all dry)
      plugin.setParameter('mix', 0);
      const dryOutput = await plugin.processOffline(inputBuffer);

      // 100% wet
      plugin.setParameter('mix', 100);
      const wetOutput = await plugin.processOffline(inputBuffer);

      const dryChannel = dryOutput.getChannelData(0);
      const wetChannel = wetOutput.getChannelData(0);
      const inputChannel = inputBuffer.getChannelData(0);

      // Dry output should be very similar to input
      let dryDiff = 0;
      for (let i = 0; i < 100; i++) {
        dryDiff += Math.abs(dryChannel[i] - inputChannel[i]);
      }

      expect(dryDiff).toBeLessThan(0.1);

      // Wet output should be different from input
      let wetDiff = 0;
      for (let i = 0; i < 100; i++) {
        wetDiff += Math.abs(wetChannel[i] - inputChannel[i]);
      }

      expect(wetDiff).toBeGreaterThan(dryDiff);
    });
  });

  describe('Performance', () => {
    test('should meet 20x real-time performance target', async () => {
      const duration = 10.0; // 10 seconds of audio
      const context = createOfflineContext(duration);
      const inputBuffer = createSineBuffer(context, 440, 0.5);

      const startTime = performance.now();
      await plugin.processOffline(inputBuffer);
      const endTime = performance.now();

      const processingTime = (endTime - startTime) / 1000; // Convert to seconds
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
