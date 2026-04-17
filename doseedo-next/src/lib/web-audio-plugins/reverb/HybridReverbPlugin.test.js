/**
 * HybridReverbPlugin Test Suite
 *
 * Tests for AudioWorklet-based Hybrid Reverb plugin
 *
 * @author Agent 6 - Reverb Plugins
 * @version 1.0.0
 */

import { HybridReverbPlugin } from './HybridReverbPlugin.js';

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
describe('HybridReverbPlugin', () => {
  let audioContext;
  let plugin;

  beforeEach(() => {
    audioContext = new AudioContext();
    plugin = new HybridReverbPlugin(audioContext);
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
      expect(plugin.name).toBe('HybridReverb');
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
        earlyLevel: -6,
        tailLevel: -6,
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
      const params = ['preDelay', 'decayTime', 'earlyLevel', 'tailLevel', 'damping', 'mix'];

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

    test('should set and get earlyLevel parameter', () => {
      plugin.setParameter('earlyLevel', -12);
      const value = plugin.getParameter('earlyLevel').param.value;
      expect(value).toBeCloseTo(-12, 1);
    });

    test('should set and get tailLevel parameter', () => {
      plugin.setParameter('tailLevel', -3);
      const value = plugin.getParameter('tailLevel').param.value;
      expect(value).toBeCloseTo(-3, 1);
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
    test('should process audio offline (impulse)', async () => {
      const context = createOfflineContext(2.0);
      const inputBuffer = createImpulseBuffer(context);

      const outputBuffer = await plugin.processOffline(inputBuffer);

      expect(outputBuffer).toBeDefined();
      expect(outputBuffer.length).toBe(inputBuffer.length);
      expect(outputBuffer.numberOfChannels).toBe(inputBuffer.numberOfChannels);

      // Check for both early and late reflections
      const channel0 = outputBuffer.getChannelData(0);

      // Early reflections (first 100ms)
      let earlyEnergy = 0;
      const earlyEnd = Math.floor(context.sampleRate * 0.1);
      for (let i = 0; i < earlyEnd; i++) {
        earlyEnergy += Math.abs(channel0[i]);
      }

      // Late reflections/tail (100ms - 500ms)
      let lateEnergy = 0;
      const lateStart = earlyEnd;
      const lateEnd = Math.floor(context.sampleRate * 0.5);
      for (let i = lateStart; i < lateEnd; i++) {
        lateEnergy += Math.abs(channel0[i]);
      }

      expect(earlyEnergy).toBeGreaterThan(0);
      expect(lateEnergy).toBeGreaterThan(0);
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

    test('should control early reflections level', async () => {
      const context1 = createOfflineContext(0.5);
      const context2 = createOfflineContext(0.5);
      const inputBuffer1 = createImpulseBuffer(context1);
      const inputBuffer2 = createImpulseBuffer(context2);

      // Low early level
      plugin.setParameter('earlyLevel', -30);
      plugin.setParameter('tailLevel', -60); // Disable tail
      plugin.setParameter('mix', 100);
      const output1 = await plugin.processOffline(inputBuffer1);

      // High early level
      plugin.setParameter('earlyLevel', -6);
      const output2 = await plugin.processOffline(inputBuffer2);

      // Measure early energy
      const earlyEnd = Math.floor(context1.sampleRate * 0.1);
      const channel1 = output1.getChannelData(0);
      const channel2 = output2.getChannelData(0);

      let energy1 = 0, energy2 = 0;
      for (let i = 0; i < earlyEnd; i++) {
        energy1 += Math.abs(channel1[i]);
        energy2 += Math.abs(channel2[i]);
      }

      // Higher early level should have more early energy
      expect(energy2).toBeGreaterThan(energy1);
    });

    test('should control tail level independently', async () => {
      const context1 = createOfflineContext(1.0);
      const context2 = createOfflineContext(1.0);
      const inputBuffer1 = createImpulseBuffer(context1);
      const inputBuffer2 = createImpulseBuffer(context2);

      // Low tail level
      plugin.setParameter('earlyLevel', -60); // Disable early
      plugin.setParameter('tailLevel', -30);
      plugin.setParameter('mix', 100);
      const output1 = await plugin.processOffline(inputBuffer1);

      // High tail level
      plugin.setParameter('tailLevel', -6);
      const output2 = await plugin.processOffline(inputBuffer2);

      // Measure tail energy
      const tailStart = Math.floor(context1.sampleRate * 0.2);
      const channel1 = output1.getChannelData(0);
      const channel2 = output2.getChannelData(0);

      let energy1 = 0, energy2 = 0;
      for (let i = tailStart; i < channel1.length; i++) {
        energy1 += Math.abs(channel1[i]);
        energy2 += Math.abs(channel2[i]);
      }

      // Higher tail level should have more tail energy
      expect(energy2).toBeGreaterThan(energy1);
    });

    test('should apply decay time to tail', async () => {
      const context1 = createOfflineContext(3.0);
      const context2 = createOfflineContext(3.0);
      const inputBuffer1 = createImpulseBuffer(context1);
      const inputBuffer2 = createImpulseBuffer(context2);

      // Short decay
      plugin.setParameter('decayTime', 0.5);
      plugin.setParameter('mix', 100);
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
