/**
 * Modulation Plugins Test Suite
 *
 * Tests for AudioWorklet-based modulation plugins:
 * - ChorusPlugin
 * - FlangerPlugin
 * - PhaserPlugin
 * - TremoloPlugin
 *
 * @author Agent 4 - Modulation Plugins
 * @version 1.0.0
 */

import { ChorusPlugin } from '../ChorusPlugin.js';
import { FlangerPlugin } from '../FlangerPlugin.js';
import { PhaserPlugin } from '../PhaserPlugin.js';
import { TremoloPlugin } from '../TremoloPlugin.js';

/**
 * Test utilities
 */
class TestRunner {
  constructor() {
    this.tests = [];
    this.results = {
      passed: 0,
      failed: 0,
      total: 0
    };
  }

  async test(name, fn) {
    this.results.total++;
    try {
      await fn();
      this.results.passed++;
      console.log(`✓ ${name}`);
    } catch (error) {
      this.results.failed++;
      console.error(`✗ ${name}`);
      console.error(`  ${error.message}`);
    }
  }

  async runAll() {
    console.log('=== Modulation Plugins Test Suite ===\n');
    for (const test of this.tests) {
      await test();
    }
    console.log(`\n=== Results ===`);
    console.log(`Passed: ${this.results.passed}/${this.results.total}`);
    console.log(`Failed: ${this.results.failed}/${this.results.total}`);
  }
}

/**
 * Helper: Create test audio buffer
 */
function createTestBuffer(audioContext, duration = 1.0, frequency = 440) {
  const sampleRate = audioContext.sampleRate;
  const length = Math.floor(duration * sampleRate);
  const buffer = audioContext.createBuffer(2, length, sampleRate);

  for (let channel = 0; channel < 2; channel++) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < length; i++) {
      // Generate sine wave
      data[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.5;
    }
  }

  return buffer;
}

/**
 * Helper: Measure processing time
 */
async function measureProcessingTime(plugin, inputBuffer) {
  const startTime = performance.now();
  const outputBuffer = await plugin.processOffline(inputBuffer);
  const endTime = performance.now();

  const processingTime = (endTime - startTime) / 1000; // Convert to seconds
  const audioLength = inputBuffer.length / inputBuffer.sampleRate;
  const realtimeFactor = audioLength / processingTime;

  return {
    processingTime,
    audioLength,
    realtimeFactor
  };
}

/**
 * Helper: Assert
 */
function assert(condition, message) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

/**
 * Test Suite
 */
async function runTests() {
  const runner = new TestRunner();
  const audioContext = new AudioContext();

  // ===== CHORUS PLUGIN TESTS =====
  await runner.test('ChorusPlugin: Initialization', async () => {
    const plugin = new ChorusPlugin(audioContext);
    await plugin.initialize();
    assert(plugin.initialized, 'Plugin should be initialized');
    assert(plugin.workletNode !== null, 'WorkletNode should be created');
    assert(plugin.usesAudioWorklet() === true, 'Should use AudioWorklet');
    plugin.dispose();
  });

  await runner.test('ChorusPlugin: Parameter registration', async () => {
    const plugin = new ChorusPlugin(audioContext);
    await plugin.initialize();

    const params = plugin.getAllParameters();
    assert(params.has('rate'), 'Should have rate parameter');
    assert(params.has('depth'), 'Should have depth parameter');
    assert(params.has('voices'), 'Should have voices parameter');
    assert(params.has('spread'), 'Should have spread parameter');
    assert(params.has('feedback'), 'Should have feedback parameter');
    assert(params.has('mix'), 'Should have mix parameter');
    assert(params.has('delay'), 'Should have delay parameter');
    assert(params.has('waveform'), 'Should have waveform parameter');

    plugin.dispose();
  });

  await runner.test('ChorusPlugin: Parameter setting', async () => {
    const plugin = new ChorusPlugin(audioContext);
    await plugin.initialize();

    plugin.setParameter('rate', 1.5);
    plugin.setParameter('depth', 75);
    plugin.setParameter('voices', 6);
    plugin.setWaveformType('triangle');

    assert(plugin.getParameter('rate').param.value === 1.5, 'Rate should be set');
    assert(plugin.getParameter('depth').param.value === 75, 'Depth should be set');
    assert(plugin.getParameter('voices').param.value === 6, 'Voices should be set');
    assert(plugin.getParameter('waveform').param.value === 1, 'Waveform should be triangle');

    plugin.dispose();
  });

  await runner.test('ChorusPlugin: Audio processing', async () => {
    const plugin = new ChorusPlugin(audioContext);
    await plugin.initialize();

    const inputBuffer = createTestBuffer(audioContext, 0.5);
    const outputBuffer = await plugin.processOffline(inputBuffer);

    assert(outputBuffer !== null, 'Output buffer should not be null');
    assert(outputBuffer.length === inputBuffer.length, 'Output length should match input');
    assert(outputBuffer.numberOfChannels === inputBuffer.numberOfChannels, 'Channels should match');

    // Check that output is not silent
    const outputData = outputBuffer.getChannelData(0);
    const hasSignal = outputData.some(sample => Math.abs(sample) > 0.001);
    assert(hasSignal, 'Output should have signal');

    plugin.dispose();
  });

  await runner.test('ChorusPlugin: Performance (20x real-time target)', async () => {
    const plugin = new ChorusPlugin(audioContext);
    await plugin.initialize();

    const inputBuffer = createTestBuffer(audioContext, 5.0);
    const metrics = await measureProcessingTime(plugin, inputBuffer);

    console.log(`  Processing: ${metrics.processingTime.toFixed(3)}s for ${metrics.audioLength.toFixed(1)}s audio`);
    console.log(`  Real-time factor: ${metrics.realtimeFactor.toFixed(1)}x`);

    assert(metrics.realtimeFactor > 20, `Should exceed 20x real-time (got ${metrics.realtimeFactor.toFixed(1)}x)`);

    plugin.dispose();
  });

  // ===== FLANGER PLUGIN TESTS =====
  await runner.test('FlangerPlugin: Initialization', async () => {
    const plugin = new FlangerPlugin(audioContext);
    await plugin.initialize();
    assert(plugin.initialized, 'Plugin should be initialized');
    assert(plugin.workletNode !== null, 'WorkletNode should be created');
    plugin.dispose();
  });

  await runner.test('FlangerPlugin: Parameter registration', async () => {
    const plugin = new FlangerPlugin(audioContext);
    await plugin.initialize();

    const params = plugin.getAllParameters();
    assert(params.has('rate'), 'Should have rate parameter');
    assert(params.has('depth'), 'Should have depth parameter');
    assert(params.has('feedback'), 'Should have feedback parameter');
    assert(params.has('manual'), 'Should have manual parameter');

    plugin.dispose();
  });

  await runner.test('FlangerPlugin: Negative feedback support', async () => {
    const plugin = new FlangerPlugin(audioContext);
    await plugin.initialize();

    plugin.setParameter('feedback', -50);
    assert(plugin.getParameter('feedback').param.value === -50, 'Should support negative feedback');

    plugin.dispose();
  });

  await runner.test('FlangerPlugin: Audio processing', async () => {
    const plugin = new FlangerPlugin(audioContext);
    await plugin.initialize();

    const inputBuffer = createTestBuffer(audioContext, 0.5);
    const outputBuffer = await plugin.processOffline(inputBuffer);

    assert(outputBuffer !== null, 'Output buffer should not be null');
    const outputData = outputBuffer.getChannelData(0);
    const hasSignal = outputData.some(sample => Math.abs(sample) > 0.001);
    assert(hasSignal, 'Output should have signal');

    plugin.dispose();
  });

  await runner.test('FlangerPlugin: Performance (20x real-time target)', async () => {
    const plugin = new FlangerPlugin(audioContext);
    await plugin.initialize();

    const inputBuffer = createTestBuffer(audioContext, 5.0);
    const metrics = await measureProcessingTime(plugin, inputBuffer);

    console.log(`  Real-time factor: ${metrics.realtimeFactor.toFixed(1)}x`);
    assert(metrics.realtimeFactor > 20, `Should exceed 20x real-time (got ${metrics.realtimeFactor.toFixed(1)}x)`);

    plugin.dispose();
  });

  // ===== PHASER PLUGIN TESTS =====
  await runner.test('PhaserPlugin: Initialization', async () => {
    const plugin = new PhaserPlugin(audioContext);
    await plugin.initialize();
    assert(plugin.initialized, 'Plugin should be initialized');
    assert(plugin.workletNode !== null, 'WorkletNode should be created');
    plugin.dispose();
  });

  await runner.test('PhaserPlugin: Parameter registration', async () => {
    const plugin = new PhaserPlugin(audioContext);
    await plugin.initialize();

    const params = plugin.getAllParameters();
    assert(params.has('stages'), 'Should have stages parameter');
    assert(params.has('frequency'), 'Should have frequency parameter');
    assert(params.has('spread'), 'Should have spread parameter');

    plugin.dispose();
  });

  await runner.test('PhaserPlugin: Variable stage count', async () => {
    const plugin = new PhaserPlugin(audioContext);
    await plugin.initialize();

    plugin.setParameter('stages', 4);
    assert(plugin.getParameter('stages').param.value === 4, 'Should support 4 stages');

    plugin.setParameter('stages', 12);
    assert(plugin.getParameter('stages').param.value === 12, 'Should support 12 stages');

    plugin.dispose();
  });

  await runner.test('PhaserPlugin: Audio processing', async () => {
    const plugin = new PhaserPlugin(audioContext);
    await plugin.initialize();

    const inputBuffer = createTestBuffer(audioContext, 0.5);
    const outputBuffer = await plugin.processOffline(inputBuffer);

    assert(outputBuffer !== null, 'Output buffer should not be null');
    const outputData = outputBuffer.getChannelData(0);
    const hasSignal = outputData.some(sample => Math.abs(sample) > 0.001);
    assert(hasSignal, 'Output should have signal');

    plugin.dispose();
  });

  await runner.test('PhaserPlugin: Performance (20x real-time target)', async () => {
    const plugin = new PhaserPlugin(audioContext);
    await plugin.initialize();

    const inputBuffer = createTestBuffer(audioContext, 5.0);
    const metrics = await measureProcessingTime(plugin, inputBuffer);

    console.log(`  Real-time factor: ${metrics.realtimeFactor.toFixed(1)}x`);
    assert(metrics.realtimeFactor > 20, `Should exceed 20x real-time (got ${metrics.realtimeFactor.toFixed(1)}x)`);

    plugin.dispose();
  });

  // ===== TREMOLO PLUGIN TESTS =====
  await runner.test('TremoloPlugin: Initialization', async () => {
    const plugin = new TremoloPlugin(audioContext);
    await plugin.initialize();
    assert(plugin.initialized, 'Plugin should be initialized');
    assert(plugin.workletNode !== null, 'WorkletNode should be created');
    plugin.dispose();
  });

  await runner.test('TremoloPlugin: Parameter registration', async () => {
    const plugin = new TremoloPlugin(audioContext);
    await plugin.initialize();

    const params = plugin.getAllParameters();
    assert(params.has('mode'), 'Should have mode parameter');
    assert(params.has('stereo'), 'Should have stereo parameter');

    plugin.dispose();
  });

  await runner.test('TremoloPlugin: Mode switching', async () => {
    const plugin = new TremoloPlugin(audioContext);
    await plugin.initialize();

    plugin.setMode('tremolo');
    assert(plugin.getParameter('mode').param.value === 0, 'Should be in tremolo mode');

    plugin.setMode('pan');
    assert(plugin.getParameter('mode').param.value === 1, 'Should be in pan mode');

    plugin.dispose();
  });

  await runner.test('TremoloPlugin: Stereo mode', async () => {
    const plugin = new TremoloPlugin(audioContext);
    await plugin.initialize();

    plugin.setStereo(true);
    assert(plugin.getParameter('stereo').param.value === 1, 'Should enable stereo mode');

    plugin.setStereo(false);
    assert(plugin.getParameter('stereo').param.value === 0, 'Should disable stereo mode');

    plugin.dispose();
  });

  await runner.test('TremoloPlugin: Audio processing', async () => {
    const plugin = new TremoloPlugin(audioContext);
    await plugin.initialize();

    const inputBuffer = createTestBuffer(audioContext, 0.5);
    const outputBuffer = await plugin.processOffline(inputBuffer);

    assert(outputBuffer !== null, 'Output buffer should not be null');
    const outputData = outputBuffer.getChannelData(0);
    const hasSignal = outputData.some(sample => Math.abs(sample) > 0.001);
    assert(hasSignal, 'Output should have signal');

    plugin.dispose();
  });

  await runner.test('TremoloPlugin: Performance (20x real-time target)', async () => {
    const plugin = new TremoloPlugin(audioContext);
    await plugin.initialize();

    const inputBuffer = createTestBuffer(audioContext, 5.0);
    const metrics = await measureProcessingTime(plugin, inputBuffer);

    console.log(`  Real-time factor: ${metrics.realtimeFactor.toFixed(1)}x`);
    assert(metrics.realtimeFactor > 20, `Should exceed 20x real-time (got ${metrics.realtimeFactor.toFixed(1)}x)`);

    plugin.dispose();
  });

  // Print results
  console.log('\n=== Test Results ===');
  console.log(`Passed: ${runner.results.passed}/${runner.results.total}`);
  console.log(`Failed: ${runner.results.failed}/${runner.results.total}`);
  console.log(`Success Rate: ${((runner.results.passed / runner.results.total) * 100).toFixed(1)}%`);

  // Close audio context
  await audioContext.close();

  return runner.results;
}

// Export for use in browser or Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runTests };
} else if (typeof window !== 'undefined') {
  window.ModulationPluginTests = { runTests };
}
