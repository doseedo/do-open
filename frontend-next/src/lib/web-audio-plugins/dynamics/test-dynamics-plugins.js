/**
 * Test Suite for Dynamics Plugins
 *
 * Tests all four dynamics processors:
 * - CompressorPlugin
 * - LimiterPlugin
 * - GatePlugin
 * - ExpanderPlugin
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

import { CompressorPlugin } from './CompressorPlugin.js';
import { LimiterPlugin } from './LimiterPlugin.js';
import { GatePlugin } from './GatePlugin.js';
import { ExpanderPlugin } from './ExpanderPlugin.js';

/**
 * Test runner class
 */
class DynamicsPluginTests {
  constructor() {
    this.testsPassed = 0;
    this.testsFailed = 0;
    this.results = [];
  }

  /**
   * Generate a test audio buffer (sine wave)
   */
  generateTestBuffer(audioContext, frequency = 440, duration = 1, amplitude = 0.5) {
    const sampleRate = audioContext.sampleRate;
    const length = Math.floor(sampleRate * duration);
    const buffer = audioContext.createBuffer(2, length, sampleRate);

    for (let channel = 0; channel < 2; channel++) {
      const data = buffer.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        data[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * amplitude;
      }
    }

    return buffer;
  }

  /**
   * Calculate RMS level of a buffer
   */
  calculateRMS(buffer) {
    let sum = 0;
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      sum += data[i] * data[i];
    }
    return Math.sqrt(sum / data.length);
  }

  /**
   * Calculate peak level of a buffer
   */
  calculatePeak(buffer) {
    let peak = 0;
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      peak = Math.max(peak, Math.abs(data[i]));
    }
    return peak;
  }

  /**
   * Run a test and record result
   */
  async runTest(name, testFn) {
    console.log(`\nRunning: ${name}`);
    try {
      await testFn();
      console.log(`✓ PASSED: ${name}`);
      this.testsPassed++;
      this.results.push({ name, status: 'passed' });
    } catch (error) {
      console.error(`✗ FAILED: ${name}`);
      console.error(`  Error: ${error.message}`);
      this.testsFailed++;
      this.results.push({ name, status: 'failed', error: error.message });
    }
  }

  /**
   * Test CompressorPlugin
   */
  async testCompressor() {
    const audioContext = new AudioContext();

    await this.runTest('Compressor: Initialize', async () => {
      const compressor = new CompressorPlugin(audioContext);
      await compressor.initialize();

      if (!compressor.initialized) {
        throw new Error('Compressor failed to initialize');
      }

      compressor.dispose();
    });

    await this.runTest('Compressor: Set parameters', async () => {
      const compressor = new CompressorPlugin(audioContext);
      await compressor.initialize();

      compressor.setParameter('threshold', -12);
      compressor.setParameter('ratio', 6);
      compressor.setParameter('attack', 0.005);
      compressor.setParameter('release', 0.200);

      const threshold = compressor.getParameter('threshold').param.value;
      if (Math.abs(threshold - (-12)) > 0.01) {
        throw new Error(`Expected threshold -12, got ${threshold}`);
      }

      compressor.dispose();
    });

    await this.runTest('Compressor: Process audio (gain reduction)', async () => {
      const compressor = new CompressorPlugin(audioContext);
      await compressor.initialize();

      // Set aggressive compression
      compressor.setParameter('threshold', -20);
      compressor.setParameter('ratio', 10);
      compressor.setParameter('mix', 1.0); // 100% wet

      // Generate loud test signal
      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.8);
      const outputBuffer = await compressor.processOffline(inputBuffer);

      const inputPeak = this.calculatePeak(inputBuffer);
      const outputPeak = this.calculatePeak(outputBuffer);

      // Compressor should reduce the peak
      if (outputPeak >= inputPeak) {
        throw new Error(`Expected compression, but output peak (${outputPeak}) >= input peak (${inputPeak})`);
      }

      console.log(`  Input peak: ${inputPeak.toFixed(3)}, Output peak: ${outputPeak.toFixed(3)}`);

      compressor.dispose();
    });

    await this.runTest('Compressor: Parallel compression (mix)', async () => {
      const compressor = new CompressorPlugin(audioContext);
      await compressor.initialize();

      compressor.setParameter('threshold', -20);
      compressor.setParameter('ratio', 10);
      compressor.setParameter('mix', 0.5); // 50% wet

      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.8);
      const outputBuffer = await compressor.processOffline(inputBuffer);

      const inputPeak = this.calculatePeak(inputBuffer);
      const outputPeak = this.calculatePeak(outputBuffer);

      // With 50% mix, output should be between compressed and dry
      if (outputPeak >= inputPeak || outputPeak <= inputPeak * 0.5) {
        throw new Error('Mix parameter not working correctly');
      }

      compressor.dispose();
    });

    await audioContext.close();
  }

  /**
   * Test LimiterPlugin
   */
  async testLimiter() {
    const audioContext = new AudioContext();

    await this.runTest('Limiter: Initialize', async () => {
      const limiter = new LimiterPlugin(audioContext);
      await limiter.initialize();

      if (!limiter.initialized) {
        throw new Error('Limiter failed to initialize');
      }

      limiter.dispose();
    });

    await this.runTest('Limiter: Hard limiting', async () => {
      const limiter = new LimiterPlugin(audioContext);
      await limiter.initialize();

      // Set threshold at -6 dB
      limiter.setParameter('threshold', -6);
      limiter.setParameter('attack', 0.001); // Fast attack

      // Generate loud signal that will exceed threshold
      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.9);
      const outputBuffer = await limiter.processOffline(inputBuffer);

      const outputPeak = this.calculatePeak(outputBuffer);
      const thresholdLinear = Math.pow(10, -6 / 20); // -6 dB in linear

      // Output peak should not significantly exceed threshold
      // Allow small overshoot due to attack time
      if (outputPeak > thresholdLinear * 1.1) {
        throw new Error(`Limiter failed to limit: output peak ${outputPeak} > threshold ${thresholdLinear}`);
      }

      console.log(`  Output peak: ${outputPeak.toFixed(3)}, Threshold: ${thresholdLinear.toFixed(3)}`);

      limiter.dispose();
    });

    await audioContext.close();
  }

  /**
   * Test GatePlugin
   */
  async testGate() {
    const audioContext = new AudioContext();

    await this.runTest('Gate: Initialize', async () => {
      const gate = new GatePlugin(audioContext);
      await gate.initialize();

      if (!gate.initialized) {
        throw new Error('Gate failed to initialize');
      }

      gate.dispose();
    });

    await this.runTest('Gate: Attenuate below threshold', async () => {
      const gate = new GatePlugin(audioContext);
      await gate.initialize();

      // Set threshold above signal level
      gate.setParameter('threshold', -10); // -10 dB
      gate.setParameter('range', -60); // Attenuate by 60 dB when closed

      // Generate quiet signal below threshold
      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.1); // Quiet
      const outputBuffer = await gate.processOffline(inputBuffer);

      const inputRMS = this.calculateRMS(inputBuffer);
      const outputRMS = this.calculateRMS(outputBuffer);

      // Gate should significantly attenuate the signal
      if (outputRMS >= inputRMS * 0.5) {
        throw new Error('Gate failed to attenuate signal below threshold');
      }

      console.log(`  Input RMS: ${inputRMS.toFixed(4)}, Output RMS: ${outputRMS.toFixed(4)}`);

      gate.dispose();
    });

    await this.runTest('Gate: Pass signal above threshold', async () => {
      const gate = new GatePlugin(audioContext);
      await gate.initialize();

      // Set threshold below signal level
      gate.setParameter('threshold', -40); // -40 dB
      gate.setParameter('range', -60);

      // Generate loud signal above threshold
      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.8); // Loud
      const outputBuffer = await gate.processOffline(inputBuffer);

      const inputRMS = this.calculateRMS(inputBuffer);
      const outputRMS = this.calculateRMS(outputBuffer);

      // Gate should pass the signal mostly unchanged
      if (Math.abs(outputRMS - inputRMS) > 0.1) {
        throw new Error('Gate failed to pass signal above threshold');
      }

      gate.dispose();
    });

    await audioContext.close();
  }

  /**
   * Test ExpanderPlugin
   */
  async testExpander() {
    const audioContext = new AudioContext();

    await this.runTest('Expander: Initialize', async () => {
      const expander = new ExpanderPlugin(audioContext);
      await expander.initialize();

      if (!expander.initialized) {
        throw new Error('Expander failed to initialize');
      }

      expander.dispose();
    });

    await this.runTest('Expander: Expand below threshold', async () => {
      const expander = new ExpanderPlugin(audioContext);
      await expander.initialize();

      // Set threshold and ratio
      expander.setParameter('threshold', -20);
      expander.setParameter('ratio', 3); // 1:3 expansion

      // Generate quiet signal below threshold
      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.2);
      const outputBuffer = await expander.processOffline(inputBuffer);

      const inputRMS = this.calculateRMS(inputBuffer);
      const outputRMS = this.calculateRMS(outputBuffer);

      // Expander should reduce the signal (but less than a gate)
      if (outputRMS >= inputRMS) {
        throw new Error('Expander failed to reduce signal below threshold');
      }

      console.log(`  Input RMS: ${inputRMS.toFixed(4)}, Output RMS: ${outputRMS.toFixed(4)}`);

      expander.dispose();
    });

    await audioContext.close();
  }

  /**
   * Performance benchmark
   */
  async benchmarkPerformance() {
    console.log('\n=== Performance Benchmarks ===\n');

    const audioContext = new AudioContext();
    const plugins = [
      { name: 'Compressor', Plugin: CompressorPlugin },
      { name: 'Limiter', Plugin: LimiterPlugin },
      { name: 'Gate', Plugin: GatePlugin },
      { name: 'Expander', Plugin: ExpanderPlugin }
    ];

    for (const { name, Plugin } of plugins) {
      const plugin = new Plugin(audioContext);
      await plugin.initialize();

      // Generate 10 seconds of test audio
      const inputBuffer = this.generateTestBuffer(audioContext, 440, 10, 0.5);

      const startTime = performance.now();
      await plugin.processOffline(inputBuffer);
      const endTime = performance.now();

      const renderTime = endTime - startTime;
      const audioLength = 10000; // 10 seconds in ms
      const speedMultiplier = audioLength / renderTime;

      console.log(`${name}: ${speedMultiplier.toFixed(1)}x real-time (${renderTime.toFixed(0)}ms)`);

      if (speedMultiplier < 20) {
        console.warn(`  ⚠ Performance below 20x target`);
      }

      plugin.dispose();
    }

    await audioContext.close();
  }

  /**
   * Run all tests
   */
  async runAll() {
    console.log('=== Dynamics Plugins Test Suite ===\n');

    await this.testCompressor();
    await this.testLimiter();
    await this.testGate();
    await this.testExpander();
    await this.benchmarkPerformance();

    console.log('\n=== Test Results ===');
    console.log(`Passed: ${this.testsPassed}`);
    console.log(`Failed: ${this.testsFailed}`);
    console.log(`Total: ${this.testsPassed + this.testsFailed}`);

    if (this.testsFailed === 0) {
      console.log('\n✓ All tests passed!');
    } else {
      console.log('\n✗ Some tests failed');
      console.log('\nFailed tests:');
      this.results.filter(r => r.status === 'failed').forEach(r => {
        console.log(`  - ${r.name}: ${r.error}`);
      });
    }

    return this.testsFailed === 0;
  }
}

// Run tests if executed directly
if (typeof window !== 'undefined') {
  const tests = new DynamicsPluginTests();
  tests.runAll().then(success => {
    console.log(success ? '\nAll dynamics plugins working correctly!' : '\nSome tests failed - check output above');
  }).catch(error => {
    console.error('Test suite crashed:', error);
  });
}

export default DynamicsPluginTests;
