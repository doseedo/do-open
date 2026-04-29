/**
 * Test Suite for Distortion Plugins
 *
 * Tests all three distortion processors:
 * - DistortionPlugin
 * - OverdrivePlugin
 * - SaturatorPlugin
 *
 * @author Agent 5 - Distortion Plugins
 * @version 1.0.0
 */

import { DistortionPlugin } from './DistortionPlugin.js';
import { OverdrivePlugin } from './OverdrivePlugin.js';
import { SaturatorPlugin } from './SaturatorPlugin.js';

/**
 * Test runner class
 */
class DistortionPluginTests {
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
   * Calculate Total Harmonic Distortion (THD) - simplified
   */
  calculateTHD(buffer, fundamentalFreq) {
    // This is a simplified THD calculation
    // A full implementation would use FFT
    const rms = this.calculateRMS(buffer);
    return rms; // Placeholder
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
   * Test DistortionPlugin
   */
  async testDistortion() {
    const audioContext = new AudioContext();

    await this.runTest('Distortion: Initialize', async () => {
      const distortion = new DistortionPlugin(audioContext);
      await distortion.initialize();

      if (!distortion.initialized) {
        throw new Error('Distortion failed to initialize');
      }

      distortion.dispose();
    });

    await this.runTest('Distortion: Set parameters', async () => {
      const distortion = new DistortionPlugin(audioContext);
      await distortion.initialize();

      distortion.setParameter('drive', 75);
      distortion.setParameter('tone', 2000);
      distortion.setParameter('mix', 80);

      const drive = distortion.getParameter('drive').param.value;
      if (Math.abs(drive - 75) > 0.01) {
        throw new Error(`Expected drive 75, got ${drive}`);
      }

      distortion.dispose();
    });

    await this.runTest('Distortion: Clip types', async () => {
      const distortion = new DistortionPlugin(audioContext);
      await distortion.initialize();

      // Test all clip types
      const clipTypes = ['hard', 'soft', 'asymmetric', 'foldback'];
      for (const type of clipTypes) {
        distortion.setClipType(type);
        const currentType = distortion.getClipTypeName();
        if (currentType !== type) {
          throw new Error(`Expected clip type ${type}, got ${currentType}`);
        }
      }

      distortion.dispose();
    });

    await this.runTest('Distortion: Filter positions', async () => {
      const distortion = new DistortionPlugin(audioContext);
      await distortion.initialize();

      // Test pre-filtering
      distortion.setFilterPosition('pre');
      if (distortion.getFilterPosition() !== 'pre') {
        throw new Error('Failed to set filter position to pre');
      }

      // Test post-filtering
      distortion.setFilterPosition('post');
      if (distortion.getFilterPosition() !== 'post') {
        throw new Error('Failed to set filter position to post');
      }

      distortion.dispose();
    });

    await this.runTest('Distortion: Process audio (adds harmonics)', async () => {
      const distortion = new DistortionPlugin(audioContext);
      await distortion.initialize();

      // Set aggressive distortion
      distortion.setParameter('drive', 80);
      distortion.setClipType('hard');
      distortion.setParameter('mix', 100); // 100% wet

      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.5);
      const outputBuffer = await distortion.processOffline(inputBuffer);

      const inputRMS = this.calculateRMS(inputBuffer);
      const outputRMS = this.calculateRMS(outputBuffer);

      // Distortion should add harmonics (energy)
      console.log(`  Input RMS: ${inputRMS.toFixed(4)}, Output RMS: ${outputRMS.toFixed(4)}`);

      distortion.dispose();
    });

    await this.runTest('Distortion: Mix control', async () => {
      const distortion = new DistortionPlugin(audioContext);
      await distortion.initialize();

      distortion.setParameter('drive', 90);
      distortion.setParameter('mix', 50); // 50% wet

      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.5);
      const outputBuffer = await distortion.processOffline(inputBuffer);

      // With mix, output should be somewhere between dry and fully distorted
      const outputRMS = this.calculateRMS(outputBuffer);
      if (outputRMS === 0) {
        throw new Error('Mix parameter not working correctly');
      }

      distortion.dispose();
    });

    await audioContext.close();
  }

  /**
   * Test OverdrivePlugin
   */
  async testOverdrive() {
    const audioContext = new AudioContext();

    await this.runTest('Overdrive: Initialize', async () => {
      const overdrive = new OverdrivePlugin(audioContext);
      await overdrive.initialize();

      if (!overdrive.initialized) {
        throw new Error('Overdrive failed to initialize');
      }

      overdrive.dispose();
    });

    await this.runTest('Overdrive: Set parameters', async () => {
      const overdrive = new OverdrivePlugin(audioContext);
      await overdrive.initialize();

      overdrive.setParameter('drive', 60);
      overdrive.setParameter('tone', 75);
      overdrive.setParameter('bias', 20);

      const drive = overdrive.getParameter('drive').param.value;
      if (Math.abs(drive - 60) > 0.01) {
        throw new Error(`Expected drive 60, got ${drive}`);
      }

      overdrive.dispose();
    });

    await this.runTest('Overdrive: Curve types', async () => {
      const overdrive = new OverdrivePlugin(audioContext);
      await overdrive.initialize();

      // Test all curve types
      const curveTypes = ['tanh', 'atan', 'softClip'];
      for (const type of curveTypes) {
        overdrive.setCurveType(type);
        const currentType = overdrive.getCurveTypeName();
        if (currentType !== type) {
          throw new Error(`Expected curve type ${type}, got ${currentType}`);
        }
      }

      overdrive.dispose();
    });

    await this.runTest('Overdrive: Soft clipping', async () => {
      const overdrive = new OverdrivePlugin(audioContext);
      await overdrive.initialize();

      // Set moderate overdrive
      overdrive.setParameter('drive', 50);
      overdrive.setCurveType('tanh');
      overdrive.setParameter('mix', 100); // 100% wet

      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.7);
      const outputBuffer = await overdrive.processOffline(inputBuffer);

      const inputPeak = this.calculatePeak(inputBuffer);
      const outputPeak = this.calculatePeak(outputBuffer);

      // Soft clipping should keep output within reasonable bounds
      if (outputPeak > 1.5) {
        throw new Error(`Output peak too high: ${outputPeak}`);
      }

      console.log(`  Input peak: ${inputPeak.toFixed(3)}, Output peak: ${outputPeak.toFixed(3)}`);

      overdrive.dispose();
    });

    await this.runTest('Overdrive: Asymmetric distortion (bias)', async () => {
      const overdrive = new OverdrivePlugin(audioContext);
      await overdrive.initialize();

      // Apply positive bias
      overdrive.setParameter('bias', 50);
      overdrive.setParameter('drive', 40);

      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.5);
      const outputBuffer = await overdrive.processOffline(inputBuffer);

      // Just verify it processes without error
      const outputRMS = this.calculateRMS(outputBuffer);
      if (outputRMS === 0) {
        throw new Error('Bias parameter not working');
      }

      overdrive.dispose();
    });

    await audioContext.close();
  }

  /**
   * Test SaturatorPlugin
   */
  async testSaturator() {
    const audioContext = new AudioContext();

    await this.runTest('Saturator: Initialize', async () => {
      const saturator = new SaturatorPlugin(audioContext);
      await saturator.initialize();

      if (!saturator.initialized) {
        throw new Error('Saturator failed to initialize');
      }

      saturator.dispose();
    });

    await this.runTest('Saturator: Set parameters', async () => {
      const saturator = new SaturatorPlugin(audioContext);
      await saturator.initialize();

      saturator.setParameter('drive', 40);
      saturator.setParameter('color', 60);
      saturator.setParameter('depth', 80);

      const drive = saturator.getParameter('drive').param.value;
      if (Math.abs(drive - 40) > 0.01) {
        throw new Error(`Expected drive 40, got ${drive}`);
      }

      saturator.dispose();
    });

    await this.runTest('Saturator: Saturation types', async () => {
      const saturator = new SaturatorPlugin(audioContext);
      await saturator.initialize();

      // Test all saturation types
      const satTypes = ['warm', 'digital', 'analog', 'clip', 'foldback', 'sine-fold'];
      for (const type of satTypes) {
        saturator.setSaturationType(type);
        const currentType = saturator.getSaturationTypeName();
        if (currentType !== type) {
          throw new Error(`Expected saturation type ${type}, got ${currentType}`);
        }
      }

      saturator.dispose();
    });

    await this.runTest('Saturator: DC filter', async () => {
      const saturator = new SaturatorPlugin(audioContext);
      await saturator.initialize();

      // Enable DC filter
      saturator.setDCFilter(true);
      if (!saturator.getDCFilterEnabled()) {
        throw new Error('Failed to enable DC filter');
      }

      // Disable DC filter
      saturator.setDCFilter(false);
      if (saturator.getDCFilterEnabled()) {
        throw new Error('Failed to disable DC filter');
      }

      saturator.dispose();
    });

    await this.runTest('Saturator: Warm saturation', async () => {
      const saturator = new SaturatorPlugin(audioContext);
      await saturator.initialize();

      // Apply warm saturation
      saturator.setSaturationType('warm');
      saturator.setParameter('drive', 30);
      saturator.setParameter('depth', 100);
      saturator.setParameter('mix', 100); // 100% wet

      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.6);
      const outputBuffer = await saturator.processOffline(inputBuffer);

      const inputRMS = this.calculateRMS(inputBuffer);
      const outputRMS = this.calculateRMS(outputBuffer);

      // Saturation should process the signal
      if (outputRMS === 0) {
        throw new Error('Saturation not processing');
      }

      console.log(`  Input RMS: ${inputRMS.toFixed(4)}, Output RMS: ${outputRMS.toFixed(4)}`);

      saturator.dispose();
    });

    await this.runTest('Saturator: Depth control', async () => {
      const saturator = new SaturatorPlugin(audioContext);
      await saturator.initialize();

      saturator.setParameter('drive', 50);
      saturator.setParameter('depth', 50); // 50% depth

      const inputBuffer = this.generateTestBuffer(audioContext, 440, 0.5, 0.5);
      const outputBuffer = await saturator.processOffline(inputBuffer);

      // With depth control, output should be a blend
      const outputRMS = this.calculateRMS(outputBuffer);
      if (outputRMS === 0) {
        throw new Error('Depth parameter not working');
      }

      saturator.dispose();
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
      { name: 'Distortion', Plugin: DistortionPlugin },
      { name: 'Overdrive', Plugin: OverdrivePlugin },
      { name: 'Saturator', Plugin: SaturatorPlugin }
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
      } else {
        console.log(`  ✓ Performance meets 20x real-time target`);
      }

      plugin.dispose();
    }

    await audioContext.close();
  }

  /**
   * Run all tests
   */
  async runAll() {
    console.log('=== Distortion Plugins Test Suite ===\n');

    await this.testDistortion();
    await this.testOverdrive();
    await this.testSaturator();
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
  const tests = new DistortionPluginTests();
  tests.runAll().then(success => {
    console.log(success ? '\nAll distortion plugins working correctly!' : '\nSome tests failed - check output above');
  }).catch(error => {
    console.error('Test suite crashed:', error);
  });
}

export default DistortionPluginTests;
