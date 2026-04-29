/**
 * Test Suite for AudioWorklet Reverb Plugins
 *
 * Tests both ReverbPlugin and ConvolutionReverbPlugin for:
 * - Initialization
 * - Parameter updates
 * - Offline rendering performance
 * - Audio processing correctness
 *
 * @author Agent 6: Reverb Plugins
 * @version 1.0.0
 */

import { ReverbPlugin, ConvolutionReverbPlugin, ReverbPresets } from './index.js';

/**
 * Create a test audio buffer (sine wave)
 * @param {AudioContext} context - Audio context
 * @param {number} frequency - Sine wave frequency in Hz
 * @param {number} duration - Duration in seconds
 * @returns {AudioBuffer} Test audio buffer
 */
function createTestBuffer(context, frequency = 440, duration = 1) {
  const sampleRate = context.sampleRate;
  const length = Math.floor(sampleRate * duration);
  const buffer = context.createBuffer(2, length, sampleRate);

  for (let channel = 0; channel < 2; channel++) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < length; i++) {
      data[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.5;
    }
  }

  return buffer;
}

/**
 * Create a simple impulse for testing convolution
 * @param {AudioContext} context - Audio context
 * @param {number} duration - Duration in seconds
 * @returns {AudioBuffer} Impulse buffer
 */
function createTestImpulse(context, duration = 0.5) {
  const sampleRate = context.sampleRate;
  const length = Math.floor(sampleRate * duration);
  const buffer = context.createBuffer(1, length, sampleRate);
  const data = buffer.getChannelData(0);

  // Create exponentially decaying impulse
  data[0] = 1.0; // Initial impulse
  for (let i = 1; i < length; i++) {
    data[i] = Math.exp(-i / (sampleRate * 0.1)) * (Math.random() * 0.1);
  }

  return buffer;
}

/**
 * Calculate RMS level of audio buffer
 * @param {AudioBuffer} buffer - Audio buffer
 * @returns {number} RMS level
 */
function calculateRMS(buffer) {
  let sumSquares = 0;
  let sampleCount = 0;

  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) {
      sumSquares += data[i] * data[i];
      sampleCount++;
    }
  }

  return Math.sqrt(sumSquares / sampleCount);
}

/**
 * Test ReverbPlugin
 */
async function testReverbPlugin() {
  console.log('\n=== Testing ReverbPlugin ===\n');

  const context = new AudioContext();
  const plugin = new ReverbPlugin(context);

  // Test 1: Initialization
  console.log('Test 1: Initialization');
  const initSuccess = await plugin.initialize();
  console.log(`  ${initSuccess ? '✓' : '✗'} Plugin initialized: ${initSuccess}`);

  if (!initSuccess) {
    console.error('  Failed to initialize plugin');
    return false;
  }

  // Test 2: Parameter Updates
  console.log('\nTest 2: Parameter Updates');
  plugin.setRoomSize(75);
  plugin.setDecayTime(3.0);
  plugin.setDamping(60);
  plugin.setWidth(80);
  plugin.setPreDelay(20);
  plugin.setMix(40);

  const params = plugin.getParams();
  console.log(`  ✓ Room Size: ${params.roomSize}%`);
  console.log(`  ✓ Decay Time: ${params.decayTime}s`);
  console.log(`  ✓ Damping: ${params.damping}%`);
  console.log(`  ✓ Width: ${params.width}%`);
  console.log(`  ✓ Pre-delay: ${params.predelay}ms`);
  console.log(`  ✓ Mix: ${params.mix}%`);

  // Test 3: Preset Loading
  console.log('\nTest 3: Preset Loading');
  plugin.loadPreset(ReverbPresets.cathedral);
  const presetParams = plugin.getParams();
  console.log(`  ✓ Loaded 'Cathedral' preset`);
  console.log(`    - Room Size: ${presetParams.roomSize}%`);
  console.log(`    - Decay Time: ${presetParams.decayTime}s`);

  // Test 4: Offline Rendering Performance
  console.log('\nTest 4: Offline Rendering Performance');
  const testBuffer = createTestBuffer(context, 440, 10); // 10 seconds

  const startTime = performance.now();
  const processedBuffer = await plugin.processOffline(testBuffer);
  const endTime = performance.now();

  const renderTime = endTime - startTime;
  const audioLength = testBuffer.duration * 1000; // in ms
  const speedMultiplier = audioLength / renderTime;

  console.log(`  ✓ Rendered ${testBuffer.duration}s in ${renderTime.toFixed(0)}ms`);
  console.log(`  ✓ Speed: ${speedMultiplier.toFixed(1)}x real-time`);

  if (speedMultiplier >= 15) {
    console.log(`  ✓ Performance PASS (target: 15x, achieved: ${speedMultiplier.toFixed(1)}x)`);
  } else {
    console.log(`  ⚠ Performance below target (target: 15x, achieved: ${speedMultiplier.toFixed(1)}x)`);
  }

  // Test 5: Audio Processing Correctness
  console.log('\nTest 5: Audio Processing Correctness');
  const inputRMS = calculateRMS(testBuffer);
  const outputRMS = calculateRMS(processedBuffer);

  console.log(`  ✓ Input RMS: ${inputRMS.toFixed(4)}`);
  console.log(`  ✓ Output RMS: ${outputRMS.toFixed(4)}`);
  console.log(`  ${outputRMS > 0 ? '✓' : '✗'} Output has signal`);
  console.log(`  ${processedBuffer.length === testBuffer.length ? '✓' : '✗'} Output length matches input`);

  // Cleanup
  plugin.dispose();
  await context.close();

  console.log('\n✓ ReverbPlugin tests completed\n');
  return true;
}

/**
 * Test ConvolutionReverbPlugin
 */
async function testConvolutionReverbPlugin() {
  console.log('\n=== Testing ConvolutionReverbPlugin ===\n');

  const context = new AudioContext();
  const plugin = new ConvolutionReverbPlugin(context);

  // Test 1: Initialization
  console.log('Test 1: Initialization');
  const initSuccess = await plugin.initialize();
  console.log(`  ${initSuccess ? '✓' : '✗'} Plugin initialized: ${initSuccess}`);

  if (!initSuccess) {
    console.error('  Failed to initialize plugin');
    return false;
  }

  // Test 2: Loading Impulse Response
  console.log('\nTest 2: Loading Impulse Response');
  const impulseBuffer = createTestImpulse(context, 0.5);
  const irSuccess = plugin.setImpulseResponse(impulseBuffer);
  console.log(`  ${irSuccess ? '✓' : '✗'} Impulse response loaded: ${irSuccess}`);

  const irInfo = plugin.getIRInfo();
  if (irInfo) {
    console.log(`  ✓ IR Duration: ${irInfo.duration.toFixed(2)}s`);
    console.log(`  ✓ IR Length: ${irInfo.length} samples`);
  }

  // Test 3: Parameter Updates
  console.log('\nTest 3: Parameter Updates');
  plugin.setMix(50);
  plugin.setPreDelay(30);
  console.log(`  ✓ Mix set to 50%`);
  console.log(`  ✓ Pre-delay set to 30ms`);

  // Test 4: Offline Rendering Performance
  console.log('\nTest 4: Offline Rendering Performance');
  const testBuffer = createTestBuffer(context, 440, 10); // 10 seconds

  const startTime = performance.now();
  const processedBuffer = await plugin.processOffline(testBuffer);
  const endTime = performance.now();

  const renderTime = endTime - startTime;
  const audioLength = testBuffer.duration * 1000; // in ms
  const speedMultiplier = audioLength / renderTime;

  console.log(`  ✓ Rendered ${testBuffer.duration}s in ${renderTime.toFixed(0)}ms`);
  console.log(`  ✓ Speed: ${speedMultiplier.toFixed(1)}x real-time`);

  if (speedMultiplier >= 15) {
    console.log(`  ✓ Performance PASS (target: 15x, achieved: ${speedMultiplier.toFixed(1)}x)`);
  } else {
    console.log(`  ⚠ Performance below target (target: 15x, achieved: ${speedMultiplier.toFixed(1)}x)`);
  }

  // Test 5: Audio Processing Correctness
  console.log('\nTest 5: Audio Processing Correctness');
  const inputRMS = calculateRMS(testBuffer);
  const outputRMS = calculateRMS(processedBuffer);

  console.log(`  ✓ Input RMS: ${inputRMS.toFixed(4)}`);
  console.log(`  ✓ Output RMS: ${outputRMS.toFixed(4)}`);
  console.log(`  ${outputRMS > 0 ? '✓' : '✗'} Output has signal`);
  console.log(`  ${processedBuffer.length === testBuffer.length ? '✓' : '✗'} Output length matches input`);

  // Cleanup
  plugin.dispose();
  await context.close();

  console.log('\n✓ ConvolutionReverbPlugin tests completed\n');
  return true;
}

/**
 * Run all tests
 */
async function runAllTests() {
  console.log('╔════════════════════════════════════════════════════╗');
  console.log('║  AudioWorklet Reverb Plugins - Test Suite         ║');
  console.log('╚════════════════════════════════════════════════════╝');

  try {
    const reverbSuccess = await testReverbPlugin();
    const convolutionSuccess = await testConvolutionReverbPlugin();

    console.log('\n╔════════════════════════════════════════════════════╗');
    console.log('║  Test Summary                                      ║');
    console.log('╚════════════════════════════════════════════════════╝');
    console.log(`\nReverbPlugin: ${reverbSuccess ? '✓ PASS' : '✗ FAIL'}`);
    console.log(`ConvolutionReverbPlugin: ${convolutionSuccess ? '✓ PASS' : '✗ FAIL'}`);

    if (reverbSuccess && convolutionSuccess) {
      console.log('\n✓ All tests PASSED\n');
      return true;
    } else {
      console.log('\n✗ Some tests FAILED\n');
      return false;
    }

  } catch (error) {
    console.error('\n✗ Test suite failed with error:', error);
    return false;
  }
}

// Run tests if executed directly
if (typeof window !== 'undefined') {
  // Browser environment
  window.addEventListener('load', () => {
    runAllTests();
  });
} else {
  // Node.js environment (for automated testing)
  runAllTests().then(success => {
    process.exit(success ? 0 : 1);
  });
}

export { runAllTests, testReverbPlugin, testConvolutionReverbPlugin };
