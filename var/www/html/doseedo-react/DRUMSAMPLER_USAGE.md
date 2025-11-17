# DrumSampler Space API Usage Guide

## 🥁 Overview

This integration allows you to call your **DrumSampler Hugging Face Space** directly from your React app and receive generated drum audio files.

**Your Space**: https://huggingface.co/spaces/doseedo/DrumSampler

---

## 📦 Installation

Already installed! The following packages are now in your project:
- `@gradio/client` - For connecting to Gradio Spaces
- DrumSampler API service created

---

## 🚀 Basic Usage

### Method 1: In React Components

```javascript
import drumSamplerAPI from './services/drumSamplerAPI';

// In your component:
async function handleGenerateDrum() {
  try {
    const result = await drumSamplerAPI.generateDrumSample({
      prompt: 'deep kick drum',
      duration: 1.0,
      temperature: 1.0
    });

    console.log('Generated files:', result.files);

    // result.files contains URLs or Blobs of the generated audio
  } catch (error) {
    console.error('Generation failed:', error);
  }
}
```

### Method 2: Browser Console (for testing)

Open your React app in browser, then in console:

```javascript
// Test connection
await testDrumSampler();

// Generate and play a drum
await generateAndPlayDrum({ prompt: 'kick drum' });

// Generate a full drum kit
const kit = await generateDrumKit(['kick', 'snare', 'hi-hat', 'crash']);

// Download a sample
await generateAndDownloadDrum({ prompt: 'snare drum' }, 'my_snare.wav');
```

---

## 📚 API Reference

### `generateDrumSample(params)`

Generate a drum sample from your DrumSampler Space.

**Parameters:**
- `prompt` (string) - Description of the drum sound (e.g., "kick drum", "snare", "hi-hat")
- `duration` (number, optional) - Length in seconds (default: 1.0)
- `temperature` (number, optional) - Generation randomness (default: 1.0)

**Returns:**
```javascript
{
  success: true,
  data: [...],
  files: [audio_url_or_blob, ...]
}
```

**Example:**
```javascript
const result = await drumSamplerAPI.generateDrumSample({
  prompt: 'crisp snare drum',
  duration: 0.8,
  temperature: 0.9
});
```

---

### `generateAndDownloadDrumSamples(params)`

Generate drum samples and automatically download them as Blobs.

**Parameters:** Same as `generateDrumSample`

**Returns:** `Array<Blob>` - Array of audio Blobs

**Example:**
```javascript
const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples({
  prompt: 'hi-hat'
});

// Use the blob
const url = URL.createObjectURL(blobs[0]);
const audio = new Audio(url);
audio.play();
```

---

### `generateMultipleDrumSamples(prompts)`

Generate multiple drum samples in sequence.

**Parameters:**
- `prompts` (Array<Object>) - Array of parameter objects

**Returns:** `Array<Object>` - Array of results

**Example:**
```javascript
const results = await drumSamplerAPI.generateMultipleDrumSamples([
  { prompt: 'kick drum', duration: 1.0 },
  { prompt: 'snare drum', duration: 0.5 },
  { prompt: 'hi-hat', duration: 0.3 }
]);
```

---

### `getDrumSamplerInfo()`

Get information about your DrumSampler Space API.

**Returns:** Object with API structure and available endpoints

**Example:**
```javascript
const info = await drumSamplerAPI.getDrumSamplerInfo();
console.log('Available endpoints:', Object.keys(info.named_endpoints));
```

---

## 🎵 Example Use Cases

### Use Case 1: Generate Drum Kit on Button Click

```javascript
import { useState } from 'react';
import drumSamplerAPI from './services/drumSamplerAPI';

function DrumGenerator() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples({
        prompt: 'kick drum',
        duration: 1.0
      });

      const url = URL.createObjectURL(blobs[0]);
      setAudioUrl(url);
    } catch (error) {
      console.error('Generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div>
      <button onClick={handleGenerate} disabled={isGenerating}>
        {isGenerating ? 'Generating...' : 'Generate Kick Drum'}
      </button>

      {audioUrl && (
        <audio controls src={audioUrl} />
      )}
    </div>
  );
}
```

---

### Use Case 2: Generate Drum Sequence

```javascript
async function createDrumSequence() {
  const drumTypes = ['kick', 'snare', 'hi-hat', 'tom', 'crash'];
  const drumKit = [];

  for (const type of drumTypes) {
    const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples({
      prompt: `${type} drum`,
      duration: 1.0
    });

    drumKit.push({
      name: type,
      audioBlob: blobs[0],
      audioUrl: URL.createObjectURL(blobs[0])
    });
  }

  // Now you have a complete drum kit!
  return drumKit;
}
```

---

### Use Case 3: Add to DAW Track

```javascript
async function addDrumToTrack(trackId, drumType) {
  try {
    // Generate the drum
    const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples({
      prompt: `${drumType} drum`
    });

    // Convert blob to audio buffer
    const arrayBuffer = await blobs[0].arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Add to your DAW track
    dispatch({
      type: 'ADD_AUDIO_TO_TRACK',
      payload: {
        trackId,
        audioBuffer,
        name: `${drumType}_sample`
      }
    });

    console.log(`✅ Added ${drumType} to track ${trackId}`);

  } catch (error) {
    console.error('Failed to add drum to track:', error);
  }
}
```

---

## 🧪 Testing

### Test from Command Line (Node.js):

```bash
cd /var/www/html/doseedo-react
node test-drumsampler.js
```

This will:
1. Connect to your DrumSampler Space
2. Get API information
3. Generate several drum samples
4. Save them to `./test-drum-samples/` folder

### Test from Browser Console:

```javascript
// Quick connection test
await testDrumSampler();

// Generate and play
await generateAndPlayDrum({ prompt: 'snare drum' });
```

---

## ⚙️ Configuration

### Adjusting the Endpoint

If your Space uses a different endpoint name than `/generate`, update it in `drumSamplerAPI.js`:

```javascript
const result = await client.predict('/your_endpoint_name', {
  prompt: params.prompt,
  // ... other params
});
```

### Adding More Parameters

Your Space might accept additional parameters. Add them to the API service:

```javascript
export async function generateDrumSample(params = {}) {
  const result = await client.predict('/generate', {
    prompt: params.prompt || 'kick drum',
    duration: params.duration || 1.0,
    temperature: params.temperature || 1.0,
    seed: params.seed || -1,  // Add new parameter
    variation: params.variation || 0  // Add new parameter
  });

  return result;
}
```

---

## 🔧 Troubleshooting

### Space Not Responding

1. **Check if Space is running**: https://huggingface.co/spaces/doseedo/DrumSampler
2. **Check Space logs** for errors
3. **Verify endpoint names** match your Space's API

### Finding Endpoint Names

```javascript
// Get your Space's API structure
const info = await drumSamplerAPI.getDrumSamplerInfo();
console.log(info);
```

This will show all available endpoints and their parameters.

### Connection Timeouts

If the Space is sleeping, first request might take 30-60 seconds to wake it up:

```javascript
console.log('Waking up Space...');
const result = await drumSamplerAPI.generateDrumSample({
  prompt: 'test'
});
// Subsequent requests will be faster
```

---

## 📁 Files Created

1. **`src/services/drumSamplerAPI.js`** - Main API service
2. **`src/utils/testDrumSampler.js`** - Browser-friendly test utilities
3. **`test-drumsampler.js`** - Node.js test script
4. **`DRUMSAMPLER_USAGE.md`** - This guide

---

## 🎯 Integration with Your App

### Option 1: Add to Generation Panel

Modify `src/components/GenerationPanel/GenerationPanelOptimized.js`:

```javascript
import drumSamplerAPI from '../../services/drumSamplerAPI';

// Add drum generation option
const handleGenerateDrums = async () => {
  const drumBlob = await drumSamplerAPI.generateAndDownloadDrumSamples({
    prompt: 'kick drum'
  });

  // Add to your track
  addAudioToTrack(drumBlob[0]);
};
```

### Option 2: Create Drum Generator Component

Create a new component specifically for drum generation with buttons for different drum types.

### Option 3: Use in MIDI Mode

Generate drums based on MIDI patterns:

```javascript
async function generateDrumsFromMIDI(midiNotes) {
  for (const note of midiNotes) {
    const drumType = mapMIDIToDrum(note.pitch);
    const blob = await drumSamplerAPI.generateDrumSample({
      prompt: drumType
    });
    // Place at note.time position
  }
}
```

---

## ✅ Next Steps

1. **Test the connection**: Run `node test-drumsampler.js`
2. **Try in browser**: Open app and run `testDrumSampler()` in console
3. **Integrate**: Add to your Generation Panel or create dedicated drum UI
4. **Customize**: Adjust parameters based on your Space's capabilities

---

**Questions?**
- Check your Space: https://huggingface.co/spaces/doseedo/DrumSampler
- View Space API: Click "Use via API" button on Space page
- Test locally: Use the test scripts provided

Happy drumming! 🥁🎵
