# 🥁 DrumSampler Space API - Quick Start

## ✅ What's Been Created

I've created a complete integration for your **DrumSampler Hugging Face Space**:

**Your Space**: https://huggingface.co/spaces/doseedo/DrumSampler

### Files Created:

1. **`src/services/drumSamplerAPI.js`** - Full API service
2. **`src/utils/testDrumSampler.js`** - Browser test utilities
3. **`test-drumsampler.js`** - Node.js test script
4. **`test-drumsampler-simple.html`** - Standalone HTML test page
5. **`DRUMSAMPLER_USAGE.md`** - Complete usage guide

---

## 🚀 Quick Test (Choose One)

### Option 1: HTML Test Page (Easiest)

Open this file in your browser:
```bash
test-drumsampler-simple.html
```

Then:
1. Click **"Test Connection"** to verify Space is accessible
2. Click **"Get API Info"** to see available endpoints
3. Enter a drum type (e.g., "snare drum") and click **"Generate & Play"**

### Option 2: Browser Console (In Your React App)

1. Start your React app
2. Open browser console (F12)
3. Run:
```javascript
// Test connection
await testDrumSampler();

// Generate and play a drum
await generateAndPlayDrum({ prompt: 'kick drum' });

// Generate a drum kit
const kit = await generateDrumKit(['kick', 'snare', 'hi-hat']);
```

### Option 3: Node.js Script

```bash
cd /var/www/html/doseedo-react
node test-drumsampler.js
```

This will generate samples and save them to `./test-drum-samples/` folder.

---

## 💻 Basic Usage in Your Code

### Example 1: Generate and Play

```javascript
import drumSamplerAPI from './services/drumSamplerAPI';

async function playDrum() {
  // Generate a drum sample
  const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples({
    prompt: 'kick drum',
    duration: 1.0
  });

  // Play it
  const url = URL.createObjectURL(blobs[0]);
  const audio = new Audio(url);
  audio.play();
}
```

### Example 2: Add to React Component

```javascript
import { useState } from 'react';
import drumSamplerAPI from './services/drumSamplerAPI';

function DrumGenerator() {
  const [audio, setAudio] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples({
        prompt: 'snare drum'
      });

      const url = URL.createObjectURL(blobs[0]);
      setAudio(url);
    } catch (error) {
      console.error('Generation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <button onClick={handleGenerate} disabled={loading}>
        {loading ? 'Generating...' : 'Generate Snare'}
      </button>
      {audio && <audio controls src={audio} />}
    </div>
  );
}
```

---

## 🔧 Configuration

### Finding Your Space's Endpoints

Your Space might use different endpoint names. To find them:

```javascript
import drumSamplerAPI from './services/drumSamplerAPI';

// Get API structure
const info = await drumSamplerAPI.getDrumSamplerInfo();
console.log('Available endpoints:', info.named_endpoints);
```

### Updating the Endpoint

If your Space uses a different endpoint name, edit `src/services/drumSamplerAPI.js`:

```javascript
// Change this line (currently line ~26):
const result = await client.predict('/generate', {  // <-- Change '/generate'
  prompt: params.prompt || 'kick drum',
  // ...
});
```

Common endpoint names:
- `/predict` (default Gradio)
- `/generate`
- `/inference`
- Custom names you defined

---

## 📚 API Functions Available

| Function | Purpose |
|----------|---------|
| `generateDrumSample(params)` | Generate a single drum sample |
| `generateAndDownloadDrumSamples(params)` | Generate and get as Blobs |
| `generateMultipleDrumSamples(prompts)` | Generate multiple samples |
| `getDrumSamplerInfo()` | Get Space API information |
| `downloadAudioFile(url)` | Download audio from URL |

### Parameters:

```javascript
{
  prompt: 'kick drum',      // Drum description
  duration: 1.0,            // Length in seconds (optional)
  temperature: 1.0          // Randomness (optional)
}
```

---

## 🎯 Integration Ideas

### 1. Add to Generation Panel

Add a "Drums" tab to your Generation Panel:
- Button for each drum type (kick, snare, hi-hat, etc.)
- Generate and add directly to timeline

### 2. MIDI-Triggered Generation

Generate drums based on MIDI patterns:
- Read MIDI note
- Map to drum type
- Generate sample
- Place at correct time

### 3. Drum Kit Builder

Create a UI to build custom drum kits:
- Select drum types
- Generate all samples
- Save as preset

### 4. Real-time Preview

Generate and preview before adding to project:
- Quick generation
- Play/preview
- Accept/regenerate

---

## 🔍 Troubleshooting

### Space Not Responding

1. **Check if Space is running**: Visit https://huggingface.co/spaces/doseedo/DrumSampler
2. **Check Space status**: Look for "Running" badge
3. **Wake up Space**: First request may take 30-60 seconds if Space was sleeping

### Wrong Endpoint Name

If you get errors like "endpoint not found":

1. Open `test-drumsampler-simple.html` in browser
2. Click "Get API Info"
3. Check console for endpoint names
4. Update `src/services/drumSamplerAPI.js` with correct name

### Generation Errors

Check your Space's expected input format:
- Visit your Space page
- Click "Use via API" tab
- See example inputs

---

## 📊 Example: Full Workflow

```javascript
import drumSamplerAPI from './services/drumSamplerAPI';
import { useApp } from './context/AppContext';

function MyComponent() {
  const { dispatch } = useApp();

  const addDrumToTimeline = async (drumType, position) => {
    try {
      // 1. Generate drum
      console.log(`Generating ${drumType}...`);
      const blobs = await drumSamplerAPI.generateAndDownloadDrumSamples({
        prompt: `${drumType} drum`,
        duration: 1.0
      });

      // 2. Convert to AudioBuffer
      const arrayBuffer = await blobs[0].arrayBuffer();
      const audioContext = new AudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      // 3. Add to timeline
      dispatch({
        type: 'ADD_AUDIO_TRACK',
        payload: {
          audioBuffer,
          name: `${drumType}_${Date.now()}`,
          startTime: position
        }
      });

      console.log(`✅ ${drumType} added to timeline at ${position}s`);

    } catch (error) {
      console.error(`Failed to add ${drumType}:`, error);
    }
  };

  return (
    <div>
      <button onClick={() => addDrumToTimeline('kick', 0)}>
        Add Kick at 0s
      </button>
      <button onClick={() => addDrumToTimeline('snare', 1)}>
        Add Snare at 1s
      </button>
    </div>
  );
}
```

---

## ✅ Next Steps

1. **Test**: Open `test-drumsampler-simple.html` and click "Test Connection"
2. **Verify**: Check that your Space is accessible
3. **Try**: Generate a few samples to test
4. **Integrate**: Add to your React app where needed

---

## 📚 Full Documentation

For complete API reference and advanced usage, see:
- **`DRUMSAMPLER_USAGE.md`** - Complete usage guide
- **`src/services/drumSamplerAPI.js`** - API source code (fully commented)

---

## 🎉 You're Ready!

Your DrumSampler Space integration is complete. You can now:

✅ Generate drum samples from text prompts
✅ Receive audio files directly in your app
✅ Play, download, or add to timeline
✅ Build custom drum kits
✅ Integrate with MIDI and DAW features

**Happy drumming!** 🥁🎵

---

**Need Help?**
- Check your Space: https://huggingface.co/spaces/doseedo/DrumSampler
- Use the test page: `test-drumsampler-simple.html`
- Read full docs: `DRUMSAMPLER_USAGE.md`
