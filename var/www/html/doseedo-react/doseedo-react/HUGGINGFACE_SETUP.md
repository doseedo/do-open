# Hugging Face API Setup Guide

This guide will help you connect your Doseedo React app with Hugging Face Spaces for AI-powered audio generation.

## Prerequisites

1. A Hugging Face account (free to create)
2. Node.js and npm installed
3. This Doseedo React application

## Step 1: Get Your Hugging Face API Token

1. Go to [Hugging Face](https://huggingface.co/) and sign in (or create an account)
2. Navigate to your [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Click **"New token"**
4. Give it a name (e.g., "Doseedo App")
5. Select **"Read"** permission (or "Write" if you plan to upload models)
6. Click **"Generate"**
7. **Copy the token** (you won't be able to see it again!)

## Step 2: Configure Your Environment

1. Open the `.env` file in the root of your React project (`/var/www/html/doseedo-react/.env`)

2. Paste your API token:
   ```bash
   REACT_APP_HF_API_TOKEN=hf_your_token_here
   ```

3. (Optional) Configure specific models:
   ```bash
   # Default models are already set, but you can customize:
   REACT_APP_HF_AUDIO_MODEL=facebook/musicgen-small
   REACT_APP_HF_TEXT_TO_MUSIC_MODEL=facebook/musicgen-melody
   REACT_APP_HF_VOICE_MODEL=suno/bark
   ```

4. Save the file

## Step 3: Install Dependencies

The Hugging Face API service is already included in your project. No additional packages needed!

## Step 4: Test Your Connection

### Option 1: Browser Console Test

1. Start your development server (if not already running):
   ```bash
   npm start
   ```

2. Open the app in your browser
3. Open the browser console (F12 or right-click > Inspect > Console)
4. Run:
   ```javascript
   testHFConnection()
   ```

5. You should see:
   ```
   🧪 Testing Hugging Face API connection...
   ✓ Configuration check: PASS
   ✓ Authentication check: PASS
   ✓ Model status: READY (or LOADING)
   ✅ All tests passed!
   ```

### Option 2: Test Music Generation

In the browser console, try generating a short audio clip:

```javascript
testMusicGeneration().then(audioBlob => {
  // Play the generated audio
  const url = URL.createObjectURL(audioBlob);
  const audio = new Audio(url);
  audio.play();
});
```

## Step 5: Using HF API in Your App

### Import the HF API Service

```javascript
import * as hfAPI from './services/huggingfaceAPI';
```

### Generate Music from Text

```javascript
// Generate music from a text prompt
const audioBlob = await hfAPI.generateMusicFromText(
  'upbeat electronic dance music with a catchy melody',
  { duration: 250 } // ~10 seconds
);

// Create a URL and play it
const audioUrl = URL.createObjectURL(audioBlob);
const audio = new Audio(audioUrl);
audio.play();
```

### Generate Music from Melody (Audio Conditioning)

```javascript
// Use an existing audio file as a melody
const audioFile = document.querySelector('input[type="file"]').files[0];

const audioBlob = await hfAPI.generateMusicFromMelody(
  audioFile,
  'transform this into orchestral music',
  { duration: 250 }
);
```

### Generate Speech/Voice

```javascript
const speechBlob = await hfAPI.generateSpeech(
  'Hello, this is an AI-generated voice'
);
```

## Available Models

The default models configured are:

1. **facebook/musicgen-small** - Fast text-to-music generation
2. **facebook/musicgen-melody** - Melody-conditioned music generation
3. **suno/bark** - Text-to-speech with voice cloning

You can browse more models at [Hugging Face Models](https://huggingface.co/models?pipeline_tag=text-to-audio)

## Integrating with Generation Panel

To integrate HF generation into your existing Generation Panel, you can modify `src/hooks/useGeneration.js`:

```javascript
import * as hfAPI from '../services/huggingfaceAPI';

// In your generation function:
const handleGenerate = async () => {
  if (hfAPI.isHFConfigured() && useHuggingFace) {
    // Use Hugging Face
    const audioBlob = await hfAPI.generateMusicFromText(prompt, options);
    // Process the blob...
  } else {
    // Use your existing backend
    const result = await generationAPI.startGeneration(params);
    // ...
  }
};
```

## Troubleshooting

### "HF API token not configured"
- Make sure you've added the token to `.env`
- Make sure the variable name is exactly `REACT_APP_HF_API_TOKEN`
- Restart your dev server after editing `.env`

### "Model is loading" / 503 errors
- Hugging Face models "cold start" and take 1-2 minutes to load on first use
- Use the `waitForModel()` function to wait until ready:
  ```javascript
  await hfAPI.waitForModel('facebook/musicgen-small', 120);
  ```

### "Unauthorized" / 401 errors
- Double-check your API token is correct
- Make sure the token has proper permissions
- Try regenerating your token on Hugging Face

### Rate Limits
- Free tier has rate limits
- Consider upgrading to Hugging Face Pro for higher limits
- Implement caching and request throttling

## Security Best Practices

1. ✅ **Never commit `.env` to git** (it's in `.gitignore`)
2. ✅ **Never expose tokens in frontend code** (use environment variables)
3. ✅ **For production, use a backend proxy** to hide your token
4. ✅ **Rotate tokens regularly** for security

## Next Steps

- Explore [Hugging Face Spaces](https://huggingface.co/spaces) for pre-built demos
- Try different audio generation models
- Create custom inference endpoints for better performance
- Consider deploying your own Hugging Face Space

## Support

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Hugging Face Community Forums](https://discuss.huggingface.co/)
- [API Documentation](https://huggingface.co/docs/api-inference/)

---

**Note**: The Hugging Face Inference API has usage limits on the free tier. For production use, consider:
- Hugging Face Pro subscription
- Inference Endpoints (dedicated instances)
- Self-hosting models
