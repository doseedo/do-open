# Quick Start Guide - Doseedo React

Get up and running with the React version of Doseedo in minutes!

## 🚀 Installation

```bash
# Navigate to the React project
cd /var/www/html/doseedo-react

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

The app will automatically open in your browser at `http://localhost:3000`

## 🎮 Using the App

### 1. Upload a File (Optional)
- Click **"Upload Audio/MIDI"** button
- Select an audio file (.wav, .mp3) or MIDI file (.mid, .midi)
- Preview appears with file information
- Audio files have a built-in player to preview

### 2. Configure Generation Parameters

**Instrument Selection:**
- Choose instrument **Group** (Strings, Winds, Brass, Keys)
- Select **Subgroup** (Violin, Cello, Flute, Trumpet, etc.)
- Pick musical **Key** (C, C#, D, etc.)

**Mode Selection:**
- Toggle **MIDI Mode** for MIDI file processing
- Enable **Monophonic Mode** for voice separation
- Turn on **Fatten Mode** to double voices

**Generation Parameters:**
- **Seed**: Random seed for reproducibility (0-10000)
- **Steps**: Diffusion steps for quality (10-100)
- **Noise Level**: Amount of noise to inject (0-1.0)

### 3. Set Automation (Optional)
- Click **"Automation"** button to show envelope editor
- **Left-click** on canvas to add volume automation points
- **Drag** points to adjust volume over time
- **Right-click** to delete a point
- Click **"Clear"** to remove all points

### 4. Generate Audio
- Click the **"Generate"** button
- Watch the progress bar as audio generates
- Generated tracks appear in the **Audio Workspace**

### 5. Play Generated Audio
- Click a track in the track list to load it
- Use **Play/Pause** button to control playback
- **Rewind** (-5s) or **Forward** (+5s) to skip
- Adjust **Volume** slider (0-100%)
- Use **Zoom** to magnify waveform (1x-100x)

## 📁 Project Structure

```
doseedo-react/
├── src/
│   ├── components/          # React components
│   │   ├── Navbar/         # Top navigation
│   │   ├── Sidebar/        # Side menu
│   │   ├── AudioWorkspace/ # Waveform player
│   │   ├── GenerationPanel/# Controls
│   │   └── AutomationWindow/# Automation
│   ├── context/            # State management
│   ├── hooks/              # Custom hooks
│   ├── utils/              # Helper functions
│   └── assets/             # CSS and images
├── public/                 # Static files
├── package.json            # Dependencies
└── README.md              # Documentation
```

## 🔧 Backend Setup

The React app communicates with a FastAPI backend. Make sure it's running:

```bash
# In a separate terminal
cd /home/arlo/Data
python3 run_fastapi_server.py --port 8070
```

**Backend URL:** `http://localhost:8070`

To change the API URL, edit `/src/utils/api.js`:
```javascript
const API_BASE_URL = 'http://localhost:8070';
```

## 🐛 Troubleshooting

### "Module not found" errors
```bash
npm install
```

### Port already in use
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use a different port
PORT=3001 npm start
```

### API connection errors
1. Make sure backend is running at port 8070
2. Check CORS settings in backend
3. Open browser DevTools → Network tab to see requests

### Waveform not showing
1. Make sure you selected a track from the list
2. Check browser console for errors
3. Verify the audio file URL is valid

## ⌨️ Keyboard Shortcuts (Coming Soon)

- `Space` - Play/Pause
- `←` - Rewind 5s
- `→` - Forward 5s
- `Ctrl+S` - Save project
- `Ctrl+Z` - Undo
- `Ctrl+Y` - Redo

## 📚 Learn More

- **README.md** - Complete project documentation
- **CONVERSION_GUIDE.md** - How we converted from HTML to React
- **PHASE_2_COMPLETE.md** - What was accomplished in Phase 2

## 🆘 Common Issues

### File Upload Not Working
- Make sure file is audio (.wav, .mp3, .ogg) or MIDI (.mid, .midi)
- Check file size (large files may take time to upload)

### Generation Fails
- Check backend API is running
- Open browser console for error messages
- Verify backend logs for errors

### Audio Doesn't Play
- Click a track to load it first
- Make sure browser supports Web Audio API
- Check volume is not muted

### Automation Window Empty
- Click "Automation" button to show window
- Canvas will appear after a brief delay
- Try resizing window if canvas doesn't render

## 🎯 Tips

1. **Start Simple:** Try generating without uploading a file first
2. **Use Defaults:** Default parameters work well for testing
3. **Save Often:** Use the File → Save menu (coming soon)
4. **Experiment:** Try different instrument combinations
5. **Check Console:** Browser DevTools show helpful error messages

## 🚀 What's Next?

After getting comfortable with the basics:
1. Try different generation parameters
2. Upload your own audio/MIDI files
3. Use automation to create volume envelopes
4. Enable advanced features (Monophonic, Fatten Mode)
5. Explore the code to customize

---

**Enjoy creating music with Doseedo React! 🎵**

For questions or issues, check the main README.md or refer to the original doseedo2.html.
