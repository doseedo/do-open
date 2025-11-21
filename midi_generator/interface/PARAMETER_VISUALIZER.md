# MIDI Parameter Visualizer

A real-time web interface for visualizing and adjusting hierarchical MIDI parameters.

## Features

- **Drag & Drop Upload**: Easily upload MIDI files for analysis
- **Hierarchical Visualization**: View parameters organized in 3 levels
  - Level 1: Global Context (tempo, key, genre, etc.)
  - Level 2: Universal Dimensions (harmony, melody, rhythm, dynamics, texture)
  - Level 3: Genre-Specific Details
- **Real-Time Parameter Adjustment**: Interactive sliders and controls
- **Visual Feedback**: Color-coded categories and modified parameters
- **Export Functionality**: Save modified parameters as JSON
- **Search & Filter**: Quickly find specific parameters
- **Responsive Design**: Modern, dark-themed interface

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors
```

### 2. Start the Backend Server

```bash
cd /Users/hydroadmin/Downloads/Do/midi_generator/interface
python3 server.py
```

The server will start on `http://localhost:5000`

### 3. Open the Web Interface

```bash
# Option 1: Serve with Python HTTP server
python3 -m http.server 8000

# Then visit: http://localhost:8000
```

Or simply open `index.html` directly in your browser.

### 4. Upload a MIDI File

- Drag and drop a `.mid` file onto the upload area
- Or click to browse and select a file

### 5. Adjust Parameters

- Use sliders to adjust numeric parameters
- Use dropdowns for categorical values
- Modified parameters are highlighted
- Click **Export** to save your changes

## Files

- `index.html` - Main interface
- `styles.css` - Styling (dark theme)
- `app.js` - Frontend logic
- `server.py` - Flask API backend

## API Endpoints

### POST /analyze
Upload and analyze a MIDI file

```bash
curl -X POST -F "file=@song.mid" http://localhost:5000/analyze
```

Returns hierarchical parameters in JSON format.

## Troubleshooting

**Server won't start**: Make sure Flask is installed (`pip install flask flask-cors`)

**CORS errors**: Ensure the server is running before opening the interface

**File won't upload**: Check that the file is a valid `.mid` or `.midi` file

## Architecture

```
interface/
├── index.html          # Main HTML structure
├── styles.css          # CSS styling
├── app.js             # JavaScript logic
├── server.py          # Flask API server
└── PARAMETER_VISUALIZER.md  # This file
```

## License

Part of the midi_generator project.
