# Doseedo React - Audio Production Web App

React conversion of the Doseedo audio production application (originally doseedo2.html).

## 🎯 Project Overview

This is a modern React reimplementation of the Doseedo audio production web application, featuring:
- **Audio/MIDI file processing and generation**
- **WaveSurfer.js integration** for waveform visualization
- **Real-time automation envelope editing**
- **Multi-track audio workspace**
- **Comprehensive generation parameter controls**
- **Project save/load functionality**

## 📁 Project Structure

```
doseedo-react/
├── public/
│   └── index.html              # HTML entry point
├── src/
│   ├── components/
│   │   ├── Navbar/             # Top navigation bar
│   │   │   ├── Navbar.js
│   │   │   └── Navbar.css
│   │   ├── Sidebar/            # Collapsible sidebar menu
│   │   │   ├── Sidebar.js
│   │   │   └── Sidebar.css
│   │   ├── AudioWorkspace/     # Main audio waveform display
│   │   │   ├── AudioWorkspace.js
│   │   │   └── AudioWorkspace.css
│   │   ├── GenerationPanel/    # Parameter controls
│   │   │   ├── GenerationPanel.js
│   │   │   └── GenerationPanel.css
│   │   ├── AutomationWindow/   # Volume automation envelope
│   │   │   ├── AutomationWindow.js
│   │   │   └── AutomationWindow.css
│   │   └── ProjectManager/     # (To be implemented)
│   ├── context/
│   │   └── AppContext.js       # Global state management
│   ├── hooks/
│   │   └── useWaveSurfer.js    # WaveSurfer.js custom hook
│   ├── utils/
│   │   ├── api.js              # API communication
│   │   └── audioUtils.js       # Audio utility functions
│   ├── assets/
│   │   ├── css/
│   │   │   └── App.css         # Global styles
│   │   └── images/             # (For future assets)
│   ├── App.js                  # Main app component
│   └── index.js                # React entry point
├── package.json
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Node.js 16+ and npm
- Backend API running at `http://localhost:8070` (or update in `src/utils/api.js`)

### Installation

```bash
cd /var/www/html/doseedo-react
npm install
```

### Development

```bash
npm start
```

The app will open at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

## 🔧 Component Architecture

### 1. **AppContext** (`src/context/AppContext.js`)
Global state management using React Context API and useReducer.

**State includes:**
- Project metadata (name, auth status)
- Audio tracks array
- Generation parameters (instrument, key, seed, steps, etc.)
- Automation window state
- UI state (sidebar, uploaded files)

**Actions:**
- `SET_PROJECT_NAME`
- `UPDATE_GENERATION_PARAMS`
- `ADD_AUDIO_TRACK` / `REMOVE_AUDIO_TRACK`
- `TOGGLE_AUTOMATION_WINDOW`
- `UPDATE_AUTOMATION_POINTS`
- And more...

### 2. **Navbar** (`src/components/Navbar/`)
Top navigation bar with:
- File dropdown (New, Open, Save, Export)
- Project name display
- Undo/Redo buttons
- Session information

### 3. **Sidebar** (`src/components/Sidebar/`)
Collapsible sidebar navigation with:
- Menu toggle
- Links to Home, Dashboard, Upgrade
- Generation-related links

### 4. **AudioWorkspace** (`src/components/AudioWorkspace/`)
Main audio visualization area:
- WaveSurfer.js integration
- Play/pause controls
- Track list management
- Waveform display

### 5. **GenerationPanel** (`src/components/GenerationPanel/`)
Parameter control panel with collapsible sections:
- **Instrument Selection**: Group, subgroup, key
- **Mode Selection**: MIDI mode, monophonic mode, fatten mode
- **Generation Parameters**: Seed, steps, noise level
- **File Upload**: Audio/MIDI file input
- **Generate Button**: Triggers audio generation

### 6. **AutomationWindow** (`src/components/AutomationWindow/`)
Volume automation envelope editor:
- Canvas-based drawing
- Point addition/removal
- Grid visualization
- Show/hide functionality

## 📋 Conversion Checklist

### ✅ Phase 1: Foundation (COMPLETED)
- [x] Project structure setup
- [x] Package.json with dependencies
- [x] Global state management (Context API)
- [x] Main App component
- [x] Basic component scaffolding
- [x] CSS modularization
- [x] Utility functions (API, audio utils)
- [x] Custom hooks (useWaveSurfer)

### 🔄 Phase 2: Core Features (IN PROGRESS)
- [ ] Complete WaveSurfer.js integration
- [ ] Implement API communication
- [ ] File upload handling
- [ ] Audio generation logic
- [ ] Track management (add/remove/organize)
- [ ] Automation canvas interaction (drag points)
- [ ] Project save/load functionality
- [ ] All parameter controls wired up

### 📝 Phase 3: Advanced Features (PENDING)
- [ ] Undo/Redo functionality
- [ ] Multi-track mixing
- [ ] Export functionality (WAV, MP3, MIDI)
- [ ] Keyboard shortcuts
- [ ] Scene changes support
- [ ] Inpainting mode
- [ ] Fast mode variants
- [ ] Test-time enhancement features
- [ ] Video generation support

### 🎨 Phase 4: Polish (PENDING)
- [ ] Responsive design (mobile/tablet)
- [ ] Loading states and spinners
- [ ] Error handling and user feedback
- [ ] Animations and transitions
- [ ] Accessibility improvements
- [ ] Performance optimization
- [ ] Testing (unit, integration, e2e)

## 🔗 API Integration

The app communicates with a FastAPI backend. Update the API base URL in `src/utils/api.js`:

```javascript
const API_BASE_URL = 'http://localhost:8070';
```

### API Endpoints Used:
- `POST /generate-risers` - Generate audio
- `POST /upload` - Upload files

## 💾 State Management

The app uses React Context API for global state. To access state in any component:

```javascript
import { useApp } from '../../context/AppContext';

function MyComponent() {
  const { state, dispatch } = useApp();

  // Read state
  console.log(state.projectName);

  // Update state
  dispatch({ type: 'SET_PROJECT_NAME', payload: 'New Name' });

  return <div>...</div>;
}
```

## 🎨 Styling

- Global styles: `src/assets/css/App.css`
- Component-specific styles: Each component has its own CSS file
- Color scheme:
  - Background: `#0a0a0a`
  - Primary gradient: `#667eea → #764ba2`
  - Success: `#4CAF50`
  - Text: `#ffffff`

## 📦 Dependencies

### Main Dependencies:
- `react` & `react-dom` - Core React
- `wavesurfer.js` - Audio waveform visualization
- `axios` - HTTP client for API calls
- `interactjs` - Drag and drop interactions

### Dev Dependencies:
- `react-scripts` - Build tooling
- `@types/react` - TypeScript support (optional)

## 🐛 Known Issues / TODO

1. **WaveSurfer Integration**: Needs full implementation in AudioWorkspace
2. **API Calls**: Generate function needs proper error handling
3. **Automation Dragging**: Points can be added but not dragged yet
4. **Project Persistence**: Save/load uses localStorage (needs proper serialization)
5. **File Upload**: Backend integration not complete
6. **Responsive Design**: Mobile layout needs work

## 📚 Additional Resources

### Original HTML File
- Location: `/var/www/html/doseedo2.html` (14,038 lines)
- **DO NOT MODIFY** - Keep as reference

### Related Files
- Original CSS: `/var/www/html/style5.css`
- Original JS: Inline in doseedo2.html (lines ~5000-14000)

## 🤝 Development Workflow

1. **Keep original HTML intact** - Use as reference only
2. **Build incrementally** - One component at a time
3. **Test frequently** - Verify each feature works before moving on
4. **Commit often** - Track progress with git
5. **Document as you go** - Update README with new features

## 🔮 Future Enhancements

- **TypeScript conversion** for type safety
- **Redux or Zustand** if state becomes too complex
- **React Query** for API state management
- **Tailwind CSS** for utility-first styling
- **Storybook** for component documentation
- **Jest + React Testing Library** for testing
- **PWA support** for offline usage
- **WebSocket** for real-time collaboration

## 📞 Support

For questions about the original implementation, refer to:
- `/var/www/html/doseedo2.html` - Original HTML
- `/home/arlo/Data/genfrominterface.py` - Backend Python code

---

**Status**: Foundation Complete ✅
**Last Updated**: 2025-10-16
**Original HTML**: 14,038 lines → React: Modular components
