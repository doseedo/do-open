# Doseedo HTML → React Conversion Guide

## 📖 Overview

This document outlines the conversion strategy from the monolithic `doseedo2.html` (14,038 lines) to a modern React application.

## 🎯 Conversion Philosophy

**Goals:**
1. **Maintain functionality** - All features from original HTML should work
2. **Improve maintainability** - Modular components, clear separation of concerns
3. **Enhance performance** - Virtual DOM, efficient re-renders
4. **Enable scalability** - Easy to add new features
5. **Better developer experience** - Modern tooling, hot reload, debugging

**Non-Goals:**
- Don't rewrite everything from scratch
- Don't change the visual design (yet)
- Don't modify backend API contracts

## 🗺️ Component Mapping

### Original HTML Structure → React Components

| Original Section (doseedo2.html) | React Component | Status |
|----------------------------------|-----------------|--------|
| Lines 60-233: Navbar & Auth | `Navbar.js` | ✅ Created |
| Lines 236-350: Sidebar | `Sidebar.js` | ✅ Created |
| Lines 400-1000: Generation Controls | `GenerationPanel.js` | ✅ Created |
| Lines 2000-2500: Waveform Display | `AudioWorkspace.js` | ✅ Created |
| Lines 5400-5700: Automation Window | `AutomationWindow.js` | ✅ Created |
| Lines 5800-14000: JavaScript Logic | Context, Hooks, Utils | ✅ Created |
| Inline CSS | Modular CSS files | ✅ Created |

## 📦 State Management Migration

### Original: Global Variables & jQuery
```javascript
// doseedo2.html (lines ~5800+)
let projectName = "Untitled Session";
let audioTracks = [];
let generationParams = { seed: 0, steps: 20, ... };
```

### React: Context API
```javascript
// src/context/AppContext.js
const initialState = {
  projectName: "Untitled Session",
  audioTracks: [],
  generationParams: { seed: 0, steps: 20, ... }
};

function appReducer(state, action) { ... }
```

**Migration Steps:**
1. ✅ Identify all global variables in original HTML
2. ✅ Create initial state object in AppContext
3. ✅ Define reducer actions for state updates
4. ⏳ Replace jQuery selectors with React state
5. ⏳ Replace event handlers with dispatch calls

## 🔄 Event Handling Migration

### Original: jQuery Event Handlers
```javascript
// doseedo2.html
$('#generate-btn').click(function() {
  const seed = $('#seed').val();
  generateAudio(seed);
});

$('#automation-toggle').click(function() {
  $('#automation-window').toggle();
});
```

### React: Component Event Handlers
```javascript
// GenerationPanel.js
const handleGenerate = () => {
  const seed = state.generationParams.seed;
  generateAudio(seed);
};

// AutomationWindow.js
const toggleWindow = () => {
  dispatch({ type: 'TOGGLE_AUTOMATION_WINDOW' });
};
```

## 🎨 CSS Migration Strategy

### Original: Monolithic CSS
- `style5.css` - ~3000 lines
- Inline `<style>` tags in HTML

### React: Modular CSS
- `App.css` - Global styles
- `Navbar.css` - Navbar-specific
- `Sidebar.css` - Sidebar-specific
- etc.

**Benefits:**
- Component-scoped styles
- Easier to find and modify
- Better code organization
- Can use CSS modules if needed

## 🔌 API Integration

### Original: jQuery AJAX
```javascript
$.ajax({
  url: 'http://localhost:8070/generate-risers',
  type: 'POST',
  data: formData,
  success: function(data) { ... },
  error: function(err) { ... }
});
```

### React: Axios + Async/Await
```javascript
// src/utils/api.js
export async function generateAudio(formData) {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/generate-risers`,
      formData
    );
    return response.data;
  } catch (error) {
    console.error('Generation error:', error);
    throw error;
  }
}
```

## 📝 Feature-by-Feature Conversion Plan

### Phase 1: Core UI (✅ COMPLETED)
- [x] Project structure
- [x] Navbar component
- [x] Sidebar component
- [x] Generation panel layout
- [x] Audio workspace layout
- [x] Automation window layout

### Phase 2: Interactivity (🔄 IN PROGRESS)
- [ ] File upload & preview
- [ ] Parameter controls (sliders, dropdowns)
- [ ] Automation canvas drawing
- [ ] Project save/load
- [ ] Track management

### Phase 3: Audio Processing (📝 PENDING)
- [ ] WaveSurfer.js integration
- [ ] Play/pause controls
- [ ] Waveform visualization
- [ ] Multi-track support
- [ ] Audio export

### Phase 4: Generation Features (📝 PENDING)
- [ ] API integration
- [ ] Generation with parameters
- [ ] MIDI mode support
- [ ] Fast mode variants
- [ ] Scene changes
- [ ] Inpainting
- [ ] Best-of-N sampling
- [ ] Self-consistency ensembling

### Phase 5: Polish (📝 PENDING)
- [ ] Loading states
- [ ] Error handling
- [ ] Responsive design
- [ ] Keyboard shortcuts
- [ ] Tooltips/help
- [ ] Performance optimization

## 🔍 Code Patterns

### Pattern 1: Form Inputs
**Original:**
```html
<input type="range" id="seed" min="0" max="10000" value="0">
<span id="seed-value">0</span>
```
```javascript
$('#seed').on('input', function() {
  $('#seed-value').text($(this).val());
});
```

**React:**
```javascript
<input
  type="range"
  min="0"
  max="10000"
  value={state.generationParams.seed}
  onChange={(e) => updateParam('seed', parseInt(e.target.value))}
/>
<span>{state.generationParams.seed}</span>
```

### Pattern 2: Conditional Rendering
**Original:**
```javascript
if (midiMode) {
  $('#midi-controls').show();
} else {
  $('#midi-controls').hide();
}
```

**React:**
```javascript
{state.generationParams.midiMode && (
  <div className="midi-controls">
    {/* MIDI controls */}
  </div>
)}
```

### Pattern 3: Dynamic Lists
**Original:**
```javascript
audioTracks.forEach(track => {
  $('#track-list').append(`
    <div class="track">
      <span>${track.name}</span>
      <button onclick="removeTrack(${track.id})">Delete</button>
    </div>
  `);
});
```

**React:**
```javascript
{state.audioTracks.map(track => (
  <div key={track.id} className="track">
    <span>{track.name}</span>
    <button onClick={() => removeTrack(track.id)}>Delete</button>
  </div>
))}
```

## 🚧 Common Challenges & Solutions

### Challenge 1: WaveSurfer.js Integration
**Problem:** WaveSurfer modifies DOM directly, conflicts with React

**Solution:**
- Use `useRef` to create container
- Initialize in `useEffect`
- Destroy on unmount
- Create custom hook `useWaveSurfer`

### Challenge 2: Canvas Automation
**Problem:** Canvas drawing is imperative, React is declarative

**Solution:**
- Use `useRef` for canvas element
- Use `useEffect` to redraw when points change
- Store points in state, render via canvas API

### Challenge 3: Large Form State
**Problem:** Many generation parameters to manage

**Solution:**
- Group parameters in nested state object
- Create helper function `updateParam(key, value)`
- Use reducer for complex state updates

### Challenge 4: File Upload
**Problem:** File inputs are tricky in React

**Solution:**
- Use `useRef` for file input
- Handle via onChange event
- Store File object in state
- Create FormData for API calls

## 📊 Progress Tracking

### Lines of Code Converted
- Original HTML: ~14,000 lines
- React Components: ~1,500 lines (across all files)
- **Reduction: ~89%** 🎉

### Features Implemented
- Basic UI: 100%
- State Management: 80%
- Event Handling: 40%
- API Integration: 20%
- Audio Processing: 10%

### Estimated Completion
- **Phase 1 (Foundation):** ✅ 100% Complete
- **Phase 2 (Interactivity):** 🔄 20% Complete
- **Phase 3 (Audio):** 📝 0% Complete
- **Phase 4 (Generation):** 📝 0% Complete
- **Phase 5 (Polish):** 📝 0% Complete

**Overall Progress: ~25%**

## 🎓 Learning Resources

### React Concepts Needed
- ✅ Components & Props
- ✅ State & Hooks (useState, useEffect, useRef, useReducer)
- ✅ Context API
- ⏳ Custom Hooks
- ⏳ Performance Optimization (memo, useCallback, useMemo)

### Libraries to Learn
- ⏳ WaveSurfer.js - Audio visualization
- ⏳ Axios - HTTP requests
- ⏳ Interact.js - Drag & drop

## 🔧 Development Tips

1. **Use the browser DevTools**
   - React DevTools extension
   - Component inspector
   - State viewer

2. **Hot reload is your friend**
   - Edit and see changes instantly
   - No need to refresh page

3. **Console.log strategically**
   - Log state changes
   - Log API responses
   - Log component renders

4. **Test incrementally**
   - Don't build everything at once
   - Verify each feature works before moving on

5. **Refer to original HTML**
   - Keep `/var/www/html/doseedo2.html` open
   - Copy logic, not syntax
   - Understand the intent

## 📞 Next Steps

### Immediate (Next Session)
1. Complete WaveSurfer integration in AudioWorkspace
2. Wire up all parameter controls in GenerationPanel
3. Implement file upload handling
4. Test automation canvas point dragging

### Short-term (This Week)
1. Implement API communication
2. Test audio generation end-to-end
3. Add loading states and error handling
4. Implement project save/load

### Long-term (This Month)
1. Complete all features from original HTML
2. Add responsive design
3. Performance optimization
4. Testing and bug fixes
5. Deployment

---

**Remember:** The original `doseedo2.html` is preserved at `/var/www/html/doseedo2.html`

**Status**: Foundation Complete, Moving to Interactivity Phase
**Last Updated**: 2025-10-16
