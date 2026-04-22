import React from 'react';
import ReactDOM from 'react-dom/client';
import { initSentry } from './lib/sentry';
import './styles/colors.css'; // Master color palette (legacy dark theme)
import './styles/theme-workbench.css'; // Master workbench theme — body.workbench-theme on /studio, /dashboard, /projects, ...
import './styles/theme-hifi-purple.css'; // Dev variant override (body.theme-hifi-purple on /studio)
import './styles/glass-theme-background.css'; // Glass theme backgrounds
import './styles/liquid-glass.css'; // Glass theme button & panel effects
// original-style5.css (3.5k lines, pre-workbench DAW chrome) is now
// dynamic-imported by App.js ONLY on the /studio-legacy route — the
// workbench routes get to load without its button/range/body rules
// bleeding in. Kept out of this eager bundle intentionally.
import './assets/css/App.css';
import App from './App';

// Sentry init BEFORE React mounts so early-boot errors are captured.
// No-ops if REACT_APP_SENTRY_DSN isn't set (local dev without DSN).
initSentry();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
