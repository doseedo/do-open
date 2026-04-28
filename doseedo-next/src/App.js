import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, useNavigate, useLocation } from 'react-router-dom';
import { AppProvider, useApp } from './context/AppContext';
import { useKeyboardControls } from './hooks/useKeyboardControls';
import { useSessionSync } from './hooks/useSessionSync';
import * as authService from './services/authService';
import * as sessionService from './services/sessionService';

// === STUDIO COMPONENTS (original) ===
import Navbar from './components/Navbar/Navbar';
import LeftSidebar from './components/Sidebar/LeftSidebar/LeftSidebar';
import TrackInfoSidebar from './components/Sidebar/RightSidebar/TrackInfoSidebar';
import ChatWindow from './components/ChatWindow/ChatWindow';
import MIDIChart from './components/MIDIChart/MIDIChart';
import AudioWaveform from './components/AudioWaveform/AudioWaveform';
import DAWOptimized from './components/DAW/DAWOptimized';
import ResizeBar from './components/ResizeBar/ResizeBar';
import VerticalResizeBar from './components/ResizeBar/VerticalResizeBar';
import Dashboard from './components/Dashboard/Dashboard';
import Projects from './components/Projects/Projects';
import Home from './components/Home/Home';
import Search from './components/Search/Search';
import UserInfo from './components/UserInfo/UserInfo';
import Settings from './components/Settings/Settings';
import Tools from './components/Tools/Tools';
import WhatsNew from './components/WhatsNew/WhatsNew';
import Research from './components/Research/Research';
import Downloads from './components/Downloads/Downloads';
import Plugins from './components/Plugins/Plugins';
import PublicProfile from './components/PublicProfile/PublicProfile';
import About from './components/Legal/About';
import Privacy from './components/Legal/Privacy';
import Terms from './components/Legal/Terms';
import Help from './components/Legal/Help';
import Feedback from './components/Legal/Feedback';
import Plans from './components/Legal/Plans';
import Verify from './components/Legal/Verify';
import Docs from './components/Legal/Docs';
import Models from './components/Models/Models';
import CreationView from './components/CreationView/CreationView';
import StudioDev from './components/StudioDev/StudioDev';

const PROTECTED_PASSWORD = process.env.NEXT_PUBLIC_PROTECTED_PASSWORD || '***REDACTED***';

function PasswordGate({ children, routeName }) {
  const [unlocked, setUnlocked] = useState(() => {
    return sessionStorage.getItem(`pw_${routeName}`) === '1';
  });
  const [input, setInput] = useState('');
  const [error, setError] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input === PROTECTED_PASSWORD) {
      sessionStorage.setItem(`pw_${routeName}`, '1');
      setUnlocked(true);
    } else {
      setError(true);
      setTimeout(() => setError(false), 1500);
    }
  };

  if (unlocked) return children;

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      minHeight: '100vh',
      background: 'var(--wb-bg, #e8e6e1)',
      color: 'var(--wb-ink, #15181c)',
      fontFamily: 'var(--wb-font-sans, Inter, -apple-system, sans-serif)',
    }}>
      <form onSubmit={handleSubmit} style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16,
        background: 'var(--wb-surface, #f2f0ea)',
        border: '1px solid var(--wb-rule-strong, #a5a29a)',
        padding: '40px 48px',
        boxShadow: '0 8px 24px -6px rgba(20, 22, 26, 0.15)',
        minWidth: 320,
      }}>
        <i className="fa-solid fa-lock" style={{
          fontSize: 24, color: 'var(--wb-accent, #1d4c7a)', marginBottom: 4,
        }} />
        <div style={{
          fontFamily: 'var(--wb-font-mono, "JetBrains Mono", monospace)',
          fontSize: 10, letterSpacing: '0.8px', textTransform: 'uppercase',
          color: 'var(--wb-ink-mute, #7c7e85)',
        }}>
          Early Access
        </div>
        <h2 style={{
          margin: 0, fontSize: 22, fontWeight: 600,
          letterSpacing: '-0.3px', color: 'var(--wb-ink, #15181c)',
        }}>
          Password Required
        </h2>
        <p style={{
          margin: 0, fontSize: 13, lineHeight: 1.5, textAlign: 'center',
          color: 'var(--wb-ink-soft, #3a3d44)', maxWidth: 260,
        }}>
          Enter the password to access {routeName}.
        </p>
        <input
          type="password"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Password"
          autoFocus
          style={{
            padding: '10px 14px', fontSize: 13,
            border: '1px solid var(--wb-rule, #c8c5bd)',
            background: 'var(--wb-bg, #e8e6e1)',
            color: 'var(--wb-ink, #15181c)',
            fontFamily: 'var(--wb-font-mono, "JetBrains Mono", monospace)',
            outline: 'none', width: 240, borderRadius: 0,
          }}
        />
        <button type="submit" style={{
          padding: '10px 32px',
          fontFamily: 'var(--wb-font-mono, "JetBrains Mono", monospace)',
          fontSize: 10, fontWeight: 500, letterSpacing: '0.8px',
          textTransform: 'uppercase', border: '1px solid var(--wb-ink, #15181c)',
          background: 'var(--wb-ink, #15181c)',
          color: 'var(--wb-bg, #e8e6e1)',
          cursor: 'pointer', borderRadius: 0,
        }}>
          Unlock
        </button>
        {error && (
          <p style={{
            margin: 0, fontSize: 11,
            fontFamily: 'var(--wb-font-mono, "JetBrains Mono", monospace)',
            letterSpacing: '0.5px', textTransform: 'uppercase',
            color: 'var(--wb-accent-warm, #c94f2c)',
          }}>
            Incorrect password
          </p>
        )}
      </form>
    </div>
  );
}

/**
 * AppContent - Inner component with access to context
 */
function AppContent() {
  const { state, dispatch } = useApp();
  const navigate = useNavigate();
  const location = useLocation();
  // Track SPA page views in Google Analytics
  useEffect(() => {
    if (window.gtag) {
      window.gtag('event', 'page_view', { page_path: location.pathname + location.search });
    }
  }, [location]);

  // Workbench theme: the master cream/JetBrains-Mono theme defined in
  // src/styles/theme-workbench.css is opt-in per route via body class.
  // Every page listed below reads from var(--wb-*) tokens and flips to
  // the workbench look as soon as this effect tags <body>. Legacy dark
  // routes (landing marketing pages, etc.) are untouched.
  useEffect(() => {
    const p = location.pathname;
    const isWorkbench =
      p === '/studio' ||
      p === '/dashboard' ||
      p === '/projects' ||
      p === '/profile' ||
      p.startsWith('/profile/') ||
      p === '/settings' ||
      p === '/search' ||
      p === '/tools' ||
      p === '/plans' ||
      p === '/verify' ||
      p === '/whats-new' ||
      p === '/help' ||
      p === '/docs' ||
      p.startsWith('/docs/') ||
      p === '/feedback' ||
      p === '/about' ||
      p === '/privacy' ||
      p === '/terms' ||
      p.startsWith('/research') ||
      p === '/downloads' ||
      p.startsWith('/plugins') ||
      p === '/models' ||
      p.startsWith('/creation/');
    // Keep the legacy hifi-purple class in lockstep — /studio (and the
    // /plugins/create gate that piggybacks on it) still uses it for the
    // password-gate skin override.
    const isHifiPurple = p === '/studio' || p.startsWith('/plugins/create');
    if (typeof document !== 'undefined') {
      document.body.classList.toggle('workbench-theme', isWorkbench);
      document.body.classList.toggle('theme-hifi-purple', isHifiPurple);
      return () => {
        document.body.classList.remove('workbench-theme');
        document.body.classList.remove('theme-hifi-purple');
      };
    }
  }, [location.pathname]);

  const [contentMode, setContentMode] = useState('video'); // 'video', 'midi', 'audio', 'image', or 'fx'
  const [showMidiBrowser, setShowMidiBrowser] = useState(false); // Toggle between generation panel and MIDI browser
  const [showChatWindow, setShowChatWindow] = useState(false); // Toggle for chat window
  const [showBookmarks, setShowBookmarks] = useState(false);   // Toggle for saved/bookmarks panel
  // The LeftSidebar tool rail (collapsed-state quick-toggles) lives at
  // the App root, but the panel it should swap into is the /studio
  // route's own sd-sidebar (sidebarPanel state inside StudioDev). We
  // bridge that with a window CustomEvent — App handlers fire it; the
  // StudioDev component listens and calls setSidebarPanel(detail).
  // Avoids a deep prop-drill or shared context for what's essentially a
  // four-button command channel.
  const fireStudioSidebar = (panel) => {
    if (typeof window === 'undefined') return;
    try {
      window.dispatchEvent(new CustomEvent('doseedo:studio-sidebar', { detail: panel }));
    } catch (_) {}
  };
  // panelWidth stores the BUS-LABEL COLUMN WIDTH (not the bar's absolute X).
  // The bar's visible X position is derived at render time as leftOffset +
  // panelWidth. This way, when the left sidebar collapses/expands, the bar
  // slides with it but the column width stays locked — no "snap" or sudden
  // DAW-column resize when the user toggles the sidebar.
  const [panelWidth, setPanelWidth] = useState(340); // Initial bus-label column width (px)
  const [panelHeight, setPanelHeight] = useState(420); // Initial height (user preference)
  const [dawTracksHeight, setDawTracksHeight] = useState(600); // Height for DAW tracks scrollable area
  const [pluginDawHeight, setPluginDawHeight] = useState(200); // Height for DAW in plugin mode
  const [minWidth, setMinWidth] = useState(200);
  const [maxWidth, setMaxWidth] = useState(Math.floor(window.innerWidth * 0.3));
  const [leftOffset, setLeftOffset] = useState(0); // Track left offset for DAW alignment
  const contentRef = React.useRef(null);

  // Prevent pinch-to-zoom on touch devices
  useEffect(() => {
    const preventDefault = (e) => {
      if (e.touches.length > 1) {
        e.preventDefault();
      }
    };

    const preventGestureZoom = (e) => {
      e.preventDefault();
    };

    // Prevent pinch zoom
    document.addEventListener('touchmove', preventDefault, { passive: false });
    document.addEventListener('gesturestart', preventGestureZoom, { passive: false });
    document.addEventListener('gesturechange', preventGestureZoom, { passive: false });
    document.addEventListener('gestureend', preventGestureZoom, { passive: false });

    return () => {
      document.removeEventListener('touchmove', preventDefault);
      document.removeEventListener('gesturestart', preventGestureZoom);
      document.removeEventListener('gesturechange', preventGestureZoom);
      document.removeEventListener('gestureend', preventGestureZoom);
    };
  }, []);

  // Ensure plugin mode is disabled on initial load
  useEffect(() => {
    if (state.pluginMode) {
      dispatch({ type: 'TOGGLE_PLUGIN_MODE' });
    }
  }, []); // Run only once on mount

  // Adjust panel height when plugin mode toggles
  useEffect(() => {
    if (state.pluginMode) {
      setPanelHeight(600); // Taller in plugin mode
    } else {
      setPanelHeight(420); // Normal height when not in plugin mode
    }
  }, [state.pluginMode]);

  // Determine current view from URL. /dashboard is the workbench "home"
  // (slideshow + Jump Back In + community feeds), /projects is the full
  // sessions table. Distinct routes → distinct components → distinct
  // sidebar highlights (Home vs Projects).
  const currentView = location.pathname === '/dashboard' ? 'home' :
                      location.pathname === '/projects' ? 'projects' :
                      location.pathname === '/search' ? 'search' :
                      location.pathname.startsWith('/profile/') && location.pathname.split('/').length >= 3 ? 'publicProfile' :
                      location.pathname === '/profile' ? 'userinfo' :
                      location.pathname === '/settings' ? 'settings' :
                      location.pathname === '/tools' ? 'tools' :
                      location.pathname === '/whats-new' ? 'whatsnew' :
                      location.pathname === '/about' ? 'about' :
                      location.pathname === '/privacy' ? 'privacy' :
                      location.pathname === '/terms' ? 'terms' :
                      location.pathname === '/help' ? 'help' :
                      location.pathname === '/docs' || location.pathname.startsWith('/docs/') ? 'docs' :
                      location.pathname === '/feedback' ? 'feedback' :
                      location.pathname === '/plans' ? 'plans' :
                      location.pathname === '/verify' ? 'verify' :
                      location.pathname.startsWith('/research') ? 'research' :
                      location.pathname === '/downloads' ? 'downloads' :
                      location.pathname.startsWith('/creation/') ? 'creation' :
                      location.pathname.startsWith('/plugins') ? 'plugins' :
                      location.pathname === '/models' ? 'models' :
                      location.pathname === '/studio' ? 'daw' :
                      location.pathname === '/' ? 'home' : 'home';

  // Redirect root based on auth status - retry multiple times to handle cookie timing.
  // Note: the primary gate lives in public/index.html as an inline <script> that
  // runs before React mounts, so unauthenticated visitors are usually gone before
  // this effect ever fires. This effect is the backstop for the edge case where
  // the auth cookie is written *after* first paint (e.g. race with /verify-
  // google-id-token), so we still need the retry loop here.
  useEffect(() => {
    if (location.pathname === '/' || location.pathname === '/home') {
      let attempts = 0;
      const maxAttempts = 5;
      const checkInterval = 150; // ms between checks

      const checkAuth = () => {
        attempts++;
        const isAuth = authService.isAuthenticated();
        console.log(`Auth check attempt ${attempts}: isAuthenticated=${isAuth}, cookies=${document.cookie}`);

        if (isAuth) {
          // Authenticated users go to dashboard
          console.log('Redirecting to /dashboard');
          navigate('/dashboard', { replace: true });
        } else if (attempts >= maxAttempts) {
          // After 750ms of retries, assume not authenticated — leave the SPA.
          // /home is routed to the doseedo-frontend Cloud Run service which
          // proxies the Framer marketing site, so the browser stays on
          // doseedo.com and sees Framer content with no visible cross-origin
          // redirect.
          console.log('Max attempts reached, redirecting to /home');
          window.location.replace('/home');
        } else {
          // Try again
          setTimeout(checkAuth, checkInterval);
        }
      };

      // Start checking after initial 100ms delay
      const timeoutId = setTimeout(checkAuth, 100);
      return () => clearTimeout(timeoutId);
    }
  }, [location.pathname, navigate]);

  // Check for active project on mount and auto-load if exists (only on first load)
  const [hasCheckedAutoLoad, setHasCheckedAutoLoad] = useState(false);

  useEffect(() => {
    if (!hasCheckedAutoLoad && location.pathname === '/' && authService.isAuthenticated()) {
      const activeProject = sessionService.getActiveProject();
      if (activeProject) {
        const sessionData = sessionService.loadSession(activeProject);
        if (sessionData) {
          dispatch({ type: 'LOAD_SESSION', payload: sessionData.state });
          navigate('/studio', { replace: true });
          console.log(`✅ Auto-loaded session: ${activeProject}`);

          // Force disable plugin mode after loading session
          if (sessionData.state.pluginMode) {
            dispatch({ type: 'TOGGLE_PLUGIN_MODE' });
          }
        }
      }
      setHasCheckedAutoLoad(true);
    }
  }, [dispatch, navigate, location.pathname, hasCheckedAutoLoad]);

  // Enable keyboard controls (spacebar for play/pause)
  useKeyboardControls(dispatch, state.isPlaying);

  // Hydrate the DAW from /studio?session=<uuid> and poll for drift.
  useSessionSync(dispatch);

  // Track the sidebar's left edge with a ResizeObserver — when the left
  // sidebar collapses/expands or the window resizes, leftOffset updates but
  // we DO NOT force-shrink panelWidth (that was the source of the "snap").
  // The user's chosen bar position is sticky; only the ResizeBar's own drag
  // clamps against minWidth/maxWidth.
  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;

    const recompute = () => {
      const leftEdge = el.getBoundingClientRect().left;
      setLeftOffset(leftEdge);
      // ResizeBar clamp bounds are in ABSOLUTE X (that's ResizeBar's API).
      // Content width bounds are 200 (min) to 30vw (max); progressive-hide
      // in the DAW controls (see showTempoLabels/showMetronome/etc.)
      // keeps the header readable at narrow widths, so we can let users
      // shrink the sidebar further than the old 310px floor.
      const MIN_CONTENT = 275;
      setMinWidth(leftEdge + MIN_CONTENT);
      setMaxWidth(Math.max(leftEdge + MIN_CONTENT, leftEdge + Math.floor(window.innerWidth * 0.3)));
    };
    recompute();

    const ro = new ResizeObserver(recompute);
    ro.observe(el);
    // Also watch window resize for viewport-dependent maxWidth.
    window.addEventListener('resize', recompute);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', recompute);
    };
  }, []);

  // panelWidth IS the bus-label column width. Single source of truth.
  const busLabelWidth = panelWidth;

  // Publish as CSS variable + custom event. Prod components receive
  // busLabelWidth as a prop; the custom event is kept for any listeners
  // that still read off it.
  useEffect(() => {
    document.documentElement.style.setProperty('--bus-label-width', `${busLabelWidth}px`);
    window.dispatchEvent(new CustomEvent('busLabelWidthChanged', { detail: busLabelWidth }));
  }, [busLabelWidth]);

  // ResizeBar reports the bar's absolute X. Translate back to column width
  // by subtracting the current leftOffset.
  const handleResize = (newBarX) => {
    setPanelWidth(Math.max(0, newBarX - leftOffset));
  };

  const handleVerticalResize = (newTop) => {
    setPanelHeight(newTop);
  };

  const handlePluginDawResize = (newTop) => {
    // Calculate DAW height from the resize position
    // newTop is the Y position, we need to calculate height from the bottom
    const containerHeight = panelHeight;
    const dawHeight = containerHeight - newTop;
    setPluginDawHeight(Math.max(100, Math.min(dawHeight, containerHeight - 100)));
  };

  // Update CSS custom property for panel height when it changes
  useEffect(() => {
    document.documentElement.style.setProperty('--panel-height', `${panelHeight}px`);
    // Dispatch custom event to notify components
    window.dispatchEvent(new CustomEvent('panelHeightChanged', { detail: panelHeight }));
  }, [panelHeight]);

  // Handle create new project from dashboard
  const handleCreateNew = (projectName) => {
    // Dashboard already handles RESET_SESSION and setting active project
    // Just navigate to studio
    navigate('/studio');
  };

  // Handle load project from dashboard
  const handleLoadProject = (projectName) => {
    // The Dashboard component already handles loading via dispatch
    // Just navigate to studio - the state is already loaded
    navigate('/studio');
  };

  // Handle navigation to home
  const handleGoToHome = () => {
    navigate('/dashboard');
  };

  // Handle back to dashboard
  const handleBackToDashboard = () => {
    navigate('/projects');
  };

  // Handle navigation to search
  const handleGoToSearch = () => {
    navigate('/search');
  };

  // Handle navigation to user info
  const handleGoToUserInfo = () => {
    navigate('/profile');
  };

  // Handle navigation to tools
  const handleGoToTools = () => {
    navigate('/tools');
  };

  // Handle navigation to what's new
  const handleGoToWhatsNew = () => {
    navigate('/whats-new');
  };

  // Handle navigation to research
  const handleGoToResearch = () => {
    navigate('/research');
  };

  // Handle navigation to downloads
  const handleGoToDownloads = () => {
    navigate('/downloads');
  };

  // Handle navigation to plugins
  const handleGoToPlugins = () => {
    navigate('/plugins');
  };

  // Handle navigation to Models catalog
  const handleGoToModels = () => {
    navigate('/models');
  };

  // Toggle MIDI browser
  const handleToggleSearch = () => {
    setShowMidiBrowser(prev => !prev);
    setShowChatWindow(false); // Hide chat when showing MIDI browser
  };

  // Show the instruments panel (wand icon). The wand used to swap the
  // sidebar to the standalone Generate window, but Generate is now
  // embedded at the bottom of the Instruments panel — so the wand
  // returns to the default instruments view, where the user picks a
  // preset and the embedded Generate form runs from the same window.
  const handleShowGenerationPanel = () => {
    setShowMidiBrowser(false);
    setShowChatWindow(false);
    setShowBookmarks(false);
    fireStudioSidebar('instruments');
  };

  // Show MIDI browser (search icon)
  const handleShowMidiBrowser = () => {
    setShowMidiBrowser(true); // true = MIDI browser
    setShowChatWindow(false);
    setShowBookmarks(false);
    fireStudioSidebar('browse');
  };

  // Toggle chat window
  const handleToggleChat = () => {
    setShowChatWindow(prev => {
      const next = !prev;
      // Fire the matching panel each transition so the studio sidebar
      // closes back to instruments when chat is toggled off.
      fireStudioSidebar(next ? 'chat' : 'instruments');
      return next;
    });
    setShowMidiBrowser(false); // Hide MIDI browser when showing chat
    setShowBookmarks(false);
  };

  // Show bookmarks / saved panel
  const handleShowBookmarks = () => {
    setShowBookmarks(true);
    setShowMidiBrowser(false);
    setShowChatWindow(false);
    fireStudioSidebar('saved');
  };

  // ---- Single LeftSidebar instance for the entire SPA ----
  // The sidebar renders ONCE at the App root so navigating between
  // workbench routes (/dashboard, /studio, /projects, …) doesn't
  // unmount/remount it — the same DOM node persists, so width
  // transitions are smooth and no auto-expand effect re-fires per
  // route change. Each route below returns just its content.
  const viewFlags = {
    isHomeView:      currentView === 'home',
    isDashboardView: currentView === 'dashboard' || currentView === 'projects',
    isSearchView:    currentView === 'search',
    isUserInfoView:  currentView === 'userinfo',
    isToolsView:     currentView === 'tools',
    isWhatsNewView:  currentView === 'whatsnew',
    isResearchView:  currentView === 'research',
    isDownloadsView: currentView === 'downloads',
    isPluginsView:   currentView === 'plugins',
    isModelsView:    currentView === 'models',
  };

  // Route content selector — returns just the route's content, no chrome.
  const renderRouteContent = () => {
    // /studio is the workbench-themed StudioDev: gated, renders the
    // from-scratch studio; the LeftSidebar above stays mounted.
    if (location.pathname === '/studio') {
      return (
        <PasswordGate routeName="Studio">
          <StudioDev />
        </PasswordGate>
      );
    }
    // /plugins/create routes through the same studio gate — the legacy
    // PluginCreator page is retired.
    if (location.pathname.startsWith('/plugins/create')) {
      return (
        <PasswordGate routeName="Studio">
          <StudioDev />
        </PasswordGate>
      );
    }
    if (currentView === 'home') {
      return (
        <Dashboard
          onCreateNew={handleCreateNew}
          onLoadProject={handleLoadProject}
        />
      );
    }
    if (currentView === 'projects') {
      return (
        <Projects
          onCreateNew={handleCreateNew}
          onLoadProject={handleLoadProject}
        />
      );
    }
    if (currentView === 'search') return <Search />;
    if (currentView === 'userinfo') return <UserInfo onLogout={handleGoToHome} />;
    if (currentView === 'settings') return <Settings />;
    if (currentView === 'tools') {
      return (
        <PasswordGate routeName="Tools">
          <Tools />
        </PasswordGate>
      );
    }
    if (currentView === 'whatsnew') return <WhatsNew />;
    if (currentView === 'docs') return <Docs />;
    if (currentView === 'research') return <Research />;
    if (currentView === 'downloads') return <Downloads />;
    if (
      currentView === 'about' ||
      currentView === 'privacy' ||
      currentView === 'terms' ||
      currentView === 'help' ||
      currentView === 'feedback' ||
      currentView === 'plans' ||
      currentView === 'verify'
    ) {
      const LegalComponent =
        currentView === 'about'    ? About    :
        currentView === 'privacy'  ? Privacy  :
        currentView === 'terms'    ? Terms    :
        currentView === 'help'     ? Help     :
        currentView === 'feedback' ? Feedback :
        currentView === 'verify'   ? Verify   :
        Plans;
      return <LegalComponent />;
    }
    if (currentView === 'plugins') return <Plugins />;
    if (currentView === 'models') return <Models />;
    if (currentView === 'creation') {
      const cid = location.pathname.split('/')[2];
      return <CreationView creationId={cid} />;
    }
    if (currentView === 'publicProfile') {
      const profileUsername = location.pathname.split('/')[2];
      return <PublicProfile username={profileUsername} />;
    }
    if (currentView === 'dashboard') {
      return (
        <div id="main-content">
          <Dashboard
            onCreateNew={handleCreateNew}
            onLoadProject={handleLoadProject}
          />
        </div>
      );
    }
    return null;
  };

  return (
    <div className="App">
      <LeftSidebar
        onBackToDashboard={handleBackToDashboard}
        onGoToHome={handleGoToHome}
        onGoToSearch={handleGoToSearch}
        onGoToUserInfo={handleGoToUserInfo}
        onGoToTools={handleGoToTools}
        onGoToWhatsNew={handleGoToWhatsNew}
        onGoToResearch={handleGoToResearch}
        onGoToDownloads={handleGoToDownloads}
        onGoToPlugins={handleGoToPlugins}
        onGoToModels={handleGoToModels}
        onToggleSearch={handleToggleSearch}
        onShowGenerationPanel={handleShowGenerationPanel}
        onShowMidiBrowser={handleShowMidiBrowser}
        showMidiBrowser={showMidiBrowser}
        onToggleChat={handleToggleChat}
        showChatWindow={showChatWindow}
        onShowBookmarks={handleShowBookmarks}
        showBookmarks={showBookmarks}
        {...viewFlags}
      />
      {renderRouteContent()}
    </div>
  );
}

/**
 * App Component - Wraps everything in Router and AppProvider
 */
function App() {
  return (
    <Router>
      <AppProvider>
        <AppContent />
      </AppProvider>
    </Router>
  );
}

export default App;
