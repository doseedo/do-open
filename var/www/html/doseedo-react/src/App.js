import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, useNavigate, useLocation } from 'react-router-dom';
import { AppProvider, useApp } from './context/AppContext';
import { useKeyboardControls } from './hooks/useKeyboardControls';
import * as authService from './services/authService';
import * as sessionService from './services/sessionService';
import Navbar from './components/Navbar/Navbar';
import LeftSidebar from './components/Sidebar/LeftSidebar/LeftSidebar';
import TrackInfoSidebar from './components/Sidebar/RightSidebar/TrackInfoSidebar';
import GenerationPanelOptimized from './components/GenerationPanel/GenerationPanelOptimized';
import MIDIBrowser from './components/MIDIBrowser/MIDIBrowser';
import ChatWindow from './components/ChatWindow/ChatWindow';
import VideoUploadOptimized from './components/VideoUpload/VideoUploadOptimized';
import ModeSelector from './components/ModeSelector/ModeSelector';
import MIDIChart from './components/MIDIChart/MIDIChart';
import AudioWaveform from './components/AudioWaveform/AudioWaveform';
import ImageViewer from './components/ImageViewer/ImageViewer';
import FXView from './components/FXView/FXView';
import DAWOptimized from './components/DAW/DAWOptimized';
import ResizeBar from './components/ResizeBar/ResizeBar';
import VerticalResizeBar from './components/ResizeBar/VerticalResizeBar';
import Dashboard from './components/Dashboard/Dashboard';
import Home from './components/Home/Home';
import Search from './components/Search/Search';
import UserInfo from './components/UserInfo/UserInfo';
import Tools from './components/Tools/Tools';
import WhatsNew from './components/WhatsNew/WhatsNew';
import ChordWindow from './components/ChordWindow/ChordWindow';
import AudioLabeler from './components/AudioLabeler/AudioLabeler';
import DataMonitor from './components/DataMonitor/DataMonitor';
// Real-time theme editor UI enabled:
import ThemeEditor from './components/ThemeEditor/ThemeEditor';
import LiquidGlassFilters from './components/LiquidGlassFilters/LiquidGlassFilters';

/**
 * AppContent - Inner component with access to context
 */
function AppContent() {
  const { state, dispatch } = useApp();
  const navigate = useNavigate();
  const location = useLocation();
  const [contentMode, setContentMode] = useState('video'); // 'video', 'midi', 'audio', 'image', or 'fx'
  const [showMidiBrowser, setShowMidiBrowser] = useState(false); // Toggle between generation panel and MIDI browser
  const [showChatWindow, setShowChatWindow] = useState(false); // Toggle for chat window
  const [panelWidth, setPanelWidth] = useState(340); // Initial width (15% smaller: 400 * 0.85)
  const [panelHeight, setPanelHeight] = useState(420); // Initial height (user preference)
  const [dawTracksHeight, setDawTracksHeight] = useState(600); // Height for DAW tracks scrollable area
  const [pluginDawHeight, setPluginDawHeight] = useState(200); // Height for DAW in plugin mode
  const [minWidth, setMinWidth] = useState(200);
  const [maxWidth, setMaxWidth] = useState(1400);
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

  // Determine current view from URL
  const currentView = location.pathname === '/dashboard' ? 'home' :
                      location.pathname === '/projects' ? 'dashboard' :
                      location.pathname === '/search' ? 'search' :
                      location.pathname === '/profile' ? 'userinfo' :
                      location.pathname === '/tools' ? 'tools' :
                      location.pathname === '/whats-new' ? 'whatsnew' :
                      location.pathname === '/studio' ? 'daw' :
                      location.pathname === '/label' ? 'label' :
                      location.pathname === '/monitor' ? 'monitor' :
                      location.pathname === '/' ? 'home' : 'home';

  // Redirect root based on auth status - retry multiple times to handle cookie timing
  useEffect(() => {
    if (location.pathname === '/') {
      let attempts = 0;
      const maxAttempts = 5;
      const checkInterval = 150; // ms between checks

      const checkAuth = () => {
        attempts++;
        if (authService.isAuthenticated()) {
          // Authenticated users go to dashboard
          navigate('/dashboard', { replace: true });
        } else if (attempts >= maxAttempts) {
          // After 750ms of retries, assume not authenticated
          window.location.href = '/home';
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
    if (!hasCheckedAutoLoad && location.pathname === '/') {
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

  // Calculate initial position and constraints from rendered element
  useEffect(() => {
    if (contentRef.current) {
      const rect = contentRef.current.getBoundingClientRect();
      // Don't override panelWidth - let initial state control it

      // Calculate proper min/max based on left edge
      const leftEdge = rect.left;
      setLeftOffset(leftEdge); // Capture left offset for DAW alignment
      setMinWidth(leftEdge + 200); // Min width of 200px
      setMaxWidth(leftEdge + 1400); // Max width of 1400px

      // Also set CSS variable immediately
      const actualWidth = rect.width;
      document.documentElement.style.setProperty('--bus-label-width', `${actualWidth}px`);
      window.dispatchEvent(new CustomEvent('busLabelWidthChanged', { detail: actualWidth }));
    }
  }, []);

  // Update CSS custom property when panel width changes
  useEffect(() => {
    if (!contentRef.current) return;
    // Calculate width exactly as GenerationPanel does: panelWidth - current left position
    const currentLeft = contentRef.current.getBoundingClientRect().left;
    const actualWidth = panelWidth - currentLeft;
    document.documentElement.style.setProperty('--bus-label-width', `${actualWidth}px`);

    // Dispatch custom event to notify DAW of width change
    window.dispatchEvent(new CustomEvent('busLabelWidthChanged', { detail: actualWidth }));
  }, [panelWidth]);

  // Force alignment recalculation after initial render when layout is complete
  useEffect(() => {
    // Use requestAnimationFrame to ensure DOM is fully laid out
    const rafId = requestAnimationFrame(() => {
      if (!contentRef.current) return;
      const currentLeft = contentRef.current.getBoundingClientRect().left;
      const actualWidth = panelWidth - currentLeft;
      document.documentElement.style.setProperty('--bus-label-width', `${actualWidth}px`);
      window.dispatchEvent(new CustomEvent('busLabelWidthChanged', { detail: actualWidth }));
    });

    return () => cancelAnimationFrame(rafId);
  }, [currentView, panelWidth]); // Re-run when view changes or initial panelWidth is set

  const handleResize = (newLeft) => {
    setPanelWidth(newLeft);
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

  // Toggle MIDI browser
  const handleToggleSearch = () => {
    setShowMidiBrowser(prev => !prev);
    setShowChatWindow(false); // Hide chat when showing MIDI browser
  };

  // Show generation panel (wand icon)
  const handleShowGenerationPanel = () => {
    setShowMidiBrowser(false); // false = generation panel
    setShowChatWindow(false);
  };

  // Show MIDI browser (search icon)
  const handleShowMidiBrowser = () => {
    setShowMidiBrowser(true); // true = MIDI browser
    setShowChatWindow(false);
  };

  // Toggle chat window
  const handleToggleChat = () => {
    setShowChatWindow(prev => !prev);
    setShowMidiBrowser(false); // Hide MIDI browser when showing chat
  };

  // Show home view with sidebar
  if (currentView === 'home') {
    return (
      <div className="App">
        <LiquidGlassFilters />
        <LeftSidebar
          onBackToDashboard={handleBackToDashboard}
          onGoToHome={handleGoToHome}
          onGoToSearch={handleGoToSearch}
          onGoToUserInfo={handleGoToUserInfo}
          onGoToTools={handleGoToTools}
          onGoToWhatsNew={handleGoToWhatsNew}
          onToggleSearch={handleToggleSearch}
          onShowGenerationPanel={handleShowGenerationPanel}
          onShowMidiBrowser={handleShowMidiBrowser}
          showMidiBrowser={showMidiBrowser}
          onToggleChat={handleToggleChat}
          showChatWindow={showChatWindow}
          isHomeView={true}
        />
        <Home />
        {/* Real-time theme editor enabled: */}
        <ThemeEditor />
      </div>
    );
  }

  // Show search view with sidebar
  if (currentView === 'search') {
    return (
      <div className="App">
        <LiquidGlassFilters />
        <LeftSidebar
          onBackToDashboard={handleBackToDashboard}
          onGoToHome={handleGoToHome}
          onGoToSearch={handleGoToSearch}
          onGoToUserInfo={handleGoToUserInfo}
          onGoToTools={handleGoToTools}
          onGoToWhatsNew={handleGoToWhatsNew}
          onToggleSearch={handleToggleSearch}
          onShowGenerationPanel={handleShowGenerationPanel}
          onShowMidiBrowser={handleShowMidiBrowser}
          showMidiBrowser={showMidiBrowser}
          onToggleChat={handleToggleChat}
          showChatWindow={showChatWindow}
          isSearchView={true}
        />
        <Search />
        {/* Real-time theme editor enabled: */}
        <ThemeEditor />
      </div>
    );
  }

  // Show user info view with sidebar
  if (currentView === 'userinfo') {
    return (
      <div className="App">
        <LiquidGlassFilters />
        <LeftSidebar
          onBackToDashboard={handleBackToDashboard}
          onGoToHome={handleGoToHome}
          onGoToSearch={handleGoToSearch}
          onGoToUserInfo={handleGoToUserInfo}
          onGoToTools={handleGoToTools}
          onGoToWhatsNew={handleGoToWhatsNew}
          onToggleSearch={handleToggleSearch}
          onShowGenerationPanel={handleShowGenerationPanel}
          onShowMidiBrowser={handleShowMidiBrowser}
          showMidiBrowser={showMidiBrowser}
          onToggleChat={handleToggleChat}
          showChatWindow={showChatWindow}
          isUserInfoView={true}
        />
        <UserInfo onLogout={handleGoToHome} />
        {/* Real-time theme editor enabled: */}
        <ThemeEditor />
      </div>
    );
  }

  // Show tools view with sidebar
  if (currentView === 'tools') {
    return (
      <div className="App">
        <LiquidGlassFilters />
        <LeftSidebar
          onBackToDashboard={handleBackToDashboard}
          onGoToHome={handleGoToHome}
          onGoToSearch={handleGoToSearch}
          onGoToUserInfo={handleGoToUserInfo}
          onGoToTools={handleGoToTools}
          onGoToWhatsNew={handleGoToWhatsNew}
          onToggleSearch={handleToggleSearch}
          onShowGenerationPanel={handleShowGenerationPanel}
          onShowMidiBrowser={handleShowMidiBrowser}
          showMidiBrowser={showMidiBrowser}
          onToggleChat={handleToggleChat}
          showChatWindow={showChatWindow}
          isToolsView={true}
        />
        <Tools />
        {/* Real-time theme editor enabled: */}
        <ThemeEditor />
      </div>
    );
  }

  // Show what's new view with sidebar
  if (currentView === 'whatsnew') {
    return (
      <div className="App">
        <LiquidGlassFilters />
        <LeftSidebar
          onBackToDashboard={handleBackToDashboard}
          onGoToHome={handleGoToHome}
          onGoToSearch={handleGoToSearch}
          onGoToUserInfo={handleGoToUserInfo}
          onGoToTools={handleGoToTools}
          onGoToWhatsNew={handleGoToWhatsNew}
          onToggleSearch={handleToggleSearch}
          onShowGenerationPanel={handleShowGenerationPanel}
          onShowMidiBrowser={handleShowMidiBrowser}
          showMidiBrowser={showMidiBrowser}
          onToggleChat={handleToggleChat}
          showChatWindow={showChatWindow}
          isWhatsNewView={true}
        />
        <WhatsNew />
        {/* Real-time theme editor enabled: */}
        <ThemeEditor />
      </div>
    );
  }

  // Show audio labeler view (standalone, no sidebar needed)
  if (currentView === 'label') {
    return (
      <div className="App">
        <AudioLabeler />
      </div>
    );
  }

  // Show data monitor view (standalone, no sidebar needed)
  if (currentView === 'monitor') {
    return (
      <div className="App">
        <DataMonitor />
      </div>
    );
  }

  // Show dashboard view with sidebar
  if (currentView === 'dashboard') {
    return (
      <div className="App">
        <LiquidGlassFilters />
        <LeftSidebar
          onBackToDashboard={handleBackToDashboard}
          onGoToHome={handleGoToHome}
          onGoToSearch={handleGoToSearch}
          onGoToUserInfo={handleGoToUserInfo}
          onGoToTools={handleGoToTools}
          onGoToWhatsNew={handleGoToWhatsNew}
          onToggleSearch={handleToggleSearch}
          onShowGenerationPanel={handleShowGenerationPanel}
          onShowMidiBrowser={handleShowMidiBrowser}
          showMidiBrowser={showMidiBrowser}
          onToggleChat={handleToggleChat}
          showChatWindow={showChatWindow}
          isDashboardView={true}
        />
        <div id="main-content">
          <Dashboard
            onCreateNew={handleCreateNew}
            onLoadProject={handleLoadProject}
          />
        </div>
        {/* Real-time theme editor enabled: */}
        <ThemeEditor />
      </div>
    );
  }

  // Show DAW view
  return (
    <div className="App">
      <LiquidGlassFilters />
      <Navbar />
      <LeftSidebar
        onBackToDashboard={handleBackToDashboard}
        onGoToHome={handleGoToHome}
        onGoToSearch={handleGoToSearch}
        onGoToUserInfo={handleGoToUserInfo}
        onGoToTools={handleGoToTools}
          onGoToWhatsNew={handleGoToWhatsNew}
        onToggleSearch={handleToggleSearch}
        showMidiBrowser={showMidiBrowser}
        onToggleChat={handleToggleChat}
        showChatWindow={showChatWindow}
      />
      {/* Show track info sidebar for both tracks and buses */}
      <TrackInfoSidebar />
      {/* Real-time theme editor enabled: */}
      <ThemeEditor />
      <div id="main-content">
        <div id="wrapper" style={{ position: 'relative' }}>
          {/* Top Left: Generation Panel or MIDI Browser or Chat Window */}
          <div
            ref={contentRef}
            className="content"
            style={{
              width: contentRef.current ? `${panelWidth - contentRef.current.getBoundingClientRect().left}px` : 'auto',
              height: `${panelHeight}px`,
              overflow: 'hidden'
            }}
          >
            <div className="startcontainer scrollable" style={{ height: '100%', overflowY: 'auto' }}>
              {showChatWindow ? (
                <ChatWindow onClose={() => setShowChatWindow(false)} />
              ) : showMidiBrowser ? (
                <MIDIBrowser onClose={() => setShowMidiBrowser(false)} />
              ) : (
                <GenerationPanelOptimized />
              )}
            </div>
          </div>

          {/* Horizontal Resize Bar */}
          {!state.pluginMode && (
            <ResizeBar
              leftPosition={panelWidth}
              onResize={handleResize}
              minWidth={minWidth}
              maxWidth={maxWidth}
            />
          )}

          {/* Top Right: Mode Selector + Video Upload or MIDI Chart (with DAW in bottom third in Plugin Mode) */}
          <div style={{
            height: `${panelHeight}px`,
            flex: 1,
            minWidth: 0,
            display: 'flex',
            flexDirection: 'row',
            background: 'var(--gradient-glow-subtle)',
            position: 'relative'
          }}>
            {/* Mode Selector Column - now absolutely positioned over content */}
            <ModeSelector
              selectedMode={contentMode}
              onModeChange={setContentMode}
            />

            {/* Content Area - In plugin mode, split into video/midi/audio content (top 2/3) and DAW (bottom 1/3) */}
            <div style={{
              flex: 1,
              minWidth: 0,
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
              zIndex: 1
            }}>
              {/* Top section: Video/MIDI/Audio/Image/FX content */}
              <div style={{
                height: state.pluginMode ? `${panelHeight - pluginDawHeight}px` : '100%',
                minHeight: 0,
                overflow: 'hidden',
                position: 'relative'  // For chord window positioning
              }}>
                {contentMode === 'fx' ? (
                  <FXView />
                ) : contentMode === 'video' ? (
                  <VideoUploadOptimized />
                ) : contentMode === 'midi' ? (
                  <MIDIChart />
                ) : contentMode === 'audio' ? (
                  <AudioWaveform />
                ) : (
                  <ImageViewer />
                )}

                {/* Chord Window - appears in lower 1/3, but not in FX mode or plugin mode */}
                {contentMode !== 'fx' && !state.pluginMode && <ChordWindow />}
              </div>

              {/* Vertical Resize Bar for Plugin Mode - between content and DAW */}
              {state.pluginMode && (
                <VerticalResizeBar
                  topPosition={panelHeight - pluginDawHeight}
                  onResize={handlePluginDawResize}
                  minHeight={100}
                  maxHeight={panelHeight - 100}
                />
              )}

              {/* DAW in lower third - only shown in plugin mode */}
              {state.pluginMode && (
                <div style={{
                  height: `${pluginDawHeight}px`,
                  minHeight: 0,
                  overflow: 'hidden',
                  background: 'var(--daw-bg)'
                }}>
                  <DAWOptimized
                    maxTracksHeight={pluginDawHeight - 50}
                    onTracksHeightChange={() => {}}
                    panelWidth={panelWidth}
                    pluginMode={state.pluginMode}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Vertical Resize Bar - only in normal mode */}
          {!state.pluginMode && (
            <VerticalResizeBar
              topPosition={panelHeight}
              onResize={handleVerticalResize}
              minHeight={300}
              maxHeight={1000}
            />
          )}

          {/* Bottom: DAW - only shown in normal mode */}
          {!state.pluginMode && (
            <div style={{
              marginTop: '10px',
              width: `calc(100% - ${leftOffset}px)`,
              position: 'relative',
              marginLeft: '0px',
              overflowX: 'auto',
              overflowY: 'visible'
            }}>
              <DAWOptimized
                maxTracksHeight={dawTracksHeight}
                onTracksHeightChange={setDawTracksHeight}
                panelWidth={panelWidth}
                pluginMode={state.pluginMode}
              />
            </div>
          )}
        </div>
      </div>
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
