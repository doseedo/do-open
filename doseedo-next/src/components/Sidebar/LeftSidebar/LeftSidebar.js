import React, { useCallback, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../../context/AppContext';
import { getCurrentUser } from '../../../services/authService';
import styles from './LeftSidebar.module.css';

/**
 * LeftSidebar — workbench cream theme rewrite.
 *
 * DOM + CSS restyled to match the mockup at /Users/hydroadmin/Downloads/daw 2/dashboard.
 * All prop names, handlers, and expand/collapse state are unchanged from the previous
 * implementation — App.js keeps working without any prop updates.
 */

// Inline SVG icon set — mirrors the mockup's stroke icons. Sticking to SVG (vs
// Font Awesome) keeps the cream theme consistent with the mockup's hairline style.
const Icon = {
  home: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M3 12l9-8 9 8M5 10v10h14V10" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  search: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="11" cy="11" r="6" /><path d="M20 20l-4-4" strokeLinecap="round" />
    </svg>
  ),
  folder: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M3 6h7l2 2h9v11H3z" strokeLinejoin="round" />
    </svg>
  ),
  plus: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M12 5v14M5 12h14" strokeLinecap="round" />
    </svg>
  ),
  wrench: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M14 4l6 6-9 9H5v-6z" strokeLinejoin="round" />
    </svg>
  ),
  plugin: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M6 4h8l4 4v12H6z M14 4v4h4" strokeLinejoin="round" />
    </svg>
  ),
  bolt: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M13 3L5 14h6l-2 7 8-11h-6z" strokeLinejoin="round" />
    </svg>
  ),
  models: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="5" cy="6" r="1.7" /><circle cx="5" cy="18" r="1.7" />
      <circle cx="12" cy="12" r="1.7" />
      <circle cx="19" cy="6" r="1.7" /><circle cx="19" cy="18" r="1.7" />
      <path d="M6.4 6.9L10.6 11M6.4 17.1L10.6 13M13.4 11L17.6 6.9M13.4 13L17.6 17.1" strokeLinecap="round" />
    </svg>
  ),
  research: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M6 3v18M6 3h10l3 4v14" />
    </svg>
  ),
  news: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <rect x="4" y="5" width="16" height="14" rx="1" /><path d="M4 9h16" />
    </svg>
  ),
  download: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M12 3v12M7 11l5 5 5-5M4 21h16" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  more: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <circle cx="6" cy="12" r="1.5" /><circle cx="12" cy="12" r="1.5" /><circle cx="18" cy="12" r="1.5" />
    </svg>
  ),
  help: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="12" cy="12" r="9" /><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.8.4-1 .9-1 1.7M12 17h.01" strokeLinecap="round" />
    </svg>
  ),
  info: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" strokeLinecap="round" />
    </svg>
  ),
  message: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M4 5h16v11H7l-3 3z" strokeLinejoin="round" />
    </svg>
  ),
  shield: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M12 3l8 3v6c0 4.5-3.3 8.3-8 9-4.7-.7-8-4.5-8-9V6l8-3z" strokeLinejoin="round" />
    </svg>
  ),
  doc: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M7 3h7l5 5v13H7z" strokeLinejoin="round" /><path d="M14 3v5h5M9 13h6M9 17h6" strokeLinecap="round" />
    </svg>
  ),
  hamburger: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M4 6h16M4 12h16M4 18h16" strokeLinecap="round" />
    </svg>
  ),
  close: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M6 6l12 12M6 18L18 6" strokeLinecap="round" />
    </svg>
  ),
  wand: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M4 20l9-9M14 5l1 2 2 1-2 1-1 2-1-2-2-1 2-1zM19 10l.7 1.3L21 12l-1.3.7L19 14l-.7-1.3L17 12l1.3-.7z" strokeLinejoin="round" />
    </svg>
  ),
  chat: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M4 5h16v10H8l-4 4z" strokeLinejoin="round" />
    </svg>
  ),
  bookmark: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <path d="M6 4h12v17l-6-4-6 4z" strokeLinejoin="round" />
    </svg>
  ),
};

// Single sidebar entry. Rendered as a <button> when interactive so keyboard +
// screen-reader behaviour is correct; disabled items render a plain div.
const SideItem = ({ icon, label, active, disabled, muted, onClick, tooltip }) => {
  const cls = [
    styles.sideItem,
    active && styles.sideItemActive,
    muted && styles.sideItemMuted,
    disabled && styles.sideItemDisabled,
  ].filter(Boolean).join(' ');

  if (disabled) {
    return (
      <div className={cls} title={tooltip || ''}>
        <span className={styles.sideIcon}>{icon}</span>
        {label}
        {tooltip && <span className={styles.sideTooltip}>{tooltip}</span>}
      </div>
    );
  }

  return (
    <button type="button" className={cls} onClick={onClick}>
      <span className={styles.sideIcon}>{icon}</span>
      {label}
    </button>
  );
};

const LeftSidebar = React.memo(({
  onBackToDashboard, onGoToHome, onGoToSearch, onGoToUserInfo, onGoToTools,
  onGoToWhatsNew, onGoToResearch, onGoToDownloads, onGoToPlugins, onGoToModels,
  onToggleSearch: onToggleMidiBrowser, onShowGenerationPanel, onShowMidiBrowser,
  showMidiBrowser, onToggleChat, showChatWindow,
  isDashboardView, isHomeView, isSearchView, isUserInfoView, isToolsView,
  isWhatsNewView, isResearchView, isDownloadsView, isPluginsView, isModelsView,
}) => {
  const navigate = useNavigate();
  const { state, dispatch } = useApp();
  const [userInfo, setUserInfo] = useState(null);
  const [showMoreMenu, setShowMoreMenu] = useState(false);
  const [isMobile, setIsMobile] = useState(
    typeof window !== 'undefined' ? window.innerWidth <= 768 : false
  );

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth <= 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    setUserInfo(getCurrentUser());
  }, []);

  // Auto-expand on non-DAW views (desktop only).
  const isSpecialView = isDashboardView || isHomeView || isSearchView || isUserInfoView
    || isToolsView || isWhatsNewView || isResearchView || isDownloadsView
    || isPluginsView || isModelsView;

  useEffect(() => {
    if (!isMobile && isSpecialView && !state.sidebar.isExpanded) {
      dispatch({ type: 'TOGGLE_SIDEBAR' });
    }
  }, [isSpecialView, state.sidebar.isExpanded, dispatch, isMobile]);

  const toggleSidebar = useCallback(() => {
    if (isMobile || !isSpecialView) {
      dispatch({ type: 'TOGGLE_SIDEBAR' });
    }
  }, [isSpecialView, dispatch, isMobile]);

  const expanded = (!isMobile && isSpecialView) || state.sidebar.isExpanded;
  const userInitial = (userInfo?.username || 'G').charAt(0).toUpperCase();
  const userTier = userInfo?.subscriptionStatus || 'Free';

  return (
    <>
      {isMobile && (
        <button className={styles.mobileMenuBtn} onClick={toggleSidebar} aria-label="Toggle menu">
          {state.sidebar.isExpanded ? Icon.close : Icon.hamburger}
        </button>
      )}
      {isMobile && (
        <div
          className={`${styles.mobileOverlay} ${state.sidebar.isExpanded ? styles.mobileOverlayVisible : ''}`}
          onClick={toggleSidebar}
        />
      )}

      <aside className={`${styles.side} ${expanded ? styles.sideExpanded : ''}`}>

        {/* Brand header */}
        {expanded ? (
          <div className={styles.sideBrand}>
            <div className={styles.sideMark}>D</div>
            <div className={styles.sideBrandname}>
              doseedo<span> / v0.3</span>
            </div>
            {!isSpecialView && (
              <button className={styles.sideCollapseBtn} onClick={toggleSidebar} aria-label="Collapse">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M15 6l-6 6 6 6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            )}
          </div>
        ) : (
          <button className={styles.sideToggle} onClick={toggleSidebar} aria-label="Open menu">
            {Icon.hamburger}
          </button>
        )}

        {/* Menu groups (expanded only) */}
        {expanded && (
          <>
            <div className={styles.sideGroup}>
              <SideItem icon={Icon.home} label="Home" active={isHomeView} onClick={onGoToHome} />
              <SideItem icon={Icon.search} label="Search" active={isSearchView} onClick={onGoToSearch} />
            </div>

            <div className={styles.sideGroup}>
              <div className={styles.sideLabel}>Create</div>
              <SideItem icon={Icon.folder} label="Projects" active={isDashboardView} onClick={onBackToDashboard} />
              <SideItem icon={Icon.plus} label="New Session" muted disabled tooltip="Coming soon" />
              <SideItem icon={Icon.wrench} label="Tools" disabled tooltip="Coming soon" />
              <SideItem icon={Icon.plugin} label="Plugins" active={isPluginsView} onClick={onGoToPlugins} />
              <SideItem icon={Icon.models} label="Models" active={isModelsView} onClick={onGoToModels} />
            </div>

            <div className={styles.sideGroup}>
              <div className={styles.sideLabel}>Info</div>
              <SideItem icon={Icon.research} label="Research" active={isResearchView} onClick={onGoToResearch} />
              <SideItem icon={Icon.news} label="What's New" active={isWhatsNewView} onClick={onGoToWhatsNew} />
              <SideItem icon={Icon.download} label="Downloads" active={isDownloadsView} onClick={onGoToDownloads} />

              <div className={styles.moreMenu}>
                <SideItem
                  icon={Icon.more}
                  label="More"
                  active={showMoreMenu}
                  onClick={() => setShowMoreMenu((v) => !v)}
                />
                {showMoreMenu && (
                  <div className={styles.moreDropdown}>
                    <button
                      type="button"
                      className={styles.moreItem}
                      onClick={() => { setShowMoreMenu(false); navigate('/help'); }}
                    >
                      <span className={styles.sideIcon}>{Icon.help}</span>
                      <span>Help</span>
                    </button>
                    <button
                      type="button"
                      className={styles.moreItem}
                      onClick={() => { setShowMoreMenu(false); navigate('/about'); }}
                    >
                      <span className={styles.sideIcon}>{Icon.info}</span>
                      <span>About</span>
                    </button>
                    <button
                      type="button"
                      className={styles.moreItem}
                      onClick={() => { setShowMoreMenu(false); navigate('/feedback'); }}
                    >
                      <span className={styles.sideIcon}>{Icon.message}</span>
                      <span>Feedback</span>
                    </button>
                    <button
                      type="button"
                      className={styles.moreItem}
                      onClick={() => { setShowMoreMenu(false); navigate('/privacy'); }}
                    >
                      <span className={styles.sideIcon}>{Icon.shield}</span>
                      <span>Privacy Policy</span>
                    </button>
                    <button
                      type="button"
                      className={styles.moreItem}
                      onClick={() => { setShowMoreMenu(false); navigate('/terms'); }}
                    >
                      <span className={styles.sideIcon}>{Icon.doc}</span>
                      <span>Terms of Service</span>
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div className={styles.sideSpacer} />

            {/* User chip */}
            <button type="button" className={styles.sideUser} onClick={onGoToUserInfo}>
              <div className={styles.sideAvatar}>{userInitial}</div>
              <div>
                <div className={styles.sideUserName}>{userInfo?.username || 'Guest'}</div>
                <div className={styles.sideUserTier}>{userTier}</div>
              </div>
            </button>
          </>
        )}

        {/* Collapsed-state tool rail (DAW quick-toggles) */}
        {!expanded && (
          <div className={styles.toolbar}>
            <button className={styles.toolbarBtn} title="Bookmarks">{Icon.bookmark}</button>
            <button
              className={`${styles.toolbarBtn} ${!showMidiBrowser && !showChatWindow ? styles.toolbarBtnActive : ''}`}
              onClick={onShowGenerationPanel}
              title="Generation Panel"
            >{Icon.wand}</button>
            <button
              className={`${styles.toolbarBtn} ${showMidiBrowser ? styles.toolbarBtnActive : ''}`}
              onClick={onShowMidiBrowser}
              title="Browse MIDI Files"
            >{Icon.search}</button>
            <button
              className={`${styles.toolbarBtn} ${showChatWindow ? styles.toolbarBtnActive : ''}`}
              onClick={onToggleChat}
              title="AI Chat Assistant"
            >{Icon.chat}</button>
          </div>
        )}

      </aside>
    </>
  );
});

LeftSidebar.displayName = 'LeftSidebar';

export default LeftSidebar;
