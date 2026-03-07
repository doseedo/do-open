import React, { useCallback, useState, useEffect } from 'react';
import { useApp } from '../../../context/AppContext';
import { getCurrentUser } from '../../../services/authService';
import SidebarLink from './SidebarLink';
import SidebarSection from './SidebarSection';
import styles from './LeftSidebar.module.css';
import logoImg from '../../../assets/transparentlogo.png';

/**
 * LeftSidebar Component
 * Main navigation sidebar with collapsible menu
 */
const LeftSidebar = React.memo(({ onBackToDashboard, onGoToHome, onGoToSearch, onGoToUserInfo, onGoToTools, onGoToWhatsNew, onGoToResearch, onGoToPlugins, onGoToDO1, onToggleSearch: onToggleMidiBrowser, onShowGenerationPanel, onShowMidiBrowser, showMidiBrowser, onToggleChat, showChatWindow, isDashboardView, isHomeView, isSearchView, isUserInfoView, isToolsView, isWhatsNewView, isResearchView, isPluginsView, isDO1View }) => {
  const { state, dispatch } = useApp();
  const [userInfo, setUserInfo] = useState(null);
  const [showMoreMenu, setShowMoreMenu] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);

  // Track viewport size
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth <= 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Get user info on mount
  useEffect(() => {
    const user = getCurrentUser();
    setUserInfo(user);
  }, []);

  // Force sidebar to be expanded when not in DAW view (desktop only)
  const isSpecialView = isDashboardView || isHomeView || isSearchView || isUserInfoView || isToolsView || isWhatsNewView || isResearchView || isPluginsView || isDO1View;
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

  return (
    <>
    {isMobile && (
      <button className={styles.mobileMenuBtn} onClick={toggleSidebar}>
        <i className={`fa-solid ${state.sidebar.isExpanded ? 'fa-xmark' : 'fa-bars'}`}></i>
      </button>
    )}
    {isMobile && (
      <div
        className={`${styles.mobileOverlay} ${state.sidebar.isExpanded ? styles.visible : ''}`}
        onClick={toggleSidebar}
      />
    )}
    <div className={`${styles.sidebar} ${(!isMobile && isSpecialView) || state.sidebar.isExpanded ? styles.expanded : ''}`}>
      {/* Logo and toggle button at top */}
      {state.sidebar.isExpanded ? (
        <div className={styles.logoHeader}>
          <img src={logoImg} alt="Doseedo" className={styles.logo} />
          {!isSpecialView && (
            <button className={styles.collapseBtn} onClick={toggleSidebar}>
              <i className="fa-solid fa-chevron-left"></i>
            </button>
          )}
        </div>
      ) : (
        <h3 className={styles.toggleBtn} onClick={toggleSidebar}>
          <i className="fa-solid fa-bars"></i>
        </h3>
      )}

      {/* Menu Links (visible when expanded) */}
      <div className={styles.menuLinks}>
        {/* Main Navigation */}
        <SidebarSection>
          <SidebarLink
            href="#"
            icon="fa-solid fa-house"
            label="Home"
            active={isHomeView}
            onClick={(e) => {
              e.preventDefault();
              onGoToHome();
            }}
          />
          <SidebarLink
            href="#"
            icon="fa-solid fa-magnifying-glass"
            label="Search"
            active={isSearchView}
            onClick={(e) => {
              e.preventDefault();
              onGoToSearch();
            }}
          />
        </SidebarSection>

        {/* Create Section */}
        <SidebarSection title="Create" showDivider>
          <SidebarLink
            href="#"
            icon="fa-solid fa-folder-closed"
            label="Projects"
            active={isDashboardView}
            onClick={(e) => {
              e.preventDefault();
              onBackToDashboard();
            }}
          />
          <SidebarLink href="" icon="fa-solid fa-plus" label="New Session" highlighted disabled tooltip="Coming soon" />
          <SidebarLink
            href="#"
            icon="fa-solid fa-wrench"
            label="Tools"
            disabled
            tooltip="Coming soon"
          />
          <SidebarLink
            href="#"
            icon="fa-solid fa-puzzle-piece"
            label="Plugins"
            active={isPluginsView}
            onClick={(e) => {
              e.preventDefault();
              onGoToPlugins();
            }}
          />
          <SidebarLink
            href="#"
            icon="fa-solid fa-bolt"
            label="DO1"
            active={isDO1View}
            onClick={(e) => {
              e.preventDefault();
              if (onGoToDO1) onGoToDO1();
            }}
          />
        </SidebarSection>

        {/* Info Section */}
        <SidebarSection title="Info" showDivider>
          <SidebarLink
            href="#"
            icon="fa-solid fa-flask"
            label="Research"
            active={isResearchView}
            onClick={(e) => {
              e.preventDefault();
              onGoToResearch();
            }}
          />
          <SidebarLink
            href="#"
            icon="fa-solid fa-newspaper"
            label="What's New"
            active={isWhatsNewView}
            onClick={(e) => {
              e.preventDefault();
              onGoToWhatsNew();
            }}
          />
          <div className={styles.moreMenu}>
            <SidebarLink
              href="#"
              icon="fa-solid fa-ellipsis"
              label="More"
              onClick={(e) => {
                e.preventDefault();
                setShowMoreMenu(!showMoreMenu);
              }}
            />
            {showMoreMenu && (
              <div className={styles.moreDropdown}>
                <a href="https://docs.doseedo.com" target="_blank" rel="noopener noreferrer" className={styles.moreItem}>
                  <i className="fa-solid fa-circle-question"></i>
                  <span>Help</span>
                </a>
                <a href="https://doseedo.com/about" target="_blank" rel="noopener noreferrer" className={styles.moreItem}>
                  <i className="fa-solid fa-info-circle"></i>
                  <span>About</span>
                </a>
                <a href="https://doseedo.com/feedback" target="_blank" rel="noopener noreferrer" className={styles.moreItem}>
                  <i className="fa-solid fa-message"></i>
                  <span>Feedback</span>
                </a>
              </div>
            )}
          </div>
        </SidebarSection>
      </div>

      {/* Toolbar (visible when collapsed) */}
      <div className={styles.toolbar}>
        <button className={styles.toolbarBtn}>
          <i className="fa-solid fa-bookmark"></i>
        </button>
        <button
          className={`${styles.toolbarBtn} ${!showMidiBrowser && !showChatWindow ? styles.active : ''}`}
          onClick={onShowGenerationPanel}
          title="Generation Panel"
        >
          <i className="fa-solid fa-wand-magic-sparkles"></i>
        </button>
        <button
          className={`${styles.toolbarBtn} ${showMidiBrowser ? styles.active : ''}`}
          onClick={onShowMidiBrowser}
          title="Browse MIDI Files"
        >
          <i className="fa-solid fa-magnifying-glass"></i>
        </button>
        <button
          className={`${styles.toolbarBtn} ${showChatWindow ? styles.active : ''}`}
          onClick={onToggleChat}
          title="AI Chat Assistant"
        >
          <i className="fa-solid fa-comments"></i>
        </button>
      </div>

      {/* User Info Footer (visible when expanded) */}
      {state.sidebar.isExpanded && (
        <div
          className={styles.userInfoFooter}
          onClick={() => onGoToUserInfo()}
        >
          <div className={styles.userAvatar}>
            <i className="fa-solid fa-user"></i>
          </div>
          <div className={styles.userDetails}>
            <div className={styles.userName}>{userInfo?.username || 'Guest'}</div>
            <div className={styles.userPlan}>{userInfo?.subscriptionStatus || 'Free'}</div>
          </div>
        </div>
      )}
    </div>
    </>
  );
});

LeftSidebar.displayName = 'LeftSidebar';

export default LeftSidebar;
