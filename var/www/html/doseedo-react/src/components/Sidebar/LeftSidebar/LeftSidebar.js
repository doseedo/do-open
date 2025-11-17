import React, { useCallback, useState, useEffect } from 'react';
import { useApp } from '../../../context/AppContext';
import { getCurrentUser } from '../../../services/authService';
import SidebarLink from './SidebarLink';
import SidebarSection from './SidebarSection';
import styles from './LeftSidebar.module.css';

/**
 * LeftSidebar Component
 * Main navigation sidebar with collapsible menu
 */
const LeftSidebar = React.memo(({ onBackToDashboard, onGoToHome, onGoToSearch, onGoToUserInfo, onGoToTools, onGoToWhatsNew, onToggleSearch: onToggleMidiBrowser, onShowGenerationPanel, onShowMidiBrowser, showMidiBrowser, onToggleChat, showChatWindow, isDashboardView, isHomeView, isSearchView, isUserInfoView, isToolsView, isWhatsNewView }) => {
  const { state, dispatch } = useApp();
  const [userInfo, setUserInfo] = useState(null);
  const [showMoreMenu, setShowMoreMenu] = useState(false);

  // Get user info on mount
  useEffect(() => {
    const user = getCurrentUser();
    setUserInfo(user);
  }, []);

  // Force sidebar to be expanded when not in DAW view
  const isSpecialView = isDashboardView || isHomeView || isSearchView || isUserInfoView || isToolsView || isWhatsNewView;
  useEffect(() => {
    if (isSpecialView && !state.sidebar.isExpanded) {
      dispatch({ type: 'TOGGLE_SIDEBAR' });
    }
  }, [isSpecialView, state.sidebar.isExpanded, dispatch]);

  const toggleSidebar = useCallback(() => {
    // Only allow toggle if in DAW view
    if (!isSpecialView) {
      dispatch({ type: 'TOGGLE_SIDEBAR' });
    }
  }, [isSpecialView, dispatch]);

  return (
    <div className={`${styles.sidebar} ${state.sidebar.isExpanded || isSpecialView ? styles.expanded : ''}`}>
      {/* Logo and toggle button at top */}
      {state.sidebar.isExpanded ? (
        <div className={styles.logoHeader}>
          <img src="/assets/icons/dologotp.png" alt="Doseedo" className={styles.logo} />
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
          <SidebarLink href="" icon="fa-solid fa-plus" label="New Session" highlighted />
          <SidebarLink
            href="#"
            icon="fa-solid fa-wrench"
            label="Tools"
            active={isToolsView}
            onClick={(e) => {
              e.preventDefault();
              onGoToTools();
            }}
          />
        </SidebarSection>

        {/* Info Section */}
        <SidebarSection title="Info" showDivider>
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
  );
});

LeftSidebar.displayName = 'LeftSidebar';

export default LeftSidebar;
