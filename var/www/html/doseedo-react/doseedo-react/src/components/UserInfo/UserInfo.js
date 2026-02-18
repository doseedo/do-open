import React, { useState, useEffect } from 'react';
import { getCurrentUser, logoutUser } from '../../services/authService';
import styles from './UserInfo.module.css';

/**
 * UserInfo Component
 * User profile and account settings page
 */
const UserInfo = ({ onLogout }) => {
  const [userInfo, setUserInfo] = useState(null);

  useEffect(() => {
    const user = getCurrentUser();
    setUserInfo(user);
  }, []);

  const handleLogout = async () => {
    if (window.confirm('Are you sure you want to logout?')) {
      await logoutUser();
      if (onLogout) {
        onLogout();
      } else {
        window.location.href = '/login.html';
      }
    }
  };

  return (
    <div className={styles.userInfoContainer}>
      <div className={styles.userInfoHeader}>
        <h1 className={styles.userInfoTitle}>Account Settings</h1>
        <p className={styles.userInfoSubtitle}>Manage your profile and preferences</p>
      </div>

      <div className={styles.profileSection}>
        <div className={styles.profileCard}>
          <div className={styles.avatarSection}>
            <div className={styles.avatarLarge}>
              <i className="fa-solid fa-user"></i>
            </div>
            <button className={styles.changeAvatarBtn}>
              <i className="fa-solid fa-camera"></i>
              Change Photo
            </button>
          </div>

          <div className={styles.profileDetails}>
            <div className={styles.detailRow}>
              <label>Username</label>
              <div className={styles.detailValue}>
                {userInfo?.username || 'Guest'}
              </div>
            </div>

            <div className={styles.detailRow}>
              <label>Subscription Plan</label>
              <div className={styles.detailValue}>
                <span className={styles.planBadge}>
                  {userInfo?.subscriptionStatus || 'Free'}
                </span>
                {userInfo?.subscriptionStatus === 'Free' && (
                  <a href="/plans.html" className={styles.upgradeLink}>
                    Upgrade to Pro+
                  </a>
                )}
              </div>
            </div>

            <div className={styles.detailRow}>
              <label>Account Status</label>
              <div className={styles.detailValue}>
                <span className={styles.statusBadge}>Active</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.settingsSection}>
        <h2 className={styles.sectionTitle}>Preferences</h2>
        <div className={styles.settingsCard}>
          <div className={styles.settingRow}>
            <div className={styles.settingInfo}>
              <h3>Email Notifications</h3>
              <p>Receive updates about your projects and new features</p>
            </div>
            <label className={styles.switch}>
              <input type="checkbox" defaultChecked />
              <span className={styles.slider}></span>
            </label>
          </div>

          <div className={styles.settingRow}>
            <div className={styles.settingInfo}>
              <h3>Auto-save Projects</h3>
              <p>Automatically save your work every 5 minutes</p>
            </div>
            <label className={styles.switch}>
              <input type="checkbox" defaultChecked />
              <span className={styles.slider}></span>
            </label>
          </div>

          <div className={styles.settingRow}>
            <div className={styles.settingInfo}>
              <h3>High Quality Audio</h3>
              <p>Export audio in highest quality (may take longer)</p>
            </div>
            <label className={styles.switch}>
              <input type="checkbox" />
              <span className={styles.slider}></span>
            </label>
          </div>
        </div>
      </div>

      <div className={styles.actionsSection}>
        <button className={styles.logoutBtn} onClick={handleLogout}>
          <i className="fa-solid fa-arrow-right-from-bracket"></i>
          Logout
        </button>
        <button className={styles.deleteAccountBtn}>
          <i className="fa-solid fa-trash"></i>
          Delete Account
        </button>
      </div>
    </div>
  );
};

export default UserInfo;
