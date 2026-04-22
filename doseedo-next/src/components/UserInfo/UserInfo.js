import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCurrentUser, logoutUser } from '../../services/authService';
import { getProfile, updateProfile, uploadAvatar, getMyLikes, getMyFavorites, getMyDownloads, toggleLike, toggleFavorite } from '../../services/communityAPI';
import styles from './UserInfo.module.css';

const UserInfo = ({ onLogout }) => {
  const navigate = useNavigate();
  const [userInfo, setUserInfo] = useState(null);
  const [profileData, setProfileData] = useState(null);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [saved, setSaved] = useState(false);
  const avatarInputRef = useRef(null);

  // Editable fields
  const [displayName, setDisplayName] = useState('');
  const [bio, setBio] = useState('');
  const [website, setWebsite] = useState('');
  const [socialTwitter, setSocialTwitter] = useState('');
  const [socialGithub, setSocialGithub] = useState('');
  const [socialSoundcloud, setSocialSoundcloud] = useState('');

  // Library
  const [libraryTab, setLibraryTab] = useState('liked');
  const [libraryItems, setLibraryItems] = useState([]);
  const [libraryLoading, setLibraryLoading] = useState(false);

  useEffect(() => {
    const user = getCurrentUser();
    setUserInfo(user);
    if (user?.username) {
      setLoadingProfile(true);
      getProfile(user.username)
        .then(data => {
          setProfileData(data);
          setDisplayName(data.display_name || '');
          setBio(data.bio || '');
          setWebsite(data.website || '');
          setSocialTwitter(data.social_twitter || '');
          setSocialGithub(data.social_github || '');
          setSocialSoundcloud(data.social_soundcloud || '');
        })
        .catch(() => {})
        .finally(() => setLoadingProfile(false));
    }
  }, []);

  // Fetch library items
  useEffect(() => {
    if (!userInfo?.username) return;
    setLibraryLoading(true);
    const fetcher = libraryTab === 'liked' ? getMyLikes : libraryTab === 'favorited' ? getMyFavorites : getMyDownloads;
    fetcher({ type: 'plugin', limit: 50 })
      .then(data => setLibraryItems(data.creations || []))
      .catch(() => setLibraryItems([]))
      .finally(() => setLibraryLoading(false));
  }, [libraryTab, userInfo]);

  const handleSaveProfile = async () => {
    setSavingProfile(true);
    setSaved(false);
    try {
      const res = await updateProfile({
        display_name: displayName,
        bio,
        website,
        social_twitter: socialTwitter,
        social_github: socialGithub,
        social_soundcloud: socialSoundcloud,
      });
      setProfileData(prev => ({ ...prev, ...res }));
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) { console.error('Profile save failed:', err); }
    finally { setSavingProfile(false); }
  };

  const handleAvatarUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await uploadAvatar(file);
      setProfileData(prev => ({ ...prev, avatar_url: res.avatar_url }));
    } catch (err) { console.error('Avatar upload failed:', err); }
  };

  const handleToggleLibraryLike = useCallback(async (id, e) => {
    if (e) e.stopPropagation();
    try {
      const res = await toggleLike(id);
      setLibraryItems(prev => prev.map(c =>
        c.id === id ? { ...c, user_liked: res.liked, like_count: res.like_count } : c
      ));
    } catch (err) { console.error(err); }
  }, []);

  const handleToggleLibraryFav = useCallback(async (id, e) => {
    if (e) e.stopPropagation();
    try {
      const res = await toggleFavorite(id);
      setLibraryItems(prev => prev.map(c =>
        c.id === id ? { ...c, user_favorited: res.favorited, favorite_count: res.favorite_count } : c
      ));
    } catch (err) { console.error(err); }
  }, []);

  const handleLogout = async () => {
    if (!window.confirm('Are you sure you want to logout?')) return;
    await logoutUser();
    if (onLogout) {
      onLogout();
      return;
    }
    // Full reload (not SPA navigation): resets the Clerk context and any
    // in-memory user state. Landing on `/` shows the public home which
    // renders sign-in affordances for unauthenticated visitors.
    window.location.href = '/';
  };

  // --- Guest state ----------------------------------------------------
  // No signed-in user: strip the profile editor, library tabs, and logout
  // button (all useless without a user). Show one clear sign-in CTA and
  // a short list of what signing in unlocks.
  if (!userInfo) {
    return (
      <div className={styles.userInfoContainer}>
        <div className={styles.userInfoHeader}>
          <h1 className={styles.userInfoTitle}>Account</h1>
          <p className={styles.userInfoSubtitle}>You're browsing as a guest.</p>
        </div>

        <div className={styles.guestCard}>
          <div className={styles.guestIcon}>
            <i className="fa-solid fa-user"></i>
          </div>
          <h2 className={styles.guestHeading}>Sign in to doseedo</h2>
          <p className={styles.guestBody}>
            Save your projects to the cloud, publish creations, like and
            favorite plugins, and pick up where you left off on any device.
          </p>
          <div className={styles.guestActions}>
            <button
              className={styles.guestSignInBtn}
              onClick={() => { window.location.href = '/sign-in'; }}
            >
              <i className="fa-solid fa-arrow-right-to-bracket"></i>
              Sign in
            </button>
            <button
              className={styles.guestSignUpBtn}
              onClick={() => { window.location.href = '/sign-up'; }}
            >
              Create account
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.userInfoContainer}>
      <div className={styles.userInfoHeader}>
        <h1 className={styles.userInfoTitle}>Account Settings</h1>
        <p className={styles.userInfoSubtitle}>Manage your profile and preferences</p>
      </div>

      <div className={styles.profileSection}>
        <div className={styles.profileCard}>
          <div className={styles.avatarSection}>
            {profileData?.avatar_url ? (
              <img src={profileData.avatar_url} alt="" className={styles.avatarLargeImg} />
            ) : (
              <div className={styles.avatarLarge}>
                <i className="fa-solid fa-user"></i>
              </div>
            )}
            <input
              ref={avatarInputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={handleAvatarUpload}
            />
            <button className={styles.changeAvatarBtn} onClick={() => avatarInputRef.current?.click()}>
              <i className="fa-solid fa-camera"></i>
              Change Photo
            </button>
          </div>

          <div className={styles.profileDetails}>
            <div className={styles.detailRow}>
              <label>Username</label>
              <div className={styles.detailValue}>
                {userInfo?.username || 'Guest'}
                {userInfo?.username && (
                  <button
                    className={styles.viewProfileLink}
                    onClick={() => navigate(`/profile/${userInfo.username}`)}
                  >
                    View Public Profile
                  </button>
                )}
              </div>
            </div>

            <div className={styles.detailRow}>
              <label>Display Name</label>
              <input
                type="text"
                className={styles.inputField}
                value={displayName}
                onChange={e => setDisplayName(e.target.value)}
                placeholder="Your display name"
              />
            </div>

            <div className={styles.detailRow}>
              <label>Bio</label>
              <textarea
                className={styles.textareaField}
                value={bio}
                onChange={e => setBio(e.target.value)}
                placeholder="Tell us about yourself..."
                rows={3}
              />
            </div>

            <div className={styles.detailRow}>
              <label>Website</label>
              <input
                type="url"
                className={styles.inputField}
                value={website}
                onChange={e => setWebsite(e.target.value)}
                placeholder="https://yoursite.com"
              />
            </div>

            <div className={styles.socialRow}>
              <div className={styles.socialField}>
                <label><i className="fa-brands fa-x-twitter"></i> Twitter</label>
                <input
                  type="text"
                  className={styles.inputField}
                  value={socialTwitter}
                  onChange={e => setSocialTwitter(e.target.value)}
                  placeholder="@handle"
                />
              </div>
              <div className={styles.socialField}>
                <label><i className="fa-brands fa-github"></i> GitHub</label>
                <input
                  type="text"
                  className={styles.inputField}
                  value={socialGithub}
                  onChange={e => setSocialGithub(e.target.value)}
                  placeholder="username"
                />
              </div>
              <div className={styles.socialField}>
                <label><i className="fa-brands fa-soundcloud"></i> SoundCloud</label>
                <input
                  type="text"
                  className={styles.inputField}
                  value={socialSoundcloud}
                  onChange={e => setSocialSoundcloud(e.target.value)}
                  placeholder="username"
                />
              </div>
            </div>

            <div className={styles.detailRow}>
              <label>Subscription Plan</label>
              <div className={styles.detailValue}>
                <span className={styles.planBadge}>
                  {userInfo?.subscriptionStatus || 'Free'}
                </span>
                {userInfo?.subscriptionStatus === 'Free' && (
                  <a href="/plans" className={styles.upgradeLink}>
                    Upgrade to Pro+
                  </a>
                )}
              </div>
            </div>

            <button
              className={styles.saveBtn}
              onClick={handleSaveProfile}
              disabled={savingProfile}
            >
              {savingProfile ? (
                <><i className="fa-solid fa-spinner fa-spin"></i> Saving...</>
              ) : saved ? (
                <><i className="fa-solid fa-check"></i> Saved!</>
              ) : (
                <><i className="fa-solid fa-floppy-disk"></i> Save Profile</>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* ── My Library ── */}
      <div className={styles.settingsSection}>
        <h2 className={styles.sectionTitle}>My Library</h2>
        <div className={styles.libraryTabs}>
          {[
            { id: 'liked', label: 'Liked', icon: 'fa-solid fa-heart' },
            { id: 'favorited', label: 'Saved', icon: 'fa-solid fa-bookmark' },
            { id: 'downloaded', label: 'Downloaded', icon: 'fa-solid fa-download' },
          ].map(t => (
            <button
              key={t.id}
              className={`${styles.libraryTab} ${libraryTab === t.id ? styles.libraryTabActive : ''}`}
              onClick={() => setLibraryTab(t.id)}
            >
              <i className={t.icon}></i> {t.label}
            </button>
          ))}
        </div>
        <div className={styles.libraryGrid}>
          {libraryLoading ? (
            <div className={styles.libraryEmpty}>
              <i className="fa-solid fa-spinner fa-spin"></i> Loading...
            </div>
          ) : libraryItems.length === 0 ? (
            <div className={styles.libraryEmpty}>
              <i className="fa-solid fa-folder-open" style={{ fontSize: 24, marginBottom: 8, display: 'block' }}></i>
              No {libraryTab} creations yet.
            </div>
          ) : (
            libraryItems.map(c => (
              <div
                key={c.id}
                className={styles.libraryCard}
                onClick={() => navigate(`/plugins`)}
              >
                {c.thumbnail_data ? (
                  <div className={styles.libraryThumb}>
                    <img src={c.thumbnail_data} alt="" />
                  </div>
                ) : (
                  <div className={styles.libraryThumbPlaceholder}>
                    <i className="fa-solid fa-puzzle-piece"></i>
                  </div>
                )}
                <div className={styles.libraryCardInfo}>
                  <h3>{c.name}</h3>
                  <p>by {c.author?.display_name || c.author?.username || 'Anonymous'}</p>
                  <div className={styles.libraryCardStats}>
                    <button
                      onClick={e => handleToggleLibraryLike(c.id, e)}
                      style={{ color: c.user_liked ? '#ff6b8a' : 'rgba(255,255,255,0.25)', background: 'none', border: 'none', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}
                    >
                      <i className={c.user_liked ? 'fa-solid fa-heart' : 'fa-regular fa-heart'}></i> {c.like_count || 0}
                    </button>
                    <button
                      onClick={e => handleToggleLibraryFav(c.id, e)}
                      style={{ color: c.user_favorited ? '#fbbf24' : 'rgba(255,255,255,0.25)', background: 'none', border: 'none', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}
                    >
                      <i className={c.user_favorited ? 'fa-solid fa-bookmark' : 'fa-regular fa-bookmark'}></i>
                    </button>
                    <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: 12 }}>
                      <i className="fa-solid fa-download"></i> {c.download_count || 0}
                    </span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* ── Preferences ── */}
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
