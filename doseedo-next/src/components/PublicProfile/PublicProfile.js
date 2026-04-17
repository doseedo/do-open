import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getProfile, toggleLike, toggleFavorite } from '../../services/communityAPI';
import * as authService from '../../services/authService';
import styles from './PublicProfile.module.css';

const PublicProfile = ({ username }) => {
  const navigate = useNavigate();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const currentUser = authService.getCurrentUser();
  const isOwnProfile = currentUser?.username === username;

  useEffect(() => {
    if (!username) return;
    setLoading(true);
    setError(null);
    getProfile(username)
      .then(data => { setProfile(data); setLoading(false); })
      .catch(err => { setError('User not found'); setLoading(false); });
  }, [username]);

  const handleToggleLike = useCallback(async (creationId, e) => {
    if (e) e.stopPropagation();
    if (!authService.isAuthenticated()) return;
    try {
      const res = await toggleLike(creationId);
      setProfile(prev => ({
        ...prev,
        creations: prev.creations.map(c =>
          c.id === creationId ? { ...c, user_liked: res.liked, like_count: res.like_count } : c
        ),
      }));
    } catch (err) { console.error('Like failed:', err); }
  }, []);

  const handleToggleFavorite = useCallback(async (creationId, e) => {
    if (e) e.stopPropagation();
    if (!authService.isAuthenticated()) return;
    try {
      const res = await toggleFavorite(creationId);
      setProfile(prev => ({
        ...prev,
        creations: prev.creations.map(c =>
          c.id === creationId ? { ...c, user_favorited: res.favorited, favorite_count: res.favorite_count } : c
        ),
      }));
    } catch (err) { console.error('Favorite failed:', err); }
  }, []);

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>
          <i className="fa-solid fa-spinner fa-spin"></i> Loading profile...
        </div>
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className={styles.container}>
        <button className={styles.backBtn} onClick={() => navigate(-1)}>
          <i className="fa-solid fa-arrow-left"></i> Back
        </button>
        <div className={styles.error}>{error || 'Profile not found'}</div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <button className={styles.backBtn} onClick={() => navigate(-1)}>
        <i className="fa-solid fa-arrow-left"></i> Back
      </button>

      {/* Profile Header */}
      <div className={styles.profileHeader}>
        <div className={styles.avatarSection}>
          {profile.avatar_url ? (
            <img src={profile.avatar_url} alt={profile.username} className={styles.avatar} />
          ) : (
            <div className={styles.avatarPlaceholder}>
              <i className="fa-solid fa-user"></i>
            </div>
          )}
        </div>
        <div className={styles.profileInfo}>
          <h1 className={styles.displayName}>{profile.display_name || profile.username}</h1>
          <p className={styles.username}>@{profile.username}</p>
          {profile.bio && <p className={styles.bio}>{profile.bio}</p>}
          <div className={styles.stats}>
            <span className={styles.stat}>
              <strong>{profile.creation_count || 0}</strong> creations
            </span>
            <span className={styles.stat}>
              <strong>{profile.total_likes || 0}</strong> likes received
            </span>
          </div>
          <div className={styles.socials}>
            {profile.website && (
              <a href={profile.website} target="_blank" rel="noopener noreferrer" className={styles.socialLink}>
                <i className="fa-solid fa-globe"></i> Website
              </a>
            )}
            {profile.social_twitter && (
              <a href={`https://twitter.com/${profile.social_twitter.replace('@', '')}`} target="_blank" rel="noopener noreferrer" className={styles.socialLink}>
                <i className="fa-brands fa-x-twitter"></i> {profile.social_twitter}
              </a>
            )}
            {profile.social_github && (
              <a href={`https://github.com/${profile.social_github}`} target="_blank" rel="noopener noreferrer" className={styles.socialLink}>
                <i className="fa-brands fa-github"></i> {profile.social_github}
              </a>
            )}
            {profile.social_soundcloud && (
              <a href={`https://soundcloud.com/${profile.social_soundcloud}`} target="_blank" rel="noopener noreferrer" className={styles.socialLink}>
                <i className="fa-brands fa-soundcloud"></i> {profile.social_soundcloud}
              </a>
            )}
          </div>
          {isOwnProfile && (
            <button className={styles.editBtn} onClick={() => navigate('/profile')}>
              <i className="fa-solid fa-pen"></i> Edit Profile
            </button>
          )}
        </div>
      </div>

      {/* Creations Grid */}
      <div className={styles.creationsSection}>
        <h2 className={styles.sectionTitle}>
          {isOwnProfile ? 'My Published Creations' : `${profile.display_name || profile.username}'s Creations`}
        </h2>
        {profile.creations?.length === 0 ? (
          <div className={styles.empty}>
            <i className="fa-solid fa-puzzle-piece"></i>
            <p>No published creations yet.</p>
          </div>
        ) : (
          <div className={styles.grid}>
            {profile.creations?.map(c => (
              <div
                key={c.id}
                className={styles.card}
                onClick={() => navigate(`/plugins?view=${c.id}`)}
              >
                {c.thumbnail_data ? (
                  <div className={styles.cardThumb}>
                    <img src={c.thumbnail_data} alt="" />
                  </div>
                ) : (
                  <div className={styles.cardThumbPlaceholder}>
                    <i className="fa-solid fa-puzzle-piece"></i>
                  </div>
                )}
                <h3 className={styles.cardName}>{c.name}</h3>
                <p className={styles.cardDesc}>
                  {c.description || (c.dsp_summary ? `${c.dsp_summary.pluginType} - ${c.dsp_summary.nodeCount} nodes` : c.creation_type)}
                </p>
                <div className={styles.cardFooter}>
                  <span className={styles.cardStat}>
                    <i className="fa-solid fa-download"></i> {c.download_count || 0}
                  </span>
                  <button
                    onClick={e => handleToggleLike(c.id, e)}
                    className={styles.likeBtn}
                    style={{ color: c.user_liked ? '#ff6b8a' : undefined }}
                  >
                    <i className={c.user_liked ? 'fa-solid fa-heart' : 'fa-regular fa-heart'}></i> {c.like_count || 0}
                  </button>
                  <button
                    onClick={e => handleToggleFavorite(c.id, e)}
                    className={styles.favBtn}
                    style={{ color: c.user_favorited ? '#fbbf24' : undefined }}
                  >
                    <i className={c.user_favorited ? 'fa-solid fa-bookmark' : 'fa-regular fa-bookmark'}></i>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default PublicProfile;
