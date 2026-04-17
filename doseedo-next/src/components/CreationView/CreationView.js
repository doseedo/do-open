import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCreation, toggleLike, forkCreation } from '../../services/communityAPI';
import * as authService from '../../services/authService';
import styles from './CreationView.module.css';

const CreationView = ({ creationId }) => {
  const navigate = useNavigate();
  const user = authService.getCurrentUser();
  const [creation, setCreation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [forking, setForking] = useState(false);

  useEffect(() => {
    if (!creationId) return;
    setLoading(true);
    getCreation(creationId)
      .then(c => { setCreation(c); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [creationId]);

  const handleLike = async () => {
    if (!user) return navigate('/login');
    try {
      const res = await toggleLike(creationId);
      setCreation(prev => ({ ...prev, likes_count: res.likes_count }));
    } catch {}
  };

  const handleFork = async () => {
    if (!user) return navigate('/login');
    setForking(true);
    try {
      const forked = await forkCreation(creationId);
      navigate(`/creation/${forked.id}`);
    } catch (e) {
      setError('Fork failed: ' + e.message);
    } finally {
      setForking(false);
    }
  };

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.skeleton} style={{ height: 200, borderRadius: 16 }} />
        <div className={styles.skeleton} style={{ height: 32, width: '60%', borderRadius: 8, marginTop: 24 }} />
        <div className={styles.skeleton} style={{ height: 18, width: '40%', borderRadius: 6, marginTop: 12 }} />
      </div>
    );
  }

  if (error || !creation) {
    return (
      <div className={styles.container}>
        <div className={styles.errorCard}>
          <h2>Creation not found</h2>
          <p>{error || 'This creation may have been removed.'}</p>
          <button className={styles.backBtn} onClick={() => navigate('/dashboard')}>
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const daw = creation.metadata?.daw || 'Doseedo';
  const trackCount = creation.metadata?.trackCount || creation.metadata?.stems_count || 0;

  return (
    <div className={styles.container}>
      <button className={styles.backLink} onClick={() => navigate(-1)}>
        <i className="fa-solid fa-arrow-left" /> Back
      </button>

      <div className={styles.hero}>
        <div className={styles.waveformPlaceholder}>
          <div className={styles.waveformBars} />
          <div className={styles.dawBadge} data-daw={daw}>{daw}</div>
        </div>
      </div>

      <div className={styles.meta}>
        <h1 className={styles.title}>{creation.title}</h1>
        <a href={`/profile/${creation.author}`} className={styles.author}>
          <span className={styles.authorAvatar}>{creation.author?.[0]?.toUpperCase()}</span>
          {creation.author}
        </a>

        {creation.description && (
          <p className={styles.description}>{creation.description}</p>
        )}

        {creation.tags && creation.tags.length > 0 && (
          <div className={styles.tags}>
            {creation.tags.map(tag => (
              <span key={tag} className={styles.tag}>{tag}</span>
            ))}
          </div>
        )}

        <div className={styles.stats}>
          <span><i className="fa-solid fa-play" /> {creation.plays_count || 0} plays</span>
          <span><i className="fa-solid fa-heart" /> {creation.likes_count || 0} likes</span>
          <span><i className="fa-solid fa-code-fork" /> {creation.forks_count || 0} forks</span>
          <span><i className="fa-solid fa-layer-group" /> {trackCount} stems</span>
        </div>

        {creation.fork_of && (
          <p className={styles.forkNote}>
            <i className="fa-solid fa-code-fork" /> Forked from{' '}
            <a href={`/creation/${creation.fork_of}`}>original</a>
          </p>
        )}
      </div>

      <div className={styles.actions}>
        <button className={styles.actionPrimary} onClick={handleLike}>
          <i className="fa-solid fa-heart" /> Like
        </button>
        <button
          className={styles.actionPrimary}
          onClick={handleFork}
          disabled={forking}
        >
          <i className="fa-solid fa-code-fork" /> {forking ? 'Forking...' : 'Fork to Studio'}
        </button>
      </div>
    </div>
  );
};

export default CreationView;
