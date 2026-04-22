import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import * as sessionService from '../../services/sessionService';
import * as dashboardService from '../../services/dashboardService';
import ProjectCard from './ProjectCard';
import styles from './Dashboard.module.css';

// Deterministic 0..1 pseudo-random from a numeric seed. Used to keep the
// waveform bars stable across renders for the same session name.
const seedRand = (s) => {
  const x = Math.sin(s) * 10000;
  return x - Math.floor(x);
};

// Hash a string to a stable positive int for seeding waveforms + swatches.
const stringHash = (str) => {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  }
  return Math.abs(h) || 1;
};

// Matches --wb-track-* tokens in theme-workbench.css.
const TRACK_SWATCHES = ['#c94f2c', '#7a5d3a', '#2f6b4e', '#4a3d6b', '#1d4c7a', '#5c7a8a'];

const makeWave = ({ seed, bars = 44, envPad = 0.05, envFloor = 0.35, noiseFloor = 0.55 }) => {
  const out = new Array(bars);
  for (let i = 0; i < bars; i++) {
    const t = i / bars;
    const env = envFloor + (1 - envFloor) * Math.pow(
      Math.sin(Math.PI * (t * (1 - 2 * envPad) + envPad)), 2
    );
    const noise = noiseFloor + seedRand(seed * 13 + i * 3.1) * (1 - noiseFloor);
    out[i] = (env * noise * 100).toFixed(1) + '%';
  }
  return out;
};

const Wave = ({ seed, className }) => {
  const heights = useMemo(() => makeWave({ seed }), [seed]);
  return (
    <div className={className}>
      {heights.map((h, i) => (<i key={i} style={{ height: h }} />))}
    </div>
  );
};

// A session card in the Jump Back In rail. Backed by real session data from
// dashboardService (local + cloud dedup), clicking anywhere loads the session.
const SessionCard = ({ session, onLoad }) => {
  const seed = useMemo(() => stringHash(session.name), [session.name]);
  const swatch = TRACK_SWATCHES[seed % TRACK_SWATCHES.length];
  const tracks = session.trackCount || 0;
  const meta = `${session.daw || 'Doseedo'} · ${tracks} track${tracks === 1 ? '' : 's'} · ${session.time || '—'}`;

  return (
    <div
      className={styles.sessCard}
      style={{ '--swatch': swatch }}
      onClick={() => onLoad(session.name)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') onLoad(session.name); }}
    >
      <div className={styles.sessCover}>
        <div className={styles.sessSwatch} />
        <div className={styles.sessMark}>DSD</div>
        <Wave seed={seed} className={styles.sessWave} />
      </div>
      <div className={styles.sessBody}>
        <div className={styles.sessName}>{session.name}</div>
        <div className={styles.sessMeta}>{meta}</div>
      </div>
      <div className={styles.sessActions}>
        <button
          className={`${styles.chip} ${styles.chipPrimary}`}
          onClick={(e) => { e.stopPropagation(); onLoad(session.name); }}
        >
          <svg width="7" height="7" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4l13 8-13 8z" /></svg>
          Resume
        </button>
      </div>
    </div>
  );
};

/**
 * Dashboard — workbench home.
 *
 * Top:    Jump Back In rail (new) — themed session cards sourced from
 *         dashboardService.getRecentSessions() (local + cloud, deduped).
 * Below:  My Sessions table (unchanged) — full project list with the original
 *         create / load / delete / rename handlers.
 */
const Dashboard = ({ onCreateNew, onLoadProject }) => {
  const { state, dispatch } = useApp();
  const [projects, setProjects] = useState([]);
  const [recent, setRecent] = useState([]);
  const [newProjectName, setNewProjectName] = useState('');

  useEffect(() => {
    refreshProjects();
    loadRecent();
  }, []);

  const refreshProjects = useCallback(() => {
    setProjects(sessionService.getProjects());
  }, []);

  const loadRecent = useCallback(async () => {
    try {
      const list = await dashboardService.getRecentSessions();
      setRecent(list);
    } catch {
      setRecent([]);
    }
  }, []);

  const handleCreateNew = useCallback((e) => {
    if (e && e.preventDefault) e.preventDefault();

    // Auto-generate unique project name like legacy dashboard
    let baseName = newProjectName.trim() || 'Untitled';
    let projectName = baseName;
    let counter = 2;

    if (!newProjectName.trim()) {
      projectName = 'Untitled';
      while (projects.includes(projectName)) {
        projectName = `Untitled (${counter})`;
        counter++;
      }
    } else {
      while (projects.includes(projectName)) {
        projectName = `${baseName} (${counter})`;
        counter++;
      }
    }

    console.log(`🆕 Creating project: ${projectName}`);
    dispatch({ type: 'RESET_SESSION', payload: { projectName } });
    sessionService.setActiveProject(projectName);
    sessionService.saveSession(projectName, { ...state, projectName, buses: [] });
    refreshProjects();
    loadRecent();
    setNewProjectName('');
    if (onCreateNew) onCreateNew(projectName);
  }, [newProjectName, projects, state, dispatch, refreshProjects, loadRecent, onCreateNew]);

  const handleLoadProject = useCallback((projectName) => {
    const sessionData = sessionService.loadSession(projectName);
    if (!sessionData) {
      alert('Could not load project');
      return;
    }
    sessionService.setActiveProject(projectName);
    dispatch({ type: 'LOAD_SESSION', payload: sessionData.state });
    if (onLoadProject) onLoadProject(projectName);
  }, [dispatch, onLoadProject]);

  const handleDeleteProject = useCallback((projectName) => {
    if (window.confirm(`Delete project "${projectName}"? This cannot be undone.`)) {
      sessionService.deleteProject(projectName);
      refreshProjects();
      loadRecent();
    }
  }, [refreshProjects, loadRecent]);

  const handleRenameProject = useCallback((oldName, newName) => {
    if (!newName || newName.trim() === '') {
      alert('Project name cannot be empty');
      return;
    }
    const trimmedName = newName.trim();
    if (trimmedName === oldName) return;
    if (projects.includes(trimmedName)) {
      alert('A project with this name already exists');
      return;
    }
    sessionService.renameProject(oldName, trimmedName);
    refreshProjects();
    loadRecent();
  }, [projects, refreshProjects, loadRecent]);

  return (
    <div className={styles.dashboard}>

      {/* ============ Jump Back In rail ============ */}
      <section className={styles.sect}>
        <div className={styles.sectHead}>
          <h2 className={styles.sectTitle}>Jump Back In</h2>
          <span className={styles.sectCount}>last touched</span>
          <div className={styles.sectSpacer} />
        </div>

        <div className={styles.jbi}>
          <button
            className={`${styles.sessCard} ${styles.newCard}`}
            onClick={handleCreateNew}
            type="button"
          >
            <div className={styles.newCardPlus}>＋</div>
            <div className={styles.newCardTitle}>Start a new session</div>
            <div className={styles.newCardSub}>blank · or from a memo</div>
          </button>

          {recent.map((s) => (
            <SessionCard key={s.id || s.name} session={s} onLoad={handleLoadProject} />
          ))}
        </div>
      </section>

      {/* ============ My Sessions table (unchanged) ============ */}
      <div className={styles.header}>
        <h1 className={styles.title}>My Sessions</h1>
        <button onClick={handleCreateNew} className={styles.createBtn}>
          <i className="fa-solid fa-plus"></i>
          <span>New Session</span>
        </button>
      </div>

      {newProjectName !== null && (
        <div className={styles.sessionInputContainer}>
          <input
            type="text"
            placeholder="Session name (optional)"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleCreateNew(e);
              if (e.key === 'Escape') setNewProjectName('');
            }}
            className={styles.sessionInput}
            autoFocus
          />
        </div>
      )}

      <div className={styles.sessionsContainer}>
        {projects.length === 0 ? (
          <div className={styles.emptyState}>
            <i className="fa-solid fa-music"></i>
            <h3>No sessions yet</h3>
            <p>Create your first session to get started</p>
          </div>
        ) : (
          <div className={styles.sessionsList}>
            <div className={styles.sessionsHeader}>
              <div className={styles.headerNumber}>#</div>
              <div className={styles.headerTitle}>Title</div>
              <div className={styles.headerTracks}>Tracks</div>
              <div className={styles.headerDate}>Date Modified</div>
              <div className={styles.headerDuration}>Duration</div>
            </div>

            {projects.map((projectName, index) => (
              <ProjectCard
                key={projectName}
                projectName={projectName}
                index={index + 1}
                onLoad={handleLoadProject}
                onDelete={handleDeleteProject}
                onRename={handleRenameProject}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
