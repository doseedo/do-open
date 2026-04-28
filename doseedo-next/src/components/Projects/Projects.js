import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import * as sessionService from '../../services/sessionService';
import * as sessionSyncAPI from '../../services/sessionSyncAPI';
import ProjectCard from './ProjectCard';
import styles from './Projects.module.css';

const CHAT_STORAGE_PREFIX = 'chat-';

// Mirrors StudioDevChat's persistence key. Returns the message count
// (0 if no thread, or if the entry is malformed). Used to render the
// 💬 badge on Projects rows so the user can see at-a-glance which
// sessions already have a chat going.
function _chatMessageCount(sessionId) {
  if (!sessionId) return 0;
  try {
    const raw = localStorage.getItem(`${CHAT_STORAGE_PREFIX}${sessionId}`);
    if (!raw) return 0;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.length : 0;
  } catch { return 0; }
}

/**
 * Projects — flat sessions table.
 *
 * Distinct from /dashboard (which is the workbench home: slideshow +
 * Jump Back In + community feeds). /projects is the place to audit
 * every session with its track count, last-modified date, and access
 * to rename/delete. Uses the existing ProjectCard row component so the
 * wiring (save, rename, delete, load) is unchanged from the legacy
 * Dashboard table.
 */
const Projects = ({ onCreateNew, onLoadProject }) => {
  const { state, dispatch } = useApp();
  const navigate = useNavigate();
  const [projects, setProjects] = useState([]);
  const [syncedSessions, setSyncedSessions] = useState([]);    // server rows from /api/sessions
  const [syncedError, setSyncedError] = useState(null);
  const [newProjectName, setNewProjectName] = useState('');

  useEffect(() => { refreshProjects(); }, []);

  const refreshProjects = useCallback(() => {
    setProjects(sessionService.getProjects());
    // Refresh the server-synced list in parallel — same blend desktop's
    // chat_server.py:1442-1522 does (local DAW projects + remote sessions).
    sessionSyncAPI.listMySessions({ limit: 100 })
      .then((res) => setSyncedSessions(Array.isArray(res?.items) ? res.items : []))
      .catch((err) => {
        // Silent on auth-not-ready (Clerk still loading); surface anything else.
        if (err?.status !== 401 && err?.status !== 403) {
          setSyncedError(err?.message || String(err));
        }
        setSyncedSessions([]);
      });
  }, []);

  const handleCreateNew = useCallback((e) => {
    if (e && e.preventDefault) e.preventDefault();
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

    dispatch({ type: 'RESET_SESSION', payload: { projectName } });
    sessionService.setActiveProject(projectName);
    sessionService.saveSession(projectName, { ...state, projectName, buses: [] });
    refreshProjects();
    setNewProjectName('');
    if (onCreateNew) onCreateNew(projectName);
  }, [newProjectName, projects, state, dispatch, refreshProjects, onCreateNew]);

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
    }
  }, [refreshProjects]);

  const handleRenameProject = useCallback((oldName, newName) => {
    if (!newName || newName.trim() === '') {
      alert('Project name cannot be empty');
      return;
    }
    const trimmed = newName.trim();
    if (trimmed === oldName) return;
    if (projects.includes(trimmed)) {
      alert('A project with this name already exists');
      return;
    }
    sessionService.renameProject(oldName, trimmed);
    refreshProjects();
  }, [projects, refreshProjects]);

  return (
    <div className={styles.projects}>
      <div className={styles.header}>
        <h1 className={styles.title}>Projects</h1>
        <button onClick={handleCreateNew} className={styles.createBtn}>
          <i className="fa-solid fa-plus"></i>
          <span>New Session</span>
        </button>
      </div>

      <div className={styles.sessionInputContainer}>
        <input
          type="text"
          placeholder="Session name (optional) — press Enter to create"
          value={newProjectName}
          onChange={(e) => setNewProjectName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleCreateNew(e);
            if (e.key === 'Escape') setNewProjectName('');
          }}
          className={styles.sessionInput}
        />
      </div>

      <div className={styles.sessionsContainer}>
        {/* Synced section — server-side sessions from /api/sessions.
            These are the projects the desktop app pushes (Logic / FL /
            Ableton / web sessions bootstrapped via createSession).
            Click to open in the studio with `?session=<uuid>`; if a
            chat thread exists in localStorage for that uuid, show the
            💬 badge with the message count. True web↔desktop chat sync
            requires Phase B (auth-service chat_messages endpoint). */}
        {(syncedSessions.length > 0 || syncedError) && (
          <div className={styles.sessionsList} style={{ marginBottom: 24 }}>
            <div className={styles.sessionsHeader}>
              <div className={styles.headerNumber}>SYNC</div>
              <div className={styles.headerTitle}>Title</div>
              <div className={styles.headerTracks}>Type</div>
              <div className={styles.headerDate}>Updated</div>
              <div className={styles.headerDuration}>Chat</div>
            </div>
            {syncedError && (
              <div className={styles.emptyState} style={{ padding: '16px 0', fontSize: 12, opacity: 0.7 }}>
                Couldn't load synced sessions: {syncedError}
              </div>
            )}
            {syncedSessions.map((s) => {
              const messageCount = _chatMessageCount(s.id);
              const updated = s.updated_at ? new Date(s.updated_at).toLocaleDateString() : '—';
              const open = () => {
                navigate(`/studio?session=${encodeURIComponent(s.id)}`);
                if (onLoadProject) onLoadProject(s.name || s.id);
              };
              return (
                <div
                  key={s.id}
                  className={styles.sessionRow}
                  onClick={open}
                  style={{ cursor: 'pointer' }}
                >
                  <div className={styles.sessionNumber}>
                    <i className="fa-solid fa-cloud" title="Server-synced session" />
                  </div>
                  <div className={styles.sessionTitle}>
                    <span className={styles.sessionName}>{s.name || s.id.slice(0, 8)}</span>
                  </div>
                  <div className={styles.sessionTracks}>{s.type || 'project'}</div>
                  <div className={styles.sessionDate}>{updated}</div>
                  <div className={styles.sessionDuration}>
                    {messageCount > 0 ? `💬 ${messageCount}` : '—'}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {projects.length === 0 && syncedSessions.length === 0 ? (
          <div className={styles.emptyState}>
            <i className="fa-solid fa-music"></i>
            <h3>No sessions yet</h3>
            <p>Create your first session to get started</p>
          </div>
        ) : projects.length > 0 && (
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

export default Projects;
