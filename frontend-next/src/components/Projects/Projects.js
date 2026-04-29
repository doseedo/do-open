import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import * as sessionService from '../../services/sessionService';
import * as sessionSyncAPI from '../../services/sessionSyncAPI';
import ProjectCard from './ProjectCard';
import styles from './Projects.module.css';
import PageTopbar from '../Sidebar/PageTopbar';
import PageEyebrow from '../Sidebar/PageEyebrow';

const CHAT_STORAGE_PREFIX = 'chat-';
const DAW_TYPES = new Set(['logic', 'ableton', 'fl', 'protools', 'desktop', 'daw']);

// Mirrors StudioDevChat's persistence key. Returns the message count
// (0 if no thread, or if the entry is malformed).
function _chatMessageCount(sessionId) {
  if (!sessionId) return 0;
  try {
    const raw = localStorage.getItem(`${CHAT_STORAGE_PREFIX}${sessionId}`);
    if (!raw) return 0;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.length : 0;
  } catch { return 0; }
}

// Classify a server-synced session into a source bucket. The type
// column came from the desktop bridge; visibility "private" stays sync/
// web, anything else (unlisted/public) is treated as collab so shareable
// sessions surface together.
function _classifySynced(s) {
  if (s.visibility && s.visibility !== 'private') return 'collab';
  if (s.type && DAW_TYPES.has(String(s.type).toLowerCase())) return 'sync';
  return 'web';
}

const SOURCE_LABEL = {
  local:  'Local',
  sync:   'Sync',
  web:    'Web',
  collab: 'Collab',
};

/**
 * Projects — flat sessions table.
 *
 * Local + server-synced sessions are merged into a single list sorted
 * by last-modified date (newest first). A source-filter dropdown +
 * name search lets the user narrow without losing chronology — fixes
 * the prior bug where the SYNC section always rendered above local
 * regardless of date.
 */
const Projects = ({ onCreateNew, onLoadProject }) => {
  const { state, dispatch } = useApp();
  const navigate = useNavigate();
  const [projects, setProjects] = useState([]);              // local project names
  const [syncedSessions, setSyncedSessions] = useState([]);  // /api/sessions
  const [syncedError, setSyncedError] = useState(null);
  const [newProjectName, setNewProjectName] = useState('');

  // Filter state
  const [sourceFilter, setSourceFilter] = useState('all');   // all | local | sync | web | collab
  const [nameQuery, setNameQuery] = useState('');

  useEffect(() => { refreshProjects(); }, []);

  const refreshProjects = useCallback(() => {
    setProjects(sessionService.getProjects());
    sessionSyncAPI.listMySessions({ limit: 100 })
      .then((res) => setSyncedSessions(Array.isArray(res?.items) ? res.items : []))
      .catch((err) => {
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

  // Build the unified, source-tagged, date-sorted list. Local sessions
  // need a localStorage hit per name to read the timestamp; we cap that
  // by memoizing on `projects` + `syncedSessions`.
  const unifiedItems = useMemo(() => {
    const localItems = projects.map((name) => {
      const s = sessionService.loadSession(name);
      const ts = s?.timestamp ? new Date(s.timestamp) : new Date(0);
      return {
        key: `local:${name}`,
        source: 'local',
        name,
        updatedAt: ts,
        // Carry the local-specific payload through to ProjectCard.
        local: { name },
      };
    });

    const remoteItems = syncedSessions.map((s) => {
      const ts = s.updated_at ? new Date(s.updated_at)
              : s.created_at ? new Date(s.created_at)
              : new Date(0);
      return {
        key: `sync:${s.id}`,
        source: _classifySynced(s),
        name: s.name || s.id?.slice(0, 8) || 'Untitled',
        updatedAt: ts,
        sync: s,
      };
    });

    return [...localItems, ...remoteItems].sort(
      (a, b) => b.updatedAt - a.updatedAt
    );
  }, [projects, syncedSessions]);

  // Apply filter + name search.
  const visibleItems = useMemo(() => {
    const q = nameQuery.trim().toLowerCase();
    return unifiedItems.filter((it) => {
      if (sourceFilter !== 'all' && it.source !== sourceFilter) return false;
      if (q && !it.name.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [unifiedItems, sourceFilter, nameQuery]);

  const sourceCounts = useMemo(() => {
    const c = { all: unifiedItems.length, local: 0, sync: 0, web: 0, collab: 0 };
    for (const it of unifiedItems) c[it.source] = (c[it.source] || 0) + 1;
    return c;
  }, [unifiedItems]);

  const renderSyncedRow = (it, index) => {
    const s = it.sync;
    const messageCount = _chatMessageCount(s.id);
    const updated = it.updatedAt.getTime() ? it.updatedAt.toLocaleDateString() : '—';
    const open = () => {
      navigate(`/studio?session=${encodeURIComponent(s.id)}`);
      if (onLoadProject) onLoadProject(s.name || s.id);
    };
    const iconClass = it.source === 'collab' ? 'fa-solid fa-users'
                    : it.source === 'sync'   ? 'fa-solid fa-cloud-arrow-up'
                    :                          'fa-solid fa-cloud';
    return (
      <div
        key={it.key}
        className={styles.sessionRow}
        onClick={open}
        style={{ cursor: 'pointer' }}
      >
        <div className={styles.sessionNumber}>
          <i className={iconClass} title={SOURCE_LABEL[it.source]} />
        </div>
        <div className={styles.sessionTitle}>
          <span className={styles.sessionName}>{it.name}</span>
          <span className={styles.sourceBadge} data-source={it.source}>{SOURCE_LABEL[it.source]}</span>
        </div>
        <div className={styles.sessionTracks}>{s.type || 'project'}</div>
        <div className={styles.sessionDate}>{updated}</div>
        <div className={styles.sessionDuration}>
          {messageCount > 0 ? `💬 ${messageCount}` : '—'}
        </div>
      </div>
    );
  };

  return (
    <div className={styles.projects}>
      <PageTopbar section="Create" title="Projects" />
      <PageEyebrow section="Projects" description="Sessions in your library" />
      <div className={styles.header}>
        <h1 className={`${styles.title} page-title`}>Projects</h1>
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

      {/* Filters */}
      <div className={styles.filterBar}>
        <input
          type="search"
          placeholder="Search by name…"
          value={nameQuery}
          onChange={(e) => setNameQuery(e.target.value)}
          className={styles.filterSearch}
          aria-label="Search projects by name"
        />
        <select
          value={sourceFilter}
          onChange={(e) => setSourceFilter(e.target.value)}
          className={styles.filterSelect}
          aria-label="Filter projects by source"
        >
          <option value="all">All sources ({sourceCounts.all})</option>
          <option value="local">Local ({sourceCounts.local})</option>
          <option value="sync">Sync ({sourceCounts.sync})</option>
          <option value="web">Web ({sourceCounts.web})</option>
          <option value="collab">Collab ({sourceCounts.collab})</option>
        </select>
        {syncedError && (
          <span className={styles.filterError} title={syncedError}>
            sync unavailable
          </span>
        )}
      </div>

      <div className={styles.sessionsContainer}>
        {visibleItems.length === 0 ? (
          <div className={styles.emptyState}>
            <i className="fa-solid fa-music"></i>
            <h3>{unifiedItems.length === 0 ? 'No sessions yet' : 'No matches'}</h3>
            <p>
              {unifiedItems.length === 0
                ? 'Create your first session to get started'
                : 'Try a different filter or search term'}
            </p>
          </div>
        ) : (
          <div className={styles.sessionsList}>
            <div className={styles.sessionsHeader}>
              <div className={styles.headerNumber}>#</div>
              <div className={styles.headerTitle}>Title</div>
              <div className={styles.headerTracks}>Type</div>
              <div className={styles.headerDate}>Date Modified</div>
              <div className={styles.headerDuration}>Chat</div>
            </div>

            {visibleItems.map((it, index) =>
              it.source === 'local' ? (
                <ProjectCard
                  key={it.key}
                  projectName={it.local.name}
                  index={index + 1}
                  onLoad={handleLoadProject}
                  onDelete={handleDeleteProject}
                  onRename={handleRenameProject}
                />
              ) : (
                renderSyncedRow(it, index)
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Projects;
