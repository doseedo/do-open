import React, { useState, useEffect, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import * as sessionService from '../../services/sessionService';
import ProjectCard from './ProjectCard';
import styles from './Dashboard.module.css';

/**
 * Dashboard Component
 * Shows list of saved projects, allows creating new sessions
 */
const Dashboard = ({ onCreateNew, onLoadProject }) => {
  const { state, dispatch } = useApp();
  const [projects, setProjects] = useState([]);
  const [newProjectName, setNewProjectName] = useState('');

  // Load projects on mount
  useEffect(() => {
    refreshProjects();
  }, []);

  const refreshProjects = useCallback(() => {
    const projectList = sessionService.getProjects();
    setProjects(projectList);
  }, []);

  const handleCreateNew = useCallback((e) => {
    e.preventDefault();

    // Auto-generate unique project name like legacy dashboard
    let baseName = newProjectName.trim() || 'Untitled';
    let projectName = baseName;
    let counter = 2;

    // If user didn't provide a name, use Untitled numbering
    if (!newProjectName.trim()) {
      projectName = 'Untitled';
      while (projects.includes(projectName)) {
        projectName = `Untitled (${counter})`;
        counter++;
      }
    } else {
      // User provided name - add number if it exists
      while (projects.includes(projectName)) {
        projectName = `${baseName} (${counter})`;
        counter++;
      }
    }

    console.log(`🆕 Creating project: ${projectName}`);

    // Reset to fresh session state
    dispatch({ type: 'RESET_SESSION', payload: { projectName } });

    // Set as active project
    sessionService.setActiveProject(projectName);

    // Save initial empty session
    sessionService.saveSession(projectName, {
      ...state,
      projectName,
      buses: []
    });

    // Refresh project list
    refreshProjects();

    // Clear input
    setNewProjectName('');

    // Notify parent (to switch to DAW view)
    if (onCreateNew) {
      onCreateNew(projectName);
    }
  }, [newProjectName, projects, state, dispatch, refreshProjects, onCreateNew]);

  const handleLoadProject = useCallback((projectName) => {
    const sessionData = sessionService.loadSession(projectName);

    if (!sessionData) {
      alert('Could not load project');
      return;
    }

    // Set as active project
    sessionService.setActiveProject(projectName);

    // Load state into app
    // We'll dispatch a LOAD_SESSION action to restore the full state
    dispatch({ type: 'LOAD_SESSION', payload: sessionData.state });

    // Notify parent to switch to DAW view
    if (onLoadProject) {
      onLoadProject(projectName);
    }
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

    const trimmedName = newName.trim();
    if (trimmedName === oldName) {
      return; // No change
    }

    // Check if name already exists
    if (projects.includes(trimmedName)) {
      alert('A project with this name already exists');
      return;
    }

    sessionService.renameProject(oldName, trimmedName);
    refreshProjects();
  }, [projects, refreshProjects]);

  return (
    <div className={styles.dashboard}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>My Sessions</h1>

        {/* Create New Session Button */}
        <button onClick={handleCreateNew} className={styles.createBtn}>
          <i className="fa-solid fa-plus"></i>
          <span>New Session</span>
        </button>
      </div>

      {/* Session Input (when creating) */}
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

      {/* Sessions List */}
      <div className={styles.sessionsContainer}>
        {projects.length === 0 ? (
          <div className={styles.emptyState}>
            <i className="fa-solid fa-music"></i>
            <h3>No sessions yet</h3>
            <p>Create your first session to get started</p>
          </div>
        ) : (
          <div className={styles.sessionsList}>
            {/* Table Header */}
            <div className={styles.sessionsHeader}>
              <div className={styles.headerNumber}>#</div>
              <div className={styles.headerTitle}>Title</div>
              <div className={styles.headerTracks}>Tracks</div>
              <div className={styles.headerDate}>Date Modified</div>
              <div className={styles.headerDuration}>Duration</div>
            </div>

            {/* Sessions */}
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
