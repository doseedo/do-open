import React, { useState, useEffect, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import * as sessionService from '../../services/sessionService';
import ProjectCard from '../Dashboard/ProjectCard';
import styles from './Projects.module.css';

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
  const [projects, setProjects] = useState([]);
  const [newProjectName, setNewProjectName] = useState('');

  useEffect(() => { refreshProjects(); }, []);

  const refreshProjects = useCallback(() => {
    setProjects(sessionService.getProjects());
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

export default Projects;
