/*
 * StudioDevFileMenu — themed File dropdown for /studio-dev.
 *
 * Wraps saveService the production Navbar uses. Items:
 *   • New Session         — RESET_STATE + fresh project name
 *   • Open Project        — navigate /projects
 *   • Save                — saveService.quickSave
 *   • Save As…            — prompt → quickSave with new name
 *   • Export Audio…       — alerts (original feature still stub)
 *   • Rename Project…     — prompt → SET_PROJECT_NAME
 */
import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import * as saveService from '../../services/saveService';

export default function StudioDevFileMenu() {
  const { state, dispatch } = useApp();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState('');
  const rootRef = useRef(null);

  useEffect(() => {
    const onDoc = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        save();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  const flash = (msg) => {
    setStatus(msg);
    setTimeout(() => setStatus(''), 1400);
  };

  const save = async (nameOverride) => {
    const projectName = nameOverride || state.projectName || 'Untitled Session';
    try {
      setSaving(true);
      const result = await saveService.quickSave(projectName, state);
      if (!result?.success) throw new Error(result?.error || 'save failed');
      if (nameOverride) dispatch({ type: 'SET_PROJECT_NAME', payload: nameOverride });
      flash(`Saved · ${projectName}`);
    } catch (e) {
      flash(`Save failed: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  const newSession = () => {
    if (!window.confirm('Create a new session? Unsaved changes will be lost.')) return;
    dispatch({ type: 'RESET_STATE' });
    dispatch({ type: 'SET_PROJECT_NAME', payload: 'Untitled Session' });
    setOpen(false);
    flash('New session');
  };

  const saveAs = async () => {
    const name = window.prompt('Save project as…', state.projectName || 'Untitled');
    if (!name) return;
    setOpen(false);
    await save(name.trim());
  };

  const rename = () => {
    const name = window.prompt('Rename project', state.projectName || 'Untitled');
    if (!name) return;
    dispatch({ type: 'SET_PROJECT_NAME', payload: name.trim() });
    setOpen(false);
    flash('Renamed');
  };

  const openProject = () => {
    setOpen(false);
    navigate('/projects');
  };

  const exportAudio = () => {
    setOpen(false);
    flash('Export: not implemented yet');
  };

  return (
    <div className="sd-filemenu" ref={rootRef}>
      <button
        className={`sd-filemenu-btn ${open ? 'open' : ''}`}
        onClick={() => setOpen((v) => !v)}
      >
        <i className="fa-solid fa-bars" style={{ fontSize: 11 }} />
        <span>File</span>
        <i className={`fa-solid fa-chevron-${open ? 'up' : 'down'}`} style={{ fontSize: 9, opacity: 0.6 }} />
      </button>
      {status && <div className="sd-filemenu-status">{status}</div>}
      {open && (
        <div className="sd-filemenu-dropdown" role="menu">
          <button className="sd-filemenu-item" onClick={newSession}>
            <i className="fa-solid fa-file-circle-plus" /> <span>New session</span>
            <span className="sd-filemenu-kbd">⌘N</span>
          </button>
          <button className="sd-filemenu-item" onClick={openProject}>
            <i className="fa-solid fa-folder-open" /> <span>Open project…</span>
            <span className="sd-filemenu-kbd">⌘O</span>
          </button>
          <div className="sd-filemenu-div" />
          <button className="sd-filemenu-item" onClick={() => { setOpen(false); save(); }} disabled={saving}>
            <i className="fa-solid fa-floppy-disk" />
            <span>{saving ? 'Saving…' : 'Save'}</span>
            <span className="sd-filemenu-kbd">⌘S</span>
          </button>
          <button className="sd-filemenu-item" onClick={saveAs} disabled={saving}>
            <i className="fa-solid fa-copy" /> <span>Save as…</span>
            <span className="sd-filemenu-kbd">⇧⌘S</span>
          </button>
          <button className="sd-filemenu-item" onClick={rename}>
            <i className="fa-solid fa-pen" /> <span>Rename project…</span>
          </button>
          <div className="sd-filemenu-div" />
          <button className="sd-filemenu-item" onClick={exportAudio}>
            <i className="fa-solid fa-arrow-up-from-bracket" /> <span>Export audio…</span>
          </button>
        </div>
      )}
    </div>
  );
}
