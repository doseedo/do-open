import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import * as videoAPI from '../../services/videoAPI';
import * as saveService from '../../services/saveService';
import * as sessionService from '../../services/sessionService';
import './Navbar.css';

function Navbar() {
  const { state, dispatch } = useApp();
  const [isExporting, setIsExporting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(saveService.SaveStatus.SAVED);
  const [lastSaveTime, setLastSaveTime] = useState(null);

  // Subscribe to save status changes
  useEffect(() => {
    const unsubscribe = saveService.onSaveStatusChange((status, localTime, cloudTime) => {
      setSaveStatus(status);
      setLastSaveTime(cloudTime || localTime);
      setIsSaving(status === saveService.SaveStatus.SAVING);
    });

    return unsubscribe;
  }, []);

  // Keyboard shortcut for save (Cmd+S / Ctrl+S)
  useEffect(() => {
    const handleKeyDown = (e) => {
      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const cmdOrCtrl = isMac ? e.metaKey : e.ctrlKey;

      // Cmd/Ctrl + S for save
      if (cmdOrCtrl && e.key === 's') {
        e.preventDefault();
        handleSaveProject(e);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleNewProject = (e) => {
    e.preventDefault();
    if (window.confirm('Create new project? Unsaved changes will be lost.')) {
      dispatch({ type: 'RESET_STATE' });
      dispatch({ type: 'SET_PROJECT_NAME', payload: 'Untitled Session' });
    }
  };

  const handleSaveProject = async (e) => {
    e.preventDefault();

    const projectName = state.projectName || 'Untitled Session';

    try {
      setIsSaving(true);

      // Use quick save (local + background cloud)
      const result = await saveService.quickSave(projectName, state);

      if (result.success) {
        // Show success notification
        const notification = document.createElement('div');
        notification.className = 'save-notification';
        notification.innerHTML = `
          <i class="fa-solid fa-check-circle"></i>
          <span>Saved: ${projectName}</span>
        `;
        notification.style.cssText = `
          position: fixed;
          top: 80px;
          right: 20px;
          background: rgba(34, 197, 94, 0.9);
          color: white;
          padding: 12px 20px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 14px;
          font-weight: 500;
          z-index: 10000;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
          animation: slideIn 0.3s ease;
        `;
        document.body.appendChild(notification);

        setTimeout(() => {
          notification.style.animation = 'slideOut 0.3s ease';
          setTimeout(() => document.body.removeChild(notification), 300);
        }, 2000);

        console.log('✅ Project saved successfully');
      } else {
        throw new Error(result.error || 'Save failed');
      }
    } catch (error) {
      console.error('Save error:', error);
      alert(`Failed to save project: ${error.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleOpenProject = (e) => {
    e.preventDefault();
    // Navigate to projects page
    window.location.href = '/projects';
  };

  const handleExportAudio = (e) => {
    e.preventDefault();
    // TODO: Implement audio export
    console.log('Export audio');
    alert('Export audio functionality to be implemented');
  };

  const handleExportAudioToVideo = async (e) => {
    e.preventDefault();

    // Check if user has uploaded a video
    if (!state.video.videoFile) {
      alert('Please upload a video file first. Use the Video Upload panel to upload a video.');
      return;
    }

    if (isExporting) {
      alert('Export already in progress...');
      return;
    }

    try {
      setIsExporting(true);

      // Create file input for audio file selection
      const audioInput = document.createElement('input');
      audioInput.type = 'file';
      audioInput.accept = 'audio/*,.wav,.mp3,.m4a,.aac';

      audioInput.onchange = async (event) => {
        const audioFile = event.target.files[0];

        if (!audioFile) {
          setIsExporting(false);
          return;
        }

        try {
          console.log('📤 Starting export with video:', state.video.fileName, 'and audio:', audioFile.name);

          // Upload to backend for processing
          const result = await videoAPI.exportAudioToVideo(state.video.videoFile, audioFile);
          const { task_id } = result;

          console.log('⏳ Waiting for export to complete...');

          // Poll for completion
          const exportResult = await videoAPI.pollExportUntilComplete(
            task_id,
            (progress) => {
              console.log(`Export progress: attempt ${progress.attempts}, status: ${progress.status}`);
            }
          );

          console.log('✅ Export complete!', exportResult);

          // Download the result
          if (exportResult.output_video) {
            // The backend returns a file path, we need to construct the download URL
            // Based on main.py, the file is in TEMP_EXPORT_DIR and returned via FileResponse
            const downloadUrl = exportResult.output_video;

            // Trigger download
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = 'video_with_audio.mp4';
            link.click();

            alert('Export complete! Download should start automatically.');
          } else {
            alert('Export completed but no output file path was returned.');
          }

        } catch (error) {
          console.error('Export error:', error);
          alert('Export failed: ' + error.message);
        } finally {
          setIsExporting(false);
        }
      };

      audioInput.click();

    } catch (error) {
      console.error('Export audio to video error:', error);
      alert('Failed to export audio to video: ' + error.message);
      setIsExporting(false);
    }
  };

  return (
    <div id="navbar">
      <div className="filedropdown">
        <button className="dropbtn">
          File <i className="fa-solid fa-angle-down"></i>
        </button>
        <div className="file-content">
          <a href="#" onClick={handleNewProject}>New Project</a>
          <a href="#" onClick={handleOpenProject}>Open</a>
          <a href="#" onClick={handleSaveProject}>Save</a>
          <div className="submenu-item">
            <a href="#">
              Export <i className="fa-solid fa-angle-right"></i>
            </a>
            <div className="submenu-content">
              <a href="#" onClick={handleExportAudioToVideo}>Export Audio to Video</a>
              <a href="#" onClick={handleExportAudio}>Export Audio</a>
            </div>
          </div>
        </div>
      </div>

      <div id="project-name-display">
        <i className="fa-solid fa-folder-open"></i>
        <span id="current-project-name">{state.projectName}</span>
        {/* Save Status Indicator */}
        {isSaving && (
          <span className="save-status saving">
            <i className="fa-solid fa-spinner fa-spin"></i>
            Saving...
          </span>
        )}
        {!isSaving && saveStatus === saveService.SaveStatus.SAVED && lastSaveTime && (
          <span className="save-status saved">
            <i className="fa-solid fa-cloud-check"></i>
            Saved
          </span>
        )}
        {saveStatus === saveService.SaveStatus.UNSAVED && (
          <span className="save-status unsaved">
            <i className="fa-solid fa-circle"></i>
            Unsaved changes
          </span>
        )}
        {saveStatus === saveService.SaveStatus.ERROR && (
          <span className="save-status error">
            <i className="fa-solid fa-exclamation-circle"></i>
            Save failed
          </span>
        )}
      </div>

      <button className="nav-btn" title="Undo">
        <i className="fa-solid fa-undo"></i>
      </button>

      <button className="nav-btn" title="Redo">
        <i className="fa-solid fa-redo"></i>
      </button>

      <button
        className={`nav-btn ${state.pluginMode ? 'active' : ''}`}
        title="Toggle Plugin Mode"
        onClick={() => dispatch({ type: 'TOGGLE_PLUGIN_MODE' })}
        style={{
          backgroundColor: state.pluginMode ? 'rgba(102, 126, 234, 0.3)' : 'transparent',
          border: state.pluginMode ? '1px solid rgba(102, 126, 234, 0.5)' : '1px solid transparent'
        }}
      >
        <i className="fa-solid fa-plug"></i>
      </button>

      <div id="sessiondiv">
        <p id="session">{state.projectName}</p>
      </div>
    </div>
  );
}

export default Navbar;
