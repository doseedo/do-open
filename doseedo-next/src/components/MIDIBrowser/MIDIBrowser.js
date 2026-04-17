import React, { useState, useEffect, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { parseMIDIFile } from '../../utils/midiParser';
import styles from './MIDIBrowser.module.css';

/**
 * MIDIBrowser - Browse and search MIDI files from the server
 */
const MIDIBrowser = ({ onClose }) => {
  const { state, dispatch } = useApp();
  const [midiFiles, setMidiFiles] = useState([]);
  const [filteredFiles, setFilteredFiles] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load MIDI files on mount
  useEffect(() => {
    loadMidiFiles();
  }, []);

  // Filter files when search term changes
  useEffect(() => {
    if (searchTerm.trim() === '') {
      setFilteredFiles(midiFiles);
    } else {
      const term = searchTerm.toLowerCase();
      setFilteredFiles(midiFiles.filter(file => file.toLowerCase().includes(term)));
    }
  }, [searchTerm, midiFiles]);

  const loadMidiFiles = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/list-midi-files');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setMidiFiles(data.files || []);
      setFilteredFiles(data.files || []);
    } catch (err) {
      console.error('Error loading MIDI files:', err);
      setError('MIDI library not available. Please upload MIDI files directly using the Upload button.');
      setMidiFiles([]);
      setFilteredFiles([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileClick = useCallback((filename) => {
    setSelectedFile(filename);
  }, []);

  const handleFileDoubleClick = useCallback(async (filename) => {
    try {
      // Fetch the MIDI file
      const response = await fetch(`/api/get-midi-file/${encodeURIComponent(filename)}`);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: 'audio/midi' });

      // Parse MIDI data
      const midiData = await parseMIDIFile(file);

      // Set as uploaded file in context
      dispatch({
        type: 'SET_UPLOADED_FILE',
        payload: {
          file,
          fileType: 'midi',
          previewUrl: null
        }
      });

      // Find Music bus or use a consistent ID for auto-creation
      const musicBus = state.buses.find(bus => bus.type.toLowerCase() === 'music');
      const busId = musicBus ? musicBus.id : `music-${Date.now()}`;

      const baseName = filename.replace('.mid', '').replace('.midi', '');

      // Filter tracks that have notes
      const tracksWithNotes = midiData.tracks.filter(track => track.notes && track.notes.length > 0);

      if (tracksWithNotes.length > 1) {
        // Multitrack MIDI - create one track per MIDI track
        console.log(`🎹 Creating ${tracksWithNotes.length} tracks for multitrack MIDI file`);

        tracksWithNotes.forEach((midiTrack, index) => {
          const trackId = `track-${Date.now()}-${index}`;
          const newTrack = {
            id: trackId,
            name: `${baseName} - Track ${index + 1}`,
            type: 'midi',
            midiData: {
              ...midiData,
              notes: midiTrack.notes, // Only this track's notes
              trackIndex: index, // Store which track this is
              isMultitrack: true,
              allTracks: midiData.tracks // Keep reference to all tracks for composite view
            },
            file: file,
            duration: midiData.duration,
            startPosition: 0,
            color: `hsl(${(index * 360 / tracksWithNotes.length)}, 70%, 60%)`,
            solo: false,
            mute: false,
            volume: 1.0,
            pan: 0,
            cropStart: 0,
            cropEnd: 0
          };

          dispatch({
            type: 'ADD_TRACK',
            payload: {
              busId: busId,
              track: newTrack
            }
          });
        });

        console.log(`✅ MIDI file loaded with ${tracksWithNotes.length} tracks:`, filename);
      } else {
        // Single track MIDI - create one track
        const trackId = `track-${Date.now()}`;
        const newTrack = {
          id: trackId,
          name: baseName,
          type: 'midi',
          midiData: midiData,
          file: file,
          duration: midiData.duration,
          startPosition: 0,
          color: `hsl(${Math.random() * 360}, 70%, 60%)`,
          solo: false,
          mute: false,
          volume: 1.0,
          pan: 0,
          cropStart: 0,
          cropEnd: 0
        };

        dispatch({
          type: 'ADD_TRACK',
          payload: {
            busId: busId,
            track: newTrack
          }
        });

        console.log('✅ MIDI file loaded and track created:', filename);
      }

      // Close the browser and return to generation panel
      if (onClose) {
        onClose();
      }
    } catch (error) {
      console.error('Error loading MIDI file:', error);
      alert(`Failed to load MIDI file: ${error.message}`);
    }
  }, [dispatch, onClose]);

  const handleUseSelected = useCallback(async () => {
    if (selectedFile) {
      await handleFileDoubleClick(selectedFile);
    }
  }, [selectedFile, handleFileDoubleClick]);

  return (
    <div className={styles.browserContainer}>
      {/* Header */}
      <div className={styles.header}>
        <h4 className={styles.title}>
          <i className="fa-solid fa-folder-music"></i> MIDI File Browser
        </h4>
        <i className={`fa-solid fa-gear ${styles.settingsIcon}`}></i>
      </div>

      {/* Search Input */}
      <div className={styles.searchSection}>
        <h5 className={styles.sectionTitle}>Search MIDI Files</h5>
        <input
          type="text"
          className={styles.searchInput}
          placeholder="Search MIDI files..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {/* File List */}
      <div className={styles.fileListContainer}>
        {isLoading && (
          <p className={styles.message}>
            <i className="fa-solid fa-spinner fa-spin"></i> Loading MIDI files...
          </p>
        )}

        {error && (
          <p className={styles.error}>
            <i className="fa-solid fa-exclamation-triangle"></i> {error}
          </p>
        )}

        {!isLoading && !error && filteredFiles.length === 0 && (
          <p className={styles.message}>No MIDI files found</p>
        )}

        {!isLoading && !error && filteredFiles.length > 0 && (
          <div className={styles.fileList}>
            {filteredFiles.map((file, index) => (
              <div
                key={index}
                className={`${styles.fileItem} ${selectedFile === file ? styles.selected : ''}`}
                onClick={() => handleFileClick(file)}
                onDoubleClick={() => handleFileDoubleClick(file)}
              >
                <i className="fa-solid fa-file-audio"></i>
                <span>{file}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Selected File Info */}
      {selectedFile && (
        <div className={styles.selectedInfo}>
          <span className={styles.selectedLabel}>Selected:</span>
          <span className={styles.selectedName}>{selectedFile}</span>
        </div>
      )}

      {/* Use Selected Button */}
      {selectedFile && (
        <button
          className={styles.useButton}
          onClick={handleUseSelected}
        >
          <i className="fa-solid fa-upload"></i> Use Selected MIDI
        </button>
      )}

      {/* Instructions */}
      <div className={styles.instructions}>
        <p>
          <i className="fa-solid fa-info-circle"></i>
          Click to select, double-click to load
        </p>
      </div>
    </div>
  );
};

export default MIDIBrowser;
