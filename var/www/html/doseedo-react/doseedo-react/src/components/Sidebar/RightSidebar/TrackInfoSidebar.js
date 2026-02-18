import React, { useCallback, useState, useEffect } from 'react';
import { useApp } from '../../../context/AppContext';
import * as generationAPI from '../../../services/generationAPI';
import GlassButtonWrapper from '../../GlassButton/GlassButtonWrapper';
import { logFeedback } from '../../../utils/feedbackLogger';
import styles from './TrackInfoSidebar.module.css';

/**
 * TrackInfoSidebar Component
 * Right sidebar showing selected track information and actions
 *
 * Features:
 * - Track info display (instrument group, subgroup, source file)
 * - Download track
 * - Regenerate track (for generated tracks)
 * - Inpaint selection (for generated tracks)
 * - Stem separation (for video audio tracks)
 * - Track FX (reverb, fade in/out)
 */
const TrackInfoSidebar = React.memo(() => {
  const { state, dispatch } = useApp();

  const isCollapsed = state.stemsSidebar?.isCollapsed ?? true;
  const selectedTrack = state.selectedTrack;
  const selectedBus = state.selectedBus;

  // Local UI state
  const [stemProcessing, setStemProcessing] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [clarifying, setClarifying] = useState(false);
  const [showFX, setShowFX] = useState(false);
  const [showParams, setShowParams] = useState(false);
  const [showPluginFX, setShowPluginFX] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(null); // 'like' or 'dislike'
  const [trumpetMuted, setTrumpetMuted] = useState(false); // For trumpet mute toggle
  const [muteProcessing, setMuteProcessing] = useState(false); // Loading state for mute FX

  // Sync trumpetMuted state with selected track's metadata
  useEffect(() => {
    if (selectedTrack?.metadata?.isMuted !== undefined) {
      setTrumpetMuted(selectedTrack.metadata.isMuted);
    } else {
      setTrumpetMuted(false);
    }
  }, [selectedTrack?.id, selectedTrack?.metadata?.isMuted]);

  // Plugin FX state
  const [rc20Enabled, setRc20Enabled] = useState(false);
  const [rc20Settings, setRc20Settings] = useState({
    magnitude: 50,
    noise: 20,
    wobble: 0,
    distortion: 0,
    magnitudeAmount: 30
  });

  const [speccraftEnabled, setSpeccraftEnabled] = useState(false);
  const [speccraftSettings, setSpeccraftSettings] = useState({
    threshold: -30,
    ratio: 65,
    attack: 3,
    release: 75,
    makeup: 0,
    knee: 0,
    slope: 8,
    // AI Vocal De-artifact bands
    band1Enabled: true,
    band1Freq: 400,
    band1Gain: 3,
    band1Q: 2.0,
    band2Enabled: true,
    band2Freq: 3000,
    band2Gain: -2,
    band2Q: 2.5,
    band3Enabled: true,
    band3Freq: 7000,
    band3Gain: -4,
    band3Q: 2.0,
    band4Enabled: true,
    band4Freq: 450,
    band4Gain: -6,
    band4Q: 3.0
  });

  // Determine track type based on metadata
  const isGeneratedTrack = selectedTrack?.metadata?.type === 'generated';
  const isVideoAudioTrack = selectedTrack?.metadata?.type === 'video_audio';
  const isUploadedTrack = selectedTrack?.metadata?.type === 'uploaded';

  // Get version history info
  const versions = selectedTrack?.metadata?.versions || [];
  const currentVersionIndex = selectedTrack?.metadata?.currentVersionIndex ?? 0;
  const hasMultipleVersions = versions.length > 1;

  // Check if there's a pending inpaint selection for this track
  const hasInpaintSelection = state.inpaintSelection && selectedTrack && state.inpaintSelection.trackId === selectedTrack.id;
  const inpaintSelectionTimes = hasInpaintSelection ? {
    start: state.inpaintSelection.startTime,
    end: state.inpaintSelection.endTime,
    duration: state.inpaintSelection.endTime - state.inpaintSelection.startTime
  } : null;

  const toggleSidebar = useCallback(() => {
    dispatch({ type: 'TOGGLE_STEMS_SIDEBAR' });
  }, [dispatch]);

  const handleDownload = useCallback(() => {
    if (!selectedTrack) return;

    // Create a link and trigger download
    const link = document.createElement('a');
    link.href = selectedTrack.audioUrl;
    link.download = selectedTrack.name || 'track.wav';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    console.log('📥 Downloading track:', selectedTrack.name);
  }, [selectedTrack]);

  // Download track with FX applied
  const handleDownloadWithFX = useCallback(async () => {
    if (!selectedTrack) return;

    try {
      console.log('🎛️ Downloading track with FX applied...');

      // Find which bus contains this track
      let trackBus = null;
      for (const bus of state.buses) {
        if (bus.tracks.find(t => t.id === selectedTrack.id)) {
          trackBus = bus;
          break;
        }
      }

      // Prepare RC-20 settings
      const rc20FX = rc20Enabled ? {
        enabled: true,
        magnitude: rc20Settings.magnitude,
        noise: rc20Settings.noise,
        wobble: rc20Settings.wobble,
        distortion: rc20Settings.distortion,
        magnitudeAmount: rc20Settings.magnitudeAmount
      } : { enabled: false };

      // Prepare SpecCraft settings
      const speccraftFX = speccraftEnabled ? {
        enabled: true,
        threshold: speccraftSettings.threshold,
        ratio: speccraftSettings.ratio,
        attack: speccraftSettings.attack,
        release: speccraftSettings.release,
        makeup: speccraftSettings.makeup,
        knee: speccraftSettings.knee,
        slope: speccraftSettings.slope,
        band1: {
          enabled: speccraftSettings.band1Enabled,
          freq: speccraftSettings.band1Freq,
          gain: speccraftSettings.band1Gain,
          q: speccraftSettings.band1Q
        },
        band2: {
          enabled: speccraftSettings.band2Enabled,
          freq: speccraftSettings.band2Freq,
          gain: speccraftSettings.band2Gain,
          q: speccraftSettings.band2Q
        },
        band3: {
          enabled: speccraftSettings.band3Enabled,
          freq: speccraftSettings.band3Freq,
          gain: speccraftSettings.band3Gain,
          q: speccraftSettings.band3Q
        },
        band4: {
          enabled: speccraftSettings.band4Enabled,
          freq: speccraftSettings.band4Freq,
          gain: speccraftSettings.band4Gain,
          q: speccraftSettings.band4Q
        }
      } : { enabled: false };

      // Call the API to process and download with FX
      const processedBlob = await generationAPI.downloadTrackWithFX(
        selectedTrack,
        trackBus,
        state.masterFX,
        rc20FX,
        speccraftFX
      );

      // Create download link with processed audio
      const url = URL.createObjectURL(processedBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedTrack.name?.replace('.wav', '') || 'track'}_with_fx.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up the object URL
      URL.revokeObjectURL(url);

      console.log('✅ Downloaded track with FX:', selectedTrack.name);
    } catch (error) {
      console.error('❌ Error downloading track with FX:', error);
      alert(`Failed to download track with FX: ${error.message}`);
    }
  }, [selectedTrack, state.buses, state.masterFX, rc20Enabled, rc20Settings, speccraftEnabled, speccraftSettings]);

  // Render track with FX applied (replaces track on timeline, keeps original in version history)
  const handleRenderWithFX = useCallback(async () => {
    if (!selectedTrack) return;

    try {
      console.log('🎛️ Rendering track with FX applied...');

      // Find which bus contains this track
      let trackBus = null;
      for (const bus of state.buses) {
        if (bus.tracks.find(t => t.id === selectedTrack.id)) {
          trackBus = bus;
          break;
        }
      }

      // Prepare RC-20 settings
      const rc20FX = rc20Enabled ? {
        enabled: true,
        magnitude: rc20Settings.magnitude,
        noise: rc20Settings.noise,
        wobble: rc20Settings.wobble,
        distortion: rc20Settings.distortion,
        magnitudeAmount: rc20Settings.magnitudeAmount
      } : { enabled: false };

      // Prepare SpecCraft settings
      const speccraftFX = speccraftEnabled ? {
        enabled: true,
        threshold: speccraftSettings.threshold,
        ratio: speccraftSettings.ratio,
        attack: speccraftSettings.attack,
        release: speccraftSettings.release,
        makeup: speccraftSettings.makeup,
        knee: speccraftSettings.knee,
        slope: speccraftSettings.slope,
        band1: {
          enabled: speccraftSettings.band1Enabled,
          freq: speccraftSettings.band1Freq,
          gain: speccraftSettings.band1Gain,
          q: speccraftSettings.band1Q
        },
        band2: {
          enabled: speccraftSettings.band2Enabled,
          freq: speccraftSettings.band2Freq,
          gain: speccraftSettings.band2Gain,
          q: speccraftSettings.band2Q
        },
        band3: {
          enabled: speccraftSettings.band3Enabled,
          freq: speccraftSettings.band3Freq,
          gain: speccraftSettings.band3Gain,
          q: speccraftSettings.band3Q
        },
        band4: {
          enabled: speccraftSettings.band4Enabled,
          freq: speccraftSettings.band4Freq,
          gain: speccraftSettings.band4Gain,
          q: speccraftSettings.band4Q
        }
      } : { enabled: false };

      // Call the API to process with FX
      const processedBlob = await generationAPI.downloadTrackWithFX(
        selectedTrack,
        trackBus,
        state.masterFX,
        rc20FX,
        speccraftFX
      );

      // Create a new object URL for the processed audio
      const newAudioUrl = URL.createObjectURL(processedBlob);

      // Save the current version to history before replacing
      const currentVersionName = selectedTrack.metadata?.versions?.[currentVersionIndex]?.name || 'Original';

      // Update the track with the processed audio
      dispatch({
        type: 'UPDATE_TRACK_AUDIO',
        payload: {
          trackId: selectedTrack.id,
          newAudioUrl: newAudioUrl,
          versionName: `${currentVersionName} + FX`
        }
      });

      console.log('✅ Rendered track with FX and saved to version history:', selectedTrack.name);
    } catch (error) {
      console.error('❌ Error rendering track with FX:', error);
      alert(`Failed to render track with FX: ${error.message}`);
    }
  }, [selectedTrack, state.buses, state.masterFX, rc20Enabled, rc20Settings, speccraftEnabled, speccraftSettings, currentVersionIndex, dispatch]);

  const handleDownloadBus = useCallback(async () => {
    if (!selectedBus) return;

    console.log('📦 Downloading all tracks from bus:', selectedBus.name);

    // Dynamically import JSZip
    const JSZip = (await import('jszip')).default;
    const zip = new JSZip();

    // Add each track to the zip
    for (let i = 0; i < selectedBus.tracks.length; i++) {
      const track = selectedBus.tracks[i];

      try {
        // Fetch the audio file as blob
        const response = await fetch(track.audioUrl);
        const blob = await response.blob();

        // Add to zip with a numbered filename
        const extension = track.name?.split('.').pop() || 'wav';
        const filename = `${i + 1}_${track.name || `track_${i + 1}.${extension}`}`;
        zip.file(filename, blob);

        console.log(`✅ Added to zip: ${filename}`);
      } catch (error) {
        console.error(`❌ Failed to add track ${track.name}:`, error);
      }
    }

    // Generate and download the zip file
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(zipBlob);
    link.download = `${selectedBus.name}_tracks.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);

    console.log('📦 Bus tracks downloaded as zip');
  }, [selectedBus]);

  // Submit generation feedback (like/dislike)
  const handleFeedback = useCallback(async (isLiked) => {
    if (!selectedTrack?.metadata?.params) {
      console.warn('No generation parameters found for feedback');
      return;
    }

    try {
      console.log(`${isLiked ? '👍' : '👎'} Submitting generation feedback...`);

      // Log feedback locally first
      const logEntry = logFeedback(selectedTrack, isLiked);
      console.log('📝 Logged to local storage:', logEntry);

      const response = await fetch('/api/generation-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          liked: isLiked,
          params: selectedTrack.metadata.params
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();
      console.log('✅ Feedback submitted:', result);

      setFeedbackSubmitted(isLiked ? 'like' : 'dislike');

      // Reset feedback UI after 2 seconds
      setTimeout(() => {
        setFeedbackSubmitted(null);
      }, 2000);

    } catch (error) {
      console.error('❌ Error submitting feedback:', error);
      // Even if API call fails, we've already logged locally
    }
  }, [selectedTrack]);

  // Split voices for composite MIDI tracks
  const handleSplitVoices = useCallback(() => {
    if (!selectedTrack || !selectedTrack.isComposite || !selectedBus) return;

    console.log('🎼 Expanding composite MIDI bus to show individual voices');

    // Dispatch action to expand the bus (showing individual tracks instead of composite)
    dispatch({
      type: 'TOGGLE_BUS_EXPANDED',
      payload: { busId: selectedBus.id }
    });

    // Deselect the composite track
    dispatch({ type: 'SELECT_TRACK', payload: { trackId: null, busId: null } });
  }, [selectedTrack, selectedBus, dispatch]);

  // Version control functions
  const switchToVersion = useCallback((versionIndex) => {
    if (!selectedTrack || !versions[versionIndex]) return;

    console.log(`🔄 Switching to version ${versionIndex + 1}...`);

    const version = versions[versionIndex];

    // Find the bus this track belongs to
    let trackBusId = null;
    for (const bus of state.buses) {
      if (bus.tracks.find(t => t.id === selectedTrack.id)) {
        trackBusId = bus.id;
        break;
      }
    }

    if (!trackBusId) return;

    // Update track with version's audio URL, inputFiles, and metadata
    const updates = {
      audioUrl: version.audioUrl,
      metadata: {
        ...selectedTrack.metadata,
        currentVersionIndex: versionIndex,
        // Restore the inputFiles from this version (includes MIDI paths)
        inputFiles: version.inputFiles || selectedTrack.metadata.inputFiles,
        // Restore generation params from this version
        params: version.params || selectedTrack.metadata.params,
        // Restore MIDI data from this version (if it exists)
        editedMidi: version.editedMidi || null,
        midiData: version.midiData || null
      }
    };

    console.log(`📂 Version ${versionIndex + 1} data:`, {
      inputFiles: version.inputFiles,
      editedMidi: version.editedMidi,
      midiData: version.midiData
    });

    dispatch({
      type: 'UPDATE_TRACK',
      payload: {
        busId: trackBusId,
        trackId: selectedTrack.id,
        updates
      }
    });

    // Refresh the selected track to show current version
    dispatch({ type: 'SELECT_TRACK', payload: { trackId: selectedTrack.id } });

    console.log(`✅ Switched to version ${versionIndex + 1}/${versions.length}`);
  }, [selectedTrack, versions, state.buses, dispatch]);

  const previousVersion = useCallback(() => {
    if (currentVersionIndex > 0) {
      switchToVersion(currentVersionIndex - 1);
    }
  }, [currentVersionIndex, switchToVersion]);

  const nextVersion = useCallback(() => {
    if (currentVersionIndex < versions.length - 1) {
      switchToVersion(currentVersionIndex + 1);
    }
  }, [currentVersionIndex, versions.length, switchToVersion]);

  /**
   * Toggle trumpet mute on/off for brass tracks.
   * When toggling ON: if mutedAudioUrl exists, swap instantly; otherwise call /api/apply-fx.
   * When toggling OFF: restore originalAudioUrl.
   */
  const handleMuteToggle = useCallback(async (enableMute) => {
    if (!selectedTrack) return;

    // Find the bus this track belongs to
    let trackBusId = null;
    for (const bus of state.buses) {
      if (bus.tracks.find(t => t.id === selectedTrack.id)) {
        trackBusId = bus.id;
        break;
      }
    }
    if (!trackBusId) return;

    if (enableMute) {
      // MUTE ON
      // If we already have a muted URL, swap instantly
      if (selectedTrack.metadata?.mutedAudioUrl) {
        setTrumpetMuted(true);
        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId: trackBusId,
            trackId: selectedTrack.id,
            updates: {
              audioUrl: selectedTrack.metadata.mutedAudioUrl,
              metadata: {
                ...selectedTrack.metadata,
                isMuted: true,
                originalAudioUrl: selectedTrack.metadata.originalAudioUrl || selectedTrack.audioUrl,
              }
            }
          }
        });
        return;
      }

      // No cached muted URL — call backend
      setMuteProcessing(true);
      setTrumpetMuted(true);
      try {
        const originalUrl = selectedTrack.audioUrl;
        const result = await generationAPI.applyFx(originalUrl, 'trumpet_mute');

        // Poll for completion
        if (result.status === 'processing' && result.task_id) {
          const maxAttempts = 60;
          let attempts = 0;
          let fxResult = result;

          while (attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 2000));
            attempts++;
            const statusResult = await generationAPI.pollFxStatus(fxResult.task_id);

            if (statusResult.status === 'completed') {
              fxResult = statusResult;
              break;
            } else if (statusResult.status === 'failed') {
              throw new Error(statusResult.error || 'Mute processing failed');
            }
          }

          if (attempts >= maxAttempts) {
            throw new Error('Mute processing timed out');
          }

          if (!fxResult.fx_url) {
            throw new Error('No muted audio URL returned');
          }

          const mutedUrl = fxResult.fx_url.startsWith('http') || fxResult.fx_url.startsWith('blob:')
            ? fxResult.fx_url
            : `https://doseedo.com${fxResult.fx_url}`;

          dispatch({
            type: 'UPDATE_TRACK',
            payload: {
              busId: trackBusId,
              trackId: selectedTrack.id,
              updates: {
                audioUrl: mutedUrl,
                metadata: {
                  ...selectedTrack.metadata,
                  isMuted: true,
                  originalAudioUrl: originalUrl,
                  mutedAudioUrl: mutedUrl,
                }
              }
            }
          });
        }
      } catch (error) {
        console.error('Mute processing failed:', error);
        setTrumpetMuted(false);
        alert(`Mute processing failed: ${error.message}`);
      } finally {
        setMuteProcessing(false);
      }
    } else {
      // MUTE OFF — restore original
      const originalUrl = selectedTrack.metadata?.originalAudioUrl;
      if (!originalUrl) {
        setTrumpetMuted(false);
        return;
      }

      setTrumpetMuted(false);
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId: trackBusId,
          trackId: selectedTrack.id,
          updates: {
            audioUrl: originalUrl,
            metadata: {
              ...selectedTrack.metadata,
              isMuted: false,
            }
          }
        }
      });
    }
  }, [selectedTrack, state.buses, dispatch]);

  /**
   * Clarify audio to improve instrument timbre quality
   */
  const handleClarify = useCallback(async () => {
    if (!selectedTrack) return;

    const instrumentGroup = selectedTrack.metadata?.params?.instrumentGroup || selectedTrack.instrumentGroup;
    const instrumentSubgroup = selectedTrack.metadata?.params?.instrumentSubgroup || selectedTrack.instrumentSubgroup;

    if (!instrumentGroup || !instrumentSubgroup) {
      alert('This track does not have instrument group/subgroup information required for clarification.');
      return;
    }

    console.log(`🎺 Clarifying track: ${selectedTrack.name}`);
    console.log(`   Group: ${instrumentGroup}, Subgroup: ${instrumentSubgroup}`);
    setClarifying(true);

    try {
      // Find the bus this track belongs to
      let trackBusId = null;
      for (const bus of state.buses) {
        if (bus.tracks.find(t => t.id === selectedTrack.id)) {
          trackBusId = bus.id;
          break;
        }
      }

      if (!trackBusId) {
        throw new Error('Could not find bus for this track');
      }

      // Call the clarify API
      const formData = new FormData();
      formData.append('audioUrl', selectedTrack.audioUrl);
      formData.append('instrumentGroup', instrumentGroup);
      formData.append('instrumentSubgroup', instrumentSubgroup);

      console.log('📤 Sending to /clarify-audio...');
      const clarifyResponse = await fetch('/clarify-audio', {
        method: 'POST',
        body: formData
      });

      if (!clarifyResponse.ok) {
        const errorText = await clarifyResponse.text();
        throw new Error(`HTTP ${clarifyResponse.status}: ${errorText}`);
      }

      const initialResult = await clarifyResponse.json();
      console.log('✅ Clarify task queued:', initialResult);

      // Poll for completion
      let result = initialResult;
      if (result.status === 'processing' && result.task_id) {
        console.log('⏳ Polling for clarify completion...');
        const maxAttempts = 60; // 60 attempts * 2s = 2 minutes max
        let attempts = 0;

        while (attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
          attempts++;

          const statusResponse = await fetch(`/clarify-audio/status/${result.task_id}`);
          if (!statusResponse.ok) {
            throw new Error(`Failed to check status: ${statusResponse.status}`);
          }

          const statusResult = await statusResponse.json();
          console.log(`🔄 Attempt ${attempts}/${maxAttempts} - Status:`, statusResult.status);

          if (statusResult.status === 'completed') {
            result = statusResult;
            console.log('✅ Clarification complete:', result);
            break;
          } else if (statusResult.status === 'failed') {
            throw new Error(statusResult.error || 'Clarification failed');
          }
        }

        if (attempts >= maxAttempts) {
          throw new Error('Clarification timed out');
        }
      }

      // Get the clarified audio URL
      if (!result.clarified_url) {
        throw new Error('No clarified audio URL in response');
      }

      const fullClarifiedUrl = result.clarified_url.startsWith('http') || result.clarified_url.startsWith('blob:')
        ? result.clarified_url
        : `https://doseedo.com${result.clarified_url}`;

      // Initialize version history if it doesn't exist
      let versionHistory = selectedTrack.metadata?.versions || [];
      if (versionHistory.length === 0) {
        versionHistory = [{
          audioUrl: selectedTrack.audioUrl,
          timestamp: Date.now(),
          type: 'original',
          name: 'Original',
          params: selectedTrack.metadata?.params || {}
        }];
      }

      // Add new version to history
      const newVersion = {
        audioUrl: fullClarifiedUrl,
        timestamp: Date.now(),
        type: 'clarified',
        name: `Clarified ${versionHistory.filter(v => v.type === 'clarified').length + 1}`,
        params: selectedTrack.metadata?.params || {}
      };

      versionHistory.push(newVersion);
      const newVersionIndex = versionHistory.length - 1;

      // Update track with clarified audio
      const updates = {
        audioUrl: fullClarifiedUrl,
        metadata: {
          ...selectedTrack.metadata,
          versions: versionHistory,
          currentVersionIndex: newVersionIndex
        }
      };

      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId: trackBusId,
          trackId: selectedTrack.id,
          updates
        }
      });

      // Refresh the selected track
      dispatch({ type: 'SELECT_TRACK', payload: { trackId: selectedTrack.id } });

      console.log(`✅ Clarification complete! Now at version ${newVersionIndex + 1}/${versionHistory.length}`);

    } catch (error) {
      console.error('❌ Clarification failed:', error);
      alert(`Clarification failed: ${error.message}`);
    } finally {
      setClarifying(false);
    }
  }, [selectedTrack, state.buses, dispatch]);

  const handleRegenerate = useCallback(async () => {
    if (!selectedTrack || !selectedTrack.metadata?.params) {
      alert('This track has no generation parameters');
      return;
    }

    console.log('🔄 Regenerating track with same settings...');
    setRegenerating(true);

    try {
      // Get the bus this track belongs to AND get the FRESH track from state
      let trackBusId = null;
      let freshTrack = null;
      for (const bus of state.buses) {
        const track = bus.tracks.find(t => t.id === selectedTrack.id);
        if (track) {
          trackBusId = bus.id;
          freshTrack = track; // Use fresh track from Redux state
          break;
        }
      }

      if (!trackBusId || !freshTrack) {
        throw new Error('Could not find bus or track in current state');
      }

      console.log('🔍 Using fresh track from Redux state (not stale prop)');

      // Initialize version history if it doesn't exist
      let versionHistory = freshTrack.metadata.versions || [];
      if (versionHistory.length === 0) {
        // Create initial version from current track
        versionHistory = [{
          audioUrl: freshTrack.audioUrl,
          timestamp: Date.now(),
          type: 'original',
          name: 'Original',
          inputFiles: freshTrack.metadata?.inputFiles || {},
          params: freshTrack.metadata?.params || {},
          editedMidi: freshTrack.metadata?.editedMidi || null,
          midiData: freshTrack.metadata?.midiData || null
        }];
      }

      // Get the original generation parameters
      let params = { ...freshTrack.metadata.params };

      // CRITICAL: If this is a monophonic voice track, set inpaintVoiceIndex so backend only regenerates this voice
      // DO NOT use inpaint mode for regeneration - that's for time-based inpainting
      // Voice regeneration uses inpaintVoiceIndex to select which voice to generate
      if (freshTrack.metadata?.voiceNumber !== undefined && params.monophonicMode) {
        params.inpaintMode = false; // Regeneration is NOT time-based inpainting
        params.inpaintVoiceIndex = freshTrack.metadata.voiceNumber; // Select which voice to regenerate
        params.inpaintStartTime = undefined; // Not used for regeneration
        params.inpaintEndTime = undefined; // Not used for regeneration
        console.log(`🔄 Regenerating voice ${params.inpaintVoiceIndex} only (full track regeneration, not inpaint)`);
      }

      console.log('📋 Regenerating with params:', params);

      // Check if there's edited MIDI in metadata - if so, send it as JSON for the backend to use
      // Check both editedMidi (from edited generated tracks) and midiData (from replaced MIDI tracks)
      let inputFile = null;
      const inputFiles = freshTrack.metadata?.inputFiles;
      const editedMidi = freshTrack.metadata?.editedMidi || freshTrack.metadata?.midiData;

      console.log('🔍 REGENERATION - Track metadata check:');
      console.log('   Track ID:', freshTrack.id);
      console.log('   Voice Number:', freshTrack.metadata?.voiceNumber);
      console.log('   Full metadata:', freshTrack.metadata);
      console.log('   hasInputFiles:', !!inputFiles);
      console.log('   inputFiles:', inputFiles);
      console.log('   hasEditedMidi:', !!editedMidi);
      console.log('   editedMidi source:', freshTrack.metadata?.editedMidi ? 'editedMidi' : freshTrack.metadata?.midiData ? 'midiData' : 'none');
      console.log('   editedMidi:', editedMidi);
      console.log('   editedMidiNotes:', editedMidi?.notes?.length || 0);
      console.log('   editedMidiTempo:', editedMidi?.tempo || 'N/A');

      // If edited MIDI exists, send it as JSON for backend to regenerate from
      if (editedMidi && editedMidi.notes && editedMidi.notes.length > 0) {
        console.log(`🎹 Using EDITED MIDI for regeneration (${editedMidi.notes.length} notes, tempo: ${editedMidi.tempo} BPM)`);

        // Remove tempoOverride to use the MIDI's embedded tempo instead
        if (params.tempoOverride) {
          console.log(`🎵 Removing tempoOverride (${params.tempoOverride} BPM) to use MIDI tempo (${editedMidi.tempo} BPM)`);
          params = { ...params, tempoOverride: null };
        }

        // Create a JSON blob with the edited MIDI data
        const midiDataJson = JSON.stringify({
          type: 'edited_midi',
          notes: editedMidi.notes,
          duration: editedMidi.duration,
          tempo: editedMidi.tempo || 120
        });

        const voiceNumber = freshTrack.metadata?.voiceNumber;
        const voiceLabel = voiceNumber !== undefined ? voiceNumber : 'master';
        const midiBlob = new Blob([midiDataJson], { type: 'application/json' });
        inputFile = new File([midiBlob], `${voiceLabel}_edited_midi.json`, { type: 'application/json' });
        console.log(`✅ Edited MIDI prepared for regeneration (${editedMidi.notes.length} notes)`);
        console.log(`🔍 File object created:`, {
          name: inputFile.name,
          type: inputFile.type,
          size: inputFile.size,
          lastModified: inputFile.lastModified
        });
      } else if (inputFiles) {
        // No edited MIDI, use original input file
        // Determine which input file to use based on type
        let inputFileUrl = null;
        let inputFileType = 'wav';

        if (inputFiles.type === 'midi' && inputFiles.midiPath) {
          inputFileUrl = inputFiles.midiPath;
          inputFileType = 'midi';
        } else if (inputFiles.type === 'wav' && inputFiles.renderPath) {
          inputFileUrl = inputFiles.renderPath;
          inputFileType = 'wav';
        }

        // Fetch the input file
        if (inputFileUrl && inputFileUrl !== 'null' && inputFileUrl !== null) {
          try {
            console.log(`📥 Fetching original input file for regeneration (${inputFileType}): ${inputFileUrl}`);
            const fullUrl = inputFileUrl.startsWith('http') || inputFileUrl.startsWith('blob:')
              ? inputFileUrl
              : `https://doseedo.com${inputFileUrl}`;

            const inputResponse = await fetch(fullUrl);
            if (inputResponse.ok) {
              const inputBlob = await inputResponse.blob();
              const voiceNumber = freshTrack.metadata?.voiceNumber;
              const voiceLabel = voiceNumber !== undefined ? voiceNumber : 'master';
              const fileName = inputFileType === 'midi'
                ? `${voiceLabel}_input.mid`
                : `${voiceLabel}_input_render.wav`;
              const mimeType = inputFileType === 'midi' ? 'audio/midi' : 'audio/wav';

              inputFile = new File([inputBlob], fileName, { type: mimeType });
              console.log(`✅ Original input file loaded for regeneration: ${inputFile.name}`);
            } else {
              console.warn(`⚠️ Input file not found (${inputResponse.status}): ${inputFileUrl}`);
            }
          } catch (error) {
            console.warn(`⚠️ Error fetching input file for regeneration:`, error.message);
          }
        }
      }

      // Start generation with original input file
      const startResult = await generationAPI.startGeneration(params, inputFile);
      console.log('✅ Regeneration started:', startResult);

      if (!startResult.task_id) {
        throw new Error('No task_id received from server');
      }

      const { task_id } = startResult;

      // Poll until complete
      const result = await generationAPI.pollUntilComplete(
        task_id,
        (progressData) => {
          console.log('📊 Regeneration progress:', progressData);
        },
        null // No partial results for version control
      );

      console.log('✅ Regeneration completed:', result);

      // Get the regenerated audio URL
      if (result.file_paths && result.file_paths.length > 0) {
        // For monophonic mode, get the specific voice that was regenerated
        // For polyphonic, get the main output (voice 1 or first non-zero)
        const match = result.file_paths[0].match(/\/(\d+)\.wav$/);
        const voiceNumber = match ? parseInt(match[1], 10) : null;

        let newAudioUrl = result.file_paths[0];

        // If monophonic mode and we have a voice number, find the matching voice
        if (params.monophonicMode && freshTrack.metadata?.voiceNumber !== undefined) {
          const targetVoice = freshTrack.metadata.voiceNumber;
          const matchingFile = result.file_paths.find(path => {
            const m = path.match(/\/(\d+)\.wav$/);
            return m && parseInt(m[1], 10) === targetVoice;
          });
          if (matchingFile) {
            newAudioUrl = matchingFile;
          }
        } else {
          // Skip voice 0 if available
          const nonZeroFile = result.file_paths.find(path => {
            const m = path.match(/\/(\d+)\.wav$/);
            return m && parseInt(m[1], 10) !== 0;
          });
          if (nonZeroFile) {
            newAudioUrl = nonZeroFile;
          }
        }

        // Add new version to history
        const newVersion = {
          audioUrl: newAudioUrl,
          timestamp: Date.now(),
          type: 'regenerated',
          name: `Regeneration ${versionHistory.filter(v => v.type === 'regenerated').length + 1}`,
          params: params, // Save the params used for this version
          inputFiles: result.input_files || {}, // Save the input files from this generation
          // Store the MIDI data that was used for this generation
          editedMidi: editedMidi || null,
          midiData: freshTrack.metadata?.midiData || null
        };

        versionHistory.push(newVersion);
        const newVersionIndex = versionHistory.length - 1;

        // Update track with new audio, version history, and fresh input files
        const updates = {
          audioUrl: newAudioUrl,
          metadata: {
            ...freshTrack.metadata,
            versions: versionHistory,
            currentVersionIndex: newVersionIndex,
            inputFiles: result.input_files || freshTrack.metadata.inputFiles, // Update input files
            editedMidi: freshTrack.metadata.editedMidi // Keep edited MIDI (frontend will continue displaying it)
          }
        };

        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId: trackBusId,
            trackId: freshTrack.id,
            updates
          }
        });

        // Refresh the selected track to show updated version history
        dispatch({ type: 'SELECT_TRACK', payload: { trackId: freshTrack.id } });

        console.log(`✅ Regeneration complete! Now at version ${newVersionIndex + 1}/${versionHistory.length}`);
        alert(`Regeneration complete! Switched to version ${newVersionIndex + 1}`);
      }

    } catch (error) {
      console.error('❌ Regeneration failed:', error);
      alert(`Regeneration failed: ${error.message}`);
    } finally {
      setRegenerating(false);
    }
  }, [selectedTrack, state.buses, dispatch]);

  // Convert AudioBuffer to WAV blob
  const audioBufferToWav = (buffer) => {
    return new Promise((resolve) => {
      const numberOfChannels = buffer.numberOfChannels;
      const length = buffer.length * numberOfChannels * 2;
      const arrayBuffer = new ArrayBuffer(44 + length);
      const view = new DataView(arrayBuffer);

      // Write WAV header
      const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      };

      writeString(0, 'RIFF');
      view.setUint32(4, 36 + length, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, numberOfChannels, true);
      view.setUint32(24, buffer.sampleRate, true);
      view.setUint32(28, buffer.sampleRate * numberOfChannels * 2, true);
      view.setUint16(32, numberOfChannels * 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, length, true);

      // Write audio data
      const offset = 44;
      const channels = [];
      for (let i = 0; i < numberOfChannels; i++) {
        channels.push(buffer.getChannelData(i));
      }

      let index = 0;
      for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
          const sample = Math.max(-1, Math.min(1, channels[channel][i]));
          view.setInt16(offset + index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
          index++;
        }
      }

      resolve(new Blob([arrayBuffer], { type: 'audio/wav' }));
    });
  };

  // Splice inpainted audio segment into original audio
  const spliceInpaintedAudio = useCallback(async (originalAudioUrl, inpaintedSegmentUrl, startTime, endTime) => {
    try {
      // Build full URLs
      const originalUrl = originalAudioUrl.startsWith('http') || originalAudioUrl.startsWith('blob:')
        ? originalAudioUrl
        : `https://doseedo.com${originalAudioUrl}`;

      const inpaintedUrl = inpaintedSegmentUrl.startsWith('http') || inpaintedSegmentUrl.startsWith('blob:')
        ? inpaintedSegmentUrl
        : `https://doseedo.com${inpaintedSegmentUrl}`;

      console.log('📥 Fetching audio for splicing...');
      console.log('  Original:', originalUrl);
      console.log('  Inpainted segment:', inpaintedUrl);

      // Fetch both audio files
      const [originalResponse, inpaintedResponse] = await Promise.all([
        fetch(originalUrl),
        fetch(inpaintedUrl)
      ]);

      if (!originalResponse.ok || !inpaintedResponse.ok) {
        throw new Error('Failed to fetch audio files for splicing');
      }

      const [originalArrayBuffer, inpaintedArrayBuffer] = await Promise.all([
        originalResponse.arrayBuffer(),
        inpaintedResponse.arrayBuffer()
      ]);

      // Create AudioContext
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContext();

      console.log('🎵 Decoding audio buffers...');
      const [originalBuffer, inpaintedBuffer] = await Promise.all([
        audioContext.decodeAudioData(originalArrayBuffer.slice(0)),
        audioContext.decodeAudioData(inpaintedArrayBuffer.slice(0))
      ]);

      // Calculate sample positions
      const sampleRate = originalBuffer.sampleRate;
      const numberOfChannels = originalBuffer.numberOfChannels;
      const startSample = Math.floor(startTime * sampleRate);
      const endSample = Math.floor(endTime * sampleRate);
      const inpaintedLength = inpaintedBuffer.length;

      console.log(`🔧 Splicing at samples ${startSample} to ${endSample}`);
      console.log(`  Original length: ${originalBuffer.length} samples (${originalBuffer.duration.toFixed(2)}s)`);
      console.log(`  Inpainted length: ${inpaintedLength} samples (${inpaintedBuffer.duration.toFixed(2)}s)`);
      console.log(`  Region to replace: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`);
      console.log(`  Backend already extracted exact inpaint region (no context)`);

      // Create new buffer with spliced audio
      // Backend returns ONLY the inpainted region without context
      const newLength = originalBuffer.length - (endSample - startSample) + inpaintedLength;
      const newBuffer = audioContext.createBuffer(numberOfChannels, newLength, sampleRate);

      console.log(`  New total length: ${newLength} samples (${(newLength / sampleRate).toFixed(2)}s)`);

      // Copy audio data for each channel
      for (let channel = 0; channel < numberOfChannels; channel++) {
        const originalData = originalBuffer.getChannelData(channel);
        const inpaintedData = inpaintedBuffer.getChannelData(Math.min(channel, inpaintedBuffer.numberOfChannels - 1));
        const newData = newBuffer.getChannelData(channel);

        // Copy before inpainted region
        for (let i = 0; i < startSample; i++) {
          newData[i] = originalData[i];
        }

        // Copy inpainted segment (backend already cropped it)
        for (let i = 0; i < inpaintedLength; i++) {
          newData[startSample + i] = inpaintedData[i];
        }

        // Copy after inpainted region
        for (let i = endSample; i < originalBuffer.length; i++) {
          newData[startSample + inpaintedLength + (i - endSample)] = originalData[i];
        }
      }

      console.log(`✅ Created spliced buffer: ${newLength} samples`);

      // Convert AudioBuffer to WAV blob
      const wavBlob = await audioBufferToWav(newBuffer);
      const blobUrl = URL.createObjectURL(wavBlob);

      console.log('✅ Splicing complete, blob URL:', blobUrl);
      return blobUrl;

    } catch (error) {
      console.error('❌ Audio splicing failed:', error);
      throw error;
    }
  }, []);

  const performInpainting = useCallback(async (startTime, endTime) => {
    if (!selectedTrack || !selectedTrack.metadata?.params) {
      alert('Cannot inpaint: missing generation parameters');
      return;
    }

    console.log(`🎨 Performing inpainting: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s...`);
    setRegenerating(true);

    try {
      // Get the bus this track belongs to
      let trackBusId = null;
      for (const bus of state.buses) {
        if (bus.tracks.find(t => t.id === selectedTrack.id)) {
          trackBusId = bus.id;
          break;
        }
      }

      if (!trackBusId) {
        throw new Error('Could not find bus for this track');
      }

      // Initialize version history if it doesn't exist
      let versionHistory = selectedTrack.metadata.versions || [];
      if (versionHistory.length === 0) {
        versionHistory = [{
          audioUrl: selectedTrack.audioUrl,
          timestamp: Date.now(),
          type: 'original',
          name: 'Original'
        }];
      }

      // Get the original generation parameters
      const params = selectedTrack.metadata.params;

      console.log('🔍 Original track params:', params);
      console.log('🔍 Track voiceNumber:', selectedTrack.metadata?.voiceNumber);
      console.log('🔍 Track monophonicMode:', params.monophonicMode);

      // Add inpainting parameters
      const inpaintParams = {
        ...params,
        inpaintMode: true,
        inpaintStartTime: startTime,
        inpaintEndTime: endTime,
        inpaintDuration: endTime - startTime
      };

      // If monophonic mode, specify which voice to inpaint
      if (selectedTrack.metadata?.voiceNumber !== undefined) {
        inpaintParams.inpaintVoiceIndex = selectedTrack.metadata.voiceNumber;
        console.log(`🎵 Inpainting voice ${inpaintParams.inpaintVoiceIndex} ONLY`);
        console.log(`🎵 inpaintVoiceIndex set to: ${inpaintParams.inpaintVoiceIndex}`);
      } else {
        console.warn('⚠️ WARNING: voiceNumber is undefined! Inpainting may generate all voices!');
      }

      console.log('📤 FINAL inpainting params being sent to backend:', JSON.stringify(inpaintParams, null, 2));

      // Determine voice information
      const voiceNumber = selectedTrack.metadata?.voiceNumber;
      const isMonophonicTrack = voiceNumber !== undefined && voiceNumber !== null;

      // For monophonic tracks, set the specific voice index
      // For non-monophonic tracks, we'll inpaint the master track (voice index not set)
      if (isMonophonicTrack) {
        if (voiceNumber === 0) {
          console.warn('⚠️ Inpainting voice 0 (mix track) - this will regenerate the mix');
        } else {
          console.log(`🎵 Inpainting voice ${voiceNumber}`);
        }
      } else {
        console.log(`🎵 Inpainting non-monophonic track - will use master generation params`);
        // Don't set inpaintVoiceIndex for non-monophonic tracks
        delete inpaintParams.inpaintVoiceIndex;
      }

      // Fetch the input file that was used for the original generation
      let inputFile = null;
      const inputFiles = selectedTrack.metadata?.inputFiles;

      console.log('🔍 Track inputFiles metadata:', inputFiles);

      // Determine which input file to use based on what was stored
      let inputFileUrl = null;
      let inputFileType = 'wav';

      if (inputFiles) {
        // For monophonic mode: use FluidSynth render OR split MIDI (whichever was used)
        // Priority: renderPath (FluidSynth) > midiPath (fast mode split MIDI)
        if (inputFiles.renderPath) {
          inputFileUrl = inputFiles.renderPath;
          inputFileType = 'wav';
          console.log(`🎵 Using FluidSynth render for voice ${voiceNumber}: ${inputFileUrl}`);
        } else if (inputFiles.midiPath) {
          inputFileUrl = inputFiles.midiPath;
          inputFileType = 'midi';
          console.log(`🎵 Using split MIDI for voice ${voiceNumber}: ${inputFileUrl}`);
        } else {
          console.warn('⚠️ inputFiles object exists but has no renderPath or midiPath');
        }
      } else {
        console.warn('⚠️ No inputFiles metadata found in track');
      }

      // For monophonic mode, audio file is REQUIRED
      if (params.monophonicMode && !inputFileUrl) {
        throw new Error('Monophonic inpaint mode requires the original input file (FluidSynth render or MIDI). The track metadata does not contain inputFiles information. This may be an old track generated before input file tracking was added.');
      }

      // Fetch the audio file
      if (inputFileUrl && inputFileUrl !== 'null' && inputFileUrl !== null) {
        try {
          console.log(`📥 Fetching audio file for inpainting (${inputFileType}): ${inputFileUrl}`);
          const fullUrl = inputFileUrl.startsWith('http') || inputFileUrl.startsWith('blob:')
            ? inputFileUrl
            : `https://doseedo.com${inputFileUrl}`;

          const inputResponse = await fetch(fullUrl);
          if (inputResponse.ok) {
            const inputBlob = await inputResponse.blob();

            // Use voice number for filename, or 'master' for non-monophonic
            const voiceLabel = isMonophonicTrack ? voiceNumber : 'master';
            const fileName = inputFileType === 'midi'
              ? `${voiceLabel}_input.mid`
              : `${voiceLabel}_conditioning.wav`;
            const mimeType = inputFileType === 'midi' ? 'audio/midi' : 'audio/wav';

            inputFile = new File([inputBlob], fileName, { type: mimeType });
            console.log(`✅ Audio file loaded for inpainting: ${inputFile.name} (${(inputBlob.size / 1024).toFixed(2)} KB)`);
          } else {
            if (params.monophonicMode) {
              throw new Error(`Failed to fetch audio file: ${inputResponse.status} - ${inputFileUrl}`);
            }
            console.warn(`⚠️ Input file not found (${inputResponse.status}): ${inputFileUrl}`);
            console.warn(`⚠️ Will send only params - backend should use same seed/settings to recreate similar output`);
          }
        } catch (error) {
          if (params.monophonicMode) {
            throw new Error(`Failed to load audio file for monophonic inpainting: ${error.message}`);
          }
          console.warn(`⚠️ Error fetching input file:`, error.message);
          console.warn(`⚠️ Will send only params - backend should use same seed/settings to recreate similar output`);
        }
      } else if (params.monophonicMode) {
        throw new Error('Monophonic inpaint mode requires an audio file, but no URL was found.');
      } else {
        console.warn(`⚠️ No input file URL saved (backend didn't return input files)`);
        console.warn(`⚠️ Will send only params - backend should use same seed/settings to recreate similar output`);
      }

      // Start generation with inpaint params and input file
      const startResult = await generationAPI.startGeneration(inpaintParams, inputFile);
      console.log('✅ Inpainting started:', startResult);

      if (!startResult.task_id) {
        throw new Error('No task_id received from server');
      }

      const { task_id } = startResult;

      // Poll until complete
      const result = await generationAPI.pollUntilComplete(
        task_id,
        (progressData) => {
          console.log('📊 Inpainting progress:', progressData);
        },
        null
      );

      console.log('✅ Inpainting completed:', result);

      // Get the inpainted audio URL
      if (result.file_paths && result.file_paths.length > 0) {
        let inpaintedSegmentUrl = result.file_paths[0];

        // If monophonic mode, find the matching voice
        if (inpaintParams.inpaintVoiceIndex !== undefined) {
          const matchingFile = result.file_paths.find(path => {
            const m = path.match(/\/(\d+)\.wav$/);
            return m && parseInt(m[1], 10) === inpaintParams.inpaintVoiceIndex;
          });
          if (matchingFile) {
            inpaintedSegmentUrl = matchingFile;
          }
        }

        // SPLICE the inpainted segment into the original audio
        console.log('🔧 Splicing inpainted audio into original...');
        console.log(`   Original track: ${selectedTrack.audioUrl}`);
        console.log(`   Inpainted segment: ${inpaintedSegmentUrl}`);
        console.log(`   Splice range: ${startTime.toFixed(3)}s - ${endTime.toFixed(3)}s (duration: ${(endTime - startTime).toFixed(3)}s)`);

        const splicedAudioUrl = await spliceInpaintedAudio(
          selectedTrack.audioUrl,
          inpaintedSegmentUrl,
          startTime,
          endTime
        );

        // Add new version to history
        const newVersion = {
          audioUrl: splicedAudioUrl,
          timestamp: Date.now(),
          type: 'inpainted',
          name: `Inpaint ${versionHistory.filter(v => v.type === 'inpainted').length + 1} (${startTime.toFixed(1)}s-${endTime.toFixed(1)}s)`,
          inputFiles: result.input_files || {},
          params: inpaintParams,
          editedMidi: selectedTrack.metadata?.editedMidi || null,
          midiData: selectedTrack.metadata?.midiData || null
        };

        versionHistory.push(newVersion);
        const newVersionIndex = versionHistory.length - 1;

        // Update track with spliced audio and version history
        const updates = {
          audioUrl: splicedAudioUrl,
          metadata: {
            ...selectedTrack.metadata,
            versions: versionHistory,
            currentVersionIndex: newVersionIndex
          }
        };

        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId: trackBusId,
            trackId: selectedTrack.id,
            updates
          }
        });

        // Refresh the selected track to show updated version history
        dispatch({ type: 'SELECT_TRACK', payload: { trackId: selectedTrack.id } });

        console.log(`✅ Inpainting complete! Now at version ${newVersionIndex + 1}/${versionHistory.length}`);
        alert(`Inpainting complete! Switched to version ${newVersionIndex + 1}`);
      }

    } catch (error) {
      console.error('❌ Inpainting failed:', error);
      alert(`Inpainting failed: ${error.message}`);
    } finally {
      setRegenerating(false);
    }
  }, [selectedTrack, state.buses, dispatch]);

  const confirmInpaint = useCallback(() => {
    if (!inpaintSelectionTimes) return;

    const { start, end } = inpaintSelectionTimes;
    performInpainting(start, end);

    // Clear the selection after starting inpainting
    dispatch({ type: 'CLEAR_INPAINT_SELECTION' });
  }, [inpaintSelectionTimes, performInpainting, dispatch]);

  const cancelInpaintSelection = useCallback(() => {
    // Clear the selection without inpainting
    dispatch({ type: 'CLEAR_INPAINT_SELECTION' });
  }, [dispatch]);

  const toggleInpaintMode = useCallback(() => {
    if (!selectedTrack || !selectedTrack.metadata?.params) {
      alert('This track has no generation parameters');
      return;
    }

    const newMode = !state.inpaintMode?.enabled;

    if (newMode) {
      const voiceNumber = selectedTrack.metadata?.voiceNumber;
      const hasInputFile = !!(selectedTrack.metadata?.inputFiles?.midiPath || selectedTrack.metadata?.inputFiles?.renderPath);

      if (voiceNumber !== undefined && voiceNumber !== null) {
        console.log('🎨 Inpainting mode activated - drag on waveform to select region');
        console.log(`🎨 Will inpaint voice ${voiceNumber}${hasInputFile ? ' using saved input file' : ''}`);
      } else {
        console.log('🎨 Inpainting mode activated for non-monophonic track');
        console.log(`🎨 Will inpaint entire track${hasInputFile ? ' using saved input file' : ' with same generation params'}`);
      }

      dispatch({ type: 'SET_INPAINT_MODE', payload: { trackId: selectedTrack.id, enabled: true } });
    } else {
      console.log('🎨 Inpainting mode deactivated');
      dispatch({ type: 'SET_INPAINT_MODE', payload: { trackId: null, enabled: false } });
    }
  }, [selectedTrack, state.inpaintMode, dispatch]);

  const handleSeparateStem = useCallback(async (stemType, versionName) => {
    if (!selectedTrack) return;

    console.log(`🎵 Separating ${stemType} stem from selected track...`);
    setStemProcessing(true);

    try {
      // Fetch the audio file
      const response = await fetch(selectedTrack.audioUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch audio: ${response.status}`);
      }

      const audioBlob = await response.blob();
      console.log('✅ Audio fetched, size:', audioBlob.size);

      // Create FormData
      const formData = new FormData();
      formData.append('audioFile', audioBlob, 'audio.wav');

      console.log('📤 Sending to /separate-stems...');
      const stemResponse = await fetch('/separate-stems', {
        method: 'POST',
        body: formData
      });

      if (!stemResponse.ok) {
        const errorText = await stemResponse.text();
        throw new Error(`HTTP ${stemResponse.status}: ${errorText}`);
      }

      const result = await stemResponse.json();
      console.log('✅ Stem separation complete:', result);

      // Get the requested stem
      let stemUrl;
      if (stemType === 'vocals') {
        stemUrl = result.stems.vocals;
        if (!stemUrl) {
          throw new Error('No vocals stem found in result');
        }
      } else if (stemType === 'accompaniment') {
        stemUrl = result.stems.other || result.stems.accompaniment;
        if (!stemUrl) {
          throw new Error('No accompaniment stem found in result');
        }
      }

      const fullStemUrl = stemUrl.startsWith('http') || stemUrl.startsWith('blob:')
        ? stemUrl
        : `https://doseedo.com${stemUrl}`;

      // Find the bus this track belongs to
      let trackBusId = null;
      for (const bus of state.buses) {
        if (bus.tracks.find(t => t.id === selectedTrack.id)) {
          trackBusId = bus.id;
          break;
        }
      }

      if (!trackBusId) {
        throw new Error('Could not find bus for this track');
      }

      // Initialize version history if it doesn't exist
      let versionHistory = selectedTrack.metadata?.versions || [];
      if (versionHistory.length === 0) {
        // Create initial version from current track
        versionHistory = [{
          audioUrl: selectedTrack.audioUrl,
          timestamp: Date.now(),
          type: 'original',
          name: 'Original',
          inputFiles: selectedTrack.metadata?.inputFiles || {},
          params: selectedTrack.metadata?.params || {},
          editedMidi: selectedTrack.metadata?.editedMidi || null,
          midiData: selectedTrack.metadata?.midiData || null
        }];
      }

      // Add new version with separated stem
      const newVersion = {
        audioUrl: fullStemUrl,
        timestamp: Date.now(),
        type: `stem_${stemType}`,
        name: versionName,
        inputFiles: selectedTrack.metadata?.inputFiles || {},
        params: selectedTrack.metadata?.params || {},
        editedMidi: selectedTrack.metadata?.editedMidi || null,
        midiData: selectedTrack.metadata?.midiData || null
      };

      versionHistory.push(newVersion);
      const newVersionIndex = versionHistory.length - 1;

      // Update track with stem audio and version history
      const updates = {
        audioUrl: fullStemUrl,
        metadata: {
          ...selectedTrack.metadata,
          versions: versionHistory,
          currentVersionIndex: newVersionIndex
        }
      };

      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId: trackBusId,
          trackId: selectedTrack.id,
          updates
        }
      });

      // Refresh the selected track to show updated version history
      dispatch({ type: 'SELECT_TRACK', payload: { trackId: selectedTrack.id } });

      console.log(`✅ Stem separation complete! Now at version ${newVersionIndex + 1}/${versionHistory.length}`);
      alert(`Stem separation complete! Switched to version ${newVersionIndex + 1}`);

    } catch (error) {
      console.error('❌ Stem separation failed:', error);
      alert(`Stem separation failed: ${error.message}`);
    } finally {
      setStemProcessing(false);
    }
  }, [selectedTrack, state.buses, dispatch]);

  const handleRemoveMusic = useCallback(() => {
    handleSeparateStem('vocals', 'Dialogue Only');
  }, [handleSeparateStem]);

  const handleRemoveDialogue = useCallback(() => {
    handleSeparateStem('accompaniment', 'Music Only');
  }, [handleSeparateStem]);

  /**
   * Separate track into 6 stems and add to new bus
   */
  const handleSeparateIntoStems = useCallback(async () => {
    if (!selectedTrack) return;

    console.log(`🎵 Separating track into 6 stems...`);
    setStemProcessing(true);

    try {
      // Fetch the audio file
      const response = await fetch(selectedTrack.audioUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch audio: ${response.status}`);
      }

      const audioBlob = await response.blob();
      console.log('✅ Audio fetched, size:', audioBlob.size);

      // Create FormData
      const formData = new FormData();
      formData.append('audioFile', audioBlob, selectedTrack.name || 'audio.wav');

      console.log('📤 Sending to /separate-stems...');
      const stemResponse = await fetch('/separate-stems', {
        method: 'POST',
        body: formData
      });

      if (!stemResponse.ok) {
        const errorText = await stemResponse.text();
        throw new Error(`HTTP ${stemResponse.status}: ${errorText}`);
      }

      const initialResult = await stemResponse.json();
      console.log('✅ Stem separation task queued:', initialResult);

      // Poll for completion if task is processing
      let result = initialResult;
      if (result.status === 'processing' && result.task_id) {
        console.log('⏳ Polling for stem separation completion...');
        const maxAttempts = 60; // 60 attempts * 2s = 2 minutes max
        let attempts = 0;

        while (attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
          attempts++;

          const statusResponse = await fetch(`/separate-stems/status/${result.task_id}`);
          if (!statusResponse.ok) {
            throw new Error(`Failed to check status: ${statusResponse.status}`);
          }

          const statusResult = await statusResponse.json();
          console.log(`🔄 Attempt ${attempts}/${maxAttempts} - Status:`, statusResult.status);

          if (statusResult.status === 'completed') {
            result = statusResult;
            console.log('✅ Stem separation complete:', result);
            break;
          } else if (statusResult.status === 'failed') {
            throw new Error(statusResult.error || 'Stem separation failed');
          }
        }

        if (attempts >= maxAttempts) {
          throw new Error('Stem separation timed out');
        }
      }

      // Check if stems are present in the response
      if (!result.stems || typeof result.stems !== 'object') {
        console.error('❌ Invalid stem separation response:', result);
        throw new Error('Invalid response: stems object not found');
      }

      // Expected stems: drums, bass, other, vocals, guitar, piano
      const stemNames = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'];
      const availableStems = stemNames.filter(name => result.stems[name]);

      if (availableStems.length === 0) {
        throw new Error('No stems found in result');
      }

      console.log(`📦 Found ${availableStems.length} stems:`, availableStems);

      // Find the bus that contains the original track
      let originalTrackBusId = null;
      for (const bus of state.buses) {
        if (bus.tracks.find(t => t.id === selectedTrack.id)) {
          originalTrackBusId = bus.id;
          break;
        }
      }

      if (!originalTrackBusId) {
        throw new Error('Could not find bus for original track');
      }

      console.log(`📍 Adding stems to same bus as original track: ${originalTrackBusId}`);

      // Add each stem as a separate track to the same bus
      for (const stemName of availableStems) {
        const stemUrl = result.stems[stemName];
        const fullStemUrl = stemUrl.startsWith('http') || stemUrl.startsWith('blob:')
          ? stemUrl
          : `https://doseedo.com${stemUrl}`;

        const stemTrack = {
          id: `track-${Date.now()}-${stemName}-${Math.random()}`,
          name: `${stemName.charAt(0).toUpperCase() + stemName.slice(1)}`,
          audioUrl: fullStemUrl,
          duration: selectedTrack.duration || 0,
          startPosition: selectedTrack.startPosition || 0,
          gain: 1.0,
          isMuted: false,
          isSolo: false,
          cropStart: 0,
          cropEnd: 0,
          fx: {
            reverb: 0,
            fadeIn: 0.2,
            fadeOut: 1.0
          },
          metadata: {
            type: 'stem',
            stemType: stemName,
            originalTrackId: selectedTrack.id,
            originalTrackName: selectedTrack.name
          },
          instrumentGroup: stemName === 'vocals' ? 'vocal' : 'instrument',
          instrumentSubgroup: stemName
        };

        dispatch({
          type: 'ADD_TRACK',
          payload: {
            busId: originalTrackBusId,
            track: stemTrack
          }
        });

        console.log(`✅ Added ${stemName} stem to bus`);
      }

      // Mute the original track
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId: originalTrackBusId,
          trackId: selectedTrack.id,
          updates: { isMuted: true }
        }
      });
      console.log(`🔇 Muted original track: ${selectedTrack.name}`);

      console.log(`✅ Stem separation complete! ${availableStems.length} stems added to bus`);
      alert(`Stem separation complete! ${availableStems.length} stems added to same bus`);

    } catch (error) {
      console.error('❌ Stem separation failed:', error);
      alert(`Stem separation failed: ${error.message}`);
    } finally {
      setStemProcessing(false);
    }
  }, [selectedTrack, state.buses, dispatch]);

  /**
   * Use this track as input for generation
   * Loads the track's audio or MIDI file as if it was uploaded via the file input
   */
  const handleUseAsInput = useCallback(async () => {
    if (!selectedTrack) return;

    console.log(`🎵 Using track "${selectedTrack.name}" as input for generation`);

    try {
      // Handle MIDI tracks with F0 audio (sine wave from F0 contour)
      if (selectedTrack.type === 'midi' && selectedTrack.f0Audio) {
        console.log('🎵 Loading F0 sine wave audio as input for conditioning');

        // Fetch the F0 audio blob
        const response = await fetch(selectedTrack.f0Audio);
        if (!response.ok) {
          throw new Error(`Failed to fetch F0 audio: ${response.status}`);
        }

        const audioBlob = await response.blob();
        console.log('✅ F0 audio fetched, size:', audioBlob.size);

        // Create filename
        let fileName = selectedTrack.name || 'f0-audio';
        if (!fileName.endsWith('.wav')) {
          fileName += '_f0.wav';
        }

        // Create File object
        const audioFile = new File([audioBlob], fileName, { type: 'audio/wav' });

        // Find which bus this track belongs to
        let trackBusId = null;
        for (const bus of state.buses) {
          if (bus.tracks.find(t => t.id === selectedTrack.id)) {
            trackBusId = bus.id;
            break;
          }
        }

        // Set as uploaded file in state
        dispatch({
          type: 'SET_UPLOADED_FILE',
          payload: {
            file: audioFile,
            fileType: 'audio',
            previewUrl: selectedTrack.f0Audio,
            sourceTrack: {
              busId: trackBusId,
              trackId: selectedTrack.id,
              name: selectedTrack.name
            }
          }
        });

        console.log('✅ F0 sine wave audio loaded as input for conditioning');
        alert(`"${selectedTrack.name}" F0 audio loaded for conditioning`);
        return;
      }

      // Handle MIDI tracks (without F0 audio)
      if (selectedTrack.type === 'midi') {
        console.log('📊 Loading MIDI track as input');

        if (!selectedTrack.midiData || !selectedTrack.midiData.notes || selectedTrack.midiData.notes.length === 0) {
          alert('This MIDI track has no notes. Please add some notes first.');
          return;
        }

        // Notes are stored in seconds at the track's original tempo
        // We need to scale them to match the timeline BPM before creating MIDI file
        const trackTempo = selectedTrack.midiData.tempo || 120;
        const timelineTempo = state.bpm || 120;
        const tempoScale = trackTempo / timelineTempo; // Scale factor for time conversion

        // Scale note times to match timeline tempo
        const scaledNotes = selectedTrack.midiData.notes.map(note => ({
          ...note,
          time: note.time * tempoScale,
          duration: note.duration * tempoScale
        }));

        console.log(`🎹 Creating MIDI for conditioning: Track tempo=${trackTempo}, Timeline tempo=${timelineTempo}, Scale=${tempoScale.toFixed(3)}x`);

        // Create MIDI file at timeline tempo
        const midiData = createMIDIFile(scaledNotes, timelineTempo);
        const midiBlob = new Blob([midiData], { type: 'audio/midi' });

        // Create filename
        let fileName = selectedTrack.name || 'midi-track';
        if (!fileName.endsWith('.mid')) {
          fileName += '.mid';
        }

        // Create File object
        const midiFile = new File([midiBlob], fileName, { type: 'audio/midi' });

        // Find which bus this track belongs to
        let trackBusId = null;
        for (const bus of state.buses) {
          if (bus.tracks.find(t => t.id === selectedTrack.id)) {
            trackBusId = bus.id;
            break;
          }
        }

        // Set as uploaded file in state with source track info for replacement
        dispatch({
          type: 'SET_UPLOADED_FILE',
          payload: {
            file: midiFile,
            fileType: 'midi',
            previewUrl: null,
            sourceTrack: {
              busId: trackBusId,
              trackId: selectedTrack.id,
              midiData: selectedTrack.midiData, // Preserve MIDI data
              name: selectedTrack.name
            }
          }
        });

        console.log('✅ MIDI track loaded as input for generation');
        alert(`"${selectedTrack.name}" loaded as MIDI input for generation`);
        return;
      }

      // Handle audio tracks
      if (!selectedTrack.audioUrl) {
        alert('This track does not have an audio file.');
        return;
      }

      // Fetch the audio file
      const response = await fetch(selectedTrack.audioUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch audio: ${response.status}`);
      }

      const audioBlob = await response.blob();
      console.log('✅ Audio fetched, size:', audioBlob.size);

      // Ensure filename has proper extension
      let fileName = selectedTrack.name || 'audio';

      // Add .wav extension if no extension present
      if (!fileName.match(/\.(wav|mp3|m4a|flac|ogg|aac)$/i)) {
        // Try to determine extension from URL or blob type
        const urlExtension = selectedTrack.audioUrl ? selectedTrack.audioUrl.match(/\.(wav|mp3|m4a|flac|ogg|aac)(\?|$)/i)?.[1] : null;
        if (urlExtension) {
          fileName += `.${urlExtension.toLowerCase()}`;
        } else {
          // Default to .wav
          fileName += '.wav';
        }
      }

      console.log(`📝 Using filename: ${fileName}`);

      // Create a File object from the blob
      const audioFile = new File([audioBlob], fileName, { type: audioBlob.type || 'audio/wav' });

      // Set as uploaded file in state
      dispatch({
        type: 'SET_UPLOADED_FILE',
        payload: {
          file: audioFile,
          fileType: 'audio',
          previewUrl: selectedTrack.audioUrl
        }
      });

      console.log('✅ Track loaded as input for generation');
      alert(`"${selectedTrack.name}" loaded as input for generation`);
    } catch (error) {
      console.error('❌ Failed to use track as input:', error);
      alert(`Failed to load track as input: ${error.message}`);
    }
  }, [selectedTrack, dispatch, state.buses]);

  /**
   * Create a MIDI file from notes
   * Helper function for exporting MIDI tracks
   */
  function createMIDIFile(notes, tempo = 120) {
    // MIDI file structure: Header + Track
    const header = new Uint8Array([
      // MThd (MIDI header chunk)
      0x4D, 0x54, 0x68, 0x64, // "MThd"
      0x00, 0x00, 0x00, 0x06, // Header length (6 bytes)
      0x00, 0x00, // Format type (0 = single track)
      0x00, 0x01, // Number of tracks (1)
      0x00, 0x60  // Ticks per quarter note (96)
    ]);

    // Build track events
    const events = [];
    const ticksPerBeat = 96;

    // Add tempo event
    const microsecondsPerBeat = Math.round(60000000 / tempo);
    events.push({
      tick: 0,
      data: [
        0xFF, 0x51, 0x03, // Tempo meta event
        (microsecondsPerBeat >> 16) & 0xFF,
        (microsecondsPerBeat >> 8) & 0xFF,
        microsecondsPerBeat & 0xFF
      ]
    });

    // Sort notes by time
    const sortedNotes = [...notes].sort((a, b) => a.time - b.time);

    // Convert seconds to ticks using tempo
    // Formula: ticks = seconds * (tempo / 60) * ticksPerBeat
    // This converts: seconds -> beats -> ticks
    const secondsToTicks = (seconds) => {
      const beats = seconds * (tempo / 60); // Convert seconds to beats
      return Math.round(beats * ticksPerBeat); // Convert beats to ticks
    };

    // Add note on/off events
    sortedNotes.forEach(note => {
      const startTick = secondsToTicks(note.time);
      const endTick = secondsToTicks(note.time + note.duration);

      // Note ON event
      events.push({
        tick: startTick,
        data: [0x90, note.note, note.velocity || 100] // Note ON, channel 0
      });

      // Note OFF event
      events.push({
        tick: endTick,
        data: [0x80, note.note, 0x40] // Note OFF, channel 0
      });
    });

    // Sort events by tick
    events.sort((a, b) => a.tick - b.tick);

    // Build track data with delta times
    const trackData = [];
    let lastTick = 0;

    events.forEach(event => {
      const delta = event.tick - lastTick;
      trackData.push(...encodeVariableLength(delta));
      trackData.push(...event.data);
      lastTick = event.tick;
    });

    // End of track
    trackData.push(0x00, 0xFF, 0x2F, 0x00);

    // MTrk chunk
    const track = new Uint8Array([
      // MTrk (MIDI track chunk)
      0x4D, 0x54, 0x72, 0x6B, // "MTrk"
      ...intToBytes(trackData.length, 4), // Track length
      ...trackData
    ]);

    // Combine header and track
    const midiFile = new Uint8Array(header.length + track.length);
    midiFile.set(header, 0);
    midiFile.set(track, header.length);

    return midiFile;
  }

  // Helper: Encode variable-length quantity (for MIDI delta times)
  function encodeVariableLength(value) {
    const bytes = [];
    bytes.push(value & 0x7F);
    value >>= 7;
    while (value > 0) {
      bytes.unshift((value & 0x7F) | 0x80);
      value >>= 7;
    }
    return bytes;
  }

  // Helper: Convert integer to byte array
  function intToBytes(value, numBytes) {
    const bytes = [];
    for (let i = numBytes - 1; i >= 0; i--) {
      bytes.push((value >> (i * 8)) & 0xFF);
    }
    return bytes;
  }

  const handleFXChange = useCallback((fxType, value) => {
    if (!selectedTrack) return;

    // Find the bus this track belongs to
    let trackBusId = null;
    for (const bus of state.buses) {
      if (bus.tracks.find(t => t.id === selectedTrack.id)) {
        trackBusId = bus.id;
        break;
      }
    }

    if (!trackBusId) return;

    // Update track with new FX value
    const updates = {
      fx: {
        ...selectedTrack.fx,
        [fxType]: value
      }
    };

    dispatch({
      type: 'UPDATE_TRACK',
      payload: {
        busId: trackBusId,
        trackId: selectedTrack.id,
        updates
      }
    });

    console.log(`🎛️ Updated ${fxType} to ${value}`);
  }, [selectedTrack, state.buses, dispatch]);

  return (
    <div className={`${styles.sidebar} ${isCollapsed ? styles.collapsed : ''}`}>
      {/* Toggle Button */}
      <div className={styles.toggle} onClick={toggleSidebar}>
        <i className={`fa-solid fa-chevron-left ${styles.toggleIcon}`}></i>
      </div>

      {/* Sidebar Content */}
      <div className={styles.content}>
        {/* Header */}
        <div className={styles.header}>
          <h3>
            <i className="fa-solid fa-info-circle"></i> {selectedBus ? 'Bus Info' : 'Track Info'}
          </h3>
          <button className={styles.closeBtn} onClick={toggleSidebar}>
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>

        {/* Track Info Display */}
        <div className={styles.infoDisplay}>
          {!selectedTrack && !selectedBus ? (
            // Empty State
            <div className={styles.emptyState}>
              <i className="fa-solid fa-music"></i>
              <p>No track or bus selected</p>
              <p className={styles.hint}>Select a track or bus to view its info</p>
            </div>
          ) : selectedBus ? (
            // Bus Info Content
            <div className={styles.infoContent}>
              {/* Bus Name */}
              <div className={styles.section}>
                <h4>
                  <i className="fa-solid fa-layer-group"></i> {selectedBus.name}
                </h4>
                <div className={styles.item}>
                  <span className={styles.label}>Type:</span>
                  <span className={styles.value}>{selectedBus.type}</span>
                </div>
                <div className={styles.item}>
                  <span className={styles.label}>Tracks:</span>
                  <span className={styles.value}>{selectedBus.tracks.length}</span>
                </div>
              </div>

              {/* Bus Icon - Show trumpet for brass bus */}
              {selectedBus.name && selectedBus.name.toLowerCase().includes('brass') && (
                <div className={styles.instrument3DSection}>
                  <div className={styles.instrument3DIcon}>
                    <img
                      src={`${process.env.PUBLIC_URL}/assets/2d/drytp.png`}
                      alt="Brass"
                      style={{
                        width: '100%',
                        height: 'auto',
                        maxHeight: '280px',
                        objectFit: 'contain',
                        filter: 'invert(1)'
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Track List */}
              <div className={styles.section}>
                <h4>
                  <i className="fa-solid fa-list"></i> Tracks in Bus
                </h4>
                {selectedBus.tracks.map((track, index) => (
                  <div key={track.id} className={styles.item} style={{ paddingLeft: '12px' }}>
                    <span className={styles.label}>#{index + 1}:</span>
                    <span className={styles.value}>{track.name || 'Untitled'}</span>
                  </div>
                ))}
              </div>

              {/* Download All Tracks */}
              <div className={styles.section}>
                <h4>
                  <i className="fa-solid fa-download"></i> Actions
                </h4>
                <button className={styles.actionButton} onClick={handleDownloadBus}>
                  <i className="fa-solid fa-file-zipper"></i> Download All Tracks (ZIP)
                </button>
              </div>
            </div>
          ) : (
            // Track Info Content
            <div className={styles.infoContent}>
              {/* 3D Instrument Icon Section */}
              {selectedTrack && (
                <div className={styles.instrument3DSection}>
                  {(() => {
                    // Check track instrument info
                    const instrumentGroup = selectedTrack.instrumentGroup ||
                      selectedTrack.metadata?.params?.instrumentGroup ||
                      selectedTrack.metadata?.instrumentGroup;
                    const instrumentSubgroup = selectedTrack.instrumentSubgroup ||
                      selectedTrack.metadata?.params?.instrumentSubgroup ||
                      selectedTrack.metadata?.instrumentSubgroup;
                    const trackName = selectedTrack.name?.toLowerCase() || '';

                    // Check for trumpet/brass
                    const isTrumpet = instrumentGroup === 'brass' ||
                      instrumentSubgroup === 'trumpet' ||
                      instrumentSubgroup === 'ensemble_brass' ||
                      trackName.includes('trumpet') ||
                      trackName.includes('brass');

                    // Check for electric guitar
                    const isElectricGuitar = instrumentSubgroup === 'electric_guitar' ||
                      trackName.includes('electric guitar') ||
                      trackName.includes('elec guitar') ||
                      (instrumentGroup === 'guitar' && (
                        instrumentSubgroup === 'electric_guitar' ||
                        trackName.includes('electric')
                      ));

                    if (isTrumpet) {
                      return (
                        <>
                          <div className={styles.instrument3DIcon}>
                            <img
                              src={trumpetMuted
                                ? `${process.env.PUBLIC_URL}/assets/2d/mutetp.png`
                                : `${process.env.PUBLIC_URL}/assets/2d/drytp.png`
                              }
                              alt={trumpetMuted ? "Muted Trumpet" : "Trumpet"}
                              style={{
                                width: '100%',
                                height: 'auto',
                                maxHeight: '280px',
                                objectFit: 'contain',
                                filter: 'invert(1)'
                              }}
                            />
                          </div>
                          <div className={styles.instrument3DControls}>
                            <label className={styles.muteToggle}>
                              <input
                                type="checkbox"
                                checked={trumpetMuted}
                                disabled={muteProcessing}
                                onChange={(e) => handleMuteToggle(e.target.checked)}
                              />
                              <span>{muteProcessing ? 'Processing...' : 'Muted'}</span>
                            </label>
                          </div>
                        </>
                      );
                    } else if (isElectricGuitar) {
                      return (
                        <div className={styles.instrument3DIcon}>
                          <img
                            src={`${process.env.PUBLIC_URL}/assets/3d/egtr.png`}
                            alt="Electric Guitar"
                            style={{
                              width: '100%',
                              height: 'auto',
                              maxHeight: '180px',
                              objectFit: 'contain'
                            }}
                          />
                        </div>
                      );
                    } else {
                      // Default to trumpet icon for other instruments
                      return (
                        <div className={styles.instrument3DIcon}>
                          <img
                            src={`${process.env.PUBLIC_URL}/assets/2d/drytp.png`}
                            alt="Instrument"
                            style={{
                              width: '100%',
                              height: 'auto',
                              maxHeight: '280px',
                              objectFit: 'contain',
                              filter: 'invert(1)'
                            }}
                          />
                        </div>
                      );
                    }
                  })()}
                </div>
              )}

              {/* Composite MIDI Track Info */}
              {selectedTrack && selectedTrack.isComposite && selectedTrack.midiData && (
                <div className={styles.section}>
                  <h4>
                    <i className="fa-solid fa-layer-group"></i> Composite MIDI Track
                  </h4>
                  <div className={styles.item}>
                    <span className={styles.label}>Voice Count:</span>
                    <span className={styles.value}>
                      {selectedTrack.midiData.trackCount || selectedBus?.tracks.length || '-'}
                    </span>
                  </div>
                  <div className={styles.item}>
                    <span className={styles.label}>Total Notes:</span>
                    <span className={styles.value}>
                      {selectedTrack.midiData.notes?.length || 0}
                    </span>
                  </div>
                  <div className={styles.item}>
                    <span className={styles.label}>Duration:</span>
                    <span className={styles.value}>
                      {selectedTrack.duration?.toFixed(2) || 0}s
                    </span>
                  </div>

                  {/* Load for Generation Button */}
                  <button
                    className={styles.downloadBtn}
                    onClick={handleUseAsInput}
                    style={{ marginTop: '12px', background: 'var(--gradient-variant-3)' }}
                  >
                    <i className="fa-solid fa-arrow-right"></i> Load for Generation
                  </button>
                  <div style={{ fontSize: '11px', color: '#888', marginTop: '8px', lineHeight: '1.4' }}>
                    Use this multitrack MIDI as input for generating new audio
                  </div>

                  {/* Split Voices Button */}
                  <button
                    className={styles.actionButton}
                    onClick={handleSplitVoices}
                    style={{ marginTop: '12px', background: 'var(--gradient-variant-1)' }}
                  >
                    <i className="fa-solid fa-layer-group"></i> Split Voices
                  </button>
                  <div style={{ fontSize: '11px', color: '#888', marginTop: '8px', lineHeight: '1.4' }}>
                    Expand to view and edit individual MIDI voices
                  </div>
                </div>
              )}

              {/* Instrument Info (for generated tracks) */}
              {isGeneratedTrack && selectedTrack.metadata?.params && (
                <div className={styles.section}>
                  <h4>
                    <i className="fa-solid fa-guitar"></i> Instrument
                  </h4>
                  <div className={styles.item}>
                    <span className={styles.label}>Group:</span>
                    <span className={styles.value}>
                      {selectedTrack.metadata.params.instrumentGroup || selectedTrack.instrumentGroup || '-'}
                    </span>
                  </div>
                  <div className={styles.item}>
                    <span className={styles.label}>Subgroup:</span>
                    <span className={styles.value}>
                      {selectedTrack.metadata.params.instrumentSubgroup || selectedTrack.instrumentSubgroup || '-'}
                    </span>
                  </div>

                  {/* Clarify Button - Improve timbre quality */}
                  {(selectedTrack.metadata.params.instrumentGroup || selectedTrack.instrumentGroup) && (
                    <>
                      <button
                        className={styles.actionButton}
                        onClick={handleClarify}
                        disabled={clarifying}
                        style={{ marginTop: '12px', background: 'linear-gradient(135deg, #9c82c8 0%, #667eea 100%)' }}
                      >
                        {clarifying ? (
                          <>
                            <i className="fa-solid fa-spinner fa-spin"></i> Clarifying...
                          </>
                        ) : (
                          <>
                            <i className="fa-solid fa-wand-magic-sparkles"></i> Clarify Timbre
                          </>
                        )}
                      </button>
                      <div style={{ fontSize: '11px', color: '#888', marginTop: '8px', lineHeight: '1.4' }}>
                        Improves instrument timbre quality using AI
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* Input Files Info (for generated tracks) */}
              {isGeneratedTrack && selectedTrack.metadata?.inputFiles && (
                <div className={styles.section}>
                  <h4>
                    <i className="fa-solid fa-file-audio"></i> Input Files
                  </h4>

                  {/* Voice Number */}
                  {selectedTrack.metadata.voiceNumber !== undefined && (
                    <div className={styles.item}>
                      <span className={styles.label}>Voice:</span>
                      <span className={styles.value}>
                        {selectedTrack.metadata.voiceNumber}
                      </span>
                    </div>
                  )}

                  {/* Original Input (MIDI or WAV) */}
                  {(selectedTrack.metadata.inputFiles.midiPath || selectedTrack.metadata.inputFiles.renderPath) && (
                    <div className={styles.item} style={{ marginTop: '8px' }}>
                      <span className={styles.label} style={{ fontWeight: 'bold' }}>Original Input:</span>
                    </div>
                  )}

                  {selectedTrack.metadata.inputFiles.midiPath && (
                    <div className={styles.item} style={{ paddingLeft: '12px' }}>
                      <span className={styles.label}>MIDI:</span>
                      <span className={styles.value} style={{ fontSize: '11px' }}>
                        <a href={selectedTrack.metadata.inputFiles.midiPath} download style={{ color: '#667eea', textDecoration: 'none' }}>
                          <i className="fa-solid fa-download"></i> Download
                        </a>
                      </span>
                    </div>
                  )}

                  {selectedTrack.metadata.inputFiles.renderPath && (
                    <div className={styles.item} style={{ paddingLeft: '12px' }}>
                      <span className={styles.label}>WAV (FluidSynth):</span>
                      <span className={styles.value} style={{ fontSize: '11px' }}>
                        <a href={selectedTrack.metadata.inputFiles.renderPath} download style={{ color: '#667eea', textDecoration: 'none' }}>
                          <i className="fa-solid fa-download"></i> Download
                        </a>
                      </span>
                    </div>
                  )}

                  {/* Piano Roll MIDI Input */}
                  {selectedTrack.metadata.inputFiles.basicPitchMidiPath && (
                    <>
                      <div className={styles.item} style={{ marginTop: '8px' }}>
                        <span className={styles.label} style={{ fontWeight: 'bold' }}>Piano Roll Source:</span>
                      </div>
                      <div className={styles.item} style={{ paddingLeft: '12px' }}>
                        <span className={styles.label}>Basic Pitch MIDI:</span>
                        <span className={styles.value} style={{ fontSize: '11px' }}>
                          <a href={selectedTrack.metadata.inputFiles.basicPitchMidiPath} download style={{ color: '#667eea', textDecoration: 'none' }}>
                            <i className="fa-solid fa-download"></i> Download
                          </a>
                        </span>
                      </div>
                    </>
                  )}

                  {/* Master MIDI (for scene changes) */}
                  {selectedTrack.metadata.inputFiles.masterMidiPath && (
                    <>
                      <div className={styles.item} style={{ marginTop: '8px' }}>
                        <span className={styles.label} style={{ fontWeight: 'bold' }}>Master/Scene MIDI:</span>
                      </div>
                      <div className={styles.item} style={{ paddingLeft: '12px' }}>
                        <span className={styles.label}>Concatenated MIDI:</span>
                        <span className={styles.value} style={{ fontSize: '11px' }}>
                          <a href={selectedTrack.metadata.inputFiles.masterMidiPath} download style={{ color: '#667eea', textDecoration: 'none' }}>
                            <i className="fa-solid fa-download"></i> Download
                          </a>
                        </span>
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* Generation Parameters (for generated tracks) */}
              {isGeneratedTrack && selectedTrack.metadata?.params && (
                <div className={styles.section}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <button
                      className={styles.paramsToggle}
                      onClick={() => setShowParams(!showParams)}
                      style={{ flex: 1, textAlign: 'left' }}
                    >
                      <i className={`fa-solid fa-${showParams ? 'chevron-down' : 'chevron-right'}`}></i>
                      Generation Parameters
                    </button>

                    {/* Feedback Buttons */}
                    <div style={{ display: 'flex', gap: '8px', marginLeft: '8px' }}>
                      <button
                        onClick={() => handleFeedback(true)}
                        disabled={feedbackSubmitted !== null}
                        style={{
                          background: feedbackSubmitted === 'like' ? '#4CAF50' : '#2a2a3e',
                          color: feedbackSubmitted === 'like' ? 'white' : '#888',
                          border: '1px solid #444',
                          borderRadius: '4px',
                          padding: '4px 8px',
                          cursor: feedbackSubmitted ? 'default' : 'pointer',
                          fontSize: '14px',
                          transition: 'all 0.2s ease',
                          opacity: feedbackSubmitted && feedbackSubmitted !== 'like' ? 0.5 : 1
                        }}
                        title="Like these parameters"
                      >
                        <i className="fa-solid fa-thumbs-up"></i>
                      </button>
                      <button
                        onClick={() => handleFeedback(false)}
                        disabled={feedbackSubmitted !== null}
                        style={{
                          background: feedbackSubmitted === 'dislike' ? '#f44336' : '#2a2a3e',
                          color: feedbackSubmitted === 'dislike' ? 'white' : '#888',
                          border: '1px solid #444',
                          borderRadius: '4px',
                          padding: '4px 8px',
                          cursor: feedbackSubmitted ? 'default' : 'pointer',
                          fontSize: '14px',
                          transition: 'all 0.2s ease',
                          opacity: feedbackSubmitted && feedbackSubmitted !== 'dislike' ? 0.5 : 1
                        }}
                        title="Dislike these parameters"
                      >
                        <i className="fa-solid fa-thumbs-down"></i>
                      </button>
                    </div>
                  </div>

                  {showParams && (
                    <div className={styles.paramsContent}>
                      {Object.entries(selectedTrack.metadata.params).map(([key, value]) => {
                        // Skip file keys and internal properties
                        if (key.includes('Key') || key.includes('audioFile') || key.includes('conditioningAudio')) {
                          return null;
                        }

                        // Format the value for display
                        let displayValue = value;
                        if (typeof value === 'boolean') {
                          displayValue = value ? 'Yes' : 'No';
                        } else if (typeof value === 'number') {
                          displayValue = value.toFixed(2);
                        } else if (typeof value === 'object') {
                          displayValue = JSON.stringify(value, null, 2);
                        }

                        return (
                          <div key={key} className={styles.paramItem}>
                            <span className={styles.paramLabel}>{key}:</span>
                            <span className={styles.paramValue}>{String(displayValue)}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* Download Buttons */}
              <div className={styles.section}>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <GlassButtonWrapper
                    className={styles.downloadBtn}
                    onClick={handleDownload}
                    style={{ flex: 1 }}
                  >
                    <i className="fa-solid fa-download"></i> Download Track
                  </GlassButtonWrapper>

                  {/* Like/Dislike buttons - only for generated tracks */}
                  {isGeneratedTrack && selectedTrack.metadata?.params && (
                    <div style={{ display: 'flex', gap: '4px' }}>
                      <button
                        className={styles.feedbackBtn}
                        onClick={() => handleFeedback(true)}
                        disabled={feedbackSubmitted === 'like'}
                        title="Like this generation"
                        style={{
                          background: feedbackSubmitted === 'like' ? 'var(--gradient-primary)' : 'var(--color-bg-light)',
                          border: '1px solid var(--color-bg-lighter)',
                          borderRadius: '6.4px',
                          width: '40px',
                          height: '40px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: feedbackSubmitted === 'like' ? 'default' : 'pointer',
                          transition: 'all 0.2s',
                          color: feedbackSubmitted === 'like' ? '#fff' : '#aaa'
                        }}
                      >
                        <i className="fa-solid fa-thumbs-up"></i>
                      </button>
                      <button
                        className={styles.feedbackBtn}
                        onClick={() => handleFeedback(false)}
                        disabled={feedbackSubmitted === 'dislike'}
                        title="Dislike this generation"
                        style={{
                          background: feedbackSubmitted === 'dislike' ? 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)' : 'var(--color-bg-light)',
                          border: '1px solid var(--color-bg-lighter)',
                          borderRadius: '6.4px',
                          width: '40px',
                          height: '40px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: feedbackSubmitted === 'dislike' ? 'default' : 'pointer',
                          transition: 'all 0.2s',
                          color: feedbackSubmitted === 'dislike' ? '#fff' : '#aaa'
                        }}
                      >
                        <i className="fa-solid fa-thumbs-down"></i>
                      </button>
                    </div>
                  )}
                </div>

                {/* Download with FX button - only for audio tracks (not MIDI) */}
                {selectedTrack.type !== 'midi' && (
                  <>
                    <GlassButtonWrapper
                      className={styles.downloadBtn}
                      onClick={handleDownloadWithFX}
                      style={{
                        marginTop: '8px',
                        background: 'var(--gradient-variant-1)'
                      }}
                    >
                      <i className="fa-solid fa-wand-magic-sparkles"></i> Download with FX
                    </GlassButtonWrapper>

                    {/* Render with FX button - replaces track on timeline */}
                    <GlassButtonWrapper
                      className={styles.downloadBtn}
                      onClick={handleRenderWithFX}
                      style={{
                        marginTop: '8px',
                        background: 'var(--gradient-variant-2)'
                      }}
                    >
                      <i className="fa-solid fa-layer-group"></i> Render with FX
                    </GlassButtonWrapper>

                    {/* Plugin FX Controls */}
                    <button
                      className={styles.fxToggle}
                      onClick={() => setShowPluginFX(!showPluginFX)}
                      style={{ marginTop: '8px' }}
                    >
                      <i className="fa-solid fa-plug"></i> Plugin FX Settings
                      <i className={`fa-solid fa-chevron-down ${styles.chevron} ${showPluginFX ? styles.up : ''}`}></i>
                    </button>

                    {showPluginFX && (
                      <div className={styles.fxPanel} style={{ marginTop: '8px' }}>
                        {/* RC-20 Retro Color */}
                        <div style={{ marginBottom: '16px' }}>
                          <label style={{ display: 'flex', alignItems: 'center', marginBottom: '8px', cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={rc20Enabled}
                              onChange={(e) => setRc20Enabled(e.target.checked)}
                              style={{ marginRight: '8px' }}
                            />
                            <strong>RC-20 Retro Color (Lo-Fi)</strong>
                          </label>

                          {rc20Enabled && (
                            <div style={{ paddingLeft: '16px' }}>
                              <div className={styles.fxControl}>
                                <label>
                                  <span>Magnitude:</span>
                                  <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="1"
                                    value={rc20Settings.magnitude}
                                    onChange={(e) => setRc20Settings({ ...rc20Settings, magnitude: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{rc20Settings.magnitude}</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Noise:</span>
                                  <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="1"
                                    value={rc20Settings.noise}
                                    onChange={(e) => setRc20Settings({ ...rc20Settings, noise: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{rc20Settings.noise}</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Wobble:</span>
                                  <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="1"
                                    value={rc20Settings.wobble}
                                    onChange={(e) => setRc20Settings({ ...rc20Settings, wobble: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{rc20Settings.wobble}</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Distortion:</span>
                                  <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="1"
                                    value={rc20Settings.distortion}
                                    onChange={(e) => setRc20Settings({ ...rc20Settings, distortion: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{rc20Settings.distortion}</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Mag Amount:</span>
                                  <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="1"
                                    value={rc20Settings.magnitudeAmount}
                                    onChange={(e) => setRc20Settings({ ...rc20Settings, magnitudeAmount: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{rc20Settings.magnitudeAmount}</span>
                                </label>
                              </div>
                            </div>
                          )}
                        </div>

                        {/* SpecCraft */}
                        <div>
                          <label style={{ display: 'flex', alignItems: 'center', marginBottom: '8px', cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={speccraftEnabled}
                              onChange={(e) => setSpeccraftEnabled(e.target.checked)}
                              style={{ marginRight: '8px' }}
                            />
                            <strong>SpecCraft (Spectral Compression)</strong>
                          </label>

                          {speccraftEnabled && (
                            <div style={{ paddingLeft: '16px' }}>
                              <div className={styles.fxControl}>
                                <label>
                                  <span>Threshold:</span>
                                  <input
                                    type="range"
                                    min="-60"
                                    max="0"
                                    step="1"
                                    value={speccraftSettings.threshold}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, threshold: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{speccraftSettings.threshold} dB</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Ratio:</span>
                                  <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="1"
                                    value={speccraftSettings.ratio}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, ratio: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{speccraftSettings.ratio}%</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Attack:</span>
                                  <input
                                    type="range"
                                    min="0.1"
                                    max="50"
                                    step="0.1"
                                    value={speccraftSettings.attack}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, attack: parseFloat(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{speccraftSettings.attack} ms</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Release:</span>
                                  <input
                                    type="range"
                                    min="10"
                                    max="1000"
                                    step="10"
                                    value={speccraftSettings.release}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, release: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{speccraftSettings.release} ms</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Makeup Gain:</span>
                                  <input
                                    type="range"
                                    min="-12"
                                    max="24"
                                    step="1"
                                    value={speccraftSettings.makeup}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, makeup: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{speccraftSettings.makeup} dB</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Knee:</span>
                                  <input
                                    type="range"
                                    min="0"
                                    max="12"
                                    step="0.5"
                                    value={speccraftSettings.knee}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, knee: parseFloat(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{speccraftSettings.knee} dB</span>
                                </label>
                              </div>

                              <div className={styles.fxControl}>
                                <label>
                                  <span>Slope:</span>
                                  <input
                                    type="range"
                                    min="0"
                                    max="12"
                                    step="1"
                                    value={speccraftSettings.slope}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, slope: parseInt(e.target.value) })}
                                  />
                                  <span className={styles.fxValue}>{speccraftSettings.slope} dB</span>
                                </label>
                              </div>

                              <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                                <strong style={{ fontSize: '11px', opacity: 0.7 }}>SPECTRAL SHAPING BANDS</strong>
                              </div>

                              {/* Band 1 - Low-Mid Boost (AI muddiness) */}
                              <div style={{ marginTop: '8px' }}>
                                <label style={{ display: 'flex', alignItems: 'center', marginBottom: '4px', fontSize: '11px' }}>
                                  <input
                                    type="checkbox"
                                    checked={speccraftSettings.band1Enabled}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band1Enabled: e.target.checked })}
                                    style={{ marginRight: '6px', width: '14px', height: '14px' }}
                                  />
                                  Band 1: Low-Mid (AI muddiness cleanup)
                                </label>
                                {speccraftSettings.band1Enabled && (
                                  <>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Freq:</span>
                                        <input
                                          type="range"
                                          min="200"
                                          max="800"
                                          step="10"
                                          value={speccraftSettings.band1Freq}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band1Freq: parseInt(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band1Freq} Hz</span>
                                      </label>
                                    </div>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Gain:</span>
                                        <input
                                          type="range"
                                          min="-12"
                                          max="12"
                                          step="0.5"
                                          value={speccraftSettings.band1Gain}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band1Gain: parseFloat(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band1Gain} dB</span>
                                      </label>
                                    </div>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Q:</span>
                                        <input
                                          type="range"
                                          min="0.5"
                                          max="5"
                                          step="0.1"
                                          value={speccraftSettings.band1Q}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band1Q: parseFloat(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band1Q}</span>
                                      </label>
                                    </div>
                                  </>
                                )}
                              </div>

                              {/* Band 2 - Presence Cleanup */}
                              <div style={{ marginTop: '8px' }}>
                                <label style={{ display: 'flex', alignItems: 'center', marginBottom: '4px', fontSize: '11px' }}>
                                  <input
                                    type="checkbox"
                                    checked={speccraftSettings.band2Enabled}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band2Enabled: e.target.checked })}
                                    style={{ marginRight: '6px', width: '14px', height: '14px' }}
                                  />
                                  Band 2: Presence (metallic harshness)
                                </label>
                                {speccraftSettings.band2Enabled && (
                                  <>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Freq:</span>
                                        <input
                                          type="range"
                                          min="2000"
                                          max="4000"
                                          step="50"
                                          value={speccraftSettings.band2Freq}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band2Freq: parseInt(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band2Freq} Hz</span>
                                      </label>
                                    </div>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Gain:</span>
                                        <input
                                          type="range"
                                          min="-12"
                                          max="12"
                                          step="0.5"
                                          value={speccraftSettings.band2Gain}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band2Gain: parseFloat(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band2Gain} dB</span>
                                      </label>
                                    </div>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Q:</span>
                                        <input
                                          type="range"
                                          min="0.5"
                                          max="5"
                                          step="0.1"
                                          value={speccraftSettings.band2Q}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band2Q: parseFloat(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band2Q}</span>
                                      </label>
                                    </div>
                                  </>
                                )}
                              </div>

                              {/* Band 3 - High-Mid Taming */}
                              <div style={{ marginTop: '8px' }}>
                                <label style={{ display: 'flex', alignItems: 'center', marginBottom: '4px', fontSize: '11px' }}>
                                  <input
                                    type="checkbox"
                                    checked={speccraftSettings.band3Enabled}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band3Enabled: e.target.checked })}
                                    style={{ marginRight: '6px', width: '14px', height: '14px' }}
                                  />
                                  Band 3: High-Mid (digital sizzle)
                                </label>
                                {speccraftSettings.band3Enabled && (
                                  <>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Freq:</span>
                                        <input
                                          type="range"
                                          min="4000"
                                          max="8000"
                                          step="100"
                                          value={speccraftSettings.band3Freq}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band3Freq: parseInt(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band3Freq} Hz</span>
                                      </label>
                                    </div>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Gain:</span>
                                        <input
                                          type="range"
                                          min="-12"
                                          max="12"
                                          step="0.5"
                                          value={speccraftSettings.band3Gain}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band3Gain: parseFloat(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band3Gain} dB</span>
                                      </label>
                                    </div>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Q:</span>
                                        <input
                                          type="range"
                                          min="0.5"
                                          max="5"
                                          step="0.1"
                                          value={speccraftSettings.band3Q}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band3Q: parseFloat(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band3Q}</span>
                                      </label>
                                    </div>
                                  </>
                                )}
                              </div>

                              {/* Band 4 - Boxy AI artifacts */}
                              <div style={{ marginTop: '8px' }}>
                                <label style={{ display: 'flex', alignItems: 'center', marginBottom: '4px', fontSize: '11px' }}>
                                  <input
                                    type="checkbox"
                                    checked={speccraftSettings.band4Enabled}
                                    onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band4Enabled: e.target.checked })}
                                    style={{ marginRight: '6px', width: '14px', height: '14px' }}
                                  />
                                  Band 4: Boxy (AI vocal artifacts)
                                </label>
                                {speccraftSettings.band4Enabled && (
                                  <>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Freq:</span>
                                        <input
                                          type="range"
                                          min="300"
                                          max="600"
                                          step="10"
                                          value={speccraftSettings.band4Freq}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band4Freq: parseInt(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band4Freq} Hz</span>
                                      </label>
                                    </div>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Gain:</span>
                                        <input
                                          type="range"
                                          min="-12"
                                          max="12"
                                          step="0.5"
                                          value={speccraftSettings.band4Gain}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band4Gain: parseFloat(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band4Gain} dB</span>
                                      </label>
                                    </div>
                                    <div className={styles.fxControl}>
                                      <label>
                                        <span style={{ fontSize: '10px' }}>Q:</span>
                                        <input
                                          type="range"
                                          min="0.5"
                                          max="5"
                                          step="0.1"
                                          value={speccraftSettings.band4Q}
                                          onChange={(e) => setSpeccraftSettings({ ...speccraftSettings, band4Q: parseFloat(e.target.value) })}
                                        />
                                        <span className={styles.fxValue}>{speccraftSettings.band4Q}</span>
                                      </label>
                                    </div>
                                  </>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Use as Input (for uploaded tracks and stems) */}
              {(isUploadedTrack || (!isGeneratedTrack && !isVideoAudioTrack)) && (
                <div className={styles.section}>
                  <h4>
                    <i className="fa-solid fa-upload"></i> Use as Input
                  </h4>
                  <button
                    className={styles.downloadBtn}
                    onClick={handleUseAsInput}
                    style={{ background: 'var(--gradient-variant-3)' }}
                  >
                    <i className="fa-solid fa-arrow-right"></i> Load for Generation
                  </button>
                  <div style={{ fontSize: '11px', color: '#888', marginTop: '8px', lineHeight: '1.4' }}>
                    Use this track as input for generating new audio
                  </div>
                </div>
              )}

              {/* Stem Separation (for uploaded tracks) */}
              {(isUploadedTrack || (!isGeneratedTrack && !isVideoAudioTrack)) && (
                <div className={styles.section}>
                  <h4>
                    <i className="fa-solid fa-layer-group"></i> Stem Separation
                  </h4>
                  <button
                    className={styles.stemBtn}
                    onClick={handleSeparateIntoStems}
                    disabled={stemProcessing}
                  >
                    {stemProcessing ? (
                      <>
                        <i className="fa-solid fa-spinner fa-spin"></i> Separating...
                      </>
                    ) : (
                      <>
                        <i className="fa-solid fa-layer-group"></i> Separate into 6 Stems
                      </>
                    )}
                  </button>
                  <div style={{ fontSize: '11px', color: '#888', marginTop: '8px', lineHeight: '1.4' }}>
                    Separates track into: vocals, drums, bass, guitar, piano, and other
                  </div>
                </div>
              )}

              {/* Regenerate Button (for generated tracks only) */}
              {isGeneratedTrack && (
                <div className={styles.section}>
                  <button
                    className={styles.regenerateBtn}
                    onClick={handleRegenerate}
                    disabled={regenerating}
                  >
                    {regenerating ? (
                      <>
                        <i className="fa-solid fa-spinner fa-spin"></i> Regenerating...
                      </>
                    ) : (
                      <>
                        <i className="fa-solid fa-rotate"></i> Regenerate
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Inpaint Button (for generated tracks only) */}
              {isGeneratedTrack && (
                <div className={styles.section}>
                  <button
                    className={`${styles.inpaintBtn} ${state.inpaintMode?.enabled && state.inpaintMode?.trackId === selectedTrack.id ? styles.active : ''}`}
                    onClick={toggleInpaintMode}
                    disabled={regenerating}
                  >
                    {state.inpaintMode?.enabled && state.inpaintMode?.trackId === selectedTrack.id ? (
                      <>
                        <i className="fa-solid fa-times"></i> Cancel Inpaint Mode
                      </>
                    ) : (
                      <>
                        <i className="fa-solid fa-paintbrush"></i> Inpaint Region
                      </>
                    )}
                  </button>

                  {/* Show selection details if a region has been selected */}
                  {hasInpaintSelection && inpaintSelectionTimes && (
                    <>
                      <div className={styles.inpaintHint} style={{ background: 'rgba(102, 126, 234, 0.1)', borderLeftColor: '#667eea', color: '#667eea' }}>
                        <i className="fa-solid fa-check-circle"></i>
                        <span>
                          Selected: {inpaintSelectionTimes.start.toFixed(2)}s - {inpaintSelectionTimes.end.toFixed(2)}s
                          ({inpaintSelectionTimes.duration.toFixed(2)}s duration)
                        </span>
                      </div>
                      <button
                        className={styles.regenerateBtn}
                        onClick={confirmInpaint}
                        disabled={regenerating}
                        style={{ marginTop: '10px' }}
                      >
                        {regenerating ? (
                          <>
                            <i className="fa-solid fa-spinner fa-spin"></i> Inpainting...
                          </>
                        ) : (
                          <>
                            <i className="fa-solid fa-check"></i> Confirm Inpaint
                          </>
                        )}
                      </button>
                      <button
                        className={styles.downloadBtn}
                        onClick={cancelInpaintSelection}
                        disabled={regenerating}
                        style={{ marginTop: '10px', background: 'linear-gradient(135deg, #6c757d 0%, #495057 100%)' }}
                      >
                        <i className="fa-solid fa-times"></i> Cancel Selection
                      </button>
                    </>
                  )}

                  {/* Show hint if in inpaint mode but no selection yet */}
                  {state.inpaintMode?.enabled && state.inpaintMode?.trackId === selectedTrack.id && !hasInpaintSelection && (
                    <div className={styles.inpaintHint}>
                      <i className="fa-solid fa-info-circle"></i> Drag on waveform to select region
                    </div>
                  )}
                </div>
              )}

              {/* Version Control (for any track with multiple versions) */}
              {hasMultipleVersions && (
                <div className={styles.section}>
                  <h4>
                    <i className="fa-solid fa-clock-rotate-left"></i> Version History
                  </h4>
                  <div className={styles.versionControls}>
                    <button
                      className={styles.versionBtn}
                      onClick={previousVersion}
                      disabled={currentVersionIndex === 0}
                      title="Previous version"
                    >
                      <i className="fa-solid fa-chevron-left"></i>
                    </button>
                    <span className={styles.versionLabel}>
                      {versions[currentVersionIndex]?.name || `Version ${currentVersionIndex + 1}`} ({currentVersionIndex + 1}/{versions.length})
                    </span>
                    <button
                      className={styles.versionBtn}
                      onClick={nextVersion}
                      disabled={currentVersionIndex === versions.length - 1}
                      title="Next version"
                    >
                      <i className="fa-solid fa-chevron-right"></i>
                    </button>
                  </div>
                </div>
              )}

              {/* Stem Separation (for video audio tracks) */}
              {isVideoAudioTrack && (
                <div className={styles.section}>
                  <h4>
                    <i className="fa-solid fa-layer-group"></i> Stem Separation
                  </h4>
                  <button
                    className={styles.stemBtn}
                    onClick={handleRemoveMusic}
                    disabled={stemProcessing}
                    style={{ marginBottom: '10px' }}
                  >
                    {stemProcessing ? (
                      <>
                        <i className="fa-solid fa-spinner fa-spin"></i> Separating...
                      </>
                    ) : (
                      <>
                        <i className="fa-solid fa-microphone"></i> Remove Music
                      </>
                    )}
                  </button>
                  <button
                    className={styles.stemBtn}
                    onClick={handleRemoveDialogue}
                    disabled={stemProcessing}
                  >
                    {stemProcessing ? (
                      <>
                        <i className="fa-solid fa-spinner fa-spin"></i> Separating...
                      </>
                    ) : (
                      <>
                        <i className="fa-solid fa-music"></i> Remove Dialogue
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Track FX Panel */}
              <div className={styles.section}>
                <button
                  className={styles.fxToggle}
                  onClick={() => setShowFX(!showFX)}
                >
                  <i className="fa-solid fa-sliders"></i> Track FX
                  <i className={`fa-solid fa-chevron-down ${styles.chevron} ${showFX ? styles.up : ''}`}></i>
                </button>
                {showFX && (
                  <div className={styles.fxPanel}>
                    <div className={styles.fxControl}>
                      <label>
                        <span>Reverb:</span>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={selectedTrack.fx?.reverb || 0}
                          onChange={(e) => handleFXChange('reverb', parseFloat(e.target.value))}
                        />
                        <span className={styles.fxValue}>
                          {(selectedTrack.fx?.reverb || 0).toFixed(2)}
                        </span>
                      </label>
                    </div>

                    <div className={styles.fxControl}>
                      <label>
                        <span>Fade In:</span>
                        <input
                          type="range"
                          min="0"
                          max="10"
                          step="0.1"
                          value={selectedTrack.fx?.fadeIn || 0.2}
                          onChange={(e) => handleFXChange('fadeIn', parseFloat(e.target.value))}
                        />
                        <span className={styles.fxValue}>
                          {(selectedTrack.fx?.fadeIn || 0.2).toFixed(1)}s
                        </span>
                      </label>
                    </div>

                    <div className={styles.fxControl}>
                      <label>
                        <span>Fade Out:</span>
                        <input
                          type="range"
                          min="0"
                          max="10"
                          step="0.1"
                          value={selectedTrack.fx?.fadeOut || 1.0}
                          onChange={(e) => handleFXChange('fadeOut', parseFloat(e.target.value))}
                        />
                        <span className={styles.fxValue}>
                          {(selectedTrack.fx?.fadeOut || 1.0).toFixed(1)}s
                        </span>
                      </label>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

TrackInfoSidebar.displayName = 'TrackInfoSidebar';

export default TrackInfoSidebar;
