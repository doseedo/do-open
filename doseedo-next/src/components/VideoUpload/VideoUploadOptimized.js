import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { useVideoProcessing } from '../../hooks/useVideoProcessing';
import styles from './VideoUpload.module.css';

/**
 * VideoUploadOptimized - Optimized video upload component
 *
 * Key Optimizations:
 * 1. CSS Modules for scoped styling
 * 2. GPU-accelerated transforms (translateZ, scale)
 * 3. Memoization with useCallback and useMemo
 * 4. Cleaner component structure
 * 5. Responsive design with media queries
 * 6. Integrated video processing with scene detection
 * 7. Synced playback with timeline (play, pause, seek)
 */

const VideoUploadOptimized = React.memo(() => {
  const { state, dispatch } = useApp();
  const { isProcessing, processingStatus, processingError, processVideo, clearVideo } = useVideoProcessing();

  const [videoFile, setVideoFile] = useState(null);
  const [showVideo, setShowVideo] = useState(false);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [useWhisper, setUseWhisper] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [videoFrames, setVideoFrames] = useState([]);

  // Ref for video element to control playback
  const videoRef = useRef(null);
  const isSeekingRef = useRef(false);
  const frameCanvasRef = useRef(null);

  // Common function to process video file
  const processVideoFile = useCallback(async (file) => {
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('video/')) {
      alert('Please upload a video file');
      return;
    }

    setVideoFile(file);

    // Create preview URL for local playback
    const previewUrl = URL.createObjectURL(file);
    setVideoPreviewUrl(previewUrl);
    setShowVideo(true);

    console.log('🎥 Video uploaded:', {
      name: file.name,
      size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
      type: file.type
    });

    // Start video processing (scene detection + audio extraction)
    try {
      await processVideo(file, useWhisper);
      console.log('✅ Video processing complete!');
    } catch (error) {
      console.error('❌ Video processing failed:', error);
      alert(`Video processing failed: ${error.message}`);
    }
  }, [processVideo, useWhisper]);

  // Handle video file upload from input
  const handleVideoUpload = useCallback(async (e) => {
    const file = e.target.files[0];
    await processVideoFile(file);
  }, [processVideoFile]);

  // Handle exit/clear video
  const handleExitVideo = useCallback(() => {
    // Revoke preview URL to free memory
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
    }

    setShowVideo(false);
    setVideoFile(null);
    setVideoPreviewUrl(null);

    dispatch({
      type: 'SET_UPLOADED_FILE',
      payload: {
        file: null,
        fileType: null,
        previewUrl: null
      }
    });

    // Clear video processing state
    clearVideo();

    console.log('❌ Video cleared');
  }, [dispatch, videoPreviewUrl, clearVideo]);

  // Handle export (placeholder)
  const handleExport = useCallback(() => {
    if (!videoFile) return;

    console.log('📥 Exporting video:', videoFile.name);
    alert('Export functionality coming soon!');
  }, [videoFile]);

  // Drag and drop handlers
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      await processVideoFile(file);
    }
  }, [processVideoFile]);

  // Determine visibility states
  const showCloseButton = useMemo(() => showVideo, [showVideo]);
  const showExportButton = useMemo(() => showVideo, [showVideo]);

  // Sync video playback position with timeline playhead (seek)
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !showVideo) return;

    // Only seek if the difference is significant (> 0.2s) to avoid constant seeking
    const timeDiff = Math.abs(video.currentTime - state.playheadPosition);
    if (timeDiff > 0.2) {
      // Set seeking flag to prevent play/pause conflicts during seek
      isSeekingRef.current = true;

      // Store the play state before seeking
      const wasPlaying = !video.paused;

      video.currentTime = state.playheadPosition;
      console.log(`🎯 Video seeked to ${state.playheadPosition.toFixed(2)}s`);

      // After seeking, restore play state if needed
      setTimeout(() => {
        isSeekingRef.current = false;

        // Resume playback if timeline is playing
        if (state.isPlaying && !wasPlaying) {
          video.play().catch(err => {
            console.warn('⚠️ Video play after seek failed:', err);
          });
        }
      }, 50);
    }
  }, [state.playheadPosition, showVideo, state.isPlaying]);

  // Sync video playback with timeline play/pause state
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !showVideo) return;

    // Skip if currently seeking
    if (isSeekingRef.current) {
      console.log('⏳ Skipping play/pause update during seek');
      return;
    }

    if (state.isPlaying) {
      // Play video if not already playing
      if (video.paused) {
        video.play().catch(err => {
          console.warn('⚠️ Video play failed:', err);
        });
        console.log('▶️ Video playing (synced with timeline)');
      }
    } else {
      // Pause video if not already paused
      if (!video.paused) {
        video.pause();
        console.log('⏸️ Video paused (synced with timeline)');
      }
    }
  }, [state.isPlaying, showVideo]);

  // Reset video when cleared
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !showVideo) return;

    // Reset to beginning when playhead is at 0
    if (state.playheadPosition === 0 && !state.isPlaying) {
      video.currentTime = 0;
    }
  }, [state.playheadPosition, state.isPlaying, showVideo]);

  // Sync state when video controls are used directly
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !showVideo) return;

    const handlePlay = () => {
      if (!state.isPlaying && !isSeekingRef.current) {
        console.log('▶️ Video played from controls - syncing state');
        dispatch({ type: 'SET_PLAYING', payload: true });
      }
    };

    const handlePause = () => {
      if (state.isPlaying && !isSeekingRef.current) {
        console.log('⏸️ Video paused from controls - syncing state');
        dispatch({ type: 'SET_PLAYING', payload: false });
      }
    };

    const handleSeeked = () => {
      const timeDiff = Math.abs(video.currentTime - state.playheadPosition);
      if (timeDiff > 0.2) {
        console.log(`🎯 Video seeked from controls to ${video.currentTime.toFixed(2)}s - syncing state`);
        dispatch({ type: 'SET_PLAYHEAD_POSITION', payload: video.currentTime });
      }
    };

    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('seeked', handleSeeked);

    return () => {
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('seeked', handleSeeked);
    };
  }, [showVideo, state.isPlaying, state.playheadPosition, dispatch]);

  // Extract video frames for timeline preview
  useEffect(() => {
    if (!videoPreviewUrl || !showVideo || !state.video?.duration) return;

    const extractFrames = async () => {
      // Create a hidden video element for frame extraction
      const hiddenVideo = document.createElement('video');
      hiddenVideo.src = videoPreviewUrl;
      hiddenVideo.muted = true;
      hiddenVideo.style.display = 'none';
      document.body.appendChild(hiddenVideo);

      const duration = state.video.duration;
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      // Set canvas size (thumbnail dimensions)
      canvas.width = 160;
      canvas.height = 90;

      const frames = [];
      const frameInterval = 1; // Extract a frame every 1 second
      const frameCount = Math.ceil(duration / frameInterval);

      console.log(`🎬 Extracting ${frameCount} frames from video...`);

      // Wait for video metadata to load
      await new Promise((resolve) => {
        hiddenVideo.onloadedmetadata = resolve;
      });

      for (let i = 0; i < frameCount; i++) {
        const time = i * frameInterval;

        // Seek to frame time on hidden video
        await new Promise((resolve) => {
          hiddenVideo.currentTime = time;
          hiddenVideo.onseeked = () => {
            // Draw video frame to canvas
            ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height);

            // Convert to data URL
            const frameData = canvas.toDataURL('image/jpeg', 0.7);
            frames.push({
              time,
              dataUrl: frameData
            });

            resolve();
          };
        });
      }

      setVideoFrames(frames);
      console.log(`✅ Extracted ${frames.length} frames`);

      // Store frames in global state so DAW can access them
      dispatch({ type: 'SET_VIDEO_FRAMES', payload: frames });

      // Clean up hidden video element
      document.body.removeChild(hiddenVideo);
    };

    // Delay extraction to ensure video is fully loaded
    const timer = setTimeout(extractFrames, 500);

    return () => clearTimeout(timer);
  }, [showVideo, state.video?.duration, videoPreviewUrl, dispatch]);

  return (
    <div className={styles.uploadGrid}>
      {/* Settings Button - Top Right */}
      <button
        className={styles.settingsButton}
        title="Video Settings"
        onClick={(e) => {
          e.preventDefault();
          setShowSettings(!showSettings);
        }}
      >
        <i className="fa-solid fa-gear"></i>
      </button>

      {/* Settings Dropdown */}
      <div className={`${styles.settingsDropdown} ${showSettings ? styles.open : ''}`}>
        <div
          className={styles.settingsOption}
          onClick={() => setUseWhisper(!useWhisper)}
          title="Use audio transcription"
        >
          <i className="fa-solid fa-microphone"></i>
          <input
            type="checkbox"
            checked={useWhisper}
            onChange={(e) => setUseWhisper(e.target.checked)}
            onClick={(e) => e.stopPropagation()}
          />
          <span>Use audio transcription</span>
        </div>
      </div>

      {/* Close Button */}
      <i
        onClick={handleExitVideo}
        className={`fa-regular fa-circle-xmark ${styles.closeButton} ${!showCloseButton ? styles.hidden : ''}`}
        title="Close video"
      />

      {/* Export Button */}
      <button
        type="button"
        className={`${styles.exportButton} ${!showExportButton ? styles.hidden : ''}`}
        onClick={handleExport}
        title="Export video"
      >
        <i className="fas fa-download"></i> Export
      </button>

      {/* Upload Container */}
      {!showVideo && (
        <div
          className={`${styles.glowContainer} ${isDragging ? styles.dragging : ''}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <h4 className={styles.uploadHeader}>
            {isDragging ? 'Drop video here' : 'Upload a Video File'}
          </h4>

          <label htmlFor="videoFile" style={{ cursor: 'pointer' }}>
            <i className={`fa-solid ${isDragging ? 'fa-file-video' : 'fa-plus'} ${styles.uploadIcon}`}></i>
          </label>

          <form id="videoUploadForm" encType="multipart/form-data">
            <input
              type="file"
              name="file"
              id="videoFile"
              accept="video/*"
              onChange={handleVideoUpload}
              className={styles.fileInput}
            />
          </form>

          <p className={styles.dragHint}>
            {isDragging ? 'Release to upload' : 'Click + or drag & drop a video file'}
          </p>

          {/* Whisper Transcription Toggle */}
          <div className={styles.whisperToggle}>
            <label>
              <input
                type="checkbox"
                checked={useWhisper}
                onChange={(e) => setUseWhisper(e.target.checked)}
              />
              <span>Use Whisper for audio transcription</span>
            </label>
          </div>
        </div>
      )}

      {/* Video Preview Container */}
      {(showVideo || isProcessing) && (
        <div className={styles.videoContainer}>
          {/* Processing Status */}
          {isProcessing && (
            <div className={styles.processingOverlay}>
              <img
                src="/loading.gif"
                alt="Loading..."
                className={styles.loadingSpinner}
              />
              <p className={styles.processingStatus}>
                {processingStatus === 'uploading' && 'Uploading video...'}
                {processingStatus === 'detecting_scenes' && 'Detecting scene changes...'}
                {processingStatus?.startsWith('processing') && processingStatus}
              </p>
            </div>
          )}

          {/* Processing Error */}
          {processingError && (
            <div className={styles.errorMessage}>
              <i className="fas fa-exclamation-triangle"></i>
              {processingError}
            </div>
          )}

          {/* Video Player */}
          {showVideo && videoPreviewUrl && (
            <video
              ref={videoRef}
              id="player"
              className={styles.videoPlayer}
              playsInline
              muted
              src={videoPreviewUrl}
            >
              <source type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          )}

          {/* Scene Change Summary */}
          {processingStatus === 'completed' && state.video?.sceneChanges?.length > 0 && (
            <div className={styles.sceneSummary}>
              <i className="fas fa-film"></i>
              <span>
                {state.video.sceneChanges.length - 1} scenes detected
                ({state.video.duration?.toFixed(1)}s total)
              </span>
            </div>
          )}

          {/* Video Timeline Preview - Frame-by-frame preview that syncs with zoom */}
          {showVideo && videoFrames.length > 0 && state.video?.duration && (
            <div className={styles.videoTimeline}>
              <div
                className={styles.videoTimelineTrack}
                style={{
                  width: `${(state.timelineWidth || 700) * (state.zoomLevel || 1)}px`
                }}
              >
                {videoFrames.map((frame, index) => {
                  const frameWidth = ((state.timelineWidth || 700) * (state.zoomLevel || 1)) / videoFrames.length;

                  return (
                    <div
                      key={index}
                      className={styles.videoFrame}
                      style={{
                        width: `${frameWidth}px`,
                        flexShrink: 0,
                        backgroundImage: `url(${frame.dataUrl})`,
                        backgroundSize: 'cover',
                        backgroundPosition: 'center'
                      }}
                    >
                      <span className={styles.frameTime}>{frame.time}s</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

VideoUploadOptimized.displayName = 'VideoUploadOptimized';

export default VideoUploadOptimized;
