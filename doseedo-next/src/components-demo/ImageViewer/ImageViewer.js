import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import styles from './ImageViewer.module.css';

/**
 * ImageViewer - Display image associated with selected track
 */
const ImageViewer = () => {
  const { state, dispatch } = useApp();
  const [imageSrc, setImageSrc] = useState(null);
  const [trackName, setTrackName] = useState('');
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [generating, setGenerating] = useState(false);

  // Get selected track
  const selectedTrack = state.selectedTrack;

  // Load image when track is selected
  useEffect(() => {
    if (!selectedTrack) {
      setImageSrc(null);
      setTrackName('');
      setImageError(false);
      return;
    }

    setTrackName(selectedTrack.name || 'Track');
    setImageLoading(true);
    setImageError(false);

    // Check if track has a custom image URL
    if (selectedTrack.imageUrl) {
      setImageSrc(selectedTrack.imageUrl);
      setImageLoading(false);
    } else {
      // Use placeholder image based on instrument/track type
      const placeholderImage = getPlaceholderImage(selectedTrack);
      setImageSrc(placeholderImage);
      setImageLoading(false);
    }
  }, [selectedTrack]);

  // Get placeholder image based on track metadata
  const getPlaceholderImage = (track) => {
    // Use track metadata to determine appropriate placeholder
    const instrumentGroup = track.instrumentGroup || 'default';
    const instrumentSubgroup = track.instrumentSubgroup || '';

    // For now, return a placeholder based on instrument group
    // These will be placeholder images until we add real ones
    const placeholderMap = {
      'piano': 'https://via.placeholder.com/800x600/667eea/ffffff?text=Piano',
      'guitar': 'https://via.placeholder.com/800x600/f093fb/ffffff?text=Guitar',
      'bass': 'https://via.placeholder.com/800x600/11998e/ffffff?text=Bass',
      'strings': 'https://via.placeholder.com/800x600/764ba2/ffffff?text=Strings',
      'brass': 'https://via.placeholder.com/800x600/fa709a/ffffff?text=Brass',
      'winds': 'https://via.placeholder.com/800x600/38ef7d/ffffff?text=Winds',
      'drums': 'https://via.placeholder.com/800x600/f5576c/ffffff?text=Drums',
      'default': 'https://via.placeholder.com/800x600/444444/ffffff?text=Music+Track'
    };

    return placeholderMap[instrumentGroup] || placeholderMap['default'];
  };

  const handleImageError = () => {
    setImageError(true);
    setImageLoading(false);
  };

  const handleImageLoad = () => {
    setImageLoading(false);
  };

  const handleGenerateImage = async () => {
    if (!selectedTrack) return;

    setGenerating(true);
    setImageError(false);

    try {
      const instrumentGroup = selectedTrack.instrumentGroup || 'default';
      const instrumentSubgroup = selectedTrack.instrumentSubgroup || '';

      console.log(`🎨 Generating image for: ${instrumentGroup} / ${instrumentSubgroup}`);

      // Create form data
      const formData = new FormData();
      formData.append('instrumentGroup', instrumentGroup);
      formData.append('instrumentSubgroup', instrumentSubgroup);
      formData.append('trackName', selectedTrack.name || 'Track');

      // Call backend API
      const response = await fetch('/api/generate-track-image', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ Image generated:', data);

      // Update the image source with the URL
      const fullImageUrl = data.imageUrl;
      setImageSrc(fullImageUrl);
      setImageLoading(false);

      // Update track with the new image URL
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          trackId: selectedTrack.id,
          updates: {
            imageUrl: fullImageUrl
          }
        }
      });

      alert('Image generated successfully!');
    } catch (error) {
      console.error('❌ Error generating image:', error);
      setImageError(true);
      alert(`Failed to generate image: ${error.message}`);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <i className="fa-solid fa-image"></i>
        <span className={styles.title}>
          {selectedTrack ? trackName : 'No Track Selected'}
        </span>
      </div>

      <div className={styles.imageWrapper}>
        {!selectedTrack ? (
          <div className={styles.emptyState}>
            <i className="fa-solid fa-circle-info"></i>
            <p>Select a track to view its image</p>
          </div>
        ) : imageLoading ? (
          <div className={styles.loadingState}>
            <i className="fa-solid fa-spinner fa-spin"></i>
            <p>Loading image...</p>
          </div>
        ) : imageError ? (
          <div className={styles.errorState}>
            <i className="fa-solid fa-triangle-exclamation"></i>
            <p>Failed to load image</p>
          </div>
        ) : (
          <img
            src={imageSrc}
            alt={trackName}
            className={styles.image}
            onError={handleImageError}
            onLoad={handleImageLoad}
          />
        )}
      </div>

      {selectedTrack && (
        <>
          {!imageError && (
            <div className={styles.info}>
              <div className={styles.infoItem}>
                <i className="fa-solid fa-layer-group"></i>
                <span>
                  {selectedTrack.instrumentGroup
                    ? `${selectedTrack.instrumentGroup.charAt(0).toUpperCase() + selectedTrack.instrumentGroup.slice(1)}`
                    : 'Track'}
                </span>
              </div>
              {selectedTrack.instrumentSubgroup && (
                <div className={styles.infoItem}>
                  <i className="fa-solid fa-tag"></i>
                  <span>
                    {selectedTrack.instrumentSubgroup.split('_').map(w =>
                      w.charAt(0).toUpperCase() + w.slice(1)
                    ).join(' ')}
                  </span>
                </div>
              )}
            </div>
          )}

          <div className={styles.actions}>
            <button
              className={styles.generateBtn}
              onClick={handleGenerateImage}
              disabled={generating}
            >
              <i className={`fa-solid ${generating ? 'fa-spinner fa-spin' : 'fa-wand-magic-sparkles'}`}></i>
              <span>{generating ? 'Generating...' : 'Generate AI Image'}</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default ImageViewer;
