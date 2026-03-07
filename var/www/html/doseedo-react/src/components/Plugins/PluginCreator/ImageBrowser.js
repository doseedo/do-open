import React, { useState, useCallback } from 'react';
import { generateImage, searchImages } from '../../../services/chatAPI';
import styles from './PluginCreator.module.css';

const MAX_UPLOAD_SIZE = 2 * 1024 * 1024; // 2MB

const ImageBrowser = ({ onSelect, onClose }) => {
  const [tab, setTab] = useState('generate'); // 'generate' | 'search' | 'upload'
  const [prompt, setPrompt] = useState('');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null); // data URL
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    setGeneratedImage(null);
    try {
      const data = await generateImage({ prompt: prompt.trim(), size: '1024x1024' });
      setGeneratedImage(data);
    } catch (err) {
      setError(err.message || 'Image generation failed');
    } finally {
      setLoading(false);
    }
  }, [prompt, loading]);

  const handleSearch = useCallback(async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResults([]);
    try {
      const data = await searchImages({ query: query.trim(), per_page: 12 });
      setResults(data.results || []);
    } catch (err) {
      setError(err.message || 'Image search failed');
    } finally {
      setLoading(false);
    }
  }, [query, loading]);

  const handleFileUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setUploadedImage(null);
    if (!file.type.match(/^image\/(png|jpe?g|webp)$/)) {
      setError('Only PNG, JPG, and WebP images are supported');
      return;
    }
    if (file.size > MAX_UPLOAD_SIZE) {
      setError(`Image too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Max 2MB.`);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setUploadedImage(reader.result);
    reader.onerror = () => setError('Failed to read file');
    reader.readAsDataURL(file);
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      if (tab === 'generate') handleGenerate();
      else handleSearch();
    }
  };

  return (
    <div className={styles.imageBrowserOverlay} onClick={onClose}>
      <div className={styles.imageBrowser} onClick={(e) => e.stopPropagation()}>
        <div className={styles.imageBrowserHeader}>
          <span>Image Browser</span>
          <button className={styles.layersCloseBtn} onClick={onClose}>
            <i className="fa-solid fa-xmark" />
          </button>
        </div>

        {/* Tabs */}
        <div className={styles.imageBrowserTabs}>
          <button
            className={`${styles.ibTab} ${tab === 'generate' ? styles.ibTabActive : ''}`}
            onClick={() => setTab('generate')}
          >
            <i className="fa-solid fa-wand-magic-sparkles" /> Generate
          </button>
          <button
            className={`${styles.ibTab} ${tab === 'search' ? styles.ibTabActive : ''}`}
            onClick={() => setTab('search')}
          >
            <i className="fa-solid fa-magnifying-glass" /> Search
          </button>
          <button
            className={`${styles.ibTab} ${tab === 'upload' ? styles.ibTabActive : ''}`}
            onClick={() => setTab('upload')}
          >
            <i className="fa-solid fa-upload" /> Upload
          </button>
        </div>

        {/* Input */}
        {tab !== 'upload' && (
        <div className={styles.ibInputRow}>
          {tab === 'generate' ? (
            <>
              <input
                className={styles.ibInput}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe the image to generate..."
                autoFocus
              />
              <button className={styles.ibGoBtn} onClick={handleGenerate} disabled={loading || !prompt.trim()}>
                {loading ? <i className="fa-solid fa-spinner fa-spin" /> : <i className="fa-solid fa-bolt" />}
              </button>
            </>
          ) : (
            <>
              <input
                className={styles.ibInput}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Search stock images..."
                autoFocus
              />
              <button className={styles.ibGoBtn} onClick={handleSearch} disabled={loading || !query.trim()}>
                {loading ? <i className="fa-solid fa-spinner fa-spin" /> : <i className="fa-solid fa-magnifying-glass" />}
              </button>
            </>
          )}
        </div>
        )}

        {error && (
          <div className={styles.ibError}>
            <i className="fa-solid fa-triangle-exclamation" /> {error}
          </div>
        )}

        {/* Results */}
        <div className={styles.ibResults}>
          {tab === 'upload' && (
            <div className={styles.ibGenResult}>
              <label
                className={styles.ibUploadArea}
                style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                  padding: '32px 16px', border: '2px dashed rgba(186,156,255,0.3)', borderRadius: 12,
                  cursor: 'pointer', textAlign: 'center', color: 'rgba(255,255,255,0.4)', fontSize: 13,
                  background: 'rgba(255,255,255,0.03)', marginBottom: 12,
                }}
              >
                <i className="fa-solid fa-cloud-arrow-up" style={{ fontSize: 28, marginBottom: 8, color: 'rgba(186,156,255,0.5)' }} />
                <span>Click to upload an image</span>
                <span style={{ fontSize: 11, marginTop: 4 }}>PNG, JPG, WebP &middot; Max 2MB</span>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
              </label>
              {uploadedImage && (
                <>
                  <img src={uploadedImage} alt="Uploaded" style={{ maxWidth: '100%', borderRadius: 8 }} />
                  <button className={styles.ibSelectBtn} onClick={() => onSelect(uploadedImage)}>
                    Use This Image
                  </button>
                </>
              )}
            </div>
          )}

          {tab === 'generate' && generatedImage && (
            <div className={styles.ibGenResult}>
              <img src={generatedImage.url} alt={generatedImage.revised_prompt || 'Generated'} />
              {generatedImage.revised_prompt && (
                <p className={styles.ibCaption}>{generatedImage.revised_prompt}</p>
              )}
              <button className={styles.ibSelectBtn} onClick={() => onSelect(generatedImage.url)}>
                Use This Image
              </button>
            </div>
          )}

          {tab === 'search' && results.length > 0 && (
            <div className={styles.ibGrid}>
              {results.map(img => (
                <div key={img.id} className={styles.ibGridItem} onClick={() => onSelect(img.url)}>
                  <img src={img.thumb} alt={img.description || ''} />
                  <span className={styles.ibAuthor}>{img.author}</span>
                </div>
              ))}
            </div>
          )}

          {!loading && tab === 'search' && results.length === 0 && query && (
            <div className={styles.ibEmpty}>No results. Try a different search term.</div>
          )}

          {loading && (
            <div className={styles.ibLoading}>
              <i className="fa-solid fa-spinner fa-spin" /> {tab === 'generate' ? 'Generating image...' : 'Searching...'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageBrowser;
