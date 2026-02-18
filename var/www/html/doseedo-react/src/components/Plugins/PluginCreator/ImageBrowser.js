import React, { useState, useCallback } from 'react';
import { generateImage, searchImages } from '../../../services/chatAPI';
import styles from './PluginCreator.module.css';

const ImageBrowser = ({ onSelect, onClose }) => {
  const [tab, setTab] = useState('generate'); // 'generate' | 'search'
  const [prompt, setPrompt] = useState('');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [generatedImage, setGeneratedImage] = useState(null);
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
        </div>

        {/* Input */}
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

        {error && (
          <div className={styles.ibError}>
            <i className="fa-solid fa-triangle-exclamation" /> {error}
          </div>
        )}

        {/* Results */}
        <div className={styles.ibResults}>
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
