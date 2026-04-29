import React, { useState, useEffect } from 'react';
import styles from './Search.module.css';
import * as sessionAPI from '../../services/sessionAPI';
import PageTopbar from '../Sidebar/PageTopbar';
import PageEyebrow from '../Sidebar/PageEyebrow';

/**
 * Search Component
 * Search page for finding projects, files, and content
 */
const Search = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('all');

  const categories = [
    { id: 'all', label: 'All', icon: 'fa-grid-2' },
    { id: 'loops', label: 'Loops', icon: 'fa-repeat' },
    { id: 'presets', label: 'Presets', icon: 'fa-sliders' },
    { id: 'projects', label: 'Projects', icon: 'fa-folder' },
    { id: 'midi', label: 'MIDI', icon: 'fa-music' }
  ];

  const handleCategoryClick = (categoryId) => {
    setSelectedCategory(categoryId);
    console.log('Category selected:', categoryId);
    // TODO: Filter results by category
  };

  // Load public sessions on mount and category change
  useEffect(() => {
    loadPublicSessions();
  }, [selectedCategory]);

  const loadPublicSessions = async () => {
    try {
      const filters = selectedCategory !== 'all' ? { type: selectedCategory } : {};
      const data = await sessionAPI.getPublicSessions(filters);

      // Convert sessions to search result format
      const results = (data.sessions || []).map(session => ({
        id: session.id,
        type: session.type,
        name: session.name,
        description: session.description,
        date: new Date(session.created_at).toLocaleDateString(),
        thumbnail: session.thumbnail_url,
        author: session.user_name || 'Anonymous',
        files: session.files || []
      }));

      setSearchResults(results);
    } catch (error) {
      console.error('Failed to load public sessions:', error);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();

    if (!searchQuery.trim()) {
      loadPublicSessions();
      return;
    }

    try {
      const filters = selectedCategory !== 'all' ? { type: selectedCategory } : {};
      const data = await sessionAPI.searchSessions(searchQuery, filters);

      // Convert sessions to search result format
      const results = (data.results || []).map(session => ({
        id: session.id,
        type: session.type,
        name: session.name,
        description: session.description,
        date: new Date(session.created_at).toLocaleDateString(),
        thumbnail: session.thumbnail_url,
        author: session.user_name || 'Anonymous',
        files: session.files || []
      }));

      setSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
    }
  };

  return (
    <div className={styles.searchContainer}>
      <PageTopbar title="Search" meta="find projects, files, content" />
      <PageEyebrow section="Search" description="Find projects, files, and content" />
      <div className={styles.searchHeader}>
        <h1 className={`${styles.searchTitle} page-title`}>Search</h1>
        <p className={styles.searchSubtitle}>Find your projects, files, and content</p>
      </div>

      <form onSubmit={handleSearch} className={styles.searchForm}>
        <div className={styles.searchInputWrapper}>
          <i className="fa-solid fa-magnifying-glass"></i>
          <input
            type="text"
            placeholder="Search for projects, MIDI files, audio files..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className={styles.searchInput}
            autoFocus
          />
        </div>
        <button type="submit" className={styles.searchButton}>
          Search
        </button>
      </form>

      {/* Category Boxes */}
      <div className={styles.categoriesSection}>
        {categories.map((category) => (
          <button
            key={category.id}
            className={`${styles.categoryBox} ${selectedCategory === category.id ? styles.active : ''}`}
            onClick={() => handleCategoryClick(category.id)}
          >
            <i className={`fa-solid ${category.icon}`}></i>
            <span>{category.label}</span>
          </button>
        ))}
      </div>

      {searchResults.length > 0 && (
        <div className={styles.resultsSection}>
          <h2 className={styles.resultsTitle}>Results ({searchResults.length})</h2>
          <div className={styles.resultsList}>
            {searchResults.map((result, index) => (
              <div key={result.id || index} className={styles.resultCard}>
                {result.thumbnail ? (
                  <div className={styles.resultThumbnail}>
                    <img src={result.thumbnail} alt={result.name} />
                  </div>
                ) : (
                  <div className={styles.resultIcon}>
                    <i className={`fa-solid ${
                      result.type === 'project' ? 'fa-folder' :
                      result.type === 'loop' ? 'fa-repeat' :
                      result.type === 'preset' ? 'fa-sliders' :
                      result.type === 'midi' ? 'fa-music' :
                      'fa-volume-high'
                    }`}></i>
                  </div>
                )}
                <div className={styles.resultInfo}>
                  <h3 className={styles.resultName}>{result.name}</h3>
                  {result.description && (
                    <p className={styles.resultDescription}>{result.description}</p>
                  )}
                  <p className={styles.resultMeta}>
                    <span className={styles.resultType}>{result.type?.toUpperCase()}</span>
                    {result.author && <span className={styles.resultAuthor}>by {result.author}</span>}
                    <span className={styles.resultDate}>{result.date}</span>
                  </p>
                </div>
                <div className={styles.resultActions}>
                  <button className={styles.actionButton} title="Preview">
                    <i className="fa-solid fa-play"></i>
                  </button>
                  <button className={styles.actionButton} title="Download">
                    <i className="fa-solid fa-download"></i>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {searchQuery && searchResults.length === 0 && (
        <div className={styles.noResults}>
          <i className="fa-solid fa-search"></i>
          <p>No results found. Try a different search term.</p>
        </div>
      )}
    </div>
  );
};

export default Search;
