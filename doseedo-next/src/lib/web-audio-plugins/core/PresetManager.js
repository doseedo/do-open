/**
 * PresetManager - Preset save/load and library management
 *
 * Manages individual plugin presets, chain presets, categories,
 * import/export functionality, and preset browsing.
 *
 * @class PresetManager
 * @author Agent 10 - Integration & Routing System
 */

class PresetManager {
  /**
   * Create a new PresetManager instance
   * @param {Object} options - Configuration options
   * @param {boolean} options.useLocalStorage - Use localStorage for persistence (default: false)
   */
  constructor(options = {}) {
    this.options = {
      useLocalStorage: options.useLocalStorage || false,
      storageKey: options.storageKey || 'webaudio_presets'
    };

    // Preset storage: presetId -> preset object
    this.presets = new Map();

    // Category storage: category -> [presetIds]
    this.categories = new Map();

    // Tags index: tag -> [presetIds]
    this.tags = new Map();

    // Default categories
    this.initializeDefaultCategories();

    // Load from localStorage if enabled
    if (this.options.useLocalStorage) {
      this.loadFromLocalStorage();
    }
  }

  /**
   * Initialize default preset categories
   * @private
   */
  initializeDefaultCategories() {
    const defaultCategories = [
      'Factory',
      'User',
      'Imported',
      'Dynamics',
      'EQ',
      'Filter',
      'Distortion',
      'Time-Based',
      'Modulation',
      'Spatial',
      'Utility',
      'Chains'
    ];

    defaultCategories.forEach(cat => {
      this.categories.set(cat, []);
    });
  }

  /**
   * Save a plugin preset
   * @param {string} name - Preset name
   * @param {BasePlugin} plugin - Plugin instance
   * @param {Object} options - Save options
   * @param {string} options.category - Category name (default: 'User')
   * @param {string[]} options.tags - Tags array
   * @param {string} options.description - Preset description
   * @param {string} options.author - Preset author
   * @returns {string} Preset ID
   */
  savePreset(name, plugin, options = {}) {
    if (!name || !plugin) {
      throw new Error('Name and plugin are required');
    }

    const category = options.category || 'User';
    const preset = plugin.savePreset(name);

    // Add metadata
    preset.category = category;
    preset.description = options.description || '';
    preset.author = options.author || 'Unknown';
    preset.tags = options.tags || [];
    preset.id = this.generatePresetId(category, name);

    // Store preset
    this.presets.set(preset.id, preset);

    // Add to category
    if (!this.categories.has(category)) {
      this.categories.set(category, []);
    }
    if (!this.categories.get(category).includes(preset.id)) {
      this.categories.get(category).push(preset.id);
    }

    // Index tags
    preset.tags.forEach(tag => {
      if (!this.tags.has(tag)) {
        this.tags.set(tag, []);
      }
      if (!this.tags.get(tag).includes(preset.id)) {
        this.tags.get(tag).push(preset.id);
      }
    });

    // Save to localStorage if enabled
    if (this.options.useLocalStorage) {
      this.saveToLocalStorage();
    }

    return preset.id;
  }

  /**
   * Load a preset onto a plugin
   * @param {string} presetId - Preset ID
   * @param {BasePlugin} plugin - Plugin instance to load onto
   * @param {number} rampTime - Ramp time for parameter changes
   * @returns {boolean} Success status
   */
  loadPreset(presetId, plugin, rampTime = 0) {
    const preset = this.presets.get(presetId);

    if (!preset) {
      console.error(`Preset ${presetId} not found`);
      return false;
    }

    return plugin.loadPreset(preset, rampTime);
  }

  /**
   * Get a preset by ID
   * @param {string} presetId - Preset ID
   * @returns {Object|null} Preset object or null
   */
  getPreset(presetId) {
    const preset = this.presets.get(presetId);
    return preset ? { ...preset } : null;
  }

  /**
   * Get all presets
   * @returns {Array} Array of preset objects
   */
  getAllPresets() {
    const presets = [];
    this.presets.forEach((preset, id) => {
      presets.push({ ...preset });
    });
    return presets;
  }

  /**
   * Get presets for a specific plugin type
   * @param {string} pluginType - Plugin type (constructor name)
   * @returns {Array} Array of preset objects
   */
  getPresetsForPlugin(pluginType) {
    const filtered = [];

    this.presets.forEach((preset, id) => {
      if (preset.type === pluginType) {
        filtered.push({ ...preset });
      }
    });

    // Sort by name
    return filtered.sort((a, b) => a.name.localeCompare(b.name));
  }

  /**
   * Get presets in a category
   * @param {string} category - Category name
   * @returns {Array} Array of preset objects
   */
  getPresetsInCategory(category) {
    const presetIds = this.categories.get(category) || [];
    return presetIds
      .map(id => ({ ...this.presets.get(id) }))
      .filter(p => p !== null)
      .sort((a, b) => a.name.localeCompare(b.name));
  }

  /**
   * Get all categories
   * @returns {Array} Array of category names
   */
  getCategories() {
    return Array.from(this.categories.keys()).sort();
  }

  /**
   * Get presets by tag
   * @param {string} tag - Tag name
   * @returns {Array} Array of preset objects
   */
  getPresetsByTag(tag) {
    const presetIds = this.tags.get(tag) || [];
    return presetIds
      .map(id => ({ ...this.presets.get(id) }))
      .filter(p => p !== null)
      .sort((a, b) => a.name.localeCompare(b.name));
  }

  /**
   * Get all tags
   * @returns {Array} Array of tag names
   */
  getAllTags() {
    return Array.from(this.tags.keys()).sort();
  }

  /**
   * Search presets
   * @param {Object} criteria - Search criteria
   * @param {string} criteria.query - Search query (searches name and description)
   * @param {string} criteria.type - Plugin type
   * @param {string} criteria.category - Category
   * @param {string[]} criteria.tags - Required tags
   * @returns {Array} Array of matching preset objects
   */
  searchPresets(criteria = {}) {
    let results = this.getAllPresets();

    // Filter by query
    if (criteria.query) {
      const query = criteria.query.toLowerCase();
      results = results.filter(preset =>
        preset.name.toLowerCase().includes(query) ||
        preset.description.toLowerCase().includes(query)
      );
    }

    // Filter by type
    if (criteria.type) {
      results = results.filter(preset => preset.type === criteria.type);
    }

    // Filter by category
    if (criteria.category) {
      results = results.filter(preset => preset.category === criteria.category);
    }

    // Filter by tags
    if (criteria.tags && criteria.tags.length > 0) {
      results = results.filter(preset =>
        criteria.tags.every(tag => preset.tags.includes(tag))
      );
    }

    return results.sort((a, b) => a.name.localeCompare(b.name));
  }

  /**
   * Delete a preset
   * @param {string} presetId - Preset ID
   * @returns {boolean} Success status
   */
  deletePreset(presetId) {
    const preset = this.presets.get(presetId);
    if (!preset) {
      console.warn(`Preset ${presetId} not found`);
      return false;
    }

    // Remove from category
    const category = this.categories.get(preset.category);
    if (category) {
      const index = category.indexOf(presetId);
      if (index > -1) {
        category.splice(index, 1);
      }
    }

    // Remove from tag indices
    preset.tags.forEach(tag => {
      const tagList = this.tags.get(tag);
      if (tagList) {
        const index = tagList.indexOf(presetId);
        if (index > -1) {
          tagList.splice(index, 1);
        }
      }
    });

    // Remove preset
    this.presets.delete(presetId);

    // Save to localStorage if enabled
    if (this.options.useLocalStorage) {
      this.saveToLocalStorage();
    }

    return true;
  }

  /**
   * Update preset metadata
   * @param {string} presetId - Preset ID
   * @param {Object} updates - Fields to update
   * @returns {boolean} Success status
   */
  updatePreset(presetId, updates) {
    const preset = this.presets.get(presetId);
    if (!preset) {
      console.error(`Preset ${presetId} not found`);
      return false;
    }

    // Handle category change
    if (updates.category && updates.category !== preset.category) {
      // Remove from old category
      const oldCategory = this.categories.get(preset.category);
      if (oldCategory) {
        const index = oldCategory.indexOf(presetId);
        if (index > -1) oldCategory.splice(index, 1);
      }

      // Add to new category
      if (!this.categories.has(updates.category)) {
        this.categories.set(updates.category, []);
      }
      this.categories.get(updates.category).push(presetId);
    }

    // Handle tag changes
    if (updates.tags) {
      // Remove from old tags
      preset.tags.forEach(tag => {
        const tagList = this.tags.get(tag);
        if (tagList) {
          const index = tagList.indexOf(presetId);
          if (index > -1) tagList.splice(index, 1);
        }
      });

      // Add to new tags
      updates.tags.forEach(tag => {
        if (!this.tags.has(tag)) {
          this.tags.set(tag, []);
        }
        if (!this.tags.get(tag).includes(presetId)) {
          this.tags.get(tag).push(presetId);
        }
      });
    }

    // Update preset
    Object.assign(preset, updates);

    // Save to localStorage if enabled
    if (this.options.useLocalStorage) {
      this.saveToLocalStorage();
    }

    return true;
  }

  /**
   * Export a preset as JSON
   * @param {string} presetId - Preset ID
   * @param {boolean} pretty - Pretty print JSON (default: true)
   * @returns {string|null} JSON string or null
   */
  exportPreset(presetId, pretty = true) {
    const preset = this.presets.get(presetId);
    if (!preset) {
      console.error(`Preset ${presetId} not found`);
      return null;
    }

    return JSON.stringify(preset, null, pretty ? 2 : 0);
  }

  /**
   * Import a preset from JSON
   * @param {string} jsonString - JSON preset string
   * @param {Object} options - Import options
   * @param {boolean} options.overwrite - Overwrite if preset exists (default: false)
   * @returns {string|null} Preset ID or null on failure
   */
  importPreset(jsonString, options = {}) {
    try {
      const preset = JSON.parse(jsonString);

      if (!preset.name || !preset.type) {
        console.error('Invalid preset format: missing name or type');
        return null;
      }

      // Generate new ID
      const category = preset.category || 'Imported';
      const presetId = this.generatePresetId(category, preset.name);

      // Check if exists
      if (this.presets.has(presetId) && !options.overwrite) {
        console.error(`Preset ${presetId} already exists. Use overwrite option to replace.`);
        return null;
      }

      preset.id = presetId;
      preset.category = category;

      // Store preset
      this.presets.set(presetId, preset);

      // Add to category
      if (!this.categories.has(category)) {
        this.categories.set(category, []);
      }
      if (!this.categories.get(category).includes(presetId)) {
        this.categories.get(category).push(presetId);
      }

      // Index tags
      (preset.tags || []).forEach(tag => {
        if (!this.tags.has(tag)) {
          this.tags.set(tag, []);
        }
        if (!this.tags.get(tag).includes(presetId)) {
          this.tags.get(tag).push(presetId);
        }
      });

      // Save to localStorage if enabled
      if (this.options.useLocalStorage) {
        this.saveToLocalStorage();
      }

      return presetId;
    } catch (error) {
      console.error('Failed to import preset:', error);
      return null;
    }
  }

  /**
   * Export all presets
   * @param {boolean} pretty - Pretty print JSON (default: true)
   * @returns {string} JSON string
   */
  exportAllPresets(pretty = true) {
    const all = {};

    this.presets.forEach((preset, id) => {
      all[id] = preset;
    });

    return JSON.stringify(all, null, pretty ? 2 : 0);
  }

  /**
   * Import multiple presets
   * @param {string} jsonString - JSON string with multiple presets
   * @param {Object} options - Import options
   * @param {boolean} options.overwrite - Overwrite existing presets (default: false)
   * @returns {Object} Import results { success: number, failed: number, ids: [] }
   */
  importAllPresets(jsonString, options = {}) {
    const results = {
      success: 0,
      failed: 0,
      ids: []
    };

    try {
      const all = JSON.parse(jsonString);

      Object.entries(all).forEach(([originalId, preset]) => {
        try {
          const presetJson = JSON.stringify(preset);
          const id = this.importPreset(presetJson, options);

          if (id) {
            results.success++;
            results.ids.push(id);
          } else {
            results.failed++;
          }
        } catch (error) {
          console.error(`Failed to import preset ${originalId}:`, error);
          results.failed++;
        }
      });

      return results;
    } catch (error) {
      console.error('Failed to parse presets JSON:', error);
      return results;
    }
  }

  /**
   * Generate a preset ID
   * @param {string} category - Category name
   * @param {string} name - Preset name
   * @returns {string} Preset ID
   * @private
   */
  generatePresetId(category, name) {
    const sanitized = name.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
    return `${category}/${sanitized}_${Date.now()}`;
  }

  /**
   * Save presets to localStorage
   * @private
   */
  saveToLocalStorage() {
    try {
      const data = {
        presets: {},
        categories: {},
        tags: {}
      };

      this.presets.forEach((preset, id) => {
        data.presets[id] = preset;
      });

      this.categories.forEach((ids, category) => {
        data.categories[category] = ids;
      });

      this.tags.forEach((ids, tag) => {
        data.tags[tag] = ids;
      });

      localStorage.setItem(this.options.storageKey, JSON.stringify(data));
    } catch (error) {
      console.error('Failed to save to localStorage:', error);
    }
  }

  /**
   * Load presets from localStorage
   * @private
   */
  loadFromLocalStorage() {
    try {
      const dataStr = localStorage.getItem(this.options.storageKey);
      if (!dataStr) return;

      const data = JSON.parse(dataStr);

      // Load presets
      if (data.presets) {
        Object.entries(data.presets).forEach(([id, preset]) => {
          this.presets.set(id, preset);
        });
      }

      // Load categories
      if (data.categories) {
        Object.entries(data.categories).forEach(([category, ids]) => {
          this.categories.set(category, ids);
        });
      }

      // Load tags
      if (data.tags) {
        Object.entries(data.tags).forEach(([tag, ids]) => {
          this.tags.set(tag, ids);
        });
      }
    } catch (error) {
      console.error('Failed to load from localStorage:', error);
    }
  }

  /**
   * Clear all presets
   * @param {boolean} clearFactory - Also clear factory presets (default: false)
   */
  clearAll(clearFactory = false) {
    if (clearFactory) {
      this.presets.clear();
      this.categories.clear();
      this.tags.clear();
      this.initializeDefaultCategories();
    } else {
      // Only clear non-factory presets
      const toDelete = [];
      this.presets.forEach((preset, id) => {
        if (preset.category !== 'Factory') {
          toDelete.push(id);
        }
      });

      toDelete.forEach(id => this.deletePreset(id));
    }

    // Save to localStorage if enabled
    if (this.options.useLocalStorage) {
      this.saveToLocalStorage();
    }
  }

  /**
   * Get statistics
   * @returns {Object} Statistics object
   */
  getStats() {
    const stats = {
      totalPresets: this.presets.size,
      categories: this.categories.size,
      tags: this.tags.size,
      presetsByCategory: {},
      presetsByType: {}
    };

    // Count by category
    this.categories.forEach((ids, category) => {
      stats.presetsByCategory[category] = ids.length;
    });

    // Count by type
    this.presets.forEach(preset => {
      stats.presetsByType[preset.type] = (stats.presetsByType[preset.type] || 0) + 1;
    });

    return stats;
  }

  /**
   * String representation
   * @returns {string} String description
   */
  toString() {
    return `PresetManager(${this.presets.size} presets, ${this.categories.size} categories)`;
  }
}

export default PresetManager;
