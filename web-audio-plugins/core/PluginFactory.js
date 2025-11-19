/**
 * PluginFactory - Plugin registration and instantiation system
 *
 * Manages plugin registry and provides factory methods for creating plugin instances.
 * All plugins must register with the factory to be available for instantiation.
 *
 * @author Agent 10 (Core Infrastructure)
 * @version 1.0.0
 */

class PluginFactoryClass {
  constructor() {
    // Plugin registry: name -> { constructor, metadata }
    this._registry = new Map();

    // Category index for quick lookups
    this._categoryIndex = new Map();
  }

  /**
   * Register a plugin with the factory
   * @param {string} name - Unique plugin name
   * @param {Function} PluginClass - Plugin constructor (must extend BasePlugin)
   * @param {Object} metadata - Plugin metadata
   * @param {string} metadata.category - Plugin category
   * @param {string} metadata.description - Plugin description
   * @param {Array<string>} metadata.tags - Search tags
   * @param {string} metadata.version - Plugin version
   * @param {string} metadata.author - Plugin author
   */
  register(name, PluginClass, metadata = {}) {
    if (this._registry.has(name)) {
      console.warn(`Plugin ${name} is already registered. Overwriting.`);
    }

    if (typeof PluginClass !== 'function') {
      throw new Error(`Plugin ${name} must be a constructor function`);
    }

    const pluginEntry = {
      constructor: PluginClass,
      metadata: {
        name,
        category: metadata.category || 'uncategorized',
        description: metadata.description || '',
        tags: metadata.tags || [],
        version: metadata.version || '1.0.0',
        author: metadata.author || 'Unknown'
      }
    };

    this._registry.set(name, pluginEntry);

    // Update category index
    const category = pluginEntry.metadata.category;
    if (!this._categoryIndex.has(category)) {
      this._categoryIndex.set(category, []);
    }
    this._categoryIndex.get(category).push(name);
  }

  /**
   * Create an instance of a registered plugin
   * @param {string} name - Plugin name
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Plugin initialization options
   * @returns {BasePlugin} Plugin instance
   */
  create(name, audioContext, options = {}) {
    const entry = this._registry.get(name);

    if (!entry) {
      throw new Error(`Plugin ${name} is not registered`);
    }

    if (!audioContext) {
      throw new Error('AudioContext is required to create plugin');
    }

    try {
      const instance = new entry.constructor(audioContext, options);
      return instance;
    } catch (error) {
      console.error(`Failed to create plugin ${name}:`, error);
      throw error;
    }
  }

  /**
   * Check if a plugin is registered
   * @param {string} name - Plugin name
   * @returns {boolean} True if registered
   */
  isRegistered(name) {
    return this._registry.has(name);
  }

  /**
   * Get plugin metadata
   * @param {string} name - Plugin name
   * @returns {Object|null} Plugin metadata or null
   */
  getMetadata(name) {
    const entry = this._registry.get(name);
    return entry ? entry.metadata : null;
  }

  /**
   * Get all registered plugin names
   * @returns {Array<string>} Plugin names
   */
  getAllPluginNames() {
    return Array.from(this._registry.keys());
  }

  /**
   * Get plugins by category
   * @param {string} category - Category name
   * @returns {Array<string>} Plugin names in category
   */
  getPluginsByCategory(category) {
    return this._categoryIndex.get(category) || [];
  }

  /**
   * Get all categories
   * @returns {Array<string>} Category names
   */
  getAllCategories() {
    return Array.from(this._categoryIndex.keys());
  }

  /**
   * Search plugins by tag
   * @param {string} tag - Tag to search for
   * @returns {Array<string>} Matching plugin names
   */
  searchByTag(tag) {
    const results = [];

    this._registry.forEach((entry, name) => {
      if (entry.metadata.tags.includes(tag)) {
        results.push(name);
      }
    });

    return results;
  }

  /**
   * Get complete registry information
   * @returns {Array<Object>} All plugins with metadata
   */
  getRegistry() {
    const registry = [];

    this._registry.forEach((entry, name) => {
      registry.push({
        name,
        ...entry.metadata
      });
    });

    return registry;
  }

  /**
   * Unregister a plugin
   * @param {string} name - Plugin name
   * @returns {boolean} True if unregistered successfully
   */
  unregister(name) {
    const entry = this._registry.get(name);

    if (!entry) {
      return false;
    }

    // Remove from registry
    this._registry.delete(name);

    // Remove from category index
    const category = entry.metadata.category;
    const categoryPlugins = this._categoryIndex.get(category);
    if (categoryPlugins) {
      const index = categoryPlugins.indexOf(name);
      if (index !== -1) {
        categoryPlugins.splice(index, 1);
      }

      // Remove empty categories
      if (categoryPlugins.length === 0) {
        this._categoryIndex.delete(category);
      }
    }

    return true;
  }

  /**
   * Clear all registered plugins
   */
  clear() {
    this._registry.clear();
    this._categoryIndex.clear();
  }
}

// Singleton instance
const PluginFactory = new PluginFactoryClass();

export default PluginFactory;
export { PluginFactory, PluginFactoryClass };
