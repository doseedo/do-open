/**
 * PluginFactory - Factory for creating and managing plugin instances
 *
 * @description
 * Centralized factory for:
 * - Plugin registration
 * - Plugin instantiation
 * - Plugin discovery
 * - Plugin metadata management
 *
 * All plugins MUST register with the factory to be discoverable
 *
 * @example
 * // Register a plugin
 * PluginFactory.register('TapeEmulation', TapeEmulation, {
 *   category: 'vintage',
 *   description: 'Analog tape emulation',
 *   tags: ['tape', 'saturation', 'vintage']
 * });
 *
 * // Create instance
 * const tape = PluginFactory.create('TapeEmulation', audioContext);
 */

class PluginFactoryClass {
  constructor() {
    this.plugins = new Map();
    this.instances = new Map();
  }

  /**
   * Register a plugin class
   * @param {string} name - Plugin name
   * @param {class} PluginClass - Plugin class (must extend BasePlugin)
   * @param {Object} metadata - Plugin metadata
   */
  register(name, PluginClass, metadata = {}) {
    if (this.plugins.has(name)) {
      console.warn(`Plugin "${name}" is already registered. Overwriting...`);
    }

    this.plugins.set(name, {
      name,
      class: PluginClass,
      category: metadata.category || 'uncategorized',
      description: metadata.description || '',
      tags: metadata.tags || [],
      version: metadata.version || '1.0.0',
      author: metadata.author || '',
      metadata
    });

    console.log(`✅ Registered plugin: ${name} (${metadata.category})`);

    return this;
  }

  /**
   * Create a plugin instance
   * @param {string} name - Plugin name
   * @param {AudioContext} audioContext - Web Audio context
   * @param {Object} options - Plugin options
   * @returns {BasePlugin} Plugin instance
   */
  create(name, audioContext, options = {}) {
    const pluginInfo = this.plugins.get(name);

    if (!pluginInfo) {
      throw new Error(`Plugin "${name}" not found. Available plugins: ${this.getPluginNames().join(', ')}`);
    }

    const instance = new pluginInfo.class(audioContext, options);

    // Track instance
    if (!this.instances.has(name)) {
      this.instances.set(name, []);
    }
    this.instances.get(name).push(instance);

    return instance;
  }

  /**
   * Get plugin metadata
   * @param {string} name - Plugin name
   * @returns {Object} Plugin metadata
   */
  getPluginInfo(name) {
    const pluginInfo = this.plugins.get(name);
    if (!pluginInfo) {
      return null;
    }

    return {
      name: pluginInfo.name,
      category: pluginInfo.category,
      description: pluginInfo.description,
      tags: pluginInfo.tags,
      version: pluginInfo.version,
      author: pluginInfo.author
    };
  }

  /**
   * Get all registered plugin names
   * @returns {Array<string>} Plugin names
   */
  getPluginNames() {
    return Array.from(this.plugins.keys());
  }

  /**
   * Get plugins by category
   * @param {string} category - Category name
   * @returns {Array<Object>} Plugin info objects
   */
  getPluginsByCategory(category) {
    const plugins = [];

    for (const [name, info] of this.plugins.entries()) {
      if (info.category === category) {
        plugins.push({
          name,
          description: info.description,
          tags: info.tags,
          version: info.version
        });
      }
    }

    return plugins;
  }

  /**
   * Get plugins by tag
   * @param {string} tag - Tag to search for
   * @returns {Array<Object>} Plugin info objects
   */
  getPluginsByTag(tag) {
    const plugins = [];

    for (const [name, info] of this.plugins.entries()) {
      if (info.tags.includes(tag)) {
        plugins.push({
          name,
          category: info.category,
          description: info.description,
          version: info.version
        });
      }
    }

    return plugins;
  }

  /**
   * Get all categories
   * @returns {Array<string>} Unique categories
   */
  getCategories() {
    const categories = new Set();

    for (const info of this.plugins.values()) {
      categories.add(info.category);
    }

    return Array.from(categories);
  }

  /**
   * Get all tags
   * @returns {Array<string>} Unique tags
   */
  getTags() {
    const tags = new Set();

    for (const info of this.plugins.values()) {
      for (const tag of info.tags) {
        tags.add(tag);
      }
    }

    return Array.from(tags);
  }

  /**
   * Get all plugin instances of a specific type
   * @param {string} name - Plugin name
   * @returns {Array<BasePlugin>} Plugin instances
   */
  getInstances(name) {
    return this.instances.get(name) || [];
  }

  /**
   * Get all active instances (all plugins)
   * @returns {Array<BasePlugin>} All plugin instances
   */
  getAllInstances() {
    const allInstances = [];

    for (const instances of this.instances.values()) {
      allInstances.push(...instances);
    }

    return allInstances;
  }

  /**
   * Dispose of a specific instance
   * @param {BasePlugin} instance - Plugin instance to dispose
   */
  disposeInstance(instance) {
    for (const [name, instances] of this.instances.entries()) {
      const index = instances.indexOf(instance);
      if (index !== -1) {
        instances.splice(index, 1);
        instance.dispose();
        console.log(`Disposed instance of ${name}`);
        return;
      }
    }
  }

  /**
   * Dispose of all instances of a plugin
   * @param {string} name - Plugin name
   */
  disposeAllInstances(name) {
    const instances = this.instances.get(name);
    if (instances) {
      for (const instance of instances) {
        instance.dispose();
      }
      this.instances.set(name, []);
      console.log(`Disposed all instances of ${name}`);
    }
  }

  /**
   * Clear all instances
   */
  clearAllInstances() {
    for (const instances of this.instances.values()) {
      for (const instance of instances) {
        instance.dispose();
      }
    }
    this.instances.clear();
    console.log('Cleared all plugin instances');
  }

  /**
   * Check if a plugin is registered
   * @param {string} name - Plugin name
   * @returns {boolean} True if registered
   */
  has(name) {
    return this.plugins.has(name);
  }

  /**
   * Get plugin count
   * @returns {number} Number of registered plugins
   */
  getPluginCount() {
    return this.plugins.size;
  }

  /**
   * Get statistics
   * @returns {Object} Factory statistics
   */
  getStats() {
    return {
      registeredPlugins: this.plugins.size,
      categories: this.getCategories().length,
      tags: this.getTags().length,
      totalInstances: this.getAllInstances().length,
      instancesByPlugin: Object.fromEntries(
        Array.from(this.instances.entries()).map(([name, instances]) => [name, instances.length])
      )
    };
  }
}

// Export singleton instance
const PluginFactory = new PluginFactoryClass();
export default PluginFactory;
