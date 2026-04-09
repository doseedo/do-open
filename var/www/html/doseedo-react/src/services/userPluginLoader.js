/**
 * userPluginLoader — fetches a /plugins/create plugin from the
 * chatbot-backend and instantiates it as a DSPPresetSlot in the studio.
 *
 * Three sources are unified behind a single API:
 *   1. Built-in DSP presets (src/lib/dsp-presets/) — synchronous lookup
 *   2. The user's saved plugins        (GET /_chat/api/plugin-projects/my)
 *   3. Community plugins by ID/slug    (GET /_chat/api/plugin-projects/{id})
 *
 * Studio FX picker code uses:
 *   - listAvailablePlugins({mine: true}) → flat list with {key,name,source}
 *   - createPluginSlot(audioContext, ref) where ref is either:
 *       {source:'builtin', key:'compressor'}
 *       {source:'user',    id:'<plugin-project-id-or-slug>'}
 *
 * Loaded community/user plugins are cached in memory by id so a slot
 * swap doesn't re-hit the network.
 */

import { DSPPresetSlot } from './dspPresetSlot.js';
import { DSP_PRESETS, listPresets } from '../lib/dsp-presets/index.js';
import {
  loadProject,
  listMyProjects,
  listCommunityProjects,
} from './pluginProjectsAPI.js';

const _projectCache = new Map(); // id-or-slug → project JSON

/**
 * Fetch a /plugins/create project by id or slug, cached.
 * Returns the full project (including dsp_config, plugin_config, components).
 */
export async function fetchUserPlugin(idOrSlug) {
  if (_projectCache.has(idOrSlug)) {
    return _projectCache.get(idOrSlug);
  }
  const project = await loadProject(idOrSlug);
  _projectCache.set(idOrSlug, project);
  return project;
}

/** Drop the cache for one plugin (call after save/edit). */
export function invalidateUserPlugin(idOrSlug) {
  _projectCache.delete(idOrSlug);
}

/** Drop the entire user-plugin cache (e.g. on logout). */
export function clearUserPluginCache() {
  _projectCache.clear();
}

/**
 * List every plugin the studio FX picker can offer, in a uniform shape.
 *
 * @param {Object} [opts]
 * @param {boolean} [opts.builtin=true]   include built-in DSP presets
 * @param {boolean} [opts.mine=false]     include the user's saved plugins
 * @param {boolean} [opts.community=false] include public community plugins
 * @param {Object}  [opts.communityParams] forwarded to listCommunityProjects
 *
 * Returns: Array<{
 *   key,             // unique handle for the picker
 *   source,          // 'builtin' | 'user' | 'community'
 *   id,              // builtin: preset key; user/community: project id
 *   name,
 *   category,
 *   description,
 * }>
 */
export async function listAvailablePlugins({
  builtin = true,
  mine = false,
  community = false,
  communityParams,
} = {}) {
  const out = [];

  if (builtin) {
    for (const p of listPresets()) {
      out.push({
        key:         `builtin:${p.key}`,
        source:      'builtin',
        id:          p.key,
        name:        p.name,
        category:    p.category,
        description: p.description,
      });
    }
  }

  if (mine) {
    try {
      const mineList = await listMyProjects();
      for (const p of mineList || []) {
        out.push({
          key:         `user:${p.id}`,
          source:      'user',
          id:          p.id,
          name:        p.name || 'Untitled',
          category:    p.category || 'user',
          description: p.description || '',
        });
      }
    } catch (e) {
      console.warn('[userPluginLoader] listMyProjects failed:', e.message);
    }
  }

  if (community) {
    try {
      const res = await listCommunityProjects(communityParams || {});
      for (const p of (res?.projects || [])) {
        out.push({
          key:         `community:${p.slug || p.id}`,
          source:      'community',
          id:          p.slug || p.id,
          name:        p.name || 'Untitled',
          category:    p.category || 'community',
          description: p.description || '',
        });
      }
    } catch (e) {
      console.warn('[userPluginLoader] listCommunityProjects failed:', e.message);
    }
  }

  return out;
}

/**
 * Resolve a plugin ref to a `dspConfig` JSON, fetching if needed.
 * Built-in presets are returned synchronously; user/community plugins
 * are fetched (and cached) from the chatbot-backend.
 *
 * @param {{source:string, id:string}} ref
 * @returns {Promise<{dspConfig:Object, meta:Object}>}
 */
export async function resolvePluginRef(ref) {
  if (!ref || !ref.source || !ref.id) {
    throw new Error('resolvePluginRef: ref must have {source,id}');
  }

  if (ref.source === 'builtin') {
    const preset = DSP_PRESETS[ref.id];
    if (!preset) throw new Error(`unknown built-in preset: ${ref.id}`);
    return {
      dspConfig: preset.dspConfig,
      meta: {
        name: preset.name,
        category: preset.category,
        description: preset.description,
        source: 'builtin',
      },
    };
  }

  if (ref.source === 'user' || ref.source === 'community') {
    const project = await fetchUserPlugin(ref.id);
    // saveProject() persists `dsp_config` (snake_case) on the wire;
    // accept both keys to be future-proof.
    const dspConfig = project.dsp_config || project.dspConfig;
    if (!dspConfig) {
      throw new Error(`plugin ${ref.id} has no dsp_config`);
    }
    return {
      dspConfig,
      meta: {
        name: project.name,
        category: project.category || ref.source,
        description: project.description || '',
        source: ref.source,
        author: project.author_name,
        thumbnail: project.thumbnail_data,
      },
    };
  }

  throw new Error(`unknown plugin source: ${ref.source}`);
}

/**
 * One-call helper for the studio FX picker:
 *   const slot = await createPluginSlot(ctx, {source:'user', id:'abc'});
 *   slot.input  // → connect upstream node
 *   slot.output // → connect downstream node
 */
export async function createPluginSlot(audioContext, ref) {
  const { dspConfig, meta } = await resolvePluginRef(ref);
  return new DSPPresetSlot(audioContext, dspConfig, meta);
}
