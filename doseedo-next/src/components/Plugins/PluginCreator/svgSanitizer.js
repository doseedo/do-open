/**
 * SVG Sanitizer — lightweight allowlist-based sanitizer using native DOMParser.
 * No npm dependencies.
 */

// NOTE: tagName is compared via .toLowerCase(), so all entries MUST be lowercase.
// SVG camelCase names (linearGradient, feGaussianBlur, etc.) must be lowercased here.
const ALLOWED_ELEMENTS = new Set([
  'svg', 'g', 'defs', 'symbol', 'use',
  'circle', 'ellipse', 'line', 'path', 'polygon', 'polyline', 'rect',
  'text', 'tspan', 'textpath',
  'clippath', 'mask', 'pattern', 'image',
  'lineargradient', 'radialgradient', 'stop',
  'filter', 'fegaussianblur', 'feoffset', 'femerge', 'femergenode',
  'feflood', 'fecomposite', 'feblend', 'fecolormatrix', 'fedropshadow',
  'feturbulence', 'fedisplacementmap', 'femorphology',
  'fediffuselighting', 'fespecularlighting',
  'fedistantlight', 'fepointlight', 'fespotlight',
  'feconvolvematrix', 'fecomponenttransfer', 'fefuncr', 'fefuncg', 'fefuncb', 'fefunca',
  'feimage', 'fetile',
  'title', 'desc', 'metadata',
]);

const BANNED_ELEMENTS = new Set([
  'script', 'foreignObject', 'iframe', 'object', 'embed', 'applet',
  'form', 'input', 'button', 'select', 'textarea',
]);

/**
 * Sanitize an SVG string — removes scripts, event handlers, and external references.
 * @param {string} svgString - Raw SVG markup
 * @returns {string|null} Sanitized SVG string, or null if parsing failed
 */
export function sanitizeSVG(svgString) {
  if (!svgString || typeof svgString !== 'string') return null;

  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(svgString, 'image/svg+xml');

    // Check for parse errors
    const errorNode = doc.querySelector('parsererror');
    if (errorNode) return null;

    const svgEl = doc.documentElement;
    if (svgEl.tagName !== 'svg') return null;

    walkAndSanitize(svgEl);

    // Ensure viewBox exists BEFORE we override width/height
    if (!svgEl.getAttribute('viewBox')) {
      const w = parseFloat(svgEl.getAttribute('width')) || 100;
      const h = parseFloat(svgEl.getAttribute('height')) || 100;
      svgEl.setAttribute('viewBox', `0 0 ${w} ${h}`);
    }
    // Force SVG to fill its container div
    svgEl.setAttribute('width', '100%');
    svgEl.setAttribute('height', '100%');

    const serializer = new XMLSerializer();
    return serializer.serializeToString(svgEl);
  } catch {
    return null;
  }
}

function walkAndSanitize(element) {
  // Process children in reverse so removal doesn't skip elements
  const children = Array.from(element.children);
  for (const child of children) {
    const tag = child.tagName.toLowerCase();

    if (BANNED_ELEMENTS.has(tag) || (!ALLOWED_ELEMENTS.has(tag) && tag !== 'svg')) {
      child.remove();
      continue;
    }

    // Remove dangerous attributes
    const attrs = Array.from(child.attributes);
    for (const attr of attrs) {
      const name = attr.name.toLowerCase();
      // Remove event handlers
      if (name.startsWith('on')) {
        child.removeAttribute(attr.name);
        continue;
      }
      // Check href/xlink:href for external or javascript URLs
      if (name === 'href' || name === 'xlink:href') {
        const val = attr.value.trim();
        if (val.startsWith('javascript:') || val.startsWith('data:text/html')) {
          child.removeAttribute(attr.name);
          continue;
        }
        // Allow internal references (#id) and data: image URLs
        if (!val.startsWith('#') && !val.startsWith('data:image/')) {
          // For <image> elements, allow https URLs (for textures/sprites)
          if (tag === 'image' && val.startsWith('https://')) {
            // allowed
          } else {
            child.removeAttribute(attr.name);
          }
        }
      }
    }

    walkAndSanitize(child);
  }
}

/**
 * Inject a texture pattern into an SVG string.
 * Finds elements with fill="texture:{regionId}" and replaces with a <pattern> reference.
 * @param {string} svgString - Sanitized SVG markup
 * @param {string} regionId - Texture region identifier (e.g. "body", "cap")
 * @param {string} imageUrl - Resolved image URL (https or data:)
 * @returns {string} Updated SVG string
 */
export function injectTexturePattern(svgString, regionId, imageUrl) {
  if (!svgString || !regionId || !imageUrl) return svgString;

  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(svgString, 'image/svg+xml');
    const svgEl = doc.documentElement;
    if (!svgEl || svgEl.tagName !== 'svg') return svgString;

    // Get or create <defs>
    let defs = svgEl.querySelector('defs');
    if (!defs) {
      defs = doc.createElementNS('http://www.w3.org/2000/svg', 'defs');
      svgEl.insertBefore(defs, svgEl.firstChild);
    }

    const patternId = `texture-${regionId}`;

    // Remove existing pattern with same id
    const existing = defs.querySelector(`#${patternId}`);
    if (existing) existing.remove();

    // Create pattern
    const pattern = doc.createElementNS('http://www.w3.org/2000/svg', 'pattern');
    pattern.setAttribute('id', patternId);
    pattern.setAttribute('patternUnits', 'userSpaceOnUse');
    pattern.setAttribute('width', '128');
    pattern.setAttribute('height', '128');

    const img = doc.createElementNS('http://www.w3.org/2000/svg', 'image');
    img.setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', imageUrl);
    img.setAttribute('href', imageUrl);
    img.setAttribute('width', '128');
    img.setAttribute('height', '128');
    img.setAttribute('preserveAspectRatio', 'xMidYMid slice');
    pattern.appendChild(img);
    defs.appendChild(pattern);

    // Replace fill="texture:{regionId}" with fill="url(#texture-{regionId})"
    const placeholder = `texture:${regionId}`;
    const replacement = `url(#${patternId})`;
    const allElements = svgEl.querySelectorAll('*');
    for (const el of allElements) {
      if (el.getAttribute('fill') === placeholder) {
        el.setAttribute('fill', replacement);
      }
      if (el.getAttribute('stroke') === placeholder) {
        el.setAttribute('stroke', replacement);
      }
    }

    const serializer = new XMLSerializer();
    return serializer.serializeToString(svgEl);
  } catch {
    return svgString;
  }
}
