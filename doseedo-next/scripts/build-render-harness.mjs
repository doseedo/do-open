#!/usr/bin/env node
/**
 * Build a standalone HTML renderer for a PluginDSL.
 *
 * Inlines every refRenderer .js file (after stripping import/export
 * lines) into <script type="text/babel"> tags so Babel-standalone can
 * parse the JSX in the browser — same pattern the reference plugin
 * editor uses at /Users/hydroadmin/Downloads/plugin editor/helix/.
 *
 *   node scripts/build-render-harness.mjs <dsl.json> <out.html>
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..');
const refDir = path.join(repoRoot, 'src/components/Plugins/PluginCreator/refRenderer');

const dslPath = process.argv[2];
const outPath = process.argv[3];
if (!dslPath || !outPath) {
  console.error('usage: build-render-harness.mjs <dsl.json> <out.html>');
  process.exit(1);
}
const dsl = JSON.parse(fs.readFileSync(dslPath, 'utf8'));
const css = fs.readFileSync(path.join(refDir, 'refRenderer.css'), 'utf8');

// Strip ES module syntax so the inlined JSX works inside a global-scope
// script tag. "import X from …" lines are dropped wholesale (the
// components they pull in are inlined below them in dep order). "export"
// keywords are removed to let declarations land on the global scope.
function stripESM(src) {
  return src
    .replace(/^\s*import[\s\S]*?;\s*$/gm, '')
    .replace(/^\s*export\s+default\s+/gm, '')
    .replace(/^\s*export\s+/gm, '')
    .replace(/\bReact\.useState\b/g, 'React.useState') // safe no-op
    // React hooks imported from 'react' are now referenced via window.React
    // directly (we inject `const { useState, useRef, useEffect } = React;`
    // at the top of each file's inlined block).
  ;
}

function loadInlined(relPath) {
  const src = fs.readFileSync(path.join(refDir, relPath), 'utf8');
  return stripESM(src);
}

// Files in dependency order
const files = [
  'primitives/index.js',
  'displays/index.js',
  'modules/index.js',
  'rows/index.js',
  'renderDSL.js',
];

const blocks = files
  .map((f) => `
  // ─── ${f} ───────────────────────────────────────────────
  {
    const { useState, useRef, useEffect } = React;
    ${loadInlined(f)}
  }
`)
  .join('\n');

// Emit components to window so later blocks can consume them across
// script-tag boundaries. We do this by NOT wrapping in a block scope —
// switch to top-level. Replaced `{ ... }` wrappers with flat sections.
const flatBlocks = files
  .map((f) => {
    const body = loadInlined(f);
    return `// ─── ${f} ───\n${body}\n`;
  })
  .join('\n');

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${dsl.meta.name} — ref-renderer preview</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone@7.24.7/babel.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <style>
    html, body { margin: 0; padding: 0; background: #0a0908; color: #d9d4cd;
      font-family: system-ui, sans-serif; min-height: 100vh; }
    #root { display: flex; align-items: center; justify-content: center;
      min-height: 100vh; padding: 20px; box-sizing: border-box; }
    ${css}
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel" data-presets="env,react" data-type="module">
    const { useState, useRef, useEffect, useMemo } = React;

    ${flatBlocks}

    const DSL = ${JSON.stringify(dsl)};

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(<RenderDSL dsl={DSL} />);

    // Signal ready for Playwright
    window.__renderReady = true;
  </script>
</body>
</html>
`;

fs.writeFileSync(outPath, html);
console.log('wrote', outPath, '(', html.length, 'bytes )');
