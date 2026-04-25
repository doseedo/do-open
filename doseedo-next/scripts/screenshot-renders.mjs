#!/usr/bin/env node
/**
 * Screenshot each harness HTML via headless Chromium.
 *
 *   node scripts/screenshot-renders.mjs
 */
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));

const jobs = [
  { html: 'nebula.html',   out: 'nebula.png',   viewport: { width: 1440, height: 900 } },
  { html: 'tapeloom.html', out: 'tapeloom.png', viewport: { width: 720, height: 820 } },
];

const browser = await chromium.launch({ headless: true });
for (const job of jobs) {
  const ctx = await browser.newContext({ viewport: job.viewport, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  const url = pathToFileURL(path.join(here, job.html)).href;
  console.log('→', url);
  page.on('pageerror', (err) => console.error('  PAGE ERROR:', err.message));
  page.on('console',   (msg) => { if (msg.type() === 'error') console.error('  console.error:', msg.text()); });
  await page.goto(url, { waitUntil: 'networkidle' });
  // wait for our harness flag or up to 10s for Babel+render
  await page.waitForFunction(() => window.__renderReady === true, { timeout: 15000 }).catch(() => {});
  // allow a tick for React to paint
  await page.waitForTimeout(1200);
  const outPath = path.join(here, job.out);
  // Screenshot only the rendered shell, not the page letterboxing
  const shell = await page.$('.rr-shell');
  if (shell) {
    await shell.screenshot({ path: outPath });
    console.log('  wrote', outPath);
  } else {
    console.error('  no .rr-shell found — full-page fallback');
    await page.screenshot({ path: outPath, fullPage: true });
  }
  await ctx.close();
}
await browser.close();
