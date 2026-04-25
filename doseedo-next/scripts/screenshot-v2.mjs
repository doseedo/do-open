#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const jobs = [
  { html: 'vision-v2-helixref.html',  out: 'vision-v2-helixref.png',  viewport: { width: 1520, height: 960 } },
  { html: 'vision-v2-strataref.html', out: 'vision-v2-strataref.png', viewport: { width: 600, height: 860 } },
];
const browser = await chromium.launch({ headless: true });
for (const job of jobs) {
  const ctx = await browser.newContext({ viewport: job.viewport, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  await page.goto(pathToFileURL(path.join(here, job.html)).href, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.__renderReady === true, { timeout: 15000 }).catch(() => {});
  await page.waitForTimeout(1500);
  const shell = await page.$('.rr-shell');
  if (shell) { await shell.screenshot({ path: path.join(here, job.out) }); console.log('wrote', job.out); }
  else { await page.screenshot({ path: path.join(here, job.out), fullPage: true }); }
  await ctx.close();
}
await browser.close();
