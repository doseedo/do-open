/**
 * useEditStream.test.js — unit tests for the inbound op-translation
 * layer in useEditStream.js. Doesn't open an EventSource; tests drive
 * `applyOp` directly with synthetic state + a mock dispatch.
 *
 * Covers the plugin-op symmetry that makes peer-to-peer plugin sync
 * actually work: each inbound op should both dispatch a state mutation
 * (so the rack UI reflects the change) AND apply to the live slot
 * (so audio updates immediately) — without echoing the edit back.
 *
 * Run with: node src/hooks/useEditStream.test.js
 */

import { strict as assert } from 'node:assert';
import { applyOp } from './editStreamApply.js';
import liveTrackChainRegistry from '../lib/liveTrackChainRegistry.js';

let passed = 0, failed = 0;
const failures = [];
async function test(name, fn) {
  try { await fn(); passed++; process.stdout.write(`  ok  ${name}\n`); }
  catch (err) { failed++; failures.push({ name, err }); process.stdout.write(`  fail ${name}\n`); }
}

// ── Helpers ─────────────────────────────────────────────────────────────

function buildState(trackUuid = 'aaaa1234') {
  return {
    buses: [{
      id: 'bus-1',
      tracks: [{
        id: 'track-1',
        uuid: trackUuid,
        logicTrackIndex: 0,
        logicPlugins: [
          {
            plugin_id: 154,
            plugin_name: 'Compressor',
            parameters: [
              { id: 0, name: 'Threshold', value: -20 },
              { id: 1, name: 'Ratio', value: 4 },
            ],
            bypassed: false,
          },
        ],
      }],
    }],
    selectedTrack: null,
  };
}

function makeDispatch() {
  const calls = [];
  const dispatch = (action) => calls.push(action);
  return { calls, dispatch };
}

function registerFakeChain(trackId, trackUuid, slots) {
  liveTrackChainRegistry.register(trackId, {
    trackIndex: 0, ginstid: null,
    trackUuid: (trackUuid || '').toLowerCase(),
    slots,
  });
}

function makeFakeSlot() {
  const calls = [];
  return {
    calls,
    setLogicParam: (id, value, opts) => calls.push(['param', id, value, !!opts?.broadcast]),
    setBypassed: (b, opts) => calls.push(['bypass', b, !!opts?.broadcast]),
  };
}

// ── Tests ───────────────────────────────────────────────────────────────

await test('set_plugin_param dispatches UPDATE_TRACK_LOGIC_PLUGINS + applies to slot', async () => {
  const state = buildState();
  const { calls, dispatch } = makeDispatch();
  const slot = makeFakeSlot();
  registerFakeChain('track-1', 'aaaa1234', [slot]);

  applyOp({
    op: 'set_plugin_param',
    args: { track_uuid: 'aaaa1234', slot: 0, param_id: 0, value: -45 },
  }, dispatch, state);

  // 1. Dispatched the reducer action carrying the new logicPlugins array
  //    with param 0's value swapped to -45.
  assert.equal(calls.length, 1);
  assert.equal(calls[0].type, 'UPDATE_TRACK_LOGIC_PLUGINS');
  assert.equal(calls[0].payload.trackUuid, 'aaaa1234');
  const newParams = calls[0].payload.logicPlugins[0].parameters;
  const threshold = newParams.find((p) => p.id === 0);
  assert.equal(threshold.value, -45);

  // 2. Applied to the live slot, broadcast suppressed (broadcast:false →
  //    the boolean we recorded is false).
  assert.equal(slot.calls.length, 1);
  assert.deepEqual(slot.calls[0], ['param', 0, -45, false]);

  liveTrackChainRegistry.clear();
});

await test('set_plugin_bypass updates state + slot with broadcast:false', async () => {
  const state = buildState();
  const { calls, dispatch } = makeDispatch();
  const slot = makeFakeSlot();
  registerFakeChain('track-1', 'aaaa1234', [slot]);

  applyOp({
    op: 'set_plugin_bypass',
    args: { track_uuid: 'aaaa1234', slot: 0, bypassed: true },
  }, dispatch, state);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].payload.logicPlugins[0].bypassed, true);
  assert.deepEqual(slot.calls[0], ['bypass', true, false]);

  liveTrackChainRegistry.clear();
});

await test('set_plugin_params_batch applies all params at once', async () => {
  const state = buildState();
  const { calls, dispatch } = makeDispatch();
  const slot = makeFakeSlot();
  registerFakeChain('track-1', 'aaaa1234', [slot]);

  applyOp({
    op: 'set_plugin_params_batch',
    args: {
      track_uuid: 'aaaa1234', slot: 0,
      params: [
        { param_id: 0, value: -30 },
        { param_id: 1, value: 8 },
      ],
    },
  }, dispatch, state);

  assert.equal(calls.length, 1);
  const params = calls[0].payload.logicPlugins[0].parameters;
  assert.equal(params.find((p) => p.id === 0).value, -30);
  assert.equal(params.find((p) => p.id === 1).value, 8);
  // Slot received both writes.
  assert.equal(slot.calls.length, 2);
  assert.equal(slot.calls[0][3], false); // broadcast suppressed
  assert.equal(slot.calls[1][3], false);

  liveTrackChainRegistry.clear();
});

await test('add_plugin appends a stub entry to the track', async () => {
  const state = buildState();
  const { calls, dispatch } = makeDispatch();

  applyOp({
    op: 'add_plugin',
    args: { track_uuid: 'aaaa1234', plugin_name: 'Tape Delay' },
  }, dispatch, state);

  assert.equal(calls.length, 1);
  const newPlugins = calls[0].payload.logicPlugins;
  assert.equal(newPlugins.length, 2);
  assert.equal(newPlugins[1].plugin_name, 'Tape Delay');
  assert.equal(newPlugins[1]._pending, true);
});

await test('remove_plugin drops the slot at the given index', async () => {
  const state = buildState();
  const { calls, dispatch } = makeDispatch();

  applyOp({
    op: 'remove_plugin',
    args: { track_uuid: 'aaaa1234', slot: 0 },
  }, dispatch, state);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].payload.logicPlugins.length, 0);
});

await test('unknown op is silently ignored', async () => {
  const state = buildState();
  const { calls, dispatch } = makeDispatch();

  applyOp({ op: 'set_plugin_typo', args: {} }, dispatch, state);

  assert.equal(calls.length, 0, 'no dispatch should fire for unknown op');
});

await test('set_plugin_param missing slot in registry → state still updates', async () => {
  // No registered chain — state should still update so the rack UI
  // reflects the inbound peer edit. Missing-slot is the bounce-cache
  // fallback; not having a live engine is acceptable for state sync.
  const state = buildState();
  const { calls, dispatch } = makeDispatch();
  liveTrackChainRegistry.clear();

  applyOp({
    op: 'set_plugin_param',
    args: { track_uuid: 'aaaa1234', slot: 0, param_id: 0, value: -55 },
  }, dispatch, state);

  assert.equal(calls.length, 1);
  const newParams = calls[0].payload.logicPlugins[0].parameters;
  assert.equal(newParams.find((p) => p.id === 0).value, -55);
});

// ── Done ────────────────────────────────────────────────────────────────

process.stdout.write(`\n${passed} passed, ${failed} failed\n`);
if (failed > 0) {
  for (const { name, err } of failures) {
    process.stderr.write(`\n--- ${name} ---\n${err.stack || err.message}\n`);
  }
  process.exit(1);
}
