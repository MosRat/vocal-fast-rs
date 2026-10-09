import test from 'node:test';
import assert from 'node:assert/strict';
import { initializeWorkspace } from '../src/workspace-startup.ts';

function fixture(overrides = {}) {
  const calls = [];
  return {
    calls,
    io: {
      subscribe: async () => { calls.push('subscribe'); },
      waitForShell: async () => {},
      snapshot: async () => ({ boot: { phase: 'ready' }, modelRestorePending: true, configRevision: 7 }),
      resources: async () => null,
      present: () => { calls.push('present'); },
      afterPaint: async () => { calls.push('paint'); return true; },
      restore: async revision => { calls.push(`restore:${revision}`); },
      refreshLibrary: async () => { calls.push('scan'); },
      report: error => { calls.push(error); },
      ...overrides,
    },
  };
}

test('a slow engine and library scan cannot delay a connected painted workspace', async () => {
  const neverCompletes = new Promise(() => {});
  const { io, calls } = fixture();
  io.restore = async revision => { calls.push(`restore:${revision}`); await neverCompletes; };
  io.refreshLibrary = async () => { calls.push('scan'); await neverCompletes; };
  await initializeWorkspace(io);
  assert.deepEqual(calls, ['subscribe', 'present', 'paint', 'restore:7', 'scan']);
});

test('restore waits for both initial state and the first visible paint', async () => {
  let paint;
  const { io, calls } = fixture({ afterPaint: () => new Promise(resolve => { paint = resolve; }) });
  const connecting = initializeWorkspace(io);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls, ['subscribe', 'present']);
  paint(true);
  await connecting;
  assert.deepEqual(calls, ['subscribe', 'present', 'restore:7', 'scan']);
});

test('a hidden webview waits for lightweight bootstrap before reading resources', async () => {
  let shellReady;
  const { io, calls } = fixture({ waitForShell: () => new Promise(resolve => { shellReady = resolve; }) });
  const connecting = initializeWorkspace(io);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls, ['subscribe']);
  shellReady();
  await connecting;
  assert.deepEqual(calls, ['subscribe', 'present', 'paint', 'restore:7', 'scan']);
});

test('auxiliary windows, an already claimed restore and failed boot never restore a model', async () => {
  for (const override of [
    { afterPaint: async () => false },
    { snapshot: async () => ({ boot: { phase: 'ready' }, modelRestorePending: false }) },
    { snapshot: async () => ({ boot: { phase: 'failed' }, modelRestorePending: true }) },
  ]) {
    const { io, calls } = fixture(override);
    await initializeWorkspace(io);
    assert(!calls.some(call => String(call).startsWith('restore:')));
    assert(calls.includes('scan'));
  }
});

test('an engine failure is reported after presenting the usable workspace', async () => {
  const failure = new Error('engine allocation failed');
  const { io, calls } = fixture({ restore: async () => { throw failure; } });
  await initializeWorkspace(io);
  await new Promise(resolve => setImmediate(resolve));
  assert(calls.indexOf('present') < calls.indexOf(failure));
});
