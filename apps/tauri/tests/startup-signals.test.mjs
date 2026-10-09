import test from 'node:test';
import assert from 'node:assert/strict';
import { BootGate } from '../src/boot-gate.ts';
import { waitForVisible } from '../src/window-visibility.ts';
import { normalizeError, errorTitle } from '../src/error-report.ts';

const boot = (revision, phase) => ({ revision, phase, message: null, elapsedMs: 0, recoverable: true });

test('a ready Channel event beating the initial reply never regresses readiness', async () => {
  const gate = new BootGate();
  gate.observe(boot(4, 'ready'));
  const result = await gate.wait(boot(1, 'restoringConfig'));
  assert.equal(result.phase, 'ready');
  assert.equal(gate.observe(boot(3, 'detectingDevices')).revision, 4);
});

test('the boot gate stays pending until a terminal event without polling', async () => {
  const gate = new BootGate();
  let completed = false;
  const ready = gate.wait(boot(0, 'starting')).then(result => { completed = true; return result; });
  gate.observe(boot(2, 'initializingIo'));
  await Promise.resolve();
  assert.equal(completed, false);
  gate.observe(boot(3, 'failed'));
  assert.equal((await ready).phase, 'failed');
});

test('visibility subscription handles Show racing registration and cleans its listener', async () => {
  let visible = false;
  let removed = 0;
  await waitForVisible({
    isVisible: async () => visible,
    observe: async callback => { visible = true; callback(); return () => { removed++; }; },
  });
  assert.equal(removed, 1);
});

test('an already visible window needs no listener', async () => {
  await waitForVisible({ isVisible: async () => true, observe: async () => { throw new Error('should not subscribe'); } });
});

test('audio command failures retain suggestions and details without displaying JSON as the message', () => {
  const details = { kind: 'deviceBusy', message: 'Input device already in use.', direction: 'capture', backend: 'nativeWasapi', hresult: '0x8889000A', deviceId: 'mic', deviceName: 'Microphone', technicalDetails: ['InitializeSharedAudioStream'], suggestions: ['Close exclusive recording, then retry.'] };
  const report = normalizeError({ kind: 'audio_startup', details });
  assert.equal(report.code, 'audio_device_busy');
  assert.equal(report.message, details.message);
  assert.deepEqual(report.details, details);
  assert.equal(errorTitle(report), 'Audio device in use');
});
