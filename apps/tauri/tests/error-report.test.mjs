import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeError, errorTitle } from '../src/error-report.ts';

test('idle pipeline failure presents the cause and retains the active settings', () => {
  const report = normalizeError({ kind: 'config_update', details: { kind: 'engine', details: 'candidate CUDA context failed' } });
  assert.equal(report.code, 'config_update_failed');
  assert.match(report.message, /previous settings remain active/);
  assert.match(report.message, /candidate CUDA context failed/);
  assert.equal(report.recoverable, true);
  assert.equal(errorTitle(report), 'Settings update failed');
});

test('optimistic revision conflicts retain their original command code', () => {
  assert.equal(normalizeError({ kind: 'config_conflict', details: { expected: 1, actual: 2 } }).code, 'config_conflict');
});
