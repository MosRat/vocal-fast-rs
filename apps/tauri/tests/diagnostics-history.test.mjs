import test from 'node:test';
import assert from 'node:assert/strict';
import { DiagnosticsHistory, MAX_POINTS, HISTORY_MS } from '../src/diagnostics-history.ts';

const audio = (elapsedMs, sessionId = '1', sequence = 1) => ({
  active: true, sessionId, sample: { sequence, elapsedMs, windowMs: 500 },
});

test('duplicate snapshots and late replies cannot fabricate samples or freshness', () => {
  const history = new DiagnosticsHistory();
  history.observe(audio(1000), 100);
  history.observe(audio(1000), 5000);
  history.observe(audio(500), 6000);
  assert.equal(history.values.length, 1);
  assert.equal(history.state(true, true, 6000), 'stale');
  history.observe(audio(7000), 7000);
  assert.equal(history.state(true, true, 7001), 'live');
  assert.equal(history.values[1].breakBefore, true);
});

test('history has both a fixed point cap and a 60-second time boundary', () => {
  const history = new DiagnosticsHistory();
  for (let i = 1; i <= 1000; i++) history.observe(audio(i * 100), i * 100);
  assert.equal(history.values.length, MAX_POINTS);
  const latest = history.latest.elapsedMs;
  assert(history.values.every(point => point.sample.elapsedMs >= latest - HISTORY_MS));
  history.observe(audio(200000), 200000);
  assert.equal(history.values.length, 1);
});

test('stop retains charts while a new session resets them before the first sample', () => {
  const history = new DiagnosticsHistory();
  history.observe(audio(1000), 1000);
  history.observe({ active: false, sessionId: null, sample: null }, 2000);
  assert.equal(history.state(false, true, 2000), 'paused');
  assert.equal(history.values.length, 1);
  history.observe({ active: true, sessionId: '2', sample: null }, 3000);
  assert.equal(history.latest, null);
  assert.equal(history.state(true, true, 3000), 'waiting');
  history.observe(audio(500, '2', 0), 3500);
  history.observe(audio(2000, '1', 2), 3600);
  assert.equal(history.values.length, 1);
  assert.equal(history.latest.elapsedMs, 500);
});

test('sequence wrap preserves order and invalid sample times are discarded', () => {
  const history = new DiagnosticsHistory();
  history.observe(audio(500, '1', 0xffffffff), 500);
  history.observe(audio(1000, '1', 0), 1000);
  for (const time of [NaN, Infinity, -1]) history.observe(audio(time), 2000);
  history.observe({ ...audio(1500), sample: { elapsedMs: 1500, windowMs: 0 } }, 1500);
  assert.equal(history.values.length, 2);
  assert.equal(history.latest.sequence, 0);
  assert.equal(history.state(true, false, 1000), 'stale');
});
