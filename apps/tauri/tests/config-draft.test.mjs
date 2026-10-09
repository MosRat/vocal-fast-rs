import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultConfig } from '../src/types.ts';
import { automaticConfig, rebaseConfigDraft, sameConfig } from '../src/config-draft.ts';

test('automatic settings never commit a staged profile or selected model path', () => {
  const saved = defaultConfig();
  const draft = structuredClone(saved);
  draft.outputDevice = 'render-endpoint';
  draft.tensorrt.profile = 'voice_changer';
  draft.modelPath = 'another-model.vfrm';
  const config = automaticConfig(draft, saved);
  assert.equal(config.outputDevice, 'render-endpoint');
  assert.equal(config.tensorrt.profile, saved.tensorrt.profile);
  assert.equal(config.modelPath, saved.modelPath);
  assert.equal(draft.tensorrt.profile, 'voice_changer');
});

test('saving routing preserves a profile draft and rebases its revision', () => {
  const base = defaultConfig();
  const draft = structuredClone(base);
  draft.outputDevice = 'render-endpoint';
  draft.tensorrt.profile = 'voice_changer';
  const saved = automaticConfig(draft, base);
  const rebased = rebaseConfigDraft(draft, saved, saved);
  assert.deepEqual(rebased.conflicts, []);
  assert.equal(rebased.config.tensorrt.profile, 'voice_changer');
  assert(sameConfig(automaticConfig(rebased.config, saved), saved));
});

test('new edits and reversals made during a save survive acknowledgement', () => {
  const base = defaultConfig();
  const sent = { ...base, outputDevice: 'first' };
  for (const outputDevice of [null, 'second']) {
    const draft = { ...sent, outputDevice };
    const rebased = rebaseConfigDraft(draft, sent, sent);
    assert.equal(rebased.config.outputDevice, outputDevice);
    assert.deepEqual(rebased.conflicts, []);
  }
});

test('disjoint edits from two windows merge, including nested conversion fields', () => {
  const base = defaultConfig();
  const draft = structuredClone(base);
  const saved = structuredClone(base);
  draft.outputDevice = 'render-endpoint';
  draft.conversion.pitchSemitones = 3;
  saved.inputDevice = 'capture-endpoint';
  saved.conversion.protect = 0.7;
  const rebased = rebaseConfigDraft(draft, base, saved);
  assert.deepEqual(rebased.conflicts, []);
  assert.equal(rebased.config.inputDevice, 'capture-endpoint');
  assert.equal(rebased.config.outputDevice, 'render-endpoint');
  assert.equal(rebased.config.conversion.pitchSemitones, 3);
  assert.equal(rebased.config.conversion.protect, 0.7);
});

test('overlapping edits report the precise field and preserve the local draft', () => {
  const base = defaultConfig();
  const draft = { ...base, outputDevice: 'local' };
  const saved = { ...base, outputDevice: 'remote' };
  const rebased = rebaseConfigDraft(draft, base, saved);
  assert.deepEqual(rebased.conflicts, ['outputDevice']);
  assert.equal(rebased.config.outputDevice, 'local');
});
