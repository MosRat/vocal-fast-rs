import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultConfig } from '../src/types.ts';
import { automaticConfig, rebaseConfigDraft } from '../src/config-draft.ts';
import { speakerIdMaximum, validateSpeakerId } from '../src/speaker-policy.ts';

test('speaker selection validates the runtime representation and model embedding range', () => {
  assert.equal(defaultConfig().speakerId, 0);
  for (const invalid of [-1, 0.5, NaN, Infinity, 2_147_483_648]) {
    assert.ok(validateSpeakerId(invalid));
  }
  assert.equal(validateSpeakerId(2_147_483_647), null);
  assert.equal(validateSpeakerId(0, 1), null);
  assert.ok(validateSpeakerId(1, 1));
  assert.equal(validateSpeakerId(2, 3), null);
  assert.ok(validateSpeakerId(3, 3));
  for (const unknown of [undefined, null]) {
    assert.equal(speakerIdMaximum(unknown), 2_147_483_647);
    assert.equal(validateSpeakerId(7, unknown), null);
  }
});

test('speaker and CPU policy autosave for every native backend while the profile remains staged', () => {
  for (const backend of ['rvcCpu', 'directMl', 'tensorRtRtx']) {
    const base = { ...defaultConfig(), backend };
    const draft = structuredClone(base);
    draft.speakerId = 2;
    draft.cpu.threads = 4;
    draft.tensorrt.profile = 'realtime_44960';
    const sent = automaticConfig(draft, base);
    assert.equal(sent.speakerId, 2);
    assert.equal(sent.cpu.threads, 4);
    assert.equal(sent.tensorrt.profile, base.tensorrt.profile);
    const acknowledged = rebaseConfigDraft(draft, sent, sent);
    assert.deepEqual(acknowledged.conflicts, []);
    assert.equal(acknowledged.config.speakerId, 2);
    assert.equal(acknowledged.config.tensorrt.profile, 'realtime_44960');
  }
});

test('speaker edits merge with remote thread changes and report concurrent speaker conflicts', () => {
  const base = defaultConfig();
  const draft = { ...base, speakerId: 1 };
  const remote = { ...base, cpu: { ...base.cpu, threads: 2 } };
  const merged = rebaseConfigDraft(draft, base, remote);
  assert.deepEqual(merged.conflicts, []);
  assert.equal(merged.config.speakerId, 1);
  assert.equal(merged.config.cpu.threads, 2);
  const conflict = rebaseConfigDraft(draft, base, { ...remote, speakerId: 2 });
  assert.deepEqual(conflict.conflicts, ['speakerId']);
  assert.equal(conflict.config.speakerId, 1);
});
