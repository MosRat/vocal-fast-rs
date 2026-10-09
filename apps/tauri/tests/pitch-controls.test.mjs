import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultConfig } from '../src/types.ts';
import { renderPitchMatching } from '../src/pitch-controls.ts';
import { automaticConfig, rebaseConfigDraft } from '../src/config-draft.ts';

test('matching defaults on and idle edits auto-save without committing a profile', () => {
  const saved = defaultConfig();
  assert.deepEqual(saved.conversion.autoPitch, { enabled: true, targetHz: 255 });
  const draft = structuredClone(saved);
  draft.conversion.autoPitch = { enabled: false, targetHz: 155 };
  draft.tensorrt.profile = 'voice_changer';
  const next = automaticConfig(draft, saved);
  assert.deepEqual(next.conversion.autoPitch, draft.conversion.autoPitch);
  assert.equal(next.tensorrt.profile, saved.tensorrt.profile);
  const remote = structuredClone(saved);
  remote.conversion.pitchSemitones = 2;
  const merged = rebaseConfigDraft(draft, saved, remote);
  assert.deepEqual(merged.conflicts, []);
  assert.equal(merged.config.conversion.pitchSemitones, 2);
  assert.equal(merged.config.conversion.autoPitch.enabled, false);
});

test('session lock disables every matching control and manual pitch remains additive', () => {
  const config = defaultConfig().conversion;
  const html = renderPitchMatching(config, true);
  assert.match(html, /role="switch"[^>]*checked[^>]*disabled/);
  assert.match(html, /id="autoPitchTarget"[^>]*disabled/);
  assert.equal((html.match(/data-pitch-target="\d+" disabled/g) ?? []).length, 2);
  assert.match(html, /Additional pitch below is added/);
  config.autoPitch.enabled = false;
  const off = renderPitchMatching(config, false);
  assert.doesNotMatch(off, /role="switch"[^>]*checked/);
  assert.match(off, /id="autoPitchTarget"[^>]*disabled/);
});
