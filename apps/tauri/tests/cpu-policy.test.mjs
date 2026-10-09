import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultConfig } from '../src/types.ts';
import { automaticConfig, rebaseConfigDraft } from '../src/config-draft.ts';
import { cpuStreamingTiming, validateCpuConfig } from '../src/cpu-policy.ts';

test('CPU configuration autosaves independently of the manual session profile', () => {
  const saved = defaultConfig();
  const draft = structuredClone(saved);
  draft.cpu.threads = 2;
  draft.cpu.pinToPerformanceCores = true;
  draft.tensorrt.profile = 'realtime_44960';
  const sent = automaticConfig(draft, saved);
  assert.equal(sent.cpu.threads, 2);
  assert.equal(sent.cpu.pinToPerformanceCores, true);
  assert.equal(sent.tensorrt.profile, saved.tensorrt.profile);
  const remote = structuredClone(sent);
  remote.outputDevice = 'new-endpoint';
  const rebased = rebaseConfigDraft(draft, saved, remote);
  assert.deepEqual(rebased.conflicts, []);
  assert.equal(rebased.config.cpu.threads, 2);
  assert.equal(rebased.config.outputDevice, 'new-endpoint');
  assert.equal(rebased.config.tensorrt.profile, 'realtime_44960');
});

test('CPU settings reject invalid pools and profiles expose their buffering cost', () => {
  const cpu = defaultConfig().cpu;
  assert.equal(validateCpuConfig(cpu), null);
  for (const threads of [-1, 0.1, 257, NaN]) assert.ok(validateCpuConfig({ ...cpu, threads }));
  assert.ok(validateCpuConfig({ ...cpu, spinWaitUs: 1001 }));
  assert.deepEqual(cpuStreamingTiming('reference'), { chunkMs: 512, warmupMs: 1536 });
  assert.deepEqual(cpuStreamingTiming('voice_changer'), { chunkMs: 512, warmupMs: 1536 });
  assert.deepEqual(cpuStreamingTiming('realtime_44960'), { chunkMs: 250, warmupMs: 3250 });
});
