import type { GuiConfig, ModelComponents, ProcessingBackendCatalog } from './types';
import { escapeHtml } from './format';
import { cpuStreamingTiming } from './cpu-policy';
import './cpu-settings.css';

export function renderCpuSettings(config: GuiConfig, catalog: ProcessingBackendCatalog | null, locked: boolean, prefix: string, components: ModelComponents | null): string {
  const cpu = config.cpu;
  const info = catalog?.cpu;
  const recommended = info?.recommendedThreads ?? 6;
  const logical = info?.logicalProcessors;
  const counts = [...new Set([1, 2, 3, 4, 6, 8, 12, 16, ...(logical ? [logical] : []), cpu.threads])].filter(n => n > 0 && n <= 256).sort((a, b) => a - b);
  const timing = cpuStreamingTiming(config.tensorrt.profile);
  const disabled = locked ? 'disabled' : '';
  return `<div class="cpu-settings">
    <div class="cpu-settings-grid"><label class="field-label" for="${prefix}cpuThreads">Inference threads<select id="${prefix}cpuThreads" data-cpu-setting="threads" ${disabled}>
      <option value="0" ${cpu.threads === 0 ? 'selected' : ''}>Automatic · ${recommended} threads</option>
      ${counts.map(n => `<option value="${n}" ${cpu.threads === n ? 'selected' : ''}>${n} threads${n === recommended ? ' · suggested' : ''}</option>`).join('')}
    </select></label><div class="cpu-engine-summary"><strong>Native FP32 · oneDNN</strong><span>${logical ? `${logical} logical processors · ` : ''}${escapeHtml(info?.operatorVersion ?? 'Standalone CPU operators')}</span><span>No GPU driver or TensorRT plans required</span></div></div>
    <label class="cpu-toggle"><input type="checkbox" data-cpu-setting="pinToPerformanceCores" ${cpu.pinToPerformanceCores ? 'checked' : ''} ${disabled}/><span>Prefer performance cores<small>Uses Windows CPU topology; falls back to OS scheduling when unavailable.</small></span></label>
    <div class="cpu-stream-note"><strong>${timing.chunkMs} ms output chunks · ${(timing.warmupMs / 1000).toFixed(2)} s initial context warmup</strong><p>These are buffering intervals, not measured end-to-end latency. Sustained inference must finish within each chunk. Use a GPU if diagnostics show queue drops or repeated underflows.${config.tensorrt.profile === 'realtime_44960' ? ' The full-context window can exceed the realtime budget on desktop CPUs.' : ''}</p></div>
    <p class="cpu-model-note">${components?.automaticTensorRt ? 'Complete model · Load extracts acoustic weights and prepares HuBERT, RMVPE, retrieval and Synth on the CPU.' : 'Supports RVC v1 / 256-feature and v2 / 768-feature Synth. Provide matching HuBERT and RMVPE weights in a complete VFRM bundle or companion sources; v1 requires the learned HuBERT projection.'}</p>
    <details class="cpu-advanced"><summary>CPU scheduling &amp; diagnostics</summary><label class="field-label" for="${prefix}cpuSpin">Worker spin wait<select id="${prefix}cpuSpin" data-cpu-setting="spinWaitUs" ${disabled}>${[...new Set([0, 50, 100, 250, cpu.spinWaitUs])].sort((a,b) => a-b).map(n => `<option value="${n}" ${cpu.spinWaitUs === n ? 'selected' : ''}>${n === 0 ? 'Sleep between operators · recommended' : `${n} µs · increased CPU use`}</option>`).join('')}</select></label><label class="cpu-toggle"><input type="checkbox" data-cpu-setting="profile" ${cpu.profile ? 'checked' : ''} ${disabled}/><span>Operator profiling<small>Adds per-operator timing overhead; leave off for normal use.</small></span></label></details>
    <p class="cpu-model-note">CPU settings save automatically while idle. Session profiles require Save profile: Reference and Voice changer use the compact CPU context; Applio uses full context. Thread policy or context changes prepare a replacement engine; pitch and protect reuse it.</p>
  </div>`;
}
