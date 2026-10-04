import { Channel, invoke } from '@tauri-apps/api/core';
import './style.css';

type Lifecycle = 'created' | 'loading' | 'ready' | 'running' | 'stopping' | 'stopped' | 'failed';
type Backend = 'passthrough' | 'rvcCpu';

interface ConversionConfig {
  pitchSemitones: number;
  indexRatio: number;
  protect: number;
  rmsMixRate: number;
}

interface GuiConfig {
  schemaVersion: number;
  inputFormat: { sampleRate: number; channels: number };
  outputFormat: { sampleRate: number; channels: number };
  conversion: ConversionConfig;
  backend: Backend;
  modelPath: string | null;
  inputDevice: string | null;
  outputDevice: string | null;
  blockFrames: number;
  queueBlocks: number;
}

interface AudioDeviceInfo {
  id: string;
  name: string;
  input: boolean;
  output: boolean;
  defaultInput: boolean;
  defaultOutput: boolean;
  sampleRate: number | null;
  channels: number | null;
}

interface AudioDeviceCatalog {
  host: string;
  devices: AudioDeviceInfo[];
}

interface AudioTelemetry {
  active: boolean;
  state: Lifecycle;
  inputDevice: string | null;
  outputDevice: string | null;
  inputFrames: number;
  outputFrames: number;
  processingMicros: number;
  failures: number;
  underflowSamples: number;
  queueOverflows: number;
  latencyP95Micros: number | null;
  streamState: string;
}

interface ErrorReport {
  code: string;
  message: string;
  recoverable: boolean;
}

interface Snapshot {
  lifecycle: Lifecycle;
  status: { session: Lifecycle; model: Lifecycle; engine: Lifecycle; audio: Lifecycle };
  config: GuiConfig;
  configRevision: number;
  model: {
    id: string;
    displayName: string;
    architecture: string;
    version: string;
    sampleRate: number;
    featureDims: number;
    usesF0: boolean;
  } | null;
  metrics: { inputFrames: number; processingMicros: number; outputFrames: number };
  engine: { state: Lifecycle; processedChunks: number; failures: number; lastError: string | null } | null;
  audio: AudioTelemetry;
  lastError: ErrorReport | null;
}

interface BackendEvent {
  type: 'snapshot' | 'metrics' | 'error';
  sequence: number;
  payload: Snapshot | Snapshot['metrics'] | ErrorReport;
}

const root = document.querySelector<HTMLDivElement>('#app');
if (!root) throw new Error('missing #app');
const appRoot = root;

let snapshot: Snapshot | null = null;
let devices: AudioDeviceCatalog | null = null;
let channelConnected = false;
let busy = false;
let localError: ErrorReport | null = null;
let refreshInFlight = false;
let lastEventSequence = 0;
let resyncInFlight = false;

const call = <T>(command: string, args?: Record<string, unknown>) => invoke<T>(command, args);

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character] ?? character);
}

function defaultConfig(): GuiConfig {
  return {
    schemaVersion: 1,
    inputFormat: { sampleRate: 48000, channels: 1 },
    outputFormat: { sampleRate: 48000, channels: 1 },
    conversion: { pitchSemitones: 0, indexRatio: 0, protect: 0.33, rmsMixRate: 1 },
    backend: 'passthrough',
    modelPath: null,
    inputDevice: null,
    outputDevice: null,
    blockFrames: 480,
    queueBlocks: 3,
  };
}

function deviceOptions(kind: 'input' | 'output', selected: string | null): string {
  const available = devices?.devices.filter((device) => kind === 'input' ? device.input : device.output) ?? [];
  const options = [`<option value="" ${selected ? '' : 'selected'}>System default</option>`];
  for (const device of available) {
    const suffix = device.sampleRate ? ` · ${device.sampleRate} Hz` : '';
    const selectedAttr = selected === device.id ? 'selected' : '';
    const isDefault = kind === 'input' ? device.defaultInput : device.defaultOutput;
    options.push(`<option value="${escapeHtml(device.id)}" ${selectedAttr}>${escapeHtml(device.name)}${suffix}${isDefault ? ' · default' : ''}</option>`);
  }
  return options.join('');
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
}

function render(): void {
  const current = snapshot;
  const config = current?.config ?? defaultConfig();
  const audio = current?.audio;
  const report = localError ?? current?.lastError;
  const running = current?.lifecycle === 'running';
  const transitional = current?.lifecycle === 'loading' || current?.lifecycle === 'stopping';
  const inputDevice = audio?.inputDevice ?? config.inputDevice;
  const outputDevice = audio?.outputDevice ?? config.outputDevice;
  appRoot.innerHTML = `
    <main class="shell">
      <header class="topbar">
        <div class="brand"><div class="brand-mark">V</div><div><p class="eyebrow">VOCAL-FAST / CONTROL ROOM</p><h1>Realtime voice conversion</h1><p class="muted">Rust audio worker · WASAPI/CPAL device bridge · revision ${current?.configRevision ?? 0}</p></div></div>
        <div class="status ${current?.lifecycle ?? 'created'}"><span></span><strong>${current?.lifecycle ?? 'created'}</strong><small>${audio?.active ? 'audio stream online' : channelConnected ? 'control channel online' : 'connecting'}</small></div>
      </header>

      <section class="hero card">
        <div class="hero-copy"><span class="label">ACTIVE MODEL</span><h2>${current?.model ? escapeHtml(current.model.displayName) : 'No model loaded'}</h2><p>${current?.model ? `${escapeHtml(current.model.architecture)} · ${current.model.sampleRate} Hz · ${current.model.featureDims} feature dims${current.model.usesF0 ? ' · F0 enabled' : ''}` : 'Load a .vfr cache or a model manifest to prepare a session.'}</p></div>
        <div class="hero-actions"><button id="start" class="primary" ${busy || transitional || !current?.model || running ? 'disabled' : ''}>Start stream</button><button id="stop" ${busy || current?.lifecycle !== 'running' ? 'disabled' : ''}>Stop</button><button id="demo" class="quiet" ${busy || !running || audio?.active ? 'disabled' : ''}>Test block</button></div>
      </section>

      <section class="grid">
        <article class="card span-2">
          <div class="card-title"><div><span class="eyebrow">MODEL LIBRARY</span><h2>Load and manage a model</h2></div><span class="pill">VFRM / RVC</span></div>
          <div class="field-line"><label class="grow">Cache or manifest path<input id="modelPath" value="${escapeHtml(config.modelPath ?? '')}" placeholder="C:\\models\\voice.vfr" /></label><button id="load" ${busy || transitional ? 'disabled' : ''}>Load model</button><button id="unload" class="quiet" ${busy || transitional || !current?.model || running ? 'disabled' : ''}>Unload</button></div>
        </article>

        <article class="card">
          <div class="card-title"><div><span class="eyebrow">AUDIO ROUTING</span><h2>Devices</h2></div><button id="refreshDevices" class="icon-button" title="Refresh device list">↻</button></div>
          <p class="muted small">${devices ? `${devices.host} · ${devices.devices.length} devices` : 'Device list not loaded'}</p>
          <label>Capture<select id="inputDevice" ${running ? 'disabled' : ''}>${deviceOptions('input', inputDevice)}</select></label>
          <label>Render<select id="outputDevice" ${running ? 'disabled' : ''}>${deviceOptions('output', outputDevice)}</select></label>
          <p class="hint">Selections are saved with the next configuration revision.</p>
        </article>

        <article class="card">
          <div class="card-title"><div><span class="eyebrow">SESSION CONFIG</span><h2>Stream and engine</h2></div><span class="pill">optimistic sync</span></div>
          <label>Backend<select id="backend" ${running ? 'disabled' : ''}><option value="passthrough" ${config.backend === 'passthrough' ? 'selected' : ''}>Passthrough</option><option value="rvcCpu" ${config.backend === 'rvcCpu' ? 'selected' : ''}>RVC CPU</option></select></label>
          <div class="two-col"><label>Sample rate<input id="inputRate" type="number" min="1" value="${config.inputFormat.sampleRate}" ${running ? 'disabled' : ''} /></label><label>Channels<input id="channels" type="number" min="1" max="8" value="${config.inputFormat.channels}" ${running ? 'disabled' : ''} /></label></div>
          <div class="two-col"><label>Block frames<input id="blockFrames" type="number" min="1" max="8192" value="${config.blockFrames}" ${running ? 'disabled' : ''} /></label><label>Queue blocks<input id="queueBlocks" type="number" min="1" max="64" value="${config.queueBlocks}" ${running ? 'disabled' : ''} /></label></div>
          <button id="save" class="primary full" ${busy || running ? 'disabled' : ''}>Save configuration</button>
        </article>

        <article class="card">
          <div class="card-title"><div><span class="eyebrow">CONVERSION</span><h2>Voice controls</h2></div><span class="pill">live-safe</span></div>
          <label>Pitch shift <output id="pitchValue">${config.conversion.pitchSemitones.toFixed(1)} st</output><input id="pitch" type="range" min="-24" max="24" step="0.1" value="${config.conversion.pitchSemitones}" /></label>
          <label>Index ratio <output id="indexValue">${config.conversion.indexRatio.toFixed(2)}</output><input id="index" type="range" min="0" max="1" step="0.01" value="${config.conversion.indexRatio}" /></label>
          <label>Protect <output id="protectValue">${config.conversion.protect.toFixed(2)}</output><input id="protect" type="range" min="0" max="1" step="0.01" value="${config.conversion.protect}" /></label>
          <label>RMS mix <output id="rmsValue">${config.conversion.rmsMixRate.toFixed(2)}</output><input id="rms" type="range" min="0" max="1" step="0.01" value="${config.conversion.rmsMixRate}" /></label>
        </article>

        <article class="card span-2 telemetry-card">
          <div class="card-title"><div><span class="eyebrow">TELEMETRY</span><h2>Runtime health</h2></div><span class="pill ${audio?.active ? 'live' : ''}">${audio?.active ? 'streaming' : 'idle'}</span></div>
          <dl class="metrics"><div><dt>Input frames</dt><dd>${formatNumber(current?.metrics.inputFrames ?? 0)}</dd></div><div><dt>Output frames</dt><dd>${formatNumber(current?.metrics.outputFrames ?? 0)}</dd></div><div><dt>Processing</dt><dd>${current ? (current.metrics.processingMicros / 1000).toFixed(2) : '0.00'} ms</dd></div><div><dt>p95 audio age</dt><dd>${audio?.latencyP95Micros ? `${(audio.latencyP95Micros / 1000).toFixed(2)} ms` : '—'}</dd></div><div><dt>Underflow samples</dt><dd>${formatNumber(audio?.underflowSamples ?? 0)}</dd></div><div><dt>Queue overflows</dt><dd>${formatNumber(audio?.queueOverflows ?? 0)}</dd></div></dl>
          <div class="telemetry-foot"><span>${audio?.streamState ?? 'Stopped'} · ${audio?.inputDevice ? escapeHtml(audio.inputDevice) : 'default input'} → ${audio?.outputDevice ? escapeHtml(audio.outputDevice) : 'default output'}</span><span>worker failures: ${audio?.failures ?? current?.engine?.failures ?? 0}</span></div>
        </article>
      </section>
      ${report ? `<section class="error"><strong>${escapeHtml(report.code)}</strong><span>${escapeHtml(report.message)}</span>${report.recoverable ? '<button id="clearError">Dismiss</button>' : ''}</section>` : ''}
      <footer><span>Audio blocks stay on the Rust side; the UI receives snapshots and metrics.</span><span>Configuration revision ${current?.configRevision ?? 0}</span></footer>
    </main>`;
  wireHandlers();
}

function readConfigForm(): GuiConfig {
  const current = snapshot?.config ?? defaultConfig();
  const number = (id: string, fallback: number) => Number(document.querySelector<HTMLInputElement>(`#${id}`)?.value ?? fallback);
  const sampleRate = number('inputRate', current.inputFormat.sampleRate);
  const channels = number('channels', current.inputFormat.channels);
  return {
    ...current,
    backend: (document.querySelector<HTMLSelectElement>('#backend')?.value as Backend) ?? current.backend,
    inputFormat: { sampleRate, channels },
    outputFormat: { sampleRate, channels },
    inputDevice: document.querySelector<HTMLSelectElement>('#inputDevice')?.value || null,
    outputDevice: document.querySelector<HTMLSelectElement>('#outputDevice')?.value || null,
    blockFrames: number('blockFrames', current.blockFrames),
    queueBlocks: number('queueBlocks', current.queueBlocks),
    conversion: {
      ...current.conversion,
      pitchSemitones: number('pitch', current.conversion.pitchSemitones),
      indexRatio: number('index', current.conversion.indexRatio),
      protect: number('protect', current.conversion.protect),
      rmsMixRate: number('rms', current.conversion.rmsMixRate),
    },
  };
}

async function runAction(action: () => Promise<unknown>): Promise<void> {
  busy = true;
  localError = null;
  try {
    const result = await action();
    if (isSnapshot(result)) snapshot = result;
  } catch (error) {
    localError = normalizeError(error);
  } finally {
    busy = false;
    render();
  }
}

function isSnapshot(value: unknown): value is Snapshot {
  return typeof value === 'object' && value !== null && 'lifecycle' in value && 'configRevision' in value;
}

function normalizeError(error: unknown): ErrorReport {
  if (typeof error === 'object' && error !== null && 'kind' in error) {
    const typed = error as { kind: string; details?: unknown };
    return { code: typed.kind, message: typeof typed.details === 'string' ? typed.details : JSON.stringify(typed.details ?? typed.kind), recoverable: true };
  }
  return { code: 'command_error', message: String(error), recoverable: true };
}

function wireHandlers(): void {
  const byId = <T extends HTMLElement>(id: string) => document.querySelector<T>(`#${id}`);
  byId<HTMLButtonElement>('load')?.addEventListener('click', () => runAction(async () => { const path = byId<HTMLInputElement>('modelPath')?.value.trim(); if (!path) throw new Error('Model path is required'); return call<Snapshot>('load_model', { path }); }));
  byId<HTMLButtonElement>('unload')?.addEventListener('click', () => runAction(() => call<Snapshot>('unload_model')));
  byId<HTMLButtonElement>('start')?.addEventListener('click', () => runAction(() => call<Snapshot>('start_session')));
  byId<HTMLButtonElement>('stop')?.addEventListener('click', () => runAction(() => call<Snapshot>('stop_session')));
  byId<HTMLButtonElement>('clearError')?.addEventListener('click', () => runAction(() => call<Snapshot>('clear_error')));
  byId<HTMLButtonElement>('refreshDevices')?.addEventListener('click', () => runAction(async () => { devices = await call<AudioDeviceCatalog>('list_audio_devices'); return devices; }));
  byId<HTMLButtonElement>('save')?.addEventListener('click', () => runAction(() => call<Snapshot>('update_config', { config: readConfigForm(), expectedRevision: snapshot?.configRevision ?? 0 })));
  byId<HTMLButtonElement>('demo')?.addEventListener('click', () => runAction(async () => { const rate = snapshot?.config.inputFormat.sampleRate ?? 48000; const samples = Array.from({ length: snapshot?.config.blockFrames ?? 480 }, (_, index) => Math.sin(index * 2 * Math.PI * 440 / rate) * 0.05); return call<Snapshot>('process_block', { samples, sampleRate: rate, channels: 1, sequence: Date.now() }); }));
  const sliderFields: ReadonlyArray<readonly [string, string, number]> = [['pitch', 'pitchValue', 1], ['index', 'indexValue', 2], ['protect', 'protectValue', 2], ['rms', 'rmsValue', 2]];
  for (const [id, outputId, digits] of sliderFields) {
    byId<HTMLInputElement>(id)?.addEventListener('input', (event) => { const value = Number((event.target as HTMLInputElement).value); const output = byId<HTMLOutputElement>(outputId); if (output) output.value = value.toFixed(digits) + (id === 'pitch' ? ' st' : ''); });
  }
}

async function connect(): Promise<void> {
  const events = new Channel<BackendEvent>();
  events.onmessage = (event) => {
    if (event.sequence <= lastEventSequence) return;
    if (lastEventSequence !== 0 && event.sequence > lastEventSequence + 1) {
      void resyncSnapshot();
    }
    lastEventSequence = event.sequence;
    if (event.type === 'snapshot') {
      snapshot = event.payload as Snapshot;
      if (!snapshot.lastError) localError = null;
    }
    if (event.type === 'metrics' && snapshot) snapshot.metrics = event.payload as Snapshot['metrics'];
    if (event.type === 'error') localError = event.payload as ErrorReport;
    render();
  };
  try {
    snapshot = await call<Snapshot>('subscribe', { channel: events });
    channelConnected = true;
    devices = await call<AudioDeviceCatalog>('list_audio_devices');
  } catch (error) {
    localError = normalizeError(error);
  }
  render();
}

async function resyncSnapshot(): Promise<void> {
  if (resyncInFlight) return;
  resyncInFlight = true;
  try {
    snapshot = await call<Snapshot>('get_snapshot');
    render();
  } catch {
    // The periodic refresh will retry while the control channel is alive.
  } finally {
    resyncInFlight = false;
  }
}

window.setInterval(async () => {
  if (!channelConnected || busy || refreshInFlight) return;
  refreshInFlight = true;
  try {
    snapshot = await call<Snapshot>('get_snapshot');
    render();
  } catch {
    // Keep the last usable snapshot; command errors are sent through the channel.
  } finally {
    refreshInFlight = false;
  }
}, 1000);

render();
void connect();
