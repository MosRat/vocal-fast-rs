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
  inputFormat: { sampleRate: number; channels: number };
  outputFormat: { sampleRate: number; channels: number };
  conversion: ConversionConfig;
  backend: Backend;
  modelPath: string | null;
}

interface ErrorReport {
  code: string;
  message: string;
  recoverable: boolean;
}

interface Snapshot {
  lifecycle: Lifecycle;
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
  lastError: ErrorReport | null;
}

interface BackendEvent {
  type: 'snapshot' | 'metrics' | 'error';
  payload: Snapshot | Snapshot['metrics'] | ErrorReport;
}

const root = document.querySelector<HTMLDivElement>('#app');
if (!root) throw new Error('missing #app');
const appRoot = root;

let snapshot: Snapshot | null = null;
let channelConnected = false;
let busy = false;
let localError: ErrorReport | null = null;

const call = <T>(command: string, args?: Record<string, unknown>) => invoke<T>(command, args);

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character] ?? character);
}

function defaultConfig(): GuiConfig {
  return { inputFormat: { sampleRate: 16000, channels: 1 }, outputFormat: { sampleRate: 16000, channels: 1 }, conversion: { pitchSemitones: 0, indexRatio: 0, protect: 0.33, rmsMixRate: 1 }, backend: 'passthrough', modelPath: null };
}

function render(): void {
  const current = snapshot;
  const config = current?.config ?? defaultConfig();
  const report = localError ?? current?.lastError;
  appRoot.innerHTML = `
    <main class="shell">
      <header class="topbar">
        <div><p class="eyebrow">VOCAL-FAST</p><h1>Realtime voice conversion</h1><p class="muted">Rust control plane · Tauri channel · revision ${current?.configRevision ?? 0}</p></div>
        <div class="status ${current?.lifecycle ?? 'created'}"><span></span>${current?.lifecycle ?? 'created'}</div>
      </header>
      <section class="grid">
        <article class="card span-2">
          <div class="card-title"><h2>Session</h2><span class="pill">${channelConnected ? 'channel connected' : 'connecting'}</span></div>
          <div class="session-row"><div class="model-box"><span class="label">Model</span><strong>${current?.model ? escapeHtml(current.model.displayName) : 'No model loaded'}</strong><small>${current?.model ? `${escapeHtml(current.model.architecture)} · ${current.model.sampleRate} Hz · ${current.model.featureDims} dims` : 'Load a VFRM cache or model manifest to begin.'}</small></div><div class="actions"><button id="start" class="primary" ${busy || !current?.model || current.lifecycle === 'running' ? 'disabled' : ''}>Start</button><button id="stop" ${busy || current?.lifecycle !== 'running' ? 'disabled' : ''}>Stop</button><button id="demo" ${busy || current?.lifecycle !== 'running' ? 'disabled' : ''}>Process test block</button></div></div>
          <div class="field-line"><label>Model path <input id="modelPath" value="${escapeHtml(config.modelPath ?? '')}" placeholder="C:\\models\\voice.vfr" /></label><button id="load" ${busy ? 'disabled' : ''}>Load model</button></div>
        </article>
        <article class="card"><div class="card-title"><h2>Configuration</h2><span class="pill">optimistic sync</span></div><label>Backend<select id="backend"><option value="passthrough" ${config.backend === 'passthrough' ? 'selected' : ''}>Passthrough</option><option value="rvcCpu" ${config.backend === 'rvcCpu' ? 'selected' : ''}>RVC CPU</option></select></label><div class="two-col"><label>Input Hz<input id="inputRate" type="number" min="1" value="${config.inputFormat.sampleRate}" /></label><label>Output Hz<input id="outputRate" type="number" min="1" value="${config.outputFormat.sampleRate}" /></label></div><label>Pitch shift <output id="pitchValue">${config.conversion.pitchSemitones.toFixed(1)} st</output><input id="pitch" type="range" min="-24" max="24" step="0.1" value="${config.conversion.pitchSemitones}" /></label><label>Index ratio <output id="indexValue">${config.conversion.indexRatio.toFixed(2)}</output><input id="index" type="range" min="0" max="1" step="0.01" value="${config.conversion.indexRatio}" /></label><button id="save" class="primary" ${busy ? 'disabled' : ''}>Save configuration</button></article>
        <article class="card"><div class="card-title"><h2>Telemetry</h2><span class="pill">control plane</span></div><dl class="metrics"><div><dt>Input frames</dt><dd>${current?.metrics.inputFrames ?? 0}</dd></div><div><dt>Output frames</dt><dd>${current?.metrics.outputFrames ?? 0}</dd></div><div><dt>Processing</dt><dd>${current ? (current.metrics.processingMicros / 1000).toFixed(2) : '0.00'} ms</dd></div><div><dt>Engine failures</dt><dd>${current?.engine?.failures ?? 0}</dd></div></dl></article>
      </section>
      ${report ? `<section class="error"><strong>${escapeHtml(report.code)}</strong><span>${escapeHtml(report.message)}</span>${report.recoverable ? '<button id="clearError">Dismiss</button>' : ''}</section>` : ''}
      <footer><span>Configuration writes are atomic and revision checked.</span><span>Audio blocks stay on the Rust side.</span></footer>
    </main>`;
  wireHandlers();
}

function readConfigForm(): GuiConfig {
  const current = snapshot?.config ?? defaultConfig();
  const number = (id: string, fallback: number) => Number((document.querySelector<HTMLInputElement>(`#${id}`)?.value ?? fallback));
  return {
    ...current,
    backend: (document.querySelector<HTMLSelectElement>('#backend')?.value as Backend) ?? current.backend,
    inputFormat: { ...current.inputFormat, sampleRate: number('inputRate', current.inputFormat.sampleRate) },
    outputFormat: { ...current.outputFormat, sampleRate: number('outputRate', current.outputFormat.sampleRate) },
    conversion: { ...current.conversion, pitchSemitones: number('pitch', current.conversion.pitchSemitones), indexRatio: number('index', current.conversion.indexRatio) },
  };
}

async function runAction(action: () => Promise<unknown>): Promise<void> {
  busy = true; localError = null; render();
  try {
    const result = await action();
    if (isSnapshot(result)) snapshot = result;
  } catch (error) { localError = normalizeError(error); } finally { busy = false; render(); }
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
  byId<HTMLButtonElement>('start')?.addEventListener('click', () => runAction(() => call<Snapshot>('start_session')));
  byId<HTMLButtonElement>('stop')?.addEventListener('click', () => runAction(() => call<Snapshot>('stop_session')));
  byId<HTMLButtonElement>('clearError')?.addEventListener('click', () => runAction(() => call('clear_error')));
  byId<HTMLButtonElement>('save')?.addEventListener('click', () => runAction(() => call<Snapshot>('update_config', { config: readConfigForm(), expectedRevision: snapshot?.configRevision ?? 0 })));
  byId<HTMLButtonElement>('demo')?.addEventListener('click', () => runAction(async () => { const rate = snapshot?.config.inputFormat.sampleRate ?? 16000; const samples = Array.from({ length: 160 }, (_, index) => Math.sin(index * 2 * Math.PI * 440 / rate) * 0.05); return call<Snapshot>('process_block', { samples, sampleRate: rate, channels: 1, sequence: Date.now() }); }));
  byId<HTMLInputElement>('pitch')?.addEventListener('input', (event) => { const value = Number((event.target as HTMLInputElement).value); const output = byId<HTMLOutputElement>('pitchValue'); if (output) output.value = `${value.toFixed(1)} st`; });
  byId<HTMLInputElement>('index')?.addEventListener('input', (event) => { const value = Number((event.target as HTMLInputElement).value); const output = byId<HTMLOutputElement>('indexValue'); if (output) output.value = value.toFixed(2); });
}

async function connect(): Promise<void> {
  const events = new Channel<BackendEvent>();
  events.onmessage = (event) => { if (event.type === 'snapshot') snapshot = event.payload as Snapshot; if (event.type === 'metrics' && snapshot) snapshot.metrics = event.payload as Snapshot['metrics']; if (event.type === 'error') localError = event.payload as ErrorReport; render(); };
  try { snapshot = await call<Snapshot>('subscribe', { channel: events }); channelConnected = true; } catch (error) { localError = normalizeError(error); }
  render();
}

render();
void connect();
