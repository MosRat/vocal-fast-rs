import { Channel, invoke } from '@tauri-apps/api/core';
import { setTheme as setNativeTheme } from '@tauri-apps/api/app';
import { getCurrentWindow } from '@tauri-apps/api/window';
import './style.css';

type Lifecycle = 'created' | 'loading' | 'ready' | 'running' | 'stopping' | 'stopped' | 'failed';
type Backend = 'passthrough' | 'rvcCpu';
type ThemePreference = 'system' | 'light' | 'dark';
type IconName = 'waveform' | 'play' | 'stop' | 'sparkle' | 'folder' | 'refresh' | 'settings' | 'speaker' | 'sliders' | 'chart' | 'moon' | 'sun' | 'desktop' | 'minimize' | 'maximize' | 'restore' | 'close' | 'chevron' | 'check' | 'alert' | 'cpu' | 'database' | 'keyboard';

interface ConversionConfig { pitchSemitones: number; indexRatio: number; protect: number; rmsMixRate: number; }
interface GuiConfig {
  schemaVersion: number; inputFormat: { sampleRate: number; channels: number }; outputFormat: { sampleRate: number; channels: number };
  conversion: ConversionConfig; backend: Backend; modelPath: string | null; inputDevice: string | null; outputDevice: string | null; blockFrames: number; queueBlocks: number;
}
interface AudioDeviceInfo { id: string; name: string; input: boolean; output: boolean; defaultInput: boolean; defaultOutput: boolean; sampleRate: number | null; channels: number | null; }
interface AudioDeviceCatalog { host: string; devices: AudioDeviceInfo[]; }
interface AudioTelemetry { active: boolean; state: Lifecycle; inputDevice: string | null; outputDevice: string | null; inputFrames: number; outputFrames: number; processingMicros: number; failures: number; underflowSamples: number; queueOverflows: number; latencyP95Micros: number | null; streamState: string; }
interface ErrorReport { code: string; message: string; recoverable: boolean; }
interface Snapshot {
  lifecycle: Lifecycle; status: { session: Lifecycle; model: Lifecycle; engine: Lifecycle; audio: Lifecycle }; config: GuiConfig; configRevision: number;
  model: { id: string; displayName: string; architecture: string; version: string; sampleRate: number; featureDims: number; usesF0: boolean } | null;
  metrics: { inputFrames: number; processingMicros: number; outputFrames: number };
  engine: { state: Lifecycle; processedChunks: number; failures: number; lastError: string | null } | null; audio: AudioTelemetry; lastError: ErrorReport | null;
}
interface BackendEvent { type: 'snapshot' | 'metrics' | 'error'; sequence: number; payload: Snapshot | Snapshot['metrics'] | ErrorReport; }

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
let themePreference: ThemePreference = 'system';
let windowMaximized = false;
let activeSection = 'studio';
let closeInFlight = false;
let closeHookInstalled = false;
let draftConfig: GuiConfig | null = null;
let renderTimer: number | undefined;
const lastActionAt = new Map<string, number>();
const MIN_ACTION_GAP_MS = 240;
const systemTheme = window.matchMedia?.('(prefers-color-scheme: dark)');
const call = <T>(command: string, args?: Record<string, unknown>) => invoke<T>(command, args);
function runningInTauri(): boolean { return '__TAURI_INTERNALS__' in window; }

function escapeHtml(value: string): string { return value.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character] ?? character); }

function icon(name: IconName, size = 18): string {
  const paths: Record<IconName, string> = {
    waveform: '<path d="M3 12h2l2.2-6 3.6 12 2.2-8 1.8 4H21"/>', play: '<path d="m8 5 11 7-11 7V5Z" fill="currentColor" stroke="none"/>', stop: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>',
    sparkle: '<path d="m12 3 1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3Z"/><path d="m19 15 .6 2.4L22 18l-2.4.6L19 21l-.6-2.4L16 18l2.4-.6L19 15Z"/>',
    folder: '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h4l2 2H18.5A2.5 2.5 0 0 1 21 9.5v7A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5v-9Z"/>', refresh: '<path d="M20 11a8 8 0 0 0-14.9-4L3 10"/><path d="M3 5v5h5"/><path d="M4 13a8 8 0 0 0 14.9 4L21 14"/><path d="M21 19v-5h-5"/>',
    settings: '<path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z"/><path d="m19.4 15 .1.1a1.8 1.8 0 0 1-2.5 2.5l-.1-.1a1.8 1.8 0 0 0-3 .8v.2a1.8 1.8 0 0 1-3.6 0v-.2a1.8 1.8 0 0 0-3-.8l-.1.1a1.8 1.8 0 0 1-2.5-2.5l.1-.1a1.8 1.8 0 0 0-.8-3h-.2a1.8 1.8 0 0 1 0-3.6h.2a1.8 1.8 0 0 0 .8-3l-.1-.1a1.8 1.8 0 0 1 2.5-2.5l.1.1a1.8 1.8 0 0 0 3-.8v-.2a1.8 1.8 0 0 1 3.6 0v.2a1.8 1.8 0 0 0 3 .8l.1-.1a1.8 1.8 0 0 1 2.5 2.5l-.1.1a1.8 1.8 0 0 0 .8 3h.2a1.8 1.8 0 0 1 0 3.6h-.2a1.8 1.8 0 0 0-.8 3Z"/>',
    speaker: '<path d="M4 10v4h4l5 4V6L8 10H4Z"/><path d="M17 9a4.3 4.3 0 0 1 0 6M19.5 6.5a8 8 0 0 1 0 11"/>', sliders: '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2" fill="var(--surface-1)"/><circle cx="15" cy="12" r="2" fill="var(--surface-1)"/><circle cx="11" cy="18" r="2" fill="var(--surface-1)"/>', chart: '<path d="M4 19V5M4 19h17"/><path d="m7 15 3-4 3 2 5-7"/>',
    moon: '<path d="M20 15.5A8 8 0 0 1 8.5 4 8 8 0 1 0 20 15.5Z"/>', sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>', desktop: '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
    minimize: '<path d="M5 12h14"/>', maximize: '<rect x="5" y="5" width="14" height="14" rx="1"/>', restore: '<path d="M8 8h11v11H8z"/><path d="M5 16V5h11"/>', close: '<path d="m6 6 12 12M18 6 6 18"/>', chevron: '<path d="m8 10 4 4 4-4"/>', check: '<path d="m5 12 4 4L19 6"/>', alert: '<path d="M12 3 2.8 20h18.4L12 3Z"/><path d="M12 9v4M12 16h.01"/>',
    cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 9h6v6H9zM9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>', database: '<ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7"/>', keyboard: '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M6 10h.01M9 10h.01M12 10h.01M15 10h.01M18 10h.01M7 14h10"/>',
  };
  return `<svg class="icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]}</svg>`;
}

function defaultConfig(): GuiConfig { return { schemaVersion: 1, inputFormat: { sampleRate: 48000, channels: 1 }, outputFormat: { sampleRate: 48000, channels: 1 }, conversion: { pitchSemitones: 0, indexRatio: 0, protect: 0.33, rmsMixRate: 1 }, backend: 'passthrough', modelPath: null, inputDevice: null, outputDevice: null, blockFrames: 480, queueBlocks: 3 }; }
function formatNumber(value: number): string { return new Intl.NumberFormat().format(value); }
function lifecycleLabel(value: Lifecycle): string { return value.charAt(0).toUpperCase() + value.slice(1); }
function statusTone(value: Lifecycle): string { return value === 'running' || value === 'ready' ? 'positive' : value === 'failed' ? 'critical' : value === 'loading' || value === 'stopping' ? 'progress' : 'neutral'; }
function themeLabel(): string { return themePreference === 'system' ? 'System theme' : themePreference === 'dark' ? 'Dark theme' : 'Light theme'; }

function deviceOptions(kind: 'input' | 'output', selected: string | null): string {
  const available = devices?.devices.filter((device) => kind === 'input' ? device.input : device.output) ?? [];
  const options = [`<option value="" ${selected ? '' : 'selected'}>System default</option>`];
  for (const device of available) { const suffix = device.sampleRate ? ` · ${device.sampleRate} Hz` : ''; const selectedAttr = selected === device.id ? 'selected' : ''; const isDefault = kind === 'input' ? device.defaultInput : device.defaultOutput; options.push(`<option value="${escapeHtml(device.id)}" ${selectedAttr}>${escapeHtml(device.name)}${suffix}${isDefault ? ' · default' : ''}</option>`); }
  return options.join('');
}

function button(id: string, label: string, iconName?: IconName, kind = 'secondary', disabled = false, title?: string): string { return `<button id="${id}" class="button ${kind}" ${disabled ? 'disabled' : ''} ${title ? `title="${escapeHtml(title)}"` : ''}>${iconName ? icon(iconName, 16) : ''}<span>${label}</span></button>`; }
function panelHeader(kicker: string, title: string, action = ''): string { return `<div class="panel-header"><div><div class="kicker">${kicker}</div><h2>${title}</h2></div>${action}</div>`; }
function metric(label: string, value: string, iconName: IconName, tone = ''): string { return `<div class="metric ${tone}"><div class="metric-icon">${icon(iconName, 16)}</div><div><dt>${label}</dt><dd>${value}</dd></div></div>`; }
function slider(id: string, label: string, value: string, number: number, min: number, max: number, step: number): string { return `<label class="slider-field"><span><span>${label}</span><output id="${id}Value">${value}</output></span><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${number}"/></label>`; }
function scheduleRender(delay = 70): void { if (renderTimer !== undefined) return; renderTimer = window.setTimeout(() => { renderTimer = undefined; render(); }, delay); }

function renderTitlebar(): string { return `<header class="titlebar" data-tauri-drag-region><div class="titlebar-brand"><div class="app-glyph">${icon('waveform', 17)}</div><span>vocal-fast</span><span class="titlebar-divider"></span><span class="titlebar-context">Studio</span></div><div class="titlebar-actions"><button id="minimizeWindow" class="window-button" title="Minimize">${icon('minimize', 15)}</button><button id="maximizeWindow" class="window-button" title="Maximize">${icon(windowMaximized ? 'restore' : 'maximize', 14)}</button><button id="closeWindow" class="window-button close" title="Close">${icon('close', 15)}</button></div></header>`; }

function renderSidebar(): string {
  const navItem = (id: string, label: string, iconName: IconName) => `<button class="nav-item ${activeSection === id ? 'active' : ''}" data-section="${id}">${icon(iconName, 17)}<span>${label}</span></button>`;
  return `<aside class="sidebar"><div class="sidebar-heading"><div class="brand-mark">V</div><div><strong>Vocal Studio</strong><span>Realtime workspace</span></div></div><div class="nav-label">Workspace</div><nav class="nav-list" aria-label="Workspace navigation">${navItem('studio', 'Studio', 'waveform')}${navItem('audio', 'Audio routing', 'speaker')}${navItem('tuning', 'Voice controls', 'sliders')}${navItem('diagnostics', 'Diagnostics', 'chart')}</nav><div class="sidebar-spacer"></div><div class="shortcut-hint">${icon('keyboard', 15)}<span><kbd>Space</kbd> Start or stop</span></div><button id="themeButton" class="theme-button" title="Change appearance">${icon(themePreference === 'dark' ? 'moon' : themePreference === 'light' ? 'sun' : 'desktop', 16)}<span>${themeLabel()}</span><span class="chevron">${icon('chevron', 14)}</span></button><div class="sidebar-footer"><span class="connection-dot ${channelConnected ? 'online' : ''}"></span>${channelConnected ? 'Control channel connected' : 'Connecting to runtime…'}</div></aside>`;
}

function render(): void {
  const current = snapshot; const config = draftConfig ?? current?.config ?? defaultConfig(); const audio = current?.audio; const report = localError ?? current?.lastError; const running = current?.lifecycle === 'running'; const transitional = current?.lifecycle === 'loading' || current?.lifecycle === 'stopping'; const model = current?.model; const latency = audio?.latencyP95Micros ? `${(audio.latencyP95Micros / 1000).toFixed(2)} ms` : '—'; const status = current?.lifecycle ?? 'created';
  const inputName = audio?.active ? audio.inputDevice : (config.inputDevice ? devices?.devices.find((item) => item.id === config.inputDevice)?.name : null); const outputName = audio?.active ? audio.outputDevice : (config.outputDevice ? devices?.devices.find((item) => item.id === config.outputDevice)?.name : null);
  appRoot.innerHTML = `${renderTitlebar()}<div class="app-layout">${renderSidebar()}<main class="content"><div class="page-heading"><div><div class="kicker">REALTIME VOICE WORKSPACE</div><h1>Make every take sound like you.</h1><p>Configure your voice chain, start the stream, and keep an eye on the audio path.</p></div><div class="headline-status ${statusTone(status)}"><span class="status-dot"></span><div><strong>${lifecycleLabel(status)}</strong><small>${audio?.active ? 'Audio stream online' : channelConnected ? 'Runtime ready for control' : 'Waiting for runtime'}</small></div></div></div>
    <section class="hero-panel ${running ? 'is-running' : ''}"><div class="hero-glow"></div><div class="hero-grid"></div><div class="hero-content"><div class="eyebrow-row"><span class="eyebrow-chip">${running ? 'LIVE SESSION' : 'SESSION CONTROL'}</span><span class="revision">Revision ${current?.configRevision ?? 0}</span></div><h2>${model ? escapeHtml(model.displayName) : 'No model loaded'}</h2><p>${model ? `${escapeHtml(model.architecture)} · ${model.sampleRate.toLocaleString()} Hz · ${model.featureDims} feature dimensions${model.usesF0 ? ' · F0 enabled' : ''}` : 'Load a .vfr cache or model manifest to prepare your voice chain.'}</p><div class="hero-wave" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div><div class="hero-actions">${button('start', running ? 'Streaming' : 'Start stream', running ? 'check' : 'play', 'primary', busy || transitional || !model || running)}${button('stop', 'Stop', 'stop', 'secondary', busy || !running)}${button('demo', 'Test block', 'sparkle', 'ghost', busy || !running || !!audio?.active, 'Run one synthetic block through the engine')}</div></div><div class="hero-orb"><div class="orb-ring ring-one"></div><div class="orb-ring ring-two"></div><div class="orb-core">${icon('waveform', 42)}</div><span class="orb-caption">${running ? 'Listening' : 'Standby'}</span></div></section>
    <section class="workspace-grid"><article class="panel model-panel span-2" data-section-target="studio">${panelHeader('MODEL LIBRARY', 'Voice model', `<span class="panel-badge">${model ? icon('check', 13) + ' Loaded' : icon('database', 13) + ' Awaiting model'}</span>`)}<div class="model-layout"><div class="model-avatar">${model ? escapeHtml(model.displayName.slice(0, 1).toUpperCase()) : icon('database', 25)}</div><div class="model-details"><strong>${model ? escapeHtml(model.displayName) : 'Choose a model to begin'}</strong><span>${model ? `${escapeHtml(model.id)} · v${escapeHtml(model.version)}` : 'VFRM caches and RVC manifests are supported'}</span>${model ? `<div class="tag-row"><span class="tag">${escapeHtml(model.architecture)}</span><span class="tag">${model.sampleRate.toLocaleString()} Hz</span>${model.usesF0 ? '<span class="tag accent">F0 enabled</span>' : ''}</div>` : ''}</div><div class="model-actions">${button('unload', 'Unload', undefined, 'ghost', busy || transitional || !model || running)}<div class="model-path-row"><input id="modelPath" aria-label="Model path" value="${escapeHtml(config.modelPath ?? '')}" placeholder="Paste a model path…"/><button id="load" class="button primary compact" ${busy || transitional ? 'disabled' : ''} title="Load model">${icon('folder', 15)}<span>Load</span></button></div></div></div></article>
    <article class="panel audio-panel" data-section-target="audio">${panelHeader('AUDIO ROUTING', 'Devices', button('refreshDevices', 'Refresh', 'refresh', 'icon-only', false, 'Refresh device list'))}<div class="host-line"><span class="live-indicator ${audio?.active ? 'active' : ''}"></span><span>${devices ? escapeHtml(devices.host) : 'Audio host not queried'}</span><span class="host-count">${devices ? `${devices.devices.length} devices` : '—'}</span></div><label class="field-label">Capture<select id="inputDevice" ${running ? 'disabled' : ''}>${deviceOptions('input', config.inputDevice)}</select></label><label class="field-label">Render<select id="outputDevice" ${running ? 'disabled' : ''}>${deviceOptions('output', config.outputDevice)}</select></label><div class="route-summary"><span>${icon('speaker', 14)}${escapeHtml(inputName ?? 'System default')}</span><span class="route-arrow">→</span><span>${escapeHtml(outputName ?? 'System default')}</span></div></article>
    <article class="panel tuning-panel" data-section-target="tuning">${panelHeader('VOICE CONTROLS', 'Shape the sound', '<span class="panel-badge subtle">Live-safe</span>')}<div class="slider-list">${slider('pitch', 'Pitch shift', `${config.conversion.pitchSemitones.toFixed(1)} st`, config.conversion.pitchSemitones, -24, 24, 0.1)}${slider('index', 'Index ratio', config.conversion.indexRatio.toFixed(2), config.conversion.indexRatio, 0, 1, 0.01)}${slider('protect', 'Unvoiced protect', config.conversion.protect.toFixed(2), config.conversion.protect, 0, 1, 0.01)}${slider('rms', 'RMS mix', config.conversion.rmsMixRate.toFixed(2), config.conversion.rmsMixRate, 0, 1, 0.01)}</div></article>
    <article class="panel config-panel" data-section-target="studio">${panelHeader('ENGINE & STREAM', 'Session configuration', '<span class="panel-badge subtle">Optimistic sync</span>')}<label class="field-label">Backend<select id="backend" ${running ? 'disabled' : ''}><option value="passthrough" ${config.backend === 'passthrough' ? 'selected' : ''}>Passthrough</option><option value="rvcCpu" ${config.backend === 'rvcCpu' ? 'selected' : ''}>RVC CPU</option></select></label><div class="form-grid"><label class="field-label">Sample rate<input id="inputRate" type="number" min="1" value="${config.inputFormat.sampleRate}" ${running ? 'disabled' : ''}/></label><label class="field-label">Channels<input id="channels" type="number" min="1" max="8" value="${config.inputFormat.channels}" ${running ? 'disabled' : ''}/></label><label class="field-label">Block frames<input id="blockFrames" type="number" min="1" max="8192" value="${config.blockFrames}" ${running ? 'disabled' : ''}/></label><label class="field-label">Queue depth<input id="queueBlocks" type="number" min="1" max="64" value="${config.queueBlocks}" ${running ? 'disabled' : ''}/></label></div>${button('save', 'Save configuration', 'check', 'primary full', busy || running)}</article>
    <article class="panel telemetry-panel span-2" data-section-target="diagnostics">${panelHeader('RUNTIME HEALTH', 'Audio telemetry', `<span class="stream-state ${audio?.active ? 'active' : ''}"><span class="status-dot"></span>${audio?.active ? 'Streaming' : 'Idle'}</span>`)}<div class="metrics-grid">${metric('Input frames', formatNumber(current?.metrics.inputFrames ?? 0), 'waveform')}${metric('Output frames', formatNumber(current?.metrics.outputFrames ?? 0), 'speaker')}${metric('Processing', current ? `${(current.metrics.processingMicros / 1000).toFixed(2)} ms` : '0.00 ms', 'cpu')}${metric('p95 audio age', latency, 'chart', audio?.latencyP95Micros ? 'positive' : '')}${metric('Underflows', formatNumber(audio?.underflowSamples ?? 0), 'alert', audio?.underflowSamples ? 'critical' : '')}${metric('Queue drops', formatNumber(audio?.queueOverflows ?? 0), 'database', audio?.queueOverflows ? 'critical' : '')}</div><div class="telemetry-footer"><span>${escapeHtml(audio?.streamState ?? 'Stopped')} · ${escapeHtml(inputName ?? 'default input')} → ${escapeHtml(outputName ?? 'default output')}</span><span>Worker failures: ${audio?.failures ?? current?.engine?.failures ?? 0}</span></div></article></section>
    ${report ? `<section class="error-banner"><div class="error-icon">${icon('alert', 18)}</div><div><strong>${escapeHtml(report.code)}</strong><span>${escapeHtml(report.message)}</span></div>${report.recoverable ? button('clearError', 'Dismiss', 'close', 'ghost') : ''}</section>` : ''}<footer class="app-footer"><span>Audio stays in the Rust worker; this surface receives snapshots and metrics.</span><span>${icon('keyboard', 13)} <kbd>Space</kbd> session · <kbd>Ctrl R</kbd> refresh</span></footer></main></div>`;
  wireHandlers();
}

function readConfigForm(): GuiConfig {
  const current = draftConfig ?? snapshot?.config ?? defaultConfig(); const number = (id: string, fallback: number) => Number(document.querySelector<HTMLInputElement>(`#${id}`)?.value ?? fallback); const sampleRate = number('inputRate', current.inputFormat.sampleRate); const channels = number('channels', current.inputFormat.channels);
  return { ...current, backend: (document.querySelector<HTMLSelectElement>('#backend')?.value as Backend) ?? current.backend, inputFormat: { sampleRate, channels }, outputFormat: { sampleRate, channels }, inputDevice: document.querySelector<HTMLSelectElement>('#inputDevice')?.value || null, outputDevice: document.querySelector<HTMLSelectElement>('#outputDevice')?.value || null, blockFrames: number('blockFrames', current.blockFrames), queueBlocks: number('queueBlocks', current.queueBlocks), conversion: { ...current.conversion, pitchSemitones: number('pitch', current.conversion.pitchSemitones), indexRatio: number('index', current.conversion.indexRatio), protect: number('protect', current.conversion.protect), rmsMixRate: number('rms', current.conversion.rmsMixRate) } };
}

async function runAction(action: () => Promise<unknown>, key = 'command'): Promise<void> {
  const now = performance.now();
  if (busy || now - (lastActionAt.get(key) ?? -Infinity) < MIN_ACTION_GAP_MS) return;
  lastActionAt.set(key, now); busy = true; localError = null; render();
  try { const result = await action(); if (isSnapshot(result)) { snapshot = result; draftConfig = null; } }
  catch (error) { localError = normalizeError(error); }
  finally { busy = false; render(); }
}
function isSnapshot(value: unknown): value is Snapshot { return typeof value === 'object' && value !== null && 'lifecycle' in value && 'configRevision' in value; }
function normalizeError(error: unknown): ErrorReport { if (typeof error === 'object' && error !== null && 'kind' in error) { const typed = error as { kind: string; details?: unknown }; return { code: typed.kind, message: typeof typed.details === 'string' ? typed.details : JSON.stringify(typed.details ?? typed.kind), recoverable: true }; } return { code: 'command_error', message: String(error), recoverable: true }; }

function wireHandlers(): void {
  const byId = <T extends HTMLElement>(id: string) => document.querySelector<T>(`#${id}`);
  byId<HTMLButtonElement>('load')?.addEventListener('click', () => runAction(async () => { const path = byId<HTMLInputElement>('modelPath')?.value.trim(); if (!path) throw new Error('Model path is required'); return call<Snapshot>('load_model', { path }); }, 'load-model'));
  byId<HTMLInputElement>('modelPath')?.addEventListener('keydown', (event) => { if (event.key === 'Enter') byId<HTMLButtonElement>('load')?.click(); });
  byId<HTMLButtonElement>('unload')?.addEventListener('click', () => runAction(() => call<Snapshot>('unload_model'), 'unload-model')); byId<HTMLButtonElement>('start')?.addEventListener('click', () => runAction(() => call<Snapshot>('start_session'), 'start-session')); byId<HTMLButtonElement>('stop')?.addEventListener('click', () => runAction(() => call<Snapshot>('stop_session'), 'stop-session')); byId<HTMLButtonElement>('clearError')?.addEventListener('click', () => runAction(() => call<Snapshot>('clear_error'), 'clear-error'));
  byId<HTMLButtonElement>('refreshDevices')?.addEventListener('click', () => runAction(async () => { devices = await call<AudioDeviceCatalog>('list_audio_devices'); return devices; }, 'refresh-devices')); byId<HTMLButtonElement>('save')?.addEventListener('click', () => runAction(() => call<Snapshot>('update_config', { config: readConfigForm(), expectedRevision: snapshot?.configRevision ?? 0 }), 'save-config'));
  byId<HTMLButtonElement>('demo')?.addEventListener('click', () => runAction(async () => { const rate = snapshot?.config.inputFormat.sampleRate ?? 48000; const samples = Array.from({ length: snapshot?.config.blockFrames ?? 480 }, (_, index) => Math.sin(index * 2 * Math.PI * 440 / rate) * 0.05); return call<Snapshot>('process_block', { samples, sampleRate: rate, channels: 1, sequence: Date.now() }); }, 'test-block'));
  const sliderFields: ReadonlyArray<readonly [string, string, number]> = [['pitch', 'pitchValue', 1], ['index', 'indexValue', 2], ['protect', 'protectValue', 2], ['rms', 'rmsValue', 2]]; for (const [id, outputId, digits] of sliderFields) byId<HTMLInputElement>(id)?.addEventListener('input', (event) => { const value = Number((event.target as HTMLInputElement).value); const output = byId<HTMLOutputElement>(outputId); if (output) output.value = value.toFixed(digits) + (id === 'pitch' ? ' st' : ''); draftConfig = readConfigForm(); });
  ['backend', 'inputRate', 'channels', 'blockFrames', 'queueBlocks', 'inputDevice', 'outputDevice'].forEach((id) => byId<HTMLInputElement | HTMLSelectElement>(id)?.addEventListener('change', () => { draftConfig = readConfigForm(); }));
  document.querySelectorAll<HTMLButtonElement>('[data-section]').forEach((item) => item.addEventListener('click', () => { activeSection = item.dataset.section ?? 'studio'; render(); requestAnimationFrame(() => document.querySelector(`[data-section-target="${activeSection}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })); }));
  byId<HTMLButtonElement>('themeButton')?.addEventListener('click', () => { themePreference = themePreference === 'system' ? 'dark' : themePreference === 'dark' ? 'light' : 'system'; applyTheme(); void setNativeTheme(themePreference === 'system' ? null : themePreference).catch(() => undefined); render(); }); byId<HTMLButtonElement>('minimizeWindow')?.addEventListener('click', () => void windowCommand('minimize')); byId<HTMLButtonElement>('maximizeWindow')?.addEventListener('click', () => void windowCommand('maximize')); byId<HTMLButtonElement>('closeWindow')?.addEventListener('click', () => void windowCommand('close'));
}

async function windowCommand(command: 'minimize' | 'maximize' | 'close'): Promise<void> { try { const current = getCurrentWindow(); if (command === 'minimize') await current.minimize(); else if (command === 'maximize') { await current.toggleMaximize(); await syncWindowState(); } else await current.close(); } catch (error) { localError = normalizeError(error); render(); } }
async function syncWindowState(): Promise<void> {
  try {
    const current = getCurrentWindow();
    windowMaximized = await current.isMaximized();
    if (!closeHookInstalled) {
      closeHookInstalled = true;
      await current.onCloseRequested(async (event) => {
        if (snapshot?.lifecycle !== 'running' || closeInFlight || busy) return;
        event.preventDefault();
        closeInFlight = true;
        try {
          await runAction(() => call<Snapshot>('stop_session'), 'close-stop');
          if (snapshot?.lifecycle !== 'running') await current.close();
        } finally { closeInFlight = false; }
      });
    }
    render();
  } catch { /* Browser preview has no native window. */ }
}
function applyTheme(): void { const dark = themePreference === 'dark' || (themePreference === 'system' && (systemTheme?.matches ?? true)); document.documentElement.dataset.theme = dark ? 'dark' : 'light'; document.documentElement.style.colorScheme = dark ? 'dark' : 'light'; }

async function connect(): Promise<void> {
  if (!runningInTauri()) {
    // Keep the Vite/Playwright preview useful without a native IPC bridge.
    render();
    return;
  }
  try {
    const events = new Channel<BackendEvent>();
    events.onmessage = (event) => { if (event.sequence <= lastEventSequence) return; if (lastEventSequence !== 0 && event.sequence > lastEventSequence + 1) void resyncSnapshot(); lastEventSequence = event.sequence; if (event.type === 'snapshot') { snapshot = event.payload as Snapshot; if (!snapshot.lastError) localError = null; } if (event.type === 'metrics' && snapshot) snapshot.metrics = event.payload as Snapshot['metrics']; if (event.type === 'error') localError = event.payload as ErrorReport; scheduleRender(); };
    snapshot = await call<Snapshot>('subscribe', { channel: events }); channelConnected = true; devices = await call<AudioDeviceCatalog>('list_audio_devices');
  } catch (error) { localError = normalizeError(error); }
  render();
}
async function resyncSnapshot(): Promise<void> { if (resyncInFlight) return; resyncInFlight = true; try { snapshot = await call<Snapshot>('get_snapshot'); render(); } catch { /* Periodic refresh will retry. */ } finally { resyncInFlight = false; } }

window.addEventListener('keydown', (event) => { const target = event.target as HTMLElement | null; const editing = target?.matches('input, select, textarea'); if (event.key === 'Escape' && localError) { localError = null; render(); return; } if (editing) return; if (event.code === 'Space') { event.preventDefault(); void runAction(() => call<Snapshot>(snapshot?.lifecycle === 'running' ? 'stop_session' : 'start_session'), 'space-session'); } if (event.ctrlKey && event.key.toLowerCase() === 'r') { event.preventDefault(); void resyncSnapshot(); } });
systemTheme?.addEventListener('change', () => { if (themePreference === 'system') { applyTheme(); render(); } });
window.setInterval(async () => { if (!channelConnected || busy || refreshInFlight) return; refreshInFlight = true; try { snapshot = await call<Snapshot>('get_snapshot'); render(); } catch { /* Keep the last usable snapshot. */ } finally { refreshInFlight = false; } }, 1000);
applyTheme(); render(); void syncWindowState(); void connect();
