import { Channel, invoke } from '@tauri-apps/api/core';
import { setTheme as setNativeTheme } from '@tauri-apps/api/app';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { WebviewWindow } from '@tauri-apps/api/webviewWindow';
import { emit, listen } from '@tauri-apps/api/event';
import './style.css';
import './effects.css';
import { icon, type IconName } from './icons';
import { formatNumber, lifecycleLabel, statusTone, themeLabel } from './format';
import {
  defaultConfig,
  type AudioDeviceCatalog,
  type Backend,
  type BackendEvent,
  type ErrorReport,
  type GuiConfig,
  type Lifecycle,
  type Snapshot,
  type ThemePreference,
  type WorkspaceRoute,
  isWorkspaceRoute,
} from './types';
import { navigate as commitRoute, routeCopy, routeFromLocation, type ViewRoute } from './router';

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
const THEME_EVENT = 'vocal-fast://theme-changed';
const THEME_STORAGE_KEY = 'vocal-fast.theme';
let themePreference: ThemePreference = readStoredTheme();
let windowMaximized = false;
let activeSection: ViewRoute = routeFromLocation();
let windowLabel = 'main';
let errorDialogOpen = false;
let splashStartedAt = performance.now();
let draftConfig: GuiConfig | null = null;
let renderTimer: number | undefined;
const lastActionAt = new Map<string, number>();
const MIN_ACTION_GAP_MS = 240;
const systemTheme = window.matchMedia?.('(prefers-color-scheme: dark)');
const call = <T>(command: string, args?: Record<string, unknown>) => invoke<T>(command, args);
function runningInTauri(): boolean {
  return '__TAURI_INTERNALS__' in window;
}

function isNativeInspectorShortcut(event: KeyboardEvent): boolean {
  const modifier = event.ctrlKey || event.metaKey;
  const key = event.key.toLowerCase();
  return (
    event.key === 'F12' ||
    event.key === 'ContextMenu' ||
    (event.shiftKey && event.key === 'F10') ||
    (modifier && event.shiftKey && (key === 'i' || key === 'j' || key === 'c')) ||
    (modifier && key === 'u')
  );
}

/** Keep Tauri windows feeling like a native shell while leaving Vite preview usable. */
function installNativeInteractionGuards(): void {
  if (!runningInTauri()) return;
  document.addEventListener(
    'contextmenu',
    (event) => {
      event.preventDefault();
      event.stopPropagation();
    },
    { capture: true },
  );
  document.addEventListener(
    'auxclick',
    (event) => {
      if (event.button === 2) {
        event.preventDefault();
        event.stopPropagation();
      }
    },
    { capture: true },
  );
  window.addEventListener(
    'keydown',
    (event) => {
      if (!isNativeInspectorShortcut(event)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
    },
    { capture: true },
  );
}

installNativeInteractionGuards();

function readStoredTheme(): ThemePreference {
  try {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    return value === 'dark' || value === 'light' || value === 'system' ? value : 'system';
  } catch {
    return 'system';
  }
}

function setThemePreference(next: ThemePreference, broadcast = true): void {
  themePreference = next;
  try {
    localStorage.setItem(THEME_STORAGE_KEY, next);
  } catch {
    // The native theme still applies when browser storage is unavailable.
  }
  applyTheme();
  void setNativeTheme(next === 'system' ? null : next).catch(() => undefined);
  if (broadcast && runningInTauri()) void emit<ThemePreference>(THEME_EVENT, next).catch(() => undefined);
  render();
}

if (runningInTauri()) {
  try {
    windowLabel = getCurrentWindow().label;
  } catch {
    // Tauri may still be initializing during a hot reload; syncWindowState retries below.
  }
}

function isSplashWindow(): boolean {
  return windowLabel === 'splash' || window.location.hash === '#/splash';
}

function navigateTo(route: ViewRoute): void {
  activeSection = route;
  commitRoute(route, render);
}

type AuxiliaryWindow = 'settings' | 'diagnostics';

async function openAuxiliaryWindow(kind: AuxiliaryWindow): Promise<void> {
  const label = `vocal-${kind}`;
  if (!runningInTauri()) {
    window.open(`/#/${kind}`, `${label}-preview`, 'noopener,noreferrer');
    return;
  }
  try {
    const existing = await WebviewWindow.getByLabel(label);
    if (existing) {
      await existing.show();
      await existing.setFocus();
      return;
    }
    const child = new WebviewWindow(label, {
      url: `/#/${kind}`,
      title: kind === 'settings' ? 'vocal-fast · Settings' : 'vocal-fast · Diagnostics',
      width: kind === 'settings' ? 760 : 980,
      height: kind === 'settings' ? 680 : 720,
      minWidth: kind === 'settings' ? 560 : 760,
      minHeight: 520,
      resizable: true,
      decorations: false,
      shadow: true,
      visible: false,
      devtools: true,
      zoomHotkeysEnabled: false,
      noRedirectionBitmap: true,
      parent: 'main',
      focus: true,
      center: true,
    });
    await child.once('tauri://created', async () => {
      await child.show();
      await child.setFocus();
    });
    await child.once('tauri://error', (event) => {
      localError = {
        code: 'window_error',
        message: String(event.payload ?? 'Unable to open window'),
        recoverable: true,
      };
      render();
    });
  } catch (error) {
    localError = normalizeError(error);
    render();
  }
}

function openInspector(): void {
  void runAction(
    async () => {
      if (!runningInTauri()) {
        throw new Error('The native WebView inspector is available in the desktop app.');
      }
      await call('open_devtools');
    },
    'open-devtools',
  );
}

function escapeHtml(value: string): string {
  return value.replace(
    /[&<>"']/g,
    (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character] ?? character,
  );
}

function deviceOptions(kind: 'input' | 'output', selected: string | null): string {
  const available = devices?.devices.filter((device) => (kind === 'input' ? device.input : device.output)) ?? [];
  const options = [`<option value="" ${selected ? '' : 'selected'}>System default</option>`];
  for (const device of available) {
    const suffix = device.sampleRate ? ` · ${device.sampleRate} Hz` : '';
    const selectedAttr = selected === device.id ? 'selected' : '';
    const isDefault = kind === 'input' ? device.defaultInput : device.defaultOutput;
    options.push(
      `<option value="${escapeHtml(device.id)}" ${selectedAttr}>${escapeHtml(device.name)}${suffix}${isDefault ? ' · default' : ''}</option>`,
    );
  }
  return options.join('');
}

function button(
  id: string,
  label: string,
  iconName?: IconName,
  kind = 'secondary',
  disabled = false,
  title?: string,
): string {
  return `<button id="${id}" class="button ${kind}" ${disabled ? 'disabled' : ''} ${title ? `title="${escapeHtml(title)}"` : ''}>${iconName ? icon(iconName, 16) : ''}<span>${label}</span></button>`;
}
function panelHeader(kicker: string, title: string, action = ''): string {
  return `<div class="panel-header"><div><div class="kicker">${kicker}</div><h2>${title}</h2></div>${action}</div>`;
}
function metric(label: string, value: string, iconName: IconName, tone = ''): string {
  return `<div class="metric ${tone}"><div class="metric-icon">${icon(iconName, 16)}</div><div><dt>${label}</dt><dd>${value}</dd></div></div>`;
}
function slider(
  id: string,
  label: string,
  value: string,
  number: number,
  min: number,
  max: number,
  step: number,
): string {
  return `<label class="slider-field"><span><span>${label}</span><output id="${id}Value">${value}</output></span><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${number}"/></label>`;
}
function scheduleRender(delay = 70): void {
  if (renderTimer !== undefined) return;
  renderTimer = window.setTimeout(() => {
    renderTimer = undefined;
    render();
  }, delay);
}

function renderSplash(): string {
  const elapsed = Math.max(0, performance.now() - splashStartedAt);
  const seconds = Math.floor(elapsed / 1000);
  return `<main class="splash-shell" aria-busy="true"><div class="splash-glow"></div><div class="splash-card"><div class="splash-mark">${icon('waveform', 27)}</div><div class="splash-kicker">VOCAL-FAST</div><h1>Preparing your studio</h1><p>Restoring the audio path and the last session.</p><div class="splash-progress"><span></span></div><div class="splash-status"><span class="splash-dot"></span><span>Starting runtime${seconds > 2 ? ' · still working' : '…'}</span></div></div></main>`;
}

function renderErrorDialog(report: ErrorReport): string {
  return `<div class="error-dialog-backdrop" role="presentation"><section class="error-dialog" role="alertdialog" aria-modal="true" aria-labelledby="error-dialog-title"><div class="error-dialog-icon">${icon('alert', 20)}</div><div class="error-dialog-body"><div class="kicker">RUNTIME NOTICE</div><h2 id="error-dialog-title">${escapeHtml(report.code)}</h2><p>${escapeHtml(report.message)}</p><details><summary>Technical details</summary><code>${escapeHtml(JSON.stringify(report, null, 2))}</code></details><div class="error-dialog-actions"><button id="copyError" class="button secondary">${icon('keyboard', 15)}<span>Copy details</span></button>${report.recoverable ? button('dismissError', 'Dismiss', 'close', 'ghost') : ''}</div></div></section></div>`;
}

function renderTitlebar(): string {
  return `<header class="titlebar" data-tauri-drag-region><div class="titlebar-brand"><div class="app-glyph">${icon('waveform', 17)}</div><span>vocal-fast</span><span class="titlebar-divider"></span><span class="titlebar-context">${activeSection === 'settings' ? 'Settings' : activeSection === 'diagnostics' ? 'Diagnostics' : 'Studio'}</span></div><div class="titlebar-tools"><button id="openDiagnostics" class="titlebar-tool" title="Open diagnostics window">${icon('chart', 15)}<span>Diagnostics</span></button><button id="openSettings" class="titlebar-tool" title="Open settings window">${icon('settings', 15)}<span>Settings</span></button><button id="openInspect" class="titlebar-tool" title="Open WebView inspector">${icon('inspect', 15)}<span>Inspect</span></button></div><div class="titlebar-actions"><button id="minimizeWindow" class="window-button" title="Minimize">${icon('minimize', 15)}</button><button id="maximizeWindow" class="window-button" title="Maximize">${icon(windowMaximized ? 'restore' : 'maximize', 14)}</button><button id="closeWindow" class="window-button close" title="Close">${icon('close', 15)}</button></div></header>`;
}

function renderTabs(): string {
  const tabs: ReadonlyArray<readonly [WorkspaceRoute, string, IconName]> = [
    ['studio', 'Studio', 'waveform'],
    ['audio', 'Audio routing', 'speaker'],
    ['tuning', 'Voice controls', 'sliders'],
    ['diagnostics', 'Diagnostics', 'chart'],
  ];
  return `<div class="workspace-tabs" role="tablist" aria-label="Workspace views">${tabs.map(([route, label, iconName]) => `<button id="${route}-tab" class="workspace-tab ${activeSection === route ? 'active' : ''}" data-route="${route}" role="tab" aria-selected="${activeSection === route}" aria-controls="${route}-view">${icon(iconName, 15)}<span>${label}</span></button>`).join('')}</div>`;
}

function renderSettingsPanel(config: GuiConfig): string {
  return `<section class="settings-panel panel ${activeSection === 'settings' ? 'is-active' : ''}" aria-labelledby="settings-title"><div class="settings-heading"><div class="kicker">APPLICATION SETTINGS</div><h2 id="settings-title">Personalize your workspace</h2><p>These controls apply to the current desktop session. Audio state remains shared with the Studio window.</p></div><div class="settings-grid"><article class="settings-card"><div class="settings-card-icon">${icon(themePreference === 'dark' ? 'moon' : themePreference === 'light' ? 'sun' : 'desktop', 18)}</div><div><h3>Appearance</h3><p>${themeLabel(themePreference)} with Fluent surfaces and motion-aware effects.</p></div><button id="settingsTheme" class="button secondary">Change theme</button></article><article class="settings-card"><div class="settings-card-icon">${icon('waveform', 18)}</div><div><h3>Audio defaults</h3><p>${config.inputFormat.sampleRate.toLocaleString()} Hz · ${config.inputFormat.channels} channel · ${config.blockFrames} frame blocks</p></div><button id="settingsStudio" class="button secondary">Back to Studio</button></article><article class="settings-card"><div class="settings-card-icon">${icon('keyboard', 18)}</div><div><h3>Keyboard</h3><p><kbd>Space</kbd> starts or stops a session. <kbd>Ctrl R</kbd> refreshes the snapshot.</p></div></article></div></section>`;
}

function renderSidebar(): string {
  const navItem = (id: string, label: string, iconName: IconName) =>
    `<button class="nav-item ${activeSection === id ? 'active' : ''}" data-section="${id}" aria-label="${label}" aria-current="${activeSection === id ? 'page' : 'false'}" title="${label}">${icon(iconName, 17)}<span>${label}</span></button>`;
  return `<aside class="sidebar"><div class="sidebar-heading"><div class="brand-mark">V</div><div><strong>Vocal Studio</strong><span>Realtime workspace</span></div></div><div class="nav-label">Workspace</div><nav class="nav-list" aria-label="Workspace navigation">${navItem('studio', 'Studio', 'waveform')}${navItem('audio', 'Audio routing', 'speaker')}${navItem('tuning', 'Voice controls', 'sliders')}${navItem('diagnostics', 'Diagnostics', 'chart')}</nav><div class="sidebar-spacer"></div><div class="shortcut-hint">${icon('keyboard', 15)}<span><kbd>Space</kbd> Start or stop</span></div><button id="themeButton" class="theme-button" title="Change appearance">${icon(themePreference === 'dark' ? 'moon' : themePreference === 'light' ? 'sun' : 'desktop', 16)}<span>${themeLabel(themePreference)}</span><span class="chevron">${icon('chevron', 14)}</span></button><div class="sidebar-footer"><span class="connection-dot ${channelConnected ? 'online' : ''}"></span>${channelConnected ? 'Control channel connected' : 'Connecting to runtime…'}</div></aside>`;
}

function render(): void {
  if (isSplashWindow()) {
    document.documentElement.classList.add('splash-window');
    appRoot.innerHTML = renderSplash();
    return;
  }
  document.documentElement.classList.remove('splash-window');
  const current = snapshot;
  const config = draftConfig ?? current?.config ?? defaultConfig();
  const audio = current?.audio;
  const report = localError ?? current?.lastError;
  const running = current?.lifecycle === 'running';
  const transitional = current?.lifecycle === 'loading' || current?.lifecycle === 'stopping';
  const model = current?.model;
  const latency = audio?.latencyP95Micros ? `${(audio.latencyP95Micros / 1000).toFixed(2)} ms` : '—';
  const status = current?.lifecycle ?? 'created';
  const boot = current?.boot;
  const bootPending = boot && boot.phase !== 'ready' && boot.phase !== 'failed';
  const bootMessage = boot?.message ?? (bootPending ? 'Preparing runtime…' : null);
  const inputName = audio?.active
    ? audio.inputDevice
    : config.inputDevice
      ? devices?.devices.find((item) => item.id === config.inputDevice)?.name
      : null;
  const outputName = audio?.active
    ? audio.outputDevice
    : config.outputDevice
      ? devices?.devices.find((item) => item.id === config.outputDevice)?.name
      : null;
  appRoot.innerHTML = `${renderTitlebar()}<div class="app-layout">${renderSidebar()}<main class="content"><div class="page-heading"><div><div class="kicker">REALTIME VOICE WORKSPACE</div><h1>Make every take sound like you.</h1><p>Configure your voice chain, start the stream, and keep an eye on the audio path.</p></div><div class="headline-status ${statusTone(status)}"><span class="status-dot"></span><div><strong>${lifecycleLabel(status)}</strong><small>${audio?.active ? 'Audio stream online' : channelConnected ? 'Runtime ready for control' : 'Waiting for runtime'}</small></div></div></div>
    <section class="hero-panel ${running ? 'is-running' : ''}"><div class="hero-glow"></div><div class="hero-grid"></div><div class="hero-content"><div class="eyebrow-row"><span class="eyebrow-chip">${running ? 'LIVE SESSION' : 'SESSION CONTROL'}</span><span class="revision">Revision ${current?.configRevision ?? 0}</span></div><h2>${model ? escapeHtml(model.displayName) : 'No model loaded'}</h2><p>${model ? `${escapeHtml(model.architecture)} · ${model.sampleRate.toLocaleString()} Hz · ${model.featureDims} feature dimensions${model.usesF0 ? ' · F0 enabled' : ''}` : 'Load a .vfr cache or model manifest to prepare your voice chain.'}</p><div class="hero-wave" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div><div class="hero-actions">${button('start', running ? 'Streaming' : 'Start stream', running ? 'check' : 'play', 'primary', busy || transitional || !model || running)}${button('stop', 'Stop', 'stop', 'secondary', busy || !running)}${button('demo', 'Test block', 'sparkle', 'ghost', busy || !running || !!audio?.active, 'Run one synthetic block through the engine')}</div></div><div class="hero-orb"><div class="orb-ring ring-one"></div><div class="orb-ring ring-two"></div><div class="orb-core">${icon('waveform', 42)}</div><span class="orb-caption">${running ? 'Listening' : 'Standby'}</span></div></section>
    <section class="workspace-grid"><article class="panel model-panel span-2" data-section-target="studio">${panelHeader('MODEL LIBRARY', 'Voice model', `<span class="panel-badge">${model ? icon('check', 13) + ' Loaded' : icon('database', 13) + ' Awaiting model'}</span>`)}<div class="model-layout"><div class="model-avatar">${model ? escapeHtml(model.displayName.slice(0, 1).toUpperCase()) : icon('database', 25)}</div><div class="model-details"><strong>${model ? escapeHtml(model.displayName) : 'Choose a model to begin'}</strong><span>${model ? `${escapeHtml(model.id)} · v${escapeHtml(model.version)}` : 'VFRM caches and RVC manifests are supported'}</span>${model ? `<div class="tag-row"><span class="tag">${escapeHtml(model.architecture)}</span><span class="tag">${model.sampleRate.toLocaleString()} Hz</span>${model.usesF0 ? '<span class="tag accent">F0 enabled</span>' : ''}</div>` : ''}</div><div class="model-actions">${button('unload', 'Unload', undefined, 'ghost', busy || transitional || !model || running)}<div class="model-path-row"><input id="modelPath" aria-label="Model path" value="${escapeHtml(config.modelPath ?? '')}" placeholder="Paste a model path…"/><button id="load" class="button primary compact" ${busy || transitional ? 'disabled' : ''} title="Load model">${icon('folder', 15)}<span>Load</span></button></div></div></div></article>
    <article class="panel audio-panel" data-section-target="audio">${panelHeader('AUDIO ROUTING', 'Devices', button('refreshDevices', 'Refresh', 'refresh', 'icon-only', false, 'Refresh device list'))}<div class="host-line"><span class="live-indicator ${audio?.active ? 'active' : ''}"></span><span>${devices ? escapeHtml(devices.host) : 'Audio host not queried'}</span><span class="host-count">${devices ? `${devices.devices.length} devices` : '—'}</span></div><label class="field-label">Capture<select id="inputDevice" ${running ? 'disabled' : ''}>${deviceOptions('input', config.inputDevice)}</select></label><label class="field-label">Render<select id="outputDevice" ${running ? 'disabled' : ''}>${deviceOptions('output', config.outputDevice)}</select></label><div class="route-summary"><span>${icon('speaker', 14)}${escapeHtml(inputName ?? 'System default')}</span><span class="route-arrow">→</span><span>${escapeHtml(outputName ?? 'System default')}</span></div></article>
    <article class="panel tuning-panel" data-section-target="tuning">${panelHeader('VOICE CONTROLS', 'Shape the sound', '<span class="panel-badge subtle">Live-safe</span>')}<div class="slider-list">${slider('pitch', 'Pitch shift', `${config.conversion.pitchSemitones.toFixed(1)} st`, config.conversion.pitchSemitones, -24, 24, 0.1)}${slider('index', 'Index ratio', config.conversion.indexRatio.toFixed(2), config.conversion.indexRatio, 0, 1, 0.01)}${slider('protect', 'Unvoiced protect', config.conversion.protect.toFixed(2), config.conversion.protect, 0, 1, 0.01)}${slider('rms', 'RMS mix', config.conversion.rmsMixRate.toFixed(2), config.conversion.rmsMixRate, 0, 1, 0.01)}</div></article>
    <article class="panel config-panel" data-section-target="studio">${panelHeader('ENGINE & STREAM', 'Session configuration', '<span class="panel-badge subtle">Optimistic sync</span>')}<label class="field-label">Backend<select id="backend" ${running ? 'disabled' : ''}><option value="passthrough" ${config.backend === 'passthrough' ? 'selected' : ''}>Passthrough</option><option value="rvcCpu" ${config.backend === 'rvcCpu' ? 'selected' : ''}>RVC CPU</option></select></label><div class="form-grid"><label class="field-label">Sample rate<input id="inputRate" type="number" min="1" value="${config.inputFormat.sampleRate}" ${running ? 'disabled' : ''}/></label><label class="field-label">Channels<input id="channels" type="number" min="1" max="8" value="${config.inputFormat.channels}" ${running ? 'disabled' : ''}/></label><label class="field-label">Block frames<input id="blockFrames" type="number" min="1" max="8192" value="${config.blockFrames}" ${running ? 'disabled' : ''}/></label><label class="field-label">Queue depth<input id="queueBlocks" type="number" min="1" max="64" value="${config.queueBlocks}" ${running ? 'disabled' : ''}/></label></div>${button('save', 'Save configuration', 'check', 'primary full', busy || running)}</article>
    <article class="panel telemetry-panel span-2" data-section-target="diagnostics">${panelHeader('RUNTIME HEALTH', 'Audio telemetry', `<span class="stream-state ${audio?.active ? 'active' : ''}"><span class="status-dot"></span>${audio?.active ? 'Streaming' : 'Idle'}</span>`)}<div class="metrics-grid">${metric('Input frames', formatNumber(current?.metrics.inputFrames ?? 0), 'waveform')}${metric('Output frames', formatNumber(current?.metrics.outputFrames ?? 0), 'speaker')}${metric('Processing', current ? `${(current.metrics.processingMicros / 1000).toFixed(2)} ms` : '0.00 ms', 'cpu')}${metric('p95 audio age', latency, 'chart', audio?.latencyP95Micros ? 'positive' : '')}${metric('Underflows', formatNumber(audio?.underflowSamples ?? 0), 'alert', audio?.underflowSamples ? 'critical' : '')}${metric('Queue drops', formatNumber(audio?.queueOverflows ?? 0), 'database', audio?.queueOverflows ? 'critical' : '')}</div><div class="telemetry-footer"><span>${escapeHtml(audio?.streamState ?? 'Stopped')} · ${escapeHtml(inputName ?? 'default input')} → ${escapeHtml(outputName ?? 'default output')}</span><span>Worker failures: ${audio?.failures ?? current?.engine?.failures ?? 0}</span></div></article></section>
    ${bootMessage ? `<section class="boot-strip ${bootPending ? 'pending' : ''}"><span class="boot-strip-dot"></span><div><strong>${bootPending ? 'Starting vocal-fast' : 'Startup notice'}</strong><span>${escapeHtml(bootMessage)}</span></div></section>` : ''}${report ? `<section class="error-banner" role="alert"><div class="error-icon">${icon('alert', 18)}</div><div><strong>${escapeHtml(report.code)}</strong><span>${escapeHtml(report.message)}</span></div><div class="error-actions"><button id="showErrorDetails" class="button ghost">Details</button>${report.recoverable ? button('clearError', 'Dismiss', 'close', 'ghost') : ''}</div></section>` : ''}<footer class="app-footer"><span>Audio stays in the Rust worker; this surface receives snapshots and metrics.</span><span>${icon('keyboard', 13)} <kbd>Space</kbd> session · <kbd>Ctrl R</kbd> refresh</span></footer></main></div>${report && errorDialogOpen ? renderErrorDialog(report) : ''}`;
  const content = appRoot.querySelector<HTMLElement>('main.content');
  if (content) {
    content.classList.toggle('settings-route', activeSection === 'settings');
    const copy = routeCopy(activeSection);
    const pageTitle = content.querySelector<HTMLElement>('.page-heading h1');
    const pageDescription = content.querySelector<HTMLElement>('.page-heading p');
    if (pageTitle) pageTitle.textContent = copy.title;
    if (pageDescription) pageDescription.textContent = copy.description;
    const heading = content.querySelector('.page-heading');
    heading?.insertAdjacentHTML('afterend', `${renderTabs()}${renderSettingsPanel(config)}`);
    const workspace = content.querySelector<HTMLElement>('.workspace-grid');
    workspace?.classList.add(`route-${activeSection}`);
    const seenTargets = new Set<string>();
    content.querySelectorAll<HTMLElement>('[data-section-target]').forEach((panel) => {
      const target = panel.dataset.sectionTarget;
      if (target && !seenTargets.has(target)) {
        panel.id = `${target}-view`;
        seenTargets.add(target);
      }
      panel.classList.add('route-panel');
      panel.setAttribute('role', 'tabpanel');
      if (target) panel.setAttribute('aria-labelledby', `${target}-tab`);
      panel.classList.toggle('is-active', target === activeSection);
    });
  }
  wireHandlers();
}

function readConfigForm(): GuiConfig {
  const current = draftConfig ?? snapshot?.config ?? defaultConfig();
  const number = (id: string, fallback: number) =>
    Number(document.querySelector<HTMLInputElement>(`#${id}`)?.value ?? fallback);
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

async function runAction(action: () => Promise<unknown>, key = 'command'): Promise<void> {
  const now = performance.now();
  if (busy || now - (lastActionAt.get(key) ?? -Infinity) < MIN_ACTION_GAP_MS) return;
  lastActionAt.set(key, now);
  busy = true;
  localError = null;
  render();
  try {
    const result = await action();
    if (isSnapshot(result)) {
      snapshot = result;
      draftConfig = null;
    }
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
    return {
      code: typed.kind,
      message: typeof typed.details === 'string' ? typed.details : JSON.stringify(typed.details ?? typed.kind),
      recoverable: true,
    };
  }
  return { code: 'command_error', message: String(error), recoverable: true };
}

function wireHandlers(): void {
  const byId = <T extends HTMLElement>(id: string) => document.querySelector<T>(`#${id}`);
  byId<HTMLButtonElement>('load')?.addEventListener('click', () =>
    runAction(async () => {
      const path = byId<HTMLInputElement>('modelPath')?.value.trim();
      if (!path) throw new Error('Model path is required');
      return call<Snapshot>('load_model', { path });
    }, 'load-model'),
  );
  byId<HTMLInputElement>('modelPath')?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') byId<HTMLButtonElement>('load')?.click();
  });
  byId<HTMLButtonElement>('unload')?.addEventListener('click', () =>
    runAction(() => call<Snapshot>('unload_model'), 'unload-model'),
  );
  byId<HTMLButtonElement>('start')?.addEventListener('click', () =>
    runAction(() => call<Snapshot>('start_session'), 'start-session'),
  );
  byId<HTMLButtonElement>('stop')?.addEventListener('click', () =>
    runAction(() => call<Snapshot>('stop_session'), 'stop-session'),
  );
  byId<HTMLButtonElement>('clearError')?.addEventListener('click', () =>
    runAction(async () => {
      if (localError) {
        localError = null;
        errorDialogOpen = false;
        render();
        return snapshot;
      }
      errorDialogOpen = false;
      return call<Snapshot>('clear_error');
    }, 'clear-error'),
  );
  byId<HTMLButtonElement>('showErrorDetails')?.addEventListener('click', () => {
    errorDialogOpen = true;
    render();
  });
  byId<HTMLButtonElement>('dismissError')?.addEventListener('click', () => {
    errorDialogOpen = false;
    if (localError) localError = null;
    render();
  });
  byId<HTMLButtonElement>('copyError')?.addEventListener('click', () => {
    const report = localError ?? snapshot?.lastError;
    if (!report) return;
    void navigator.clipboard
      ?.writeText(`[${report.code}] ${report.message}`)
      .catch(() => undefined);
  });
  byId<HTMLButtonElement>('refreshDevices')?.addEventListener('click', () =>
    runAction(async () => {
      devices = await call<AudioDeviceCatalog>('list_audio_devices');
      return devices;
    }, 'refresh-devices'),
  );
  byId<HTMLButtonElement>('save')?.addEventListener('click', () =>
    runAction(
      () =>
        call<Snapshot>('update_config', {
          config: readConfigForm(),
          expectedRevision: snapshot?.configRevision ?? 0,
        }),
      'save-config',
    ),
  );
  byId<HTMLButtonElement>('demo')?.addEventListener('click', () =>
    runAction(async () => {
      const rate = snapshot?.config.inputFormat.sampleRate ?? 48000;
      const samples = Array.from(
        { length: snapshot?.config.blockFrames ?? 480 },
        (_, index) => Math.sin((index * 2 * Math.PI * 440) / rate) * 0.05,
      );
      return call<Snapshot>('process_block', {
        samples,
        sampleRate: rate,
        channels: 1,
        sequence: Date.now(),
      });
    }, 'test-block'),
  );
  const sliderFields: ReadonlyArray<readonly [string, string, number]> = [
    ['pitch', 'pitchValue', 1],
    ['index', 'indexValue', 2],
    ['protect', 'protectValue', 2],
    ['rms', 'rmsValue', 2],
  ];
  for (const [id, outputId, digits] of sliderFields)
    byId<HTMLInputElement>(id)?.addEventListener('input', (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      const output = byId<HTMLOutputElement>(outputId);
      if (output) output.value = value.toFixed(digits) + (id === 'pitch' ? ' st' : '');
      draftConfig = readConfigForm();
    });
  ['backend', 'inputRate', 'channels', 'blockFrames', 'queueBlocks', 'inputDevice', 'outputDevice'].forEach((id) =>
    byId<HTMLInputElement | HTMLSelectElement>(id)?.addEventListener('change', () => {
      draftConfig = readConfigForm();
    }),
  );
  document.querySelectorAll<HTMLButtonElement>('[data-section]').forEach((item) =>
    item.addEventListener('click', () => {
      const route = item.dataset.section;
      if (route && isWorkspaceRoute(route)) navigateTo(route);
    }),
  );
  document.querySelectorAll<HTMLButtonElement>('[data-route]').forEach((item) =>
    item.addEventListener('click', () => {
      const route = item.dataset.route;
      if (route && isWorkspaceRoute(route)) navigateTo(route);
    }),
  );
  const cycleTheme = () => {
    setThemePreference(themePreference === 'system' ? 'dark' : themePreference === 'dark' ? 'light' : 'system');
  };
  byId<HTMLButtonElement>('themeButton')?.addEventListener('click', cycleTheme);
  byId<HTMLButtonElement>('settingsTheme')?.addEventListener('click', cycleTheme);
  byId<HTMLButtonElement>('settingsStudio')?.addEventListener('click', () => navigateTo('studio'));
  byId<HTMLButtonElement>('openDiagnostics')?.addEventListener('click', () => void openAuxiliaryWindow('diagnostics'));
  byId<HTMLButtonElement>('openSettings')?.addEventListener('click', () => void openAuxiliaryWindow('settings'));
  byId<HTMLButtonElement>('openInspect')?.addEventListener('click', openInspector);
  byId<HTMLButtonElement>('minimizeWindow')?.addEventListener('click', () => void windowCommand('minimize'));
  byId<HTMLButtonElement>('maximizeWindow')?.addEventListener('click', () => void windowCommand('maximize'));
  byId<HTMLButtonElement>('closeWindow')?.addEventListener('click', () => void windowCommand('close'));
}

async function windowCommand(command: 'minimize' | 'maximize' | 'close'): Promise<void> {
  try {
    const current = getCurrentWindow();
    if (command === 'minimize') await current.minimize();
    else if (command === 'maximize') {
      await current.toggleMaximize();
      await syncWindowState();
    } else {
      if (snapshot?.lifecycle === 'running') {
        await runAction(() => call<Snapshot>('stop_session'), 'close-stop');
      }
      await current.hide();
    }
  } catch (error) {
    localError = normalizeError(error);
    render();
  }
}
async function syncWindowState(): Promise<void> {
  try {
    const current = getCurrentWindow();
    windowLabel = current.label;
    windowMaximized = await current.isMaximized();
    render();
  } catch {
    /* Browser preview has no native window. */
  }
}
function applyTheme(): void {
  const dark = themePreference === 'dark' || (themePreference === 'system' && (systemTheme?.matches ?? true));
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  document.documentElement.style.colorScheme = dark ? 'dark' : 'light';
}

async function connect(): Promise<void> {
  if (isSplashWindow()) return;
  if (!runningInTauri()) {
    // Keep the Vite/Playwright preview useful without a native IPC bridge.
    render();
    return;
  }
  try {
    await call('harden_webview').catch((error) => {
      // A non-Windows runtime may not expose WebView2 settings. The DOM
      // guards still provide the same interaction policy in that case.
      localError = normalizeError(error);
    });
    const events = new Channel<BackendEvent>();
    events.onmessage = (event) => {
      if (event.sequence <= lastEventSequence) return;
      if (lastEventSequence !== 0 && event.sequence > lastEventSequence + 1) void resyncSnapshot();
      lastEventSequence = event.sequence;
      if (event.type === 'snapshot') {
        snapshot = event.payload as Snapshot;
        if (!snapshot.lastError) localError = null;
      }
      if (event.type === 'metrics' && snapshot) snapshot.metrics = event.payload as Snapshot['metrics'];
      if (event.type === 'error') {
        localError = event.payload as ErrorReport;
        errorDialogOpen = !localError.recoverable || localError.code === 'startup_timeout';
      }
      scheduleRender();
    };
    snapshot = await call<Snapshot>('subscribe', { windowLabel, channel: events });
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
    /* Periodic refresh will retry. */
  } finally {
    resyncInFlight = false;
  }
}

window.addEventListener('hashchange', () => {
  activeSection = routeFromLocation();
  render();
});
window.addEventListener('popstate', () => {
  activeSection = routeFromLocation();
  render();
});

window.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement | null;
  const editing = target?.matches('input, select, textarea');
  if (event.key === 'Escape' && localError) {
    localError = null;
    render();
    return;
  }
  if (editing) return;
  if (event.code === 'Space') {
    event.preventDefault();
    void runAction(
      () => call<Snapshot>(snapshot?.lifecycle === 'running' ? 'stop_session' : 'start_session'),
      'space-session',
    );
  }
  if (event.ctrlKey && event.key.toLowerCase() === 'r') {
    event.preventDefault();
    void resyncSnapshot();
  }
});
systemTheme?.addEventListener('change', () => {
  if (themePreference === 'system') {
    applyTheme();
    render();
  }
});
if (runningInTauri()) {
  void listen<ThemePreference>(THEME_EVENT, (event) => {
    if (event.payload === 'dark' || event.payload === 'light' || event.payload === 'system')
      setThemePreference(event.payload, false);
  });
  void listen<string>('vocal-fast://open-route', (event) => {
    if (event.payload === 'settings' || event.payload === 'diagnostics') {
      void openAuxiliaryWindow(event.payload);
    } else if (isWorkspaceRoute(event.payload)) {
      navigateTo(event.payload as ViewRoute);
    }
  });
}
window.setInterval(async () => {
  if (isSplashWindow() || document.visibilityState !== 'visible' || !channelConnected || busy || refreshInFlight) return;
  refreshInFlight = true;
  try {
    snapshot = await call<Snapshot>('get_snapshot');
    render();
  } catch {
    /* Keep the last usable snapshot. */
  } finally {
    refreshInFlight = false;
  }
}, 1000);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') void resyncSnapshot();
});
applyTheme();
render();
void syncWindowState();
void connect();
