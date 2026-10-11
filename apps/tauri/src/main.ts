import { Channel, invoke } from '@tauri-apps/api/core';
import { setTheme as setNativeTheme } from '@tauri-apps/api/app';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { WebviewWindow } from '@tauri-apps/api/webviewWindow';
import { emit, listen } from '@tauri-apps/api/event';
import './style.css';
import './effects.css';
import './pitch-controls.css';
import { renderPitchMatching } from './pitch-controls';
import { icon, type IconName } from './icons';
import { displayPath, escapeHtml, formatBytes, formatNumber, lifecycleLabel, samePath, statusTone, themeLabel } from './format';
import {
  defaultConfig,
  type AudioDeviceCatalog,
  type BackendEvent,
  type BootStatus,
  type ErrorReport,
  type GuiConfig,
  type Lifecycle,
  type ModelCatalog,
  type ModelArtifactInspection,
  type ModelComponents,
  type ModelDependencyStatus,
  type ProcessingBackendCatalog,
  type Snapshot,
  type StartupResources,
  type PreparationStatus,
  type ThemePreference,
  type WorkspaceRoute,
  isWorkspaceRoute,
} from './types';
import { on, patchView } from './view';
import { renderSessionProfiles } from './session-profiles';
import { renderSaveStatus } from './save-status';
import { renderAudioRoute } from './audio-routing';
import { renderDiagnostics } from './diagnostics';
import { DiagnosticsHistory } from './diagnostics-history';
import { automaticConfig, rebaseConfigDraft, reconcileConfigDraft, sameConfig } from './config-draft';
import { renderStartup } from './startup';
import { initializeWorkspace } from './workspace-startup';
import { errorTitle, normalizeError } from './error-report';
import { BootGate } from './boot-gate';
import { waitForVisible } from './window-visibility';
import { backendAvailable, backendLoadSelection, backendSelectionChanged, renderProcessingBackend as renderBackendView, resolvedBackend } from './processing-backend';
import { validateCpuConfig } from './cpu-policy';
import { validateSpeakerId } from './speaker-policy';
import { renderComponentBadges, renderComponentDetails } from './model-components';
import { observePreparation, preparationActive, renderPreparationProgress } from './preparation-progress';
import { navigate as commitRoute, routeCopy, routeFromLocation, type ViewRoute } from './router';
import './model-management.css';
import { ModelManagement } from './model-management';
import type { ModelImportStatus } from './model-library-types';

const root = document.querySelector<HTMLDivElement>('#app');
if (!root) throw new Error('missing #app');
const appRoot = root;
let snapshot: Snapshot | null = null;
const diagnosticsHistory = new DiagnosticsHistory();
let devices: AudioDeviceCatalog | null = null;
let modelCatalog: ModelCatalog | null = null;
let processingBackends: ProcessingBackendCatalog | null = null;
let channelConnected = false;
let busy = false;
let activeAction: string | null = null;
let actionStartedAt = 0;
let actionPreparationId = 0;
let preparationStatus: PreparationStatus | null = null;
let bootStatus: BootStatus | null = null;
const bootGate = new BootGate();
let startupResources: StartupResources | null = null;
let preparationObservedAt = performance.now();
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
const splashStartedAt = performance.now();
let draftConfig: GuiConfig | null = null;
let draftRevision: number | null = null;
let draftBaseConfig: GuiConfig | null = null;
let draftConflict = false;
let automaticSaveTimer: number | undefined;
let automaticSaveState: 'idle' | 'pending' | 'saving' | 'saved' | 'failed' = 'idle';
let automaticSaveError: ErrorReport | null = null;
let automaticSaveRequest: { config: GuiConfig; revision: number } | null = null;
let savedFeedbackUntil = 0;
let savedFeedbackTimer: number | undefined;
let automaticConflictRetries = 0;
const AUTO_SAVE_DELAY_MS = 350;
// Choosing a file is independent of session settings and the loaded model.
let selectedModelPath: string | undefined;
let modelInspection: (ModelArtifactInspection & { selectionPath: string }) | null = null;
let modelInspectionError: string | null = null;
let modelDependencyStatus: ModelDependencyStatus | null = null;
let dependencyHubertSource: string | null = null;
let dependencyRmvpeSource: string | null = null;
let dependencyInspectionSequence = 0;
let inspectionSequence = 0;
let inspectionTimer: number | undefined;
let renderTimer: number | undefined;
const lastActionAt = new Map<string, number>();
const MIN_ACTION_GAP_MS = 240;
const systemTheme = window.matchMedia?.('(prefers-color-scheme: dark)');
const call = <T>(command: string, args?: Record<string, unknown>) => invoke<T>(command, args);
const modelManagement = new ModelManagement({
  invoke:call,changed:()=>scheduleRender(),native:runningInTauri,
  completeBundle:()=>modelCatalog?.entries.find(entry=>entry.components.automaticTensorRt)?.path??(selectedComponents()?.automaticTensorRt?modelSelectionPath()||null:null),
  config:()=>draftConfig??snapshot?.config??defaultConfig(),
  selected:path=>{selectedModelPath=path;inspectSelectedModel();commitRoute('studio',()=>{activeSection='studio';render();});},
  completed:async directory=>{
    if(!directory)return;
    modelCatalog=await call<ModelCatalog>('list_models',{path:directory});
    inspectSelectedModel();
    if(!snapshot?.audio.active && snapshot?.lifecycle!=='running')setDraftConfig({...draftConfig??snapshot?.config??defaultConfig(),modelDirectory:directory});
    scheduleRender();
  },
});
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

function setDraftConfig(next: GuiConfig, automatic = true): void {
  const base = snapshot?.config ?? defaultConfig();
  if (sameConfig(next, base) && !automaticSaveRequest) {
    clearDraftConfig();
    return;
  }
  if (draftConfig === null) {
    draftRevision = snapshot?.configRevision ?? 0;
    draftBaseConfig = base;
  }
  draftConfig = next;
  if (automatic) {
    automaticSaveError = null;
    automaticConflictRetries = 0;
    automaticSaveState = automaticSaveRequest ? 'saving' : 'pending';
    queueAutomaticSave();
  }
  scheduleRender();
}

function clearDraftConfig(): void {
  draftConfig = null;
  draftRevision = null;
  draftBaseConfig = null;
  draftConflict = false;
  automaticSaveError = null;
  window.clearTimeout(automaticSaveTimer);
  automaticSaveTimer = undefined;
}

function configIsDirty(): boolean {
  if (!draftConfig) return false;
  return JSON.stringify(draftConfig) !== JSON.stringify(snapshot?.config ?? defaultConfig());
}

function profileIsDirty(): boolean {
  return !!draftConfig && draftConfig.tensorrt.profile !== snapshot?.config.tensorrt.profile;
}

function automaticConfigIsDirty(): boolean {
  return !!draftConfig && !!snapshot && !sameConfig(automaticConfig(draftConfig, snapshot.config), snapshot.config);
}

function configLocked(): boolean {
  return busy || pipelineBusy() || !!snapshot?.audio.active || snapshot?.lifecycle === 'running'
    || snapshot?.lifecycle === 'loading' || snapshot?.lifecycle === 'stopping';
}

function queueAutomaticSave(): void {
  window.clearTimeout(automaticSaveTimer);
  automaticSaveTimer = undefined;
  if (!automaticConfigIsDirty()) {
    if (!automaticSaveRequest && automaticSaveState !== 'idle') automaticSaveState = 'saved';
    return;
  }
  if (!runningInTauri() || automaticSaveRequest || automaticSaveState === 'failed' || draftConflict || configLocked()) return;
  automaticSaveState = 'pending';
  automaticSaveTimer = window.setTimeout(() => { void saveAutomaticConfig(); }, AUTO_SAVE_DELAY_MS);
}

async function saveAutomaticConfig(): Promise<void> {
  automaticSaveTimer = undefined;
  if (!snapshot || !draftConfig || automaticSaveRequest || draftConflict || configLocked() || !automaticConfigIsDirty()) return;
  const config = automaticConfig(draftConfig, snapshot.config);
  const validation = validateConfig(config);
  if (validation) {
    automaticSaveError = { code: 'invalid_config', message: validation, recoverable: true };
    automaticSaveState = 'failed';
    render();
    return;
  }
  automaticSaveRequest = { config, revision: snapshot.configRevision };
  automaticSaveState = 'saving';
  render();
  try {
    acceptSnapshot(await call<Snapshot>('update_config', { config, expectedRevision: automaticSaveRequest.revision }));
    automaticSaveState = 'saved';
    automaticSaveError = null;
    savedFeedbackUntil = performance.now() + 2600;
    window.clearTimeout(savedFeedbackTimer);
    savedFeedbackTimer = window.setTimeout(scheduleRender, 2600);
  } catch (error) {
    const report = normalizeError(error);
    // A disjoint edit can win the revision race before its Channel snapshot arrives.
    // Rebase once; repeated or overlapping conflicts require the user's attention.
    if (report.code === 'config_conflict' && automaticConflictRetries++ < 1) {
      try { acceptSnapshot(await call<Snapshot>('get_snapshot')); } catch { /* Preserve the last usable draft. */ }
      if (!draftConflict) automaticSaveState = 'pending';
      else { automaticSaveError = report; automaticSaveState = 'failed'; }
    } else { automaticSaveError = report; automaticSaveState = 'failed'; }
  } finally {
    automaticSaveRequest = null;
    queueAutomaticSave();
    render();
  }
}

function renderConfigPersistence(): string {
  return renderSaveStatus({
    connected: channelConnected,
    conflict: draftConflict,
    error: automaticSaveError?.message ?? null,
    pending: automaticConfigIsDirty(),
    saving: automaticSaveState === 'saving',
    profileDirty: profileIsDirty(),
    locked: configLocked(),
    recentlySaved: performance.now() < savedFeedbackUntil,
  });
}

function modelSelectionPath(): string {
  return selectedModelPath ?? snapshot?.config.modelPath ?? '';
}

function dependencyRequest(path: string, allowDownload = false): Record<string, unknown> {
  const config = draftConfig ?? snapshot?.config ?? defaultConfig();
  return {
    request: {
      modelPath: path,
      libraryDirectory: config.modelDirectory,
      cacheDirectory: null,
      allowDownload,
      hubertSource: dependencyHubertSource,
      rmvpeSource: dependencyRmvpeSource,
    },
  };
}

function selectedComponents(): ModelComponents | null {
  const path = modelSelectionPath();
  return modelCatalog?.entries.find(entry => samePath(entry.path, path))?.components
    ?? (modelInspection && (samePath(modelInspection.path, path) || samePath(modelInspection.selectionPath, path)) ? modelInspection.components : null);
}

function renderProcessingBackend(config: GuiConfig, catalog: ProcessingBackendCatalog | null, locked: boolean, prefix = ''): string {
  const components = selectedComponents();
  const pending = busy && (activeAction === 'load-model' || activeAction === 'prepare-tensorrt' || activeAction === 'save-config')
    && (!preparationStatus || preparationStatus.operationId <= actionPreparationId);
  const progress = pending ? {
    operationId: -1, revision: 0, kind: 'loadModel' as const, phase: 'inspecting' as const,
    message: activeAction === 'save-config' ? 'Sending configuration to the runtime…' : 'Starting model preparation…',
    elapsedMs: performance.now() - actionStartedAt, completedSteps: 0, totalSteps: null, cacheHit: false,
  } : preparationStatus;
  return renderComponentDetails(components, modelInspectionError)
    + renderBackendView(config, catalog, locked, prefix, components, snapshot && progress ? { ...snapshot, preparation: progress } : snapshot, modelSelectionPath())
    + (!snapshot ? renderPreparationProgress(progress, prefix) : '');
}

function inspectSelectedModel(): void {
  window.clearTimeout(inspectionTimer);
  const sequence = ++inspectionSequence;
  const dependencySequence = ++dependencyInspectionSequence;
  modelInspection = null;
  modelInspectionError = null;
  modelDependencyStatus = null;
  dependencyHubertSource = null;
  dependencyRmvpeSource = null;
  const path = modelSelectionPath().trim();
  if (!path || !runningInTauri()) return;
  inspectionTimer = window.setTimeout(() => {
    void Promise.all([
      call<ModelArtifactInspection>('inspect_model_artifact', { path }),
      call<ModelDependencyStatus>('inspect_model_dependencies', dependencyRequest(path)),
    ]).then(([result, dependencies]) => {
      if (sequence !== inspectionSequence || dependencySequence !== dependencyInspectionSequence || !samePath(modelSelectionPath(), path)) return;
      modelInspection = { ...result, selectionPath: path };
      modelDependencyStatus = dependencies;
      render();
    }).catch(error => {
      if (sequence !== inspectionSequence || dependencySequence !== dependencyInspectionSequence || !samePath(modelSelectionPath(), path)) return;
      modelInspectionError = normalizeError(error).message;
      render();
    });
  }, 300);
}

function modelSelectionLabel(): string {
  const path = modelSelectionPath().trim();
  if (!path) return 'Choose a model to load';
  if (selectedComponents()?.synth.present === false) return 'Acoustic component · choose a voice model to load';
  if (modelDependencyStatus && modelDependencyStatus.phase !== 'ready') return 'Set up HuBERT and RMVPE to continue';
  return snapshot?.model && samePath(path, snapshot.config.modelPath)
    ? snapshot.engine ? 'Loaded' : 'Model loaded · prepare plans'
    : 'Selected · click Load';
}

function syncDraftControls(): void {
  const locked = configLocked() || !!automaticSaveRequest;
  const disabled = locked || !profileIsDirty() || automaticConfigIsDirty() || draftConflict;
  for (const id of ['save', 'settingsSave']) {
    const control = document.querySelector<HTMLButtonElement>(`#${id}`);
    if (control) control.disabled = disabled;
  }
  const dirty = configIsDirty();
  const start = document.querySelector<HTMLButtonElement>('#start');
  if (start) start.disabled = locked || dirty || !snapshot?.engine || !backendAvailable({ ...snapshot.config, backendAuto: false }, processingBackends) || profileBuildRequired();
  const load = document.querySelector<HTMLButtonElement>('#load');
  if (load) load.disabled = locked || automaticConfigIsDirty() || !modelSelectionPath().trim() || selectedComponents()?.synth.present === false || !backendAvailable(draftConfig ?? snapshot?.config ?? defaultConfig(), processingBackends);
  document.querySelectorAll<HTMLButtonElement>('[data-prepare-backend]').forEach((control) => {
    const config = draftConfig ?? snapshot?.config ?? defaultConfig();
    control.disabled = locked || configIsDirty() || draftConflict || !modelSelectionPath().trim()
      || !config.tensorrt.hubertSource || !config.tensorrt.rmvpeSource
      || !backendAvailable(config, processingBackends);
  });
  const unload = document.querySelector<HTMLButtonElement>('#unload');
  if (unload) unload.disabled = locked || !snapshot?.model;
  const selectionStatus = document.querySelector<HTMLElement>('#modelSelectionStatus');
  if (selectionStatus) selectionStatus.textContent = modelSelectionLabel();
}

function profileBuildRequired(): boolean {
  return snapshot?.config.backend === 'tensorRtRtx' && snapshot.lastConfigUpdate === 'planBuildRequired';
}

function pipelineBusy(): boolean {
  return preparationActive(preparationStatus) || snapshot?.tensorrtPipeline.phase === 'building' || snapshot?.tensorrtPipeline.phase === 'checkingCache';
}

function acceptSnapshot(next: Snapshot): void {
  // A late command reply must not roll back a newer configuration from another window.
  if (snapshot && next.configRevision < snapshot.configRevision) return;
  if (draftConfig && draftBaseConfig) {
    const acknowledged = automaticSaveRequest && next.configRevision > automaticSaveRequest.revision
      && sameConfig(next.config, automaticSaveRequest.config);
    const rebased = rebaseConfigDraft(draftConfig, acknowledged ? automaticSaveRequest!.config : draftBaseConfig, next.config);
    draftConfig = rebased.config;
    draftConflict ||= rebased.conflicts.length > 0;
    draftRevision = next.configRevision;
    draftBaseConfig = next.config;
  }
  snapshot = next;
  diagnosticsHistory.observe(next.audio, performance.now());
  if (draftConfig && !configIsDirty()) clearDraftConfig();
  acceptPreparation(next.preparation);
  if (automaticSaveTimer === undefined) queueAutomaticSave();
}

function acceptPreparation(status: PreparationStatus | undefined): void {
  if (!status || (preparationStatus && (status.operationId < preparationStatus.operationId
    || status.operationId === preparationStatus.operationId && status.revision < preparationStatus.revision))) return;
  preparationStatus = status;
  preparationObservedAt = performance.now();
  observePreparation(status);
}

function validateConfig(config: GuiConfig): string | null {
  const cpuError = validateCpuConfig(config.cpu);
  if (cpuError) return cpuError;
  const speakerError = validateSpeakerId(config.speakerId, samePath(modelSelectionPath(), config.modelPath) ? selectedComponents()?.speakerCount : null);
  if (speakerError) return speakerError;
  const numbers: ReadonlyArray<readonly [string, number, number, number]> = [
    ['sample rate', config.inputFormat.sampleRate, 8_000, 192_000],
    ['channels', config.inputFormat.channels, 1, 8],
    ['block frames', config.blockFrames, 1, 8_192],
    ['queue depth', config.queueBlocks, 1, 64],
    ['pitch shift', config.conversion.pitchSemitones, -24, 24],
    ['automatic pitch target', config.conversion.autoPitch.targetHz, 50, 1200],
    ['index ratio', config.conversion.indexRatio, 0, 1],
    ['unvoiced protect', config.conversion.protect, 0, 1],
    ['RMS mix', config.conversion.rmsMixRate, 0, 1],
  ];
  for (const [label, value, minimum, maximum] of numbers) {
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      return `${label} must be between ${minimum} and ${maximum}.`;
    }
  }
  if (!Number.isInteger(config.inputFormat.sampleRate)
    || !Number.isInteger(config.inputFormat.channels)
    || !Number.isInteger(config.blockFrames)
    || !Number.isInteger(config.queueBlocks)) {
    return 'Sample rate, channels, block frames, and queue depth must be whole numbers.';
  }
  return null;
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

function deviceOptions(kind: 'input' | 'output', selected: string | null): string {
  const available = devices?.devices.filter((device) => (kind === 'input' ? device.input : device.output)) ?? [];
  const options = [`<option value="" ${selected ? '' : 'selected'}>System default</option>`];
  if (selected && !available.some((device) => device.id === selected)) {
    options.push(`<option value="${escapeHtml(selected)}" selected disabled>Selected device unavailable</option>`);
  }
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
  const work: Record<string, [string, string]> = {
    load: ['load-model', 'Loading…'], unload: ['unload-model', 'Unloading…'],
    scanModels: ['scan-models', 'Scanning…'], settingsScanModels: ['settings-scan-models', 'Scanning…'],
    browseModels: ['browse-models', 'Opening…'], settingsBrowseModels: ['settings-browse-models', 'Opening…'],
    save: ['save-config', 'Saving…'], settingsSave: ['save-config', 'Saving…'],
    refreshDevices: ['refresh-devices', 'Detecting…'], settingsRefreshDevices: ['refresh-devices', 'Detecting…'],
    start: ['start-session', 'Starting…'], stop: ['stop-session', 'Stopping…'],
  };
  const loading = !!work[id] && (activeAction === work[id][0] || id === 'load' && preparationActive(preparationStatus) && preparationStatus?.kind === 'loadModel');
  return `<button id="${id}" class="button ${kind}" ${disabled ? 'disabled' : ''} ${loading ? 'aria-busy="true"' : ''} ${title ? `title="${escapeHtml(title)}"` : ''}>${loading ? '<span class="button-spinner" aria-hidden="true"></span>' : iconName ? icon(iconName, 16) : ''}<span>${loading ? work[id][1] : label}</span></button>`;
}
function panelHeader(kicker: string, title: string, action = ''): string {
  return `<div class="panel-header"><div><div class="kicker">${kicker}</div><h2>${title}</h2></div>${action}</div>`;
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
  return `<label class="slider-field"><span><span>${label}</span><output id="${id}Value">${value}</output></span><input id="${id}" ${configLocked() ? 'disabled' : ''} type="range" min="${min}" max="${max}" step="${step}" value="${number}"/></label>`;
}
function scheduleRender(delay = 70): void {
  if (renderTimer !== undefined) return;
  renderTimer = window.setTimeout(() => {
    renderTimer = undefined;
    render();
  }, delay);
}

function renderSplash(): string {
  return renderStartup(bootStatus, performance.now() - splashStartedAt);
}

function renderWorkspacePreparation(): string {
  if (!preparationActive(preparationStatus) && (snapshot?.modelRestorePending || (activeAction === 'restore-model' && (preparationStatus?.operationId ?? 0) <= actionPreparationId))) {
    return renderPreparationProgress({
      operationId: -1, revision: 0, kind: 'loadModel', phase: 'inspecting',
      message: 'Preparing your saved voice model in this workspace…',
      elapsedMs: 0, completedSteps: 0, totalSteps: null, cacheHit: false,
    }, 'workspace-');
  }
  return renderPreparationProgress(preparationStatus, 'workspace-');
}

function renderErrorDialog(report: ErrorReport): string {
  const context = report.details;
  const recovery = context ? `<div class="audio-error-recovery">${context.deviceName ? `<strong>${escapeHtml(context.deviceName)}</strong>` : ''}${context.hresult ? `<span class="tag">${escapeHtml(context.hresult)}</span>` : ''}<ul>${context.suggestions.map(suggestion => `<li>${escapeHtml(suggestion)}</li>`).join('')}</ul></div>` : '';
  return `<div class="error-dialog-backdrop" role="presentation"><section class="error-dialog" role="alertdialog" aria-modal="true" aria-labelledby="error-dialog-title"><div class="error-dialog-icon">${icon('alert', 20)}</div><div class="error-dialog-body"><div class="kicker">RUNTIME NOTICE</div><h2 id="error-dialog-title">${escapeHtml(errorTitle(report))}</h2><p>${escapeHtml(report.message)}</p>${recovery}<details><summary>Technical details</summary><code>${escapeHtml(JSON.stringify(report, null, 2))}</code></details><div class="error-dialog-actions">${context ? button('errorAudioRouting', 'Audio routing', 'speaker', 'primary') : ''}<button id="copyError" class="button secondary">${icon('keyboard', 15)}<span>Copy details</span></button>${report.recoverable ? button('dismissError', 'Dismiss', 'close', 'ghost') : ''}</div></div></section></div>`;
}

function renderTitlebar(): string {
  return `<header class="titlebar" data-tauri-drag-region><div class="titlebar-brand"><div class="app-glyph">${icon('waveform', 17)}</div><span>vocal-fast</span><span class="titlebar-divider"></span><span class="titlebar-context">${activeSection === 'settings' ? 'Settings' : activeSection === 'diagnostics' ? 'Diagnostics' : 'Studio'}</span></div><div class="titlebar-tools"><button id="openInspect" class="titlebar-tool" title="Open WebView inspector">${icon('inspect', 15)}<span>Inspect</span></button></div><div class="titlebar-actions"><button id="minimizeWindow" class="window-button" title="Minimize">${icon('minimize', 15)}</button><button id="maximizeWindow" class="window-button" title="Maximize">${icon(windowMaximized ? 'restore' : 'maximize', 14)}</button><button id="closeWindow" class="window-button close" title="Close">${icon('close', 15)}</button></div></header>`;
}

function renderTabs(): string {
  const tabs: ReadonlyArray<readonly [WorkspaceRoute, string, IconName]> = [
    ['studio', 'Studio', 'waveform'],
    ['models', 'Models', 'database'],
    ['audio', 'Audio routing', 'speaker'],
    ['tuning', 'Voice controls', 'sliders'],
    ['diagnostics', 'Diagnostics', 'chart'],
  ];
  return `<div class="workspace-tabs" role="tablist" aria-label="Workspace views">${tabs.map(([route, label, iconName]) => `<button id="${route}-tab" class="workspace-tab ${activeSection === route ? 'active' : ''}" data-route="${route}" role="tab" aria-selected="${activeSection === route}" aria-controls="${route}-view">${icon(iconName, 15)}<span>${label}</span></button>`).join('')}</div>`;
}

function renderSettingsPanel(config: GuiConfig): string {
  const running = snapshot?.lifecycle === 'running';
  const locked = busy || pipelineBusy() || running || !!snapshot?.audio.active || snapshot?.lifecycle === 'loading' || snapshot?.lifecycle === 'stopping';
  const dirty = profileIsDirty();
  const inputDeviceCount = devices?.devices.filter((device) => device.input).length ?? 0;
  const outputDeviceCount = devices?.devices.filter((device) => device.output).length ?? 0;
  const rates = [...new Set([16_000, 24_000, 44_100, 48_000, 96_000, config.inputFormat.sampleRate])].sort((a, b) => a - b);
  const rateOptions = rates
    .map((rate) => `<option value="${rate}" ${config.inputFormat.sampleRate === rate ? 'selected' : ''}>${rate.toLocaleString()} Hz</option>`)
    .join('');
  const blockFrames = [...new Set([240, 480, 960, 1920, config.blockFrames])].sort((a, b) => a - b);
  const blockFrameOptions = blockFrames
    .map((frames) => `<option value="${frames}" ${config.blockFrames === frames ? 'selected' : ''}>${frames} · ${(frames / config.inputFormat.sampleRate * 1000).toFixed(1)} ms</option>`)
    .join('');
  const themeOptions = (['system', 'light', 'dark'] as const)
    .map((theme) => `<button type="button" class="settings-choice ${themePreference === theme ? 'active' : ''}" data-theme-choice="${theme}" aria-pressed="${themePreference === theme}">${icon(theme === 'system' ? 'desktop' : theme === 'light' ? 'sun' : 'moon', 15)}<span>${themeLabel(theme)}</span></button>`)
    .join('');
  const conflict = draftConflict
    ? `<div class="settings-conflict" role="alert">${icon('alert', 15)}<span>This configuration changed in another window. Reload it before saving your draft.</span><button id="settingsReload" class="button ghost compact">Reload</button></div>`
    : '';
  return `<section class="settings-panel panel ${activeSection === 'settings' ? 'is-active' : ''}" aria-labelledby="settings-title">
    <div class="settings-heading"><div class="kicker">APPLICATION SETTINGS</div><h2 id="settings-title">Control the session from one place.</h2><p>Audio, model library, and realtime budget share the same revisioned configuration as Studio.</p></div>
    <div class="settings-toolbar"><div class="settings-toolbar-caption"><strong>Session profile</strong><span>Save to apply your selected preset.</span></div><div class="settings-toolbar-actions">${button('settingsReset', 'Discard profile', 'refresh', 'ghost', locked || !dirty)}${button('settingsSave', 'Save profile', 'check', 'primary', locked || !dirty || draftConflict || automaticConfigIsDirty())}</div></div>
    ${conflict}
    <div class="settings-grid">
      <article class="settings-card settings-card-wide settings-appearance"><div class="settings-card-heading"><div class="settings-card-icon">${icon(themePreference === 'dark' ? 'moon' : themePreference === 'light' ? 'sun' : 'desktop', 18)}</div><div><h3>Appearance</h3><p>Choose the surface contrast for every vocal-fast window.</p></div></div><div class="settings-choice-list" role="group" aria-label="Theme preference">${themeOptions}</div></article>
      <article class="settings-card settings-card-wide"><div class="settings-card-heading"><div class="settings-card-icon">${icon('speaker', 18)}</div><div><h3>Audio routing</h3><p>Saved automatically. Device changes apply on the next Start.</p></div><div class="settings-card-actions">${button('settingsRefreshDevices', 'Refresh', 'refresh', 'icon-only', busy, 'Refresh audio devices')}<span class="settings-card-badge">${devices ? `${inputDeviceCount} in · ${outputDeviceCount} out` : 'Not queried'}</span></div></div><div class="settings-form-grid"><label class="field-label">Capture<select id="settings-inputDevice" ${locked ? 'disabled' : ''}>${deviceOptions('input', config.inputDevice)}</select></label><label class="field-label">Render<select id="settings-outputDevice" ${locked ? 'disabled' : ''}>${deviceOptions('output', config.outputDevice)}</select></label><label class="field-label">Worker sample rate<select id="settings-inputRate" disabled>${rateOptions}</select></label><label class="field-label">Worker channels<input id="settings-channels" type="number" min="1" max="8" step="1" value="${config.inputFormat.channels}" disabled/></label></div>${renderAudioRoute(snapshot?.audio.route ?? null)}</article>
      <article class="settings-card settings-card-wide"><div class="settings-card-heading"><div class="settings-card-icon">${icon('cpu', 18)}</div><div><h3>Session profile</h3><p>Choose a preset while idle, then click Save profile.</p></div></div>${renderSessionProfiles(config.tensorrt.profile, locked, 'settings-')}</article>
      <article class="settings-card"><div class="settings-card-heading"><div class="settings-card-icon">${icon('chart', 18)}</div><div><h3>Realtime budget</h3><p>Keep the worker responsive without starving device callbacks.</p></div></div><div class="settings-form-grid"><label class="field-label">Block frames<select id="settings-blockFrames" disabled>${blockFrameOptions}</select></label><label class="field-label">Queue depth<input id="settings-queueBlocks" type="number" min="1" max="64" step="1" value="${config.queueBlocks}" disabled/></label></div><div class="settings-budget"><span>${icon('chart', 13)} Estimated buffered age</span><strong>${((config.blockFrames * (config.queueBlocks + 1) / config.inputFormat.sampleRate) * 1000).toFixed(1)} ms</strong></div></article>
      <article class="settings-card"><div class="settings-card-heading"><div class="settings-card-icon">${icon('database', 18)}</div><div><h3>Model library</h3><p>Persist the folder used for model discovery.</p></div></div><label class="field-label">Library folder<input id="settings-modelDirectory" value="${escapeHtml(config.modelDirectory ?? '')}" placeholder="Choose a model folder…" ${locked ? 'disabled' : ''}/></label><div class="settings-inline-actions">${button('settingsBrowseModels', 'Browse', 'folder', 'secondary', locked || !runningInTauri())}${button('settingsScanModels', 'Scan library', 'refresh', 'ghost', locked)}</div></article>
      <article class="settings-card settings-card-wide">${renderProcessingBackend(config, processingBackends, locked, 'settings-')}</article>
      <article class="settings-card settings-card-wide settings-about"><div class="settings-card-heading"><div class="settings-card-icon">${icon('keyboard', 18)}</div><div><h3>Window and keyboard</h3><p><kbd>Space</kbd> starts or stops a session. <kbd>Ctrl R</kbd> refreshes the runtime snapshot. Settings are shared across Studio and auxiliary windows.</p></div><button id="settingsStudio" class="button secondary">Open Studio</button></div></article>
    </div>
  </section>`;
}

function renderSidebar(): string {
  const navItem = (id: string, label: string, iconName: IconName) =>
    `<button class="nav-item ${activeSection === id ? 'active' : ''}" data-section="${id}" aria-label="${label}" aria-current="${activeSection === id ? 'page' : 'false'}" title="${label}">${icon(iconName, 17)}<span>${label}</span></button>`;
  return `<aside class="sidebar">
    <div class="sidebar-heading"><div class="brand-mark">${icon('waveform', 17)}</div><div><strong>Vocal Studio</strong><span>Realtime workspace</span></div></div>
    <div class="sidebar-navigation"><div class="nav-label">Workspace</div>
    <nav class="nav-list" aria-label="Workspace navigation">${navItem('studio', 'Studio', 'waveform')}${navItem('models', 'Models', 'database')}${navItem('audio', 'Audio routing', 'speaker')}${navItem('tuning', 'Voice controls', 'sliders')}${navItem('diagnostics', 'Diagnostics', 'chart')}</nav>
    <div class="nav-label nav-label-settings">Application</div>
    <nav class="nav-list" aria-label="Application navigation">${navItem('settings', 'Settings', 'settings')}</nav></div>
    <div class="sidebar-bottom"><div class="shortcut-hint">${icon('keyboard', 15)}<span><kbd>Space</kbd> Start or stop</span></div>
    <button id="themeButton" class="theme-button" title="Change appearance">${icon(themePreference === 'dark' ? 'moon' : themePreference === 'light' ? 'sun' : 'desktop', 16)}<span>${themeLabel(themePreference)}</span><span class="chevron">${icon('chevron', 14)}</span></button>
    <div class="sidebar-footer">${renderConfigPersistence()}<div class="sidebar-connection" title="${channelConnected ? 'Control channel connected' : 'Connecting to runtime…'}"><span class="connection-dot ${channelConnected ? 'online' : ''}"></span><span class="connection-label">${channelConnected ? 'Control channel connected' : 'Connecting to runtime…'}</span></div></div></div>
  </aside>`;
}

function renderModelControls(locked: boolean): string {
  const path = displayPath(modelSelectionPath());
  return `<div class="model-actions"><div class="model-path-row"><input id="modelPath" aria-label="Model path" title="${escapeHtml(path)}" value="${escapeHtml(path)}" placeholder="Paste a model path…" ${locked ? 'disabled' : ''}/></div>${renderDependencySetup(locked)}<div class="model-action-row"><span id="modelSelectionStatus" class="model-selection-status" role="status">${modelSelectionLabel()}</span><div class="model-buttons">${button('load', 'Load', 'folder', 'primary compact', locked || !path.trim() || !backendAvailable(draftConfig ?? snapshot?.config ?? defaultConfig(), processingBackends) || (!!modelDependencyStatus && modelDependencyStatus.phase !== 'ready'), 'Load selected model')}${button('unload', 'Unload', undefined, 'secondary compact', locked || !snapshot?.model, 'Unload the current model')}</div></div></div>`;
}

function renderDependencySetup(locked: boolean): string {
  const status = modelDependencyStatus;
  if (!status || status.phase === 'idle' || status.phase === 'ready') return '';
  const item = (label: string, value: ModelDependencyStatus['hubert']) => `<span class="dependency-item"><strong>${label}</strong><span>${value.present ? `Ready · ${formatBytes(value.sizeBytes)}` : `Missing${value.expectedSizeBytes ? ` · ${formatBytes(value.expectedSizeBytes)}` : ''}`}</span></span>`;
  const active = ['queued', 'listing', 'downloading', 'binding', 'cancelling'].includes(status.phase);
  const progress = status.totalBytes ? ` · ${formatBytes(status.downloadedBytes)} / ${formatBytes(status.totalBytes)}` : '';
  const actions = status.phase === 'needsConsent' || status.phase === 'binding'
    ? `<button id="installModelDependencies" class="button primary compact" ${locked ? 'disabled' : ''}>${icon(status.phase === 'binding' ? 'sparkle' : 'download', 14)}${status.phase === 'binding' ? 'Prepare and continue' : 'Download and continue'}</button><button id="pickHubertDependency" class="button ghost compact" ${locked ? 'disabled' : ''}>Choose HuBERT</button><button id="pickRmvpeDependency" class="button ghost compact" ${locked ? 'disabled' : ''}>Choose RMVPE</button>`
    : active ? `<button id="cancelModelDependencies" class="button ghost compact">Cancel</button>`
      : status.phase === 'failed' || status.phase === 'cancelled' ? `<button id="installModelDependencies" class="button secondary compact" ${locked ? 'disabled' : ''}>Retry</button>` : '';
  return `<section class="model-dependency-setup ${status.phase === 'failed' ? 'failed' : ''}" role="status" aria-live="polite"><div class="dependency-heading"><strong>${escapeHtml(status.message)}</strong><span>${active ? `${escapeHtml(status.phase)}${progress}` : 'Setup required'}</span></div><div class="dependency-items">${item('HuBERT', status.hubert)}${item('RMVPE', status.rmvpe)}</div>${status.phase === 'needsConsent' ? '<p>Shared acoustic models are downloaded once and reused by every voice model on this device.</p>' : ''}<div class="dependency-actions">${actions}</div></section>`;
}

function renderModelCatalog(locked: boolean): string {
  if (activeAction?.includes('scan-models') || activeAction?.includes('browse-models')) return renderPreparationProgress({
    operationId: -1, revision: 0, kind: null, phase: 'inspecting', message: 'Scanning model metadata and component directories…',
    elapsedMs: performance.now() - actionStartedAt, completedSteps: 0, totalSteps: null, cacheHit: false,
  }, 'library-');
  if (!modelCatalog) {
    return `<div class="model-library-empty"><span class="model-library-empty-icon">${icon('database', 17)}</span><span>Choose a model library folder and scan it to discover VFRM caches and manifests.</span></div>`;
  }
  const entries = modelCatalog.entries
    .map((entry) => {
      const selected = samePath(modelSelectionPath(), entry.path);
      const loaded = !!snapshot?.model && samePath(snapshot.config.modelPath, entry.path);
      return `<button class="model-library-entry ${selected ? 'selected' : ''}" data-model-path="${escapeHtml(entry.path)}" title="${escapeHtml(displayPath(entry.path))}" aria-pressed="${selected}" ${locked ? 'disabled' : ''}><span class="model-library-entry-mark">${icon(entry.kind === 'vfrm' ? 'database' : 'folder', 16)}</span><span class="model-library-entry-copy"><strong>${escapeHtml(entry.displayName)}</strong><small>${escapeHtml(entry.architecture)} · v${escapeHtml(entry.version)} · ${entry.sampleRate.toLocaleString()} Hz</small>${renderComponentBadges(entry.components)}</span><span class="model-library-entry-meta"><small>${formatBytes(entry.sizeBytes)}</small><small>${loaded ? 'Loaded' : selected ? 'Selected' : entry.components?.automaticTensorRt ? 'Complete' : entry.kind.toUpperCase()}</small></span></button>`;
    })
    .join('');
  const issue = modelCatalog.issues.length
    ? `<details class="model-library-issues"><summary>${icon('alert', 14)}<span>${modelCatalog.issues.length} artifact${modelCatalog.issues.length === 1 ? '' : 's'} skipped</span></summary><ul>${modelCatalog.issues.map((item) => `<li><strong>${escapeHtml(displayPath(item.path))}</strong><span>${escapeHtml(item.message)}</span></li>`).join('')}</ul></details>`
    : '';
  return `<div class="model-library-summary"><span>${icon('folder', 14)}${escapeHtml(displayPath(modelCatalog.root))}</span><span>${modelCatalog.entries.length} model${modelCatalog.entries.length === 1 ? '' : 's'}</span></div>${entries || '<div class="model-library-empty compact">No supported model manifests or VFRM caches found.</div>'}${issue}`;
}

function render(): void {
  const view = appRoot.cloneNode(false) as HTMLElement;
  if (isSplashWindow()) {
    document.documentElement.classList.add('splash-window');
    view.innerHTML = renderSplash();
    patchView(appRoot, view);
    return;
  }
  document.documentElement.classList.remove('splash-window');
  const current = snapshot;
  const config = draftConfig ?? current?.config ?? defaultConfig();
  const audio = current?.audio;
  const report = localError ?? current?.lastError;
  const running = current?.lifecycle === 'running';
  const transitional = pipelineBusy() || current?.lifecycle === 'loading' || current?.lifecycle === 'stopping';
  const modelLocked = busy || transitional || running || !!audio?.active;
  const model = current?.model;
  const displayedModel = model ?? ((current?.modelRestorePending || activeAction === 'restore-model') ? startupResources?.savedModel?.summary : null);
  const preparing = preparationActive(preparationStatus) || !!current?.modelRestorePending;
  const status = preparing ? 'loading' : current?.lifecycle ?? 'created';
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
  view.innerHTML = `${renderTitlebar()}<div class="app-layout">${renderSidebar()}<div class="content-scroll" id="content-scroll"><main class="content"><div class="page-heading"><div><div class="kicker">REALTIME VOICE WORKSPACE</div><h1>Make every take sound like you.</h1><p>Configure your voice chain, start the stream, and keep an eye on the audio path.</p></div><div class="headline-status ${statusTone(status)}"><span class="status-dot"></span><div><strong>${lifecycleLabel(status)}</strong><small>${audio?.active ? 'Audio stream online' : channelConnected ? 'Runtime ready for control' : 'Waiting for runtime'}</small></div></div></div>
    <section class="hero-panel ${running ? 'is-running' : ''}"><div class="hero-glow"></div><div class="hero-grid"></div><div class="hero-content"><div class="eyebrow-row"><span class="eyebrow-chip">${running ? 'LIVE SESSION' : 'SESSION CONTROL'}</span><span class="revision">${profileIsDirty() ? 'Save session profile before starting' : automaticConfigIsDirty() ? 'Saving settings…' : `Revision ${current?.configRevision ?? 0}`}</span></div><h2>${model ? escapeHtml(model.displayName) : 'No model loaded'}</h2><p>${model ? `${escapeHtml(model.architecture)} · ${model.sampleRate.toLocaleString()} Hz · ${model.featureDims} feature dimensions${model.usesF0 ? ' · F0 enabled' : ''}` : 'Load a .vfr cache or model manifest to prepare your voice chain.'}</p><div class="hero-wave" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div><div class="hero-actions">${button('start', running ? 'Streaming' : 'Start stream', running ? 'check' : 'play', 'primary', busy || transitional || !current?.engine || running || configIsDirty() || profileBuildRequired() || !backendAvailable(config, processingBackends))}${button('stop', 'Stop', 'stop', 'secondary', busy || !running)}${button('demo', 'Test block', 'sparkle', 'ghost', busy || !running || !!audio?.active, 'Run one synthetic block through the engine')}</div>${renderAudioRoute(audio?.route ?? null, true)}</div><div class="hero-orb"><div class="orb-ring ring-one"></div><div class="orb-ring ring-two"></div><div class="orb-core">${icon('waveform', 42)}</div><span class="orb-caption">${running ? 'Listening' : 'Standby'}</span></div></section>
    <section class="workspace-grid"><article class="panel model-panel span-2" data-section-target="studio">${panelHeader('MODEL LIBRARY', 'Voice model', `<span class="panel-badge">${model ? icon('check', 13) + ' Loaded' : icon('database', 13) + ' Awaiting model'}</span>`)}<div class="model-library-controls"><label class="field-label"><span>Library folder</span><input id="modelDirectory" aria-label="Model library folder" value="${escapeHtml(config.modelDirectory ?? '')}" placeholder="Choose a folder containing model artifacts..." ${busy || transitional || running ? 'disabled' : ''}/></label><div class="model-library-actions">${button('browseModels', 'Browse', 'folder', 'secondary', busy || transitional || running || !runningInTauri())}${button('scanModels', 'Scan library', 'refresh', 'secondary', busy || transitional || running)}</div></div><div class="model-library-list">${renderModelCatalog(modelLocked)}</div><div class="model-layout"><div class="model-avatar">${model ? escapeHtml(model.displayName.slice(0, 1).toUpperCase()) : icon('database', 25)}</div><div class="model-details"><strong>${model ? escapeHtml(model.displayName) : 'Choose a model to begin'}</strong><span>${model ? `${escapeHtml(model.id)} · v${escapeHtml(model.version)}` : 'VFRM caches and RVC manifests are supported'}</span>${model ? `<div class="tag-row"><span class="tag">${escapeHtml(model.architecture)}</span><span class="tag">${model.sampleRate.toLocaleString()} Hz</span>${model.usesF0 ? '<span class="tag accent">F0 enabled</span>' : ''}</div>` : ''}</div>${renderModelControls(modelLocked)}</div>${renderProcessingBackend(config, processingBackends, modelLocked)}</article>
    <article class="panel audio-panel" data-section-target="audio">${panelHeader('AUDIO ROUTING', 'Devices', button('refreshDevices', 'Refresh', 'refresh', 'icon-only', false, 'Refresh device list'))}<div class="host-line"><span class="live-indicator ${audio?.active ? 'active' : ''}"></span><span>${devices ? escapeHtml(devices.host) : 'Audio host not queried'}</span><span class="host-count">${devices ? `${devices.devices.length} devices` : '—'}</span></div><label class="field-label">Capture<select id="inputDevice" ${modelLocked ? 'disabled' : ''}>${deviceOptions('input', config.inputDevice)}</select></label><label class="field-label">Render<select id="outputDevice" ${modelLocked ? 'disabled' : ''}>${deviceOptions('output', config.outputDevice)}</select></label><div class="route-summary"><span>${icon('speaker', 14)}${escapeHtml(inputName ?? 'System default')}</span><span class="route-arrow">→</span><span>${escapeHtml(outputName ?? 'System default')}</span></div>${renderAudioRoute(audio?.route ?? null)}</article>
    <article class="panel tuning-panel" data-section-target="tuning">${panelHeader('VOICE CONTROLS', 'Shape the sound', '<span class="panel-badge subtle">Auto-save · idle only</span>')}${renderPitchMatching(config.conversion, modelLocked)}<div class="slider-list">${slider('pitch', 'Additional pitch', `${config.conversion.pitchSemitones.toFixed(1)} st`, config.conversion.pitchSemitones, -24, 24, 0.1)}${slider('index', 'Index ratio', config.conversion.indexRatio.toFixed(2), config.conversion.indexRatio, 0, 1, 0.01)}${slider('protect', 'Unvoiced protect', config.conversion.protect.toFixed(2), config.conversion.protect, 0, 1, 0.01)}${slider('rms', 'RMS mix', config.conversion.rmsMixRate.toFixed(2), config.conversion.rmsMixRate, 0, 1, 0.01)}</div></article>
    <article class="panel config-panel span-2" data-section-target="studio">${panelHeader('ENGINE & STREAM', 'Session configuration', '<span class="panel-badge subtle">Three presets</span>')}${renderSessionProfiles(config.tensorrt.profile, busy || transitional || running)}<div class="session-config-footer"><p>Profile changes require Save. Devices, voice controls and model library settings save automatically.</p><div class="session-config-actions">${button('save', 'Save profile', 'check', 'primary', busy || transitional || running || !profileIsDirty() || draftConflict || automaticConfigIsDirty())}</div></div></article>
    ${activeSection === 'diagnostics' ? renderDiagnostics(audio, diagnosticsHistory, channelConnected, button('diagnosticStart', 'Start stream', 'play', 'primary', busy || transitional || !current?.engine || running || configIsDirty() || profileBuildRequired() || !backendAvailable(config, processingBackends)) + button('diagnosticStop', 'Stop', 'stop', 'secondary', busy || !running)) : ''}</section>
    ${bootMessage ? `<section class="boot-strip ${bootPending ? 'pending' : ''}"><span class="boot-strip-dot"></span><div><strong>${bootPending ? 'Starting vocal-fast' : 'Startup notice'}</strong><span>${escapeHtml(bootMessage)}</span></div></section>` : ''}${report ? `<section class="error-banner" role="alert"><div class="error-icon">${icon('alert', 18)}</div><div><strong>${escapeHtml(report.code)}</strong><span>${escapeHtml(report.message)}</span></div><div class="error-actions"><button id="showErrorDetails" class="button ghost">Details</button>${report.recoverable ? button('clearError', 'Dismiss', 'close', 'ghost') : ''}</div></section>` : ''}<footer class="app-footer"><span>Audio stays in the Rust worker; this surface receives snapshots and metrics.</span><span>${icon('keyboard', 13)} <kbd>Space</kbd> session · <kbd>Ctrl R</kbd> refresh</span></footer></main></div></div>${report && errorDialogOpen ? renderErrorDialog(report) : ''}`;
  const content = view.querySelector<HTMLElement>('main.content');
  if (content) {
    if (preparing) {
      const detail = content.querySelector('.headline-status small');
      if (detail) detail.textContent = 'Audio starts after preparation';
    }
    if (!model && displayedModel) {
      const title = content.querySelector('.hero-content h2');
      const description = content.querySelector('.hero-content > p');
      const caption = content.querySelector('.orb-caption');
      if (title) title.textContent = displayedModel.displayName;
      if (description) description.textContent = 'Preparing the saved model · audio becomes available when the engine is ready.';
      if (caption) caption.textContent = 'Preparing';
    }
    content.classList.toggle('settings-route', activeSection === 'settings');
    content.classList.toggle('diagnostics-route', activeSection === 'diagnostics');
    const copy = routeCopy(activeSection);
    const pageTitle = content.querySelector<HTMLElement>('.page-heading h1');
    const pageDescription = content.querySelector<HTMLElement>('.page-heading p');
    if (pageTitle) pageTitle.textContent = copy.title;
    if (pageDescription) pageDescription.textContent = copy.description;
    const heading = content.querySelector('.page-heading');
    heading?.insertAdjacentHTML('afterend', `${renderTabs()}${renderWorkspacePreparation()}${renderSettingsPanel(config)}`);
    const workspace = content.querySelector<HTMLElement>('.workspace-grid');
    workspace?.classList.add(`route-${activeSection}`);
    workspace?.insertAdjacentHTML('beforeend',modelManagement.render(modelCatalog,modelLocked));
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
  if (patchView(appRoot, view)) wireHandlers();
  syncDraftControls();
}

function readConfigForm(): GuiConfig {
  const current = draftConfig ?? snapshot?.config ?? defaultConfig();
  const formId = (id: string) => activeSection === 'settings' ? `settings-${id}` : id;
  const element = <T extends HTMLElement>(id: string): T | null =>
    document.querySelector<T>(`#${formId(id)}`) ?? document.querySelector<T>(`#${id}`);
  const number = (id: string, fallback: number) => Number(element<HTMLInputElement>(id)?.value ?? fallback);
  const sampleRate = current.inputFormat.sampleRate;
  const channels = current.inputFormat.channels;
  return {
    ...current,
    backend: current.backend,
    speakerId: element<HTMLInputElement>('speakerId')?.valueAsNumber ?? current.speakerId,
    inputFormat: { sampleRate, channels },
    outputFormat: { ...current.outputFormat },
    modelPath: snapshot?.config.modelPath ?? null,
    modelDirectory: (element<HTMLInputElement>('modelDirectory')?.value ?? current.modelDirectory)?.trim() || null,
    inputDevice: element<HTMLSelectElement>('inputDevice') ? element<HTMLSelectElement>('inputDevice')!.value || null : current.inputDevice,
    outputDevice: element<HTMLSelectElement>('outputDevice') ? element<HTMLSelectElement>('outputDevice')!.value || null : current.outputDevice,
    blockFrames: current.blockFrames,
    queueBlocks: current.queueBlocks,
    tensorrt: {
      ...current.tensorrt,
      profile: document.querySelector<HTMLInputElement>(`input[name="${activeSection === 'settings' ? 'settings-' : ''}sessionProfile"]:checked`)?.value ?? current.tensorrt.profile,
    },
    conversion: {
      ...current.conversion,
      pitchSemitones: number('pitch', current.conversion.pitchSemitones),
      autoPitch: { enabled: element<HTMLInputElement>('autoPitchEnabled')?.checked ?? current.conversion.autoPitch.enabled, targetHz: number('autoPitchTarget', current.conversion.autoPitch.targetHz) },
      indexRatio: number('index', current.conversion.indexRatio),
      protect: number('protect', current.conversion.protect),
      rmsMixRate: number('rms', current.conversion.rmsMixRate),
    },
  };
}

async function runAction(action: () => Promise<unknown>, key = 'command', applyDraft = false, backendChange = false): Promise<void> {
  const now = performance.now();
  if (busy || automaticSaveRequest || now - (lastActionAt.get(key) ?? -Infinity) < MIN_ACTION_GAP_MS) return;
  lastActionAt.set(key, now);
  const previousSnapshot = snapshot;
  const previousConflict = draftConflict;
  busy = true;
  window.clearTimeout(automaticSaveTimer);
  automaticSaveTimer = undefined;
  activeAction = key;
  actionStartedAt = performance.now();
  actionPreparationId = preparationStatus?.operationId ?? 0;
  localError = null;
  render();
  try {
    const result = await action();
    if (isSnapshot(result)) {
      if (snapshot && result.configRevision < snapshot.configRevision) return;
      const preserved = !applyDraft && draftConfig && previousSnapshot
        ? reconcileConfigDraft(draftConfig, previousSnapshot, result, draftRevision, previousConflict, backendChange)
        : null;
      snapshot = result;
      diagnosticsHistory.observe(result.audio, performance.now());
      acceptPreparation(result.preparation);
      if (preserved) {
        draftConfig = preserved.config;
        draftRevision = preserved.revision;
        draftBaseConfig = result.config;
        draftConflict = preserved.conflict;
        if (!configIsDirty()) clearDraftConfig();
      } else {
        clearDraftConfig();
      }
    }
  } catch (error) {
    localError = normalizeError(error);
  } finally {
    busy = false;
    activeAction = null;
    queueAutomaticSave();
    render();
  }
}
function isSnapshot(value: unknown): value is Snapshot {
  return typeof value === 'object' && value !== null && 'lifecycle' in value && 'configRevision' in value;
}

function dependencyActive(status: ModelDependencyStatus | null): boolean {
  return !!status && ['queued', 'listing', 'downloading', 'binding', 'cancelling'].includes(status.phase);
}

async function waitForDependency(jobId: number): Promise<ModelDependencyStatus> {
  for (;;) {
    const status = await call<ModelDependencyStatus>('get_model_dependency_status');
    if (status.jobId !== jobId) throw new Error('Dependency operation was replaced by another request');
    modelDependencyStatus = status;
    render();
    if (!dependencyActive(status)) return status;
    await new Promise<void>(resolve => window.setTimeout(resolve, 220));
  }
}

async function prepareDependenciesAndLoad(path: string, allowDownload: boolean): Promise<Snapshot> {
  const started = await call<ModelDependencyStatus>('start_model_dependency_install', dependencyRequest(path, allowDownload));
  modelDependencyStatus = started;
  render();
  const finished = dependencyActive(started) ? await waitForDependency(started.jobId) : started;
  if (finished.phase !== 'ready' || !finished.resolvedModelPath) {
    throw new Error(finished.error?.message ?? finished.message);
  }
  selectedModelPath = finished.resolvedModelPath;
  inspectSelectedModel();
  const config = draftConfig ?? snapshot?.config ?? defaultConfig();
  const backendChanged = backendSelectionChanged(config, snapshot?.config ?? defaultConfig(), processingBackends);
  return call<Snapshot>('load_model', {
    path: finished.resolvedModelPath,
    selection: backendChanged ? backendLoadSelection(config, processingBackends, draftRevision ?? snapshot?.configRevision ?? 0) : null,
  });
}

function wireHandlers(): void {
  modelManagement.wire(busy || pipelineBusy() || !!snapshot?.audio.active || snapshot?.lifecycle==='running' || snapshot?.lifecycle==='loading' || snapshot?.lifecycle==='stopping');
  const byId = <T extends HTMLElement>(id: string) => document.querySelector<T>(`#${id}`);
  on(byId('diagnosticStart'), 'click', () => byId<HTMLButtonElement>('start')?.click());
  on(byId('diagnosticStop'), 'click', () => byId<HTMLButtonElement>('stop')?.click());
  on(byId('errorAudioRouting'), 'click', () => { errorDialogOpen = false; navigateTo('audio'); });
  on(byId<HTMLButtonElement>('load'), 'click', () => {
    const config = draftConfig ?? snapshot?.config ?? defaultConfig();
    const backendChanged = backendSelectionChanged(config, snapshot?.config ?? defaultConfig(), processingBackends);
    void runAction(async () => {
      const path = modelSelectionPath().trim();
      if (!path) throw new Error('Model path is required');
      if (!backendAvailable(config, processingBackends)) throw new Error('The selected processing backend is unavailable. Refresh devices or choose another implementation.');
      const dependencies = await call<ModelDependencyStatus>('inspect_model_dependencies', dependencyRequest(path));
      modelDependencyStatus = dependencies;
      if (dependencies.phase !== 'ready') {
        render();
        return snapshot;
      }
      const resolved = dependencies.resolvedModelPath ?? path;
      selectedModelPath = resolved;
      const loaded = await call<Snapshot>('load_model', {
        path: resolved,
        selection: backendChanged ? backendLoadSelection(config, processingBackends, draftRevision ?? snapshot?.configRevision ?? 0) : null,
      });
      selectedModelPath = undefined;
      return loaded;
    }, 'load-model', false, backendChanged || resolvedBackend(config, processingBackends) === 'tensorRtRtx');
  });
  on(byId<HTMLButtonElement>('installModelDependencies'), 'click', () => {
    runAction(async () => {
      const path = modelSelectionPath().trim();
      if (!path) throw new Error('Model path is required');
      return prepareDependenciesAndLoad(path, true);
    }, 'install-model-dependencies', false, true);
  });
  on(byId<HTMLButtonElement>('pickHubertDependency'), 'click', () => {
    runAction(async () => {
      const path = await call<string | null>('pick_model_file', { kind: 'acoustic' });
      if (path) { dependencyHubertSource = path; modelDependencyStatus = { ...(modelDependencyStatus as ModelDependencyStatus), message: 'HuBERT selected; choose RMVPE or download the remaining dependency' }; render(); }
      return snapshot;
    }, 'pick-hubert-dependency');
  });
  on(byId<HTMLButtonElement>('pickRmvpeDependency'), 'click', () => {
    runAction(async () => {
      const path = await call<string | null>('pick_model_file', { kind: 'acoustic' });
      if (path) { dependencyRmvpeSource = path; modelDependencyStatus = { ...(modelDependencyStatus as ModelDependencyStatus), message: 'RMVPE selected; choose HuBERT or continue' }; render(); }
      return snapshot;
    }, 'pick-rmvpe-dependency');
  });
  on(byId<HTMLButtonElement>('cancelModelDependencies'), 'click', () => runAction(async () => {
    if (!modelDependencyStatus?.jobId) return snapshot;
    modelDependencyStatus = await call<ModelDependencyStatus>('cancel_model_dependency_install', { jobId: modelDependencyStatus.jobId });
    return snapshot;
  }, 'cancel-model-dependencies'));
  on(byId<HTMLInputElement>('modelPath'), 'keydown', (event) => {
    if (event.key === 'Enter') byId<HTMLButtonElement>('load')?.click();
  });
  on(byId<HTMLButtonElement>('browseModels'), 'click', () =>
    runAction(async () => {
      const directory = await call<string | null>('pick_model_directory');
      if (!directory) return undefined;
      const directoryInput = activeSection === 'settings'
        ? byId<HTMLInputElement>('settings-modelDirectory')
        : byId<HTMLInputElement>('modelDirectory');
      if (directoryInput) directoryInput.value = directory;
      modelCatalog = await call<ModelCatalog>('list_models', { path: directory });
      setDraftConfig(readConfigForm());
      return undefined;
    }, 'browse-models'));
  on(byId<HTMLButtonElement>('settingsBrowseModels'), 'click', () =>
    runAction(async () => {
      const directory = await call<string | null>('pick_model_directory');
      if (!directory) return undefined;
      const directoryInput = byId<HTMLInputElement>('settings-modelDirectory');
      if (directoryInput) directoryInput.value = directory;
      modelCatalog = await call<ModelCatalog>('list_models', { path: directory });
      setDraftConfig(readConfigForm());
      return undefined;
    }, 'settings-browse-models'));
  on(byId<HTMLButtonElement>('scanModels'), 'click', () =>
    runAction(async () => {
      const directory = (activeSection === 'settings'
        ? byId<HTMLInputElement>('settings-modelDirectory')
        : byId<HTMLInputElement>('modelDirectory'))?.value.trim();
      if (!directory) throw new Error('Model library folder is required');
      modelCatalog = await call<ModelCatalog>('list_models', { path: directory });
      setDraftConfig({ ...(draftConfig ?? snapshot?.config ?? defaultConfig()), modelDirectory: directory });
      return undefined;
    }, 'scan-models'));
  on(byId<HTMLButtonElement>('settingsScanModels'), 'click', () =>
    runAction(async () => {
      const directory = byId<HTMLInputElement>('settings-modelDirectory')?.value.trim();
      if (!directory) throw new Error('Model library folder is required');
      modelCatalog = await call<ModelCatalog>('list_models', { path: directory });
      setDraftConfig({ ...(draftConfig ?? snapshot?.config ?? defaultConfig()), modelDirectory: directory });
      return undefined;
    }, 'settings-scan-models'));
  document.querySelectorAll<HTMLButtonElement>('[data-model-path]').forEach((item) =>
    on(item, 'click', () => {
      const path = item.dataset.modelPath;
      if (!path) return;
      selectedModelPath = path;
      inspectSelectedModel();
      render();
    }),
  );
  on(byId<HTMLButtonElement>('unload'), 'click', () =>
    runAction(async () => {
      const path = modelSelectionPath();
      const unloaded = await call<Snapshot>('unload_model');
      selectedModelPath = path;
      return unloaded;
    }, 'unload-model'));
  on(byId<HTMLButtonElement>('start'), 'click', () =>
    runAction(() => call<Snapshot>('start_session'), 'start-session'));
  on(byId<HTMLButtonElement>('stop'), 'click', () =>
    runAction(() => call<Snapshot>('stop_session'), 'stop-session'));
  on(byId<HTMLButtonElement>('clearError'), 'click', () =>
    runAction(async () => {
      if (localError) {
        localError = null;
        errorDialogOpen = false;
        render();
        return snapshot;
      }
      errorDialogOpen = false;
      return call<Snapshot>('clear_error');
    }, 'clear-error'));
  on(byId<HTMLButtonElement>('showErrorDetails'), 'click', () => {
    errorDialogOpen = true;
    render();
  });
  on(byId<HTMLButtonElement>('dismissError'), 'click', () => {
    errorDialogOpen = false;
    if (localError) localError = null;
    render();
  });
  on(byId<HTMLButtonElement>('copyError'), 'click', () => {
    const report = localError ?? snapshot?.lastError;
    if (!report) return;
    void navigator.clipboard
      ?.writeText(JSON.stringify(report, null, 2))
      .catch(() => undefined);
  });
  on(byId<HTMLButtonElement>('refreshDevices'), 'click', () =>
    runAction(async () => {
      devices = await call<AudioDeviceCatalog>('list_audio_devices');
      return devices;
    }, 'refresh-devices'));
  on(byId<HTMLButtonElement>('settingsRefreshDevices'), 'click', () =>
    runAction(async () => {
      devices = await call<AudioDeviceCatalog>('list_audio_devices');
      return devices;
    }, 'refresh-devices'));
  const saveConfiguration = () => runAction(
    async () => {
      if (!snapshot || !draftConfig || !profileIsDirty()) return undefined;
      const config = { ...snapshot.config, tensorrt: { ...snapshot.config.tensorrt, profile: draftConfig.tensorrt.profile } };
      const validation = validateConfig(config);
      if (validation) throw new Error(validation);
      if (draftConflict) throw new Error('Configuration changed in another window. Reload before saving.');
      return call<Snapshot>('update_config', {
        config,
        expectedRevision: draftRevision ?? snapshot?.configRevision ?? 0,
      });
    },
    'save-config',
    true,
  );
  on(byId<HTMLButtonElement>('save'), 'click', saveConfiguration);
  on(byId<HTMLButtonElement>('settingsSave'), 'click', saveConfiguration);
  on(byId<HTMLButtonElement>('settingsReset'), 'click', () => {
    if (!draftConfig || !snapshot || !profileIsDirty()) return;
    setDraftConfig({ ...draftConfig, tensorrt: { ...draftConfig.tensorrt, profile: snapshot.config.tensorrt.profile } }, false);
    render();
  });
  on(byId<HTMLButtonElement>('settingsReload'), 'click', () => {
    clearDraftConfig();
    localError = null;
    render();
  });
  document.querySelectorAll<HTMLButtonElement>('[data-retry-autosave]').forEach(control => on(control, 'click', () => {
    automaticSaveError = null;
    automaticSaveState = 'pending';
    automaticConflictRetries = 0;
    queueAutomaticSave();
    render();
  }));
  document.querySelectorAll<HTMLButtonElement>('[data-reload-config]').forEach(control => on(control, 'click', () => {
    clearDraftConfig();
    automaticSaveState = 'idle';
    render();
  }));
  document.querySelectorAll<HTMLButtonElement>('[data-theme-choice]').forEach((item) =>
    on(item, 'click', () => {
      const choice = item.dataset.themeChoice;
      if (choice === 'system' || choice === 'light' || choice === 'dark') setThemePreference(choice);
    }),
  );
  document.querySelectorAll<HTMLButtonElement>('[data-prepare-backend]').forEach((control) => on(control, 'click', () =>
    runAction(async () => {
      const form = snapshot?.config ?? defaultConfig();
      if (draftConflict) throw new Error('Configuration changed in another window. Reload before building.');
      const path = modelSelectionPath().trim();
      if (!path) throw new Error('Choose a voice model before building plans.');
      const saved = await call<Snapshot>('update_config', {
        config: { ...form, modelPath: path },
        expectedRevision: draftRevision ?? snapshot?.configRevision ?? 0,
      });
      snapshot = saved;
      diagnosticsHistory.observe(saved.audio, performance.now());
      clearDraftConfig();
      const prepared = await call<Snapshot>('prepare_tensorrt_pipeline', {
        request: {
          expectedRevision: saved.configRevision,
          modelPath: path,
          hubertSource: form.tensorrt.hubertSource,
          rmvpeSource: form.tensorrt.rmvpeSource,
          cacheDirectory: form.tensorrt.cacheDirectory,
        },
      });
      snapshot = prepared;
      diagnosticsHistory.observe(prepared.audio, performance.now());
      const loaded = await call<Snapshot>('load_model', { path });
      selectedModelPath = undefined;
      return loaded;
    }, 'prepare-tensorrt', true)));

  document.querySelectorAll<HTMLSelectElement>('[data-processing-backend]').forEach((control) => on(control, 'change', () => {
    const backend = control.value === 'auto' ? processingBackends?.recommended : processingBackends?.backends.find((item) => item.backend === control.value && item.available)?.backend;
    if (!backend) return;
    setDraftConfig({ ...readConfigForm(), backend, backendAuto: control.value === 'auto' });
    render();
  }));
  document.querySelectorAll<HTMLSelectElement>('[data-directml-device]').forEach((control) => on(control, 'change', () => {
    const id = control.value === 'auto' ? null : control.value;
    if (id && !processingBackends?.backends.find(item => item.backend === 'directMl')?.devices.some(device => device.id === id && device.available)) return;
    setDraftConfig({ ...readConfigForm(), directmlDeviceId: id });
    render();
  }));
  document.querySelectorAll<HTMLInputElement | HTMLSelectElement>('[data-cpu-setting]').forEach((control) => on(control, 'change', () => {
    if (configLocked()) return;
    const config = readConfigForm();
    const key = control.dataset.cpuSetting;
    const cpu = { ...config.cpu };
    if (key === 'threads' || key === 'spinWaitUs') cpu[key] = Number(control.value);
    else if (key === 'pinToPerformanceCores' || key === 'profile') cpu[key] = (control as HTMLInputElement).checked;
    else return;
    setDraftConfig({ ...config, cpu });
    render();
  }));
  document.querySelectorAll<HTMLInputElement>('[data-model-speaker]').forEach((control) => on(control, 'change', () => {
    if (configLocked()) return;
    setDraftConfig({ ...readConfigForm(), speakerId: control.valueAsNumber });
    render();
  }));
  document.querySelectorAll<HTMLInputElement>('[data-backend-path]').forEach((control) => on(control, 'input', () => {
    const config = readConfigForm();
    const value = control.value.trim() || null;
    const key = control.dataset.backendPath;
    if (key === 'artifacts') {
      config.tensorrt.artifactsPath = value;
      config.tensorrtArtifactsPath = value;
    } else if (key === 'hubert') config.tensorrt.hubertSource = value;
    else if (key === 'rmvpe') config.tensorrt.rmvpeSource = value;
    else if (key === 'dmlSources') config.directmlSourcesPath = value;
    setDraftConfig(config);
    syncDraftControls();
  }));
  document.querySelectorAll<HTMLButtonElement>('[data-refresh-backends]').forEach((control) => on(control, 'click', () =>
    runAction(async () => {
      processingBackends = await call<ProcessingBackendCatalog>('list_processing_backends');
      return undefined;
    }, 'refresh-backends')));
  on(byId<HTMLButtonElement>('demo'), 'click', () =>
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
    }, 'test-block'));
  const sliderFields: ReadonlyArray<readonly [string, string, number]> = [
    ['pitch', 'pitchValue', 1],
    ['index', 'indexValue', 2],
    ['protect', 'protectValue', 2],
    ['rms', 'rmsValue', 2],
  ];
  for (const [id, outputId, digits] of sliderFields)
    on(byId<HTMLInputElement>(id), 'input', (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      const output = byId<HTMLOutputElement>(outputId);
      if (output) output.value = value.toFixed(digits) + (id === 'pitch' ? ' st' : '');
      setDraftConfig(readConfigForm());
      syncDraftControls();
    });
  on(byId<HTMLInputElement>('autoPitchEnabled'), 'change', () => {
    setDraftConfig(readConfigForm());
    render();
  });
  on(byId<HTMLInputElement>('autoPitchTarget'), 'change', () => {
    setDraftConfig(readConfigForm());
    syncDraftControls();
  });
  document.querySelectorAll<HTMLButtonElement>('[data-pitch-target]').forEach((item) =>
    on(item, 'click', () => {
      const target = byId<HTMLInputElement>('autoPitchTarget');
      if (target) target.value = item.dataset.pitchTarget!;
      setDraftConfig(readConfigForm());
      render();
    }),
  );
  ['modelDirectory', 'inputDevice', 'outputDevice'].forEach((id) =>
    on(byId<HTMLInputElement | HTMLSelectElement>(id), 'change', () => {
      setDraftConfig(readConfigForm());
      syncDraftControls();
    }),
  );
  document.querySelectorAll<HTMLInputElement>('[data-session-profile]').forEach((item) =>
    on(item, 'change', () => {
      setDraftConfig(readConfigForm(), false);
      render();
    }),
  );
  ['modelDirectory', 'settings-modelDirectory'].forEach((id) =>
    on(byId<HTMLInputElement>(id), 'input', () => {
      setDraftConfig(readConfigForm());
      syncDraftControls();
    }),
  );
  for (const event of ['input', 'change'] as const)
    on(byId<HTMLInputElement>('modelPath'), event, (event) => {
      selectedModelPath = (event.target as HTMLInputElement).value;
      inspectSelectedModel();
      syncDraftControls();
    });
  ['settings-modelDirectory', 'settings-inputDevice', 'settings-outputDevice'].forEach((id) =>
    on(byId<HTMLInputElement | HTMLSelectElement>(id), 'change', () => {
      setDraftConfig(readConfigForm());
      render();
    }),
  );
  document.querySelectorAll<HTMLButtonElement>('[data-section]').forEach((item) =>
    on(item, 'click', () => {
      const route = item.dataset.section;
      if (route && (route === 'settings' || isWorkspaceRoute(route))) navigateTo(route as ViewRoute);
    }),
  );
  document.querySelectorAll<HTMLButtonElement>('[data-route]').forEach((item) =>
    on(item, 'click', () => {
      const route = item.dataset.route;
      if (route && (route === 'settings' || isWorkspaceRoute(route))) navigateTo(route as ViewRoute);
    }),
  );
  const cycleTheme = () => {
    setThemePreference(themePreference === 'system' ? 'dark' : themePreference === 'dark' ? 'light' : 'system');
  };
  on(byId<HTMLButtonElement>('themeButton'), 'click', cycleTheme);
  on(byId<HTMLButtonElement>('settingsTheme'), 'click', cycleTheme);
  on(byId<HTMLButtonElement>('settingsStudio'), 'click', () => navigateTo('studio'));
  on(byId<HTMLButtonElement>('openInspect'), 'click', openInspector);
  on(byId<HTMLButtonElement>('minimizeWindow'), 'click', () => void windowCommand('minimize'));
  on(byId<HTMLButtonElement>('maximizeWindow'), 'click', () => void windowCommand('maximize'));
  on(byId<HTMLButtonElement>('closeWindow'), 'click', () => void windowCommand('close'));
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
  if (isSplashWindow()) {
    if (!runningInTauri()) return;
    const events = new Channel<BackendEvent>();
    events.onmessage = event => {
      if (event.type === 'boot') {
        bootStatus = bootGate.observe(event.payload as BootStatus);
        scheduleRender();
      }
    };
    try {
      bootStatus = bootGate.observe(await call<BootStatus>('subscribe_boot', { windowLabel, channel: events }));
      render();
    } catch { /* Keep startup feedback visible if the IPC bridge is not ready yet. */ }
    return;
  }
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
      // Import has its own job/revision ordering. A later snapshot delivered by
      // another publisher must not hide a terminal import event arriving late.
      if (event.type === 'modelImport') modelManagement.observe(event.payload as ModelImportStatus);
      if (event.type === 'modelDependency') {
        modelDependencyStatus = event.payload as ModelDependencyStatus;
        scheduleRender();
      }
      if (event.sequence <= lastEventSequence) return;
      if (lastEventSequence !== 0 && event.sequence > lastEventSequence + 1) void resyncSnapshot();
      lastEventSequence = event.sequence;
      if (event.type === 'boot') bootStatus = bootGate.observe(event.payload as BootStatus);
      if (event.type === 'snapshot') {
        acceptSnapshot(event.payload as Snapshot);
        if (!snapshot?.lastError) localError = null;
      }
      if (event.type === 'preparation') acceptPreparation(event.payload as PreparationStatus);
      if (event.type === 'error') {
        localError = event.payload as ErrorReport;
        errorDialogOpen = !localError.recoverable;
      }
      scheduleRender();
    };
    await initializeWorkspace({
      subscribe: async () => {
        // Register this Channel once; another registration drops its first sender.
        acceptPreparation(await call<PreparationStatus>('subscribe_preparation', { windowLabel, channel: events }));
      },
      waitForShell: async () => {
        bootStatus = await bootGate.wait(await call<BootStatus>('get_boot_status'));
      },
      snapshot: () => call<Snapshot>('get_snapshot'),
      resources: () => call<StartupResources | null>('get_startup_resources'),
      present: (initial, resources) => {
        acceptSnapshot(initial);
        startupResources = resources;
        channelConnected = true;
        void modelManagement.initialize();
        if (resources) {
          processingBackends = resources.processingBackends;
          devices = resources.audioDevices;
          if (resources.savedModel && samePath(modelSelectionPath(), resources.savedModel.path))
            modelInspection = { path: resources.savedModel.path, selectionPath: resources.savedModel.path, components: resources.savedModel.components };
        }
        performance.mark('vocal-workspace-state-ready');
        render();
      },
      afterPaint: async () => {
        if (windowLabel !== 'main') return false;
        const nativeWindow = getCurrentWindow();
        await waitForVisible({ isVisible: () => nativeWindow.isVisible(), observe: callback => nativeWindow.onFocusChanged(callback) });
        await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
        performance.mark('vocal-workspace-first-paint');
        return true;
      },
      restore: async revision => {
        performance.mark('vocal-model-restore-requested');
        await runAction(() => call<Snapshot>('restore_saved_model', { expectedRevision: revision }), 'restore-model');
        performance.mark(snapshot?.engine ? 'vocal-model-restore-ready' : 'vocal-model-restore-failed');
        performance.measure('vocal-model-restore-duration', 'vocal-model-restore-requested');
      },
      refreshLibrary: async () => {
        if (!startupResources) {
          [processingBackends, devices] = await Promise.all([
            call<ProcessingBackendCatalog>('list_processing_backends'), call<AudioDeviceCatalog>('list_audio_devices'),
          ]);
        }
        const directory = snapshot?.config.modelDirectory;
        if (directory) {
          const catalog = await call<ModelCatalog>('list_models', { path: directory }).catch(() => null);
          if (samePath(directory, snapshot?.config.modelDirectory)) modelCatalog = catalog;
        }
        if (!selectedComponents()) inspectSelectedModel();
        render();
      },
      report: error => { localError = normalizeError(error); render(); },
    });
  } catch (error) {
    localError = normalizeError(error);
  }
  render();
}
async function resyncSnapshot(): Promise<void> {
  if (resyncInFlight) return;
  resyncInFlight = true;
  try {
    acceptSnapshot(await call<Snapshot>('get_snapshot'));
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
    const button = document.querySelector<HTMLButtonElement>(snapshot?.lifecycle === 'running' ? '#stop' : '#start');
    if (button && !button.disabled) button.click();
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
  if (isSplashWindow()) {
    const clock = document.querySelector('.startup-elapsed');
    if (clock) clock.textContent = `${Math.floor((performance.now() - splashStartedAt) / 1000)} s`;
    return;
  }
  if (activeSection === 'diagnostics' && document.visibilityState === 'visible' && snapshot?.audio.active
    && diagnosticsHistory.state(true, channelConnected, performance.now()) === 'stale') render();
  if (document.visibilityState !== 'visible' || !channelConnected || busy || automaticSaveRequest || refreshInFlight) return;
  // Recover a missed terminal event in an auxiliary window. At most one
  // snapshot is outstanding; the local timer stays independent of its reply.
  if (preparationActive(preparationStatus) && performance.now() - preparationObservedAt < 5000) return;
  refreshInFlight = true;
  try {
    acceptSnapshot(await call<Snapshot>('get_snapshot'));
    render();
  } catch {
    /* Keep the last usable snapshot. */
  } finally {
    refreshInFlight = false;
  }
}, 1000);
document.addEventListener('visibilitychange', () => {
  if (!isSplashWindow() && document.visibilityState === 'visible') void resyncSnapshot();
});
applyTheme();
render();
void syncWindowState();
void connect();
