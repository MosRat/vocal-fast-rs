export type Lifecycle = 'created' | 'loading' | 'ready' | 'running' | 'stopping' | 'stopped' | 'failed';
export type BootPhase = 'starting' | 'restoringConfig' | 'initializingIo' | 'detectingDevices' | 'ready' | 'failed' | 'shuttingDown';
export type Backend = 'passthrough' | 'rvcCpu' | 'tensorRtRtx' | 'directMl';
export interface ProcessingDevice {
  id: string;
  name: string;
  vendorId: number;
  dedicatedMemoryBytes: number;
  sharedMemoryBytes: number;
  available: boolean;
  reason: string | null;
  kind: 'gpu' | 'virtual' | 'remote' | 'software';
}
export interface ProcessingBackendCatalog {
  recommended: Backend;
  cpu?: { logicalProcessors: number; recommendedThreads: number; operatorVersion: string | null };
  backends: Array<{
    backend: Backend;
    available: boolean;
    reason: string | null;
    device: string | null;
    description: string;
    devices: ProcessingDevice[];
    recommendedDeviceId: string | null;
  }>;
}
export type ConfigUpdateKind =
  | 'unchanged'
  | 'hotReload'
  | 'rebuildEngine'
  | 'reloadTensorRtPipeline'
  | 'planBuildRequired'
  | 'loadModelRequired';
export type ThemePreference = 'system' | 'light' | 'dark';
export type WorkspaceRoute = 'studio' | 'audio' | 'tuning' | 'diagnostics' | 'models';

export const WORKSPACE_ROUTES: readonly WorkspaceRoute[] = ['studio', 'audio', 'tuning', 'diagnostics', 'models'];

export function isWorkspaceRoute(value: string): value is WorkspaceRoute {
  return WORKSPACE_ROUTES.includes(value as WorkspaceRoute);
}

export interface ConversionConfig {
  pitchSemitones: number;
  autoPitch: { enabled: boolean; targetHz: number };
  indexRatio: number;
  protect: number;
  rmsMixRate: number;
}

export interface TensorRtPipelineConfig {
  artifactsPath: string | null;
  hubertSource: string | null;
  rmvpeSource: string | null;
  cacheDirectory: string | null;
  profile: string;
  windowSamples: number | null;
  precision: string;
  hubertPrecision: string;
  rmvpePrecision: string;
  synthPrecision: string;
  graphMode: string;
  cudaGraph: boolean;
  nativeCudaGraph: boolean;
  graphCacheSize: number;
  parallelAcoustic: boolean;
  asyncAudioD2h: boolean;
  rmvpeFft: string;
  synthFullFp16: boolean;
}

export interface GuiConfig {
  schemaVersion: number;
  inputFormat: { sampleRate: number; channels: number };
  outputFormat: { sampleRate: number; channels: number };
  conversion: ConversionConfig;
  backend: Backend;
  backendAuto: boolean;
  speakerId: number;
  modelPath: string | null;
  modelDirectory: string | null;
  inputDevice: string | null;
  outputDevice: string | null;
  blockFrames: number;
  queueBlocks: number;
  tensorrtArtifactsPath: string | null;
  tensorrtWindowSamples: number | null;
  tensorrt: TensorRtPipelineConfig;
  cpu: CpuExecutionConfig;
  directmlSourcesPath: string | null;
  directmlDeviceId: string | null;
}

export interface CpuExecutionConfig {
  threads: number;
  spinWaitUs: number;
  pinToPerformanceCores: boolean;
  profile: boolean;
}

export interface AudioDeviceInfo {
  id: string;
  name: string;
  input: boolean;
  output: boolean;
  defaultInput: boolean;
  defaultOutput: boolean;
  sampleRate: number | null;
  channels: number | null;
}

export interface AudioDeviceCatalog {
  host: string;
  devices: AudioDeviceInfo[];
}

export type ModelArtifactKind = 'vfrm' | 'manifest';

export interface ModelComponent {
  present: boolean;
  embedded: boolean;
  tensorCount: number;
  sizeBytes: number;
  source: string | null;
}
export interface ModelComponents {
  synth: ModelComponent;
  speakerCount?: number | null;
  hubert: ModelComponent;
  rmvpe: ModelComponent;
  retrieval: ModelComponent;
  automaticTensorRt: boolean;
}
export interface ModelArtifactInspection {
  path: string;
  components: ModelComponents;
}

export interface ModelCatalogEntry {
  path: string;
  kind: ModelArtifactKind;
  sizeBytes: number;
  modifiedAtMs: number;
  components: ModelComponents;
  id: string;
  displayName: string;
  architecture: string;
  version: string;
  sampleRate: number;
  featureDims: number;
  usesF0: boolean;
}

export interface ModelCatalogIssue {
  path: string;
  message: string;
}

export interface ModelCatalog {
  root: string;
  entries: ModelCatalogEntry[];
  issues: ModelCatalogIssue[];
}

export interface AudioTelemetry {
  active: boolean;
  state: Lifecycle;
  inputDevice: string | null;
  outputDevice: string | null;
  sessionId: string | null;
  sample: AudioSample | null;
  streamState: string;
  route: AudioRouteInfo | null;
}

export interface DirectionTelemetry {
  queueDropsPerSecond: number;
  staleDropsPerSecond: number;
  queuePeakBlocks: number;
  callbackPeriodMs: number | null;
  callbackJitterP95Ms: number | null;
  endpointPaddingP95Ms: number | null;
  streamLatencyMs: number | null;
  clockDriftPpm: number | null;
}

export interface AudioSample {
  sequence: number;
  elapsedMs: number;
  windowMs: number;
  inputFramesPerSecond: number;
  outputFramesPerSecond: number;
  processingMs: number | null;
  blockBudgetMs: number | null;
  processingLoadPercent: number | null;
  workerFailuresPerSecond: number;
  underflowMsPerSecond: number;
  audioAgeMs: number | null;
  audioAgeP50Ms: number | null;
  audioAgeP95Ms: number | null;
  audioAgeP99Ms: number | null;
  capture: DirectionTelemetry;
  render: DirectionTelemetry;
}

export interface AudioRouteInfo {
  backend: 'nativeWasapi' | 'cpal';
  processingBatchFrames: number;
  playbackPrefillMicros: number;
  fallbackReason: string | null;
  captureFormat: GuiConfig['inputFormat'];
  renderFormat: GuiConfig['outputFormat'];
  processingFormat: GuiConfig['inputFormat'];
  conversionDelayMicros: number;
}

export interface ErrorReport {
  code: string;
  message: string;
  recoverable: boolean;
  details?: AudioStartDetails;
}

export interface AudioStartDetails {
  kind: 'deviceBusy' | 'deviceUnavailable' | 'permissionDenied' | 'serviceUnavailable' | 'unsupportedFormat' | 'initialization';
  message: string;
  backend: string;
  direction: string | null;
  deviceId: string | null;
  deviceName: string | null;
  hresult: string | null;
  technicalDetails: string[];
  suggestions: string[];
}

export interface BootStatus {
  revision: number;
  phase: BootPhase;
  message: string | null;
  elapsedMs: number;
  recoverable: boolean;
}

export interface Snapshot {
  boot: BootStatus;
  lifecycle: Lifecycle;
  status: {
    session: Lifecycle;
    model: Lifecycle;
    engine: Lifecycle;
    audio: Lifecycle;
  };
  config: GuiConfig;
  configRevision: number;
  lastConfigUpdate: ConfigUpdateKind;
  model: {
    id: string;
    displayName: string;
    architecture: string;
    version: string;
    sampleRate: number;
    featureDims: number;
    usesF0: boolean;
  } | null;
  engine: {
    state: Lifecycle;
    lastError: string | null;
  } | null;
  audio: AudioTelemetry;
  tensorrtPipeline: TensorRtPipelineStatus;
  preparation: PreparationStatus;
  lastError: ErrorReport | null;
  modelRestorePending: boolean;
}

/** Metadata only; savedModel is a selection, never a loaded engine. */
export interface StartupResources {
  audioDevices: AudioDeviceCatalog | null;
  processingBackends: ProcessingBackendCatalog;
  savedModel: { path: string; summary: NonNullable<Snapshot['model']>; components: ModelComponents } | null;
  notices: string[];
}

export type TensorRtPipelinePhase = 'idle' | 'checkingCache' | 'building' | 'ready' | 'failed';
export interface TensorRtPipelineStatus {
  phase: TensorRtPipelinePhase;
  message: string | null;
  artifactsPath: string | null;
  cacheHit: boolean;
  elapsedMs: number;
}

export interface BackendEvent {
  type: 'boot' | 'snapshot' | 'error' | 'preparation' | 'modelImport';
  sequence: number;
  payload: BootStatus | Snapshot | ErrorReport | PreparationStatus | import('./model-library-types').ModelImportStatus;
}

export type PreparationPhase = 'idle' | 'inspecting' | 'extracting' | 'loadingWeights' | 'checkingCache' | 'building' | 'publishing' | 'loadingEngine' | 'ready' | 'failed';
export interface PreparationStatus {
  operationId: number;
  revision: number;
  kind: 'loadModel' | 'prepareTensorRt' | 'updateConfig' | null;
  phase: PreparationPhase;
  message: string;
  elapsedMs: number;
  completedSteps: number;
  totalSteps: number | null;
  cacheHit: boolean;
}

export function defaultConfig(): GuiConfig {
  return {
    schemaVersion: 1,
    inputFormat: { sampleRate: 48000, channels: 1 },
    outputFormat: { sampleRate: 48000, channels: 1 },
    conversion: {
      pitchSemitones: 0,
      autoPitch: { enabled: true, targetHz: 255 },
      indexRatio: 0,
      protect: 0.33,
      rmsMixRate: 1,
    },
    backend: 'passthrough',
    backendAuto: true,
    speakerId: 0,
    modelPath: null,
    modelDirectory: null,
    inputDevice: null,
    outputDevice: null,
    blockFrames: 480,
    queueBlocks: 3,
    tensorrtArtifactsPath: null,
    tensorrtWindowSamples: null,
    directmlSourcesPath: null,
    directmlDeviceId: null,
    cpu: { threads: 0, spinWaitUs: 0, pinToPerformanceCores: false, profile: false },
    tensorrt: {
      artifactsPath: null,
      hubertSource: null,
      rmvpeSource: null,
      cacheDirectory: null,
      profile: 'reference',
      windowSamples: null,
      precision: 'fp32',
      hubertPrecision: 'fp16',
      rmvpePrecision: 'fp32',
      synthPrecision: 'fp16',
      graphMode: 'native_subgraphs',
      cudaGraph: true,
      nativeCudaGraph: true,
      graphCacheSize: 2,
      parallelAcoustic: false,
      asyncAudioD2h: true,
      rmvpeFft: 'cufft',
      synthFullFp16: false,
    },
  };
}
