export type Lifecycle = 'created' | 'loading' | 'ready' | 'running' | 'stopping' | 'stopped' | 'failed';
export type BootPhase = 'starting' | 'restoringConfig' | 'restoringModel' | 'ready' | 'failed' | 'shuttingDown';
export type Backend = 'passthrough' | 'rvcCpu' | 'tensorRtRtx';
export type ConfigUpdateKind =
  | 'unchanged'
  | 'hotReload'
  | 'rebuildEngine'
  | 'reloadTensorRtPipeline'
  | 'planBuildRequired'
  | 'loadModelRequired';
export type ThemePreference = 'system' | 'light' | 'dark';
export type WorkspaceRoute = 'studio' | 'audio' | 'tuning' | 'diagnostics';

export const WORKSPACE_ROUTES: readonly WorkspaceRoute[] = ['studio', 'audio', 'tuning', 'diagnostics'];

export function isWorkspaceRoute(value: string): value is WorkspaceRoute {
  return WORKSPACE_ROUTES.includes(value as WorkspaceRoute);
}

export interface ConversionConfig {
  pitchSemitones: number;
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
  modelPath: string | null;
  modelDirectory: string | null;
  inputDevice: string | null;
  outputDevice: string | null;
  blockFrames: number;
  queueBlocks: number;
  tensorrtArtifactsPath: string | null;
  tensorrtWindowSamples: number | null;
  tensorrt: TensorRtPipelineConfig;
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

export interface ModelCatalogEntry {
  path: string;
  kind: ModelArtifactKind;
  sizeBytes: number;
  modifiedAtMs: number;
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
  inputFrames: number;
  outputFrames: number;
  processingMicros: number;
  failures: number;
  underflowSamples: number;
  queueOverflows: number;
  latencyP95Micros: number | null;
  streamState: string;
}

export interface ErrorReport {
  code: string;
  message: string;
  recoverable: boolean;
}

export interface BootStatus {
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
  metrics: {
    inputFrames: number;
    processingMicros: number;
    outputFrames: number;
  };
  engine: {
    state: Lifecycle;
    processedChunks: number;
    failures: number;
    lastError: string | null;
  } | null;
  audio: AudioTelemetry;
  tensorrtPipeline: TensorRtPipelineStatus;
  lastError: ErrorReport | null;
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
  type: 'snapshot' | 'metrics' | 'error';
  sequence: number;
  payload: Snapshot | Snapshot['metrics'] | ErrorReport;
}

export function defaultConfig(): GuiConfig {
  return {
    schemaVersion: 1,
    inputFormat: { sampleRate: 48000, channels: 1 },
    outputFormat: { sampleRate: 48000, channels: 1 },
    conversion: {
      pitchSemitones: 0,
      indexRatio: 0,
      protect: 0.33,
      rmsMixRate: 1,
    },
    backend: 'passthrough',
    modelPath: null,
    modelDirectory: null,
    inputDevice: null,
    outputDevice: null,
    blockFrames: 480,
    queueBlocks: 3,
    tensorrtArtifactsPath: null,
    tensorrtWindowSamples: null,
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
