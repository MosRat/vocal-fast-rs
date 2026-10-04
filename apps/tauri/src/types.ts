export type Lifecycle = 'created' | 'loading' | 'ready' | 'running' | 'stopping' | 'stopped' | 'failed';
export type Backend = 'passthrough' | 'rvcCpu';
export type ThemePreference = 'system' | 'light' | 'dark';

export interface ConversionConfig {
  pitchSemitones: number;
  indexRatio: number;
  protect: number;
  rmsMixRate: number;
}

export interface GuiConfig {
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

export interface Snapshot {
  lifecycle: Lifecycle;
  status: {
    session: Lifecycle;
    model: Lifecycle;
    engine: Lifecycle;
    audio: Lifecycle;
  };
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
  lastError: ErrorReport | null;
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
    inputDevice: null,
    outputDevice: null,
    blockFrames: 480,
    queueBlocks: 3,
  };
}
