import type { ModelArtifactInspection } from './types';

export interface ModelManifest {
  id: string; display_name: string; architecture: string; version: string;
  sample_rate: number; model_file: string; index_file: string | null; source: string; notes: string | null;
}
export interface ModelMetadataEdit {
  displayName: string; source: string; notes: string | null;
  author: string; language: string; license: string; tags: string[];
}
export interface ModelMetadataView {
  inspection: ModelArtifactInspection & { manifest: ModelManifest };
  revision: string; editable: boolean; fields: ModelMetadataEdit;
}
export type ModelProvider = 'huggingFace' | 'hfMirror' | 'modelScope';
export interface RemoteRepository { provider: ModelProvider; repository: string; revision: string | null }
export interface RemoteFile { path: string; sizeBytes: number | null; sha256: string | null; kind: 'checkpoint' | 'vfrm' | 'index' | 'archive' }
export interface RemoteModelCatalog { repository: RemoteRepository; files: RemoteFile[]; totalFiles: number }
export type ImportPhase = 'idle' | 'queued' | 'listing' | 'downloading' | 'verifying' | 'converting' | 'bundling' | 'publishing' | 'ready' | 'cancelling' | 'cancelled' | 'failed';
export interface ModelImportStatus {
  jobId: number; revision: number; phase: ImportPhase; message: string; file: string | null;
  downloadedBytes: number; totalBytes: number | null; elapsedMs: number;
  outputPath: string | null; complete: boolean; warnings: string[];
  libraryDirectory?: string | null;
  error: { code: string; message: string; retryable: boolean } | null;
}
export function importActive(phase: ImportPhase): boolean {
  return !['idle', 'ready', 'failed', 'cancelled'].includes(phase);
}
export function observeImport(current: ModelImportStatus | null, next: ModelImportStatus): ModelImportStatus | null {
  if (current && (next.jobId < current.jobId || (next.jobId === current.jobId && next.revision <= current.revision))) return current;
  return next;
}
export function suggestIndex(model: string, files: RemoteFile[]): string {
  const name = model.split('/').pop()?.replace(/\.(pth|vfr|vfrm|zip)$/i, '').toLowerCase() ?? '';
  const indices = files.filter(f => f.kind === 'index');
  const exact = indices.find(f => f.path.split('/').pop()?.replace(/\.index$/i, '').toLowerCase() === name);
  return exact?.path ?? (indices.length === 1 ? indices[0].path : '');
}
