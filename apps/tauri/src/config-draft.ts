import type { GuiConfig, Snapshot } from './types';

export function sameConfig(left: GuiConfig, right: GuiConfig): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

/** Profile changes are committed only by Save profile; model paths belong to Load. */
export function automaticConfig(draft: GuiConfig, saved: GuiConfig): GuiConfig {
  return { ...draft, modelPath: saved.modelPath, tensorrt: { ...draft.tensorrt, profile: saved.tensorrt.profile } };
}

/** Merge disjoint edits from other windows without overwriting either window's edits. */
export function rebaseConfigDraft(draft: GuiConfig, base: GuiConfig, saved: GuiConfig): { config: GuiConfig; conflicts: string[] } {
  const conflicts: string[] = [];
  function merge(local: unknown, previous: unknown, remote: unknown, path: string): unknown {
    if (JSON.stringify(local) === JSON.stringify(previous) || JSON.stringify(local) === JSON.stringify(remote)) return remote;
    if (JSON.stringify(previous) === JSON.stringify(remote)) return local;
    if (local && previous && remote && typeof local === 'object' && typeof previous === 'object' && typeof remote === 'object') {
      const result: Record<string, unknown> = {};
      for (const key of Object.keys(remote)) {
        result[key] = merge((local as Record<string, unknown>)[key], (previous as Record<string, unknown>)[key], (remote as Record<string, unknown>)[key], path ? `${path}.${key}` : key);
      }
      return result;
    }
    conflicts.push(path);
    return local;
  }
  const config = merge(draft, base, saved, '') as GuiConfig;
  // Selecting or unloading a model is an authoritative runtime operation.
  config.modelPath = saved.modelPath;
  return { config, conflicts: conflicts.filter(path => path !== 'modelPath') };
}

/** Preserve edits when a model operation changes only the persisted model path. */
export function reconcileConfigDraft(
  draft: GuiConfig,
  base: Snapshot,
  next: Snapshot,
  revision: number | null,
  alreadyConflicted: boolean,
  backendChange = false,
): { config: GuiConfig; revision: number | null; conflict: boolean } {
  const settings = (config: GuiConfig) => backendChange
    ? { ...config, modelPath: null, backend: null, backendAuto: null, directmlSourcesPath: null, directmlDeviceId: null, tensorrtArtifactsPath: null, tensorrt: { ...config.tensorrt, artifactsPath: null, hubertSource: null, rmvpeSource: null } }
    : { ...config, modelPath: null };
  const unchangedSettings = JSON.stringify(settings(base.config)) === JSON.stringify(settings(next.config));
  const delta = next.configRevision - base.configRevision;
  const canRebase = unchangedSettings && delta >= 0 && delta <= 1
    && (revision === base.configRevision || revision === next.configRevision);
  return {
    config: { ...draft, modelPath: next.config.modelPath, ...(backendChange ? {
      backend: next.config.backend,
      backendAuto: next.config.backendAuto,
      directmlSourcesPath: next.config.directmlSourcesPath,
      directmlDeviceId: next.config.directmlDeviceId,
      tensorrtArtifactsPath: next.config.tensorrtArtifactsPath,
      tensorrt: { ...draft.tensorrt, artifactsPath: next.config.tensorrt.artifactsPath, hubertSource: next.config.tensorrt.hubertSource, rmvpeSource: next.config.tensorrt.rmvpeSource },
    } : {}) },
    revision: canRebase ? next.configRevision : revision,
    conflict: alreadyConflicted || !canRebase,
  };
}
