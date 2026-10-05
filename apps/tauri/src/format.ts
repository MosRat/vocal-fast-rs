import type { ConfigUpdateKind, Lifecycle, ThemePreference } from './types';

export function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value < 1024) return `${Math.max(0, value)} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let size = value;
  let unit = units[0];
  for (const candidate of units) {
    size /= 1024;
    unit = candidate;
    if (size < 1024 || candidate === units[units.length - 1]) break;
  }
  return `${size.toFixed(size >= 100 ? 0 : size >= 10 ? 1 : 2)} ${unit}`;
}

export function lifecycleLabel(value: Lifecycle): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function statusTone(value: Lifecycle): string {
  return value === 'running' || value === 'ready'
    ? 'positive'
    : value === 'failed'
      ? 'critical'
      : value === 'loading' || value === 'stopping'
        ? 'progress'
        : 'neutral';
}

export function themeLabel(preference: ThemePreference): string {
  return preference === 'system' ? 'System theme' : preference === 'dark' ? 'Dark theme' : 'Light theme';
}

export function configUpdateLabel(value: ConfigUpdateKind): string {
  switch (value) {
    case 'hotReload': return 'Idle settings applied';
    case 'rebuildEngine': return 'Engine rebuilt';
    case 'reloadTensorRtPipeline': return 'TensorRT pipeline reloaded';
    case 'planBuildRequired': return 'TensorRT build required';
    case 'loadModelRequired': return 'Load model to apply';
    default: return 'No engine change';
  }
}
