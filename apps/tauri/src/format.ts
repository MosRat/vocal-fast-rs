import type { Lifecycle, ThemePreference } from './types';

export function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
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
