import type { AudioStartDetails, ErrorReport } from './types';

const AUDIO_CODES: Record<AudioStartDetails['kind'], string> = {
  deviceBusy: 'audio_device_busy', deviceUnavailable: 'audio_device_unavailable',
  permissionDenied: 'audio_permission_denied', serviceUnavailable: 'audio_service_unavailable',
  unsupportedFormat: 'audio_format_unsupported', initialization: 'audio_start_failed',
};

/** Command rejections and Channel reports use the same user-facing shape. */
export function normalizeError(error: unknown): ErrorReport {
  if (typeof error === 'object' && error !== null && 'kind' in error) {
    const typed = error as { kind: string; details?: unknown };
    if (typed.kind === 'config_update') {
      const cause = normalizeError(typed.details);
      return { code: 'config_update_failed', message: `Settings update failed; previous settings remain active: ${cause.message}`, recoverable: true };
    }
    if (typed.kind === 'audio_startup' && typeof typed.details === 'object' && typed.details !== null && 'kind' in typed.details) {
      const details = typed.details as AudioStartDetails;
      return { code: AUDIO_CODES[details.kind] ?? 'audio_start_failed', message: details.message, recoverable: true, details };
    }
    return { code: typed.kind, message: typeof typed.details === 'string' ? typed.details : JSON.stringify(typed.details ?? typed.kind), recoverable: true };
  }
  return { code: 'command_error', message: String(error), recoverable: true };
}

export function errorTitle(report: ErrorReport): string {
  const names: Record<string, string> = {
    config_update_failed: 'Settings update failed',
    audio_device_busy: 'Audio device in use', audio_device_unavailable: 'Audio device unavailable',
    audio_permission_denied: 'Microphone access denied', audio_service_unavailable: 'Windows audio unavailable',
    audio_format_unsupported: 'Audio format unavailable', audio_start_failed: 'Could not start audio',
  };
  return names[report.code] ?? report.code;
}
