import './save-status.css';
import { escapeHtml } from './format';
import { icon, type IconName } from './icons';

export interface SaveStatus {
  connected: boolean;
  conflict: boolean;
  error: string | null;
  pending: boolean;
  saving: boolean;
  profileDirty: boolean;
  locked: boolean;
  recentlySaved: boolean;
}

/** A single quiet status in the shell, independent of the current page. */
export function renderSaveStatus(status: SaveStatus): string {
  let state = 'idle';
  let label = 'Auto-save enabled';
  let detail = 'Settings save automatically while idle. Session profiles use Save profile.';
  let glyph: IconName = 'check';
  let action = '';

  if (status.conflict) {
    state = 'error'; label = 'Settings conflict'; glyph = 'alert';
    detail = 'Settings changed in another window. Reload to resolve overlapping edits.';
    action = '<button class="config-save-action" data-reload-config aria-label="Reload settings" title="Reload settings and discard conflicting edits">' + icon('refresh', 13) + '<span>Reload</span></button>';
  } else if (status.error) {
    state = 'error'; label = 'Save failed'; glyph = 'alert'; detail = status.error;
    action = '<button class="config-save-action" data-retry-autosave aria-label="Retry saving settings" title="Retry saving settings">' + icon('refresh', 13) + '<span>Retry</span></button>';
  } else if (status.saving) {
    state = 'saving'; label = 'Saving settings…'; glyph = 'refresh';
    detail = 'Your changes are being saved.';
  } else if (status.pending) {
    state = 'pending'; label = 'Changes pending'; glyph = 'sliders';
    detail = 'Your changes will save automatically when the runtime is idle.';
  } else if (status.profileDirty) {
    state = 'profile'; label = 'Profile not saved'; glyph = 'sliders';
    detail = 'Click Save profile to apply your selected session preset.';
  } else if (!status.connected) {
    label = 'Preparing settings'; detail = 'Connecting to the runtime configuration.';
  } else if (status.locked) {
    label = 'Settings locked'; glyph = 'sliders';
    detail = 'Stop audio or wait for model preparation to finish before changing settings.';
  } else if (status.recentlySaved) {
    state = 'saved'; label = 'Settings saved'; detail = 'Your latest settings have been saved.';
  }

  return `<div id="config-save-status" class="config-persistence is-${state}" data-save-state="${state}" title="${escapeHtml(label + '. ' + detail)}"><span class="config-save-icon" aria-hidden="true">${icon(glyph, 15)}</span><span class="config-save-label" role="status" aria-live="polite" aria-atomic="true">${label}</span>${action}</div>`;
}
