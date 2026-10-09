import type { BootStatus } from './types';
import { escapeHtml } from './format';
import { icon } from './icons';
import './startup.css';

/** Shell startup has no dependency on weights, plan caches or GPU contexts. */
export function renderStartup(status: BootStatus | null, elapsedMs: number): string {
  const failed = status?.phase === 'failed';
  const active = !status || !['ready', 'failed', 'shuttingDown'].includes(status.phase);
  const message = status?.message ?? 'Initializing application settings and devices…';
  const label = failed ? 'Unable to start' : status?.phase === 'ready' ? 'Opening your studio' : 'Starting your studio';
  return `<main class="startup-surface ${failed ? 'has-error' : ''}" aria-busy="${active}" data-tauri-drag-region>
    <header class="startup-brand" data-tauri-drag-region><span class="startup-mark" aria-hidden="true">${icon('waveform', 24)}</span><div><strong>vocal-fast</strong><span>Realtime voice studio</span></div></header>
    <section class="startup-copy"><h1 id="startup-status">${label}</h1><p role="status" aria-live="polite" title="${escapeHtml(message)}">${escapeHtml(message)}</p></section>
    <footer class="startup-footer"><span>${failed ? icon('alert', 13) + 'Startup notice' : 'Preparing workspace &amp; devices'}</span><span class="startup-elapsed" aria-label="Elapsed time">${Math.floor(Math.max(0, elapsedMs) / 1000)} s</span></footer>
    ${active ? '<div class="startup-progress is-indeterminate" role="progressbar" aria-labelledby="startup-status" aria-valuetext="Starting"><span></span></div>' : ''}
  </main>`;
}
