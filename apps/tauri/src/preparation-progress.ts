import type { PreparationPhase, PreparationStatus } from './types';
import { escapeHtml } from './format';
import { icon } from './icons';
import './preparation-progress.css';

const PHASE_NAMES: Record<PreparationPhase, string> = {
  idle: 'Standby', inspecting: 'Inspecting model', extracting: 'Preparing acoustic sources',
  loadingWeights: 'Reading model weights', checkingCache: 'Checking plan cache',
  building: 'Building TensorRT plans', publishing: 'Publishing cache',
  loadingEngine: 'Loading processing engine', ready: 'Preparation complete', failed: 'Preparation failed',
};

let observed: { id: number; elapsed: number; at: number } | null = null;

export function preparationActive(status: PreparationStatus | null | undefined): boolean {
  return !!status && status.phase !== 'idle' && status.phase !== 'ready' && status.phase !== 'failed';
}

export function observePreparation(status: PreparationStatus | undefined): void {
  if (!status) return;
  if (!observed || observed.id !== status.operationId || observed.elapsed !== status.elapsedMs)
    observed = { id: status.operationId, elapsed: status.elapsedMs, at: performance.now() };
}

function elapsed(status: PreparationStatus): number {
  return status.elapsedMs + (preparationActive(status) && observed?.id === status.operationId ? Math.max(0, performance.now() - observed.at) : 0);
}

export function renderPreparationProgress(status: PreparationStatus | null | undefined, prefix = ''): string {
  if (!status || status.phase === 'idle') return '';
  const active = preparationActive(status);
  const counted = active && !!status.totalSteps;
  const percent = counted ? Math.min(100, Math.max(0, status.completedSteps / status.totalSteps! * 100)) : 0;
  return `<section class="preparation-progress ${active ? 'is-active' : status.phase === 'failed' ? 'has-error' : 'is-ready'}" data-preparation-phase="${status.phase}" aria-label="Model preparation" aria-busy="${active}">
    <div class="preparation-heading"><span class="preparation-mark">${active ? '<span class="button-spinner" aria-hidden="true"></span>' : icon(status.phase === 'failed' ? 'alert' : 'check', 16)}</span><div><strong id="${prefix}preparation-title">${PHASE_NAMES[status.phase]}</strong>${status.cacheHit ? '<span class="preparation-cache">Cache hit</span>' : ''}<p role="status" aria-live="polite" aria-atomic="true">${escapeHtml(status.message)}</p></div><span class="preparation-time" data-progress-elapsed data-elapsed="${elapsed(status)}" data-clock-start="${performance.now()}" data-clock-active="${active}" aria-label="Elapsed time">${(elapsed(status) / 1000).toFixed(1)} s</span></div>
    ${active ? `<div class="preparation-track ${counted ? 'is-counted' : 'is-indeterminate'}" role="progressbar" aria-labelledby="${prefix}preparation-title" aria-valuemin="0" aria-valuemax="${counted ? status.totalSteps : 100}" ${counted ? `aria-valuenow="${status.completedSteps}" aria-valuetext="${status.completedSteps} of ${status.totalSteps} plans complete"` : 'aria-valuetext="Working"'}><span style="${counted ? `width:${percent}%` : ''}"></span></div><div class="preparation-footnote"><span>${counted ? `${status.completedSteps} / ${status.totalSteps} plans complete` : status.phase === 'loadingEngine' ? 'Creating execution contexts and preparing device memory' : status.phase === 'checkingCache' && status.cacheHit ? 'Cached plans found · compilation skipped' : 'Working · this window remains responsive'}</span><span>${status.phase === 'building' ? 'First builds may take several minutes' : 'Audio starts after preparation'}</span></div>` : ''}
  </section>`;
}

// Update only the timer text; do not rerender forms or query a busy runtime actor.
window.setInterval(() => {
  if (document.visibilityState !== 'visible') return;
  document.querySelectorAll<HTMLElement>('[data-progress-elapsed][data-clock-active="true"]').forEach(node => {
    node.textContent = `${((Number(node.dataset.elapsed) + Math.max(0, performance.now() - Number(node.dataset.clockStart))) / 1000).toFixed(1)} s`;
  });
}, 200);
