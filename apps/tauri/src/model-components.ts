import type { ModelComponents } from './types';
import { displayPath, escapeHtml, formatBytes } from './format';

const COMPONENTS = [['synth', 'Synth'], ['hubert', 'HuBERT'], ['rmvpe', 'RMVPE'], ['retrieval', 'Index']] as const;

export function renderComponentBadges(components?: ModelComponents): string {
  if (!components) return '';
  return `<span class="model-component-badges">${COMPONENTS.map(([key, name]) => {
    const component = components[key];
    const status = component.present ? component.embedded ? 'embedded' : 'external' : key === 'retrieval' ? 'optional · absent' : 'missing';
    return `<span class="model-component-badge ${component.present ? 'present' : ''}" title="${escapeHtml(`${name}: ${status}${component.present ? ` · ${formatBytes(component.sizeBytes)}` : ''}`)}">${component.present ? '✓' : '—'} ${name}</span>`;
  }).join('')}</span>`;
}

export function renderComponentDetails(components: ModelComponents | null, error: string | null): string {
  if (!components) return error ? `<p class="model-inspection-note">${escapeHtml(error)}</p>` : '';
  return `<details class="model-component-details"><summary>Model contents ${renderComponentBadges(components)}</summary><div class="model-component-grid">${COMPONENTS.map(([key, name]) => {
    const component = components[key];
    return `<div><strong>${name}${key === 'retrieval' ? ' · optional' : ''}</strong><span>${component.present ? `${component.embedded ? 'Embedded' : 'External'} · ${formatBytes(component.sizeBytes)}${component.tensorCount ? ` · ${component.tensorCount} tensors` : ''}` : key === 'retrieval' ? 'No retrieval index' : 'Not included'}</span>${component.source ? `<small title="${escapeHtml(displayPath(component.source))}">${escapeHtml(displayPath(component.source))}</small>` : ''}</div>`;
  }).join('')}</div></details>`;
}
