import type { ModelComponents } from './types';
import { escapeHtml } from './format';
import { speakerIdMaximum, validateSpeakerId } from './speaker-policy';

/** One model-level control shared by the CPU, DirectML and TensorRT engines. */
export function renderSpeakerSettings(speakerId: number, components: ModelComponents | null, locked: boolean, prefix: string): string {
  const count = components?.speakerCount;
  const error = validateSpeakerId(speakerId, count);
  const id = `${prefix}speakerId`;
  const description = count === 1 ? 'Single-speaker model · use ID 0.'
    : count != null && count > 1 ? `${count} speakers · IDs 0–${speakerIdMaximum(count)}.`
      : 'IDs start at 0. The model loader checks the range when weights are available.';
  return `<div class="backend-source-grid"><label class="field-label" for="${escapeHtml(id)}">Speaker ID<input id="${escapeHtml(id)}" data-model-speaker type="number" min="0" max="${speakerIdMaximum(count)}" step="1" required value="${Number.isFinite(speakerId) ? speakerId : ''}" aria-invalid="${!!error}" aria-describedby="${escapeHtml(id)}-description" ${locked ? 'disabled' : ''}/></label><p id="${escapeHtml(id)}-description" class="backend-footnote" ${error ? 'role="alert"' : ''}>${escapeHtml(error ?? description)} Changes save automatically while audio is stopped.</p></div>`;
}
