import type { ConversionConfig } from './types';

/** Session calibration changes remain ordinary idle, automatically saved controls. */
export function renderPitchMatching(config: ConversionConfig, locked: boolean): string {
  const { enabled, targetHz } = config.autoPitch;
  return `<div class="pitch-matching ${enabled ? 'enabled' : ''}">
    <label class="pitch-match-toggle"><span><strong>Automatic pitch matching</strong><small>Calibrate your voice to the model's register</small></span><input id="autoPitchEnabled" type="checkbox" role="switch" aria-label="Automatic pitch matching" ${enabled ? 'checked' : ''} ${locked ? 'disabled' : ''}/></label>
    <div class="pitch-match-target"><label class="field-label"><span>Target register · Hz</span><input id="autoPitchTarget" type="number" min="50" max="1200" step="1" value="${targetHz}" ${locked || !enabled ? 'disabled' : ''}/></label><div class="pitch-match-presets"><button class="button ghost" data-pitch-target="155" ${locked || !enabled ? 'disabled' : ''}>155 · Lower</button><button class="button ghost" data-pitch-target="255" ${locked || !enabled ? 'disabled' : ''}>255 · Higher</button></div></div>
    <p>Matches once after voiced input, then stays steady until the next session. Additional pitch below is added to the match.</p>
  </div>`;
}
