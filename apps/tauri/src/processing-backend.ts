import type { Backend, GuiConfig, ModelComponents, ProcessingBackendCatalog, Snapshot } from './types';
import { displayPath, escapeHtml, samePath } from './format';
import { icon } from './icons';
import { preparationActive, renderPreparationProgress } from './preparation-progress';
import { selectedDirectMlDevice, renderDirectMlSettings } from './directml-settings';
import { renderCpuSettings } from './cpu-settings';
import { renderSpeakerSettings } from './speaker-settings';

export const BACKEND_NAMES: Record<Backend, string> = {
  tensorRtRtx: 'TensorRT RTX',
  directMl: 'DirectML',
  rvcCpu: 'RVC CPU · oneDNN',
  passthrough: 'Passthrough',
};

export function resolvedBackend(config: GuiConfig, catalog: ProcessingBackendCatalog | null): Backend {
  return config.backendAuto && catalog ? catalog.recommended : config.backend;
}

export function backendAvailable(config: GuiConfig, catalog: ProcessingBackendCatalog | null): boolean {
  const selected = resolvedBackend(config, catalog);
  return !!catalog?.backends.some((entry) => entry.backend === selected && entry.available)
    && (selected !== 'directMl' || !!selectedDirectMlDevice(config, catalog)?.available);
}

export function backendLoadSelection(config: GuiConfig, catalog: ProcessingBackendCatalog | null, expectedRevision: number) {
  return {
    expectedRevision,
    backend: resolvedBackend(config, catalog),
    automatic: config.backendAuto,
    artifactsPath: config.tensorrt.artifactsPath || config.tensorrtArtifactsPath || null,
    directmlSourcesPath: config.directmlSourcesPath,
    directmlDeviceId: config.directmlDeviceId,
  };
}

export function backendSelectionChanged(config: GuiConfig, persisted: GuiConfig, catalog: ProcessingBackendCatalog | null): boolean {
  return resolvedBackend(config, catalog) !== persisted.backend
    || config.directmlSourcesPath !== persisted.directmlSourcesPath
    || (resolvedBackend(config, catalog) === 'directMl' && config.directmlDeviceId !== persisted.directmlDeviceId)
    || config.backendAuto !== persisted.backendAuto
    || (config.tensorrt.artifactsPath || config.tensorrtArtifactsPath || null)
      !== (persisted.tensorrt.artifactsPath || persisted.tensorrtArtifactsPath || null);
}

/** Shared by Studio and the independently designed settings window. */
export function renderProcessingBackend(config: GuiConfig, catalog: ProcessingBackendCatalog | null, locked: boolean, prefix = '', components: ModelComponents | null = null, loaded: Snapshot | null = null, selectedPath = ''): string {
  const selected = resolvedBackend(config, catalog);
  const preparing = preparationActive(loaded?.preparation);
  const entry = catalog?.backends.find((item) => item.backend === selected);
  const available = backendAvailable(config, catalog);
  const dmlDevice = selected === 'directMl' ? selectedDirectMlDevice(config, catalog) : null;
  const options = catalog?.backends.map((item) => `<option value="${item.backend}" ${!config.backendAuto && config.backend === item.backend ? 'selected' : ''} ${item.available ? '' : 'disabled'} title="${escapeHtml(item.reason ?? item.description)}">${BACKEND_NAMES[item.backend]}${item.available ? '' : ' · unavailable'}</option>`).join('') ?? '';
  const gpu = selected === 'tensorRtRtx' && entry?.available;
  const artifacts = config.tensorrt.artifactsPath || config.tensorrtArtifactsPath;
  const automatic = !!components?.automaticTensorRt;
  const boundModel = loaded?.config.backend === 'tensorRtRtx' && !!loaded.engine && samePath(selectedPath, loaded.config.modelPath);
  const prepared = boundModel && loaded.lastConfigUpdate !== 'planBuildRequired' && loaded.lastConfigUpdate !== 'loadModelRequired';
  const source = (key: 'hubert' | 'rmvpe') => boundModel ? loaded.config.tensorrt[key === 'hubert' ? 'hubertSource' : 'rmvpeSource'] : components?.[key].source;
  const automaticPaths = automatic ? `<div class="backend-automatic"><strong>${icon('sparkle', 14)}Complete model · automatic plans</strong><p>Load extracts its acoustic weights and builds or reuses plans for the saved session profile. The first build can take several minutes.</p><label class="field-label">Pipeline manifest<input data-auto-path="artifacts" value="${escapeHtml(displayPath(prepared ? loaded.config.tensorrt.artifactsPath ?? '' : 'Generated automatically on Load'))}" readonly/></label><div class="backend-source-grid">${(['hubert', 'rmvpe'] as const).map(key => `<label class="field-label">${key === 'hubert' ? 'HuBERT' : 'RMVPE'} source<input data-auto-path="${key}" value="${escapeHtml(displayPath(source(key) ?? 'Embedded in selected model'))}" title="${escapeHtml(displayPath(source(key) ?? ''))}" readonly/></label>`).join('')}</div>${loaded?.tensorrtPipeline.phase === 'building' || loaded?.tensorrtPipeline.phase === 'checkingCache' ? '<span class="backend-build-note" role="status">Preparing model plans…</span>' : prepared ? `<span class="backend-build-note">${loaded.tensorrtPipeline.cacheHit ? 'Reused cached plans' : 'Plans ready'} · ${(loaded.tensorrtPipeline.elapsedMs / 1000).toFixed(1)} s</span>` : ''}</div>` : '';
  const note = !catalog ? 'Detecting processing devices…'
    : !entry?.available ? entry?.reason ?? 'Selected backend is unavailable'
      : gpu ? automatic ? prepared ? 'Model plans ready.' : 'All weights included · Load prepares plans automatically.' : artifacts ? 'Plans configured. Load validates the model and GPU compatibility.' : 'Device ready · prepare plans before starting audio.'
        : selected === 'directMl' ? !available ? dmlDevice?.reason ?? 'Selected GPU unavailable · choose another device below.' : automatic ? 'All weights included · Load prepares the graph on your GPU.' : config.directmlSourcesPath ? 'Custom sources configured · ready to load on your GPU.' : 'Choose a complete model or add acoustic sources below.'
          : entry.description;
  const input = (key: string, label: string, value: string | null, placeholder: string) => `<label class="field-label">${label}<input id="${prefix}backend-${key}" data-backend-path="${key}" value="${escapeHtml(displayPath(value ?? ''))}" placeholder="${placeholder}" ${locked ? 'disabled' : ''}/></label>`;
  return `<section class="processing-backend" aria-label="Processing backend">
    <div class="backend-heading"><div><h3>${icon('cpu', 16)}Processing backend</h3><p>${catalog ? `Recommended: ${BACKEND_NAMES[catalog.recommended]}` : 'Device discovery'}</p></div><button class="button ghost compact" data-refresh-backends ${locked ? 'disabled' : ''}>${icon('refresh', 14)}Refresh devices</button></div>
    <div class="backend-selection"><label class="field-label" for="${prefix}processingBackend">Implementation<select id="${prefix}processingBackend" data-processing-backend ${locked || !catalog ? 'disabled' : ''}><option value="auto" ${config.backendAuto ? 'selected' : ''}>Automatic${catalog ? ` · ${BACKEND_NAMES[catalog.recommended]}` : ''}</option>${options}</select></label><div class="backend-status" role="status"><span class="status-dot ${available ? 'positive' : ''}"></span><div><strong>${escapeHtml(selected === 'directMl' ? dmlDevice?.name ?? 'No GPU selected' : entry?.device ?? 'No device selected')}</strong><span>${escapeHtml(note)}</span></div></div></div>
    ${renderPreparationProgress(loaded?.preparation, prefix)}
    ${selected !== 'passthrough' ? renderSpeakerSettings(config.speakerId, samePath(selectedPath, config.modelPath) ? components : null, locked, prefix) : ''}
    ${selected === 'directMl' ? renderDirectMlSettings(config, catalog, locked, prefix, components, loaded, selectedPath) : ''}
    ${selected === 'rvcCpu' && entry?.available ? renderCpuSettings(config, catalog, locked, prefix, components) : ''}
    ${gpu && automatic ? automaticPaths : ''}
    ${gpu && !automatic ? `<details class="backend-preparation" ${!artifacts ? 'open' : ''}><summary>TensorRT model plans</summary><p>Use an existing pipeline, or build one from the shared acoustic models. Preparation runs while audio is stopped.</p>${input('artifacts', 'Pipeline manifest · optional', artifacts, 'Path to pipeline.json')}<div class="backend-source-grid">${input('hubert', 'HuBERT model', config.tensorrt.hubertSource, 'Path to HuBERT .vfr / .vfrm')}${input('rmvpe', 'RMVPE model', config.tensorrt.rmvpeSource, 'Path to RMVPE .vfr / .vfrm')}</div><button id="${prefix}prepareTensorRt" data-prepare-backend class="button secondary compact" ${preparing ? 'aria-busy="true"' : ''} ${locked || !config.tensorrt.hubertSource || !config.tensorrt.rmvpeSource ? 'disabled' : ''}>${preparing ? '<span class="button-spinner" aria-hidden="true"></span>Preparing…' : icon('sparkle', 14) + 'Build plans &amp; load model'}</button><span class="backend-build-note">Choose a voice model above, then build or reuse its cached plans.</span></details>` : ''}
    ${catalog?.backends.some((item) => !item.available) ? `<details class="backend-unavailable"><summary>Unavailable implementations</summary><ul>${catalog.backends.filter((item) => !item.available).map((item) => `<li><strong>${BACKEND_NAMES[item.backend]}</strong><span>${escapeHtml(item.reason ?? '')}</span></li>`).join('')}</ul></details>` : ''}
    <p class="backend-footnote">Backend, speaker, device and path selections save automatically while audio is stopped. Session profile changes require Save profile. Load prepares the selected model.</p>
  </section>`;
}
