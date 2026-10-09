import type { GuiConfig, ModelComponents, ProcessingBackendCatalog, ProcessingDevice, Snapshot } from './types';
import { displayPath, escapeHtml, samePath } from './format';
import { icon } from './icons';
import './directml-settings.css';

export function selectedDirectMlDevice(config: GuiConfig, catalog: ProcessingBackendCatalog | null): ProcessingDevice | null {
  const backend = catalog?.backends.find(item => item.backend === 'directMl');
  const id = config.directmlDeviceId || backend?.recommendedDeviceId;
  return backend?.devices.find(device => device.id.toLowerCase() === id?.toLowerCase()) ?? null;
}

function vendorName(vendor: number): string {
  return ({ 0x10de: 'NVIDIA', 0x1002: 'AMD', 0x8086: 'Intel', 0x1414: 'Microsoft' } as Record<number, string>)[vendor] ?? 'Graphics adapter';
}

function deviceLabel(device: ProcessingDevice): string {
  const suffix = device.kind === 'virtual' ? ' · virtual adapter' : device.kind === 'remote' ? ' · remote session' : '';
  return device.name + suffix;
}

function memorySize(bytes: number): string {
  return bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GiB` : `${(bytes / 2 ** 20).toFixed(0)} MiB`;
}

export function renderDirectMlSettings(config: GuiConfig, catalog: ProcessingBackendCatalog | null, locked: boolean, prefix: string, components: ModelComponents | null, loaded: Snapshot | null, selectedPath: string): string {
  const backend = catalog?.backends.find(item => item.backend === 'directMl');
  const devices = backend?.devices ?? [];
  const device = selectedDirectMlDevice(config, catalog);
  const missing = !!config.directmlDeviceId && !device;
  const available = !!device?.available;
  const recommended = devices.find(item => item.id === backend?.recommendedDeviceId);
  const complete = !!components?.automaticTensorRt;
  const boundModel = loaded?.config.backend === 'directMl' && !!loaded.engine && samePath(selectedPath, loaded.config.modelPath);
  const changed = boundModel && ((loaded.config.directmlDeviceId ?? null) !== config.directmlDeviceId || loaded.config.directmlSourcesPath !== config.directmlSourcesPath);
  const ready = boundModel && !changed && loaded.status.engine === 'ready';
  const deviceNote = missing ? 'The saved GPU is no longer connected. Choose Automatic or select an available device.'
    : device && !device.available ? device.reason ?? 'This GPU does not support DirectML.'
      : !catalog ? 'Detecting graphics devices…'
        : !available ? backend?.reason ?? 'No compatible DirectML GPU found.'
          : changed ? 'Applying GPU selection automatically while idle…'
            : config.directmlDeviceId ? ready ? 'Device selection saved · this model is loaded on the selected GPU.' : 'Manual selection · Load uses this GPU for the model.'
              : 'Automatic prefers a compatible hardware GPU in Windows high-performance order.';
  const options = devices.map(item => `<option value="${escapeHtml(item.id)}" ${config.directmlDeviceId && device?.id === item.id ? 'selected' : ''} ${item.available ? '' : 'disabled'} title="${escapeHtml(item.reason ?? deviceLabel(item))}">${escapeHtml(deviceLabel(item))}${item.available ? item.id === backend?.recommendedDeviceId ? ' · recommended' : '' : ' · unavailable'}</option>`).join('');
  const memory = device ? [
    device.dedicatedMemoryBytes > 0 ? `${memorySize(device.dedicatedMemoryBytes)} dedicated` : 'Shared graphics',
    device.sharedMemoryBytes > 0 ? `${memorySize(device.sharedMemoryBytes)} shared RAM` : '',
  ].filter(Boolean) : [];
  const modelTitle = config.directmlSourcesPath ? 'Custom model sources' : complete ? 'Complete model · automatic preparation' : 'Acoustic model sources required';
  const modelNote = config.directmlSourcesPath ? 'Load uses your source manifest and compiles its graph on the selected GPU.'
    : complete ? 'Load prepares the bundled Synth, HuBERT and RMVPE weights, then compiles the graph on your GPU.'
      : 'Choose a complete VFRM bundle, or provide the acoustic models in Advanced sources below.';
  return `<div class="dml-settings">
    <div class="dml-device-grid">
      <label class="field-label" for="${prefix}directmlDevice">GPU device<select id="${prefix}directmlDevice" data-directml-device ${locked || !catalog ? 'disabled' : ''} aria-describedby="${prefix}dmlDeviceNote">
        <option value="auto" ${!config.directmlDeviceId ? 'selected' : ''}>Automatic${recommended ? ` · ${escapeHtml(recommended.name)}` : ' · no compatible GPU'}</option>
        ${missing ? `<option value="${escapeHtml(config.directmlDeviceId!)}" selected disabled>Previously selected GPU · unavailable</option>` : ''}${options}
      </select></label>
      <div class="dml-device-summary ${available ? '' : 'unavailable'}"><span class="dml-device-icon">${icon('cpu', 20)}</span><div><strong>${escapeHtml(device ? deviceLabel(device) : missing ? 'Saved GPU unavailable' : 'No compatible GPU')}</strong><span>${escapeHtml(device ? vendorName(device.vendorId) : 'DirectML · Windows GPU')}</span><div class="dml-device-tags">${memory.map(item => `<span>${escapeHtml(item)}</span>`).join('')}</div></div></div>
    </div>
    <p id="${prefix}dmlDeviceNote" class="dml-device-note ${missing || device && !available ? 'unavailable' : ''}" role="status">${escapeHtml(deviceNote)}</p>
    <div class="dml-model-preparation"><span class="dml-model-icon">${icon(ready ? 'check' : config.directmlSourcesPath || complete ? 'sparkle' : 'alert', 18)}</span><div><div class="dml-model-heading"><strong>${modelTitle}</strong><span class="dml-state ${ready ? 'ready' : changed ? 'pending' : ''}">${ready ? 'Ready' : changed ? 'Reload to apply' : complete || config.directmlSourcesPath ? 'Prepared on Load' : 'Choose sources'}</span></div><p>${modelNote}</p></div></div>
    <details class="backend-preparation dml-advanced" ${!complete && !config.directmlSourcesPath ? 'open' : ''}>
      <summary>${icon('settings', 13)}Advanced sources<span>${config.directmlSourcesPath ? 'Custom manifest' : 'Optional'}</span></summary>
      <p>A source manifest overrides the bundle’s acoustic models and graph settings. The GPU selection above always applies.</p>
      <label class="field-label" for="${prefix}backend-dmlSources">Source manifest<input id="${prefix}backend-dmlSources" data-backend-path="dmlSources" value="${escapeHtml(displayPath(config.directmlSourcesPath ?? ''))}" placeholder="Path to sources.json" ${locked ? 'disabled' : ''}/></label>
    </details>
  </div>`;
}
