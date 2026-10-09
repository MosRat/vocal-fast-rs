import './diagnostics.css';
import { escapeHtml } from './format';
import { icon } from './icons';
import { renderAudioRoute } from './audio-routing';
import { DiagnosticsHistory, HISTORY_MS, type HistoryPoint } from './diagnostics-history';
import type { AudioSample, AudioTelemetry, DirectionTelemetry } from './types';

interface Series {
  label: string;
  color: 'blue' | 'violet' | 'amber' | 'red' | 'green';
  value: (sample: AudioSample) => number | null;
  dashed?: boolean;
}

const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
export function diagnosticNumber(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return '—';
  if (Math.abs(value) >= 1e12) return value.toExponential(1);
  return Math.abs(value) >= 1000 ? compact.format(value) : value.toFixed(digits);
}

function reading(value: number | null | undefined, unit: string, digits = 1): string {
  return `<strong>${diagnosticNumber(value, digits)}</strong><span>${unit}</span>`;
}

function legend(series: readonly Series[]): string {
  return series.map(item => `<span><i class="${item.color}${item.dashed ? ' dashed' : ''}"></i>${item.label}</span>`).join('');
}

function paths(points: readonly HistoryPoint[], series: Series, end: number, ceiling: number, width: number): string[] {
  const segments: string[] = [];
  let segment = '';
  for (const point of points) {
    const value = series.value(point.sample);
    if (point.breakBefore || value == null || !Number.isFinite(value) || value < 0) {
      if (segment) segments.push(segment);
      segment = '';
    }
    if (value == null || !Number.isFinite(value) || value < 0) continue;
    const x = 34 + Math.max(0, Math.min(1, (point.sample.elapsedMs - end + HISTORY_MS) / HISTORY_MS)) * (width - 46);
    const y = 145 - Math.min(1, value / ceiling) * 127;
    segment += `${segment ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)} `;
  }
  if (segment) segments.push(segment);
  return segments;
}

function chart(id: string, points: readonly HistoryPoint[], series: readonly Series[], minimum: number, unit: string): string {
  const values = points.flatMap(point => series.map(item => item.value(point.sample)))
    .filter((value): value is number => value != null && Number.isFinite(value) && value >= 0);
  // Keep SVG scale arithmetic finite even if a corrupt source sends a huge number.
  const peak = Math.min(1e12, Math.max(minimum, ...values));
  const step = 10 ** Math.floor(Math.log10(Math.max(peak, 1)));
  const ceiling = Math.ceil(peak * 1.15 / step) * step;
  const end = points.at(-1)?.sample.elapsedMs ?? HISTORY_MS;
  const width = id === 'audio-age' || id === 'processing-time' ? 480 : 320;
  const right = width - 12;
  const grid = [0, .5, 1].map(ratio => {
    const y = 145 - ratio * 127;
    return `<line class="chart-grid" x1="34" y1="${y}" x2="${right}" y2="${y}"/><text class="chart-axis" x="27" y="${y + 3}" text-anchor="end">${diagnosticNumber(ceiling * ratio, 0)}</text>`;
  }).join('');
  const lines = series.map((item, index) => paths(points, item, end, ceiling, width).map(path => {
    const first = path.match(/^M([\d.]+),/)?.[1];
    const last = [...path.matchAll(/[ML]([\d.]+),/g)].at(-1)?.[1];
    const fill = index === 0 && !item.dashed && first && last
      ? `<path class="chart-area ${item.color}" d="${path}L${last},145 L${first},145 Z" fill="url(#${id}-fill)"/>` : '';
    return `${fill}<path class="chart-line ${item.color}${item.dashed ? ' dashed' : ''}" d="${path}"/>${path.trim().split(' ').length === 1 ? `<circle class="chart-point ${item.color}" cx="${first}" cy="${path.match(/,([\d.]+)/)?.[1]}" r="2.5"/>` : ''}`;
  }).join('')).join('');
  const hits = points.map(point => {
    const x = 34 + Math.max(0, Math.min(1, (point.sample.elapsedMs - end + HISTORY_MS) / HISTORY_MS)) * (width - 46);
    const label = `${((point.sample.elapsedMs - end) / 1000).toFixed(1)} s · ${series.map(item => `${item.label}: ${diagnosticNumber(item.value(point.sample))} ${unit}`).join(' · ')}`;
    return `<g class="chart-hit"><title>${escapeHtml(label)}</title><rect x="${(x - 3).toFixed(2)}" y="18" width="6" height="127" fill="transparent"/><line x1="${x.toFixed(2)}" y1="18" x2="${x.toFixed(2)}" y2="145"/></g>`;
  }).join('');
  return `<div class="diagnostic-chart" data-chart="${id}"><svg viewBox="0 0 ${width} 174" role="img" aria-label="${escapeHtml(series.map(item => item.label).join(' and '))} over the last 60 seconds, in ${escapeHtml(unit)}"><defs><linearGradient id="${id}-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="currentColor" stop-opacity=".2"/><stop offset="100%" stop-color="currentColor" stop-opacity="0"/></linearGradient></defs>${grid}${lines}${hits}<text class="chart-axis" x="34" y="166">−60 s</text><text class="chart-axis" x="${(34 + right) / 2}" y="166" text-anchor="middle">−30 s</text><text class="chart-axis" x="${right}" y="166" text-anchor="end">Latest</text></svg>${values.length ? '' : '<span class="chart-empty">Waiting for audio measurements</span>'}</div>`;
}

function directionDetails(label: string, direction: DirectionTelemetry | undefined): string {
  const rows: Array<[string, number | null | undefined, string]> = [
    ['Callback period · mean', direction?.callbackPeriodMs, 'ms'],
    ['Callback jitter · p95', direction?.callbackJitterP95Ms, 'ms'],
    ['Endpoint padding · p95', direction?.endpointPaddingP95Ms, 'ms'],
    ['Reported stream latency', direction?.streamLatencyMs, 'ms'],
    ['Queue high-water', direction?.queuePeakBlocks, 'blocks'],
    ['Device clock drift', direction?.clockDriftPpm, 'ppm'],
  ];
  return `<section class="direction-details"><h3>${label}</h3><dl>${rows.map(([title, value, unit]) => `<div><dt>${title}</dt><dd>${diagnosticNumber(value)} <span>${unit}</span></dd></div>`).join('')}</dl></section>`;
}

export function renderDiagnostics(audio: AudioTelemetry | undefined, history: DiagnosticsHistory, connected: boolean, controls: string): string {
  const state = history.state(!!audio?.active, connected, performance.now());
  const latest = history.latest;
  const live = state === 'live';
  const current = live ? latest : null;
  const points = history.values;
  const age: Series[] = [{ label: 'Audio age', color: 'blue', value: sample => sample.audioAgeMs }];
  const work: Series[] = [
    { label: 'Processing / block', color: 'violet', value: sample => sample.processingMs },
    { label: 'Audio budget / block', color: 'amber', value: sample => sample.blockBudgetMs, dashed: true },
  ];
  const flow: Series[] = [
    { label: 'Consumed input', color: 'blue', value: sample => sample.inputFramesPerSecond / 1000 },
    { label: 'Produced output', color: 'violet', value: sample => sample.outputFramesPerSecond / 1000 },
  ];
  const drops: Series[] = [
    { label: 'Input full', color: 'amber', value: sample => sample.capture.queueDropsPerSecond },
    { label: 'Output full', color: 'red', value: sample => sample.render.queueDropsPerSecond },
    { label: 'Stale skipped', color: 'violet', value: sample => sample.capture.staleDropsPerSecond + sample.render.staleDropsPerSecond },
  ];
  const silence: Series[] = [{ label: 'Missing output', color: 'red', value: sample => sample.underflowMsPerSecond }];
  const totalDrops = current ? current.capture.queueDropsPerSecond + current.render.queueDropsPerSecond : null;
  const stateText = { live: 'Live', waiting: 'Sampling…', paused: 'Paused', stale: 'Updates delayed' }[state];
  return `<article class="diagnostics-panel span-2" data-section-target="diagnostics">
    <header class="diagnostics-toolbar"><div class="diagnostics-live ${state}" role="status"><span class="status-dot"></span>${stateText}<span class="diagnostic-window">60-second history</span></div><div class="diagnostic-controls">${controls}</div></header>
    ${state === 'paused' ? `<div class="diagnostics-notice">${icon('chart', 16)}<span>${points.length ? 'Stream stopped. The last session’s charts are retained until you start again.' : 'Start an audio stream to see latency and queue health in real time.'}</span></div>` : state === 'stale' ? `<div class="diagnostics-notice warning">${icon('alert', 16)}<span>Measurements are delayed. Values are paused until fresh data arrives.</span></div>` : ''}
    <div class="diagnostics-primary">
      <section class="diagnostic-card latency-card"><div class="diagnostic-card-heading"><div><span class="kicker">CAPTURE → RENDER</span><h2>Audio age</h2></div><div class="diagnostic-symbol blue">${icon('waveform', 19)}</div></div><div class="diagnostic-reading">${reading(current?.audioAgeMs, 'ms')}<small>latest measured block</small></div>${chart('audio-age', points, age, 20, 'ms')}<div class="diagnostic-percentiles"><span>Session p50 <b>${diagnosticNumber(latest?.audioAgeP50Ms)} ms</b></span><span>p95 <b>${diagnosticNumber(latest?.audioAgeP95Ms)} ms</b></span><span>p99 <b>${diagnosticNumber(latest?.audioAgeP99Ms)} ms</b></span></div><p class="diagnostic-caption">Timestamp age at the render callback; CPAL adds reported playback timing. Hardware round-trip latency requires a loopback measurement.</p></section>
      <section class="diagnostic-card processing-card"><div class="diagnostic-card-heading"><div><span class="kicker">WORKER PERFORMANCE</span><h2>Processing time</h2></div><div class="diagnostic-symbol violet">${icon('cpu', 19)}</div></div><div class="diagnostic-reading">${reading(current?.processingMs, 'ms / block')}<small class="${(current?.processingLoadPercent ?? 0) > 100 ? 'warning' : ''}">${diagnosticNumber(current?.processingLoadPercent)}% of audio budget</small></div>${chart('processing-time', points, work, 10, 'ms')}<div class="diagnostic-legend">${legend(work)}</div><p class="diagnostic-caption">Mean inference time in each sample interval. The budget is the audio duration per processed block; sustained values above it can cause gaps.</p></section>
    </div>
    <div class="diagnostics-secondary">
      <section class="diagnostic-card"><div class="diagnostic-card-heading"><h2>Worker throughput</h2>${icon('speaker', 16)}</div><div class="diagnostic-reading small">${reading(current ? current.inputFramesPerSecond / 1000 : null, 'kframes/s in')}<em>${diagnosticNumber(current ? current.outputFramesPerSecond / 1000 : null)} out</em></div>${chart('worker-throughput', points, flow, 48, 'kframes/s')}<div class="diagnostic-legend">${legend(flow)}</div><p class="diagnostic-caption">Native endpoint frames. Different device sample rates produce different frame rates.</p></section>
      <section class="diagnostic-card"><div class="diagnostic-card-heading"><h2>Queue drops</h2>${icon('database', 16)}</div><div class="diagnostic-reading small ${(totalDrops ?? 0) > 0 ? 'warning' : ''}">${reading(totalDrops, 'blocks/s full')}</div>${chart('queue-drops', points, drops, 2, 'blocks/s')}<div class="diagnostic-legend">${legend(drops)}</div><p class="diagnostic-caption">Queue-full drops and deliberate stale-input skips are measured separately.</p></section>
      <section class="diagnostic-card"><div class="diagnostic-card-heading"><h2>Output underflow</h2>${icon('alert', 16)}</div><div class="diagnostic-reading small ${(current?.underflowMsPerSecond ?? 0) > 0 ? 'warning' : ''}">${reading(current?.underflowMsPerSecond, 'ms missing / s')}</div>${chart('output-underflow', points, silence, 5, 'ms missing/s')}<div class="diagnostic-legend">${legend(silence)}</div><p class="diagnostic-caption">Missing interleaved samples converted to silence duration using the output device’s rate and channels.</p></section>
    </div>
    <details class="diagnostic-details"><summary>${icon('sliders', 16)}<span>Endpoint details</span><small>Session statistics · capture / render</small>${icon('chevron', 14)}</summary><div class="direction-grid">${directionDetails('Capture', latest?.capture)}${directionDetails('Render', latest?.render)}</div><p class="diagnostic-caption">Percentiles, means and queue peaks cover the session. Clock drift is an estimate. “—” means the backend has no measurement.</p>${renderAudioRoute(audio?.route ?? null)}<p class="diagnostic-caption">Worker errors: ${diagnosticNumber(current?.workerFailuresPerSecond)} / s</p></details>
    <div class="diagnostic-help"><section><span class="diagnostic-help-icon amber">${icon('database', 18)}</span><div><h3>What is a queue drop?</h3><p>A full queue cannot accept a new audio block, so the block is discarded. A stale skip deliberately discards older input to keep the voice current. Frequent drops can mean the worker or output is falling behind.</p></div></section><section><span class="diagnostic-help-icon red">${icon('speaker', 18)}</span><div><h3>What is underflow?</h3><p>The output device requests audio before enough processed samples are ready. The missing part is filled with silence, which may sound like a break or click. This is missing audio duration, not an event count.</p></div></section></div>
    <footer class="diagnostic-footer"><span>${escapeHtml(audio?.inputDevice ?? 'System input')}<b>→</b>${escapeHtml(audio?.outputDevice ?? 'System output')}</span><span>${live && current ? `Sample interval ${(current.windowMs / 1000).toFixed(1)} s` : 'No live measurement'}</span></footer>
  </article>`;
}
