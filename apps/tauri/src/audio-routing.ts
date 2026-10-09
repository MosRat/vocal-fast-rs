import type { AudioRouteInfo } from './types';
import { escapeHtml } from './format';
import { icon } from './icons';
import './audio-routing.css';

function formatLabel(format: AudioRouteInfo['captureFormat']): string {
  return `${(format.sampleRate / 1000).toLocaleString()} kHz · ${format.channels} ch`;
}

/** The same actual route and fallback notice is used by Studio, Settings and diagnostics. */
export function renderAudioRoute(route: AudioRouteInfo | null, compact = false): string {
  if (!route) return compact ? '' : '<p class="audio-route-idle">Device sample rates and channels are selected automatically when the stream starts.</p>';
  const fallback = route.backend === 'cpal' && !!route.fallbackReason;
  const batchMs = route.processingBatchFrames * 1000 / route.processingFormat.sampleRate;
  const buffering = Number.isFinite(batchMs) && route.playbackPrefillMicros > 0
    ? `<div class="audio-route-buffering"><span>Model batch <strong>${batchMs.toFixed(0)} ms</strong></span><span>Playback prefill <strong>${(route.playbackPrefillMicros / 1000).toFixed(0)} ms</strong></span><span>Continuous input · FIFO</span></div>`
    : '';
  return `<div class="audio-route ${compact ? 'compact' : ''} ${fallback ? 'fallback' : ''}" data-audio-route>
    <div class="audio-route-heading">${icon(fallback ? 'alert' : 'speaker', 15)}<strong>${route.backend === 'nativeWasapi' ? 'Native WASAPI · low-period shared' : fallback ? 'CPAL fallback active' : 'CPAL audio'}</strong><span>Audio driver</span></div>
    ${fallback ? `<div class="audio-route-warning" role="alert"><p>Native WASAPI could not start. Audio is running through CPAL.</p><details><summary>Fallback reason</summary><p>${escapeHtml(route.fallbackReason!)}</p></details></div>` : ''}
    ${compact && !fallback ? '' : `${buffering}<dl class="audio-route-formats"><div><dt>Capture</dt><dd>${escapeHtml(formatLabel(route.captureFormat))}</dd></div><div><dt>Worker</dt><dd>${escapeHtml(formatLabel(route.processingFormat))}</dd></div><div><dt>Render</dt><dd>${escapeHtml(formatLabel(route.renderFormat))}</dd></div></dl><p class="audio-route-conversion">Automatic streaming resampling &amp; channel mapping${route.conversionDelayMicros ? ` · ${(route.conversionDelayMicros / 1000).toFixed(2)} ms filter delay` : ''}</p>`}
  </div>`;
}
