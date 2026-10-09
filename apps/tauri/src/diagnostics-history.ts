import type { AudioSample, AudioTelemetry } from './types';

export const HISTORY_MS = 60_000;
export const MAX_POINTS = 128;
export const STALE_MS = 3_000;

export interface HistoryPoint {
  sample: AudioSample;
  breakBefore: boolean;
}

/** Window-local bounded history of shared Rust samples. Reads never count as samples. */
export class DiagnosticsHistory {
  private session: string | null = null;
  private receivedAt = -Infinity;
  private points: HistoryPoint[] = [];

  observe(audio: AudioTelemetry | undefined, now: number): void {
    if (!audio?.active || audio.sessionId == null) return;
    if (audio.sessionId !== this.session) {
      // Native session IDs are decimal u64 strings. A delayed reply from an
      // earlier session must not clear the new session's history.
      if (this.session && /^\d+$/.test(this.session) && /^\d+$/.test(audio.sessionId)
        && BigInt(audio.sessionId) < BigInt(this.session)) return;
      this.session = audio.sessionId;
      this.points = [];
      this.receivedAt = -Infinity;
    }
    const sample = audio.sample;
    if (!sample || !Number.isFinite(sample.elapsedMs) || sample.elapsedMs < 0
      || !Number.isFinite(sample.windowMs) || sample.windowMs <= 0) return;
    const previous = this.points.at(-1)?.sample;
    // elapsedMs also handles sequence wrap and late command responses.
    if (previous && sample.elapsedMs <= previous.elapsedMs) return;
    const breakBefore = !!previous && (sample.elapsedMs - previous.elapsedMs > STALE_MS
      || now - this.receivedAt > STALE_MS);
    this.points.push({ sample, breakBefore });
    this.receivedAt = now;
    const cutoff = sample.elapsedMs - HISTORY_MS;
    this.points = this.points.filter(point => point.sample.elapsedMs >= cutoff).slice(-MAX_POINTS);
  }

  get values(): readonly HistoryPoint[] { return this.points; }
  get latest(): AudioSample | null { return this.points.at(-1)?.sample ?? null; }

  state(active: boolean, connected: boolean, now: number): 'live' | 'waiting' | 'paused' | 'stale' {
    if (!active) return 'paused';
    if (!connected) return 'stale';
    if (!this.latest) return 'waiting';
    return now - this.receivedAt > STALE_MS ? 'stale' : 'live';
  }
}
