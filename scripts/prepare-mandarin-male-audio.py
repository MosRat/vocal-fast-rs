"""Create compact input-test copies without changing the original recordings.

Only outer silence and the middle of long near-silent intervals are removed.
Short pauses and speech remain in order; volume is adjusted once per utterance.
This is preparation of test input, not processing in the audio runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

QUIET_AMPLITUDE = 10 ** (-55 / 20)
MIN_PAUSE_SECONDS = .35
KEEP_PAUSE_SECONDS = .18
EDGE_SECONDS = .06
PEAK = 10 ** (-1 / 20)
TARGET_ACTIVE_RMS = 10 ** (-21 / 20)


def envelope(audio: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    hop = round(rate * .01)
    padded = np.pad(audio, (0, -len(audio) % hop))
    return np.sqrt(np.mean(padded.reshape(-1, hop) ** 2, axis=1)), hop


def prepare(audio: np.ndarray, rate: int) -> tuple[np.ndarray, dict]:
    rms, hop = envelope(audio, rate)
    quiet = rms < QUIET_AMPLITUDE
    audible = np.flatnonzero(~quiet)
    if not audible.size:
        raise ValueError('recording has no usable speech energy')
    margin = round(rate * EDGE_SECONDS)
    start = max(0, int(audible[0] * hop) - margin)
    end = min(len(audio), int((audible[-1] + 1) * hop) + margin)
    edges = np.diff(np.r_[False, quiet, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    cuts = []
    for first, last in zip(starts, ends):
        a, b = first * hop, min(len(audio), last * hop)
        if a <= start or b >= end or (b - a) / rate < MIN_PAUSE_SECONDS:
            continue
        keep_half = round(rate * KEEP_PAUSE_SECONDS / 2)
        cuts.append((int(a + keep_half), int(b - keep_half)))
    pieces = []
    cursor = start
    for a, b in cuts:
        pieces.append(audio[cursor:a].copy())
        cursor = b
    pieces.append(audio[cursor:end].copy())
    # Edits lie in near-silence. A short taper further avoids a splice click.
    fade = round(rate * .005)
    for i, piece in enumerate(pieces):
        if len(piece) < 2 * fade:
            raise ValueError('unexpected short retained segment')
        if i or start:
            piece[:fade] *= np.linspace(0, 1, fade)
        if i < len(pieces) - 1 or end < len(audio):
            piece[-fade:] *= np.linspace(1, 0, fade)
    compact = np.concatenate(pieces)
    frames, _ = envelope(compact, rate)
    active = frames >= QUIET_AMPLITUDE
    active_rms = float(np.sqrt(np.mean(frames[active] ** 2)))
    gain = min(16.0, TARGET_ACTIVE_RMS / active_rms, PEAK / float(np.max(np.abs(compact))))
    return compact * gain, {
        'original_seconds': round(len(audio) / rate, 3),
        'prepared_seconds': round(len(compact) / rate, 3),
        'head_trim_seconds': round(start / rate, 3),
        'tail_trim_seconds': round((len(audio) - end) / rate, 3),
        'shortened_pauses': len(cuts),
        'removed_internal_silence_seconds': round(sum(b-a for a,b in cuts) / rate, 3),
        'gain_db': round(20 * math.log10(gain), 3),
        'active_rms_after_dbfs': round(20 * math.log10(active_rms * gain), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('target/test-assets/mandarin-male'))
    args = parser.parse_args()
    root = args.directory.resolve()
    source = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    report = {'source_manifest': 'manifest.json', 'quiet_threshold_dbfs': -55,
              'minimum_pause_seconds': MIN_PAUSE_SECONDS, 'retained_pause_seconds': KEEP_PAUSE_SECONDS,
              'utterance_gap_seconds': .18, 'edge_margin_seconds': EDGE_SECONDS,
              'target_active_rms_dbfs': -21, 'speakers': []}
    for index, speaker in enumerate(source['speakers'], 1):
        chunks, edits = [], []
        for record in speaker['originals']:
            path = root / record['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'source changed: {path}')
            audio, rate = sf.read(path, dtype='float64')
            if rate != 44100 or audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError(f'unexpected source audio: {path}')
            prepared, edit = prepare(audio, rate)
            chunks.append(prepared)
            edits.append({'file': record['file'], **edit})
        gap = np.zeros(round(rate * .18))
        joined = np.concatenate([part for i, chunk in enumerate(chunks)
                                 for part in ([gap, chunk] if i else [chunk])])
        divisor = math.gcd(rate, 16000)
        output = resample_poly(joined, 16000 // divisor, rate // divisor, window=('kaiser', 8.6))
        output *= min(1.0, PEAK / float(np.max(np.abs(output))))
        name = f"male-{index:02d}-{speaker['speaker']}-compact-16k.wav"
        path = root / name
        temporary = path.with_suffix('.partial.wav')
        sf.write(temporary, output, 16000, subtype='PCM_16')
        decoded, actual_rate = sf.read(temporary, dtype='float64')
        if actual_rate != 16000 or not np.isfinite(decoded).all() or np.max(np.abs(decoded)) >= .999:
            raise ValueError(f'invalid prepared WAV: {path}')
        temporary.replace(path)
        result = {'speaker': speaker['speaker'], 'gender': speaker['gender'], 'file': name,
                  'seconds': round(len(decoded) / actual_rate, 3), 'sample_rate': actual_rate, 'channels': 1,
                  'peak_dbfs': round(20 * math.log10(float(np.max(np.abs(decoded)))), 3),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'utterances': edits}
        report['speakers'].append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    (root / 'compact-manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
