"""Independent NumPy reference for the current speech-safe Rust SOLA boundary.

Versioned traces choose this policy explicitly. Older validation traces retain
the legacy sine-squared crossfade implemented in the reference runner.
"""
import numpy as np


class SpeechSafeJoiner:
    def __init__(self, overlap=800, search=160, sample_rate=16000):
        self.overlap, self.search, self.sample_rate = overlap, search, sample_rate
        self.previous = np.zeros(overlap, np.float32)
        self.offset = 0
        self.initialized = False

    def process(self, audio, block, fixed_offset=None):
        audio = np.asarray(audio, np.float32)
        if len(audio) < block + self.overlap + self.search:
            raise ValueError("short decoder tail")
        previous = self.previous.astype(np.float64)
        prior_energy = float(np.sum(previous ** 2))
        if prior_energy >= self.overlap * 1e-8:
            candidates = np.lib.stride_tricks.sliding_window_view(
                audio[:self.overlap + self.search].astype(np.float64), self.overlap)
            energies = np.sum(candidates ** 2, axis=1)
            scores = np.zeros(len(candidates), np.float64)
            valid = energies >= self.overlap * 1e-8
            scores[valid] = np.sum(candidates[valid] * previous, axis=1) / np.sqrt(prior_energy * energies[valid])
            best = int(np.argmax(scores))
            if self.initialized and scores[best] >= 0.35:
                self.offset = best
        if fixed_offset is not None:
            self.offset = fixed_offset
        current = audio[self.offset:]
        output = current[:block].copy()
        overlap = min(self.overlap, block)
        old = previous[:overlap]
        new = current[:overlap].astype(np.float64)
        a2, b2 = float(np.sum(old ** 2)), float(np.sum(new ** 2))
        denominator = np.sqrt(a2 * b2)
        correlation = np.float32(np.clip(np.sum(old * new) / denominator, 0, 1)) if denominator > 1e-12 else np.float32(0)
        ramp = min(max(self.sample_rate // 200, 1), overlap)
        onset = not self.initialized or a2 * 16 < b2
        release = not onset and b2 * 16 < a2
        start, length = (0, ramp) if onset else ((overlap - ramp, ramp) if release else (0, overlap))
        ids = np.arange(overlap)
        phase = np.ones(overlap, np.float32) if length <= 1 else np.minimum((ids-start).astype(np.float32) / np.float32(length-1), np.float32(1))
        phase[ids < start] = 0
        weight = np.sin(np.float32(0.5) * np.float32(np.pi) * phase) ** 2
        old_weight = np.float32(1) - weight
        power = old_weight * old_weight + weight * weight + np.float32(2) * correlation * old_weight * weight
        divisor = np.maximum(np.sqrt(power), np.float32(1 / np.sqrt(2)))
        output[:overlap] = (self.previous[:overlap] * old_weight + current[:overlap] * weight) / divisor
        self.previous = current[block:block + self.overlap].copy()
        self.initialized = True
        return output, self.offset
