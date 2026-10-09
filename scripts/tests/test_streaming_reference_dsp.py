"""Boundary regressions: weak speech, release and uncorrelated noise power."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming_reference_dsp import SpeechSafeJoiner


class StreamingReferenceTests(unittest.TestCase):
    def test_silence_onset_keeps_weak_consonant(self):
        joiner = SpeechSafeJoiner()
        joiner.process(np.zeros(4960, np.float32), 4000)
        audio = np.zeros(4960, np.float32)
        audio[160:320] = 0.01
        actual, offset = joiner.process(audio, 4000)
        self.assertEqual(offset, 0)
        np.testing.assert_array_equal(actual[160:320], audio[160:320])

    def test_silent_revision_keeps_previous_release(self):
        joiner = SpeechSafeJoiner()
        audio = np.zeros(4960, np.float32)
        audio[4000:4400] = 0.1
        joiner.process(audio, 4000)
        actual, _ = joiner.process(np.zeros_like(audio), 4000)
        np.testing.assert_array_equal(actual[:400], audio[4000:4400])
        self.assertTrue((actual[800:] == 0).all())

    def test_uncorrelated_noise_overlap_keeps_power(self):
        joiner = SpeechSafeJoiner(search=0)
        first = np.zeros(4800, np.float32)
        first[4000:] = np.tile([0.1, -0.1], 400)
        joiner.process(first, 4000)
        second = np.tile([0.1, 0.1, -0.1, -0.1], 1200).astype(np.float32)
        actual, _ = joiner.process(second, 4000)
        self.assertAlmostEqual(float(np.sqrt(np.mean(actual[:800] ** 2))), 0.1, delta=0.001)

    def test_quiet_residue_does_not_change_previous_offset(self):
        joiner = SpeechSafeJoiner()
        joiner.initialized, joiner.offset = True, 23
        joiner.previous.fill(1e-5)
        _, offset = joiner.process(np.full(4960, 1e-5, np.float32), 4000)
        self.assertEqual(offset, 23)


if __name__ == "__main__":
    unittest.main()
