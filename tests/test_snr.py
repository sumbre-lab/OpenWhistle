"""Checks for the audio-derived SNR figure."""
import csv
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cnn/analysis'))
from snr import decode_audio, estimate_snr, SNRConfig
from plot_snr import compute_dataset, parse_args


class SNRTests(unittest.TestCase):
    def test_more_noise_reduces_snr_and_scaling_preserves_it(self):
        fs = 96000
        t = np.arange(fs // 2) / fs
        tone = np.sin(2 * np.pi * 10000 * t)
        noise = np.random.default_rng(7).normal(size=len(t))
        clean = tone + .1 * noise
        quiet, status, _ = estimate_snr(clean, fs)
        noisy, noisy_status, _ = estimate_snr(tone + noise, fs)
        scaled, _, _ = estimate_snr(clean * .01, fs)
        self.assertEqual((status, noisy_status), ('ok', 'ok'))
        self.assertGreater(quiet - noisy, 15)
        self.assertAlmostEqual(quiet, scaled, places=4)

    def test_silence_and_short_clips_do_not_produce_valid_snr(self):
        self.assertEqual(estimate_snr(np.zeros(96000), 96000)[1], 'no_ridge_detected')
        self.assertEqual(estimate_snr(np.zeros(10), 96000)[1], 'too_short')

    def test_embedded_audio_bytes_and_stereo(self):
        buf = BytesIO()
        sf.write(buf, np.ones((100, 2)) * [.2, .6], 96000, format='WAV', subtype='FLOAT')
        audio, fs = decode_audio({'bytes': buf.getvalue(), 'path': 'unused.wav'})
        self.assertEqual(fs, 96000)
        np.testing.assert_allclose(audio, .4)

    def test_hf_splits_and_limit_are_applied_across_dataset(self):
        buf = BytesIO()
        sf.write(buf, np.zeros(3000), 96000, format='WAV')
        payload = {'audio': {'bytes': buf.getvalue()}}

        class Stream(list):
            def cast_column(self, *args):
                return self
            def take(self, count):
                return self[:count]

        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', return_value=Stream([payload])) as load:
            out = Path(tmp) / 'scores.csv'
            values, summary = compute_dataset('org/ds', 'all', ('train', 'validation', 'test'),
                                              'pinned-sha', 2, out, SNRConfig())
            self.assertEqual(summary['processed'], 2)
            self.assertEqual(len(values), 0)
            self.assertEqual([c.kwargs['split'] for c in load.call_args_list], ['train', 'validation'])
            for call in load.call_args_list:
                self.assertEqual(call.kwargs['name'], 'all')
                self.assertEqual(call.kwargs['revision'], 'pinned-sha')
                self.assertTrue(call.kwargs['streaming'])
                self.assertEqual(call.kwargs['columns'], ['audio'])
            with out.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_full_corpus_is_default(self):
        self.assertEqual(parse_args([]).limit, 0)

    def test_parallel_scores_and_order_match_serial(self):
        payloads = []
        fs = 96000
        t = np.arange(fs // 4) / fs
        rng = np.random.default_rng(17)
        for frequency in (5000, 9000, 12000, 17000):
            buf = BytesIO()
            sf.write(buf, np.sin(2 * np.pi * frequency * t) + .1 * rng.normal(size=len(t)),
                     fs, format='WAV', subtype='FLOAT')
            payloads.append({'audio': {'bytes': buf.getvalue()}})

        class Stream(list):
            def cast_column(self, *args):
                return self

        with tempfile.TemporaryDirectory() as tmp, patch('datasets.load_dataset', return_value=Stream(payloads)):
            results = []
            for workers in (1, 4):
                path = Path(tmp) / f'{workers}.csv'
                scores, summary = compute_dataset('org/ds', 'all', ('train',), 'sha',
                                                  0, path, SNRConfig(), workers=workers)
                results.append((path.read_bytes(), scores, summary))
            self.assertEqual(results[0][0], results[1][0])
            np.testing.assert_array_equal(results[0][1], results[1][1])
            self.assertEqual(results[0][2], results[1][2])


if __name__ == '__main__':
    unittest.main()
