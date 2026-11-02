Classification configuration: `all`. Scores in percent.

| Method | Pretraining | Classification Macro-F1 | Detection mAP |
|---|---|---:|---:|
| Chance level | Label permutation / random ranking | 10.00 | 7.84 |
| Spectral features | -- | 14.15 ± 0.67 | 26.34 ± 1.17 |
| MFCCs | -- | 21.22 ± 0.99 | 33.91 ± 1.77 |
| Mean spectrogram | -- | 32.05 ± 1.30 | 47.80 ± 2.10 |
| AVES-core | General audio (AudioSet, FSD50K) | 51.60 ± 2.28 | 57.39 ± 2.07 |
| BioLingual | Audio-text (AnimalSpeak) | 54.03 ± 2.72 | 66.52 ± 2.23 |
| AVES-bio | Animal vocalizations (AudioSet, VGGSound) | 54.45 ± 2.58 | 65.01 ± 2.29 |
| AVES-bio (from scratch) | OpenWhistle (from scratch) | 60.97 ± 3.76 | 76.17 ± 1.97 |
| Wav2Vec2.0 | OpenWhistle (from scratch) | 61.95 ± 2.79 | **79.59 ± 1.83** |
| AVES-bio (continual) | AVES-bio checkpoint to OpenWhistle | **64.75 ± 2.64** | 73.87 ± 2.06 |

Performance on whistle-type classification (macro-F1) and whistle-type detection (mAP). Models are grouped by representation and pretraining strategy. All use linear probing with a logistic regression classifier on frozen representations. Scores are reported as percentages. Each entry gives the full-test metric $\pm$ the standard deviation over 1,000 bootstrap resamples of the test set.
Continual AVES uses three 16 kHz band views concatenated into one embedding; scratch AVES uses native 44.1 kHz audio.
