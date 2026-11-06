| Method | Pretraining | Macro-F1 unbalanced | Macro-F1 balanced | Detection mAP |
|---|---|---:|---:|---:|
| Chance level | Label permutation / random ranking | 10.00 | 16.67 | 7.84 |
| Spectral features | -- | 14.15 ± 0.67 | 33.52 ± 2.06 | 26.34 ± 1.17 |
| MFCCs | -- | 21.22 ± 0.99 | 44.20 ± 2.20 | 33.91 ± 1.77 |
| Mean spectrogram | -- | 32.05 ± 1.30 | 54.95 ± 2.15 | 47.80 ± 2.10 |
| AVES-core | General audio (AudioSet, FSD50K) | 51.60 ± 2.28 | 67.60 ± 2.11 | 57.39 ± 2.07 |
| BioLingual | Audio-text (AnimalSpeak) | 54.03 ± 2.72 | 70.98 ± 2.12 | 66.52 ± 2.23 |
| AVES-bio | Animal vocalizations (AudioSet, VGGSound) | 54.45 ± 2.58 | 75.13 ± 1.96 | 65.01 ± 2.29 |
| AVES-bio (from scratch) | OpenWhistle (from scratch) | 60.97 ± 3.76 | 79.89 ± 1.81 | 76.17 ± 1.97 |
| wav2vec 2.0 | OpenWhistle (from scratch) | 61.95 ± 2.79 | 81.40 ± 1.78 | 79.59 ± 1.83 |
| AVES-bio (continual) | AVES-bio checkpoint to OpenWhistle | 64.75 ± 2.64 | 75.24 ± 2.01 | 73.87 ± 2.06 |
