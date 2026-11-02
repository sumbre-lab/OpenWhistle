Classification configuration: `balanced`. Scores in percent.

| Method | Pretraining | Classification Macro-F1 | Detection mAP |
|---|---|---:|---:|
| Chance level | Label permutation / random ranking | 16.67 | 7.84 |
| Spectral features | -- | 33.52 ± 2.06 | 26.34 ± 1.17 |
| MFCCs | -- | 44.20 ± 2.20 | 33.91 ± 1.77 |
| Mean spectrogram | -- | 54.95 ± 2.15 | 47.80 ± 2.10 |
| AVES-core | General audio (AudioSet, FSD50K) | 67.60 ± 2.11 | 57.39 ± 2.07 |
| BioLingual | Audio-text (AnimalSpeak) | 70.98 ± 2.12 | 66.52 ± 2.23 |
| AVES-bio | Animal vocalizations (AudioSet, VGGSound) | 75.13 ± 1.96 | 65.01 ± 2.29 |
| AVES-bio (from scratch) | OpenWhistle (from scratch) | 79.89 ± 1.81 | 76.17 ± 1.97 |
| wav2vec 2.0 | OpenWhistle (from scratch) | **81.40 ± 1.78** | **79.59 ± 1.83** |
| AVES-bio (continual) | AVES-bio checkpoint to OpenWhistle | 75.24 ± 2.01 | 73.87 ± 2.06 |

Performance on whistle-type classification (balanced, macro-F1) and whistle-type detection (mAP). Models are grouped by representation and pretraining strategy. All use linear probing with a logistic regression classifier on frozen representations. Scores are reported as percentages. Each entry gives the full-test metric $\pm$ the standard deviation over 1,000 bootstrap resamples of the test set.
Continual AVES uses three 16 kHz band views concatenated into one embedding; scratch AVES uses native 44.1 kHz audio.
