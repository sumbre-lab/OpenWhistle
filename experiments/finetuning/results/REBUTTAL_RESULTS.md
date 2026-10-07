# Authoritative rebuttal results

Only the values in this file and the source files named below should be used for
the rebuttal. Percentages are reported as mean ± standard deviation.

Historical accuracy-selected linear-probing reports are kept in the local
archive described in [results organization](../../../docs/results-organization.md).
They must not be mixed into a macro-F1 table.

## Linear-probing classification baseline in Macro-F1

Source:
`experiments/benchmark/results/rebuttal_linear_probing_macro_f1.csv`.

| Model | Balanced | Unbalanced | All |
| --- | ---: | ---: | ---: |
| AVES-Core | 67.39 ± 2.14 | 51.00 ± 2.68 | 51.57 ± 2.34 |
| BioLingual | 70.98 ± 2.05 | 52.93 ± 2.11 | 54.00 ± 2.77 |
| AVES-Bio | 75.13 ± 2.03 | 51.24 ± 2.52 | 52.69 ± 2.31 |

The point estimate comes from one deterministic logistic-regression fit after
selecting C on the validation set. The standard deviation is estimated by
bootstrap resampling of the test set (1,000 resamples for each of 10 RNG
seeds); it is not variation across 10 independently trained probes.
BioLingual audio is processed at 48 kHz even though the CSV currently records
44.1 kHz in its `target_sample_rate` metadata field.

## End-to-end fine-tuning

Classification source:
`experiments/finetuning/results/classification_v2/<model>/<config>/seed_<seed>/results.json`.
Wav2Vec2 source: `experiments/finetuning/results/wav2vec2_ow/all/seed_<seed>/results.json`.
Each result below uses seeds 42, 43, and 44; its standard deviation is computed
across those three fine-tuning runs.

| Model | Config | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| AVES-Core | balanced | 77.63 ± 0.93 | 77.37 ± 0.85 |
| AVES-Core | unbalanced | 74.70 ± 1.65 | 61.24 ± 4.58 |
| AVES-Core | all | 78.11 ± 0.62 | 57.29 ± 1.73 |
| BioLingual | balanced | 79.33 ± 1.46 | 78.99 ± 1.56 |
| BioLingual | unbalanced | 75.40 ± 1.96 | 57.55 ± 1.46 |
| BioLingual | all | 81.99 ± 0.64 | 54.32 ± 2.79 |
| AVES-Bio | balanced | 79.70 ± 0.68 | 79.60 ± 0.62 |
| AVES-Bio | unbalanced | 75.84 ± 0.80 | 62.06 ± 0.89 |
| AVES-Bio | all | 79.65 ± 0.91 | 59.54 ± 1.05 |
| Wav2Vec2 OpenWhistle | all | 82.58 ± 0.56 | 62.82 ± 0.40 |

Detection sources:

- `experiments/finetuning/results/aves_core/detection_map`
- `experiments/finetuning/results/biolingual/detection_map_b8_lr1e5`
- `experiments/finetuning/results/aves_bio/detection_map`
- `experiments/finetuning/results/wav2vec2_ow/detection`

| Model | Detection (mAP) |
| --- | ---: |
| AVES-Core | 74.10 ± 1.26 |
| BioLingual | 72.26 ± 0.44 |
| AVES-Bio | 77.02 ± 1.31 |
| Wav2Vec2 OpenWhistle | 82.27 ± 0.37 |

## Fine-tuning protocol

The encoder and dropout-linear classification head were optimized end-to-end.
The classification head has 6 outputs for `balanced` and 10 outputs for
`unbalanced`/`all`; the detection head has 7 outputs. AVES/BioLingual classification used a
batch size of 32, up to 50 epochs, patience 15, and three seeds. Encoder
learning rates were 1e-5 for AVES-Core/AVES-Bio and 5e-6 for BioLingual; the
head learning rate was 1e-4.
