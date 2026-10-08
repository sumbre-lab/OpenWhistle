# Pretraining checkpoints used in the rebuttal

The mapping below comes from the rebuttal PDF and the recorded
`results/hf_collections_benchmark.csv` (`masked-padding-btb3-v5`). Scores have
been checked against the stored CSV, not recomputed on the full benchmark.
Classification uses the full ten-class `all` split and macro-F1; detection uses
`default` and mAP. The dispersion is `test_std` as recorded by the original runner.

| Rebuttal row | Exact model ID under `dolphinteam/` | Macro-F1 all | Detection mAP |
|---|---|---:|---:|
| AVES from scratch | `AVES-OpenWhistle-Stage-2-320-100pct` | 61.26 ± 3.86 | 76.18 ± 2.09 |
| AVES continual | `AVES-bio-OpenWhistle-100pct-44kHz` (BTB3 concat) | 63.53 ± 2.93 | 73.93 ± 2.13 |
| Wav2Vec2 10% | `wav2vec2-44k-stride960-10pct-40k` | 22.65 ± 1.07 | 20.76 ± 1.34 |
| Wav2Vec2 50% | `wav2vec2-44k-stride960-50pct-200k` | 56.53 ± 2.61 | 77.03 ± 2.09 |
| Wav2Vec2 100% | `OpenWhistle-Wav2Vec2.0` | 64.37 ± 2.67 | 75.76 ± 2.02 |

Do not substitute `wav2vec2-44k-stride960-transformers` for the last row: that
checkpoint has different weights and gives 60.57 ± 2.98 macro-F1 / 80.80 ± 1.91
mAP in the stored CSV. Both remain in the complete collection table.
Likewise, AVES Stage-1 stride 960 is an intermediate variant (54.43 ± 2.98
macro-F1 / 66.51 ± 2.27 mAP), not the from-scratch result in the rebuttal.

## Native AVES loading

Stage-1 and Stage-2 contain TorchAudio weights plus a model configuration under
`models/` and `config/`; they are not Transformers exports. The frozen probe
uses `--model aves_hf` and the existing AVES loader. Downloads are pinned to
the revisions in `AVES_SCRATCH_MODELS`. Sampling rate is 44.1 kHz. The config's
`sample_rate` metadata is removed from a temporary copy before passing it to
TorchAudio; the downloaded Hub blob is never rewritten. Native inference uses
float32, even when the collection runner enables mixed precision for its
Transformers backbones. The reported precision now reflects the actual backend.

`HF_COLLECTION_MODELS` remains the Transformers-only list used for supervised
fine-tuning. `FROZEN_COLLECTION_MODELS` adds the two native AVES checkpoints for
linear probing and tables; this does not introduce unsupported native exports
into the HF fine-tuning runner.

## Commands

Rebuild the exact rebuttal table from existing results:

```bash
python experiments/benchmark/src/build_rebuttal_table.py --table pretraining
```

The resulting [Markdown table](results/rebuttal_pretraining.md) and
[LaTeX table](results/rebuttal_pretraining.tex) explicitly select the intended
five checkpoints. Detection scores for the smaller Wav2Vec2 budgets are
included from the local CSV; the rebuttal's size-ablation table reports only
classification for those budgets.

Inspect the ten evaluation commands (five checkpoints × two tasks):

```bash
python experiments/benchmark/src/run_hf_collections.py \
  --collection rebuttal_pretraining --classification_configs all \
  --embedding_batch_size 64 --results_csv /tmp/pretraining-rerun.csv --dry_run
```

Remove `--dry_run` to execute in the benchmark environment with dataset/model
access. The separate output CSV keeps historical reports intact. Rebuild its
table with `--table pretraining --results_csv /tmp/pretraining-rerun.csv`, using
`--markdown` and `--latex` to choose separate output paths if desired.

To inspect Stage-1 separately, use the full collection and select its exact ID:

```bash
python experiments/benchmark/src/run_hf_collections.py \
  --model_ids dolphinteam/AVES-OpenWhistle-Stage-1-960-100pct \
  --classification_configs all --results_csv /tmp/aves-stage1-rerun.csv --dry_run
```

## Verified provenance and remaining work

On 2026-10-05, the authenticated Hub metadata confirms both AVES exports and
Wav2Vec2 10%, 50%, and the alternative 100% export are **private**. The main
`OpenWhistle-Wav2Vec2.0` export is public. This records their current access,
not a change to visibility. They need appropriate public access for release.
Revisions, weight hashes and cached architecture metadata are recorded in
[`docs/provenance/pretraining-checkpoints.json`](../../docs/provenance/pretraining-checkpoints.json).
The historical score CSV does not record model revisions, so the revisions
observed today must not be treated as proven revisions of those old runs.

The local Stage-2 weight file has SHA-256
`aacb6b474be576c2340879a43e841a83574cd6b43f1d610fc56fcf877e74fe15`, matching
the Hub's weight metadata. A CPU smoke test with those actual weights produced
finite 768-dimensional embeddings for a synthetic 0.5-second waveform, through
both single-example and two-example batch inference. Hub downloads were
redirected to the matching existing local weights for that check; no weight
download or full dataset evaluation was performed.

The four inspected Wav2Vec2 exports have the same principal encoder dimensions,
convolutional kernels and total stride 960 at 44.1 kHz. However, the actual
pretraining manifests, random sampling seed, stratification implementation and
training logs for the 10/50/100% comparison remain to be identified. Model names
alone do not verify the year/channel/duration stratification promised in the
rebuttal, nor equal training policy across budgets.

The main 100% export's cached config contains a source-path reference ending in
`checkpoint-182000`, while the preprint describes 400k training steps. This
could reflect selection of an earlier checkpoint or stale export metadata;
the config alone cannot establish which explanation is correct. Locate the
source checkpoint/trainer state and document the actual selection rule before
claiming the final training recipe is reproduced.

Update, 2026-10-07: SSH verification found identical main-model weights on
Jean Zay, with `trainer_state.json` reporting **191,000 completed steps** and
400,000 planned steps. The 10% and 50% weights match exports from a separate
Fairseq pipeline. See the [cluster verification](../../docs/provenance/jeanzay-pretraining-verification.md).

The AVES recipe must likewise include its MFCC and Stage-2 HuBERT pseudo-label
generation and the exact checkpoint selection. The local Stage-2 model card
identifies `checkpoint_best.pt`, not the final 100k-update checkpoint. Its old
model-card evaluation uses different selection/weighting and yields 75.51 mAP;
the rebuttal's 76.18 mAP is the later v5 CSV result. Keep those protocols separate.
