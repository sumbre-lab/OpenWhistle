# Camera-ready branch review

Reviewed on 2026-10-05 after fetching every branch from `origin`.
Integration target: `camera-ready/pablo`, based on `462149c` (also `origin/main`).
The original working tree on local `pablo/main` was preserved.

## Branch inventory and decisions

| Branch | Tip | Changes outside camera-ready | Decision |
|---|---|---|---|
| `origin/main` | `462149c` | None | Keep current public README, citation and links. |
| `origin/neurips/main` | `0d44606` | None; all commits are ancestors | Already included. |
| `origin/pablo/main` | `6adcac1` | None; all commits are ancestors | Already includes external inference helper and pretraining launch fixes. |
| local `main` | historical | None; ancestor | No integration required. |
| local `pablo/main` | `e432f97` | Uncommitted rebuttal code and results | Import changed CNN/benchmark code, learning curves, fine-tuning, checkpoint registry, BTB3 transform, ROC code, structured reports and figures. |
| `origin/faadil/F1` | `bbe3eee` | 2 commits: macro-F1/config selection, pretraining-size models | Local rebuttal implementation already supplies macro-F1 and a broader checkpoint runner. Add `--classification_config` compatibility alias and adapted `run_pretrain_variants.sh`; preserve the branch's three separately named F1/accuracy result files. |
| `origin/faadil/leave-one-out` | `9326bda` | 1 commit: open-set probe | Import open-set script, runner and existing results; expose dataset config and label validation selection as macro-F1 after integration. |
| `origin/fm/umap` | `43b1ee7` | 1 commit: embedding UMAP | Import plotter and figure; add explicit split selection to the current loader while retaining its default train/test behavior. |
| `origin/fm/finetuning` | `b2c16c4` | 2 commits against an old repository layout | Historical implementation, superseded by local `experiments/finetuning/`. Do not import the obsolete top-level `benchmark/` tree: hardcoded `/lustre/` and `/media/` paths, local CSV/audio dependencies, old evaluation protocol and additional dependencies. Original commits remain accessible on the branch. |
| `origin/camera-ready/pablo` | `462149c` | Target | Integrate selectively to retain the current layout and avoid regressions. |

## Integration choices

- Keep both classification and detection in `run_all.sh`; the F1 branch had disabled detection.
- Keep balanced as the standard classification default; all/unbalanced remain explicit options.
- Keep the local `masked-padding-btb3-v5` pipeline: padded-frame masking, batched embeddings, and three 16 kHz BTB3 views pooled separately and concatenated. The old F1 branch fed all AVES variants 44.1 kHz directly, including BTB3 checkpoints. Those historical numbers must not replace v5 results.
- Use the shared `HF_COLLECTION_MODELS` registry for both linear probing and fine-tuning instead of duplicating the old variant classes/configuration.
- Preserve the SLURM launcher's current configurable defaults; do not introduce local `gpu_p2` / `ioc@v100` defaults. The import-path and `slurm_account` fixes already exist in main.
- Align code and examples with the `dolphinteam/` resources already linked by the public README. Original result reports retain the IDs recorded when they were generated.
- Retain existing reports, CSVs and figures. Exclude weights, embedding caches, Python caches and training logs. Correct `.gitignore` paths for `datasets_figures/`; existing tracked figures remain tracked.
- Correct multiclass PR average precision to use binarized targets, accept both CLAP feature return formats, and fail benchmark extraction if an individual audio example fails rather than reporting metrics on an undeclared partial split.
- Correct LaTeX escaping of percent signs in generated ablation tables.

## Result provenance

`results/hf_collections_benchmark.csv` and its Markdown/LaTeX tables are the local v5 frozen-embedding ablation results. `results/rebuttal_linear_probing*.csv`, baseline detection CSVs, per-run `linear_parts/`, and ROC exports are preserved from the local rebuttal workspace.

`results/results_classification_{all_f1,balanced_f1,balanced_accuracy}.txt` come from `origin/faadil/F1`. The branch's replacement of the generic `results_classification.txt` and PR plots was not imported: it would overwrite existing outputs with ambiguous configuration/metric provenance. All content remains on the source branch.

`results/results_openset.txt` comes from `origin/faadil/leave-one-out`, before the integrated macro-F1 selection change. Its original validation metric is accuracy. Re-running the imported script now uses macro-F1, so these historical numbers are not claimed to be reproduced by the updated selector. The rejection threshold is calibrated before refitting, following the original rebuttal implementation; acceptance on the refitted model is not guaranteed.

`results/umap_dolph2vec_classification.png` comes from `origin/fm/umap`. UMAP uses all three splits by default and is descriptive.

`experiments/finetuning/results/` and `cnn/rebuttal/results/` contain existing lightweight reports and figures, imported without re-running training. No historical numbers were rewritten to match new protocols.

## Validation performed

- Parsed all 65 Python files and checked every shell script with `bash -n`.
- Five regression tests passed: macro-F1 selection, multiclass micro-AP, UMAP split selection, rejection of partial embedding extraction, and CLAP tensor/pooled output compatibility.
- Benchmark, open-set, UMAP, ROC and supervised fine-tuning CLI help checks passed. Classification/detection checkpoint runner dry-runs passed, including BTB3 configuration.
- Rebuilt the ablation Markdown/LaTeX tables and the seven-model ROC comparison from the imported reports.
- Validated 89 CNN/fine-tuning JSON reports and all local Markdown links. `git diff --check` passed.
- CNN learning-curve CLI execution is not validated: the current interpreter lacks the declared `wandb` dependency. Pretraining execution is not validated: it lacks `submitit`. No GPU training or full dataset/model download was run.

## Remaining release checks

- Choose the exact results and experiments referenced by the final manuscript; the manuscript sources and reviewer instructions are not in this repository.
- Confirm public access to all `dolphinteam/` datasets, base checkpoints and ablation checkpoints. Code/README consistency alone does not establish access or equivalence of migrated datasets.
- Run a clean-environment end-to-end smoke test, then reproduce the final selected GPU experiments. Local structural/synthetic checks do not reproduce published scores.
- Record exact dependency versions, model revisions and dataset revisions for the final release. Current requirements leave several core ML packages unpinned.
- Decide the code license: `LICENSE-DATA` specifies CC BY 4.0 for datasets, but the repository has no explicit code license.
- Regenerate open-set results if the manuscript uses macro-F1 selection; preserve historical accuracy-selected reports as such.
