# CNN performance versus training-set size

## Question

> The detection CNN works quite well: how much data is actually required to
> achieve this level of success? It would be good to plot performance as a
> function of training set size to understand this.

## Protocol

Train the published VGG16 detector on nested subsets containing 5%, 10%, 25%,
50%, and 100% of the training sessions, with three seeds per size.

The important controls are:

- reduce only the training split;
- keep the published validation and test splits identical for every run;
- sample entire sessions rather than individual 0.4 s windows;
- make subsets nested within each seed;
- keep architecture, preprocessing, optimizer, learning rate, batch size, and
  early-stopping rule fixed;
- report individual runs and mean test F1 ± standard deviation;
- put the actual number of sessions, windows, and window-hours in the CSV.

Sampling by session avoids treating strongly correlated windows from the same
recording as independent data. Because session lengths differ, the figure uses
the observed number of 0.4 s windows converted to window-hours on the x-axis.
This is not the duration of the original continuous recordings.

For the verbal answer to “how much data is required,” use the smallest subset
whose mean test F1 is within one absolute percentage point of the mean
full-data F1. This descriptive tolerance is fixed before seeing the results and
must be stated explicitly; the complete curve should still be reported.

## Why hyperparameters remain fixed

This experiment estimates a **fixed-recipe learning curve**: what happens when
the published detector and training recipe receive less data? Fixing the recipe
isolates the effect of data quantity.

It does not estimate the best possible performance at every dataset size.
Optimal learning rate, regularization, stopping schedule, model capacity, and
convolutional kernel design can depend on sample size. An independent grid
search at every point would answer that second question, but at much greater
computational cost and with additional model-selection variance. This
qualification must be stated in the rebuttal and figure caption.

## Reproduce the experiment

Run from the repository root in the CNN environment. New outputs default to
ignored `cnn/runs/learning_curve/`; saved manuscript results are kept separately.

### Rebuild the manuscript figure without training

```bash
conda run -n openwhistle-cnn \
  python cnn/learning_curve/run_learning_curve.py \
  --plot-only \
  --runs-csv cnn/learning_curve/results/learning_curve_runs.csv
```

This reads the 15 saved scores and rebuilds the summary and PNG/PDF figure in
`cnn/runs/learning_curve/`. It does not need the archived individual reports or
download audio or checkpoints. Fractions and seeds can select a subset of the CSV.

### Run new trainings

One-seed pilot with the current default dataset:

```bash
conda run -n openwhistle-cnn \
  python cnn/learning_curve/run_learning_curve.py --seeds 7
```

The saved experiment used `OpenWhistleNeurIPS26/OpenWhistle-CNN` and eight data
workers. Its recorded settings are in [protocol.json](results/protocol.json):

```bash
conda run -n openwhistle-cnn \
  python cnn/learning_curve/run_learning_curve.py \
  --dataset-source OpenWhistleNeurIPS26/OpenWhistle-CNN --num-workers 8
```

The historical dataset revision was not recorded. Equivalence with the current
`dolphinteam/OpenWhistle-CNN` default remains to be verified. Session sampling is
deterministic for a fixed dataset: sorted session IDs, a NumPy permutation with
the run seed, and the first `ceil(fraction * number_of_sessions)` sessions.

Training is resumable: completed combinations with matching settings are skipped.
`--dry-run` lists planned runs without training or writing results. `--plot-only`
without `--runs-csv` aggregates reports from the selected output directory.

### Files kept in the repository

```text
results/
├── protocol.json
├── learning_curve_runs.csv
├── learning_curve_summary.csv
├── learning_curve_test_f1.png
└── learning_curve_test_f1.pdf
```

The CSV keeps each seed's metrics, training-set sizes and best epoch. Individual
training reports and metadata are in the [local archive](../../docs/results-organization.md).
Their saved paths retain the historical `cnn/rebuttal/` location. New runs generate
checkpoints and reports under `fraction_*/seed_*/` in the ignored output folder.

## Rebuttal context

> We thank the reviewer for this suggestion. We added a learning-curve
> experiment in which the VGG16 detector is trained on nested, session-level
> subsets comprising 5%, 10%, 25%, 50%, and 100% of the training sessions,
> using three random seeds per size. Validation and test sets are held fixed and
> session-disjoint, and we report mean test F1 with standard deviation. To
> isolate the effect of training-data quantity, we keep the architecture and
> training hyperparameters fixed to those used for the reported model. This is
> therefore a conditional, fixed-recipe learning curve: because optimal
> hyperparameters may depend on sample size, it should not be interpreted as
> the maximum attainable performance at each size. Retuning a full grid at
> every size was computationally infeasible.
