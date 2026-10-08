# Rebuttal: CNN performance versus training-set size

## Question

> The detection CNN works quite well: how much data is actually required to
> achieve this level of success? It would be good to plot performance as a
> function of training set size to understand this.

## Proposed experiment

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

## Intended commands

Do not run these commands until the protocol has been approved.

One-seed pilot:

```bash
conda run -n openwhistle-cnn \
  python cnn/learning_curve/run_learning_curve.py --seeds 7
```

Full experiment (15 trainings):

```bash
conda run -n openwhistle-cnn \
  python cnn/learning_curve/run_learning_curve.py
```

The command is resumable: completed combinations are skipped. `--plot-only`
rebuilds the tables and figure, while `--dry-run` only lists the planned runs.
All outputs stay under `cnn/learning_curve/results/`.

Expected outputs:

- `learning_curve_test_f1.png` and `.pdf`;
- `learning_curve_runs.csv`, one row per run;
- `learning_curve_summary.csv`, mean and standard deviation per size;
- `protocol.json`;
- normal checkpoints and reports under one directory per size and seed.

## Suggested rebuttal text

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
