# Dataset overview figures

This folder builds the two **OpenWhistle** manuscript figures that summarize the pretraining corpus and the classification finetuning subset. Outputs are written to `output/` (PNG and PDF).

## Scripts

From the repository root:

```bash
python datasets_figures/plot_dataset_overview.py
python datasets_figures/plot_classification_overview.py
```

Input data: CSVs under `data/` where a statistic is not taken directly from the Hugging Face datasets; the classification plot also loads the HF classification dataset by default.

---

## Figure captions

### `fig_dataset_overview`

**OpenWhistle: Longitudinal Extent and Temporal Distribution of the Dataset.**

- **A)** Cumulative recording hours over time, showing dataset growth and changes in pod composition.
- **B)** Distribution of recording hours across the day, indicating alignment with periods of human activity.
- **C)** Cumulative detected whistling hours over time, obtained by applying the whistle presence detection CNN to the raw recordings.
- **D)** Confusion matrix of the whistle presence detection CNN on the test set, indicating high reliability of the detected whistle segments used to derive panel C.

### `fig_classification_overview`

**Analyses of Whistle Properties in OpenWhistle.**

- **A–B)** Temporal structure of dolphin vocalizations: distributions of inter-whistle intervals (A) and whistle sequence durations (B).
- **C–E)** Properties of the expert-annotated subset: temporal coverage (C), class distribution (D), and whistle duration (E).
- **F)** Data quality assessment through signal-to-noise ratio (SNR) for the full dataset and the expert-annotated subset.
