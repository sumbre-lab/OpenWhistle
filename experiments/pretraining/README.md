# Pretraining

The pretraining code now accepts either:

- a local Hugging Face dataset directory saved with `save_to_disk(...)`
- a remote Hugging Face dataset id such as `OpenWhistleNeurIPS26/OpenWhistle-Pretraining`

For the OpenWhistle pretraining corpus, the recommended source is:

- dataset id: `OpenWhistleNeurIPS26/OpenWhistle-Pretraining`
- config: `default`

The dataset exposes raw `audio` at 96 kHz plus metadata columns. During
training, the loader now accepts that schema directly and casts the audio to the
44.1 kHz rate expected by the Wav2Vec2 preprocessor.

## Example

With the submitit launcher:

```bash
python experiments/pretraining/ANNpretraining/pretraining/submit_dolphin.py \
  --path_data OpenWhistleNeurIPS26/OpenWhistle-Pretraining \
  --path_data_config default
```

For a small smoke test, use the deterministic review subset:

```bash
python experiments/pretraining/ANNpretraining/pretraining/submit_dolphin.py \
  --path_data OpenWhistleNeurIPS26/OpenWhistle-Pretraining \
  --path_data_config review-sample
```

## Anonymous Defaults

The pretraining code has been cleaned so it no longer depends on personal paths,
hard-coded SLURM accounts, or a committed Hugging Face token.

- Local artifacts default to `experiments/pretraining/artifacts/`
- Hugging Face authentication is read from `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`
- Hugging Face repo ownership is read from `HF_REPO_OWNER`, `HF_USERNAME`, or `HF_NAMESPACE`
- Optional cluster settings are read from `OPENWHISTLE_SLURM_PARTITION` and `OPENWHISTLE_SLURM_ACCOUNT`
