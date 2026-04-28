# CNN Training

This folder contains the Torch/VGG16 CNN training code used for binary dolphin
whistle detection.

## Files

- `train.py`: end-to-end training and evaluation script.
- `modeling_components.py`: shared model and spectrogram utilities.
- `requirements.txt`: Python dependencies needed by this extraction.

## Dataset

By default, training reads the public Hugging Face dataset:

```bash
dolphinteam/OpenWhistle-1.0-CNN
```

Override it with `TRAIN_DATASET_SOURCE`. The value can be either a Hugging Face
dataset repo id or a local `datasets.DatasetDict` saved with `save_to_disk`.

## Example

```bash
git clone https://github.com/dolphinteam/OpenWhistle.git
cd OpenWhistle
python -m pip install -r experiments/cnn/requirements.txt
python experiments/cnn/train.py
```

Useful environment variables:

- `TRAIN_BATCH_SIZE` default: `4`
- `TRAIN_NUM_EPOCHS` default: `50`
- `TRAIN_PATIENCE` default: `10`
- `TRAIN_LEARNING_RATE` default: `1e-5`
- `TRAIN_INPUT_SOURCE` default: `spectrogram`, can be `audio` to regenerate
  spectrograms from waveform payloads.
- `TRAIN_PRETRAINED_BACKBONE` default: `1`
- `TRAIN_FREEZE_BACKBONE` default: `0`
- `TRAIN_NORMALIZATION_MEAN`/`TRAIN_NORMALIZATION_STD`: defaults are the
  torchvision ImageNet normalization used by pretrained image backbones.
- `TRAIN_EVAL_ONLY` default: `0`
- `TRAIN_CHECKPOINT_PATH` default: the run's `model_best.pt`
- `WANDB_ENABLED` default: `1`

The script writes checkpoints, figures, and reports under
`artifacts/cnn` relative to `experiments/cnn` by default.
