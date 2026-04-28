# CNN Training

This folder contains the Torch/VGG16 CNN training code used for binary dolphin
whistle detection.

## Files

- `train_torch.py`: end-to-end training and evaluation script.
- `whistle_torch.py`: shared model and spectrogram utilities.
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
cd /home/pablo/Documents/OpenWhistle
python experiments/cnn/train_torch.py
```

Useful environment variables:

- `TRAIN_BATCH_SIZE` default: `4`
- `TRAIN_NUM_EPOCHS` default: `50`
- `TRAIN_PATIENCE` default: `10`
- `TRAIN_LEARNING_RATE` default: `1e-5`
- `TRAIN_INPUT_SOURCE` default: `audio`, can be `spectrogram`
- `TRAIN_EVAL_ONLY` default: `0`
- `TRAIN_CHECKPOINT_PATH` default: `models/run3/model_vgg_best.pt`
- `WANDB_ENABLED` default: `1`

The script writes checkpoints, figures, and reports to `models/run3`,
`figs/run3`, and `reports/run3` relative to the working directory.
