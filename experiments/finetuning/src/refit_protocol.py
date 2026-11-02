"""Shared final train+validation refit for native and Hugging Face encoders."""
import json
import math

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from checkpoint_io import save_checkpoint


def add_refit_argument(parser):
    parser.add_argument("--refit_train_validation", action="store_true",
                        help="Select an epoch on validation, then restart on train+validation.")


def validate_refit_output(args):
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("Epochs and patience must be positive.")
    if args.refit_train_validation and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Train+validation refit requires a new or empty output directory.")


def refit_model(args, best_epoch, model_factory, loaders, loader_kwargs, train_epoch,
                device, set_seed, label_names, metric, validation_score):
    """Reinitialize all weights and optimizer state; never evaluate validation/test here."""
    set_seed(args.seed)
    model = model_factory().to(device)
    dataset = ConcatDataset([loaders["train"].dataset, loaders["validation"].dataset])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        [{"params": list(model.backbone_parameters()), "lr": args.learning_rate},
         {"params": list(model.head_parameters()), "lr": args.head_learning_rate}],
        weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(total_updates * args.warmup_ratio),
        num_training_steps=total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.mixed_precision == "float16")
    print(f"Refit from pretrained initialization: train+validation, {len(dataset)} clips, "
          f"{best_epoch} fixed epochs; schedule horizon={args.epochs} epochs.", flush=True)
    history = []
    for epoch in range(1, best_epoch + 1):
        loss = train_epoch(model, loader, optimizer, scheduler, scaler, device, args)
        history.append(dict(epoch=epoch, train_loss=loss))
        print(json.dumps(dict(phase="refit_train_validation", **history[-1])), flush=True)
    final_training = dict(split="train+validation", epochs=best_epoch, checkpoint="final.pt",
                          num_examples=len(dataset), seed=args.seed,
                          schedule_horizon_epochs=args.epochs, updates_per_epoch=updates_per_epoch)
    path = args.output_dir / "final.pt"
    save_checkpoint(dict(model=model.checkpoint_state(), epoch=best_epoch, label_names=label_names,
                    args=vars(args), final_training=final_training,
                    **{f"selection_validation_{metric}": validation_score}), path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_checkpoint_state(checkpoint["model"])
    return model, final_training, history
