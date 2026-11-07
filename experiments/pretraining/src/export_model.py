#!/usr/bin/env python3
"""Export native Fairseq weights after strict mapping and embedding parity."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import torch
import transformers
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
)


def model_config(checkpoint):
    cfg = checkpoint.get("cfg")
    if cfg is not None and cfg.get("model") is not None:
        from omegaconf import OmegaConf

        model = cfg["model"]
        return (
            OmegaConf.to_container(model, resolve=True)
            if OmegaConf.is_config(model)
            else dict(model)
        )
    legacy = checkpoint.get("args")
    if legacy is not None:
        return vars(legacy) if not isinstance(legacy, dict) else legacy
    raise ValueError("Checkpoint model configuration (cfg or args) is required.")


def hf_config(cfg):
    for name, expected in {
        "quantize_input": False,
        "target_glu": False,
        "quantizer_depth": 1,
        "pos_conv_depth": 1,
        "layer_type": "transformer",
        "conv_bias": False,
    }.items():
        if cfg.get(name, expected) != expected:
            raise ValueError(f"Unsupported architecture: {name}={cfg[name]!r}")
    if not cfg.get("quantize_targets", False):
        raise ValueError("Expected quantized Wav2Vec2 pretraining targets.")
    conv = ast.literal_eval(cfg["conv_feature_layers"])
    final_dim = cfg.get("final_dim", 0)
    if final_dim <= 0:
        final_dim = cfg["encoder_embed_dim"]
    latent_dim = cfg.get("latent_dim", 0)
    if latent_dim <= 0:
        latent_dim = final_dim
    return Wav2Vec2Config(
        hidden_size=cfg["encoder_embed_dim"],
        num_hidden_layers=cfg["encoder_layers"],
        intermediate_size=cfg["encoder_ffn_embed_dim"],
        num_attention_heads=cfg["encoder_attention_heads"],
        hidden_act=cfg.get("activation_fn", "gelu"),
        hidden_dropout=cfg.get("dropout", 0.1),
        attention_dropout=cfg.get("attention_dropout", 0.1),
        activation_dropout=cfg.get("activation_dropout", 0.0),
        feat_proj_dropout=cfg.get("dropout_input", 0.0),
        feat_quantizer_dropout=cfg.get("dropout_features", 0.0),
        layerdrop=cfg.get("encoder_layerdrop", 0.0),
        feat_extract_activation="gelu",
        feat_extract_norm="group"
        if cfg.get("extractor_mode", "default") == "default"
        else "layer",
        conv_dim=[c[0] for c in conv],
        conv_kernel=[c[1] for c in conv],
        conv_stride=[c[2] for c in conv],
        conv_bias=False,
        num_conv_pos_embeddings=cfg.get("conv_pos", 128),
        num_conv_pos_embedding_groups=cfg.get("conv_pos_groups", 16),
        do_stable_layer_norm=cfg.get("layer_norm_first", False),
        mask_time_prob=cfg.get("mask_prob", 0.65),
        mask_time_length=cfg.get("mask_length", 10),
        mask_time_min_masks=2,
        mask_feature_prob=cfg.get("mask_channel_prob", 0.0),
        mask_feature_length=cfg.get("mask_channel_length", 10),
        num_codevector_groups=cfg.get("latent_groups", 2),
        num_codevectors_per_group=cfg.get("latent_vars", 320),
        codevector_dim=latent_dim,
        proj_codevector_dim=final_dim,
        diversity_loss_weight=0.1,
        contrastive_logits_temperature=cfg.get("logit_temp", 0.1),
        num_negatives=cfg.get("num_negatives", 100),
        return_dict=True,
    )


def convert_state(fs_sd, model):
    target = model.state_dict()
    converted, consumed = {}, set()

    def copy(src, dst):
        sources = (src,) if isinstance(src, str) else src
        destinations = (dst,) if isinstance(dst, str) else dst
        found_src = next((s for s in sources if s in fs_sd), None)
        found_dst = next((d for d in destinations if d in target), None)
        if found_src is None or found_dst is None:
            raise ValueError(f"Missing mapping: {sources} -> {destinations}")
        if fs_sd[found_src].shape != target[found_dst].shape:
            raise ValueError(f"Shape mismatch: {found_src} -> {found_dst}")
        converted[found_dst] = fs_sd[found_src]
        consumed.add(found_src)

    for i in range(len(model.config.conv_dim)):
        copy(
            f"feature_extractor.conv_layers.{i}.0.weight",
            f"wav2vec2.feature_extractor.conv_layers.{i}.conv.weight",
        )
        if model.config.feat_extract_norm == "layer" or i == 0:
            norm = "2.1" if model.config.feat_extract_norm == "layer" else "2"
            for suffix in ("weight", "bias"):
                copy(
                    f"feature_extractor.conv_layers.{i}.{norm}.{suffix}",
                    f"wav2vec2.feature_extractor.conv_layers.{i}.layer_norm.{suffix}",
                )
    for suffix in ("weight", "bias"):
        copy(f"layer_norm.{suffix}", f"wav2vec2.feature_projection.layer_norm.{suffix}")
        copy(
            f"post_extract_proj.{suffix}",
            f"wav2vec2.feature_projection.projection.{suffix}",
        )
        copy(f"encoder.layer_norm.{suffix}", f"wav2vec2.encoder.layer_norm.{suffix}")
    if "wav2vec2.masked_spec_embed" in target:
        copy("mask_emb", "wav2vec2.masked_spec_embed")
    for old, new in (("weight_g", "original0"), ("weight_v", "original1")):
        copy(
            (
                f"encoder.pos_conv.0.{old}",
                f"encoder.pos_conv.0.parametrizations.weight.{new}",
            ),
            (
                f"wav2vec2.encoder.pos_conv_embed.conv.{old}",
                f"wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.{new}",
            ),
        )
    copy("encoder.pos_conv.0.bias", "wav2vec2.encoder.pos_conv_embed.conv.bias")
    for i in range(model.config.num_hidden_layers):
        fs, hf = f"encoder.layers.{i}", f"wav2vec2.encoder.layers.{i}"
        names = {
            f"self_attn.{p}": f"attention.{p}"
            for p in ("q_proj", "k_proj", "v_proj", "out_proj")
        }
        names.update(
            {
                "self_attn_layer_norm": "layer_norm",
                "final_layer_norm": "final_layer_norm",
                "fc1": "feed_forward.intermediate_dense",
                "fc2": "feed_forward.output_dense",
            }
        )
        for src, dst in names.items():
            for suffix in ("weight", "bias"):
                copy(f"{fs}.{src}.{suffix}", f"{hf}.{dst}.{suffix}")
    copy("quantizer.vars", "quantizer.codevectors")
    for src, dst in (
        ("quantizer.weight_proj", "quantizer.weight_proj"),
        ("project_q", "project_q"),
        ("final_proj", "project_hid"),
    ):
        for suffix in ("weight", "bias"):
            copy(f"{src}.{suffix}", f"{dst}.{suffix}")
    missing, unused = set(target) - set(converted), set(fs_sd) - consumed
    unused -= {"_float_tensor"}  # Historical non-parameter dtype marker.
    if missing or unused:
        raise ValueError(
            f"Incomplete conversion: missing HF={sorted(missing)}, unused Fairseq={sorted(unused)}"
        )
    model.load_state_dict(converted, strict=True)
    return len(converted)


def verify_embeddings(fs_sd, cfg, hf_model, sample_rate, normalize=True):
    import fairseq
    from fairseq.models.wav2vec.wav2vec2 import (
        Wav2Vec2Config as FSConfig,
        Wav2Vec2Model,
    )

    allowed = {f.name for f in dataclasses.fields(FSConfig)}
    native = Wav2Vec2Model(FSConfig(**{k: v for k, v in cfg.items() if k in allowed}))
    native.load_state_dict(
        {k: v for k, v in fs_sd.items() if k != "_float_tensor"}, strict=True
    )
    native.float().eval()
    hf_model.float().eval()
    generator = torch.Generator().manual_seed(42)
    largest_error = 0.0
    with torch.no_grad():
        for duration in (0.5, 0.73):
            audio = torch.randn(2, int(sample_rate * duration), generator=generator)
            if normalize:
                audio = (audio - audio.mean(-1, keepdim=True)) / (
                    audio.var(-1, keepdim=True, unbiased=False) + 1e-7
                ).sqrt()
            expected = native.extract_features(audio, padding_mask=None, mask=False)[
                "x"
            ]
            actual = hf_model.wav2vec2(audio).last_hidden_state
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            largest_error = max(largest_error, (actual - expected).abs().max().item())
    return {
        "max_absolute_error": largest_error,
        "atol": 1e-4,
        "rtol": 1e-4,
        "fairseq_version": fairseq.__version__,
        "inputs": f"two seeded {'normalized' if normalize else 'raw'} batches; no waveform padding",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument(
        "--stride",
        type=int,
        choices=[960],
        default=960,
        help="Expected OpenWhistle architecture stride (960 samples).",
    )
    p.add_argument("--sample_rate", type=int, default=44100)
    args = p.parse_args()
    if args.sample_rate <= 0:
        p.error("--sample_rate must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = model_config(checkpoint)
    config = hf_config(cfg)
    stride = math.prod(config.conv_stride)
    if args.stride is not None and args.stride != stride:
        raise ValueError(f"Actual stride {stride} differs from --stride {args.stride}")
    task = (checkpoint.get("cfg") or {}).get("task", {})
    if task.get("sample_rate", args.sample_rate) != args.sample_rate:
        raise ValueError(
            "Sample rate disagrees with the checkpoint task configuration."
        )
    normalize = task.get("normalize", True)
    model = Wav2Vec2ForPreTraining(config)
    count = convert_state(checkpoint["model"], model)
    parity = verify_embeddings(
        checkpoint["model"], cfg, model, args.sample_rate, normalize=normalize
    )
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    Wav2Vec2FeatureExtractor(
        sampling_rate=args.sample_rate,
        do_normalize=normalize,
        return_attention_mask=False,
    ).save_pretrained(args.output_dir)
    history = checkpoint.get("optimizer_history") or [{}]
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "num_updates": history[-1].get("num_updates"),
        "fairseq_model_config": cfg,
        "sample_rate": args.sample_rate,
        "normalize": normalize,
        "stride": stride,
        "copied_tensors": count,
        "parity": parity,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "usage": "Inference export; Transformers pretraining does not reproduce the native Fairseq recipe.",
    }
    (args.output_dir / "export_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        f"Exported {count} tensors; max parity error {parity['max_absolute_error']:.3g}; {args.output_dir}"
    )


if __name__ == "__main__":
    main()
