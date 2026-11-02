# ruff: noqa: E402  # Check the native Python environment before importing Fairseq.
"""Native model and conversion checks using small random models, no downloads."""

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

if sys.version_info[:2] != (3, 10):
    raise unittest.SkipTest(
        "Native pretraining tests require the Python 3.10 environment."
    )

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf
import torch
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model
from fairseq.criterions.wav2vec_criterion import Wav2vecCriterion
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining

ROOT = Path(__file__).resolve().parents[1] / "experiments/pretraining"
EXPORT = ROOT / "src/export_model.py"
spec = importlib.util.spec_from_file_location("strict_export", EXPORT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
torch.set_num_threads(1)


def tiny_model():
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = Wav2Vec2Config(
        conv_feature_layers=str([(8, 30, 15)] + [(8, 3, 2)] * 4 + [(8, 2, 2)] * 2),
        encoder_layers=2,
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=32,
        encoder_attention_heads=2,
        conv_pos=8,
        conv_pos_groups=2,
        layer_norm_first=True,
        quantize_targets=True,
        latent_vars=4,
        latent_groups=2,
        final_dim=8,
        num_negatives=2,
        mask_length=2,
        feature_grad_mult=0.1,
        required_seq_len_multiple=2,
    )
    return cfg, Wav2Vec2Model(cfg)


def test_complete_conversion_preserves_embeddings():
    cfg, native = tiny_model()
    data = dataclasses.asdict(cfg)
    hf = Wav2Vec2ForPreTraining(export.hf_config(data))
    assert export.convert_state(native.state_dict(), hf) == len(hf.state_dict())
    parity = export.verify_embeddings(native.state_dict(), data, hf, sample_rate=8000)
    assert parity["max_absolute_error"] < 1e-4


@pytest.mark.parametrize("damage", ["missing", "shape", "extra"])
def test_invalid_conversion_is_rejected(damage):
    cfg, native = tiny_model()
    hf = Wav2Vec2ForPreTraining(export.hf_config(dataclasses.asdict(cfg)))
    state = dict(native.state_dict())
    if damage == "missing":
        del state["final_proj.weight"]
    elif damage == "shape":
        state["final_proj.weight"] = state["final_proj.weight"][:1]
    else:
        state["unmapped_parameter"] = torch.zeros(1)
    with pytest.raises(ValueError):
        export.convert_state(state, hf)


def test_native_pretraining_has_finite_gradients_and_temperature_schedule():
    cfg, native = tiny_model()
    criterion = Wav2vecCriterion(
        None, infonce=True, loss_weights=[0.1, 10], log_keys=["temp"]
    )
    optimizer = torch.optim.AdamW(native.parameters(), lr=0.0005)
    original = native.final_proj.weight.detach().clone()
    for step in (0, 191000):
        native.set_num_updates(step)
        assert native.quantizer.curr_temp == pytest.approx(max(2 * 0.999995**step, 0.5))
        optimizer.zero_grad()
        loss, size, logs = criterion(
            native,
            {"id": torch.arange(2), "net_input": {"source": torch.randn(2, 16000)}},
        )
        assert size > 0 and torch.isfinite(loss)
        (loss / size).backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in native.parameters()
            if p.grad is not None
        )
        optimizer.step()
    assert not torch.equal(native.final_proj.weight, original)


@pytest.mark.parametrize("normalize", [True, False])
def test_checkpoint_cli_exports_provenance_and_reloadable_weights(tmp_path, normalize):
    cfg, native = tiny_model()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": native.state_dict(),
            "cfg": {
                "model": dataclasses.asdict(cfg),
                "task": {"sample_rate": 8000, "normalize": normalize},
            },
            "optimizer_history": [{"num_updates": 20}],
        },
        checkpoint,
    )
    destination = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            str(EXPORT),
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(destination),
            "--sample_rate",
            "8000",
            "--stride",
            "960",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((destination / "export_provenance.json").read_text())
    assert meta["num_updates"] == 20 and len(meta["checkpoint_sha256"]) == 64
    assert meta["normalize"] == normalize
    assert (
        Wav2Vec2FeatureExtractor.from_pretrained(destination).do_normalize == normalize
    )
    loaded = Wav2Vec2ForPreTraining.from_pretrained(destination)
    hf = Wav2Vec2ForPreTraining(export.hf_config(dataclasses.asdict(cfg)))
    export.convert_state(native.state_dict(), hf)
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, hf.state_dict()[name])


def test_export_rejects_nonpaper_stride(tmp_path):
    cfg, native = tiny_model()
    data = dataclasses.asdict(cfg)
    # Shape-compatible first convolution, but a different temporal stride.
    data["conv_feature_layers"] = str([(8, 30, 5)] + [(8, 3, 2)] * 4 + [(8, 2, 2)] * 2)
    checkpoint = tmp_path / "other_stride.pt"
    torch.save({"model": native.state_dict(), "cfg": {"model": data}}, checkpoint)
    destination = tmp_path / "export"
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORT),
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Actual stride 320 differs from --stride 960" in result.stderr
    assert not destination.exists()


def test_native_training_cli_saves_and_resumes_with_literal_paths(tmp_path):
    cfg, _ = tiny_model()
    recipe = OmegaConf.load(ROOT / "src/pretraining_config.yaml")
    recipe.model = OmegaConf.create(dataclasses.asdict(cfg))
    recipe.model._name = "wav2vec2"
    recipe.common.cpu = True
    recipe.common.fp16 = False
    recipe.task.min_sample_size = 12000
    recipe.lr_scheduler.warmup_updates = 0
    recipe.optimization.max_update = 2
    recipe.checkpoint.save_interval_updates = 1
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    OmegaConf.save(recipe, config_dir / "tiny.yaml")
    data = tmp_path / "data's [test]"
    data.mkdir()
    generator = np.random.default_rng(42)
    for split, count in (("train", 8), ("valid", 2)):
        rows = [str(data)]
        for index in range(count):
            name = f"{split}_{index}.flac"
            sf.write(data / name, generator.normal(0, 0.15, 16000), 44100)
            rows.append(f"{name}\t16000")
        (data / f"{split}.tsv").write_text("\n".join(rows) + "\n")
    env = dict(os.environ, HYDRA_FULL_ERROR="1", OMP_NUM_THREADS="1")
    output_dir = "checkpoints/run's [test]"
    command = [
        sys.executable,
        str(ROOT / "src/run_pretraining.py"),
        "--data_dir",
        str(data),
        "--output_dir",
        output_dir,
        "--config",
        str(config_dir / "tiny.yaml"),
        "--max_tokens",
        "32000",
        "--max_sample_size",
        "16000",
        "--no-fp16",
    ]
    first = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert first.returncode == 0, first.stdout[-5000:] + first.stderr
    checkpoint = tmp_path / output_dir / "checkpoint_last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["optimizer_history"][-1]["num_updates"] == 2
    assert state["last_optimizer_state"]
    resumed = subprocess.run(
        command + ["optimization.max_update=3"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stdout[-5000:] + resumed.stderr
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["optimizer_history"][-1]["num_updates"] == 3
    assert "loaded checkpoint" in resumed.stdout.lower()
    assert all(torch.isfinite(value).all() for value in state["model"].values())
