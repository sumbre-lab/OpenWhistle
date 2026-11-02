"""Offline manifest integrity checks using synthetic audio rows."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1] / "experiments/pretraining"
spec = importlib.util.spec_from_file_location(
    "prepare_audio", ROOT / "src/prepare_data.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def options(tmp_path):
    return argparse.Namespace(
        dataset_name="synthetic",
        config_name="default",
        revision="fixed-commit",
        audio_dir=tmp_path / "audio",
        manifest_dir=tmp_path / "manifests",
        target_sample_rate=44100,
        audio_format="flac",
        overwrite=False,
        train_fraction=1.0,
        seed=42,
    )


def test_manifest_uses_real_file_lengths_and_records_dataset_revision(
    tmp_path, monkeypatch
):
    rows = [
        {"audio": {"array": np.linspace(-0.9, 0.9, n), "sampling_rate": 44100}}
        for n in (1000, 2000)
    ]
    monkeypatch.setattr(prepare, "load_split", lambda args, split: rows)
    args = options(tmp_path)
    prepare.prepare_split(args, "train", "train.tsv", None)
    lines = (args.manifest_dir / "train.tsv").read_text().splitlines()
    for row, n in zip(lines[1:], (1000, 2000)):
        name, count = row.split("\t")
        assert int(count) == sf.info(args.audio_dir / name).frames == n
    assert (
        json.loads((args.manifest_dir / "train.json").read_text())["dataset_revision"]
        == "fixed-commit"
    )
    with pytest.raises(FileExistsError):
        prepare.prepare_split(args, "train", "train.tsv", None)


def test_stale_audio_is_rejected_without_publishing_a_manifest(tmp_path, monkeypatch):
    args = options(tmp_path)
    wrong = args.audio_dir / "train/00000000.flac"
    wrong.parent.mkdir(parents=True)
    sf.write(wrong, np.zeros(50), 16000)
    monkeypatch.setattr(
        prepare,
        "load_split",
        lambda args, split: [
            {"audio": {"array": np.zeros(100), "sampling_rate": 44100}}
        ],
    )
    with pytest.raises(ValueError, match="Existing audio"):
        prepare.prepare_split(args, "train", "train.tsv", None)
    assert not (args.manifest_dir / "train.tsv").exists()


def test_nonfinite_audio_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        prepare.audio_array({"audio": {"array": [0.0, float("nan")]}})


def test_subsets_are_nested_and_reproducible():
    subsets = [
        {i for i in range(1000) if prepare.selected(i, 42, fraction)}
        for fraction in (0.1, 0.5, 1.0)
    ]
    assert subsets[0] < subsets[1] < subsets[2]
    assert subsets[0] == {i for i in range(1000) if prepare.selected(i, 42, 0.1)}


def test_same_length_stale_audio_is_rejected(tmp_path, monkeypatch):
    args = options(tmp_path)
    stale = args.audio_dir / "train/00000000.flac"
    stale.parent.mkdir(parents=True)
    sf.write(stale, np.zeros(100), 44100)
    rows = [{"audio": {"array": np.full(100, 0.25), "sampling_rate": 44100}}]
    monkeypatch.setattr(prepare, "load_split", lambda args, split: rows)
    with pytest.raises(ValueError, match="Existing audio"):
        prepare.prepare_split(args, "train", "train.tsv", None)
    assert not (args.manifest_dir / "train.tsv").exists()
    assert not (args.manifest_dir / "train.tsv.tmp").exists()


def test_existing_manifest_fails_before_loading_dataset(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.manifest_dir.mkdir()
    manifest = args.manifest_dir / "train.tsv"
    manifest.write_text("existing manifest\n")

    def unexpected_load(*args):
        pytest.fail("existing manifests must be checked before dataset access")

    monkeypatch.setattr(prepare, "load_split", unexpected_load)
    with pytest.raises(FileExistsError):
        prepare.prepare_split(args, "train", "train.tsv", None)
    assert manifest.read_text() == "existing manifest\n"


def test_row_limit_does_not_decode_an_extra_example(tmp_path, monkeypatch):
    def rows():
        yield {"audio": {"array": np.zeros(100), "sampling_rate": 44100}}
        raise AssertionError("an example beyond the requested limit was decoded")

    monkeypatch.setattr(prepare, "load_split", lambda args, split: rows())
    args = options(tmp_path)
    prepare.prepare_split(args, "train", "train.tsv", 1)
    assert len((args.manifest_dir / "train.tsv").read_text().splitlines()) == 2


def test_failed_preparation_keeps_published_manifest_and_cleans_temporary_file(
    tmp_path, monkeypatch
):
    args = options(tmp_path)
    args.overwrite = True
    args.manifest_dir.mkdir()
    manifest = args.manifest_dir / "train.tsv"
    manifest.write_text("previous manifest\n")
    monkeypatch.setattr(
        prepare,
        "load_split",
        lambda args, split: [
            {"audio": {"array": [float("nan")], "sampling_rate": 44100}},
        ],
    )
    with pytest.raises(ValueError, match="finite"):
        prepare.prepare_split(args, "train", "train.tsv", None)
    assert manifest.read_text() == "previous manifest\n"
    assert not (args.manifest_dir / "train.tsv.tmp").exists()


@pytest.mark.parametrize("audio_format", ["flac", "wav"])
def test_valid_quantized_audio_can_be_reused(tmp_path, monkeypatch, audio_format):
    args = options(tmp_path)
    args.audio_format = audio_format
    array = np.linspace(-0.9, 0.9, 100, dtype=np.float32)
    rows = [{"audio": {"array": array, "sampling_rate": 44100}}]
    monkeypatch.setattr(prepare, "load_split", lambda args, split: rows)
    prepare.prepare_split(args, "train", "train.tsv", None)
    (args.manifest_dir / "train.tsv").unlink()

    def unexpected_write(*args):
        pytest.fail("matching audio must be reused")

    monkeypatch.setattr(prepare, "write_audio", unexpected_write)
    prepare.prepare_split(args, "train", "train.tsv", None)
    assert (args.manifest_dir / "train.tsv").is_file()


@pytest.mark.parametrize(
    "option,value",
    [("--max_train", "0"), ("--max_validation", "-1"), ("--target_sample_rate", "0")],
)
def test_cli_rejects_nonpositive_limits_before_dataset_access(tmp_path, option, value):
    with pytest.raises(SystemExit) as error:
        prepare.parse_args(
            [
                "--audio_dir",
                str(tmp_path / "audio"),
                "--manifest_dir",
                str(tmp_path / "manifests"),
                option,
                value,
            ]
        )
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_failed_audio_write_preserves_existing_file(tmp_path, monkeypatch):
    destination = tmp_path / "audio.flac"
    destination.write_bytes(b"previous complete audio")

    def broken_write(path, *args, **kwargs):
        Path(path).write_bytes(b"partial audio")
        raise OSError("simulated write failure")

    monkeypatch.setattr(prepare.sf, "write", broken_write)
    with pytest.raises(OSError, match="simulated"):
        prepare.write_audio(destination, np.zeros(100), 44100, "flac")
    assert destination.read_bytes() == b"previous complete audio"
    assert list(tmp_path.iterdir()) == [destination]
