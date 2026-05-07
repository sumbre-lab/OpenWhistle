#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DWE_ROOT = Path.home() / "Documents" / "DolphinWhistleExtractor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the external wmmsd/dclde folders expected by "
            "cnn/run_inference_dataset.py."
        )
    )
    parser.add_argument(
        "--dwe-root",
        type=Path,
        default=Path(os.environ.get("DOLPHIN_WHISTLE_EXTRACTOR_ROOT", DEFAULT_DWE_ROOT)),
        help="Path to DolphinWhistleExtractor.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Do not create .venv or install cnn/requirements.txt.",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=REPO_ROOT / ".venv",
        help="Virtual environment to create/reuse.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Base Python executable used to create the virtual environment.",
    )
    parser.add_argument(
        "--prepare-wmmsd-hf",
        action="store_true",
        help="Build benchmark_data/wmmsd_first_pass from confit/wmms-parquet.",
    )
    parser.add_argument(
        "--wmmsd-source-root",
        type=Path,
        default=None,
        help="Local WMMSD tree used to build benchmark_data/wmmsd_first_pass.",
    )
    parser.add_argument(
        "--download-dclde",
        action="store_true",
        help="Download NOAA DCLDE evaluation data if benchmark_data/dclde_source is missing.",
    )
    parser.add_argument(
        "--no-link-fallback",
        action="store_true",
        help="Fail instead of linking to already prepared fallback folders.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run python cnn/run_inference_dataset.py wmmsd dclde after setup.",
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional local checkpoint for the final --run step.",
    )
    return parser.parse_args()


def info(message: str) -> None:
    print(f"[openwhistle-setup] {message}")


def audio_count(flat_dir: Path) -> int:
    if not flat_dir.is_dir():
        return 0
    return sum(
        1
        for path in flat_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
    )


def is_prepared_dataset(path: Path) -> bool:
    return audio_count(path / "flat_recordings") > 0


def run(command: list[str], cwd: Path) -> None:
    info(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def setup_python(args: argparse.Namespace) -> Path:
    if args.skip_deps:
        info(f"Skipping dependency installation; using {sys.executable}")
        return Path(sys.executable)

    venv_dir = args.venv_dir.expanduser().resolve()
    python_bin = venv_python(venv_dir)
    if not python_bin.exists():
        info(f"Creating virtual environment: {venv_dir}")
        subprocess.run([str(args.python.expanduser()), "-m", "venv", str(venv_dir)], check=True)
    else:
        info(f"Reusing virtual environment: {venv_dir}")

    info("Installing CNN requirements.")
    subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", "-r", str(REPO_ROOT / "cnn" / "requirements.txt")],
        check=True,
    )
    return python_bin


def link_dataset(target: Path, source: Path, label: str) -> None:
    if not is_prepared_dataset(source):
        raise FileNotFoundError(f"{label} fallback is not prepared: {source}")
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise FileExistsError(
            f"{label} target exists but is not a usable prepared dataset: {target}"
        )
    target.symlink_to(source, target_is_directory=True)
    info(f"{label}: linked {target} -> {source}")


def first_prepared(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if is_prepared_dataset(candidate):
            return candidate
    return None


def prepare_wmmsd(
    args: argparse.Namespace,
    dwe_root: Path,
    benchmark_root: Path,
    python_bin: Path,
) -> Path:
    target = benchmark_root / "wmmsd_first_pass"
    if is_prepared_dataset(target):
        info(f"WMMSD ready: {target} ({audio_count(target / 'flat_recordings')} audio files)")
        return target

    if args.wmmsd_source_root is not None:
        run(
            [
                str(python_bin),
                "external_benchmarks/wmmsd/prepare_wmmsd_manifest.py",
                "--source-root",
                str(args.wmmsd_source_root.expanduser().resolve()),
                "--output-dir",
                str(target),
                "--species",
                "Bottlenose_Dolphin",
                "Common_Dolphin",
            ],
            cwd=dwe_root,
        )
    elif args.prepare_wmmsd_hf:
        run(
            [
                str(python_bin),
                "external_benchmarks/wmmsd/prepare_wmmsd_manifest.py",
                "--hf-dataset-id",
                "confit/wmms-parquet",
                "--output-dir",
                str(target),
                "--species",
                "Bottlenose_Dolphin",
                "Common_Dolphin",
            ],
            cwd=dwe_root,
        )
    elif not args.no_link_fallback:
        fallback = first_prepared(
            [
                benchmark_root / "wmmsd_binary_clip",
                benchmark_root / "wmmsd_ultimate",
                benchmark_root / "wmmsd_smoke",
            ]
        )
        if fallback is None:
            raise FileNotFoundError(
                "WMMSD is missing. Use --prepare-wmmsd-hf or --wmmsd-source-root."
            )
        info(f"WMMSD canonical folder missing; using fallback {fallback}")
        link_dataset(target, fallback, "WMMSD")
    else:
        raise FileNotFoundError(
            f"WMMSD is missing: {target}. Use --prepare-wmmsd-hf or allow fallback links."
        )

    if not is_prepared_dataset(target):
        raise RuntimeError(f"WMMSD setup did not create audio files under {target}")
    return target


def prepare_dclde(
    args: argparse.Namespace,
    dwe_root: Path,
    benchmark_root: Path,
    python_bin: Path,
) -> Path:
    source_root = benchmark_root / "dclde_source"
    target = benchmark_root / "dclde_clip"
    if is_prepared_dataset(target):
        info(f"DCLDE ready: {target} ({audio_count(target / 'flat_recordings')} audio files)")
        return target

    if source_root.is_dir():
        run(
            [
                str(python_bin),
                "external_benchmarks/dclde/prepare_dclde_clip_benchmark.py",
                "--source-root",
                str(source_root),
                "--output-dir",
                str(target),
                "--subset",
                "evaluation",
                "--annotation-version",
                "2011",
                "--window-seconds",
                "0.4",
                "--margin-seconds",
                "0.4",
                "--noise-to-whistle-ratio",
                "1.0",
            ],
            cwd=dwe_root,
        )
    elif args.download_dclde:
        run(
            [
                str(python_bin),
                "external_benchmarks/dclde/download_dclde_subset.py",
                "--output-dir",
                str(source_root),
                "--subset",
                "evaluation",
                "--annotated-only",
                "--include-docs",
            ],
            cwd=dwe_root,
        )
        return prepare_dclde(args, dwe_root, benchmark_root, python_bin)
    elif not args.no_link_fallback:
        fallback = first_prepared([benchmark_root / "dclde_smoke_clip"])
        if fallback is None:
            raise FileNotFoundError(
                "DCLDE is missing. Use --download-dclde or provide benchmark_data/dclde_source."
            )
        info(f"DCLDE canonical folder missing; using fallback {fallback}")
        link_dataset(target, fallback, "DCLDE")
    else:
        raise FileNotFoundError(
            f"DCLDE is missing: {target}. Use --download-dclde or allow fallback links."
        )

    if not is_prepared_dataset(target):
        raise RuntimeError(f"DCLDE setup did not create audio files under {target}")
    return target


def write_env_file(dwe_root: Path, wmmsd_dir: Path, dclde_dir: Path) -> None:
    env_path = REPO_ROOT / "cnn" / ".external_inference.env"
    env_path.write_text(
        "\n".join(
            [
                "# Generated by cnn/setup_external_inference.py",
                f"export DOLPHIN_WHISTLE_EXTRACTOR_ROOT={shlex_quote(dwe_root)}",
                f"export OPENWHISTLE_CNN_WMMSD_DIR={shlex_quote(wmmsd_dir)}",
                f"export OPENWHISTLE_CNN_DCLDE_DIR={shlex_quote(dclde_dir)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    info(f"Wrote {env_path}")


def shlex_quote(path: Path) -> str:
    import shlex

    return shlex.quote(str(path))


def run_inference(
    args: argparse.Namespace,
    dwe_root: Path,
    wmmsd_dir: Path,
    dclde_dir: Path,
    python_bin: Path,
) -> None:
    command = [
        str(python_bin),
        "cnn/run_inference_dataset.py",
        "wmmsd",
        "dclde",
        "--batch-size",
        str(args.batch_size),
    ]
    if args.cpu_only:
        command.append("--cpu-only")
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    if args.checkpoint_path is not None:
        command.extend(["--checkpoint-path", str(args.checkpoint_path.expanduser().resolve())])

    env = os.environ.copy()
    env["DOLPHIN_WHISTLE_EXTRACTOR_ROOT"] = str(dwe_root)
    env["OPENWHISTLE_CNN_WMMSD_DIR"] = str(wmmsd_dir)
    env["OPENWHISTLE_CNN_DCLDE_DIR"] = str(dclde_dir)
    info(" ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    dwe_root = args.dwe_root.expanduser().resolve()
    benchmark_root = dwe_root / "benchmark_data"

    if not dwe_root.is_dir():
        raise FileNotFoundError(f"DolphinWhistleExtractor not found: {dwe_root}")
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"benchmark_data not found: {benchmark_root}")
    if shutil.which(str(args.python)) is None and not args.python.exists():
        raise RuntimeError(f"Python executable not found: {args.python}")

    python_bin = setup_python(args)
    wmmsd_dir = prepare_wmmsd(args, dwe_root, benchmark_root, python_bin)
    dclde_dir = prepare_dclde(args, dwe_root, benchmark_root, python_bin)
    write_env_file(dwe_root, wmmsd_dir, dclde_dir)

    if args.run:
        run_inference(args, dwe_root, wmmsd_dir, dclde_dir, python_bin)
    else:
        print()
        print("Next:")
        if not args.skip_deps:
            print(f"  source {args.venv_dir.expanduser().resolve() / 'bin' / 'activate'}")
        print("  source cnn/.external_inference.env")
        print("  python cnn/run_inference_dataset.py wmmsd dclde")


if __name__ == "__main__":
    main()
