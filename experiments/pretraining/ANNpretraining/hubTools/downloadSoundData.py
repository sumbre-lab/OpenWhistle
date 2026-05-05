import argparse

import datasets

from ANNpretraining.hubTools.token_hub import login_to_hub
from ANNpretraining.runtime import artifact_dir, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Download a Hugging Face audio dataset and save it locally.")
    parser.add_argument("--repo_id", required=True, help="Hugging Face dataset id to download.")
    parser.add_argument(
        "--cache_dir",
        default=str(artifact_dir("datasets", "hf_cache")),
        help="Directory used by the datasets cache.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(artifact_dir("datasets", "saved_dataset")),
        help="Target directory for datasets.save_to_disk(...).",
    )
    parser.add_argument("--config", default="", help="Optional dataset config name.")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of dataset workers.")
    return parser.parse_args()


def main():
    args = parse_args()
    login_to_hub()
    cache_dir = ensure_dir(args.cache_dir)
    output_dir = ensure_dir(args.output_dir)

    if args.config:
        dataset = datasets.load_dataset(
            args.repo_id,
            args.config,
            cache_dir=str(cache_dir),
            num_proc=args.num_proc,
        )
    else:
        dataset = datasets.load_dataset(
            args.repo_id,
            cache_dir=str(cache_dir),
            num_proc=args.num_proc,
        )
    dataset.save_to_disk(str(output_dir))


if __name__ == "__main__":
    main()
