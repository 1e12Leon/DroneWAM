"""Build strict trajectory-level train/validation manifests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dronewam.data_manifest import (
    discover_records,
    manifest_summary,
    stratified_split,
    write_manifest,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--minimum-frames", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir = Path(args.output_dir)
    records = discover_records(Path(args.dataset_root), args.minimum_frames)
    train, val = stratified_split(records, args.val_fraction, args.seed)
    write_manifest(train, output_dir / "airsim6d_train.csv")
    write_manifest(val, output_dir / "airsim6d_val.csv")
    summary = manifest_summary(train, val, seed=args.seed, val_fraction=args.val_fraction)
    write_summary(summary, output_dir / "airsim6d_split.json")
    logging.info("manifest summary=%s", summary)


if __name__ == "__main__":
    main()
