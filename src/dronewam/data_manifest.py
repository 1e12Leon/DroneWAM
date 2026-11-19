"""Deterministic trajectory-level manifests for the AirSim 6-DoF dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

from dronewam.data import STATE_COLUMNS


@dataclass(frozen=True)
class ManifestRecord:
    clip_id: str
    trajectory: Path
    num_frames: int
    world: str


def _world_seed(seed: int, world: str) -> int:
    digest = hashlib.sha256(world.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:8], "big")


def inspect_trajectory(trajectory: Path, dataset_root: Path, minimum_frames: int) -> ManifestRecord:
    state_path = trajectory / "state_log.csv"
    image_dir = trajectory / "images"
    if not state_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f"Incomplete trajectory: {trajectory}")

    frame_ids = []
    timestamps = []
    with state_path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(set(STATE_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{state_path} is missing columns: {missing}")
        for row in reader:
            frame_ids.append(int(row["frame_id"]))
            timestamps.append(float(row["timestamp"]))
    if len(frame_ids) < minimum_frames:
        raise ValueError(f"{trajectory} has {len(frame_ids)} frames, requires at least {minimum_frames}")
    if frame_ids != list(range(len(frame_ids))):
        raise ValueError(f"{state_path} frame_id must be contiguous from zero")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{state_path} timestamps must be strictly increasing")

    image_indices = sorted(
        int(path.stem)
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png" and path.stem.isdigit()
    )
    if image_indices != frame_ids:
        raise ValueError(f"{image_dir} PNG indices do not match state_log.csv")
    relative = trajectory.relative_to(dataset_root)
    if len(relative.parts) != 2:
        raise ValueError(f"Expected <world>/<trajectory>, got {relative}")
    return ManifestRecord(
        clip_id=f"{relative.parts[0]}/{relative.parts[1]}",
        trajectory=trajectory.resolve(),
        num_frames=len(frame_ids),
        world=relative.parts[0],
    )


def discover_records(dataset_root: Path, minimum_frames: int) -> list[ManifestRecord]:
    root = dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    trajectories = sorted(path for path in root.glob("*/Traj_*") if path.is_dir())
    if not trajectories:
        raise ValueError(f"No <world>/Traj_* directories under {root}")
    return [inspect_trajectory(path, root, minimum_frames) for path in trajectories]


def stratified_split(
    records: Sequence[ManifestRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1)")
    by_world: Dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_world[record.world].append(record)
    train = []
    val = []
    for world, world_records in sorted(by_world.items()):
        if len(world_records) < 2:
            raise ValueError(f"World {world!r} needs at least two trajectories")
        shuffled = sorted(world_records, key=lambda record: record.clip_id)
        random.Random(_world_seed(seed, world)).shuffle(shuffled)
        val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_fraction)))
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])
    return sorted(train, key=lambda record: record.clip_id), sorted(val, key=lambda record: record.clip_id)


def write_manifest(records: Iterable[ManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("clip_id", "traj_dir", "n_frames"))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "clip_id": record.clip_id,
                    "traj_dir": str(record.trajectory),
                    "n_frames": record.num_frames,
                }
            )


def manifest_summary(
    train: Sequence[ManifestRecord],
    val: Sequence[ManifestRecord],
    *,
    seed: int,
    val_fraction: float,
) -> dict:
    def counts(records: Sequence[ManifestRecord]) -> dict[str, int]:
        result: Dict[str, int] = defaultdict(int)
        for record in records:
            result[record.world] += 1
        return dict(sorted(result.items()))

    return {
        "split_unit": "trajectory",
        "stratified_by": "world",
        "seed": seed,
        "val_fraction": val_fraction,
        "train_trajectories": len(train),
        "val_trajectories": len(val),
        "train_by_world": counts(train),
        "val_by_world": counts(val),
    }


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
