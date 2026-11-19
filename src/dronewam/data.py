"""Single-camera 6-DoF drone dataset used by DroneWAM."""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from dronewam.config import DataConfig
from dronewam.geometry import states_to_ego_actions

STATE_COLUMNS = ("frame_id", "timestamp", "pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw")
PREFERENCE_WINDOW_SAMPLING = "stratified_interior_midpoints_v1"


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    directory: Path
    num_frames: int


class ImageTransform:
    """Crop, resize and ImageNet-normalize a raw RGB clip."""

    def __init__(self, resolution: Sequence[int], crop_top_bottom: int) -> None:
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.crop_top_bottom = int(crop_top_bottom)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1, 1)

    def __call__(self, clip: np.ndarray) -> torch.Tensor:
        if clip.ndim != 4 or clip.shape[-1] != 3:
            raise ValueError(f"Expected clip [T,H,W,3], got {clip.shape}")
        tensor = torch.as_tensor(clip, dtype=torch.float32)
        if self.crop_top_bottom:
            if tensor.shape[1] <= 2 * self.crop_top_bottom:
                raise ValueError("Image height is too small for crop_top_bottom")
            tensor = tensor[:, self.crop_top_bottom : -self.crop_top_bottom]
        tensor = tensor.permute(0, 3, 1, 2).div(255.0)
        tensor = F.interpolate(tensor, size=self.resolution, mode="bilinear", align_corners=False)
        tensor = tensor.permute(1, 0, 2, 3).contiguous()
        return ((tensor - self.mean) / self.std).contiguous()


def load_records(csv_path: str | Path) -> List[ClipRecord]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    records: List[ClipRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            directory = Path(row["traj_dir"])
            if not directory.is_dir():
                raise FileNotFoundError(f"Trajectory directory does not exist: {directory}")
            records.append(
                ClipRecord(
                    clip_id=row["clip_id"],
                    directory=directory,
                    num_frames=int(row["n_frames"]),
                )
            )
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def load_states(directory: Path) -> np.ndarray:
    path = directory / "state_log.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(set(STATE_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        rows = [{key: float(row[key]) for key in STATE_COLUMNS} for row in reader]
    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two states")

    states = np.zeros((len(rows), 7), dtype=np.float32)
    for index, row in enumerate(rows):
        states[index, :6] = [
            row["pos_x"],
            row["pos_y"],
            row["pos_z"],
            row["roll"],
            row["pitch"],
            row["yaw"],
        ]
    for index in range(1, len(rows)):
        dt = rows[index]["timestamp"] - rows[index - 1]["timestamp"]
        if dt <= 0:
            raise ValueError(f"Non-positive timestamp delta at {path}:{index}")
        delta = states[index, :3] - states[index - 1, :3]
        states[index, 6] = float(np.linalg.norm(delta) / dt)
    return states


class DroneDataset(Dataset):
    """One fixed-length clip per CSV trajectory row."""

    def __init__(
        self,
        config: DataConfig,
        split: str,
        resolution: Sequence[int],
        fixed_window_starts: Optional[Sequence[int]] = None,
        sample_windows: Optional[Sequence[tuple[int, int]]] = None,
    ) -> None:
        if split not in {"train", "val", "preference"}:
            raise ValueError(f"split must be train, val or preference, got {split!r}")
        self.config = config
        self.split = split
        self.records = load_records(config.val_csv if split == "val" else config.train_csv)
        if fixed_window_starts is not None and len(fixed_window_starts) != len(self.records):
            raise ValueError("fixed_window_starts must align one-to-one with dataset records")
        if fixed_window_starts is not None and sample_windows is not None:
            raise ValueError("fixed_window_starts and sample_windows are mutually exclusive")
        self.fixed_window_starts = None if fixed_window_starts is None else tuple(int(v) for v in fixed_window_starts)
        self.sample_windows = (
            None
            if sample_windows is None
            else tuple((int(record_index), int(window_start)) for record_index, window_start in sample_windows)
        )
        self.transform = ImageTransform(resolution, config.crop_top_bottom)
        if config.base_fps % config.fps:
            raise ValueError("data.base_fps must be divisible by data.fps")
        self.sample_stride = config.base_fps // config.fps
        self.required_source_frames = 1 + (config.num_frames - 1) * self.sample_stride
        if self.sample_windows is not None:
            for record_index, window_start in self.sample_windows:
                if not 0 <= record_index < len(self.records):
                    raise ValueError(f"Sample record index {record_index} is outside the dataset")
                max_start = self.records[record_index].num_frames - self.required_source_frames
                if not 0 <= window_start <= max_start:
                    raise ValueError(
                        f"Sample window start {window_start} is invalid for "
                        f"{self.records[record_index].clip_id}, max={max_start}"
                    )

    def __len__(self) -> int:
        return len(self.records) if self.sample_windows is None else len(self.sample_windows)

    def _window_start(self, record: ClipRecord, index: int) -> int:
        max_start = record.num_frames - self.required_source_frames
        if max_start < 0:
            raise ValueError(
                f"{record.clip_id} has {record.num_frames} frames, needs {self.required_source_frames}"
            )
        if self.fixed_window_starts is not None:
            start = self.fixed_window_starts[index]
            if not 0 <= start <= max_start:
                raise ValueError(f"Fixed window start {start} is invalid for {record.clip_id}, max={max_start}")
            return start
        if self.split == "train":
            return random.randint(0, max_start)
        if self.split == "val":
            return max_start // 2
        digest = hashlib.sha256(f"{self.split}:{record.clip_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (max_start + 1)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.sample_windows is None:
            record_index = index
            record = self.records[record_index]
            start = self._window_start(record, record_index)
        else:
            record_index, start = self.sample_windows[index]
            record = self.records[record_index]
        states_all = load_states(record.directory)
        if len(states_all) != record.num_frames:
            raise ValueError(
                f"{record.clip_id} CSV declares {record.num_frames} frames but state_log has {len(states_all)}"
            )
        indices = start + np.arange(self.config.num_frames) * self.sample_stride
        images = []
        for frame_index in indices:
            image_path = record.directory / "images" / f"{int(frame_index):06d}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        goal_index = record.num_frames - 1
        goal_path = record.directory / "images" / f"{goal_index:06d}.png"
        if not goal_path.is_file():
            raise FileNotFoundError(goal_path)
        with Image.open(goal_path) as image:
            goal_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        states = torch.from_numpy(states_all[indices].copy())
        actions = states_to_ego_actions(states.unsqueeze(0)).squeeze(0)
        return {
            "frames": self.transform(np.stack(images)),
            "goal_frame": self.transform(goal_image[None])[:, 0],
            "states": states,
            "actions": actions,
            "indices": torch.as_tensor(indices, dtype=torch.long),
            "record_index": torch.tensor(record_index, dtype=torch.long),
            "window_start": torch.tensor(start, dtype=torch.long),
            "goal_index": torch.tensor(goal_index, dtype=torch.long),
        }


def deterministic_window_samples(
    config: DataConfig,
    records: Sequence[ClipRecord],
    windows_per_record: int,
    record_indices: Optional[Sequence[int]] = None,
) -> list[tuple[int, int]]:
    """Enumerate uniformly spaced, deterministic windows for each trajectory."""
    if windows_per_record < 1:
        raise ValueError("windows_per_record must be positive")
    if config.base_fps % config.fps:
        raise ValueError("data.base_fps must be divisible by data.fps")
    sample_stride = config.base_fps // config.fps
    required_source_frames = 1 + (config.num_frames - 1) * sample_stride
    selected = range(len(records)) if record_indices is None else [int(index) for index in record_indices]
    samples = []
    for record_index in selected:
        if not 0 <= record_index < len(records):
            raise ValueError(f"Record index {record_index} is outside the dataset")
        max_start = records[record_index].num_frames - required_source_frames
        if max_start < 0:
            raise ValueError(
                f"{records[record_index].clip_id} has {records[record_index].num_frames} frames, "
                f"needs {required_source_frames}"
            )
        count = min(windows_per_record, max_start + 1)
        if count == 1:
            starts = [max_start // 2]
        else:
            # Midpoint stratification avoids over-representing exact trajectory endpoints.
            starts = np.floor(
                (np.arange(count, dtype=np.float64) + 0.5) * (max_start + 1) / count
            ).astype(np.int64).tolist()
        samples.extend((record_index, int(start)) for start in starts)
    if not samples:
        raise ValueError("Deterministic window sampling produced zero samples")
    return samples


def build_preference_dataset(
    config: DataConfig,
    resolution: Sequence[int],
    windows_per_record: int,
    record_indices: Optional[Sequence[int]] = None,
    source_split: str = "train",
) -> DroneDataset:
    """Build reproducible multi-window samples without crossing trajectory groups."""
    if source_split not in {"train", "val"}:
        raise ValueError("source_split must be train or val")
    records = load_records(config.train_csv if source_split == "train" else config.val_csv)
    sample_windows = deterministic_window_samples(
        config,
        records,
        windows_per_record,
        record_indices,
    )
    return DroneDataset(
        config,
        source_split,
        resolution,
        sample_windows=sample_windows,
    )


def build_loader(
    config: DataConfig,
    resolution: Sequence[int],
    split: str,
    rank: int,
    world_size: int,
    record_indices: Optional[Sequence[int]] = None,
) -> DataLoader:
    dataset = DroneDataset(config, split, resolution)
    if record_indices is not None:
        dataset = Subset(dataset, [int(index) for index in record_indices])
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=split == "train",
        drop_last=split == "train",
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        drop_last=split == "train",
    )


def build_exact_eval_loader(
    config: DataConfig,
    resolution: Sequence[int],
    rank: int,
    world_size: int,
) -> DataLoader:
    """Build a validation loader without DistributedSampler padding."""
    dataset = DroneDataset(config, "val", resolution)
    local_dataset = Subset(dataset, exact_eval_indices(len(dataset), rank, world_size))
    return DataLoader(
        local_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        drop_last=False,
    )


def exact_eval_indices(length: int, rank: int, world_size: int) -> range:
    """Partition indices across ranks exactly, without padding or duplication."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return range(rank, length, world_size)
