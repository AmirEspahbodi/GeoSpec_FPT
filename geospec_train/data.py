from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets, transforms

from .config import TrainConfig


def _pil_loader(path: str, channels: int) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        if channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        return img


def _normalize_stats(cfg: TrainConfig) -> Tuple[list[float], list[float]]:
    mean = list(cfg.mean)
    std = list(cfg.std)

    channels = int(cfg.image_channels)

    if channels == 1:
        if len(mean) != 1:
            mean = [sum(mean) / max(1, len(mean))]
        if len(std) != 1:
            std = [sum(std) / max(1, len(std))]
    elif channels == 3:
        if len(mean) == 1:
            mean = mean * 3
        if len(std) == 1:
            std = std * 3
    else:
        if len(mean) != channels:
            mean = [0.5] * channels
        if len(std) != channels:
            std = [0.5] * channels

    return [float(x) for x in mean], [float(x) for x in std]


def build_transforms(cfg: TrainConfig, is_train: bool) -> transforms.Compose:
    """
    Conservative deterministic preprocessing by default.

    Random augmentation is disabled by default because preloaded FPT+
    key/value caches are not augmentation-aware.
    """
    mean, std = _normalize_stats(cfg)

    ops = []

    if is_train and cfg.use_augmentation:
        # Deliberately minimal. Do not enable unless caches are regenerated.
        ops.append(transforms.RandomHorizontalFlip(p=0.5))

    ops.extend(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transforms.Compose(ops)


def resolve_preload_dir(cfg: TrainConfig, split: str) -> Path:
    explicit = getattr(cfg, f"{split}_preload_dir", "")
    if explicit:
        return Path(explicit)

    preload_path = Path(cfg.preload_path)
    split_dir = preload_path / split

    if split_dir.exists():
        return split_dir

    return preload_path


def load_domain_map(path: Optional[str | Path]) -> Dict[str, int]:
    """
    Optional domain metadata CSV.

    Expected format:
      path,domain
      train/class_a/img1.jpg,0
      img1,1

    The path column may be:
      - relative path from split root,
      - file name,
      - file stem.
    """
    if not path:
        return {}

    path = Path(path)
    if not path.exists():
        return {}

    domain_map: Dict[str, int] = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue

            p = row[0].strip()
            d = row[1].strip()

            if not p or not d:
                continue

            if p.lower() == "path":
                continue

            try:
                domain = int(d)
            except ValueError:
                continue

            domain_map[p] = domain
            domain_map[Path(p).name] = domain
            domain_map[Path(p).stem] = domain

    return domain_map


class PreloadedImageDataset(datasets.ImageFolder):
    """
    ImageFolder dataset that additionally loads precomputed FPT+ states:

      key_states, value_states = safetensors file

    Returned item:
      image_tensor, key_states, value_states, label, domain_label

    domain_label is -1 when unavailable.
    """

    def __init__(
        self,
        root: str | Path,
        preload_dir: str | Path,
        transform: Optional[Any] = None,
        image_channels: int = 3,
        domain_map: Optional[Dict[str, int]] = None,
    ):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        self.preload_dir = Path(preload_dir)
        if not self.preload_dir.exists():
            raise FileNotFoundError(
                f"Preload directory does not exist: {self.preload_dir}"
            )

        self.image_channels = int(image_channels)
        self.domain_map = domain_map or {}

        super().__init__(
            root=str(root),
            transform=transform,
            loader=lambda p: _pil_loader(p, self.image_channels),
        )

    def _relative_path(self, path: str) -> Path:
        try:
            return Path(path).relative_to(self.root)
        except ValueError:
            return Path(path)

    def _find_states_file(self, path: str) -> Path:
        rel = self._relative_path(path)

        candidates = [
            self.preload_dir / rel.with_suffix(".safetensors"),
            self.preload_dir / f"{Path(path).stem}.safetensors",
            self.preload_dir / f"{Path(path).name}.safetensors",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Could not find preloaded states for image.\n"
            f"image: {path}\n"
            f"preload_dir: {self.preload_dir}\n"
            f"candidates: {[str(x) for x in candidates]}"
        )

    def _load_states(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load preloaded key/value states. "
                "Install it with: pip install safetensors"
            ) from exc

        states_path = self._find_states_file(path)
        states = load_file(str(states_path))

        if "key_states" not in states or "value_states" not in states:
            raise KeyError(
                f"Safetensors file {states_path} must contain 'key_states' and 'value_states'."
            )

        return states["key_states"], states["value_states"]

    def _load_domain(self, path: str) -> int:
        if not self.domain_map:
            return -1

        rel = self._relative_path(path)

        keys = [
            str(rel),
            rel.as_posix(),
            rel.name,
            rel.stem,
            str(Path(path)),
            Path(path).name,
            Path(path).stem,
        ]

        for key in keys:
            if key in self.domain_map:
                return int(self.domain_map[key])

        return -1

    def __getitem__(self, index: int):
        path, target = self.samples[index]

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)

        key_states, value_states = self._load_states(path)
        domain = self._load_domain(path)

        return image, key_states, value_states, int(target), int(domain)


def _collate_states(states: list[Any], transpose_tensor: bool = True) -> Any:
    """
    Collate preloaded states.

    The reference loading contract transposes batched tensor states on dims 0 and 1:
      [B, layers, ...] -> [layers, B, ...]

    If states are already provided as a list/tuple, we preserve that structure
    and do not transpose inner tensors, because list structures usually already
    represent layer-first containers.
    """
    first = states[0]

    if torch.is_tensor(first):
        if all(torch.is_tensor(x) and x.shape == first.shape for x in states):
            stacked = torch.stack(states, dim=0)
        else:
            stacked = default_collate(states)

        if transpose_tensor and stacked.dim() >= 2:
            stacked = stacked.transpose(0, 1)

        return stacked.contiguous()

    if isinstance(first, (list, tuple)):
        return type(first)(
            _collate_states([s[i] for s in states], transpose_tensor=False)
            for i in range(len(first))
        )

    return default_collate(states)


def preload_collate(
    batch: list[Any],
) -> Tuple[Any, Any, Any, torch.Tensor, torch.Tensor]:
    images = default_collate([item[0] for item in batch])
    key_states = _collate_states([item[1] for item in batch], transpose_tensor=True)
    value_states = _collate_states([item[2] for item in batch], transpose_tensor=True)
    labels = default_collate([item[3] for item in batch])
    domains = default_collate([item[4] for item in batch])

    return images, key_states, value_states, labels, domains


def _make_dataset(
    cfg: TrainConfig,
    split: str,
    transform: transforms.Compose,
) -> Optional[PreloadedImageDataset]:
    root = Path(cfg.data_path) / split
    if not root.exists():
        return None

    preload_dir = resolve_preload_dir(cfg, split)
    domain_map = load_domain_map(cfg.domain_metadata_path)

    return PreloadedImageDataset(
        root=root,
        preload_dir=preload_dir,
        transform=transform,
        image_channels=cfg.image_channels,
        domain_map=domain_map,
    )


def build_dataloaders(
    cfg: TrainConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    train_transform = build_transforms(cfg, is_train=True)
    eval_transform = build_transforms(cfg, is_train=False)

    train_dataset = _make_dataset(cfg, cfg.train_split, train_transform)
    if train_dataset is None:
        raise FileNotFoundError(
            f"Training split not found: {Path(cfg.data_path) / cfg.train_split}"
        )

    val_dataset = _make_dataset(cfg, cfg.val_split, eval_transform)
    test_dataset = _make_dataset(cfg, cfg.test_split, eval_transform)

    common_loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=preload_collate,
        persistent_workers=cfg.num_workers > 0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader_kwargs,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs,
        )

    return train_loader, val_loader, test_loader
