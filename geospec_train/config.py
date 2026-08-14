from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        return value

    return {}


def _filter_fields(cls: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


@dataclass
class DatasetConfig:
    name: str = ""
    data_path: str = "./dataset"
    preload_path: str = "./preload_dataset"
    output_dir: str = ""
    save_path: str = ""
    input_size: int = 384
    image_channels: int = 3
    num_classes: int = 2
    mean: Union[Tuple[float, ...], str] = (0.5, 0.5, 0.5)
    std: Union[Tuple[float, ...], str] = (0.5, 0.5, 0.5)
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    train_preload_dir: str = ""
    val_preload_dir: str = ""
    test_preload_dir: str = ""
    domain_metadata_path: str = ""
    num_domains: int = 0
    use_augmentation: bool = False
    data_augmentation_args: list[str] = field(default_factory=list[str])

    def __post_init__(self) -> None:
        if not isinstance(self.mean, str):
            self.mean = tuple(float(x) for x in self.mean)

        if not isinstance(self.std, str):
            self.std = tuple(float(x) for x in self.std)

    @classmethod
    def from_dict(cls, data: Any) -> DatasetConfig:
        return cls(**_filter_fields(cls, _as_dict(data)))


@dataclass
class BaseConfig:
    device: str = "cuda"
    random_seed: int = -1

    @classmethod
    def from_dict(cls, data: Any) -> BaseConfig:
        return cls(**_filter_fields(cls, _as_dict(data)))


@dataclass
class TrainSectionConfig:
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    epochs: int = 50
    warmup_epochs: float = 2.0
    grad_clip: float = 1.0
    amp: bool = True

    @classmethod
    def from_dict(cls, data: Any) -> TrainSectionConfig:
        return cls(**_filter_fields(cls, _as_dict(data)))


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.0005
    min_lr: float = 0.000001
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 0.00000001

    @classmethod
    def from_dict(cls, data: Any) -> OptimizerConfig:
        return cls(**_filter_fields(cls, _as_dict(data)))


@dataclass
class LossConfig:
    ce_weight: float = 1.0
    evid_weight: float = 0.1
    evid_ramp_epochs: float = 2.0
    evid_kl_lambda: float = 1.0
    evid_kl_anneal_epochs: int = 10
    curvature_weight: float = 0.1
    disentangle_weight: float = 0.1
    bayesian_weight: float = 0.1
    branch_weight: float = 1.0
    sym_weight: float = 0.0
    sym_start_epoch: int = 1
    sym_ramp_epochs: int = 1
    domain_weight: float = 0.0
    domain_start_epoch: int = 5
    domain_ramp_epochs: int = 5
    class_weights: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.class_weights is not None:
            self.class_weights = [float(x) for x in self.class_weights]

    @classmethod
    def from_dict(cls, data: Any) -> LossConfig:
        return cls(**_filter_fields(cls, _as_dict(data)))


@dataclass
class LoggingConfig:
    seed: int = 42
    deterministic: bool = False
    output_dir: str = "./runs/geospec"
    log_interval: int = 20
    eval_interval: int = 1
    save_interval: int = 1
    resume: str = ""
    max_nonfinite_steps: int = 5

    @classmethod
    def from_dict(cls, data: Any) -> LoggingConfig:
        return cls(**_filter_fields(cls, _as_dict(data)))


@dataclass
class NetworkConfig:
    num_prompts: int = 16
    side_reduction_ratio: int = 8
    prompt_reduction_ratio: int = 1
    prompt_norm: bool = True
    prompt_proj: bool = False
    layers_to_extract: Union[str, Tuple[int, ...], List[int]] = "6-11"
    token_ratio: float = 0.2
    token_imp: str = "global"
    side_input_size: int = 128
    pretrained_path: str = "google/vit-base-patch16-384"
    input_size: int = 384
    backbone_input_size: int = 224
    vit_feature_branch: Tuple[int, int] = (0, 2)
    fold_id: int = 0
    drop_path_rate: float = 0.2
    prototype_momentum: float = 0.99
    tta_lr: float = 0.001
    tta_lambda_conf: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.layers_to_extract, int):
            self.layers_to_extract = (int(self.layers_to_extract),)
        elif isinstance(self.layers_to_extract, list):
            self.layers_to_extract = tuple(int(x) for x in self.layers_to_extract)
        elif isinstance(self.layers_to_extract, tuple):
            self.layers_to_extract = tuple(int(x) for x in self.layers_to_extract)

        if not isinstance(self.vit_feature_branch, tuple):
            self.vit_feature_branch = tuple(int(x) for x in self.vit_feature_branch)

        if len(self.vit_feature_branch) != 2:
            raise ValueError(
                "network.vit_feature_branch must contain exactly two integers."
            )

    @classmethod
    def from_dict(cls, data: Any) -> NetworkConfig:
        return cls(**_filter_fields(cls, _as_dict(data)))


@dataclass
class CoreConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    base: BaseConfig = field(default_factory=BaseConfig)
    train: TrainSectionConfig = field(default_factory=TrainSectionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetConfig):
            self.dataset = DatasetConfig.from_dict(self.dataset)

        if not isinstance(self.base, BaseConfig):
            self.base = BaseConfig.from_dict(self.base)

        if not isinstance(self.train, TrainSectionConfig):
            self.train = TrainSectionConfig.from_dict(self.train)

        if not isinstance(self.optimizer, OptimizerConfig):
            self.optimizer = OptimizerConfig.from_dict(self.optimizer)

        if not isinstance(self.loss, LossConfig):
            self.loss = LossConfig.from_dict(self.loss)

        if not isinstance(self.logging, LoggingConfig):
            self.logging = LoggingConfig.from_dict(self.logging)

        if not isinstance(self.network, NetworkConfig):
            self.network = NetworkConfig.from_dict(self.network)

    @classmethod
    def from_dict(cls, data: Any) -> CoreConfig:
        data = _as_dict(data)

        return cls(
            dataset=DatasetConfig.from_dict(data.get("dataset")),
            base=BaseConfig.from_dict(data.get("base")),
            train=TrainSectionConfig.from_dict(data.get("train")),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer")),
            loss=LossConfig.from_dict(data.get("loss")),
            logging=LoggingConfig.from_dict(data.get("logging")),
            network=NetworkConfig.from_dict(data.get("network")),
        )

    @classmethod
    def from_hydra(cls, cfg: Any) -> CoreConfig:
        return cls.from_dict(cfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Union[Path, str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: Union[Path, str]) -> CoreConfig:
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
