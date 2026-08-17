import os
import sys
import argparse
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geospec_train.config import CoreConfig
from fpt_old.func import *
from geospec_model.fpt_modules.builder import build_frozen_encoder


original_check_help = argparse.ArgumentParser._check_help


def patched_check_help(self, action):
    if getattr(action, "help", None) is None:
        return
    if type(action.help).__name__ == "LazyCompletionHelp":
        return
    original_check_help(self, action)


argparse.ArgumentParser._check_help = patched_check_help


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class FineImageFolder(datasets.ImageFolder):
    def __init__(self, root, lpm_transform=None, loader=pil_loader):
        super(FineImageFolder, self).__init__(root, loader=loader)
        self.lpm_features = {}
        self.lpm_transform = lpm_transform

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        lpm_sample = self.lpm_transform(sample)
        return path, lpm_sample


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(hydra_cfg: DictConfig) -> None:
    # Convert Hydra DictConfig to CoreConfig.
    cfg = CoreConfig.from_hydra(hydra_cfg)

    if os.path.exists(cfg.dataset.preload_path):
        print_msg(
            "Preload path {} exists.".format(cfg.dataset.preload_path),
            warning=True,
        )
        print(
            'WARNING: Please update the "preload_path" in "/configs/dataset" '
            'or add an argument "++dataset.preload_path=new_path" in the command.'
        )
        return

    print("Preloading {} dataset...".format(cfg.dataset.name))

    frozen_encoder = build_frozen_encoder(cfg).to(cfg.base.device)
    dataset = generate_dataset(cfg)

    preload_dataset(cfg, dataset, frozen_encoder)

    print("Preloading done.")


def preload_dataset(cfg: CoreConfig, dataset, frozen_encoder):
    train_dataset, test_dataset, val_dataset = dataset

    print("Preloading train dataset...")
    preload(cfg, train_dataset, frozen_encoder, cfg.dataset.preload_path)

    print("Preloading test dataset...")
    preload(cfg, test_dataset, frozen_encoder, cfg.dataset.preload_path)

    print("Preloading val dataset...")
    preload(cfg, val_dataset, frozen_encoder, cfg.dataset.preload_path)


def generate_dataset(cfg: CoreConfig):
    data_path = cfg.dataset.data_path

    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    val_path = os.path.join(data_path, "val")

    preprocess = transforms.Compose(
        [
            transforms.Resize((cfg.dataset.input_size, cfg.dataset.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.dataset.mean, cfg.dataset.std),
        ]
    )

    train_dataset = FineImageFolder(train_path, preprocess)
    test_dataset = FineImageFolder(test_path, preprocess)
    val_dataset = FineImageFolder(val_path, preprocess)

    return train_dataset, test_dataset, val_dataset


def preload(
    cfg: CoreConfig,
    dataset,
    frozen_encoder,
    preload_path: str = "./preload_data",
):
    os.makedirs(preload_path, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        drop_last=False,
        pin_memory=cfg.train.pin_memory,
    )

    for img_paths, X in tqdm(loader):
        img_names = [Path(path).stem for path in img_paths]
        X = X.to(cfg.base.device)

        with torch.no_grad():
            _, key_states, value_states = frozen_encoder(
                X,
                interpolate_pos_encoding=True,
            )

            key_states = key_states.cpu()
            value_states = value_states.cpu()

            for i in range(len(img_names)):
                states = {
                    "key_states": key_states[:, i].contiguous(),
                    "value_states": value_states[:, i].contiguous(),
                }

                save_path = os.path.join(
                    preload_path,
                    img_names[i] + ".safetensors",
                )
                save_file(states, save_path)


if __name__ == "__main__":
    main()
