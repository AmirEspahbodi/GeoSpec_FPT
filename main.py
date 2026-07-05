import argparse
import json
import os
import random
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from model.rff_axial import QueryEnhancedSideViTClassifier_rff_axial
from omegaconf import ListConfig, OmegaConf

from data.builder import generate_dataset, generate_model, load_weights
from evaluate_model import evaluate_model
from model.GeoSpec_FPT import GeoSpecClassifier
from parameters_count import model_params_summary
from train import train as train_a
from utils.func import DataLoader, add_path_suffix, print_msg
from utils.metrics import Estimator


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg):
    # print configuration
    print_msg("LOADING CONFIG FILE")
    print(OmegaConf.to_yaml(cfg))

    # --- Start of new validation block ---
    # This block validates the command-line arguments after they've been loaded by Hydra.
    print_msg("VALIDATING RUNTIME ARGUMENTS")
    validation_passed = True

    # Validate backbone_trainable_layers (btl)
    if hasattr(cfg.network, "backbone_trainable_layers"):
        btl = cfg.network.backbone_trainable_layers
        # Ensure it's a list and all values are within the allowed range
        if not isinstance(btl, ListConfig) or not all(x in [1, 2, 3, 4] for x in btl):
            print(
                f"ERROR: Invalid 'backbone_trainable_layers': {btl}. All values must be in [1, 2, 3, 4]."
            )
            validation_passed = False
        else:
            print(f"  - OK: Backbone trainable layers set to: {btl}")

    # Validate vit1_feature_strame (v1fs)
    if hasattr(cfg.network, "vit1_feature_strame"):
        v1fs = cfg.network.vit1_feature_strame
        # Ensure it's a list of two numbers, each between 1 and 4
        if (
            isinstance(v1fs, ListConfig)
            and (len(v1fs) == 1 or len(v1fs) == 2)
            and all(1 <= x <= 4 for x in v1fs)
        ):
            print(f"  - OK: ViT1 Feature Strame set to: {v1fs}")
        else:
            print(
                f"ERROR: Invalid 'vit1_feature_strame': {v1fs}. Must be one or two numbers, each between 1 and 4."
            )
            validation_passed = False

    # Validate vit2_feature_strame (v2fs)
    if hasattr(cfg.network, "vit2_feature_strame"):
        v2fs = cfg.network.vit2_feature_strame
        # Ensure it's a list of two numbers, each between 1 and 4
        if (
            isinstance(v2fs, ListConfig)
            and (len(v2fs) == 1 or len(v2fs) == 2)
            and all(1 <= x <= 4 for x in v2fs)
        ):
            print(f"  - OK: ViT2 Feature Strame set to: {v2fs}")
        else:
            print(
                f"ERROR: Invalid 'vit2_feature_strame': {v2fs}. Must be two numbers, each between 1 and 4."
            )
            validation_passed = False
    if hasattr(cfg.base, "training_plan"):
        training_plan = cfg.base.training_plan.strip()
        if training_plan not in ["A", "B", "C", "D"]:
            print(
                f"ERROR: Invalid 'training_plan': {training_plan.strip()}. Must be two str, just A, B, C. D."
            )
            validation_passed = False
    if not validation_passed:
        print_msg("Argument validation failed. Exiting.", warning=True)
        sys.exit(1)
    # --- End of new validation block ---

    # change save path epecially for the model with current configuration, create unqie path for each model+configuration
    cfg.dataset.save_path = (
        f"{cfg.dataset.save_path}\\{cfg.network.model}_{cfg.network.model_id}"
    )

    # create folder
    save_path = cfg.dataset.save_path
    if os.path.exists(save_path):
        if cfg.base.overwrite:
            print_msg(
                f"Save path {save_path} exists and will be overwritten.", warning=True
            )
        else:
            new_save_path = add_path_suffix(save_path)
            cfg.dataset.save_path = new_save_path
            warning = f"Save path {save_path} exists. New save path is set to {new_save_path}."
            print_msg(warning, warning=True)

    os.makedirs(cfg.dataset.save_path, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.dataset.save_path, "cfg.yaml"))

    # check preloading
    if cfg.dataset.preload_path:
        print(f"cfg.dataset.preload_path = {cfg.dataset.preload_path}")
        assert os.path.exists(cfg.dataset.preload_path), "Preload path does not exist."
        print_msg(f"Preloading is enabled using {cfg.dataset.preload_path}")

    if cfg.base.random_seed >= 0:
        set_seed(cfg.base.random_seed, cfg.base.cudnn_deterministic)

    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    frozen_encoder, side_vit_model1 = generate_model(cfg)
    frozen_encoder2, side_vit_model2 = generate_model(cfg)
    del frozen_encoder2

    print(f"type cfg = {type(cfg)}")
    classifier_with_side_vits = GeoSpecClassifier(
        side_vit_b1=side_vit_model1,
        side_vit_b2=side_vit_model2,
        cfg=cfg,
    ).to(cfg.base.device)

    estimator = Estimator(
        cfg.train.metrics, cfg.dataset.num_classes, cfg.train.criterion
    )
    train_pipeline = train_a
    model_summary_json = model_params_summary(classifier_with_side_vits)
    finall_resul_path = os.path.join(cfg.dataset.save_path, "model_summary_json.json")

    with open(finall_resul_path, "w") as fp:
        json.dump(
            model_summary_json,
            fp,
            indent=4,
        )

    used_loss_function = train_pipeline(
        cfg=cfg,
        frozen_encoder=frozen_encoder,
        model=classifier_with_side_vits,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
    )

    print("This is the performance of the best validation model:")
    checkpoint = os.path.join(cfg.dataset.save_path, "best_validation_weights.pt")
    load_weights(classifier_with_side_vits, checkpoint)

    save_metics(
        cfg,
        frozen_encoder,
        classifier_with_side_vits,
        test_dataset,
        "best_validation_weights",
        used_loss_function,
    )

    print("This is the performance of the final model:")
    checkpoint = os.path.join(cfg.dataset.save_path, "final_weights.pt")
    load_weights(classifier_with_side_vits, checkpoint)
    save_metics(
        cfg,
        frozen_encoder,
        classifier_with_side_vits,
        test_dataset,
        "final_weights",
        used_loss_function,
    )


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def save_metics(cfg, frozen_encoder, model, dataset, model_name, used_loss_function):
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

    acc, f1, auc, precision, recall, confusion_matrix = evaluate_model(
        cfg, frozen_encoder, model, dataloader, used_loss_function, cfg.base.device
    )
    confusion_matrix_path = os.path.join(
        cfg.dataset.save_path, f"{model_name}_confusion_matrix.png"
    )
    save_confusion_matrix(confusion_matrix, confusion_matrix_path)

    finall_resul_path = os.path.join(
        cfg.dataset.save_path, f"{model_name}_results.json"
    )

    with open(finall_resul_path, "w") as fp:
        json.dump(
            {
                "acc": acc,
                "f1": f1,
                "auc": auc,
                "precision": precision,
                "recall": recall,
            },
            fp,
            indent=4,
        )


def save_confusion_matrix(confusion_matrix: np.ndarray, save_path: str):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    n = confusion_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(confusion_matrix, cmap="Blues")
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(
            j,
            i,
            f"{val}",
            ha="center",
            va="center",
            fontsize=12,
            color="white" if val > confusion_matrix.max() / 2 else "black",
        )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training with custom arguments for Hydra."
    )
    parser.add_argument(
        "--tp", "--training_plan", type=str, help="Train Plan (e.g., A B C)."
    )
    parser.add_argument(
        "--btl",
        "--backbone_trainable_layers",
        type=int,
        nargs="*",
        help="List of backbone layers to train (e.g., 1 2 3 4).",
    )
    parser.add_argument(
        "--v1fs",
        "--vit1_feature_strame",
        type=int,
        nargs="+",
        help="Two numbers for ViT1 feature stride, each between 1 and 4 (e.g., 1 4).",
    )
    parser.add_argument(
        "--v2fs",
        "--vit2_feature_strame",
        type=int,
        nargs="+",
        help="Two numbers for ViT2 feature stride, each between 1 and 4 (e.g., 2 3).",
    )

    args, unknown_args = parser.parse_known_args()

    hydra_overrides = []
    if args.btl is not None:
        btl_str = "[" + ",".join(map(str, args.btl)) + "]"
        hydra_overrides.append(f"+network.backbone_trainable_layers={btl_str}")

    if args.v1fs is not None:
        v1fs_str = "[" + ",".join(map(str, args.v1fs)) + "]"
        hydra_overrides.append(f"+network.vit1_feature_strame={v1fs_str}")

    if args.v2fs is not None:
        v2fs_str = "[" + ",".join(map(str, args.v2fs)) + "]"
        hydra_overrides.append(f"+network.vit2_feature_strame={v2fs_str}")

    if args.tp is not None:
        tp_str = args.tp
        hydra_overrides.append(f"base.training_plan={tp_str}")

    model_id = f"btl{''.join(map(str, args.btl))}_v1fs{''.join(map(str, args.v1fs))}_v2fs{''.join(map(str, args.v2fs))}_tp{tp_str}"
    hydra_overrides.append(f"+network.model_id={model_id}")

    sys.argv = [sys.argv[0]] + hydra_overrides + unknown_args

    main()
