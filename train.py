import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
from omegaconf import DictConfig

from geospec_train.config import CoreConfig
from geospec_train.data import build_dataloaders
from geospec_train.engine import Trainer
from geospec_train.utils import ensure_dir, set_seed
from geospec_model.fpt_modules.builder import build_model
from geospec_model.geospec_fpt import GeoSpecClassifier


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    core_cfg = CoreConfig.from_hydra(cfg)

    output_dir = Path(core_cfg.logging.output_dir)
    ensure_dir(output_dir)

    core_cfg.save_json(output_dir / "config.json")

    seed = core_cfg.base.random_seed
    if seed < 0:
        seed = core_cfg.logging.seed

    set_seed(
        seed,
        deterministic=core_cfg.logging.deterministic,
    )

    train_loader, val_loader, test_loader = build_dataloaders(core_cfg)

    side_vit_1 = build_model(core_cfg)
    side_vit_2 = build_model(core_cfg)

    model = GeoSpecClassifier(
        side_vit_b1=side_vit_1,
        side_vit_b2=side_vit_2,
        cfg=core_cfg,
    )

    trainer = Trainer(
        cfg=core_cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    trainer.fit()

    if test_loader is not None:
        trainer.evaluate_best_on_test()


if __name__ == "__main__":
    main()
