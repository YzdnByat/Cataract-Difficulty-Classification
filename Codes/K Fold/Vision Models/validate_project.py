"""Validate configuration coverage and model/optimizer scope compatibility."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from config_loader import load_winning_configs, models_for_task


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "winning_configs.json"))
    parser.add_argument("--build-models", action="store_true", help="Also instantiate models without downloading weights")
    args = parser.parse_args()
    configs = load_winning_configs(args.config)
    print(f"Validated {len(configs)} task/model configurations")
    for task in ("severity", "poor_dilation"):
        print(f"{task}: {', '.join(models_for_task(configs, task))}")

    if not args.build_models:
        return
    from models import build_model, build_optimizer_groups, summarize_trainable_parameters

    for key, config in sorted(configs.items()):
        model = build_model(
            config["model"],
            config["num_classes"],
            config["fine_tuning"]["unfreeze_scopes"],
            pretrained=False,
        )
        groups = build_optimizer_groups(model, config["optimizer"]["parameter_groups"])
        counts = summarize_trainable_parameters(model)
        print(f"{key}: groups={len(groups)}, trainable={counts['trainable']:,}/{counts['total']:,}")
        del groups, model
        gc.collect()


if __name__ == "__main__":
    main()
