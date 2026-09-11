"""Load and validate task/model-specific notebook-winning configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


SUPPORTED_TASKS = {"severity", "poor_dilation"}
REQUIRED_FIELDS = {
    "task", "model", "num_classes", "pretrained", "input_resolution",
    "batch_size", "max_epochs", "loss", "fine_tuning", "optimizer",
    "scheduler", "sampler", "augmentation", "checkpoint_monitor",
}


def normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_task_name(name: str) -> str:
    task = name.strip().lower().replace("-", "_").replace(" ", "_")
    return {"poordilation": "poor_dilation", "dilation": "poor_dilation"}.get(task, task)


def _require_keys(container: Dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = set(keys) - set(container)
    if missing:
        raise ValueError(f"{context} is missing required fields: {sorted(missing)}")


def _validate_row(row: Dict[str, Any], context: str) -> None:
    _require_keys(row, REQUIRED_FIELDS, context)
    if row["task"] not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task in {context}: {row['task']}")
    if int(row["num_classes"]) < 2:
        raise ValueError(f"num_classes must be at least 2 in {context}")
    if int(row["input_resolution"]) <= 0 or int(row["batch_size"]) <= 0:
        raise ValueError(f"Resolution and batch size must be positive in {context}")
    if int(row["max_epochs"]) <= 0:
        raise ValueError(f"max_epochs must be positive in {context}")

    _require_keys(row["loss"], {"name", "label_smoothing", "class_weights"}, f"{context}.loss")
    weights = row["loss"]["class_weights"]
    if weights is not None and len(weights) != int(row["num_classes"]):
        raise ValueError(f"Loss-weight count does not match num_classes in {context}")

    _require_keys(row["fine_tuning"], {"unfreeze_scopes"}, f"{context}.fine_tuning")
    if not row["fine_tuning"]["unfreeze_scopes"]:
        raise ValueError(f"No unfreeze scopes declared in {context}")

    _require_keys(row["optimizer"], {"name", "parameter_groups"}, f"{context}.optimizer")
    if row["optimizer"]["name"] != "AdamW":
        raise ValueError(f"Only notebook-used AdamW is supported in {context}")
    groups = row["optimizer"]["parameter_groups"]
    if not groups:
        raise ValueError(f"No optimizer parameter groups in {context}")
    declared_scopes = []
    for index, group in enumerate(groups):
        _require_keys(group, {"scopes", "learning_rate", "weight_decay"}, f"{context}.optimizer.parameter_groups[{index}]")
        if not group["scopes"] or float(group["learning_rate"]) <= 0:
            raise ValueError(f"Invalid optimizer group {index} in {context}")
        declared_scopes.extend(group["scopes"])
    if len(declared_scopes) != len(set(declared_scopes)):
        raise ValueError(f"Optimizer scopes overlap by declaration in {context}")

    unfreeze = set(row["fine_tuning"]["unfreeze_scopes"])
    if set(declared_scopes) != unfreeze:
        raise ValueError(
            f"Optimizer scopes must exactly cover unfreeze scopes in {context}: "
            f"optimizer={sorted(set(declared_scopes))}, unfreeze={sorted(unfreeze)}"
        )

    _require_keys(row["scheduler"], {"name", "mode", "factor", "patience", "min_learning_rate"}, f"{context}.scheduler")
    _require_keys(row["sampler"], {"name", "replacement", "target_per_class"}, f"{context}.sampler")
    if len(row["sampler"]["target_per_class"]) != int(row["num_classes"]):
        raise ValueError(f"Sampler quota count does not match num_classes in {context}")


def load_winning_configs(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Return validated configs indexed by ``task:model``."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    rows = document.get("configs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No configurations found in {config_path}")

    indexed: Dict[str, Dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        row["model"] = normalize_model_name(str(row.get("model", "")))
        row["task"] = normalize_task_name(str(row.get("task", "")))
        context = f"{row['task']}:{row['model']}"
        _validate_row(row, context)
        if context in indexed:
            raise ValueError(f"Duplicate configuration: {context}")
        indexed[context] = row
    return indexed


def get_winning_config(configs: Dict[str, Dict[str, Any]], task: str, model: str) -> Dict[str, Any]:
    key = f"{normalize_task_name(task)}:{normalize_model_name(model)}"
    if key not in configs:
        available = ", ".join(sorted(configs))
        raise KeyError(f"No notebook-winning configuration for {key}. Available: {available}") from None
    return dict(configs[key])


def models_for_task(configs: Dict[str, Dict[str, Any]], task: str) -> list[str]:
    prefix = f"{normalize_task_name(task)}:"
    return sorted(key.split(":", 1)[1] for key in configs if key.startswith(prefix))
