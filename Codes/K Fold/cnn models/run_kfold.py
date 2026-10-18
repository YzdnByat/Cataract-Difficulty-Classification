"""Command-line entry point for grouped K-fold CNN evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import traceback
from pathlib import Path

from config_loader import (
    get_winning_config,
    load_winning_configs,
    models_for_task,
    normalize_model_name,
    normalize_task_name,
)
from folds import build_fold_audit, load_metadata, make_grouped_folds


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "winning_configs.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video-isolated stratified K-fold evaluation using Colab winning configs"
    )
    parser.add_argument(
        "--model", default="all",
        help="Canonical model, comma-separated model list, or 'all'",
    )
    parser.add_argument(
        "--task", required=True, choices=["severity", "poor_dilation"],
        help="Run exactly one task per command/session",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Run only one zero-based fold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", default="/kaggle/working/outputs/kfold")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=None,
        help="Protocol-changing emergency override for GPU memory limits",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate and audit folds only")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def default_paths(task: str):
    if task == "severity":
        return (
            "/kaggle/working/data/Dataset_Severity/SEV_metadata.csv",
            "/kaggle/working/data/Dataset_Severity/images",
        )
    if task == "poor_dilation":
        return (
            "/kaggle/working/data/Dataset_Poordilation/PD_metadata.csv",
            "/kaggle/working/data/Dataset_Poordilation/images",
        )
    raise ValueError(f"Unsupported task: {task}")


def ensure_consistent_json(path: Path, payload: dict) -> None:
    """Prevent resumed jobs from mixing outputs from different protocols."""
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise RuntimeError(
                f"Existing run settings conflict with this command: {path}. "
                "Use a new --output-dir or remove the old run after reviewing it."
            )
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    task = normalize_task_name(args.task)
    default_csv, default_images = default_paths(task)
    csv_path = args.csv_path or default_csv
    image_dir = args.image_dir or default_images
    configs = load_winning_configs(args.config)
    available_models = models_for_task(configs, task)
    if not available_models:
        raise ValueError(
            f"No winning configurations are available for task={task}. "
            "Do not substitute severity settings for another task."
        )
    if args.list_models:
        print("\n".join(available_models))
        return
    models = available_models if args.model == "all" else [
        normalize_model_name(value) for value in args.model.split(",") if value.strip()
    ]
    unknown = [model for model in models if model not in available_models]
    if unknown:
        raise ValueError(f"Unknown model(s) for {task}: {unknown}. Available: {available_models}")
    if args.fold is not None and not 0 <= args.fold < args.n_splits:
        raise ValueError(f"--fold must be between 0 and {args.n_splits - 1}")

    output_root = Path(args.output_dir) / task
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(csv_path, task)
    folded = make_grouped_folds(metadata, args.n_splits, args.seed)
    fold_bytes = folded[
        ["row_id", "file_name", "video_id", "label", "fold"]
    ].to_csv(index=False).encode("utf-8")
    protocol = {
        "task": task,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "splitter": "StratifiedGroupKFold",
        "group_column": "video_id",
        "fold_assignment_sha256": hashlib.sha256(fold_bytes).hexdigest(),
    }
    ensure_consistent_json(output_root / "protocol.json", protocol)
    folded.to_csv(output_root / "fold_assignments.csv", index=False)
    build_fold_audit(folded).to_csv(output_root / "fold_audit.csv", index=False)
    print(
        f"Prepared {args.n_splits} leakage-free folds from "
        f"{folded['video_id'].nunique()} videos and {len(folded)} frames."
    )
    if args.dry_run:
        print(f"Fold audit saved to {output_root}")
        return

    import torch
    from kfold_engine import aggregate_completed_folds, train_fold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds_to_run = [args.fold] if args.fold is not None else list(range(args.n_splits))
    failures = []
    for model_name in models:
        model_config = get_winning_config(configs, task, model_name)
        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        resolved_run = {
            "winning_config": model_config,
            "execution": {
                "n_splits": args.n_splits,
                "seed": args.seed,
                "batch_size_override": args.batch_size_override,
            },
        }
        ensure_consistent_json(model_dir / "resolved_config.json", resolved_run)
        if model_config["batch_size"] >= 64 and model_config["input_resolution"] >= 380:
            print(
                f"WARNING: {model_name} uses notebook batch_size=64 at "
                f"{model_config['input_resolution']}px and may exceed Kaggle GPU memory. "
                "An override changes the original protocol."
            )

        for fold in folds_to_run:
            metrics_path = model_dir / f"fold_{fold}" / "metrics.json"
            if args.skip_existing and metrics_path.exists():
                print(f"Skipping completed {model_name} fold {fold}")
                continue
            train_df = folded[folded["fold"] != fold].reset_index(drop=True)
            val_df = folded[folded["fold"] == fold].reset_index(drop=True)
            try:
                train_fold(
                    model_config=model_config,
                    task=task,
                    fold=fold,
                    train_df=train_df,
                    val_df=val_df,
                    image_dir=image_dir,
                    output_dir=model_dir,
                    device=device,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    batch_size_override=args.batch_size_override,
                )
            except Exception as error:
                failures.append({"model": model_name, "fold": fold, "error": str(error)})
                traceback.print_exc()
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        aggregate_completed_folds(model_dir, task, range(args.n_splits))

    if failures:
        failure_path = output_root / "failures.json"
        with failure_path.open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2)
        raise RuntimeError(f"{len(failures)} fold run(s) failed; see {failure_path}")


if __name__ == "__main__":
    main()
