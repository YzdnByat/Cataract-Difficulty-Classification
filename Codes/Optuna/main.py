"""
main.py
-------
Master orchestration script for benchmarking multiple CNN architectures
across cataract severity and poor-dilation classification tasks.

Workflow for every model/task pair:

1. Create video-isolated train/validation/test splits.
2. Run Optuna hyperparameter optimization.
3. Save the best Optuna hyperparameters.
4. Build a fresh pretrained model.
5. Train with the best hyperparameters for up to MAX_EPOCHS.
6. Use early stopping when validation Macro-F1 stops improving.
7. Save the best validation checkpoint.
8. Evaluate that checkpoint on the held-out test set.
9. Save metrics, plots, logs and benchmark summary.
"""

import os
import gc
import csv
import json
import argparse
import traceback

import torch

from config import PathConfig, ModelConfig
from dataset import build_dataloaders
from optimize import run_hpo
from train import train_model


# =============================================================
# HELPERS
# =============================================================

def normalize_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "_")


def append_summary(
    summary_path: str,
    model_name: str,
    task: str,
    img_size: int,
    batch_size: int,
    best_params: dict,
    results: dict,
):
    """
    Appends one completed model/task experiment to the global CSV.
    """

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    file_exists = os.path.isfile(summary_path)

    row = {
        "model": model_name,
        "task": task,
        "input_resolution": img_size,
        "batch_size": batch_size,

        "lr_head": best_params["lr_head"],
        "lr_backbone": best_params["lr_backbone"],
        "weight_decay": best_params["weight_decay"],
        "label_smoothing": best_params["label_smoothing"],

        "epochs_trained": results["epochs_trained"],
        "best_epoch": results["best_epoch"],

        "best_val_macro_f1": results["best_val_macro_f1"],

        "test_accuracy": results["test_metrics"]["accuracy"],
        "test_macro_f1": results["test_metrics"]["macro_f1"],
        "test_weighted_f1": results["test_metrics"]["weighted_f1"],
        "test_roc_auc": results["test_metrics"]["roc_auc"],
    }

    with open(summary_path, "a", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# =============================================================
# SINGLE MODEL / TASK PIPELINE
# =============================================================

def run_pipeline(
    model_name: str,
    task: str,
    run_optuna: bool = True
):

    model_name = normalize_model_name(model_name)

    if model_name not in ModelConfig.MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: {ModelConfig.MODELS}"
        )

    print("\n" + "=" * 80)
    print(
        f"STARTING BENCHMARK | "
        f"MODEL: {model_name} | "
        f"TASK: {task}"
    )
    print("=" * 80)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # Model-specific configuration
    # ---------------------------------------------------------

    img_size = ModelConfig.MODEL_RESOLUTIONS[model_name]

    batch_size = ModelConfig.MODEL_BATCH_SIZES[model_name]

    print(
        f"Input resolution: "
        f"{img_size}x{img_size}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    output_dir = os.path.join(
        PathConfig.OUTPUT_ROOT_DIR,
        f"{model_name}_{task}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Build data loaders
    # ---------------------------------------------------------

    train_loader, val_loader, test_loader, split_dfs = (
        build_dataloaders(
            task=task,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=ModelConfig.NUM_WORKERS,
        )
    )

    print("\nVideo-isolated splits:")

    print(
        f"  Train frames: "
        f"{len(split_dfs['train'])}"
    )

    print(
        f"  Val frames:   "
        f"{len(split_dfs['val'])}"
    )

    print(
        f"  Test frames:  "
        f"{len(split_dfs['test'])}"
    )

    # ---------------------------------------------------------
    # Hyperparameter optimization
    # ---------------------------------------------------------

    if run_optuna:

        best_params = run_hpo(
            model_name=model_name,
            task=task,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=ModelConfig.OPTUNA_TRIALS,
            device=device,
        )

    else:

        # Baseline values when Optuna is disabled

        if "efficientnet" in model_name:

            best_params = {
                "lr_head": 1e-3,
                "lr_backbone": 2e-5,
                "weight_decay": 1e-4,
                "label_smoothing": 0.05,
            }

        else:

            best_params = {
                "lr_head": 3e-4,
                "lr_backbone": 3e-5,
                "weight_decay": 1e-4,
                "label_smoothing": 0.05,
            }

    # ---------------------------------------------------------
    # Save selected HPO configuration
    # ---------------------------------------------------------

    best_params_path = os.path.join(
        output_dir,
        "best_hyperparameters.json"
    )

    with open(
        best_params_path,
        "w"
    ) as f:

        json.dump(
            best_params,
            f,
            indent=4
        )

    print("\nSelected hyperparameters:")

    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # ---------------------------------------------------------
    # Final full training
    # ---------------------------------------------------------

    print(
        f"\n>>> Final Training: "
        f"maximum {ModelConfig.MAX_EPOCHS} epochs <<<"
    )

    results = train_model(
        model_name=model_name,
        task=task,

        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,

        num_epochs=ModelConfig.MAX_EPOCHS,

        lr_head=best_params["lr_head"],
        lr_backbone=best_params["lr_backbone"],
        weight_decay=best_params["weight_decay"],
        label_smoothing=best_params["label_smoothing"],

        output_dir=output_dir,
        device=device,

        early_stopping_patience=(
            ModelConfig.EARLY_STOPPING_PATIENCE
        ),

        early_stopping_min_delta=(
            ModelConfig.EARLY_STOPPING_MIN_DELTA
        ),
    )

    # ---------------------------------------------------------
    # Global benchmark summary
    # ---------------------------------------------------------

    summary_path = os.path.join(
        PathConfig.OUTPUT_ROOT_DIR,
        "benchmark_summary.csv"
    )

    append_summary(
        summary_path=summary_path,
        model_name=model_name,
        task=task,
        img_size=img_size,
        batch_size=batch_size,
        best_params=best_params,
        results=results,
    )

    print("\nFinished benchmark:")
    print(f"  Model: {model_name}")
    print(f"  Task:  {task}")
    print(f"  Output: {output_dir}")

    # ---------------------------------------------------------
    # GPU cleanup
    # ---------------------------------------------------------

    del train_loader
    del val_loader
    del test_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# =============================================================
# MAIN
# =============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Cataract multi-model benchmarking pipeline"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "run1", "run2"] + ModelConfig.MODELS,
        help="Run one model, run1, run2, or all models.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=[
            "severity",
            "poor_dilation",
            "all",
        ],
        help=(
            "Clinical task. "
            "Use 'all' for both datasets."
        ),
    )

    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help=(
            "Skip Optuna and use baseline "
            "hyperparameters."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Determine models
    # ---------------------------------------------------------

    if args.model == "all":
        models_to_run = ModelConfig.MODELS

    elif args.model == "run1":
        models_to_run = ModelConfig.RUN_GROUPS["run1"]

    elif args.model == "run2":
        models_to_run = ModelConfig.RUN_GROUPS["run2"]

    else:
        models_to_run = [args.model]

    # ---------------------------------------------------------
    # Determine tasks
    # ---------------------------------------------------------

    if args.task == "all":

        tasks_to_run = [
            "severity",
            "poor_dilation",
        ]

    else:

        tasks_to_run = [
            args.task
        ]

    print("\n" + "=" * 80)
    print("BENCHMARK PLAN")
    print("=" * 80)

    print(
        f"Models: {len(models_to_run)}"
    )

    for model_name in models_to_run:
        print(f"  - {model_name}")

    print(
        f"\nTasks: {len(tasks_to_run)}"
    )

    for task in tasks_to_run:
        print(f"  - {task}")

    print(
        f"\nTotal experiments: "
        f"{len(models_to_run) * len(tasks_to_run)}"
    )

    # ---------------------------------------------------------
    # Run every model/task combination
    # ---------------------------------------------------------

    failed_runs = []

    for model_name in models_to_run:

        for task in tasks_to_run:

            try:

                run_pipeline(
                    model_name=model_name,
                    task=task,
                    run_optuna=not args.skip_optuna,
                )

            except Exception as exc:

                print("\n" + "!" * 80)

                print(
                    f"FAILED: "
                    f"{model_name} | {task}"
                )

                print(
                    f"Error: {exc}"
                )

                traceback.print_exc()

                print("!" * 80)

                failed_runs.append(
                    {
                        "model": model_name,
                        "task": task,
                        "error": str(exc),
                    }
                )

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("ALL REQUESTED BENCHMARKS FINISHED")
    print("=" * 80)

    if failed_runs:

        print(
            f"\nFailed experiments: "
            f"{len(failed_runs)}"
        )

        for failure in failed_runs:

            print(
                f"  {failure['model']} | "
                f"{failure['task']} | "
                f"{failure['error']}"
            )

    else:

        print(
            "\nAll experiments completed successfully."
        )


if __name__ == "__main__":
    main()