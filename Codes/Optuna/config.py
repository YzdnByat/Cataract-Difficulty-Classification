"""
config.py
---------
Configuration parameters, directory mappings, label encodings,
and hyperparameter settings for Cataract Grading and Dilation tasks.
"""

import os
from typing import Dict, List


class PathConfig:

    # Set these environment variables in Kaggle BEFORE launching main.py.
    DATA_ROOT: str = os.environ.get("CATARACT_DATA_ROOT", "/kaggle/working/data")
    SEVERITY_CSV_PATH: str = os.environ.get(
        "SEVERITY_CSV_PATH", os.path.join(DATA_ROOT, "metadata", "Dataset_Severity_V2.csv"))
    SEVERITY_IMG_DIR: str = os.environ.get(
        "SEVERITY_IMG_DIR", os.path.join(DATA_ROOT, "Dataset_Severity_V2_extracted", "Dataset_Severity_V2"))
    DILATION_CSV_PATH: str = os.environ.get(
        "DILATION_CSV_PATH", os.path.join(DATA_ROOT, "metadata", "Dataset_poordilation.csv"))
    DILATION_IMG_DIR: str = os.environ.get(
        "DILATION_IMG_DIR", os.path.join(DATA_ROOT, "Dataset_Poordilation_extracted", "Dataset_poordilation"))
    OUTPUT_ROOT_DIR: str = os.environ.get(
        "OPTUNA_OUTPUT_DIR", "/kaggle/working/outputs/optuna_original_ids_v1")

class TaskConfig:

    # -------------------------------------------------------------
    # Label Encodings
    # -------------------------------------------------------------
    SEVERITY_LABEL_MAP: Dict[str, int] = {
        "low": 0,
        "dense": 1,
        "mature": 2,
        "brunescent": 3,
    }

    DILATION_LABEL_MAP: Dict[str, int] = {
        "normal": 0,
        "poor dilation": 1,
    }

    # -------------------------------------------------------------
    # Class Balancing Quotas
    # -------------------------------------------------------------
    SEVERITY_TARGET_PER_CLASS = {
        0: 550,  # low
        1: 700,  # dense
        2: 450,  # mature
        3: 150,  # brunescent
    }

    DILATION_TARGET_PER_CLASS: Dict[int, int] = {
        0: 1000,  # normal
        1: 1000,  # poor dilation
    }

    # -------------------------------------------------------------
    # Loss Weights
    # -------------------------------------------------------------
    SEVERITY_LOSS_WEIGHTS = [
        1.0,
        1.0,
        1.0,
        2.0,
    ]

    DILATION_LOSS_WEIGHTS: List[float] = [
        1.0,
        1.0,
    ]

class ModelConfig:
    MODELS = [
            "resnet18",
            "resnet34",
            "resnet50",
            "densenet121",
            "densenet169",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
        ]

    RUN_GROUPS = {
            "run1": [
                "resnet18",
                "resnet34",
                "resnet50",
                "densenet121",
                "densenet169",
                "efficientnet_b1",
            ],

            "run2": [
                "efficientnet_b2",
                "efficientnet_b3",
                "efficientnet_b4",
                "efficientnet_b5",
            ],
        }

    MODEL_RESOLUTIONS = {
            "resnet18": 224,
            "resnet34": 224,
            "resnet50": 224,
            "densenet121": 224,
            "densenet169": 224,
            "efficientnet_b1": 240,
            "efficientnet_b2": 288,
            "efficientnet_b3": 300,
            "efficientnet_b4": 380,
            "efficientnet_b5": 456,
        }


    # ============================================================
    # Model-specific batch sizes
    #
    # Conservative values for a 16 GB T4.
    # These can later be increased if GPU memory allows.
    # ============================================================

    MODEL_BATCH_SIZES = {
        "resnet18": 64,
        "resnet34": 64,
        "resnet50": 32,

        "densenet121": 32,
        "densenet169": 24,

        "efficientnet_b1": 48,
        "efficientnet_b2": 32,
        "efficientnet_b3": 24,
        "efficientnet_b4": 12,
        "efficientnet_b5": 8,
    }

    # ============================================================
    # Default / fallback batch size
    # ============================================================

    BATCH_SIZE: int = 64

    # ============================================================
    # Final training
    # ============================================================

    MAX_EPOCHS: int = 50

    # Stop if validation Macro-F1 has not improved for this many
    # consecutive epochs.
    EARLY_STOPPING_PATIENCE: int = 10

    EARLY_STOPPING_MIN_DELTA: float = 1e-4

    # ============================================================
    # DataLoader
    # ============================================================

    NUM_WORKERS: int = max(
        1,
        os.cpu_count() // 2 if os.cpu_count() else 2
    )

    RANDOM_STATE: int = 42

    # ============================================================
    # Optuna
    # ============================================================

    OPTUNA_TRIALS: int = 12
    OPTUNA_EPOCHS: int = 8