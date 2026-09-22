from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\Yazdan\Desktop\Thesis\pipeline\data"
)

SEVERITY_DIR = DATA_ROOT / "Dataset_Severity"
POOR_DILATION_DIR = DATA_ROOT / "Dataset_Poordilation"

SEVERITY_CSV = SEVERITY_DIR / "SEV_metadata.csv"
POOR_DILATION_CSV = POOR_DILATION_DIR / "PD_metadata.csv"

SEVERITY_IMAGE_DIR = SEVERITY_DIR / "images"
POOR_DILATION_IMAGE_DIR = POOR_DILATION_DIR / "images"

OUTPUT_DIR = DATA_ROOT / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LABEL DEFINITIONS
# ============================================================

SEVERITY_LABELS = {
    0: "Low",
    1: "Dense",
    2: "Mature",
    3: "Brunescent",
}

POOR_DILATION_LABELS = {
    0: "Normal",
    1: "Poor dilation",
}


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze_dataset(
    csv_path: Path,
    image_dir: Path,
    dataset_name: str,
    label_map: dict,
):
    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found:\n{csv_path}")

    df = pd.read_csv(csv_path)

    print(f"\nMetadata file: {csv_path}")
    print(f"Metadata rows: {len(df):,}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = {"file_name", "video_id", "label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Frame count per class
    # --------------------------------------------------------

    counts = (
        df["label"]
        .value_counts()
        .reindex(label_map.keys(), fill_value=0)
        .sort_index()
    )

    results = pd.DataFrame({
        "label": counts.index,
        "class_name": [label_map[label] for label in counts.index],
        "frame_count": counts.values,
    })

    results["percentage"] = (
        results["frame_count"] /
        results["frame_count"].sum() *
        100
    )

    # --------------------------------------------------------
    # Unique videos per class
    # --------------------------------------------------------

    video_counts = (
        df.groupby("label")["video_id"]
        .nunique()
        .reindex(label_map.keys(), fill_value=0)
    )

    results["unique_videos"] = video_counts.values

    print("\nClass distribution:")
    print(results.to_string(index=False))

    print(f"\nTotal frames: {results['frame_count'].sum():,}")
    print(f"Total unique videos: {df['video_id'].nunique():,}")

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    actual_labels = set(df["label"].dropna().unique())
    expected_labels = set(label_map.keys())

    missing_classes = expected_labels - actual_labels
    unknown_classes = actual_labels - expected_labels

    if missing_classes:
        print(
            f"\nWARNING: Dataset contains no frames for "
            f"expected label(s): {sorted(missing_classes)}"
        )

    if unknown_classes:
        print(
            f"\nWARNING: Dataset contains unexpected label(s): "
            f"{sorted(unknown_classes)}"
        )

    # --------------------------------------------------------
    # Verify image files
    # --------------------------------------------------------

    if image_dir.exists():

        image_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
        }

        actual_image_files = {
            file.name
            for file in image_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in image_extensions
        }

        metadata_files = set(df["file_name"].astype(str))

        missing_images = metadata_files - actual_image_files
        extra_images = actual_image_files - metadata_files

        print("\nImage verification:")
        print(f"Images physically found: {len(actual_image_files):,}")
        print(f"Images listed in CSV:    {len(metadata_files):,}")
        print(f"Missing images:          {len(missing_images):,}")
        print(f"Unlisted images:         {len(extra_images):,}")

        if missing_images:
            print("\nFirst missing images:")
            for file in sorted(missing_images)[:10]:
                print(f"  {file}")

    else:
        print(f"\nWARNING: Image directory not found:\n{image_dir}")

    # --------------------------------------------------------
    # Save results table
    # --------------------------------------------------------

    safe_name = dataset_name.lower().replace(" ", "_")

    csv_output = OUTPUT_DIR / f"{safe_name}_class_distribution.csv"

    results.to_csv(csv_output, index=False)

    # --------------------------------------------------------
    # Plot frame counts
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        results["class_name"],
        results["frame_count"],
    )

    ax.set_title(
        f"{dataset_name} - Frame Distribution by Class",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Frames")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    # Put number above each bar
    for bar, count in zip(bars, results["frame_count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()

    plot_output = OUTPUT_DIR / f"{safe_name}_frame_distribution.png"

    plt.savefig(
        plot_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved table:")
    print(csv_output)

    print("\nSaved plot:")
    print(plot_output)

    return results


# ============================================================
# RUN ANALYSIS
# ============================================================

severity_results = analyze_dataset(
    csv_path=SEVERITY_CSV,
    image_dir=SEVERITY_IMAGE_DIR,
    dataset_name="Severity",
    label_map=SEVERITY_LABELS,
)

poor_dilation_results = analyze_dataset(
    csv_path=POOR_DILATION_CSV,
    image_dir=POOR_DILATION_IMAGE_DIR,
    dataset_name="Poor Dilation",
    label_map=POOR_DILATION_LABELS,
)


# ============================================================
# COMBINED SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nSeverity:")
print(severity_results.to_string(index=False))

print("\nPoor Dilation:")
print(poor_dilation_results.to_string(index=False))

print("\nAnalysis complete.")
print(f"Results saved to:\n{OUTPUT_DIR}")