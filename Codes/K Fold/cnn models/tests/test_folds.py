import unittest

import pandas as pd
import tempfile
from pathlib import Path

from folds import build_fold_audit, load_metadata, make_grouped_folds, validate_fold_assignments


class FoldTests(unittest.TestCase):
    def test_grouped_folds_have_no_video_leakage_and_full_class_coverage(self):
        rows = []
        row_id = 0
        for label in range(4):
            for video_number in range(10):
                video_id = f"class_{label}_video_{video_number}"
                for frame in range(3):
                    rows.append(
                        {
                            "row_id": row_id,
                            "file_name": f"{video_id}_{frame}.jpg",
                            "video_id": video_id,
                            "label": label,
                        }
                    )
                    row_id += 1
        df = pd.DataFrame(rows)
        folded = make_grouped_folds(df, n_splits=5, random_state=42)
        validate_fold_assignments(folded, 5)
        self.assertEqual(folded.groupby("video_id")["fold"].nunique().max(), 1)
        self.assertEqual(len(build_fold_audit(folded)), 20)

    def test_poor_dilation_legacy_columns_are_normalized(self):
        rows = []
        for label in range(2):
            for video_number in range(8):
                rows.append({
                    "filename": f"frame_{label}_{video_number}.jpg",
                    "videoname": f"video_{label}_{video_number}",
                    "label": "normal" if label == 0 else "poor_dilation",
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            loaded = load_metadata(path, "poor_dilation")
        self.assertEqual(set(loaded.columns), {"row_id", "file_name", "video_id", "label"})
        folded = make_grouped_folds(loaded, n_splits=4, random_state=42)
        validate_fold_assignments(folded, 4)


if __name__ == "__main__":
    unittest.main()
