from pathlib import Path
import unittest

from config_loader import get_winning_config, load_winning_configs


class ConfigLoaderTests(unittest.TestCase):
    def test_all_notebook_configs_load_with_canonical_names(self):
        project_root = Path(__file__).resolve().parents[1]
        configs = load_winning_configs(project_root / "configs" / "winning_configs.json")
        self.assertEqual(len(configs), 21)
        self.assertEqual(
            get_winning_config(configs, "severity", "EfficientNet-B5")["model"],
            "efficientnet_b5",
        )
        self.assertEqual(
            get_winning_config(configs, "severity", "vgg16")["optimizer"]["parameter_groups"][1]["learning_rate"],
            1e-4,
        )


if __name__ == "__main__":
    unittest.main()
