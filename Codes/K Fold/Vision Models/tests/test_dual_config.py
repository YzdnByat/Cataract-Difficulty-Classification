from pathlib import Path
import unittest

from config_loader import get_winning_config, load_winning_configs, models_for_task


class DualTaskConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.configs = load_winning_configs(root / "configs" / "winning_configs.json")

    def test_task_coverage(self):
        self.assertEqual(len(models_for_task(self.configs, "severity")), 11)
        self.assertEqual(len(models_for_task(self.configs, "poor_dilation")), 10)
        self.assertNotIn("vgg16", models_for_task(self.configs, "poor_dilation"))

    def test_poor_dilation_b2_exact_groups(self):
        config = get_winning_config(self.configs, "poor-dilation", "EfficientNet-B2")
        groups = config["optimizer"]["parameter_groups"]
        observed = {
            group["scopes"][0]: group["learning_rate"] for group in groups
        }
        self.assertEqual(observed, {
            "features.5": 5e-6,
            "features.6": 1e-5,
            "features.7": 1e-5,
            "classifier": 1.5e-4,
        })
        self.assertEqual(config["input_resolution"], 224)
        self.assertIsNone(config["loss"]["class_weights"])

    def test_severity_and_poor_dilation_configs_are_distinct(self):
        severity = get_winning_config(self.configs, "severity", "efficientnet_b5")
        poor = get_winning_config(self.configs, "poor_dilation", "efficientnet_b5")
        self.assertEqual(severity["input_resolution"], 456)
        self.assertEqual(poor["input_resolution"], 224)
        self.assertNotEqual(
            severity["optimizer"]["parameter_groups"],
            poor["optimizer"]["parameter_groups"],
        )


if __name__ == "__main__":
    unittest.main()
