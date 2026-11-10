import unittest

import numpy as np
import pandas as pd
from PIL import Image

from cataract_swin.color_constancy import correct_rgb
from cataract_swin.comparison import FARABI, metric_row
from cataract_swin.experiments import select_preset
from cataract_swin.preprocessing import DEFAULT_PREPROCESSING, FitPad


class ColorConstancyTests(unittest.TestCase):
    def test_gray_world_neutralizes_a_uniform_rgb_cast(self):
        image = Image.new('RGB', (16, 9), (160, 100, 40))
        self.assertEqual(correct_rgb(image, 'gray_world').getpixel((0, 0)), (100, 100, 100))

    def test_white_patch_saturated_channels_and_zero_channel(self):
        pixels = np.array([[[255, 40, 0], [60, 255, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(correct_rgb(Image.fromarray(pixels), 'white_patch'), pixels)
        self.assertEqual(correct_rgb(Image.new('RGB', (1, 1), (125, 100, 0)),
                                     'white_patch').getpixel((0, 0)), (255, 255, 0))

    def test_only_noor_is_corrected_before_the_same_padding(self):
        image = Image.new('RGB', (16, 9), (160, 100, 40))
        pad = FitPad()
        corrected = FitPad({**DEFAULT_PREPROCESSING, 'color_constancy': 'gray_world'})
        np.testing.assert_array_equal(corrected(image, source='Farabi'), pad(image))
        np.testing.assert_array_equal(corrected(image, source='Noor'), pad(correct_rgb(image, 'gray_world')))
        with self.assertRaisesRegex(ValueError, 'source is required'):
            corrected(image)


class PresetTests(unittest.TestCase):
    def test_shared_three_class_evaluation_and_train_membership(self):
        rows = []
        for split in ('train', 'val', 'test'):
            for source, label in [('Farabi', 0), ('Farabi', 1), ('Farabi', 2),
                                  ('Noor', 0), ('YouTube_external', 3)]:
                rows.append({'split': split, 'source': source, 'label': label,
                             'video_id': f'{split}_{source}_{label}'})
        master = pd.DataFrame(rows)
        farabi = select_preset(master, 'farabi3')
        pooled = select_preset(master, 'pooled3')
        pd.testing.assert_frame_equal(farabi[farabi.split.ne('train')], pooled[pooled.split.ne('train')])
        self.assertEqual(pooled.loc[pooled.split.eq('train'), 'source'].tolist(), ['Farabi'] * 3 + ['Noor'])
        pd.testing.assert_frame_equal(select_preset(master, 'gray_world'), master)

    def test_shared_metrics_keep_brunescent_calls_visible(self):
        frame = pd.DataFrame({
            'video_id': ['a', 'b', 'c'], 'label': [0, 1, 2], 'prediction': [3, 1, 2],
            'prob_0': [.2, .0, .0], 'prob_1': [.1, 1., .0], 'prob_2': [.1, .0, 1.],
            'prob_3': [.6, .0, .0],
        })
        result = metric_row(frame, 'baseline', FARABI, 4, 1)
        self.assertEqual(result['macro_f1'], 1.)
        self.assertAlmostEqual(result['native_accuracy'], 2 / 3)
        self.assertEqual(result['native_class_3_calls'], 1)
