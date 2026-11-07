"""Preview both color-constancy methods on Noor training frames."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .color_constancy import correct_rgb
from .experiments import read_prepared
from .io_utils import require
from .reporting import save_figure
from .tracking import write_csv
import matplotlib.pyplot as plt


def preview_noor(data_dir, output_dir, count=6):
    require(count > 0, 'Preview count must be positive.')
    frame, _ = read_prepared(data_dir)
    examples = (frame.loc[frame.source.eq('Noor') & frame.split.eq('train')]
                .sort_values(['video_id', 'file_name']).drop_duplicates('video_id').head(count))
    require(len(examples) > 0, 'The shared training split has no Noor frames.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [('none', 'Original'), ('gray_world', 'Gray World'), ('white_patch', 'White Patch')]
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 2.9 * len(examples)), squeeze=False)
    stats = []
    for row_index, row in enumerate(examples.itertuples()):
        with Image.open(row.image_path) as original:
            rgb = original.convert('RGB')
        original_pixels = np.asarray(rgb)
        for column, (method, title) in enumerate(methods):
            image = correct_rgb(rgb, method)
            pixels = np.asarray(image)
            ax = axes[row_index, column]
            ax.imshow(pixels)
            ax.set_title(f'{title} | {row.video_id}', fontsize=9)
            ax.axis('off')
            mean = pixels.mean(axis=(0, 1))
            maximum = pixels.max(axis=(0, 1))
            stats.append({
                'video_id': row.video_id, 'file_name': row.file_name, 'label': row.label,
                'split': row.split, 'method': method,
                **{f'mean_{channel}': float(mean[i]) for i, channel in enumerate('rgb')},
                **{f'max_{channel}': int(maximum[i]) for i, channel in enumerate('rgb')},
                'fraction_pixels_changed': float(np.any(pixels != original_pixels, axis=-1).mean()),
                'fraction_pixels_with_a_channel_at_255': float(np.any(pixels == 255, axis=-1).mean()),
            })
    fig.tight_layout()
    path = output_dir / 'noor_color_comparison.png'
    save_figure(fig, path)
    write_csv(pd.DataFrame(stats), output_dir / 'noor_color_channel_statistics.csv')
    print(f'Noor color preview: {path}')
    return path
