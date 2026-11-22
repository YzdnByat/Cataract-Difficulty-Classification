"""Save geometry for every frame and deterministic training-only padding previews."""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from .io_utils import save_json
from .config import DEFAULT_MODEL, model_preprocessing_metadata
from .preprocessing import FitPad, geometry


def save_preprocessing_audit(manifest, classes, task_dir, config, *, model=DEFAULT_MODEL):
    task_dir = Path(task_dir)
    spec = model_preprocessing_metadata(config, model)
    save_json(task_dir/'preprocessing.json', spec)
    records = []
    for row in manifest.itertuples():
        if hasattr(row, 'original_width') and hasattr(row, 'original_height'):
            width, height = int(row.original_width), int(row.original_height)
        else:
            with Image.open(row.image_path) as image:
                width, height = image.size
        records.append({'file_name': row.file_name, 'video_id': row.video_id,
                        'label': row.label, 'class_name': classes[row.label], 'split': row.split,
                        'source': getattr(row, 'source', None),
                        **geometry(width, height, config['input_size'])})
    table = pd.DataFrame(records)
    table.to_csv(task_dir/'preprocessing_frames.csv', index=False)
    table.groupby(['split', 'label', 'class_name']).agg(
        frames=('file_name', 'size'), mean_padding_fraction=('padding_fraction', 'mean'),
        max_padding_fraction=('padding_fraction', 'max')).reset_index().to_csv(
            task_dir/'preprocessing_summary.csv', index=False)
    # Inspect training data only here; neither augmentation RNG nor torch is used.
    paths = manifest.set_index(['video_id', 'file_name']).image_path
    directory = task_dir/'preprocessing_previews'
    directory.mkdir(exist_ok=True)
    transform = FitPad(config)
    selected = []
    for label, group in table[table.split == 'train'].groupby('label', sort=True):
        row = group.sort_values(['padding_fraction', 'video_id', 'file_name'],
                                ascending=[False, True, True]).iloc[0]
        with Image.open(paths.loc[(row.video_id, row.file_name)]) as original:
            rgb = original.convert('RGB')
            view = transform(rgb, source=row.source)
            preview = Image.new('RGB', (728, 320), '#eef1f4')
            draw = ImageDraw.Draw(preview)
            draw.text((12, 10), f'{classes[label]} | train | {row.video_id}', fill='#172635')
            draw.text((12, 35), f'Original {rgb.width} x {rgb.height} (display scaled)', fill='#172635')
            thumb = ImageOps.contain(rgb, (450, 250), Image.Resampling.LANCZOS)
            preview.paste(thumb, (12+(450-thumb.width)//2, 60+(250-thumb.height)//2))
            size=config['input_size']
            draw.text((480, 35), f'Model RGB: {size} x {size} (scaled)', fill='#172635')
            preview.paste(view.resize((224,224),Image.Resampling.LANCZOS), (480, 60))
            name = f'class_{label}.png'
            preview.save(directory/name)
        selected.append({**row.to_dict(), 'preview': name,
                         'selection': 'training frame with largest padding fraction; tie by video/filename'})
    pd.DataFrame(selected).to_csv(directory/'index.csv', index=False)
