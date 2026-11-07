from pathlib import Path
from collections import defaultdict
import json, zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from .io_utils import require, file_hash, table_hash, save_json

DATA_ROOT = None
OUTPUT_ROOT = None
SOURCES = {}
LOCAL_FILES = {}
CONFIG = {}

def configure(*, cache_dir, output_root, sources, local_files, training):
    global DATA_ROOT, OUTPUT_ROOT, SOURCES, LOCAL_FILES, CONFIG
    DATA_ROOT = Path(cache_dir)
    OUTPUT_ROOT = Path(output_root)
    SOURCES = sources
    LOCAL_FILES = local_files
    CONFIG = training

def fetch(task, kind):
    spec = SOURCES[task]
    override = LOCAL_FILES[task][kind]
    if override:
        p = Path(override)
        require(p.is_file(), f'Missing local file: {p}')
        return p
    p = DATA_ROOT / task / ('images.zip' if kind == 'zip' else 'labels.' + spec['ext'])
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_file():
        return p
    part = p.with_suffix(p.suffix + '.part')
    if kind == 'metadata' and spec['ext'] == 'xlsx':
        import requests
        response = requests.get('https://docs.google.com/spreadsheets/d/' +
                                spec['metadata_id'] + '/export?format=xlsx', timeout=180)
        response.raise_for_status()
        part.write_bytes(response.content)
    else:
        import gdown
        result = gdown.download(id=spec['zip_id'] if kind == 'zip' else spec['metadata_id'],
                                output=str(part), quiet=False)
        require(result is not None, 'Drive download failed. Check sharing or use LOCAL_FILES.')
    require(part.is_file() and part.stat().st_size > 0, f'Empty download: {p}')
    if kind == 'zip' or spec['ext'] == 'xlsx':
        require(zipfile.is_zipfile(part), f'Invalid ZIP/XLSX: {p}')
    part.replace(p)
    return p

def canonical_metadata(path, task):
    spec = SOURCES[task]
    raw = pd.read_excel(path, sheet_name='Dataset_Severity_V2') if spec['ext'] == 'xlsx' else pd.read_csv(path)
    raw.columns = raw.columns.astype(str).str.strip()
    columns = ['filename', 'videoname'] + spec['classes']
    require(set(columns) <= set(raw.columns), 'Missing metadata columns')
    require(not raw[columns].isna().any().any(), 'Missing metadata values')
    labels = raw[spec['classes']]
    require(labels.isin([0, 1]).all().all() and labels.sum(axis=1).eq(1).all(), 'Invalid one-hot labels')
    df = pd.DataFrame(dict(file_name=raw.filename.astype(str).str.strip(),
                           video_id=raw.videoname.astype(str).str.strip(),
                           label=labels.to_numpy().argmax(axis=1)))
    require(df.file_name.ne('').all() and df.video_id.ne('').all(), 'Empty identifiers')
    require(df.groupby(['video_id', 'file_name']).label.nunique().max() == 1,
            'Conflicting duplicate labels; do not silently discard these')
    before = len(df)
    df = df.drop_duplicates().sort_values(['video_id', 'file_name']).reset_index(drop=True)
    require(not df.file_name.duplicated().any(), 'Filename assigned to multiple videos')
    require(len(df) == spec['rows'] and df.video_id.nunique() == spec['videos'], 'Unexpected dataset size')
    require(df.label.value_counts().sort_index().to_dict() == spec['counts'], 'Unexpected class counts')
    require(table_hash(df, ['file_name', 'video_id', 'label']) == spec['expected_hash'],
            'Metadata differs from the audited CNN label files. Investigate before training.')
    if task == 'severity':
        require(df.groupby('video_id').label.nunique().max() == 1, 'Mixed severity labels')
    print(task, 'removed exact duplicate rows:', before-len(df))
    return df

def extract_archive(path, task):
    digest = file_hash(path)
    root = DATA_ROOT / task / ('extracted_' + digest[:16])
    marker = root / 'complete.json'
    if marker.exists():
        require(json.loads(marker.read_text())['sha256'] == digest, 'Extraction checksum mismatch')
        return root
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        names = set()
        for info in z.infolist():
            dest = (root / info.filename).resolve()
            require(dest == root.resolve() or root.resolve() in dest.parents, 'Unsafe ZIP path')
            require((info.external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlink rejected')
            if not info.is_dir():
                require(dest not in names, 'Duplicate ZIP member')
                names.add(dest)
        z.extractall(root)
    save_json(marker, {'sha256':digest})
    return root

def attach_images(df, root):
    from PIL import Image
    index = defaultdict(list)
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}:
            index[(p.parent.name.strip(), p.name.strip())].append(p)
    df = df.copy()
    paths, hashes, widths, heights = [], [], [], []
    for row in df.itertuples():
        matches = index[(row.video_id, row.file_name)]
        require(len(matches) == 1, f'Missing/ambiguous image: {row.video_id}/{row.file_name}')
        p = matches[0]
        with Image.open(p) as im:
            im.convert('RGB').load()  # Fail before training on unreadable images.
            widths.append(im.width); heights.append(im.height)
        paths.append(str(p)); hashes.append(file_hash(p))
    df['image_path'], df['image_sha256'] = paths, hashes
    df['original_width'], df['original_height'] = widths, heights
    print('Unreferenced archive images:', sum(map(len,index.values()))-len(df))
    return df

def assign_splits(df, seed):
    groups = df.groupby('video_id', sort=True).label.agg(lambda x: ','.join(map(str,sorted(set(x)))))
    train_ids, rest_ids = train_test_split(groups.index.to_numpy(), test_size=.30,
                                          stratify=groups.to_numpy(), random_state=seed)
    val_ids, test_ids = train_test_split(rest_ids, test_size=.50,
                                        stratify=groups.loc[rest_ids].to_numpy(), random_state=seed)
    mapping = {v:s for s,ids in [('train',train_ids),('val',val_ids),('test',test_ids)] for v in ids}
    out = df.copy(); out['split'] = out.video_id.map(mapping)
    require(out.groupby('video_id').split.nunique().eq(1).all(), 'Patient leakage')
    for split in ['train','val','test']:
        require(set(out.loc[out.split==split,'label']) == set(out.label), f'Missing class in {split}')
    if 'image_sha256' in out:
        require(out.groupby('image_sha256').split.nunique().max() == 1,
                'Identical image bytes cross splits. Resolve source identities before training.')
    return out

def prepare_task(task):
    meta_path, zip_path = fetch(task,'metadata'), fetch(task,'zip')
    df = canonical_metadata(meta_path, task)
    df = attach_images(df, extract_archive(zip_path, task))
    df = assign_splits(df, CONFIG['seed'])
    directory = OUTPUT_ROOT / task
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'split_manifest.csv'
    cols = ['file_name','video_id','label','image_sha256','split']
    if manifest.exists():
        require(table_hash(pd.read_csv(manifest),cols)==table_hash(df,cols),
                'Dataset or split changed for this RUN_NAME. Use a new RUN_NAME.')
    df.to_csv(manifest, index=False)
    save_json(directory/'data_sources.json', {**SOURCES[task], 'metadata_sha256':file_hash(meta_path),
                                              'archive_sha256':file_hash(zip_path)})
    print(task, df.groupby('split').agg(frames=('file_name','size'), videos=('video_id','nunique')))
    print(pd.crosstab(df.split,df.label))
    return df
