from pathlib import Path
import hashlib, json
def require(condition, message):
    if not condition:
        raise ValueError(message)

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def table_hash(df, columns):
    data = df[columns].sort_values(['video_id', 'file_name']).reset_index(drop=True)
    return hashlib.sha256(data.to_csv(index=False, lineterminator='\n').encode()).hexdigest()

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))
    temp.replace(path)
