"""Archive the complete run, including every .pth and copied test image."""
from pathlib import Path
import argparse
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',help='Default: a ZIP next to the run directory.')
    parser.add_argument('--compress',action='store_true',help='Compress files; slower for large checkpoints.')
    args=parser.parse_args();root=Path(args.run_dir).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    destination=Path(args.output).expanduser().resolve() if args.output else root.parent/(root.name+'.zip')
    if destination==root or root in destination.parents:raise ValueError('Write the archive outside the run directory.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+'.tmp')
    count=0
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp') and path.name!='run.lock':
                z.write(path,path.relative_to(root.parent));count+=1
    temporary.replace(destination)
    print(f'Archived {count} files: {destination}')


if __name__=='__main__':main()
