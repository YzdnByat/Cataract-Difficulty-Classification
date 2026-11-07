"""Prepare shared data, inspect Noor corrections or compare completed runs."""
import argparse
from pathlib import Path

from cataract_swin.experiments import prepare_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare = subparsers.add_parser('prepare', help='Validate the severity archive and make one shared split.')
    prepare.add_argument('--data-dir', type=Path, required=True)
    prepare.add_argument('--images-zip', type=Path, help='Local severity image ZIP.')
    prepare.add_argument('--metadata', type=Path, help='Local original severity XLSX.')
    prepare.add_argument('--images-drive-id', help='Drive file ID for the severity image ZIP.')
    prepare.add_argument('--metadata-drive-id', help='Drive file ID for the original severity XLSX.')

    preview = subparsers.add_parser('preview', help='Save original, Gray World and White Patch panels for Noor training frames.')
    preview.add_argument('--data-dir', type=Path, required=True)
    preview.add_argument('--output-dir', type=Path, required=True)
    preview.add_argument('--count', type=int, default=6)

    compare = subparsers.add_parser('compare', help='Score completed runs on matching test cohorts.')
    compare.add_argument('--runs-dir', type=Path, action='append', required=True,
                         help='A run folder or its parent. Repeat for separate saved outputs.')
    compare.add_argument('--output-dir', type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare_data(args.data_dir, images_zip=args.images_zip, metadata=args.metadata,
                     images_drive_id=args.images_drive_id, metadata_drive_id=args.metadata_drive_id)
    elif args.command == 'preview':
        from cataract_swin.color_preview import preview_noor
        preview_noor(args.data_dir, args.output_dir, count=args.count)
    else:
        from cataract_swin.comparison import compare_runs
        compare_runs(args.runs_dir, args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
