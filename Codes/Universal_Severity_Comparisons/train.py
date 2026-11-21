"""Command-line entry point; CUDA is selected before the worker imports torch."""
from cataract_swin.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
