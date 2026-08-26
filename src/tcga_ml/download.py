from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import urllib.request

from .provenance import build_provenance_record, write_provenance
from .sources import SOURCES, get_source


def download_source(key: str, outdir: str | Path, *, force: bool = False) -> Path:
    source = get_source(key)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.filename
    provenance_path = destination.with_suffix(destination.suffix + ".provenance.json")

    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists; pass --force to replace it")

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        request = urllib.request.Request(source.url, headers={"User-Agent": "tcga-rnaseq-ml/0.1"})
        with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    record = build_provenance_record(destination, source_url=source.url, source_key=source.key)
    write_provenance(provenance_path, record)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download open TCGA PanCancer Atlas source files from the GDC.")
    parser.add_argument("--list", action="store_true", help="List registered source keys and exit.")
    parser.add_argument("--source", choices=sorted(SOURCES), help="Source key to download.")
    parser.add_argument("--outdir", default="data/raw", help="Destination directory (default: data/raw).")
    parser.add_argument("--force", action="store_true", help="Replace an existing destination file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for key, source in SOURCES.items():
            print(f"{key}\t{source.filename}\t{source.url}")
        return 0
    if args.source is None:
        print("error: --source is required unless --list is used", file=sys.stderr)
        return 2
    path = download_source(args.source, args.outdir, force=args.force)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
