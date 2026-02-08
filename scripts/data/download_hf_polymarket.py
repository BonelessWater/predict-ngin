#!/usr/bin/env python3
"""
Download Polymarket dataset from HuggingFace.

Source: https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data

Files available:
  - markets.parquet    (68MB)   - Market metadata (268K markets)
  - quant.parquet      (21GB)   - Clean trades, unified YES perspective (170M records)
  - users.parquet      (23GB)   - User behavior, maker/taker split (340M records)
  - trades.parquet     (32GB)   - Processed trades with market linkage (293M records)
  - orderfilled.parquet (31GB)  - Raw blockchain events (293M records)

Usage:
    # Download essential files (markets + clean trades) ~21GB
    python scripts/download_hf_polymarket.py

    # Download everything (~107GB)
    python scripts/download_hf_polymarket.py --all

    # Download specific files
    python scripts/download_hf_polymarket.py --files markets.parquet quant.parquet users.parquet

    # Download to custom directory
    python scripts/download_hf_polymarket.py --output-dir data/polymarket/hf
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ID = "SII-WANGZJ/Polymarket_data"

ESSENTIAL_FILES = [
    "markets.parquet",   # 68MB - market metadata
    "quant.parquet",     # 21GB - clean trades (unified YES perspective)
]

ALL_FILES = [
    "markets.parquet",       # 68MB
    "quant.parquet",         # 21GB
    "users.parquet",         # 23GB
    "trades.parquet",        # 32GB
    "orderfilled.parquet",   # 31GB
]

FILE_SIZES = {
    "markets.parquet": "68 MB",
    "quant.parquet": "21 GB",
    "users.parquet": "23 GB",
    "trades.parquet": "32 GB",
    "orderfilled.parquet": "31 GB",
}


def download_files(files: list[str], output_dir: Path) -> None:
    """Download files from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub[hf_transfer]"])
        from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    for i, filename in enumerate(files, 1):
        size = FILE_SIZES.get(filename, "?")
        print(f"\n[{i}/{len(files)}] Downloading {filename} ({size})...")

        dest = output_dir / filename
        if dest.exists():
            existing_mb = dest.stat().st_size / 1024 / 1024
            print(f"  Already exists ({existing_mb:.0f} MB) - skipping. Use --force to re-download.")
            continue

        start = time.time()
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
            elapsed = time.time() - start
            file_size = Path(path).stat().st_size / 1024 / 1024
            speed = file_size / elapsed if elapsed > 0 else 0
            print(f"  Done: {file_size:.0f} MB in {elapsed:.0f}s ({speed:.1f} MB/s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Try: huggingface-cli download {REPO_ID} {filename} --repo-type dataset --local-dir {output_dir}")

    total_elapsed = time.time() - total_start
    print(f"\nAll downloads complete in {total_elapsed / 60:.1f} minutes")

    # Summary
    print("\nFiles:")
    for f in files:
        p = output_dir / f
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            if size_mb > 1024:
                print(f"  {f}: {size_mb / 1024:.1f} GB")
            else:
                print(f"  {f}: {size_mb:.0f} MB")
        else:
            print(f"  {f}: MISSING")


def main():
    parser = argparse.ArgumentParser(
        description="Download Polymarket dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Download all files (~107GB)")
    parser.add_argument("--files", nargs="+", choices=ALL_FILES, help="Specific files to download")
    parser.add_argument("--output-dir", default="data/polymarket", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.files:
        files = args.files
    elif args.all:
        files = ALL_FILES
    else:
        files = ESSENTIAL_FILES

    total_size = sum(
        float(FILE_SIZES[f].split()[0]) for f in files
    )
    unit = "GB"

    print("=" * 60)
    print("POLYMARKET HUGGINGFACE DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Source: huggingface.co/datasets/{REPO_ID}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(files)} ({total_size:.0f} {unit} total)")
    print()
    for f in files:
        print(f"  - {f} ({FILE_SIZES[f]})")
    print("=" * 60)

    download_files(files, output_dir)


if __name__ == "__main__":
    main()
