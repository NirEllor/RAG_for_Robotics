#!/usr/bin/env python
"""Download and extract a RLBench mirror from Hugging Face."""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="hqfang/rlbench-18-tasks",
        help="Hugging Face dataset repo to download.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=["data/train/*.zip"],
        help="Glob pattern to download from the repo. Repeatable.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repository revision to download.",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "rlbench" / "raw" / "_hf_stage",
        help="Temporary download directory.",
    )
    parser.add_argument(
        "--extract-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "rlbench" / "raw",
        help="Destination directory for extracted RLBench data.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded zip archives after extraction.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the Hugging Face download step and only extract an existing stage dir.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files are already cached.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel extraction workers.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print a live progress line after this many extracted files.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=5.0,
        help="Print a live progress line if this many seconds passed without output.",
    )
    return parser.parse_args()


def _run_download(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. "
            "Run `python3 -m pip install -e .` inside the project venv."
        ) from exc

    print("Downloading from Hugging Face via huggingface_hub.snapshot_download")
    print(f"repo_id={args.repo_id} repo_type={args.repo_type} revision={args.revision}")
    print(f"local_dir={args.stage_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=args.stage_dir,
        local_dir_use_symlinks=False,
        max_workers=4,
        allow_patterns=list(args.include),
        force_download=args.force_download,
    )


def _extract_one_archive(
    archive: Path,
    stage_dir: Path,
    extract_root: Path,
    keep_archives: bool,
    *,
    progress_every: int,
    progress_seconds: float,
) -> str:
    relative_parts = archive.relative_to(stage_dir).parts
    if len(relative_parts) < 3:
        raise ValueError(f"Unexpected archive path: {archive}")
    split_name = relative_parts[1]
    destination_root = extract_root / split_name
    destination_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    start = time.monotonic()
    last_progress = start
    with zipfile.ZipFile(archive, "r") as handle:
        members = handle.infolist()
        total_members = len(members)
        print(
            f"[extract-start] {archive.name} -> {destination_root} "
            f"({total_members} files)",
            flush=True,
        )
        for member in members:
            handle.extract(member, destination_root)
            extracted += 1
            now = time.monotonic()
            should_log = extracted == total_members
            if progress_every > 0 and extracted % progress_every == 0:
                should_log = True
            if progress_seconds > 0 and (now - last_progress) >= progress_seconds:
                should_log = True
            if should_log:
                elapsed = now - start
                rate = extracted / elapsed if elapsed > 0 else 0.0
                print(
                    f"[extract-progress] {archive.name}: "
                    f"{extracted}/{total_members} files "
                    f"({rate:.1f} files/s)",
                    flush=True,
                )
                last_progress = now
    if not keep_archives:
        archive.unlink()
    elapsed = time.monotonic() - start
    print(
        f"[extract-done] {archive.name}: {extracted}/{extracted} files "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    return str(archive)


def _extract_archives(
    stage_dir: Path,
    extract_root: Path,
    keep_archives: bool,
    workers: int,
    *,
    progress_every: int,
    progress_seconds: float,
) -> None:
    zip_files = sorted(stage_dir.rglob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip archives found under {stage_dir}")
    workers = max(1, int(workers))
    if workers == 1:
        total = len(zip_files)
        print(f"Extracting {total} archives with 1 worker", flush=True)
        for index, archive in enumerate(zip_files, start=1):
            print(f"[archive {index}/{total}] starting {archive.name}", flush=True)
            _extract_one_archive(
                archive,
                stage_dir,
                extract_root,
                keep_archives,
                progress_every=progress_every,
                progress_seconds=progress_seconds,
            )
        return

    print(f"Extracting {len(zip_files)} archives with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _extract_one_archive,
                archive,
                stage_dir,
                extract_root,
                keep_archives,
                progress_every=progress_every,
                progress_seconds=progress_seconds,
            ): archive
            for archive in zip_files
        }
        completed = 0
        total = len(zip_files)
        for future in as_completed(futures):
            archive = futures[future]
            future.result()
            completed += 1
            print(f"[archive {completed}/{total}] done {archive.name}", flush=True)


def main() -> int:
    args = _parse_args()
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    args.extract_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        _run_download(args)
    _extract_archives(
        args.stage_dir,
        args.extract_root,
        args.keep_archives,
        args.workers,
        progress_every=args.progress_every,
        progress_seconds=args.progress_seconds,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
