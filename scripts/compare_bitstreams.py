"""Verify that two frozen-codec evaluation runs emitted identical files."""

import argparse
import hashlib
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_directories(baseline_dir, gs_residual_dir):
    baseline_files = {
        path.relative_to(baseline_dir)
        for path in baseline_dir.rglob("*")
        if path.is_file()
    }
    gs_residual_files = {
        path.relative_to(gs_residual_dir)
        for path in gs_residual_dir.rglob("*")
        if path.is_file()
    }
    missing_from_gs = sorted(baseline_files - gs_residual_files)
    missing_from_baseline = sorted(gs_residual_files - baseline_files)
    mismatched = [
        relative_path
        for relative_path in sorted(baseline_files & gs_residual_files)
        if file_digest(baseline_dir / relative_path)
        != file_digest(gs_residual_dir / relative_path)
    ]
    return missing_from_gs, missing_from_baseline, mismatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("gs_residual_dir", type=Path)
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    gs_residual_dir = args.gs_residual_dir.resolve()
    if not baseline_dir.is_dir() or not gs_residual_dir.is_dir():
        parser.error("both arguments must be bitstream directories")

    results = compare_directories(baseline_dir, gs_residual_dir)
    missing_from_gs, missing_from_baseline, mismatched = results
    file_count = sum(1 for path in baseline_dir.rglob("*") if path.is_file())
    if file_count == 0:
        parser.error("baseline bitstream directory is empty")

    if missing_from_gs or missing_from_baseline or mismatched:
        for path in missing_from_gs:
            print(f"missing from candidate: {path}")
        for path in missing_from_baseline:
            print(f"missing from baseline: {path}")
        for path in mismatched:
            print(f"content mismatch: {path}")
        raise SystemExit(1)

    print(f"PASS: {file_count} paired bitstreams are byte-identical")


if __name__ == "__main__":
    main()
