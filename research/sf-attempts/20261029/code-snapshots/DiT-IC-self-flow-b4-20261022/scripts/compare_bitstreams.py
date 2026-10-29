import argparse
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root):
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Require two codec-output trees to be byte-identical."
    )
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference_root = Path(args.reference)
    candidate_root = Path(args.candidate)
    if not reference_root.is_dir() or not candidate_root.is_dir():
        raise SystemExit("both inputs must be existing directories")

    reference = collect(reference_root)
    candidate = collect(candidate_root)
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        unexpected = sorted(candidate.keys() - reference.keys())
        raise SystemExit(
            f"file sets differ; missing={missing}, unexpected={unexpected}"
        )
    changed = [
        name for name in reference if reference[name] != candidate[name]
    ]
    if changed:
        raise SystemExit(f"bitstreams differ: {changed}")
    print(f"Byte-identical: {len(reference)} files")


if __name__ == "__main__":
    main()
