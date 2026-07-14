import argparse
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifest for ACC-style load testing (2 images per user)."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root folder containing user subfolders (same style as test_ACC.py).",
    )
    parser.add_argument(
        "--output",
        default="acc_manifest.json",
        help="Output manifest JSON path. Default: acc_manifest.json",
    )
    parser.add_argument(
        "--image-glob",
        default="*.jpg",
        help="Image glob inside each user folder. Default: *.jpg",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Limit number of users. 0 means all users.",
    )
    parser.add_argument(
        "--folder-regex",
        default=r"^\d+$",
        help="Regex for valid user folder names. Default: digits only.",
    )
    return parser.parse_args()


def sorted_user_dirs(dataset_root: Path, folder_regex: str) -> list[Path]:
    pat = re.compile(folder_regex)
    dirs = [d for d in dataset_root.iterdir() if d.is_dir() and pat.match(d.name)]

    def sort_key(p: Path):
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)

    return sorted(dirs, key=sort_key)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"[ERROR] Dataset root does not exist: {dataset_root}")
        return 1

    users = sorted_user_dirs(dataset_root, args.folder_regex)
    if args.max_users > 0:
        users = users[: args.max_users]

    manifest = []
    skipped = 0

    for user_dir in users:
        images = sorted(user_dir.glob(args.image_glob))
        if len(images) < 2:
            skipped += 1
            continue

        # Mimic ACC-style pairing: use the first two images of each user.
        enroll_img = images[0].resolve()
        verify_img = images[1].resolve()

        manifest.append(
            {
                "username": user_dir.name,
                "enroll_image_path": str(enroll_img),
                "verify_image_path": str(verify_img),
            }
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("ACC Manifest Builder")
    print("=" * 70)
    print(f"Dataset root          : {dataset_root.resolve()}")
    print(f"Output file           : {output}")
    print(f"Total user folders    : {len(users)}")
    print(f"Users in manifest     : {len(manifest)}")
    print(f"Skipped (<2 images)   : {skipped}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
