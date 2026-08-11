#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish a local TIC bundle + SSOT model card to a Hugging Face model repo.

Weights are never read from this git tree. Pass --bundle-dir to a local compile
output (e.g. .../Qwen2.5-7B-Instruct-TIC). The model card defaults to
model-cards/<repo-basename>/README.md under this repository.

Uploads every file in the bundle directory (compile layout as-is). The Hub
README is always overlaid from model-cards SSOT (bundle README.md is skipped).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_card(repo_id: str) -> Path:
    name = repo_id.split("/")[-1]
    return _repo_root() / "model-cards" / name / "README.md"


def _validate_bundle(bundle_dir: Path) -> None:
    if not bundle_dir.is_dir():
        raise SystemExit(f"bundle dir not found: {bundle_dir}")
    tic = bundle_dir / "model.tic"
    if not tic.is_file():
        raise SystemExit(f"missing model.tic in {bundle_dir}")


def _stage(bundle_dir: Path, card_path: Path, stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(bundle_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == "README.md":
            continue
        dest = stage_dir / p.name
        os.symlink(p.resolve(), dest)
    shutil.copy2(card_path, stage_dir / "README.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hub model id, e.g. isiroai/Qwen2.5-7B-Instruct-TIC",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Local TIC bundle directory containing model.tic",
    )
    parser.add_argument(
        "--card",
        type=Path,
        default=None,
        help="Path to SSOT model card (default: model-cards/<name>/README.md)",
    )
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Upload only README.md from the SSOT card path",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private (default: public)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage and print paths without calling the Hub",
    )
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.expanduser().resolve()
    card_path = (args.card or _default_card(args.repo_id)).expanduser().resolve()
    if not card_path.is_file():
        raise SystemExit(f"model card not found: {card_path}")

    _validate_bundle(bundle_dir)

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: "
            "pip install -U huggingface_hub"
        ) from e

    api = HfApi()

    if args.card_only:
        print(f"upload card -> {args.repo_id}:README.md from {card_path}")
        if args.dry_run:
            return 0
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update model card from isiro model-cards SSOT",
        )
        print(f"https://huggingface.co/{args.repo_id}")
        return 0

    with tempfile.TemporaryDirectory(prefix="isiro-hf-publish-") as tmp:
        stage_dir = Path(tmp) / "bundle"
        _stage(bundle_dir, card_path, stage_dir)
        staged = sorted(p.name for p in stage_dir.iterdir())
        print(f"repo:   {args.repo_id}")
        print(f"bundle: {bundle_dir}")
        print(f"card:   {card_path}")
        print(f"staged: {', '.join(staged)}")
        if args.dry_run:
            return 0

        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(stage_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Publish TIC bundle + model card",
        )

    print(f"https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
