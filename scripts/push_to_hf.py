"""Push a local folder to the HuggingFace Hub as a private repo.

Covers both project upload cases — they only differ by ``--repo-type`` and the
source folder:

- LoRA adapter (model):  runs/finetune/<run-name>/adapter/
- formatted dataset:     data/finetune/<corpus>/formatted/<model>/

Uploads the folder as-is via ``HfApi.upload_folder`` — it does NOT reload the
gated base model, so it is fast and needs no extra download. Existing remote
files not present locally are left untouched (e.g. a dataset card uploaded
separately survives a data re-push).

License note: SBCSAE is CC BY-ND 3.0 US (No Derivatives). Adapters and formatted
data are derivative works, so repos default to **private**. Do not flip to
public without resolving the license question. The HF token must have write
access to the target namespace (an org needs org write permission, not just
the user scope, or create_repo returns 403).

Usage:
    # adapter (model repo):
    uv run python scripts/push_to_hf.py \
        --folder runs/finetune/t5gemma2-1b-SBCSAE-lora-es/adapter \
        --repo-id aip-scripttuner-team/scripttuner-t5gemma2-1b-sbcsae-casual

    # formatted dataset:
    uv run python scripts/push_to_hf.py --repo-type dataset \
        --folder data/finetune/SBCSAE/formatted/t5gemma2-1b \
        --repo-id aip-scripttuner-team/scripttuner-sbcsae-formatted-t5gemma2-1b
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import dotenv
from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        required=True,
        help="Local folder to upload (its contents become the repo root).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HF repo, e.g. aip-scripttuner-team/scripttuner-...",
    )
    parser.add_argument(
        "--repo-type",
        choices=["model", "dataset"],
        default="model",
        help="HF repo type (default: model).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a PUBLIC repo. Off by default — SBCSAE CC BY-ND forbids it.",
    )
    args = parser.parse_args()

    dotenv.load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN not found (set it in .env or the environment).")

    folder = Path(args.folder)
    if not folder.is_dir():
        parser.error(f"folder not found: {folder}")

    api = HfApi(token=token)
    api.create_repo(
        args.repo_id, repo_type=args.repo_type, private=not args.public, exist_ok=True
    )
    visibility = "PUBLIC" if args.public else "private"
    print(f"Uploading {folder} -> {args.repo_id} ({args.repo_type}, {visibility}) ...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(folder),
        commit_message=f"Upload from {folder.name}",
    )
    prefix = "datasets/" if args.repo_type == "dataset" else ""
    print(f"Done: https://huggingface.co/{prefix}{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
