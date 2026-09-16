"""Upload local model checkpoints or dataset shards to Hugging Face Hub."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi

from train.utils.log import log


def upload(
    folder_path: str | Path,
    repo_id: str,
    repo_type: str = "model",
    commit_message: str | None = None,
) -> None:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        log(f"Error: folder '{folder}' does not exist or is not a directory.", print_console=True)
        sys.exit(1)

    api = HfApi()
    user = api.whoami()
    log(f"Authenticated as HF user: {user.get('name')} ({user.get('fullname')})", print_console=True)
    log(f"Target repository: [{repo_type}] {repo_id}", print_console=True)
    log(f"Source folder: {folder}", print_console=True)

    files = list(folder.glob("*"))
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    log(f"Found {len(files)} files ({total_bytes / (1024**3):.2f} GiB total)", print_console=True)
    for f in sorted(files, key=lambda x: x.name):
        if f.is_file():
            log(f"  - {f.name}: {f.stat().st_size / (1024**2):.1f} MB", print_console=True)

    if commit_message is None:
        commit_message = f"Upload {folder.name} ({total_bytes / (1024**3):.2f} GiB)"

    log(f"Starting upload to {repo_id}...", print_console=True)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type=repo_type,
        commit_message=commit_message,
    )
    log(f"Upload complete! Commit: {commit_info}", print_console=True)
    log(f"Repository URL: https://huggingface.co/{repo_id if repo_type == 'model' else f'datasets/{repo_id}'}", print_console=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload checkpoint or data directory to Hugging Face Hub.")
    parser.add_argument("folder", help="Path to local directory to upload")
    parser.add_argument("--repo-id", default="Ouroboros-Research/Our1-2b", help="HF repo ID (e.g. org/model_name)")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"], help="HF repo type")
    parser.add_argument("--message", default=None, help="Commit message")

    args = parser.parse_args()
    upload(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.message,
    )


if __name__ == "__main__":
    main()
