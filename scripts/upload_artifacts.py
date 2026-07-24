"""Upload a local checkpoint tree to the HF dataset (maintainer utility).

Expects a directory shaped like grok_lens/<run>/final.pt, lego/<run>/step_*.pt.

Usage: HF_TOKEN=... uv run python scripts/upload_artifacts.py <folder>
"""

import os
import sys

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    repo_id="brendanlong/lens-loss-grokking-experiment",
    repo_type="dataset",
    folder_path=sys.argv[1],
    commit_message="Update checkpoints",
)
