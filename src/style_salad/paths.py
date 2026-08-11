"""Project path helpers used by the module entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
CONFIGS_DIR = REPO_ROOT / "configs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def add_repo_root_to_path() -> None:
    """Keep legacy top-level imports working during the cleanup migration."""
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
