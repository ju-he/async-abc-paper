"""Git-related helpers shared by plotting and metadata exports."""

from __future__ import annotations

import subprocess
from pathlib import Path


def find_repo_root(anchor: Path | None = None) -> Path | None:
    """Return the nearest parent directory that looks like a git worktree."""
    start = (anchor or Path(__file__)).resolve()
    candidates = [start] + list(start.parents)
    for path in candidates:
        if (path / ".git").exists():
            return path
    return None


def get_git_hash(anchor: Path | None = None) -> str:
    """Return the short HEAD git hash, or ``'unknown'`` on failure."""
    repo_root = find_repo_root(anchor)
    if repo_root is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
