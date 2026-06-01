"""Sync the source-only RWR++ vendor tree into MENTOR-RL."""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = Path("/lustre/orion/syb114/proj-shared/Personal/smithkp/sandbox/rwr_hpc")
DEFAULT_TARGET_DIR = REPO_ROOT / "external" / "rwr_hpc"
ALLOWED_TARGET_ROOT = DEFAULT_TARGET_DIR.resolve()

INCLUDE_PATHS = (
    Path("README.md"),
    Path("LICENSE"),
    Path("CMakeLists.txt"),
    Path("CMakeLists_frontier.txt"),
    Path("apps"),
    Path("libs"),
    Path("python"),
    Path("R"),
    Path("scripts") / "compile_frontier.sh",
    Path("scripts") / "skeleton_rwr_frontier.sh",
)

IGNORE_PATTERNS = (
    ".git",
    ".vscode",
    "__pycache__",
    "build*",
    "results",
    "logs",
    "slurm",
    "*.o",
    "*.e",
    "*.pyc",
    "*.profraw",
    "Rplots.pdf",
    ".DS_Store",
)


def _ignore_names(_directory: str, names: list[str]) -> set[str]:
    """Return names that should be skipped inside copied source directories."""

    ignored = set()
    for name in names:
        if any(fnmatch.fnmatch(name, pattern) for pattern in IGNORE_PATTERNS):
            ignored.add(name)
    return ignored


def _remove_existing(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_target_dir(target_dir: Path) -> None:
    if target_dir != ALLOWED_TARGET_ROOT and not _is_relative_to(target_dir, ALLOWED_TARGET_ROOT):
        raise ValueError(
            "target_dir must be external/rwr_hpc or one of its subdirectories; "
            f"got {target_dir}"
        )


def _copy_path(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing(target)

    if source.is_dir():
        shutil.copytree(source, target, ignore=_ignore_names)
        return

    shutil.copy2(source, target)


def sync_rwr_hpc_source(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    target_dir: Path = DEFAULT_TARGET_DIR,
    dry_run: bool = True,
) -> list[tuple[Path, Path]]:
    """Sync selected RWR++ source paths and return the planned copy actions."""

    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    _validate_target_dir(target_dir)

    copy_actions = []
    missing_paths = []
    for relative_path in INCLUDE_PATHS:
        source_path = source_dir / relative_path
        target_path = target_dir / relative_path
        if not source_path.exists():
            missing_paths.append(relative_path)
            continue
        copy_actions.append((source_path, target_path))

    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required source paths are missing under {source_dir}: {missing}")

    for source_path, target_path in copy_actions:
        action = "Would copy" if dry_run else "Copying"
        print(f"{action}: {source_path} -> {target_path}")
        if not dry_run:
            _copy_path(source_path, target_path)

    return copy_actions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync the source-only RWR++ vendor tree into external/rwr_hpc."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"RWR++ source directory. Defaults to {DEFAULT_SOURCE_DIR}.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Destination vendor directory. Defaults to {DEFAULT_TARGET_DIR}.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copy actions without writing files. This is the default.",
    )
    mode.add_argument(
        "--copy",
        action="store_true",
        help="Actually copy the selected source files and directories.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    dry_run = not args.copy
    try:
        sync_rwr_hpc_source(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            dry_run=dry_run,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
