from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def case_key(value: str) -> str:
    """Return a lookup-only key. Never use this value as an output name."""
    return value.strip().strip("'").strip('"').casefold()


def build_unique_stem_index(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
        if not path.is_file() or path.suffix.casefold() not in suffixes:
            continue
        key = path.stem.casefold()
        if key in result:
            raise RuntimeError(
                "Case-insensitive stem collision:\n"
                f"  {result[key]}\n  {path}"
            )
        result[key] = path
    return result


def clear_owned_files(directory: Path, suffixes: set[str]) -> None:
    """Clear only files in a script-owned directory, then reject leftovers."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix.casefold() in suffixes:
            path.unlink()
        elif path.is_dir():
            raise RuntimeError(f"Unexpected subdirectory in owned output: {path}")
    leftovers = [p for p in directory.iterdir() if p.is_file() and p.suffix.casefold() in suffixes]
    if leftovers:
        raise RuntimeError(f"Failed to clear old output files in {directory}: {leftovers[:5]}")


def recreate_owned_directory(directory: Path) -> None:
    """Recreate one explicitly owned output directory."""
    if directory.exists():
        if not directory.is_dir():
            raise RuntimeError(f"Owned output is not a directory: {directory}")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=False)


def validate_yolo_file(path: Path, class_count: int, allow_empty: bool = True) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    boxes = 0
    for line_no, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 YOLO fields, got {len(parts)}")
        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid numeric YOLO data") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_no}: class {class_id} outside [0, {class_count - 1}]")
        if not all(0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"{path}:{line_no}: normalized coordinate outside [0, 1]: {coords}")
        if coords[2] <= 0.0 or coords[3] <= 0.0:
            raise ValueError(f"{path}:{line_no}: width/height must be positive: {coords[2:]}")
        boxes += 1
    if not allow_empty and boxes == 0:
        raise ValueError(f"Empty label is not allowed: {path}")
    return boxes


def assert_exact_output(directory: Path, expected_names: Iterable[str], suffix: str = ".txt") -> None:
    expected = set(expected_names)
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.suffix.casefold() == suffix}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"Output-set mismatch in {directory}: expected={len(expected)}, actual={len(actual)}, "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    folded: dict[str, str] = {}
    for name in actual:
        key = Path(name).stem.casefold()
        if key in folded:
            raise RuntimeError(f"Case-insensitive duplicate output stems: {folded[key]} / {name}")
        folded[key] = name
