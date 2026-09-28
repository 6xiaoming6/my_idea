#!/usr/bin/env python3
"""Rename existing v24 custom experiment folders and repair indexed paths."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from stmoe_imputer.utils.run_naming import experiment_label

OUTPUT = ROOT / "outputs/v24-COE"
STAMP = re.compile(r"^\d{8}_\d{6}")


def moves() -> list[tuple[Path, Path]]:
    planned = []
    for dataset in ("TaxiBJ", "BikeNYC"):
        custom = OUTPUT / dataset / "custom"
        if not custom.is_dir():
            continue
        for old in custom.iterdir():
            if not old.is_dir() or not re.match(r"^v\d+_", old.name):
                continue
            stamps = [match.group() for path in old.rglob("*") if path.is_dir()
                      if (match := STAMP.match(path.name))]
            if not stamps:
                raise ValueError(f"No dated run under {old}")
            seed = re.search(r"_seed(\d+)$", old.name)
            seeds = ({seed.group(1)} if seed else
                     {match.group(1) for path in old.rglob("*") if path.is_dir()
                      if (match := re.search(r"_seed(\d+)(?:_|$)", path.name))})
            if len(seeds) != 1:
                raise ValueError(f"Cannot infer a unique seed from {old}: {seeds}")
            new = old.with_name(f"{min(stamps)}_{experiment_label(old.name)}_seed{next(iter(seeds))}")
            if new.exists() or any(existing == new for _, existing in planned):
                raise FileExistsError(new)
            planned.append((old, new))
    return sorted(planned)


def apply(planned: list[tuple[Path, Path]]) -> int:
    replacements = []
    for old, new in planned:
        replacements.append((str(old), str(new)))
        replacements.append((str(old.relative_to(ROOT)), str(new.relative_to(ROOT))))
    updated = 0
    for path in OUTPUT.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".csv", ".jsonl"}:
            continue
        if "source_snapshot" in path.parts:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        modified = original
        for old, new in replacements:
            modified = modified.replace(old, new)
        if modified != original:
            path.write_text(modified, encoding="utf-8")
            updated += 1
    for old, new in planned:
        old.rename(new)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply verified directory and path updates")
    args = parser.parse_args()
    planned = moves()
    print(f"Planned {len(planned)} directory renames")
    for old, new in planned:
        print(f"{old.relative_to(ROOT)} -> {new.name}")
    if args.apply:
        updated = apply(planned)
        print(f"Renamed {len(planned)} directories; updated {updated} JSON/CSV index files")
        (OUTPUT / "summary").mkdir(exist_ok=True)
        (OUTPUT / "summary/output_name_migration.json").write_text(
            json.dumps({"directories": [{"old": str(old), "new": str(new)}
                                        for old, new in planned],
                        "updated_index_files": updated}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
