#!/usr/bin/env python3
"""Front matter injection utility.
Scans markdown files and prepends standardized YAML front matter if missing.
Derived from `_meta/implementation_plan.md` script outline.
"""
from __future__ import annotations
import os, sys, datetime
from pathlib import Path

EXCLUDE_DIRS = {'.git', '.github', 'node_modules', 'tools', '_meta'}
EXT = {'.md'}

TEMPLATE = """---
# Auto-generated front matter
Title: {title}
LastUpdated: {ts}
Tags: []
Status: draft
---\n\n"""

def front_matter_template(title: str) -> str:
    return TEMPLATE.format(title=title, ts=datetime.datetime.utcnow().isoformat())

def has_front_matter(lines: list[str]) -> bool:
    return len(lines) >= 2 and lines[0].strip() == '---'

def process_file(path: Path) -> bool:
    if path.suffix not in EXT:
        return False
    with path.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    if has_front_matter(lines):
        return False
    title = path.stem.replace('_', ' ').title()
    fm = front_matter_template(title)
    with path.open('w', encoding='utf-8') as f:
        f.write(fm)
        f.writelines(lines)
    return True

def should_skip_dir(p: Path) -> bool:
    return p.name in EXCLUDE_DIRS

def main(root: str) -> None:
    root_path = Path(root)
    injected = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                if process_file(p):
                    injected += 1
            except Exception as e:
                print(f"Error processing {p}: {e}", file=sys.stderr)
    print(f"Injected front matter into {injected} files.")

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(target)
