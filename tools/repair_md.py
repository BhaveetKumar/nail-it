#!/usr/bin/env python3
"""
repair_md.py

Scan Markdown files, detect internal links, create placeholder files for missing targets,
add missing anchors, seed short files with starter content, and generate a JSON report.

AUTO-GENERATED sections are clearly marked for manual review.
"""
import os
import re
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_CREATE = 50

link_re = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
heading_re = re.compile(r"^#{1,6}\s+(.*)", re.M)

def slugify_heading(h):
    # simple GitHub-like slug
    s = h.strip().lower()
    s = re.sub(r"[^a-z0-9 \-]", "", s)
    s = s.replace(' ', '-')
    s = re.sub(r"-+", '-', s)
    return s

report = {
    'files_scanned': 0,
    'links_found': 0,
    'links_fixed': 0,
    'files_created': [],
    'anchors_added': [],
    'files_seeded': [],
    'errors': []
}

def is_external(link):
    return link.startswith('http:') or link.startswith('https:') or link.startswith('mailto:') or link.startswith('#')

def resolve_target(from_path, link):
    # split fragment
    if '#' in link:
        path_part, frag = link.split('#', 1)
    else:
        path_part, frag = link, None

    # handle absolute path relative to repo
    if path_part.startswith('/'):
        target = ROOT.joinpath(path_part.lstrip('/'))
    else:
        target = (from_path.parent / path_part).resolve()

    # if path is directory, point to README.md
    if target.is_dir():
        for name in ('README.md', 'readme.md', 'index.md'):
            candidate = target / name
            if candidate.exists():
                target = candidate
                break
        else:
            target = target / 'README.md'

    # append .md if no ext and file exists
    if not target.exists() and target.suffix == '':
        candidate = Path(str(target) + '.md')
        if candidate.exists():
            target = candidate

    return target, frag

def ensure_file(path: Path):
    if path.exists():
        return False
    if len(report['files_created']) >= MAX_CREATE:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        title = path.stem.replace('-', ' ').replace('_', ' ').title()
        with open(path, 'w', encoding='utf8') as f:
            f.write(f"# {title}\n\n<!-- AUTO-GENERATED - REVIEW REQUIRED -->\n\nThis file was created automatically because an internal link pointed here. Please review and expand.\n")
        report['files_created'].append(str(path.relative_to(ROOT)))
        return True
    except Exception as e:
        report['errors'].append({'op': 'create', 'path': str(path), 'error': str(e)})
        return False

def ensure_anchor(target: Path, frag: str):
    if not frag:
        return False
    try:
        text = target.read_text(encoding='utf8')
    except Exception as e:
        report['errors'].append({'op': 'read', 'path': str(target), 'error': str(e)})
        return False

    # generate possible anchors from headings
    headings = heading_re.findall(text)
    slugs = {slugify_heading(h): h for h in headings}
    if frag in slugs:
        return False

    # fuzzy: try compare frag to heading slugs
    for slug, h in slugs.items():
        if frag in slug or slug in frag:
            return False

    # add anchor heading at end
    new_heading = frag.replace('-', ' ').replace('_', ' ').title()
    add_text = f"\n\n## {new_heading}\n\n<!-- AUTO-GENERATED ANCHOR: originally referenced as #{frag} -->\n\nPlaceholder content. Please replace with proper section.\n"
    try:
        with open(target, 'a', encoding='utf8') as f:
            f.write(add_text)
        report['anchors_added'].append({'file': str(target.relative_to(ROOT)), 'anchor': frag})
        return True
    except Exception as e:
        report['errors'].append({'op': 'append', 'path': str(target), 'error': str(e)})
        return False

def seed_short_file(path: Path):
    try:
        text = path.read_text(encoding='utf8')
    except Exception:
        return False
    words = len(text.split())
    if words >= 200:
        return False
    # gather related files in same dir
    rel_links = []
    for ext in ('*.go', '*.rs', '*.js'):
        for p in path.parent.glob(ext):
            rel_links.append(p.name)
    seed = '\n\n---\n\n## AUTO-GENERATED: Starter Content\n\n<!-- AUTO-GENERATED - REVIEW REQUIRED -->\n\nThis section seeds the document with a short introduction, learning objectives, and related links to code samples.\n\n**Learning objectives:**\n- Understand the core concepts.\n- See practical code examples.\n\n**Related files:**\n'
    for r in rel_links[:10]:
        seed += f"- [{r}](./{r})\n"
    seed += '\nPlease replace this auto-generated section with curated content.\n'
    try:
        with open(path, 'a', encoding='utf8') as f:
            f.write(seed)
        report['files_seeded'].append(str(path.relative_to(ROOT)))
        return True
    except Exception as e:
        report['errors'].append({'op': 'seed', 'path': str(path), 'error': str(e)})
        return False

def process_file(mdpath: Path):
    report['files_scanned'] += 1
    try:
        txt = mdpath.read_text(encoding='utf8')
    except Exception as e:
        report['errors'].append({'op': 'read', 'path': str(mdpath), 'error': str(e)})
        return

    for m in link_re.finditer(txt):
        report['links_found'] += 1
        link = m.group(2).strip()
        if is_external(link):
            continue
        target, frag = resolve_target(mdpath, link)
        # if target exists but fragment missing, add anchor
        if target.exists():
            if frag:
                if ensure_anchor(target, frag):
                    report['links_fixed'] += 1
            continue
        # try common alternatives
        candidates = []
        if not target.exists():
            candidates.append(Path(str(target) + '.md'))
            candidates.append(target.with_name('README.md'))
            candidates.append(target.with_name('index.md'))
        fixed = False
        for c in candidates:
            if c.exists():
                # rewrite link in source file to c relative path
                rel = os.path.relpath(c, mdpath.parent)
                txt = txt.replace(link, rel)
                fixed = True
                report['links_fixed'] += 1
                break
        if fixed:
            continue
        # create placeholder
        created = ensure_file(target)
        if created:
            report['links_fixed'] += 1
            # if fragment present, add anchor
            if frag:
                ensure_anchor(target, frag)

    # after processing links, write back any path fixes
    try:
        mdpath.write_text(txt, encoding='utf8')
    except Exception as e:
        report['errors'].append({'op': 'write', 'path': str(mdpath), 'error': str(e)})

    # seed short files
    seed_short_file(mdpath)

def main():
    md_files = list(ROOT.rglob('*.md'))
    # exclude .git and tools dir and node_modules
    md_files = [p for p in md_files if '.git' not in p.parts and 'node_modules' not in p.parts and 'tools' not in p.parts]

    for p in md_files:
        process_file(p)

    # write report
    out = ROOT / 'repo-heal-report.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print('Done. Report written to', out)

if __name__ == '__main__':
    main()
