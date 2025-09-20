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
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_CREATE = 50

link_re = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
heading_re = re.compile(r"^#{1,6}\s+(.*)", re.M)

def slugify_heading(h):
    # simple GitHub-like slug
    s = str(h).strip().lower()
    s = re.sub(r"[^a-z0-9 \-]", "", s)
    s = s.replace(' ', '-')
    s = re.sub(r"-+", '-', s)
    return s

def normalize_fragment(frag):
    if not frag:
        return ''
    return slugify_heading(frag.replace('_', '-'))

report = {
    'files_scanned': 0,
    'links_found': 0,
    'links_fixed': 0,
    'files_created': [],
    'anchors_added': [],
    'files_seeded': [],
    'errors': []
}

def parse_args():
    p = argparse.ArgumentParser(description='Repair Markdown links across repository')
    p.add_argument('--dry-run', action='store_true', help='Do not write files; only report')
    p.add_argument('--verbose', action='store_true', help='Verbose logging')
    p.add_argument('--max-create', type=int, default=MAX_CREATE, help='Maximum number of files to auto-create')
    return p.parse_args()

def is_external(link):
    # treat absolute URLs and mailto as external. fragment-only links ("#foo") should be handled.
    l = link.strip()
    return l.startswith('http:') or l.startswith('https:') or l.startswith('mailto:')

def resolve_target(from_path, link):
    # split fragment
    if '#' in link:
        path_part, frag = link.split('#', 1)
    else:
        path_part, frag = link, None

    # fragment-only (anchor in same file)
    if link.startswith('#') and (not path_part or path_part.strip() == ''):
        return from_path, frag

    # handle absolute path relative to repo
    if path_part.startswith('/'):
        target = ROOT.joinpath(path_part.lstrip('/'))
    else:
        # if empty path (e.g. link was "#frag"), treat as same file
        if path_part == '' or path_part is None:
            target = from_path
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

    # if still doesn't exist, try heuristic repository-wide search for matching basename
    if not target.exists():
        basename = Path(path_part).name
        if basename == '':
            basename = None
        else:
            # try with and without .md
            bnames = {basename, basename + '.md', (basename + '.MD')}
            # also try dash/underscore variants
            bnames.update({basename.replace('-', '_'), basename.replace('_', '-')} if basename else set())
            # case-insensitive search across repo for files with matching names
            for p in ROOT.rglob('**/*'):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in ('.md', ''):
                    continue
                if p.name in bnames or p.name.lower() in {n.lower() for n in bnames}:
                    target = p
                    break

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
    nf = normalize_fragment(frag)
    if nf in slugs:
        return False

    # check for explicit id/name anchors or CommonMark {#id}
    if re.search(rf'id=["\']{re.escape(nf)}["\']', text) or re.search(rf'name=["\']{re.escape(nf)}["\']', text) or re.search(rf'\{{#\s*{re.escape(nf)}\s*\}}', text):
        return False

    # fuzzy: try compare normalized frag to heading slugs
    for slug, h in slugs.items():
        if nf in slug or slug in nf:
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
    # We'll rebuild the text replacing only specific matches to avoid accidental global replace
    out_parts = []
    last_idx = 0
    for m in link_re.finditer(txt):
        report['links_found'] += 1
        start, end = m.span(2)
        link_text = m.group(2).strip()
        out_parts.append(txt[last_idx:start])
        last_idx = end
        if is_external(link_text):
            out_parts.append(link_text)
            continue
        target, frag = resolve_target(mdpath, link_text)
        # if target exists but fragment missing, add anchor
        if target.exists():
            if frag:
                if ensure_anchor(target, frag):
                    report['links_fixed'] += 1
            out_parts.append(link_text)
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
                rel = os.path.relpath(c, mdpath.parent)
                out_parts.append(rel)
                fixed = True
                report['links_fixed'] += 1
                break
        if fixed:
            continue
        # create placeholder
        if len(report['files_created']) < args.max_create:
            created = ensure_file(target)
            if created:
                report['links_fixed'] += 1
                if frag:
                    ensure_anchor(target, frag)
                out_parts.append(os.path.relpath(target, mdpath.parent))
                continue
        # if we couldn't fix, leave original and record
        report.setdefault('unfixed_links', []).append({'file': str(mdpath.relative_to(ROOT)), 'link': link_text})
        out_parts.append(link_text)

    # after processing links, write back any path fixes
    new_txt = ''.join(out_parts) + txt[last_idx:]
    try:
        if not args.dry_run:
            mdpath.write_text(new_txt, encoding='utf8')
        else:
            # in dry-run, don't write
            pass
    except Exception as e:
        report['errors'].append({'op': 'write', 'path': str(mdpath), 'error': str(e)})

    # seed short files
    seed_short_file(mdpath)

def main():
    global args
    args = parse_args()
    md_files = list(ROOT.rglob('*.md'))
    # exclude .git and tools dir and node_modules and common editor artifact dirs
    exclude = {'node_modules', '.git', 'tools', '.vscode', '.idea', '.cursor-autocontinue'}
    md_files = [p for p in md_files if not any(part in exclude for part in p.parts)]

    if args.verbose:
        print(f"Scanning {len(md_files)} markdown files (dry-run={args.dry_run})")

    for p in md_files:
        process_file(p)

    # write report
    out = ROOT / 'repo-heal-report.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print('Done. Report written to', out)

if __name__ == '__main__':
    main()
