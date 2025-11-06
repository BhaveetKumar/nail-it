# Implementation Plan (Post-Scaffold)

## Phase A: Canonicalization

- [ ] Add front matter to all files in `02_system_design/patterns/`.
- [ ] Merge duplicate microservices docs; keep one canonical.
- [ ] Normalize file names to `snake_case` (log rename list).

## Phase B: Automation

- [ ] Script to insert front matter if missing.
- [ ] Generate `catalog.json` with metadata (size, hash, tags).
- [ ] Duplicate hash detector integrated into CI (mark threshold >2 duplicates).

### Automation Script Outline (front_matter_inject.py)

```python
#!/usr/bin/env python3
import pathlib, datetime, yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GUIDE_DIRS = ["02_system_design", "01_backend_core", "03_platform_and_performance"]

def front_matter_template(title: str):
    return {
        'title': title,
        'slug': title.lower().replace(' ', '_'),
        'last_updated': datetime.date.today().isoformat(),
        'level': 'intermediate',
        'tags': [],
        'summary': f"Auto summary placeholder for {title}. Update manually."
    }

def has_front_matter(text: str) -> bool:
    return text.strip().startswith('---')

def process_file(p: pathlib.Path):
    content = p.read_text(encoding='utf-8')
    if has_front_matter(content):
        return False
    title = p.stem.replace('_', ' ').title()
    fm = yaml.dump(front_matter_template(title), sort_keys=False)
    new = f"---\n{fm}---\n\n" + content
    p.write_text(new, encoding='utf-8')
    return True

def main():
    inserted = 0
    for d in GUIDE_DIRS:
        for p in (REPO_ROOT / d).rglob('*.md'):
            if process_file(p):
                inserted += 1
    print(f"Inserted front matter in {inserted} files")

if __name__ == '__main__':
    main()
```

## Phase C: Content Depth

- [ ] Create latency budgeting deep dive.
- [ ] Add multi-region failover doc.
- [ ] Add distributed transactions comparative analysis (saga vs 2PC vs outbox).

## Phase D: Project Execution

- [ ] Scaffold repositories under `06_projects_portfolio/implementations/`.
- [ ] Add k6 load scripts for payment + feature flag services.
- [ ] Add Go benchmarks for Kafka stream processor.

## Phase E: Interview Assets

- [ ] Expand FAANG prompts to 25 total. (DONE)
- [ ] Add answer exemplars for 5 prompts. (DONE)
- [ ] Populate 10 behavioral stories fully fleshed.

## Phase F: Quality & Governance

- [ ] Add CODE_OF_CONDUCT.md and SECURITY.md (if missing here).
- [ ] Add PR template emphasizing front matter + duplication check.
- [ ] Introduce stale file detector (last_updated older than 180 days).

## Tracking Metrics

| Metric | Target |
|--------|--------|
| Front matter coverage | 80%+ in 30 days |
| Duplicate pattern docs | 0 |
| Project implementations | 4 complete |
| Design docs polished | 4 |
| Mock strong-hire rate | ≥70% |

## Review Cadence

- Weekly check on catalog diff.
- Bi-weekly architecture doc peer review.
