---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.372040
Tags: []
Status: draft
---

# Feature Flag Platform (Prompt 5)

Skeleton for rule evaluation + caching + exposure logging.

## Components

- Flag store (in-memory map) with version
- Rule engine (simple predicates now)
- Exposure log buffer (flush batch)

## Run

```bash
cd 06_projects_portfolio/implementations/feature_flag_platform
go run ./...
```

## Extend

- Add Redis pub/sub invalidation
- Add per-flag rollout strategy DSL

## Testing

- Rule evaluation edge cases
- Version bump invalidation path
