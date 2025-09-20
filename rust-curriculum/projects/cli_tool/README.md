---
stage: draft
updated_at: 2025-09-19T00:00:00Z
---

# Project: CLI Log Processor

## Goal

Build a fast, robust CLI to parse logs (JSON/NDJSON and text), filter, aggregate, and output summaries.

## Requirements

- Subcommands: `stats`, `filter`, `tail`.
- Input: file(s) or stdin; auto-detect format; streaming.
- Filtering: field match, range, regex.
- Aggregations: count, sum, avg, percentiles (p50/p95/p99).
- Output: table, JSON, CSV; color and quiet modes.
- Exit codes: non-zero on parse errors unless `--ignore-errors`.

## Tech Choices

- `clap` for CLI, `serde`/`serde_json`, `regex`, `itertools`, `anyhow`/`thiserror`.

## Milestones

- M1: Skeleton CLI + basic `stats` on NDJSON.
- M2: Streaming parser and filters on JSON and text via regex.
- M3: Aggregations and multiple outputs; tests and benches.
- M4: Packaging, docs, and CI (fmt/clippy/test/audit).

## CI Notes

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo audit`

## Stretch

- Plugins via trait objects; async reading; gzip support.
