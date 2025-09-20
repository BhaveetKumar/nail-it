---
stage: draft
updated_at: 2025-09-19T00:00:00Z
---

# Project: Async Web Service

## Goal

Implement a small async web service with request tracing, persistence, and graceful shutdown.

## Requirements

- Endpoints: `POST /items`, `GET /items/:id`, `GET /health`.
- Storage: `sqlx` (SQLite) or `sled` for simplicity.
- Concurrency: Tokio; graceful shutdown on SIGINT/SIGTERM.
- Observability: `tracing` + `tracing-subscriber`, request ids, latency histograms.
- Config: env vars and `--config` file.

## Tech Choices

- `axum` or `hyper`, `tokio`, `serde`, `sqlx`/`sled`, `tracing`.

## Milestones

- M1: HTTP server + healthz + shutdown.
- M2: Items CRUD with storage; migrations for sqlx.
- M3: Tracing middleware and metrics.
- M4: Dockerfile, Makefile targets, CI (fmt/clippy/test/audit).

## CI Notes

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo audit`

## Stretch

- Rate limiting, backpressure tests, chaos tests.


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
