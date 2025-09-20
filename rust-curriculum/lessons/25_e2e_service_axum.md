---
verified: true
---

# Lesson 25: E2E Service with Axum + Serde + Tracing

Build a tiny service that stores typed items in-memory with JSON over HTTP. Demonstrates integrating Axum routing, Serde (including custom base64), and Tracing.

## Endpoints

- `GET /health` → 200 OK
- `POST /items/:key` → 201 Created (body: `Item`)
- `GET /items/:key` → 200 OK with `Item` or 404

## Data Model

- `Item` enum with adjacently tagged JSON: `Text(String)` and `Binary(Vec<u8>)` where Binary uses base64 via `#[serde(with = "b64")]`.

## Observability

- `tracing` spans on handlers via `#[instrument]` and test output via `with_test_writer`.

## Run the tests

- `cargo test -p lesson25_e2e_service`

## Further Reading

- [Axum Guide](https://docs.rs/axum/latest/axum/)
- [Tracing](https://docs.rs/tracing/latest/tracing/)
- [Serde Attributes](https://serde.rs/attributes.html)
