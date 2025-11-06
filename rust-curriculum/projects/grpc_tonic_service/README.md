---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.262334
Tags: []
Status: draft
---

# gRPC Service with Tonic

- Prereqs: `protoc` installed (CI uses arduino/setup-protoc).

- Build:

```bash
cargo build -p grpc_tonic_service
```

- Run:

```bash
cargo run -p grpc_tonic_service
```

- Test (automated): `cargo test -p grpc_tonic_service` runs an integration test.

- Test (manual) with grpcurl:

```bash
grpcurl -plaintext -d '{"message":"hello"}' localhost:50051 echo.Echo/Say
```


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**
- [build.rs](./build.rs)

Please replace this auto-generated section with curated content.


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**
- [build.rs](./build.rs)

Please replace this auto-generated section with curated content.


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**
- [build.rs](./build.rs)

Please replace this auto-generated section with curated content.
