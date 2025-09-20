---
verified: true
---

# Lesson 23: Serde Advanced — Custom (De)Serializers and Enums

This lesson explores advanced Serde features: tagged enums and custom (de)serializers, using base64 encoding for binary data in JSON.

## Key Concepts

- Adjacently tagged enums via `#[serde(tag = "type", content = "data")]`.
- Field-level custom (de)serializers with `#[serde(with = "module")]`.
- Ensuring JSON-friendly representations (e.g., base64 for binary).

## Example

The example crate `examples/lesson23_serde_advanced` includes:

- `Message` enum with `Text` and `Binary(Vec<u8>)`; `Binary` uses `#[serde(with = "b64")]`.
- `b64` module providing `serialize`/`deserialize` with `base64` crate.
- Tests: round-trip for both variants, and JSON tagging assertions.

## Run the tests

- `cargo test -p lesson23_serde_advanced`

## Further Reading

- [Serde Attributes Reference](https://serde.rs/attributes.html)
- [`base64` crate](https://crates.io/crates/base64)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
