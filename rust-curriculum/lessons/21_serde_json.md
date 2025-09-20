---
verified: true
---

# Lesson 21: Data Serialization with Serde (JSON)

In this lesson, you will learn how to serialize and deserialize Rust data structures using Serde and the `serde_json` crate. We'll cover key attributes like `rename` and `skip_serializing_if`, and implement a small roundtrip example with tests.

## Why Serde?

Serde is Rust's de-facto framework for data serialization. It is highly performant, extensible, and supports multiple formats (JSON, YAML, CBOR, MessagePack, etc.).

## Key Concepts

- Derive `Serialize`/`Deserialize` on your types.
- Use attributes like `rename`, `default`, `skip_serializing_if` to control the JSON shape.
- Roundtrip tests help ensure compatibility and correctness.

## Example

The example crate `examples/lesson21_serde` defines:

- A `User` struct with a renamed field and an optional field that is omitted when `None`.
- Helper functions `to_json` and `from_json` that use `serde_json` and a small error type.
- Tests that validate round-trip behavior and attribute effects.

## Run the tests

- `cargo test -p lesson21_serde`

## Further Reading

- [Serde Docs](https://serde.rs/)
- [`serde` crate](https://crates.io/crates/serde)
- [`serde_json` crate](https://crates.io/crates/serde_json)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
