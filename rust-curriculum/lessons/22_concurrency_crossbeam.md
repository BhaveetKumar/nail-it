---
verified: true
---

# Lesson 22: Concurrency with Crossbeam

This lesson introduces Crossbeam's scoped threads and channels for ergonomic, safe concurrency beyond the standard library. We'll implement a parallel sum with `crossbeam::scope` and a simple worker pool using `crossbeam-channel`.

## Key Concepts

- `crossbeam::scope`: spawn threads that can borrow from the parent stack safely.
- `crossbeam-channel`: MPMC channels with powerful combinators and selection.
- Fan-out/fan-in patterns using channels.

## Example

The example crate `examples/lesson22_crossbeam` includes:

- `scoped_sum`: splits work into chunks and sums in parallel using scoped threads.
- `worker_pool`: bounded job queue + unbounded result queue to process jobs concurrently.

## Run the tests

- `cargo test -p lesson22_crossbeam`

## Further Reading

- [Crossbeam Crate](https://crates.io/crates/crossbeam)
- [crossbeam-channel Crate](https://crates.io/crates/crossbeam-channel)
- [Crossbeam GitHub](https://github.com/crossbeam-rs/crossbeam)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
