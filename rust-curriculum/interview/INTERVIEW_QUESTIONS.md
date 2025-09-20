---
verified: false
---

# Rust Interview Questions (Draft)

Note: This is a structured draft. Expand to 200+ items; categorize by level.

- Junior
  - Explain ownership and borrowing with a small code example.
  - What is the difference between `&T` and `&mut T`? Provide a scenario for each.
  - What does `Option<T>` represent? How do you handle `None` idiomatically?
  - Show a `Result<T, E>` function using `?` and a custom error with `thiserror`.
- Mid-level
  - Explain lifetimes and when explicit annotations are necessary.
  - How do `Send` and `Sync` work? How to ensure a type is safe to share?
  - When would you use `Arc<Mutex<T>>` vs `RwLock<T>` vs channels?
  - Show how to write an async function with `tokio` tasks and cancellation.
- Senior
  - Design an abstraction over unsafe code (FFI or pointer types) with invariants.
  - Compare `axum` vs `actix-web` tradeoffs; when choose each?
  - Implement a property-based test with `proptest` validating an invariant.
  - Discuss backpressure and timeouts in async services.
- Staff
  - Outline steps to contribute to rustc or an RFC workflow.
  - Architect a high-throughput, low-latency pipeline using `tokio`, `mio`, or custom runtime internals.
  - Discuss memory layout optimizations (niche optimization, enum layout) and performance profiling workflows.
