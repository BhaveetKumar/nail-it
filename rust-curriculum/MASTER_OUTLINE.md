# Rust Curriculum Master Outline

Status: DRAFT
Generated: 2025-09-19

## Resources

- Curated: resources/RESOURCES.md

## Beginner → Intermediate → Advanced → Expert/Core

- Beginner (Foundations)
  - Objectives: Syntax, ownership/borrowing basics, cargo, testing
  - Prereqs: Basic programming
  - Lessons: 12 (suggested practice: 60–90m/lesson)
  - Outcomes: Build/test simple CLI; understand ownership rules
- Intermediate (Idioms & Ecosystem)
  - Objectives: Traits, generics, error handling, iterators, macros (intro)
  - Lessons: 14 (60–120m/lesson)
  - Outcomes: Write ergonomic libraries with tests and docs
- Advanced (Concurrency, Async, Unsafe)
  - Objectives: Threads, async/await, pinning, FFI, unsafe invariants
  - Lessons: 16 (90–150m/lesson)
  - Outcomes: Build async services; reason about safety and FFI
- Expert/Core (Internals & Systems)
  - Objectives: Compiler/MIR, proc-macros, allocators, embedded, WASM, performance
  - Lessons: 18 (120–180m/lesson)
  - Outcomes: Contribute to compiler or core crates; production-grade systems

## Modules

1. Rust Basics (Beginner)
   - Objectives: variables, mutability, types, pattern matching, collections
   - Prereqs: any language basics
   - Lessons: 4
   - Outcome: Implement small CLI with tests
2. Ownership & Borrowing (Beginner)
   - Objectives: move/borrow rules, slices, lifetimes
   - Lessons: 3
   - Outcome: Compile mental model; avoid common borrow-checker errors
3. Tooling & Workflows (Beginner)
   - Objectives: rustup, cargo, fmt, clippy, tests, docs
   - Lessons: 2
   - Outcome: Create workspace; CI-ready project
4. Error Handling (Intermediate)
   - Objectives: Result/Option, anyhow/thiserror, propagation
   - Lessons: 2
   - Outcome: Robust error architecture
5. Traits & Generics (Intermediate)
   - Objectives: trait bounds, associated types, trait objects
   - Lessons: 3
   - Outcome: Generic libraries; object safety understanding
6. Iterators & Closures (Intermediate)
   - Objectives: adapter chains, custom iterators
   - Lessons: 2
   - Outcome: Idiomatic functional style
7. Macros (Intermediate→Advanced)
   - Objectives: macro_rules!, proc-macros, hygiene
   - Lessons: 3
   - Outcome: Author derive macro crate
8. Concurrency (Advanced)
   - Objectives: Send/Sync, channels, atomics
   - Lessons: 3
   - Outcome: Correct concurrent primitives
9. Async Rust (Advanced)
   - Objectives: futures, pinning, executors, tokio/async-std
   - Lessons: 4
   - Outcome: Build async HTTP service with tracing
10. Unsafe & FFI (Advanced)
   - Objectives: invariants, raw pointers, extern "C"
   - Lessons: 3
   - Outcome: Safe wrapper around unsafe
11. Ecosystems: Web, DB, WASM (Expert)
   - Objectives: hyper/reqwest, tonic, sqlx/diesel, wasm-bindgen/yew
   - Lessons: 4
   - Outcome: Full-stack Rust deliverable
12. Embedded & no_std (Expert)
   - Objectives: HALs, RTIC, memory constraints
   - Lessons: 3
   - Outcome: Blink + sensor + OTA pattern
13. Performance & Profiling (Expert)
   - Objectives: criterion, flamegraph, tokio-console, tracing
   - Lessons: 2
   - Outcome: Profiling workflow & regressions CI
14. Compiler Internals & Contribution (Expert/Core)
   - Objectives: MIR, lints, contributing to rustc
   - Lessons: 3
   - Outcome: First rustc PR plan

## Citations (seed sources)

- The Rust Programming Language (TRPL) — https://doc.rust-lang.org/book/ — fetched_at: 2025-09-19T00:00:00Z
- Rust By Example — https://doc.rust-lang.org/rust-by-example/ — fetched_at: 2025-09-19T00:00:00Z
- Rustonomicon — https://doc.rust-lang.org/nomicon/ — fetched_at: 2025-09-19T00:00:00Z
- Async Book — https://rust-lang.github.io/async-book/ — fetched_at: 2025-09-19T00:00:00Z
- Tokio — https://tokio.rs/tokio/tutorial — fetched_at: 2025-09-19T00:00:00Z
