---
verified: true
---

# Rust Flashcards (Beginner → Advanced)

- Q: What is ownership in Rust? A: Ownership is Rust’s compile-time memory management model where each value has a single owner; when the owner goes out of scope, the value is dropped.
- Q: What does `&T` mean? A: Shared reference to `T` (immutable borrow).
- Q: What does `&mut T` mean? A: Exclusive mutable reference to `T` (mutable borrow).
- Q: What trait bounds are needed to move a type across threads? A: `Send` for moving to another thread; `Sync` to share references across threads.
- Q: What’s the difference between `Box<T>` and `Arc<T>`? A: `Box<T>` provides heap allocation with single ownership; `Arc<T>` provides thread-safe reference-counted shared ownership.
- Q: What is `?` operator? A: Sugar for early-return on `Result`/`Option` with `From`-based conversion.
- Q: When do you need `Pin<T>`? A: For types that must not move after being pinned (e.g., async futures that self-reference).
- Q: What is `unsafe` used for? A: To perform operations the compiler cannot verify for safety (raw pointers, FFI), requiring manual invariants.
- Q: What are `Send` and `Sync` auto traits? A: Marker traits auto-implemented when safe; you should not implement them manually in most cases.
- Q: Difference between `Iterator::map` and `into_iter`? A: `map` transforms elements; `into_iter` consumes the collection to produce an iterator.
