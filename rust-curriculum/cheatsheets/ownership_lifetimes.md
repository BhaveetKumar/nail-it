# Ownership & Lifetimes Cheat-sheet

- Moves vs Copies; `Copy` trait
- Borrow rules: many `&T` or one `&mut T`
- Slices `&[T]`, `&str` are borrowed views
- Lifetimes name relationships, not durations
- Return references only to valid data

Sources:
- [TRPL Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html) — fetched_at: 2025-09-19T00:00:00Z
