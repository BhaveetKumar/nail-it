---
verified: true
---

# Lesson 3: Tooling & Workflows

## Overview
Use rustup and cargo effectively. Format, lint, test, document.

## Concepts

- rustup toolchains and components
- cargo commands: new, build, run, test, bench, doc, fmt, clippy
- rust-analyzer setup

## Example: Library with tests & docs

```bash
cargo new lesson03_tooling --lib --vcs none
cd lesson03_tooling
```

Create `Cargo.toml`:

```toml
[package]
name = "lesson03_tooling"
version = "0.1.0"
edition = "2021"

[dev-dependencies]
criterion = "0.5"
```

Create `src/lib.rs`:

```rust
/// Adds two numbers.
///
/// # Examples
/// ```
/// assert_eq!(lesson03_tooling::add(2,2), 4);
/// ```
pub fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() { assert_eq!(add(1,2), 3); }
}
```

Run:

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
cargo doc --open
```

## Further Reading
- [rustup](https://rustup.rs/) — fetched_at: 2025-09-19T00:00:00Z
- [Cargo Book](https://doc.rust-lang.org/cargo/) — fetched_at: 2025-09-19T00:00:00Z
