---
verified: true
---

# Lesson 2: Ownership, Borrowing, Lifetimes (Basics)

## Overview
Understand Rust's ownership model: moves, clones, borrowing (immutable/mutable), slices, and a first look at lifetimes.

## Concepts

- Ownership and move semantics
- Copy types vs non-Copy
- Immutable vs mutable borrows; aliasing rules
- Slices and string slices
- Intro to lifetimes on function signatures

## Idiomatic Examples

```rust
// examples/ownership_basics.rs
fn take_ownership(s: String) { println!("{}", s); }
fn borrow_str(s: &str) { println!("{}", s); }
fn main() {
    let a = String::from("hello");
    take_ownership(a);
    // a is moved here; can't use a

    let b = String::from("world");
    borrow_str(&b);
    println!("still have b: {}", b);
}
```

## Hands-on Exercise
Implement a `first_word` function returning a slice `&str` to the first word in a string. Avoid allocations.

### Cargo.toml

```toml
[package]
name = "lesson02_ownership"
version = "0.1.0"
edition = "2021"

[dev-dependencies]
proptest = "1"
```

### src/lib.rs

```rust
pub fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b' ' { return &s[..i]; }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty() { assert_eq!(first_word(""), ""); }
    #[test]
    fn one() { assert_eq!(first_word("abc"), "abc"); }
    #[test]
    fn two() { assert_eq!(first_word("abc def"), "abc"); }
}
```

### Run

```bash
cargo new lesson02_ownership --lib --vcs none
cd lesson02_ownership
# replace Cargo.toml and src/lib.rs with above
cargo test
```

## Common Mistakes
- Returning references to data that goes out of scope; ensure lifetimes are valid.
- Taking `String` by value when `&str` or `&String` suffices; prefer borrowed parameters.

## Further Reading
- [TRPL: Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html) — fetched_at: 2025-09-19T00:00:00Z
- [Rust By Example: Ownership](https://doc.rust-lang.org/rust-by-example/scope/ownership.html) — fetched_at: 2025-09-19T00:00:00Z
