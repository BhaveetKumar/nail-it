---
verified: true
---

# Lesson 9: Macros Intro (`macro_rules!`)

## Overview

Learn declarative macros, pattern matching, repetition, hygiene basics, and best practices.

## Concepts

- `macro_rules!` patterns: literals, idents, expr, tt
- Repetition: `*`, `+`, separators, nested
- Hygiene: why names don’t clash; use of `$crate`

## Idiomatic Examples

```rust
#[macro_export]
macro_rules! vec_of_strings {
    ($($x:expr),* $(,)?) => {{
        let mut v = Vec::new();
        $( v.push($x.to_string()); )*
        v
    }}
}
```

## Hands-on Exercise

Write `maplit!{ key => value, ... }` macro to build `HashMap<String, i32>`.

### Cargo.toml

```toml
[package]
name = "lesson09_macros_intro"
version = "0.1.0"
edition = "2021"
```

### src/lib.rs

```rust
use std::collections::HashMap;

#[macro_export]
macro_rules! maplit {
    ( $( $k:expr => $v:expr ),* $(,)? ) => {{
        let mut m: HashMap<String, i32> = HashMap::new();
        $( m.insert($k.to_string(), $v as i32); )*
        m
    }}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn works() {
        let m = maplit!{"a" => 1, "b" => 2};
        assert_eq!(m.get("a"), Some(&1));
        assert_eq!(m.get("b"), Some(&2));
    }
}
```

### Run

```bash
cargo new lesson09_macros_intro --lib --vcs none
cd lesson09_macros_intro
# replace files
cargo test
```

## Common Mistakes

- Overly broad matchers; be explicit about tokens and separators.

## Further Reading

- [TRPL: Macros](https://doc.rust-lang.org/book/ch19-06-macros.html) — fetched_at: 2025-09-19T00:00:00Z
