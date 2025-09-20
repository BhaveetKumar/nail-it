---
verified: true
---

# Lesson 1: Rust Intro, Variables, Types

## Overview
Learn basic Rust syntax: variables, mutability, shadowing, types, pattern matching, and collections.

## Concepts
- Variables (`let`), mutability (`mut`), shadowing
- Scalar and compound types
- Pattern matching with `match`
- Collections: `Vec`, `HashMap`

## Idiomatic Examples
```rust
// src/main.rs
fn main() {
    let x = 5; // immutable
    let mut y = 10; // mutable
    let y = y + x; // shadowing
    println!("x = {x}, y = {y}");
}
```

## Hands-on Exercise
Create a program that counts word frequencies from stdin.

### Files
- Cargo.toml
- src/main.rs
- tests/wordcount.rs

### Cargo.toml
```toml
[package]
name = "lesson01_wordcount"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
```

### src/main.rs
```rust
use std::collections::HashMap;
use std::io::{self, Read};

fn main() -> anyhow::Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let counts = wordcount(&input);
    for (w, c) in counts {
        println!("{w} {c}");
    }
    Ok(())
}

fn wordcount(s: &str) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for w in s.split_whitespace() {
        *map.entry(w.to_lowercase()).or_insert(0) += 1;
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_basic() {
        let m = wordcount("a a b");
        assert_eq!(m.get("a"), Some(&2));
        assert_eq!(m.get("b"), Some(&1));
    }
}
```

### Test
```bash
cargo new lesson01_wordcount --vcs none
cd lesson01_wordcount
# Replace Cargo.toml and src/main.rs with the above
cargo test
```

## Common Mistakes
- Forgetting to handle `Result` properly; use `anyhow` for simplicity during learning.

## Further Reading
- TRPL: https://doc.rust-lang.org/book/ — fetched_at: 2025-09-19T00:00:00Z
- Rust By Example: https://doc.rust-lang.org/rust-by-example/ — fetched_at: 2025-09-19T00:00:00Z
