---
verified: true
---

# Lesson 5: Collections and Iterators

## Overview

Learn `Vec`, `HashMap`, ownership with collections, iterator adaptors, and writing custom iterators.

## Concepts

- Collection ownership and borrowing patterns
- Iterator trait, adapters (`map`, `filter`, `collect`, `fold`)
- IntoIterator vs Iterator, references and moved items

## Idiomatic Examples

```rust
fn squares(n: u32) -> Vec<u32> {
    (0..n).map(|x| x * x).collect()
}

struct Counter { i: u32, n: u32 }
impl Iterator for Counter {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.n { let v = self.i; self.i += 1; Some(v) } else { None }
    }
}
```

## Hands-on Exercise

Compute word frequency top-k using iterators only.

### Cargo.toml

```toml
[package]
name = "lesson05_iterators"
version = "0.1.0"
edition = "2021"

[dependencies]
itertools = "0.13"
```

### src/lib.rs

```rust
use std::collections::HashMap;
use itertools::Itertools;

pub fn top_k(s: &str, k: usize) -> Vec<(String, usize)> {
    let mut map: HashMap<String, usize> = HashMap::new();
    for w in s.split_whitespace() {
        *map.entry(w.to_lowercase()).or_insert(0) += 1;
    }
    map.into_iter()
        .sorted_by(|a,b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)))
        .take(k)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic() {
        let v = top_k("a a b c c c", 2);
        assert_eq!(v, vec![("c".into(),3),("a".into(),2)]);
    }
}
```

### Run

```bash
cargo new lesson05_iterators --lib --vcs none
cd lesson05_iterators
# replace files
cargo test
```

## Common Mistakes

- Consuming iterators inadvertently; borrow when needed with `iter()`.

## Further Reading

- [TRPL: Collections](https://doc.rust-lang.org/book/ch08-00-common-collections.html) — fetched_at: 2025-09-19T00:00:00Z
- [TRPL: Iterators](https://doc.rust-lang.org/book/ch13-02-iterators.html) — fetched_at: 2025-09-19T00:00:00Z
- [Rust By Example: Iterators](https://doc.rust-lang.org/rust-by-example/trait/iter.html) — fetched_at: 2025-09-19T00:00:00Z