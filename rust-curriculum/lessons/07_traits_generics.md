---
verified: true
---

# Lesson 7: Traits & Generics

## Overview

Understand generics, trait bounds, associated types, default methods, and blanket impls.

## Concepts

- Generic functions and impls
- Trait bounds (`T: Trait`, `where`), default methods
- Associated types vs generic parameters
- Blanket implementations, coherence (orphan rules)

## Idiomatic Examples

```rust
pub trait Storage {
    type Key: Ord;
    type Value;
    fn put(&mut self, k: Self::Key, v: Self::Value);
    fn get(&self, k: &Self::Key) -> Option<&Self::Value>;
}

pub struct MapStore<K: Ord, V> { inner: std::collections::BTreeMap<K, V> }
impl<K: Ord, V> Default for MapStore<K, V> { fn default() -> Self { Self { inner: Default::default() } } }

impl<K: Ord, V> Storage for MapStore<K, V> {
    type Key = K;
    type Value = V;
    fn put(&mut self, k: K, v: V) { self.inner.insert(k, v); }
    fn get(&self, k: &K) -> Option<&V> { self.inner.get(k) }
}
```

## Hands-on Exercise

Implement `min_by_key<T, F, K>(slice: &[T], f: F) -> Option<&T>` where `F: Fn(&T) -> K` and `K: Ord`.

### Cargo.toml

```toml
[package]
name = "lesson07_traits_generics"
version = "0.1.0"
edition = "2021"
```

### src/lib.rs

```rust
pub fn min_by_key<T, F, K>(slice: &[T], f: F) -> Option<&T>
where
    F: Fn(&T) -> K,
    K: Ord,
{
    let mut it = slice.iter();
    let first = it.next()?;
    let mut best = first;
    let mut best_k = f(best);
    for item in it {
        let k = f(item);
        if k < best_k { best = item; best_k = k; }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn works() {
        let a = [3,1,2];
        assert_eq!(min_by_key(&a, |x| *x), Some(&1));
    }
}
```

### Run

```bash
cargo new lesson07_traits_generics --lib --vcs none
cd lesson07_traits_generics
# replace files
cargo test
```

## Common Mistakes

- Over-constraining trait bounds; push bounds to call sites when possible.

## Further Reading

- [TRPL: Generics](https://doc.rust-lang.org/book/ch10-00-generics.html) — fetched_at: 2025-09-19T00:00:00Z
- [TRPL: Traits](https://doc.rust-lang.org/book/ch10-02-traits.html) — fetched_at: 2025-09-19T00:00:00Z
