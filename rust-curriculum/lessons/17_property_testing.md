---
verified: true
---


# Lesson 17: Property Testing with proptest

## Overview

Use property-based testing to validate invariants across many generated inputs. Model laws like commutativity, idempotence, and symmetry.

## Concepts

- Strategies generate random but shrinking-friendly data.
- Properties are invariant assertions over inputs.
- Shrinking finds minimal failing counterexamples to ease debugging.

## Hands-on Exercise

Add three properties:

- `add` is commutative on bounded integers.
- `sorted` returns a non-decreasing sequence and is idempotent.
- `is_palindrome` matches manual reverse on sanitized strings.


### Cargo.toml

```toml
[package]
name = "lesson17_proptest"
version = "0.1.0"
edition = "2021"

[dependencies]
proptest = "1"
rand = "0.8"
```

### src/lib.rs

```rust
pub fn is_palindrome(s: &str) -> bool {
    let cleaned: String = s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect();
    cleaned.chars().eq(cleaned.chars().rev())
}

pub fn add(a: i64, b: i64) -> i64 { a + b }

pub fn sorted(mut v: Vec<i32>) -> Vec<i32> { v.sort(); v }

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn add_commutative(a in -1_000_000i64..=1_000_000, b in -1_000_000i64..=1_000_000) {
            prop_assert_eq!(add(a,b), add(b,a));
        }

        #[test]
        fn sort_non_decreasing(mut data in proptest::collection::vec(-1000i32..=1000, 0..100)) {
            let s1 = sorted(data.clone());
            prop_assert!(s1.windows(2).all(|w| w[0] <= w[1]));
            let s2 = sorted(s1.clone());
            prop_assert_eq!(s1, s2);
            prop_assert_eq!(s1.len(), data.len());
        }

        #[test]
        fn palindrome_symmetry(s in ".{0,64}") {
            let cleaned: String = s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect();
            let is_pal = cleaned.chars().eq(cleaned.chars().rev());
            prop_assert_eq!(is_palindrome(&s), is_pal);
        }
    }
}
```

### Run

```bash
cargo new lesson17_proptest --lib --vcs none
cd lesson17_proptest
# replace files
cargo test
```

## Pitfalls

- Constrain domains to avoid overflow/UB in arithmetic.
- Keep generated sizes reasonable to keep test time down.

## Further Reading

- proptest book — fetched_at: 2025-09-20T00:00:00Z
- proptest crate docs — fetched_at: 2025-09-20T00:00:00Z
