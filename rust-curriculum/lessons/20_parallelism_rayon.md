verified: true
 

# Lesson 20: Data Parallelism with Rayon

## Overview

Use Rayon to parallelize independent work over collections with `par_iter` and sort large arrays with `par_sort`.

## Concepts

- Fork-join data parallelism; work-stealing scheduler.
- `rayon::prelude::*` for `ParallelIterator` traits.
- Parallel sum and parallel sort patterns.

## Hands-on Exercise

Implement:

- `parallel_sum(&[i64]) -> i64` using `par_iter().cloned().sum()`.
- `parallel_sort(Vec<i32>) -> Vec<i32>` using `par_sort()`.


### Cargo.toml

```toml
[package]
name = "lesson20_rayon"
version = "0.1.0"
edition = "2021"

[dependencies]
rayon = "1.10"
rand = "0.8"
```

### src/lib.rs

```rust
use rayon::prelude::*;

pub fn parallel_sum(v: &[i64]) -> i64 {
    v.par_iter().cloned().sum()
}

pub fn parallel_sort(mut v: Vec<i32>) -> Vec<i32> {
    v.par_sort();
    v
}
```

### Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn sums_matches_sequential() {
        let mut rng = rand::thread_rng();
        let data: Vec<i64> = (0..10_000).map(|_| rng.gen_range(-1000..=1000)).collect();
        let s1: i64 = data.iter().sum();
        let s2 = parallel_sum(&data);
        assert_eq!(s1, s2);
    }

    #[test]
    fn sort_is_ordered() {
        let mut rng = rand::thread_rng();
        let data: Vec<i32> = (0..10_000).map(|_| rng.gen_range(-1000..=1000)).collect();
        let sorted = parallel_sort(data);
        assert!(sorted.windows(2).all(|w| w[0] <= w[1]));
    }
}
```

### Run

```bash
cargo new lesson20_rayon --lib --vcs none
cd lesson20_rayon
# replace files
cargo test
```

## Pitfalls

- Beware non-determinism with side effects in parallel iterators.
- Avoid excessive task overhead for tiny workloads; batch work or stick to sequential.

## Further Reading

- Rayon Guide — fetched_at: 2025-09-20T00:00:00Z
- docs.rs: rayon — fetched_at: 2025-09-20T00:00:00Z
