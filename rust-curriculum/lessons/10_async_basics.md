---
verified: true
---

# Lesson 10: Async Basics (Tokio)

## Overview

Understand Rust async fundamentals: futures, executors, `.await`, tasks, and sleeping.

## Concepts

- Futures: lazy computations; poll, wake, and executors.
- Tokio runtime: multi-threaded vs current-thread; spawning tasks.
- `.await` suspension; not holding locks/borrows across `.await`.
- Timers: `tokio::time::sleep`, `timeout`.

## Hands-on Exercise

Implement `fetch_both` that concurrently runs two async operations and returns both results after the longer completes using `join!`.

### Cargo.toml

```toml
[package]
name = "lesson10_async_basics"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.40", features = ["rt", "macros", "time"] }
```

### src/lib.rs

```rust
use tokio::time::{sleep, Duration};

pub async fn slow_add(a: i32, b: i32, ms: u64) -> i32 {
    sleep(Duration::from_millis(ms)).await;
    a + b
}

pub async fn fetch_both() -> (i32, i32) {
    let f1 = slow_add(1, 2, 50);
    let f2 = slow_add(3, 4, 10);
    let (a, b) = tokio::join!(f1, f2);
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn join_works() {
        let (a, b) = fetch_both().await;
        assert_eq!((a, b), (3, 7));
    }
}
```

### Run

```bash
cargo new lesson10_async_basics --lib --vcs none
cd lesson10_async_basics
# replace files
cargo test
```

## Pitfalls

- Do not block: avoid `std::thread::sleep`; prefer `tokio::time::sleep`.
- Use `join!` or `try_join!` for concurrency; avoid sequential awaits when independent.

## Further Reading

- Tokio: Getting Started — fetched_at: 2025-09-20T00:00:00Z
- Async Book: Execution — fetched_at: 2025-09-20T00:00:00Z
