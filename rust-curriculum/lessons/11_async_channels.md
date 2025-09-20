---
verified: true
---

# Lesson 11: Async Channels (mpsc)

## Overview

Use `tokio::sync::mpsc` channels for message passing between tasks, handling backpressure and graceful shutdown.

## Concepts

- Bounded vs unbounded channels; send backpressure.
- Receiver loops, `recv().await` yielding `Option<T>` on close.
- Graceful shutdown with `drop(tx)` and `select!` for cancellation.

## Hands-on Exercise

Implement `sum_worker` that receives integers over an `mpsc` channel and returns the sum when the sender closes.

### Cargo.toml

```toml
[package]
name = "lesson11_async_channels"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.40", features = ["rt", "macros", "sync"] }
```

### src/lib.rs

```rust
use tokio::sync::mpsc;

pub async fn sum_worker(mut rx: mpsc::Receiver<i32>) -> i32 {
    let mut sum = 0;
    while let Some(v) = rx.recv().await {
        sum += v;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sums_values() {
        let (tx, rx) = mpsc::channel(8);
        let handle = tokio::spawn(sum_worker(rx));
        for v in [1, 2, 3, 4] { tx.send(v).await.unwrap(); }
        drop(tx);
        let res = handle.await.unwrap();
        assert_eq!(res, 10);
    }
}
```

### Run

```bash
cargo new lesson11_async_channels --lib --vcs none
cd lesson11_async_channels
# replace files
cargo test
```

## Pitfalls

- Avoid unbounded channels in hot paths; prefer bounded to exert backpressure.
- Always close the sender (`drop(tx)`) to let receivers terminate cleanly.

## Further Reading

- Tokio: Channels — fetched_at: 2025-09-20T00:00:00Z
