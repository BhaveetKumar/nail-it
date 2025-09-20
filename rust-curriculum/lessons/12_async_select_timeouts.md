---
verified: true
---

# Lesson 12: Async select! and Timeouts

## Overview

Combine multiple async operations with `tokio::select!`, handle cancellations, and add timeouts safely.

## Concepts

- `tokio::select!` races branches; first ready wins; others are cancelled.
- Cancellation safety: Drop guards, be careful with partially-consumed futures.
- Timeouts: `tokio::time::timeout` returns `Result` with `Elapsed` error.

## Hands-on Exercise

Implement `first_ok` that races two fallible futures and returns the first `Ok(T)`, otherwise the last `Err` if both fail. Add a `with_timeout` wrapper to bound latency.

### Cargo.toml

```toml
[package]
name = "lesson12_async_select"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.40", features = ["rt", "macros", "time"] }
anyhow = "1.0.100"
```

### src/lib.rs

```rust
use anyhow::{anyhow, Result};
use tokio::time::{sleep, timeout, Duration};

pub async fn maybe_after(ms: u64, ok: bool) -> Result<&'static str> {
    sleep(Duration::from_millis(ms)).await;
    if ok { Ok("ok") } else { Err(anyhow!("bad")) }
}

pub async fn first_ok<A, B>(a: A, b: B) -> Result<&'static str>
where
    A: std::future::Future<Output = Result<&'static str>>,
    B: std::future::Future<Output = Result<&'static str>>,
{
    tokio::select! {
        ra = a => ra,
        rb = b => rb,
    }
}

pub async fn with_timeout<F, T>(dur: Duration, fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    Ok(timeout(dur, fut).await??)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn picks_first_ok() {
        let r = first_ok(maybe_after(5, true), maybe_after(1, false)).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn times_out() {
        let r = with_timeout(Duration::from_millis(1), maybe_after(50, true)).await;
        assert!(r.is_err());
    }
}
```

### Run

```bash
cargo new lesson12_async_select --lib --vcs none
cd lesson12_async_select
# replace files
cargo test
```

## Pitfalls

- If you need both results, use `join!`; `select!` cancels other branches.
- Avoid borrowing across `.await` inside `select!` branches.

## Further Reading

- Tokio: select! — fetched_at: 2025-09-20T00:00:00Z
- Tokio: timeouts — fetched_at: 2025-09-20T00:00:00Z
