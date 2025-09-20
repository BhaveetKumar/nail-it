---
verified: false
---

# Lesson 13: Async I/O (tokio::io)

## Overview

Learn `tokio::io` primitives (`AsyncRead`, `AsyncWrite`), helpers like `copy`, and in-memory testing with `duplex`.

## Concepts

- Traits: `AsyncRead`, `AsyncWrite`, `AsyncBufRead` and extension methods.
- Utilities: `copy`, `split`, `BufReader`/`BufWriter`.
- In-memory testing: `tokio::io::duplex` for deterministic tests.

## Hands-on Exercise

Use `duplex` to exchange bytes between two endpoints and assert payload integrity.

### Cargo.toml

```toml
[package]
name = "lesson13_async_io"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.40", features = ["rt", "macros", "io-util", "time"] }
```

### src/lib.rs

```rust
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub async fn roundtrip(msg: &[u8]) -> Vec<u8> {
    let (mut a, mut b) = tokio::io::duplex(64);
    // Writer task
    let m = msg.to_vec();
    let w = tokio::spawn(async move {
        a.write_all(&m).await.unwrap();
        // Close write half to signal EOF
        a.shutdown().await.unwrap();
    });
    // Reader path
    let mut buf = vec![0u8; msg.len()];
    b.read_exact(&mut buf).await.unwrap();
    w.await.unwrap();
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn duplex_roundtrip() {
        let got = roundtrip(b"hello").await;
        assert_eq!(got, b"hello");
    }
}
```

### Run

```bash
cargo new lesson13_async_io --lib --vcs none
cd lesson13_async_io
# replace files
cargo test
```

## Pitfalls

- Always close writers (`shutdown`) to let readers finish.
- Beware of deadlocks when both sides wait to read.

## Further Reading

- Tokio: I/O — fetched_at: 2025-09-20T00:00:00Z
