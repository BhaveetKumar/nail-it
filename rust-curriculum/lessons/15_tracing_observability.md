---
verified: true
---

# Lesson 15: Tracing & Observability

## Overview

Instrument apps with `tracing`, use spans and fields, and initialize a subscriber. Capture logs in tests to assert behavior.

## Concepts

- Spans vs events; structured fields.
- Subscribers: `tracing_subscriber::fmt` for human logs; filtering with `RUST_LOG` or code.
- In tests, use `tracing_test` or a test-specific subscriber to capture output.

## Hands-on Exercise

Implement `compute` instrumented with a span and an event, and verify emitted output in a test.

### Cargo.toml

```toml
[package]
name = "lesson15_tracing"
version = "0.1.0"
edition = "2021"

[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter"] }
```

### src/lib.rs

```rust
use tracing::{info, span, Level};

pub fn init_for_tests() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();
}

pub fn compute(x: i32) -> i32 {
    let s = span!(Level::INFO, "compute", input = x);
    let _e = s.enter();
    let y = x + 1;
    info!(result = y, "computed");
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_logs() {
        init_for_tests();
        let r = compute(41);
        assert_eq!(r, 42);
    }
}
```

### Run

```bash
cargo new lesson15_tracing --lib --vcs none
cd lesson15_tracing
# replace files
cargo test
```

## Pitfalls

- Initialize the subscriber only once per process; use `try_init` and ignore the error.
- Prefer fields over string logs for machine parsing.

## Further Reading

- tracing: Overview — fetched_at: 2025-09-20T00:00:00Z
- tracing-subscriber: fmt, EnvFilter — fetched_at: 2025-09-20T00:00:00Z
