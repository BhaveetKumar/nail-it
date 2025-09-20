---
verified: true
---

# Lesson 6: Error Handling with Result, anyhow, thiserror

## Overview

Use `Result` and `Option` effectively; design error types with `thiserror`; use `anyhow` for app-level ergonomics.

## Concepts

- Result and the `?` operator
- Custom error enums with `thiserror::Error`
- Converting errors with `From` and `#[from]`
- When to use `anyhow::Result` vs typed errors

## Idiomatic Examples

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MyError {
    #[error("parse int: {0}")]
    ParseInt(#[from] std::num::ParseIntError),
}

pub fn parse_sum(a: &str, b: &str) -> Result<i64, MyError> {
    let a: i64 = a.parse()?;
    let b: i64 = b.parse()?;
    Ok(a + b)
}
```

## Hands-on Exercise

Build a CSV sum utility with robust errors.

### Cargo.toml

```toml
[package]
name = "lesson06_errors"
version = "0.1.0"
edition = "2021"

[dependencies]
thiserror = "1"
anyhow = "1"
```

### src/lib.rs

```rust
use anyhow::{Context, Result};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CsvSumError {
    #[error("empty input")]
    Empty,
}

pub fn csv_sum(line: &str) -> Result<i64> {
    if line.trim().is_empty() { return Err(CsvSumError::Empty.into()); }
    line.split(',')
        .map(|s| s.trim().parse::<i64>().context("parse int"))
        .try_fold(0i64, |acc, r| r.map(|v| acc + v))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ok() { assert_eq!(csv_sum("1, 2, 3").unwrap(), 6); }
    #[test]
    fn empty() { assert!(csv_sum("").is_err()); }
    #[test]
    fn bad() { assert!(csv_sum("a,2").is_err()); }
}
```

### Run

```bash
cargo new lesson06_errors --lib --vcs none
cd lesson06_errors
# replace files
cargo test
```

## Common Mistakes

- Overusing `anyhow` in libraries; prefer typed errors for public APIs.

## Further Reading

- [TRPL: Error Handling](https://doc.rust-lang.org/book/ch09-00-error-handling.html) — fetched_at: 2025-09-19T00:00:00Z
- [thiserror](https://crates.io/crates/thiserror) — fetched_at: 2025-09-19T00:00:00Z
- [anyhow](https://crates.io/crates/anyhow) — fetched_at: 2025-09-19T00:00:00Z
