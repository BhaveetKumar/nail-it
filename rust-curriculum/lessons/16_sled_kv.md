---
verified: true
---

# Lesson 16: Embedded KV with sled

## Overview

Use `sled`, a fast embedded KV store, for simple persistence with atomic ops and iterators.

## Concepts

- `Db` open, `Tree`, `insert`, `get`, `remove`, `flush`.
- Atomic compare-and-swap (CAS) with `compare_and_swap`.
- Iteration: range scans.

## Hands-on Exercise

Implement a tiny KV wrapper with `put/get/del` and test roundtrip using a temporary directory.

### Cargo.toml

```toml
[package]
name = "lesson16_sled_kv"
version = "0.1.0"
edition = "2021"

[dependencies]
sled = "0.34"
tempfile = "3"
```

### src/lib.rs

```rust
use sled::IVec;

pub struct Kv {
    db: sled::Db,
}

impl Kv {
    pub fn open(path: &std::path::Path) -> sled::Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }
    pub fn put(&self, k: &str, v: &[u8]) -> sled::Result<Option<IVec>> {
        self.db.insert(k, v)
    }
    pub fn get(&self, k: &str) -> sled::Result<Option<IVec>> {
        self.db.get(k)
    }
    pub fn del(&self, k: &str) -> sled::Result<Option<IVec>> {
        self.db.remove(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn roundtrip() {
        let dir = TempDir::new().unwrap();
        let kv = Kv::open(dir.path()).unwrap();
        kv.put("a", b"1").unwrap();
        assert_eq!(kv.get("a").unwrap().unwrap().as_ref(), b"1");
        assert_eq!(kv.del("a").unwrap().unwrap().as_ref(), b"1");
        assert!(kv.get("a").unwrap().is_none());
    }
}
```

### Run

```bash
cargo new lesson16_sled_kv --lib --vcs none
cd lesson16_sled_kv
# replace files
cargo test
```

## Pitfalls

- Always close/flush on shutdown to ensure durability guarantees.
- Keys and values are bytes; handle encoding/decoding.

## Further Reading

- sled: docs & guide — fetched_at: 2025-09-20T00:00:00Z
