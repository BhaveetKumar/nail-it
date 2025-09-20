---
verified: true
---

# Lesson 18: Unsafe Basics

## Overview

Explore core unsafe building blocks: raw pointers, manual slice splitting, and memory copy. Understand when and how to encapsulate unsafe safely.

## Concepts

- The five unsafe superpowers: deref raw pointers, call unsafe fns, mutate statics, implement unsafe traits, access union fields.
- `std::ptr::copy_nonoverlapping` and aliasing rules.
- Creating slices from raw parts: `from_raw_parts_mut`.

## Hands-on Exercise

Implement two functions and test them:

- `copy_bytes(src: *const u8, dst: *mut u8, len: usize)` using `copy_nonoverlapping`.
- `split_at_mut_manual(&mut [u8], mid) -> (&mut [u8], &mut [u8])` using raw parts.


### Cargo.toml

```toml
[package]
name = "lesson18_unsafe"
version = "0.1.0"
edition = "2021"
```

### src/lib.rs

```rust
pub unsafe fn copy_bytes(src: *const u8, dst: *mut u8, len: usize) {
    std::ptr::copy_nonoverlapping(src, dst, len);
}

pub unsafe fn split_at_mut_manual(slice: &mut [u8], mid: usize) -> (&mut [u8], &mut [u8]) {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    assert!(mid <= len);
    (
        std::slice::from_raw_parts_mut(ptr, mid),
        std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
    )
}
```

### Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_bytes() {
        let src = b"hello";
        let mut dst = [0u8; 5];
        unsafe { copy_bytes(src.as_ptr(), dst.as_mut_ptr(), src.len()) };
        assert_eq!(&dst, src);
    }

    #[test]
    fn splits_mut_slice() {
        let mut buf = *b"abcdef";
        let (a,b) = unsafe { split_at_mut_manual(&mut buf, 2) };
        assert_eq!(a, b"ab");
        assert_eq!(b, b"cdef");
        b[0] = b'X';
        assert_eq!(&buf, b"abXdef");
    }
}
```

### Run

```bash
cargo new lesson18_unsafe --lib --vcs none
cd lesson18_unsafe
# replace files
cargo test
```

## Pitfalls

- Keep unsafe blocks tiny and well-documented; uphold invariants at API boundaries.
- Use `copy` (overlapping) vs `copy_nonoverlapping` appropriately.

## Further Reading

- TRPL: Unsafe Rust — fetched_at: 2025-09-20T00:00:00Z
- std::ptr — fetched_at: 2025-09-20T00:00:00Z
