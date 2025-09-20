---
verified: true
---

# Lesson 19: FFI Basics (C interop)

## Overview

Call C from Rust and export Rust functions to C. Use `cc` to compile C sources and ABI-correct types for signatures.

## Concepts

- `extern "C"` blocks for imports and `#[no_mangle] extern "C"` for exports.
- Build C code with a `build.rs` using the `cc` crate.
- Use `libc` types where needed (e.g., `c_int`) and native Rust types (`i64`) when they match the C ABI on your target.

## Hands-on Exercise

- Compile `native.c` providing `add_i64` and `mul_i32`.
- Call them from Rust, and export `rust_square` callable from C.

### Cargo.toml

```toml
[package]
name = "lesson19_ffi"
version = "0.1.0"
edition = "2021"
build = "build.rs"

[dependencies]
libc = "0.2"

[build-dependencies]
cc = "1"
```

### build.rs

```rust
fn main() {
    cc::Build::new()
        .file("src/native.c")
        .compile("nativeffi");
    println!("cargo:rerun-if-changed=src/native.c");
}
```

### src/native.c

```c
#include <stdint.h>

int64_t add_i64(int64_t a, int64_t b) {
    return a + b;
}

int mul_i32(int a, int b) {
    return a * b;
}
```

### src/lib.rs

```rust
use libc::c_int;

extern "C" {
    fn add_i64(a: i64, b: i64) -> i64;
    fn mul_i32(a: c_int, b: c_int) -> c_int;
}

#[no_mangle]
pub extern "C" fn rust_square(x: c_int) -> c_int {
    x * x
}

pub fn add_via_c(a: i64, b: i64) -> i64 {
    unsafe { add_i64(a as i64, b as i64) as i64 }
}

pub fn mul_via_c(a: i32, b: i32) -> i32 {
    unsafe { mul_i32(a as c_int, b as c_int) as i32 }
}
```

### Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_c_and_rust_export() {
        assert_eq!(add_via_c(2, 40), 42);
        assert_eq!(mul_via_c(6, 7), 42);
        extern "C" { fn rust_square(x: c_int) -> c_int; }
        let r = unsafe { rust_square(7) };
        assert_eq!(r, 49);
    }
}
```

### Run

```bash
cargo new lesson19_ffi --lib --vcs none
cd lesson19_ffi
# add build.rs and src/native.c, replace src/lib.rs
cargo test
```

## Pitfalls

- Ensure C types match Rust `libc` types to avoid UB.
- Beware name mangling; use `#[no_mangle]` on exports.

## Further Reading

- Rustonomicon: FFI — fetched_at: 2025-09-20T00:00:00Z
- libc crate docs — fetched_at: 2025-09-20T00:00:00Z
- cc crate docs — fetched_at: 2025-09-20T00:00:00Z
