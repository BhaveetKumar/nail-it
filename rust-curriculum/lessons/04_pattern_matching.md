---
verified: true
---

# Lesson 4: Pattern Matching Deep Dive

## Overview

Master `match`, if-let, while-let, destructuring, guards, and exhaustive handling.

## Concepts

- match exhaustiveness and ordering
- Destructuring structs, enums, tuples
- Guards (`if` in patterns), bindings (`@`), `_` wildcard

## Idiomatic Examples

```rust
#[derive(Debug)]
enum Token { Int(i64), Ident(String), Plus, EOF }

fn classify(t: Token) -> &'static str {
    match t {
        Token::Int(n) if n < 0 => "neg-int",
        Token::Int(_) => "int",
        Token::Ident(ref s) if s == "let" => "keyword",
        Token::Ident(_) => "ident",
        Token::Plus => "+",
        Token::EOF => "eof",
    }
}
```

## Hands-on Exercise

Implement a simple calculator parsing `a + b` returning i64.

### Cargo.toml

```toml
[package]
name = "lesson04_patterns"
version = "0.1.0"
edition = "2021"
```

### src/lib.rs

```rust
pub fn eval(expr: &str) -> Option<i64> {
    let parts: Vec<_> = expr.split('+').map(|s| s.trim()).collect();
    match parts.as_slice() {
        [a,b] => {
            let (Ok(a), Ok(b)) = (a.parse::<i64>(), b.parse::<i64>()) else { return None };
            Some(a + b)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ok() { assert_eq!(eval("2 + 3"), Some(5)); }
    #[test]
    fn bad() { assert_eq!(eval("x + 3"), None); }
}
```

### Run

```bash
cargo new lesson04_patterns --lib --vcs none
cd lesson04_patterns
# replace files
cargo test
```

## Common Mistakes

- Non-exhaustive matches; prefer `_` catch-all only when justified.

## Further Reading

- [TRPL: Control Flow](https://doc.rust-lang.org/book/ch06-00-enums.html) — fetched_at: 2025-09-19T00:00:00Z
- [Rust By Example: Pattern Matching](https://doc.rust-lang.org/rust-by-example/flow_control/match.html) — fetched_at: 2025-09-19T00:00:00Z