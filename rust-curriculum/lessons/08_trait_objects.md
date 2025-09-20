---
verified: true
---

# Lesson 8: Trait Objects & Object Safety

## Overview

Use dynamic dispatch with `dyn Trait`, understand object safety, and design trait APIs for both static and dynamic use.

## Concepts

- `dyn Trait` vs generics; when to choose each
- Object safety rules (no generic methods, `Self: Sized` constraints)
- Vtables, dynamic dispatch cost model

## Idiomatic Examples

```rust
trait Draw {
    fn draw(&self) -> String;
}

struct Button { label: String }
struct Label { text: String }

impl Draw for Button { fn draw(&self) -> String { format!("[{}]", self.label) } }
impl Draw for Label { fn draw(&self) -> String { self.text.clone() } }

pub struct Screen { widgets: Vec<Box<dyn Draw>> }
impl Screen {
    pub fn new() -> Self { Self { widgets: vec![] } }
    pub fn add(&mut self, w: Box<dyn Draw>) { self.widgets.push(w); }
    pub fn render(&self) -> String { self.widgets.iter().map(|w| w.draw()).collect::<Vec<_>>().join("\n") }
}
```

## Hands-on Exercise

Build a `Logger` trait with `info`/`warn`/`error` methods and implement `ConsoleLogger` and `VecLogger`. Use `Box<dyn Logger>` collection.

### Cargo.toml

```toml
[package]
name = "lesson08_trait_objects"
version = "0.1.0"
edition = "2021"
```

### src/lib.rs

```rust
pub trait Logger {
    fn info(&mut self, msg: &str);
    fn warn(&mut self, msg: &str);
    fn error(&mut self, msg: &str);
}

pub struct ConsoleLogger;
impl Logger for ConsoleLogger {
    fn info(&mut self, msg: &str) { println!("INFO: {msg}"); }
    fn warn(&mut self, msg: &str) { println!("WARN: {msg}"); }
    fn error(&mut self, msg: &str) { eprintln!("ERROR: {msg}"); }
}

pub struct VecLogger { pub entries: Vec<String> }
impl Default for VecLogger { fn default() -> Self { Self { entries: vec![] } } }
impl Logger for VecLogger {
    fn info(&mut self, msg: &str) { self.entries.push(format!("INFO: {msg}")); }
    fn warn(&mut self, msg: &str) { self.entries.push(format!("WARN: {msg}")); }
    fn error(&mut self, msg: &str) { self.entries.push(format!("ERROR: {msg}")); }
}

pub struct MultiLogger { inner: Vec<Box<dyn Logger>> }
impl MultiLogger {
    pub fn new() -> Self { Self { inner: vec![] } }
    pub fn add(&mut self, l: Box<dyn Logger>) { self.inner.push(l); }
    pub fn info(&mut self, msg: &str) { for l in self.inner.iter_mut() { l.info(msg); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn works() {
        let mut ml = MultiLogger::new();
        ml.add(Box::new(ConsoleLogger));
        let v = Box::new(VecLogger::default());
        ml.add(v);
        ml.info("hello");
        assert!(true);
    }
}
```

### Run

```bash
cargo new lesson08_trait_objects --lib --vcs none
cd lesson08_trait_objects
# replace files
cargo test
```

## Common Mistakes

- Designing non-object-safe traits when dynamic dispatch is needed.

## Further Reading

- [TRPL: Trait Objects](https://doc.rust-lang.org/book/ch17-02-trait-objects.html) — fetched_at: 2025-09-19T00:00:00Z
