# Rust Ecosystem Quick Reference Guide

> **Level**: Expert  
> **Last Updated**: 2024-12-19T00:00:00Z  
> **Rust Version**: 1.75.0

---

## 🚀 **Quick Start Commands**

### **Crate Management**
```bash
# Search for crates
cargo search <crate-name>

# Add dependency
cargo add <crate-name>

# Add dependency with features
cargo add <crate-name> --features <feature1>,<feature2>

# Update dependencies
cargo update

# Check outdated dependencies
cargo outdated

# Audit security vulnerabilities
cargo audit
```

### **Development Workflow**
```bash
# Check code quality
cargo clippy -- -D warnings

# Format code
cargo fmt

# Run tests
cargo test

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open

# Check for unused dependencies
cargo machete
```

---

## 📦 **Essential Crate Categories**

### **Core Utilities**
| Crate | Purpose | Version | Notes |
|-------|---------|---------|-------|
| `anyhow` | Error handling | 1.0 | `Result<T, anyhow::Error>` |
| `thiserror` | Custom error types | 1.0 | Derive macros for errors |
| `serde` | Serialization | 1.0 | JSON, YAML, TOML support |
| `tokio` | Async runtime | 1.35 | Most popular async runtime |
| `futures` | Async utilities | 0.3 | Streams, combinators |

### **Web Development**
| Crate | Purpose | Version | Notes |
|-------|---------|---------|-------|
| `axum` | Web framework | 0.7 | Built on tower |
| `warp` | Web framework | 0.3 | Functional style |
| `actix-web` | Web framework | 4.4 | Actor-based |
| `tower` | Middleware | 0.4 | Service abstraction |
| `reqwest` | HTTP client | 0.11 | Async HTTP client |

### **Database**
| Crate | Purpose | Version | Notes |
|-------|---------|---------|-------|
| `sqlx` | SQL toolkit | 0.7 | Async, compile-time checked |
| `diesel` | ORM | 2.1 | Type-safe ORM |
| `redis` | Redis client | 0.24 | Async Redis client |
| `mongodb` | MongoDB driver | 2.6 | Official MongoDB driver |

### **Serialization**
| Crate | Purpose | Version | Notes |
|-------|---------|---------|-------|
| `serde_json` | JSON | 1.0 | JSON serialization |
| `serde_yaml` | YAML | 0.9 | YAML serialization |
| `toml` | TOML | 0.8 | TOML parsing |
| `bincode` | Binary | 1.3 | Compact binary format |
| `ron` | RON | 0.8 | Rusty Object Notation |

---

## 🔧 **Development Tools**

### **Code Quality**
```bash
# Install tools
cargo install cargo-audit
cargo install cargo-outdated
cargo install cargo-machete
cargo install cargo-expand
cargo install cargo-tree

# Run all quality checks
cargo clippy -- -D warnings
cargo fmt --check
cargo test
cargo audit
cargo outdated
```

### **Documentation**
```bash
# Generate docs
cargo doc --open

# Generate docs for dependencies
cargo doc --open --no-deps

# Check doc links
cargo doc --document-private-items

# Generate book
mdbook build
```

### **Testing**
```bash
# Run tests
cargo test

# Run specific test
cargo test test_name

# Run integration tests
cargo test --test integration

# Run with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
```

---

## 📊 **Crate Quality Assessment**

### **Quick Quality Check**
```rust
// Check these factors when evaluating crates:
pub struct CrateQuality {
    pub downloads: u64,           // > 1M = excellent, > 100K = good
    pub recent_activity: bool,    // Updated in last 6 months
    pub documentation: bool,      // Has docs.rs page
    pub examples: bool,           // Has usage examples
    pub tests: bool,              // Has test suite
    pub license: String,          // Compatible license
    pub maintenance: bool,        // Actively maintained
}
```

### **Dependency Analysis**
```bash
# Analyze dependency tree
cargo tree

# Check for duplicate dependencies
cargo tree --duplicates

# Check for unused dependencies
cargo machete

# Check for security vulnerabilities
cargo audit
```

---

## 🎯 **Best Practices**

### **Cargo.toml Structure**
```toml
[package]
name = "my-crate"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <email@example.com>"]
description = "Brief description"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/repo"
homepage = "https://my-crate.com"
documentation = "https://docs.rs/my-crate"
keywords = ["keyword1", "keyword2"]
categories = ["development-tools"]

[dependencies]
# Core dependencies
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }

[dev-dependencies]
# Development dependencies
criterion = "0.5"
proptest = "1.0"

[features]
default = ["std"]
std = []
no_std = []

[profile.dev]
opt-level = 1
debug = true

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

### **Version Management**
```toml
# Use semantic versioning
version = "1.0.0"  # Major.Minor.Patch

# Version constraints
serde = "1.0"           # ^1.0.0 (compatible)
serde = "~1.0.0"        # ~1.0.0 (patch only)
serde = "=1.0.0"        # =1.0.0 (exact)
serde = ">=1.0.0, <2.0" # Range
```

---

## 🚀 **Publishing Workflow**

### **Pre-Publish Checklist**
```bash
# 1. Update version
cargo version patch  # or minor, major

# 2. Update CHANGELOG.md
# 3. Run quality checks
cargo clippy -- -D warnings
cargo fmt
cargo test
cargo audit

# 4. Generate docs
cargo doc --no-deps

# 5. Check package
cargo package --dry-run

# 6. Publish
cargo publish
```

### **Publishing Commands**
```bash
# Dry run
cargo package --dry-run

# Publish to crates.io
cargo publish

# Publish to alternative registry
cargo publish --registry <registry-name>

# Yank a version
cargo yank --vers 1.0.0
```

---

## 🔍 **Troubleshooting**

### **Common Issues**
```bash
# Dependency resolution issues
cargo update
cargo clean
cargo build

# Feature flag conflicts
cargo tree --features <feature>

# Compilation errors
cargo check
cargo clippy
cargo expand  # Show macro expansions
```

### **Debug Commands**
```bash
# Verbose output
cargo build --verbose

# Show dependency resolution
cargo tree --duplicates

# Check feature flags
cargo check --features <feature>

# Show compilation units
cargo build --verbose 2>&1 | grep "Compiling"
```

---

## 📚 **Useful Resources**

### **Documentation**
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Rust Cookbook](https://rust-lang-nursery.github.io/rust-cookbook/)

### **Community**
- [crates.io](https://crates.io/) - Package registry
- [docs.rs](https://docs.rs/) - Documentation hosting
- [Rust Users Forum](https://users.rust-lang.org/)
- [Rust Internals Forum](https://internals.rust-lang.org/)

### **Tools**
- [cargo-audit](https://github.com/RustSec/cargo-audit) - Security auditing
- [cargo-outdated](https://github.com/kbknapp/cargo-outdated) - Check outdated deps
- [cargo-machete](https://github.com/bnjbvr/cargo-machete) - Find unused deps
- [cargo-expand](https://github.com/dtolnay/cargo-expand) - Expand macros

---

## 🎯 **Quick Reference**

### **Essential Commands**
```bash
# Project setup
cargo new <project-name>
cargo init

# Development
cargo build
cargo run
cargo test
cargo check

# Code quality
cargo clippy
cargo fmt
cargo audit

# Documentation
cargo doc
cargo doc --open

# Dependencies
cargo add <crate>
cargo update
cargo tree
cargo outdated

# Publishing
cargo package
cargo publish
```

### **Common Patterns**
```rust
// Error handling
use anyhow::Result;

fn main() -> Result<()> {
    // Your code here
    Ok(())
}

// Async main
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Your async code here
    Ok(())
}

// Feature flags
#[cfg(feature = "std")]
fn std_function() {}

#[cfg(not(feature = "std"))]
fn no_std_function() {}
```

---

**Cheat Sheet Version**: 1.0  
**Rust Version**: 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z
