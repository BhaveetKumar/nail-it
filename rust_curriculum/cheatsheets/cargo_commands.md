# Cargo Commands Cheat Sheet

> **Quick Reference for Rust Package Manager**  
> **Rust Version**: 1.75.0  
> **Last Updated**: 2024-12-19T00:00:00Z

---

## 🚀 **Basic Commands**

### **Project Management**
```bash
# Create new project
cargo new my_project
cargo new --lib my_library
cargo new --bin my_binary

# Initialize project in existing directory
cargo init
cargo init --lib

# Build project
cargo build
cargo build --release

# Run project
cargo run
cargo run --release
cargo run --bin my_binary

# Check code without building
cargo check
cargo check --release
```

### **Testing**
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Run integration tests
cargo test --test integration_test

# Run benchmarks
cargo bench
```

---

## 🔧 **Development Commands**

### **Code Quality**
```bash
# Format code
cargo fmt
cargo fmt -- --check

# Run linter
cargo clippy
cargo clippy -- -D warnings

# Check for security vulnerabilities
cargo audit

# Check for outdated dependencies
cargo outdated
```

### **Documentation**
```bash
# Build documentation
cargo doc
cargo doc --open
cargo doc --no-deps

# Build documentation for all dependencies
cargo doc --all

# Serve documentation locally
cargo doc --open --no-deps
```

---

## 📦 **Dependency Management**

### **Adding Dependencies**
```bash
# Add dependency
cargo add serde
cargo add serde --features derive
cargo add tokio --features full

# Add development dependency
cargo add --dev tokio-test

# Add build dependency
cargo add --build cc

# Add specific version
cargo add serde@1.0.195
cargo add serde@^1.0
cargo add serde@~1.0.0
```

### **Managing Dependencies**
```bash
# Update dependencies
cargo update
cargo update package_name

# Remove dependency
cargo remove package_name

# Show dependency tree
cargo tree
cargo tree --duplicates
cargo tree --format "{p} {f}"
```

---

## 🏗️ **Build Configuration**

### **Build Profiles**
```bash
# Debug build (default)
cargo build

# Release build
cargo build --release

# Custom profile
cargo build --profile custom

# Build specific target
cargo build --target x86_64-unknown-linux-gnu
cargo build --target wasm32-unknown-unknown
```

### **Feature Flags**
```bash
# Build with specific features
cargo build --features feature1,feature2
cargo build --no-default-features

# Run with features
cargo run --features async
cargo test --features integration
```

---

## 🔍 **Analysis and Debugging**

### **Code Analysis**
```bash
# Show unused dependencies
cargo machete

# Analyze dependency graph
cargo tree --duplicates
cargo tree --invert package_name

# Check for unused code
cargo check --all-targets
```

### **Debugging**
```bash
# Build with debug info
cargo build --profile dev

# Run with debug output
RUST_LOG=debug cargo run
RUST_BACKTRACE=1 cargo run

# Run with specific toolchain
cargo +nightly run
cargo +stable run
```

---

## 🌐 **Cross-Platform Development**

### **Target Management**
```bash
# List installed targets
rustup target list --installed

# Add target
rustup target add x86_64-unknown-linux-gnu
rustup target add wasm32-unknown-unknown

# Build for target
cargo build --target x86_64-unknown-linux-gnu
cargo build --target wasm32-unknown-unknown
```

### **Cross-Compilation**
```bash
# Install cross-compilation tools
rustup target add x86_64-pc-windows-gnu
rustup target add aarch64-unknown-linux-gnu

# Build for Windows from Linux
cargo build --target x86_64-pc-windows-gnu

# Build for ARM from x86_64
cargo build --target aarch64-unknown-linux-gnu
```

---

## 📊 **Performance and Profiling**

### **Benchmarking**
```bash
# Run benchmarks
cargo bench

# Run specific benchmark
cargo bench benchmark_name

# Generate flamegraph
cargo install flamegraph
cargo flamegraph --bin my_binary

# Profile with perf
cargo build --release
perf record ./target/release/my_binary
perf report
```

### **Optimization**
```bash
# Build with optimizations
cargo build --release

# Build with size optimizations
cargo build --release --profile size-optimized

# Strip debug symbols
strip target/release/my_binary
```

---

## 🧪 **Testing and Quality**

### **Test Configuration**
```bash
# Run tests in parallel
cargo test --jobs 4

# Run tests sequentially
cargo test --jobs 1

# Run tests with specific pattern
cargo test -- test_pattern

# Run tests in specific module
cargo test module_name::
```

### **Code Coverage**
```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html

# Generate coverage with tests
cargo tarpaulin --tests --out Html
```

---

## 🔒 **Security and Maintenance**

### **Security Auditing**
```bash
# Check for security vulnerabilities
cargo audit

# Install cargo-audit if not available
cargo install cargo-audit

# Check for outdated dependencies
cargo install cargo-outdated
cargo outdated
```

### **Dependency Management**
```bash
# Update all dependencies
cargo update

# Update specific dependency
cargo update package_name

# Check for unused dependencies
cargo install cargo-machete
cargo machete
```

---

## 🚀 **Publishing and Distribution**

### **Publishing to crates.io**
```bash
# Login to crates.io
cargo login

# Check package before publishing
cargo package
cargo publish --dry-run

# Publish package
cargo publish

# Publish with specific version
cargo publish --allow-dirty
```

### **Creating Releases**
```bash
# Build for multiple targets
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target x86_64-pc-windows-gnu
cargo build --release --target x86_64-apple-darwin

# Create archive
tar -czf my_project-linux.tar.gz target/x86_64-unknown-linux-gnu/release/my_project
```

---

## 🛠️ **Advanced Commands**

### **Workspace Management**
```bash
# Build entire workspace
cargo build --workspace

# Test entire workspace
cargo test --workspace

# Run specific package
cargo run -p package_name
cargo test -p package_name
```

### **Custom Commands**
```bash
# Install custom cargo commands
cargo install cargo-expand
cargo install cargo-watch
cargo install cargo-udeps

# Use custom commands
cargo expand
cargo watch -x run
cargo udeps
```

---

## 📋 **Common Workflows**

### **Daily Development**
```bash
# Check code quality
cargo clippy -- -D warnings
cargo fmt

# Run tests
cargo test

# Build and run
cargo run
```

### **CI/CD Pipeline**
```bash
# Install dependencies
cargo fetch

# Check code
cargo check --all-targets
cargo clippy -- -D warnings
cargo fmt -- --check

# Run tests
cargo test --all

# Build release
cargo build --release
```

### **Release Process**
```bash
# Update version
cargo set-version 1.2.3

# Check package
cargo package

# Publish
cargo publish

# Create git tag
git tag v1.2.3
git push origin v1.2.3
```

---

## 🎯 **Best Practices**

### **Project Structure**
- Use `cargo new` for new projects
- Organize code in `src/` directory
- Use `tests/` for integration tests
- Use `examples/` for example code

### **Dependency Management**
- Pin major versions with `^`
- Use `cargo update` regularly
- Remove unused dependencies
- Use `cargo audit` for security

### **Build Optimization**
- Use `--release` for production builds
- Enable LTO for final binaries
- Strip debug symbols for smaller binaries
- Use appropriate target for deployment

---

## 📚 **Further Reading**

### **Official Documentation**
- [Cargo Book](https://doc.rust-lang.org/cargo/) - Fetched: 2024-12-19T00:00:00Z
- [Cargo Reference](https://doc.rust-lang.org/cargo/reference/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Cargo Commands](https://doc.rust-lang.org/cargo/commands/) - Fetched: 2024-12-19T00:00:00Z
- [Cargo Workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html) - Fetched: 2024-12-19T00:00:00Z

---

**Cheat Sheet Version**: 1.0  
**Rust Version**: 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z
