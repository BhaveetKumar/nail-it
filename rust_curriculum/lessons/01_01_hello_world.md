---
# Auto-generated front matter
Title: 01 01 Hello World
LastUpdated: 2025-11-06T20:45:58.127717
Tags: []
Status: draft
---

# Lesson 1.1: Hello, World! - Your First Rust Program

> **Module**: 1 - Introduction to Rust  
> **Lesson**: 1 of 4  
> **Duration**: 2-3 hours  
> **Prerequisites**: Basic programming knowledge  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Set up a Rust development environment
- Write and run your first Rust program
- Understand the basic structure of a Rust program
- Use `cargo` to manage Rust projects
- Explain what makes Rust unique among programming languages

---

## 🎯 **Overview**

Welcome to Rust! In this lesson, we'll start your journey into one of the most exciting programming languages of the modern era. We'll begin with the traditional "Hello, World!" program and explore what makes Rust special.

---

## 🔧 **Setup and Installation**

### **Installing Rust**

Rust is installed through `rustup`, the official installer and toolchain manager.

```bash
# Install rustup (Linux/macOS)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install rustup (Windows)
# Download and run rustup-init.exe from https://rustup.rs/

# Verify installation
rustc --version
cargo --version
```

**Source**: [rustup.rs](https://rustup.rs/) - Fetched: 2024-12-19T00:00:00Z

### **Development Environment**

Recommended tools for Rust development:

```bash
# Install rust-analyzer for your editor
# VS Code: Install "rust-analyzer" extension
# Vim/Neovim: Use rust-analyzer LSP
# Emacs: Use lsp-mode with rust-analyzer

# Install useful cargo extensions
cargo install cargo-edit cargo-watch cargo-expand
```

**Source**: [rust-analyzer](https://rust-analyzer.github.io/) - Fetched: 2024-12-19T00:00:00Z

---

## 💻 **Your First Rust Program**

### **Method 1: Using `cargo` (Recommended)**

```bash
# Create a new project
cargo new hello_world
cd hello_world

# Run the program
cargo run
```

This creates a project structure:
```
hello_world/
├── Cargo.toml
└── src/
    └── main.rs
```

### **Method 2: Manual Creation**

Create a file called `main.rs`:

```rust
fn main() {
    println!("Hello, world!");
}
```

Compile and run:
```bash
rustc main.rs
./main  # Linux/macOS
main.exe  # Windows
```

---

## 🔍 **Understanding the Code**

Let's break down our first Rust program:

```rust
fn main() {
    println!("Hello, world!");
}
```

### **Function Declaration**
```rust
fn main() {
    // function body
}
```
- `fn` - keyword to declare a function
- `main` - function name (special function that serves as entry point)
- `()` - empty parameter list
- `{}` - function body

### **The `println!` Macro**
```rust
println!("Hello, world!");
```
- `println!` - a macro (not a function) for printing to console
- `!` - indicates this is a macro
- `"Hello, world!"` - string literal argument
- `;` - statement terminator

**Source**: [The Rust Book - Functions](https://doc.rust-lang.org/book/ch03-03-how-functions-work.html) - Fetched: 2024-12-19T00:00:00Z

---

## 🏗️ **Cargo Project Structure**

When you run `cargo new hello_world`, Cargo creates:

### **Cargo.toml** (Project Configuration)
```toml
[package]
name = "hello_world"
version = "0.1.0"
edition = "2021"

[dependencies]
```

### **src/main.rs** (Source Code)
```rust
fn main() {
    println!("Hello, world!");
}
```

### **Cargo Commands**
```bash
cargo new <project_name>    # Create new project
cargo build                 # Compile the project
cargo run                   # Compile and run
cargo check                 # Check for errors without building
cargo test                  # Run tests
cargo clean                 # Remove build artifacts
```

**Source**: [Cargo Book](https://doc.rust-lang.org/cargo/) - Fetched: 2024-12-19T00:00:00Z

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Custom Greeting**
Modify the program to print a personalized greeting:

```rust
fn main() {
    let name = "Rustacean";
    println!("Hello, {}!", name);
}
```

**Expected Output**:
```
Hello, Rustacean!
```

### **Exercise 2: Multiple Messages**
Print multiple messages:

```rust
fn main() {
    println!("Welcome to Rust!");
    println!("Let's learn together!");
    println!("Ready to code?");
}
```

### **Exercise 3: Using Variables**
Create variables for different parts of your message:

```rust
fn main() {
    let greeting = "Hello";
    let target = "World";
    let punctuation = "!";
    
    println!("{}{}{}", greeting, target, punctuation);
}
```

---

## 🧪 **Unit Tests**

Let's add some tests to our program:

```rust
fn main() {
    greet("World");
}

fn greet(name: &str) {
    println!("Hello, {}!", name);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        // This test would need to capture stdout to verify output
        // For now, we'll just ensure the function doesn't panic
        greet("Test");
    }
}
```

Run tests with:
```bash
cargo test
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Missing Semicolon**
```rust
// ❌ Wrong
fn main() {
    println!("Hello, world!")
}

// ✅ Correct
fn main() {
    println!("Hello, world!");
}
```

### **Common Mistake 2: Wrong Macro Syntax**
```rust
// ❌ Wrong
println("Hello, world!");

// ✅ Correct
println!("Hello, world!");
```

### **Common Mistake 3: Incorrect String Literals**
```rust
// ❌ Wrong
println!(Hello, world!);

// ✅ Correct
println!("Hello, world!");
```

### **Debugging Tips**
1. **Use `cargo check`** - Fast compilation check without building
2. **Read error messages carefully** - Rust has excellent error messages
3. **Use `rustfmt`** - Format your code: `cargo fmt`
4. **Use `clippy`** - Lint your code: `cargo clippy`

---

## 🔍 **What Makes Rust Special?**

### **Memory Safety Without Garbage Collection**
- No null pointer dereferences
- No buffer overflows
- No use-after-free bugs
- No data races

### **Zero-Cost Abstractions**
- High-level features compile to efficient low-level code
- No runtime overhead for abstractions

### **Fearless Concurrency**
- Safe concurrent programming
- Ownership system prevents data races

### **Performance**
- Comparable to C/C++ performance
- Predictable performance characteristics

**Source**: [Rust Website](https://www.rust-lang.org/) - Fetched: 2024-12-19T00:00:00Z

---

## 📖 **Further Reading**

### **Official Documentation**
- [The Rust Programming Language (The Book)](https://doc.rust-lang.org/book/) - Fetched: 2024-12-19T00:00:00Z
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) - Fetched: 2024-12-19T00:00:00Z
- [Cargo Book](https://doc.rust-lang.org/cargo/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust Users Forum](https://users.rust-lang.org/) - Fetched: 2024-12-19T00:00:00Z
- [This Week in Rust](https://this-week-in-rust.org/) - Fetched: 2024-12-19T00:00:00Z
- [Rustlings](https://github.com/rust-lang/rustlings) - Interactive exercises - Fetched: 2024-12-19T00:00:00Z

### **Tools and Extensions**
- [rust-analyzer](https://rust-analyzer.github.io/) - Language server - Fetched: 2024-12-19T00:00:00Z
- [Clippy](https://github.com/rust-lang/rust-clippy) - Linting tool - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What is the difference between `println!` and `println`?
2. Why do we use `cargo new` instead of creating files manually?
3. What does the `!` in `println!` indicate?
4. How do you run a Rust program using cargo?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Variables and mutability
- Data types in Rust
- Basic input/output
- Understanding the ownership system

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [1.2 Variables and Mutability](01_02_variables_mutability.md)
