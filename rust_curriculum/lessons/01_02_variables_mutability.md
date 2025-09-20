# Lesson 1.2: Variables and Mutability

> **Module**: 1 - Introduction to Rust  
> **Lesson**: 2 of 4  
> **Duration**: 2-3 hours  
> **Prerequisites**: Lesson 1.1 (Hello, World!)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Understand the difference between `let` and `let mut`
- Explain Rust's approach to mutability
- Use variable shadowing effectively
- Understand the concept of constants
- Apply best practices for variable naming

---

## 🎯 **Overview**

Rust's approach to variables is unique among programming languages. By default, variables are immutable, which means they cannot be changed after assignment. This design choice helps prevent bugs and makes code more predictable. In this lesson, we'll explore variables, mutability, and shadowing in detail.

---

## 🔧 **Basic Variables**

### **Immutable Variables (Default)**

```rust
fn main() {
    let x = 5;
    println!("The value of x is: {}", x);
    
    // This would cause a compilation error:
    // x = 6; // Error: cannot assign to immutable variable
}
```

**Key Points**:
- Variables are immutable by default
- Use `let` to declare variables
- The compiler prevents accidental modifications

### **Mutable Variables**

```rust
fn main() {
    let mut x = 5;
    println!("The value of x is: {}", x);
    
    x = 6; // This is allowed because x is mutable
    println!("The value of x is: {}", x);
}
```

**Key Points**:
- Use `let mut` to declare mutable variables
- Mutable variables can be changed after assignment
- Choose mutability explicitly for better code clarity

---

## 🔄 **Variable Shadowing**

Rust allows you to declare a new variable with the same name as a previous variable. This is called "shadowing."

```rust
fn main() {
    let x = 5;
    let x = x + 1; // Shadowing: x is now 6
    let x = x * 2; // Shadowing: x is now 12
    
    println!("The value of x is: {}", x);
}
```

### **Shadowing vs Mutability**

```rust
fn main() {
    // Using shadowing
    let spaces = "   ";
    let spaces = spaces.len(); // spaces is now a number
    
    // This would not work with mutability:
    // let mut spaces = "   ";
    // spaces = spaces.len(); // Error: mismatched types
}
```

**Key Differences**:
- **Shadowing**: Creates a new variable, can change type
- **Mutability**: Modifies existing variable, same type required

---

## 📊 **Data Types**

### **Scalar Types**

```rust
fn main() {
    // Integers
    let a: i32 = 42;        // 32-bit signed integer
    let b: u64 = 100;       // 64-bit unsigned integer
    let c = 1_000_000;      // Underscores for readability
    
    // Floating-point
    let d: f64 = 3.14;      // 64-bit floating point
    let e = 2.0;            // Type inference: f64
    
    // Boolean
    let f: bool = true;
    let g = false;          // Type inference: bool
    
    // Character
    let h: char = 'z';      // Unicode scalar value
    let i = '🦀';           // Emoji character
}
```

### **Compound Types**

```rust
fn main() {
    // Tuple
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    let (x, y, z) = tup;    // Destructuring
    println!("The value of y is: {}", y);
    
    // Array
    let arr = [1, 2, 3, 4, 5];
    let first = arr[0];     // Access by index
    println!("The first element is: {}", first);
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Basic Variables**
Create variables for different data types and print them:

```rust
fn main() {
    let name = "Rustacean";
    let age = 25;
    let height = 5.9;
    let is_student = true;
    
    println!("Name: {}", name);
    println!("Age: {}", age);
    println!("Height: {}", height);
    println!("Is student: {}", is_student);
}
```

**Expected Output**:
```
Name: Rustacean
Age: 25
Height: 5.9
Is student: true
```

### **Exercise 2: Variable Shadowing**
Use shadowing to transform a string into a number:

```rust
fn main() {
    let text = "42";
    let text = text.len();  // Shadow: text is now a number
    let text = text * 2;    // Shadow: text is now 4
    
    println!("The value is: {}", text);
}
```

### **Exercise 3: Mutable vs Immutable**
Compare mutable and immutable variables:

```rust
fn main() {
    // Immutable
    let x = 5;
    println!("Immutable x: {}", x);
    
    // Mutable
    let mut y = 5;
    y = y + 1;
    println!("Mutable y: {}", y);
    
    // Shadowing
    let z = 5;
    let z = z + 1;
    println!("Shadowed z: {}", z);
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_immutable_variable() {
        let x = 5;
        assert_eq!(x, 5);
    }

    #[test]
    fn test_mutable_variable() {
        let mut x = 5;
        x = 6;
        assert_eq!(x, 6);
    }

    #[test]
    fn test_variable_shadowing() {
        let x = 5;
        let x = x + 1;
        let x = x * 2;
        assert_eq!(x, 12);
    }

    #[test]
    fn test_different_types() {
        let text = "hello";
        let text = text.len();
        assert_eq!(text, 5);
    }
}
```

Run tests with:
```bash
cargo test
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Forgetting `mut`**
```rust
// ❌ Wrong
let x = 5;
x = 6; // Error: cannot assign to immutable variable

// ✅ Correct
let mut x = 5;
x = 6; // OK
```

### **Common Mistake 2: Type Mismatch with Mutability**
```rust
// ❌ Wrong
let mut spaces = "   ";
spaces = spaces.len(); // Error: mismatched types

// ✅ Correct - Use shadowing
let spaces = "   ";
let spaces = spaces.len(); // OK: creates new variable
```

### **Common Mistake 3: Unused Variables**
```rust
// ❌ Warning: unused variable
let x = 5;

// ✅ Correct - Use the variable or prefix with underscore
let x = 5;
println!("{}", x);

// Or suppress warning
let _x = 5;
```

### **Debugging Tips**
1. **Read error messages carefully** - Rust has excellent error messages
2. **Use `cargo check`** - Fast compilation check without building
3. **Use `rust-analyzer`** - IDE integration for better error detection
4. **Enable warnings** - Use `cargo clippy` for additional checks

---

## 📖 **Constants**

Constants are different from variables:

```rust
const MAX_POINTS: u32 = 100_000;

fn main() {
    println!("Maximum points: {}", MAX_POINTS);
}
```

**Key Differences**:
- **Constants**: Always immutable, must have type annotation
- **Variables**: Can be mutable, type annotation optional
- **Constants**: Computed at compile time
- **Variables**: Computed at runtime

---

## 🎯 **Best Practices**

### **Variable Naming**
```rust
// ✅ Good naming
let user_name = "Alice";
let max_retries = 3;
let is_authenticated = true;

// ❌ Poor naming
let x = "Alice";
let n = 3;
let flag = true;
```

### **When to Use `mut`**
```rust
// ✅ Use mut when you need to change the value
let mut counter = 0;
counter += 1;

// ❌ Don't use mut if you're not changing the value
let mut name = "Alice"; // Unnecessary mut
```

### **When to Use Shadowing**
```rust
// ✅ Use shadowing to change type
let input = "42";
let input: i32 = input.parse().unwrap();

// ✅ Use shadowing for transformations
let value = 5;
let value = value * 2;
let value = value + 1;
```

---

## 🔍 **Type Inference**

Rust can often infer types:

```rust
fn main() {
    let x = 5;        // i32
    let y = 3.14;     // f64
    let z = true;     // bool
    let w = 'a';      // char
    
    // Sometimes you need to specify
    let a = 42;       // i32 by default
    let b: u8 = 42;   // u8 explicitly
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [The Rust Book - Variables and Mutability](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html) - Fetched: 2024-12-19T00:00:00Z
- [The Rust Book - Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust by Example - Variable Bindings](https://doc.rust-lang.org/rust-by-example/variable_bindings.html) - Fetched: 2024-12-19T00:00:00Z
- [Rustlings - Variables](https://github.com/rust-lang/rustlings) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What's the difference between `let` and `let mut`?
2. How does variable shadowing differ from mutability?
3. When should you use constants instead of variables?
4. What are the benefits of Rust's default immutability?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Functions and parameters
- Return values and expressions
- Statements vs expressions
- Understanding scope and lifetime

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [1.3 Functions and Control Flow](01_03_functions_control_flow.md)
