# Lesson 3.1: Ownership Basics

> **Module**: 3 - Ownership and Borrowing  
> **Lesson**: 1 of 8  
> **Duration**: 3-4 hours  
> **Prerequisites**: Module 2 (Basic Syntax and Data Types)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Understand Rust's ownership system and its three rules
- Explain the difference between stack and heap memory
- Identify when values are moved vs copied
- Understand the concept of ownership transfer
- Apply ownership principles to write memory-safe code

---

## 🎯 **Overview**

Ownership is Rust's most unique feature and the key to memory safety without garbage collection. It's a system of rules that the compiler checks at compile time, ensuring memory safety without runtime overhead. This lesson introduces the fundamental concepts of ownership in Rust.

---

## 🧠 **Memory Management in Rust**

### **Stack vs Heap**

```rust
fn main() {
    // Stack-allocated data
    let x = 5;                    // i32 on stack
    let y = x;                    // Copy of x on stack
    
    // Heap-allocated data
    let s1 = String::from("hello"); // String on heap
    let s2 = s1;                    // s1 is moved to s2
    // println!("{}", s1);         // Error: s1 no longer valid
}
```

**Key Differences**:
- **Stack**: Fast, fixed size, automatic cleanup
- **Heap**: Slower, dynamic size, manual management
- **Rust**: Automatic heap management through ownership

---

## 📋 **The Three Rules of Ownership**

### **Rule 1: Each value has one owner**

```rust
fn main() {
    let s = String::from("hello"); // s owns the string
    // Only s can access and modify this string
}
```

### **Rule 2: Only one owner at a time**

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2
    // s1 is no longer valid here
    println!("{}", s2); // OK: s2 owns the string
}
```

### **Rule 3: When owner goes out of scope, value is dropped**

```rust
fn main() {
    {
        let s = String::from("hello"); // s comes into scope
        // s is valid here
    } // s goes out of scope, string is dropped
    // s is no longer valid here
}
```

---

## 🔄 **Move Semantics**

### **What Happens During a Move**

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2
    
    // s1 is no longer valid
    // s2 now owns the string data
    println!("{}", s2); // OK
    // println!("{}", s1); // Error: value used after move
}
```

### **Move vs Copy**

```rust
fn main() {
    // Copy semantics (stack-only data)
    let x = 5;
    let y = x; // x is copied to y
    println!("x: {}, y: {}", x, y); // Both valid
    
    // Move semantics (heap data)
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2
    // println!("{}", s1); // Error: s1 no longer valid
    println!("{}", s2); // OK
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Understanding Moves**

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1; // Move occurs here
    
    // Try to use s1 - this will cause a compilation error
    // println!("s1: {}", s1);
    
    println!("s2: {}", s2);
}
```

**Expected Output**:
```
s2: hello
```

### **Exercise 2: Copy vs Move**

```rust
fn main() {
    // Integer (implements Copy)
    let x = 5;
    let y = x;
    println!("x: {}, y: {}", x, y); // Both work
    
    // String (moved)
    let s1 = String::from("world");
    let s2 = s1;
    // println!("s1: {}", s1); // Error
    println!("s2: {}", s2);
}
```

### **Exercise 3: Ownership in Functions**

```rust
fn takes_ownership(s: String) {
    println!("{}", s);
} // s goes out of scope and is dropped

fn makes_copy(x: i32) {
    println!("{}", x);
} // x goes out of scope, but i32 is Copy

fn main() {
    let s = String::from("hello");
    takes_ownership(s); // s is moved into the function
    // s is no longer valid here
    
    let x = 5;
    makes_copy(x); // x is copied into the function
    println!("x is still valid: {}", x); // x is still valid
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_semantics() {
        let s1 = String::from("hello");
        let s2 = s1; // Move
        assert_eq!(s2, "hello");
    }

    #[test]
    fn test_copy_semantics() {
        let x = 5;
        let y = x; // Copy
        assert_eq!(x, 5);
        assert_eq!(y, 5);
    }

    #[test]
    fn test_ownership_transfer() {
        let s = String::from("test");
        let result = takes_and_returns(s);
        assert_eq!(result, "test");
    }

    fn takes_and_returns(s: String) -> String {
        s // Return transfers ownership back
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Using Moved Value**
```rust
// ❌ Wrong
let s1 = String::from("hello");
let s2 = s1;
println!("{}", s1); // Error: value used after move

// ✅ Correct
let s1 = String::from("hello");
let s2 = s1;
println!("{}", s2); // Use s2 instead
```

### **Common Mistake 2: Returning Ownership**
```rust
// ❌ Wrong - loses ownership
fn bad_function(s: String) {
    println!("{}", s);
    // s is dropped here, ownership lost
}

// ✅ Correct - return ownership
fn good_function(s: String) -> String {
    println!("{}", s);
    s // Return s to give ownership back
}
```

### **Common Mistake 3: Confusing Move and Copy**
```rust
// ❌ Wrong - trying to use after move
let s1 = String::from("hello");
let s2 = s1;
let s3 = s1; // Error: s1 was moved to s2

// ✅ Correct - use clone for heap data
let s1 = String::from("hello");
let s2 = s1.clone(); // Clone creates a copy
let s3 = s1; // Now this works
```

---

## 🔍 **Understanding the String Type**

### **String vs &str**

```rust
fn main() {
    // String: owned, growable, heap-allocated
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2
    
    // &str: borrowed, immutable, can point to string literal or String
    let s3 = "hello"; // string literal
    let s4 = s3; // s3 is copied (it's a reference)
    println!("s3: {}, s4: {}", s3, s4); // Both valid
}
```

### **Why String is Moved**

```rust
fn main() {
    let s1 = String::from("hello");
    // s1 contains: ptr, len, capacity
    // ptr points to heap memory containing "hello"
    
    let s2 = s1; // s1's data is moved to s2
    // s1 is no longer valid to prevent double-free
}
```

---

## 📊 **Types That Implement Copy**

```rust
fn main() {
    // All of these implement Copy
    let x = 5;        // i32
    let y = 3.14;     // f64
    let z = true;     // bool
    let c = 'a';      // char
    let t = (1, 2);   // tuple with Copy types
    
    // These can be copied
    let x2 = x;
    let y2 = y;
    let z2 = z;
    let c2 = c;
    let t2 = t;
    
    // All original variables are still valid
    println!("x: {}, x2: {}", x, x2);
}
```

**Types that implement Copy**:
- All integer types (`i32`, `u64`, etc.)
- Boolean type (`bool`)
- Floating point types (`f32`, `f64`)
- Character type (`char`)
- Tuples with only `Copy` types
- Arrays with only `Copy` types

---

## 🎯 **Best Practices**

### **When to Use Clone**
```rust
// ✅ Use clone when you need a copy of heap data
let s1 = String::from("hello");
let s2 = s1.clone(); // Explicit copy
println!("s1: {}, s2: {}", s1, s2);

// ❌ Don't clone unnecessarily
let x = 5;
let y = x.clone(); // Unnecessary - i32 is Copy
```

### **Function Design**
```rust
// ✅ Design functions to take ownership when appropriate
fn process_string(s: String) -> String {
    // Process the string
    s.to_uppercase()
}

// ✅ Or use references to avoid taking ownership
fn process_string_ref(s: &String) -> String {
    s.to_uppercase()
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [The Rust Book - Understanding Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html) - Fetched: 2024-12-19T00:00:00Z
- [The Rust Book - What is Ownership?](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust by Example - Move Semantics](https://doc.rust-lang.org/rust-by-example/scope/move.html) - Fetched: 2024-12-19T00:00:00Z
- [Rustlings - Ownership](https://github.com/rust-lang/rustlings) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What are the three rules of ownership in Rust?
2. What's the difference between move and copy semantics?
3. Why does `String` use move semantics while `i32` uses copy semantics?
4. What happens to a value when its owner goes out of scope?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- References and borrowing
- Mutable and immutable references
- Borrowing rules and restrictions
- Slices and string slices

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [3.2 References and Borrowing](03_02_references_borrowing.md)
