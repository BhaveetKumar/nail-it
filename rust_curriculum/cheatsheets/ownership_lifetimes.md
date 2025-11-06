---
# Auto-generated front matter
Title: Ownership Lifetimes
LastUpdated: 2025-11-06T20:45:58.130359
Tags: []
Status: draft
---

# 🦀 Rust Ownership & Lifetimes Cheat Sheet

> **Quick Reference for Rust's Memory Management System**  
> **Last Updated**: 2024-12-19T00:00:00Z  
> **Rust Version**: 1.75.0

---

## 📋 **Ownership Rules**

### **The Three Rules of Ownership**
1. **Each value has one owner**
2. **Only one owner at a time**
3. **When owner goes out of scope, value is dropped**

```rust
// ✅ Valid: s1 owns the string
let s1 = String::from("hello");

// ❌ Invalid: s2 tries to own the same string
let s2 = s1; // s1 is moved to s2, s1 is no longer valid

// ❌ Invalid: using s1 after move
println!("{}", s1); // Error: value used after move
```

---

## 🔄 **Moving vs Copying**

### **Move Semantics (Default)**
```rust
let s1 = String::from("hello");
let s2 = s1; // s1 is moved to s2
// s1 is no longer valid here
```

### **Copy Semantics (Stack-Only Types)**
```rust
let x = 5;
let y = x; // x is copied to y
// Both x and y are valid
```

### **Types That Implement Copy**
- All integer types (`i32`, `u64`, etc.)
- Boolean type (`bool`)
- Floating point types (`f32`, `f64`)
- Character type (`char`)
- Tuples with only `Copy` types
- Arrays with only `Copy` types

---

## 📖 **Borrowing**

### **Immutable Borrows**
```rust
let s1 = String::from("hello");
let len = calculate_length(&s1); // &s1 is an immutable borrow
// s1 is still valid here

fn calculate_length(s: &String) -> usize {
    s.len() // s is a reference to s1
} // s goes out of scope, but s1 is not dropped
```

### **Mutable Borrows**
```rust
let mut s = String::from("hello");
change(&mut s); // &mut s is a mutable borrow

fn change(s: &mut String) {
    s.push_str(", world");
}
```

### **Borrowing Rules**
1. **At any time, you can have either:**
   - One mutable reference, OR
   - Any number of immutable references
2. **References must always be valid**

```rust
// ❌ Invalid: multiple mutable borrows
let mut s = String::from("hello");
let r1 = &mut s;
let r2 = &mut s; // Error: cannot borrow as mutable more than once

// ❌ Invalid: mutable and immutable borrows
let r1 = &s;
let r2 = &mut s; // Error: cannot borrow as mutable while borrowed as immutable
```

---

## ⏰ **Lifetimes**

### **Basic Lifetime Syntax**
```rust
// Lifetime parameter in function
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

### **Lifetime in Structs**
```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a '.'");
    let i = ImportantExcerpt {
        part: first_sentence,
    };
}
```

### **Lifetime Elision Rules**
```rust
// These are equivalent due to lifetime elision:

// With explicit lifetimes
fn first_word<'a>(s: &'a str) -> &'a str { /* ... */ }

// With elided lifetimes (Rust infers them)
fn first_word(s: &str) -> &str { /* ... */ }
```

---

## 🎯 **Common Patterns**

### **Returning References**
```rust
// ✅ Valid: returning a reference to input
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    &s[..]
}

// ❌ Invalid: returning reference to local variable
fn dangle() -> &String {
    let s = String::from("hello");
    &s // Error: s goes out of scope
}
```

### **Multiple Lifetimes**
```rust
fn longest_with_an_announcement<'a, 'b>(
    x: &'a str,
    y: &'b str,
    ann: &str,
) -> &'a str
where
    'a: 'b, // 'a must live at least as long as 'b
{
    println!("Announcement! {}", ann);
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

### **Static Lifetime**
```rust
// 'static lifetime: lives for entire program duration
let s: &'static str = "I have a static lifetime.";
```

---

## 🔧 **Practical Examples**

### **String vs &str**
```rust
// String: owned, growable
let owned_string = String::from("hello");
let owned_string2 = owned_string; // moved

// &str: borrowed, immutable slice
let string_slice = "hello";
let string_slice2 = string_slice; // copied (it's a reference)
```

### **Vector Ownership**
```rust
let v = vec![1, 2, 3, 4, 5];
let first = &v[0]; // immutable borrow
// v.push(6); // Error: cannot borrow as mutable while borrowed as immutable
println!("The first element is: {}", first);
```

### **Closure Ownership**
```rust
let list = vec![1, 2, 3];
let only_borrows = || println!("From closure: {:?}", list);
only_borrows();
println!("From main: {:?}", list); // list is still valid

let list2 = vec![1, 2, 3];
let takes_ownership = move || println!("From closure: {:?}", list2);
takes_ownership();
// println!("From main: {:?}", list2); // Error: list2 was moved
```

---

## 🚨 **Common Errors and Solutions**

### **Error: "value used after move"**
```rust
// ❌ Problem
let s1 = String::from("hello");
let s2 = s1;
println!("{}", s1); // Error: value used after move

// ✅ Solution 1: Clone
let s1 = String::from("hello");
let s2 = s1.clone();
println!("{}", s1); // OK

// ✅ Solution 2: Borrow
let s1 = String::from("hello");
let s2 = &s1;
println!("{}", s1); // OK
```

### **Error: "cannot borrow as mutable"**
```rust
// ❌ Problem
let mut v = vec![1, 2, 3];
let first = &v[0];
v.push(4); // Error: cannot borrow as mutable

// ✅ Solution: Limit scope of immutable borrow
let mut v = vec![1, 2, 3];
{
    let first = &v[0];
    println!("First: {}", first);
} // first goes out of scope
v.push(4); // OK
```

### **Error: "lifetime may not live long enough"**
```rust
// ❌ Problem
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// ✅ Solution: Add lifetime parameters
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

---

## 🎯 **Quick Reference**

### **Ownership Transfer**
```rust
// Move (default for non-Copy types)
let s1 = String::from("hello");
let s2 = s1; // s1 moved to s2

// Clone (explicit copy)
let s1 = String::from("hello");
let s2 = s1.clone(); // s1 copied to s2

// Borrow (reference)
let s1 = String::from("hello");
let s2 = &s1; // s2 borrows s1
```

### **Borrowing Rules**
```rust
// Immutable borrows
let r1 = &s;
let r2 = &s; // OK: multiple immutable borrows

// Mutable borrow
let r3 = &mut s; // OK: only one mutable borrow

// Mixed borrows
let r1 = &s;
let r2 = &mut s; // Error: cannot mix immutable and mutable
```

### **Lifetime Annotations**
```rust
// Function with lifetime
fn func<'a>(x: &'a str) -> &'a str { x }

// Struct with lifetime
struct S<'a> { field: &'a str }

// Method with lifetime
impl<'a> S<'a> {
    fn method<'b>(&self, other: &'b str) -> &'a str { self.field }
}
```

---

## 📚 **Further Reading**

- [The Rust Book - Understanding Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html) - Fetched: 2024-12-19T00:00:00Z
- [The Rust Book - References and Borrowing](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html) - Fetched: 2024-12-19T00:00:00Z
- [The Rust Book - The Slice Type](https://doc.rust-lang.org/book/ch04-03-slices.html) - Fetched: 2024-12-19T00:00:00Z
- [The Rust Book - Validating References with Lifetimes](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html) - Fetched: 2024-12-19T00:00:00Z

---

**Cheat Sheet Version**: 1.0  
**Rust Version**: 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z
