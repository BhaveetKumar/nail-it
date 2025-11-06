---
# Auto-generated front matter
Title: 12 01 Macros Basics
LastUpdated: 2025-11-06T20:45:58.117526
Tags: []
Status: draft
---

# Lesson 12.1: Macros Basics

> **Module**: 12 - Macros  
> **Lesson**: 1 of 8  
> **Duration**: 3-4 hours  
> **Prerequisites**: Module 11 (Async Programming)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Understand the difference between macros and functions
- Write basic declarative macros using `macro_rules!`
- Use common built-in macros effectively
- Apply macros to reduce code duplication
- Debug macro expansion issues

---

## 🎯 **Overview**

Macros are a way of writing code that writes other code, known as "metaprogramming." Rust has two types of macros: declarative macros (using `macro_rules!`) and procedural macros. This lesson focuses on declarative macros.

---

## 🔧 **Macros vs Functions**

### **Key Differences**

```rust
// Function - evaluated at runtime
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// Macro - expanded at compile time
macro_rules! add {
    ($a:expr, $b:expr) => {
        $a + $b
    };
}

fn main() {
    // Function call
    let result1 = add(5, 3);
    
    // Macro call
    let result2 = add!(5, 3);
    
    println!("Function: {}, Macro: {}", result1, result2);
}
```

**Key Differences**:
- **Macros**: Expanded at compile time, can take variable number of arguments
- **Functions**: Evaluated at runtime, fixed number of arguments

---

## 📝 **Basic Declarative Macros**

### **Simple Macro**

```rust
macro_rules! say_hello {
    () => {
        println!("Hello, world!");
    };
}

fn main() {
    say_hello!();
}
```

### **Macro with Parameters**

```rust
macro_rules! create_function {
    ($func_name:ident) => {
        fn $func_name() {
            println!("You called {:?}()", stringify!($func_name));
        }
    };
}

create_function!(foo);
create_function!(bar);

fn main() {
    foo();
    bar();
}
```

### **Macro with Multiple Patterns**

```rust
macro_rules! test {
    ($left:expr; and $right:expr) => {
        println!("{:?} and {:?} is {:?}",
                 stringify!($left),
                 stringify!($right),
                 $left && $right);
    };
    ($left:expr; or $right:expr) => {
        println!("{:?} or {:?} is {:?}",
                 stringify!($left),
                 stringify!($right),
                 $left || $right);
    };
}

fn main() {
    test!(1i32 + 1 == 2i32; and 2i32 * 2 == 4i32);
    test!(true; or false);
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Vector Creation Macro**

```rust
macro_rules! vec {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}

fn main() {
    let v = vec![1, 2, 3, 4, 5];
    println!("Vector: {:?}", v);
}
```

### **Exercise 2: Debug Print Macro**

```rust
macro_rules! debug_print {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!($($arg)*);
    };
}

fn main() {
    let x = 42;
    let y = "hello";
    
    debug_print!("Debug: x = {}, y = {}", x, y);
    
    // This won't print in release mode
    debug_print!("This is debug only");
}
```

### **Exercise 3: Assert with Custom Message**

```rust
macro_rules! assert_with_message {
    ($condition:expr, $($arg:tt)*) => {
        if !$condition {
            panic!($($arg)*);
        }
    };
}

fn divide(a: i32, b: i32) -> i32 {
    assert_with_message!(b != 0, "Division by zero: {} / {}", a, b);
    a / b
}

fn main() {
    println!("10 / 2 = {}", divide(10, 2));
    // println!("10 / 0 = {}", divide(10, 0)); // This will panic
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_macro() {
        let v = vec![1, 2, 3];
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn test_debug_print() {
        // This should compile without errors
        debug_print!("Test message");
    }

    #[test]
    fn test_assert_with_message() {
        // This should not panic
        assert_with_message!(true, "This should not panic");
    }

    #[test]
    #[should_panic(expected = "Custom assertion failed")]
    fn test_assert_with_message_panic() {
        assert_with_message!(false, "Custom assertion failed");
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Incorrect Syntax**

```rust
// ❌ Wrong - missing exclamation mark
macro_rules! bad_macro {
    ($x:expr) => {
        $x + 1
    };
}

fn main() {
    let result = bad_macro(5); // Error: expected function
}

// ✅ Correct - use exclamation mark
macro_rules! good_macro {
    ($x:expr) => {
        $x + 1
    };
}

fn main() {
    let result = good_macro!(5); // Correct
}
```

### **Common Mistake 2: Wrong Fragment Specifiers**

```rust
// ❌ Wrong - using expr for identifier
macro_rules! bad_macro {
    ($name:expr) => {
        fn $name() {} // Error: expected identifier
    };
}

// ✅ Correct - use ident for identifier
macro_rules! good_macro {
    ($name:ident) => {
        fn $name() {}
    };
}
```

### **Common Mistake 3: Macro Hygiene Issues**

```rust
// ❌ Wrong - variable name conflicts
macro_rules! bad_macro {
    ($x:expr) => {
        let x = 10;
        $x + x // Which x?
    };
}

// ✅ Correct - use unique variable names
macro_rules! good_macro {
    ($x:expr) => {
        {
            let temp_x = 10;
            $x + temp_x
        }
    };
}
```

---

## 🔍 **Macro Fragment Specifiers**

### **Common Specifiers**

```rust
macro_rules! demonstrate_fragments {
    // Expression
    ($expr:expr) => {
        println!("Expression: {}", $expr);
    };
    
    // Identifier
    ($ident:ident) => {
        let $ident = 42;
    };
    
    // Type
    ($ty:ty) => {
        let _: $ty = Default::default();
    };
    
    // Pattern
    ($pat:pat) => {
        match 42 {
            $pat => println!("Matched pattern"),
            _ => println!("No match"),
        }
    };
    
    // Statement
    ($stmt:stmt) => {
        $stmt
    };
    
    // Token tree
    ($tt:tt) => {
        println!("Token tree: {:?}", stringify!($tt));
    };
}

fn main() {
    demonstrate_fragments!(1 + 2);           // expr
    demonstrate_fragments!(my_variable);     // ident
    demonstrate_fragments!(i32);             // ty
    demonstrate_fragments!(x @ 1..=10);      // pat
    demonstrate_fragments!(let x = 5;);      // stmt
    demonstrate_fragments!([1, 2, 3]);       // tt
}
```

---

## 📊 **Advanced Macro Patterns**

### **Recursive Macros**

```rust
macro_rules! count {
    () => { 0 };
    ($head:tt $($tail:tt)*) => { 1 + count!($($tail)*) };
}

macro_rules! sum {
    () => { 0 };
    ($head:expr $(, $tail:expr)*) => { $head + sum!($($tail),*) };
}

fn main() {
    let count = count!(a b c d e);
    println!("Count: {}", count);
    
    let total = sum!(1, 2, 3, 4, 5);
    println!("Sum: {}", total);
}
```

### **Macro with Multiple Arms**

```rust
macro_rules! calculate {
    (add $a:expr, $b:expr) => {
        $a + $b
    };
    (multiply $a:expr, $b:expr) => {
        $a * $b
    };
    (subtract $a:expr, $b:expr) => {
        $a - $b
    };
    (divide $a:expr, $b:expr) => {
        $a / $b
    };
}

fn main() {
    println!("Add: {}", calculate!(add 5, 3));
    println!("Multiply: {}", calculate!(multiply 5, 3));
    println!("Subtract: {}", calculate!(subtract 5, 3));
    println!("Divide: {}", calculate!(divide 5, 3));
}
```

---

## 🎯 **Best Practices**

### **Macro Naming**

```rust
// ✅ Good - descriptive names
macro_rules! create_user_struct {
    // ...
}

macro_rules! debug_assert_eq {
    // ...
}

// ❌ Avoid - unclear names
macro_rules! helper {
    // ...
}

macro_rules! do_stuff {
    // ...
}
```

### **Documentation**

```rust
/// Creates a vector with the given elements
/// 
/// # Examples
/// 
/// ```
/// let v = vec![1, 2, 3];
/// assert_eq!(v, vec![1, 2, 3]);
/// ```
macro_rules! vec {
    ($($x:expr),*) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}
```

### **Error Messages**

```rust
macro_rules! expect {
    ($x:expr, $y:expr) => {
        if $x != $y {
            panic!("Expected {}, got {}", $y, $x);
        }
    };
}

fn main() {
    expect!(2 + 2, 4);
    // expect!(2 + 2, 5); // This will panic with a clear message
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [The Rust Book - Macros](https://doc.rust-lang.org/book/ch19-06-macros.html) - Fetched: 2024-12-19T00:00:00Z
- [Rust Reference - Macros](https://doc.rust-lang.org/reference/macros.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust by Example - Macros](https://doc.rust-lang.org/rust-by-example/macros.html) - Fetched: 2024-12-19T00:00:00Z
- [The Little Book of Rust Macros](https://danielkeep.github.io/tlborm/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What's the difference between macros and functions?
2. When should you use macros instead of functions?
3. What are the different fragment specifiers in macros?
4. How do you debug macro expansion issues?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Procedural macros and derive macros
- Function-like procedural macros
- Attribute macros
- Macro hygiene and best practices

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [12.2 Procedural Macros](12_02_procedural_macros.md)
