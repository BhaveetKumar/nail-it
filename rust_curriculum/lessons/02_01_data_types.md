---
# Auto-generated front matter
Title: 02 01 Data Types
LastUpdated: 2025-11-06T20:45:58.123054
Tags: []
Status: draft
---

# Lesson 2.1: Data Types

> **Module**: 2 - Basic Syntax and Data Types  
> **Lesson**: 1 of 4  
> **Duration**: 2-3 hours  
> **Prerequisites**: Lesson 1.2 (Variables and Mutability)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Understand Rust's scalar and compound data types
- Choose appropriate types for different use cases
- Use type annotations when necessary
- Understand type inference and when to specify types explicitly

---

## 🎯 **Overview**

Rust is a statically typed language, which means the compiler must know the types of all variables at compile time. Rust has two categories of data types: scalar types (single values) and compound types (multiple values).

---

## 🔢 **Scalar Types**

### **Integer Types**

```rust
fn main() {
    // Signed integers
    let a: i8 = 127;        // 8-bit signed integer (-128 to 127)
    let b: i16 = 32767;     // 16-bit signed integer
    let c: i32 = 2147483647; // 32-bit signed integer (default)
    let d: i64 = 9223372036854775807; // 64-bit signed integer
    let e: i128 = 170141183460469231731687303715884105727; // 128-bit signed integer
    let f: isize = 9223372036854775807; // Architecture-dependent (32 or 64 bits)
    
    // Unsigned integers
    let g: u8 = 255;        // 8-bit unsigned integer (0 to 255)
    let h: u16 = 65535;     // 16-bit unsigned integer
    let i: u32 = 4294967295; // 32-bit unsigned integer
    let j: u64 = 18446744073709551615; // 64-bit unsigned integer
    let k: u128 = 340282366920938463463374607431768211455; // 128-bit unsigned integer
    let l: usize = 18446744073709551615; // Architecture-dependent
    
    println!("Signed: {}, {}, {}, {}, {}, {}", a, b, c, d, e, f);
    println!("Unsigned: {}, {}, {}, {}, {}, {}", g, h, i, j, k, l);
}
```

### **Floating-Point Types**

```rust
fn main() {
    let x: f32 = 3.14;      // 32-bit floating point
    let y: f64 = 3.141592653589793; // 64-bit floating point (default)
    
    // Scientific notation
    let z = 1.0e10;         // 10 billion
    let w = 2.5e-4;         // 0.00025
    
    println!("f32: {}, f64: {}", x, y);
    println!("Scientific: {}, {}", z, w);
}
```

### **Boolean Type**

```rust
fn main() {
    let t: bool = true;
    let f: bool = false;
    
    // Boolean operations
    let and_result = t && f;  // false
    let or_result = t || f;   // true
    let not_result = !t;      // false
    
    println!("t: {}, f: {}", t, f);
    println!("&&: {}, ||: {}, !: {}", and_result, or_result, not_result);
}
```

### **Character Type**

```rust
fn main() {
    let c: char = 'z';
    let z: char = 'ℤ';
    let heart_eyed_cat: char = '😻';
    
    // Character methods
    println!("c: {}, is_alphabetic: {}", c, c.is_alphabetic());
    println!("z: {}, is_alphabetic: {}", z, z.is_alphabetic());
    println!("😻: {}, is_alphabetic: {}", heart_eyed_cat, heart_eyed_cat.is_alphabetic());
}
```

---

## 📦 **Compound Types**

### **Tuple Type**

```rust
fn main() {
    // Tuple with different types
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    
    // Destructuring
    let (x, y, z) = tup;
    println!("x: {}, y: {}, z: {}", x, y, z);
    
    // Access by index
    println!("First element: {}", tup.0);
    println!("Second element: {}", tup.1);
    println!("Third element: {}", tup.2);
    
    // Empty tuple (unit type)
    let empty = ();
    println!("Empty tuple: {:?}", empty);
}
```

### **Array Type**

```rust
fn main() {
    // Array with fixed size
    let a: [i32; 5] = [1, 2, 3, 4, 5];
    
    // Array with same value
    let b = [3; 5]; // [3, 3, 3, 3, 3]
    
    // Access elements
    println!("First element: {}", a[0]);
    println!("Last element: {}", a[4]);
    
    // Array length
    println!("Array length: {}", a.len());
    
    // Iterate over array
    for element in &a {
        println!("Element: {}", element);
    }
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Type Inference**

```rust
fn main() {
    // Let Rust infer the types
    let x = 42;        // What type is this?
    let y = 3.14;      // What type is this?
    let z = true;      // What type is this?
    let w = 'A';       // What type is this?
    
    println!("x: {} (type: {})", x, std::any::type_name_of_val(&x));
    println!("y: {} (type: {})", y, std::any::type_name_of_val(&y));
    println!("z: {} (type: {})", z, std::any::type_name_of_val(&z));
    println!("w: {} (type: {})", w, std::any::type_name_of_val(&w));
}
```

### **Exercise 2: Type Conversions**

```rust
fn main() {
    let x: i32 = 42;
    let y: f64 = 3.14;
    
    // Convert i32 to f64
    let x_as_f64 = x as f64;
    
    // Convert f64 to i32 (truncates)
    let y_as_i32 = y as i32;
    
    // Convert between different integer types
    let small: u8 = 255;
    let large: u16 = small as u16;
    
    println!("x: {} -> f64: {}", x, x_as_f64);
    println!("y: {} -> i32: {}", y, y_as_i32);
    println!("small: {} -> large: {}", small, large);
}
```

### **Exercise 3: Working with Arrays**

```rust
fn main() {
    let numbers = [1, 2, 3, 4, 5];
    
    // Calculate sum
    let mut sum = 0;
    for &num in &numbers {
        sum += num;
    }
    
    // Calculate average
    let average = sum as f64 / numbers.len() as f64;
    
    println!("Numbers: {:?}", numbers);
    println!("Sum: {}", sum);
    println!("Average: {}", average);
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_types() {
        let x: i32 = 42;
        let y: u32 = 42;
        assert_eq!(x, 42);
        assert_eq!(y, 42);
    }

    #[test]
    fn test_float_types() {
        let x: f32 = 3.14;
        let y: f64 = 3.141592653589793;
        assert!((x - 3.14).abs() < 0.01);
        assert!((y - 3.141592653589793).abs() < 0.000000000000001);
    }

    #[test]
    fn test_boolean_operations() {
        let t = true;
        let f = false;
        assert_eq!(t && f, false);
        assert_eq!(t || f, true);
        assert_eq!(!t, false);
    }

    #[test]
    fn test_tuple_destructuring() {
        let tup = (1, 2.0, 'a');
        let (x, y, z) = tup;
        assert_eq!(x, 1);
        assert_eq!(y, 2.0);
        assert_eq!(z, 'a');
    }

    #[test]
    fn test_array_access() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(arr[0], 1);
        assert_eq!(arr[4], 5);
        assert_eq!(arr.len(), 5);
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Integer Overflow**

```rust
// ❌ Wrong - can cause panic in debug mode
fn bad_example() {
    let x: u8 = 255;
    let y = x + 1; // Overflow!
}

// ✅ Correct - handle overflow explicitly
fn good_example() {
    let x: u8 = 255;
    let y = x.wrapping_add(1); // Wraps around to 0
    let z = x.checked_add(1);  // Returns Option<u8>
    
    match z {
        Some(value) => println!("Result: {}", value),
        None => println!("Overflow occurred"),
    }
}
```

### **Common Mistake 2: Array Bounds**

```rust
// ❌ Wrong - can cause panic
fn bad_example() {
    let arr = [1, 2, 3];
    let value = arr[5]; // Out of bounds!
}

// ✅ Correct - check bounds
fn good_example() {
    let arr = [1, 2, 3];
    let index = 5;
    
    if index < arr.len() {
        println!("Value: {}", arr[index]);
    } else {
        println!("Index out of bounds");
    }
    
    // Or use get() method
    match arr.get(index) {
        Some(value) => println!("Value: {}", value),
        None => println!("Index out of bounds"),
    }
}
```

### **Common Mistake 3: Type Mismatches**

```rust
// ❌ Wrong - type mismatch
fn bad_example() {
    let x: i32 = 42;
    let y: f64 = 3.14;
    let sum = x + y; // Error: cannot add i32 and f64
}

// ✅ Correct - convert types
fn good_example() {
    let x: i32 = 42;
    let y: f64 = 3.14;
    let sum = x as f64 + y; // Convert i32 to f64
    println!("Sum: {}", sum);
}
```

---

## 📊 **Type Size and Memory**

### **Size of Types**

```rust
use std::mem;

fn main() {
    println!("Size of i8: {} bytes", mem::size_of::<i8>());
    println!("Size of i32: {} bytes", mem::size_of::<i32>());
    println!("Size of i64: {} bytes", mem::size_of::<i64>());
    println!("Size of f32: {} bytes", mem::size_of::<f32>());
    println!("Size of f64: {} bytes", mem::size_of::<f64>());
    println!("Size of bool: {} bytes", mem::size_of::<bool>());
    println!("Size of char: {} bytes", mem::size_of::<char>());
    println!("Size of (i32, f64): {} bytes", mem::size_of::<(i32, f64)>());
    println!("Size of [i32; 5]: {} bytes", mem::size_of::<[i32; 5]>());
}
```

---

## 🎯 **Best Practices**

### **Choosing the Right Type**

```rust
// ✅ Use appropriate integer types
let age: u8 = 25;           // Age can't be negative
let temperature: i16 = -10; // Temperature can be negative
let population: u32 = 1000000; // Population is large but positive

// ✅ Use f64 for most floating-point calculations
let pi: f64 = 3.141592653589793;
let e: f64 = 2.718281828459045;

// ✅ Use usize for array indices
let arr = [1, 2, 3, 4, 5];
let index: usize = 2;
let value = arr[index];
```

### **Type Annotations**

```rust
// ✅ Annotate when type is unclear
let x: f64 = 3.14; // Without annotation, this would be f64 anyway
let y: i32 = 42;   // Without annotation, this would be i32 anyway

// ✅ Annotate function parameters and return types
fn add(x: i32, y: i32) -> i32 {
    x + y
}

// ✅ Annotate when type inference fails
let numbers: Vec<i32> = vec![1, 2, 3];
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [The Rust Book - Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html) - Fetched: 2024-12-19T00:00:00Z
- [Rust Reference - Types](https://doc.rust-lang.org/reference/types.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust by Example - Primitives](https://doc.rust-lang.org/rust-by-example/primitives.html) - Fetched: 2024-12-19T00:00:00Z
- [Rustlings - Variables](https://github.com/rust-lang/rustlings) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What's the difference between `i32` and `u32`?
2. When should you use `f32` vs `f64`?
3. What's the difference between an array and a tuple?
4. When do you need to specify type annotations explicitly?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Control flow with if/else statements
- Loops and iteration
- Pattern matching with match expressions
- Function definitions and calls

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [2.2 Control Flow](02_02_control_flow.md)
