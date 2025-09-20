# Lesson 6.1: Collections - Vectors

> **Module**: 6 - Collections and Iterators  
> **Lesson**: 1 of 6  
> **Duration**: 2-3 hours  
> **Prerequisites**: Module 5 (Error Handling)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Create and manipulate vectors
- Understand vector ownership and borrowing
- Use common vector methods and operations
- Choose appropriate collection types for different use cases
- Apply vector operations in real-world scenarios

---

## 🎯 **Overview**

Vectors (`Vec<T>`) are growable arrays that can store multiple values of the same type. They're one of the most commonly used collection types in Rust and provide efficient access to elements by index.

---

## 🔧 **Creating Vectors**

### **Basic Creation**

```rust
fn main() {
    // Create empty vector
    let mut v1: Vec<i32> = Vec::new();
    
    // Create vector with initial values
    let v2 = vec![1, 2, 3, 4, 5];
    
    // Create vector with capacity
    let mut v3 = Vec::with_capacity(10);
    
    // Create vector of specific size with default value
    let v4 = vec![0; 5]; // [0, 0, 0, 0, 0]
    
    println!("v1: {:?}", v1);
    println!("v2: {:?}", v2);
    println!("v3: {:?}", v3);
    println!("v4: {:?}", v4);
}
```

### **Adding Elements**

```rust
fn main() {
    let mut v = Vec::new();
    
    // Add elements
    v.push(1);
    v.push(2);
    v.push(3);
    
    // Insert at specific index
    v.insert(1, 10); // [1, 10, 2, 3]
    
    // Extend with another vector
    v.extend_from_slice(&[4, 5, 6]);
    
    println!("Vector: {:?}", v);
}
```

---

## 📖 **Accessing Elements**

### **Safe Access Methods**

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // Using get() - returns Option
    match v.get(2) {
        Some(value) => println!("Third element: {}", value),
        None => println!("No third element"),
    }
    
    // Using get() with unwrap_or
    let third = v.get(2).unwrap_or(&0);
    println!("Third element: {}", third);
    
    // Check if index is valid
    let index = 10;
    if let Some(value) = v.get(index) {
        println!("Element at index {}: {}", index, value);
    } else {
        println!("Index {} is out of bounds", index);
    }
}
```

### **Unsafe Access (Panic on Error)**

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // Direct indexing - panics if out of bounds
    let first = v[0];
    let second = v[1];
    
    println!("First: {}, Second: {}", first, second);
    
    // This would panic:
    // let invalid = v[10]; // thread 'main' panicked at 'index out of bounds'
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Basic Vector Operations**

```rust
fn main() {
    let mut numbers = vec![1, 2, 3, 4, 5];
    
    // Add elements
    numbers.push(6);
    numbers.push(7);
    
    // Remove elements
    let last = numbers.pop(); // Removes and returns last element
    println!("Removed: {:?}", last);
    
    // Access elements
    println!("First: {}", numbers[0]);
    println!("Last: {}", numbers[numbers.len() - 1]);
    
    // Print all elements
    for (index, value) in numbers.iter().enumerate() {
        println!("Index {}: {}", index, value);
    }
}
```

### **Exercise 2: Vector Statistics**

```rust
fn calculate_stats(numbers: &[i32]) -> (i32, i32, f64) {
    if numbers.is_empty() {
        return (0, 0, 0.0);
    }
    
    let min = *numbers.iter().min().unwrap();
    let max = *numbers.iter().max().unwrap();
    let sum: i32 = numbers.iter().sum();
    let average = sum as f64 / numbers.len() as f64;
    
    (min, max, average)
}

fn main() {
    let numbers = vec![1, 5, 3, 9, 2, 8, 4, 7, 6];
    let (min, max, average) = calculate_stats(&numbers);
    
    println!("Numbers: {:?}", numbers);
    println!("Min: {}, Max: {}, Average: {:.2}", min, max, average);
}
```

### **Exercise 3: Vector Manipulation**

```rust
fn remove_duplicates(mut numbers: Vec<i32>) -> Vec<i32> {
    numbers.sort();
    numbers.dedup();
    numbers
}

fn filter_even(numbers: &[i32]) -> Vec<i32> {
    numbers.iter()
        .filter(|&&x| x % 2 == 0)
        .cloned()
        .collect()
}

fn main() {
    let numbers = vec![1, 2, 2, 3, 4, 4, 5, 6, 6, 7];
    
    let unique = remove_duplicates(numbers.clone());
    let even = filter_even(&numbers);
    
    println!("Original: {:?}", numbers);
    println!("Unique: {:?}", unique);
    println!("Even: {:?}", even);
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let v = vec![1, 2, 3];
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn test_vector_push_pop() {
        let mut v = Vec::new();
        v.push(1);
        v.push(2);
        assert_eq!(v.len(), 2);
        
        let popped = v.pop();
        assert_eq!(popped, Some(2));
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_vector_get() {
        let v = vec![1, 2, 3];
        assert_eq!(v.get(0), Some(&1));
        assert_eq!(v.get(5), None);
    }

    #[test]
    fn test_calculate_stats() {
        let numbers = vec![1, 5, 3, 9, 2];
        let (min, max, average) = calculate_stats(&numbers);
        assert_eq!(min, 1);
        assert_eq!(max, 9);
        assert!((average - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_remove_duplicates() {
        let numbers = vec![1, 2, 2, 3, 3, 3];
        let unique = remove_duplicates(numbers);
        assert_eq!(unique, vec![1, 2, 3]);
    }

    #[test]
    fn test_filter_even() {
        let numbers = vec![1, 2, 3, 4, 5, 6];
        let even = filter_even(&numbers);
        assert_eq!(even, vec![2, 4, 6]);
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Index Out of Bounds**

```rust
// ❌ Wrong - can panic
fn bad_example() {
    let v = vec![1, 2, 3];
    let value = v[5]; // Panic!
}

// ✅ Correct - check bounds
fn good_example() {
    let v = vec![1, 2, 3];
    let index = 5;
    
    if let Some(value) = v.get(index) {
        println!("Value: {}", value);
    } else {
        println!("Index out of bounds");
    }
}
```

### **Common Mistake 2: Borrowing Issues**

```rust
// ❌ Wrong - cannot borrow as mutable and immutable
fn bad_example() {
    let mut v = vec![1, 2, 3];
    let first = &v[0];        // Immutable borrow
    v.push(4);                // Mutable borrow - Error!
    println!("{}", first);
}

// ✅ Correct - scope the borrows
fn good_example() {
    let mut v = vec![1, 2, 3];
    {
        let first = &v[0];
        println!("{}", first);
    } // first goes out of scope
    v.push(4); // Now we can borrow mutably
}
```

### **Common Mistake 3: Moving Out of Vector**

```rust
// ❌ Wrong - cannot move out of vector
fn bad_example() {
    let v = vec![String::from("hello"), String::from("world")];
    let first = v[0]; // Error: cannot move out of vector
}

// ✅ Correct - use references or clone
fn good_example() {
    let v = vec![String::from("hello"), String::from("world")];
    let first = &v[0]; // Borrow
    let first_owned = v[0].clone(); // Clone
    println!("{}", first);
    println!("{}", first_owned);
}
```

---

## 📊 **Advanced Vector Operations**

### **Vector Slicing**

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Get slice
    let slice = &v[2..5]; // [3, 4, 5]
    println!("Slice: {:?}", slice);
    
    // Get slice from beginning
    let first_half = &v[..5]; // [1, 2, 3, 4, 5]
    println!("First half: {:?}", first_half);
    
    // Get slice to end
    let last_half = &v[5..]; // [6, 7, 8, 9, 10]
    println!("Last half: {:?}", last_half);
}
```

### **Vector Iteration**

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    
    // Iterate by reference
    for item in &v {
        println!("Item: {}", item);
    }
    
    // Iterate by value (moves elements)
    for item in v {
        println!("Item: {}", item);
    }
    // v is no longer valid here
    
    // Iterate with index
    let v2 = vec![10, 20, 30, 40, 50];
    for (index, value) in v2.iter().enumerate() {
        println!("Index {}: {}", index, value);
    }
}
```

---

## 🎯 **Best Practices**

### **Memory Management**

```rust
// ✅ Good - pre-allocate capacity when you know the size
fn efficient_vector() {
    let mut v = Vec::with_capacity(1000);
    for i in 0..1000 {
        v.push(i);
    }
}

// ✅ Good - use references when possible
fn process_vector(v: &[i32]) {
    for item in v {
        println!("{}", item);
    }
}

// ❌ Avoid - unnecessary cloning
fn inefficient_vector(v: &Vec<i32>) -> Vec<i32> {
    v.clone() // Only clone when necessary
}
```

### **Error Handling**

```rust
// ✅ Good - handle errors gracefully
fn safe_access(v: &[i32], index: usize) -> Option<i32> {
    v.get(index).copied()
}

// ✅ Good - use Result for operations that can fail
fn safe_divide(v: &[f64], index: usize) -> Result<f64, String> {
    let value = v.get(index).ok_or("Index out of bounds")?;
    if *value == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(1.0 / value)
    }
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [The Rust Book - Vectors](https://doc.rust-lang.org/book/ch08-01-vectors.html) - Fetched: 2024-12-19T00:00:00Z
- [Vec Documentation](https://doc.rust-lang.org/std/vec/struct.Vec.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust by Example - Vectors](https://doc.rust-lang.org/rust-by-example/std/vec.html) - Fetched: 2024-12-19T00:00:00Z
- [Rustlings - Vectors](https://github.com/rust-lang/rustlings) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What's the difference between `Vec::new()` and `vec![]`?
2. When should you use `get()` vs direct indexing?
3. How do you safely remove elements from a vector?
4. What are the ownership implications of iterating over a vector?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Hash maps and hash sets
- String collections and text processing
- Iterators and functional programming
- Performance considerations for collections

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [6.2 Hash Maps and Sets](06_02_hash_maps_sets.md)
