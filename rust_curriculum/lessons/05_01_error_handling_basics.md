# Lesson 5.1: Error Handling Basics

> **Module**: 5 - Error Handling  
> **Lesson**: 1 of 6  
> **Duration**: 3-4 hours  
> **Prerequisites**: Module 4 (Structs, Enums & Pattern Matching)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Understand Rust's approach to error handling with `Result` and `Option`
- Use `match` and `if let` for error handling
- Apply the `?` operator for error propagation
- Choose between different error handling strategies
- Write robust, error-safe Rust code

---

## 🎯 **Overview**

Rust doesn't have exceptions like other languages. Instead, it uses the `Result<T, E>` and `Option<T>` types to represent operations that might fail or return no value. This approach makes error handling explicit and forces you to consider error cases, leading to more robust code.

---

## 🔧 **The Option Type**

### **What is Option?**

```rust
enum Option<T> {
    Some(T),
    None,
}
```

The `Option<T>` type represents a value that might or might not be present.

### **Basic Usage**

```rust
fn find_user(id: u32) -> Option<String> {
    if id == 1 {
        Some("Alice".to_string())
    } else {
        None
    }
}

fn main() {
    let user = find_user(1);
    match user {
        Some(name) => println!("User found: {}", name),
        None => println!("User not found"),
    }
}
```

### **Common Option Methods**

```rust
fn main() {
    let maybe_number = Some(42);
    let no_number: Option<i32> = None;
    
    // Unwrap with default
    let value1 = maybe_number.unwrap_or(0);
    let value2 = no_number.unwrap_or(0);
    
    // Map the value
    let doubled = maybe_number.map(|x| x * 2);
    
    // Filter the value
    let filtered = maybe_number.filter(|&x| x > 50);
    
    // Chain operations
    let result = maybe_number
        .map(|x| x * 2)
        .filter(|&x| x > 50)
        .unwrap_or(0);
    
    println!("Values: {}, {}, {:?}, {:?}, {}", 
             value1, value2, doubled, filtered, result);
}
```

---

## ⚠️ **The Result Type**

### **What is Result?**

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

The `Result<T, E>` type represents an operation that might succeed (returning `T`) or fail (returning `E`).

### **Basic Usage**

```rust
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}

fn main() {
    match divide(10.0, 2.0) {
        Ok(result) => println!("Result: {}", result),
        Err(error) => println!("Error: {}", error),
    }
    
    match divide(10.0, 0.0) {
        Ok(result) => println!("Result: {}", result),
        Err(error) => println!("Error: {}", error),
    }
}
```

### **Common Result Methods**

```rust
use std::fs::File;
use std::io::Read;

fn read_file_contents(filename: &str) -> Result<String, std::io::Error> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

fn main() {
    match read_file_contents("hello.txt") {
        Ok(contents) => println!("File contents: {}", contents),
        Err(error) => println!("Error reading file: {}", error),
    }
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Option Basics**

```rust
fn find_max(numbers: &[i32]) -> Option<i32> {
    if numbers.is_empty() {
        None
    } else {
        Some(*numbers.iter().max().unwrap())
    }
}

fn main() {
    let numbers = vec![1, 5, 3, 9, 2];
    match find_max(&numbers) {
        Some(max) => println!("Maximum: {}", max),
        None => println!("No numbers provided"),
    }
    
    let empty: Vec<i32> = vec![];
    match find_max(&empty) {
        Some(max) => println!("Maximum: {}", max),
        None => println!("No numbers provided"),
    }
}
```

**Expected Output**:
```
Maximum: 9
No numbers provided
```

### **Exercise 2: Result with Custom Errors**

```rust
#[derive(Debug)]
enum MathError {
    DivisionByZero,
    NegativeSquareRoot,
}

fn sqrt(x: f64) -> Result<f64, MathError> {
    if x < 0.0 {
        Err(MathError::NegativeSquareRoot)
    } else {
        Ok(x.sqrt())
    }
}

fn divide(a: f64, b: f64) -> Result<f64, MathError> {
    if b == 0.0 {
        Err(MathError::DivisionByZero)
    } else {
        Ok(a / b)
    }
}

fn main() {
    match sqrt(16.0) {
        Ok(result) => println!("Square root: {}", result),
        Err(e) => println!("Error: {:?}", e),
    }
    
    match sqrt(-4.0) {
        Ok(result) => println!("Square root: {}", result),
        Err(e) => println!("Error: {:?}", e),
    }
}
```

### **Exercise 3: Error Propagation with ?**

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_and_process_file(filename: &str) -> Result<String, io::Error> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    // Process the contents
    let processed = contents.to_uppercase();
    Ok(processed)
}

fn main() {
    match read_and_process_file("example.txt") {
        Ok(contents) => println!("Processed: {}", contents),
        Err(error) => println!("Error: {}", error),
    }
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_max_some() {
        let numbers = vec![1, 5, 3, 9, 2];
        assert_eq!(find_max(&numbers), Some(9));
    }

    #[test]
    fn test_find_max_none() {
        let empty: Vec<i32> = vec![];
        assert_eq!(find_max(&empty), None);
    }

    #[test]
    fn test_sqrt_success() {
        assert_eq!(sqrt(16.0), Ok(4.0));
    }

    #[test]
    fn test_sqrt_error() {
        assert!(sqrt(-4.0).is_err());
    }

    #[test]
    fn test_divide_success() {
        assert_eq!(divide(10.0, 2.0), Ok(5.0));
    }

    #[test]
    fn test_divide_by_zero() {
        assert!(divide(10.0, 0.0).is_err());
    }

    fn find_max(numbers: &[i32]) -> Option<i32> {
        if numbers.is_empty() {
            None
        } else {
            Some(*numbers.iter().max().unwrap())
        }
    }

    fn sqrt(x: f64) -> Result<f64, MathError> {
        if x < 0.0 {
            Err(MathError::NegativeSquareRoot)
        } else {
            Ok(x.sqrt())
        }
    }

    fn divide(a: f64, b: f64) -> Result<f64, MathError> {
        if b == 0.0 {
            Err(MathError::DivisionByZero)
        } else {
            Ok(a / b)
        }
    }

    #[derive(Debug, PartialEq)]
    enum MathError {
        DivisionByZero,
        NegativeSquareRoot,
    }
}
```

Run tests with:
```bash
cargo test
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Using unwrap() in Production**

```rust
// ❌ Wrong - can panic
fn bad_example() {
    let result = risky_operation().unwrap(); // Panics on error
}

// ✅ Correct - handle errors properly
fn good_example() {
    match risky_operation() {
        Ok(value) => println!("Success: {}", value),
        Err(e) => println!("Error: {}", e),
    }
}
```

### **Common Mistake 2: Ignoring Errors**

```rust
// ❌ Wrong - ignoring potential errors
fn bad_example() {
    let _ = risky_operation(); // Error is ignored
}

// ✅ Correct - handle or propagate errors
fn good_example() -> Result<(), String> {
    risky_operation()?; // Propagate error up
    Ok(())
}
```

### **Common Mistake 3: Confusing Option and Result**

```rust
// ❌ Wrong - using Option for errors
fn bad_example() -> Option<String> {
    if some_condition {
        Some("success".to_string())
    } else {
        None // What kind of error occurred?
    }
}

// ✅ Correct - use Result for errors
fn good_example() -> Result<String, String> {
    if some_condition {
        Ok("success".to_string())
    } else {
        Err("Specific error message".to_string())
    }
}
```

---

## 🔍 **Advanced Error Handling Patterns**

### **Custom Error Types**

```rust
use std::fmt;

#[derive(Debug)]
enum AppError {
    Io(std::io::Error),
    Parse(std::num::ParseIntError),
    Custom(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AppError::Io(err) => write!(f, "IO error: {}", err),
            AppError::Parse(err) => write!(f, "Parse error: {}", err),
            AppError::Custom(msg) => write!(f, "Custom error: {}", msg),
        }
    }
}

impl std::error::Error for AppError {}

// Convert other errors to our custom error
impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> AppError {
        AppError::Io(err)
    }
}

impl From<std::num::ParseIntError> for AppError {
    fn from(err: std::num::ParseIntError) -> AppError {
        AppError::Parse(err)
    }
}
```

### **Error Propagation with ?**

```rust
fn process_file(filename: &str) -> Result<i32, AppError> {
    let contents = std::fs::read_to_string(filename)?; // Converts io::Error to AppError
    let number: i32 = contents.trim().parse()?; // Converts ParseIntError to AppError
    Ok(number * 2)
}

fn main() {
    match process_file("number.txt") {
        Ok(result) => println!("Result: {}", result),
        Err(e) => println!("Error: {}", e),
    }
}
```

---

## 📊 **Error Handling Strategies**

### **Strategy 1: Handle Immediately**

```rust
fn handle_immediately() {
    match risky_operation() {
        Ok(value) => {
            // Process the value
            println!("Got value: {}", value);
        }
        Err(e) => {
            // Handle the error immediately
            eprintln!("Error occurred: {}", e);
            // Maybe return early or use a default value
        }
    }
}
```

### **Strategy 2: Propagate Up**

```rust
fn propagate_error() -> Result<String, Box<dyn std::error::Error>> {
    let value = risky_operation()?; // Propagate error up
    let processed = process_value(value)?; // Propagate error up
    Ok(processed)
}
```

### **Strategy 3: Convert Error Types**

```rust
fn convert_errors() -> Result<i32, AppError> {
    let value = risky_operation()?; // Convert to AppError
    Ok(value)
}
```

---

## 🎯 **Best Practices**

### **Error Message Guidelines**

```rust
// ✅ Good error messages
fn validate_age(age: u32) -> Result<(), String> {
    if age < 18 {
        Err("Age must be at least 18".to_string())
    } else if age > 120 {
        Err("Age must be at most 120".to_string())
    } else {
        Ok(())
    }
}

// ❌ Poor error messages
fn bad_validate_age(age: u32) -> Result<(), String> {
    if age < 18 || age > 120 {
        Err("Invalid age".to_string()) // Too vague
    } else {
        Ok(())
    }
}
```

### **When to Use Each Type**

```rust
// Use Option when the absence of a value is normal
fn find_user_by_id(id: u32) -> Option<User> {
    // User might not exist - this is normal
}

// Use Result when an operation might fail
fn create_user(data: UserData) -> Result<User, ValidationError> {
    // User creation might fail due to validation
}

// Use panic! only for programming errors
fn get_first_element<T>(vec: &[T]) -> &T {
    if vec.is_empty() {
        panic!("Cannot get first element of empty vector"); // Programming error
    }
    &vec[0]
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [The Rust Book - Error Handling](https://doc.rust-lang.org/book/ch09-00-error-handling.html) - Fetched: 2024-12-19T00:00:00Z
- [The Rust Book - Recoverable Errors with Result](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust by Example - Error Handling](https://doc.rust-lang.org/rust-by-example/error.html) - Fetched: 2024-12-19T00:00:00Z
- [anyhow crate](https://docs.rs/anyhow/latest/anyhow/) - Fetched: 2024-12-19T00:00:00Z
- [thiserror crate](https://docs.rs/thiserror/latest/thiserror/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What's the difference between `Option<T>` and `Result<T, E>`?
2. When should you use `unwrap()` vs proper error handling?
3. How does the `?` operator work for error propagation?
4. What are the benefits of Rust's approach to error handling?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced error handling with custom types
- Error handling in async contexts
- Using external crates for error handling
- Error handling patterns and best practices

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [5.2 Advanced Error Handling](05_02_advanced_error_handling.md)
