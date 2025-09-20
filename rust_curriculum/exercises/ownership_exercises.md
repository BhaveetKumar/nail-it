# Ownership Exercises

> **Module**: 3 - Ownership and Borrowing  
> **Difficulty**: Beginner to Intermediate  
> **Estimated Time**: 2-3 hours  
> **Prerequisites**: Lesson 3.1 (Ownership Basics)

---

## 🎯 **Exercise Overview**

These exercises will help you master Rust's ownership system through hands-on practice. Start with the beginner exercises and work your way up to more complex scenarios.

---

## 🟢 **Beginner Exercises**

### **Exercise 1: Understanding Moves**

**Task**: Fix the compilation errors in the following code by understanding move semantics.

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;
    
    // This line causes a compilation error. Why?
    println!("s1: {}", s1);
    println!("s2: {}", s2);
}
```

**Expected Learning**: Understanding that `String` values are moved, not copied.

**Solution**:
```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2
    
    // s1 is no longer valid after the move
    // println!("s1: {}", s1); // This would cause an error
    
    println!("s2: {}", s2); // s2 owns the string now
}
```

### **Exercise 2: Copy vs Move**

**Task**: Identify which variables can be used after assignment and which cannot.

```rust
fn main() {
    let x = 5;
    let y = x;
    println!("x: {}, y: {}", x, y); // Does this work?
    
    let s1 = String::from("hello");
    let s2 = s1;
    println!("s1: {}, s2: {}", s1, s2); // Does this work?
}
```

**Expected Learning**: Understanding the difference between `Copy` and `Move` semantics.

**Solution**:
```rust
fn main() {
    let x = 5;
    let y = x; // i32 implements Copy
    println!("x: {}, y: {}", x, y); // ✅ This works
    
    let s1 = String::from("hello");
    let s2 = s1; // String is moved
    // println!("s1: {}, s2: {}", s1, s2); // ❌ This doesn't work
    println!("s2: {}", s2); // ✅ Only s2 is valid
}
```

### **Exercise 3: Ownership in Functions**

**Task**: Fix the function to properly handle ownership.

```rust
fn take_ownership(s: String) {
    println!("{}", s);
}

fn main() {
    let s = String::from("hello");
    take_ownership(s);
    println!("{}", s); // This causes an error. Why?
}
```

**Expected Learning**: Understanding that function parameters take ownership.

**Solution**:
```rust
fn take_ownership(s: String) {
    println!("{}", s);
} // s goes out of scope and is dropped

fn main() {
    let s = String::from("hello");
    take_ownership(s); // s is moved into the function
    // println!("{}", s); // ❌ s is no longer valid
    
    // If you need to use s after the function call, you need to return it
}
```

---

## 🟡 **Intermediate Exercises**

### **Exercise 4: Returning Ownership**

**Task**: Modify the function to return ownership so the caller can use the value.

```rust
fn take_and_give_back(s: String) -> String {
    println!("{}", s);
    s // Return s to give ownership back
}

fn main() {
    let s1 = String::from("hello");
    let s2 = take_and_give_back(s1);
    println!("s2: {}", s2); // This should work now
}
```

**Expected Learning**: Understanding how to return ownership from functions.

### **Exercise 5: Multiple Ownership Transfers**

**Task**: Create a function that takes ownership of two strings and returns a new string.

```rust
fn combine_strings(s1: String, s2: String) -> String {
    format!("{} {}", s1, s2)
}

fn main() {
    let s1 = String::from("Hello");
    let s2 = String::from("World");
    let combined = combine_strings(s1, s2);
    println!("{}", combined);
    
    // Can you use s1 or s2 here? Why or why not?
}
```

**Expected Learning**: Understanding that ownership is transferred to function parameters.

### **Exercise 6: Clone vs Move**

**Task**: Use `clone()` to create copies when you need to keep the original value.

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone(); // Create a copy
    println!("s1: {}, s2: {}", s1, s2); // Both should work now
    
    // When would you use clone() vs move?
    // What are the performance implications?
}
```

**Expected Learning**: Understanding when to use `clone()` and its implications.

---

## 🟠 **Advanced Exercises**

### **Exercise 7: Ownership with Structs**

**Task**: Create a struct that owns a `String` and implement methods that transfer ownership.

```rust
#[derive(Debug)]
struct Person {
    name: String,
    age: u32,
}

impl Person {
    fn new(name: String, age: u32) -> Person {
        Person { name, age }
    }
    
    fn get_name(self) -> String {
        self.name // Move the name out of the struct
    }
    
    fn get_age(self) -> u32 {
        self.age
    }
}

fn main() {
    let person = Person::new(String::from("Alice"), 30);
    let name = person.get_name();
    println!("Name: {}", name);
    
    // Can you use person after get_name()? Why or why not?
    // println!("Age: {}", person.age); // What happens here?
}
```

**Expected Learning**: Understanding ownership with structs and method calls.

### **Exercise 8: Ownership with Vectors**

**Task**: Work with vectors and understand how ownership works with collections.

```rust
fn main() {
    let mut vec = vec![String::from("hello"), String::from("world")];
    
    // Move elements out of the vector
    let first = vec.remove(0); // This moves the first element
    println!("First element: {}", first);
    
    // What's left in the vector?
    println!("Remaining elements: {:?}", vec);
    
    // Can you access the first element after remove()? Why or why not?
}
```

**Expected Learning**: Understanding ownership with collections and how `remove()` works.

### **Exercise 9: Ownership with Tuples**

**Task**: Understand how ownership works with tuples containing owned values.

```rust
fn main() {
    let tuple = (String::from("hello"), 42);
    let (s, n) = tuple; // Destructuring moves the values
    
    println!("String: {}, Number: {}", s, n);
    
    // Can you use tuple after destructuring? Why or why not?
    // println!("Tuple: {:?}", tuple); // What happens here?
}
```

**Expected Learning**: Understanding ownership with tuples and destructuring.

---

## 🔴 **Expert Exercises**

### **Exercise 10: Complex Ownership Scenarios**

**Task**: Create a function that takes ownership of a vector of strings and returns a new vector with modified strings.

```rust
fn process_strings(strings: Vec<String>) -> Vec<String> {
    strings.into_iter()
        .map(|s| s.to_uppercase())
        .collect()
}

fn main() {
    let words = vec![
        String::from("hello"),
        String::from("world"),
        String::from("rust"),
    ];
    
    let processed = process_strings(words);
    println!("Processed: {:?}", processed);
    
    // Can you use words after the function call? Why or why not?
}
```

**Expected Learning**: Understanding ownership with iterators and complex data transformations.

### **Exercise 11: Ownership with Error Handling**

**Task**: Create a function that might fail and properly handle ownership in error cases.

```rust
use std::num::ParseIntError;

fn parse_and_double(s: String) -> Result<i32, ParseIntError> {
    let num: i32 = s.parse()?;
    Ok(num * 2)
}

fn main() {
    let input = String::from("42");
    match parse_and_double(input) {
        Ok(result) => println!("Result: {}", result),
        Err(e) => println!("Error: {}", e),
    }
    
    // Can you use input after the function call? Why or why not?
}
```

**Expected Learning**: Understanding ownership with error handling and the `?` operator.

### **Exercise 12: Ownership with Closures**

**Task**: Understand how ownership works with closures that capture values.

```rust
fn main() {
    let s = String::from("hello");
    
    let closure = move || {
        println!("{}", s);
    };
    
    closure();
    
    // Can you use s after creating the closure? Why or why not?
    // println!("{}", s); // What happens here?
}
```

**Expected Learning**: Understanding ownership with closures and the `move` keyword.

---

## 🧪 **Testing Your Solutions**

Create a test file to verify your solutions:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ownership_basics() {
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
        let result = take_and_give_back(s);
        assert_eq!(result, "test");
    }

    fn take_and_give_back(s: String) -> String {
        s
    }
}
```

Run tests with:
```bash
cargo test
```

---

## 🎯 **Learning Objectives Checklist**

After completing these exercises, you should be able to:

- [ ] Explain the difference between move and copy semantics
- [ ] Identify when values are moved vs copied
- [ ] Understand ownership transfer in function calls
- [ ] Use `clone()` appropriately when needed
- [ ] Handle ownership with structs and collections
- [ ] Understand ownership with error handling
- [ ] Apply ownership principles to complex scenarios

---

## 📚 **Further Reading**

### **Official Documentation**
- [The Rust Book - Understanding Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html) - Fetched: 2024-12-19T00:00:00Z
- [Rust by Example - Move Semantics](https://doc.rust-lang.org/rust-by-example/scope/move.html) - Fetched: 2024-12-19T00:00:00Z

### **Practice Resources**
- [Rustlings - Ownership](https://github.com/rust-lang/rustlings) - Fetched: 2024-12-19T00:00:00Z
- [Exercism Rust Track](https://exercism.org/tracks/rust) - Fetched: 2024-12-19T00:00:00Z

---

**Exercise Set Version**: 1.0  
**Rust Version**: 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z
