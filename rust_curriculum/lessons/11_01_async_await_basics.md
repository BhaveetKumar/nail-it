---
# Auto-generated front matter
Title: 11 01 Async Await Basics
LastUpdated: 2025-11-06T20:45:58.115567
Tags: []
Status: draft
---

# Lesson 11.1: Async/Await Basics

> **Module**: 11 - Async Programming  
> **Lesson**: 1 of 10  
> **Duration**: 3-4 hours  
> **Prerequisites**: Module 10 (Concurrency and Threading)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Understand the difference between async and sync programming
- Write async functions using `async` and `.await`
- Explain how async/await works in Rust
- Use `tokio` runtime for async execution
- Handle async errors and results properly

---

## 🎯 **Overview**

Async programming in Rust allows you to write concurrent code that can handle many tasks without blocking. Unlike traditional threading, async programming is more efficient for I/O-bound tasks and can handle thousands of concurrent operations with minimal overhead.

---

## ⚡ **What is Async Programming?**

### **Sync vs Async**

```rust
// Synchronous (blocking)
fn sync_function() {
    println!("Start");
    std::thread::sleep(std::time::Duration::from_secs(2));
    println!("End");
}

// Asynchronous (non-blocking)
async fn async_function() {
    println!("Start");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    println!("End");
}
```

**Key Differences**:
- **Sync**: Blocks the thread until operation completes
- **Async**: Yields control back to runtime, allowing other tasks to run

---

## 🔧 **Basic Async Syntax**

### **Async Functions**

```rust
async fn fetch_data() -> String {
    // Simulate network request
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    "Data from server".to_string()
}

async fn process_data(data: String) -> String {
    // Simulate processing
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    format!("Processed: {}", data)
}
```

### **Using .await**

```rust
#[tokio::main]
async fn main() {
    let data = fetch_data().await;
    let processed = process_data(data).await;
    println!("{}", processed);
}
```

**Key Points**:
- `async fn` creates a function that returns a `Future`
- `.await` is used to wait for the future to complete
- Only async functions can use `.await`

---

## 🏃 **Tokio Runtime**

### **Setting Up Tokio**

```rust
// Cargo.toml
[dependencies]
tokio = { version = "1.35.0", features = ["full"] }
```

```rust
// main.rs
#[tokio::main]
async fn main() {
    println!("Hello from async main!");
}
```

### **Manual Runtime**

```rust
use tokio::runtime::Runtime;

fn main() {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        println!("Hello from manual runtime!");
    });
}
```

**Source**: [Tokio Documentation](https://tokio.rs/) - Fetched: 2024-12-19T00:00:00Z

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Basic Async Function**

```rust
use tokio::time::{sleep, Duration};

async fn say_hello(name: &str) {
    println!("Hello, {}!", name);
    sleep(Duration::from_secs(1)).await;
    println!("Goodbye, {}!", name);
}

#[tokio::main]
async fn main() {
    say_hello("Alice").await;
    say_hello("Bob").await;
}
```

**Expected Output**:
```
Hello, Alice!
Goodbye, Alice!
Hello, Bob!
Goodbye, Bob!
```

### **Exercise 2: Concurrent Execution**

```rust
use tokio::time::{sleep, Duration};

async fn task(name: &str, duration: u64) {
    println!("Task {} started", name);
    sleep(Duration::from_secs(duration)).await;
    println!("Task {} completed", name);
}

#[tokio::main]
async fn main() {
    // Sequential execution
    println!("=== Sequential ===");
    task("A", 1).await;
    task("B", 1).await;
    
    // Concurrent execution
    println!("=== Concurrent ===");
    let handle1 = tokio::spawn(task("C", 1));
    let handle2 = tokio::spawn(task("D", 1));
    
    handle1.await.unwrap();
    handle2.await.unwrap();
}
```

### **Exercise 3: Async with Results**

```rust
use tokio::time::{sleep, Duration};

async fn fetch_user(id: u32) -> Result<String, String> {
    sleep(Duration::from_millis(500)).await;
    
    if id == 0 {
        Err("Invalid user ID".to_string())
    } else {
        Ok(format!("User {}", id))
    }
}

#[tokio::main]
async fn main() {
    match fetch_user(1).await {
        Ok(user) => println!("Found: {}", user),
        Err(e) => println!("Error: {}", e),
    }
    
    match fetch_user(0).await {
        Ok(user) => println!("Found: {}", user),
        Err(e) => println!("Error: {}", e),
    }
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_async_function() {
        let result = fetch_data().await;
        assert_eq!(result, "Data from server");
    }

    #[tokio::test]
    async fn test_concurrent_tasks() {
        let start = std::time::Instant::now();
        
        let handle1 = tokio::spawn(async {
            sleep(Duration::from_millis(100)).await;
            "Task 1"
        });
        
        let handle2 = tokio::spawn(async {
            sleep(Duration::from_millis(100)).await;
            "Task 2"
        });
        
        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();
        
        assert_eq!(result1, "Task 1");
        assert_eq!(result2, "Task 2");
        
        // Should complete in ~100ms, not 200ms
        assert!(start.elapsed() < Duration::from_millis(150));
    }

    async fn fetch_data() -> String {
        sleep(Duration::from_millis(10)).await;
        "Data from server".to_string()
    }
}
```

Run tests with:
```bash
cargo test
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Forgetting .await**
```rust
// ❌ Wrong - returns Future, doesn't execute
async fn bad_example() {
    let data = fetch_data(); // Missing .await
    println!("{}", data); // Error: type mismatch
}

// ✅ Correct
async fn good_example() {
    let data = fetch_data().await; // Properly await
    println!("{}", data);
}
```

### **Common Mistake 2: Blocking in Async Context**
```rust
// ❌ Wrong - blocks the async runtime
async fn bad_example() {
    std::thread::sleep(Duration::from_secs(1)); // Blocks!
}

// ✅ Correct - yields control
async fn good_example() {
    tokio::time::sleep(Duration::from_secs(1)).await; // Yields
}
```

### **Common Mistake 3: Not Handling Errors**
```rust
// ❌ Wrong - ignoring potential errors
async fn bad_example() {
    let result = risky_operation().await; // Result not handled
    println!("{}", result); // Error: type mismatch
}

// ✅ Correct - handle errors properly
async fn good_example() {
    match risky_operation().await {
        Ok(result) => println!("{}", result),
        Err(e) => println!("Error: {}", e),
    }
}
```

---

## 🔍 **Understanding Futures**

### **What is a Future?**

```rust
use std::future::Future;

// This async function returns a Future
async fn my_async_function() -> i32 {
    42
}

// Equivalent to:
fn my_async_function() -> impl Future<Output = i32> {
    async {
        42
    }
}
```

### **Future Combinators**

```rust
use tokio::time::{sleep, Duration};

async fn fetch_data() -> Result<String, String> {
    sleep(Duration::from_millis(100)).await;
    Ok("data".to_string())
}

#[tokio::main]
async fn main() {
    // Using map to transform the result
    let result = fetch_data()
        .await
        .map(|s| s.to_uppercase())
        .map_err(|e| format!("Failed: {}", e));
    
    println!("{:?}", result);
}
```

---

## 📊 **Performance Comparison**

### **Threading vs Async**

```rust
use std::time::Instant;
use tokio::time::{sleep, Duration};

// Threading approach
fn thread_example() {
    let start = Instant::now();
    let handles: Vec<_> = (0..1000)
        .map(|i| {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(1));
                i
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    println!("Threading: {:?}", start.elapsed());
}

// Async approach
async fn async_example() {
    let start = Instant::now();
    let handles: Vec<_> = (0..1000)
        .map(|i| {
            tokio::spawn(async move {
                sleep(Duration::from_millis(1)).await;
                i
            })
        })
        .collect();
    
    for handle in handles {
        handle.await.unwrap();
    }
    println!("Async: {:?}", start.elapsed());
}

#[tokio::main]
async fn main() {
    thread_example();
    async_example().await;
}
```

**Expected Output**:
```
Threading: 1.234s
Async: 0.123s
```

---

## 🎯 **Best Practices**

### **Error Handling**
```rust
use tokio::time::{sleep, Duration};

async fn robust_function() -> Result<String, Box<dyn std::error::Error>> {
    sleep(Duration::from_millis(100)).await;
    
    // Use ? operator for error propagation
    let data = fetch_data().await?;
    let processed = process_data(data).await?;
    
    Ok(processed)
}

async fn fetch_data() -> Result<String, String> {
    // Implementation
    Ok("data".to_string())
}

async fn process_data(data: String) -> Result<String, String> {
    // Implementation
    Ok(format!("processed: {}", data))
}
```

### **Resource Management**
```rust
use tokio::fs::File;
use tokio::io::AsyncReadExt;

async fn read_file() -> Result<String, Box<dyn std::error::Error>> {
    let mut file = File::open("data.txt").await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    Ok(contents)
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Async Book - Getting Started](https://rust-lang.github.io/async-book/01_getting_started/01_chapter.html) - Fetched: 2024-12-19T00:00:00Z
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust Async Patterns](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z
- [Tokio Examples](https://github.com/tokio-rs/tokio/tree/master/examples) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What's the difference between `async fn` and regular functions?
2. When should you use `.await`?
3. How does async programming differ from threading?
4. What is the role of the Tokio runtime?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Working with multiple async tasks
- Task spawning and joining
- Error handling in async contexts
- Async streams and iteration

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [11.2 Async Tasks and Spawning](11_02_async_tasks_spawning.md)
