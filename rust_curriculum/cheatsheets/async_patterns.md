---
# Auto-generated front matter
Title: Async Patterns
LastUpdated: 2025-11-06T20:45:58.130736
Tags: []
Status: draft
---

# Async Patterns Cheat Sheet

> **Quick Reference for Async Programming in Rust**  
> **Rust Version**: 1.75.0  
> **Last Updated**: 2024-12-19T00:00:00Z

---

## 🚀 **Basic Async Syntax**

### **Async Functions**
```rust
async fn fetch_data() -> Result<String, Box<dyn std::error::Error>> {
    // Async operation
    Ok("data".to_string())
}

async fn process_data(data: String) -> String {
    // Process data
    data.to_uppercase()
}
```

### **Using .await**
```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = fetch_data().await?;
    let processed = process_data(data).await;
    println!("{}", processed);
    Ok(())
}
```

---

## 🔧 **Tokio Runtime**

### **Basic Setup**
```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    println!("Hello from async main!");
}

// Manual runtime
use tokio::runtime::Runtime;

fn main() {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        println!("Hello from manual runtime!");
    });
}
```

### **Runtime Configuration**
```rust
use tokio::runtime::Runtime;

let rt = Runtime::new()
    .unwrap()
    .with_worker_threads(4)
    .with_thread_name("my-worker")
    .with_thread_stack_size(3 * 1024 * 1024);
```

---

## 🎯 **Task Spawning**

### **Basic Spawning**
```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let handle = tokio::spawn(async {
        sleep(Duration::from_secs(1)).await;
        "Task completed"
    });
    
    let result = handle.await.unwrap();
    println!("{}", result);
}
```

### **Spawning with Data**
```rust
#[tokio::main]
async fn main() {
    let data = vec![1, 2, 3, 4, 5];
    
    let handle = tokio::spawn(async move {
        data.iter().sum::<i32>()
    });
    
    let sum = handle.await.unwrap();
    println!("Sum: {}", sum);
}
```

---

## 🔄 **Concurrency Patterns**

### **Join Multiple Tasks**
```rust
use tokio::time::{sleep, Duration};

async fn task1() -> i32 {
    sleep(Duration::from_millis(100)).await;
    1
}

async fn task2() -> i32 {
    sleep(Duration::from_millis(200)).await;
    2
}

#[tokio::main]
async fn main() {
    let (result1, result2) = tokio::join!(task1(), task2());
    println!("Results: {}, {}", result1, result2);
}
```

### **Select from Multiple Tasks**
```rust
use tokio::time::{sleep, Duration, timeout};
use tokio::select;

async fn slow_task() -> &'static str {
    sleep(Duration::from_secs(5)).await;
    "Slow task completed"
}

async fn fast_task() -> &'static str {
    sleep(Duration::from_millis(100)).await;
    "Fast task completed"
}

#[tokio::main]
async fn main() {
    select! {
        result = slow_task() => println!("Slow: {}", result),
        result = fast_task() => println!("Fast: {}", result),
        _ = sleep(Duration::from_secs(1)) => println!("Timeout"),
    }
}
```

---

## 🌐 **HTTP Requests**

### **Basic HTTP Client**
```rust
use reqwest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let response = client
        .get("https://httpbin.org/get")
        .send()
        .await?;
    
    let body = response.text().await?;
    println!("{}", body);
    Ok(())
}
```

### **Concurrent HTTP Requests**
```rust
use reqwest;
use tokio::time::{sleep, Duration};

async fn fetch_url(url: &str) -> Result<String, reqwest::Error> {
    let client = reqwest::Client::new();
    let response = client.get(url).send().await?;
    response.text().await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let urls = vec![
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/3",
    ];
    
    let handles: Vec<_> = urls
        .into_iter()
        .map(|url| tokio::spawn(fetch_url(url)))
        .collect();
    
    for handle in handles {
        match handle.await? {
            Ok(response) => println!("Response: {}", response),
            Err(e) => println!("Error: {}", e),
        }
    }
    
    Ok(())
}
```

---

## 📊 **Channels and Communication**

### **Basic Channel**
```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(32);
    
    // Spawn sender
    tokio::spawn(async move {
        for i in 0..10 {
            tx.send(i).await.unwrap();
        }
    });
    
    // Receive messages
    while let Some(message) = rx.recv().await {
        println!("Received: {}", message);
    }
}
```

### **Broadcast Channel**
```rust
use tokio::sync::broadcast;

#[tokio::main]
async fn main() {
    let (tx, mut rx1) = broadcast::channel(16);
    let mut rx2 = tx.subscribe();
    
    // Send message
    tx.send("Hello").unwrap();
    
    // Receive from both receivers
    println!("rx1: {}", rx1.recv().await.unwrap());
    println!("rx2: {}", rx2.recv().await.unwrap());
}
```

---

## 🔒 **Synchronization**

### **Mutex**
```rust
use tokio::sync::Mutex;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let data = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for i in 0..10 {
        let data = Arc::clone(&data);
        let handle = tokio::spawn(async move {
            let mut num = data.lock().await;
            *num += i;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    println!("Final value: {}", *data.lock().await);
}
```

### **RwLock**
```rust
use tokio::sync::RwLock;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let data = Arc::new(RwLock::new(0));
    
    // Multiple readers
    let readers: Vec<_> = (0..5)
        .map(|_| {
            let data = Arc::clone(&data);
            tokio::spawn(async move {
                let num = data.read().await;
                println!("Reader: {}", *num);
            })
        })
        .collect();
    
    // One writer
    let writer = {
        let data = Arc::clone(&data);
        tokio::spawn(async move {
            let mut num = data.write().await;
            *num += 1;
        })
    };
    
    // Wait for all tasks
    for handle in readers {
        handle.await.unwrap();
    }
    writer.await.unwrap();
}
```

---

## ⏰ **Timers and Delays**

### **Basic Delays**
```rust
use tokio::time::{sleep, Duration, Instant};

#[tokio::main]
async fn main() {
    // Sleep for 1 second
    sleep(Duration::from_secs(1)).await;
    
    // Sleep until specific time
    let instant = Instant::now() + Duration::from_secs(5);
    tokio::time::sleep_until(instant).await;
}
```

### **Interval Timer**
```rust
use tokio::time::{interval, Duration};

#[tokio::main]
async fn main() {
    let mut interval = interval(Duration::from_secs(1));
    
    for i in 0..5 {
        interval.tick().await;
        println!("Tick {}", i);
    }
}
```

### **Timeout**
```rust
use tokio::time::{timeout, Duration};

async fn slow_operation() -> &'static str {
    tokio::time::sleep(Duration::from_secs(5)).await;
    "Operation completed"
}

#[tokio::main]
async fn main() {
    match timeout(Duration::from_secs(2), slow_operation()).await {
        Ok(result) => println!("{}", result),
        Err(_) => println!("Operation timed out"),
    }
}
```

---

## 🎯 **Error Handling**

### **Basic Error Handling**
```rust
use tokio::time::{sleep, Duration};

async fn risky_operation() -> Result<String, String> {
    sleep(Duration::from_millis(100)).await;
    
    if rand::random::<bool>() {
        Ok("Success".to_string())
    } else {
        Err("Operation failed".to_string())
    }
}

#[tokio::main]
async fn main() {
    match risky_operation().await {
        Ok(result) => println!("Success: {}", result),
        Err(e) => println!("Error: {}", e),
    }
}
```

### **Error Propagation**
```rust
use tokio::time::{sleep, Duration};

async fn operation1() -> Result<String, Box<dyn std::error::Error>> {
    sleep(Duration::from_millis(100)).await;
    Ok("Operation 1".to_string())
}

async fn operation2() -> Result<String, Box<dyn std::error::Error>> {
    sleep(Duration::from_millis(200)).await;
    Ok("Operation 2".to_string())
}

async fn combined_operation() -> Result<String, Box<dyn std::error::Error>> {
    let result1 = operation1().await?;
    let result2 = operation2().await?;
    Ok(format!("{} + {}", result1, result2))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = combined_operation().await?;
    println!("{}", result);
    Ok(())
}
```

---

## 🔄 **Streams and Iterators**

### **Basic Stream**
```rust
use tokio_stream::{self as stream, StreamExt};

#[tokio::main]
async fn main() {
    let mut stream = stream::iter(1..=5);
    
    while let Some(value) = stream.next().await {
        println!("Value: {}", value);
    }
}
```

### **Stream Processing**
```rust
use tokio_stream::{self as stream, StreamExt};

#[tokio::main]
async fn main() {
    let numbers = stream::iter(1..=10);
    
    let doubled: Vec<i32> = numbers
        .map(|x| x * 2)
        .filter(|&x| x > 10)
        .collect()
        .await;
    
    println!("Doubled and filtered: {:?}", doubled);
}
```

---

## 🎯 **Best Practices**

### **Resource Management**
```rust
use tokio::time::{sleep, Duration};

// ✅ Good - use RAII for resources
async fn good_example() {
    let client = reqwest::Client::new();
    let response = client.get("https://example.com").send().await;
    // Client is automatically dropped
}

// ❌ Avoid - holding resources too long
async fn bad_example() {
    let client = reqwest::Client::new();
    // Don't hold client in a long-running task
    tokio::time::sleep(Duration::from_secs(3600)).await;
}
```

### **Error Handling**
```rust
// ✅ Good - handle errors appropriately
async fn robust_function() -> Result<String, Box<dyn std::error::Error>> {
    let result = risky_operation().await?;
    Ok(result)
}

// ✅ Good - use timeout for operations
async fn safe_operation() -> Result<String, Box<dyn std::error::Error>> {
    timeout(Duration::from_secs(5), risky_operation()).await?
}
```

### **Performance**
```rust
// ✅ Good - use appropriate concurrency
async fn efficient_processing() {
    let handles: Vec<_> = (0..100)
        .map(|i| tokio::spawn(process_item(i)))
        .collect();
    
    for handle in handles {
        handle.await.unwrap();
    }
}

// ❌ Avoid - unnecessary sequential processing
async fn inefficient_processing() {
    for i in 0..100 {
        process_item(i).await;
    }
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Async Book](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Tokio Examples](https://github.com/tokio-rs/tokio/tree/master/examples) - Fetched: 2024-12-19T00:00:00Z
- [Async Patterns](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

---

**Cheat Sheet Version**: 1.0  
**Rust Version**: 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z
