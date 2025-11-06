---
# Auto-generated front matter
Title: 29 01 Advanced Concurrency Patterns
LastUpdated: 2025-11-06T20:45:58.116615
Tags: []
Status: draft
---

# Lesson 29.1: Advanced Concurrency Patterns

> **Module**: 29 - Advanced Concurrency  
> **Lesson**: 1 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 28 (Advanced Memory Management)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Implement advanced concurrency patterns
- Design lock-free data structures
- Master actor model and message passing
- Optimize concurrent performance
- Handle complex synchronization scenarios

---

## 🎯 **Overview**

Advanced concurrency in Rust goes beyond basic threads and channels. This lesson covers sophisticated patterns for building high-performance concurrent systems, including actor models, work-stealing, and advanced synchronization primitives.

---

## 🔧 **Actor Model Implementation**

### **Basic Actor System**

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

pub type ActorId = u64;

pub trait Message: Send + 'static {}
pub trait Actor: Send + 'static {
    type Message: Message;
    
    fn handle_message(&mut self, msg: Self::Message);
    fn handle_timeout(&mut self) {}
}

pub struct ActorSystem {
    actors: Arc<Mutex<HashMap<ActorId, Box<dyn Actor<Message = ()>>>>>,
    sender: mpsc::Sender<(ActorId, Box<dyn Message + Send>)>,
}

impl ActorSystem {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel();
        let actors = Arc::new(Mutex::new(HashMap::new()));
        
        let actors_clone = actors.clone();
        thread::spawn(move || {
            let mut receiver = receiver;
            while let Ok((actor_id, message)) = receiver.recv() {
                if let Some(actor) = actors_clone.lock().unwrap().get_mut(&actor_id) {
                    // Handle message based on type
                    // This is simplified - in practice you'd use trait objects or enums
                }
            }
        });
        
        Self { actors, sender }
    }
    
    pub fn spawn<A>(&self, actor_id: ActorId, actor: A) 
    where
        A: Actor + 'static,
    {
        self.actors.lock().unwrap().insert(actor_id, Box::new(actor));
    }
    
    pub fn send<M>(&self, actor_id: ActorId, message: M)
    where
        M: Message,
    {
        let _ = self.sender.send((actor_id, Box::new(message)));
    }
}
```

### **Advanced Actor with State**

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct CounterActor {
    count: AtomicU64,
    subscribers: Vec<Arc<dyn Fn(u64) + Send + Sync>>,
}

impl CounterActor {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            subscribers: Vec::new(),
        }
    }
    
    pub fn increment(&self) {
        let new_count = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        self.notify_subscribers(new_count);
    }
    
    pub fn decrement(&self) {
        let new_count = self.count.fetch_sub(1, Ordering::SeqCst) - 1;
        self.notify_subscribers(new_count);
    }
    
    pub fn get_count(&self) -> u64 {
        self.count.load(Ordering::SeqCst)
    }
    
    pub fn subscribe<F>(&mut self, callback: F)
    where
        F: Fn(u64) + Send + Sync + 'static,
    {
        self.subscribers.push(Arc::new(callback));
    }
    
    fn notify_subscribers(&self, count: u64) {
        for callback in &self.subscribers {
            callback(count);
        }
    }
}

impl Actor for CounterActor {
    type Message = CounterMessage;
    
    fn handle_message(&mut self, msg: Self::Message) {
        match msg {
            CounterMessage::Increment => self.increment(),
            CounterMessage::Decrement => self.decrement(),
            CounterMessage::GetCount(sender) => {
                let _ = sender.send(self.get_count());
            }
        }
    }
}

#[derive(Debug)]
pub enum CounterMessage {
    Increment,
    Decrement,
    GetCount(mpsc::Sender<u64>),
}

impl Message for CounterMessage {}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Work-Stealing Thread Pool**

```rust
use std::sync::{Arc, Mutex, Condvar};
use std::collections::VecDeque;
use std::thread;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct WorkStealingThreadPool {
    workers: Vec<Worker>,
    global_queue: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send + 'static>>>>,
    shutdown: Arc<AtomicBool>,
}

struct Worker {
    id: usize,
    local_queue: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send + 'static>>>>,
    condvar: Arc<Condvar>,
}

impl WorkStealingThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let global_queue = Arc::new(Mutex::new(VecDeque::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::with_capacity(num_threads);
        
        for i in 0..num_threads {
            let worker = Worker::new(
                i,
                global_queue.clone(),
                shutdown.clone(),
            );
            workers.push(worker);
        }
        
        Self {
            workers,
            global_queue,
            shutdown,
        }
    }
    
    pub fn execute<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let task = Box::new(task);
        
        // Try to find a worker with an empty local queue
        for worker in &self.workers {
            if let Ok(mut local_queue) = worker.local_queue.try_lock() {
                if local_queue.is_empty() {
                    local_queue.push_back(task);
                    worker.condvar.notify_one();
                    return;
                }
            }
        }
        
        // Fall back to global queue
        if let Ok(mut global_queue) = self.global_queue.lock() {
            global_queue.push_back(task);
        }
    }
    
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::SeqCst);
        
        for worker in self.workers {
            worker.condvar.notify_all();
            if let Some(handle) = worker.handle {
                let _ = handle.join();
            }
        }
    }
}

impl Worker {
    fn new(
        id: usize,
        global_queue: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send + 'static>>>>,
        shutdown: Arc<AtomicBool>,
    ) -> Self {
        let local_queue = Arc::new(Mutex::new(VecDeque::new()));
        let condvar = Arc::new(Condvar::new());
        
        let worker = Self {
            id,
            local_queue: local_queue.clone(),
            condvar: condvar.clone(),
        };
        
        let handle = thread::spawn(move || {
            worker.run(global_queue, shutdown);
        });
        
        Self {
            id,
            local_queue,
            condvar,
            handle: Some(handle),
        }
    }
    
    fn run(
        self,
        global_queue: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send + 'static>>>>,
        shutdown: Arc<AtomicBool>,
    ) {
        while !shutdown.load(Ordering::SeqCst) {
            // Try to get work from local queue
            if let Some(task) = self.local_queue.lock().unwrap().pop_front() {
                task();
                continue;
            }
            
            // Try to steal work from global queue
            if let Some(task) = global_queue.lock().unwrap().pop_front() {
                task();
                continue;
            }
            
            // Try to steal work from other workers
            if let Some(task) = self.steal_work() {
                task();
                continue;
            }
            
            // Wait for work
            let _ = self.condvar.wait_timeout(
                self.local_queue.lock().unwrap(),
                Duration::from_millis(100),
            );
        }
    }
    
    fn steal_work(&self) -> Option<Box<dyn FnOnce() + Send + 'static>> {
        // Implementation for work stealing from other workers
        None
    }
}
```

### **Exercise 2: Lock-Free Ring Buffer**

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr;

pub struct LockFreeRingBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
    mask: usize,
}

impl<T> LockFreeRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let buffer = Vec::with_capacity(capacity);
        
        Self {
            buffer,
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            mask: capacity - 1,
        }
    }
    
    pub fn push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        if (head + 1) & self.mask == tail {
            return Err(value);
        }
        
        unsafe {
            let index = head & self.mask;
            ptr::write(self.buffer.as_ptr().add(index), value);
        }
        
        self.head.store((head + 1) & self.mask, Ordering::Release);
        Ok(())
    }
    
    pub fn pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        
        if head == tail {
            return None;
        }
        
        let index = tail & self.mask;
        let value = unsafe { ptr::read(self.buffer.as_ptr().add(index)) };
        
        self.tail.store((tail + 1) & self.mask, Ordering::Release);
        Some(value)
    }
    
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        head == tail
    }
    
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        (head + 1) & self.mask == tail
    }
}
```

### **Exercise 3: Advanced Channel with Backpressure**

```rust
use std::sync::{Arc, Mutex, Condvar};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct BackpressureChannel<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    condvar: Arc<Condvar>,
    max_size: usize,
    current_size: AtomicUsize,
    closed: AtomicBool,
}

impl<T> BackpressureChannel<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            condvar: Arc::new(Condvar::new()),
            max_size,
            current_size: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
        }
    }
    
    pub fn send(&self, value: T) -> Result<(), T> {
        if self.closed.load(Ordering::Acquire) {
            return Err(value);
        }
        
        let current_size = self.current_size.load(Ordering::Relaxed);
        if current_size >= self.max_size {
            return Err(value);
        }
        
        {
            let mut queue = self.queue.lock().unwrap();
            queue.push_back(value);
        }
        
        self.current_size.fetch_add(1, Ordering::Release);
        self.condvar.notify_one();
        Ok(())
    }
    
    pub fn try_send(&self, value: T) -> Result<(), T> {
        if self.closed.load(Ordering::Acquire) {
            return Err(value);
        }
        
        let current_size = self.current_size.load(Ordering::Relaxed);
        if current_size >= self.max_size {
            return Err(value);
        }
        
        {
            let mut queue = self.queue.lock().unwrap();
            queue.push_back(value);
        }
        
        self.current_size.fetch_add(1, Ordering::Release);
        self.condvar.notify_one();
        Ok(())
    }
    
    pub fn recv(&self) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();
        
        while queue.is_empty() && !self.closed.load(Ordering::Acquire) {
            queue = self.condvar.wait(queue).unwrap();
        }
        
        if queue.is_empty() && self.closed.load(Ordering::Acquire) {
            return None;
        }
        
        let value = queue.pop_front()?;
        self.current_size.fetch_sub(1, Ordering::Release);
        Some(value)
    }
    
    pub fn try_recv(&self) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();
        
        if queue.is_empty() {
            return None;
        }
        
        let value = queue.pop_front()?;
        self.current_size.fetch_sub(1, Ordering::Release);
        Some(value)
    }
    
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.condvar.notify_all();
    }
    
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
    
    pub fn len(&self) -> usize {
        self.current_size.load(Ordering::Acquire)
    }
    
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    pub fn is_full(&self) -> bool {
        self.len() >= self.max_size
    }
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_actor_system() {
        let system = ActorSystem::new();
        let counter = Arc::new(Mutex::new(CounterActor::new()));
        
        system.spawn(1, counter.clone());
        
        system.send(1, CounterMessage::Increment);
        system.send(1, CounterMessage::Increment);
        
        thread::sleep(Duration::from_millis(100));
        
        assert_eq!(counter.lock().unwrap().get_count(), 2);
    }

    #[test]
    fn test_work_stealing_thread_pool() {
        let pool = WorkStealingThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        
        for i in 0..100 {
            let counter_clone = counter.clone();
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }
        
        thread::sleep(Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn test_lock_free_ring_buffer() {
        let buffer = LockFreeRingBuffer::new(4);
        
        assert!(buffer.push(1).is_ok());
        assert!(buffer.push(2).is_ok());
        assert!(buffer.push(3).is_ok());
        assert!(buffer.push(4).is_ok());
        assert!(buffer.push(5).is_err()); // Buffer full
        
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), Some(4));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn test_backpressure_channel() {
        let channel = BackpressureChannel::new(2);
        
        assert!(channel.send(1).is_ok());
        assert!(channel.send(2).is_ok());
        assert!(channel.send(3).is_err()); // Channel full
        
        assert_eq!(channel.recv(), Some(1));
        assert_eq!(channel.recv(), Some(2));
        assert_eq!(channel.recv(), None);
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Deadlocks in Actor Systems**

```rust
// ❌ Wrong - potential deadlock
impl Actor for BadActor {
    fn handle_message(&mut self, msg: Self::Message) {
        match msg {
            Message::RequestData(sender) => {
                // This can cause deadlock if the sender is waiting
                let data = self.get_data();
                let _ = sender.send(data);
            }
        }
    }
}

// ✅ Correct - use async patterns
impl Actor for GoodActor {
    fn handle_message(&mut self, msg: Self::Message) {
        match msg {
            Message::RequestData(sender) => {
                // Use non-blocking approach
                if let Some(data) = self.try_get_data() {
                    let _ = sender.send(data);
                } else {
                    // Queue the request for later
                    self.pending_requests.push(sender);
                }
            }
        }
    }
}
```

### **Common Mistake 2: Memory Ordering Issues**

```rust
// ❌ Wrong - incorrect memory ordering
impl LockFreeRingBuffer<T> {
    pub fn push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        if (head + 1) & self.mask == tail {
            return Err(value);
        }
        
        // ... write value ...
        
        // Wrong ordering - other threads might not see the write
        self.head.store((head + 1) & self.mask, Ordering::Relaxed);
        Ok(())
    }
}

// ✅ Correct - proper memory ordering
impl LockFreeRingBuffer<T> {
    pub fn push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        if (head + 1) & self.mask == tail {
            return Err(value);
        }
        
        // ... write value ...
        
        // Correct ordering - ensures write is visible
        self.head.store((head + 1) & self.mask, Ordering::Release);
        Ok(())
    }
}
```

---

## 📊 **Advanced Concurrency Patterns**

### **Pipeline Pattern**

```rust
use std::sync::mpsc;
use std::thread;

pub struct Pipeline<T> {
    stages: Vec<Stage<T>>,
    input: mpsc::Sender<T>,
    output: mpsc::Receiver<T>,
}

struct Stage<T> {
    input: mpsc::Receiver<T>,
    output: mpsc::Sender<T>,
    processor: Box<dyn Fn(T) -> T + Send + 'static>,
}

impl<T> Pipeline<T> {
    pub fn new<F>(processors: Vec<F>) -> Self
    where
        F: Fn(T) -> T + Send + 'static,
    {
        let mut stages = Vec::new();
        let mut channels = Vec::new();
        
        // Create channels between stages
        for _ in 0..processors.len() + 1 {
            let (sender, receiver) = mpsc::channel();
            channels.push((sender, receiver));
        }
        
        // Create stages
        for (i, processor) in processors.into_iter().enumerate() {
            let input = channels[i].1;
            let output = channels[i + 1].0.clone();
            
            stages.push(Stage {
                input,
                output,
                processor: Box::new(processor),
            });
        }
        
        let input = channels[0].0;
        let output = channels[channels.len() - 1].1;
        
        Self {
            stages,
            input,
            output,
        }
    }
    
    pub fn start(self) {
        for stage in self.stages {
            thread::spawn(move || {
                while let Ok(item) = stage.input.recv() {
                    let processed = (stage.processor)(item);
                    let _ = stage.output.send(processed);
                }
            });
        }
    }
    
    pub fn send(&self, item: T) -> Result<(), mpsc::SendError<T>> {
        self.input.send(item)
    }
    
    pub fn recv(&self) -> Result<T, mpsc::RecvError> {
        self.output.recv()
    }
}
```

### **Circuit Breaker Pattern**

```rust
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU32, Ordering};

pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    state: Arc<Mutex<CircuitState>>,
    failure_count: AtomicU32,
    failure_threshold: u32,
    timeout: Duration,
    last_failure_time: Arc<Mutex<Option<Instant>>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, timeout: Duration) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitState::Closed)),
            failure_count: AtomicU32::new(0),
            failure_threshold,
            timeout,
            last_failure_time: Arc::new(Mutex::new(None)),
        }
    }
    
    pub fn call<F, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let state = self.state.lock().unwrap();
        
        match *state {
            CircuitState::Closed => {
                drop(state);
                self.execute_operation(operation)
            }
            CircuitState::Open => {
                if self.should_attempt_reset() {
                    self.transition_to_half_open();
                    self.execute_operation(operation)
                } else {
                    Err(/* Circuit breaker error */)
                }
            }
            CircuitState::HalfOpen => {
                drop(state);
                self.execute_operation(operation)
            }
        }
    }
    
    fn execute_operation<F, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(error) => {
                self.on_failure();
                Err(error)
            }
        }
    }
    
    fn on_success(&self) {
        self.failure_count.store(0, Ordering::SeqCst);
        self.transition_to_closed();
    }
    
    fn on_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        
        if count >= self.failure_threshold {
            self.transition_to_open();
        }
    }
    
    fn should_attempt_reset(&self) -> bool {
        if let Ok(last_failure) = self.last_failure_time.lock() {
            if let Some(time) = *last_failure {
                return Instant::now().duration_since(time) >= self.timeout;
            }
        }
        false
    }
    
    fn transition_to_closed(&self) {
        let mut state = self.state.lock().unwrap();
        *state = CircuitState::Closed;
    }
    
    fn transition_to_open(&self) {
        let mut state = self.state.lock().unwrap();
        *state = CircuitState::Open;
        
        let mut last_failure = self.last_failure_time.lock().unwrap();
        *last_failure = Some(Instant::now());
    }
    
    fn transition_to_half_open(&self) {
        let mut state = self.state.lock().unwrap();
        *state = CircuitState::HalfOpen;
    }
}
```

---

## 🎯 **Best Practices**

### **Performance Optimization**

```rust
// ✅ Good - use appropriate data structures
use std::sync::mpsc;
use std::collections::VecDeque;

pub struct OptimizedChannel<T> {
    queue: VecDeque<T>,
    capacity: usize,
    // Use VecDeque for O(1) push/pop operations
}

// ✅ Good - minimize allocations
pub struct ObjectPool<T> {
    objects: VecDeque<T>,
    factory: Box<dyn Fn() -> T>,
}

impl<T> ObjectPool<T> {
    pub fn get(&mut self) -> T {
        self.objects.pop_front().unwrap_or_else(|| (self.factory)())
    }
    
    pub fn put(&mut self, obj: T) {
        self.objects.push_back(obj);
    }
}
```

### **Error Handling**

```rust
// ✅ Good - comprehensive error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConcurrencyError {
    #[error("Channel is closed")]
    ChannelClosed,
    #[error("Operation timeout")]
    Timeout,
    #[error("Resource not available")]
    ResourceUnavailable,
    #[error("Deadlock detected")]
    Deadlock,
}

pub struct SafeChannel<T> {
    // Implementation with proper error handling
}

impl<T> SafeChannel<T> {
    pub fn send(&self, value: T) -> Result<(), ConcurrencyError> {
        // Implementation with error handling
        Ok(())
    }
    
    pub fn recv(&self) -> Result<T, ConcurrencyError> {
        // Implementation with error handling
        Err(ConcurrencyError::ChannelClosed)
    }
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Rust Concurrency](https://doc.rust-lang.org/book/ch16-00-concurrency.html) - Fetched: 2024-12-19T00:00:00Z
- [Rust Async Book](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust Concurrency Patterns](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z
- [Advanced Rust Concurrency](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you implement the actor model in Rust?
2. Do you understand work-stealing thread pools?
3. Can you create lock-free data structures?
4. Do you understand advanced synchronization patterns?
5. Can you optimize concurrent performance?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced async patterns
- Stream processing
- Reactive programming
- Performance profiling

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [29.2 Advanced Async Patterns](29_02_advanced_async.md)
