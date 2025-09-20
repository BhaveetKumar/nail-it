# Advanced Rust Exercises

> **Modules**: 25-30 (Expert Level)  
> **Difficulty**: Expert  
> **Estimated Time**: 4-6 hours  
> **Prerequisites**: All previous modules

---

## 🎯 **Exercise Overview**

These exercises will challenge your understanding of advanced Rust concepts including compiler internals, memory management, concurrency, and systems programming. Complete these to demonstrate expert-level Rust proficiency.

---

## 🔴 **Expert Level Exercises**

### **Exercise 1: Custom Memory Allocator**

**Task**: Implement a custom memory allocator that uses a buddy system for memory management.

```rust
use std::alloc::{Allocator, Layout, GlobalAlloc};
use std::ptr::NonNull;
use std::sync::Mutex;

// Implement a buddy system allocator
pub struct BuddyAllocator {
    // Your implementation here
}

unsafe impl GlobalAlloc for BuddyAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Implement buddy system allocation
        todo!()
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Implement buddy system deallocation
        todo!()
    }
}

#[global_allocator]
static ALLOCATOR: BuddyAllocator = BuddyAllocator::new();

// Test your allocator
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buddy_allocator() {
        // Test allocation and deallocation
        let layout = Layout::new::<i32>();
        let ptr = unsafe { ALLOCATOR.alloc(layout) };
        assert!(!ptr.is_null());
        unsafe { ALLOCATOR.dealloc(ptr, layout) };
    }
}
```

**Expected Learning**: Understanding memory management, allocator design, and unsafe Rust.

### **Exercise 2: Lock-Free Data Structure**

**Task**: Implement a lock-free stack using atomic operations.

```rust
use std::sync::atomic::{AtomicPtr, Ordering};
use std::ptr;

pub struct LockFreeStack<T> {
    head: AtomicPtr<Node<T>>,
}

struct Node<T> {
    data: T,
    next: *mut Node<T>,
}

impl<T> LockFreeStack<T> {
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }
    
    pub fn push(&self, data: T) {
        // Implement lock-free push
        todo!()
    }
    
    pub fn pop(&self) -> Option<T> {
        // Implement lock-free pop
        todo!()
    }
}

// Test your lock-free stack
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_lock_free_stack() {
        let stack = LockFreeStack::new();
        
        // Test single-threaded operations
        stack.push(1);
        stack.push(2);
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
    }
    
    #[test]
    fn test_concurrent_operations() {
        let stack = LockFreeStack::new();
        let mut handles = vec![];
        
        // Test concurrent push operations
        for i in 0..10 {
            let stack = &stack;
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    stack.push(i * 100 + j);
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all elements were pushed
        let mut count = 0;
        while stack.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 1000);
    }
}
```

**Expected Learning**: Understanding atomic operations, memory ordering, and lock-free programming.

### **Exercise 3: Compiler Pass**

**Task**: Write a custom lint pass that detects unused variables.

```rust
use rustc_lint::{EarlyLintPass, LintArray, LintPass};
use rustc_ast as ast;
use rustc_span::symbol::sym;
use rustc_session::lint;

declare_lint! {
    pub UNUSED_VARIABLE,
    Warn,
    "unused variable"
}

declare_lint_pass!(UnusedVariableLint => [UNUSED_VARIABLE]);

impl EarlyLintPass for UnusedVariableLint {
    fn check_local(&mut self, cx: &EarlyContext<'_>, local: &ast::Local) {
        // Implement unused variable detection
        todo!()
    }
}

// Test your lint pass
#[cfg(test)]
mod tests {
    use super::*;
    use rustc_span::create_session_if_not_set_then;
    
    #[test]
    fn test_unused_variable_lint() {
        create_session_if_not_set_then(|_| {
            let source = r#"
                fn main() {
                    let unused = 42;
                    let used = 24;
                    println!("{}", used);
                }
            "#;
            
            // Test that the lint detects unused variables
            let result = run_lint_pass(source);
            assert!(result.contains("unused variable"));
        });
    }
}
```

**Expected Learning**: Understanding compiler internals, AST traversal, and lint development.

### **Exercise 4: Async Runtime**

**Task**: Implement a simple async runtime with task scheduling.

```rust
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::future::Future;
use std::pin::Pin;

pub struct Task {
    future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    waker: Waker,
}

pub struct Executor {
    ready_queue: Arc<Mutex<VecDeque<Task>>>,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            ready_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
    
    pub fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        // Implement task spawning
        todo!()
    }
    
    pub fn run(&self) {
        // Implement task execution loop
        todo!()
    }
}

// Test your async runtime
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;
    
    #[test]
    fn test_async_runtime() {
        let executor = Executor::new();
        let counter = Arc::new(AtomicU32::new(0));
        
        let counter_clone = counter.clone();
        executor.spawn(async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        
        executor.run();
        
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
```

**Expected Learning**: Understanding async programming, task scheduling, and runtime implementation.

### **Exercise 5: FFI Integration**

**Task**: Create a safe Rust wrapper for a C library.

```rust
use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_int};

// Define the C function signature
extern "C" {
    fn strlen(s: *const c_char) -> c_int;
    fn strcpy(dest: *mut c_char, src: *const c_char) -> *mut c_char;
}

// Create a safe Rust wrapper
pub struct CStringWrapper {
    inner: CString,
}

impl CStringWrapper {
    pub fn new(s: &str) -> Result<Self, std::ffi::NulError> {
        Ok(Self {
            inner: CString::new(s)?,
        })
    }
    
    pub fn length(&self) -> usize {
        unsafe { strlen(self.inner.as_ptr()) as usize }
    }
    
    pub fn copy_to(&self, dest: &mut [u8]) -> Result<(), &'static str> {
        if dest.len() < self.length() + 1 {
            return Err("Destination buffer too small");
        }
        
        unsafe {
            strcpy(dest.as_mut_ptr() as *mut c_char, self.inner.as_ptr());
        }
        
        Ok(())
    }
}

// Test your FFI wrapper
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_c_string_wrapper() {
        let wrapper = CStringWrapper::new("Hello, World!").unwrap();
        assert_eq!(wrapper.length(), 13);
        
        let mut dest = [0u8; 20];
        wrapper.copy_to(&mut dest).unwrap();
        assert_eq!(CStr::from_bytes_until_nul(&dest).unwrap().to_str().unwrap(), "Hello, World!");
    }
}
```

**Expected Learning**: Understanding FFI, unsafe Rust, and C interoperability.

---

## 🧪 **Advanced Testing Strategies**

### **Property-Based Testing**

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_lock_free_stack_properties(
        values in prop::collection::vec(any::<i32>(), 0..1000)
    ) {
        let stack = LockFreeStack::new();
        
        // Push all values
        for &value in &values {
            stack.push(value);
        }
        
        // Pop all values and verify order
        let mut popped = Vec::new();
        while let Some(value) = stack.pop() {
            popped.push(value);
        }
        
        // Values should be popped in reverse order
        assert_eq!(popped, values.into_iter().rev().collect::<Vec<_>>());
    }
}
```

### **Concurrency Testing**

```rust
#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_concurrent_allocator() {
        let allocator = Arc::new(BuddyAllocator::new());
        let mut handles = vec![];
        
        for i in 0..10 {
            let allocator = allocator.clone();
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let layout = Layout::from_size_align(1 << (j % 10), 8).unwrap();
                    let ptr = unsafe { allocator.alloc(layout) };
                    assert!(!ptr.is_null());
                    unsafe { allocator.dealloc(ptr, layout) };
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
```

---

## 🚨 **Common Pitfalls and Solutions**

### **Memory Safety Issues**

```rust
// ❌ Wrong - potential use-after-free
unsafe fn bad_example() {
    let ptr = std::alloc::alloc(Layout::new::<i32>());
    std::alloc::dealloc(ptr, Layout::new::<i32>());
    *ptr = 42; // Use after free!
}

// ✅ Correct - proper memory management
unsafe fn good_example() {
    let layout = Layout::new::<i32>();
    let ptr = std::alloc::alloc(layout);
    if !ptr.is_null() {
        *ptr = 42;
        std::alloc::dealloc(ptr, layout);
    }
}
```

### **Atomic Operation Issues**

```rust
// ❌ Wrong - incorrect memory ordering
use std::sync::atomic::{AtomicPtr, Ordering};

fn bad_atomic_example() {
    let ptr = AtomicPtr::new(std::ptr::null_mut());
    ptr.store(some_ptr, Ordering::Relaxed); // May not be visible to other threads
}

// ✅ Correct - proper memory ordering
fn good_atomic_example() {
    let ptr = AtomicPtr::new(std::ptr::null_mut());
    ptr.store(some_ptr, Ordering::Release); // Ensures visibility
}
```

---

## 🎯 **Performance Optimization**

### **Memory Pool Optimization**

```rust
use std::sync::Mutex;
use std::collections::VecDeque;

pub struct ObjectPool<T> {
    objects: Mutex<VecDeque<T>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> ObjectPool<T> {
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            objects: Mutex::new(VecDeque::new()),
            factory: Box::new(factory),
        }
    }
    
    pub fn get(&self) -> T {
        self.objects
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| (self.factory)())
    }
    
    pub fn put(&self, obj: T) {
        self.objects.lock().unwrap().push_back(obj);
    }
}
```

### **Cache-Friendly Data Structures**

```rust
use std::mem;

pub struct CacheFriendlyArray<T> {
    data: Vec<T>,
    indices: Vec<usize>,
}

impl<T> CacheFriendlyArray<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            indices: Vec::new(),
        }
    }
    
    pub fn push(&mut self, value: T) -> usize {
        let index = self.data.len();
        self.data.push(value);
        self.indices.push(index);
        index
    }
    
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
}
```

---

## 📚 **Further Reading**

### **Advanced Topics**
- [Rustonomicon](https://doc.rust-lang.org/nomicon/) - Fetched: 2024-12-19T00:00:00Z
- [Rust Unsafe Code Guidelines](https://rust-lang.github.io/unsafe-code-guidelines/) - Fetched: 2024-12-19T00:00:00Z
- [Rust Compiler Development Guide](https://rustc-dev-guide.rust-lang.org/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust Internals Forum](https://internals.rust-lang.org/) - Fetched: 2024-12-19T00:00:00Z
- [Rust Performance Book](https://nnethercote.github.io/perf-book/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you implement a custom memory allocator?
2. Do you understand lock-free programming concepts?
3. Can you write compiler passes and lints?
4. Do you understand async runtime implementation?
5. Can you create safe FFI wrappers?

---

**Exercise Set Version**: 1.0  
**Rust Version**: 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z
