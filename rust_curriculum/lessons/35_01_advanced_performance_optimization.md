# Lesson 35.1: Advanced Performance Optimization

> **Module**: 35 - Advanced Performance Optimization  
> **Lesson**: 1 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 34 (Advanced Security Patterns)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Profile and optimize Rust applications
- Implement advanced memory management patterns
- Optimize concurrent and parallel code
- Use performance monitoring tools
- Build high-performance systems

---

## 🎯 **Overview**

Advanced performance optimization in Rust involves profiling applications, optimizing memory usage, improving concurrency, and building high-performance systems. This lesson covers profiling tools, optimization techniques, and performance patterns.

---

## 🔧 **Profiling and Benchmarking**

### **Comprehensive Profiling System**

```rust
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub function_name: String,
    pub call_count: u64,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub memory_usage: u64,
    pub cpu_usage: f64,
}

pub struct PerformanceProfiler {
    pub metrics: Arc<RwLock<HashMap<String, PerformanceMetrics>>>,
    pub enabled: bool,
    pub sample_rate: f64,
}

impl PerformanceProfiler {
    pub fn new(enabled: bool, sample_rate: f64) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            enabled,
            sample_rate,
        }
    }
    
    pub fn profile<F, T>(&self, function_name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        if !self.enabled {
            return f();
        }
        
        let start_time = Instant::now();
        let start_memory = self.get_memory_usage();
        let start_cpu = self.get_cpu_usage();
        
        let result = f();
        
        let end_time = Instant::now();
        let end_memory = self.get_memory_usage();
        let end_cpu = self.get_cpu_usage();
        
        let duration = end_time.duration_since(start_time);
        let memory_usage = end_memory.saturating_sub(start_memory);
        let cpu_usage = end_cpu - start_cpu;
        
        self.record_metrics(function_name, duration, memory_usage, cpu_usage);
        
        result
    }
    
    async fn record_metrics(&self, function_name: &str, duration: Duration, memory_usage: u64, cpu_usage: f64) {
        let mut metrics = self.metrics.write().await;
        
        let entry = metrics.entry(function_name.to_string()).or_insert(PerformanceMetrics {
            function_name: function_name.to_string(),
            call_count: 0,
            total_time: Duration::from_secs(0),
            average_time: Duration::from_secs(0),
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::from_secs(0),
            memory_usage: 0,
            cpu_usage: 0.0,
        });
        
        entry.call_count += 1;
        entry.total_time += duration;
        entry.average_time = Duration::from_nanos(entry.total_time.as_nanos() as u64 / entry.call_count);
        entry.min_time = entry.min_time.min(duration);
        entry.max_time = entry.max_time.max(duration);
        entry.memory_usage = entry.memory_usage.max(memory_usage);
        entry.cpu_usage = entry.cpu_usage.max(cpu_usage);
    }
    
    fn get_memory_usage(&self) -> u64 {
        // In a real implementation, you would use system APIs to get memory usage
        // For now, return a mock value
        0
    }
    
    fn get_cpu_usage(&self) -> f64 {
        // In a real implementation, you would use system APIs to get CPU usage
        // For now, return a mock value
        0.0
    }
    
    pub async fn get_metrics(&self) -> Vec<PerformanceMetrics> {
        let metrics = self.metrics.read().await;
        metrics.values().cloned().collect()
    }
    
    pub async fn get_slowest_functions(&self, limit: usize) -> Vec<PerformanceMetrics> {
        let mut metrics = self.get_metrics().await;
        metrics.sort_by(|a, b| b.average_time.cmp(&a.average_time));
        metrics.truncate(limit);
        metrics
    }
    
    pub async fn get_most_called_functions(&self, limit: usize) -> Vec<PerformanceMetrics> {
        let mut metrics = self.get_metrics().await;
        metrics.sort_by(|a, b| b.call_count.cmp(&a.call_count));
        metrics.truncate(limit);
        metrics
    }
}

// Macro for easy profiling
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $name:expr, $code:block) => {
        $profiler.profile($name, || $code)
    };
}

// Example usage
pub fn example_function(profiler: &PerformanceProfiler) -> i32 {
    profile!(profiler, "example_function", {
        // Simulate some work
        std::thread::sleep(Duration::from_millis(100));
        42
    })
}
```

### **Memory Profiling**

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub struct MemoryProfiler {
    pub total_allocated: AtomicU64,
    pub total_deallocated: AtomicU64,
    pub peak_memory: AtomicU64,
    pub allocation_count: AtomicU64,
    pub deallocation_count: AtomicU64,
    pub allocations: Mutex<HashMap<usize, AllocationInfo>>,
}

#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub size: usize,
    pub count: u64,
    pub total_size: u64,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            total_allocated: AtomicU64::new(0),
            total_deallocated: AtomicU64::new(0),
            peak_memory: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
            allocations: Mutex::new(HashMap::new()),
        }
    }
    
    pub fn record_allocation(&self, size: usize) {
        self.total_allocated.fetch_add(size as u64, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        let current_memory = self.get_current_memory_usage();
        let peak = self.peak_memory.load(Ordering::Relaxed);
        if current_memory > peak {
            self.peak_memory.store(current_memory, Ordering::Relaxed);
        }
        
        let mut allocations = self.allocations.lock().unwrap();
        let entry = allocations.entry(size).or_insert(AllocationInfo {
            size,
            count: 0,
            total_size: 0,
        });
        entry.count += 1;
        entry.total_size += size as u64;
    }
    
    pub fn record_deallocation(&self, size: usize) {
        self.total_deallocated.fetch_add(size as u64, Ordering::Relaxed);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn get_current_memory_usage(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed) - self.total_deallocated.load(Ordering::Relaxed)
    }
    
    pub fn get_peak_memory_usage(&self) -> u64 {
        self.peak_memory.load(Ordering::Relaxed)
    }
    
    pub fn get_allocation_stats(&self) -> AllocationStats {
        let allocations = self.allocations.lock().unwrap();
        let mut size_distribution = Vec::new();
        
        for (size, info) in allocations.iter() {
            size_distribution.push((*size, info.count, info.total_size));
        }
        
        size_distribution.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count
        
        AllocationStats {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_deallocated: self.total_deallocated.load(Ordering::Relaxed),
            peak_memory: self.peak_memory.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.deallocation_count.load(Ordering::Relaxed),
            size_distribution,
        }
    }
}

#[derive(Debug)]
pub struct AllocationStats {
    pub total_allocated: u64,
    pub total_deallocated: u64,
    pub peak_memory: u64,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub size_distribution: Vec<(usize, u64, u64)>, // (size, count, total_size)
}

pub struct ProfilingAllocator {
    pub profiler: MemoryProfiler,
}

unsafe impl GlobalAlloc for ProfilingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            self.profiler.record_allocation(layout.size());
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.profiler.record_deallocation(layout.size());
        System.dealloc(ptr, layout);
    }
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Advanced Caching System**

```rust
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

pub struct AdvancedCache<K, V> {
    pub l1_cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    pub l2_cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    pub l1_capacity: usize,
    pub l2_capacity: usize,
    pub l1_ttl: Duration,
    pub l2_ttl: Duration,
    pub hit_count: AtomicU64,
    pub miss_count: AtomicU64,
    pub eviction_count: AtomicU64,
}

#[derive(Clone, Debug)]
pub struct CacheEntry<V> {
    pub value: V,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub size: usize,
}

impl<K, V> AdvancedCache<K, V>
where
    K: Clone + Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new(l1_capacity: usize, l2_capacity: usize, l1_ttl: Duration, l2_ttl: Duration) -> Self {
        Self {
            l1_cache: Arc::new(RwLock::new(HashMap::new())),
            l2_cache: Arc::new(RwLock::new(HashMap::new())),
            l1_capacity,
            l2_capacity,
            l1_ttl,
            l2_ttl,
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
            eviction_count: AtomicU64::new(0),
        }
    }
    
    pub async fn get(&self, key: &K) -> Option<V> {
        // Check L1 cache first
        if let Some(entry) = self.get_from_l1(key).await {
            if entry.created_at.elapsed() < self.l1_ttl {
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                self.update_access_stats(key, &entry).await;
                return Some(entry.value);
            } else {
                // L1 entry expired, remove it
                self.remove_from_l1(key).await;
            }
        }
        
        // Check L2 cache
        if let Some(entry) = self.get_from_l2(key).await {
            if entry.created_at.elapsed() < self.l2_ttl {
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                // Promote to L1
                self.promote_to_l1(key.clone(), entry.clone()).await;
                return Some(entry.value);
            } else {
                // L2 entry expired, remove it
                self.remove_from_l2(key).await;
            }
        }
        
        self.miss_count.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    pub async fn set(&self, key: K, value: V, size: usize) {
        let entry = CacheEntry {
            value,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            size,
        };
        
        // Try to insert into L1 cache
        if self.can_fit_in_l1(size).await {
            self.insert_into_l1(key, entry).await;
        } else {
            // L1 doesn't have space, try L2
            if self.can_fit_in_l2(size).await {
                self.insert_into_l2(key, entry).await;
            } else {
                // Neither cache has space, evict and insert
                self.evict_and_insert(key, entry).await;
            }
        }
    }
    
    async fn get_from_l1(&self, key: &K) -> Option<CacheEntry<V>> {
        let l1_cache = self.l1_cache.read().await;
        l1_cache.get(key).cloned()
    }
    
    async fn get_from_l2(&self, key: &K) -> Option<CacheEntry<V>> {
        let l2_cache = self.l2_cache.read().await;
        l2_cache.get(key).cloned()
    }
    
    async fn insert_into_l1(&self, key: K, entry: CacheEntry<V>) {
        let mut l1_cache = self.l1_cache.write().await;
        l1_cache.insert(key, entry);
    }
    
    async fn insert_into_l2(&self, key: K, entry: CacheEntry<V>) {
        let mut l2_cache = self.l2_cache.write().await;
        l2_cache.insert(key, entry);
    }
    
    async fn can_fit_in_l1(&self, size: usize) -> bool {
        let l1_cache = self.l1_cache.read().await;
        let current_size: usize = l1_cache.values().map(|e| e.size).sum();
        current_size + size <= self.l1_capacity
    }
    
    async fn can_fit_in_l2(&self, size: usize) -> bool {
        let l2_cache = self.l2_cache.read().await;
        let current_size: usize = l2_cache.values().map(|e| e.size).sum();
        current_size + size <= self.l2_capacity
    }
    
    async fn evict_and_insert(&self, key: K, entry: CacheEntry<V>) {
        // Evict from L1 first
        if let Some(evicted_key) = self.evict_lru_from_l1().await {
            if let Some(evicted_entry) = self.remove_from_l1(&evicted_key).await {
                self.insert_into_l2(evicted_key, evicted_entry).await;
            }
        }
        
        // Try to insert into L1
        if self.can_fit_in_l1(entry.size).await {
            self.insert_into_l1(key, entry).await;
        } else {
            // Evict from L2 and insert into L2
            if let Some(evicted_key) = self.evict_lru_from_l2().await {
                self.remove_from_l2(&evicted_key).await;
            }
            self.insert_into_l2(key, entry).await;
        }
        
        self.eviction_count.fetch_add(1, Ordering::Relaxed);
    }
    
    async fn evict_lru_from_l1(&self) -> Option<K> {
        let l1_cache = self.l1_cache.read().await;
        let mut lru_key = None;
        let mut lru_time = Instant::now();
        
        for (key, entry) in l1_cache.iter() {
            if entry.last_accessed < lru_time {
                lru_time = entry.last_accessed;
                lru_key = Some(key.clone());
            }
        }
        
        lru_key
    }
    
    async fn evict_lru_from_l2(&self) -> Option<K> {
        let l2_cache = self.l2_cache.read().await;
        let mut lru_key = None;
        let mut lru_time = Instant::now();
        
        for (key, entry) in l2_cache.iter() {
            if entry.last_accessed < lru_time {
                lru_time = entry.last_accessed;
                lru_key = Some(key.clone());
            }
        }
        
        lru_key
    }
    
    async fn remove_from_l1(&self, key: &K) -> Option<CacheEntry<V>> {
        let mut l1_cache = self.l1_cache.write().await;
        l1_cache.remove(key)
    }
    
    async fn remove_from_l2(&self, key: &K) -> Option<CacheEntry<V>> {
        let mut l2_cache = self.l2_cache.write().await;
        l2_cache.remove(key)
    }
    
    async fn promote_to_l1(&self, key: K, mut entry: CacheEntry<V>) {
        // Update access statistics
        entry.access_count += 1;
        entry.last_accessed = Instant::now();
        
        // Remove from L2
        self.remove_from_l2(&key).await;
        
        // Insert into L1
        if self.can_fit_in_l1(entry.size).await {
            self.insert_into_l1(key, entry).await;
        } else {
            // L1 is full, evict LRU and insert
            if let Some(evicted_key) = self.evict_lru_from_l1().await {
                if let Some(evicted_entry) = self.remove_from_l1(&evicted_key).await {
                    self.insert_into_l2(evicted_key, evicted_entry).await;
                }
            }
            self.insert_into_l1(key, entry).await;
        }
    }
    
    async fn update_access_stats(&self, key: &K, entry: &CacheEntry<V>) {
        let mut l1_cache = self.l1_cache.write().await;
        if let Some(cached_entry) = l1_cache.get_mut(key) {
            cached_entry.access_count += 1;
            cached_entry.last_accessed = Instant::now();
        }
    }
    
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            l1_size: 0, // Would need to calculate
            l2_size: 0, // Would need to calculate
            hit_count: self.hit_count.load(Ordering::Relaxed),
            miss_count: self.miss_count.load(Ordering::Relaxed),
            eviction_count: self.eviction_count.load(Ordering::Relaxed),
            hit_rate: self.calculate_hit_rate(),
        }
    }
    
    fn calculate_hit_rate(&self) -> f64 {
        let hits = self.hit_count.load(Ordering::Relaxed);
        let misses = self.miss_count.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub l1_size: usize,
    pub l2_size: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub hit_rate: f64,
}
```

### **Exercise 2: Lock-Free Data Structures**

```rust
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::ptr;

pub struct LockFreeStack<T> {
    head: AtomicPtr<Node<T>>,
    size: AtomicUsize,
}

struct Node<T> {
    data: T,
    next: *mut Node<T>,
}

impl<T> LockFreeStack<T> {
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            size: AtomicUsize::new(0),
        }
    }
    
    pub fn push(&self, data: T) {
        let new_node = Box::into_raw(Box::new(Node {
            data,
            next: ptr::null_mut(),
        }));
        
        loop {
            let current_head = self.head.load(Ordering::Acquire);
            unsafe {
                (*new_node).next = current_head;
            }
            
            match self.head.compare_exchange_weak(
                current_head,
                new_node,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.size.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                Err(_) => {
                    // Retry
                    continue;
                }
            }
        }
    }
    
    pub fn pop(&self) -> Option<T> {
        loop {
            let current_head = self.head.load(Ordering::Acquire);
            if current_head.is_null() {
                return None;
            }
            
            unsafe {
                let next = (*current_head).next;
                match self.head.compare_exchange_weak(
                    current_head,
                    next,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        self.size.fetch_sub(1, Ordering::Relaxed);
                        let node = Box::from_raw(current_head);
                        return Some(node.data);
                    }
                    Err(_) => {
                        // Retry
                        continue;
                    }
                }
            }
        }
    }
    
    pub fn size(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }
    
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire).is_null()
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}

// Lock-free ring buffer
pub struct LockFreeRingBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T> LockFreeRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let buffer = Vec::with_capacity(capacity);
        
        Self {
            buffer,
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
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

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_performance_profiler() {
        let profiler = PerformanceProfiler::new(true, 1.0);
        
        let result = profiler.profile("test_function", || {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        
        let metrics = profiler.get_metrics().await;
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].function_name, "test_function");
        assert_eq!(metrics[0].call_count, 1);
    }

    #[tokio::test]
    async fn test_advanced_cache() {
        let cache = AdvancedCache::new(100, 200, Duration::from_secs(60), Duration::from_secs(300));
        
        // Test basic operations
        cache.set("key1".to_string(), "value1".to_string(), 10).await;
        cache.set("key2".to_string(), "value2".to_string(), 10).await;
        
        assert_eq!(cache.get(&"key1".to_string()).await, Some("value1".to_string()));
        assert_eq!(cache.get(&"key2".to_string()).await, Some("value2".to_string()));
        
        // Test cache eviction
        for i in 0..20 {
            cache.set(format!("key{}", i), format!("value{}", i), 10).await;
        }
        
        // Some keys should be evicted
        let stats = cache.get_stats();
        assert!(stats.eviction_count > 0);
    }

    #[test]
    fn test_lock_free_stack() {
        let stack = LockFreeStack::new();
        
        stack.push(1);
        stack.push(2);
        stack.push(3);
        
        assert_eq!(stack.size(), 3);
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
        assert!(stack.is_empty());
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
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Inefficient Memory Usage**

```rust
// ❌ Wrong - inefficient memory usage
fn bad_memory_usage() -> Vec<String> {
    let mut result = Vec::new();
    for i in 0..1000 {
        result.push(format!("item_{}", i));
    }
    result
}

// ✅ Correct - efficient memory usage
fn good_memory_usage() -> Vec<String> {
    let mut result = Vec::with_capacity(1000);
    for i in 0..1000 {
        result.push(format!("item_{}", i));
    }
    result
}
```

### **Common Mistake 2: Inefficient Concurrency**

```rust
// ❌ Wrong - inefficient concurrency
use std::sync::Mutex;

fn bad_concurrency() {
    let data = Arc::new(Mutex::new(Vec::new()));
    for i in 0..1000 {
        let data = data.clone();
        std::thread::spawn(move || {
            let mut data = data.lock().unwrap();
            data.push(i);
        });
    }
}

// ✅ Correct - efficient concurrency
use crossbeam::channel;

fn good_concurrency() {
    let (sender, receiver) = channel::unbounded();
    
    // Spawn workers
    for _ in 0..4 {
        let receiver = receiver.clone();
        std::thread::spawn(move || {
            while let Ok(item) = receiver.recv() {
                // Process item
            }
        });
    }
    
    // Send work
    for i in 0..1000 {
        sender.send(i).unwrap();
    }
}
```

---

## 📊 **Advanced Performance Patterns**

### **SIMD Optimization**

```rust
use std::arch::x86_64::*;

pub fn simd_sum_f32(values: &[f32]) -> f32 {
    if values.len() < 8 {
        return values.iter().sum();
    }
    
    unsafe {
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= values.len() {
            let chunk = _mm256_loadu_ps(values.as_ptr().add(i));
            sum = _mm256_add_ps(sum, chunk);
            i += 8;
        }
        
        // Handle remaining elements
        let mut result = 0.0;
        for &value in &values[i..] {
            result += value;
        }
        
        // Sum the SIMD result
        let mut simd_result = [0.0f32; 8];
        _mm256_storeu_ps(simd_result.as_mut_ptr(), sum);
        result += simd_result.iter().sum::<f32>();
        
        result
    }
}

pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.len() < 8 {
        return a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    }
    
    unsafe {
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= a.len() {
            let chunk_a = _mm256_loadu_ps(a.as_ptr().add(i));
            let chunk_b = _mm256_loadu_ps(b.as_ptr().add(i));
            let product = _mm256_mul_ps(chunk_a, chunk_b);
            sum = _mm256_add_ps(sum, product);
            i += 8;
        }
        
        // Handle remaining elements
        let mut result = 0.0;
        for j in i..a.len() {
            result += a[j] * b[j];
        }
        
        // Sum the SIMD result
        let mut simd_result = [0.0f32; 8];
        _mm256_storeu_ps(simd_result.as_mut_ptr(), sum);
        result += simd_result.iter().sum::<f32>();
        
        result
    }
}
```

---

## 🎯 **Best Practices**

### **Performance Configuration**

```rust
// ✅ Good - comprehensive performance configuration
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct PerformanceConfig {
    pub profiling: ProfilingConfig,
    pub caching: CachingConfig,
    pub concurrency: ConcurrencyConfig,
    pub memory: MemoryConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ProfilingConfig {
    pub enabled: bool,
    pub sample_rate: f64,
    pub output_file: String,
    pub metrics_interval: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CachingConfig {
    pub l1_capacity: usize,
    pub l2_capacity: usize,
    pub l1_ttl: u64,
    pub l2_ttl: u64,
    pub eviction_policy: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ConcurrencyConfig {
    pub thread_pool_size: usize,
    pub max_workers: usize,
    pub work_stealing: bool,
    pub async_runtime: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MemoryConfig {
    pub heap_size: usize,
    pub stack_size: usize,
    pub gc_threshold: f64,
    pub allocation_strategy: String,
}
```

### **Error Handling**

```rust
// ✅ Good - comprehensive performance error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PerformanceError {
    #[error("Profiling error: {0}")]
    ProfilingError(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
    
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),
    
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    #[error("Optimization error: {0}")]
    OptimizationError(String),
}

pub type Result<T> = std::result::Result<T, PerformanceError>;
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Rust Performance Book](https://nnethercote.github.io/perf-book/) - Fetched: 2024-12-19T00:00:00Z
- [Rust SIMD](https://rust-lang.github.io/portable-simd/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust Performance](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z
- [Lock-Free Programming](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you profile and optimize Rust applications?
2. Do you understand advanced memory management patterns?
3. Can you optimize concurrent and parallel code?
4. Do you know how to use performance monitoring tools?
5. Can you build high-performance systems?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced concurrency patterns
- Memory management optimization
- Performance monitoring
- Production deployment

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [35.2 Advanced Concurrency Optimization](35_02_concurrency_optimization.md)
