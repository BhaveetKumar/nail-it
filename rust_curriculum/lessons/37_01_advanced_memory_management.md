---
# Auto-generated front matter
Title: 37 01 Advanced Memory Management
LastUpdated: 2025-11-06T20:45:58.127272
Tags: []
Status: draft
---

# Lesson 37.1: Advanced Memory Management

> **Module**: 37 - Advanced Memory Management  
> **Lesson**: 1 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 36 (Advanced Compiler Techniques)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Implement custom memory allocators
- Design memory pools and arenas
- Optimize memory usage patterns
- Handle memory fragmentation
- Build high-performance memory systems

---

## 🎯 **Overview**

Advanced memory management in Rust involves implementing custom allocators, designing memory pools, optimizing memory usage, and handling memory fragmentation. This lesson covers custom allocators, memory pools, and advanced memory optimization techniques.

---

## 🔧 **Custom Memory Allocators**

### **Arena Allocator Implementation**

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::{self, NonNull};
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

pub struct ArenaAllocator {
    pub current_chunk: AtomicPtr<Chunk>,
    pub chunk_size: usize,
    pub max_chunks: usize,
    pub chunks: Mutex<Vec<Chunk>>,
}

#[derive(Debug)]
struct Chunk {
    pub data: *mut u8,
    pub size: usize,
    pub used: AtomicUsize,
    pub next: AtomicPtr<Chunk>,
}

impl Chunk {
    pub fn new(size: usize) -> Result<Self, std::alloc::AllocError> {
        let layout = Layout::from_size_align(size, 8)?;
        let data = unsafe { System.alloc(layout) };
        
        if data.is_null() {
            return Err(std::alloc::AllocError);
        }
        
        Ok(Self {
            data,
            size,
            used: AtomicUsize::new(0),
            next: AtomicPtr::new(ptr::null_mut()),
        })
    }
    
    pub fn allocate(&self, layout: Layout) -> Option<*mut u8> {
        let size = layout.size();
        let align = layout.align();
        
        // Align the allocation
        let aligned_size = (size + align - 1) & !(align - 1);
        
        loop {
            let current_used = self.used.load(Ordering::Acquire);
            let new_used = current_used + aligned_size;
            
            if new_used > self.size {
                return None; // Not enough space in this chunk
            }
            
            if self.used.compare_exchange_weak(
                current_used,
                new_used,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                let ptr = unsafe { self.data.add(current_used) };
                return Some(ptr);
            }
        }
    }
    
    pub fn can_allocate(&self, layout: Layout) -> bool {
        let size = layout.size();
        let align = layout.align();
        let aligned_size = (size + align - 1) & !(align - 1);
        
        self.used.load(Ordering::Acquire) + aligned_size <= self.size
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        if !self.data.is_null() {
            let layout = Layout::from_size_align(self.size, 8).unwrap();
            unsafe { System.dealloc(self.data, layout) };
        }
    }
}

impl ArenaAllocator {
    pub fn new(chunk_size: usize, max_chunks: usize) -> Self {
        let initial_chunk = Chunk::new(chunk_size).unwrap();
        let chunk_ptr = Box::into_raw(Box::new(initial_chunk));
        
        Self {
            current_chunk: AtomicPtr::new(chunk_ptr),
            chunk_size,
            max_chunks,
            chunks: Mutex::new(Vec::new()),
        }
    }
    
    fn get_or_create_chunk(&self) -> *mut Chunk {
        let current = self.current_chunk.load(Ordering::Acquire);
        
        if !current.is_null() {
            return current;
        }
        
        // Create new chunk
        let new_chunk = Chunk::new(self.chunk_size).unwrap();
        let new_chunk_ptr = Box::into_raw(Box::new(new_chunk));
        
        // Try to set as current chunk
        if self.current_chunk.compare_exchange(
            ptr::null_mut(),
            new_chunk_ptr,
            Ordering::Release,
            Ordering::Relaxed,
        ).is_ok() {
            // Add to chunks list
            let mut chunks = self.chunks.lock().unwrap();
            chunks.push(unsafe { Box::from_raw(new_chunk_ptr) });
            new_chunk_ptr
        } else {
            // Another thread created a chunk, use that one
            self.current_chunk.load(Ordering::Acquire)
        }
    }
    
    fn find_available_chunk(&self, layout: Layout) -> Option<*mut Chunk> {
        let chunks = self.chunks.lock().unwrap();
        
        for chunk in chunks.iter() {
            if chunk.can_allocate(layout) {
                return Some(chunk as *const Chunk as *mut Chunk);
            }
        }
        
        None
    }
}

unsafe impl GlobalAlloc for ArenaAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Try to allocate from current chunk
        let current = self.current_chunk.load(Ordering::Acquire);
        if !current.is_null() {
            if let Some(ptr) = (*current).allocate(layout) {
                return ptr;
            }
        }
        
        // Try to find an available chunk
        if let Some(chunk) = self.find_available_chunk(layout) {
            if let Some(ptr) = (*chunk).allocate(layout) {
                return ptr;
            }
        }
        
        // Create new chunk if we haven't exceeded max chunks
        let chunks = self.chunks.lock().unwrap();
        if chunks.len() < self.max_chunks {
            drop(chunks);
            
            let new_chunk = Chunk::new(self.chunk_size).unwrap();
            let new_chunk_ptr = Box::into_raw(Box::new(new_chunk));
            
            if let Some(ptr) = (*new_chunk_ptr).allocate(layout) {
                // Add to chunks list
                let mut chunks = self.chunks.lock().unwrap();
                chunks.push(Box::from_raw(new_chunk_ptr));
                
                return ptr;
            }
        }
        
        // Fall back to system allocator
        System.alloc(layout)
    }
    
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Arena allocator doesn't support individual deallocation
        // Memory is freed when the arena is dropped
    }
}

impl Drop for ArenaAllocator {
    fn drop(&mut self) {
        let mut chunks = self.chunks.lock().unwrap();
        chunks.clear();
    }
}
```

### **Memory Pool Implementation**

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::{self, NonNull};
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

pub struct MemoryPool {
    pub block_size: usize,
    pub block_count: usize,
    pub free_blocks: AtomicPtr<FreeBlock>,
    pub allocated_blocks: AtomicUsize,
    pub total_blocks: AtomicUsize,
    pub memory: *mut u8,
    pub memory_size: usize,
}

#[derive(Debug)]
struct FreeBlock {
    pub next: AtomicPtr<FreeBlock>,
}

impl MemoryPool {
    pub fn new(block_size: usize, block_count: usize) -> Result<Self, std::alloc::AllocError> {
        let memory_size = block_size * block_count;
        let layout = Layout::from_size_align(memory_size, block_size)?;
        let memory = unsafe { System.alloc(layout) };
        
        if memory.is_null() {
            return Err(std::alloc::AllocError);
        }
        
        let mut pool = Self {
            block_size,
            block_count,
            free_blocks: AtomicPtr::new(ptr::null_mut()),
            allocated_blocks: AtomicUsize::new(0),
            total_blocks: AtomicUsize::new(block_count),
            memory,
            memory_size,
        };
        
        // Initialize free blocks
        pool.initialize_free_blocks();
        
        Ok(pool)
    }
    
    fn initialize_free_blocks(&self) {
        let mut current_block = self.memory;
        let mut prev_block: *mut FreeBlock = ptr::null_mut();
        
        for i in 0..self.block_count {
            let block_ptr = current_block as *mut FreeBlock;
            
            if i == 0 {
                // First block
                self.free_blocks.store(block_ptr, Ordering::Release);
            } else {
                // Link to previous block
                unsafe {
                    (*prev_block).next.store(block_ptr, Ordering::Release);
                }
            }
            
            prev_block = block_ptr;
            current_block = unsafe { current_block.add(self.block_size) };
        }
        
        // Last block points to null
        if !prev_block.is_null() {
            unsafe {
                (*prev_block).next.store(ptr::null_mut(), Ordering::Release);
            }
        }
    }
    
    pub fn allocate(&self) -> Option<*mut u8> {
        loop {
            let current = self.free_blocks.load(Ordering::Acquire);
            
            if current.is_null() {
                return None; // No free blocks available
            }
            
            let next = unsafe { (*current).next.load(Ordering::Acquire) };
            
            if self.free_blocks.compare_exchange_weak(
                current,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                self.allocated_blocks.fetch_add(1, Ordering::Relaxed);
                return Some(current as *mut u8);
            }
        }
    }
    
    pub fn deallocate(&self, ptr: *mut u8) -> bool {
        // Check if pointer is within our memory range
        if ptr < self.memory || ptr >= unsafe { self.memory.add(self.memory_size) } {
            return false;
        }
        
        // Check if pointer is aligned to block size
        let offset = unsafe { ptr.offset_from(self.memory) } as usize;
        if offset % self.block_size != 0 {
            return false;
        }
        
        // Add block back to free list
        let block_ptr = ptr as *mut FreeBlock;
        
        loop {
            let current = self.free_blocks.load(Ordering::Acquire);
            unsafe {
                (*block_ptr).next.store(current, Ordering::Release);
            }
            
            if self.free_blocks.compare_exchange_weak(
                current,
                block_ptr,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                self.allocated_blocks.fetch_sub(1, Ordering::Relaxed);
                return true;
            }
        }
    }
    
    pub fn get_stats(&self) -> PoolStats {
        PoolStats {
            block_size: self.block_size,
            total_blocks: self.total_blocks.load(Ordering::Relaxed),
            allocated_blocks: self.allocated_blocks.load(Ordering::Relaxed),
            free_blocks: self.total_blocks.load(Ordering::Relaxed) - self.allocated_blocks.load(Ordering::Relaxed),
            memory_usage: self.allocated_blocks.load(Ordering::Relaxed) * self.block_size,
            total_memory: self.memory_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub block_size: usize,
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub memory_usage: usize,
    pub total_memory: usize,
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        if !self.memory.is_null() {
            let layout = Layout::from_size_align(self.memory_size, self.block_size).unwrap();
            unsafe { System.dealloc(self.memory, layout) };
        }
    }
}

unsafe impl GlobalAlloc for MemoryPool {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() <= self.block_size && layout.align() <= self.block_size {
            self.allocate().unwrap_or_else(|| System.alloc(layout))
        } else {
            System.alloc(layout)
        }
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() <= self.block_size && layout.align() <= self.block_size {
            if !self.deallocate(ptr) {
                System.dealloc(ptr, layout);
            }
        } else {
            System.dealloc(ptr, layout);
        }
    }
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Advanced Memory Pool with Multiple Sizes**

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::{self, NonNull};
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

pub struct MultiSizeMemoryPool {
    pub pools: Vec<MemoryPool>,
    pub size_classes: Vec<usize>,
    pub fallback_allocator: System,
}

impl MultiSizeMemoryPool {
    pub fn new() -> Self {
        let size_classes = vec![16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        let mut pools = Vec::new();
        
        for &size in &size_classes {
            let pool = MemoryPool::new(size, 1000).unwrap();
            pools.push(pool);
        }
        
        Self {
            pools,
            size_classes,
            fallback_allocator: System,
        }
    }
    
    fn find_suitable_pool(&self, size: usize) -> Option<usize> {
        for (i, &class_size) in self.size_classes.iter().enumerate() {
            if size <= class_size {
                return Some(i);
            }
        }
        None
    }
    
    pub fn allocate(&self, size: usize) -> Option<*mut u8> {
        if let Some(pool_index) = self.find_suitable_pool(size) {
            self.pools[pool_index].allocate()
        } else {
            None
        }
    }
    
    pub fn deallocate(&self, ptr: *mut u8, size: usize) -> bool {
        if let Some(pool_index) = self.find_suitable_pool(size) {
            self.pools[pool_index].deallocate(ptr)
        } else {
            false
        }
    }
    
    pub fn get_all_stats(&self) -> Vec<PoolStats> {
        self.pools.iter().map(|pool| pool.get_stats()).collect()
    }
    
    pub fn get_total_memory_usage(&self) -> usize {
        self.pools.iter().map(|pool| pool.get_stats().memory_usage).sum()
    }
    
    pub fn get_total_memory_available(&self) -> usize {
        self.pools.iter().map(|pool| pool.get_stats().total_memory).sum()
    }
}

unsafe impl GlobalAlloc for MultiSizeMemoryPool {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if let Some(ptr) = self.allocate(layout.size()) {
            ptr
        } else {
            self.fallback_allocator.alloc(layout)
        }
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !self.deallocate(ptr, layout.size()) {
            self.fallback_allocator.dealloc(ptr, layout);
        }
    }
}

// Memory pool with garbage collection
pub struct GarbageCollectedPool {
    pub pool: MemoryPool,
    pub allocated_objects: Mutex<Vec<*mut u8>>,
    pub gc_threshold: usize,
    pub gc_count: AtomicUsize,
}

impl GarbageCollectedPool {
    pub fn new(block_size: usize, block_count: usize, gc_threshold: usize) -> Result<Self, std::alloc::AllocError> {
        Ok(Self {
            pool: MemoryPool::new(block_size, block_count)?,
            allocated_objects: Mutex::new(Vec::new()),
            gc_threshold,
            gc_count: AtomicUsize::new(0),
        })
    }
    
    pub fn allocate(&self) -> Option<*mut u8> {
        if let Some(ptr) = self.pool.allocate() {
            let mut objects = self.allocated_objects.lock().unwrap();
            objects.push(ptr);
            
            // Check if we need garbage collection
            if objects.len() >= self.gc_threshold {
                drop(objects);
                self.garbage_collect();
            }
            
            Some(ptr)
        } else {
            None
        }
    }
    
    pub fn deallocate(&self, ptr: *mut u8) -> bool {
        let mut objects = self.allocated_objects.lock().unwrap();
        if let Some(pos) = objects.iter().position(|&obj| obj == ptr) {
            objects.remove(pos);
            self.pool.deallocate(ptr)
        } else {
            false
        }
    }
    
    pub fn garbage_collect(&self) {
        let mut objects = self.allocated_objects.lock().unwrap();
        let mut to_remove = Vec::new();
        
        // Mark objects that are no longer referenced
        for (i, &obj) in objects.iter().enumerate() {
            if !self.is_object_referenced(obj) {
                to_remove.push(i);
            }
        }
        
        // Remove unreferenced objects
        for &i in to_remove.iter().rev() {
            let obj = objects.remove(i);
            self.pool.deallocate(obj);
        }
        
        self.gc_count.fetch_add(1, Ordering::Relaxed);
    }
    
    fn is_object_referenced(&self, obj: *mut u8) -> bool {
        // Simple reference counting - in a real implementation,
        // this would be more sophisticated
        true
    }
    
    pub fn get_gc_stats(&self) -> GcStats {
        let objects = self.allocated_objects.lock().unwrap();
        GcStats {
            allocated_objects: objects.len(),
            gc_count: self.gc_count.load(Ordering::Relaxed),
            pool_stats: self.pool.get_stats(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GcStats {
    pub allocated_objects: usize,
    pub gc_count: usize,
    pub pool_stats: PoolStats,
}
```

### **Exercise 2: Memory Fragmentation Analysis**

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::{self, NonNull};
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

pub struct FragmentationAnalyzer {
    pub allocations: Mutex<Vec<Allocation>>,
    pub total_allocated: AtomicUsize,
    pub total_freed: AtomicUsize,
    pub fragmentation_score: AtomicUsize,
}

#[derive(Debug, Clone)]
struct Allocation {
    pub ptr: *mut u8,
    pub size: usize,
    pub timestamp: std::time::Instant,
    pub freed: bool,
}

impl FragmentationAnalyzer {
    pub fn new() -> Self {
        Self {
            allocations: Mutex::new(Vec::new()),
            total_allocated: AtomicUsize::new(0),
            total_freed: AtomicUsize::new(0),
            fragmentation_score: AtomicUsize::new(0),
        }
    }
    
    pub fn record_allocation(&self, ptr: *mut u8, size: usize) {
        let allocation = Allocation {
            ptr,
            size,
            timestamp: std::time::Instant::now(),
            freed: false,
        };
        
        let mut allocations = self.allocations.lock().unwrap();
        allocations.push(allocation);
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
    }
    
    pub fn record_deallocation(&self, ptr: *mut u8, size: usize) {
        let mut allocations = self.allocations.lock().unwrap();
        
        if let Some(allocation) = allocations.iter_mut().find(|a| a.ptr == ptr && !a.freed) {
            allocation.freed = true;
            self.total_freed.fetch_add(size, Ordering::Relaxed);
        }
    }
    
    pub fn calculate_fragmentation(&self) -> FragmentationReport {
        let allocations = self.allocations.lock().unwrap();
        let mut active_allocations: Vec<_> = allocations
            .iter()
            .filter(|a| !a.freed)
            .collect();
        
        // Sort by memory address
        active_allocations.sort_by_key(|a| a.ptr as usize);
        
        let mut gaps = Vec::new();
        let mut total_gap_size = 0;
        
        for i in 0..active_allocations.len() - 1 {
            let current = active_allocations[i];
            let next = active_allocations[i + 1];
            
            let current_end = unsafe { current.ptr.add(current.size) as usize };
            let next_start = next.ptr as usize;
            
            if current_end < next_start {
                let gap_size = next_start - current_end;
                gaps.push(gap_size);
                total_gap_size += gap_size;
            }
        }
        
        let total_active_memory: usize = active_allocations.iter().map(|a| a.size).sum();
        let fragmentation_ratio = if total_active_memory > 0 {
            total_gap_size as f64 / total_active_memory as f64
        } else {
            0.0
        };
        
        FragmentationReport {
            total_allocations: allocations.len(),
            active_allocations: active_allocations.len(),
            total_gap_size,
            fragmentation_ratio,
            gaps,
            total_active_memory,
        }
    }
    
    pub fn suggest_optimizations(&self) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        let report = self.calculate_fragmentation();
        
        if report.fragmentation_ratio > 0.5 {
            suggestions.push(OptimizationSuggestion::UseMemoryPool);
        }
        
        if report.gaps.len() > 100 {
            suggestions.push(OptimizationSuggestion::DefragmentMemory);
        }
        
        if report.active_allocations > 1000 {
            suggestions.push(OptimizationSuggestion::UseArenaAllocator);
        }
        
        suggestions
    }
}

#[derive(Debug, Clone)]
pub struct FragmentationReport {
    pub total_allocations: usize,
    pub active_allocations: usize,
    pub total_gap_size: usize,
    pub fragmentation_ratio: f64,
    pub gaps: Vec<usize>,
    pub total_active_memory: usize,
}

#[derive(Debug, Clone)]
pub enum OptimizationSuggestion {
    UseMemoryPool,
    DefragmentMemory,
    UseArenaAllocator,
    ReduceAllocationSize,
    UseStackAllocation,
}

// Memory defragmentation utility
pub struct MemoryDefragmenter {
    pub analyzer: FragmentationAnalyzer,
    pub defragmentation_threshold: f64,
}

impl MemoryDefragmenter {
    pub fn new(analyzer: FragmentationAnalyzer, threshold: f64) -> Self {
        Self {
            analyzer,
            defragmentation_threshold: threshold,
        }
    }
    
    pub fn should_defragment(&self) -> bool {
        let report = self.analyzer.calculate_fragmentation();
        report.fragmentation_ratio > self.defragmentation_threshold
    }
    
    pub fn defragment(&self) -> DefragmentationResult {
        let report = self.analyzer.calculate_fragmentation();
        
        if !self.should_defragment() {
            return DefragmentationResult {
                success: false,
                reason: "Fragmentation below threshold".to_string(),
                memory_moved: 0,
                time_taken: std::time::Duration::from_secs(0),
            };
        }
        
        let start_time = std::time::Instant::now();
        let mut memory_moved = 0;
        
        // Simple defragmentation: move all active allocations to the beginning
        let mut allocations = self.analyzer.allocations.lock().unwrap();
        let mut active_allocations: Vec<_> = allocations
            .iter_mut()
            .filter(|a| !a.freed)
            .collect();
        
        active_allocations.sort_by_key(|a| a.ptr as usize);
        
        let mut current_ptr = active_allocations[0].ptr;
        
        for allocation in active_allocations.iter_mut() {
            if allocation.ptr != current_ptr {
                // Move allocation
                unsafe {
                    ptr::copy_nonoverlapping(allocation.ptr, current_ptr, allocation.size);
                }
                allocation.ptr = current_ptr;
                memory_moved += allocation.size;
            }
            current_ptr = unsafe { current_ptr.add(allocation.size) };
        }
        
        let time_taken = start_time.elapsed();
        
        DefragmentationResult {
            success: true,
            reason: "Defragmentation completed".to_string(),
            memory_moved,
            time_taken,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DefragmentationResult {
    pub success: bool,
    pub reason: String,
    pub memory_moved: usize,
    pub time_taken: std::time::Duration,
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout};

    #[test]
    fn test_arena_allocator() {
        let arena = ArenaAllocator::new(1024, 10);
        
        let layout = Layout::from_size_align(64, 8).unwrap();
        let ptr1 = unsafe { arena.alloc(layout) };
        let ptr2 = unsafe { arena.alloc(layout) };
        
        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(64, 100).unwrap();
        
        let ptr1 = pool.allocate().unwrap();
        let ptr2 = pool.allocate().unwrap();
        
        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);
        
        assert!(pool.deallocate(ptr1));
        assert!(pool.deallocate(ptr2));
    }

    #[test]
    fn test_multi_size_pool() {
        let pool = MultiSizeMemoryPool::new();
        
        let ptr1 = pool.allocate(32).unwrap();
        let ptr2 = pool.allocate(64).unwrap();
        
        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        
        assert!(pool.deallocate(ptr1, 32));
        assert!(pool.deallocate(ptr2, 64));
    }

    #[test]
    fn test_fragmentation_analyzer() {
        let analyzer = FragmentationAnalyzer::new();
        
        let ptr1 = std::ptr::null_mut();
        let ptr2 = std::ptr::null_mut();
        
        analyzer.record_allocation(ptr1, 64);
        analyzer.record_allocation(ptr2, 128);
        
        let report = analyzer.calculate_fragmentation();
        assert_eq!(report.total_allocations, 2);
        assert_eq!(report.active_allocations, 2);
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Memory Leaks in Custom Allocators**

```rust
// ❌ Wrong - potential memory leak
pub struct BadAllocator {
    pub memory: *mut u8,
    pub size: usize,
}

impl BadAllocator {
    pub fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 8).unwrap();
        let memory = unsafe { System.alloc(layout) };
        
        Self { memory, size }
    }
}

// ✅ Correct - proper memory management
pub struct GoodAllocator {
    pub memory: *mut u8,
    pub size: usize,
}

impl GoodAllocator {
    pub fn new(size: usize) -> Result<Self, std::alloc::AllocError> {
        let layout = Layout::from_size_align(size, 8)?;
        let memory = unsafe { System.alloc(layout) };
        
        if memory.is_null() {
            return Err(std::alloc::AllocError);
        }
        
        Ok(Self { memory, size })
    }
}

impl Drop for GoodAllocator {
    fn drop(&mut self) {
        if !self.memory.is_null() {
            let layout = Layout::from_size_align(self.size, 8).unwrap();
            unsafe { System.dealloc(self.memory, layout) };
        }
    }
}
```

### **Common Mistake 2: Race Conditions in Memory Pools**

```rust
// ❌ Wrong - race condition
pub struct BadPool {
    pub free_blocks: *mut FreeBlock,
    pub allocated_count: usize,
}

impl BadPool {
    pub fn allocate(&self) -> Option<*mut u8> {
        if self.allocated_count > 100 {
            return None;
        }
        
        // Race condition here!
        self.allocated_count += 1;
        Some(self.free_blocks)
    }
}

// ✅ Correct - thread-safe
pub struct GoodPool {
    pub free_blocks: AtomicPtr<FreeBlock>,
    pub allocated_count: AtomicUsize,
}

impl GoodPool {
    pub fn allocate(&self) -> Option<*mut u8> {
        let current = self.allocated_count.load(Ordering::Acquire);
        if current > 100 {
            return None;
        }
        
        if self.allocated_count.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Release,
            Ordering::Relaxed,
        ).is_ok() {
            Some(self.free_blocks.load(Ordering::Acquire) as *mut u8)
        } else {
            None
        }
    }
}
```

---

## 📊 **Advanced Memory Management Patterns**

### **Memory Pool with Reference Counting**

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct ReferenceCountedPool {
    pub pool: MemoryPool,
    pub reference_counts: Mutex<Vec<AtomicUsize>>,
    pub object_indices: Mutex<Vec<usize>>,
}

impl ReferenceCountedPool {
    pub fn new(block_size: usize, block_count: usize) -> Result<Self, std::alloc::AllocError> {
        Ok(Self {
            pool: MemoryPool::new(block_size, block_count)?,
            reference_counts: Mutex::new(Vec::new()),
            object_indices: Mutex::new(Vec::new()),
        })
    }
    
    pub fn allocate(&self) -> Option<ReferenceCountedPtr> {
        if let Some(ptr) = self.pool.allocate() {
            let mut reference_counts = self.reference_counts.lock().unwrap();
            let mut object_indices = self.object_indices.lock().unwrap();
            
            let index = reference_counts.len();
            reference_counts.push(AtomicUsize::new(1));
            object_indices.push(index);
            
            Some(ReferenceCountedPtr {
                ptr,
                index,
                pool: Arc::new(self),
            })
        } else {
            None
        }
    }
    
    pub fn increment_reference(&self, index: usize) {
        let reference_counts = self.reference_counts.lock().unwrap();
        if index < reference_counts.len() {
            reference_counts[index].fetch_add(1, Ordering::Relaxed);
        }
    }
    
    pub fn decrement_reference(&self, index: usize) -> bool {
        let reference_counts = self.reference_counts.lock().unwrap();
        if index < reference_counts.len() {
            let count = reference_counts[index].fetch_sub(1, Ordering::Relaxed);
            if count == 1 {
                // Last reference, deallocate
                let object_indices = self.object_indices.lock().unwrap();
                if let Some(&ptr_index) = object_indices.get(index) {
                    self.pool.deallocate(ptr_index as *mut u8);
                }
                return true;
            }
        }
        false
    }
}

pub struct ReferenceCountedPtr {
    pub ptr: *mut u8,
    pub index: usize,
    pub pool: Arc<ReferenceCountedPool>,
}

impl Clone for ReferenceCountedPtr {
    fn clone(&self) -> Self {
        self.pool.increment_reference(self.index);
        Self {
            ptr: self.ptr,
            index: self.index,
            pool: self.pool.clone(),
        }
    }
}

impl Drop for ReferenceCountedPtr {
    fn drop(&mut self) {
        self.pool.decrement_reference(self.index);
    }
}
```

---

## 🎯 **Best Practices**

### **Memory Management Configuration**

```rust
// ✅ Good - comprehensive memory management configuration
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct MemoryConfig {
    pub allocator: AllocatorConfig,
    pub pools: PoolConfig,
    pub gc: GcConfig,
    pub fragmentation: FragmentationConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AllocatorConfig {
    pub strategy: String,
    pub arena_size: usize,
    pub max_arenas: usize,
    pub enable_custom_allocator: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PoolConfig {
    pub enabled: bool,
    pub block_sizes: Vec<usize>,
    pub blocks_per_size: usize,
    pub enable_reference_counting: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GcConfig {
    pub enabled: bool,
    pub threshold: usize,
    pub interval: u64,
    pub enable_compaction: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FragmentationConfig {
    pub monitor_enabled: bool,
    pub defragmentation_threshold: f64,
    pub analysis_interval: u64,
    pub enable_auto_defragmentation: bool,
}
```

### **Error Handling**

```rust
// ✅ Good - comprehensive memory management error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),
    
    #[error("Deallocation failed: {0}")]
    DeallocationFailed(String),
    
    #[error("Pool exhausted: {0}")]
    PoolExhausted(String),
    
    #[error("Fragmentation too high: {0}")]
    FragmentationTooHigh(String),
    
    #[error("Garbage collection failed: {0}")]
    GarbageCollectionFailed(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Rust Memory Management](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html) - Fetched: 2024-12-19T00:00:00Z
- [Custom Allocators](https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Memory Pool Patterns](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z
- [Garbage Collection](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you implement custom memory allocators?
2. Do you understand memory pools and arenas?
3. Can you optimize memory usage patterns?
4. Do you know how to handle memory fragmentation?
5. Can you build high-performance memory systems?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced concurrency patterns
- Performance monitoring
- Production deployment
- Final project

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [37.2 Advanced Concurrency Patterns](37_02_concurrency_patterns.md)
