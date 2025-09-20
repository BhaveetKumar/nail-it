# Memory Management

## Overview

This module covers advanced memory management concepts including virtual memory, paging, segmentation, memory allocation algorithms, and garbage collection. These concepts are essential for understanding how operating systems efficiently manage memory resources.

## Table of Contents

1. [Virtual Memory](#virtual-memory)
2. [Paging Systems](#paging-systems)
3. [Segmentation](#segmentation)
4. [Memory Allocation](#memory-allocation)
5. [Garbage Collection](#garbage-collection)
6. [Memory Optimization](#memory-optimization)
7. [Applications](#applications)
8. [Complexity Analysis](#complexity-analysis)
9. [Follow-up Questions](#follow-up-questions)

## Virtual Memory

### Theory

Virtual memory is a memory management technique that provides an abstraction of physical memory, allowing processes to use more memory than physically available. It uses disk storage as an extension of RAM.

### Page Table Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
)

type PageTableEntry struct {
    FrameNumber int
    Valid       bool
    Dirty       bool
    Referenced  bool
    AccessTime  time.Time
}

type PageTable struct {
    entries map[int]*PageTableEntry
    size    int
}

func NewPageTable(size int) *PageTable {
    return &PageTable{
        entries: make(map[int]*PageTableEntry),
        size:    size,
    }
}

func (pt *PageTable) GetPage(pageNumber int) (*PageTableEntry, bool) {
    entry, exists := pt.entries[pageNumber]
    if exists && entry.Valid {
        entry.Referenced = true
        entry.AccessTime = time.Now()
        return entry, true
    }
    return nil, false
}

func (pt *PageTable) SetPage(pageNumber, frameNumber int) {
    pt.entries[pageNumber] = &PageTableEntry{
        FrameNumber: frameNumber,
        Valid:       true,
        Dirty:       false,
        Referenced:  true,
        AccessTime:  time.Now(),
    }
}

func (pt *PageTable) MarkDirty(pageNumber int) {
    if entry, exists := pt.entries[pageNumber]; exists {
        entry.Dirty = true
    }
}

func (pt *PageTable) Invalidate(pageNumber int) {
    if entry, exists := pt.entries[pageNumber]; exists {
        entry.Valid = false
    }
}

type VirtualMemoryManager struct {
    pageTable     *PageTable
    physicalMemory []byte
    pageSize      int
    numFrames     int
    freeFrames    []int
    frameMap      map[int]int // frame -> page mapping
}

func NewVirtualMemoryManager(pageSize, numFrames int) *VirtualMemoryManager {
    freeFrames := make([]int, numFrames)
    for i := 0; i < numFrames; i++ {
        freeFrames[i] = i
    }
    
    return &VirtualMemoryManager{
        pageTable:      NewPageTable(1024), // 1024 pages
        physicalMemory: make([]byte, pageSize*numFrames),
        pageSize:       pageSize,
        numFrames:      numFrames,
        freeFrames:     freeFrames,
        frameMap:       make(map[int]int),
    }
}

func (vmm *VirtualMemoryManager) AllocateFrame() int {
    if len(vmm.freeFrames) == 0 {
        return -1 // No free frames
    }
    
    frame := vmm.freeFrames[0]
    vmm.freeFrames = vmm.freeFrames[1:]
    return frame
}

func (vmm *VirtualMemoryManager) FreeFrame(frame int) {
    vmm.freeFrames = append(vmm.freeFrames, frame)
    delete(vmm.frameMap, frame)
}

func (vmm *VirtualMemoryManager) ReadVirtualAddress(virtualAddr int) (byte, error) {
    pageNumber := virtualAddr / vmm.pageSize
    offset := virtualAddr % vmm.pageSize
    
    entry, exists := vmm.pageTable.GetPage(pageNumber)
    if !exists {
        // Page fault - load page
        frame := vmm.handlePageFault(pageNumber)
        if frame == -1 {
            return 0, fmt.Errorf("page fault handling failed")
        }
        entry, _ = vmm.pageTable.GetPage(pageNumber)
    }
    
    physicalAddr := entry.FrameNumber*vmm.pageSize + offset
    return vmm.physicalMemory[physicalAddr], nil
}

func (vmm *VirtualMemoryManager) WriteVirtualAddress(virtualAddr int, value byte) error {
    pageNumber := virtualAddr / vmm.pageSize
    offset := virtualAddr % vmm.pageSize
    
    entry, exists := vmm.pageTable.GetPage(pageNumber)
    if !exists {
        // Page fault - load page
        frame := vmm.handlePageFault(pageNumber)
        if frame == -1 {
            return fmt.Errorf("page fault handling failed")
        }
        entry, _ = vmm.pageTable.GetPage(pageNumber)
    }
    
    physicalAddr := entry.FrameNumber*vmm.pageSize + offset
    vmm.physicalMemory[physicalAddr] = value
    vmm.pageTable.MarkDirty(pageNumber)
    
    return nil
}

func (vmm *VirtualMemoryManager) handlePageFault(pageNumber int) int {
    fmt.Printf("Page fault for page %d\n", pageNumber)
    
    // Try to allocate a free frame
    frame := vmm.AllocateFrame()
    if frame != -1 {
        vmm.pageTable.SetPage(pageNumber, frame)
        vmm.frameMap[frame] = pageNumber
        return frame
    }
    
    // No free frames - need to evict a page
    frame = vmm.evictPage()
    if frame == -1 {
        return -1
    }
    
    // Update page table
    oldPage := vmm.frameMap[frame]
    vmm.pageTable.Invalidate(oldPage)
    
    vmm.pageTable.SetPage(pageNumber, frame)
    vmm.frameMap[frame] = pageNumber
    
    return frame
}

func (vmm *VirtualMemoryManager) evictPage() int {
    // Simple FIFO eviction
    for frame, pageNumber := range vmm.frameMap {
        vmm.FreeFrame(frame)
        return frame
    }
    return -1
}

func main() {
    vmm := NewVirtualMemoryManager(4096, 4) // 4KB pages, 4 frames
    
    // Simulate memory access
    fmt.Println("Virtual Memory Manager Demo:")
    
    // Write to virtual address
    err := vmm.WriteVirtualAddress(0, 42)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Println("Successfully wrote to virtual address 0")
    }
    
    // Read from virtual address
    value, err := vmm.ReadVirtualAddress(0)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Read value %d from virtual address 0\n", value)
    }
    
    // Simulate more memory access to trigger page faults
    for i := 0; i < 10; i++ {
        virtualAddr := i * 4096
        vmm.WriteVirtualAddress(virtualAddr, byte(i))
        fmt.Printf("Wrote %d to virtual address %d\n", i, virtualAddr)
    }
}
```

## Paging Systems

### Theory

Paging is a memory management scheme that eliminates the need for contiguous allocation of physical memory. The physical memory is divided into fixed-size blocks called frames, and the logical memory is divided into pages of the same size.

### Page Replacement Algorithms

#### LRU (Least Recently Used) Implementation

##### Golang Implementation

```go
package main

import (
    "fmt"
    "time"
)

type LRUCache struct {
    capacity int
    cache    map[int]*Node
    head     *Node
    tail     *Node
}

type Node struct {
    key   int
    value int
    prev  *Node
    next  *Node
    time  time.Time
}

func NewLRUCache(capacity int) *LRUCache {
    lru := &LRUCache{
        capacity: capacity,
        cache:    make(map[int]*Node),
    }
    
    // Create dummy head and tail nodes
    lru.head = &Node{}
    lru.tail = &Node{}
    lru.head.next = lru.tail
    lru.tail.prev = lru.head
    
    return lru
}

func (lru *LRUCache) Get(key int) (int, bool) {
    if node, exists := lru.cache[key]; exists {
        // Move to head (most recently used)
        lru.moveToHead(node)
        return node.value, true
    }
    return 0, false
}

func (lru *LRUCache) Put(key, value int) {
    if node, exists := lru.cache[key]; exists {
        // Update existing node
        node.value = value
        node.time = time.Now()
        lru.moveToHead(node)
    } else {
        // Create new node
        newNode := &Node{
            key:   key,
            value: value,
            time:  time.Now(),
        }
        
        if len(lru.cache) >= lru.capacity {
            // Remove least recently used
            lru.removeTail()
        }
        
        lru.cache[key] = newNode
        lru.addToHead(newNode)
    }
}

func (lru *LRUCache) addToHead(node *Node) {
    node.prev = lru.head
    node.next = lru.head.next
    lru.head.next.prev = node
    lru.head.next = node
}

func (lru *LRUCache) removeNode(node *Node) {
    node.prev.next = node.next
    node.next.prev = node.prev
}

func (lru *LRUCache) moveToHead(node *Node) {
    lru.removeNode(node)
    lru.addToHead(node)
}

func (lru *LRUCache) removeTail() {
    if lru.tail.prev != lru.head {
        lastNode := lru.tail.prev
        lru.removeNode(lastNode)
        delete(lru.cache, lastNode.key)
    }
}

func (lru *LRUCache) PrintCache() {
    fmt.Print("Cache: ")
    current := lru.head.next
    for current != lru.tail {
        fmt.Printf("(%d:%d) ", current.key, current.value)
        current = current.next
    }
    fmt.Println()
}

func main() {
    lru := NewLRUCache(3)
    
    fmt.Println("LRU Cache Demo:")
    
    lru.Put(1, 10)
    lru.Put(2, 20)
    lru.Put(3, 30)
    lru.PrintCache()
    
    value, found := lru.Get(1)
    if found {
        fmt.Printf("Got value %d for key 1\n", value)
    }
    lru.PrintCache()
    
    lru.Put(4, 40) // This should evict key 2
    lru.PrintCache()
    
    value, found = lru.Get(2)
    if found {
        fmt.Printf("Got value %d for key 2\n", value)
    } else {
        fmt.Println("Key 2 not found (was evicted)")
    }
}
```

#### FIFO (First In, First Out) Implementation

##### Golang Implementation

```go
package main

import (
    "fmt"
    "time"
)

type FIFOCache struct {
    capacity int
    cache    map[int]*Node
    queue    []*Node
    head     int
    tail     int
}

func NewFIFOCache(capacity int) *FIFOCache {
    return &FIFOCache{
        capacity: capacity,
        cache:    make(map[int]*Node),
        queue:    make([]*Node, capacity),
        head:     0,
        tail:     0,
    }
}

func (fifo *FIFOCache) Get(key int) (int, bool) {
    if node, exists := fifo.cache[key]; exists {
        return node.value, true
    }
    return 0, false
}

func (fifo *FIFOCache) Put(key, value int) {
    if node, exists := fifo.cache[key]; exists {
        // Update existing node
        node.value = value
        node.time = time.Now()
    } else {
        // Create new node
        newNode := &Node{
            key:   key,
            value: value,
            time:  time.Now(),
        }
        
        if len(fifo.cache) >= fifo.capacity {
            // Remove oldest node
            fifo.removeOldest()
        }
        
        fifo.cache[key] = newNode
        fifo.addToQueue(newNode)
    }
}

func (fifo *FIFOCache) addToQueue(node *Node) {
    fifo.queue[fifo.tail] = node
    fifo.tail = (fifo.tail + 1) % fifo.capacity
}

func (fifo *FIFOCache) removeOldest() {
    if fifo.head != fifo.tail {
        oldestNode := fifo.queue[fifo.head]
        delete(fifo.cache, oldestNode.key)
        fifo.head = (fifo.head + 1) % fifo.capacity
    }
}

func (fifo *FIFOCache) PrintCache() {
    fmt.Print("FIFO Cache: ")
    for key, node := range fifo.cache {
        fmt.Printf("(%d:%d) ", key, node.value)
    }
    fmt.Println()
}

func main() {
    fifo := NewFIFOCache(3)
    
    fmt.Println("FIFO Cache Demo:")
    
    fifo.Put(1, 10)
    fifo.Put(2, 20)
    fifo.Put(3, 30)
    fifo.PrintCache()
    
    value, found := fifo.Get(1)
    if found {
        fmt.Printf("Got value %d for key 1\n", value)
    }
    fifo.PrintCache()
    
    fifo.Put(4, 40) // This should evict key 1 (oldest)
    fifo.PrintCache()
    
    value, found = fifo.Get(1)
    if found {
        fmt.Printf("Got value %d for key 1\n", value)
    } else {
        fmt.Println("Key 1 not found (was evicted)")
    }
}
```

## Memory Allocation

### Theory

Memory allocation algorithms determine how to allocate and deallocate memory blocks efficiently. Common algorithms include first-fit, best-fit, and worst-fit.

### First Fit Algorithm

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
)

type MemoryBlock struct {
    start   int
    size    int
    free    bool
    processID int
}

type MemoryManager struct {
    blocks []MemoryBlock
    totalSize int
}

func NewMemoryManager(totalSize int) *MemoryManager {
    return &MemoryManager{
        blocks: []MemoryBlock{
            {start: 0, size: totalSize, free: true, processID: -1},
        },
        totalSize: totalSize,
    }
}

func (mm *MemoryManager) Allocate(processID, size int) bool {
    for i, block := range mm.blocks {
        if block.free && block.size >= size {
            // Found a suitable block
            if block.size == size {
                // Exact fit
                mm.blocks[i].free = false
                mm.blocks[i].processID = processID
            } else {
                // Split the block
                newBlock := MemoryBlock{
                    start:     block.start + size,
                    size:      block.size - size,
                    free:      true,
                    processID: -1,
                }
                
                mm.blocks[i].size = size
                mm.blocks[i].free = false
                mm.blocks[i].processID = processID
                
                // Insert new block after current block
                mm.blocks = append(mm.blocks[:i+1], append([]MemoryBlock{newBlock}, mm.blocks[i+1:]...)...)
            }
            
            fmt.Printf("Allocated %d bytes to process %d at address %d\n", 
                       size, processID, block.start)
            return true
        }
    }
    
    fmt.Printf("Failed to allocate %d bytes to process %d\n", size, processID)
    return false
}

func (mm *MemoryManager) Deallocate(processID int) bool {
    for i, block := range mm.blocks {
        if !block.free && block.processID == processID {
            mm.blocks[i].free = true
            mm.blocks[i].processID = -1
            
            // Merge with adjacent free blocks
            mm.mergeAdjacentBlocks()
            
            fmt.Printf("Deallocated memory for process %d\n", processID)
            return true
        }
    }
    
    fmt.Printf("Process %d not found for deallocation\n", processID)
    return false
}

func (mm *MemoryManager) mergeAdjacentBlocks() {
    // Sort blocks by start address
    sort.Slice(mm.blocks, func(i, j int) bool {
        return mm.blocks[i].start < mm.blocks[j].start
    })
    
    // Merge adjacent free blocks
    for i := 0; i < len(mm.blocks)-1; i++ {
        if mm.blocks[i].free && mm.blocks[i+1].free &&
           mm.blocks[i].start+mm.blocks[i].size == mm.blocks[i+1].start {
            
            // Merge blocks
            mm.blocks[i].size += mm.blocks[i+1].size
            
            // Remove the second block
            mm.blocks = append(mm.blocks[:i+1], mm.blocks[i+2:]...)
            i-- // Check again from current position
        }
    }
}

func (mm *MemoryManager) PrintMemory() {
    fmt.Println("Memory Layout:")
    for _, block := range mm.blocks {
        status := "FREE"
        if !block.free {
            status = fmt.Sprintf("Process %d", block.processID)
        }
        fmt.Printf("Address %d-%d: %s (Size: %d)\n", 
                   block.start, block.start+block.size-1, status, block.size)
    }
    fmt.Println()
}

func main() {
    mm := NewMemoryManager(1000)
    
    fmt.Println("First Fit Memory Allocation Demo:")
    mm.PrintMemory()
    
    // Allocate memory for processes
    mm.Allocate(1, 200)
    mm.PrintMemory()
    
    mm.Allocate(2, 300)
    mm.PrintMemory()
    
    mm.Allocate(3, 150)
    mm.PrintMemory()
    
    // Deallocate process 2
    mm.Deallocate(2)
    mm.PrintMemory()
    
    // Allocate new process
    mm.Allocate(4, 100)
    mm.PrintMemory()
}
```

### Best Fit Algorithm

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
)

type BestFitManager struct {
    blocks []MemoryBlock
    totalSize int
}

func NewBestFitManager(totalSize int) *BestFitManager {
    return &BestFitManager{
        blocks: []MemoryBlock{
            {start: 0, size: totalSize, free: true, processID: -1},
        },
        totalSize: totalSize,
    }
}

func (bfm *BestFitManager) Allocate(processID, size int) bool {
    // Find the smallest suitable block
    bestIndex := -1
    bestSize := bfm.totalSize + 1
    
    for i, block := range bfm.blocks {
        if block.free && block.size >= size && block.size < bestSize {
            bestIndex = i
            bestSize = block.size
        }
    }
    
    if bestIndex == -1 {
        fmt.Printf("Failed to allocate %d bytes to process %d\n", size, processID)
        return false
    }
    
    block := bfm.blocks[bestIndex]
    
    if block.size == size {
        // Exact fit
        bfm.blocks[bestIndex].free = false
        bfm.blocks[bestIndex].processID = processID
    } else {
        // Split the block
        newBlock := MemoryBlock{
            start:     block.start + size,
            size:      block.size - size,
            free:      true,
            processID: -1,
        }
        
        bfm.blocks[bestIndex].size = size
        bfm.blocks[bestIndex].free = false
        bfm.blocks[bestIndex].processID = processID
        
        // Insert new block after current block
        bfm.blocks = append(bfm.blocks[:bestIndex+1], 
                           append([]MemoryBlock{newBlock}, bfm.blocks[bestIndex+1:]...)...)
    }
    
    fmt.Printf("Allocated %d bytes to process %d at address %d (Best Fit)\n", 
               size, processID, block.start)
    return true
}

func (bfm *BestFitManager) Deallocate(processID int) bool {
    for i, block := range bfm.blocks {
        if !block.free && block.processID == processID {
            bfm.blocks[i].free = true
            bfm.blocks[i].processID = -1
            
            // Merge with adjacent free blocks
            bfm.mergeAdjacentBlocks()
            
            fmt.Printf("Deallocated memory for process %d\n", processID)
            return true
        }
    }
    
    fmt.Printf("Process %d not found for deallocation\n", processID)
    return false
}

func (bfm *BestFitManager) mergeAdjacentBlocks() {
    // Sort blocks by start address
    sort.Slice(bfm.blocks, func(i, j int) bool {
        return bfm.blocks[i].start < bfm.blocks[j].start
    })
    
    // Merge adjacent free blocks
    for i := 0; i < len(bfm.blocks)-1; i++ {
        if bfm.blocks[i].free && bfm.blocks[i+1].free &&
           bfm.blocks[i].start+bfm.blocks[i].size == bfm.blocks[i+1].start {
            
            // Merge blocks
            bfm.blocks[i].size += bfm.blocks[i+1].size
            
            // Remove the second block
            bfm.blocks = append(bfm.blocks[:i+1], bfm.blocks[i+2:]...)
            i-- // Check again from current position
        }
    }
}

func (bfm *BestFitManager) PrintMemory() {
    fmt.Println("Memory Layout (Best Fit):")
    for _, block := range bfm.blocks {
        status := "FREE"
        if !block.free {
            status = fmt.Sprintf("Process %d", block.processID)
        }
        fmt.Printf("Address %d-%d: %s (Size: %d)\n", 
                   block.start, block.start+block.size-1, status, block.size)
    }
    fmt.Println()
}

func main() {
    bfm := NewBestFitManager(1000)
    
    fmt.Println("Best Fit Memory Allocation Demo:")
    bfm.PrintMemory()
    
    // Allocate memory for processes
    bfm.Allocate(1, 200)
    bfm.PrintMemory()
    
    bfm.Allocate(2, 300)
    bfm.PrintMemory()
    
    bfm.Allocate(3, 150)
    bfm.PrintMemory()
    
    // Deallocate process 2
    bfm.Deallocate(2)
    bfm.PrintMemory()
    
    // Allocate new process
    bfm.Allocate(4, 100)
    bfm.PrintMemory()
}
```

## Garbage Collection

### Theory

Garbage collection is the automatic memory management technique that reclaims memory occupied by objects that are no longer referenced by the program.

### Mark and Sweep Algorithm

#### Golang Implementation

```go
package main

import (
    "fmt"
    "time"
)

type Object struct {
    ID        int
    Size      int
    Marked    bool
    ReferencedBy []*Object
    References   []*Object
}

type GarbageCollector struct {
    objects map[int]*Object
    nextID  int
}

func NewGarbageCollector() *GarbageCollector {
    return &GarbageCollector{
        objects: make(map[int]*Object),
        nextID:  1,
    }
}

func (gc *GarbageCollector) CreateObject(size int) *Object {
    obj := &Object{
        ID:        gc.nextID,
        Size:      size,
        Marked:    false,
        ReferencedBy: make([]*Object, 0),
        References:   make([]*Object, 0),
    }
    
    gc.objects[gc.nextID] = obj
    gc.nextID++
    
    fmt.Printf("Created object %d with size %d\n", obj.ID, obj.Size)
    return obj
}

func (gc *GarbageCollector) AddReference(from, to *Object) {
    if from != nil && to != nil {
        from.References = append(from.References, to)
        to.ReferencedBy = append(to.ReferencedBy, from)
        fmt.Printf("Added reference from object %d to object %d\n", from.ID, to.ID)
    }
}

func (gc *GarbageCollector) RemoveReference(from, to *Object) {
    if from != nil && to != nil {
        // Remove from References
        for i, ref := range from.References {
            if ref.ID == to.ID {
                from.References = append(from.References[:i], from.References[i+1:]...)
                break
            }
        }
        
        // Remove from ReferencedBy
        for i, ref := range to.ReferencedBy {
            if ref.ID == from.ID {
                to.ReferencedBy = append(to.ReferencedBy[:i], to.ReferencedBy[i+1:]...)
                break
            }
        }
        
        fmt.Printf("Removed reference from object %d to object %d\n", from.ID, to.ID)
    }
}

func (gc *GarbageCollector) MarkAndSweep() {
    fmt.Println("Starting garbage collection...")
    
    // Mark phase
    gc.mark()
    
    // Sweep phase
    gc.sweep()
    
    fmt.Println("Garbage collection completed")
}

func (gc *GarbageCollector) mark() {
    fmt.Println("Mark phase:")
    
    // Mark all objects as unmarked
    for _, obj := range gc.objects {
        obj.Marked = false
    }
    
    // Mark reachable objects
    for _, obj := range gc.objects {
        if len(obj.ReferencedBy) > 0 || gc.isRoot(obj) {
            gc.markRecursive(obj)
        }
    }
}

func (gc *GarbageCollector) markRecursive(obj *Object) {
    if obj.Marked {
        return
    }
    
    obj.Marked = true
    fmt.Printf("Marked object %d\n", obj.ID)
    
    // Mark all objects referenced by this object
    for _, ref := range obj.References {
        gc.markRecursive(ref)
    }
}

func (gc *GarbageCollector) isRoot(obj *Object) bool {
    // In a real system, root objects would be global variables, stack variables, etc.
    // For simplicity, we'll consider objects with no incoming references as roots
    return len(obj.ReferencedBy) == 0
}

func (gc *GarbageCollector) sweep() {
    fmt.Println("Sweep phase:")
    
    var totalFreed int
    
    for id, obj := range gc.objects {
        if !obj.Marked {
            fmt.Printf("Sweeping object %d (size: %d)\n", obj.ID, obj.Size)
            totalFreed += obj.Size
            delete(gc.objects, id)
        }
    }
    
    fmt.Printf("Freed %d bytes of memory\n", totalFreed)
}

func (gc *GarbageCollector) PrintMemory() {
    fmt.Println("Memory Status:")
    totalSize := 0
    for _, obj := range gc.objects {
        status := "UNMARKED"
        if obj.Marked {
            status = "MARKED"
        }
        fmt.Printf("Object %d: size %d, status %s\n", obj.ID, obj.Size, status)
        totalSize += obj.Size
    }
    fmt.Printf("Total memory used: %d bytes\n", totalSize)
    fmt.Println()
}

func main() {
    gc := NewGarbageCollector()
    
    fmt.Println("Garbage Collection Demo:")
    
    // Create objects
    obj1 := gc.CreateObject(100)
    obj2 := gc.CreateObject(200)
    obj3 := gc.CreateObject(150)
    obj4 := gc.CreateObject(300)
    
    // Create references
    gc.AddReference(obj1, obj2)
    gc.AddReference(obj2, obj3)
    gc.AddReference(obj1, obj4)
    
    gc.PrintMemory()
    
    // Remove some references
    gc.RemoveReference(obj1, obj2)
    gc.RemoveReference(obj2, obj3)
    
    gc.PrintMemory()
    
    // Run garbage collection
    gc.MarkAndSweep()
    
    gc.PrintMemory()
}
```

## Follow-up Questions

### 1. Virtual Memory
**Q: What are the advantages and disadvantages of virtual memory?**
A: Advantages: Allows processes to use more memory than physically available, provides memory protection, enables efficient memory sharing. Disadvantages: Overhead from page table management, potential performance impact from page faults, complexity in implementation.

### 2. Page Replacement
**Q: When would you choose LRU over FIFO for page replacement?**
A: Choose LRU when you have good locality of reference and want to minimize page faults. Choose FIFO when you need a simple implementation and the access pattern doesn't have strong locality.

### 3. Memory Allocation
**Q: What's the difference between first-fit and best-fit allocation?**
A: First-fit allocates the first suitable block found, which is faster but may lead to fragmentation. Best-fit allocates the smallest suitable block, which reduces waste but takes longer to find and may create many small free blocks.

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| LRU | O(1) | O(n) | Using hash map and doubly linked list |
| FIFO | O(1) | O(n) | Using circular buffer |
| First Fit | O(n) | O(n) | Linear search through blocks |
| Best Fit | O(n) | O(n) | Linear search for best block |
| Mark and Sweep | O(n) | O(1) | Linear time, constant extra space |

## Applications

1. **Virtual Memory**: Operating systems, virtual machines
2. **Page Replacement**: Database systems, web servers
3. **Memory Allocation**: Dynamic memory management, embedded systems
4. **Garbage Collection**: Programming languages (Java, C#, Go)

---

**Next**: [File Systems](file-systems.md) | **Previous**: [OS Deep Dive](README.md) | **Up**: [OS Deep Dive](README.md)
