# Operating Systems Deep Dive

## Table of Contents

1. [Overview](#overview/)
2. [Process Management](#process-management/)
3. [Memory Management](#memory-management/)
4. [File Systems](#file-systems/)
5. [I/O Systems](#io-systems/)
6. [Concurrency and Synchronization](#concurrency-and-synchronization/)
7. [System Calls](#system-calls/)
8. [Performance Optimization](#performance-optimization/)
9. [Implementations](#implementations/)
10. [Follow-up Questions](#follow-up-questions/)
11. [Sources](#sources/)
12. [Projects](#projects/)

## Overview

### Learning Objectives

- Master process lifecycle and scheduling algorithms
- Understand memory management and virtual memory
- Learn file system implementation and optimization
- Master I/O operations and device drivers
- Apply concurrency and synchronization techniques
- Optimize system performance

### What is OS Deep Dive?

This module provides an in-depth understanding of operating system internals, covering process management, memory systems, file systems, I/O operations, and performance optimization techniques.

## Process Management

### 1. Process Lifecycle

#### Process States and Transitions
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type ProcessState int

const (
    New ProcessState = iota
    Ready
    Running
    Waiting
    Terminated
)

type Process struct {
    ID          int
    State       ProcessState
    Priority    int
    BurstTime   int
    ArrivalTime time.Time
    StartTime   time.Time
    EndTime     time.Time
    WaitTime    int
    TurnaroundTime int
}

type ProcessManager struct {
    processes    []*Process
    readyQueue   []*Process
    running      *Process
    mutex        sync.Mutex
    nextPID      int
}

func NewProcessManager() *ProcessManager {
    return &ProcessManager{
        processes:  make([]*Process, 0),
        readyQueue: make([]*Process, 0),
        nextPID:    1,
    }
}

func (pm *ProcessManager) CreateProcess(priority, burstTime int) *Process {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    process := &Process{
        ID:          pm.nextPID,
        State:       New,
        Priority:    priority,
        BurstTime:   burstTime,
        ArrivalTime: time.Now(),
    }
    
    pm.nextPID++
    pm.processes = append(pm.processes, process)
    
    // Move to ready state
    pm.AdmitProcess(process)
    
    return process
}

func (pm *ProcessManager) AdmitProcess(process *Process) {
    process.State = Ready
    pm.readyQueue = append(pm.readyQueue, process)
    fmt.Printf("Process %d admitted to ready queue\n", process.ID)
}

func (pm *ProcessManager) DispatchProcess() *Process {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    if len(pm.readyQueue) == 0 {
        return nil
    }
    
    // Select process from ready queue (FCFS for simplicity)
    process := pm.readyQueue[0]
    pm.readyQueue = pm.readyQueue[1:]
    
    process.State = Running
    process.StartTime = time.Now()
    pm.running = process
    
    fmt.Printf("Process %d dispatched to CPU\n", process.ID)
    return process
}

func (pm *ProcessManager) PreemptProcess() {
    if pm.running != nil {
        pm.running.State = Ready
        pm.readyQueue = append(pm.readyQueue, pm.running)
        fmt.Printf("Process %d preempted\n", pm.running.ID)
        pm.running = nil
    }
}

func (pm *ProcessManager) TerminateProcess(process *Process) {
    process.State = Terminated
    process.EndTime = time.Now()
    process.TurnaroundTime = int(process.EndTime.Sub(process.ArrivalTime).Milliseconds())
    process.WaitTime = process.TurnaroundTime - process.BurstTime
    
    if pm.running == process {
        pm.running = nil
    }
    
    fmt.Printf("Process %d terminated (Turnaround: %dms, Wait: %dms)\n", 
        process.ID, process.TurnaroundTime, process.WaitTime)
}

func main() {
    pm := NewProcessManager()
    
    // Create processes
    p1 := pm.CreateProcess(1, 100)
    p2 := pm.CreateProcess(2, 200)
    p3 := pm.CreateProcess(1, 150)
    
    // Simulate process execution
    for len(pm.readyQueue) > 0 || pm.running != nil {
        if pm.running == nil {
            pm.DispatchProcess()
        } else {
            // Simulate time quantum
            time.Sleep(50 * time.Millisecond)
            pm.TerminateProcess(pm.running)
        }
    }
    
    // Calculate average turnaround time
    totalTurnaround := 0
    for _, p := range pm.processes {
        totalTurnaround += p.TurnaroundTime
    }
    avgTurnaround := float64(totalTurnaround) / float64(len(pm.processes))
    fmt.Printf("Average Turnaround Time: %.2fms\n", avgTurnaround)
}
```

### 2. CPU Scheduling Algorithms

#### Round Robin Scheduling
```go
package main

import (
    "fmt"
    "time"
)

type Process struct {
    ID        int
    BurstTime int
    RemainingTime int
    ArrivalTime time.Time
}

type RoundRobinScheduler struct {
    processes   []Process
    timeQuantum int
}

func NewRoundRobinScheduler(quantum int) *RoundRobinScheduler {
    return &RoundRobinScheduler{
        timeQuantum: quantum,
    }
}

func (rr *RoundRobinScheduler) AddProcess(id, burstTime int) {
    process := Process{
        ID:            id,
        BurstTime:     burstTime,
        RemainingTime: burstTime,
        ArrivalTime:   time.Now(),
    }
    rr.processes = append(rr.processes, process)
}

func (rr *RoundRobinScheduler) Schedule() {
    queue := make([]*Process, 0)
    
    // Initialize queue with all processes
    for i := range rr.processes {
        queue = append(queue, &rr.processes[i])
    }
    
    currentTime := 0
    completed := 0
    
    fmt.Println("Round Robin Scheduling (Quantum =", rr.timeQuantum, ")")
    fmt.Println("Time\tProcess\tRemaining")
    
    for len(queue) > 0 {
        process := queue[0]
        queue = queue[1:]
        
        if process.RemainingTime <= rr.timeQuantum {
            // Process completes
            currentTime += process.RemainingTime
            fmt.Printf("%d\tP%d\t0 (Completed)\n", currentTime, process.ID)
            process.RemainingTime = 0
            completed++
        } else {
            // Process uses full quantum
            currentTime += rr.timeQuantum
            process.RemainingTime -= rr.timeQuantum
            fmt.Printf("%d\tP%d\t%d\n", currentTime, process.ID, process.RemainingTime)
            
            // Add back to queue
            queue = append(queue, process)
        }
    }
}

func main() {
    scheduler := NewRoundRobinScheduler(4)
    
    scheduler.AddProcess(1, 8)
    scheduler.AddProcess(2, 6)
    scheduler.AddProcess(3, 3)
    scheduler.AddProcess(4, 9)
    
    scheduler.Schedule()
}
```

#### Priority Scheduling
```go
package main

import (
    "fmt"
    "sort"
)

type PriorityProcess struct {
    ID       int
    BurstTime int
    Priority int
    ArrivalTime int
}

type PriorityScheduler struct {
    processes []PriorityProcess
}

func NewPriorityScheduler() *PriorityScheduler {
    return &PriorityScheduler{
        processes: make([]PriorityProcess, 0),
    }
}

func (ps *PriorityScheduler) AddProcess(id, burstTime, priority, arrivalTime int) {
    process := PriorityProcess{
        ID:          id,
        BurstTime:   burstTime,
        Priority:    priority,
        ArrivalTime: arrivalTime,
    }
    ps.processes = append(ps.processes, process)
}

func (ps *PriorityScheduler) Schedule() {
    // Sort by priority (lower number = higher priority)
    sort.Slice(ps.processes, func(i, j int) bool {
        if ps.processes[i].Priority == ps.processes[j].Priority {
            return ps.processes[i].ArrivalTime < ps.processes[j].ArrivalTime
        }
        return ps.processes[i].Priority < ps.processes[j].Priority
    })
    
    fmt.Println("Priority Scheduling")
    fmt.Println("Process\tPriority\tBurst Time\tWait Time\tTurnaround Time")
    
    currentTime := 0
    totalWaitTime := 0
    totalTurnaroundTime := 0
    
    for _, process := range ps.processes {
        waitTime := currentTime - process.ArrivalTime
        if waitTime < 0 {
            waitTime = 0
        }
        
        turnaroundTime := waitTime + process.BurstTime
        
        fmt.Printf("P%d\t%d\t\t%d\t\t%d\t\t%d\n", 
            process.ID, process.Priority, process.BurstTime, waitTime, turnaroundTime)
        
        totalWaitTime += waitTime
        totalTurnaroundTime += turnaroundTime
        currentTime += process.BurstTime
    }
    
    avgWaitTime := float64(totalWaitTime) / float64(len(ps.processes))
    avgTurnaroundTime := float64(totalTurnaroundTime) / float64(len(ps.processes))
    
    fmt.Printf("\nAverage Wait Time: %.2f\n", avgWaitTime)
    fmt.Printf("Average Turnaround Time: %.2f\n", avgTurnaroundTime)
}

func main() {
    scheduler := NewPriorityScheduler()
    
    scheduler.AddProcess(1, 10, 3, 0)
    scheduler.AddProcess(2, 1, 1, 0)
    scheduler.AddProcess(3, 2, 4, 0)
    scheduler.AddProcess(4, 1, 5, 0)
    scheduler.AddProcess(5, 5, 2, 0)
    
    scheduler.Schedule()
}
```

## Memory Management

### 1. Virtual Memory Implementation

#### Page Table and TLB
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
    Accessed    bool
    ReferenceTime time.Time
}

type PageTable struct {
    entries map[int]*PageTableEntry
    size    int
}

type TLB struct {
    entries map[int]int // page -> frame
    size    int
    order   []int // LRU order
}

type VirtualMemory struct {
    pageTable    *PageTable
    tlb          *TLB
    physicalFrames []bool
    pageSize     int
    numFrames    int
    pageFaults   int
    tlbHits      int
    tlbMisses    int
}

func NewVirtualMemory(pageSize, numFrames, tlbSize int) *VirtualMemory {
    return &VirtualMemory{
        pageTable: &PageTable{
            entries: make(map[int]*PageTableEntry),
            size:    0,
        },
        tlb: &TLB{
            entries: make(map[int]int),
            size:    0,
            order:   make([]int, 0),
        },
        physicalFrames: make([]bool, numFrames),
        pageSize:       pageSize,
        numFrames:      numFrames,
    }
}

func (vm *VirtualMemory) GetFrame(pageNum int) int {
    // Check TLB first
    if frame, exists := vm.tlb.entries[pageNum]; exists {
        vm.tlbHits++
        vm.updateTLBOrder(pageNum)
        return frame
    }
    
    vm.tlbMisses++
    
    // Check page table
    if entry, exists := vm.pageTable.entries[pageNum]; exists && entry.Valid {
        frame := entry.FrameNumber
        vm.addToTLB(pageNum, frame)
        return frame
    }
    
    // Page fault - allocate new frame
    frame := vm.allocateFrame()
    if frame == -1 {
        // No free frames, need to evict
        frame = vm.evictPage()
    }
    
    vm.pageFaults++
    
    // Update page table
    vm.pageTable.entries[pageNum] = &PageTableEntry{
        FrameNumber: frame,
        Valid:       true,
        Dirty:       false,
        Accessed:    true,
        ReferenceTime: time.Now(),
    }
    
    vm.addToTLB(pageNum, frame)
    return frame
}

func (vm *VirtualMemory) allocateFrame() int {
    for i, used := range vm.physicalFrames {
        if !used {
            vm.physicalFrames[i] = true
            return i
        }
    }
    return -1
}

func (vm *VirtualMemory) evictPage() int {
    // Simple FIFO eviction
    for pageNum, entry := range vm.pageTable.entries {
        if entry.Valid {
            entry.Valid = false
            vm.physicalFrames[entry.FrameNumber] = false
            delete(vm.tlb.entries, pageNum)
            return entry.FrameNumber
        }
    }
    return 0
}

func (vm *VirtualMemory) addToTLB(pageNum, frame int) {
    if vm.tlb.size >= len(vm.tlb.order) {
        // TLB is full, remove LRU entry
        lruPage := vm.tlb.order[0]
        vm.tlb.order = vm.tlb.order[1:]
        delete(vm.tlb.entries, lruPage)
        vm.tlb.size--
    }
    
    vm.tlb.entries[pageNum] = frame
    vm.tlb.order = append(vm.tlb.order, pageNum)
    vm.tlb.size++
}

func (vm *VirtualMemory) updateTLBOrder(pageNum int) {
    // Move to end (most recently used)
    for i, p := range vm.tlb.order {
        if p == pageNum {
            vm.tlb.order = append(vm.tlb.order[:i], vm.tlb.order[i+1:]...)
            vm.tlb.order = append(vm.tlb.order, pageNum)
            break
        }
    }
}

func (vm *VirtualMemory) GetStats() {
    fmt.Printf("Page Faults: %d\n", vm.pageFaults)
    fmt.Printf("TLB Hits: %d\n", vm.tlbHits)
    fmt.Printf("TLB Misses: %d\n", vm.tlbMisses)
    
    totalAccesses := vm.tlbHits + vm.tlbMisses
    if totalAccesses > 0 {
        hitRate := float64(vm.tlbHits) / float64(totalAccesses) * 100
        fmt.Printf("TLB Hit Rate: %.2f%%\n", hitRate)
    }
}

func main() {
    vm := NewVirtualMemory(4096, 4, 2) // 4KB pages, 4 frames, 2 TLB entries
    
    // Simulate memory access pattern
    rand.Seed(time.Now().UnixNano())
    
    for i := 0; i < 20; i++ {
        pageNum := rand.Intn(10) // Access pages 0-9
        frame := vm.GetFrame(pageNum)
        fmt.Printf("Access page %d -> frame %d\n", pageNum, frame)
    }
    
    vm.GetStats()
}
```

### 2. Memory Allocation Algorithms

#### Buddy System
```go
package main

import (
    "fmt"
    "math"
)

type BuddySystem struct {
    memory     []byte
    size       int
    freeLists  [][]int // Each index represents 2^i sized blocks
    maxOrder   int
}

func NewBuddySystem(size int) *BuddySystem {
    // Size must be power of 2
    actualSize := 1
    for actualSize < size {
        actualSize <<= 1
    }
    
    maxOrder := int(math.Log2(float64(actualSize)))
    
    return &BuddySystem{
        memory:    make([]byte, actualSize),
        size:      actualSize,
        freeLists: make([][]int, maxOrder+1),
        maxOrder:  maxOrder,
    }
}

func (bs *BuddySystem) Allocate(size int) int {
    if size <= 0 || size > bs.size {
        return -1
    }
    
    // Find appropriate order
    order := bs.getOrder(size)
    
    // Find free block
    block := bs.findFreeBlock(order)
    if block == -1 {
        return -1
    }
    
    // Mark as allocated
    bs.markAllocated(block, order)
    
    return block
}

func (bs *BuddySystem) Deallocate(block int) {
    if block < 0 || block >= bs.size {
        return
    }
    
    // Find the order of this block
    order := bs.findBlockOrder(block)
    if order == -1 {
        return
    }
    
    // Mark as free
    bs.markFree(block, order)
    
    // Try to merge with buddy
    bs.mergeBuddies(block, order)
}

func (bs *BuddySystem) getOrder(size int) int {
    order := 0
    blockSize := 1
    
    for blockSize < size {
        blockSize <<= 1
        order++
    }
    
    return order
}

func (bs *BuddySystem) findFreeBlock(order int) int {
    if order > bs.maxOrder {
        return -1
    }
    
    if len(bs.freeLists[order]) > 0 {
        block := bs.freeLists[order][0]
        bs.freeLists[order] = bs.freeLists[order][1:]
        return block
    }
    
    // Try to split larger block
    if order < bs.maxOrder {
        largerBlock := bs.findFreeBlock(order + 1)
        if largerBlock != -1 {
            // Split the block
            buddy := largerBlock + (1 << order)
            bs.freeLists[order] = append(bs.freeLists[order], buddy)
            return largerBlock
        }
    }
    
    return -1
}

func (bs *BuddySystem) markAllocated(block, order int) {
    // Implementation would mark the block as allocated
    fmt.Printf("Allocated block %d of order %d\n", block, order)
}

func (bs *BuddySystem) markFree(block, order int) {
    bs.freeLists[order] = append(bs.freeLists[order], block)
    fmt.Printf("Freed block %d of order %d\n", block, order)
}

func (bs *BuddySystem) findBlockOrder(block int) int {
    // This would track which blocks are allocated and their orders
    // For simplicity, return a default order
    return 0
}

func (bs *BuddySystem) mergeBuddies(block, order int) {
    if order >= bs.maxOrder {
        return
    }
    
    buddy := block ^ (1 << order)
    
    // Check if buddy is free
    for i, freeBlock := range bs.freeLists[order] {
        if freeBlock == buddy {
            // Remove buddy from free list
            bs.freeLists[order] = append(bs.freeLists[order][:i], bs.freeLists[order][i+1:]...)
            
            // Remove current block from free list
            for j, freeBlock := range bs.freeLists[order] {
                if freeBlock == block {
                    bs.freeLists[order] = append(bs.freeLists[order][:j], bs.freeLists[order][j+1:]...)
                    break
                }
            }
            
            // Add merged block to higher order
            mergedBlock := block & buddy // Take the smaller address
            bs.freeLists[order+1] = append(bs.freeLists[order+1], mergedBlock)
            
            // Recursively try to merge
            bs.mergeBuddies(mergedBlock, order+1)
            break
        }
    }
}

func (bs *BuddySystem) PrintFreeLists() {
    fmt.Println("Free Lists:")
    for order, blocks := range bs.freeLists {
        if len(blocks) > 0 {
            fmt.Printf("Order %d (size %d): %v\n", order, 1<<order, blocks)
        }
    }
}

func main() {
    bs := NewBuddySystem(1024)
    
    // Allocate some blocks
    block1 := bs.Allocate(100)
    block2 := bs.Allocate(200)
    block3 := bs.Allocate(50)
    
    fmt.Printf("Allocated blocks: %d, %d, %d\n", block1, block2, block3)
    
    // Deallocate some blocks
    bs.Deallocate(block2)
    bs.Deallocate(block3)
    
    bs.PrintFreeLists()
}
```

## File Systems

### 1. Inode-based File System

#### Inode Structure
```go
package main

import (
    "fmt"
    "time"
)

type Inode struct {
    Mode        uint16    // File type and permissions
    UID         uint16    // User ID
    GID         uint16    // Group ID
    Size        int64     // File size in bytes
    AccessTime  time.Time // Last access time
    ModifyTime  time.Time // Last modification time
    ChangeTime  time.Time // Last change time
    BlockCount  int32     // Number of blocks
    DirectBlocks [12]int32 // Direct block pointers
    IndirectBlock int32   // Single indirect block
    DoubleIndirect int32  // Double indirect block
    TripleIndirect int32  // Triple indirect block
}

type FileSystem struct {
    inodes    map[int32]*Inode
    dataBlocks [][]byte
    blockSize int
    nextInode int32
}

func NewFileSystem(blockSize, numBlocks int) *FileSystem {
    return &FileSystem{
        inodes:     make(map[int32]*Inode),
        dataBlocks: make([][]byte, numBlocks),
        blockSize:  blockSize,
        nextInode:  1,
    }
}

func (fs *FileSystem) CreateFile() int32 {
    inodeNum := fs.nextInode
    fs.nextInode++
    
    inode := &Inode{
        Mode:       0644, // Regular file with rw-r--r--
        UID:        1000,
        GID:        1000,
        Size:       0,
        AccessTime: time.Now(),
        ModifyTime: time.Now(),
        ChangeTime: time.Now(),
    }
    
    fs.inodes[inodeNum] = inode
    return inodeNum
}

func (fs *FileSystem) WriteFile(inodeNum int32, data []byte) error {
    inode, exists := fs.inodes[inodeNum]
    if !exists {
        return fmt.Errorf("inode %d not found", inodeNum)
    }
    
    // Calculate number of blocks needed
    blocksNeeded := (len(data) + fs.blockSize - 1) / fs.blockSize
    
    // Allocate blocks
    for i := 0; i < blocksNeeded; i++ {
        blockNum := fs.allocateBlock()
        if blockNum == -1 {
            return fmt.Errorf("no free blocks available")
        }
        
        // Store in direct block if available
        if i < 12 {
            inode.DirectBlocks[i] = int32(blockNum)
        } else {
            // Would use indirect blocks in real implementation
            return fmt.Errorf("file too large for direct blocks")
        }
        
        // Copy data to block
        start := i * fs.blockSize
        end := start + fs.blockSize
        if end > len(data) {
            end = len(data)
        }
        
        fs.dataBlocks[blockNum] = make([]byte, fs.blockSize)
        copy(fs.dataBlocks[blockNum], data[start:end])
    }
    
    inode.Size = int64(len(data))
    inode.ModifyTime = time.Now()
    inode.ChangeTime = time.Now()
    inode.BlockCount = int32(blocksNeeded)
    
    return nil
}

func (fs *FileSystem) ReadFile(inodeNum int32) ([]byte, error) {
    inode, exists := fs.inodes[inodeNum]
    if !exists {
        return nil, fmt.Errorf("inode %d not found", inodeNum)
    }
    
    data := make([]byte, 0, inode.Size)
    
    for i := 0; i < int(inode.BlockCount); i++ {
        if i < 12 {
            blockNum := inode.DirectBlocks[i]
            if blockNum != 0 {
                data = append(data, fs.dataBlocks[blockNum]...)
            }
        }
    }
    
    // Truncate to actual file size
    if len(data) > int(inode.Size) {
        data = data[:inode.Size]
    }
    
    inode.AccessTime = time.Now()
    return data, nil
}

func (fs *FileSystem) allocateBlock() int {
    for i, block := range fs.dataBlocks {
        if block == nil {
            return i
        }
    }
    return -1
}

func (fs *FileSystem) GetFileInfo(inodeNum int32) {
    inode, exists := fs.inodes[inodeNum]
    if !exists {
        fmt.Printf("Inode %d not found\n", inodeNum)
        return
    }
    
    fmt.Printf("Inode %d:\n", inodeNum)
    fmt.Printf("  Size: %d bytes\n", inode.Size)
    fmt.Printf("  Blocks: %d\n", inode.BlockCount)
    fmt.Printf("  Mode: %o\n", inode.Mode)
    fmt.Printf("  UID: %d, GID: %d\n", inode.UID, inode.GID)
    fmt.Printf("  Access: %s\n", inode.AccessTime.Format(time.RFC3339))
    fmt.Printf("  Modify: %s\n", inode.ModifyTime.Format(time.RFC3339))
    fmt.Printf("  Change: %s\n", inode.ChangeTime.Format(time.RFC3339))
}

func main() {
    fs := NewFileSystem(1024, 100) // 1KB blocks, 100 blocks
    
    // Create a file
    inodeNum := fs.CreateFile()
    fmt.Printf("Created file with inode %d\n", inodeNum)
    
    // Write data to file
    data := []byte("Hello, World! This is a test file.")
    err := fs.WriteFile(inodeNum, data)
    if err != nil {
        fmt.Printf("Error writing file: %v\n", err)
        return
    }
    
    // Read data from file
    readData, err := fs.ReadFile(inodeNum)
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    
    fmt.Printf("Read data: %s\n", string(readData))
    
    // Show file info
    fs.GetFileInfo(inodeNum)
}
```

## Concurrency and Synchronization

### 1. Mutex and Semaphore Implementation

#### Mutex Implementation
```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

type Mutex struct {
    state int32
    sema  uint32
}

func NewMutex() *Mutex {
    return &Mutex{}
}

func (m *Mutex) Lock() {
    if atomic.CompareAndSwapInt32(&m.state, 0, 1) {
        return
    }
    
    // Spin for a while
    for i := 0; i < 100; i++ {
        if atomic.CompareAndSwapInt32(&m.state, 0, 1) {
            return
        }
        // Yield to other goroutines
        time.Sleep(time.Nanosecond)
    }
    
    // Block and wait
    for {
        if atomic.CompareAndSwapInt32(&m.state, 0, 1) {
            return
        }
        // In real implementation, this would use futex or similar
        time.Sleep(time.Microsecond)
    }
}

func (m *Mutex) Unlock() {
    atomic.StoreInt32(&m.state, 0)
}

type Semaphore struct {
    count int32
    mutex sync.Mutex
    cond  *sync.Cond
}

func NewSemaphore(initial int) *Semaphore {
    s := &Semaphore{
        count: int32(initial),
    }
    s.cond = sync.NewCond(&s.mutex)
    return s
}

func (s *Semaphore) Wait() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    for s.count <= 0 {
        s.cond.Wait()
    }
    s.count--
}

func (s *Semaphore) Signal() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    s.count++
    s.cond.Signal()
}

func main() {
    // Test mutex
    var counter int
    var mutex sync.Mutex
    var wg sync.WaitGroup
    
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for j := 0; j < 1000; j++ {
                mutex.Lock()
                counter++
                mutex.Unlock()
            }
        }()
    }
    
    wg.Wait()
    fmt.Printf("Counter value: %d\n", counter)
    
    // Test semaphore
    sem := NewSemaphore(2)
    
    for i := 0; i < 5; i++ {
        go func(id int) {
            fmt.Printf("Goroutine %d waiting\n", id)
            sem.Wait()
            fmt.Printf("Goroutine %d acquired semaphore\n", id)
            time.Sleep(2 * time.Second)
            fmt.Printf("Goroutine %d releasing semaphore\n", id)
            sem.Signal()
        }(i)
    }
    
    time.Sleep(10 * time.Second)
}
```

### 2. Reader-Writer Lock

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type RWMutex struct {
    readerCount int32
    readerMutex sync.Mutex
    writerMutex sync.Mutex
    writerCond  *sync.Cond
    writerWaiting bool
}

func NewRWMutex() *RWMutex {
    rw := &RWMutex{}
    rw.writerCond = sync.NewCond(&rw.writerMutex)
    return rw
}

func (rw *RWMutex) RLock() {
    rw.readerMutex.Lock()
    defer rw.readerMutex.Unlock()
    
    rw.readerCount++
}

func (rw *RWMutex) RUnlock() {
    rw.readerMutex.Lock()
    defer rw.readerMutex.Unlock()
    
    rw.readerCount--
    
    if rw.readerCount == 0 && rw.writerWaiting {
        rw.writerCond.Signal()
    }
}

func (rw *RWMutex) Lock() {
    rw.writerMutex.Lock()
    defer rw.writerMutex.Unlock()
    
    rw.writerWaiting = true
    
    // Wait for all readers to finish
    for rw.readerCount > 0 {
        rw.writerCond.Wait()
    }
}

func (rw *RWMutex) Unlock() {
    rw.writerWaiting = false
    rw.writerMutex.Unlock()
}

type SharedResource struct {
    data int
    rw   *RWMutex
}

func NewSharedResource() *SharedResource {
    return &SharedResource{
        rw: NewRWMutex(),
    }
}

func (sr *SharedResource) Read() int {
    sr.rw.RLock()
    defer sr.rw.RUnlock()
    
    time.Sleep(100 * time.Millisecond) // Simulate read operation
    return sr.data
}

func (sr *SharedResource) Write(value int) {
    sr.rw.Lock()
    defer sr.rw.Unlock()
    
    time.Sleep(200 * time.Millisecond) // Simulate write operation
    sr.data = value
}

func main() {
    resource := NewSharedResource()
    
    // Start readers
    for i := 0; i < 5; i++ {
        go func(id int) {
            for j := 0; j < 3; j++ {
                value := resource.Read()
                fmt.Printf("Reader %d read: %d\n", id, value)
                time.Sleep(100 * time.Millisecond)
            }
        }(i)
    }
    
    // Start writers
    for i := 0; i < 2; i++ {
        go func(id int) {
            for j := 0; j < 2; j++ {
                resource.Write(id*10 + j)
                fmt.Printf("Writer %d wrote: %d\n", id, id*10+j)
                time.Sleep(200 * time.Millisecond)
            }
        }(i)
    }
    
    time.Sleep(5 * time.Second)
}
```

## Follow-up Questions

### 1. Process Management
**Q: What's the difference between preemptive and non-preemptive scheduling?**
A: Preemptive scheduling allows the OS to interrupt a running process and switch to another process, while non-preemptive scheduling only switches when the current process voluntarily gives up the CPU.

### 2. Memory Management
**Q: What are the advantages of virtual memory?**
A: Virtual memory allows programs to use more memory than physically available, provides memory protection between processes, enables efficient memory sharing, and simplifies program loading.

### 3. File Systems
**Q: What's the purpose of inodes in file systems?**
A: Inodes store metadata about files (permissions, size, timestamps) and pointers to data blocks, allowing efficient file access and management without storing metadata in directory entries.

## Sources

### Books
- **Operating System Concepts** by Silberschatz, Galvin, Gagne
- **Modern Operating Systems** by Andrew Tanenbaum
- **The Design and Implementation of the FreeBSD Operating System** by McKusick and Neville-Neil

### Online Resources
- **OSDev Wiki** - Operating system development
- **Linux Kernel Documentation** - Linux kernel internals
- **MIT 6.828** - Operating System Engineering course

## Projects

### 1. Mini Operating System
**Objective**: Build a minimal operating system kernel
**Requirements**: Process management, memory management, basic file system
**Deliverables**: Working kernel with basic functionality

### 2. File System Implementation
**Objective**: Implement a simple file system
**Requirements**: Inode structure, directory management, file operations
**Deliverables**: Complete file system with user interface

### 3. Process Scheduler
**Objective**: Implement various CPU scheduling algorithms
**Requirements**: Multiple scheduling algorithms, performance comparison
**Deliverables**: Scheduler with benchmarking tools

---

**Next**: [Database Systems](database-systems/README.md/) | **Previous**: [Advanced DSA](advanced-dsa/README.md/) | **Up**: [Phase 1](README.md/)

