# Operating Systems Concepts

## Table of Contents

1. [Overview](#overview/)
2. [Process Management](#process-management/)
3. [Memory Management](#memory-management/)
4. [File Systems](#file-systems/)
5. [Concurrency and Synchronization](#concurrency-and-synchronization/)
6. [Implementations](#implementations/)
7. [Follow-up Questions](#follow-up-questions/)
8. [Sources](#sources/)
9. [Projects](#projects/)

## Overview

### Learning Objectives

- Understand process management and scheduling
- Master memory management and virtual memory
- Learn file system concepts and implementation
- Apply concurrency and synchronization techniques
- Implement OS concepts in code

### What are Operating Systems?

Operating systems are system software that manage computer hardware and software resources, providing common services for computer programs.

## Process Management

### 1. Process Control Block (PCB)

#### PCB Implementation
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type ProcessState int

const (
    NEW ProcessState = iota
    READY
    RUNNING
    WAITING
    TERMINATED
)

type ProcessControlBlock struct {
    PID           int
    State         ProcessState
    Priority      int
    ProgramCounter int
    Registers     map[string]int
    MemoryLimits  MemoryLimits
    OpenFiles     []int
    ParentPID     int
    Children      []int
    StartTime     time.Time
    CPUTime       time.Duration
    IOTime        time.Duration
    mutex         sync.RWMutex
}

type MemoryLimits struct {
    BaseAddress   int
    LimitAddress  int
    StackPointer  int
    HeapPointer   int
}

func NewPCB(pid int, priority int) *ProcessControlBlock {
    return &ProcessControlBlock{
        PID:           pid,
        State:         NEW,
        Priority:      priority,
        ProgramCounter: 0,
        Registers:     make(map[string]int),
        MemoryLimits:  MemoryLimits{},
        OpenFiles:     make([]int, 0),
        ParentPID:     -1,
        Children:      make([]int, 0),
        StartTime:     time.Now(),
        CPUTime:       0,
        IOTime:        0,
    }
}

func (pcb *ProcessControlBlock) SetState(state ProcessState) {
    pcb.mutex.Lock()
    defer pcb.mutex.Unlock()
    pcb.State = state
}

func (pcb *ProcessControlBlock) GetState() ProcessState {
    pcb.mutex.RLock()
    defer pcb.mutex.RUnlock()
    return pcb.State
}

func (pcb *ProcessControlBlock) UpdateCPUTime(duration time.Duration) {
    pcb.mutex.Lock()
    defer pcb.mutex.Unlock()
    pcb.CPUTime += duration
}

func (pcb *ProcessControlBlock) UpdateIOTime(duration time.Duration) {
    pcb.mutex.Lock()
    defer pcb.mutex.Unlock()
    pcb.IOTime += duration
}

func (pcb *ProcessControlBlock) String() string {
    pcb.mutex.RLock()
    defer pcb.mutex.RUnlock()
    
    return fmt.Sprintf("PCB{PID: %d, State: %s, Priority: %d, PC: %d, CPU: %v, IO: %v}",
        pcb.PID, pcb.stateToString(), pcb.Priority, pcb.ProgramCounter, pcb.CPUTime, pcb.IOTime)
}

func (pcb *ProcessControlBlock) stateToString() string {
    switch pcb.State {
    case NEW:
        return "NEW"
    case READY:
        return "READY"
    case RUNNING:
        return "RUNNING"
    case WAITING:
        return "WAITING"
    case TERMINATED:
        return "TERMINATED"
    default:
        return "UNKNOWN"
    }
}

func main() {
    // Create sample processes
    pcb1 := NewPCB(1, 5)
    pcb2 := NewPCB(2, 3)
    pcb3 := NewPCB(3, 7)
    
    // Simulate process state changes
    pcb1.SetState(READY)
    pcb2.SetState(RUNNING)
    pcb3.SetState(WAITING)
    
    // Update timing information
    pcb1.UpdateCPUTime(100 * time.Millisecond)
    pcb2.UpdateCPUTime(200 * time.Millisecond)
    pcb3.UpdateIOTime(50 * time.Millisecond)
    
    // Print process information
    fmt.Println("Process Control Blocks:")
    fmt.Println(pcb1)
    fmt.Println(pcb2)
    fmt.Println(pcb3)
}
```

### 2. Process Scheduler

#### Round Robin Scheduler
```go
package main

import (
    "container/list"
    "fmt"
    "sync"
    "time"
)

type Scheduler struct {
    ReadyQueue    *list.List
    RunningQueue  *list.List
    WaitingQueue  *list.List
    TerminatedQueue *list.List
    TimeQuantum   time.Duration
    mutex         sync.RWMutex
}

func NewScheduler(timeQuantum time.Duration) *Scheduler {
    return &Scheduler{
        ReadyQueue:     list.New(),
        RunningQueue:   list.New(),
        WaitingQueue:   list.New(),
        TerminatedQueue: list.New(),
        TimeQuantum:    timeQuantum,
    }
}

func (s *Scheduler) AddProcess(pcb *ProcessControlBlock) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    pcb.SetState(READY)
    s.ReadyQueue.PushBack(pcb)
    fmt.Printf("Process %d added to ready queue\n", pcb.PID)
}

func (s *Scheduler) Schedule() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    // Move running process back to ready queue if time quantum expired
    if s.RunningQueue.Len() > 0 {
        running := s.RunningQueue.Front().Value.(*ProcessControlBlock)
        s.RunningQueue.Remove(s.RunningQueue.Front())
        running.SetState(READY)
        s.ReadyQueue.PushBack(running)
        fmt.Printf("Process %d moved back to ready queue\n", running.PID)
    }
    
    // Schedule next process from ready queue
    if s.ReadyQueue.Len() > 0 {
        next := s.ReadyQueue.Front().Value.(*ProcessControlBlock)
        s.ReadyQueue.Remove(s.ReadyQueue.Front())
        next.SetState(RUNNING)
        s.RunningQueue.PushBack(next)
        fmt.Printf("Process %d scheduled to run\n", next.PID)
    }
}

func (s *Scheduler) BlockProcess(pid int) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    // Find process in running queue
    for e := s.RunningQueue.Front(); e != nil; e = e.Next() {
        pcb := e.Value.(*ProcessControlBlock)
        if pcb.PID == pid {
            s.RunningQueue.Remove(e)
            pcb.SetState(WAITING)
            s.WaitingQueue.PushBack(pcb)
            fmt.Printf("Process %d blocked\n", pid)
            return
        }
    }
}

func (s *Scheduler) UnblockProcess(pid int) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    // Find process in waiting queue
    for e := s.WaitingQueue.Front(); e != nil; e = e.Next() {
        pcb := e.Value.(*ProcessControlBlock)
        if pcb.PID == pid {
            s.WaitingQueue.Remove(e)
            pcb.SetState(READY)
            s.ReadyQueue.PushBack(pcb)
            fmt.Printf("Process %d unblocked\n", pid)
            return
        }
    }
}

func (s *Scheduler) TerminateProcess(pid int) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    // Find process in running queue
    for e := s.RunningQueue.Front(); e != nil; e = e.Next() {
        pcb := e.Value.(*ProcessControlBlock)
        if pcb.PID == pid {
            s.RunningQueue.Remove(e)
            pcb.SetState(TERMINATED)
            s.TerminatedQueue.PushBack(pcb)
            fmt.Printf("Process %d terminated\n", pid)
            return
        }
    }
}

func (s *Scheduler) PrintStatus() {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    
    fmt.Println("\nScheduler Status:")
    fmt.Println("=================")
    
    fmt.Printf("Ready Queue (%d processes): ", s.ReadyQueue.Len())
    for e := s.ReadyQueue.Front(); e != nil; e = e.Next() {
        pcb := e.Value.(*ProcessControlBlock)
        fmt.Printf("%d ", pcb.PID)
    }
    fmt.Println()
    
    fmt.Printf("Running Queue (%d processes): ", s.RunningQueue.Len())
    for e := s.RunningQueue.Front(); e != nil; e = e.Next() {
        pcb := e.Value.(*ProcessControlBlock)
        fmt.Printf("%d ", pcb.PID)
    }
    fmt.Println()
    
    fmt.Printf("Waiting Queue (%d processes): ", s.WaitingQueue.Len())
    for e := s.WaitingQueue.Front(); e != nil; e = e.Next() {
        pcb := e.Value.(*ProcessControlBlock)
        fmt.Printf("%d ", pcb.PID)
    }
    fmt.Println()
    
    fmt.Printf("Terminated Queue (%d processes): ", s.TerminatedQueue.Len())
    for e := s.TerminatedQueue.Front(); e != nil; e = e.Next() {
        pcb := e.Value.(*ProcessControlBlock)
        fmt.Printf("%d ", pcb.PID)
    }
    fmt.Println()
}

func main() {
    scheduler := NewScheduler(100 * time.Millisecond)
    
    // Create and add processes
    pcb1 := NewPCB(1, 5)
    pcb2 := NewPCB(2, 3)
    pcb3 := NewPCB(3, 7)
    
    scheduler.AddProcess(pcb1)
    scheduler.AddProcess(pcb2)
    scheduler.AddProcess(pcb3)
    
    // Simulate scheduling
    for i := 0; i < 5; i++ {
        fmt.Printf("\n--- Scheduling Round %d ---\n", i+1)
        scheduler.Schedule()
        scheduler.PrintStatus()
        time.Sleep(50 * time.Millisecond)
    }
    
    // Simulate process blocking and unblocking
    scheduler.BlockProcess(1)
    scheduler.Schedule()
    scheduler.PrintStatus()
    
    scheduler.UnblockProcess(1)
    scheduler.Schedule()
    scheduler.PrintStatus()
    
    // Simulate process termination
    scheduler.TerminateProcess(2)
    scheduler.Schedule()
    scheduler.PrintStatus()
}
```

## Memory Management

### 1. Virtual Memory System

#### Page Table Implementation
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type PageTableEntry struct {
    Valid       bool
    FrameNumber int
    Dirty       bool
    Referenced  bool
    LastAccess  time.Time
}

type PageTable struct {
    Entries map[int]*PageTableEntry
    mutex   sync.RWMutex
}

func NewPageTable() *PageTable {
    return &PageTable{
        Entries: make(map[int]*PageTableEntry),
    }
}

func (pt *PageTable) GetEntry(pageNumber int) *PageTableEntry {
    pt.mutex.RLock()
    defer pt.mutex.RUnlock()
    
    if entry, exists := pt.Entries[pageNumber]; exists {
        entry.Referenced = true
        entry.LastAccess = time.Now()
        return entry
    }
    return nil
}

func (pt *PageTable) SetEntry(pageNumber int, frameNumber int) {
    pt.mutex.Lock()
    defer pt.mutex.Unlock()
    
    pt.Entries[pageNumber] = &PageTableEntry{
        Valid:       true,
        FrameNumber: frameNumber,
        Dirty:       false,
        Referenced:  true,
        LastAccess:  time.Now(),
    }
}

func (pt *PageTable) MarkDirty(pageNumber int) {
    pt.mutex.Lock()
    defer pt.mutex.Unlock()
    
    if entry, exists := pt.Entries[pageNumber]; exists {
        entry.Dirty = true
    }
}

func (pt *PageTable) Invalidate(pageNumber int) {
    pt.mutex.Lock()
    defer pt.mutex.Unlock()
    
    if entry, exists := pt.Entries[pageNumber]; exists {
        entry.Valid = false
    }
}

type VirtualMemoryManager struct {
    PageTable     *PageTable
    PhysicalMemory map[int][]byte
    PageSize      int
    TotalFrames   int
    FreeFrames    []int
    mutex         sync.RWMutex
}

func NewVirtualMemoryManager(pageSize, totalFrames int) *VirtualMemoryManager {
    freeFrames := make([]int, totalFrames)
    for i := 0; i < totalFrames; i++ {
        freeFrames[i] = i
    }
    
    return &VirtualMemoryManager{
        PageTable:     NewPageTable(),
        PhysicalMemory: make(map[int][]byte),
        PageSize:      pageSize,
        TotalFrames:   totalFrames,
        FreeFrames:    freeFrames,
    }
}

func (vmm *VirtualMemoryManager) AllocateFrame() int {
    vmm.mutex.Lock()
    defer vmm.mutex.Unlock()
    
    if len(vmm.FreeFrames) == 0 {
        return -1 // No free frames available
    }
    
    frameNumber := vmm.FreeFrames[0]
    vmm.FreeFrames = vmm.FreeFrames[1:]
    
    // Initialize frame with zeros
    vmm.PhysicalMemory[frameNumber] = make([]byte, vmm.PageSize)
    
    return frameNumber
}

func (vmm *VirtualMemoryManager) DeallocateFrame(frameNumber int) {
    vmm.mutex.Lock()
    defer vmm.mutex.Unlock()
    
    delete(vmm.PhysicalMemory, frameNumber)
    vmm.FreeFrames = append(vmm.FreeFrames, frameNumber)
}

func (vmm *VirtualMemoryManager) ReadPage(pageNumber int) ([]byte, bool) {
    entry := vmm.PageTable.GetEntry(pageNumber)
    if entry == nil || !entry.Valid {
        return nil, false
    }
    
    vmm.mutex.RLock()
    defer vmm.mutex.RUnlock()
    
    if frame, exists := vmm.PhysicalMemory[entry.FrameNumber]; exists {
        return frame, true
    }
    
    return nil, false
}

func (vmm *VirtualMemoryManager) WritePage(pageNumber int, data []byte) bool {
    entry := vmm.PageTable.GetEntry(pageNumber)
    if entry == nil || !entry.Valid {
        return false
    }
    
    vmm.mutex.Lock()
    defer vmm.mutex.Unlock()
    
    if frame, exists := vmm.PhysicalMemory[entry.FrameNumber]; exists {
        copy(frame, data)
        vmm.PageTable.MarkDirty(pageNumber)
        return true
    }
    
    return false
}

func (vmm *VirtualMemoryManager) PageFault(pageNumber int) bool {
    // Allocate a new frame
    frameNumber := vmm.AllocateFrame()
    if frameNumber == -1 {
        return false // No free frames available
    }
    
    // Set up page table entry
    vmm.PageTable.SetEntry(pageNumber, frameNumber)
    
    // Load page from disk (simulated)
    vmm.mutex.Lock()
    vmm.PhysicalMemory[frameNumber] = make([]byte, vmm.PageSize)
    vmm.mutex.Unlock()
    
    fmt.Printf("Page fault handled: Page %d -> Frame %d\n", pageNumber, frameNumber)
    return true
}

func (vmm *VirtualMemoryManager) PrintStatus() {
    vmm.mutex.RLock()
    defer vmm.mutex.RUnlock()
    
    fmt.Println("\nVirtual Memory Manager Status:")
    fmt.Println("=============================")
    fmt.Printf("Total Frames: %d\n", vmm.TotalFrames)
    fmt.Printf("Free Frames: %d\n", len(vmm.FreeFrames))
    fmt.Printf("Used Frames: %d\n", vmm.TotalFrames-len(vmm.FreeFrames))
    fmt.Printf("Page Table Entries: %d\n", len(vmm.PageTable.Entries))
    
    fmt.Println("\nPage Table:")
    for pageNum, entry := range vmm.PageTable.Entries {
        if entry.Valid {
            fmt.Printf("  Page %d -> Frame %d (Dirty: %t, Referenced: %t)\n", 
                pageNum, entry.FrameNumber, entry.Dirty, entry.Referenced)
        }
    }
}

func main() {
    vmm := NewVirtualMemoryManager(4096, 8) // 4KB pages, 8 frames
    
    // Simulate page allocation
    for i := 0; i < 5; i++ {
        frameNumber := vmm.AllocateFrame()
        if frameNumber != -1 {
            vmm.PageTable.SetEntry(i, frameNumber)
            fmt.Printf("Allocated page %d to frame %d\n", i, frameNumber)
        }
    }
    
    vmm.PrintStatus()
    
    // Simulate page fault
    vmm.PageFault(5)
    vmm.PrintStatus()
    
    // Simulate page access
    data := make([]byte, 4096)
    for i := range data {
        data[i] = byte(i % 256)
    }
    
    if vmm.WritePage(0, data) {
        fmt.Println("Successfully wrote to page 0")
    }
    
    if readData, success := vmm.ReadPage(0); success {
        fmt.Printf("Successfully read from page 0: %d bytes\n", len(readData))
    }
}
```

## File Systems

### 1. File System Implementation

#### Simple File System
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type FileType int

const (
    REGULAR_FILE FileType = iota
    DIRECTORY
    SYMLINK
)

type FileMode int

const (
    READ_ONLY FileMode = 1 << iota
    WRITE_ONLY
    READ_WRITE
    EXECUTABLE
)

type Inode struct {
    InodeNumber int
    FileType    FileType
    Mode        FileMode
    Size        int64
    BlockCount  int
    Blocks      []int
    Created     time.Time
    Modified    time.Time
    Accessed    time.Time
    Owner       int
    Group       int
    Links       int
}

type DirectoryEntry struct {
    Name     string
    InodeNum int
}

type FileSystem struct {
    Inodes      map[int]*Inode
    DataBlocks  map[int][]byte
    BlockSize   int
    TotalBlocks int
    FreeBlocks  []int
    RootInode   int
    mutex       sync.RWMutex
}

func NewFileSystem(blockSize, totalBlocks int) *FileSystem {
    freeBlocks := make([]int, totalBlocks)
    for i := 0; i < totalBlocks; i++ {
        freeBlocks[i] = i
    }
    
    fs := &FileSystem{
        Inodes:      make(map[int]*Inode),
        DataBlocks:  make(map[int][]byte),
        BlockSize:   blockSize,
        TotalBlocks: totalBlocks,
        FreeBlocks:  freeBlocks,
        RootInode:   0,
    }
    
    // Create root directory
    rootInode := &Inode{
        InodeNumber: 0,
        FileType:    DIRECTORY,
        Mode:        READ_WRITE | EXECUTABLE,
        Size:        0,
        BlockCount:  0,
        Blocks:      make([]int, 0),
        Created:     time.Now(),
        Modified:    time.Now(),
        Accessed:    time.Now(),
        Owner:       0,
        Group:       0,
        Links:       1,
    }
    
    fs.Inodes[0] = rootInode
    return fs
}

func (fs *FileSystem) AllocateBlock() int {
    fs.mutex.Lock()
    defer fs.mutex.Unlock()
    
    if len(fs.FreeBlocks) == 0 {
        return -1 // No free blocks available
    }
    
    blockNumber := fs.FreeBlocks[0]
    fs.FreeBlocks = fs.FreeBlocks[1:]
    
    // Initialize block with zeros
    fs.DataBlocks[blockNumber] = make([]byte, fs.BlockSize)
    
    return blockNumber
}

func (fs *FileSystem) DeallocateBlock(blockNumber int) {
    fs.mutex.Lock()
    defer fs.mutex.Unlock()
    
    delete(fs.DataBlocks, blockNumber)
    fs.FreeBlocks = append(fs.FreeBlocks, blockNumber)
}

func (fs *FileSystem) CreateFile(name string, parentInode int) int {
    fs.mutex.Lock()
    defer fs.mutex.Unlock()
    
    // Allocate new inode
    inodeNumber := len(fs.Inodes)
    
    inode := &Inode{
        InodeNumber: inodeNumber,
        FileType:    REGULAR_FILE,
        Mode:        READ_WRITE,
        Size:        0,
        BlockCount:  0,
        Blocks:      make([]int, 0),
        Created:     time.Now(),
        Modified:    time.Now(),
        Accessed:    time.Now(),
        Owner:       0,
        Group:       0,
        Links:       1,
    }
    
    fs.Inodes[inodeNumber] = inode
    
    // Add directory entry to parent
    if parentInode >= 0 {
        parent := fs.Inodes[parentInode]
        if parent.FileType == DIRECTORY {
            // Allocate block for directory if needed
            if len(parent.Blocks) == 0 {
                blockNumber := fs.allocateBlockUnsafe()
                if blockNumber != -1 {
                    parent.Blocks = append(parent.Blocks, blockNumber)
                    parent.BlockCount++
                }
            }
            
            // Add directory entry
            if len(parent.Blocks) > 0 {
                block := fs.DataBlocks[parent.Blocks[0]]
                entry := DirectoryEntry{Name: name, InodeNum: inodeNumber}
                // Simplified: just store the entry (in real FS, would serialize properly)
                fmt.Printf("Added directory entry: %s -> %d\n", name, inodeNumber)
            }
        }
    }
    
    return inodeNumber
}

func (fs *FileSystem) allocateBlockUnsafe() int {
    if len(fs.FreeBlocks) == 0 {
        return -1
    }
    
    blockNumber := fs.FreeBlocks[0]
    fs.FreeBlocks = fs.FreeBlocks[1:]
    fs.DataBlocks[blockNumber] = make([]byte, fs.BlockSize)
    
    return blockNumber
}

func (fs *FileSystem) WriteFile(inodeNumber int, data []byte) bool {
    fs.mutex.Lock()
    defer fs.mutex.Unlock()
    
    inode, exists := fs.Inodes[inodeNumber]
    if !exists || inode.FileType != REGULAR_FILE {
        return false
    }
    
    // Calculate blocks needed
    blocksNeeded := (len(data) + fs.BlockSize - 1) / fs.BlockSize
    
    // Allocate additional blocks if needed
    for len(inode.Blocks) < blocksNeeded {
        blockNumber := fs.allocateBlockUnsafe()
        if blockNumber == -1 {
            return false // No free blocks
        }
        inode.Blocks = append(inode.Blocks, blockNumber)
        inode.BlockCount++
    }
    
    // Write data to blocks
    bytesWritten := 0
    for i, blockNumber := range inode.Blocks {
        if bytesWritten >= len(data) {
            break
        }
        
        block := fs.DataBlocks[blockNumber]
        copySize := fs.BlockSize
        if bytesWritten+copySize > len(data) {
            copySize = len(data) - bytesWritten
        }
        
        copy(block, data[bytesWritten:bytesWritten+copySize])
        bytesWritten += copySize
    }
    
    inode.Size = int64(len(data))
    inode.Modified = time.Now()
    inode.Accessed = time.Now()
    
    return true
}

func (fs *FileSystem) ReadFile(inodeNumber int) ([]byte, bool) {
    fs.mutex.RLock()
    defer fs.mutex.RUnlock()
    
    inode, exists := fs.Inodes[inodeNumber]
    if !exists || inode.FileType != REGULAR_FILE {
        return nil, false
    }
    
    data := make([]byte, inode.Size)
    bytesRead := 0
    
    for _, blockNumber := range inode.Blocks {
        if bytesRead >= int(inode.Size) {
            break
        }
        
        block := fs.DataBlocks[blockNumber]
        copySize := fs.BlockSize
        if bytesRead+copySize > int(inode.Size) {
            copySize = int(inode.Size) - bytesRead
        }
        
        copy(data[bytesRead:], block[:copySize])
        bytesRead += copySize
    }
    
    inode.Accessed = time.Now()
    return data, true
}

func (fs *FileSystem) PrintStatus() {
    fs.mutex.RLock()
    defer fs.mutex.RUnlock()
    
    fmt.Println("\nFile System Status:")
    fmt.Println("==================")
    fmt.Printf("Total Blocks: %d\n", fs.TotalBlocks)
    fmt.Printf("Free Blocks: %d\n", len(fs.FreeBlocks))
    fmt.Printf("Used Blocks: %d\n", fs.TotalBlocks-len(fs.FreeBlocks))
    fmt.Printf("Inodes: %d\n", len(fs.Inodes))
    
    fmt.Println("\nInodes:")
    for inodeNum, inode := range fs.Inodes {
        fmt.Printf("  Inode %d: Type=%s, Size=%d, Blocks=%d, Links=%d\n",
            inodeNum, fs.fileTypeToString(inode.FileType), inode.Size, inode.BlockCount, inode.Links)
    }
}

func (fs *FileSystem) fileTypeToString(fileType FileType) string {
    switch fileType {
    case REGULAR_FILE:
        return "FILE"
    case DIRECTORY:
        return "DIR"
    case SYMLINK:
        return "LINK"
    default:
        return "UNKNOWN"
    }
}

func main() {
    fs := NewFileSystem(1024, 100) // 1KB blocks, 100 total blocks
    
    // Create some files
    file1 := fs.CreateFile("test1.txt", 0)
    file2 := fs.CreateFile("test2.txt", 0)
    
    // Write data to files
    data1 := []byte("Hello, World! This is test file 1.")
    data2 := []byte("This is test file 2 with some content.")
    
    if fs.WriteFile(file1, data1) {
        fmt.Println("Successfully wrote to test1.txt")
    }
    
    if fs.WriteFile(file2, data2) {
        fmt.Println("Successfully wrote to test2.txt")
    }
    
    // Read data from files
    if readData, success := fs.ReadFile(file1); success {
        fmt.Printf("Read from test1.txt: %s\n", string(readData))
    }
    
    if readData, success := fs.ReadFile(file2); success {
        fmt.Printf("Read from test2.txt: %s\n", string(readData))
    }
    
    fs.PrintStatus()
}
```

## Concurrency and Synchronization

### 1. Mutex and Semaphore Implementation

#### Synchronization Primitives
```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

type Semaphore struct {
    value    int32
    waiters  int32
    mutex    sync.Mutex
    cond     *sync.Cond
}

func NewSemaphore(initial int) *Semaphore {
    s := &Semaphore{
        value:   int32(initial),
        waiters: 0,
    }
    s.cond = sync.NewCond(&s.mutex)
    return s
}

func (s *Semaphore) Wait() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    for s.value <= 0 {
        s.waiters++
        s.cond.Wait()
        s.waiters--
    }
    s.value--
}

func (s *Semaphore) Signal() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    s.value++
    if s.waiters > 0 {
        s.cond.Signal()
    }
}

func (s *Semaphore) TryWait() bool {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    if s.value > 0 {
        s.value--
        return true
    }
    return false
}

type ReaderWriterLock struct {
    readers       int32
    writer        bool
    mutex         sync.Mutex
    readCondition *sync.Cond
    writeCondition *sync.Cond
}

func NewReaderWriterLock() *ReaderWriterLock {
    rw := &ReaderWriterLock{}
    rw.readCondition = sync.NewCond(&rw.mutex)
    rw.writeCondition = sync.NewCond(&rw.mutex)
    return rw
}

func (rw *ReaderWriterLock) RLock() {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    for rw.writer {
        rw.readCondition.Wait()
    }
    atomic.AddInt32(&rw.readers, 1)
}

func (rw *ReaderWriterLock) RUnlock() {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    if atomic.AddInt32(&rw.readers, -1) == 0 {
        rw.writeCondition.Signal()
    }
}

func (rw *ReaderWriterLock) Lock() {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    for rw.writer || atomic.LoadInt32(&rw.readers) > 0 {
        rw.writeCondition.Wait()
    }
    rw.writer = true
}

func (rw *ReaderWriterLock) Unlock() {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    rw.writer = false
    rw.writeCondition.Signal()
    rw.readCondition.Broadcast()
}

type Barrier struct {
    count     int
    threshold int
    mutex     sync.Mutex
    condition *sync.Cond
}

func NewBarrier(threshold int) *Barrier {
    b := &Barrier{
        count:     0,
        threshold: threshold,
    }
    b.condition = sync.NewCond(&b.mutex)
    return b
}

func (b *Barrier) Wait() {
    b.mutex.Lock()
    defer b.mutex.Unlock()
    
    b.count++
    
    if b.count >= b.threshold {
        b.count = 0
        b.condition.Broadcast()
    } else {
        b.condition.Wait()
    }
}

func main() {
    // Test Semaphore
    fmt.Println("Testing Semaphore:")
    sem := NewSemaphore(2)
    
    for i := 0; i < 5; i++ {
        go func(id int) {
            fmt.Printf("Goroutine %d waiting\n", id)
            sem.Wait()
            fmt.Printf("Goroutine %d acquired semaphore\n", id)
            time.Sleep(100 * time.Millisecond)
            fmt.Printf("Goroutine %d releasing semaphore\n", id)
            sem.Signal()
        }(i)
    }
    
    time.Sleep(1 * time.Second)
    
    // Test Reader-Writer Lock
    fmt.Println("\nTesting Reader-Writer Lock:")
    rwLock := NewReaderWriterLock()
    sharedData := 0
    
    // Writers
    for i := 0; i < 2; i++ {
        go func(id int) {
            for j := 0; j < 3; j++ {
                rwLock.Lock()
                sharedData++
                fmt.Printf("Writer %d: wrote %d\n", id, sharedData)
                time.Sleep(50 * time.Millisecond)
                rwLock.Unlock()
            }
        }(i)
    }
    
    // Readers
    for i := 0; i < 3; i++ {
        go func(id int) {
            for j := 0; j < 2; j++ {
                rwLock.RLock()
                fmt.Printf("Reader %d: read %d\n", id, sharedData)
                time.Sleep(30 * time.Millisecond)
                rwLock.RUnlock()
            }
        }(i)
    }
    
    time.Sleep(1 * time.Second)
    
    // Test Barrier
    fmt.Println("\nTesting Barrier:")
    barrier := NewBarrier(3)
    
    for i := 0; i < 3; i++ {
        go func(id int) {
            fmt.Printf("Goroutine %d: before barrier\n", id)
            time.Sleep(time.Duration(id*100) * time.Millisecond)
            barrier.Wait()
            fmt.Printf("Goroutine %d: after barrier\n", id)
        }(i)
    }
    
    time.Sleep(1 * time.Second)
}
```

## Follow-up Questions

### 1. Process Management
**Q: What's the difference between a process and a thread?**
A: A process is an independent execution unit with its own memory space, while a thread is a lightweight execution unit that shares memory with other threads in the same process.

### 2. Memory Management
**Q: How does virtual memory improve system performance?**
A: Virtual memory allows processes to use more memory than physically available, enables memory protection, and provides efficient memory sharing through paging.

### 3. File Systems
**Q: What are the advantages of inode-based file systems?**
A: Inodes provide efficient file access, support for hard links, and better performance for large directories and files.

## Sources

### Books
- **Operating Systems: Three Easy Pieces** by Remzi Arpaci-Dusseau
- **Modern Operating Systems** by Tanenbaum
- **Operating System Concepts** by Silberschatz

### Online Resources
- **MIT 6.828**: Operating System Engineering
- **Coursera**: Operating Systems courses
- **Linux Kernel Documentation**: Kernel development guides

## Projects

### 1. Process Scheduler
**Objective**: Implement a complete process scheduler
**Requirements**: Multiple scheduling algorithms, process states, context switching
**Deliverables**: Working scheduler with performance metrics

### 2. Memory Manager
**Objective**: Build a virtual memory management system
**Requirements**: Page tables, page replacement, memory allocation
**Deliverables**: Memory manager with allocation strategies

### 3. File System
**Objective**: Create a simple file system
**Requirements**: Inodes, directories, file operations, disk management
**Deliverables**: File system with basic operations

---

**Next**: [Networks & Protocols](networks-protocols.md/) | **Previous**: [Computer Organization](computer-organization.md/) | **Up**: [Phase 0](README.md/)
