# 🖥️ Operating System Concepts - Complete Guide

> **Comprehensive guide to OS concepts with Go code examples and real-world implementations**

## 📋 Table of Contents

1. [Process Management](#process-management)
2. [Memory Management](#memory-management)
3. [File Systems](#file-systems)
4. [Scheduling Algorithms](#scheduling-algorithms)
5. [Inter-Process Communication](#inter-process-communication)
6. [Deadlocks & Synchronization](#deadlocks--synchronization)
7. [System Calls](#system-calls)
8. [Virtual Memory](#virtual-memory)
9. [FAANG Interview Questions](#faang-interview-questions)

---

## 🔄 Process Management

### **1. Process Lifecycle**

#### **Process States**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// ProcessState represents the current state of a process
// These are the fundamental states in process lifecycle
type ProcessState int

const (
    New ProcessState = iota  // Process is being created
    Ready                    // Process is ready to run, waiting for CPU
    Running                  // Process is currently executing on CPU
    Waiting                  // Process is waiting for I/O or event
    Terminated               // Process has finished execution
)

// Process represents a process in the operating system
type Process struct {
    ID       int              // Unique process identifier
    State    ProcessState     // Current state of the process
    Priority int              // Process priority for scheduling
    BurstTime int             // CPU time required for execution
    ArrivalTime time.Time     // When process arrived in the system
    mutex    sync.Mutex       // Protects concurrent access to process state
}

// ChangeState safely transitions process from one state to another
// Thread-safe state transition with logging
func (p *Process) ChangeState(newState ProcessState) {
    p.mutex.Lock()         // Acquire lock for thread safety
    defer p.mutex.Unlock() // Ensure lock is released
    
    oldState := p.State
    p.State = newState     // Update process state
    
    // Log state transition for debugging/monitoring
    fmt.Printf("Process %d: %s -> %s\n", p.ID,
        getStateName(oldState), getStateName(newState))
}

// getStateName converts ProcessState enum to human-readable string
func getStateName(state ProcessState) string {
    switch state {
    case New: return "New"
    case Ready: return "Ready"
    case Running: return "Running"
    case Waiting: return "Waiting"
    case Terminated: return "Terminated"
    default: return "Unknown"
    }
}

func main() {
    // Create a new process
    process := &Process{
        ID: 1,
        State: New,              // Start in New state
        Priority: 5,             // Medium priority
        BurstTime: 10,           // 10 time units of CPU time needed
        ArrivalTime: time.Now(), // Record arrival time
    }

    // Simulate complete process lifecycle
    process.ChangeState(Ready)   // Process is ready to run
    time.Sleep(100 * time.Millisecond)
    
    process.ChangeState(Running) // Process starts executing
    time.Sleep(200 * time.Millisecond)
    
    process.ChangeState(Waiting) // Process waits for I/O
    time.Sleep(100 * time.Millisecond)
    
    process.ChangeState(Running) // Process resumes execution
    time.Sleep(300 * time.Millisecond)
    
    process.ChangeState(Terminated) // Process completes
}
```

**Key Concepts Explained:**
- **Process States**: New, Ready, Running, Waiting, Terminated
- **State Transitions**: Processes move between states based on events
- **Thread Safety**: Mutex protects process state from race conditions
- **Process Attributes**: ID, priority, burst time, arrival time
- **Lifecycle Management**: Complete process lifecycle simulation

### **2. Process Creation & Termination**

#### **Fork-like Process Creation**

```go
package main

import (
    "fmt"
    "os"
    "os/exec"
    "syscall"
)

type ProcessManager struct {
    processes map[int]*Process
    nextPID   int
    mutex     sync.Mutex
}

func NewProcessManager() *ProcessManager {
    return &ProcessManager{
        processes: make(map[int]*Process),
        nextPID:   1,
    }
}

func (pm *ProcessManager) CreateProcess(command string, args []string) (*Process, error) {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()

    pid := pm.nextPID
    pm.nextPID++

    process := &Process{
        ID: pid,
        State: New,
        Priority: 5,
        ArrivalTime: time.Now(),
    }

    pm.processes[pid] = process

    // In real OS, this would fork and exec
    go pm.executeProcess(process, command, args)

    return process, nil
}

func (pm *ProcessManager) executeProcess(process *Process, command string, args []string) {
    process.ChangeState(Ready)
    process.ChangeState(Running)

    // Simulate process execution
    time.Sleep(time.Duration(process.BurstTime) * time.Millisecond)

    process.ChangeState(Terminated)

    pm.mutex.Lock()
    delete(pm.processes, process.ID)
    pm.mutex.Unlock()
}

func (pm *ProcessManager) TerminateProcess(pid int) error {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()

    process, exists := pm.processes[pid]
    if !exists {
        return fmt.Errorf("process %d not found", pid)
    }

    process.ChangeState(Terminated)
    delete(pm.processes, pid)

    return nil
}

func main() {
    pm := NewProcessManager()

    // Create processes
    process1, _ := pm.CreateProcess("ls", []string{"-la"})
    process2, _ := pm.CreateProcess("ps", []string{"aux"})

    fmt.Printf("Created processes: %d, %d\n", process1.ID, process2.ID)

    // Wait for processes to complete
    time.Sleep(2 * time.Second)

    // Terminate a process
    pm.TerminateProcess(process1.ID)
}
```

---

## 🧠 Memory Management

### **3. Memory Allocation**

#### **First Fit Memory Allocation**

```go
package main

import (
    "fmt"
    "sort"
)

type MemoryBlock struct {
    Start   int
    Size    int
    IsFree  bool
    ProcessID int
}

type MemoryManager struct {
    blocks []MemoryBlock
    totalSize int
}

func NewMemoryManager(totalSize int) *MemoryManager {
    return &MemoryManager{
        blocks: []MemoryBlock{
            {Start: 0, Size: totalSize, IsFree: true, ProcessID: -1},
        },
        totalSize: totalSize,
    }
}

func (mm *MemoryManager) AllocateMemory(processID, size int) (int, error) {
    // First fit algorithm
    for i, block := range mm.blocks {
        if block.IsFree && block.Size >= size {
            if block.Size == size {
                // Exact fit
                mm.blocks[i].IsFree = false
                mm.blocks[i].ProcessID = processID
                return block.Start, nil
            } else {
                // Split the block
                newBlock := MemoryBlock{
                    Start: block.Start + size,
                    Size: block.Size - size,
                    IsFree: true,
                    ProcessID: -1,
                }

                mm.blocks[i].Size = size
                mm.blocks[i].IsFree = false
                mm.blocks[i].ProcessID = processID

                // Insert new block after current block
                mm.blocks = append(mm.blocks[:i+1], append([]MemoryBlock{newBlock}, mm.blocks[i+1:]...)...)

                return block.Start, nil
            }
        }
    }

    return -1, fmt.Errorf("insufficient memory")
}

func (mm *MemoryManager) DeallocateMemory(processID int) error {
    for i, block := range mm.blocks {
        if block.ProcessID == processID {
            mm.blocks[i].IsFree = true
            mm.blocks[i].ProcessID = -1

            // Merge with adjacent free blocks
            mm.mergeFreeBlocks()
            return nil
        }
    }

    return fmt.Errorf("process %d not found", processID)
}

func (mm *MemoryManager) mergeFreeBlocks() {
    // Sort blocks by start address
    sort.Slice(mm.blocks, func(i, j int) bool {
        return mm.blocks[i].Start < mm.blocks[j].Start
    })

    // Merge adjacent free blocks
    for i := 0; i < len(mm.blocks)-1; i++ {
        if mm.blocks[i].IsFree && mm.blocks[i+1].IsFree &&
           mm.blocks[i].Start+mm.blocks[i].Size == mm.blocks[i+1].Start {

            mm.blocks[i].Size += mm.blocks[i+1].Size
            mm.blocks = append(mm.blocks[:i+1], mm.blocks[i+2:]...)
            i-- // Check again from current position
        }
    }
}

func (mm *MemoryManager) PrintMemoryLayout() {
    fmt.Println("Memory Layout:")
    for _, block := range mm.blocks {
        status := "Free"
        if !block.IsFree {
            status = fmt.Sprintf("Process %d", block.ProcessID)
        }
        fmt.Printf("Address: %d-%d, Size: %d, Status: %s\n",
            block.Start, block.Start+block.Size-1, block.Size, status)
    }
}

func main() {
    mm := NewMemoryManager(1000)

    // Allocate memory for processes
    addr1, _ := mm.AllocateMemory(1, 200)
    addr2, _ := mm.AllocateMemory(2, 300)
    addr3, _ := mm.AllocateMemory(3, 150)

    fmt.Printf("Allocated addresses: %d, %d, %d\n", addr1, addr2, addr3)
    mm.PrintMemoryLayout()

    // Deallocate process 2
    mm.DeallocateMemory(2)
    fmt.Println("\nAfter deallocating process 2:")
    mm.PrintMemoryLayout()
}
```

### **4. Page Replacement Algorithms**

#### **LRU (Least Recently Used)**

```go
package main

import (
    "container/list"
    "fmt"
)

type LRUCache struct {
    capacity int
    cache    map[int]*list.Element
    list     *list.List
}

type Page struct {
    key   int
    value string
}

func NewLRUCache(capacity int) *LRUCache {
    return &LRUCache{
        capacity: capacity,
        cache:    make(map[int]*list.Element),
        list:     list.New(),
    }
}

func (lru *LRUCache) Get(key int) (string, bool) {
    if elem, exists := lru.cache[key]; exists {
        lru.list.MoveToFront(elem)
        return elem.Value.(*Page).value, true
    }
    return "", false
}

func (lru *LRUCache) Put(key int, value string) {
    if elem, exists := lru.cache[key]; exists {
        elem.Value.(*Page).value = value
        lru.list.MoveToFront(elem)
        return
    }

    if len(lru.cache) >= lru.capacity {
        // Remove least recently used
        back := lru.list.Back()
        delete(lru.cache, back.Value.(*Page).key)
        lru.list.Remove(back)
    }

    // Add new page
    page := &Page{key: key, value: value}
    elem := lru.list.PushFront(page)
    lru.cache[key] = elem
}

func (lru *LRUCache) PrintCache() {
    fmt.Print("Cache: ")
    for elem := lru.list.Front(); elem != nil; elem = elem.Next() {
        page := elem.Value.(*Page)
        fmt.Printf("%d:%s ", page.key, page.value)
    }
    fmt.Println()
}

func main() {
    lru := NewLRUCache(3)

    // Simulate page references
    pages := []int{1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5}

    for _, page := range pages {
        if _, exists := lru.Get(page); !exists {
            fmt.Printf("Page fault for page %d\n", page)
            lru.Put(page, fmt.Sprintf("data%d", page))
        } else {
            fmt.Printf("Page hit for page %d\n", page)
        }
        lru.PrintCache()
    }
}
```

---

## 📁 File Systems

### **5. File System Operations**

#### **Basic File System**

```go
package main

import (
    "fmt"
    "os"
    "path/filepath"
    "sync"
    "time"
)

type File struct {
    Name     string
    Size     int64
    Mode     os.FileMode
    ModTime  time.Time
    Content  []byte
    mutex    sync.RWMutex
}

type Directory struct {
    Name     string
    Files    map[string]*File
    Dirs     map[string]*Directory
    mutex    sync.RWMutex
}

type FileSystem struct {
    root     *Directory
    mutex    sync.RWMutex
}

func NewFileSystem() *FileSystem {
    return &FileSystem{
        root: &Directory{
            Name:  "/",
            Files: make(map[string]*File),
            Dirs:  make(map[string]*Directory),
        },
    }
}

func (fs *FileSystem) CreateFile(path, content string) error {
    fs.mutex.Lock()
    defer fs.mutex.Unlock()

    dir, filename := filepath.Split(path)
    parentDir := fs.getDirectory(dir)

    if parentDir == nil {
        return fmt.Errorf("directory not found: %s", dir)
    }

    parentDir.mutex.Lock()
    defer parentDir.mutex.Unlock()

    file := &File{
        Name:    filename,
        Size:    int64(len(content)),
        Mode:    0644,
        ModTime: time.Now(),
        Content: []byte(content),
    }

    parentDir.Files[filename] = file
    return nil
}

func (fs *FileSystem) ReadFile(path string) (string, error) {
    fs.mutex.RLock()
    defer fs.mutex.RUnlock()

    dir, filename := filepath.Split(path)
    parentDir := fs.getDirectory(dir)

    if parentDir == nil {
        return "", fmt.Errorf("directory not found: %s", dir)
    }

    parentDir.mutex.RLock()
    defer parentDir.mutex.RUnlock()

    file, exists := parentDir.Files[filename]
    if !exists {
        return "", fmt.Errorf("file not found: %s", filename)
    }

    file.mutex.RLock()
    defer file.mutex.RUnlock()

    return string(file.Content), nil
}

func (fs *FileSystem) getDirectory(path string) *Directory {
    if path == "/" || path == "" {
        return fs.root
    }

    parts := filepath.SplitList(path)
    current := fs.root

    for _, part := range parts {
        if part == "" || part == "/" {
            continue
        }

        current.mutex.RLock()
        dir, exists := current.Dirs[part]
        current.mutex.RUnlock()

        if !exists {
            return nil
        }
        current = dir
    }

    return current
}

func (fs *FileSystem) ListDirectory(path string) ([]string, error) {
    fs.mutex.RLock()
    defer fs.mutex.RUnlock()

    dir := fs.getDirectory(path)
    if dir == nil {
        return nil, fmt.Errorf("directory not found: %s", path)
    }

    dir.mutex.RLock()
    defer dir.mutex.RUnlock()

    var items []string

    for name := range dir.Files {
        items = append(items, name)
    }

    for name := range dir.Dirs {
        items = append(items, name+"/")
    }

    return items, nil
}

func main() {
    fs := NewFileSystem()

    // Create files
    fs.CreateFile("/file1.txt", "Hello, World!")
    fs.CreateFile("/file2.txt", "Go File System")

    // Read files
    content1, _ := fs.ReadFile("/file1.txt")
    content2, _ := fs.ReadFile("/file2.txt")

    fmt.Printf("File1 content: %s\n", content1)
    fmt.Printf("File2 content: %s\n", content2)

    // List directory
    items, _ := fs.ListDirectory("/")
    fmt.Printf("Root directory contents: %v\n", items)
}
```

---

## ⏰ Scheduling Algorithms

### **6. Round Robin Scheduling**

```go
package main

import (
    "fmt"
    "time"
)

type Process struct {
    ID         int
    BurstTime  int
    RemainingTime int
    ArrivalTime time.Time
}

type RoundRobinScheduler struct {
    processes []Process
    timeQuantum int
}

func NewRoundRobinScheduler(timeQuantum int) *RoundRobinScheduler {
    return &RoundRobinScheduler{
        timeQuantum: timeQuantum,
    }
}

func (rr *RoundRobinScheduler) AddProcess(process Process) {
    process.RemainingTime = process.BurstTime
    rr.processes = append(rr.processes, process)
}

func (rr *RoundRobinScheduler) Schedule() {
    queue := make([]Process, len(rr.processes))
    copy(queue, rr.processes)

    fmt.Println("Round Robin Scheduling:")
    fmt.Printf("Time Quantum: %d\n", rr.timeQuantum)
    fmt.Println("Process Execution:")

    currentTime := 0

    for len(queue) > 0 {
        process := queue[0]
        queue = queue[1:]

        if process.RemainingTime <= rr.timeQuantum {
            // Process completes
            fmt.Printf("Time %d-%d: Process %d (completes)\n",
                currentTime, currentTime+process.RemainingTime, process.ID)
            currentTime += process.RemainingTime
        } else {
            // Process uses full time quantum
            fmt.Printf("Time %d-%d: Process %d\n",
                currentTime, currentTime+rr.timeQuantum, process.ID)
            currentTime += rr.timeQuantum
            process.RemainingTime -= rr.timeQuantum

            // Add back to queue
            queue = append(queue, process)
        }
    }
}

func main() {
    scheduler := NewRoundRobinScheduler(2)

    // Add processes
    scheduler.AddProcess(Process{ID: 1, BurstTime: 5, ArrivalTime: time.Now()})
    scheduler.AddProcess(Process{ID: 2, BurstTime: 3, ArrivalTime: time.Now()})
    scheduler.AddProcess(Process{ID: 3, BurstTime: 4, ArrivalTime: time.Now()})

    scheduler.Schedule()
}
```

### **7. Priority Scheduling**

```go
package main

import (
    "fmt"
    "sort"
    "time"
)

type PriorityProcess struct {
    ID         int
    BurstTime  int
    Priority   int
    ArrivalTime time.Time
}

type PriorityScheduler struct {
    processes []PriorityProcess
}

func NewPriorityScheduler() *PriorityScheduler {
    return &PriorityScheduler{}
}

func (ps *PriorityScheduler) AddProcess(process PriorityProcess) {
    ps.processes = append(ps.processes, process)
}

func (ps *PriorityScheduler) Schedule() {
    // Sort by priority (lower number = higher priority)
    sort.Slice(ps.processes, func(i, j int) bool {
        return ps.processes[i].Priority < ps.processes[j].Priority
    })

    fmt.Println("Priority Scheduling:")
    fmt.Println("Process Execution:")

    currentTime := 0

    for _, process := range ps.processes {
        fmt.Printf("Time %d-%d: Process %d (Priority: %d)\n",
            currentTime, currentTime+process.BurstTime, process.ID, process.Priority)
        currentTime += process.BurstTime
    }
}

func main() {
    scheduler := NewPriorityScheduler()

    // Add processes with different priorities
    scheduler.AddProcess(PriorityProcess{ID: 1, BurstTime: 5, Priority: 3, ArrivalTime: time.Now()})
    scheduler.AddProcess(PriorityProcess{ID: 2, BurstTime: 3, Priority: 1, ArrivalTime: time.Now()})
    scheduler.AddProcess(PriorityProcess{ID: 3, BurstTime: 4, Priority: 2, ArrivalTime: time.Now()})

    scheduler.Schedule()
}
```

---

## 🔗 Inter-Process Communication

### **8. Shared Memory**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type SharedMemory struct {
    data  map[string]interface{}
    mutex sync.RWMutex
}

func NewSharedMemory() *SharedMemory {
    return &SharedMemory{
        data: make(map[string]interface{}),
    }
}

func (sm *SharedMemory) Write(key string, value interface{}) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()

    sm.data[key] = value
    fmt.Printf("Process wrote: %s = %v\n", key, value)
}

func (sm *SharedMemory) Read(key string) (interface{}, bool) {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()

    value, exists := sm.data[key]
    if exists {
        fmt.Printf("Process read: %s = %v\n", key, value)
    }
    return value, exists
}

func producer(sm *SharedMemory, id int) {
    for i := 0; i < 5; i++ {
        key := fmt.Sprintf("producer%d_item%d", id, i)
        value := fmt.Sprintf("data%d", i)
        sm.Write(key, value)
        time.Sleep(100 * time.Millisecond)
    }
}

func consumer(sm *SharedMemory, id int) {
    for i := 0; i < 5; i++ {
        key := fmt.Sprintf("producer%d_item%d", id, i)
        if value, exists := sm.Read(key); exists {
            fmt.Printf("Consumer %d processed: %v\n", id, value)
        }
        time.Sleep(150 * time.Millisecond)
    }
}

func main() {
    sm := NewSharedMemory()

    // Start producer and consumer goroutines
    go producer(sm, 1)
    go consumer(sm, 1)

    time.Sleep(2 * time.Second)
}
```

### **9. Message Passing**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Message struct {
    SenderID int
    Content  string
    Timestamp time.Time
}

type MessageQueue struct {
    messages chan Message
    mutex    sync.Mutex
}

func NewMessageQueue(bufferSize int) *MessageQueue {
    return &MessageQueue{
        messages: make(chan Message, bufferSize),
    }
}

func (mq *MessageQueue) Send(senderID int, content string) {
    message := Message{
        SenderID:  senderID,
        Content:   content,
        Timestamp: time.Now(),
    }

    select {
    case mq.messages <- message:
        fmt.Printf("Process %d sent: %s\n", senderID, content)
    default:
        fmt.Printf("Process %d: Queue full, message dropped\n", senderID)
    }
}

func (mq *MessageQueue) Receive() (Message, bool) {
    select {
    case message := <-mq.messages:
        fmt.Printf("Received from process %d: %s\n", message.SenderID, message.Content)
        return message, true
    case <-time.After(1 * time.Second):
        return Message{}, false
    }
}

func sender(mq *MessageQueue, id int) {
    for i := 0; i < 3; i++ {
        content := fmt.Sprintf("Message %d from sender %d", i, id)
        mq.Send(id, content)
        time.Sleep(200 * time.Millisecond)
    }
}

func receiver(mq *MessageQueue) {
    for i := 0; i < 6; i++ {
        if message, received := mq.Receive(); received {
            fmt.Printf("Receiver processed: %s\n", message.Content)
        }
        time.Sleep(100 * time.Millisecond)
    }
}

func main() {
    mq := NewMessageQueue(5)

    // Start senders and receiver
    go sender(mq, 1)
    go sender(mq, 2)
    go receiver(mq)

    time.Sleep(3 * time.Second)
}
```

---

## 🔒 Deadlocks & Synchronization

### **10. Deadlock Detection**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Resource struct {
    ID   int
    mutex sync.Mutex
}

type Process struct {
    ID        int
    Resources []*Resource
    mutex     sync.Mutex
}

type DeadlockDetector struct {
    processes map[int]*Process
    resources map[int]*Resource
    mutex     sync.RWMutex
}

func NewDeadlockDetector() *DeadlockDetector {
    return &DeadlockDetector{
        processes: make(map[int]*Process),
        resources: make(map[int]*Resource),
    }
}

func (dd *DeadlockDetector) AddResource(id int) *Resource {
    dd.mutex.Lock()
    defer dd.mutex.Unlock()

    resource := &Resource{ID: id}
    dd.resources[id] = resource
    return resource
}

func (dd *DeadlockDetector) AddProcess(id int) *Process {
    dd.mutex.Lock()
    defer dd.mutex.Unlock()

    process := &Process{ID: id}
    dd.processes[id] = process
    return process
}

func (dd *DeadlockDetector) RequestResource(processID, resourceID int) bool {
    dd.mutex.RLock()
    process := dd.processes[processID]
    resource := dd.resources[resourceID]
    dd.mutex.RUnlock()

    if process == nil || resource == nil {
        return false
    }

    // Try to acquire resource
    acquired := resource.mutex.TryLock()
    if acquired {
        process.mutex.Lock()
        process.Resources = append(process.Resources, resource)
        process.mutex.Unlock()
        fmt.Printf("Process %d acquired resource %d\n", processID, resourceID)
        return true
    } else {
        fmt.Printf("Process %d waiting for resource %d\n", processID, resourceID)
        return false
    }
}

func (dd *DeadlockDetector) ReleaseResource(processID, resourceID int) {
    dd.mutex.RLock()
    process := dd.processes[processID]
    resource := dd.resources[resourceID]
    dd.mutex.RUnlock()

    if process == nil || resource == nil {
        return
    }

    process.mutex.Lock()
    for i, res := range process.Resources {
        if res.ID == resourceID {
            process.Resources = append(process.Resources[:i], process.Resources[i+1:]...)
            break
        }
    }
    process.mutex.Unlock()

    resource.mutex.Unlock()
    fmt.Printf("Process %d released resource %d\n", processID, resourceID)
}

func (dd *DeadlockDetector) DetectDeadlock() []int {
    dd.mutex.RLock()
    defer dd.mutex.RUnlock()

    // Simple deadlock detection using cycle detection
    visited := make(map[int]bool)
    recStack := make(map[int]bool)

    var deadlockedProcesses []int

    for processID := range dd.processes {
        if !visited[processID] {
            if dd.hasCycle(processID, visited, recStack) {
                deadlockedProcesses = append(deadlockedProcesses, processID)
            }
        }
    }

    return deadlockedProcesses
}

func (dd *DeadlockDetector) hasCycle(processID int, visited, recStack map[int]bool) bool {
    visited[processID] = true
    recStack[processID] = true

    process := dd.processes[processID]
    for _, resource := range process.Resources {
        // Check if any other process is waiting for this resource
        for otherProcessID, otherProcess := range dd.processes {
            if otherProcessID != processID {
                for _, otherResource := range otherProcess.Resources {
                    if otherResource.ID == resource.ID {
                        if !visited[otherProcessID] && dd.hasCycle(otherProcessID, visited, recStack) {
                            return true
                        } else if recStack[otherProcessID] {
                            return true
                        }
                    }
                }
            }
        }
    }

    recStack[processID] = false
    return false
}

func main() {
    dd := NewDeadlockDetector()

    // Create resources
    r1 := dd.AddResource(1)
    r2 := dd.AddResource(2)

    // Create processes
    p1 := dd.AddProcess(1)
    p2 := dd.AddProcess(2)

    // Simulate deadlock scenario
    dd.RequestResource(1, 1) // P1 gets R1
    dd.RequestResource(2, 2) // P2 gets R2

    // This will cause deadlock
    dd.RequestResource(1, 2) // P1 waits for R2
    dd.RequestResource(2, 1) // P2 waits for R1

    time.Sleep(100 * time.Millisecond)

    // Detect deadlock
    deadlocked := dd.DetectDeadlock()
    if len(deadlocked) > 0 {
        fmt.Printf("Deadlock detected involving processes: %v\n", deadlocked)
    } else {
        fmt.Println("No deadlock detected")
    }
}
```

---

## 🎯 FAANG Interview Questions

### **Google Interview Questions**

#### **1. Implement a Process Scheduler**

**Question**: "Design a process scheduler that can handle different scheduling algorithms."

**Answer**:

```go
package main

import (
    "container/heap"
    "fmt"
    "time"
)

type Process struct {
    ID         int
    BurstTime  int
    Priority   int
    ArrivalTime time.Time
}

type Scheduler interface {
    Schedule(processes []Process) []Process
}

type FCFSScheduler struct{}

func (fcfs *FCFSScheduler) Schedule(processes []Process) []Process {
    // First Come First Served - sort by arrival time
    sorted := make([]Process, len(processes))
    copy(sorted, processes)

    for i := 0; i < len(sorted)-1; i++ {
        for j := i + 1; j < len(sorted); j++ {
            if sorted[i].ArrivalTime.After(sorted[j].ArrivalTime) {
                sorted[i], sorted[j] = sorted[j], sorted[i]
            }
        }
    }

    return sorted
}

type PriorityScheduler struct{}

func (ps *PriorityScheduler) Schedule(processes []Process) []Process {
    // Priority scheduling - sort by priority (lower number = higher priority)
    sorted := make([]Process, len(processes))
    copy(sorted, processes)

    for i := 0; i < len(sorted)-1; i++ {
        for j := i + 1; j < len(sorted); j++ {
            if sorted[i].Priority > sorted[j].Priority {
                sorted[i], sorted[j] = sorted[j], sorted[i]
            }
        }
    }

    return sorted
}

type SchedulerManager struct {
    scheduler Scheduler
}

func NewSchedulerManager(scheduler Scheduler) *SchedulerManager {
    return &SchedulerManager{scheduler: scheduler}
}

func (sm *SchedulerManager) ExecuteSchedule(processes []Process) {
    scheduled := sm.scheduler.Schedule(processes)

    fmt.Println("Scheduled processes:")
    currentTime := 0

    for _, process := range scheduled {
        fmt.Printf("Time %d-%d: Process %d (Burst: %d, Priority: %d)\n",
            currentTime, currentTime+process.BurstTime, process.ID, process.BurstTime, process.Priority)
        currentTime += process.BurstTime
    }
}

func main() {
    processes := []Process{
        {ID: 1, BurstTime: 5, Priority: 3, ArrivalTime: time.Now()},
        {ID: 2, BurstTime: 3, Priority: 1, ArrivalTime: time.Now().Add(time.Second)},
        {ID: 3, BurstTime: 4, Priority: 2, ArrivalTime: time.Now().Add(2 * time.Second)},
    }

    // FCFS Scheduler
    fcfsManager := NewSchedulerManager(&FCFSScheduler{})
    fmt.Println("FCFS Scheduling:")
    fcfsManager.ExecuteSchedule(processes)

    fmt.Println()

    // Priority Scheduler
    priorityManager := NewSchedulerManager(&PriorityScheduler{})
    fmt.Println("Priority Scheduling:")
    priorityManager.ExecuteSchedule(processes)
}
```

### **Meta Interview Questions**

#### **2. Implement a Memory Manager**

**Question**: "Design a memory manager with different allocation strategies."

**Answer**:

```go
package main

import (
    "fmt"
    "sort"
)

type MemoryBlock struct {
    Start   int
    Size    int
    IsFree  bool
    ProcessID int
}

type AllocationStrategy int

const (
    FirstFit AllocationStrategy = iota
    BestFit
    WorstFit
)

type MemoryManager struct {
    blocks    []MemoryBlock
    strategy  AllocationStrategy
    totalSize int
}

func NewMemoryManager(totalSize int, strategy AllocationStrategy) *MemoryManager {
    return &MemoryManager{
        blocks: []MemoryBlock{
            {Start: 0, Size: totalSize, IsFree: true, ProcessID: -1},
        },
        strategy:  strategy,
        totalSize: totalSize,
    }
}

func (mm *MemoryManager) AllocateMemory(processID, size int) (int, error) {
    switch mm.strategy {
    case FirstFit:
        return mm.firstFit(processID, size)
    case BestFit:
        return mm.bestFit(processID, size)
    case WorstFit:
        return mm.worstFit(processID, size)
    default:
        return -1, fmt.Errorf("unknown allocation strategy")
    }
}

func (mm *MemoryManager) firstFit(processID, size int) (int, error) {
    for i, block := range mm.blocks {
        if block.IsFree && block.Size >= size {
            return mm.allocateBlock(i, processID, size)
        }
    }
    return -1, fmt.Errorf("insufficient memory")
}

func (mm *MemoryManager) bestFit(processID, size int) (int, error) {
    bestIndex := -1
    bestSize := mm.totalSize + 1

    for i, block := range mm.blocks {
        if block.IsFree && block.Size >= size && block.Size < bestSize {
            bestIndex = i
            bestSize = block.Size
        }
    }

    if bestIndex == -1 {
        return -1, fmt.Errorf("insufficient memory")
    }

    return mm.allocateBlock(bestIndex, processID, size)
}

func (mm *MemoryManager) worstFit(processID, size int) (int, error) {
    worstIndex := -1
    worstSize := -1

    for i, block := range mm.blocks {
        if block.IsFree && block.Size >= size && block.Size > worstSize {
            worstIndex = i
            worstSize = block.Size
        }
    }

    if worstIndex == -1 {
        return -1, fmt.Errorf("insufficient memory")
    }

    return mm.allocateBlock(worstIndex, processID, size)
}

func (mm *MemoryManager) allocateBlock(index, processID, size int) (int, error) {
    block := mm.blocks[index]

    if block.Size == size {
        // Exact fit
        mm.blocks[index].IsFree = false
        mm.blocks[index].ProcessID = processID
        return block.Start, nil
    } else {
        // Split the block
        newBlock := MemoryBlock{
            Start: block.Start + size,
            Size: block.Size - size,
            IsFree: true,
            ProcessID: -1,
        }

        mm.blocks[index].Size = size
        mm.blocks[index].IsFree = false
        mm.blocks[index].ProcessID = processID

        // Insert new block after current block
        mm.blocks = append(mm.blocks[:index+1], append([]MemoryBlock{newBlock}, mm.blocks[index+1:]...)...)

        return block.Start, nil
    }
}

func (mm *MemoryManager) DeallocateMemory(processID int) error {
    for i, block := range mm.blocks {
        if block.ProcessID == processID {
            mm.blocks[i].IsFree = true
            mm.blocks[i].ProcessID = -1

            // Merge with adjacent free blocks
            mm.mergeFreeBlocks()
            return nil
        }
    }

    return fmt.Errorf("process %d not found", processID)
}

func (mm *MemoryManager) mergeFreeBlocks() {
    // Sort blocks by start address
    sort.Slice(mm.blocks, func(i, j int) bool {
        return mm.blocks[i].Start < mm.blocks[j].Start
    })

    // Merge adjacent free blocks
    for i := 0; i < len(mm.blocks)-1; i++ {
        if mm.blocks[i].IsFree && mm.blocks[i+1].IsFree &&
           mm.blocks[i].Start+mm.blocks[i].Size == mm.blocks[i+1].Start {

            mm.blocks[i].Size += mm.blocks[i+1].Size
            mm.blocks = append(mm.blocks[:i+1], mm.blocks[i+2:]...)
            i-- // Check again from current position
        }
    }
}

func (mm *MemoryManager) PrintMemoryLayout() {
    fmt.Println("Memory Layout:")
    for _, block := range mm.blocks {
        status := "Free"
        if !block.IsFree {
            status = fmt.Sprintf("Process %d", block.ProcessID)
        }
        fmt.Printf("Address: %d-%d, Size: %d, Status: %s\n",
            block.Start, block.Start+block.Size-1, block.Size, status)
    }
}

func main() {
    // Test different allocation strategies
    strategies := []AllocationStrategy{FirstFit, BestFit, WorstFit}
    strategyNames := []string{"First Fit", "Best Fit", "Worst Fit"}

    for i, strategy := range strategies {
        fmt.Printf("\n=== %s ===\n", strategyNames[i])

        mm := NewMemoryManager(1000, strategy)

        // Allocate memory for processes
        addr1, _ := mm.AllocateMemory(1, 200)
        addr2, _ := mm.AllocateMemory(2, 300)
        addr3, _ := mm.AllocateMemory(3, 150)

        fmt.Printf("Allocated addresses: %d, %d, %d\n", addr1, addr2, addr3)
        mm.PrintMemoryLayout()

        // Deallocate process 2
        mm.DeallocateMemory(2)
        fmt.Println("\nAfter deallocating process 2:")
        mm.PrintMemoryLayout()
    }
}
```

---

## 📚 Additional Resources

### **Books**

- [Operating System Concepts](https://www.os-book.com/) - Abraham Silberschatz
- [Modern Operating Systems](https://www.pearson.com/us/higher-education/program/Tanenbaum-Modern-Operating-Systems-4th-Edition/PGM241619.html) - Andrew Tanenbaum
- [Operating Systems: Three Easy Pieces](http://pages.cs.wisc.edu/~remzi/OSTEP/) - Remzi Arpaci-Dusseau

### **Online Resources**

- [OS Dev Wiki](https://wiki.osdev.org/) - Operating system development
- [Linux Kernel Documentation](https://www.kernel.org/doc/) - Linux kernel internals
- [FreeBSD Handbook](https://docs.freebsd.org/en/books/handbook/) - FreeBSD operating system

### **Video Resources**

- [CS 162: Operating Systems](https://www.youtube.com/playlist?list=PL-XXv-cvA_iBDyz-ba4yDskqMDY6A1w_c) - UC Berkeley
- [Operating System Concepts](https://www.youtube.com/playlist?list=PLbMVogVj5nJQ2uWfLpByyqKqBE4XkE4iR) - Tutorials Point
- [Linux Kernel Development](https://www.youtube.com/playlist?list=PLrAXtmRdnEQy6nuLMOVlF7WwvU0n1-Gxc) - Linux Foundation

---

_This comprehensive guide covers essential operating system concepts with practical Go implementations and real-world interview questions from top tech companies._
