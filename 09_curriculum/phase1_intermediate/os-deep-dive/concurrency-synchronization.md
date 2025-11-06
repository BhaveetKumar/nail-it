---
# Auto-generated front matter
Title: Concurrency-Synchronization
LastUpdated: 2025-11-06T20:45:58.444369
Tags: []
Status: draft
---

# Concurrency and Synchronization

## Overview

This module covers concurrency and synchronization concepts including threads, processes, locks, semaphores, monitors, and deadlock prevention. These concepts are essential for understanding how operating systems manage concurrent execution safely and efficiently.

## Table of Contents

1. [Thread Management](#thread-management)
2. [Synchronization Primitives](#synchronization-primitives)
3. [Deadlock Prevention](#deadlock-prevention)
4. [Producer-Consumer Problem](#producer-consumer-problem)
5. [Reader-Writer Problem](#reader-writer-problem)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## Thread Management

### Theory

Threads are lightweight processes that share the same memory space and resources. Thread management involves creating, scheduling, and synchronizing threads to achieve concurrent execution.

### Thread Pool Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Task struct {
    ID       int
    Function func() interface{}
    Result   chan interface{}
}

type ThreadPool struct {
    workers    int
    taskQueue  chan Task
    wg         sync.WaitGroup
    quit       chan bool
    mutex      sync.RWMutex
    running    bool
}

func NewThreadPool(workers int) *ThreadPool {
    return &ThreadPool{
        workers:   workers,
        taskQueue: make(chan Task, 100),
        quit:      make(chan bool),
        running:   false,
    }
}

func (tp *ThreadPool) Start() {
    tp.mutex.Lock()
    defer tp.mutex.Unlock()
    
    if tp.running {
        return
    }
    
    tp.running = true
    
    for i := 0; i < tp.workers; i++ {
        tp.wg.Add(1)
        go tp.worker(i)
    }
    
    fmt.Printf("Started thread pool with %d workers\n", tp.workers)
}

func (tp *ThreadPool) Stop() {
    tp.mutex.Lock()
    defer tp.mutex.Unlock()
    
    if !tp.running {
        return
    }
    
    tp.running = false
    close(tp.quit)
    
    tp.wg.Wait()
    fmt.Println("Thread pool stopped")
}

func (tp *ThreadPool) worker(id int) {
    defer tp.wg.Done()
    
    fmt.Printf("Worker %d started\n", id)
    
    for {
        select {
        case task := <-tp.taskQueue:
            fmt.Printf("Worker %d executing task %d\n", id, task.ID)
            
            // Execute the task
            result := task.Function()
            
            // Send result back
            select {
            case task.Result <- result:
            case <-time.After(1 * time.Second):
                fmt.Printf("Worker %d: timeout sending result for task %d\n", id, task.ID)
            }
            
        case <-tp.quit:
            fmt.Printf("Worker %d stopping\n", id)
            return
        }
    }
}

func (tp *ThreadPool) SubmitTask(id int, function func() interface{}) <-chan interface{} {
    result := make(chan interface{}, 1)
    
    task := Task{
        ID:       id,
        Function: function,
        Result:   result,
    }
    
    select {
    case tp.taskQueue <- task:
        fmt.Printf("Submitted task %d\n", id)
    case <-time.After(1 * time.Second):
        fmt.Printf("Failed to submit task %d (queue full)\n", id)
        close(result)
    }
    
    return result
}

func (tp *ThreadPool) GetQueueLength() int {
    return len(tp.taskQueue)
}

func (tp *ThreadPool) GetStatus() {
    tp.mutex.RLock()
    defer tp.mutex.RUnlock()
    
    fmt.Printf("Thread Pool Status:\n")
    fmt.Printf("  Workers: %d\n", tp.workers)
    fmt.Printf("  Running: %t\n", tp.running)
    fmt.Printf("  Queue Length: %d\n", tp.GetQueueLength())
}

func main() {
    pool := NewThreadPool(3)
    
    fmt.Println("Thread Pool Demo:")
    
    // Start the thread pool
    pool.Start()
    
    // Submit some tasks
    for i := 0; i < 5; i++ {
        taskID := i
        result := pool.SubmitTask(taskID, func() interface{} {
            // Simulate some work
            time.Sleep(time.Duration(100+taskID*50) * time.Millisecond)
            return fmt.Sprintf("Task %d completed", taskID)
        })
        
        // Wait for result
        go func(id int, res <-chan interface{}) {
            select {
            case result := <-res:
                fmt.Printf("Received result for task %d: %v\n", id, result)
            case <-time.After(2 * time.Second):
                fmt.Printf("Timeout waiting for task %d result\n", id)
            }
        }(taskID, result)
    }
    
    // Wait a bit for tasks to complete
    time.Sleep(2 * time.Second)
    
    // Check status
    pool.GetStatus()
    
    // Stop the thread pool
    pool.Stop()
}
```

## Synchronization Primitives

### Theory

Synchronization primitives are mechanisms that ensure proper coordination between threads to prevent race conditions and ensure data consistency.

### Mutex Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type SafeCounter struct {
    mu    sync.Mutex
    count int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}

func (c *SafeCounter) Decrement() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count--
}

func (c *SafeCounter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.count
}

func (c *SafeCounter) IncrementUnsafe() {
    c.count++
}

func (c *SafeCounter) DecrementUnsafe() {
    c.count--
}

func (c *SafeCounter) ValueUnsafe() int {
    return c.count
}

func main() {
    counter := &SafeCounter{}
    
    fmt.Println("Mutex Demo:")
    
    // Test with mutex
    fmt.Println("Testing with mutex:")
    var wg sync.WaitGroup
    
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.Increment()
        }()
    }
    
    wg.Wait()
    fmt.Printf("Final count with mutex: %d\n", counter.Value())
    
    // Test without mutex (race condition)
    counter.count = 0
    fmt.Println("\nTesting without mutex (race condition):")
    
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.IncrementUnsafe()
        }()
    }
    
    wg.Wait()
    fmt.Printf("Final count without mutex: %d\n", counter.ValueUnsafe())
}
```

### Semaphore Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Semaphore struct {
    permits int
    cond    *sync.Cond
}

func NewSemaphore(permits int) *Semaphore {
    return &Semaphore{
        permits: permits,
        cond:    sync.NewCond(&sync.Mutex{}),
    }
}

func (s *Semaphore) Acquire() {
    s.cond.L.Lock()
    defer s.cond.L.Unlock()
    
    for s.permits <= 0 {
        s.cond.Wait()
    }
    s.permits--
}

func (s *Semaphore) Release() {
    s.cond.L.Lock()
    defer s.cond.L.Unlock()
    
    s.permits++
    s.cond.Signal()
}

func (s *Semaphore) TryAcquire() bool {
    s.cond.L.Lock()
    defer s.cond.L.Unlock()
    
    if s.permits > 0 {
        s.permits--
        return true
    }
    return false
}

func (s *Semaphore) GetPermits() int {
    s.cond.L.Lock()
    defer s.cond.L.Unlock()
    
    return s.permits
}

func main() {
    semaphore := NewSemaphore(2) // Allow 2 concurrent processes
    var wg sync.WaitGroup
    
    fmt.Println("Semaphore Demo:")
    
    for i := 0; i < 5; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            
            fmt.Printf("Process %d waiting for semaphore\n", id)
            semaphore.Acquire()
            fmt.Printf("Process %d acquired semaphore (permits: %d)\n", id, semaphore.GetPermits())
            
            // Simulate work
            time.Sleep(2 * time.Second)
            
            fmt.Printf("Process %d releasing semaphore\n", id)
            semaphore.Release()
        }(i)
    }
    
    wg.Wait()
    fmt.Println("All processes completed")
}
```

### Condition Variable Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type BoundedBuffer struct {
    buffer    []int
    capacity  int
    count     int
    mutex     sync.Mutex
    notEmpty  *sync.Cond
    notFull   *sync.Cond
}

func NewBoundedBuffer(capacity int) *BoundedBuffer {
    bb := &BoundedBuffer{
        buffer:   make([]int, 0, capacity),
        capacity: capacity,
        count:    0,
    }
    
    bb.notEmpty = sync.NewCond(&bb.mutex)
    bb.notFull = sync.NewCond(&bb.mutex)
    
    return bb
}

func (bb *BoundedBuffer) Put(item int) {
    bb.mutex.Lock()
    defer bb.mutex.Unlock()
    
    for bb.count >= bb.capacity {
        fmt.Printf("Buffer full, waiting to put %d\n", item)
        bb.notFull.Wait()
    }
    
    bb.buffer = append(bb.buffer, item)
    bb.count++
    
    fmt.Printf("Put %d, buffer size: %d\n", item, bb.count)
    bb.notEmpty.Signal()
}

func (bb *BoundedBuffer) Get() int {
    bb.mutex.Lock()
    defer bb.mutex.Unlock()
    
    for bb.count <= 0 {
        fmt.Println("Buffer empty, waiting to get")
        bb.notEmpty.Wait()
    }
    
    item := bb.buffer[0]
    bb.buffer = bb.buffer[1:]
    bb.count--
    
    fmt.Printf("Got %d, buffer size: %d\n", item, bb.count)
    bb.notFull.Signal()
    
    return item
}

func (bb *BoundedBuffer) GetStatus() {
    bb.mutex.Lock()
    defer bb.mutex.Unlock()
    
    fmt.Printf("Buffer Status: size=%d, capacity=%d\n", bb.count, bb.capacity)
}

func main() {
    buffer := NewBoundedBuffer(3)
    var wg sync.WaitGroup
    
    fmt.Println("Condition Variable Demo:")
    
    // Producer
    wg.Add(1)
    go func() {
        defer wg.Done()
        for i := 0; i < 10; i++ {
            buffer.Put(i)
            time.Sleep(100 * time.Millisecond)
        }
    }()
    
    // Consumer
    wg.Add(1)
    go func() {
        defer wg.Done()
        for i := 0; i < 10; i++ {
            item := buffer.Get()
            time.Sleep(150 * time.Millisecond)
            _ = item // Use the item
        }
    }()
    
    // Monitor
    wg.Add(1)
    go func() {
        defer wg.Done()
        for i := 0; i < 20; i++ {
            buffer.GetStatus()
            time.Sleep(200 * time.Millisecond)
        }
    }()
    
    wg.Wait()
    fmt.Println("All operations completed")
}
```

## Deadlock Prevention

### Theory

Deadlock occurs when two or more processes are blocked forever, waiting for each other to release resources. Deadlock prevention techniques include resource ordering, timeout mechanisms, and deadlock detection.

### Deadlock Detection Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Resource struct {
    ID       int
    Owner    int
    mutex    sync.Mutex
}

type Process struct {
    ID           int
    Resources    map[int]*Resource
    WaitingFor   map[int]*Resource
    mutex        sync.Mutex
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

func (dd *DeadlockDetector) AddProcess(id int) *Process {
    dd.mutex.Lock()
    defer dd.mutex.Unlock()
    
    process := &Process{
        ID:         id,
        Resources:  make(map[int]*Resource),
        WaitingFor: make(map[int]*Resource),
    }
    
    dd.processes[id] = process
    fmt.Printf("Added process %d\n", id)
    return process
}

func (dd *DeadlockDetector) AddResource(id int) *Resource {
    dd.mutex.Lock()
    defer dd.mutex.Unlock()
    
    resource := &Resource{
        ID:    id,
        Owner: -1,
    }
    
    dd.resources[id] = resource
    fmt.Printf("Added resource %d\n", id)
    return resource
}

func (dd *DeadlockDetector) RequestResource(processID, resourceID int) bool {
    dd.mutex.Lock()
    defer dd.mutex.Unlock()
    
    process, exists := dd.processes[processID]
    if !exists {
        fmt.Printf("Process %d not found\n", processID)
        return false
    }
    
    resource, exists := dd.resources[resourceID]
    if !exists {
        fmt.Printf("Resource %d not found\n", resourceID)
        return false
    }
    
    process.mutex.Lock()
    defer process.mutex.Unlock()
    
    if resource.Owner == -1 {
        // Resource is available
        resource.Owner = processID
        process.Resources[resourceID] = resource
        fmt.Printf("Process %d acquired resource %d\n", processID, resourceID)
        return true
    } else if resource.Owner == processID {
        // Process already owns the resource
        fmt.Printf("Process %d already owns resource %d\n", processID, resourceID)
        return true
    } else {
        // Resource is owned by another process
        process.WaitingFor[resourceID] = resource
        fmt.Printf("Process %d waiting for resource %d (owned by %d)\n", 
                   processID, resourceID, resource.Owner)
        return false
    }
}

func (dd *DeadlockDetector) ReleaseResource(processID, resourceID int) bool {
    dd.mutex.Lock()
    defer dd.mutex.Unlock()
    
    process, exists := dd.processes[processID]
    if !exists {
        fmt.Printf("Process %d not found\n", processID)
        return false
    }
    
    resource, exists := dd.resources[resourceID]
    if !exists {
        fmt.Printf("Resource %d not found\n", resourceID)
        return false
    }
    
    process.mutex.Lock()
    defer process.mutex.Unlock()
    
    if resource.Owner != processID {
        fmt.Printf("Process %d does not own resource %d\n", processID, resourceID)
        return false
    }
    
    resource.Owner = -1
    delete(process.Resources, resourceID)
    delete(process.WaitingFor, resourceID)
    
    fmt.Printf("Process %d released resource %d\n", processID, resourceID)
    return true
}

func (dd *DeadlockDetector) DetectDeadlock() []int {
    dd.mutex.RLock()
    defer dd.mutex.RUnlock()
    
    // Build wait-for graph
    waitForGraph := make(map[int][]int)
    
    for _, process := range dd.processes {
        process.mutex.Lock()
        for resourceID, resource := range process.WaitingFor {
            if resource.Owner != -1 {
                waitForGraph[process.ID] = append(waitForGraph[process.ID], resource.Owner)
            }
        }
        process.mutex.Unlock()
    }
    
    // Detect cycles using DFS
    visited := make(map[int]bool)
    recStack := make(map[int]bool)
    deadlockedProcesses := make([]int, 0)
    
    for processID := range dd.processes {
        if !visited[processID] {
            if dd.hasCycle(processID, waitForGraph, visited, recStack) {
                // Find all processes in the cycle
                cycle := dd.findCycle(processID, waitForGraph)
                deadlockedProcesses = append(deadlockedProcesses, cycle...)
            }
        }
    }
    
    return deadlockedProcesses
}

func (dd *DeadlockDetector) hasCycle(processID int, graph map[int][]int, visited, recStack map[int]bool) bool {
    visited[processID] = true
    recStack[processID] = true
    
    for _, neighbor := range graph[processID] {
        if !visited[neighbor] {
            if dd.hasCycle(neighbor, graph, visited, recStack) {
                return true
            }
        } else if recStack[neighbor] {
            return true
        }
    }
    
    recStack[processID] = false
    return false
}

func (dd *DeadlockDetector) findCycle(processID int, graph map[int][]int) []int {
    visited := make(map[int]bool)
    path := make([]int, 0)
    
    return dd.dfsCycle(processID, graph, visited, path)
}

func (dd *DeadlockDetector) dfsCycle(processID int, graph map[int][]int, visited map[int]bool, path []int) []int {
    if visited[processID] {
        // Found a cycle
        for i, p := range path {
            if p == processID {
                return path[i:]
            }
        }
        return []int{}
    }
    
    visited[processID] = true
    path = append(path, processID)
    
    for _, neighbor := range graph[processID] {
        cycle := dd.dfsCycle(neighbor, graph, visited, path)
        if len(cycle) > 0 {
            return cycle
        }
    }
    
    return []int{}
}

func (dd *DeadlockDetector) PrintStatus() {
    dd.mutex.RLock()
    defer dd.mutex.RUnlock()
    
    fmt.Println("Deadlock Detector Status:")
    fmt.Println("Processes:")
    for _, process := range dd.processes {
        process.mutex.Lock()
        fmt.Printf("  Process %d:\n", process.ID)
        fmt.Printf("    Owns: ")
        for resourceID := range process.Resources {
            fmt.Printf("%d ", resourceID)
        }
        fmt.Println()
        fmt.Printf("    Waiting for: ")
        for resourceID := range process.WaitingFor {
            fmt.Printf("%d ", resourceID)
        }
        fmt.Println()
        process.mutex.Unlock()
    }
    
    fmt.Println("Resources:")
    for _, resource := range dd.resources {
        fmt.Printf("  Resource %d: owner=%d\n", resource.ID, resource.Owner)
    }
}

func main() {
    dd := NewDeadlockDetector()
    
    fmt.Println("Deadlock Detection Demo:")
    
    // Add processes and resources
    dd.AddProcess(1)
    dd.AddProcess(2)
    dd.AddProcess(3)
    
    dd.AddResource(1)
    dd.AddResource(2)
    dd.AddResource(3)
    
    // Create a deadlock scenario
    dd.RequestResource(1, 1) // Process 1 gets resource 1
    dd.RequestResource(2, 2) // Process 2 gets resource 2
    dd.RequestResource(3, 3) // Process 3 gets resource 3
    
    dd.RequestResource(1, 2) // Process 1 waits for resource 2
    dd.RequestResource(2, 3) // Process 2 waits for resource 3
    dd.RequestResource(3, 1) // Process 3 waits for resource 1
    
    dd.PrintStatus()
    
    // Detect deadlock
    deadlocked := dd.DetectDeadlock()
    if len(deadlocked) > 0 {
        fmt.Printf("Deadlock detected! Processes involved: %v\n", deadlocked)
    } else {
        fmt.Println("No deadlock detected")
    }
    
    // Break the deadlock
    fmt.Println("Breaking deadlock by releasing resource 1...")
    dd.ReleaseResource(1, 1)
    
    dd.PrintStatus()
    
    // Check again
    deadlocked = dd.DetectDeadlock()
    if len(deadlocked) > 0 {
        fmt.Printf("Deadlock detected! Processes involved: %v\n", deadlocked)
    } else {
        fmt.Println("No deadlock detected")
    }
}
```

## Producer-Consumer Problem

### Theory

The producer-consumer problem is a classic synchronization problem where producers produce items and consumers consume items from a shared buffer. The challenge is to ensure that producers don't produce when the buffer is full and consumers don't consume when the buffer is empty.

### Producer-Consumer Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type ProducerConsumer struct {
    buffer    []int
    capacity  int
    count     int
    mutex     sync.Mutex
    notEmpty  *sync.Cond
    notFull   *sync.Cond
    producers int
    consumers int
}

func NewProducerConsumer(capacity, producers, consumers int) *ProducerConsumer {
    pc := &ProducerConsumer{
        buffer:    make([]int, 0, capacity),
        capacity:  capacity,
        count:     0,
        producers: producers,
        consumers: consumers,
    }
    
    pc.notEmpty = sync.NewCond(&pc.mutex)
    pc.notFull = sync.NewCond(&pc.mutex)
    
    return pc
}

func (pc *ProducerConsumer) Produce(producerID int, item int) {
    pc.mutex.Lock()
    defer pc.mutex.Unlock()
    
    for pc.count >= pc.capacity {
        fmt.Printf("Producer %d waiting (buffer full)\n", producerID)
        pc.notFull.Wait()
    }
    
    pc.buffer = append(pc.buffer, item)
    pc.count++
    
    fmt.Printf("Producer %d produced item %d (buffer size: %d)\n", 
               producerID, item, pc.count)
    pc.notEmpty.Signal()
}

func (pc *ProducerConsumer) Consume(consumerID int) int {
    pc.mutex.Lock()
    defer pc.mutex.Unlock()
    
    for pc.count <= 0 {
        fmt.Printf("Consumer %d waiting (buffer empty)\n", consumerID)
        pc.notEmpty.Wait()
    }
    
    item := pc.buffer[0]
    pc.buffer = pc.buffer[1:]
    pc.count--
    
    fmt.Printf("Consumer %d consumed item %d (buffer size: %d)\n", 
               consumerID, item, pc.count)
    pc.notFull.Signal()
    
    return item
}

func (pc *ProducerConsumer) GetStatus() {
    pc.mutex.Lock()
    defer pc.mutex.Unlock()
    
    fmt.Printf("Buffer Status: size=%d, capacity=%d\n", pc.count, pc.capacity)
}

func (pc *ProducerConsumer) Run() {
    var wg sync.WaitGroup
    
    // Start producers
    for i := 0; i < pc.producers; i++ {
        wg.Add(1)
        go func(producerID int) {
            defer wg.Done()
            for j := 0; j < 5; j++ {
                pc.Produce(producerID, producerID*10+j)
                time.Sleep(100 * time.Millisecond)
            }
        }(i)
    }
    
    // Start consumers
    for i := 0; i < pc.consumers; i++ {
        wg.Add(1)
        go func(consumerID int) {
            defer wg.Done()
            for j := 0; j < 5; j++ {
                item := pc.Consume(consumerID)
                time.Sleep(150 * time.Millisecond)
                _ = item // Use the item
            }
        }(i)
    }
    
    // Monitor
    wg.Add(1)
    go func() {
        defer wg.Done()
        for i := 0; i < 20; i++ {
            pc.GetStatus()
            time.Sleep(200 * time.Millisecond)
        }
    }()
    
    wg.Wait()
    fmt.Println("All operations completed")
}

func main() {
    pc := NewProducerConsumer(3, 2, 2)
    
    fmt.Println("Producer-Consumer Demo:")
    pc.Run()
}
```

## Reader-Writer Problem

### Theory

The reader-writer problem is a synchronization problem where multiple readers can access a shared resource simultaneously, but writers need exclusive access. The challenge is to ensure that readers don't starve writers and vice versa.

### Reader-Writer Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type ReaderWriter struct {
    data        int
    readers     int
    writers     int
    mutex       sync.Mutex
    readCond    *sync.Cond
    writeCond   *sync.Cond
    readCount   int
    writeCount  int
}

func NewReaderWriter() *ReaderWriter {
    rw := &ReaderWriter{
        data:    0,
        readers: 0,
        writers: 0,
    }
    
    rw.readCond = sync.NewCond(&rw.mutex)
    rw.writeCond = sync.NewCond(&rw.mutex)
    
    return rw
}

func (rw *ReaderWriter) StartRead(readerID int) {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    for rw.writers > 0 {
        fmt.Printf("Reader %d waiting (writers active)\n", readerID)
        rw.readCond.Wait()
    }
    
    rw.readers++
    rw.readCount++
    fmt.Printf("Reader %d started reading (readers: %d)\n", readerID, rw.readers)
}

func (rw *ReaderWriter) EndRead(readerID int) {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    rw.readers--
    fmt.Printf("Reader %d finished reading (readers: %d)\n", readerID, rw.readers)
    
    if rw.readers == 0 {
        rw.writeCond.Signal()
    }
}

func (rw *ReaderWriter) StartWrite(writerID int) {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    for rw.readers > 0 || rw.writers > 0 {
        fmt.Printf("Writer %d waiting (readers: %d, writers: %d)\n", 
                   writerID, rw.readers, rw.writers)
        rw.writeCond.Wait()
    }
    
    rw.writers++
    rw.writeCount++
    fmt.Printf("Writer %d started writing (writers: %d)\n", writerID, rw.writers)
}

func (rw *ReaderWriter) EndWrite(writerID int) {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    rw.writers--
    fmt.Printf("Writer %d finished writing (writers: %d)\n", writerID, rw.writers)
    
    rw.writeCond.Signal()
    rw.readCond.Broadcast()
}

func (rw *ReaderWriter) Read(readerID int) int {
    rw.StartRead(readerID)
    
    // Simulate reading
    time.Sleep(100 * time.Millisecond)
    value := rw.data
    
    rw.EndRead(readerID)
    return value
}

func (rw *ReaderWriter) Write(writerID int, value int) {
    rw.StartWrite(writerID)
    
    // Simulate writing
    time.Sleep(200 * time.Millisecond)
    rw.data = value
    
    rw.EndWrite(writerID)
}

func (rw *ReaderWriter) GetStatus() {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    fmt.Printf("Reader-Writer Status: data=%d, readers=%d, writers=%d, readCount=%d, writeCount=%d\n", 
               rw.data, rw.readers, rw.writers, rw.readCount, rw.writeCount)
}

func (rw *ReaderWriter) Run() {
    var wg sync.WaitGroup
    
    // Start readers
    for i := 0; i < 3; i++ {
        wg.Add(1)
        go func(readerID int) {
            defer wg.Done()
            for j := 0; j < 3; j++ {
                value := rw.Read(readerID)
                fmt.Printf("Reader %d read value: %d\n", readerID, value)
                time.Sleep(100 * time.Millisecond)
            }
        }(i)
    }
    
    // Start writers
    for i := 0; i < 2; i++ {
        wg.Add(1)
        go func(writerID int) {
            defer wg.Done()
            for j := 0; j < 2; j++ {
                value := writerID*10 + j
                rw.Write(writerID, value)
                fmt.Printf("Writer %d wrote value: %d\n", writerID, value)
                time.Sleep(200 * time.Millisecond)
            }
        }(i)
    }
    
    // Monitor
    wg.Add(1)
    go func() {
        defer wg.Done()
        for i := 0; i < 10; i++ {
            rw.GetStatus()
            time.Sleep(300 * time.Millisecond)
        }
    }()
    
    wg.Wait()
    fmt.Println("All operations completed")
}

func main() {
    rw := NewReaderWriter()
    
    fmt.Println("Reader-Writer Demo:")
    rw.Run()
}
```

## Follow-up Questions

### 1. Thread Management
**Q: What are the advantages and disadvantages of using thread pools?**
A: Advantages: Reuse of threads reduces creation/destruction overhead, better resource management, controlled concurrency. Disadvantages: Fixed pool size may not be optimal for all workloads, potential for thread starvation if pool is too small.

### 2. Synchronization
**Q: When would you choose a mutex over a semaphore?**
A: Use a mutex for mutual exclusion (only one thread can access a resource at a time). Use a semaphore for resource counting (multiple threads can access a resource up to a limit) or for signaling between threads.

### 3. Deadlock Prevention
**Q: What are the four conditions necessary for deadlock to occur?**
A: Mutual exclusion, hold and wait, no preemption, and circular wait. Deadlock can be prevented by eliminating any one of these conditions.

## Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Thread Creation | O(1) | O(1) | Constant time |
| Mutex Operations | O(1) | O(1) | Lock/unlock operations |
| Semaphore Operations | O(1) | O(1) | Acquire/release operations |
| Deadlock Detection | O(V + E) | O(V) | Graph traversal |

## Applications

1. **Thread Management**: Web servers, database systems, game engines
2. **Synchronization**: Multi-threaded applications, operating systems
3. **Deadlock Prevention**: Database systems, operating systems
4. **Producer-Consumer**: Message queues, streaming systems
5. **Reader-Writer**: Database systems, file systems, caches

---

**Next**: [System Calls](system-calls.md) | **Previous**: [OS Deep Dive](README.md) | **Up**: [OS Deep Dive](README.md)
