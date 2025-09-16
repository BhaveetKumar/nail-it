# Process Management

## Overview

This module covers advanced process management concepts including scheduling algorithms, synchronization primitives, inter-process communication, and process lifecycle management. These concepts are essential for understanding how operating systems manage multiple processes efficiently.

## Table of Contents

1. [Process Scheduling](#process-scheduling/)
2. [Synchronization Primitives](#synchronization-primitives/)
3. [Inter-Process Communication](#inter-process-communication/)
4. [Process Lifecycle](#process-lifecycle/)
5. [Memory Management](#memory-management/)
6. [Applications](#applications/)
7. [Complexity Analysis](#complexity-analysis/)
8. [Follow-up Questions](#follow-up-questions/)

## Process Scheduling

### Theory

Process scheduling is the mechanism by which the operating system decides which process should run on the CPU at any given time. Different scheduling algorithms optimize for different criteria like throughput, response time, and fairness.

### Scheduling Algorithms

#### First-Come, First-Served (FCFS)

##### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
    "time"
)

type Process struct {
    ID           int
    ArrivalTime  int
    BurstTime    int
    StartTime    int
    FinishTime   int
    WaitingTime  int
    TurnaroundTime int
}

type FCFSScheduler struct {
    processes []Process
}

func NewFCFSScheduler() *FCFSScheduler {
    return &FCFSScheduler{
        processes: make([]Process, 0),
    }
}

func (s *FCFSScheduler) AddProcess(id, arrivalTime, burstTime int) {
    process := Process{
        ID:          id,
        ArrivalTime: arrivalTime,
        BurstTime:   burstTime,
    }
    s.processes = append(s.processes, process)
}

func (s *FCFSScheduler) Schedule() {
    // Sort processes by arrival time
    sort.Slice(s.processes, func(i, j int) bool {
        return s.processes[i].ArrivalTime < s.processes[j].ArrivalTime
    })
    
    currentTime := 0
    
    for i := range s.processes {
        // Wait for process to arrive
        if currentTime < s.processes[i].ArrivalTime {
            currentTime = s.processes[i].ArrivalTime
        }
        
        // Set start time
        s.processes[i].StartTime = currentTime
        
        // Execute process
        fmt.Printf("Time %d: Executing Process %d (Burst: %d)\n", 
                   currentTime, s.processes[i].ID, s.processes[i].BurstTime)
        
        // Simulate execution time
        time.Sleep(time.Duration(s.processes[i].BurstTime) * time.Millisecond)
        
        // Update times
        currentTime += s.processes[i].BurstTime
        s.processes[i].FinishTime = currentTime
        s.processes[i].TurnaroundTime = s.processes[i].FinishTime - s.processes[i].ArrivalTime
        s.processes[i].WaitingTime = s.processes[i].TurnaroundTime - s.processes[i].BurstTime
        
        fmt.Printf("Time %d: Process %d completed\n", currentTime, s.processes[i].ID)
    }
}

func (s *FCFSScheduler) CalculateMetrics() {
    var totalWaitingTime, totalTurnaroundTime float64
    
    fmt.Println("\nProcess Details:")
    fmt.Println("ID\tArrival\tBurst\tStart\tFinish\tWaiting\tTurnaround")
    
    for _, p := range s.processes {
        fmt.Printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
                   p.ID, p.ArrivalTime, p.BurstTime, p.StartTime,
                   p.FinishTime, p.WaitingTime, p.TurnaroundTime)
        
        totalWaitingTime += float64(p.WaitingTime)
        totalTurnaroundTime += float64(p.TurnaroundTime)
    }
    
    avgWaitingTime := totalWaitingTime / float64(len(s.processes))
    avgTurnaroundTime := totalTurnaroundTime / float64(len(s.processes))
    
    fmt.Printf("\nAverage Waiting Time: %.2f\n", avgWaitingTime)
    fmt.Printf("Average Turnaround Time: %.2f\n", avgTurnaroundTime)
}

func main() {
    scheduler := NewFCFSScheduler()
    
    // Add processes
    scheduler.AddProcess(1, 0, 5)
    scheduler.AddProcess(2, 1, 3)
    scheduler.AddProcess(3, 2, 8)
    scheduler.AddProcess(4, 3, 2)
    
    fmt.Println("FCFS Scheduling:")
    scheduler.Schedule()
    scheduler.CalculateMetrics()
}
```

#### Shortest Job First (SJF)

##### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
    "time"
)

type SJFScheduler struct {
    processes []Process
}

func NewSJFScheduler() *SJFScheduler {
    return &SJFScheduler{
        processes: make([]Process, 0),
    }
}

func (s *SJFScheduler) AddProcess(id, arrivalTime, burstTime int) {
    process := Process{
        ID:          id,
        ArrivalTime: arrivalTime,
        BurstTime:   burstTime,
    }
    s.processes = append(s.processes, process)
}

func (s *SJFScheduler) Schedule() {
    currentTime := 0
    completed := 0
    n := len(s.processes)
    
    for completed < n {
        // Find processes that have arrived and are ready
        var readyProcesses []Process
        for i := range s.processes {
            if s.processes[i].ArrivalTime <= currentTime && s.processes[i].StartTime == 0 {
                readyProcesses = append(readyProcesses, s.processes[i])
            }
        }
        
        if len(readyProcesses) == 0 {
            currentTime++
            continue
        }
        
        // Sort by burst time (SJF)
        sort.Slice(readyProcesses, func(i, j int) bool {
            return readyProcesses[i].BurstTime < readyProcesses[j].BurstTime
        })
        
        // Execute the shortest job
        selectedProcess := readyProcesses[0]
        
        // Find the process in the main slice and update it
        for i := range s.processes {
            if s.processes[i].ID == selectedProcess.ID {
                s.processes[i].StartTime = currentTime
                
                fmt.Printf("Time %d: Executing Process %d (Burst: %d)\n", 
                           currentTime, s.processes[i].ID, s.processes[i].BurstTime)
                
                // Simulate execution
                time.Sleep(time.Duration(s.processes[i].BurstTime) * time.Millisecond)
                
                currentTime += s.processes[i].BurstTime
                s.processes[i].FinishTime = currentTime
                s.processes[i].TurnaroundTime = s.processes[i].FinishTime - s.processes[i].ArrivalTime
                s.processes[i].WaitingTime = s.processes[i].TurnaroundTime - s.processes[i].BurstTime
                
                fmt.Printf("Time %d: Process %d completed\n", currentTime, s.processes[i].ID)
                completed++
                break
            }
        }
    }
}

func (s *SJFScheduler) CalculateMetrics() {
    var totalWaitingTime, totalTurnaroundTime float64
    
    fmt.Println("\nProcess Details:")
    fmt.Println("ID\tArrival\tBurst\tStart\tFinish\tWaiting\tTurnaround")
    
    for _, p := range s.processes {
        fmt.Printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
                   p.ID, p.ArrivalTime, p.BurstTime, p.StartTime,
                   p.FinishTime, p.WaitingTime, p.TurnaroundTime)
        
        totalWaitingTime += float64(p.WaitingTime)
        totalTurnaroundTime += float64(p.TurnaroundTime)
    }
    
    avgWaitingTime := totalWaitingTime / float64(len(s.processes))
    avgTurnaroundTime := totalTurnaroundTime / float64(len(s.processes))
    
    fmt.Printf("\nAverage Waiting Time: %.2f\n", avgWaitingTime)
    fmt.Printf("Average Turnaround Time: %.2f\n", avgTurnaroundTime)
}

func main() {
    scheduler := NewSJFScheduler()
    
    // Add processes
    scheduler.AddProcess(1, 0, 5)
    scheduler.AddProcess(2, 1, 3)
    scheduler.AddProcess(3, 2, 8)
    scheduler.AddProcess(4, 3, 2)
    
    fmt.Println("SJF Scheduling:")
    scheduler.Schedule()
    scheduler.CalculateMetrics()
}
```

#### Round Robin (RR)

##### Golang Implementation

```go
package main

import (
    "fmt"
    "time"
)

type RoundRobinScheduler struct {
    processes []Process
    timeQuantum int
}

func NewRoundRobinScheduler(timeQuantum int) *RoundRobinScheduler {
    return &RoundRobinScheduler{
        processes:   make([]Process, 0),
        timeQuantum: timeQuantum,
    }
}

func (s *RoundRobinScheduler) AddProcess(id, arrivalTime, burstTime int) {
    process := Process{
        ID:          id,
        ArrivalTime: arrivalTime,
        BurstTime:   burstTime,
    }
    s.processes = append(s.processes, process)
}

func (s *RoundRobinScheduler) Schedule() {
    // Create a copy of processes for scheduling
    remainingTime := make([]int, len(s.processes))
    for i := range s.processes {
        remainingTime[i] = s.processes[i].BurstTime
    }
    
    currentTime := 0
    completed := 0
    n := len(s.processes)
    
    // Sort processes by arrival time
    for i := 0; i < n-1; i++ {
        for j := 0; j < n-i-1; j++ {
            if s.processes[j].ArrivalTime > s.processes[j+1].ArrivalTime {
                s.processes[j], s.processes[j+1] = s.processes[j+1], s.processes[j]
                remainingTime[j], remainingTime[j+1] = remainingTime[j+1], remainingTime[j]
            }
        }
    }
    
    for completed < n {
        for i := range s.processes {
            if remainingTime[i] > 0 && s.processes[i].ArrivalTime <= currentTime {
                // Set start time if not set
                if s.processes[i].StartTime == 0 {
                    s.processes[i].StartTime = currentTime
                }
                
                // Execute for time quantum or remaining time
                executionTime := s.timeQuantum
                if remainingTime[i] < s.timeQuantum {
                    executionTime = remainingTime[i]
                }
                
                fmt.Printf("Time %d: Executing Process %d for %d units\n", 
                           currentTime, s.processes[i].ID, executionTime)
                
                // Simulate execution
                time.Sleep(time.Duration(executionTime) * time.Millisecond)
                
                currentTime += executionTime
                remainingTime[i] -= executionTime
                
                if remainingTime[i] == 0 {
                    s.processes[i].FinishTime = currentTime
                    s.processes[i].TurnaroundTime = s.processes[i].FinishTime - s.processes[i].ArrivalTime
                    s.processes[i].WaitingTime = s.processes[i].TurnaroundTime - s.processes[i].BurstTime
                    completed++
                    fmt.Printf("Time %d: Process %d completed\n", currentTime, s.processes[i].ID)
                }
            }
        }
    }
}

func (s *RoundRobinScheduler) CalculateMetrics() {
    var totalWaitingTime, totalTurnaroundTime float64
    
    fmt.Println("\nProcess Details:")
    fmt.Println("ID\tArrival\tBurst\tStart\tFinish\tWaiting\tTurnaround")
    
    for _, p := range s.processes {
        fmt.Printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
                   p.ID, p.ArrivalTime, p.BurstTime, p.StartTime,
                   p.FinishTime, p.WaitingTime, p.TurnaroundTime)
        
        totalWaitingTime += float64(p.WaitingTime)
        totalTurnaroundTime += float64(p.TurnaroundTime)
    }
    
    avgWaitingTime := totalWaitingTime / float64(len(s.processes))
    avgTurnaroundTime := totalTurnaroundTime / float64(len(s.processes))
    
    fmt.Printf("\nAverage Waiting Time: %.2f\n", avgWaitingTime)
    fmt.Printf("Average Turnaround Time: %.2f\n", avgTurnaroundTime)
}

func main() {
    scheduler := NewRoundRobinScheduler(2) // Time quantum = 2
    
    // Add processes
    scheduler.AddProcess(1, 0, 5)
    scheduler.AddProcess(2, 1, 3)
    scheduler.AddProcess(3, 2, 8)
    scheduler.AddProcess(4, 3, 2)
    
    fmt.Println("Round Robin Scheduling (Time Quantum = 2):")
    scheduler.Schedule()
    scheduler.CalculateMetrics()
}
```

## Synchronization Primitives

### Theory

Synchronization primitives are mechanisms that ensure proper coordination between processes or threads to prevent race conditions and ensure data consistency.

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

func (c *SafeCounter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.count
}

func (c *SafeCounter) IncrementUnsafe() {
    c.count++
}

func main() {
    counter := &SafeCounter{}
    
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
    fmt.Printf("Final count without mutex: %d\n", counter.Value())
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

func main() {
    semaphore := NewSemaphore(2) // Allow 2 concurrent processes
    var wg sync.WaitGroup
    
    for i := 0; i < 5; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            
            fmt.Printf("Process %d waiting for semaphore\n", id)
            semaphore.Acquire()
            fmt.Printf("Process %d acquired semaphore\n", id)
            
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

## Inter-Process Communication

### Theory

Inter-Process Communication (IPC) allows processes to communicate and synchronize with each other. Common IPC mechanisms include pipes, message queues, shared memory, and sockets.

### Message Queue Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Message struct {
    ID      int
    Content string
    Sender  int
}

type MessageQueue struct {
    messages []Message
    mutex    sync.Mutex
    cond     *sync.Cond
    maxSize  int
}

func NewMessageQueue(maxSize int) *MessageQueue {
    mq := &MessageQueue{
        messages: make([]Message, 0),
        maxSize:  maxSize,
    }
    mq.cond = sync.NewCond(&mq.mutex)
    return mq
}

func (mq *MessageQueue) Send(message Message) {
    mq.mutex.Lock()
    defer mq.mutex.Unlock()
    
    for len(mq.messages) >= mq.maxSize {
        mq.cond.Wait()
    }
    
    mq.messages = append(mq.messages, message)
    fmt.Printf("Process %d sent message: %s\n", message.Sender, message.Content)
    mq.cond.Signal()
}

func (mq *MessageQueue) Receive(receiverID int) Message {
    mq.mutex.Lock()
    defer mq.mutex.Unlock()
    
    for len(mq.messages) == 0 {
        mq.cond.Wait()
    }
    
    message := mq.messages[0]
    mq.messages = mq.messages[1:]
    
    fmt.Printf("Process %d received message: %s (from process %d)\n", 
               receiverID, message.Content, message.Sender)
    
    mq.cond.Signal()
    return message
}

func main() {
    mq := NewMessageQueue(3)
    var wg sync.WaitGroup
    
    // Sender processes
    for i := 0; i < 2; i++ {
        wg.Add(1)
        go func(senderID int) {
            defer wg.Done()
            for j := 0; j < 3; j++ {
                message := Message{
                    ID:      j,
                    Content: fmt.Sprintf("Hello from sender %d, message %d", senderID, j),
                    Sender:  senderID,
                }
                mq.Send(message)
                time.Sleep(1 * time.Second)
            }
        }(i)
    }
    
    // Receiver processes
    for i := 0; i < 2; i++ {
        wg.Add(1)
        go func(receiverID int) {
            defer wg.Done()
            for j := 0; j < 3; j++ {
                mq.Receive(receiverID)
                time.Sleep(500 * time.Millisecond)
            }
        }(i)
    }
    
    wg.Wait()
    fmt.Println("All messages sent and received")
}
```

### Shared Memory Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type SharedMemory struct {
    data  map[string]int
    mutex sync.RWMutex
}

func NewSharedMemory() *SharedMemory {
    return &SharedMemory{
        data: make(map[string]int),
    }
}

func (sm *SharedMemory) Write(key string, value int) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    sm.data[key] = value
    fmt.Printf("Written: %s = %d\n", key, value)
}

func (sm *SharedMemory) Read(key string) (int, bool) {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    value, exists := sm.data[key]
    if exists {
        fmt.Printf("Read: %s = %d\n", key, value)
    } else {
        fmt.Printf("Key %s not found\n", key)
    }
    return value, exists
}

func (sm *SharedMemory) ReadAll() map[string]int {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    result := make(map[string]int)
    for k, v := range sm.data {
        result[k] = v
    }
    return result
}

func main() {
    sharedMem := NewSharedMemory()
    var wg sync.WaitGroup
    
    // Writer processes
    for i := 0; i < 3; i++ {
        wg.Add(1)
        go func(writerID int) {
            defer wg.Done()
            for j := 0; j < 3; j++ {
                key := fmt.Sprintf("key_%d_%d", writerID, j)
                value := writerID*10 + j
                sharedMem.Write(key, value)
                time.Sleep(100 * time.Millisecond)
            }
        }(i)
    }
    
    // Reader processes
    for i := 0; i < 2; i++ {
        wg.Add(1)
        go func(readerID int) {
            defer wg.Done()
            for j := 0; j < 5; j++ {
                key := fmt.Sprintf("key_%d_%d", j%3, j%3)
                sharedMem.Read(key)
                time.Sleep(200 * time.Millisecond)
            }
        }(i)
    }
    
    wg.Wait()
    
    fmt.Println("\nFinal shared memory state:")
    for key, value := range sharedMem.ReadAll() {
        fmt.Printf("%s = %d\n", key, value)
    }
}
```

## Follow-up Questions

### 1. Scheduling Algorithms
**Q: When would you choose Round Robin over Shortest Job First?**
A: Choose Round Robin when you need fair CPU allocation and good response time for interactive processes. Choose SJF when you want to minimize average waiting time and can predict process burst times accurately.

### 2. Synchronization
**Q: What's the difference between a mutex and a semaphore?**
A: A mutex is a binary semaphore (0 or 1) used for mutual exclusion. A semaphore can have multiple permits and is used for resource counting and signaling between processes.

### 3. IPC Mechanisms
**Q: When would you use shared memory vs message passing?**
A: Use shared memory for high-performance communication between processes on the same machine. Use message passing for communication across network boundaries or when you need better isolation between processes.

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| FCFS | O(n) | O(n) | Simple but can have high waiting times |
| SJF | O(n²) | O(n) | Optimal for minimizing waiting time |
| Round Robin | O(n) | O(n) | Good for interactive systems |
| Mutex | O(1) | O(1) | Constant time operations |
| Semaphore | O(1) | O(1) | Constant time operations |

## Applications

1. **Process Scheduling**: Operating systems, real-time systems
2. **Synchronization**: Multi-threaded applications, database systems
3. **IPC**: Distributed systems, microservices architecture

---

**Next**: [Memory Management](memory-management.md/) | **Previous**: [OS Deep Dive](README.md/) | **Up**: [OS Deep Dive](README.md/)
