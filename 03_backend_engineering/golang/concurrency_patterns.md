---
# Auto-generated front matter
Title: Concurrency Patterns
LastUpdated: 2025-11-06T20:45:58.292401
Tags: []
Status: draft
---

# Go Concurrency Patterns - Advanced Patterns and Best Practices

## Overview

Go's concurrency model is built around goroutines and channels, providing powerful primitives for building concurrent applications. This guide covers advanced concurrency patterns, best practices, and common pitfalls.

## Key Concepts

- **Goroutines**: Lightweight threads managed by the Go runtime
- **Channels**: Communication mechanism between goroutines
- **Select Statement**: Multiplexing channel operations
- **Context**: Cancellation and timeout management
- **Sync Package**: Synchronization primitives

## Concurrency Patterns

### 1. Worker Pool Pattern

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

// Job represents a unit of work
type Job struct {
    ID       int
    Data     interface{}
    Result   chan interface{}
    Error    chan error
}

// Worker represents a worker in the pool
type Worker struct {
    ID       int
    JobQueue chan Job
    Quit     chan bool
    WG       *sync.WaitGroup
}

// WorkerPool manages a pool of workers
type WorkerPool struct {
    Workers    []*Worker
    JobQueue   chan Job
    WorkerCount int
    Quit       chan bool
    WG         *sync.WaitGroup
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(workerCount, jobQueueSize int) *WorkerPool {
    return &WorkerPool{
        Workers:     make([]*Worker, workerCount),
        JobQueue:    make(chan Job, jobQueueSize),
        WorkerCount: workerCount,
        Quit:        make(chan bool),
        WG:          &sync.WaitGroup{},
    }
}

// Start starts all workers in the pool
func (wp *WorkerPool) Start() {
    for i := 0; i < wp.WorkerCount; i++ {
        worker := &Worker{
            ID:       i,
            JobQueue: wp.JobQueue,
            Quit:     make(chan bool),
            WG:       wp.WG,
        }
        wp.Workers[i] = worker
        wp.WG.Add(1)
        go worker.Start()
    }
}

// Stop stops all workers in the pool
func (wp *WorkerPool) Stop() {
    close(wp.Quit)
    for _, worker := range wp.Workers {
        worker.Stop()
    }
    wp.WG.Wait()
}

// SubmitJob submits a job to the pool
func (wp *WorkerPool) SubmitJob(job Job) {
    wp.JobQueue <- job
}

// Start starts a worker
func (w *Worker) Start() {
    defer w.WG.Done()
    
    for {
        select {
        case job := <-w.JobQueue:
            log.Printf("Worker %d processing job %d", w.ID, job.ID)
            
            // Simulate work
            time.Sleep(100 * time.Millisecond)
            
            // Process the job
            result, err := w.processJob(job)
            if err != nil {
                job.Error <- err
            } else {
                job.Result <- result
            }
            
        case <-w.Quit:
            log.Printf("Worker %d stopping", w.ID)
            return
        }
    }
}

// Stop stops a worker
func (w *Worker) Stop() {
    w.Quit <- true
}

// processJob processes a single job
func (w *Worker) processJob(job Job) (interface{}, error) {
    // Simulate some processing
    time.Sleep(50 * time.Millisecond)
    
    // Return a result
    return fmt.Sprintf("Processed job %d by worker %d", job.ID, w.ID), nil
}

// Example usage
func main() {
    // Create worker pool
    pool := NewWorkerPool(3, 10)
    pool.Start()
    
    // Submit jobs
    for i := 0; i < 10; i++ {
        job := Job{
            ID:     i,
            Data:   fmt.Sprintf("Data for job %d", i),
            Result: make(chan interface{}, 1),
            Error:  make(chan error, 1),
        }
        
        pool.SubmitJob(job)
        
        // Handle result
        go func(j Job) {
            select {
            case result := <-j.Result:
                log.Printf("Job %d completed: %v", j.ID, result)
            case err := <-j.Error:
                log.Printf("Job %d failed: %v", j.ID, err)
            }
        }(job)
    }
    
    // Wait for all jobs to complete
    time.Sleep(2 * time.Second)
    
    // Stop the pool
    pool.Stop()
}
```

### 2. Pipeline Pattern

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

// Stage represents a stage in the pipeline
type Stage struct {
    Name     string
    Process  func(ctx context.Context, input <-chan interface{}) <-chan interface{}
    Workers  int
}

// Pipeline represents a processing pipeline
type Pipeline struct {
    Stages []Stage
    Input  chan interface{}
    Output chan interface{}
    Quit   chan bool
    WG     *sync.WaitGroup
}

// NewPipeline creates a new pipeline
func NewPipeline(stages []Stage) *Pipeline {
    return &Pipeline{
        Stages: stages,
        Input:  make(chan interface{}, 100),
        Output: make(chan interface{}, 100),
        Quit:   make(chan bool),
        WG:     &sync.WaitGroup{},
    }
}

// Start starts the pipeline
func (p *Pipeline) Start(ctx context.Context) {
    p.WG.Add(len(p.Stages))
    
    // Connect stages
    currentInput := p.Input
    for i, stage := range p.Stages {
        output := make(chan interface{}, 100)
        
        // Start stage workers
        for j := 0; j < stage.Workers; j++ {
            go func(stage Stage, input <-chan interface{}, output chan<- interface{}) {
                defer p.WG.Done()
                
                for {
                    select {
                    case data := <-input:
                        if data == nil {
                            close(output)
                            return
                        }
                        
                        // Process data
                        result := stage.Process(ctx, input)
                        for r := range result {
                            output <- r
                        }
                        
                    case <-p.Quit:
                        close(output)
                        return
                    }
                }
            }(stage, currentInput, output)
        }
        
        currentInput = output
        if i == len(p.Stages)-1 {
            p.Output = output
        }
    }
}

// Stop stops the pipeline
func (p *Pipeline) Stop() {
    close(p.Quit)
    p.WG.Wait()
}

// Example stages
func createStages() []Stage {
    return []Stage{
        {
            Name:    "Filter",
            Workers: 2,
            Process: func(ctx context.Context, input <-chan interface{}) <-chan interface{} {
                output := make(chan interface{})
                go func() {
                    defer close(output)
                    for data := range input {
                        if data != nil {
                            output <- data
                        }
                    }
                }()
                return output
            },
        },
        {
            Name:    "Transform",
            Workers: 3,
            Process: func(ctx context.Context, input <-chan interface{}) <-chan interface{} {
                output := make(chan interface{})
                go func() {
                    defer close(output)
                    for data := range input {
                        // Transform data
                        result := fmt.Sprintf("Transformed: %v", data)
                        output <- result
                    }
                }()
                return output
            },
        },
        {
            Name:    "Aggregate",
            Workers: 1,
            Process: func(ctx context.Context, input <-chan interface{}) <-chan interface{} {
                output := make(chan interface{})
                go func() {
                    defer close(output)
                    count := 0
                    for data := range input {
                        count++
                        if count%10 == 0 {
                            output <- fmt.Sprintf("Processed %d items", count)
                        }
                    }
                }()
                return output
            },
        },
    }
}

// Example usage
func main() {
    // Create pipeline
    stages := createStages()
    pipeline := NewPipeline(stages)
    
    // Start pipeline
    ctx := context.Background()
    pipeline.Start(ctx)
    
    // Send data
    go func() {
        for i := 0; i < 100; i++ {
            pipeline.Input <- fmt.Sprintf("Data %d", i)
        }
        close(pipeline.Input)
    }()
    
    // Collect results
    go func() {
        for result := range pipeline.Output {
            log.Printf("Result: %v", result)
        }
    }()
    
    // Wait for completion
    time.Sleep(5 * time.Second)
    pipeline.Stop()
}
```

### 3. Fan-In/Fan-Out Pattern

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

// FanOut distributes work across multiple workers
func FanOut(ctx context.Context, input <-chan interface{}, workerCount int, worker func(context.Context, <-chan interface{}) <-chan interface{}) []<-chan interface{} {
    outputs := make([]<-chan interface{}, workerCount)
    
    for i := 0; i < workerCount; i++ {
        outputs[i] = worker(ctx, input)
    }
    
    return outputs
}

// FanIn combines multiple channels into one
func FanIn(ctx context.Context, inputs ...<-chan interface{}) <-chan interface{} {
    output := make(chan interface{})
    var wg sync.WaitGroup
    
    // Start a goroutine for each input channel
    for _, input := range inputs {
        wg.Add(1)
        go func(input <-chan interface{}) {
            defer wg.Done()
            for data := range input {
                select {
                case output <- data:
                case <-ctx.Done():
                    return
                }
            }
        }(input)
    }
    
    // Close output when all inputs are done
    go func() {
        wg.Wait()
        close(output)
    }()
    
    return output
}

// Worker function
func worker(ctx context.Context, input <-chan interface{}) <-chan interface{} {
    output := make(chan interface{})
    
    go func() {
        defer close(output)
        for data := range input {
            // Process data
            result := fmt.Sprintf("Processed: %v", data)
            
            select {
            case output <- result:
            case <-ctx.Done():
                return
            }
        }
    }()
    
    return output
}

// Example usage
func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    // Create input channel
    input := make(chan interface{})
    
    // Fan out to multiple workers
    outputs := FanOut(ctx, input, 3, worker)
    
    // Fan in results
    result := FanIn(ctx, outputs...)
    
    // Send data
    go func() {
        defer close(input)
        for i := 0; i < 10; i++ {
            input <- fmt.Sprintf("Data %d", i)
        }
    }()
    
    // Collect results
    for data := range result {
        log.Printf("Result: %v", data)
    }
}
```

### 4. Circuit Breaker Pattern

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

// CircuitState represents the state of the circuit breaker
type CircuitState int

const (
    Closed CircuitState = iota
    Open
    HalfOpen
)

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
    Name           string
    State          CircuitState
    FailureCount   int
    SuccessCount   int
    LastFailTime   time.Time
    Timeout        time.Duration
    MaxFailures    int
    ResetTimeout   time.Duration
    Mutex          sync.RWMutex
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(name string, maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        Name:         name,
        State:        Closed,
        MaxFailures:  maxFailures,
        ResetTimeout: resetTimeout,
        Timeout:      1 * time.Second,
    }
}

// Execute executes a function with circuit breaker protection
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
    cb.Mutex.Lock()
    defer cb.Mutex.Unlock()
    
    // Check if circuit is open
    if cb.State == Open {
        if time.Since(cb.LastFailTime) > cb.ResetTimeout {
            cb.State = HalfOpen
            cb.SuccessCount = 0
        } else {
            return nil, fmt.Errorf("circuit breaker is open")
        }
    }
    
    // Execute function with timeout
    resultChan := make(chan interface{}, 1)
    errorChan := make(chan error, 1)
    
    go func() {
        result, err := fn()
        if err != nil {
            errorChan <- err
        } else {
            resultChan <- result
        }
    }()
    
    select {
    case result := <-resultChan:
        cb.onSuccess()
        return result, nil
        
    case err := <-errorChan:
        cb.onFailure()
        return nil, err
        
    case <-ctx.Done():
        cb.onFailure()
        return nil, ctx.Err()
        
    case <-time.After(cb.Timeout):
        cb.onFailure()
        return nil, fmt.Errorf("operation timed out")
    }
}

// onSuccess handles successful execution
func (cb *CircuitBreaker) onSuccess() {
    cb.SuccessCount++
    cb.FailureCount = 0
    
    if cb.State == HalfOpen {
        cb.State = Closed
    }
}

// onFailure handles failed execution
func (cb *CircuitBreaker) onFailure() {
    cb.FailureCount++
    cb.LastFailTime = time.Now()
    
    if cb.FailureCount >= cb.MaxFailures {
        cb.State = Open
    }
}

// GetState returns the current state of the circuit breaker
func (cb *CircuitBreaker) GetState() CircuitState {
    cb.Mutex.RLock()
    defer cb.Mutex.RUnlock()
    return cb.State
}

// Example usage
func main() {
    // Create circuit breaker
    cb := NewCircuitBreaker("test", 3, 5*time.Second)
    
    // Simulate some operations
    for i := 0; i < 10; i++ {
        result, err := cb.Execute(context.Background(), func() (interface{}, error) {
            // Simulate some work
            time.Sleep(100 * time.Millisecond)
            
            // Simulate failure for first few calls
            if i < 5 {
                return nil, fmt.Errorf("simulated error")
            }
            
            return fmt.Sprintf("Success %d", i), nil
        })
        
        if err != nil {
            log.Printf("Operation %d failed: %v (State: %v)", i, err, cb.GetState())
        } else {
            log.Printf("Operation %d succeeded: %v (State: %v)", i, result, cb.GetState())
        }
        
        time.Sleep(500 * time.Millisecond)
    }
}
```

## Best Practices

### 1. Context Usage

```go
// Always use context for cancellation
func processWithContext(ctx context.Context, data interface{}) error {
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
        // Process data
        return nil
    }
}
```

### 2. Channel Management

```go
// Always close channels when done
func processData(input <-chan interface{}) <-chan interface{} {
    output := make(chan interface{})
    
    go func() {
        defer close(output) // Always close output channel
        for data := range input {
            // Process data
            output <- data
        }
    }()
    
    return output
}
```

### 3. Goroutine Lifecycle

```go
// Use sync.WaitGroup for goroutine synchronization
func processConcurrently(items []interface{}) {
    var wg sync.WaitGroup
    
    for _, item := range items {
        wg.Add(1)
        go func(item interface{}) {
            defer wg.Done()
            // Process item
        }(item)
    }
    
    wg.Wait() // Wait for all goroutines to complete
}
```

## Common Pitfalls

1. **Goroutine Leaks**: Always ensure goroutines can exit
2. **Deadlocks**: Be careful with channel operations
3. **Race Conditions**: Use proper synchronization
4. **Context Cancellation**: Always check context.Done()

## Performance Considerations

1. **Goroutine Overhead**: Don't create too many goroutines
2. **Channel Buffering**: Use buffered channels when appropriate
3. **Memory Usage**: Be mindful of goroutine stack size
4. **CPU Usage**: Use worker pools for CPU-intensive tasks

## Interview Questions

1. **What's the difference between goroutines and threads?**
   - Goroutines are lighter weight, managed by Go runtime, use M:N scheduling

2. **How do you prevent goroutine leaks?**
   - Use context for cancellation, ensure all goroutines can exit

3. **When would you use channels vs sync primitives?**
   - Channels for communication, sync primitives for synchronization

4. **How do you handle timeouts in concurrent operations?**
   - Use context.WithTimeout or time.After with select

The optimal solution uses:
1. **Proper Context Usage**: Always use context for cancellation
2. **Channel Management**: Always close channels when done
3. **Synchronization**: Use appropriate sync primitives
4. **Error Handling**: Handle errors gracefully in concurrent operations
