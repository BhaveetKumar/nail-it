# ⚡ Concurrency and Parallelism in Go

## Table of Contents
1. [Concurrency vs Parallelism](#concurrency-vs-parallelism)
2. [Goroutines](#goroutines)
3. [Channels](#channels)
4. [Synchronization](#synchronization)
5. [Select Statement](#select-statement)
6. [Context Package](#context-package)
7. [Common Patterns](#common-patterns)
8. [Error Handling](#error-handling)
9. [Performance Considerations](#performance-considerations)
10. [Interview Questions](#interview-questions)

## Concurrency vs Parallelism

### Concurrency
- **Definition**: Managing multiple tasks at the same time
- **Focus**: Structure and design
- **Example**: Handling multiple HTTP requests simultaneously

### Parallelism
- **Definition**: Executing multiple tasks simultaneously
- **Focus**: Performance and execution
- **Example**: Processing data on multiple CPU cores

```go
// Concurrency: Multiple goroutines handling different tasks
func main() {
    go handleHTTPRequests()
    go processBackgroundTasks()
    go monitorSystemHealth()
    
    // Main goroutine continues
    select {}
}

// Parallelism: Multiple goroutines working on the same problem
func parallelSum(numbers []int) int {
    const numWorkers = 4
    chunkSize := len(numbers) / numWorkers
    
    results := make(chan int, numWorkers)
    
    for i := 0; i < numWorkers; i++ {
        start := i * chunkSize
        end := start + chunkSize
        if i == numWorkers-1 {
            end = len(numbers)
        }
        
        go func(chunk []int) {
            sum := 0
            for _, num := range chunk {
                sum += num
            }
            results <- sum
        }(numbers[start:end])
    }
    
    total := 0
    for i := 0; i < numWorkers; i++ {
        total += <-results
    }
    
    return total
}
```

## Goroutines

### Basic Goroutine Usage

```go
package main

import (
    "fmt"
    "time"
)

func sayHello(name string) {
    for i := 0; i < 5; i++ {
        fmt.Printf("Hello %s! (%d)\n", name, i)
        time.Sleep(100 * time.Millisecond)
    }
}

func main() {
    // Sequential execution
    sayHello("Alice")
    sayHello("Bob")
    
    // Concurrent execution
    go sayHello("Charlie")
    go sayHello("David")
    
    // Wait for goroutines to complete
    time.Sleep(1 * time.Second)
}
```

### Goroutine Lifecycle

```go
package main

import (
    "fmt"
    "runtime"
    "time"
)

func worker(id int, jobs <-chan int, results chan<- int) {
    for job := range jobs {
        fmt.Printf("Worker %d processing job %d\n", id, job)
        time.Sleep(100 * time.Millisecond)
        results <- job * 2
    }
}

func main() {
    numWorkers := 3
    numJobs := 10
    
    jobs := make(chan int, numJobs)
    results := make(chan int, numJobs)
    
    // Start workers
    for i := 1; i <= numWorkers; i++ {
        go worker(i, jobs, results)
    }
    
    // Send jobs
    for j := 1; j <= numJobs; j++ {
        jobs <- j
    }
    close(jobs)
    
    // Collect results
    for r := 1; r <= numJobs; r++ {
        result := <-results
        fmt.Printf("Result: %d\n", result)
    }
    
    fmt.Printf("Number of goroutines: %d\n", runtime.NumGoroutine())
}
```

## Channels

### Channel Types

```go
package main

import "fmt"

func main() {
    // Unbuffered channel
    ch1 := make(chan int)
    
    // Buffered channel
    ch2 := make(chan int, 3)
    
    // Send-only channel
    var sendOnly chan<- int = ch1
    
    // Receive-only channel
    var receiveOnly <-chan int = ch1
    
    // Channel operations
    go func() {
        ch1 <- 42
        ch2 <- 1
        ch2 <- 2
        ch2 <- 3
        close(ch2)
    }()
    
    // Receive from unbuffered channel
    value := <-ch1
    fmt.Printf("Received: %d\n", value)
    
    // Receive from buffered channel
    for val := range ch2 {
        fmt.Printf("Buffered: %d\n", val)
    }
}
```

### Channel Patterns

#### Fan-out Pattern
```go
package main

import (
    "fmt"
    "math/rand"
    "time"
)

func producer(numbers chan<- int) {
    defer close(numbers)
    for i := 0; i < 10; i++ {
        numbers <- i
        time.Sleep(100 * time.Millisecond)
    }
}

func worker(id int, numbers <-chan int, results chan<- int) {
    for num := range numbers {
        // Simulate work
        time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)
        results <- num * num
        fmt.Printf("Worker %d processed %d\n", id, num)
    }
}

func main() {
    numbers := make(chan int)
    results := make(chan int, 10)
    
    // Start producer
    go producer(numbers)
    
    // Start workers
    numWorkers := 3
    for i := 0; i < numWorkers; i++ {
        go worker(i, numbers, results)
    }
    
    // Collect results
    go func() {
        defer close(results)
        for i := 0; i < 10; i++ {
            result := <-results
            fmt.Printf("Result: %d\n", result)
        }
    }()
    
    time.Sleep(2 * time.Second)
}
```

#### Fan-in Pattern
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

func producer(name string, numbers chan<- int) {
    defer close(numbers)
    for i := 0; i < 5; i++ {
        numbers <- i
        fmt.Printf("%s produced %d\n", name, i)
        time.Sleep(100 * time.Millisecond)
    }
}

func fanIn(inputs ...<-chan int) <-chan int {
    output := make(chan int)
    var wg sync.WaitGroup
    
    for _, input := range inputs {
        wg.Add(1)
        go func(ch <-chan int) {
            defer wg.Done()
            for val := range ch {
                output <- val
            }
        }(input)
    }
    
    go func() {
        wg.Wait()
        close(output)
    }()
    
    return output
}

func main() {
    ch1 := make(chan int)
    ch2 := make(chan int)
    ch3 := make(chan int)
    
    go producer("Producer1", ch1)
    go producer("Producer2", ch2)
    go producer("Producer3", ch3)
    
    // Fan-in all channels
    merged := fanIn(ch1, ch2, ch3)
    
    for val := range merged {
        fmt.Printf("Merged: %d\n", val)
    }
}
```

## Synchronization

### Mutex
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Counter struct {
    mu    sync.Mutex
    value int
}

func (c *Counter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.value++
}

func (c *Counter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.value
}

func main() {
    counter := &Counter{}
    var wg sync.WaitGroup
    
    // Start 1000 goroutines
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.Increment()
        }()
    }
    
    wg.Wait()
    fmt.Printf("Final count: %d\n", counter.Value())
}
```

### RWMutex
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Cache struct {
    mu   sync.RWMutex
    data map[string]string
}

func NewCache() *Cache {
    return &Cache{
        data: make(map[string]string),
    }
}

func (c *Cache) Get(key string) (string, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    val, ok := c.data[key]
    return val, ok
}

func (c *Cache) Set(key, value string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.data[key] = value
}

func (c *Cache) Reader(id int) {
    for i := 0; i < 5; i++ {
        val, ok := c.Get("key1")
        if ok {
            fmt.Printf("Reader %d: %s\n", id, val)
        }
        time.Sleep(100 * time.Millisecond)
    }
}

func (c *Cache) Writer(id int) {
    for i := 0; i < 3; i++ {
        c.Set("key1", fmt.Sprintf("value%d", i))
        fmt.Printf("Writer %d: wrote value%d\n", id, i)
        time.Sleep(200 * time.Millisecond)
    }
}

func main() {
    cache := NewCache()
    cache.Set("key1", "initial")
    
    var wg sync.WaitGroup
    
    // Start readers
    for i := 0; i < 3; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            cache.Reader(id)
        }(i)
    }
    
    // Start writers
    for i := 0; i < 2; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            cache.Writer(id)
        }(i)
    }
    
    wg.Wait()
}
```

### WaitGroup
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    
    fmt.Printf("Worker %d starting\n", id)
    time.Sleep(time.Duration(id) * time.Second)
    fmt.Printf("Worker %d done\n", id)
}

func main() {
    var wg sync.WaitGroup
    
    for i := 1; i <= 5; i++ {
        wg.Add(1)
        go worker(i, &wg)
    }
    
    fmt.Println("Waiting for workers to complete...")
    wg.Wait()
    fmt.Println("All workers completed!")
}
```

### Once
```go
package main

import (
    "fmt"
    "sync"
)

type Singleton struct {
    data string
}

var (
    instance *Singleton
    once     sync.Once
)

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{
            data: "Initialized only once",
        }
        fmt.Println("Singleton initialized")
    })
    return instance
}

func main() {
    var wg sync.WaitGroup
    
    // Try to create multiple instances
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            instance := GetInstance()
            fmt.Printf("Goroutine %d got instance: %s\n", id, instance.data)
        }(i)
    }
    
    wg.Wait()
}
```

## Select Statement

### Basic Select
```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)
    
    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "from ch1"
    }()
    
    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "from ch2"
    }()
    
    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:
            fmt.Println(msg1)
        case msg2 := <-ch2:
            fmt.Println(msg2)
        }
    }
}
```

### Select with Default
```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch := make(chan string)
    
    go func() {
        time.Sleep(2 * time.Second)
        ch <- "message"
    }()
    
    for {
        select {
        case msg := <-ch:
            fmt.Printf("Received: %s\n", msg)
            return
        default:
            fmt.Println("No message yet, doing other work...")
            time.Sleep(500 * time.Millisecond)
        }
    }
}
```

### Select with Timeout
```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch := make(chan string)
    
    go func() {
        time.Sleep(3 * time.Second)
        ch <- "slow message"
    }()
    
    select {
    case msg := <-ch:
        fmt.Printf("Received: %s\n", msg)
    case <-time.After(2 * time.Second):
        fmt.Println("Timeout! No message received")
    }
}
```

## Context Package

### Basic Context Usage
```go
package main

import (
    "context"
    "fmt"
    "time"
)

func worker(ctx context.Context, name string) {
    for {
        select {
        case <-ctx.Done():
            fmt.Printf("Worker %s: Context cancelled\n", name)
            return
        default:
            fmt.Printf("Worker %s: Working...\n", name)
            time.Sleep(500 * time.Millisecond)
        }
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    
    go worker(ctx, "A")
    go worker(ctx, "B")
    
    time.Sleep(2 * time.Second)
    fmt.Println("Cancelling context...")
    cancel()
    
    time.Sleep(1 * time.Second)
}
```

### Context with Timeout
```go
package main

import (
    "context"
    "fmt"
    "time"
)

func longRunningTask(ctx context.Context) error {
    for i := 0; i < 10; i++ {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            fmt.Printf("Task progress: %d/10\n", i+1)
            time.Sleep(500 * time.Millisecond)
        }
    }
    return nil
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()
    
    err := longRunningTask(ctx)
    if err != nil {
        fmt.Printf("Task failed: %v\n", err)
    } else {
        fmt.Println("Task completed successfully")
    }
}
```

### Context with Value
```go
package main

import (
    "context"
    "fmt"
)

func processRequest(ctx context.Context) {
    userID := ctx.Value("userID")
    requestID := ctx.Value("requestID")
    
    fmt.Printf("Processing request %s for user %s\n", requestID, userID)
}

func main() {
    ctx := context.WithValue(context.Background(), "userID", "12345")
    ctx = context.WithValue(ctx, "requestID", "req-001")
    
    processRequest(ctx)
}
```

## Common Patterns

### Worker Pool Pattern
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Job struct {
    ID   int
    Data string
}

type Worker struct {
    ID       int
    JobChan  chan Job
    QuitChan chan bool
    WG       *sync.WaitGroup
}

func NewWorker(id int, jobChan chan Job, wg *sync.WaitGroup) *Worker {
    return &Worker{
        ID:       id,
        JobChan:  jobChan,
        QuitChan: make(chan bool),
        WG:       wg,
    }
}

func (w *Worker) Start() {
    go func() {
        defer w.WG.Done()
        for {
            select {
            case job := <-w.JobChan:
                fmt.Printf("Worker %d processing job %d: %s\n", w.ID, job.ID, job.Data)
                time.Sleep(100 * time.Millisecond) // Simulate work
            case <-w.QuitChan:
                fmt.Printf("Worker %d quitting\n", w.ID)
                return
            }
        }
    }()
}

func (w *Worker) Stop() {
    w.QuitChan <- true
}

type WorkerPool struct {
    Workers   []*Worker
    JobChan   chan Job
    QuitChan  chan bool
    WG        sync.WaitGroup
}

func NewWorkerPool(numWorkers int) *WorkerPool {
    jobChan := make(chan Job, 100)
    
    pool := &WorkerPool{
        Workers:  make([]*Worker, numWorkers),
        JobChan:  jobChan,
        QuitChan: make(chan bool),
    }
    
    for i := 0; i < numWorkers; i++ {
        pool.Workers[i] = NewWorker(i, jobChan, &pool.WG)
        pool.WG.Add(1)
        pool.Workers[i].Start()
    }
    
    return pool
}

func (p *WorkerPool) AddJob(job Job) {
    p.JobChan <- job
}

func (p *WorkerPool) Stop() {
    close(p.JobChan)
    for _, worker := range p.Workers {
        worker.Stop()
    }
    p.WG.Wait()
}

func main() {
    pool := NewWorkerPool(3)
    
    // Add jobs
    for i := 0; i < 10; i++ {
        job := Job{
            ID:   i,
            Data: fmt.Sprintf("Job data %d", i),
        }
        pool.AddJob(job)
    }
    
    time.Sleep(2 * time.Second)
    pool.Stop()
}
```

### Pipeline Pattern
```go
package main

import (
    "fmt"
    "sync"
)

func stage1(input <-chan int, output chan<- int) {
    defer close(output)
    for num := range input {
        output <- num * 2
    }
}

func stage2(input <-chan int, output chan<- int) {
    defer close(output)
    for num := range input {
        output <- num + 1
    }
}

func stage3(input <-chan int, output chan<- int) {
    defer close(output)
    for num := range input {
        output <- num * 3
    }
}

func main() {
    input := make(chan int)
    stage1Output := make(chan int)
    stage2Output := make(chan int)
    finalOutput := make(chan int)
    
    var wg sync.WaitGroup
    
    // Start pipeline stages
    wg.Add(1)
    go func() {
        defer wg.Done()
        stage1(input, stage1Output)
    }()
    
    wg.Add(1)
    go func() {
        defer wg.Done()
        stage2(stage1Output, stage2Output)
    }()
    
    wg.Add(1)
    go func() {
        defer wg.Done()
        stage3(stage2Output, finalOutput)
    }()
    
    // Send input data
    go func() {
        defer close(input)
        for i := 1; i <= 5; i++ {
            input <- i
        }
    }()
    
    // Collect results
    go func() {
        wg.Wait()
        close(finalOutput)
    }()
    
    for result := range finalOutput {
        fmt.Printf("Final result: %d\n", result)
    }
}
```

## Error Handling

### Error Propagation
```go
package main

import (
    "errors"
    "fmt"
    "sync"
)

type Result struct {
    Value int
    Error error
}

func worker(id int, input <-chan int, output chan<- Result) {
    for num := range input {
        if num%2 == 0 {
            output <- Result{Value: num * 2, Error: nil}
        } else {
            output <- Result{Value: 0, Error: errors.New("odd number")}
        }
    }
}

func main() {
    input := make(chan int)
    output := make(chan Result)
    
    var wg sync.WaitGroup
    
    // Start workers
    for i := 0; i < 3; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            worker(id, input, output)
        }(i)
    }
    
    // Send input
    go func() {
        defer close(input)
        for i := 1; i <= 10; i++ {
            input <- i
        }
    }()
    
    // Collect results
    go func() {
        wg.Wait()
        close(output)
    }()
    
    for result := range output {
        if result.Error != nil {
            fmt.Printf("Error: %v\n", result.Error)
        } else {
            fmt.Printf("Result: %d\n", result.Value)
        }
    }
}
```

## Performance Considerations

### Goroutine Overhead
```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

func sequentialSum(numbers []int) int {
    sum := 0
    for _, num := range numbers {
        sum += num
    }
    return sum
}

func parallelSum(numbers []int) int {
    const numWorkers = 4
    chunkSize := len(numbers) / numWorkers
    
    results := make(chan int, numWorkers)
    var wg sync.WaitGroup
    
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(start, end int) {
            defer wg.Done()
            sum := 0
            for j := start; j < end; j++ {
                sum += numbers[j]
            }
            results <- sum
        }(i*chunkSize, (i+1)*chunkSize)
    }
    
    go func() {
        wg.Wait()
        close(results)
    }()
    
    total := 0
    for sum := range results {
        total += sum
    }
    
    return total
}

func main() {
    // Create large slice
    numbers := make([]int, 1000000)
    for i := range numbers {
        numbers[i] = i
    }
    
    // Sequential
    start := time.Now()
    sum1 := sequentialSum(numbers)
    sequentialTime := time.Since(start)
    
    // Parallel
    start = time.Now()
    sum2 := parallelSum(numbers)
    parallelTime := time.Since(start)
    
    fmt.Printf("Sequential sum: %d, time: %v\n", sum1, sequentialTime)
    fmt.Printf("Parallel sum: %d, time: %v\n", sum2, parallelTime)
    fmt.Printf("Speedup: %.2fx\n", float64(sequentialTime)/float64(parallelTime))
    fmt.Printf("Goroutines: %d\n", runtime.NumGoroutine())
}
```

## Interview Questions

### Basic Concepts
1. **What is the difference between concurrency and parallelism?**
2. **How do goroutines differ from OS threads?**
3. **What are channels and how do they work?**
4. **Explain the select statement in Go.**
5. **What is the purpose of the context package?**

### Advanced Topics
1. **How would you implement a worker pool pattern?**
2. **Explain the fan-in and fan-out patterns.**
3. **How do you handle errors in concurrent programs?**
4. **What are the different types of channels in Go?**
5. **How would you implement a rate limiter using goroutines?**

### System Design
1. **Design a concurrent web scraper.**
2. **How would you implement a concurrent cache?**
3. **Design a message queue system.**
4. **How would you handle graceful shutdown in a concurrent application?**
5. **Design a concurrent file processor.**

## Conclusion

Concurrency in Go is powerful and elegant. Key takeaways:

- Use goroutines for concurrent tasks
- Channels for communication between goroutines
- Select for non-blocking operations
- Context for cancellation and timeouts
- Synchronization primitives for shared state
- Patterns like worker pools for structured concurrency

Master these concepts to build efficient, scalable concurrent applications in Go.
