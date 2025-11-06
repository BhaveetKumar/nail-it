---
# Auto-generated front matter
Title: Advanced Go Concurrency
LastUpdated: 2025-11-06T20:45:58.662201
Tags: []
Status: draft
---

# 🚀 **Advanced Go Concurrency**

## 📊 **Complete Guide to Go Concurrency Patterns**

---

## 🎯 **1. Advanced Goroutine Patterns**

### **Worker Pool with Dynamic Scaling**

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Dynamic Worker Pool
type DynamicWorkerPool struct {
    workers     []*Worker
    jobQueue    chan Job
    resultQueue chan Result
    config      *PoolConfig
    mutex       sync.RWMutex
    ctx         context.Context
    cancel      context.CancelFunc
}

type PoolConfig struct {
    MinWorkers    int
    MaxWorkers    int
    QueueSize     int
    ScaleUpThreshold float64
    ScaleDownThreshold float64
    CheckInterval time.Duration
}

type Worker struct {
    ID       int
    JobQueue chan Job
    Quit     chan bool
    Active   bool
    mutex    sync.RWMutex
}

type Job struct {
    ID       string
    Data     interface{}
    Priority int
    Timeout  time.Duration
}

type Result struct {
    JobID    string
    Data     interface{}
    Error    error
    Duration time.Duration
}

func NewDynamicWorkerPool(config *PoolConfig) *DynamicWorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    
    pool := &DynamicWorkerPool{
        workers:     make([]*Worker, 0, config.MaxWorkers),
        jobQueue:    make(chan Job, config.QueueSize),
        resultQueue: make(chan Result, config.QueueSize),
        config:      config,
        ctx:         ctx,
        cancel:      cancel,
    }
    
    // Start with minimum workers
    for i := 0; i < config.MinWorkers; i++ {
        pool.addWorker()
    }
    
    // Start scaling monitor
    go pool.monitorScaling()
    
    return pool
}

func (p *DynamicWorkerPool) addWorker() {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    if len(p.workers) >= p.config.MaxWorkers {
        return
    }
    
    worker := &Worker{
        ID:       len(p.workers),
        JobQueue: make(chan Job, 1),
        Quit:     make(chan bool),
        Active:   true,
    }
    
    p.workers = append(p.workers, worker)
    
    // Start worker
    go p.runWorker(worker)
}

func (p *DynamicWorkerPool) removeWorker() {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    if len(p.workers) <= p.config.MinWorkers {
        return
    }
    
    // Find least active worker
    var leastActiveWorker *Worker
    minJobs := int(^uint(0) >> 1)
    
    for _, worker := range p.workers {
        if worker.Active && len(worker.JobQueue) < minJobs {
            minJobs = len(worker.JobQueue)
            leastActiveWorker = worker
        }
    }
    
    if leastActiveWorker != nil {
        // Stop worker
        leastActiveWorker.Quit <- true
        leastActiveWorker.Active = false
        
        // Remove from workers slice
        for i, w := range p.workers {
            if w == leastActiveWorker {
                p.workers = append(p.workers[:i], p.workers[i+1:]...)
                break
            }
        }
    }
}

func (p *DynamicWorkerPool) runWorker(worker *Worker) {
    for {
        select {
        case job := <-worker.JobQueue:
            p.processJob(worker, job)
        case <-worker.Quit:
            return
        case <-p.ctx.Done():
            return
        }
    }
}

func (p *DynamicWorkerPool) processJob(worker *Worker, job Job) {
    start := time.Now()
    
    // Process job
    result := p.executeJob(job)
    result.Duration = time.Since(start)
    
    // Send result
    select {
    case p.resultQueue <- result:
    case <-p.ctx.Done():
        return
    }
}

func (p *DynamicWorkerPool) executeJob(job Job) Result {
    // Simulate job execution
    time.Sleep(100 * time.Millisecond)
    
    return Result{
        JobID: job.ID,
        Data:  fmt.Sprintf("processed-%v", job.Data),
        Error: nil,
    }
}

func (p *DynamicWorkerPool) SubmitJob(job Job) error {
    select {
    case p.jobQueue <- job:
        return nil
    case <-p.ctx.Done():
        return p.ctx.Err()
    default:
        return fmt.Errorf("job queue is full")
    }
}

func (p *DynamicWorkerPool) GetResult() <-chan Result {
    return p.resultQueue
}

func (p *DynamicWorkerPool) monitorScaling() {
    ticker := time.NewTicker(p.config.CheckInterval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            p.checkScaling()
        case <-p.ctx.Done():
            return
        }
    }
}

func (p *DynamicWorkerPool) checkScaling() {
    p.mutex.RLock()
    workerCount := len(p.workers)
    queueLength := len(p.jobQueue)
    p.mutex.RUnlock()
    
    // Calculate load
    load := float64(queueLength) / float64(workerCount)
    
    if load > p.config.ScaleUpThreshold && workerCount < p.config.MaxWorkers {
        p.addWorker()
    } else if load < p.config.ScaleDownThreshold && workerCount > p.config.MinWorkers {
        p.removeWorker()
    }
}

func (p *DynamicWorkerPool) distributeJob(job Job) {
    p.mutex.RLock()
    workers := make([]*Worker, len(p.workers))
    copy(workers, p.workers)
    p.mutex.RUnlock()
    
    // Find worker with least load
    var bestWorker *Worker
    minLoad := int(^uint(0) >> 1)
    
    for _, worker := range workers {
        if worker.Active {
            load := len(worker.JobQueue)
            if load < minLoad {
                minLoad = load
                bestWorker = worker
            }
        }
    }
    
    if bestWorker != nil {
        select {
        case bestWorker.JobQueue <- job:
        default:
            // Worker is busy, queue job
            p.jobQueue <- job
        }
    }
}

func (p *DynamicWorkerPool) Close() {
    p.cancel()
    
    // Stop all workers
    p.mutex.Lock()
    for _, worker := range p.workers {
        worker.Quit <- true
    }
    p.mutex.Unlock()
}

// Example usage
func main() {
    config := &PoolConfig{
        MinWorkers:         2,
        MaxWorkers:         10,
        QueueSize:          100,
        ScaleUpThreshold:   0.8,
        ScaleDownThreshold: 0.2,
        CheckInterval:      5 * time.Second,
    }
    
    pool := NewDynamicWorkerPool(config)
    
    // Submit jobs
    for i := 0; i < 50; i++ {
        job := Job{
            ID:       fmt.Sprintf("job-%d", i),
            Data:     fmt.Sprintf("data-%d", i),
            Priority: i % 5,
            Timeout:  30 * time.Second,
        }
        
        if err := pool.SubmitJob(job); err != nil {
            fmt.Printf("Failed to submit job: %v\n", err)
        }
    }
    
    // Process results
    go func() {
        for result := range pool.GetResult() {
            fmt.Printf("Result: %+v\n", result)
        }
    }()
    
    // Keep running
    time.Sleep(10 * time.Second)
    pool.Close()
}
```

---

## 🎯 **2. Advanced Channel Patterns**

### **Pipeline with Backpressure and Error Handling**

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Advanced Pipeline
type Pipeline struct {
    stages    []*Stage
    input     chan interface{}
    output    chan interface{}
    errors    chan error
    ctx       context.Context
    cancel    context.CancelFunc
    wg        sync.WaitGroup
}

type Stage struct {
    Name        string
    Processor   func(interface{}) (interface{}, error)
    Concurrency int
    BufferSize  int
    input       chan interface{}
    output      chan interface{}
    errors      chan error
    wg          sync.WaitGroup
}

func NewPipeline() *Pipeline {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &Pipeline{
        stages: make([]*Stage, 0),
        input:  make(chan interface{}, 100),
        output: make(chan interface{}, 100),
        errors: make(chan error, 100),
        ctx:    ctx,
        cancel: cancel,
    }
}

func (p *Pipeline) AddStage(name string, processor func(interface{}) (interface{}, error), concurrency, bufferSize int) {
    stage := &Stage{
        Name:        name,
        Processor:   processor,
        Concurrency: concurrency,
        BufferSize:  bufferSize,
        input:       make(chan interface{}, bufferSize),
        output:      make(chan interface{}, bufferSize),
        errors:      make(chan error, bufferSize),
    }
    
    p.stages = append(p.stages, stage)
}

func (p *Pipeline) Start() {
    // Connect stages
    for i := 0; i < len(p.stages); i++ {
        if i == 0 {
            // First stage gets input from pipeline
            p.stages[i].input = p.input
        } else {
            // Connect to previous stage
            p.stages[i].input = p.stages[i-1].output
        }
        
        if i == len(p.stages)-1 {
            // Last stage sends to pipeline output
            p.stages[i].output = p.output
        }
        
        // Start stage
        p.startStage(p.stages[i])
    }
    
    // Start error collector
    go p.collectErrors()
}

func (p *Pipeline) startStage(stage *Stage) {
    for i := 0; i < stage.Concurrency; i++ {
        stage.wg.Add(1)
        go p.runWorker(stage, i)
    }
}

func (p *Pipeline) runWorker(stage *Stage, workerID int) {
    defer stage.wg.Done()
    
    for {
        select {
        case data := <-stage.input:
            if data == nil {
                return
            }
            
            result, err := stage.Processor(data)
            if err != nil {
                select {
                case stage.errors <- err:
                case <-p.ctx.Done():
                    return
                }
                continue
            }
            
            select {
            case stage.output <- result:
            case <-p.ctx.Done():
                return
            }
            
        case <-p.ctx.Done():
            return
        }
    }
}

func (p *Pipeline) collectErrors() {
    for {
        select {
        case err := <-p.errors:
            fmt.Printf("Pipeline error: %v\n", err)
        case <-p.ctx.Done():
            return
        }
    }
}

func (p *Pipeline) Input() chan<- interface{} {
    return p.input
}

func (p *Pipeline) Output() <-chan interface{} {
    return p.output
}

func (p *Pipeline) Errors() <-chan error {
    return p.errors
}

func (p *Pipeline) Close() {
    p.cancel()
    
    // Close input
    close(p.input)
    
    // Wait for all stages to complete
    for _, stage := range p.stages {
        stage.wg.Wait()
        close(stage.output)
    }
    
    // Close output
    close(p.output)
}

// Fan-out Fan-in Pattern
type FanOutFanIn struct {
    input     chan interface{}
    outputs   []chan interface{}
    inputs    []chan interface{}
    output    chan interface{}
    ctx       context.Context
    cancel    context.CancelFunc
    wg        sync.WaitGroup
}

func NewFanOutFanIn(fanOutCount, fanInCount int) *FanOutFanIn {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &FanOutFanIn{
        input:   make(chan interface{}, 100),
        outputs: make([]chan interface{}, fanOutCount),
        inputs:  make([]chan interface{}, fanInCount),
        output:  make(chan interface{}, 100),
        ctx:     ctx,
        cancel:  cancel,
    }
}

func (fofi *FanOutFanIn) Start() {
    // Initialize output channels
    for i := range fofi.outputs {
        fofi.outputs[i] = make(chan interface{}, 100)
    }
    
    // Initialize input channels
    for i := range fofi.inputs {
        fofi.inputs[i] = make(chan interface{}, 100)
    }
    
    // Start fan-out
    fofi.wg.Add(1)
    go fofi.fanOut()
    
    // Start fan-in
    fofi.wg.Add(1)
    go fofi.fanIn()
}

func (fofi *FanOutFanIn) fanOut() {
    defer fofi.wg.Done()
    
    for {
        select {
        case data := <-fofi.input:
            if data == nil {
                // Close all output channels
                for _, output := range fofi.outputs {
                    close(output)
                }
                return
            }
            
            // Distribute to all outputs
            for _, output := range fofi.outputs {
                select {
                case output <- data:
                case <-fofi.ctx.Done():
                    return
                }
            }
            
        case <-fofi.ctx.Done():
            return
        }
    }
}

func (fofi *FanOutFanIn) fanIn() {
    defer fofi.wg.Done()
    
    var wg sync.WaitGroup
    
    // Start workers for each input
    for i, input := range fofi.inputs {
        wg.Add(1)
        go func(inputChan chan interface{}) {
            defer wg.Done()
            
            for data := range inputChan {
                select {
                case fofi.output <- data:
                case <-fofi.ctx.Done():
                    return
                }
            }
        }(input)
    }
    
    wg.Wait()
    close(fofi.output)
}

func (fofi *FanOutFanIn) Input() chan<- interface{} {
    return fofi.input
}

func (fofi *FanOutFanIn) Output() <-chan interface{} {
    return fofi.output
}

func (fofi *FanOutFanIn) GetOutput(index int) <-chan interface{} {
    return fofi.outputs[index]
}

func (fofi *FanOutFanIn) SetInput(index int, input chan interface{}) {
    fofi.inputs[index] = input
}

func (fofi *FanOutFanIn) Close() {
    fofi.cancel()
    close(fofi.input)
    fofi.wg.Wait()
}

// Example usage
func main() {
    // Create pipeline
    pipeline := NewPipeline()
    
    // Add stages
    pipeline.AddStage("stage1", func(data interface{}) (interface{}, error) {
        return fmt.Sprintf("processed-%v", data), nil
    }, 2, 10)
    
    pipeline.AddStage("stage2", func(data interface{}) (interface{}, error) {
        return fmt.Sprintf("final-%v", data), nil
    }, 3, 15)
    
    // Start pipeline
    pipeline.Start()
    
    // Send data
    go func() {
        for i := 0; i < 10; i++ {
            pipeline.Input() <- fmt.Sprintf("data-%d", i)
        }
        close(pipeline.Input())
    }()
    
    // Process output
    go func() {
        for result := range pipeline.Output() {
            fmt.Printf("Result: %v\n", result)
        }
    }()
    
    // Wait for completion
    time.Sleep(2 * time.Second)
    pipeline.Close()
}
```

---

## 🎯 **3. Advanced Synchronization Patterns**

### **Read-Write Locks with Priority**

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

// Priority Read-Write Lock
type PriorityRWMutex struct {
    readers    int32
    writers    int32
    readerCond *sync.Cond
    writerCond *sync.Cond
    mutex      sync.Mutex
    priority   Priority
}

type Priority int

const (
    ReaderPriority Priority = iota
    WriterPriority
    FairPriority
)

func NewPriorityRWMutex(priority Priority) *PriorityRWMutex {
    mutex := &PriorityRWMutex{
        priority: priority,
    }
    mutex.readerCond = sync.NewCond(&mutex.mutex)
    mutex.writerCond = sync.NewCond(&mutex.mutex)
    return mutex
}

func (rw *PriorityRWMutex) RLock() {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    // Wait for writers to finish
    for atomic.LoadInt32(&rw.writers) > 0 {
        rw.readerCond.Wait()
    }
    
    atomic.AddInt32(&rw.readers, 1)
}

func (rw *PriorityRWMutex) RUnlock() {
    readers := atomic.AddInt32(&rw.readers, -1)
    
    if readers == 0 {
        rw.writerCond.Signal()
    }
}

func (rw *PriorityRWMutex) Lock() {
    rw.mutex.Lock()
    defer rw.mutex.Unlock()
    
    // Wait for readers and writers to finish
    for atomic.LoadInt32(&rw.readers) > 0 || atomic.LoadInt32(&rw.writers) > 0 {
        rw.writerCond.Wait()
    }
    
    atomic.StoreInt32(&rw.writers, 1)
}

func (rw *PriorityRWMutex) Unlock() {
    atomic.StoreInt32(&rw.writers, 0)
    rw.writerCond.Signal()
    rw.readerCond.Broadcast()
}

// Advanced Semaphore
type Semaphore struct {
    permits   int32
    available int32
    waiters   []chan struct{}
    mutex     sync.Mutex
}

func NewSemaphore(permits int32) *Semaphore {
    return &Semaphore{
        permits:   permits,
        available: permits,
        waiters:   make([]chan struct{}, 0),
    }
}

func (s *Semaphore) Acquire() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    if s.available > 0 {
        s.available--
        return
    }
    
    // Wait for permit
    waiter := make(chan struct{})
    s.waiters = append(s.waiters, waiter)
    s.mutex.Unlock()
    
    <-waiter
    
    s.mutex.Lock()
    s.available--
}

func (s *Semaphore) TryAcquire() bool {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    if s.available > 0 {
        s.available--
        return true
    }
    
    return false
}

func (s *Semaphore) Release() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    if len(s.waiters) > 0 {
        // Wake up first waiter
        waiter := s.waiters[0]
        s.waiters = s.waiters[1:]
        close(waiter)
    } else {
        s.available++
    }
}

func (s *Semaphore) AvailablePermits() int32 {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    return s.available
}

// Example usage
func main() {
    // Test PriorityRWMutex
    rw := NewPriorityRWMutex(ReaderPriority)
    
    // Test readers
    for i := 0; i < 5; i++ {
        go func(id int) {
            rw.RLock()
            fmt.Printf("Reader %d acquired lock\n", id)
            time.Sleep(100 * time.Millisecond)
            rw.RUnlock()
            fmt.Printf("Reader %d released lock\n", id)
        }(i)
    }
    
    // Test writers
    for i := 0; i < 3; i++ {
        go func(id int) {
            rw.Lock()
            fmt.Printf("Writer %d acquired lock\n", id)
            time.Sleep(200 * time.Millisecond)
            rw.Unlock()
            fmt.Printf("Writer %d released lock\n", id)
        }(i)
    }
    
    // Test Semaphore
    sem := NewSemaphore(3)
    
    for i := 0; i < 10; i++ {
        go func(id int) {
            sem.Acquire()
            fmt.Printf("Worker %d acquired permit\n", id)
            time.Sleep(1 * time.Second)
            sem.Release()
            fmt.Printf("Worker %d released permit\n", id)
        }(i)
    }
    
    time.Sleep(5 * time.Second)
}
```

---

## 🎯 **Key Takeaways from Advanced Go Concurrency**

### **1. Advanced Goroutine Patterns**
- **Dynamic Worker Pools**: Auto-scaling worker pools based on load
- **Load Balancing**: Intelligent job distribution across workers
- **Resource Management**: Efficient resource allocation and cleanup
- **Monitoring**: Comprehensive monitoring and metrics collection

### **2. Advanced Channel Patterns**
- **Pipeline Processing**: Multi-stage data processing pipelines
- **Fan-out Fan-in**: Parallel processing with result aggregation
- **Backpressure**: Flow control and error handling
- **Error Propagation**: Comprehensive error handling and recovery

### **3. Advanced Synchronization**
- **Priority Locks**: Reader-writer locks with priority handling
- **Semaphores**: Advanced semaphore implementation with waiting
- **Condition Variables**: Advanced synchronization primitives
- **Deadlock Prevention**: Strategies for preventing deadlocks

### **4. Production Considerations**
- **Performance**: Optimized for high-throughput processing
- **Scalability**: Horizontal scaling with dynamic resource allocation
- **Monitoring**: Comprehensive observability and metrics
- **Error Handling**: Robust error handling and recovery mechanisms

---

**🎉 This comprehensive guide provides advanced Go concurrency patterns with production-ready implementations for high-performance concurrent systems! 🚀**
