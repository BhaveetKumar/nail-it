# 🚀 **Advanced Go Concepts - Complete Guide**

## 📊 **Deep Dive into Advanced Go Programming Techniques**

---

## 🎯 **1. Go Runtime Internals**

### **Goroutine Scheduler Deep Dive**

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

// Goroutine Pool with Work Stealing
type GoroutinePool struct {
    workers    int
    workQueue  chan func()
    done       chan struct{}
    wg         sync.WaitGroup
    mutex      sync.RWMutex
}

func NewGoroutinePool(workers int) *GoroutinePool {
    return &GoroutinePool{
        workers:   workers,
        workQueue: make(chan func(), 1000),
        done:      make(chan struct{}),
    }
}

func (gp *GoroutinePool) Start() {
    for i := 0; i < gp.workers; i++ {
        gp.wg.Add(1)
        go gp.worker(i)
    }
}

func (gp *GoroutinePool) worker(id int) {
    defer gp.wg.Done()
    
    for {
        select {
        case work := <-gp.workQueue:
            work()
        case <-gp.done:
            return
        }
    }
}

func (gp *GoroutinePool) Submit(work func()) {
    select {
    case gp.workQueue <- work:
    default:
        // Queue is full, handle overflow
        go work()
    }
}

func (gp *GoroutinePool) Stop() {
    close(gp.done)
    gp.wg.Wait()
}

// Memory Pool for Object Reuse
type MemoryPool struct {
    pool sync.Pool
    size int
}

type PooledObject struct {
    Data    []byte
    ID      int
    Created time.Time
}

func NewMemoryPool(size int) *MemoryPool {
    return &MemoryPool{
        pool: sync.Pool{
            New: func() interface{} {
                return &PooledObject{
                    Data:    make([]byte, size),
                    ID:      0,
                    Created: time.Now(),
                }
            },
        },
        size: size,
    }
}

func (mp *MemoryPool) Get() *PooledObject {
    obj := mp.pool.Get().(*PooledObject)
    obj.ID = 0
    obj.Created = time.Now()
    return obj
}

func (mp *MemoryPool) Put(obj *PooledObject) {
    if obj != nil {
        // Reset object
        obj.ID = 0
        for i := range obj.Data {
            obj.Data[i] = 0
        }
        mp.pool.Put(obj)
    }
}

// Lock-Free Data Structures
type LockFreeStack struct {
    head unsafe.Pointer
}

type stackNode struct {
    value interface{}
    next  unsafe.Pointer
}

func NewLockFreeStack() *LockFreeStack {
    return &LockFreeStack{}
}

func (s *LockFreeStack) Push(value interface{}) {
    node := &stackNode{value: value}
    
    for {
        head := atomic.LoadPointer(&s.head)
        node.next = head
        
        if atomic.CompareAndSwapPointer(&s.head, head, unsafe.Pointer(node)) {
            break
        }
    }
}

func (s *LockFreeStack) Pop() (interface{}, bool) {
    for {
        head := atomic.LoadPointer(&s.head)
        if head == nil {
            return nil, false
        }
        
        node := (*stackNode)(head)
        next := atomic.LoadPointer(&node.next)
        
        if atomic.CompareAndSwapPointer(&s.head, head, next) {
            return node.value, true
        }
    }
}

// Context with Timeout and Cancellation
type ContextManager struct {
    contexts map[string]context.Context
    mutex    sync.RWMutex
}

func NewContextManager() *ContextManager {
    return &ContextManager{
        contexts: make(map[string]context.Context),
    }
}

func (cm *ContextManager) CreateContext(id string, timeout time.Duration) context.Context {
    cm.mutex.Lock()
    defer cm.mutex.Unlock()
    
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    cm.contexts[id] = ctx
    
    // Clean up context when it's done
    go func() {
        <-ctx.Done()
        cm.mutex.Lock()
        delete(cm.contexts, id)
        cm.mutex.Unlock()
        cancel()
    }()
    
    return ctx
}

func (cm *ContextManager) CancelContext(id string) {
    cm.mutex.RLock()
    ctx, exists := cm.contexts[id]
    cm.mutex.RUnlock()
    
    if exists {
        // Context will be cleaned up by the goroutine above
        _ = ctx
    }
}

// Example usage
func main() {
    // Goroutine pool example
    pool := NewGoroutinePool(4)
    pool.Start()
    
    // Submit work
    for i := 0; i < 100; i++ {
        i := i
        pool.Submit(func() {
            fmt.Printf("Worker processing task %d\n", i)
            time.Sleep(100 * time.Millisecond)
        })
    }
    
    time.Sleep(2 * time.Second)
    pool.Stop()
    
    // Memory pool example
    memoryPool := NewMemoryPool(1024)
    
    obj := memoryPool.Get()
    obj.ID = 123
    copy(obj.Data, []byte("Hello, World!"))
    
    // Use object...
    
    memoryPool.Put(obj)
    
    // Lock-free stack example
    stack := NewLockFreeStack()
    
    stack.Push("item1")
    stack.Push("item2")
    stack.Push("item3")
    
    for {
        if value, ok := stack.Pop(); ok {
            fmt.Printf("Popped: %v\n", value)
        } else {
            break
        }
    }
    
    // Context manager example
    ctxManager := NewContextManager()
    
    ctx := ctxManager.CreateContext("task1", 5*time.Second)
    
    go func() {
        select {
        case <-ctx.Done():
            fmt.Println("Context cancelled or timed out")
        case <-time.After(3 * time.Second):
            fmt.Println("Task completed")
        }
    }()
    
    time.Sleep(6 * time.Second)
}
```

---

## 🎯 **2. Advanced Concurrency Patterns**

### **Worker Pool with Work Stealing**

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Work Stealing Scheduler
type WorkStealingScheduler struct {
    workers    []*Worker
    workQueues []chan func()
    done       chan struct{}
    wg         sync.WaitGroup
}

type Worker struct {
    ID       int
    workQueue chan func()
    scheduler *WorkStealingScheduler
}

func NewWorkStealingScheduler(workers int) *WorkStealingScheduler {
    ws := &WorkStealingScheduler{
        workers:    make([]*Worker, workers),
        workQueues: make([]chan func(), workers),
        done:       make(chan struct{}),
    }
    
    for i := 0; i < workers; i++ {
        ws.workQueues[i] = make(chan func(), 100)
        ws.workers[i] = &Worker{
            ID:       i,
            workQueue: ws.workQueues[i],
            scheduler: ws,
        }
    }
    
    return ws
}

func (ws *WorkStealingScheduler) Start() {
    for _, worker := range ws.workers {
        ws.wg.Add(1)
        go worker.run()
    }
}

func (ws *WorkStealingScheduler) Submit(work func()) {
    // Try to submit to a random worker's queue
    workerID := time.Now().UnixNano() % int64(len(ws.workers))
    select {
    case ws.workQueues[workerID] <- work:
    default:
        // Queue is full, try another worker
        for i := 0; i < len(ws.workers); i++ {
            select {
            case ws.workQueues[i] <- work:
                return
            default:
                continue
            }
        }
        // All queues are full, run in a new goroutine
        go work()
    }
}

func (ws *WorkStealingScheduler) Stop() {
    close(ws.done)
    ws.wg.Wait()
}

func (w *Worker) run() {
    defer w.scheduler.wg.Done()
    
    for {
        select {
        case work := <-w.workQueue:
            work()
        case <-w.scheduler.done:
            return
        default:
            // Try to steal work from other workers
            if w.stealWork() {
                continue
            }
            // No work available, yield
            time.Sleep(1 * time.Microsecond)
        }
    }
}

func (w *Worker) stealWork() bool {
    // Try to steal from other workers
    for i := 0; i < len(w.scheduler.workQueues); i++ {
        if i == w.ID {
            continue
        }
        
        select {
        case work := <-w.scheduler.workQueues[i]:
            work()
            return true
        default:
            continue
        }
    }
    
    return false
}

// Pipeline Pattern
type Pipeline struct {
    stages []*Stage
    mutex  sync.RWMutex
}

type Stage struct {
    ID       int
    input    chan interface{}
    output   chan interface{}
    process  func(interface{}) interface{}
    workers  int
    done     chan struct{}
    wg       sync.WaitGroup
}

func NewPipeline() *Pipeline {
    return &Pipeline{
        stages: make([]*Stage, 0),
    }
}

func (p *Pipeline) AddStage(workers int, process func(interface{}) interface{}) *Stage {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    stage := &Stage{
        ID:      len(p.stages),
        input:   make(chan interface{}, 100),
        output:  make(chan interface{}, 100),
        process: process,
        workers: workers,
        done:    make(chan struct{}),
    }
    
    p.stages = append(p.stages, stage)
    return stage
}

func (p *Pipeline) ConnectStages() {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    for i := 0; i < len(p.stages)-1; i++ {
        p.stages[i].output = p.stages[i+1].input
    }
}

func (p *Pipeline) Start() {
    p.mutex.RLock()
    defer p.mutex.RUnlock()
    
    for _, stage := range p.stages {
        for i := 0; i < stage.workers; i++ {
            stage.wg.Add(1)
            go stage.worker(i)
        }
    }
}

func (p *Pipeline) Stop() {
    p.mutex.RLock()
    defer p.mutex.RUnlock()
    
    for _, stage := range p.stages {
        close(stage.done)
        stage.wg.Wait()
    }
}

func (s *Stage) worker(id int) {
    defer s.wg.Done()
    
    for {
        select {
        case data := <-s.input:
            if s.process != nil {
                result := s.process(data)
                if s.output != nil {
                    s.output <- result
                }
            }
        case <-s.done:
            return
        }
    }
}

func (s *Stage) Input() chan<- interface{} {
    return s.input
}

func (s *Stage) Output() <-chan interface{} {
    return s.output
}

// Fan-Out Fan-In Pattern
type FanOutFanIn struct {
    input    chan interface{}
    outputs  []chan interface{}
    inputs   []chan interface{}
    output   chan interface{}
    workers  int
    mutex    sync.RWMutex
}

func NewFanOutFanIn(workers int) *FanOutFanIn {
    return &FanOutFanIn{
        input:   make(chan interface{}, 100),
        outputs: make([]chan interface{}, workers),
        inputs:  make([]chan interface{}, workers),
        output:  make(chan interface{}, 100),
        workers: workers,
    }
}

func (fofi *FanOutFanIn) Start() {
    // Fan-out: distribute input to multiple workers
    for i := 0; i < fofi.workers; i++ {
        fofi.outputs[i] = make(chan interface{}, 100)
        fofi.inputs[i] = make(chan interface{}, 100)
        
        go fofi.fanOutWorker(i)
        go fofi.fanInWorker(i)
    }
    
    // Fan-in: collect results from all workers
    go fofi.collectResults()
}

func (fofi *FanOutFanIn) fanOutWorker(id int) {
    for data := range fofi.input {
        select {
        case fofi.outputs[id] <- data:
        case <-time.After(1 * time.Second):
            // Worker is busy, skip this data
            continue
        }
    }
    close(fofi.outputs[id])
}

func (fofi *FanOutFanIn) fanInWorker(id int) {
    for data := range fofi.outputs[id] {
        // Process data
        result := fofi.process(data)
        fofi.inputs[id] <- result
    }
    close(fofi.inputs[id])
}

func (fofi *FanOutFanIn) process(data interface{}) interface{} {
    // Simulate processing
    time.Sleep(100 * time.Millisecond)
    return fmt.Sprintf("processed_%v", data)
}

func (fofi *FanOutFanIn) collectResults() {
    var wg sync.WaitGroup
    
    for i := 0; i < fofi.workers; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for result := range fofi.inputs[id] {
                fofi.output <- result
            }
        }(i)
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

// Example usage
func main() {
    // Work stealing scheduler example
    scheduler := NewWorkStealingScheduler(4)
    scheduler.Start()
    
    // Submit work
    for i := 0; i < 100; i++ {
        i := i
        scheduler.Submit(func() {
            fmt.Printf("Processing task %d\n", i)
            time.Sleep(100 * time.Millisecond)
        })
    }
    
    time.Sleep(2 * time.Second)
    scheduler.Stop()
    
    // Pipeline example
    pipeline := NewPipeline()
    
    // Add stages
    stage1 := pipeline.AddStage(2, func(data interface{}) interface{} {
        return fmt.Sprintf("stage1_%v", data)
    })
    
    stage2 := pipeline.AddStage(3, func(data interface{}) interface{} {
        return fmt.Sprintf("stage2_%v", data)
    })
    
    stage3 := pipeline.AddStage(1, func(data interface{}) interface{} {
        return fmt.Sprintf("stage3_%v", data)
    })
    
    // Connect stages
    pipeline.ConnectStages()
    
    // Start pipeline
    pipeline.Start()
    
    // Send data
    go func() {
        for i := 0; i < 10; i++ {
            stage1.Input() <- i
        }
        close(stage1.Input())
    }()
    
    // Collect results
    go func() {
        for result := range stage3.Output() {
            fmt.Printf("Pipeline result: %v\n", result)
        }
    }()
    
    time.Sleep(2 * time.Second)
    pipeline.Stop()
    
    // Fan-out fan-in example
    fofi := NewFanOutFanIn(3)
    fofi.Start()
    
    // Send data
    go func() {
        for i := 0; i < 10; i++ {
            fofi.Input() <- i
        }
        close(fofi.Input())
    }()
    
    // Collect results
    go func() {
        for result := range fofi.Output() {
            fmt.Printf("Fan-out fan-in result: %v\n", result)
        }
    }()
    
    time.Sleep(2 * time.Second)
}
```

---

## 🎯 **3. Memory Management and Optimization**

### **Advanced Memory Patterns**

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
    "unsafe"
)

// Memory Arena for Efficient Allocation
type MemoryArena struct {
    blocks    [][]byte
    current   []byte
    offset    int
    blockSize int
    mutex     sync.Mutex
}

func NewMemoryArena(blockSize int) *MemoryArena {
    return &MemoryArena{
        blockSize: blockSize,
    }
}

func (ma *MemoryArena) Allocate(size int) []byte {
    ma.mutex.Lock()
    defer ma.mutex.Unlock()
    
    if ma.current == nil || ma.offset+size > len(ma.current) {
        // Allocate new block
        ma.current = make([]byte, ma.blockSize)
        ma.blocks = append(ma.blocks, ma.current)
        ma.offset = 0
    }
    
    if ma.offset+size > len(ma.current) {
        // Size too large for current block
        return make([]byte, size)
    }
    
    result := ma.current[ma.offset : ma.offset+size]
    ma.offset += size
    return result
}

func (ma *MemoryArena) Reset() {
    ma.mutex.Lock()
    defer ma.mutex.Unlock()
    
    ma.blocks = ma.blocks[:0]
    ma.current = nil
    ma.offset = 0
}

// String Interning for Memory Efficiency
type StringInterner struct {
    strings map[string]string
    mutex   sync.RWMutex
}

func NewStringInterner() *StringInterner {
    return &StringInterner{
        strings: make(map[string]string),
    }
}

func (si *StringInterner) Intern(s string) string {
    si.mutex.RLock()
    if interned, exists := si.strings[s]; exists {
        si.mutex.RUnlock()
        return interned
    }
    si.mutex.RUnlock()
    
    si.mutex.Lock()
    defer si.mutex.Unlock()
    
    // Double-check after acquiring write lock
    if interned, exists := si.strings[s]; exists {
        return interned
    }
    
    si.strings[s] = s
    return s
}

func (si *StringInterner) GetStats() map[string]interface{} {
    si.mutex.RLock()
    defer si.mutex.RUnlock()
    
    return map[string]interface{}{
        "unique_strings": len(si.strings),
        "total_bytes":    si.calculateTotalBytes(),
    }
}

func (si *StringInterner) calculateTotalBytes() int {
    total := 0
    for s := range si.strings {
        total += len(s)
    }
    return total
}

// Object Pool with Generics (Go 1.18+)
type ObjectPool[T any] struct {
    pool    sync.Pool
    factory func() T
    reset   func(T)
}

func NewObjectPool[T any](factory func() T, reset func(T)) *ObjectPool[T] {
    return &ObjectPool[T]{
        pool: sync.Pool{
            New: func() interface{} {
                return factory()
            },
        },
        factory: factory,
        reset:   reset,
    }
}

func (op *ObjectPool[T]) Get() T {
    obj := op.pool.Get().(T)
    if op.reset != nil {
        op.reset(obj)
    }
    return obj
}

func (op *ObjectPool[T]) Put(obj T) {
    op.pool.Put(obj)
}

// Memory-mapped File
type MemoryMappedFile struct {
    data     []byte
    size     int64
    mutex    sync.RWMutex
}

func NewMemoryMappedFile(size int64) *MemoryMappedFile {
    return &MemoryMappedFile{
        data: make([]byte, size),
        size: size,
    }
}

func (mmf *MemoryMappedFile) Write(offset int64, data []byte) error {
    if offset+int64(len(data)) > mmf.size {
        return fmt.Errorf("write would exceed file size")
    }
    
    mmf.mutex.Lock()
    defer mmf.mutex.Unlock()
    
    copy(mmf.data[offset:], data)
    return nil
}

func (mmf *MemoryMappedFile) Read(offset int64, length int) ([]byte, error) {
    if offset+int64(length) > mmf.size {
        return nil, fmt.Errorf("read would exceed file size")
    }
    
    mmf.mutex.RLock()
    defer mmf.mutex.RUnlock()
    
    data := make([]byte, length)
    copy(data, mmf.data[offset:offset+int64(length)])
    return data, nil
}

// Lock-free Ring Buffer
type LockFreeRingBuffer struct {
    data     []interface{}
    head     int64
    tail     int64
    capacity int64
    mask     int64
}

func NewLockFreeRingBuffer(capacity int64) *LockFreeRingBuffer {
    // Ensure capacity is power of 2
    actualCapacity := int64(1)
    for actualCapacity < capacity {
        actualCapacity <<= 1
    }
    
    return &LockFreeRingBuffer{
        data:     make([]interface{}, actualCapacity),
        capacity: actualCapacity,
        mask:     actualCapacity - 1,
    }
}

func (rb *LockFreeRingBuffer) Push(item interface{}) bool {
    head := atomic.LoadInt64(&rb.head)
    tail := atomic.LoadInt64(&rb.tail)
    
    if (head+1)&rb.mask == tail&rb.mask {
        return false // Buffer full
    }
    
    rb.data[head&rb.mask] = item
    atomic.StoreInt64(&rb.head, head+1)
    return true
}

func (rb *LockFreeRingBuffer) Pop() (interface{}, bool) {
    tail := atomic.LoadInt64(&rb.tail)
    head := atomic.LoadInt64(&rb.head)
    
    if tail == head {
        return nil, false // Buffer empty
    }
    
    item := rb.data[tail&rb.mask]
    atomic.StoreInt64(&rb.tail, tail+1)
    return item, true
}

func (rb *LockFreeRingBuffer) Size() int64 {
    head := atomic.LoadInt64(&rb.head)
    tail := atomic.LoadInt64(&rb.tail)
    return head - tail
}

// Memory Profiler
type MemoryProfiler struct {
    samples []MemorySample
    mutex   sync.RWMutex
}

type MemorySample struct {
    Timestamp time.Time
    Alloc     uint64
    TotalAlloc uint64
    Sys       uint64
    NumGC     uint32
}

func NewMemoryProfiler() *MemoryProfiler {
    return &MemoryProfiler{
        samples: make([]MemorySample, 0),
    }
}

func (mp *MemoryProfiler) Sample() {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    mp.mutex.Lock()
    defer mp.mutex.Unlock()
    
    sample := MemorySample{
        Timestamp:  time.Now(),
        Alloc:      m.Alloc,
        TotalAlloc: m.TotalAlloc,
        Sys:        m.Sys,
        NumGC:      m.NumGC,
    }
    
    mp.samples = append(mp.samples, sample)
    
    // Keep only last 1000 samples
    if len(mp.samples) > 1000 {
        mp.samples = mp.samples[1:]
    }
}

func (mp *MemoryProfiler) GetStats() map[string]interface{} {
    mp.mutex.RLock()
    defer mp.mutex.RUnlock()
    
    if len(mp.samples) == 0 {
        return map[string]interface{}{}
    }
    
    latest := mp.samples[len(mp.samples)-1]
    
    return map[string]interface{}{
        "current_alloc":    latest.Alloc,
        "total_alloc":      latest.TotalAlloc,
        "system_memory":    latest.Sys,
        "gc_cycles":        latest.NumGC,
        "sample_count":     len(mp.samples),
    }
}

// Example usage
func main() {
    // Memory arena example
    arena := NewMemoryArena(1024)
    
    data1 := arena.Allocate(100)
    data2 := arena.Allocate(200)
    
    copy(data1, []byte("Hello"))
    copy(data2, []byte("World"))
    
    fmt.Printf("Arena data1: %s\n", string(data1))
    fmt.Printf("Arena data2: %s\n", string(data2))
    
    // String interning example
    interner := NewStringInterner()
    
    s1 := interner.Intern("hello")
    s2 := interner.Intern("hello")
    s3 := interner.Intern("world")
    
    fmt.Printf("Same string: %t\n", s1 == s2)
    fmt.Printf("Different string: %t\n", s1 == s3)
    
    stats := interner.GetStats()
    fmt.Printf("String interner stats: %+v\n", stats)
    
    // Object pool example
    pool := NewObjectPool(
        func() *PooledObject {
            return &PooledObject{
                Data:    make([]byte, 1024),
                ID:      0,
                Created: time.Now(),
            }
        },
        func(obj *PooledObject) {
            obj.ID = 0
            for i := range obj.Data {
                obj.Data[i] = 0
            }
        },
    )
    
    obj := pool.Get()
    obj.ID = 123
    copy(obj.Data, []byte("Pooled object"))
    
    pool.Put(obj)
    
    // Memory-mapped file example
    mmf := NewMemoryMappedFile(1024)
    
    data := []byte("Memory mapped data")
    mmf.Write(0, data)
    
    readData, err := mmf.Read(0, len(data))
    if err == nil {
        fmt.Printf("Memory mapped data: %s\n", string(readData))
    }
    
    // Lock-free ring buffer example
    rb := NewLockFreeRingBuffer(1024)
    
    for i := 0; i < 10; i++ {
        rb.Push(fmt.Sprintf("item_%d", i))
    }
    
    for {
        if item, ok := rb.Pop(); ok {
            fmt.Printf("Popped: %v\n", item)
        } else {
            break
        }
    }
    
    // Memory profiler example
    profiler := NewMemoryProfiler()
    
    go func() {
        for {
            profiler.Sample()
            time.Sleep(1 * time.Second)
        }
    }()
    
    time.Sleep(5 * time.Second)
    
    stats = profiler.GetStats()
    fmt.Printf("Memory profiler stats: %+v\n", stats)
}
```

---

## 🎯 **4. Advanced Error Handling**

### **Error Wrapping and Context**

```go
package main

import (
    "context"
    "errors"
    "fmt"
    "runtime"
    "time"
)

// Custom Error Types
type BusinessError struct {
    Code    string
    Message string
    Cause   error
}

func (be *BusinessError) Error() string {
    if be.Cause != nil {
        return fmt.Sprintf("%s: %s (caused by: %v)", be.Code, be.Message, be.Cause)
    }
    return fmt.Sprintf("%s: %s", be.Code, be.Message)
}

func (be *BusinessError) Unwrap() error {
    return be.Cause
}

// Error Context
type ErrorContext struct {
    Timestamp time.Time
    Stack     string
    Context   map[string]interface{}
}

func NewErrorContext() *ErrorContext {
    stack := make([]byte, 4096)
    length := runtime.Stack(stack, false)
    
    return &ErrorContext{
        Timestamp: time.Now(),
        Stack:     string(stack[:length]),
        Context:   make(map[string]interface{}),
    }
}

func (ec *ErrorContext) Add(key string, value interface{}) {
    ec.Context[key] = value
}

// Error Handler
type ErrorHandler struct {
    handlers map[string]func(error) error
    mutex    sync.RWMutex
}

func NewErrorHandler() *ErrorHandler {
    return &ErrorHandler{
        handlers: make(map[string]func(error) error),
    }
}

func (eh *ErrorHandler) RegisterHandler(errorType string, handler func(error) error) {
    eh.mutex.Lock()
    defer eh.mutex.Unlock()
    
    eh.handlers[errorType] = handler
}

func (eh *ErrorHandler) Handle(err error) error {
    if err == nil {
        return nil
    }
    
    eh.mutex.RLock()
    defer eh.mutex.RUnlock()
    
    // Try to find a handler for this error type
    errorType := fmt.Sprintf("%T", err)
    if handler, exists := eh.handlers[errorType]; exists {
        return handler(err)
    }
    
    // Default handling
    return fmt.Errorf("unhandled error: %w", err)
}

// Retry Mechanism
type RetryConfig struct {
    MaxAttempts int
    InitialDelay time.Duration
    MaxDelay     time.Duration
    Multiplier   float64
    Jitter       bool
}

func Retry(config RetryConfig, fn func() error) error {
    var lastErr error
    delay := config.InitialDelay
    
    for attempt := 0; attempt < config.MaxAttempts; attempt++ {
        if err := fn(); err == nil {
            return nil
        } else {
            lastErr = err
        }
        
        if attempt < config.MaxAttempts-1 {
            if config.Jitter {
                // Add random jitter
                jitter := time.Duration(float64(delay) * 0.1 * (0.5 - rand.Float64()))
                delay += jitter
            }
            
            time.Sleep(delay)
            delay = time.Duration(float64(delay) * config.Multiplier)
            if delay > config.MaxDelay {
                delay = config.MaxDelay
            }
        }
    }
    
    return lastErr
}

// Circuit Breaker
type CircuitBreaker struct {
    name          string
    maxRequests   uint32
    interval      time.Duration
    timeout       time.Duration
    readyToTrip   func(counts Counts) bool
    onStateChange func(name string, from State, to State)
    
    mutex      sync.Mutex
    state      State
    generation uint64
    counts     Counts
    expiry     time.Time
}

type State int

const (
    StateClosed State = iota
    StateHalfOpen
    StateOpen
)

type Counts struct {
    Requests             uint32
    TotalSuccesses       uint32
    TotalFailures        uint32
    ConsecutiveSuccesses uint32
    ConsecutiveFailures  uint32
}

func NewCircuitBreaker(name string, maxRequests uint32, interval, timeout time.Duration) *CircuitBreaker {
    cb := &CircuitBreaker{
        name:        name,
        maxRequests: maxRequests,
        interval:    interval,
        timeout:     timeout,
        readyToTrip: func(counts Counts) bool {
            return counts.ConsecutiveFailures >= 5
        },
        onStateChange: func(name string, from State, to State) {
            fmt.Printf("Circuit breaker %s changed from %s to %s\n", name, from, to)
        },
    }
    
    cb.toNewGeneration(time.Now())
    return cb
}

func (cb *CircuitBreaker) Execute(req func() (interface{}, error)) (interface{}, error) {
    generation, err := cb.beforeRequest()
    if err != nil {
        return nil, err
    }
    
    defer func() {
        e := recover()
        if e != nil {
            cb.afterRequest(generation, false)
            panic(e)
        }
    }()
    
    result, err := req()
    cb.afterRequest(generation, err == nil)
    return result, err
}

func (cb *CircuitBreaker) beforeRequest() (uint64, error) {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    now := time.Now()
    state, generation := cb.currentState(now)
    
    if state == StateOpen {
        return generation, fmt.Errorf("circuit breaker is open")
    } else if state == StateHalfOpen && cb.counts.Requests >= cb.maxRequests {
        return generation, fmt.Errorf("circuit breaker is half-open and max requests reached")
    }
    
    cb.counts.onRequest()
    return generation, nil
}

func (cb *CircuitBreaker) afterRequest(before uint64, success bool) {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    now := time.Now()
    state, generation := cb.currentState(now)
    if generation != before {
        return
    }
    
    if success {
        cb.onSuccess(state, now)
    } else {
        cb.onFailure(state, now)
    }
}

func (cb *CircuitBreaker) currentState(now time.Time) (State, uint64) {
    if cb.state == StateOpen && cb.expiry.Before(now) {
        cb.setState(StateHalfOpen, now)
        return StateHalfOpen, cb.generation
    }
    
    return cb.state, cb.generation
}

func (cb *CircuitBreaker) setState(state State, now time.Time) {
    if cb.state == state {
        return
    }
    
    prev := cb.state
    cb.state = state
    
    cb.toNewGeneration(now)
    
    if cb.onStateChange != nil {
        cb.onStateChange(cb.name, prev, state)
    }
}

func (cb *CircuitBreaker) toNewGeneration(now time.Time) {
    cb.generation++
    cb.counts = Counts{}
    cb.expiry = now.Add(cb.interval)
}

func (cb *CircuitBreaker) onSuccess(state State, now time.Time) {
    switch state {
    case StateClosed:
        cb.counts.onSuccess()
    case StateHalfOpen:
        cb.counts.onSuccess()
        if cb.counts.ConsecutiveSuccesses >= cb.maxRequests {
            cb.setState(StateClosed, now)
        }
    }
}

func (cb *CircuitBreaker) onFailure(state State, now time.Time) {
    switch state {
    case StateClosed:
        cb.counts.onFailure()
        if cb.readyToTrip(cb.counts) {
            cb.setState(StateOpen, now)
        }
    case StateHalfOpen:
        cb.setState(StateOpen, now)
    }
}

func (c *Counts) onRequest() {
    c.Requests++
}

func (c *Counts) onSuccess() {
    c.TotalSuccesses++
    c.ConsecutiveSuccesses++
    c.ConsecutiveFailures = 0
}

func (c *Counts) onFailure() {
    c.TotalFailures++
    c.ConsecutiveFailures++
    c.ConsecutiveSuccesses = 0
}

func (s State) String() string {
    switch s {
    case StateClosed:
        return "closed"
    case StateHalfOpen:
        return "half-open"
    case StateOpen:
        return "open"
    default:
        return "unknown"
    }
}

// Example usage
func main() {
    // Error handling example
    handler := NewErrorHandler()
    
    handler.RegisterHandler("*BusinessError", func(err error) error {
        if be, ok := err.(*BusinessError); ok {
            fmt.Printf("Handling business error: %s\n", be.Code)
            return nil
        }
        return err
    })
    
    // Create a business error
    businessErr := &BusinessError{
        Code:    "VALIDATION_ERROR",
        Message: "Invalid input data",
        Cause:   errors.New("field 'email' is required"),
    }
    
    if err := handler.Handle(businessErr); err != nil {
        fmt.Printf("Error not handled: %v\n", err)
    }
    
    // Retry mechanism example
    retryConfig := RetryConfig{
        MaxAttempts:  3,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     1 * time.Second,
        Multiplier:   2.0,
        Jitter:       true,
    }
    
    err := Retry(retryConfig, func() error {
        // Simulate flaky operation
        if time.Now().UnixNano()%3 == 0 {
            return errors.New("temporary failure")
        }
        return nil
    })
    
    if err != nil {
        fmt.Printf("Retry failed: %v\n", err)
    } else {
        fmt.Println("Retry succeeded")
    }
    
    // Circuit breaker example
    cb := NewCircuitBreaker("test-service", 3, 30*time.Second, 5*time.Second)
    
    for i := 0; i < 10; i++ {
        result, err := cb.Execute(func() (interface{}, error) {
            // Simulate service call
            time.Sleep(100 * time.Millisecond)
            
            // Simulate occasional failures
            if time.Now().UnixNano()%3 == 0 {
                return nil, errors.New("service unavailable")
            }
            
            return "success", nil
        })
        
        if err != nil {
            fmt.Printf("Request %d failed: %v\n", i+1, err)
        } else {
            fmt.Printf("Request %d succeeded: %v\n", i+1, result)
        }
        
        time.Sleep(1 * time.Second)
    }
}
```

---

## 🎯 **Key Takeaways**

### **1. Go Runtime Internals**
- **Goroutine scheduler**: M:N model with work stealing
- **Memory management**: GC tuning and object pooling
- **Lock-free data structures**: High-performance concurrent access
- **Context management**: Timeout and cancellation

### **2. Advanced Concurrency**
- **Work stealing**: Efficient task distribution
- **Pipeline patterns**: Sequential processing stages
- **Fan-out fan-in**: Parallel processing with aggregation
- **Worker pools**: Controlled concurrency

### **3. Memory Management**
- **Memory arenas**: Efficient allocation patterns
- **String interning**: Reduce memory usage
- **Object pooling**: Reuse objects to reduce GC pressure
- **Memory profiling**: Monitor and optimize memory usage

### **4. Error Handling**
- **Error wrapping**: Preserve error context
- **Retry mechanisms**: Handle transient failures
- **Circuit breakers**: Prevent cascade failures
- **Error handlers**: Centralized error processing

### **5. Best Practices**
- **Use sync.Pool**: For frequently allocated objects
- **Avoid memory leaks**: Proper cleanup and context cancellation
- **Profile regularly**: Monitor performance and memory usage
- **Handle errors gracefully**: Implement proper error handling strategies

---

**🎉 This comprehensive guide provides deep insights into advanced Go concepts with practical implementations! 🚀**
