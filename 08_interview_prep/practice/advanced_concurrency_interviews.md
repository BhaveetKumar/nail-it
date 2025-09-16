# Advanced Concurrency Interviews

## Table of Contents
- [Introduction](#introduction)
- [Goroutine Management](#goroutine-management)
- [Channel Patterns](#channel-patterns)
- [Synchronization Primitives](#synchronization-primitives)
- [Lock-Free Programming](#lock-free-programming)
- [Concurrent Data Structures](#concurrent-data-structures)
- [Race Condition Prevention](#race-condition-prevention)
- [Performance Optimization](#performance-optimization)

## Introduction

Advanced concurrency interviews test your understanding of complex concurrent programming concepts, synchronization mechanisms, and performance optimization techniques.

## Goroutine Management

### Goroutine Pool with Dynamic Scaling

```go
// Dynamic goroutine pool
type DynamicPool struct {
    minWorkers    int
    maxWorkers    int
    currentWorkers int
    jobQueue      chan Job
    workerQueue   chan chan Job
    quit          chan bool
    wg            sync.WaitGroup
    mutex         sync.RWMutex
    metrics       *PoolMetrics
}

type Job struct {
    ID       int
    Data     interface{}
    Process  func(interface{}) (interface{}, error)
    Result   chan JobResult
    Error    chan error
}

type JobResult struct {
    JobID  int
    Result interface{}
}

type PoolMetrics struct {
    JobsProcessed    int64
    JobsFailed       int64
    ActiveWorkers    int64
    QueueLength      int64
    AverageWaitTime  time.Duration
}

func NewDynamicPool(minWorkers, maxWorkers int, queueSize int) *DynamicPool {
    return &DynamicPool{
        minWorkers:    minWorkers,
        maxWorkers:    maxWorkers,
        currentWorkers: minWorkers,
        jobQueue:      make(chan Job, queueSize),
        workerQueue:   make(chan chan Job, maxWorkers),
        quit:          make(chan bool),
        metrics:       &PoolMetrics{},
    }
}

func (dp *DynamicPool) Start() {
    // Start minimum workers
    for i := 0; i < dp.minWorkers; i++ {
        dp.startWorker()
    }
    
    // Start pool manager
    go dp.managePool()
}

func (dp *DynamicPool) startWorker() {
    dp.mutex.Lock()
    dp.currentWorkers++
    dp.mutex.Unlock()
    
    dp.wg.Add(1)
    go func() {
        defer dp.wg.Done()
        
        workerQueue := make(chan Job)
        dp.workerQueue <- workerQueue
        
        for {
            select {
            case job := <-workerQueue:
                dp.processJob(job)
            case <-dp.quit:
                return
            }
        }
    }()
}

func (dp *DynamicPool) processJob(job Job) {
    start := time.Now()
    
    result, err := job.Process(job.Data)
    if err != nil {
        atomic.AddInt64(&dp.metrics.JobsFailed, 1)
        job.Error <- err
    } else {
        atomic.AddInt64(&dp.metrics.JobsProcessed, 1)
        job.Result <- JobResult{JobID: job.ID, Result: result}
    }
    
    // Update metrics
    waitTime := time.Since(start)
    dp.updateAverageWaitTime(waitTime)
}

func (dp *DynamicPool) managePool() {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case job := <-dp.jobQueue:
            dp.dispatchJob(job)
        case <-ticker.C:
            dp.adjustPoolSize()
        case <-dp.quit:
            return
        }
    }
}

func (dp *DynamicPool) dispatchJob(job Job) {
    select {
    case workerQueue := <-dp.workerQueue:
        workerQueue <- job
    default:
        // No available workers, queue the job
        dp.jobQueue <- job
    }
}

func (dp *DynamicPool) adjustPoolSize() {
    dp.mutex.Lock()
    defer dp.mutex.Unlock()
    
    queueLength := len(dp.jobQueue)
    atomic.StoreInt64(&dp.metrics.QueueLength, int64(queueLength))
    
    // Scale up if queue is getting full
    if queueLength > dp.currentWorkers*2 && dp.currentWorkers < dp.maxWorkers {
        dp.startWorker()
    }
    
    // Scale down if queue is empty and we have more than min workers
    if queueLength == 0 && dp.currentWorkers > dp.minWorkers {
        dp.currentWorkers--
        // Send quit signal to a worker
        select {
        case <-dp.workerQueue:
        default:
        }
    }
}

func (dp *DynamicPool) Submit(job Job) {
    dp.jobQueue <- job
}

func (dp *DynamicPool) Stop() {
    close(dp.quit)
    dp.wg.Wait()
    close(dp.jobQueue)
    close(dp.workerQueue)
}

func (dp *DynamicPool) GetMetrics() PoolMetrics {
    dp.mutex.RLock()
    defer dp.mutex.RUnlock()
    
    return PoolMetrics{
        JobsProcessed:   atomic.LoadInt64(&dp.metrics.JobsProcessed),
        JobsFailed:      atomic.LoadInt64(&dp.metrics.JobsFailed),
        ActiveWorkers:   int64(dp.currentWorkers),
        QueueLength:     atomic.LoadInt64(&dp.metrics.QueueLength),
        AverageWaitTime: dp.metrics.AverageWaitTime,
    }
}

func (dp *DynamicPool) updateAverageWaitTime(waitTime time.Duration) {
    // Simple moving average
    dp.metrics.AverageWaitTime = (dp.metrics.AverageWaitTime + waitTime) / 2
}
```

### Goroutine Leak Prevention

```go
// Goroutine leak prevention utilities
type GoroutineManager struct {
    goroutines map[string]chan bool
    mutex      sync.RWMutex
    wg         sync.WaitGroup
}

func NewGoroutineManager() *GoroutineManager {
    return &GoroutineManager{
        goroutines: make(map[string]chan bool),
    }
}

func (gm *GoroutineManager) StartGoroutine(name string, fn func(<-chan bool)) {
    gm.mutex.Lock()
    defer gm.mutex.Unlock()
    
    quit := make(chan bool)
    gm.goroutines[name] = quit
    
    gm.wg.Add(1)
    go func() {
        defer gm.wg.Done()
        fn(quit)
    }()
}

func (gm *GoroutineManager) StopGoroutine(name string) {
    gm.mutex.Lock()
    defer gm.mutex.Unlock()
    
    if quit, exists := gm.goroutines[name]; exists {
        close(quit)
        delete(gm.goroutines, name)
    }
}

func (gm *GoroutineManager) StopAll() {
    gm.mutex.Lock()
    defer gm.mutex.Unlock()
    
    for name, quit := range gm.goroutines {
        close(quit)
        delete(gm.goroutines, name)
    }
    
    gm.wg.Wait()
}

func (gm *GoroutineManager) GetActiveGoroutines() []string {
    gm.mutex.RLock()
    defer gm.mutex.RUnlock()
    
    names := make([]string, 0, len(gm.goroutines))
    for name := range gm.goroutines {
        names = append(names, name)
    }
    return names
}

// Context-based goroutine management
func RunWithContext(ctx context.Context, fn func(context.Context)) {
    go func() {
        defer func() {
            if r := recover(); r != nil {
                fmt.Printf("Goroutine panicked: %v\n", r)
            }
        }()
        
        fn(ctx)
    }()
}

func RunWithTimeout(timeout time.Duration, fn func()) error {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()
    
    done := make(chan error, 1)
    
    go func() {
        defer func() {
            if r := recover(); r != nil {
                done <- fmt.Errorf("goroutine panicked: %v", r)
            }
        }()
        
        fn()
        done <- nil
    }()
    
    select {
    case err := <-done:
        return err
    case <-ctx.Done():
        return ctx.Err()
    }
}
```

## Channel Patterns

### Fan-In/Fan-Out Pattern

```go
// Fan-out pattern: distribute work to multiple workers
func FanOut(input <-chan Job, numWorkers int) []<-chan Job {
    outputs := make([]<-chan Job, numWorkers)
    
    for i := 0; i < numWorkers; i++ {
        output := make(chan Job)
        outputs[i] = output
        
        go func() {
            defer close(output)
            for job := range input {
                output <- job
            }
        }()
    }
    
    return outputs
}

// Fan-in pattern: merge multiple channels into one
func FanIn(inputs ...<-chan Job) <-chan Job {
    output := make(chan Job)
    var wg sync.WaitGroup
    
    for _, input := range inputs {
        wg.Add(1)
        go func(input <-chan Job) {
            defer wg.Done()
            for job := range input {
                output <- job
            }
        }(input)
    }
    
    go func() {
        wg.Wait()
        close(output)
    }()
    
    return output
}

// Pipeline pattern with fan-in/fan-out
type Pipeline struct {
    stages []PipelineStage
}

type PipelineStage struct {
    Name     string
    Process  func(interface{}) interface{}
    FanOut   int
}

func NewPipeline(stages ...PipelineStage) *Pipeline {
    return &Pipeline{stages: stages}
}

func (p *Pipeline) Process(input <-chan interface{}) <-chan interface{} {
    current := input
    
    for _, stage := range p.stages {
        if stage.FanOut > 1 {
            // Fan-out
            outputs := make([]<-chan interface{}, stage.FanOut)
            for i := 0; i < stage.FanOut; i++ {
                output := make(chan interface{})
                outputs[i] = output
                
                go func() {
                    defer close(output)
                    for data := range current {
                        result := stage.Process(data)
                        output <- result
                    }
                }()
            }
            
            // Fan-in
            current = FanIn(outputs...)
        } else {
            // Single worker
            output := make(chan interface{})
            go func() {
                defer close(output)
                for data := range current {
                    result := stage.Process(data)
                    output <- result
                }
            }()
            current = output
        }
    }
    
    return current
}
```

### Channel Multiplexing

```go
// Channel multiplexer
type Multiplexer struct {
    inputs  []<-chan interface{}
    output  chan interface{}
    quit    chan bool
    wg      sync.WaitGroup
}

func NewMultiplexer(inputs []<-chan interface{}) *Multiplexer {
    return &Multiplexer{
        inputs: inputs,
        output: make(chan interface{}),
        quit:   make(chan bool),
    }
}

func (m *Multiplexer) Start() {
    for i, input := range m.inputs {
        m.wg.Add(1)
        go func(input <-chan interface{}, id int) {
            defer m.wg.Done()
            
            for {
                select {
                case data := <-input:
                    select {
                    case m.output <- data:
                    case <-m.quit:
                        return
                    }
                case <-m.quit:
                    return
                }
            }
        }(input, i)
    }
    
    go func() {
        m.wg.Wait()
        close(m.output)
    }()
}

func (m *Multiplexer) GetOutput() <-chan interface{} {
    return m.output
}

func (m *Multiplexer) Stop() {
    close(m.quit)
}

// Channel demultiplexer
type Demultiplexer struct {
    input   <-chan interface{}
    outputs []chan interface{}
    quit    chan bool
    wg      sync.WaitGroup
}

func NewDemultiplexer(input <-chan interface{}, numOutputs int) *Demultiplexer {
    outputs := make([]chan interface{}, numOutputs)
    for i := 0; i < numOutputs; i++ {
        outputs[i] = make(chan interface{})
    }
    
    return &Demultiplexer{
        input:   input,
        outputs: outputs,
        quit:    make(chan bool),
    }
}

func (d *Demultiplexer) Start() {
    d.wg.Add(1)
    go func() {
        defer d.wg.Done()
        
        for {
            select {
            case data := <-d.input:
                // Round-robin distribution
                for _, output := range d.outputs {
                    select {
                    case output <- data:
                    case <-d.quit:
                        return
                    }
                }
            case <-d.quit:
                return
            }
        }
    }()
}

func (d *Demultiplexer) GetOutputs() []<-chan interface{} {
    result := make([]<-chan interface{}, len(d.outputs))
    for i, output := range d.outputs {
        result[i] = output
    }
    return result
}

func (d *Demultiplexer) Stop() {
    close(d.quit)
    d.wg.Wait()
    
    for _, output := range d.outputs {
        close(output)
    }
}
```

### Channel Buffering Strategies

```go
// Adaptive channel buffering
type AdaptiveBuffer struct {
    channel    chan interface{}
    maxSize    int
    currentSize int
    mutex      sync.RWMutex
    metrics    *BufferMetrics
}

type BufferMetrics struct {
    TotalMessages   int64
    DroppedMessages int64
    AverageLatency  time.Duration
}

func NewAdaptiveBuffer(initialSize, maxSize int) *AdaptiveBuffer {
    return &AdaptiveBuffer{
        channel:    make(chan interface{}, initialSize),
        maxSize:    maxSize,
        currentSize: initialSize,
        metrics:    &BufferMetrics{},
    }
}

func (ab *AdaptiveBuffer) Send(data interface{}) error {
    ab.mutex.RLock()
    currentSize := ab.currentSize
    ab.mutex.RUnlock()
    
    select {
    case ab.channel <- data:
        atomic.AddInt64(&ab.metrics.TotalMessages, 1)
        return nil
    default:
        // Buffer full, try to expand
        if ab.expandBuffer() {
            select {
            case ab.channel <- data:
                atomic.AddInt64(&ab.metrics.TotalMessages, 1)
                return nil
            default:
                atomic.AddInt64(&ab.metrics.DroppedMessages, 1)
                return fmt.Errorf("buffer full, message dropped")
            }
        }
        atomic.AddInt64(&ab.metrics.DroppedMessages, 1)
        return fmt.Errorf("buffer full, message dropped")
    }
}

func (ab *AdaptiveBuffer) expandBuffer() bool {
    ab.mutex.Lock()
    defer ab.mutex.Unlock()
    
    if ab.currentSize >= ab.maxSize {
        return false
    }
    
    // Expand buffer by 50%
    newSize := ab.currentSize * 3 / 2
    if newSize > ab.maxSize {
        newSize = ab.maxSize
    }
    
    // Create new channel with larger buffer
    newChannel := make(chan interface{}, newSize)
    
    // Transfer existing messages
    for {
        select {
        case data := <-ab.channel:
            select {
            case newChannel <- data:
            default:
                // New channel is also full, keep old one
                return false
            }
        default:
            // No more messages to transfer
            ab.channel = newChannel
            ab.currentSize = newSize
            return true
        }
    }
}

func (ab *AdaptiveBuffer) Receive() <-chan interface{} {
    return ab.channel
}

func (ab *AdaptiveBuffer) GetMetrics() BufferMetrics {
    return BufferMetrics{
        TotalMessages:   atomic.LoadInt64(&ab.metrics.TotalMessages),
        DroppedMessages: atomic.LoadInt64(&ab.metrics.DroppedMessages),
        AverageLatency:  ab.metrics.AverageLatency,
    }
}
```

## Synchronization Primitives

### Advanced Mutex Patterns

```go
// Read-write mutex with priority
type PriorityRWMutex struct {
    readCount    int
    writeCount   int
    readMutex    sync.Mutex
    writeMutex   sync.Mutex
    readCond     *sync.Cond
    writeCond    *sync.Cond
    priority     bool // true = readers have priority
}

func NewPriorityRWMutex(priority bool) *PriorityRWMutex {
    rw := &PriorityRWMutex{priority: priority}
    rw.readCond = sync.NewCond(&rw.readMutex)
    rw.writeCond = sync.NewCond(&rw.writeMutex)
    return rw
}

func (rw *PriorityRWMutex) RLock() {
    rw.readMutex.Lock()
    defer rw.readMutex.Unlock()
    
    // Wait for writers to finish
    for rw.writeCount > 0 {
        rw.readCond.Wait()
    }
    
    rw.readCount++
}

func (rw *PriorityRWMutex) RUnlock() {
    rw.readMutex.Lock()
    defer rw.readMutex.Unlock()
    
    rw.readCount--
    if rw.readCount == 0 {
        rw.writeCond.Signal()
    }
}

func (rw *PriorityRWMutex) Lock() {
    rw.writeMutex.Lock()
    defer rw.writeMutex.Unlock()
    
    rw.writeCount++
    
    // Wait for readers to finish
    for rw.readCount > 0 {
        rw.writeCond.Wait()
    }
}

func (rw *PriorityRWMutex) Unlock() {
    rw.writeMutex.Lock()
    defer rw.writeMutex.Unlock()
    
    rw.writeCount--
    if rw.writeCount == 0 {
        rw.readCond.Broadcast()
    }
}

// Recursive mutex
type RecursiveMutex struct {
    mutex    sync.Mutex
    owner    int64
    count    int
    cond     *sync.Cond
}

func NewRecursiveMutex() *RecursiveMutex {
    rm := &RecursiveMutex{}
    rm.cond = sync.NewCond(&rm.mutex)
    return rm
}

func (rm *RecursiveMutex) Lock() {
    goroutineID := getGoroutineID()
    
    rm.mutex.Lock()
    defer rm.mutex.Unlock()
    
    for rm.count > 0 && rm.owner != goroutineID {
        rm.cond.Wait()
    }
    
    rm.owner = goroutineID
    rm.count++
}

func (rm *RecursiveMutex) Unlock() {
    rm.mutex.Lock()
    defer rm.mutex.Unlock()
    
    if rm.count > 0 && rm.owner == getGoroutineID() {
        rm.count--
        if rm.count == 0 {
            rm.owner = 0
            rm.cond.Signal()
        }
    }
}

func getGoroutineID() int64 {
    return runtime.NumGoroutine() // Simplified for example
}
```

### Condition Variables

```go
// Advanced condition variable with timeout
type TimeoutCondition struct {
    mutex    sync.Mutex
    cond     *sync.Cond
    signaled bool
}

func NewTimeoutCondition() *TimeoutCondition {
    tc := &TimeoutCondition{}
    tc.cond = sync.NewCond(&tc.mutex)
    return tc
}

func (tc *TimeoutCondition) Wait() {
    tc.mutex.Lock()
    defer tc.mutex.Unlock()
    
    for !tc.signaled {
        tc.cond.Wait()
    }
    tc.signaled = false
}

func (tc *TimeoutCondition) WaitWithTimeout(timeout time.Duration) bool {
    tc.mutex.Lock()
    defer tc.mutex.Unlock()
    
    if tc.signaled {
        tc.signaled = false
        return true
    }
    
    done := make(chan bool, 1)
    go func() {
        tc.cond.Wait()
        done <- true
    }()
    
    select {
    case <-done:
        tc.signaled = false
        return true
    case <-time.After(timeout):
        return false
    }
}

func (tc *TimeoutCondition) Signal() {
    tc.mutex.Lock()
    defer tc.mutex.Unlock()
    
    tc.signaled = true
    tc.cond.Signal()
}

func (tc *TimeoutCondition) Broadcast() {
    tc.mutex.Lock()
    defer tc.mutex.Unlock()
    
    tc.signaled = true
    tc.cond.Broadcast()
}
```

## Lock-Free Programming

### Lock-Free Stack

```go
// Lock-free stack using atomic operations
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
    newNode := &stackNode{value: value}
    
    for {
        head := atomic.LoadPointer(&s.head)
        newNode.next = head
        
        if atomic.CompareAndSwapPointer(&s.head, head, unsafe.Pointer(newNode)) {
            break
        }
    }
}

func (s *LockFreeStack) Pop() interface{} {
    for {
        head := atomic.LoadPointer(&s.head)
        if head == nil {
            return nil
        }
        
        node := (*stackNode)(head)
        next := atomic.LoadPointer(&node.next)
        
        if atomic.CompareAndSwapPointer(&s.head, head, next) {
            return node.value
        }
    }
}

func (s *LockFreeStack) IsEmpty() bool {
    return atomic.LoadPointer(&s.head) == nil
}
```

### Lock-Free Queue

```go
// Lock-free queue using atomic operations
type LockFreeQueue struct {
    head unsafe.Pointer
    tail unsafe.Pointer
}

type queueNode struct {
    value interface{}
    next  unsafe.Pointer
}

func NewLockFreeQueue() *LockFreeQueue {
    node := unsafe.Pointer(&queueNode{})
    return &LockFreeQueue{
        head: node,
        tail: node,
    }
}

func (q *LockFreeQueue) Enqueue(value interface{}) {
    newNode := &queueNode{value: value}
    
    for {
        tail := atomic.LoadPointer(&q.tail)
        tailNode := (*queueNode)(tail)
        
        next := atomic.LoadPointer(&tailNode.next)
        
        if next == nil {
            if atomic.CompareAndSwapPointer(&tailNode.next, nil, unsafe.Pointer(newNode)) {
                break
            }
        } else {
            atomic.CompareAndSwapPointer(&q.tail, tail, next)
        }
    }
    
    atomic.CompareAndSwapPointer(&q.tail, atomic.LoadPointer(&q.tail), unsafe.Pointer(newNode))
}

func (q *LockFreeQueue) Dequeue() interface{} {
    for {
        head := atomic.LoadPointer(&q.head)
        headNode := (*queueNode)(head)
        
        tail := atomic.LoadPointer(&q.tail)
        tailNode := (*queueNode)(tail)
        
        next := atomic.LoadPointer(&headNode.next)
        
        if head == tail {
            if next == nil {
                return nil
            }
            atomic.CompareAndSwapPointer(&q.tail, tail, next)
        } else {
            if next == nil {
                continue
            }
            
            value := (*queueNode)(next).value
            
            if atomic.CompareAndSwapPointer(&q.head, head, next) {
                return value
            }
        }
    }
}

func (q *LockFreeQueue) IsEmpty() bool {
    head := atomic.LoadPointer(&q.head)
    headNode := (*queueNode)(head)
    next := atomic.LoadPointer(&headNode.next)
    return next == nil
}
```

## Concurrent Data Structures

### Concurrent Map

```go
// Concurrent map with sharding
type ConcurrentMap struct {
    shards    []*Shard
    shardMask uint32
}

type Shard struct {
    mutex sync.RWMutex
    data  map[string]interface{}
}

func NewConcurrentMap(numShards int) *ConcurrentMap {
    if numShards <= 0 {
        numShards = 16
    }
    
    shards := make([]*Shard, numShards)
    for i := 0; i < numShards; i++ {
        shards[i] = &Shard{
            data: make(map[string]interface{}),
        }
    }
    
    return &ConcurrentMap{
        shards:    shards,
        shardMask: uint32(numShards - 1),
    }
}

func (cm *ConcurrentMap) getShard(key string) *Shard {
    hash := fnv.New32a()
    hash.Write([]byte(key))
    return cm.shards[hash.Sum32()&cm.shardMask]
}

func (cm *ConcurrentMap) Set(key string, value interface{}) {
    shard := cm.getShard(key)
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    shard.data[key] = value
}

func (cm *ConcurrentMap) Get(key string) (interface{}, bool) {
    shard := cm.getShard(key)
    shard.mutex.RLock()
    defer shard.mutex.RUnlock()
    value, exists := shard.data[key]
    return value, exists
}

func (cm *ConcurrentMap) Delete(key string) {
    shard := cm.getShard(key)
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    delete(shard.data, key)
}

func (cm *ConcurrentMap) Size() int {
    total := 0
    for _, shard := range cm.shards {
        shard.mutex.RLock()
        total += len(shard.data)
        shard.mutex.RUnlock()
    }
    return total
}

func (cm *ConcurrentMap) Range(fn func(key string, value interface{}) bool) {
    for _, shard := range cm.shards {
        shard.mutex.RLock()
        for key, value := range shard.data {
            if !fn(key, value) {
                shard.mutex.RUnlock()
                return
            }
        }
        shard.mutex.RUnlock()
    }
}
```

### Concurrent Set

```go
// Concurrent set implementation
type ConcurrentSet struct {
    mutex sync.RWMutex
    data  map[interface{}]struct{}
}

func NewConcurrentSet() *ConcurrentSet {
    return &ConcurrentSet{
        data: make(map[interface{}]struct{}),
    }
}

func (cs *ConcurrentSet) Add(item interface{}) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    cs.data[item] = struct{}{}
}

func (cs *ConcurrentSet) Remove(item interface{}) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    delete(cs.data, item)
}

func (cs *ConcurrentSet) Contains(item interface{}) bool {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    _, exists := cs.data[item]
    return exists
}

func (cs *ConcurrentSet) Size() int {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    return len(cs.data)
}

func (cs *ConcurrentSet) IsEmpty() bool {
    return cs.Size() == 0
}

func (cs *ConcurrentSet) Clear() {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    cs.data = make(map[interface{}]struct{})
}

func (cs *ConcurrentSet) ToSlice() []interface{} {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    
    result := make([]interface{}, 0, len(cs.data))
    for item := range cs.data {
        result = append(result, item)
    }
    return result
}

func (cs *ConcurrentSet) Union(other *ConcurrentSet) *ConcurrentSet {
    result := NewConcurrentSet()
    
    cs.mutex.RLock()
    for item := range cs.data {
        result.Add(item)
    }
    cs.mutex.RUnlock()
    
    other.mutex.RLock()
    for item := range other.data {
        result.Add(item)
    }
    other.mutex.RUnlock()
    
    return result
}

func (cs *ConcurrentSet) Intersection(other *ConcurrentSet) *ConcurrentSet {
    result := NewConcurrentSet()
    
    cs.mutex.RLock()
    other.mutex.RLock()
    
    // Iterate over the smaller set
    if len(cs.data) < len(other.data) {
        for item := range cs.data {
            if _, exists := other.data[item]; exists {
                result.Add(item)
            }
        }
    } else {
        for item := range other.data {
            if _, exists := cs.data[item]; exists {
                result.Add(item)
            }
        }
    }
    
    cs.mutex.RUnlock()
    other.mutex.RUnlock()
    
    return result
}
```

## Race Condition Prevention

### Race Detection

```go
// Race condition detection utilities
type RaceDetector struct {
    accesses map[string][]AccessInfo
    mutex    sync.RWMutex
}

type AccessInfo struct {
    GoroutineID int64
    Timestamp   time.Time
    Operation   string
    Stack       string
}

func NewRaceDetector() *RaceDetector {
    return &RaceDetector{
        accesses: make(map[string][]AccessInfo),
    }
}

func (rd *RaceDetector) RecordAccess(key string, operation string) {
    rd.mutex.Lock()
    defer rd.mutex.Unlock()
    
    info := AccessInfo{
        GoroutineID: getGoroutineID(),
        Timestamp:   time.Now(),
        Operation:   operation,
        Stack:       getStackTrace(),
    }
    
    rd.accesses[key] = append(rd.accesses[key], info)
    
    // Check for potential race conditions
    rd.checkRaceCondition(key)
}

func (rd *RaceDetector) checkRaceCondition(key string) {
    accesses := rd.accesses[key]
    if len(accesses) < 2 {
        return
    }
    
    // Check for concurrent access from different goroutines
    for i := 0; i < len(accesses)-1; i++ {
        for j := i + 1; j < len(accesses); j++ {
            if accesses[i].GoroutineID != accesses[j].GoroutineID {
                // Potential race condition detected
                fmt.Printf("Potential race condition detected for key %s:\n", key)
                fmt.Printf("  Goroutine %d: %s at %v\n", 
                    accesses[i].GoroutineID, accesses[i].Operation, accesses[i].Timestamp)
                fmt.Printf("  Goroutine %d: %s at %v\n", 
                    accesses[j].GoroutineID, accesses[j].Operation, accesses[j].Timestamp)
            }
        }
    }
}

func getStackTrace() string {
    buf := make([]byte, 1024)
    n := runtime.Stack(buf, false)
    return string(buf[:n])
}

// Atomic operations for race-free code
type AtomicCounter struct {
    value int64
}

func (ac *AtomicCounter) Increment() int64 {
    return atomic.AddInt64(&ac.value, 1)
}

func (ac *AtomicCounter) Decrement() int64 {
    return atomic.AddInt64(&ac.value, -1)
}

func (ac *AtomicCounter) Value() int64 {
    return atomic.LoadInt64(&ac.value)
}

func (ac *AtomicCounter) CompareAndSwap(old, new int64) bool {
    return atomic.CompareAndSwapInt64(&ac.value, old, new)
}

func (ac *AtomicCounter) Swap(new int64) int64 {
    return atomic.SwapInt64(&ac.value, new)
}
```

## Performance Optimization

### Goroutine Profiling

```go
// Goroutine profiling utilities
type GoroutineProfiler struct {
    snapshots []GoroutineSnapshot
    mutex     sync.RWMutex
}

type GoroutineSnapshot struct {
    Timestamp    time.Time
    NumGoroutines int
    StackTraces  map[string]int
}

func NewGoroutineProfiler() *GoroutineProfiler {
    return &GoroutineProfiler{
        snapshots: make([]GoroutineSnapshot, 0),
    }
}

func (gp *GoroutineProfiler) TakeSnapshot() {
    gp.mutex.Lock()
    defer gp.mutex.Unlock()
    
    snapshot := GoroutineSnapshot{
        Timestamp:     time.Now(),
        NumGoroutines: runtime.NumGoroutine(),
        StackTraces:   make(map[string]int),
    }
    
    // Get stack traces
    buf := make([]byte, 1024*1024)
    n := runtime.Stack(buf, true)
    stack := string(buf[:n])
    
    // Count goroutines by stack trace
    lines := strings.Split(stack, "\n")
    for _, line := range lines {
        if strings.HasPrefix(line, "goroutine ") {
            snapshot.StackTraces[line]++
        }
    }
    
    gp.snapshots = append(gp.snapshots, snapshot)
}

func (gp *GoroutineProfiler) GetLeaks() []string {
    gp.mutex.RLock()
    defer gp.mutex.RUnlock()
    
    if len(gp.snapshots) < 2 {
        return nil
    }
    
    last := gp.snapshots[len(gp.snapshots)-1]
    previous := gp.snapshots[len(gp.snapshots)-2]
    
    var leaks []string
    for stack, count := range last.StackTraces {
        if prevCount, exists := previous.StackTraces[stack]; exists {
            if count > prevCount {
                leaks = append(leaks, fmt.Sprintf("%s: +%d", stack, count-prevCount))
            }
        } else {
            leaks = append(leaks, fmt.Sprintf("%s: +%d", stack, count))
        }
    }
    
    return leaks
}

// Memory usage monitoring
func MonitorMemoryUsage(interval time.Duration, callback func(uint64)) {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()
    
    for range ticker.C {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        callback(m.Alloc)
    }
}

// CPU usage monitoring
func MonitorCPUUsage(interval time.Duration, callback func(float64)) {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()
    
    var lastCPU uint64
    var lastTime time.Time
    
    for range ticker.C {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        
        now := time.Now()
        if !lastTime.IsZero() {
            cpuUsage := float64(m.NumGC-lastCPU) / now.Sub(lastTime).Seconds()
            callback(cpuUsage)
        }
        
        lastCPU = m.NumGC
        lastTime = now
    }
}
```

## Conclusion

Advanced concurrency interviews test:

1. **Goroutine Management**: Understanding goroutine lifecycle and management
2. **Channel Patterns**: Complex channel usage patterns and communication
3. **Synchronization**: Advanced synchronization primitives and patterns
4. **Lock-Free Programming**: Atomic operations and lock-free data structures
5. **Concurrent Data Structures**: Thread-safe data structure implementations
6. **Race Condition Prevention**: Identifying and preventing race conditions
7. **Performance Optimization**: Profiling and optimizing concurrent code

Mastering these advanced concurrency concepts demonstrates your readiness for senior engineering roles and complex concurrent system development.

## Additional Resources

- [Go Concurrency Patterns](https://golang.org/doc/codewalk/sharemem/)
- [Advanced Go Concurrency](https://www.advancedgoconcurrency.com/)
- [Lock-Free Programming](https://www.lockfreeprogramming.com/)
- [Concurrent Data Structures](https://www.concurrentdatastructures.com/)
- [Race Condition Prevention](https://www.raceconditionprevention.com/)
- [Performance Profiling](https://www.performanceprofiling.com/)
- [Goroutine Management](https://www.goroutinemanagement.com/)
