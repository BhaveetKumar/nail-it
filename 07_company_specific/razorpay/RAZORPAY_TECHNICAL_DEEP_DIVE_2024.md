---
# Auto-generated front matter
Title: Razorpay Technical Deep Dive 2024
LastUpdated: 2025-11-06T20:45:58.500541
Tags: []
Status: draft
---

# 🎯 **Razorpay Technical Deep Dive Guide 2024**

## 📊 **Comprehensive Go Runtime, System Design & Performance Engineering**

---

## 🚀 **Go Runtime Deep Dive**

### **1. Go Scheduler & Concurrency Model**

#### **M:N Scheduler Model**

```go
// Understanding Go's Scheduler
type GoScheduler struct {
    // G: Goroutine
    // M: Machine (OS Thread)
    // P: Processor (Logical CPU)

    // Global run queue
    globalRunQueue *RunQueue

    // Per-P run queues
    localRunQueues []*RunQueue

    // Network poller
    networkPoller *NetworkPoller

    // Work stealing
    workStealing bool
}

// Goroutine States
const (
    _Gidle = iota
    _Grunnable
    _Grunning
    _Gsyscall
    _Gwaiting
    _Gdead
    _Gcopystack
    _Gpreempted
)

// Optimizing for Go Scheduler
func OptimizeForScheduler() {
    // 1. Set optimal GOMAXPROCS
    runtime.GOMAXPROCS(runtime.NumCPU())

    // 2. Use appropriate goroutine pool size
    numWorkers := runtime.NumCPU() * 2 // For I/O bound work

    // 3. Avoid excessive goroutine creation
    // Bad: Creating goroutines in tight loops
    for i := 0; i < 1000000; i++ {
        go func() {
            // Work
        }()
    }

    // Good: Use worker pools
    jobs := make(chan Job, 1000)
    for i := 0; i < numWorkers; i++ {
        go worker(jobs)
    }
}

func worker(jobs <-chan Job) {
    for job := range jobs {
        processJob(job)
    }
}
```

#### **Work Stealing Algorithm**

```go
// Work Stealing Implementation
type WorkStealer struct {
    localQueue  *deque.Deque
    globalQueue *RunQueue
    mutex       sync.Mutex
}

func (ws *WorkStealer) StealWork() *Goroutine {
    // 1. Try local queue first
    if ws.localQueue.Len() > 0 {
        return ws.localQueue.PopBack()
    }

    // 2. Try to steal from other processors
    for i := 0; i < runtime.GOMAXPROCS(0); i++ {
        if otherQueue := getOtherProcessorQueue(i); otherQueue != nil {
            if goroutine := otherQueue.Steal(); goroutine != nil {
                return goroutine
            }
        }
    }

    // 3. Try global queue
    ws.mutex.Lock()
    defer ws.mutex.Unlock()
    return ws.globalQueue.Pop()
}
```

### **2. Memory Management & Garbage Collection**

#### **Go's Garbage Collector**

```go
// GC Optimization for Payment Processing
type PaymentProcessor struct {
    // Memory pools to reduce GC pressure
    paymentPool    *sync.Pool
    requestPool    *sync.Pool
    responsePool   *sync.Pool

    // String interning for common values
    stringInterner *StringInterner

    // Object pooling
    bufferPool     *BufferPool
}

func NewPaymentProcessor() *PaymentProcessor {
    return &PaymentProcessor{
        paymentPool: &sync.Pool{
            New: func() interface{} {
                return &Payment{
                    Metadata: make(map[string]string, 16), // Pre-allocate
                    Tags:     make([]string, 0, 8),
                }
            },
        },
        requestPool: &sync.Pool{
            New: func() interface{} {
                return &PaymentRequest{
                    Headers: make(map[string]string, 10),
                }
            },
        },
        stringInterner: NewStringInterner(),
        bufferPool:     NewBufferPool(1024), // 1KB buffers
    }
}

// String Interning for Common Values
type StringInterner struct {
    cache map[string]string
    mutex sync.RWMutex
}

func (si *StringInterner) Intern(s string) string {
    si.mutex.RLock()
    if interned, exists := si.cache[s]; exists {
        si.mutex.RUnlock()
        return interned
    }
    si.mutex.RUnlock()

    si.mutex.Lock()
    defer si.mutex.Unlock()

    // Double-check pattern
    if interned, exists := si.cache[s]; exists {
        return interned
    }

    si.cache[s] = s
    return s
}

// Buffer Pool for I/O Operations
type BufferPool struct {
    pool sync.Pool
    size int
}

func NewBufferPool(size int) *BufferPool {
    return &BufferPool{
        pool: sync.Pool{
            New: func() interface{} {
                return make([]byte, size)
            },
        },
        size: size,
    }
}

func (bp *BufferPool) Get() []byte {
    return bp.pool.Get().([]byte)
}

func (bp *BufferPool) Put(buf []byte) {
    if len(buf) == bp.size {
        bp.pool.Put(buf)
    }
}
```

#### **GC Tuning for High-Throughput Systems**

```go
// GC Optimization Settings
func OptimizeGC() {
    // 1. Set GC target percentage (default: 100%)
    // Lower values = more frequent GC = lower latency
    debug.SetGCPercent(50)

    // 2. Set memory limit (Go 1.19+)
    debug.SetMemoryLimit(2 << 30) // 2GB limit

    // 3. Monitor GC metrics
    go func() {
        ticker := time.NewTicker(5 * time.Second)
        defer ticker.Stop()

        for range ticker.C {
            var m runtime.MemStats
            runtime.ReadMemStats(&m)

            log.Printf("GC Stats - Alloc: %d, Sys: %d, NumGC: %d, PauseTotal: %v",
                m.Alloc, m.Sys, m.NumGC, time.Duration(m.PauseTotalNs))
        }
    }()
}

// Memory-efficient data structures
type EfficientPayment struct {
    // Use fixed-size arrays instead of slices when possible
    ID       [16]byte // UUID as fixed array
    Amount   int64
    Currency [3]byte  // ISO currency code

    // Use bit fields for flags
    Flags uint32 // Bit 0: processed, Bit 1: failed, etc.

    // Pool-allocated metadata
    Metadata *PooledMap
}

type PooledMap struct {
    data map[string]string
    pool *sync.Pool
}

func (pm *PooledMap) Reset() {
    for k := range pm.data {
        delete(pm.data, k)
    }
}

func (pm *PooledMap) Return() {
    pm.Reset()
    pm.pool.Put(pm)
}
```

### **3. Advanced Concurrency Patterns**

#### **Lock-Free Data Structures**

```go
// Lock-free ring buffer for high-throughput logging
type LockFreeRingBuffer struct {
    buffer []interface{}
    mask   uint64
    head   uint64
    tail   uint64
}

func NewLockFreeRingBuffer(size uint64) *LockFreeRingBuffer {
    // Size must be power of 2
    if size&(size-1) != 0 {
        panic("size must be power of 2")
    }

    return &LockFreeRingBuffer{
        buffer: make([]interface{}, size),
        mask:   size - 1,
    }
}

func (rb *LockFreeRingBuffer) Enqueue(item interface{}) bool {
    head := atomic.LoadUint64(&rb.head)
    tail := atomic.LoadUint64(&rb.tail)

    // Check if buffer is full
    if (head+1)&rb.mask == tail&rb.mask {
        return false
    }

    // Store item
    rb.buffer[head&rb.mask] = item

    // Update head
    atomic.StoreUint64(&rb.head, head+1)
    return true
}

func (rb *LockFreeRingBuffer) Dequeue() (interface{}, bool) {
    tail := atomic.LoadUint64(&rb.tail)
    head := atomic.LoadUint64(&rb.head)

    // Check if buffer is empty
    if tail&rb.mask == head&rb.mask {
        return nil, false
    }

    // Get item
    item := rb.buffer[tail&rb.mask]

    // Update tail
    atomic.StoreUint64(&rb.tail, tail+1)
    return item, true
}

// Lock-free hash map for concurrent access
type LockFreeHashMap struct {
    buckets []*atomic.Value
    size    uint64
    mask    uint64
}

type Bucket struct {
    key   string
    value interface{}
    next  *Bucket
}

func NewLockFreeHashMap(size uint64) *LockFreeHashMap {
    if size&(size-1) != 0 {
        panic("size must be power of 2")
    }

    buckets := make([]*atomic.Value, size)
    for i := range buckets {
        buckets[i] = &atomic.Value{}
        buckets[i].Store((*Bucket)(nil))
    }

    return &LockFreeHashMap{
        buckets: buckets,
        size:    size,
        mask:    size - 1,
    }
}

func (hm *LockFreeHashMap) Set(key string, value interface{}) {
    hash := fnv.New64a()
    hash.Write([]byte(key))
    bucketIndex := hash.Sum64() & hm.mask

    bucket := &Bucket{
        key:   key,
        value: value,
    }

    // CAS loop for lock-free insertion
    for {
        current := hm.buckets[bucketIndex].Load().(*Bucket)
        bucket.next = current

        if hm.buckets[bucketIndex].CompareAndSwap(current, bucket) {
            return
        }
    }
}

func (hm *LockFreeHashMap) Get(key string) (interface{}, bool) {
    hash := fnv.New64a()
    hash.Write([]byte(key))
    bucketIndex := hash.Sum64() & hm.mask

    current := hm.buckets[bucketIndex].Load().(*Bucket)
    for current != nil {
        if current.key == key {
            return current.value, true
        }
        current = current.next
    }

    return nil, false
}
```

#### **Advanced Channel Patterns**

```go
// Fan-out/Fan-in Pattern for Payment Processing
type PaymentFanOut struct {
    input    <-chan Payment
    outputs  []chan Payment
    workers  int
}

func NewPaymentFanOut(input <-chan Payment, workers int) *PaymentFanOut {
    outputs := make([]chan Payment, workers)
    for i := range outputs {
        outputs[i] = make(chan Payment, 100)
    }

    return &PaymentFanOut{
        input:   input,
        outputs: outputs,
        workers: workers,
    }
}

func (pfo *PaymentFanOut) Start() {
    go func() {
        defer func() {
            for _, output := range pfo.outputs {
                close(output)
            }
        }()

        i := 0
        for payment := range pfo.input {
            // Round-robin distribution
            pfo.outputs[i] <- payment
            i = (i + 1) % pfo.workers
        }
    }()
}

// Pipeline Pattern for Payment Processing
type PaymentPipeline struct {
    stages []Stage
}

type Stage interface {
    Process(input <-chan Payment) <-chan Payment
}

type ValidationStage struct {
    validator *PaymentValidator
}

func (vs *ValidationStage) Process(input <-chan Payment) <-chan Payment {
    output := make(chan Payment, 100)

    go func() {
        defer close(output)
        for payment := range input {
            if vs.validator.Validate(payment) {
                output <- payment
            }
        }
    }()

    return output
}

type FraudDetectionStage struct {
    detector *FraudDetector
}

func (fds *FraudDetectionStage) Process(input <-chan Payment) <-chan Payment {
    output := make(chan Payment, 100)

    go func() {
        defer close(output)
        for payment := range input {
            if !fds.detector.IsFraudulent(payment) {
                output <- payment
            }
        }
    }()

    return output
}

// Pipeline execution
func (pp *PaymentPipeline) Execute(input <-chan Payment) <-chan Payment {
    current := input
    for _, stage := range pp.stages {
        current = stage.Process(current)
    }
    return current
}
```

---

## 🚀 **System Design Deep Dive**

### **1. Microservices Architecture Patterns**

#### **Service Mesh Implementation**

```go
// Service Mesh for Payment Services
type ServiceMesh struct {
    services    map[string]*Service
    loadBalancer *LoadBalancer
    circuitBreaker *CircuitBreaker
    rateLimiter *RateLimiter
    tracer      *Tracer
}

type Service struct {
    Name        string
    Endpoints   []string
    HealthCheck *HealthCheck
    Metrics     *Metrics
}

// Circuit Breaker Pattern
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
        return generation, ErrOpenState
    } else if state == StateHalfOpen && cb.counts.Requests >= cb.maxRequests {
        return generation, ErrTooManyRequests
    }

    cb.counts.onRequest()
    return generation, nil
}

// Rate Limiting with Token Bucket
type TokenBucket struct {
    capacity     int64
    tokens       int64
    refillRate   int64
    lastRefill   time.Time
    mutex        sync.Mutex
}

func NewTokenBucket(capacity, refillRate int64) *TokenBucket {
    return &TokenBucket{
        capacity:   capacity,
        tokens:     capacity,
        refillRate: refillRate,
        lastRefill: time.Now(),
    }
}

func (tb *TokenBucket) Allow() bool {
    tb.mutex.Lock()
    defer tb.mutex.Unlock()

    now := time.Now()
    tokensToAdd := int64(now.Sub(tb.lastRefill).Seconds()) * tb.refillRate
    tb.tokens = min(tb.capacity, tb.tokens+tokensToAdd)
    tb.lastRefill = now

    if tb.tokens > 0 {
        tb.tokens--
        return true
    }

    return false
}

func min(a, b int64) int64 {
    if a < b {
        return a
    }
    return b
}
```

#### **Event-Driven Architecture**

```go
// Event Sourcing for Payment Events
type EventStore struct {
    events []Event
    mutex  sync.RWMutex
}

type Event struct {
    ID        string
    Type      string
    AggregateID string
    Data      interface{}
    Timestamp time.Time
    Version   int
}

type PaymentAggregate struct {
    ID      string
    Version int
    State   PaymentState
    Events  []Event
}

type PaymentState struct {
    Amount    int64
    Status    string
    UserID    string
    CreatedAt time.Time
}

func (ps *PaymentAggregate) ApplyEvent(event Event) {
    switch event.Type {
    case "PaymentCreated":
        ps.State.Amount = event.Data.(PaymentCreatedData).Amount
        ps.State.UserID = event.Data.(PaymentCreatedData).UserID
        ps.State.Status = "created"
        ps.State.CreatedAt = event.Timestamp

    case "PaymentProcessed":
        ps.State.Status = "processed"

    case "PaymentFailed":
        ps.State.Status = "failed"
    }

    ps.Version = event.Version
    ps.Events = append(ps.Events, event)
}

// CQRS Implementation
type CommandHandler struct {
    eventStore *EventStore
    repository *PaymentRepository
}

type QueryHandler struct {
    readModel *PaymentReadModel
}

type PaymentReadModel struct {
    payments map[string]*PaymentView
    mutex    sync.RWMutex
}

type PaymentView struct {
    ID        string
    Amount    int64
    Status    string
    UserID    string
    CreatedAt time.Time
}

func (ch *CommandHandler) HandleCreatePayment(cmd CreatePaymentCommand) error {
    // Create aggregate
    aggregate := &PaymentAggregate{
        ID:      cmd.PaymentID,
        Version: 0,
    }

    // Create event
    event := Event{
        ID:          generateUUID(),
        Type:        "PaymentCreated",
        AggregateID: cmd.PaymentID,
        Data: PaymentCreatedData{
            Amount: cmd.Amount,
            UserID: cmd.UserID,
        },
        Timestamp: time.Now(),
        Version:   1,
    }

    // Apply event
    aggregate.ApplyEvent(event)

    // Store event
    ch.eventStore.AppendEvent(event)

    return nil
}

func (qh *QueryHandler) GetPayment(paymentID string) (*PaymentView, error) {
    qh.readModel.mutex.RLock()
    defer qh.readModel.mutex.RUnlock()

    payment, exists := qh.readModel.payments[paymentID]
    if !exists {
        return nil, fmt.Errorf("payment not found")
    }

    return payment, nil
}
```

### **2. Database Design & Optimization**

#### **Sharding Strategy**

```go
// Consistent Hashing for Database Sharding
type ConsistentHash struct {
    ring     map[uint32]string
    sortedKeys []uint32
    replicas int
    mutex    sync.RWMutex
}

func NewConsistentHash(replicas int) *ConsistentHash {
    return &ConsistentHash{
        ring:     make(map[uint32]string),
        replicas: replicas,
    }
}

func (ch *ConsistentHash) AddNode(node string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()

    for i := 0; i < ch.replicas; i++ {
        key := ch.hash(fmt.Sprintf("%s:%d", node, i))
        ch.ring[key] = node
        ch.sortedKeys = append(ch.sortedKeys, key)
    }

    sort.Slice(ch.sortedKeys, func(i, j int) bool {
        return ch.sortedKeys[i] < ch.sortedKeys[j]
    })
}

func (ch *ConsistentHash) GetNode(key string) (string, bool) {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()

    if len(ch.ring) == 0 {
        return "", false
    }

    hash := ch.hash(key)
    idx := sort.Search(len(ch.sortedKeys), func(i int) bool {
        return ch.sortedKeys[i] >= hash
    })

    if idx == len(ch.sortedKeys) {
        idx = 0
    }

    node := ch.ring[ch.sortedKeys[idx]]
    return node, true
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := fnv.New32a()
    h.Write([]byte(key))
    return h.Sum32()
}

// Database Connection Pooling
type ConnectionPool struct {
    connections chan *sql.DB
    factory     func() (*sql.DB, error)
    maxSize     int
    currentSize int
    mutex       sync.Mutex
}

func NewConnectionPool(factory func() (*sql.DB, error), maxSize int) *ConnectionPool {
    return &ConnectionPool{
        connections: make(chan *sql.DB, maxSize),
        factory:     factory,
        maxSize:     maxSize,
    }
}

func (cp *ConnectionPool) Get() (*sql.DB, error) {
    select {
    case conn := <-cp.connections:
        return conn, nil
    default:
        cp.mutex.Lock()
        defer cp.mutex.Unlock()

        if cp.currentSize < cp.maxSize {
            conn, err := cp.factory()
            if err != nil {
                return nil, err
            }
            cp.currentSize++
            return conn, nil
        }

        // Wait for connection to be returned
        return <-cp.connections, nil
    }
}

func (cp *ConnectionPool) Put(conn *sql.DB) {
    select {
    case cp.connections <- conn:
    default:
        // Pool is full, close connection
        conn.Close()
        cp.mutex.Lock()
        cp.currentSize--
        cp.mutex.Unlock()
    }
}
```

#### **Caching Strategies**

```go
// Multi-Level Caching
type MultiLevelCache struct {
    l1Cache *sync.Map // In-memory cache
    l2Cache *redis.Client // Redis cache
    l3Cache *sql.DB // Database
}

func (mlc *MultiLevelCache) Get(key string) (interface{}, error) {
    // L1 Cache (In-memory)
    if value, ok := mlc.l1Cache.Load(key); ok {
        return value, nil
    }

    // L2 Cache (Redis)
    if value, err := mlc.l2Cache.Get(key).Result(); err == nil {
        // Store in L1 cache
        mlc.l1Cache.Store(key, value)
        return value, nil
    }

    // L3 Cache (Database)
    var value interface{}
    err := mlc.l3Cache.QueryRow("SELECT data FROM cache WHERE key = ?", key).Scan(&value)
    if err != nil {
        return nil, err
    }

    // Store in L2 and L1 caches
    mlc.l2Cache.Set(key, value, 1*time.Hour)
    mlc.l1Cache.Store(key, value)

    return value, nil
}

func (mlc *MultiLevelCache) Set(key string, value interface{}, ttl time.Duration) error {
    // Store in all levels
    mlc.l1Cache.Store(key, value)
    mlc.l2Cache.Set(key, value, ttl)

    // Store in database with TTL
    _, err := mlc.l3Cache.Exec(`
        INSERT INTO cache (key, data, expires_at)
        VALUES (?, ?, ?)
        ON DUPLICATE KEY UPDATE data = VALUES(data), expires_at = VALUES(expires_at)
    `, key, value, time.Now().Add(ttl))

    return err
}

// Cache-Aside Pattern
type CacheAsideService struct {
    cache *redis.Client
    db    *sql.DB
}

func (cas *CacheAsideService) GetPayment(paymentID string) (*Payment, error) {
    // 1. Try cache first
    cached, err := cas.cache.Get(fmt.Sprintf("payment:%s", paymentID)).Result()
    if err == nil {
        var payment Payment
        json.Unmarshal([]byte(cached), &payment)
        return &payment, nil
    }

    // 2. Cache miss - get from database
    var payment Payment
    err = cas.db.QueryRow(`
        SELECT id, amount, status, user_id, created_at
        FROM payments
        WHERE id = ?
    `, paymentID).Scan(&payment.ID, &payment.Amount, &payment.Status, &payment.UserID, &payment.CreatedAt)

    if err != nil {
        return nil, err
    }

    // 3. Store in cache
    data, _ := json.Marshal(payment)
    cas.cache.Set(fmt.Sprintf("payment:%s", paymentID), data, 1*time.Hour)

    return &payment, nil
}
```

---

## 🚀 **Performance Engineering**

### **1. Profiling & Monitoring**

#### **Go Profiling Tools**

```go
// CPU Profiling
func StartCPUProfiling() {
    f, err := os.Create("cpu.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    if err := pprof.StartCPUProfile(f); err != nil {
        log.Fatal(err)
    }
    defer pprof.StopCPUProfile()
}

// Memory Profiling
func StartMemoryProfiling() {
    f, err := os.Create("mem.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    runtime.GC()
    if err := pprof.WriteHeapProfile(f); err != nil {
        log.Fatal(err)
    }
}

// Goroutine Profiling
func StartGoroutineProfiling() {
    f, err := os.Create("goroutine.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    if err := pprof.Lookup("goroutine").WriteTo(f, 0); err != nil {
        log.Fatal(err)
    }
}

// Custom Metrics Collection
type MetricsCollector struct {
    counters   map[string]*Counter
    histograms map[string]*Histogram
    gauges     map[string]*Gauge
    mutex      sync.RWMutex
}

type Counter struct {
    value int64
    mutex sync.RWMutex
}

func (c *Counter) Inc() {
    c.mutex.Lock()
    c.value++
    c.mutex.Unlock()
}

func (c *Counter) Add(delta int64) {
    c.mutex.Lock()
    c.value += delta
    c.mutex.Unlock()
}

func (c *Counter) Value() int64 {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    return c.value
}

type Histogram struct {
    buckets []float64
    counts  []int64
    mutex   sync.RWMutex
}

func (h *Histogram) Observe(value float64) {
    h.mutex.Lock()
    defer h.mutex.Unlock()

    for i, bucket := range h.buckets {
        if value <= bucket {
            h.counts[i]++
            return
        }
    }
    // Value exceeds all buckets
    h.counts[len(h.counts)-1]++
}

// Real-time Performance Monitoring
type PerformanceMonitor struct {
    metrics    *MetricsCollector
    alerting   *AlertingService
    dashboard  *DashboardService
}

func (pm *PerformanceMonitor) Start() {
    go pm.collectSystemMetrics()
    go pm.collectApplicationMetrics()
    go pm.checkAlerts()
}

func (pm *PerformanceMonitor) collectSystemMetrics() {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()

    for range ticker.C {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)

        // Record memory metrics
        pm.metrics.Gauge("memory.alloc").Set(float64(m.Alloc))
        pm.metrics.Gauge("memory.sys").Set(float64(m.Sys))
        pm.metrics.Gauge("memory.num_gc").Set(float64(m.NumGC))

        // Record GC pause time
        pm.metrics.Histogram("gc.pause").Observe(float64(m.PauseNs[(m.NumGC+255)%256]) / 1e6) // Convert to ms
    }
}
```

### **2. Optimization Techniques**

#### **Memory Optimization**

```go
// Object Pooling for High-Frequency Objects
type PaymentPool struct {
    pool sync.Pool
}

func NewPaymentPool() *PaymentPool {
    return &PaymentPool{
        pool: sync.Pool{
            New: func() interface{} {
                return &Payment{
                    Metadata: make(map[string]string, 16),
                    Tags:     make([]string, 0, 8),
                }
            },
        },
    }
}

func (pp *PaymentPool) Get() *Payment {
    payment := pp.pool.Get().(*Payment)
    payment.Reset()
    return payment
}

func (pp *PaymentPool) Put(payment *Payment) {
    pp.pool.Put(payment)
}

// Slice Reuse to Reduce Allocations
type SlicePool struct {
    pools map[int]*sync.Pool
    mutex sync.RWMutex
}

func NewSlicePool() *SlicePool {
    return &SlicePool{
        pools: make(map[int]*sync.Pool),
    }
}

func (sp *SlicePool) Get(size int) []byte {
    sp.mutex.RLock()
    pool, exists := sp.pools[size]
    sp.mutex.RUnlock()

    if !exists {
        sp.mutex.Lock()
        pool, exists = sp.pools[size]
        if !exists {
            pool = &sync.Pool{
                New: func() interface{} {
                    return make([]byte, size)
                },
            }
            sp.pools[size] = pool
        }
        sp.mutex.Unlock()
    }

    return pool.Get().([]byte)
}

func (sp *SlicePool) Put(slice []byte) {
    size := len(slice)
    sp.mutex.RLock()
    pool, exists := sp.pools[size]
    sp.mutex.RUnlock()

    if exists {
        pool.Put(slice)
    }
}

// String Builder for Efficient String Concatenation
type StringBuilder struct {
    buffer []byte
}

func NewStringBuilder() *StringBuilder {
    return &StringBuilder{
        buffer: make([]byte, 0, 64),
    }
}

func (sb *StringBuilder) WriteString(s string) {
    sb.buffer = append(sb.buffer, s...)
}

func (sb *StringBuilder) WriteByte(c byte) {
    sb.buffer = append(sb.buffer, c)
}

func (sb *StringBuilder) String() string {
    return string(sb.buffer)
}

func (sb *StringBuilder) Reset() {
    sb.buffer = sb.buffer[:0]
}
```

#### **Concurrency Optimization**

```go
// Worker Pool with Backpressure
type WorkerPool struct {
    workers    int
    jobQueue   chan Job
    resultQueue chan Result
    quit       chan bool
    wg         sync.WaitGroup
}

type Job struct {
    ID   string
    Data interface{}
}

type Result struct {
    JobID string
    Data  interface{}
    Error error
}

func NewWorkerPool(workers int, queueSize int) *WorkerPool {
    return &WorkerPool{
        workers:     workers,
        jobQueue:    make(chan Job, queueSize),
        resultQueue: make(chan Result, queueSize),
        quit:        make(chan bool),
    }
}

func (wp *WorkerPool) Start() {
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker(i)
    }
}

func (wp *WorkerPool) worker(id int) {
    defer wp.wg.Done()

    for {
        select {
        case job := <-wp.jobQueue:
            result := wp.processJob(job)
            wp.resultQueue <- result

        case <-wp.quit:
            return
        }
    }
}

func (wp *WorkerPool) processJob(job Job) Result {
    // Simulate work
    time.Sleep(10 * time.Millisecond)

    return Result{
        JobID: job.ID,
        Data:  fmt.Sprintf("Processed by worker %d", job.ID),
        Error: nil,
    }
}

func (wp *WorkerPool) Submit(job Job) error {
    select {
    case wp.jobQueue <- job:
        return nil
    default:
        return fmt.Errorf("job queue is full")
    }
}

func (wp *WorkerPool) Stop() {
    close(wp.quit)
    wp.wg.Wait()
    close(wp.resultQueue)
}

// Batch Processing for High Throughput
type BatchProcessor struct {
    batchSize    int
    flushTimeout time.Duration
    processor    func([]interface{}) error
    batch        []interface{}
    mutex        sync.Mutex
    flushTicker  *time.Ticker
    quit         chan bool
}

func NewBatchProcessor(batchSize int, flushTimeout time.Duration, processor func([]interface{}) error) *BatchProcessor {
    bp := &BatchProcessor{
        batchSize:    batchSize,
        flushTimeout: flushTimeout,
        processor:    processor,
        batch:        make([]interface{}, 0, batchSize),
        quit:         make(chan bool),
    }

    bp.flushTicker = time.NewTicker(flushTimeout)
    go bp.flushLoop()

    return bp
}

func (bp *BatchProcessor) Add(item interface{}) error {
    bp.mutex.Lock()
    defer bp.mutex.Unlock()

    bp.batch = append(bp.batch, item)

    if len(bp.batch) >= bp.batchSize {
        return bp.flush()
    }

    return nil
}

func (bp *BatchProcessor) flush() error {
    if len(bp.batch) == 0 {
        return nil
    }

    batch := make([]interface{}, len(bp.batch))
    copy(batch, bp.batch)
    bp.batch = bp.batch[:0]

    return bp.processor(batch)
}

func (bp *BatchProcessor) flushLoop() {
    for {
        select {
        case <-bp.flushTicker.C:
            bp.mutex.Lock()
            bp.flush()
            bp.mutex.Unlock()

        case <-bp.quit:
            return
        }
    }
}

func (bp *BatchProcessor) Stop() {
    bp.flushTicker.Stop()
    bp.quit <- true

    bp.mutex.Lock()
    bp.flush()
    bp.mutex.Unlock()
}
```

---

## 🎯 **Mock Technical Deep Dive Questions**

### **1. Go Runtime Questions**

1. "Explain how Go's garbage collector works and how you would optimize it for a high-throughput payment system."
2. "How does Go's scheduler handle work stealing, and what are the performance implications?"
3. "Describe the memory model in Go and how it affects concurrent programming."
4. "How would you implement a lock-free data structure in Go?"

### **2. System Design Questions**

1. "Design a distributed cache system with consistency guarantees."
2. "How would you implement event sourcing for a payment system?"
3. "Design a service mesh for microservices communication."
4. "How would you handle database sharding for a payment system with 1B+ transactions?"

### **3. Performance Engineering Questions**

1. "How would you profile and optimize a Go application with high memory usage?"
2. "Describe your approach to implementing circuit breakers and rate limiting."
3. "How would you design a monitoring system for a distributed payment platform?"
4. "What strategies would you use to optimize database performance for high-throughput systems?"

---

**🎉 This comprehensive technical deep dive guide covers all the advanced topics you need for Razorpay interviews! 🚀**
