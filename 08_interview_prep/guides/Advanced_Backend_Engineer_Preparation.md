# 🚀 Advanced Backend Engineer Preparation Guide - Razorpay Lead SDE

> **Comprehensive preparation for 10+ years experienced backend engineers targeting Razorpay Lead SDE role**

## 📋 Table of Contents

1. [Advanced Go Runtime & Concurrency](#advanced-go-runtime--concurrency)
2. [System Design at Scale](#system-design-at-scale)
3. [Operating System Deep Dive](#operating-system-deep-dive)
4. [Razorpay-Specific Technical Challenges](#razorpay-specific-technical-challenges)
5. [Advanced DSA & Algorithm Patterns](#advanced-dsa--algorithm-patterns)
6. [Performance Engineering](#performance-engineering)
7. [Leadership & Architecture Decisions](#leadership--architecture-decisions)

---

## 🔧 Advanced Go Runtime & Concurrency

### **Go Scheduler Deep Dive**

#### **M:N Scheduler Model**

**Detailed Explanation:**

The Go scheduler implements an M:N threading model, which is a hybrid approach that provides the benefits of both user-space and kernel-space threading:

- **M (Machine/OS Threads)**: These are actual OS threads managed by the kernel. The number is limited by `GOMAXPROCS` (typically equal to the number of CPU cores).
- **N (Goroutines)**: These are lightweight user-space threads that can number in the millions or even billions.
- **P (Processors)**: These are logical processors that provide the context for scheduling. Each P has its own run queue and can be associated with an M.

**Why M:N Model?**

1. **Efficiency**: Goroutines are much lighter than OS threads (2KB vs 2MB initial stack)
2. **Scalability**: Can handle millions of goroutines without exhausting system resources
3. **Performance**: Avoids expensive kernel context switches for most operations
4. **Simplicity**: Provides a simple concurrency model for developers

**Key Challenges with Extreme Concurrency:**

When dealing with trillions of goroutines, several challenges emerge:

1. **Memory Pressure**: Each goroutine consumes ~2KB of stack space initially
2. **Scheduling Overhead**: The scheduler must efficiently distribute work
3. **Garbage Collection**: More goroutines mean more objects to track
4. **Work Stealing**: Efficient load balancing becomes critical

```go
// Understanding the Go scheduler behavior with trillions of goroutines
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

// The Go runtime uses an M:N scheduler where:
// M = OS threads (limited by GOMAXPROCS)
// N = Goroutines (can be millions/billions)
// P = Logical processors (context for scheduling)

func demonstrateSchedulerBehavior() {
    // With trillions of goroutines, the scheduler faces:
    // 1. Memory pressure (each goroutine ~2KB stack)
    // 2. Context switching overhead
    // 3. Work stealing efficiency
    // 4. Garbage collection pressure

    var wg sync.WaitGroup
    numGoroutines := 1000000 // Simulating high concurrency

    for i := 0; i < numGoroutines; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            // Simulate work
            time.Sleep(time.Millisecond)
        }(i)
    }

    wg.Wait()
    fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
    fmt.Printf("NumGoroutine: %d\n", runtime.NumGoroutine())
}
```

**Discussion Questions & Answers:**

**Q1: How does the Go scheduler handle work distribution when you have millions of goroutines?**

**Answer:** The Go scheduler uses a work-stealing algorithm where each P (processor) has its own run queue. When a P's queue is empty, it steals work from other P's queues. This ensures load balancing across all available processors. The scheduler also maintains a global run queue for overflow situations and uses a network poller for I/O operations.

**Q2: What happens to memory usage when you create trillions of goroutines?**

**Answer:** Each goroutine starts with a 2KB stack that can grow as needed. With trillions of goroutines, you'd need at least 2TB of memory just for stacks. However, the Go runtime uses segmented stacks that can grow and shrink dynamically. The real challenge is garbage collection pressure and the overhead of managing so many goroutines.

**Q3: How does the Go scheduler prevent goroutines from blocking the entire system?**

**Answer:** Go 1.14+ introduced preemptive scheduling where goroutines can be preempted at function calls and certain points in loops. The runtime also uses cooperative scheduling where goroutines voluntarily yield control at specific points. This prevents a single goroutine from monopolizing a processor.

**Q4: What are the performance implications of having too many goroutines?**

**Answer:** Too many goroutines can lead to:

- Increased memory usage and GC pressure
- Higher context switching overhead
- Reduced cache locality
- Work stealing becomes less efficient
- Potential for goroutine leaks if not properly managed

**Q5: How would you optimize a system that needs to handle millions of concurrent operations?**

**Answer:** Several optimization strategies:

1. Use worker pools instead of creating unlimited goroutines
2. Implement backpressure mechanisms to limit concurrency
3. Use channels for communication instead of shared memory
4. Profile and monitor goroutine usage
5. Consider using sync.Pool for object reuse
6. Implement proper timeouts and cancellation

#### **Work Stealing Algorithm**

**Detailed Explanation:**

The work stealing algorithm is a key component of Go's scheduler that ensures efficient load balancing across processors. Here's how it works:

**Core Components:**

1. **Local Run Queues**: Each P (processor) has its own run queue that can hold up to 256 goroutines
2. **Global Run Queue**: A shared queue for overflow situations and newly created goroutines
3. **Network Poller**: Handles I/O operations without blocking processors
4. **Work Stealing**: When a P's local queue is empty, it steals work from other P's queues

**Algorithm Steps:**

1. Check local run queue first
2. If empty, check global run queue
3. If still empty, steal from other P's queues
4. If no work found, park the M (OS thread) until work becomes available

**Benefits:**

- **Load Balancing**: Work is distributed evenly across all processors
- **Cache Locality**: Goroutines tend to run on the same processor
- **Scalability**: Works efficiently with any number of processors
- **Responsiveness**: I/O operations don't block processors

```go
// How the Go scheduler handles work distribution
type WorkStealingScheduler struct {
    // Each P (processor) has its own run queue
    runQueues []*RunQueue
    // Global run queue for overflow
    globalQueue *GlobalQueue
    // Work stealing happens when a P's queue is empty
}

// Key behaviors with trillions of goroutines:
// 1. Local run queue (256 goroutines max per P)
// 2. Global run queue (unlimited but slower access)
// 3. Network poller (handles I/O operations)
// 4. Work stealing from other P's queues
```

**Discussion Questions & Answers:**

**Q1: How does work stealing prevent processor starvation?**

**Answer:** Work stealing ensures that no processor remains idle while others have work. When a processor's local queue is empty, it actively seeks work from other processors' queues. This creates a dynamic load balancing system where work naturally flows from busy processors to idle ones, preventing any processor from being starved of work.

**Q2: What happens when all processors are busy and there's no work to steal?**

**Answer:** When all processors are busy and there's no work to steal, the processor parks its associated OS thread (M). The thread goes into a sleep state until new work becomes available. This is efficient because it doesn't consume CPU cycles while waiting. The thread will be woken up when new goroutines are created or when I/O operations complete.

**Q3: How does the 256 goroutine limit per local queue affect performance?**

**Answer:** The 256 goroutine limit per local queue is a balance between memory usage and performance. It ensures that:

- Each processor has a reasonable amount of work to process
- Memory usage is bounded per processor
- Work stealing remains efficient (not too many goroutines to steal from)
- Cache locality is maintained for frequently accessed goroutines

**Q4: Why is the global run queue slower than local run queues?**

**Answer:** The global run queue requires synchronization (mutex locks) because it's shared across all processors. Local run queues don't need synchronization since each processor owns its queue. This makes local queue operations much faster, which is why the scheduler prioritizes local work over global work.

**Q5: How does the network poller integrate with work stealing?**

**Answer:** The network poller handles I/O operations without blocking processors. When a goroutine performs I/O, it's moved to the network poller, and the processor can continue with other work. When the I/O completes, the goroutine is moved back to a run queue. This integration ensures that I/O operations don't waste processor time and that work stealing can continue efficiently.

#### **Performance Bottlenecks with Extreme Concurrency**

**Memory Issues:**

- Each goroutine: ~2KB initial stack
- Trillions of goroutines = ~2TB+ memory
- Stack growth can cause memory fragmentation
- GC pressure increases exponentially

**Scheduling Overhead:**

- Context switching between goroutines
- Work stealing becomes less efficient
- Lock contention on global structures
- Cache locality degradation

**Solutions:**

```go
// 1. Use worker pools instead of unlimited goroutines
type WorkerPool struct {
    workers    int
    jobQueue   chan Job
    resultChan chan Result
}

func (wp *WorkerPool) Start() {
    for i := 0; i < wp.workers; i++ {
        go wp.worker()
    }
}

// 2. Implement backpressure mechanisms
type BackpressureLimiter struct {
    semaphore chan struct{}
    maxConcurrency int
}

func (bl *BackpressureLimiter) Acquire() {
    bl.semaphore <- struct{}{}
}

func (bl *BackpressureLimiter) Release() {
    <-bl.semaphore
}
```

### **Advanced Concurrency Patterns**

#### **Lock-Free Data Structures**

```go
// Implementing lock-free operations for high-performance scenarios
import "sync/atomic"

type LockFreeCounter struct {
    value int64
}

func (c *LockFreeCounter) Increment() {
    atomic.AddInt64(&c.value, 1)
}

func (c *LockFreeCounter) Get() int64 {
    return atomic.LoadInt64(&c.value)
}

// Compare-and-Swap for complex operations
type LockFreeStack struct {
    head unsafe.Pointer
}

func (s *LockFreeStack) Push(value interface{}) {
    newHead := &Node{value: value}
    for {
        oldHead := atomic.LoadPointer(&s.head)
        newHead.next = oldHead
        if atomic.CompareAndSwapPointer(&s.head, oldHead, unsafe.Pointer(newHead)) {
            return
        }
    }
}
```

#### **Advanced Channel Patterns**

```go
// Fan-out/Fan-in pattern for processing large datasets
func fanOutFanIn(input <-chan int, numWorkers int) <-chan int {
    // Fan-out: Distribute work to multiple workers
    workers := make([]<-chan int, numWorkers)
    for i := 0; i < numWorkers; i++ {
        workers[i] = worker(input)
    }

    // Fan-in: Collect results from all workers
    return fanIn(workers...)
}

// Pipeline pattern for data processing
func createPipeline(input <-chan int) <-chan int {
    stage1 := stage1(input)
    stage2 := stage2(stage1)
    stage3 := stage3(stage2)
    return stage3
}
```

---

## 🏗️ System Design at Scale

### **Payment Gateway Architecture (Razorpay-Specific)**

**Detailed Explanation:**

A payment gateway is a critical financial infrastructure that processes transactions between merchants and customers. The architecture must handle high throughput, ensure security, maintain data consistency, and provide real-time processing capabilities.

**Key Design Principles:**

1. **High Availability**: 99.99% uptime is critical for financial services
2. **Security**: PCI DSS compliance and end-to-end encryption
3. **Scalability**: Handle millions of transactions per day
4. **Consistency**: ACID properties for financial data
5. **Real-time Processing**: Low latency for user experience
6. **Fault Tolerance**: Graceful degradation and recovery

**Architecture Components:**

**Client Layer:**

- Mobile apps, web applications, and merchant dashboards
- Handles user authentication and transaction initiation
- Implements secure communication protocols

**API Gateway:**

- Single entry point for all client requests
- Rate limiting to prevent abuse
- Authentication and authorization
- Request routing and load balancing
- API versioning and documentation

**Microservices Layer:**

- **Payment Service**: Core transaction processing
- **User Service**: Customer and merchant management
- **Notification Service**: Real-time updates and alerts
- **Fraud Detection Service**: Risk assessment and prevention
- **Settlement Service**: T+1 settlement processing

**Data Layer:**

- **MySQL**: Primary database for transactional data
- **Redis**: Caching layer for frequently accessed data
- **Kafka**: Message queue for event streaming
- **Data Warehouse**: Analytics and reporting

#### **High-Level Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Web Portal    │    │   Admin Panel   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      API Gateway          │
                    │  (Rate Limiting, Auth)    │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │    Microservices Layer    │
                    │  ┌─────┐ ┌─────┐ ┌─────┐  │
                    │  │Pay  │ │User │ │Notif│  │
                    │  │Svc  │ │Svc  │ │Svc  │  │
                    │  └─────┘ └─────┘ └─────┘  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     Data Layer            │
                    │  ┌─────┐ ┌─────┐ ┌─────┐  │
                    │  │MySQL│ │Redis│ │Kafka│  │
                    │  └─────┘ └─────┘ └─────┘  │
                    └───────────────────────────┘
```

**Discussion Questions & Answers:**

**Q1: How would you handle a sudden spike in transaction volume (e.g., during Black Friday)?**

**Answer:** Several strategies to handle traffic spikes:

1. **Auto-scaling**: Implement horizontal pod autoscaling based on CPU/memory metrics
2. **Circuit Breakers**: Prevent cascade failures by temporarily disabling failing services
3. **Rate Limiting**: Implement per-merchant rate limits to ensure fair resource allocation
4. **Caching**: Use Redis for frequently accessed merchant and customer data
5. **Database Optimization**: Read replicas for read-heavy operations, connection pooling
6. **CDN**: Cache static content and API responses where appropriate
7. **Queue Management**: Use Kafka to buffer requests during peak times

**Q2: How do you ensure data consistency across multiple microservices?**

**Answer:** Use the Saga pattern for distributed transactions:

1. **Choreography**: Each service publishes events that other services consume
2. **Orchestration**: A central orchestrator coordinates the transaction flow
3. **Compensation**: Implement compensating actions for rollback scenarios
4. **Event Sourcing**: Store all events for audit and replay capabilities
5. **Idempotency**: Ensure operations can be safely retried
6. **Eventual Consistency**: Accept temporary inconsistency for better performance

**Q3: What security measures would you implement for a payment gateway?**

**Answer:** Comprehensive security strategy:

1. **Encryption**: TLS 1.3 for data in transit, AES-256 for data at rest
2. **PCI DSS Compliance**: Secure handling of card data, tokenization
3. **Authentication**: Multi-factor authentication, OAuth 2.0, JWT tokens
4. **Authorization**: Role-based access control (RBAC), API key management
5. **Fraud Detection**: Real-time ML models, behavioral analysis
6. **Audit Logging**: Complete audit trail for all transactions
7. **Network Security**: VPC, private subnets, security groups
8. **Secrets Management**: HashiCorp Vault or AWS Secrets Manager

**Q4: How would you design the database schema for a payment gateway?**

**Answer:** Database design considerations:

1. **Normalization**: Separate tables for users, merchants, transactions, settlements
2. **Partitioning**: Partition transaction table by date or merchant ID
3. **Indexing**: Composite indexes on frequently queried columns
4. **Sharding**: Shard by merchant ID for horizontal scaling
5. **Read Replicas**: Separate read and write operations
6. **Data Retention**: Archive old transactions to cold storage
7. **Backup Strategy**: Point-in-time recovery, cross-region replication

**Q5: How do you handle payment failures and retries?**

**Answer:** Robust failure handling:

1. **Exponential Backoff**: Gradually increase retry intervals
2. **Circuit Breaker**: Stop retrying after consecutive failures
3. **Dead Letter Queue**: Store permanently failed transactions
4. **Manual Review**: Flag suspicious patterns for human review
5. **Webhook Notifications**: Inform merchants of failure reasons
6. **Compensation**: Implement compensating transactions for partial failures
7. **Monitoring**: Real-time alerts for failure rate spikes

#### **Payment Processing Service Design**

```go
// Core payment processing service
type PaymentService struct {
    db          *sql.DB
    cache       *redis.Client
    queue       *kafka.Producer
    bankClient  *BankAPIClient
    riskEngine  *RiskEngine
}

type PaymentRequest struct {
    ID          string    `json:"id"`
    Amount      int64     `json:"amount"`
    Currency    string    `json:"currency"`
    Method      string    `json:"method"`
    CustomerID  string    `json:"customer_id"`
    MerchantID  string    `json:"merchant_id"`
    Timestamp   time.Time `json:"timestamp"`
}

func (ps *PaymentService) ProcessPayment(ctx context.Context, req *PaymentRequest) (*PaymentResponse, error) {
    // 1. Validate request
    if err := ps.validateRequest(req); err != nil {
        return nil, err
    }

    // 2. Risk assessment
    riskScore, err := ps.riskEngine.AssessRisk(req)
    if err != nil {
        return nil, err
    }

    // 3. Check limits and compliance
    if err := ps.checkLimits(req, riskScore); err != nil {
        return nil, err
    }

    // 4. Process payment based on method
    switch req.Method {
    case "card":
        return ps.processCardPayment(ctx, req)
    case "upi":
        return ps.processUPIPayment(ctx, req)
    case "netbanking":
        return ps.processNetBankingPayment(ctx, req)
    default:
        return nil, ErrUnsupportedPaymentMethod
    }
}
```

#### **Database Sharding Strategy**

```go
// Sharding strategy for payment data
type ShardingStrategy struct {
    shards map[string]*sql.DB
    hashFunc func(string) string
}

func (ss *ShardingStrategy) GetShard(key string) *sql.DB {
    shardKey := ss.hashFunc(key)
    return ss.shards[shardKey]
}

// Consistent hashing for shard distribution
func consistentHash(key string, shards []string) string {
    hash := fnv.New32a()
    hash.Write([]byte(key))
    hashValue := hash.Sum32()

    shardIndex := hashValue % uint32(len(shards))
    return shards[shardIndex]
}
```

### **Real-Time Risk Management System**

#### **Stream Processing Architecture**

```go
// Real-time risk assessment using stream processing
type RiskAssessmentService struct {
    streamProcessor *kafka.StreamProcessor
    mlModel        *MLModel
    rulesEngine    *RulesEngine
}

type RiskEvent struct {
    PaymentID    string            `json:"payment_id"`
    CustomerID   string            `json:"customer_id"`
    Amount       int64             `json:"amount"`
    Timestamp    time.Time         `json:"timestamp"`
    Features     map[string]float64 `json:"features"`
}

func (ras *RiskAssessmentService) ProcessRiskEvent(event *RiskEvent) (*RiskScore, error) {
    // 1. Extract features
    features := ras.extractFeatures(event)

    // 2. Apply ML model
    mlScore, err := ras.mlModel.Predict(features)
    if err != nil {
        return nil, err
    }

    // 3. Apply business rules
    ruleScore := ras.rulesEngine.Evaluate(event)

    // 4. Combine scores
    finalScore := ras.combineScores(mlScore, ruleScore)

    return &RiskScore{
        Score:     finalScore,
        Reason:    ras.getReason(finalScore),
        Timestamp: time.Now(),
    }, nil
}
```

---

## 💻 Operating System Deep Dive

### **Memory Management**

#### **Virtual Memory and Paging**

```go
// Understanding memory management in Go applications
package main

import (
    "runtime"
    "runtime/debug"
    "unsafe"
)

func demonstrateMemoryManagement() {
    // Go's memory management features:
    // 1. Garbage collection (concurrent, low-latency)
    // 2. Memory pooling
    // 3. Stack vs heap allocation

    // Force garbage collection
    runtime.GC()

    // Get memory statistics
    var m runtime.MemStats
    runtime.ReadMemStats(&m)

    fmt.Printf("Alloc = %d KB\n", m.Alloc/1024)
    fmt.Printf("TotalAlloc = %d KB\n", m.TotalAlloc/1024)
    fmt.Printf("Sys = %d KB\n", m.Sys/1024)
    fmt.Printf("NumGC = %d\n", m.NumGC)

    // Memory pooling for high-performance scenarios
    pool := &sync.Pool{
        New: func() interface{} {
            return make([]byte, 1024)
        },
    }

    // Use pooled memory
    buf := pool.Get().([]byte)
    defer pool.Put(buf)
}
```

#### **Process Scheduling**

```go
// Understanding OS-level process scheduling
// Go's runtime scheduler vs OS scheduler

type ProcessScheduling struct {
    // OS Level:
    // - Preemptive scheduling
    // - Time quantum (typically 10-100ms)
    // - Priority-based scheduling
    // - Context switching overhead

    // Go Runtime Level:
    // - Cooperative scheduling
    // - Goroutine preemption points
    // - Work stealing
    // - M:N threading model
}

// Demonstrating goroutine preemption
func demonstratePreemption() {
    // Go 1.14+ has preemptive scheduling
    // Goroutines can be preempted at function calls

    go func() {
        for {
            // This loop can be preempted
            runtime.Gosched() // Voluntary yield
        }
    }()

    // CPU-intensive work without preemption points
    go func() {
        for i := 0; i < 1000000000; i++ {
            // This might not be preempted in older Go versions
            // Go 1.14+ uses signal-based preemption
        }
    }()
}
```

### **I/O Operations and System Calls**

#### **Non-blocking I/O**

```go
// Understanding I/O operations in Go
package main

import (
    "context"
    "net"
    "syscall"
)

func demonstrateNonBlockingIO() {
    // Go's netpoller handles non-blocking I/O
    // Uses epoll (Linux), kqueue (BSD), or IOCP (Windows)

    listener, err := net.Listen("tcp", ":8080")
    if err != nil {
        panic(err)
    }

    for {
        conn, err := listener.Accept()
        if err != nil {
            continue
        }

        // Handle connection in goroutine
        go handleConnection(conn)
    }
}

// Context-based cancellation
func handleConnectionWithContext(ctx context.Context, conn net.Conn) {
    select {
    case <-ctx.Done():
        conn.Close()
        return
    default:
        // Process connection
    }
}
```

#### **System Call Optimization**

```go
// Minimizing system call overhead
type SystemCallOptimizer struct {
    // Batch operations to reduce syscall overhead
    batchSize int
    buffer    []byte
}

func (sco *SystemCallOptimizer) WriteBatch(data []byte) error {
    // Batch multiple writes into single syscall
    if len(sco.buffer)+len(data) > sco.batchSize {
        if err := sco.flush(); err != nil {
            return err
        }
    }

    sco.buffer = append(sco.buffer, data...)
    return nil
}

func (sco *SystemCallOptimizer) flush() error {
    // Single write syscall for batched data
    _, err := syscall.Write(1, sco.buffer)
    sco.buffer = sco.buffer[:0]
    return err
}
```

---

## 🎯 Razorpay-Specific Technical Challenges

### **Payment Gateway Scalability**

#### **Handling Peak Load**

```go
// Strategies for handling payment gateway peak loads
type PaymentGatewayScaler struct {
    // 1. Horizontal scaling
    loadBalancer *LoadBalancer
    instances    []*PaymentInstance

    // 2. Circuit breakers
    circuitBreakers map[string]*CircuitBreaker

    // 3. Rate limiting
    rateLimiters map[string]*RateLimiter

    // 4. Caching layers
    cache *redis.ClusterClient
}

func (pgs *PaymentGatewayScaler) HandlePaymentRequest(req *PaymentRequest) (*PaymentResponse, error) {
    // 1. Rate limiting
    if !pgs.rateLimiters[req.MerchantID].Allow() {
        return nil, ErrRateLimitExceeded
    }

    // 2. Circuit breaker check
    if pgs.circuitBreakers[req.Method].IsOpen() {
        return nil, ErrServiceUnavailable
    }

    // 3. Load balancing
    instance := pgs.loadBalancer.GetInstance()

    // 4. Process payment
    return instance.ProcessPayment(req)
}
```

#### **Data Consistency in Distributed Systems**

```go
// Implementing distributed transactions for payment processing
type DistributedTransaction struct {
    coordinator *TransactionCoordinator
    participants []*TransactionParticipant
}

// Two-Phase Commit for payment processing
func (dt *DistributedTransaction) ExecutePayment(payment *Payment) error {
    // Phase 1: Prepare
    for _, participant := range dt.participants {
        if err := participant.Prepare(payment); err != nil {
            dt.Abort()
            return err
        }
    }

    // Phase 2: Commit
    for _, participant := range dt.participants {
        if err := participant.Commit(payment); err != nil {
            // Handle partial failure
            dt.Compensate()
            return err
        }
    }

    return nil
}
```

### **Fraud Detection System**

#### **Real-Time ML Pipeline**

```go
// Real-time fraud detection using ML
type FraudDetectionService struct {
    featureExtractor *FeatureExtractor
    mlModel         *MLModel
    rulesEngine     *RulesEngine
    streamProcessor *StreamProcessor
}

type FraudFeatures struct {
    Amount          float64
    TimeOfDay       int
    DayOfWeek       int
    CustomerHistory map[string]float64
    DeviceFingerprint string
    Location        *GeoLocation
}

func (fds *FraudDetectionService) DetectFraud(payment *Payment) (*FraudScore, error) {
    // 1. Extract features in real-time
    features, err := fds.featureExtractor.Extract(payment)
    if err != nil {
        return nil, err
    }

    // 2. Apply ML model
    mlScore, err := fds.mlModel.Predict(features)
    if err != nil {
        return nil, err
    }

    // 3. Apply business rules
    ruleScore := fds.rulesEngine.Evaluate(payment, features)

    // 4. Combine scores
    finalScore := fds.combineScores(mlScore, ruleScore)

    return &FraudScore{
        Score:     finalScore,
        RiskLevel: fds.getRiskLevel(finalScore),
        Reason:    fds.getReason(finalScore),
    }, nil
}
```

---

## 🧮 Advanced DSA & Algorithm Patterns

### **Advanced Graph Algorithms**

#### **Minimum Spanning Tree for Network Optimization**

```go
// Kruskal's algorithm for network optimization
type Edge struct {
    From   int
    To     int
    Weight int
}

type UnionFind struct {
    parent []int
    rank   []int
}

func (uf *UnionFind) Find(x int) int {
    if uf.parent[x] != x {
        uf.parent[x] = uf.Find(uf.parent[x])
    }
    return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) {
    px, py := uf.Find(x), uf.Find(y)
    if px == py {
        return
    }

    if uf.rank[px] < uf.rank[py] {
        uf.parent[px] = py
    } else if uf.rank[px] > uf.rank[py] {
        uf.parent[py] = px
    } else {
        uf.parent[py] = px
        uf.rank[px]++
    }
}

func kruskalMST(edges []Edge, numVertices int) []Edge {
    sort.Slice(edges, func(i, j int) bool {
        return edges[i].Weight < edges[j].Weight
    })

    uf := &UnionFind{
        parent: make([]int, numVertices),
        rank:   make([]int, numVertices),
    }

    for i := 0; i < numVertices; i++ {
        uf.parent[i] = i
    }

    var mst []Edge
    for _, edge := range edges {
        if uf.Find(edge.From) != uf.Find(edge.To) {
            mst = append(mst, edge)
            uf.Union(edge.From, edge.To)
        }
    }

    return mst
}
```

#### **Dynamic Programming for Optimization**

```go
// Advanced DP: Longest Common Subsequence with space optimization
func lcsOptimized(s1, s2 string) int {
    m, n := len(s1), len(s2)

    // Space-optimized DP using only 2 rows
    prev := make([]int, n+1)
    curr := make([]int, n+1)

    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                curr[j] = prev[j-1] + 1
            } else {
                curr[j] = max(prev[j], curr[j-1])
            }
        }
        prev, curr = curr, prev
    }

    return prev[n]
}

// Knapsack with multiple constraints
func knapsackMultipleConstraints(weights, values []int, capacity int, maxItems int) int {
    n := len(weights)

    // DP[i][w][k] = max value with first i items, weight w, k items
    dp := make([][][]int, n+1)
    for i := range dp {
        dp[i] = make([][]int, capacity+1)
        for j := range dp[i] {
            dp[i][j] = make([]int, maxItems+1)
        }
    }

    for i := 1; i <= n; i++ {
        for w := 0; w <= capacity; w++ {
            for k := 0; k <= maxItems; k++ {
                dp[i][w][k] = dp[i-1][w][k] // Don't take item i

                if w >= weights[i-1] && k > 0 {
                    take := dp[i-1][w-weights[i-1]][k-1] + values[i-1]
                    dp[i][w][k] = max(dp[i][w][k], take)
                }
            }
        }
    }

    return dp[n][capacity][maxItems]
}
```

### **Advanced String Algorithms**

#### **Suffix Array and LCP Array**

```go
// Suffix array construction for string processing
func buildSuffixArray(s string) []int {
    n := len(s)
    suffixes := make([]Suffix, n)

    for i := 0; i < n; i++ {
        suffixes[i] = Suffix{i, s[i:]}
    }

    sort.Slice(suffixes, func(i, j int) bool {
        return suffixes[i].suffix < suffixes[j].suffix
    })

    result := make([]int, n)
    for i, suffix := range suffixes {
        result[i] = suffix.index
    }

    return result
}

type Suffix struct {
    index  int
    suffix string
}

// Longest Common Prefix array
func buildLCPArray(s string, suffixArray []int) []int {
    n := len(s)
    lcp := make([]int, n)
    rank := make([]int, n)

    for i := 0; i < n; i++ {
        rank[suffixArray[i]] = i
    }

    k := 0
    for i := 0; i < n; i++ {
        if rank[i] == n-1 {
            k = 0
            continue
        }

        j := suffixArray[rank[i]+1]
        for i+k < n && j+k < n && s[i+k] == s[j+k] {
            k++
        }

        lcp[rank[i]] = k
        if k > 0 {
            k--
        }
    }

    return lcp
}
```

---

## ⚡ Performance Engineering

### **Profiling and Optimization**

#### **Go Profiling Tools**

```go
// Comprehensive profiling setup
package main

import (
    _ "net/http/pprof"
    "net/http"
    "runtime"
    "runtime/pprof"
    "os"
)

func setupProfiling() {
    // CPU profiling
    f, err := os.Create("cpu.prof")
    if err != nil {
        panic(err)
    }
    defer f.Close()

    pprof.StartCPUProfile(f)
    defer pprof.StopCPUProfile()

    // Memory profiling
    runtime.GC()
    f2, err := os.Create("mem.prof")
    if err != nil {
        panic(err)
    }
    defer f2.Close()

    pprof.WriteHeapProfile(f2)

    // HTTP profiling endpoint
    go func() {
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()
}

// Benchmarking critical paths
func BenchmarkPaymentProcessing(b *testing.B) {
    service := NewPaymentService()
    payment := &PaymentRequest{
        Amount:   1000,
        Currency: "INR",
        Method:   "card",
    }

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := service.ProcessPayment(context.Background(), payment)
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

#### **Memory Optimization Techniques**

```go
// Memory pooling for high-performance scenarios
type ObjectPool struct {
    pool sync.Pool
    new  func() interface{}
}

func NewObjectPool(newFunc func() interface{}) *ObjectPool {
    return &ObjectPool{
        pool: sync.Pool{New: newFunc},
        new:  newFunc,
    }
}

func (op *ObjectPool) Get() interface{} {
    return op.pool.Get()
}

func (op *ObjectPool) Put(obj interface{}) {
    op.pool.Put(obj)
}

// String interning for memory efficiency
type StringInterner struct {
    strings map[string]string
    mutex   sync.RWMutex
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

    if interned, exists := si.strings[s]; exists {
        return interned
    }

    si.strings[s] = s
    return s
}
```

### **Concurrency Optimization**

#### **Lock-Free Data Structures**

```go
// Lock-free hash table for high-concurrency scenarios
type LockFreeHashMap struct {
    buckets []*atomic.Pointer
    size    int
}

type HashNode struct {
    key   string
    value interface{}
    next  *atomic.Pointer
}

func (lfhm *LockFreeHashMap) Set(key string, value interface{}) {
    hash := lfhm.hash(key)
    bucket := lfhm.buckets[hash%len(lfhm.buckets)]

    newNode := &HashNode{
        key:   key,
        value: value,
        next:  &atomic.Pointer{},
    }

    for {
        oldHead := bucket.Load()
        newNode.next.Store(oldHead)

        if bucket.CompareAndSwap(oldHead, unsafe.Pointer(newNode)) {
            return
        }
    }
}

func (lfhm *LockFreeHashMap) Get(key string) (interface{}, bool) {
    hash := lfhm.hash(key)
    bucket := lfhm.buckets[hash%len(lfhm.buckets)]

    node := (*HashNode)(bucket.Load())
    for node != nil {
        if node.key == key {
            return node.value, true
        }
        node = (*HashNode)(node.next.Load())
    }

    return nil, false
}
```

---

## 🎯 Leadership & Architecture Decisions

### **Technical Leadership Scenarios**

#### **System Migration Strategy**

```go
// Leading a complex system migration
type MigrationStrategy struct {
    // 1. Assessment phase
    currentSystem *LegacySystem
    targetSystem  *ModernSystem

    // 2. Risk mitigation
    rollbackPlan   *RollbackPlan
    monitoring     *MigrationMonitoring

    // 3. Phased approach
    phases []MigrationPhase
}

type MigrationPhase struct {
    Name        string
    Duration    time.Duration
    RiskLevel   RiskLevel
    Dependencies []string
    RollbackPlan *RollbackPlan
}

func (ms *MigrationStrategy) ExecuteMigration() error {
    for _, phase := range ms.phases {
        // 1. Pre-migration checks
        if err := ms.preMigrationChecks(phase); err != nil {
            return err
        }

        // 2. Execute phase
        if err := ms.executePhase(phase); err != nil {
            // 3. Rollback if needed
            if err := ms.rollback(phase); err != nil {
                return fmt.Errorf("migration and rollback failed: %v", err)
            }
            return err
        }

        // 4. Post-migration validation
        if err := ms.postMigrationValidation(phase); err != nil {
            return err
        }
    }

    return nil
}
```

#### **Team Mentoring and Code Review**

```go
// Code review framework for technical leadership
type CodeReviewFramework struct {
    // 1. Technical aspects
    codeQuality    *CodeQualityMetrics
    performance    *PerformanceMetrics
    security       *SecurityChecklist

    // 2. Architecture aspects
    designPatterns *DesignPatternReview
    scalability    *ScalabilityReview

    // 3. Team aspects
    knowledgeSharing *KnowledgeSharingPlan
    mentoring       *MentoringPlan
}

type CodeQualityMetrics struct {
    CyclomaticComplexity int
    TestCoverage        float64
    CodeDuplication     float64
    MaintainabilityIndex float64
}

func (crf *CodeReviewFramework) ReviewCode(pr *PullRequest) *ReviewResult {
    result := &ReviewResult{
        TechnicalScore:  crf.evaluateTechnicalAspects(pr),
        ArchitectureScore: crf.evaluateArchitecture(pr),
        TeamImpact:     crf.evaluateTeamImpact(pr),
    }

    // Provide constructive feedback
    result.Feedback = crf.generateFeedback(result)

    return result
}
```

### **Architecture Decision Records (ADRs)**

#### **Technology Selection Framework**

```go
// Framework for making architecture decisions
type ArchitectureDecision struct {
    ID          string
    Title       string
    Status      DecisionStatus
    Context     string
    Decision    string
    Consequences []string
    Alternatives []string
    TradeOffs   []TradeOff
}

type TradeOff struct {
    Aspect     string
    Pros       []string
    Cons       []string
    Impact     ImpactLevel
    Mitigation string
}

// Example: Choosing between microservices and monolith
func evaluateMicroservicesVsMonolith() *ArchitectureDecision {
    return &ArchitectureDecision{
        ID:     "ADR-001",
        Title:  "Microservices vs Monolith Architecture",
        Status: "Accepted",
        Context: "Need to scale payment processing system",
        Decision: "Adopt microservices architecture",
        Consequences: []string{
            "Improved scalability",
            "Independent deployments",
            "Technology diversity",
            "Increased complexity",
        },
        TradeOffs: []TradeOff{
            {
                Aspect: "Scalability",
                Pros:   []string{"Independent scaling", "Resource optimization"},
                Cons:   []string{"Network overhead", "Distributed complexity"},
                Impact: High,
            },
            {
                Aspect: "Development Speed",
                Pros:   []string{"Parallel development", "Smaller teams"},
                Cons:   []string{"Coordination overhead", "Integration complexity"},
                Impact: Medium,
            },
        },
    }
}
```

---

## 📚 Additional Resources

### **Advanced Go Resources**

- [Go Memory Model](https://golang.org/ref/mem/)
- [Go Scheduler Design](https://docs.google.com/document/d/1TTj4T2JO42uD5ID9e89oa0sLKhJYD0Y_kqxDv3I3XMw/edit/)
- [Go Performance Optimization](https://github.com/golang/go/wiki/Performance/)

### **System Design Resources**

- [Designing Data-Intensive Applications](https://dataintensive.net/)
- [High Scalability Case Studies](https://highscalability.com/)
- [Microservices Patterns](https://microservices.io/)

### **Interview Preparation**

- [System Design Interview](https://github.com/checkcheckzz/system-design-interview/)
- [LeetCode Advanced Problems](https://leetcode.com/problemset/all/?difficulty=HARD/)
- [Go Interview Questions](https://github.com/kevinyan815/gocookbook/)

---

_This guide provides comprehensive preparation for advanced backend engineering roles at Razorpay, covering deep technical knowledge, system design expertise, and leadership capabilities._
