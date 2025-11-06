---
# Auto-generated front matter
Title: Razorpay Practice Scenarios
LastUpdated: 2025-11-06T20:45:58.498260
Tags: []
Status: draft
---

# 🎯 Razorpay Practice Scenarios & Solutions

> **Focused practice scenarios for Round 2 and Round 3 with detailed solutions**

## 🏗️ **Round 2: System Design Scenarios**

### **Scenario 1: Design Razorpay's Payment Gateway for 1M TPS**

**Question**: "Design a payment gateway that can handle 1 million transactions per second with 99.99% availability and sub-100ms latency."

#### **Solution Framework**

**1. Requirements Analysis**

```
Throughput: 1M TPS
Availability: 99.99% (52.56 minutes downtime/year)
Latency: <100ms p99
Consistency: Eventual consistency acceptable
Durability: Strong durability required
```

**2. Capacity Planning**

```go
type SystemRequirements struct {
    Throughput    int     // 1M TPS
    Availability  float64 // 99.99%
    Latency       int     // <100ms p99
    DataIngestion int64   // 1GB/s
    Storage       int64   // 86TB/day
    Network       int64   // 10Gbps
    CPU           int     // ~1000 cores
    Memory        int64   // 100GB
}
```

**3. High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer Layer                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │   LB1   │ │   LB2   │ │   LB3   │ │   LB4   │ │   LB5   ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 API Gateway Layer                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │   GW1   │ │   GW2   │ │   GW3   │ │   GW4   │ │   GW5   ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│              Microservices Layer                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │Payment  │ │Fraud    │ │Risk     │ │Settlement│ │Notification││
│  │Service  │ │Detection│ │Engine   │ │Service  │ │Service  ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 Data Layer                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │MySQL    │ │Redis    │ │Kafka    │ │Elastic  │ │S3       ││
│  │Cluster  │ │Cluster  │ │Cluster  │ │Search   │ │Storage  ││
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│
└─────────────────────────────────────────────────────────────┘
```

**4. Payment Service Implementation**

```go
type PaymentService struct {
    db          *sql.DB
    cache       *redis.ClusterClient
    queue       *kafka.Producer
    bankClient  *BankAPIClient
    circuitBreaker *CircuitBreaker
    rateLimiter    *RateLimiter
    metrics    *MetricsCollector
}

func (ps *PaymentService) ProcessPayment(ctx context.Context, req *PaymentRequest) (*PaymentResponse, error) {
    // 1. Input validation (1ms)
    if err := ps.validateRequest(req); err != nil {
        return nil, err
    }

    // 2. Rate limiting (1ms)
    if !ps.rateLimiter.Allow(req.MerchantID) {
        return nil, ErrRateLimitExceeded
    }

    // 3. Circuit breaker check (1ms)
    if ps.circuitBreaker.IsOpen(req.Method) {
        return nil, ErrServiceUnavailable
    }

    // 4. Cache lookup (5ms)
    if cached, err := ps.cache.Get(req.ID); err == nil {
        return cached, nil
    }

    // 5. Database transaction (20ms)
    tx, err := ps.db.BeginTx(ctx, nil)
    if err != nil {
        return nil, err
    }
    defer tx.Rollback()

    // 6. Bank API call (50ms)
    bankResp, err := ps.bankClient.ProcessPayment(ctx, req)
    if err != nil {
        return nil, err
    }

    // 7. Update database (10ms)
    if err := ps.updateTransaction(tx, req, bankResp); err != nil {
        return nil, err
    }

    // 8. Commit transaction (5ms)
    if err := tx.Commit(); err != nil {
        return nil, err
    }

    // 9. Cache result (2ms)
    ps.cache.Set(req.ID, bankResp, time.Hour)

    // 10. Async notification (1ms)
    ps.queue.ProduceAsync(&NotificationEvent{
        PaymentID: req.ID,
        Status:    bankResp.Status,
    })

    return bankResp, nil
}
```

**5. Database Sharding Strategy**

```go
type ShardingStrategy struct {
    shards map[string]*sql.DB
    hashFunc func(string) string
}

func (ss *ShardingStrategy) GetShard(merchantID string) *sql.DB {
    // Consistent hashing for even distribution
    shardKey := ss.hashFunc(merchantID)
    return ss.shards[shardKey]
}

func selectShardKey(payment *PaymentRequest) string {
    // Use merchant ID for even distribution
    return payment.MerchantID
}
```

**6. Multi-Level Caching**

```go
type MultiLevelCache struct {
    l1Cache *sync.Map        // In-memory cache (1ms access)
    l2Cache *redis.Client    // Redis cache (5ms access)
    l3Cache *DatabaseCache   // Database cache (20ms access)
}

func (mlc *MultiLevelCache) Get(key string) (interface{}, error) {
    // L1 cache lookup
    if value, ok := mlc.l1Cache.Load(key); ok {
        return value, nil
    }

    // L2 cache lookup
    if value, err := mlc.l2Cache.Get(key).Result(); err == nil {
        mlc.l1Cache.Store(key, value)
        return value, nil
    }

    // L3 cache lookup
    if value, err := mlc.l3Cache.Get(key); err == nil {
        mlc.l2Cache.Set(key, value, time.Hour)
        mlc.l1Cache.Store(key, value)
        return value, nil
    }

    return nil, ErrNotFound
}
```

### **Scenario 2: Design Real-Time Fraud Detection System**

**Question**: "Design a fraud detection system that can process 100K transactions per second with sub-10ms latency for risk scoring."

#### **Solution Framework**

**1. System Architecture**

```go
type FraudDetectionSystem struct {
    streamProcessor *kafka.StreamProcessor
    featureExtractor *FeatureExtractor
    mlModel         *MLModel
    rulesEngine     *RulesEngine
    batchProcessor  *BatchProcessor
    modelTrainer    *ModelTrainer
    featureStore    *FeatureStore
    modelStore      *ModelStore
}

func (fds *FraudDetectionSystem) ProcessTransaction(tx *Transaction) (*FraudScore, error) {
    // 1. Feature extraction (2ms)
    features, err := fds.featureExtractor.ExtractRealTime(tx)
    if err != nil {
        return nil, err
    }

    // 2. ML model prediction (3ms)
    mlScore, err := fds.mlModel.Predict(features)
    if err != nil {
        return nil, err
    }

    // 3. Rules engine evaluation (2ms)
    ruleScore := fds.rulesEngine.Evaluate(tx, features)

    // 4. Score combination (1ms)
    finalScore := fds.combineScores(mlScore, ruleScore)

    // 5. Decision (1ms)
    decision := fds.makeDecision(finalScore)

    return &FraudScore{
        Score:    finalScore,
        Decision: decision,
        Latency:  time.Since(tx.Timestamp),
    }, nil
}
```

**2. Feature Engineering**

```go
type FeatureExtractor struct {
    realTimeFeatures map[string]FeatureCalculator
    historicalFeatures map[string]FeatureCalculator
    externalFeatures map[string]FeatureCalculator
}

type FeatureCalculator interface {
    Calculate(tx *Transaction) (float64, error)
    GetLatency() time.Duration
}

func (fe *FeatureExtractor) ExtractRealTime(tx *Transaction) (*Features, error) {
    features := &Features{
        RealTime: make(map[string]float64),
    }

    for name, calculator := range fe.realTimeFeatures {
        value, err := calculator.Calculate(tx)
        if err != nil {
            return nil, err
        }
        features.RealTime[name] = value
    }

    return features, nil
}
```

---

## 🔧 **Round 3: Go Runtime Deep Dive**

### **Question 1: Go Scheduler with Extreme Concurrency**

**Question**: "What happens when you create trillions of goroutines in Go? How does the scheduler behave?"

#### **Solution**

**1. Memory Impact Analysis**

```go
func calculateGoroutineMemory() {
    // Each goroutine starts with 2KB stack
    goroutineSize := 2 * 1024 // 2KB

    // Trillions of goroutines
    numGoroutines := 1_000_000_000_000 // 1 trillion

    // Total memory required
    totalMemory := goroutineSize * numGoroutines
    totalMemoryGB := totalMemory / (1024 * 1024 * 1024)

    fmt.Printf("Memory required: %d GB\n", totalMemoryGB)
    // Output: Memory required: 1862645 GB (1.8 TB)
}
```

**2. Go Scheduler Model**

```go
type SchedulerLimitations struct {
    // M:N scheduler model
    // M = OS threads (limited by GOMAXPROCS)
    // N = Goroutines (can be millions/billions)
    // P = Logical processors (context for scheduling)

    MaxOSThreads    int // Typically 1-8 per CPU core
    MaxGoroutines   int // Practically unlimited
    MaxLogicalProcs int // Equal to GOMAXPROCS
}

func demonstrateSchedulerBehavior() {
    // With trillions of goroutines:

    // 1. Memory pressure
    // - Each goroutine consumes 2KB initially
    // - Stack growth can cause memory fragmentation
    // - GC pressure increases exponentially

    // 2. Scheduling overhead
    // - Context switching between goroutines
    // - Work stealing becomes less efficient
    // - Lock contention on global structures

    // 3. Performance degradation
    // - Cache locality degradation
    // - Increased GC pause times
    // - Reduced throughput
}
```

**3. Work Stealing Algorithm**

```go
type WorkStealingScheduler struct {
    runQueues []*RunQueue
    globalQueue *GlobalQueue
    networkPoller *NetworkPoller
}

func (wss *WorkStealingScheduler) StealWork() {
    // 1. Local run queue (256 goroutines max per P)
    // 2. Global run queue (unlimited but slower access)
    // 3. Network poller (handles I/O operations)
    // 4. Work stealing from other P's queues

    // With trillions of goroutines:
    // - Local queues overflow to global queue
    // - Global queue becomes bottleneck
    // - Work stealing becomes less efficient
    // - Context switching overhead increases
}
```

**4. Performance Bottlenecks**

```go
type PerformanceBottlenecks struct {
    // Memory issues
    MemoryPressure    bool
    StackFragmentation bool
    GCPressure        bool

    // Scheduling issues
    ContextSwitchingOverhead bool
    WorkStealingInefficiency bool
    LockContention          bool

    // System issues
    CacheLocalityDegradation bool
    ReducedThroughput        bool
}

// Solutions for handling extreme concurrency
func handleExtremeConcurrency() {
    // 1. Use worker pools instead of unlimited goroutines
    workerPool := NewWorkerPool(1000) // Limit to 1000 workers

    // 2. Implement backpressure mechanisms
    semaphore := make(chan struct{}, 1000)

    // 3. Use buffered channels for communication
    jobQueue := make(chan Job, 10000)

    // 4. Implement circuit breakers
    circuitBreaker := NewCircuitBreaker(100, time.Second)

    // 5. Use connection pooling
    connectionPool := NewConnectionPool(100)
}
```

### **Question 2: Go Memory Management Optimization**

**Question**: "Explain Go's memory management, garbage collection, and how to optimize for high-performance applications."

#### **Solution**

**1. Go Memory Model**

```go
type GoMemoryModel struct {
    // Stack allocation
    StackSize    int // 2KB initial, grows as needed
    StackGrowth  int // 2x growth factor

    // Heap allocation
    HeapSize     int // Managed by GC
    HeapGrowth   int // 2x growth factor

    // Garbage collection
    GCAlgorithm  string // Concurrent, tri-color mark and sweep
    GCPause      time.Duration // Target <10ms
    GCTrigger    int // When heap size doubles
}

func demonstrateMemoryAllocation() {
    // Stack allocation (fast)
    var stackVar int = 42

    // Heap allocation (slower)
    heapVar := new(int)
    *heapVar = 42

    // Escape analysis
    // Go compiler determines if variable escapes to heap
    // Variables that escape are allocated on heap
}
```

**2. GC Optimization Strategies**

```go
type GCOptimization struct {
    // Reduce allocations
    ObjectPooling    bool
    StringInterning  bool
    SliceReuse       bool

    // Reduce GC pressure
    ReducePointers   bool
    UseValueTypes    bool
    AvoidReflection  bool

    // Tune GC parameters
    GOGC             int // Default 100
    GOMEMLIMIT       int // Memory limit
    GOMAXPROCS       int // CPU cores
}

// Object pooling for high-performance scenarios
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

**3. Memory Profiling**

```go
func setupMemoryProfiling() {
    // Enable memory profiling
    runtime.MemProfileRate = 1

    // Force garbage collection
    runtime.GC()

    // Get memory statistics
    var m runtime.MemStats
    runtime.ReadMemStats(&m)

    fmt.Printf("Alloc = %d KB\n", m.Alloc/1024)
    fmt.Printf("TotalAlloc = %d KB\n", m.TotalAlloc/1024)
    fmt.Printf("Sys = %d KB\n", m.Sys/1024)
    fmt.Printf("NumGC = %d\n", m.NumGC)
    fmt.Printf("GCCPUFraction = %f\n", m.GCCPUFraction)

    // Write heap profile
    f, err := os.Create("heap.prof")
    if err != nil {
        panic(err)
    }
    defer f.Close()

    pprof.WriteHeapProfile(f)
}
```

---

## 🎯 **Round 3: Leadership & Architecture Scenarios**

### **Scenario 1: Leading Complex System Migration**

**Question**: "You need to migrate a monolithic payment system to microservices. How would you approach this as a technical lead?"

#### **Solution Framework**

**1. Migration Strategy**

```go
type MigrationStrategy struct {
    currentSystem *LegacySystem
    targetSystem  *MicroservicesSystem
    rollbackPlan   *RollbackPlan
    monitoring     *MigrationMonitoring
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

**2. Team Management**

```go
type TeamManagement struct {
    architects    []*Architect
    developers    []*Developer
    testers       []*Tester
    devops        []*DevOps
    dailyStandups bool
    weeklyReviews bool
    monthlyRetros bool
    riskRegister  *RiskRegister
    mitigationPlans map[string]*MitigationPlan
}

func (tm *TeamManagement) CoordinateMigration() error {
    // 1. Define roles and responsibilities
    tm.defineRoles()

    // 2. Establish communication channels
    tm.establishCommunication()

    // 3. Set up monitoring and alerting
    tm.setupMonitoring()

    // 4. Implement risk management
    tm.implementRiskManagement()

    return nil
}
```

### **Scenario 2: Handling Production Incidents**

**Question**: "A critical payment processing service is down. How would you handle this incident as a technical lead?"

#### **Solution Framework**

**1. Incident Response Framework**

```go
type IncidentResponse struct {
    severity    IncidentSeverity
    impact      IncidentImpact
    category    IncidentCategory
    incidentCommander *IncidentCommander
    technicalLead     *TechnicalLead
    communications    *CommunicationsLead
    detectionTime     time.Time
    responseTime      time.Time
    resolutionTime    time.Time
    postMortemTime    time.Time
}

func (ir *IncidentResponse) HandleIncident() error {
    // 1. Immediate response
    if err := ir.immediateResponse(); err != nil {
        return err
    }

    // 2. Assessment and classification
    if err := ir.assessIncident(); err != nil {
        return err
    }

    // 3. Communication
    if err := ir.communicateIncident(); err != nil {
        return err
    }

    // 4. Resolution
    if err := ir.resolveIncident(); err != nil {
        return err
    }

    // 5. Post-mortem
    if err := ir.postMortem(); err != nil {
        return err
    }

    return nil
}
```

**2. Technical Troubleshooting**

```go
type TechnicalTroubleshooting struct {
    logs        *LogAnalyzer
    metrics     *MetricsAnalyzer
    traces      *TraceAnalyzer
    databaseIssues    bool
    networkIssues     bool
    applicationIssues bool
    infrastructureIssues bool
}

func (tt *TechnicalTroubleshooting) Troubleshoot() error {
    // 1. Check system health
    if err := tt.checkSystemHealth(); err != nil {
        return err
    }

    // 2. Analyze logs
    if err := tt.analyzeLogs(); err != nil {
        return err
    }

    // 3. Check metrics
    if err := tt.checkMetrics(); err != nil {
        return err
    }

    // 4. Analyze traces
    if err := tt.analyzeTraces(); err != nil {
        return err
    }

    // 5. Identify root cause
    if err := tt.identifyRootCause(); err != nil {
        return err
    }

    return nil
}
```

---

## 🏦 **Razorpay-Specific Technical Challenges**

### **Challenge 1: UPI Payment Processing**

**Question**: "Design a UPI payment processing system that can handle 10M transactions per day with 99.9% success rate."

#### **Solution**

**1. UPI Architecture**

```go
type UPIPaymentSystem struct {
    upiGateway    *UPIGateway
    bankConnector *BankConnector
    npciConnector *NPCIConnector
    paymentProcessor *PaymentProcessor
    settlementEngine *SettlementEngine
    reconciliation   *Reconciliation
    monitoring    *Monitoring
    alerting      *Alerting
}

func (ups *UPIPaymentSystem) ProcessUPIPayment(req *UPIPaymentRequest) (*UPIPaymentResponse, error) {
    // 1. Validate UPI request
    if err := ups.validateUPIRequest(req); err != nil {
        return nil, err
    }

    // 2. Check payer account
    payerAccount, err := ups.checkPayerAccount(req.PayerUPI)
    if err != nil {
        return nil, err
    }

    // 3. Check payee account
    payeeAccount, err := ups.checkPayeeAccount(req.PayeeUPI)
    if err != nil {
        return nil, err
    }

    // 4. Process payment
    paymentResp, err := ups.processPayment(req, payerAccount, payeeAccount)
    if err != nil {
        return nil, err
    }

    // 5. Update settlement
    if err := ups.updateSettlement(paymentResp); err != nil {
        return nil, err
    }

    return paymentResp, nil
}
```

**2. UPI Compliance**

```go
type UPICompliance struct {
    npciCompliance bool
    encryption     bool
    authentication bool
    authorization  bool
    auditTrail     bool
    reporting      bool
    reconciliation bool
}

func (uc *UPICompliance) ValidateCompliance(req *UPIPaymentRequest) error {
    // 1. NPCI compliance check
    if err := uc.checkNPCICompliance(req); err != nil {
        return err
    }

    // 2. Security validation
    if err := uc.validateSecurity(req); err != nil {
        return err
    }

    // 3. Audit trail creation
    if err := uc.createAuditTrail(req); err != nil {
        return err
    }

    return nil
}
```

### **Challenge 2: Real-Time Settlement System**

**Question**: "Design a real-time settlement system for payment processing with sub-second settlement times."

#### **Solution**

**1. Settlement Architecture**

```go
type SettlementSystem struct {
    settlementEngine *SettlementEngine
    bankConnector    *BankConnector
    ledgerSystem     *LedgerSystem
    batchProcessor   *BatchProcessor
    realTimeProcessor *RealTimeProcessor
    monitoring       *Monitoring
    alerting         *Alerting
}

func (ss *SettlementSystem) ProcessSettlement(settlement *Settlement) error {
    // 1. Validate settlement
    if err := ss.validateSettlement(settlement); err != nil {
        return err
    }

    // 2. Check account balances
    if err := ss.checkAccountBalances(settlement); err != nil {
        return err
    }

    // 3. Process settlement
    if err := ss.processSettlement(settlement); err != nil {
        return err
    }

    // 4. Update ledger
    if err := ss.updateLedger(settlement); err != nil {
        return err
    }

    // 5. Notify parties
    if err := ss.notifyParties(settlement); err != nil {
        return err
    }

    return nil
}
```

**2. Settlement Optimization**

```go
type SettlementOptimization struct {
    batchSize        int
    batchInterval    time.Duration
    realTimeThreshold int64
    realTimeTimeout   time.Duration
    netting          bool
    compression      bool
    parallelProcessing bool
}

func (so *SettlementOptimization) ProcessSettlementOptimized(settlements []*Settlement) error {
    // 1. Netting optimization
    nettedSettlements := so.netSettlements(settlements)

    // 2. Batch processing
    if len(nettedSettlements) > so.batchSize {
        return so.processBatch(nettedSettlements)
    }

    // 3. Real-time processing
    return so.processRealTime(nettedSettlements)
}
```

---

## 🎯 **Key Success Tips**

### **System Design Success Factors**

1. **Start with requirements** - Always clarify functional and non-functional requirements
2. **Think in layers** - API Gateway → Services → Data Layer
3. **Consider scalability** - Horizontal scaling, caching, database sharding
4. **Discuss trade-offs** - Performance vs consistency, availability vs consistency
5. **Be specific** - Use actual numbers, technologies, and patterns

### **Technical Deep Dive Success Factors**

1. **Know your Go** - Runtime internals, memory management, concurrency
2. **Explain reasoning** - Why you made specific technical decisions
3. **Show leadership** - How you led teams and made architecture decisions
4. **Learn from failures** - Discuss challenges and how you overcame them
5. **Stay current** - Know latest trends and best practices

### **Behavioral Success Factors**

1. **Use STAR method** - Situation, Task, Action, Result
2. **Be specific** - Use concrete examples with numbers and outcomes
3. **Show growth** - Demonstrate learning from experiences
4. **Stay positive** - Focus on solutions and improvements
5. **Ask questions** - Show interest in the role and company

---

## 🎬 **Additional System Design Scenarios**

### **Scenario 3: Design BookMyShow (Movie Ticketing Platform)**

**Question**: "Design a movie ticketing platform like BookMyShow that can handle 10M users and 100K concurrent bookings."

#### **Solution Framework**

**1. Requirements Analysis**

```
Daily Active Users: 10M
Peak Concurrent Users: 100K
Movies per day: 50K shows
Theaters: 10K across India
Peak booking rate: 10K bookings/minute
Availability: 99.9% uptime
Latency: <200ms for search, <500ms for booking
```

**2. High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   Web App   │ │  Mobile App │ │  Admin App  │ │  Theater App││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    Load Balancer                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    API Gateway                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                  Microservices Layer                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   User      │ │   Movie     │ │  Theater    │ │  Booking    ││
│  │  Service    │ │  Service    │ │  Service    │ │  Service    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │  Payment    │ │  Search     │ │  Review     │ │  Notification││
│  │  Service    │ │  Service    │ │  Service    │ │  Service    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    Data Layer                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   MySQL     │ │    Redis    │ │   Elastic   │ │   MongoDB   ││
│  │  Cluster    │ │   Cluster   │ │   Search    │ │  (Reviews)  ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**3. Core Booking Service Implementation**

```go
type BookingService struct {
    db              *sql.DB
    cache           *redis.Client
    paymentService  *PaymentService
    seatLockManager *SeatLockManager
    eventBus        *EventBus
}

func (bs *BookingService) CreateBooking(ctx context.Context, req *CreateBookingRequest) (*Booking, error) {
    // 1. Lock seats (prevent double booking)
    lockID, err := bs.seatLockManager.LockSeats(req.ShowID, req.SeatIDs, req.UserID, 10*time.Minute)
    if err != nil {
        return nil, err
    }

    // 2. Create booking
    booking := &Booking{
        ID:          generateUUID(),
        UserID:      req.UserID,
        ShowID:      req.ShowID,
        Seats:       req.Seats,
        TotalAmount: req.TotalAmount,
        Status:      "pending",
        CreatedAt:   time.Now(),
        ExpiresAt:   time.Now().Add(10 * time.Minute),
    }

    // 3. Save booking
    if err := bs.saveBooking(booking); err != nil {
        bs.seatLockManager.UnlockSeats(lockID)
        return nil, err
    }

    // 4. Process payment
    paymentResult, err := bs.paymentService.ProcessPayment(ctx, &PaymentRequest{
        Amount:    booking.TotalAmount,
        BookingID: booking.ID,
    })
    if err != nil {
        bs.seatLockManager.UnlockSeats(lockID)
        booking.Status = "failed"
        bs.updateBooking(booking)
        return nil, err
    }

    // 5. Confirm booking
    booking.Status = "confirmed"
    booking.PaymentID = paymentResult.PaymentID
    bs.updateBooking(booking)
    bs.seatLockManager.UnlockSeats(lockID)

    return booking, nil
}
```

**4. Seat Locking Strategy**

```go
type SeatLockManager struct {
    cache *redis.Client
}

func (slm *SeatLockManager) LockSeats(showID string, seatIDs []string, userID string, duration time.Duration) (string, error) {
    lockID := generateUUID()

    // Mark seats as locked
    for _, seatID := range seatIDs {
        seatKey := fmt.Sprintf("seat:locked:%s:%s", showID, seatID)
        if err := slm.cache.Set(seatKey, userID, duration).Err(); err != nil {
            return "", err
        }
    }

    return lockID, nil
}

func (slm *SeatLockManager) IsSeatLocked(showID, seatID string) (bool, string) {
    seatKey := fmt.Sprintf("seat:locked:%s:%s", showID, seatID)
    userID, err := slm.cache.Get(seatKey).Result()
    if err != nil {
        return false, ""
    }
    return true, userID
}
```

**Key Design Decisions:**

- **Seat Locking**: Redis-based locking to prevent double booking
- **Payment Integration**: Separate payment service with rollback capability
- **Event-Driven**: Async processing for notifications and analytics
- **Caching**: Multi-level caching for performance
- **Search**: Elasticsearch for fast movie and theater search

### **Scenario 4: Design WhatsApp (Messaging Platform)**

**Question**: "Design a messaging platform like WhatsApp that can handle 2B users and 100B messages per day."

#### **Solution Framework**

**1. Requirements Analysis**

```
Total Users: 2B
Daily Active Users: 1.5B
Messages per day: 100B
Peak messages per second: 1M
Message size: 1KB average
Group size: Up to 256 users
Availability: 99.9% uptime
Latency: <100ms for message delivery
```

**2. High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   Mobile    │ │   Web App   │ │  Desktop    │ │  API Client ││
│  │    App      │ │             │ │    App      │ │             ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    Load Balancer                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    API Gateway                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                  Microservices Layer                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   User      │ │  Message    │ │   Group     │ │  Presence   ││
│  │  Service    │ │  Service    │ │  Service    │ │  Service    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │  Media      │ │  Push       │ │  Analytics  │ │  Security   ││
│  │  Service    │ │  Service    │ │  Service    │ │  Service    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    Data Layer                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   MySQL     │ │    Redis    │ │   MongoDB   │ │   S3        ││
│  │  (Users)    │ │  (Sessions) │ │ (Messages)  │ │  (Media)    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**3. Message Service Implementation**

```go
type MessageService struct {
    db          *mongo.Database
    cache       *redis.Client
    pushService *PushService
    eventBus    *EventBus
}

type Message struct {
    ID          string    `bson:"_id" json:"id"`
    SenderID    string    `bson:"sender_id" json:"sender_id"`
    ReceiverID  string    `bson:"receiver_id" json:"receiver_id"`
    GroupID     string    `bson:"group_id,omitempty" json:"group_id,omitempty"`
    Content     string    `bson:"content" json:"content"`
    MessageType string    `bson:"message_type" json:"message_type"` // text, image, video, file
    MediaURL    string    `bson:"media_url,omitempty" json:"media_url,omitempty"`
    Timestamp   time.Time `bson:"timestamp" json:"timestamp"`
    Status      string    `bson:"status" json:"status"` // sent, delivered, read
}

func (ms *MessageService) SendMessage(ctx context.Context, req *SendMessageRequest) (*Message, error) {
    // 1. Create message
    message := &Message{
        ID:          generateUUID(),
        SenderID:    req.SenderID,
        ReceiverID:  req.ReceiverID,
        GroupID:     req.GroupID,
        Content:     req.Content,
        MessageType: req.MessageType,
        MediaURL:    req.MediaURL,
        Timestamp:   time.Now(),
        Status:      "sent",
    }

    // 2. Save message to database
    if err := ms.saveMessage(message); err != nil {
        return nil, err
    }

    // 3. Update sender's chat list
    ms.updateChatList(req.SenderID, req.ReceiverID, message)

    // 4. Update receiver's chat list
    ms.updateChatList(req.ReceiverID, req.SenderID, message)

    // 5. Send push notification
    go ms.pushService.SendNotification(req.ReceiverID, &PushNotification{
        Title:   "New Message",
        Body:    message.Content,
        Data:    map[string]string{"message_id": message.ID},
    })

    // 6. Publish message event
    ms.eventBus.Publish("message.sent", &MessageSentEvent{
        MessageID: message.ID,
        SenderID:  req.SenderID,
        ReceiverID: req.ReceiverID,
        GroupID:   req.GroupID,
    })

    return message, nil
}

func (ms *MessageService) GetMessages(ctx context.Context, userID, chatUserID string, limit int) ([]*Message, error) {
    // 1. Get messages from database
    messages, err := ms.getMessagesFromDB(userID, chatUserID, limit)
    if err != nil {
        return nil, err
    }

    // 2. Mark messages as read
    go ms.markMessagesAsRead(userID, chatUserID)

    return messages, nil
}
```

**4. Real-time Communication**

```go
type WebSocketManager struct {
    clients    map[string]*websocket.Conn
    register   chan *Client
    unregister chan *Client
    broadcast  chan []byte
    mutex      sync.RWMutex
}

type Client struct {
    ID     string
    UserID string
    Conn   *websocket.Conn
    Send   chan []byte
}

func (wsm *WebSocketManager) HandleClient(client *Client) {
    defer func() {
        wsm.unregister <- client
        client.Conn.Close()
    }()

    for {
        select {
        case message := <-client.Send:
            if err := client.Conn.WriteMessage(websocket.TextMessage, message); err != nil {
                return
            }
        }
    }
}

func (wsm *WebSocketManager) SendMessageToUser(userID string, message []byte) {
    wsm.mutex.RLock()
    defer wsm.mutex.RUnlock()

    if client, exists := wsm.clients[userID]; exists {
        select {
        case client.Send <- message:
        default:
            close(client.Send)
            delete(wsm.clients, userID)
        }
    }
}
```

**Key Design Decisions:**

- **Message Storage**: MongoDB for flexible schema and horizontal scaling
- **Real-time**: WebSocket for instant message delivery
- **Push Notifications**: Separate service for offline users
- **Media Handling**: S3 for media storage with CDN
- **Presence**: Redis for online/offline status

### **Scenario 5: Design Twitter (Social Media Platform)**

**Question**: "Design a social media platform like Twitter that can handle 500M users and 1B tweets per day."

#### **Solution Framework**

**1. Requirements Analysis**

```
Total Users: 500M
Daily Active Users: 200M
Tweets per day: 1B
Peak tweets per second: 10K
Tweet size: 280 characters
Timeline updates: Real-time
Availability: 99.9% uptime
```

**2. High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   Web App   │ │  Mobile App │ │  API Client │ │  Admin App  ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    Load Balancer                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    API Gateway                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                  Microservices Layer                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   User      │ │   Tweet     │ │  Timeline   │ │  Follow     ││
│  │  Service    │ │  Service    │ │  Service    │ │  Service    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │  Search     │ │  Media      │ │  Analytics  │ │  Notification││
│  │  Service    │ │  Service    │ │  Service    │ │  Service    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    Data Layer                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   MySQL     │ │    Redis    │ │   MongoDB   │ │   S3        ││
│  │  (Users)    │ │  (Timeline) │ │  (Tweets)   │ │  (Media)    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**3. Tweet Service Implementation**

```go
type TweetService struct {
    db          *mongo.Database
    cache       *redis.Client
    timelineService *TimelineService
    eventBus    *EventBus
}

type Tweet struct {
    ID          string    `bson:"_id" json:"id"`
    UserID      string    `bson:"user_id" json:"user_id"`
    Content     string    `bson:"content" json:"content"`
    MediaURLs   []string  `bson:"media_urls" json:"media_urls"`
    RetweetID   string    `bson:"retweet_id,omitempty" json:"retweet_id,omitempty"`
    ReplyToID   string    `bson:"reply_to_id,omitempty" json:"reply_to_id,omitempty"`
    Likes       int       `bson:"likes" json:"likes"`
    Retweets    int       `bson:"retweets" json:"retweets"`
    Replies     int       `bson:"replies" json:"replies"`
    Timestamp   time.Time `bson:"timestamp" json:"timestamp"`
    IsDeleted   bool      `bson:"is_deleted" json:"is_deleted"`
}

func (ts *TweetService) CreateTweet(ctx context.Context, req *CreateTweetRequest) (*Tweet, error) {
    // 1. Create tweet
    tweet := &Tweet{
        ID:        generateUUID(),
        UserID:    req.UserID,
        Content:   req.Content,
        MediaURLs: req.MediaURLs,
        ReplyToID: req.ReplyToID,
        Likes:     0,
        Retweets:  0,
        Replies:   0,
        Timestamp: time.Now(),
        IsDeleted: false,
    }

    // 2. Save tweet to database
    if err := ts.saveTweet(tweet); err != nil {
        return nil, err
    }

    // 3. Update user's timeline
    go ts.timelineService.AddToUserTimeline(req.UserID, tweet)

    // 4. Update followers' timelines
    go ts.timelineService.AddToFollowersTimelines(req.UserID, tweet)

    // 5. Publish tweet event
    ts.eventBus.Publish("tweet.created", &TweetCreatedEvent{
        TweetID: tweet.ID,
        UserID:  req.UserID,
        Content: req.Content,
    })

    return tweet, nil
}

func (ts *TweetService) GetTimeline(ctx context.Context, userID string, limit int) ([]*Tweet, error) {
    // 1. Get timeline from cache
    timeline, err := ts.timelineService.GetUserTimeline(userID, limit)
    if err != nil {
        return nil, err
    }

    // 2. Get tweet details
    tweets, err := ts.getTweetsByIDs(timeline)
    if err != nil {
        return nil, err
    }

    return tweets, nil
}
```

**4. Timeline Service**

```go
type TimelineService struct {
    cache *redis.Client
    db    *mongo.Database
}

func (ts *TimelineService) GetUserTimeline(userID string, limit int) ([]string, error) {
    // Get timeline from Redis (sorted set)
    timeline, err := ts.cache.ZRevRange(fmt.Sprintf("timeline:%s", userID), 0, int64(limit-1)).Result()
    if err != nil {
        return nil, err
    }

    return timeline, nil
}

func (ts *TimelineService) AddToUserTimeline(userID string, tweet *Tweet) error {
    // Add tweet to user's timeline
    key := fmt.Sprintf("timeline:%s", userID)
    score := float64(tweet.Timestamp.Unix())

    return ts.cache.ZAdd(key, &redis.Z{
        Score:  score,
        Member: tweet.ID,
    }).Err()
}

func (ts *TimelineService) AddToFollowersTimelines(userID string, tweet *Tweet) error {
    // Get followers
    followers, err := ts.getFollowers(userID)
    if err != nil {
        return err
    }

    // Add to each follower's timeline
    for _, followerID := range followers {
        go ts.AddToUserTimeline(followerID, tweet)
    }

    return nil
}
```

**Key Design Decisions:**

- **Timeline Generation**: Redis sorted sets for fast timeline retrieval
- **Fan-out**: Push model for active users, pull for inactive
- **Media Handling**: S3 with CDN for images and videos
- **Search**: Elasticsearch for tweet search and trending topics
- **Real-time**: WebSocket for live updates

---

## 🎯 **System Design Interview Tips**

### **1. Requirements Clarification**

- Ask about scale (users, requests, data)
- Understand functional vs non-functional requirements
- Clarify constraints and assumptions

### **2. High-Level Design**

- Start with a simple diagram
- Identify major components
- Show data flow between components

### **3. Deep Dive**

- Choose 2-3 components to detail
- Discuss database schema
- Explain API design

### **4. Scalability**

- Identify bottlenecks
- Discuss scaling strategies
- Consider caching and load balancing

### **5. Trade-offs**

- Performance vs consistency
- Cost vs scalability
- Complexity vs maintainability

---

**🎉 Practice these scenarios thoroughly and you'll be well-prepared for your Razorpay interviews! Good luck! 🚀**
