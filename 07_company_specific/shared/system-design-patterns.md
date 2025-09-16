# System Design Patterns

## 📚 Table of Contents

1. [Scalability Patterns](#scalability-patterns/)
2. [Caching Patterns](#caching-patterns/)
3. [Database Patterns](#database-patterns/)
4. [Communication Patterns](#communication-patterns/)
5. [Consistency Patterns](#consistency-patterns/)
6. [Fault Tolerance Patterns](#fault-tolerance-patterns/)
7. [Security Patterns](#security-patterns/)
8. [Monitoring Patterns](#monitoring-patterns/)

## Scalability Patterns

### 1. Load Balancing

**Purpose**: Distribute incoming requests across multiple servers to improve performance and reliability.

#### Round Robin Load Balancing

```go
type RoundRobinBalancer struct {
    servers []Server
    current int
    mutex   sync.Mutex
}

func (rr *RoundRobinBalancer) SelectServer() Server {
    rr.mutex.Lock()
    defer rr.mutex.Unlock()

    if len(rr.servers) == 0 {
        return nil
    }

    server := rr.servers[rr.current]
    rr.current = (rr.current + 1) % len(rr.servers)
    return server
}
```

#### Weighted Round Robin

```go
type WeightedServer struct {
    Server Server
    Weight int
}

type WeightedRoundRobinBalancer struct {
    servers []WeightedServer
    current int
    cw      int // current weight
    mutex   sync.Mutex
}

func (wrr *WeightedRoundRobinBalancer) SelectServer() Server {
    wrr.mutex.Lock()
    defer wrr.mutex.Unlock()

    if len(wrr.servers) == 0 {
        return nil
    }

    for {
        wrr.current = (wrr.current + 1) % len(wrr.servers)
        if wrr.current == 0 {
            wrr.cw = wrr.cw - wrr.gcd()
            if wrr.cw <= 0 {
                wrr.cw = wrr.maxWeight()
            }
        }

        if wrr.servers[wrr.current].Weight >= wrr.cw {
            return wrr.servers[wrr.current].Server
        }
    }
}
```

#### Least Connections

```go
type LeastConnectionsBalancer struct {
    servers []Server
    mutex   sync.RWMutex
}

func (lc *LeastConnectionsBalancer) SelectServer() Server {
    lc.mutex.RLock()
    defer lc.mutex.RUnlock()

    if len(lc.servers) == 0 {
        return nil
    }

    minConnections := lc.servers[0].ActiveConnections()
    selectedServer := lc.servers[0]

    for _, server := range lc.servers[1:] {
        if server.ActiveConnections() < minConnections {
            minConnections = server.ActiveConnections()
            selectedServer = server
        }
    }

    return selectedServer
}
```

### 2. Horizontal Scaling

**Purpose**: Add more machines to handle increased load.

#### Auto-scaling Configuration

```yaml
# Kubernetes HPA example
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-deployment
  minReplicas: 3
  maxReplicas: 100
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

#### Load Testing Pattern

```go
type LoadTester struct {
    targetURL string
    duration  time.Duration
    rate      int // requests per second
}

func (lt *LoadTester) RunTest() *TestResults {
    results := &TestResults{
        StartTime: time.Now(),
        Requests:  make([]RequestResult, 0),
    }

    ticker := time.NewTicker(time.Second / time.Duration(lt.rate))
    defer ticker.Stop()

    timeout := time.After(lt.duration)

    for {
        select {
        case <-ticker.C:
            go lt.makeRequest(results)
        case <-timeout:
            results.EndTime = time.Now()
            return results
        }
    }
}

func (lt *LoadTester) makeRequest(results *TestResults) {
    start := time.Now()

    resp, err := http.Get(lt.targetURL)
    duration := time.Since(start)

    result := RequestResult{
        Duration: duration,
        Success:  err == nil && resp.StatusCode == 200,
        StatusCode: resp.StatusCode,
    }

    if resp != nil {
        resp.Body.Close()
    }

    results.mutex.Lock()
    results.Requests = append(results.Requests, result)
    results.mutex.Unlock()
}
```

## Caching Patterns

### 1. Cache-Aside Pattern

**Purpose**: Application manages cache directly.

```go
type CacheAsideService struct {
    cache Cache
    db    Database
}

func (cas *CacheAsideService) Get(key string) (interface{}, error) {
    // Try cache first
    if value, found := cas.cache.Get(key); found {
        return value, nil
    }

    // Cache miss - load from database
    value, err := cas.db.Get(key)
    if err != nil {
        return nil, err
    }

    // Store in cache for next time
    cas.cache.Set(key, value, 5*time.Minute)

    return value, nil
}

func (cas *CacheAsideService) Set(key string, value interface{}) error {
    // Write to database
    if err := cas.db.Set(key, value); err != nil {
        return err
    }

    // Update cache
    cas.cache.Set(key, value, 5*time.Minute)

    return nil
}
```

### 2. Write-Through Pattern

**Purpose**: Write to both cache and database simultaneously.

```go
type WriteThroughService struct {
    cache Cache
    db    Database
}

func (wts *WriteThroughService) Set(key string, value interface{}) error {
    // Write to database first
    if err := wts.db.Set(key, value); err != nil {
        return err
    }

    // Then write to cache
    wts.cache.Set(key, value, 5*time.Minute)

    return nil
}
```

### 3. Write-Behind Pattern

**Purpose**: Write to cache immediately, database asynchronously.

```go
type WriteBehindService struct {
    cache     Cache
    db        Database
    writeQueue chan WriteOperation
    batchSize  int
    flushInterval time.Duration
}

type WriteOperation struct {
    Key   string
    Value interface{}
}

func (wbs *WriteBehindService) Set(key string, value interface{}) error {
    // Write to cache immediately
    wbs.cache.Set(key, value, 5*time.Minute)

    // Queue for database write
    select {
    case wbs.writeQueue <- WriteOperation{Key: key, Value: value}:
        return nil
    default:
        return errors.New("write queue is full")
    }
}

func (wbs *WriteBehindService) startBatchWriter() {
    ticker := time.NewTicker(wbs.flushInterval)
    defer ticker.Stop()

    batch := make([]WriteOperation, 0, wbs.batchSize)

    for {
        select {
        case op := <-wbs.writeQueue:
            batch = append(batch, op)
            if len(batch) >= wbs.batchSize {
                wbs.flushBatch(batch)
                batch = batch[:0]
            }
        case <-ticker.C:
            if len(batch) > 0 {
                wbs.flushBatch(batch)
                batch = batch[:0]
            }
        }
    }
}

func (wbs *WriteBehindService) flushBatch(batch []WriteOperation) {
    for _, op := range batch {
        if err := wbs.db.Set(op.Key, op.Value); err != nil {
            log.Printf("Failed to write %s to database: %v", op.Key, err)
        }
    }
}
```

### 4. Cache Invalidation Patterns

#### Time-based Invalidation

```go
type TTLCache struct {
    data map[string]CacheItem
    mutex sync.RWMutex
}

type CacheItem struct {
    Value     interface{}
    ExpiresAt time.Time
}

func (tc *TTLCache) Get(key string) (interface{}, bool) {
    tc.mutex.RLock()
    defer tc.mutex.RUnlock()

    item, exists := tc.data[key]
    if !exists || time.Now().After(item.ExpiresAt) {
        return nil, false
    }

    return item.Value, true
}
```

#### Event-based Invalidation

```go
type EventDrivenCache struct {
    cache     Cache
    eventBus  EventBus
    mutex     sync.RWMutex
}

func (edc *EventDrivenCache) InvalidateOnEvent(eventType string) {
    edc.eventBus.Subscribe(eventType, func(event Event) {
        if keys, ok := event.Data.([]string); ok {
            edc.mutex.Lock()
            for _, key := range keys {
                edc.cache.Delete(key)
            }
            edc.mutex.Unlock()
        }
    })
}
```

## Database Patterns

### 1. Database Sharding

**Purpose**: Distribute data across multiple databases to improve performance.

#### Range-based Sharding

```go
type RangeShardRouter struct {
    shards []Shard
    ranges []ShardRange
}

type ShardRange struct {
    Start int64
    End   int64
    Shard Shard
}

func (rsr *RangeShardRouter) GetShard(key int64) Shard {
    for _, r := range rsr.ranges {
        if key >= r.Start && key <= r.End {
            return r.Shard
        }
    }
    return nil // Key not in any range
}
```

#### Hash-based Sharding

```go
type HashShardRouter struct {
    shards []Shard
    numShards int
}

func (hsr *HashShardRouter) GetShard(key string) Shard {
    hash := fnv.New32a()
    hash.Write([]byte(key))
    shardIndex := int(hash.Sum32()) % hsr.numShards
    return hsr.shards[shardIndex]
}
```

#### Consistent Hashing

```go
type ConsistentHash struct {
    ring map[uint32]string
    sortedKeys []uint32
    mutex sync.RWMutex
}

func NewConsistentHash() *ConsistentHash {
    return &ConsistentHash{
        ring: make(map[uint32]string),
    }
}

func (ch *ConsistentHash) AddNode(node string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()

    for i := 0; i < 3; i++ { // 3 virtual nodes
        hash := ch.hash(node + strconv.Itoa(i))
        ch.ring[hash] = node
        ch.sortedKeys = append(ch.sortedKeys, hash)
    }

    sort.Slice(ch.sortedKeys, func(i, j int) bool {
        return ch.sortedKeys[i] < ch.sortedKeys[j]
    })
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()

    if len(ch.ring) == 0 {
        return ""
    }

    hash := ch.hash(key)

    for _, ringHash := range ch.sortedKeys {
        if ringHash >= hash {
            return ch.ring[ringHash]
        }
    }

    // Wrap around to first node
    return ch.ring[ch.sortedKeys[0]]
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := fnv.New32a()
    h.Write([]byte(key))
    return h.Sum32()
}
```

### 2. Database Replication

#### Master-Slave Replication

```go
type MasterSlaveDB struct {
    master Database
    slaves []Database
    mutex  sync.RWMutex
}

func (msdb *MasterSlaveDB) Write(key string, value interface{}) error {
    // Write to master
    return msdb.master.Write(key, value)
}

func (msdb *MasterSlaveDB) Read(key string) (interface{}, error) {
    msdb.mutex.RLock()
    defer msdb.mutex.RUnlock()

    // Read from a random slave
    if len(msdb.slaves) == 0 {
        return msdb.master.Read(key)
    }

    slave := msdb.slaves[rand.Intn(len(msdb.slaves))]
    return slave.Read(key)
}
```

#### Master-Master Replication

```go
type MasterMasterDB struct {
    masters []Database
    mutex   sync.RWMutex
}

func (mmdb *MasterMasterDB) Write(key string, value interface{}) error {
    mmdb.mutex.Lock()
    defer mmdb.mutex.Unlock()

    // Write to all masters
    var wg sync.WaitGroup
    errChan := make(chan error, len(mmdb.masters))

    for _, master := range mmdb.masters {
        wg.Add(1)
        go func(db Database) {
            defer wg.Done()
            if err := db.Write(key, value); err != nil {
                errChan <- err
            }
        }(master)
    }

    wg.Wait()
    close(errChan)

    // Check for errors
    for err := range errChan {
        if err != nil {
            return err
        }
    }

    return nil
}
```

## Communication Patterns

### 1. Request-Response Pattern

**Purpose**: Synchronous communication between services.

```go
type RequestResponseClient struct {
    httpClient *http.Client
    baseURL    string
}

func (rrc *RequestResponseClient) Call(endpoint string, request interface{}) (interface{}, error) {
    jsonData, err := json.Marshal(request)
    if err != nil {
        return nil, err
    }

    resp, err := rrc.httpClient.Post(
        rrc.baseURL+endpoint,
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var response interface{}
    if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
        return nil, err
    }

    return response, nil
}
```

### 2. Publish-Subscribe Pattern

**Purpose**: Asynchronous communication with multiple subscribers.

```go
type PubSub struct {
    subscribers map[string][]Subscriber
    mutex       sync.RWMutex
}

type Subscriber interface {
    Handle(message Message)
}

type Message struct {
    Topic   string
    Payload interface{}
}

func (ps *PubSub) Subscribe(topic string, subscriber Subscriber) {
    ps.mutex.Lock()
    defer ps.mutex.Unlock()

    ps.subscribers[topic] = append(ps.subscribers[topic], subscriber)
}

func (ps *PubSub) Publish(topic string, payload interface{}) {
    ps.mutex.RLock()
    defer ps.mutex.RUnlock()

    message := Message{
        Topic:   topic,
        Payload: payload,
    }

    for _, subscriber := range ps.subscribers[topic] {
        go subscriber.Handle(message)
    }
}
```

### 3. Message Queue Pattern

**Purpose**: Decouple producers and consumers using message queues.

```go
type MessageQueue struct {
    queue chan Message
    mutex sync.RWMutex
}

func NewMessageQueue(bufferSize int) *MessageQueue {
    return &MessageQueue{
        queue: make(chan Message, bufferSize),
    }
}

func (mq *MessageQueue) Publish(message Message) error {
    select {
    case mq.queue <- message:
        return nil
    default:
        return errors.New("queue is full")
    }
}

func (mq *MessageQueue) Subscribe(handler func(Message)) {
    go func() {
        for message := range mq.queue {
            handler(message)
        }
    }()
}
```

## Consistency Patterns

### 1. Strong Consistency

**Purpose**: All nodes see the same data at the same time.

```go
type StrongConsistentStore struct {
    data  map[string]interface{}
    mutex sync.RWMutex
}

func (scs *StrongConsistentStore) Write(key string, value interface{}) error {
    scs.mutex.Lock()
    defer scs.mutex.Unlock()

    scs.data[key] = value
    return nil
}

func (scs *StrongConsistentStore) Read(key string) (interface{}, error) {
    scs.mutex.RLock()
    defer scs.mutex.RUnlock()

    value, exists := scs.data[key]
    if !exists {
        return nil, errors.New("key not found")
    }

    return value, nil
}
```

### 2. Eventual Consistency

**Purpose**: System will become consistent over time.

```go
type EventualConsistentStore struct {
    data     map[string]interface{}
    replicas []Replica
    mutex    sync.RWMutex
}

func (ecs *EventualConsistentStore) Write(key string, value interface{}) error {
    ecs.mutex.Lock()
    ecs.data[key] = value
    ecs.mutex.Unlock()

    // Replicate asynchronously
    go ecs.replicate(key, value)

    return nil
}

func (ecs *EventualConsistentStore) replicate(key string, value interface{}) {
    for _, replica := range ecs.replicas {
        go func(r Replica) {
            if err := r.Write(key, value); err != nil {
                log.Printf("Replication failed: %v", err)
            }
        }(replica)
    }
}
```

### 3. Causal Consistency

**Purpose**: Causally related operations are seen in the same order by all processes.

```go
type CausalConsistentStore struct {
    data     map[string]interface{}
    versions map[string]int
    mutex    sync.RWMutex
}

func (ccs *CausalConsistentStore) Write(key string, value interface{}, version int) error {
    ccs.mutex.Lock()
    defer ccs.mutex.Unlock()

    if currentVersion, exists := ccs.versions[key]; exists && version <= currentVersion {
        return errors.New("version conflict")
    }

    ccs.data[key] = value
    ccs.versions[key] = version

    return nil
}
```

## Fault Tolerance Patterns

### 1. Circuit Breaker Pattern

**Purpose**: Prevent cascading failures by stopping calls to failing services.

```go
type CircuitBreaker struct {
    maxFailures int
    timeout     time.Duration
    state       State
    failures    int
    lastFailure time.Time
    mutex       sync.Mutex
}

type State int

const (
    Closed State = iota
    Open
    HalfOpen
)

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()

    if cb.state == Open {
        if time.Since(cb.lastFailure) < cb.timeout {
            return errors.New("circuit breaker is open")
        }
        cb.state = HalfOpen
    }

    err := fn()

    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()

        if cb.failures >= cb.maxFailures {
            cb.state = Open
        }

        return err
    }

    cb.failures = 0
    cb.state = Closed

    return nil
}
```

### 2. Retry Pattern

**Purpose**: Automatically retry failed operations with exponential backoff.

```go
type RetryConfig struct {
    MaxAttempts int
    BaseDelay   time.Duration
    MaxDelay    time.Duration
    Multiplier  float64
}

func (rc *RetryConfig) Execute(fn func() error) error {
    var lastErr error

    for attempt := 0; attempt < rc.MaxAttempts; attempt++ {
        if err := fn(); err == nil {
            return nil
        } else {
            lastErr = err
        }

        if attempt < rc.MaxAttempts-1 {
            delay := time.Duration(float64(rc.BaseDelay) * math.Pow(rc.Multiplier, float64(attempt)))
            if delay > rc.MaxDelay {
                delay = rc.MaxDelay
            }
            time.Sleep(delay)
        }
    }

    return lastErr
}
```

### 3. Bulkhead Pattern

**Purpose**: Isolate critical resources to prevent total system failure.

```go
type BulkheadPool struct {
    pools map[string]*WorkerPool
    mutex sync.RWMutex
}

func (bp *BulkheadPool) GetPool(name string, size int) *WorkerPool {
    bp.mutex.Lock()
    defer bp.mutex.Unlock()

    if pool, exists := bp.pools[name]; exists {
        return pool
    }

    pool := NewWorkerPool(size)
    bp.pools[name] = pool
    return pool
}

func (bp *BulkheadPool) Submit(poolName string, task func()) error {
    pool := bp.GetPool(poolName, 10) // Default size

    select {
    case pool.jobQueue <- task:
        return nil
    default:
        return errors.New("pool is full")
    }
}
```

## Security Patterns

### 1. Authentication Pattern

**Purpose**: Verify user identity before allowing access.

```go
type AuthService struct {
    jwtSecret []byte
    userStore UserStore
}

func (as *AuthService) Authenticate(token string) (*User, error) {
    claims, err := as.validateToken(token)
    if err != nil {
        return nil, err
    }

    user, err := as.userStore.GetByID(claims.UserID)
    if err != nil {
        return nil, err
    }

    return user, nil
}

func (as *AuthService) validateToken(token string) (*Claims, error) {
    jwtToken, err := jwt.ParseWithClaims(token, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        return as.jwtSecret, nil
    })

    if err != nil {
        return nil, err
    }

    if claims, ok := jwtToken.Claims.(*Claims); ok && jwtToken.Valid {
        return claims, nil
    }

    return nil, errors.New("invalid token")
}
```

### 2. Authorization Pattern

**Purpose**: Control access to resources based on user permissions.

```go
type RBACService struct {
    permissions map[string][]string
    mutex       sync.RWMutex
}

func (rbac *RBACService) HasPermission(userRole, resource, action string) bool {
    rbac.mutex.RLock()
    defer rbac.mutex.RUnlock()

    rolePermissions, exists := rbac.permissions[userRole]
    if !exists {
        return false
    }

    requiredPermission := fmt.Sprintf("%s:%s", resource, action)

    for _, permission := range rolePermissions {
        if permission == requiredPermission {
            return true
        }
    }

    return false
}

func (rbac *RBACService) Authorize(userRole, resource, action string) error {
    if !rbac.HasPermission(userRole, resource, action) {
        return errors.New("access denied")
    }
    return nil
}
```

### 3. Rate Limiting Pattern

**Purpose**: Prevent abuse by limiting request frequency.

```go
type RateLimiter struct {
    requests map[string][]time.Time
    limit    int
    window   time.Duration
    mutex    sync.Mutex
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
    return &RateLimiter{
        requests: make(map[string][]time.Time),
        limit:    limit,
        window:   window,
    }
}

func (rl *RateLimiter) Allow(clientID string) bool {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()

    now := time.Now()
    cutoff := now.Add(-rl.window)

    // Clean old requests
    requests := rl.requests[clientID]
    var validRequests []time.Time
    for _, reqTime := range requests {
        if reqTime.After(cutoff) {
            validRequests = append(validRequests, reqTime)
        }
    }

    if len(validRequests) >= rl.limit {
        return false
    }

    validRequests = append(validRequests, now)
    rl.requests[clientID] = validRequests

    return true
}
```

## Monitoring Patterns

### 1. Health Check Pattern

**Purpose**: Monitor service health and availability.

```go
type HealthChecker struct {
    checks map[string]HealthCheck
    mutex  sync.RWMutex
}

type HealthCheck func() error

func (hc *HealthChecker) AddCheck(name string, check HealthCheck) {
    hc.mutex.Lock()
    defer hc.mutex.Unlock()
    hc.checks[name] = check
}

func (hc *HealthChecker) CheckHealth() map[string]string {
    hc.mutex.RLock()
    defer hc.mutex.RUnlock()

    results := make(map[string]string)

    for name, check := range hc.checks {
        if err := check(); err != nil {
            results[name] = fmt.Sprintf("FAIL: %v", err)
        } else {
            results[name] = "OK"
        }
    }

    return results
}
```

### 2. Metrics Collection Pattern

**Purpose**: Collect and expose system metrics.

```go
type MetricsCollector struct {
    counters map[string]int64
    gauges   map[string]float64
    mutex    sync.RWMutex
}

func (mc *MetricsCollector) IncrementCounter(name string) {
    mc.mutex.Lock()
    defer mc.mutex.Unlock()
    mc.counters[name]++
}

func (mc *MetricsCollector) SetGauge(name string, value float64) {
    mc.mutex.Lock()
    defer mc.mutex.Unlock()
    mc.gauges[name] = value
}

func (mc *MetricsCollector) GetMetrics() map[string]interface{} {
    mc.mutex.RLock()
    defer mc.mutex.RUnlock()

    metrics := make(map[string]interface{})

    for name, value := range mc.counters {
        metrics[name] = value
    }

    for name, value := range mc.gauges {
        metrics[name] = value
    }

    return metrics
}
```

### 3. Distributed Tracing Pattern

**Purpose**: Track requests across multiple services.

```go
type TraceContext struct {
    TraceID  string
    SpanID   string
    ParentID string
    Tags     map[string]string
}

type Tracer struct {
    spans map[string]*Span
    mutex sync.RWMutex
}

type Span struct {
    TraceID    string
    SpanID     string
    ParentID   string
    Operation  string
    StartTime  time.Time
    EndTime    time.Time
    Tags       map[string]string
    Logs       []LogEntry
}

func (t *Tracer) StartSpan(operation string, parent *Span) *Span {
    span := &Span{
        TraceID:   generateID(),
        SpanID:    generateID(),
        Operation: operation,
        StartTime: time.Now(),
        Tags:      make(map[string]string),
        Logs:      make([]LogEntry, 0),
    }

    if parent != nil {
        span.TraceID = parent.TraceID
        span.ParentID = parent.SpanID
    }

    t.mutex.Lock()
    t.spans[span.SpanID] = span
    t.mutex.Unlock()

    return span
}

func (s *Span) Finish() {
    s.EndTime = time.Now()
}

func (s *Span) AddTag(key, value string) {
    s.Tags[key] = value
}

func (s *Span) AddLog(message string) {
    s.Logs = append(s.Logs, LogEntry{
        Timestamp: time.Now(),
        Message:   message,
    })
}
```

This comprehensive guide covers the essential system design patterns used in building scalable, reliable, and maintainable distributed systems. Each pattern includes practical implementations and real-world examples that can be applied in interview scenarios and production systems.
