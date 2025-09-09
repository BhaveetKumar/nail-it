# 🏗️ System Design Patterns Complete Guide

> **Master essential system design patterns for building scalable, fault-tolerant distributed systems**

## 📚 Overview

This comprehensive guide covers the most important system design patterns used in building large-scale distributed systems. These patterns are essential for system design interviews and real-world architecture decisions.

## 🎯 Table of Contents

1. [Load Balancing Patterns](#load-balancing-patterns)
2. [Caching Patterns](#caching-patterns)
3. [Database Patterns](#database-patterns)
4. [Messaging Patterns](#messaging-patterns)
5. [Microservices Patterns](#microservices-patterns)
6. [Security Patterns](#security-patterns)
7. [Monitoring Patterns](#monitoring-patterns)
8. [Data Processing Patterns](#data-processing-patterns)
9. [API Design Patterns](#api-design-patterns)
10. [Deployment Patterns](#deployment-patterns)

## ⚖️ Load Balancing Patterns

### **1. Round Robin**
Distributes requests evenly across servers in a circular manner.

```go
type RoundRobinBalancer struct {
    servers []string
    current int
    mutex   sync.Mutex
}

func (rb *RoundRobinBalancer) GetServer() string {
    rb.mutex.Lock()
    defer rb.mutex.Unlock()
    
    server := rb.servers[rb.current]
    rb.current = (rb.current + 1) % len(rb.servers)
    return server
}
```

**Use Cases:**
- Stateless applications
- Equal server capacity
- Simple load distribution

### **2. Weighted Round Robin**
Distributes requests based on server capacity or performance.

```go
type WeightedServer struct {
    Server string
    Weight int
}

type WeightedRoundRobinBalancer struct {
    servers []WeightedServer
    current int
    cw      int // current weight
    mutex   sync.Mutex
}

func (wrb *WeightedRoundRobinBalancer) GetServer() string {
    wrb.mutex.Lock()
    defer wrb.mutex.Unlock()
    
    for {
        wrb.current = (wrb.current + 1) % len(wrb.servers)
        if wrb.current == 0 {
            wrb.cw = wrb.cw - wrb.gcd()
            if wrb.cw <= 0 {
                wrb.cw = wrb.maxWeight()
            }
        }
        
        if wrb.servers[wrb.current].Weight >= wrb.cw {
            return wrb.servers[wrb.current].Server
        }
    }
}
```

### **3. Least Connections**
Routes requests to the server with the fewest active connections.

```go
type LeastConnectionsBalancer struct {
    servers map[string]int
    mutex   sync.RWMutex
}

func (lcb *LeastConnectionsBalancer) GetServer() string {
    lcb.mutex.Lock()
    defer lcb.mutex.Unlock()
    
    minConnections := math.MaxInt32
    selectedServer := ""
    
    for server, connections := range lcb.servers {
        if connections < minConnections {
            minConnections = connections
            selectedServer = server
        }
    }
    
    lcb.servers[selectedServer]++
    return selectedServer
}
```

### **4. Consistent Hashing**
Distributes requests based on hash values, minimizing redistribution when servers are added/removed.

```go
type ConsistentHash struct {
    ring     map[uint32]string
    sortedKeys []uint32
    mutex    sync.RWMutex
}

func (ch *ConsistentHash) AddServer(server string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    hash := ch.hash(server)
    ch.ring[hash] = server
    ch.sortedKeys = append(ch.sortedKeys, hash)
    sort.Slice(ch.sortedKeys, func(i, j int) bool {
        return ch.sortedKeys[i] < ch.sortedKeys[j]
    })
}

func (ch *ConsistentHash) GetServer(key string) string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    hash := ch.hash(key)
    
    for _, serverHash := range ch.sortedKeys {
        if serverHash >= hash {
            return ch.ring[serverHash]
        }
    }
    
    // Wrap around to first server
    return ch.ring[ch.sortedKeys[0]]
}
```

## 🗄️ Caching Patterns

### **1. Cache-Aside (Lazy Loading)**
Application manages cache directly.

```go
type CacheAside struct {
    cache  map[string]interface{}
    db     Database
    mutex  sync.RWMutex
}

func (ca *CacheAside) Get(key string) (interface{}, error) {
    // Try cache first
    ca.mutex.RLock()
    if value, exists := ca.cache[key]; exists {
        ca.mutex.RUnlock()
        return value, nil
    }
    ca.mutex.RUnlock()
    
    // Cache miss - load from database
    value, err := ca.db.Get(key)
    if err != nil {
        return nil, err
    }
    
    // Store in cache
    ca.mutex.Lock()
    ca.cache[key] = value
    ca.mutex.Unlock()
    
    return value, nil
}
```

### **2. Write-Through**
Data is written to both cache and database simultaneously.

```go
type WriteThrough struct {
    cache map[string]interface{}
    db    Database
    mutex sync.RWMutex
}

func (wt *WriteThrough) Set(key string, value interface{}) error {
    // Write to database first
    if err := wt.db.Set(key, value); err != nil {
        return err
    }
    
    // Then write to cache
    wt.mutex.Lock()
    wt.cache[key] = value
    wt.mutex.Unlock()
    
    return nil
}
```

### **3. Write-Behind (Write-Back)**
Data is written to cache immediately and to database asynchronously.

```go
type WriteBehind struct {
    cache     map[string]interface{}
    db        Database
    mutex     sync.RWMutex
    writeQueue chan WriteOperation
}

type WriteOperation struct {
    Key   string
    Value interface{}
}

func (wb *WriteBehind) Set(key string, value interface{}) error {
    // Write to cache immediately
    wb.mutex.Lock()
    wb.cache[key] = value
    wb.mutex.Unlock()
    
    // Queue for database write
    select {
    case wb.writeQueue <- WriteOperation{Key: key, Value: value}:
        return nil
    default:
        return errors.New("write queue full")
    }
}

func (wb *WriteBehind) processWrites() {
    for op := range wb.writeQueue {
        wb.db.Set(op.Key, op.Value)
    }
}
```

### **4. Refresh-Ahead**
Cache is refreshed before expiration.

```go
type RefreshAhead struct {
    cache      map[string]CacheEntry
    db         Database
    mutex      sync.RWMutex
    refreshTTL time.Duration
}

type CacheEntry struct {
    Value     interface{}
    ExpiresAt time.Time
    Refreshed bool
}

func (ra *RefreshAhead) Get(key string) (interface{}, error) {
    ra.mutex.RLock()
    entry, exists := ra.cache[key]
    ra.mutex.RUnlock()
    
    if !exists {
        return ra.loadAndCache(key)
    }
    
    // Check if refresh is needed
    if time.Until(entry.ExpiresAt) < ra.refreshTTL && !entry.Refreshed {
        go ra.refreshInBackground(key)
    }
    
    if time.Now().After(entry.ExpiresAt) {
        return ra.loadAndCache(key)
    }
    
    return entry.Value, nil
}
```

## 🗃️ Database Patterns

### **1. Database Sharding**
Distributes data across multiple database instances.

```go
type ShardedDatabase struct {
    shards map[int]Database
    shardCount int
}

func (sd *ShardedDatabase) GetShard(key string) Database {
    hash := sd.hash(key)
    shardID := hash % sd.shardCount
    return sd.shards[shardID]
}

func (sd *ShardedDatabase) Get(key string) (interface{}, error) {
    shard := sd.GetShard(key)
    return shard.Get(key)
}

func (sd *ShardedDatabase) Set(key string, value interface{}) error {
    shard := sd.GetShard(key)
    return shard.Set(key, value)
}
```

### **2. Read Replicas**
Separates read and write operations.

```go
type ReadReplicaDatabase struct {
    master  Database
    replicas []Database
    current int
    mutex   sync.Mutex
}

func (rrd *ReadReplicaDatabase) Read(key string) (interface{}, error) {
    rrd.mutex.Lock()
    replica := rrd.replicas[rrd.current]
    rrd.current = (rrd.current + 1) % len(rrd.replicas)
    rrd.mutex.Unlock()
    
    return replica.Get(key)
}

func (rrd *ReadReplicaDatabase) Write(key string, value interface{}) error {
    return rrd.master.Set(key, value)
}
```

### **3. Database Partitioning**
Splits large tables into smaller, manageable pieces.

```go
type PartitionedTable struct {
    partitions map[string]Database
    partitionKey string
}

func (pt *PartitionedTable) GetPartition(partitionValue string) Database {
    return pt.partitions[partitionValue]
}

func (pt *PartitionedTable) Insert(record Record) error {
    partition := pt.GetPartition(record[pt.partitionKey])
    return partition.Insert(record)
}
```

## 📨 Messaging Patterns

### **1. Publisher-Subscriber**
Decouples message producers from consumers.

```go
type PubSub struct {
    subscribers map[string][]chan Message
    mutex       sync.RWMutex
}

type Message struct {
    Topic   string
    Payload interface{}
}

func (ps *PubSub) Subscribe(topic string) <-chan Message {
    ps.mutex.Lock()
    defer ps.mutex.Unlock()
    
    ch := make(chan Message, 100)
    ps.subscribers[topic] = append(ps.subscribers[topic], ch)
    return ch
}

func (ps *PubSub) Publish(topic string, payload interface{}) {
    ps.mutex.RLock()
    defer ps.mutex.RUnlock()
    
    message := Message{Topic: topic, Payload: payload}
    for _, ch := range ps.subscribers[topic] {
        select {
        case ch <- message:
        default:
            // Channel full, skip
        }
    }
}
```

### **2. Message Queue**
Provides reliable message delivery.

```go
type MessageQueue struct {
    queue chan Message
    mutex sync.Mutex
}

func (mq *MessageQueue) Enqueue(message Message) error {
    select {
    case mq.queue <- message:
        return nil
    default:
        return errors.New("queue full")
    }
}

func (mq *MessageQueue) Dequeue() (Message, error) {
    select {
    case message := <-mq.queue:
        return message, nil
    case <-time.After(5 * time.Second):
        return Message{}, errors.New("timeout")
    }
}
```

### **3. Event Sourcing**
Stores events instead of current state.

```go
type EventStore struct {
    events []Event
    mutex  sync.RWMutex
}

type Event struct {
    ID        string
    Type      string
    Data      interface{}
    Timestamp time.Time
}

func (es *EventStore) AppendEvent(event Event) {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    es.events = append(es.events, event)
}

func (es *EventStore) GetEvents(aggregateID string) []Event {
    es.mutex.RLock()
    defer es.mutex.RUnlock()
    
    var result []Event
    for _, event := range es.events {
        if event.ID == aggregateID {
            result = append(result, event)
        }
    }
    return result
}
```

## 🔧 Microservices Patterns

### **1. API Gateway**
Single entry point for all client requests.

```go
type APIGateway struct {
    services map[string]Service
    router   *gin.Engine
}

func (gw *APIGateway) RouteRequest(path string, method string) {
    service := gw.routeToService(path)
    gw.router.Handle(method, path, gw.proxyHandler(service))
}

func (gw *APIGateway) proxyHandler(service Service) gin.HandlerFunc {
    return func(c *gin.Context) {
        // Authentication
        if !gw.authenticate(c) {
            c.JSON(401, gin.H{"error": "unauthorized"})
            return
        }
        
        // Rate limiting
        if !gw.rateLimit(c) {
            c.JSON(429, gin.H{"error": "rate limit exceeded"})
            return
        }
        
        // Proxy to service
        gw.forwardRequest(c, service)
    }
}
```

### **2. Circuit Breaker**
Prevents cascading failures.

```go
type CircuitBreaker struct {
    failureCount int
    threshold    int
    timeout      time.Duration
    state        State
    mutex        sync.Mutex
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
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = HalfOpen
        } else {
            return errors.New("circuit breaker open")
        }
    }
    
    err := fn()
    if err != nil {
        cb.failureCount++
        cb.lastFailure = time.Now()
        
        if cb.failureCount >= cb.threshold {
            cb.state = Open
        }
        return err
    }
    
    cb.failureCount = 0
    cb.state = Closed
    return nil
}
```

### **3. Service Discovery**
Automatically discovers and registers services.

```go
type ServiceRegistry struct {
    services map[string][]ServiceInstance
    mutex    sync.RWMutex
}

type ServiceInstance struct {
    ID       string
    Address  string
    Port     int
    Health   bool
    LastSeen time.Time
}

func (sr *ServiceRegistry) Register(serviceName string, instance ServiceInstance) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    instance.LastSeen = time.Now()
    sr.services[serviceName] = append(sr.services[serviceName], instance)
}

func (sr *ServiceRegistry) Discover(serviceName string) []ServiceInstance {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    instances := sr.services[serviceName]
    var healthy []ServiceInstance
    
    for _, instance := range instances {
        if instance.Health && time.Since(instance.LastSeen) < 30*time.Second {
            healthy = append(healthy, instance)
        }
    }
    
    return healthy
}
```

## 🔒 Security Patterns

### **1. OAuth 2.0**
Authorization framework for secure API access.

```go
type OAuth2Provider struct {
    clients map[string]Client
    tokens  map[string]Token
    mutex   sync.RWMutex
}

type Client struct {
    ID          string
    Secret      string
    RedirectURI string
    Scopes      []string
}

type Token struct {
    AccessToken  string
    RefreshToken string
    ExpiresAt    time.Time
    Scopes       []string
}

func (op *OAuth2Provider) Authorize(clientID, redirectURI, scope string) string {
    op.mutex.RLock()
    client, exists := op.clients[clientID]
    op.mutex.RUnlock()
    
    if !exists || client.RedirectURI != redirectURI {
        return ""
    }
    
    // Generate authorization code
    code := op.generateCode()
    op.storeAuthorizationCode(code, clientID, scope)
    
    return code
}

func (op *OAuth2Provider) ExchangeCode(code string) (*Token, error) {
    authCode := op.getAuthorizationCode(code)
    if authCode == nil {
        return nil, errors.New("invalid authorization code")
    }
    
    token := &Token{
        AccessToken:  op.generateAccessToken(),
        RefreshToken: op.generateRefreshToken(),
        ExpiresAt:    time.Now().Add(1 * time.Hour),
        Scopes:       authCode.Scopes,
    }
    
    op.mutex.Lock()
    op.tokens[token.AccessToken] = *token
    op.mutex.Unlock()
    
    return token, nil
}
```

### **2. JWT (JSON Web Token)**
Stateless authentication mechanism.

```go
type JWTProvider struct {
    secretKey []byte
}

type Claims struct {
    UserID string `json:"user_id"`
    Email  string `json:"email"`
    jwt.StandardClaims
}

func (jp *JWTProvider) GenerateToken(userID, email string) (string, error) {
    claims := Claims{
        UserID: userID,
        Email:  email,
        StandardClaims: jwt.StandardClaims{
            ExpiresAt: time.Now().Add(24 * time.Hour).Unix(),
            IssuedAt:  time.Now().Unix(),
        },
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(jp.secretKey)
}

func (jp *JWTProvider) ValidateToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        return jp.secretKey, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(*Claims); ok && token.Valid {
        return claims, nil
    }
    
    return nil, errors.New("invalid token")
}
```

## 📊 Monitoring Patterns

### **1. Health Checks**
Monitors service health and availability.

```go
type HealthChecker struct {
    checks map[string]HealthCheck
    mutex  sync.RWMutex
}

type HealthCheck interface {
    Check() error
}

type DatabaseHealthCheck struct {
    db Database
}

func (dhc *DatabaseHealthCheck) Check() error {
    return dhc.db.Ping()
}

func (hc *HealthChecker) Register(name string, check HealthCheck) {
    hc.mutex.Lock()
    defer hc.mutex.Unlock()
    hc.checks[name] = check
}

func (hc *HealthChecker) CheckAll() map[string]error {
    hc.mutex.RLock()
    defer hc.mutex.RUnlock()
    
    results := make(map[string]error)
    for name, check := range hc.checks {
        results[name] = check.Check()
    }
    return results
}
```

### **2. Metrics Collection**
Collects and aggregates system metrics.

```go
type MetricsCollector struct {
    counters map[string]int64
    gauges   map[string]float64
    histograms map[string][]float64
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

func (mc *MetricsCollector) RecordHistogram(name string, value float64) {
    mc.mutex.Lock()
    defer mc.mutex.Unlock()
    mc.histograms[name] = append(mc.histograms[name], value)
}

func (mc *MetricsCollector) GetMetrics() map[string]interface{} {
    mc.mutex.RLock()
    defer mc.mutex.RUnlock()
    
    return map[string]interface{}{
        "counters":    mc.counters,
        "gauges":      mc.gauges,
        "histograms":  mc.histograms,
    }
}
```

## 🔄 Data Processing Patterns

### **1. Map-Reduce**
Distributed data processing pattern.

```go
type MapReduce struct {
    mapper  func(interface{}) []KeyValue
    reducer func(string, []interface{}) interface{}
}

type KeyValue struct {
    Key   string
    Value interface{}
}

func (mr *MapReduce) Process(data []interface{}) map[string]interface{} {
    // Map phase
    intermediate := make(map[string][]interface{})
    for _, item := range data {
        kvs := mr.mapper(item)
        for _, kv := range kvs {
            intermediate[kv.Key] = append(intermediate[kv.Key], kv.Value)
        }
    }
    
    // Reduce phase
    result := make(map[string]interface{})
    for key, values := range intermediate {
        result[key] = mr.reducer(key, values)
    }
    
    return result
}
```

### **2. Stream Processing**
Real-time data processing.

```go
type StreamProcessor struct {
    processors []Processor
    input      <-chan interface{}
    output     chan<- interface{}
}

type Processor interface {
    Process(input interface{}) interface{}
}

func (sp *StreamProcessor) Start() {
    for data := range sp.input {
        result := data
        for _, processor := range sp.processors {
            result = processor.Process(result)
        }
        sp.output <- result
    }
}
```

## 🎯 Best Practices

### **1. Pattern Selection**
- Choose patterns based on requirements
- Consider trade-offs and complexity
- Start simple and evolve

### **2. Implementation Guidelines**
- Use appropriate concurrency primitives
- Handle errors gracefully
- Implement proper logging and monitoring
- Test thoroughly

### **3. Performance Considerations**
- Monitor resource usage
- Optimize for your use case
- Consider caching strategies
- Plan for scaling

## 🏢 Industry Examples

### **Netflix**
- **Circuit Breaker**: Hystrix for fault tolerance
- **Service Discovery**: Eureka for service registration
- **Caching**: Redis for session management

### **Uber**
- **Event Sourcing**: For trip and payment events
- **CQRS**: Separate read/write models
- **Microservices**: Domain-driven design

### **Airbnb**
- **API Gateway**: Kong for request routing
- **Database Sharding**: By geographic region
- **Caching**: Multi-level caching strategy

---

**🎉 Master these patterns to excel in system design interviews and build robust distributed systems!**

**Good luck with your system design journey! 🚀**
