# Advanced System Design Patterns

## Table of Contents
- [Introduction](#introduction)
- [Microservices Patterns](#microservices-patterns)
- [Event-Driven Patterns](#event-driven-patterns)
- [Data Patterns](#data-patterns)
- [Caching Patterns](#caching-patterns)
- [Security Patterns](#security-patterns)
- [Performance Patterns](#performance-patterns)
- [Scalability Patterns](#scalability-patterns)

## Introduction

Advanced system design patterns provide proven solutions to complex architectural challenges. This guide covers sophisticated patterns used in large-scale distributed systems.

## Microservices Patterns

### Saga Pattern

**Problem**: Managing distributed transactions across microservices without traditional ACID properties.

**Solution**: Use a sequence of local transactions with compensating actions.

```go
// Saga Orchestrator
type SagaOrchestrator struct {
    steps    []SagaStep
    state    SagaState
    compensations map[string]CompensationFunc
}

type SagaStep struct {
    ID          string
    Service     string
    Action      func() error
    Compensation func() error
    Timeout     time.Duration
}

type SagaState struct {
    CurrentStep int
    Completed   []string
    Failed      []string
    Data        map[string]interface{}
}

func (so *SagaOrchestrator) Execute() error {
    for i, step := range so.steps {
        so.state.CurrentStep = i
        
        if err := step.Action(); err != nil {
            return so.compensate(step.ID)
        }
        
        so.state.Completed = append(so.state.Completed, step.ID)
    }
    
    return nil
}

func (so *SagaOrchestrator) compensate(failedStepID string) error {
    // Execute compensations in reverse order
    for i := len(so.state.Completed) - 1; i >= 0; i-- {
        stepID := so.state.Completed[i]
        if compensation, exists := so.compensations[stepID]; exists {
            if err := compensation(); err != nil {
                log.Printf("Compensation failed for step %s: %v", stepID, err)
            }
        }
    }
    
    return fmt.Errorf("saga failed at step %s", failedStepID)
}
```

### Circuit Breaker Pattern

**Problem**: Preventing cascading failures in distributed systems.

**Solution**: Monitor service calls and open circuit when failure threshold is reached.

```go
// Circuit Breaker Implementation
type CircuitBreaker struct {
    name          string
    maxFailures   int
    timeout       time.Duration
    resetTimeout  time.Duration
    state         State
    failures      int
    lastFailure   time.Time
    mu            sync.RWMutex
}

type State int
const (
    Closed State = iota
    Open
    HalfOpen
)

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    if cb.state == Open {
        if time.Since(cb.lastFailure) < cb.resetTimeout {
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

### Bulkhead Pattern

**Problem**: Isolating critical resources to prevent total system failure.

**Solution**: Partition resources into isolated pools.

```go
// Bulkhead Resource Pool
type BulkheadPool struct {
    name        string
    maxConns    int
    maxQueue    int
    connections chan *Connection
    queue       chan *Request
    mu          sync.RWMutex
}

type Connection struct {
    ID       string
    InUse    bool
    Created  time.Time
    LastUsed time.Time
}

func (bp *BulkheadPool) GetConnection() (*Connection, error) {
    select {
    case conn := <-bp.connections:
        conn.InUse = true
        conn.LastUsed = time.Now()
        return conn, nil
    default:
        return nil, errors.New("no available connections")
    }
}

func (bp *BulkheadPool) ReturnConnection(conn *Connection) {
    conn.InUse = false
    select {
    case bp.connections <- conn:
    default:
        // Pool is full, close connection
        conn.Close()
    }
}
```

## Event-Driven Patterns

### Event Sourcing Pattern

**Problem**: Maintaining complete audit trail and enabling time-travel queries.

**Solution**: Store all changes as a sequence of events.

```go
// Event Store
type EventStore struct {
    events map[string][]Event
    mu     sync.RWMutex
}

type Event struct {
    ID        string
    Type      string
    Data      map[string]interface{}
    Timestamp time.Time
    Version   int
}

type Aggregate struct {
    ID      string
    Version int
    Events  []Event
}

func (es *EventStore) AppendEvents(streamID string, events []Event) error {
    es.mu.Lock()
    defer es.mu.Unlock()
    
    currentVersion := len(es.events[streamID])
    
    for i, event := range events {
        event.Version = currentVersion + i + 1
        es.events[streamID] = append(es.events[streamID], event)
    }
    
    return nil
}

func (es *EventStore) GetEvents(streamID string, fromVersion int) ([]Event, error) {
    es.mu.RLock()
    defer es.mu.RUnlock()
    
    events := es.events[streamID]
    if fromVersion >= len(events) {
        return []Event{}, nil
    }
    
    return events[fromVersion:], nil
}

// Event Sourced Aggregate
func (a *Aggregate) ApplyEvent(event Event) {
    a.Events = append(a.Events, event)
    a.Version = event.Version
    
    // Apply event to aggregate state
    switch event.Type {
    case "UserCreated":
        a.applyUserCreated(event)
    case "UserUpdated":
        a.applyUserUpdated(event)
    }
}
```

### CQRS Pattern

**Problem**: Optimizing read and write operations with different requirements.

**Solution**: Separate command and query responsibilities.

```go
// Command Side
type CommandHandler struct {
    eventStore *EventStore
    bus        *CommandBus
}

type CreateUserCommand struct {
    ID    string
    Name  string
    Email string
}

func (ch *CommandHandler) HandleCreateUser(cmd CreateUserCommand) error {
    // Validate command
    if err := ch.validateCreateUser(cmd); err != nil {
        return err
    }
    
    // Create events
    events := []Event{
        {
            ID:   generateEventID(),
            Type: "UserCreated",
            Data: map[string]interface{}{
                "id":    cmd.ID,
                "name":  cmd.Name,
                "email": cmd.Email,
            },
            Timestamp: time.Now(),
        },
    }
    
    // Store events
    return ch.eventStore.AppendEvents(cmd.ID, events)
}

// Query Side
type QueryHandler struct {
    readModel *ReadModel
}

type UserQuery struct {
    ID string
}

type UserView struct {
    ID    string
    Name  string
    Email string
}

func (qh *QueryHandler) HandleGetUser(query UserQuery) (*UserView, error) {
    return qh.readModel.GetUser(query.ID)
}

// Read Model
type ReadModel struct {
    users map[string]*UserView
    mu    sync.RWMutex
}

func (rm *ReadModel) UpdateFromEvent(event Event) {
    rm.mu.Lock()
    defer rm.mu.Unlock()
    
    switch event.Type {
    case "UserCreated":
        rm.users[event.Data["id"].(string)] = &UserView{
            ID:    event.Data["id"].(string),
            Name:  event.Data["name"].(string),
            Email: event.Data["email"].(string),
        }
    }
}
```

### Event Streaming Pattern

**Problem**: Processing high-volume event streams in real-time.

**Solution**: Use event streaming platforms with partitioning and consumer groups.

```go
// Event Stream Processor
type StreamProcessor struct {
    consumer    *KafkaConsumer
    processors  map[string]EventProcessor
    partitions  int
    consumers   int
}

type EventProcessor interface {
    Process(event Event) error
    GetEventType() string
}

func (sp *StreamProcessor) Start() error {
    for i := 0; i < sp.consumers; i++ {
        go sp.processEvents(i)
    }
    return nil
}

func (sp *StreamProcessor) processEvents(consumerID int) {
    for {
        events, err := sp.consumer.Consume()
        if err != nil {
            log.Printf("Consumer %d error: %v", consumerID, err)
            continue
        }
        
        for _, event := range events {
            if processor, exists := sp.processors[event.Type]; exists {
                if err := processor.Process(event); err != nil {
                    log.Printf("Processing error: %v", err)
                }
            }
        }
    }
}

// Event Processor Implementation
type UserEventProcessor struct {
    readModel *ReadModel
}

func (uep *UserEventProcessor) Process(event Event) error {
    uep.readModel.UpdateFromEvent(event)
    return nil
}

func (uep *UserEventProcessor) GetEventType() string {
    return "UserEvent"
}
```

## Data Patterns

### CQRS with Event Sourcing

**Problem**: Complex domain models with different read/write requirements.

**Solution**: Combine CQRS with event sourcing for optimal performance.

```go
// Command Side with Event Sourcing
type CommandSide struct {
    aggregates map[string]*Aggregate
    eventStore *EventStore
    bus        *CommandBus
}

func (cs *CommandSide) HandleCommand(cmd Command) error {
    aggregate := cs.getAggregate(cmd.AggregateID)
    
    // Apply command to aggregate
    events, err := aggregate.HandleCommand(cmd)
    if err != nil {
        return err
    }
    
    // Store events
    return cs.eventStore.AppendEvents(cmd.AggregateID, events)
}

// Query Side with Read Models
type QuerySide struct {
    readModels map[string]ReadModel
    projector  *EventProjector
}

type ReadModel interface {
    Update(event Event) error
    Query(query interface{}) (interface{}, error)
}

// Materialized View
type UserMaterializedView struct {
    users map[string]*UserView
    mu    sync.RWMutex
}

func (umv *UserMaterializedView) Update(event Event) error {
    umv.mu.Lock()
    defer umv.mu.Unlock()
    
    switch event.Type {
    case "UserCreated":
        umv.users[event.Data["id"].(string)] = &UserView{
            ID:    event.Data["id"].(string),
            Name:  event.Data["name"].(string),
            Email: event.Data["email"].(string),
        }
    }
    
    return nil
}

func (umv *UserMaterializedView) Query(query interface{}) (interface{}, error) {
    umv.mu.RLock()
    defer umv.mu.RUnlock()
    
    switch q := query.(type) {
    case GetUserQuery:
        return umv.users[q.ID], nil
    case ListUsersQuery:
        return umv.listUsers(q), nil
    }
    
    return nil, errors.New("unknown query type")
}
```

### Data Sharding Pattern

**Problem**: Distributing data across multiple databases for scalability.

**Solution**: Partition data based on shard key and route requests accordingly.

```go
// Shard Manager
type ShardManager struct {
    shards    map[int]*Shard
    router    *ShardRouter
    replicas  int
    mu        sync.RWMutex
}

type Shard struct {
    ID       int
    Master   *Database
    Replicas []*Database
    Range    *ShardRange
}

type ShardRange struct {
    Start string
    End   string
}

type ShardRouter struct {
    hashFunction func(string) int
    shardCount   int
}

func (sm *ShardManager) GetShard(key string) *Shard {
    shardID := sm.router.Route(key)
    return sm.shards[shardID]
}

func (sm *ShardManager) Write(key string, value interface{}) error {
    shard := sm.GetShard(key)
    
    // Write to master
    if err := shard.Master.Write(key, value); err != nil {
        return err
    }
    
    // Replicate to replicas
    for _, replica := range shard.Replicas {
        go replica.Write(key, value)
    }
    
    return nil
}

func (sm *ShardManager) Read(key string) (interface{}, error) {
    shard := sm.GetShard(key)
    
    // Try master first
    if value, err := shard.Master.Read(key); err == nil {
        return value, nil
    }
    
    // Fallback to replicas
    for _, replica := range shard.Replicas {
        if value, err := replica.Read(key); err == nil {
            return value, nil
        }
    }
    
    return nil, errors.New("key not found")
}
```

## Caching Patterns

### Cache-Aside Pattern

**Problem**: Managing cache consistency with application data.

**Solution**: Application manages cache explicitly.

```go
// Cache-Aside Implementation
type CacheAsideService struct {
    cache    Cache
    database Database
    mu       sync.RWMutex
}

func (cas *CacheAsideService) Get(key string) (interface{}, error) {
    // Try cache first
    if value, found := cas.cache.Get(key); found {
        return value, nil
    }
    
    // Cache miss - get from database
    value, err := cas.database.Get(key)
    if err != nil {
        return nil, err
    }
    
    // Update cache
    cas.cache.Set(key, value, cas.getTTL(key))
    
    return value, nil
}

func (cas *CacheAsideService) Set(key string, value interface{}) error {
    // Update database
    if err := cas.database.Set(key, value); err != nil {
        return err
    }
    
    // Update cache
    cas.cache.Set(key, value, cas.getTTL(key))
    
    return nil
}

func (cas *CacheAsideService) Delete(key string) error {
    // Delete from database
    if err := cas.database.Delete(key); err != nil {
        return err
    }
    
    // Delete from cache
    cas.cache.Delete(key)
    
    return nil
}
```

### Write-Through Pattern

**Problem**: Ensuring cache and database consistency.

**Solution**: Write to both cache and database simultaneously.

```go
// Write-Through Implementation
type WriteThroughService struct {
    cache    Cache
    database Database
    mu       sync.RWMutex
}

func (wts *WriteThroughService) Set(key string, value interface{}) error {
    wts.mu.Lock()
    defer wts.mu.Unlock()
    
    // Write to database first
    if err := wts.database.Set(key, value); err != nil {
        return err
    }
    
    // Write to cache
    wts.cache.Set(key, value, wts.getTTL(key))
    
    return nil
}

func (wts *WriteThroughService) Get(key string) (interface{}, error) {
    wts.mu.RLock()
    defer wts.mu.RUnlock()
    
    // Try cache first
    if value, found := wts.cache.Get(key); found {
        return value, nil
    }
    
    // Cache miss - get from database
    value, err := wts.database.Get(key)
    if err != nil {
        return nil, err
    }
    
    // Update cache
    wts.cache.Set(key, value, wts.getTTL(key))
    
    return value, nil
}
```

### Write-Behind Pattern

**Problem**: Optimizing write performance with eventual consistency.

**Solution**: Write to cache immediately and batch database writes.

```go
// Write-Behind Implementation
type WriteBehindService struct {
    cache      Cache
    database   Database
    writeQueue chan WriteOperation
    batchSize  int
    flushInterval time.Duration
    mu         sync.RWMutex
}

type WriteOperation struct {
    Key   string
    Value interface{}
    Time  time.Time
}

func (wbs *WriteBehindService) Set(key string, value interface{}) error {
    wbs.mu.Lock()
    defer wbs.mu.Unlock()
    
    // Write to cache immediately
    wbs.cache.Set(key, value, wbs.getTTL(key))
    
    // Queue for database write
    select {
    case wbs.writeQueue <- WriteOperation{Key: key, Value: value, Time: time.Now()}:
    default:
        // Queue is full, handle overflow
        return errors.New("write queue is full")
    }
    
    return nil
}

func (wbs *WriteBehindService) Get(key string) (interface{}, error) {
    wbs.mu.RLock()
    defer wbs.mu.RUnlock()
    
    // Always read from cache
    if value, found := wbs.cache.Get(key); found {
        return value, nil
    }
    
    return nil, errors.New("key not found")
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

func (wbs *WriteBehindService) flushBatch(batch []WriteOperation) error {
    // Batch write to database
    return wbs.database.BatchWrite(batch)
}
```

## Security Patterns

### Zero Trust Pattern

**Problem**: Securing distributed systems without implicit trust.

**Solution**: Verify every request regardless of source.

```go
// Zero Trust Security
type ZeroTrustSecurity struct {
    authenticator *Authenticator
    authorizer    *Authorizer
    auditor       *Auditor
    encryptor     *Encryptor
}

func (zts *ZeroTrustSecurity) ProcessRequest(req *Request) (*Response, error) {
    // Authenticate
    identity, err := zts.authenticator.Authenticate(req)
    if err != nil {
        return nil, err
    }
    
    // Authorize
    if err := zts.authorizer.Authorize(identity, req); err != nil {
        return nil, err
    }
    
    // Encrypt sensitive data
    if err := zts.encryptor.EncryptRequest(req); err != nil {
        return nil, err
    }
    
    // Process request
    response, err := zts.processRequest(req)
    if err != nil {
        return nil, err
    }
    
    // Encrypt response
    if err := zts.encryptor.EncryptResponse(response); err != nil {
        return nil, err
    }
    
    // Audit
    zts.auditor.Audit(identity, req, response)
    
    return response, nil
}
```

### OAuth 2.0 Pattern

**Problem**: Secure API access without sharing credentials.

**Solution**: Use OAuth 2.0 for delegated authorization.

```go
// OAuth 2.0 Implementation
type OAuth2Server struct {
    clients    map[string]*Client
    tokens     map[string]*Token
    authCodes  map[string]*AuthCode
    mu         sync.RWMutex
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
    ExpiresIn    int
    Scope        string
    ClientID     string
    UserID       string
}

func (oas *OAuth2Server) Authorize(clientID, redirectURI, scope string) (*AuthCode, error) {
    oas.mu.Lock()
    defer oas.mu.Unlock()
    
    client, exists := oas.clients[clientID]
    if !exists {
        return nil, errors.New("invalid client")
    }
    
    if client.RedirectURI != redirectURI {
        return nil, errors.New("invalid redirect URI")
    }
    
    code := &AuthCode{
        Code:        generateAuthCode(),
        ClientID:    clientID,
        RedirectURI: redirectURI,
        Scope:       scope,
        ExpiresAt:   time.Now().Add(10 * time.Minute),
    }
    
    oas.authCodes[code.Code] = code
    
    return code, nil
}

func (oas *OAuth2Server) ExchangeCode(code, clientID, clientSecret string) (*Token, error) {
    oas.mu.Lock()
    defer oas.mu.Unlock()
    
    authCode, exists := oas.authCodes[code]
    if !exists || authCode.ExpiresAt.Before(time.Now()) {
        return nil, errors.New("invalid or expired code")
    }
    
    if authCode.ClientID != clientID {
        return nil, errors.New("invalid client")
    }
    
    client, exists := oas.clients[clientID]
    if !exists || client.Secret != clientSecret {
        return nil, errors.New("invalid client secret")
    }
    
    token := &Token{
        AccessToken:  generateAccessToken(),
        RefreshToken: generateRefreshToken(),
        ExpiresIn:    3600,
        Scope:        authCode.Scope,
        ClientID:     clientID,
        UserID:       authCode.UserID,
    }
    
    oas.tokens[token.AccessToken] = token
    delete(oas.authCodes, code)
    
    return token, nil
}
```

## Performance Patterns

### Connection Pooling Pattern

**Problem**: Managing database connections efficiently.

**Solution**: Reuse connections from a pool.

```go
// Connection Pool
type ConnectionPool struct {
    connections chan *Connection
    factory     ConnectionFactory
    maxSize     int
    minSize     int
    mu          sync.RWMutex
    stats       *PoolStats
}

type ConnectionFactory interface {
    Create() (*Connection, error)
    Close(*Connection) error
    Validate(*Connection) bool
}

type PoolStats struct {
    Active    int
    Idle      int
    Total     int
    Created   int
    Destroyed int
}

func (cp *ConnectionPool) Get() (*Connection, error) {
    select {
    case conn := <-cp.connections:
        if cp.factory.Validate(conn) {
            cp.mu.Lock()
            cp.stats.Active++
            cp.stats.Idle--
            cp.mu.Unlock()
            return conn, nil
        }
        // Invalid connection, create new one
        cp.factory.Close(conn)
        cp.mu.Lock()
        cp.stats.Destroyed++
        cp.mu.Unlock()
    default:
        // No available connections
    }
    
    // Create new connection
    conn, err := cp.factory.Create()
    if err != nil {
        return nil, err
    }
    
    cp.mu.Lock()
    cp.stats.Active++
    cp.stats.Created++
    cp.mu.Unlock()
    
    return conn, nil
}

func (cp *ConnectionPool) Put(conn *Connection) {
    cp.mu.Lock()
    defer cp.mu.Unlock()
    
    if cp.stats.Idle >= cp.maxSize {
        cp.factory.Close(conn)
        cp.stats.Destroyed++
        return
    }
    
    select {
    case cp.connections <- conn:
        cp.stats.Active--
        cp.stats.Idle++
    default:
        cp.factory.Close(conn)
        cp.stats.Destroyed++
    }
}
```

### Lazy Loading Pattern

**Problem**: Optimizing resource usage by loading data only when needed.

**Solution**: Defer object creation until first access.

```go
// Lazy Loading Implementation
type LazyLoader struct {
    factory func() (interface{}, error)
    value   interface{}
    loaded  bool
    mu      sync.RWMutex
}

func (ll *LazyLoader) Get() (interface{}, error) {
    ll.mu.RLock()
    if ll.loaded {
        value := ll.value
        ll.mu.RUnlock()
        return value, nil
    }
    ll.mu.RUnlock()
    
    ll.mu.Lock()
    defer ll.mu.Unlock()
    
    if ll.loaded {
        return ll.value, nil
    }
    
    value, err := ll.factory()
    if err != nil {
        return nil, err
    }
    
    ll.value = value
    ll.loaded = true
    
    return value, nil
}

// Lazy Collection
type LazyCollection struct {
    loader func() ([]interface{}, error)
    items  []interface{}
    loaded bool
    mu     sync.RWMutex
}

func (lc *LazyCollection) Get(index int) (interface{}, error) {
    if err := lc.ensureLoaded(); err != nil {
        return nil, err
    }
    
    lc.mu.RLock()
    defer lc.mu.RUnlock()
    
    if index >= len(lc.items) {
        return nil, errors.New("index out of range")
    }
    
    return lc.items[index], nil
}

func (lc *LazyCollection) ensureLoaded() error {
    lc.mu.RLock()
    if lc.loaded {
        lc.mu.RUnlock()
        return nil
    }
    lc.mu.RUnlock()
    
    lc.mu.Lock()
    defer lc.mu.Unlock()
    
    if lc.loaded {
        return nil
    }
    
    items, err := lc.loader()
    if err != nil {
        return err
    }
    
    lc.items = items
    lc.loaded = true
    
    return nil
}
```

## Scalability Patterns

### Horizontal Scaling Pattern

**Problem**: Scaling systems to handle increased load.

**Solution**: Add more instances and distribute load.

```go
// Load Balancer
type LoadBalancer struct {
    servers    []*Server
    algorithm  LoadBalancingAlgorithm
    healthCheck *HealthChecker
    mu         sync.RWMutex
}

type LoadBalancingAlgorithm int
const (
    RoundRobin LoadBalancingAlgorithm = iota
    LeastConnections
    WeightedRoundRobin
    IPHash
)

func (lb *LoadBalancer) SelectServer() (*Server, error) {
    lb.mu.RLock()
    defer lb.mu.RUnlock()
    
    healthyServers := lb.getHealthyServers()
    if len(healthyServers) == 0 {
        return nil, errors.New("no healthy servers available")
    }
    
    switch lb.algorithm {
    case RoundRobin:
        return lb.roundRobin(healthyServers)
    case LeastConnections:
        return lb.leastConnections(healthyServers)
    case WeightedRoundRobin:
        return lb.weightedRoundRobin(healthyServers)
    case IPHash:
        return lb.ipHash(healthyServers)
    default:
        return lb.roundRobin(healthyServers)
    }
}

func (lb *LoadBalancer) roundRobin(servers []*Server) *Server {
    // Implementation of round-robin algorithm
    return servers[0] // Simplified
}

func (lb *LoadBalancer) leastConnections(servers []*Server) *Server {
    // Implementation of least connections algorithm
    return servers[0] // Simplified
}
```

### Database Sharding Pattern

**Problem**: Scaling database operations across multiple instances.

**Solution**: Partition data across multiple databases.

```go
// Database Sharding
type ShardedDatabase struct {
    shards    map[int]*Database
    router    *ShardRouter
    replicas  map[int][]*Database
    mu        sync.RWMutex
}

func (sd *ShardedDatabase) Write(key string, value interface{}) error {
    shardID := sd.router.GetShardID(key)
    
    sd.mu.RLock()
    shard := sd.shards[shardID]
    replicas := sd.replicas[shardID]
    sd.mu.RUnlock()
    
    // Write to primary shard
    if err := shard.Write(key, value); err != nil {
        return err
    }
    
    // Replicate to replicas
    for _, replica := range replicas {
        go replica.Write(key, value)
    }
    
    return nil
}

func (sd *ShardedDatabase) Read(key string) (interface{}, error) {
    shardID := sd.router.GetShardID(key)
    
    sd.mu.RLock()
    shard := sd.shards[shardID]
    replicas := sd.replicas[shardID]
    sd.mu.RUnlock()
    
    // Try primary shard first
    if value, err := shard.Read(key); err == nil {
        return value, nil
    }
    
    // Try replicas
    for _, replica := range replicas {
        if value, err := replica.Read(key); err == nil {
            return value, nil
        }
    }
    
    return nil, errors.New("key not found")
}
```

## Conclusion

Advanced system design patterns provide:

1. **Proven Solutions**: Battle-tested approaches to common problems
2. **Scalability**: Patterns that work at scale
3. **Reliability**: Patterns that improve system resilience
4. **Performance**: Patterns that optimize system performance
5. **Maintainability**: Patterns that improve code organization
6. **Flexibility**: Patterns that adapt to changing requirements
7. **Best Practices**: Industry-standard approaches to system design

Mastering these patterns will prepare you for designing complex, scalable systems and excelling in system design interviews.

## Additional Resources

- [System Design Patterns](https://www.systemdesignpatterns.com/)
- [Microservices Patterns](https://www.microservicespatterns.com/)
- [Event-Driven Architecture](https://www.eventdrivenarchitecture.com/)
- [Data Patterns](https://www.datapatterns.com/)
- [Caching Patterns](https://www.cachingpatterns.com/)
- [Security Patterns](https://www.securitypatterns.com/)
- [Performance Patterns](https://www.performancepatterns.com/)
- [Scalability Patterns](https://www.scalabilitypatterns.com/)
