---
# Auto-generated front matter
Title: Advanced System Design Comprehensive
LastUpdated: 2025-11-06T20:45:57.703228
Tags: []
Status: draft
---

# Advanced System Design Comprehensive

Comprehensive system design patterns and architectures for senior engineering interviews.

## 🎯 System Design Fundamentals

### Scalability Patterns
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Application   │    │   Database      │
│   (HAProxy)     │────│   Servers       │────│   (Sharded)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Cache Layer   │
                    │   (Redis)       │
                    └─────────────────┘
```

### High Availability Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Primary       │    │   Secondary     │    │   Tertiary      │
│   Data Center   │    │   Data Center   │    │   Data Center   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Global        │
                    │   Load Balancer │
                    └─────────────────┘
```

## 🏗️ Advanced System Design Patterns

### Microservices Architecture
```go
// Service Discovery
type ServiceRegistry struct {
    services map[string][]ServiceInstance
    mutex    sync.RWMutex
}

type ServiceInstance struct {
    ID       string
    Name     string
    Address  string
    Port     int
    Health   HealthStatus
    Metadata map[string]string
}

func (sr *ServiceRegistry) Register(instance ServiceInstance) error {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    if sr.services[instance.Name] == nil {
        sr.services[instance.Name] = []ServiceInstance{}
    }
    
    sr.services[instance.Name] = append(sr.services[instance.Name], instance)
    return nil
}

func (sr *ServiceRegistry) Discover(serviceName string) ([]ServiceInstance, error) {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    instances := sr.services[serviceName]
    if instances == nil {
        return nil, errors.New("service not found")
    }
    
    // Filter healthy instances
    healthyInstances := []ServiceInstance{}
    for _, instance := range instances {
        if instance.Health == HealthUp {
            healthyInstances = append(healthyInstances, instance)
        }
    }
    
    return healthyInstances, nil
}

// API Gateway
type APIGateway struct {
    routes      map[string]Route
    middleware  []Middleware
    loadBalancer LoadBalancer
    rateLimiter RateLimiter
}

type Route struct {
    Path        string
    Method      string
    Service     string
    Middleware  []Middleware
    RateLimit   RateLimit
}

func (ag *APIGateway) RouteRequest(req *http.Request) (*http.Response, error) {
    // Find matching route
    route := ag.findRoute(req.URL.Path, req.Method)
    if route == nil {
        return nil, errors.New("route not found")
    }
    
    // Apply middleware
    for _, middleware := range route.Middleware {
        if err := middleware.Process(req); err != nil {
            return nil, err
        }
    }
    
    // Check rate limit
    if !ag.rateLimiter.Allow(req.RemoteAddr) {
        return nil, errors.New("rate limit exceeded")
    }
    
    // Forward to service
    return ag.forwardToService(route, req)
}
```

### Event-Driven Architecture
```go
// Event Bus
type EventBus struct {
    handlers map[string][]EventHandler
    mutex    sync.RWMutex
}

type Event struct {
    ID        string
    Type      string
    Data      interface{}
    Timestamp time.Time
    Source    string
}

type EventHandler interface {
    Handle(event Event) error
    GetEventType() string
}

func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    
    if eb.handlers[eventType] == nil {
        eb.handlers[eventType] = []EventHandler{}
    }
    
    eb.handlers[eventType] = append(eb.handlers[eventType], handler)
}

func (eb *EventBus) Publish(event Event) error {
    eb.mutex.RLock()
    handlers := eb.handlers[event.Type]
    eb.mutex.RUnlock()
    
    for _, handler := range handlers {
        go func(h EventHandler) {
            if err := h.Handle(event); err != nil {
                log.Printf("Error handling event: %v", err)
            }
        }(handler)
    }
    
    return nil
}

// Event Sourcing
type EventStore struct {
    events map[string][]Event
    mutex  sync.RWMutex
}

func (es *EventStore) AppendEvents(streamID string, events []Event) error {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    if es.events[streamID] == nil {
        es.events[streamID] = []Event{}
    }
    
    es.events[streamID] = append(es.events[streamID], events...)
    return nil
}

func (es *EventStore) GetEvents(streamID string) ([]Event, error) {
    es.mutex.RLock()
    defer es.mutex.RUnlock()
    
    events := es.events[streamID]
    if events == nil {
        return []Event{}, nil
    }
    
    return events, nil
}
```

## 🔄 Data Management Patterns

### CQRS (Command Query Responsibility Segregation)
```go
// Command Side
type CommandHandler interface {
    Handle(command Command) error
}

type CreateUserCommand struct {
    ID       string
    Name     string
    Email    string
    Password string
}

type CreateUserHandler struct {
    eventStore EventStore
    repository UserRepository
}

func (h *CreateUserHandler) Handle(command CreateUserCommand) error {
    // Create user aggregate
    user := &User{
        ID:       command.ID,
        Name:     command.Name,
        Email:    command.Email,
        Password: command.Password,
    }
    
    // Generate events
    events := []Event{
        {
            ID:        generateEventID(),
            Type:      "UserCreated",
            Data:      user,
            Timestamp: time.Now(),
        },
    }
    
    // Store events
    if err := h.eventStore.AppendEvents(command.ID, events); err != nil {
        return err
    }
    
    // Update read model
    return h.repository.Save(user)
}

// Query Side
type QueryHandler interface {
    Handle(query Query) (interface{}, error)
}

type GetUserQuery struct {
    ID string
}

type GetUserHandler struct {
    readModel UserReadModel
}

func (h *GetUserHandler) Handle(query GetUserQuery) (*UserView, error) {
    return h.readModel.GetUser(query.ID)
}

type UserView struct {
    ID    string `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}
```

### Database Sharding
```go
// Shard Manager
type ShardManager struct {
    shards    map[string]*sql.DB
    router    ShardRouter
    balancer  LoadBalancer
}

type ShardRouter struct {
    hashFunction func(string) int
    shardCount   int
}

func (sr *ShardRouter) GetShard(key string) string {
    hash := sr.hashFunction(key)
    shardIndex := hash % sr.shardCount
    return fmt.Sprintf("shard_%d", shardIndex)
}

func (sm *ShardManager) ExecuteQuery(query string, args ...interface{}) (*sql.Rows, error) {
    // Determine shard based on query
    shardKey := sm.extractShardKey(query, args...)
    shardName := sm.router.GetShard(shardKey)
    
    // Get shard connection
    shard, exists := sm.shards[shardName]
    if !exists {
        return nil, errors.New("shard not found")
    }
    
    // Execute query
    return shard.Query(query, args...)
}

// Consistent Hashing
type ConsistentHash struct {
    ring     map[uint32]string
    nodes    []string
    replicas int
    mutex    sync.RWMutex
}

func (ch *ConsistentHash) AddNode(node string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    for i := 0; i < ch.replicas; i++ {
        hash := ch.hash(fmt.Sprintf("%s:%d", node, i))
        ch.ring[hash] = node
    }
    
    ch.nodes = append(ch.nodes, node)
    ch.sortRing()
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    hash := ch.hash(key)
    
    // Find the first node with hash >= key hash
    for _, nodeHash := range ch.getSortedHashes() {
        if nodeHash >= hash {
            return ch.ring[nodeHash]
        }
    }
    
    // Wrap around to first node
    return ch.ring[ch.getSortedHashes()[0]]
}
```

## 🚀 Performance Optimization

### Caching Strategies
```go
// Multi-Level Cache
type MultiLevelCache struct {
    l1Cache *cache.Cache // In-memory cache
    l2Cache *redis.Client // Redis cache
    l3Cache *sql.DB      // Database
}

func (mlc *MultiLevelCache) Get(key string) (interface{}, error) {
    // Try L1 cache first
    if value, found := mlc.l1Cache.Get(key); found {
        return value, nil
    }
    
    // Try L2 cache
    value, err := mlc.l2Cache.Get(key).Result()
    if err == nil {
        // Store in L1 cache
        mlc.l1Cache.Set(key, value, 5*time.Minute)
        return value, nil
    }
    
    // Try L3 cache (database)
    value, err = mlc.getFromDatabase(key)
    if err != nil {
        return nil, err
    }
    
    // Store in both caches
    mlc.l2Cache.Set(key, value, 1*time.Hour)
    mlc.l1Cache.Set(key, value, 5*time.Minute)
    
    return value, nil
}

// Cache-Aside Pattern
type CacheAside struct {
    cache Cache
    db    Database
}

func (ca *CacheAside) Get(key string) (interface{}, error) {
    // Try cache first
    if value, found := ca.cache.Get(key); found {
        return value, nil
    }
    
    // Get from database
    value, err := ca.db.Get(key)
    if err != nil {
        return nil, err
    }
    
    // Store in cache
    ca.cache.Set(key, value, 1*time.Hour)
    
    return value, nil
}

func (ca *CacheAside) Set(key string, value interface{}) error {
    // Update database
    if err := ca.db.Set(key, value); err != nil {
        return err
    }
    
    // Update cache
    ca.cache.Set(key, value, 1*time.Hour)
    
    return nil
}
```

### Database Optimization
```go
// Connection Pool
type ConnectionPool struct {
    connections chan *sql.DB
    factory     func() (*sql.DB, error)
    maxSize     int
    mutex       sync.Mutex
}

func (cp *ConnectionPool) Get() (*sql.DB, error) {
    select {
    case conn := <-cp.connections:
        return conn, nil
    default:
        return cp.factory()
    }
}

func (cp *ConnectionPool) Put(conn *sql.DB) {
    cp.mutex.Lock()
    defer cp.mutex.Unlock()
    
    if len(cp.connections) < cp.maxSize {
        select {
        case cp.connections <- conn:
        default:
            conn.Close()
        }
    } else {
        conn.Close()
    }
}

// Query Optimization
type QueryOptimizer struct {
    db *sql.DB
}

func (qo *QueryOptimizer) OptimizeQuery(query string) (string, error) {
    // Analyze query plan
    plan, err := qo.getQueryPlan(query)
    if err != nil {
        return "", err
    }
    
    // Check for missing indexes
    if qo.hasMissingIndex(plan) {
        return qo.suggestIndex(query), nil
    }
    
    // Check for inefficient joins
    if qo.hasInefficientJoin(plan) {
        return qo.optimizeJoins(query), nil
    }
    
    return query, nil
}

func (qo *QueryOptimizer) getQueryPlan(query string) (string, error) {
    rows, err := qo.db.Query("EXPLAIN " + query)
    if err != nil {
        return "", err
    }
    defer rows.Close()
    
    var plan strings.Builder
    for rows.Next() {
        var line string
        if err := rows.Scan(&line); err != nil {
            return "", err
        }
        plan.WriteString(line + "\n")
    }
    
    return plan.String(), nil
}
```

## 🔐 Security Patterns

### Authentication & Authorization
```go
// JWT Token Service
type JWTService struct {
    secretKey []byte
    issuer    string
    expiry    time.Duration
}

func (js *JWTService) GenerateToken(userID string, roles []string) (string, error) {
    claims := jwt.MapClaims{
        "user_id": userID,
        "roles":   roles,
        "exp":     time.Now().Add(js.expiry).Unix(),
        "iat":     time.Now().Unix(),
        "iss":     js.issuer,
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(js.secretKey)
}

func (js *JWTService) ValidateToken(tokenString string) (*Claims, error) {
    token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return js.secretKey, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
        return &Claims{
            UserID: claims["user_id"].(string),
            Roles:  claims["roles"].([]string),
        }, nil
    }
    
    return nil, errors.New("invalid token")
}

// RBAC (Role-Based Access Control)
type RBACService struct {
    roleStore      RoleStore
    permissionStore PermissionStore
    userStore      UserStore
}

func (rbac *RBACService) CheckPermission(userID, resource, action string) (bool, error) {
    // Get user roles
    roles, err := rbac.roleStore.GetUserRoles(userID)
    if err != nil {
        return false, err
    }
    
    // Check each role for permission
    for _, role := range roles {
        if rbac.hasPermission(role, resource, action) {
            return true, nil
        }
    }
    
    return false, nil
}

func (rbac *RBACService) hasPermission(role Role, resource, action string) bool {
    for _, permission := range role.Permissions {
        if permission == fmt.Sprintf("%s:%s", resource, action) {
            return true
        }
    }
    return false
}
```

## 📊 Monitoring & Observability

### Metrics Collection
```go
// Prometheus Metrics
type MetricsCollector struct {
    requestDuration prometheus.HistogramVec
    requestTotal    prometheus.CounterVec
    activeConnections prometheus.Gauge
}

func NewMetricsCollector() *MetricsCollector {
    return &MetricsCollector{
        requestDuration: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "http_request_duration_seconds",
                Help: "Duration of HTTP requests",
            },
            []string{"method", "endpoint", "status"},
        ),
        requestTotal: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "http_requests_total",
                Help: "Total number of HTTP requests",
            },
            []string{"method", "endpoint", "status"},
        ),
        activeConnections: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "active_connections",
                Help: "Number of active connections",
            },
        ),
    }
}

func (mc *MetricsCollector) RecordRequest(method, endpoint, status string, duration float64) {
    mc.requestDuration.WithLabelValues(method, endpoint, status).Observe(duration)
    mc.requestTotal.WithLabelValues(method, endpoint, status).Inc()
}

// Distributed Tracing
type TracingService struct {
    tracer trace.Tracer
}

func (ts *TracingService) StartSpan(ctx context.Context, name string) (context.Context, trace.Span) {
    return ts.tracer.Start(ctx, name)
}

func (ts *TracingService) AddSpanAttributes(span trace.Span, attrs map[string]interface{}) {
    for key, value := range attrs {
        span.SetAttributes(attribute.String(key, fmt.Sprintf("%v", value)))
    }
}
```

## 🎯 System Design Interview Questions

### Question 1: Design a URL Shortener
**Requirements:**
- Shorten long URLs to short URLs
- Redirect short URLs to original URLs
- Handle 100M URLs per day
- 99.9% availability

**Solution:**
```go
// URL Shortener Service
type URLShortener struct {
    storage    URLStorage
    cache      Cache
    generator  IDGenerator
    analytics  AnalyticsService
}

type URLStorage interface {
    Store(shortURL, longURL string) error
    Get(shortURL string) (string, error)
}

type IDGenerator interface {
    Generate() string
}

// Base62 ID Generator
type Base62Generator struct {
    characters string
}

func (bg *Base62Generator) Generate() string {
    // Generate random 6-character base62 string
    result := make([]byte, 6)
    for i := range result {
        result[i] = bg.characters[rand.Intn(len(bg.characters))]
    }
    return string(result)
}

func (us *URLShortener) ShortenURL(longURL string) (string, error) {
    // Generate short URL
    shortURL := us.generator.Generate()
    
    // Store mapping
    if err := us.storage.Store(shortURL, longURL); err != nil {
        return "", err
    }
    
    // Cache for fast access
    us.cache.Set(shortURL, longURL, 24*time.Hour)
    
    return shortURL, nil
}

func (us *URLShortener) Redirect(shortURL string) (string, error) {
    // Try cache first
    if longURL, found := us.cache.Get(shortURL); found {
        us.analytics.RecordRedirect(shortURL)
        return longURL.(string), nil
    }
    
    // Get from storage
    longURL, err := us.storage.Get(shortURL)
    if err != nil {
        return "", err
    }
    
    // Cache for future requests
    us.cache.Set(shortURL, longURL, 24*time.Hour)
    
    // Record analytics
    us.analytics.RecordRedirect(shortURL)
    
    return longURL, nil
}
```

### Question 2: Design a Chat System
**Requirements:**
- Real-time messaging
- Support 1M concurrent users
- Message persistence
- Group chats

**Solution:**
```go
// Chat Service
type ChatService struct {
    messageStore MessageStore
    userStore    UserStore
    roomStore    RoomStore
    websocket    WebSocketManager
    pubsub       PubSub
}

type Message struct {
    ID        string
    UserID    string
    RoomID    string
    Content   string
    Timestamp time.Time
}

type WebSocketManager struct {
    connections map[string]*websocket.Conn
    mutex       sync.RWMutex
}

func (ws *WebSocketManager) AddConnection(userID string, conn *websocket.Conn) {
    ws.mutex.Lock()
    defer ws.mutex.Unlock()
    ws.connections[userID] = conn
}

func (ws *WebSocketManager) SendMessage(userID string, message Message) error {
    ws.mutex.RLock()
    conn, exists := ws.connections[userID]
    ws.mutex.RUnlock()
    
    if !exists {
        return errors.New("user not connected")
    }
    
    return conn.WriteJSON(message)
}

func (cs *ChatService) SendMessage(userID, roomID, content string) error {
    // Create message
    message := Message{
        ID:        generateMessageID(),
        UserID:    userID,
        RoomID:    roomID,
        Content:   content,
        Timestamp: time.Now(),
    }
    
    // Store message
    if err := cs.messageStore.Store(message); err != nil {
        return err
    }
    
    // Publish to room
    if err := cs.pubsub.Publish(roomID, message); err != nil {
        return err
    }
    
    return nil
}

func (cs *ChatService) JoinRoom(userID, roomID string) error {
    // Add user to room
    if err := cs.roomStore.AddUser(roomID, userID); err != nil {
        return err
    }
    
    // Subscribe to room messages
    return cs.pubsub.Subscribe(roomID, func(message Message) {
        cs.websocket.SendMessage(userID, message)
    })
}
```

## 🎯 Best Practices

### System Design Principles
1. **Scalability**: Design for horizontal scaling
2. **Availability**: Implement redundancy and failover
3. **Consistency**: Choose appropriate consistency model
4. **Performance**: Optimize for latency and throughput
5. **Security**: Implement proper security measures

### Interview Preparation
1. **Clarify Requirements**: Ask clarifying questions
2. **Estimate Scale**: Calculate storage, bandwidth, and compute needs
3. **Design Core Components**: Start with basic components
4. **Identify Bottlenecks**: Find and address bottlenecks
5. **Discuss Trade-offs**: Explain design decisions and trade-offs

---

**Last Updated**: December 2024  
**Category**: Advanced System Design Comprehensive  
**Complexity**: Expert Level
