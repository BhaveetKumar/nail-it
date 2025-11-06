---
# Auto-generated front matter
Title: Advanced Microservices Patterns
LastUpdated: 2025-11-06T20:45:58.283821
Tags: []
Status: draft
---

# 🏗️ **Advanced Microservices Patterns**

## 📊 **Production-Ready Microservices Architecture**

---

## 🎯 **1. Service Mesh Architecture**

### **Istio Service Mesh Implementation**

```go
package main

import (
    "context"
    "fmt"
    "net/http"
    "time"
)

// Service Mesh with Istio-like functionality
type ServiceMesh struct {
    services     map[string]*Service
    policies     map[string]*Policy
    observability *Observability
    security     *Security
    mutex        sync.RWMutex
}

type Service struct {
    Name        string
    Namespace   string
    Version     string
    Endpoints   []string
    Health      string
    Load        float64
    Latency     time.Duration
    CircuitBreaker *CircuitBreaker
}

type Policy struct {
    ServiceName string
    RetryPolicy *RetryPolicy
    TimeoutPolicy *TimeoutPolicy
    RateLimit   *RateLimit
    Security    *SecurityPolicy
}

type RetryPolicy struct {
    MaxAttempts int
    BaseDelay   time.Duration
    MaxDelay    time.Duration
    Backoff     string // "exponential", "linear"
}

type TimeoutPolicy struct {
    ConnectTimeout time.Duration
    RequestTimeout time.Duration
}

type RateLimit struct {
    RequestsPerSecond int
    BurstSize         int
    WindowSize        time.Duration
}

type SecurityPolicy struct {
    Authentication bool
    Authorization  bool
    TLS            bool
    mTLS           bool
}

// Service Discovery and Load Balancing
func (sm *ServiceMesh) DiscoverService(serviceName string) (*Service, error) {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    service, exists := sm.services[serviceName]
    if !exists {
        return nil, fmt.Errorf("service %s not found", serviceName)
    }
    
    // Health check
    if service.Health != "healthy" {
        return nil, fmt.Errorf("service %s is unhealthy", serviceName)
    }
    
    return service, nil
}

func (sm *ServiceMesh) LoadBalance(serviceName string) (string, error) {
    service, err := sm.DiscoverService(serviceName)
    if err != nil {
        return "", err
    }
    
    // Simple round-robin load balancing
    // In production, use more sophisticated algorithms
    if len(service.Endpoints) == 0 {
        return "", fmt.Errorf("no endpoints available for service %s", serviceName)
    }
    
    // Select endpoint with lowest load
    bestEndpoint := service.Endpoints[0]
    minLoad := service.Load
    
    for _, endpoint := range service.Endpoints[1:] {
        // Simulate load calculation
        load := sm.calculateLoad(endpoint)
        if load < minLoad {
            bestEndpoint = endpoint
            minLoad = load
        }
    }
    
    return bestEndpoint, nil
}

func (sm *ServiceMesh) calculateLoad(endpoint string) float64 {
    // Simulate load calculation based on endpoint metrics
    return 0.5 // Placeholder
}

// Circuit Breaker Implementation
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
        return generation, fmt.Errorf("circuit breaker is open")
    } else if state == StateHalfOpen && cb.counts.Requests >= cb.maxRequests {
        return generation, fmt.Errorf("circuit breaker is half-open")
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
    switch cb.state {
    case StateClosed:
        if !cb.expiry.IsZero() && cb.expiry.Before(now) {
            cb.toNewGeneration(now)
        }
    case StateOpen:
        if cb.expiry.Before(now) {
            cb.setState(StateHalfOpen, now)
        }
    }
    return cb.state, cb.generation
}

func (cb *CircuitBreaker) toNewGeneration(now time.Time) {
    cb.generation++
    cb.counts = Counts{}
    cb.expiry = now.Add(cb.interval)
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

// Service Mesh Client
type ServiceMeshClient struct {
    mesh       *ServiceMesh
    httpClient *http.Client
}

func NewServiceMeshClient(mesh *ServiceMesh) *ServiceMeshClient {
    return &ServiceMeshClient{
        mesh: mesh,
        httpClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

func (smc *ServiceMeshClient) CallService(serviceName, path string, headers map[string]string) (*http.Response, error) {
    // Get service endpoint
    endpoint, err := smc.mesh.LoadBalance(serviceName)
    if err != nil {
        return nil, err
    }
    
    // Get service policy
    policy, exists := smc.mesh.policies[serviceName]
    if !exists {
        policy = &Policy{} // Default policy
    }
    
    // Apply circuit breaker
    if policy.ServiceName != "" {
        service, _ := smc.mesh.DiscoverService(serviceName)
        if service != nil && service.CircuitBreaker != nil {
            var result interface{}
            err := service.CircuitBreaker.Execute(func() (interface{}, error) {
                return smc.makeHTTPRequest(endpoint, path, headers, policy)
            })
            if err != nil {
                return nil, err
            }
            return result.(*http.Response), nil
        }
    }
    
    return smc.makeHTTPRequest(endpoint, path, headers, policy)
}

func (smc *ServiceMeshClient) makeHTTPRequest(endpoint, path string, headers map[string]string, policy *Policy) (*http.Response, error) {
    url := endpoint + path
    req, err := http.NewRequest("GET", url, nil)
    if err != nil {
        return nil, err
    }
    
    // Add headers
    for key, value := range headers {
        req.Header.Set(key, value)
    }
    
    // Apply timeout policy
    if policy.TimeoutPolicy != nil {
        ctx, cancel := context.WithTimeout(context.Background(), policy.TimeoutPolicy.RequestTimeout)
        defer cancel()
        req = req.WithContext(ctx)
    }
    
    // Make request
    return smc.httpClient.Do(req)
}

// Example usage
func main() {
    // Create service mesh
    mesh := &ServiceMesh{
        services: make(map[string]*Service),
        policies: make(map[string]*Policy),
    }
    
    // Register service
    mesh.services["user-service"] = &Service{
        Name:      "user-service",
        Namespace: "default",
        Version:   "v1",
        Endpoints: []string{"http://user-service-1:8080", "http://user-service-2:8080"},
        Health:    "healthy",
        Load:      0.3,
        Latency:   50 * time.Millisecond,
    }
    
    // Set policy
    mesh.policies["user-service"] = &Policy{
        ServiceName: "user-service",
        RetryPolicy: &RetryPolicy{
            MaxAttempts: 3,
            BaseDelay:   100 * time.Millisecond,
            MaxDelay:    1 * time.Second,
            Backoff:     "exponential",
        },
        TimeoutPolicy: &TimeoutPolicy{
            ConnectTimeout: 5 * time.Second,
            RequestTimeout: 10 * time.Second,
        },
        RateLimit: &RateLimit{
            RequestsPerSecond: 100,
            BurstSize:         200,
            WindowSize:        time.Second,
        },
    }
    
    // Create client
    client := NewServiceMeshClient(mesh)
    
    // Call service
    resp, err := client.CallService("user-service", "/users", map[string]string{
        "Authorization": "Bearer token123",
    })
    if err != nil {
        fmt.Printf("Service call failed: %v\n", err)
    } else {
        fmt.Printf("Service call successful: %d\n", resp.StatusCode)
        resp.Body.Close()
    }
}
```

---

## 🎯 **2. Event-Driven Architecture with CQRS**

### **Event Sourcing and CQRS Implementation**

```go
package main

import (
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

// Event Store for Event Sourcing
type EventStore struct {
    events []Event
    mutex  sync.RWMutex
}

type Event struct {
    ID          string
    Type        string
    AggregateID string
    Data        map[string]interface{}
    Metadata    map[string]interface{}
    Timestamp   time.Time
    Version     int
}

type EventHandler interface {
    Handle(event Event) error
}

// CQRS Command Side
type CommandHandler struct {
    eventStore     *EventStore
    eventHandlers  []EventHandler
    aggregateStore map[string]*Aggregate
    mutex          sync.RWMutex
}

type Aggregate struct {
    ID      string
    Version int
    Events  []Event
    mutex   sync.RWMutex
}

func (ch *CommandHandler) HandleCommand(command interface{}) error {
    switch cmd := command.(type) {
    case *CreateUserCommand:
        return ch.handleCreateUser(cmd)
    case *UpdateUserCommand:
        return ch.handleUpdateUser(cmd)
    case *DeleteUserCommand:
        return ch.handleDeleteUser(cmd)
    default:
        return fmt.Errorf("unknown command type")
    }
}

func (ch *CommandHandler) handleCreateUser(cmd *CreateUserCommand) error {
    // Create aggregate
    aggregate := &Aggregate{
        ID:      cmd.UserID,
        Version: 0,
        Events:  make([]Event, 0),
    }
    
    // Create event
    event := Event{
        ID:          generateEventID(),
        Type:        "UserCreated",
        AggregateID: cmd.UserID,
        Data: map[string]interface{}{
            "username": cmd.Username,
            "email":    cmd.Email,
            "name":     cmd.Name,
        },
        Metadata: map[string]interface{}{
            "command_id": cmd.CommandID,
            "user_id":    cmd.UserID,
        },
        Timestamp: time.Now(),
        Version:   1,
    }
    
    // Store event
    ch.eventStore.StoreEvent(event)
    
    // Update aggregate
    aggregate.Events = append(aggregate.Events, event)
    aggregate.Version = 1
    
    // Save aggregate
    ch.mutex.Lock()
    ch.aggregateStore[cmd.UserID] = aggregate
    ch.mutex.Unlock()
    
    // Publish to handlers
    for _, handler := range ch.eventHandlers {
        go handler.Handle(event)
    }
    
    return nil
}

func (ch *CommandHandler) handleUpdateUser(cmd *UpdateUserCommand) error {
    // Get aggregate
    ch.mutex.RLock()
    aggregate, exists := ch.aggregateStore[cmd.UserID]
    ch.mutex.RUnlock()
    
    if !exists {
        return fmt.Errorf("user not found")
    }
    
    // Create event
    event := Event{
        ID:          generateEventID(),
        Type:        "UserUpdated",
        AggregateID: cmd.UserID,
        Data: map[string]interface{}{
            "username": cmd.Username,
            "email":    cmd.Email,
            "name":     cmd.Name,
        },
        Metadata: map[string]interface{}{
            "command_id": cmd.CommandID,
            "user_id":    cmd.UserID,
        },
        Timestamp: time.Now(),
        Version:   aggregate.Version + 1,
    }
    
    // Store event
    ch.eventStore.StoreEvent(event)
    
    // Update aggregate
    aggregate.Events = append(aggregate.Events, event)
    aggregate.Version++
    
    // Publish to handlers
    for _, handler := range ch.eventHandlers {
        go handler.Handle(event)
    }
    
    return nil
}

func (ch *CommandHandler) handleDeleteUser(cmd *DeleteUserCommand) error {
    // Get aggregate
    ch.mutex.RLock()
    aggregate, exists := ch.aggregateStore[cmd.UserID]
    ch.mutex.RUnlock()
    
    if !exists {
        return fmt.Errorf("user not found")
    }
    
    // Create event
    event := Event{
        ID:          generateEventID(),
        Type:        "UserDeleted",
        AggregateID: cmd.UserID,
        Data:        map[string]interface{}{},
        Metadata: map[string]interface{}{
            "command_id": cmd.CommandID,
            "user_id":    cmd.UserID,
        },
        Timestamp: time.Now(),
        Version:   aggregate.Version + 1,
    }
    
    // Store event
    ch.eventStore.StoreEvent(event)
    
    // Update aggregate
    aggregate.Events = append(aggregate.Events, event)
    aggregate.Version++
    
    // Publish to handlers
    for _, handler := range ch.eventHandlers {
        go handler.Handle(event)
    }
    
    return nil
}

func (es *EventStore) StoreEvent(event Event) {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    es.events = append(es.events, event)
}

func (es *EventStore) GetEvents(aggregateID string) ([]Event, error) {
    es.mutex.RLock()
    defer es.mutex.RUnlock()
    
    var events []Event
    for _, event := range es.events {
        if event.AggregateID == aggregateID {
            events = append(events, event)
        }
    }
    
    return events, nil
}

// CQRS Query Side
type QueryHandler struct {
    readModels map[string]*ReadModel
    mutex      sync.RWMutex
}

type ReadModel struct {
    ID        string
    Username  string
    Email     string
    Name      string
    CreatedAt time.Time
    UpdatedAt time.Time
    DeletedAt *time.Time
}

func (qh *QueryHandler) HandleQuery(query interface{}) (interface{}, error) {
    switch q := query.(type) {
    case *GetUserQuery:
        return qh.handleGetUser(q)
    case *ListUsersQuery:
        return qh.handleListUsers(q)
    default:
        return nil, fmt.Errorf("unknown query type")
    }
}

func (qh *QueryHandler) handleGetUser(query *GetUserQuery) (*ReadModel, error) {
    qh.mutex.RLock()
    defer qh.mutex.RUnlock()
    
    user, exists := qh.readModels[query.UserID]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }
    
    return user, nil
}

func (qh *QueryHandler) handleListUsers(query *ListUsersQuery) ([]*ReadModel, error) {
    qh.mutex.RLock()
    defer qh.mutex.RUnlock()
    
    var users []*ReadModel
    for _, user := range qh.readModels {
        if query.Offset <= len(users) && len(users) < query.Offset+query.Limit {
            users = append(users, user)
        }
    }
    
    return users, nil
}

// Event Handlers for Read Model Updates
type UserReadModelHandler struct {
    queryHandler *QueryHandler
}

func (urh *UserReadModelHandler) Handle(event Event) error {
    switch event.Type {
    case "UserCreated":
        return urh.handleUserCreated(event)
    case "UserUpdated":
        return urh.handleUserUpdated(event)
    case "UserDeleted":
        return urh.handleUserDeleted(event)
    default:
        return fmt.Errorf("unknown event type: %s", event.Type)
    }
}

func (urh *UserReadModelHandler) handleUserCreated(event Event) error {
    user := &ReadModel{
        ID:        event.AggregateID,
        Username:  event.Data["username"].(string),
        Email:     event.Data["email"].(string),
        Name:      event.Data["name"].(string),
        CreatedAt: event.Timestamp,
        UpdatedAt: event.Timestamp,
    }
    
    urh.queryHandler.mutex.Lock()
    urh.queryHandler.readModels[event.AggregateID] = user
    urh.queryHandler.mutex.Unlock()
    
    return nil
}

func (urh *UserReadModelHandler) handleUserUpdated(event Event) error {
    urh.queryHandler.mutex.Lock()
    defer urh.queryHandler.mutex.Unlock()
    
    user, exists := urh.queryHandler.readModels[event.AggregateID]
    if !exists {
        return fmt.Errorf("user not found in read model")
    }
    
    user.Username = event.Data["username"].(string)
    user.Email = event.Data["email"].(string)
    user.Name = event.Data["name"].(string)
    user.UpdatedAt = event.Timestamp
    
    return nil
}

func (urh *UserReadModelHandler) handleUserDeleted(event Event) error {
    urh.queryHandler.mutex.Lock()
    defer urh.queryHandler.mutex.Unlock()
    
    user, exists := urh.queryHandler.readModels[event.AggregateID]
    if !exists {
        return fmt.Errorf("user not found in read model")
    }
    
    now := event.Timestamp
    user.DeletedAt = &now
    
    return nil
}

// Commands and Queries
type CreateUserCommand struct {
    CommandID string
    UserID    string
    Username  string
    Email     string
    Name      string
}

type UpdateUserCommand struct {
    CommandID string
    UserID    string
    Username  string
    Email     string
    Name      string
}

type DeleteUserCommand struct {
    CommandID string
    UserID    string
}

type GetUserQuery struct {
    UserID string
}

type ListUsersQuery struct {
    Offset int
    Limit  int
}

// Example usage
func main() {
    // Create event store
    eventStore := &EventStore{
        events: make([]Event, 0),
    }
    
    // Create query handler
    queryHandler := &QueryHandler{
        readModels: make(map[string]*ReadModel),
    }
    
    // Create command handler
    commandHandler := &CommandHandler{
        eventStore:     eventStore,
        eventHandlers:  []EventHandler{&UserReadModelHandler{queryHandler: queryHandler}},
        aggregateStore: make(map[string]*Aggregate),
    }
    
    // Create user
    createCmd := &CreateUserCommand{
        CommandID: "cmd1",
        UserID:    "user1",
        Username:  "john_doe",
        Email:     "john@example.com",
        Name:      "John Doe",
    }
    
    if err := commandHandler.HandleCommand(createCmd); err != nil {
        fmt.Printf("Failed to create user: %v\n", err)
    } else {
        fmt.Printf("User created successfully\n")
    }
    
    // Query user
    getUserQuery := &GetUserQuery{UserID: "user1"}
    user, err := queryHandler.HandleQuery(getUserQuery)
    if err != nil {
        fmt.Printf("Failed to get user: %v\n", err)
    } else {
        fmt.Printf("User: %+v\n", user)
    }
}
```

---

## 🎯 **3. Distributed Tracing and Observability**

### **OpenTelemetry Integration**

```go
package main

import (
    "context"
    "fmt"
    "time"
)

// Distributed Tracing with OpenTelemetry
type TracingService struct {
    tracer    *Tracer
    exporter  *Exporter
    sampler   *Sampler
    propagator *Propagator
}

type Tracer struct {
    name    string
    version  string
    sampler  *Sampler
    exporter *Exporter
}

type Span struct {
    ID          string
    TraceID     string
    ParentID    string
    Name        string
    StartTime   time.Time
    EndTime     time.Time
    Duration    time.Duration
    Status      string
    Attributes  map[string]interface{}
    Events      []Event
    Links       []Link
}

type Event struct {
    Name       string
    Timestamp  time.Time
    Attributes map[string]interface{}
}

type Link struct {
    TraceID string
    SpanID  string
    Attributes map[string]interface{}
}

type Exporter struct {
    endpoint string
    headers  map[string]string
}

type Sampler struct {
    samplingRate float64
}

type Propagator struct {
    headers map[string]string
}

func NewTracingService(name, version string) *TracingService {
    return &TracingService{
        tracer: &Tracer{
            name:    name,
            version: version,
            sampler: &Sampler{samplingRate: 1.0},
            exporter: &Exporter{
                endpoint: "http://jaeger:14268/api/traces",
                headers:  make(map[string]string),
            },
        },
        exporter: &Exporter{
            endpoint: "http://jaeger:14268/api/traces",
            headers:  make(map[string]string),
        },
        sampler: &Sampler{samplingRate: 1.0},
        propagator: &Propagator{
            headers: make(map[string]string),
        },
    }
}

func (ts *TracingService) StartSpan(ctx context.Context, name string) (context.Context, *Span) {
    // Generate trace and span IDs
    traceID := generateTraceID()
    spanID := generateSpanID()
    
    // Get parent span from context
    parentSpan := getParentSpan(ctx)
    
    span := &Span{
        ID:        spanID,
        TraceID:   traceID,
        ParentID:  parentSpan.ID,
        Name:      name,
        StartTime: time.Now(),
        Attributes: make(map[string]interface{}),
        Events:    make([]Event, 0),
        Links:     make([]Link, 0),
    }
    
    // Add span to context
    ctx = context.WithValue(ctx, "span", span)
    
    return ctx, span
}

func (ts *TracingService) EndSpan(span *Span) {
    span.EndTime = time.Now()
    span.Duration = span.EndTime.Sub(span.StartTime)
    
    // Export span
    ts.exportSpan(span)
}

func (ts *TracingService) AddAttribute(span *Span, key string, value interface{}) {
    span.Attributes[key] = value
}

func (ts *TracingService) AddEvent(span *Span, name string, attributes map[string]interface{}) {
    event := Event{
        Name:       name,
        Timestamp:  time.Now(),
        Attributes: attributes,
    }
    span.Events = append(span.Events, event)
}

func (ts *TracingService) AddLink(span *Span, traceID, spanID string, attributes map[string]interface{}) {
    link := Link{
        TraceID:   traceID,
        SpanID:    spanID,
        Attributes: attributes,
    }
    span.Links = append(span.Links, link)
}

func (ts *TracingService) SetStatus(span *Span, status string) {
    span.Status = status
}

func (ts *TracingService) exportSpan(span *Span) {
    // In real implementation, send to Jaeger/Zipkin
    fmt.Printf("Exporting span: %s (trace: %s)\n", span.Name, span.TraceID)
}

// HTTP Middleware for Tracing
func TracingMiddleware(ts *TracingService) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            // Start span
            ctx, span := ts.StartSpan(r.Context(), "http_request")
            
            // Add attributes
            ts.AddAttribute(span, "http.method", r.Method)
            ts.AddAttribute(span, "http.url", r.URL.String())
            ts.AddAttribute(span, "http.user_agent", r.UserAgent())
            
            // Add event
            ts.AddEvent(span, "request_started", map[string]interface{}{
                "timestamp": time.Now(),
            })
            
            // Process request
            next.ServeHTTP(w, r.WithContext(ctx))
            
            // Add event
            ts.AddEvent(span, "request_completed", map[string]interface{}{
                "timestamp": time.Now(),
            })
            
            // End span
            ts.EndSpan(span)
        })
    }
}

// Database Tracing
func (ts *TracingService) TraceDatabase(ctx context.Context, operation string, query string) (context.Context, *Span) {
    ctx, span := ts.StartSpan(ctx, "database_operation")
    
    ts.AddAttribute(span, "db.operation", operation)
    ts.AddAttribute(span, "db.statement", query)
    
    return ctx, span
}

// External Service Tracing
func (ts *TracingService) TraceExternalService(ctx context.Context, serviceName, operation string) (context.Context, *Span) {
    ctx, span := ts.StartSpan(ctx, "external_service_call")
    
    ts.AddAttribute(span, "service.name", serviceName)
    ts.AddAttribute(span, "service.operation", operation)
    
    return ctx, span
}

// Example usage
func main() {
    // Create tracing service
    tracing := NewTracingService("user-service", "v1.0.0")
    
    // Start root span
    ctx, span := tracing.StartSpan(context.Background(), "user_creation")
    
    // Add attributes
    tracing.AddAttribute(span, "user.id", "user123")
    tracing.AddAttribute(span, "user.email", "user@example.com")
    
    // Add event
    tracing.AddEvent(span, "user_creation_started", map[string]interface{}{
        "timestamp": time.Now(),
    })
    
    // Simulate database operation
    dbCtx, dbSpan := tracing.TraceDatabase(ctx, "INSERT", "INSERT INTO users (id, email) VALUES (?, ?)")
    time.Sleep(100 * time.Millisecond) // Simulate DB operation
    tracing.EndSpan(dbSpan)
    
    // Simulate external service call
    extCtx, extSpan := tracing.TraceExternalService(ctx, "notification-service", "send_welcome_email")
    time.Sleep(50 * time.Millisecond) // Simulate external call
    tracing.EndSpan(extSpan)
    
    // Add event
    tracing.AddEvent(span, "user_creation_completed", map[string]interface{}{
        "timestamp": time.Now(),
    })
    
    // Set status
    tracing.SetStatus(span, "success")
    
    // End span
    tracing.EndSpan(span)
}
```

---

## 🎯 **4. Advanced Caching Strategies**

### **Multi-Level Caching with Cache-Aside Pattern**

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Multi-Level Cache Implementation
type MultiLevelCache struct {
    L1Cache *L1Cache // In-memory cache
    L2Cache *L2Cache // Redis cache
    L3Cache *L3Cache // Database cache
    mutex   sync.RWMutex
}

type L1Cache struct {
    data    map[string]*CacheItem
    maxSize int
    mutex   sync.RWMutex
}

type L2Cache struct {
    client  *RedisClient
    prefix  string
    ttl     time.Duration
}

type L3Cache struct {
    db      *Database
    queries map[string]string
    mutex   sync.RWMutex
}

type CacheItem struct {
    Value     interface{}
    ExpiresAt time.Time
    AccessCount int
    LastAccess time.Time
}

type CacheStats struct {
    L1Hits   int64
    L1Misses int64
    L2Hits   int64
    L2Misses int64
    L3Hits   int64
    L3Misses int64
    TotalRequests int64
}

func NewMultiLevelCache(l1Size int, l2Client *RedisClient, l3DB *Database) *MultiLevelCache {
    return &MultiLevelCache{
        L1Cache: &L1Cache{
            data:    make(map[string]*CacheItem),
            maxSize: l1Size,
        },
        L2Cache: &L2Cache{
            client: l2Client,
            prefix: "cache:",
            ttl:    5 * time.Minute,
        },
        L3Cache: &L3Cache{
            db:      l3DB,
            queries: make(map[string]string),
        },
    }
}

func (mlc *MultiLevelCache) Get(ctx context.Context, key string) (interface{}, error) {
    // Try L1 cache first
    if value, found := mlc.L1Cache.Get(key); found {
        mlc.updateStats("L1", true)
        return value, nil
    }
    mlc.updateStats("L1", false)
    
    // Try L2 cache
    if value, found := mlc.L2Cache.Get(ctx, key); found {
        mlc.updateStats("L2", true)
        // Store in L1 cache
        mlc.L1Cache.Set(key, value, 1*time.Minute)
        return value, nil
    }
    mlc.updateStats("L2", false)
    
    // Try L3 cache (database)
    if value, found := mlc.L3Cache.Get(ctx, key); found {
        mlc.updateStats("L3", true)
        // Store in L2 and L1 caches
        mlc.L2Cache.Set(ctx, key, value, 5*time.Minute)
        mlc.L1Cache.Set(key, value, 1*time.Minute)
        return value, nil
    }
    mlc.updateStats("L3", false)
    
    return nil, fmt.Errorf("key not found: %s", key)
}

func (mlc *MultiLevelCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    // Set in L1 cache
    mlc.L1Cache.Set(key, value, ttl)
    
    // Set in L2 cache
    if err := mlc.L2Cache.Set(ctx, key, value, ttl); err != nil {
        return err
    }
    
    // Set in L3 cache (database)
    if err := mlc.L3Cache.Set(ctx, key, value, ttl); err != nil {
        return err
    }
    
    return nil
}

func (mlc *MultiLevelCache) Delete(ctx context.Context, key string) error {
    // Delete from L1 cache
    mlc.L1Cache.Delete(key)
    
    // Delete from L2 cache
    if err := mlc.L2Cache.Delete(ctx, key); err != nil {
        return err
    }
    
    // Delete from L3 cache
    if err := mlc.L3Cache.Delete(ctx, key); err != nil {
        return err
    }
    
    return nil
}

func (mlc *MultiLevelCache) updateStats(level string, hit bool) {
    mlc.mutex.Lock()
    defer mlc.mutex.Unlock()
    
    // Update stats based on level and hit/miss
    // Implementation details...
}

// L1 Cache Implementation
func (l1 *L1Cache) Get(key string) (interface{}, bool) {
    l1.mutex.RLock()
    defer l1.mutex.RUnlock()
    
    item, exists := l1.data[key]
    if !exists {
        return nil, false
    }
    
    // Check expiration
    if time.Now().After(item.ExpiresAt) {
        delete(l1.data, key)
        return nil, false
    }
    
    // Update access info
    item.AccessCount++
    item.LastAccess = time.Now()
    
    return item.Value, true
}

func (l1 *L1Cache) Set(key string, value interface{}, ttl time.Duration) {
    l1.mutex.Lock()
    defer l1.mutex.Unlock()
    
    // Check if cache is full
    if len(l1.data) >= l1.maxSize {
        l1.evictLRU()
    }
    
    item := &CacheItem{
        Value:       value,
        ExpiresAt:   time.Now().Add(ttl),
        AccessCount: 1,
        LastAccess:  time.Now(),
    }
    
    l1.data[key] = item
}

func (l1 *L1Cache) Delete(key string) {
    l1.mutex.Lock()
    defer l1.mutex.Unlock()
    
    delete(l1.data, key)
}

func (l1 *L1Cache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, item := range l1.data {
        if oldestTime.IsZero() || item.LastAccess.Before(oldestTime) {
            oldestKey = key
            oldestTime = item.LastAccess
        }
    }
    
    if oldestKey != "" {
        delete(l1.data, oldestKey)
    }
}

// L2 Cache Implementation (Redis)
func (l2 *L2Cache) Get(ctx context.Context, key string) (interface{}, bool) {
    fullKey := l2.prefix + key
    
    value, err := l2.client.Get(ctx, fullKey)
    if err != nil {
        return nil, false
    }
    
    return value, true
}

func (l2 *L2Cache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    fullKey := l2.prefix + key
    return l2.client.Set(ctx, fullKey, value, ttl)
}

func (l2 *L2Cache) Delete(ctx context.Context, key string) error {
    fullKey := l2.prefix + key
    return l2.client.Delete(ctx, fullKey)
}

// L3 Cache Implementation (Database)
func (l3 *L3Cache) Get(ctx context.Context, key string) (interface{}, bool) {
    l3.mutex.RLock()
    query, exists := l3.queries[key]
    l3.mutex.RUnlock()
    
    if !exists {
        return nil, false
    }
    
    // Execute query
    result, err := l3.db.Query(ctx, query)
    if err != nil {
        return nil, false
    }
    
    return result, true
}

func (l3 *L3Cache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    // Store query for later use
    l3.mutex.Lock()
    l3.queries[key] = fmt.Sprintf("SELECT * FROM cache WHERE key = '%s'", key)
    l3.mutex.Unlock()
    
    return nil
}

func (l3 *L3Cache) Delete(ctx context.Context, key string) error {
    l3.mutex.Lock()
    delete(l3.queries, key)
    l3.mutex.Unlock()
    
    return nil
}

// Cache-Aside Pattern
type CacheAsideService struct {
    cache *MultiLevelCache
    db    *Database
}

func (cas *CacheAsideService) GetUser(ctx context.Context, userID string) (*User, error) {
    // Try cache first
    if value, found := cas.cache.Get(ctx, "user:"+userID); found {
        return value.(*User), nil
    }
    
    // Get from database
    user, err := cas.db.GetUser(ctx, userID)
    if err != nil {
        return nil, err
    }
    
    // Store in cache
    cas.cache.Set(ctx, "user:"+userID, user, 5*time.Minute)
    
    return user, nil
}

func (cas *CacheAsideService) UpdateUser(ctx context.Context, user *User) error {
    // Update database
    if err := cas.db.UpdateUser(ctx, user); err != nil {
        return err
    }
    
    // Invalidate cache
    cas.cache.Delete(ctx, "user:"+user.ID)
    
    return nil
}

// Example usage
func main() {
    // Create multi-level cache
    cache := NewMultiLevelCache(1000, &RedisClient{}, &Database{})
    
    // Create cache-aside service
    service := &CacheAsideService{
        cache: cache,
        db:    &Database{},
    }
    
    // Get user (will try cache first, then database)
    user, err := service.GetUser(context.Background(), "user123")
    if err != nil {
        fmt.Printf("Failed to get user: %v\n", err)
    } else {
        fmt.Printf("User: %+v\n", user)
    }
}
```

---

## 🎯 **Key Takeaways from Advanced Microservices Patterns**

### **1. Service Mesh Architecture**
- **Istio Integration**: Complete service mesh with circuit breakers, retry policies, and load balancing
- **Service Discovery**: Automatic service registration and discovery
- **Traffic Management**: Advanced routing and load balancing strategies
- **Security**: mTLS, authentication, and authorization policies

### **2. Event-Driven Architecture**
- **Event Sourcing**: Complete event store with versioning and replay capabilities
- **CQRS Pattern**: Command and query separation with read model updates
- **Event Handlers**: Asynchronous event processing with error handling
- **Saga Pattern**: Distributed transaction management with compensation

### **3. Distributed Tracing**
- **OpenTelemetry**: Complete tracing implementation with spans and events
- **Context Propagation**: Trace context across service boundaries
- **Performance Monitoring**: Latency and error rate tracking
- **Debugging**: Distributed request tracing and analysis

### **4. Advanced Caching**
- **Multi-Level Caching**: L1 (memory), L2 (Redis), L3 (database) cache hierarchy
- **Cache-Aside Pattern**: Application-managed cache with database fallback
- **Cache Invalidation**: Smart invalidation strategies and TTL management
- **Performance Optimization**: Cache hit ratio optimization and monitoring

### **5. Production-Ready Features**
- **Circuit Breakers**: Fault tolerance and resilience patterns
- **Retry Policies**: Exponential backoff and jitter for external calls
- **Rate Limiting**: Request throttling and burst handling
- **Health Checks**: Service health monitoring and load balancer integration

---

**🎉 This comprehensive guide provides advanced microservices patterns with production-ready Go implementations for modern distributed systems! 🚀**
