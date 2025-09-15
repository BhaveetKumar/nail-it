# 🏗️ **Microservices Architecture Deep Dive**

## 📊 **Complete Guide to Building Scalable Microservices**

---

## 🎯 **1. Microservices Design Patterns**

### **Service Discovery and Registration**

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"

    "github.com/hashicorp/consul/api"
)

type ServiceRegistry struct {
    consulClient *api.Client
    services     map[string]*Service
    mutex        sync.RWMutex
}

type Service struct {
    ID       string
    Name     string
    Address  string
    Port     int
    Tags     []string
    Health   string
    LastSeen time.Time
}

type ServiceDiscovery struct {
    registry *ServiceRegistry
    cache    map[string][]*Service
    mutex    sync.RWMutex
}

func NewServiceRegistry() (*ServiceRegistry, error) {
    config := api.DefaultConfig()
    config.Address = "localhost:8500"

    client, err := api.NewClient(config)
    if err != nil {
        return nil, err
    }

    return &ServiceRegistry{
        consulClient: client,
        services:     make(map[string]*Service),
    }, nil
}

func (sr *ServiceRegistry) Register(service *Service) error {
    registration := &api.AgentServiceRegistration{
        ID:      service.ID,
        Name:    service.Name,
        Address: service.Address,
        Port:    service.Port,
        Tags:    service.Tags,
        Check: &api.AgentServiceCheck{
            HTTP:                           fmt.Sprintf("http://%s:%d/health", service.Address, service.Port),
            Interval:                       "10s",
            Timeout:                        "3s",
            DeregisterCriticalServiceAfter: "30s",
        },
    }

    err := sr.consulClient.Agent().ServiceRegister(registration)
    if err != nil {
        return err
    }

    sr.mutex.Lock()
    sr.services[service.ID] = service
    sr.mutex.Unlock()

    return nil
}

func (sr *ServiceRegistry) Deregister(serviceID string) error {
    err := sr.consulClient.Agent().ServiceDeregister(serviceID)
    if err != nil {
        return err
    }

    sr.mutex.Lock()
    delete(sr.services, serviceID)
    sr.mutex.Unlock()

    return nil
}

func (sr *ServiceRegistry) Discover(serviceName string) ([]*Service, error) {
    services, _, err := sr.consulClient.Health().Service(serviceName, "", true, nil)
    if err != nil {
        return nil, err
    }

    var result []*Service
    for _, service := range services {
        result = append(result, &Service{
            ID:      service.Service.ID,
            Name:    service.Service.Service,
            Address: service.Service.Address,
            Port:    service.Service.Port,
            Tags:    service.Service.Tags,
            Health:  "healthy",
        })
    }

    return result, nil
}

func NewServiceDiscovery(registry *ServiceRegistry) *ServiceDiscovery {
    return &ServiceDiscovery{
        registry: registry,
        cache:    make(map[string][]*Service),
    }
}

func (sd *ServiceDiscovery) GetService(serviceName string) (*Service, error) {
    // Check cache first
    sd.mutex.RLock()
    if services, exists := sd.cache[serviceName]; exists && len(services) > 0 {
        // Return first healthy service
        for _, service := range services {
            if service.Health == "healthy" {
                sd.mutex.RUnlock()
                return service, nil
            }
        }
    }
    sd.mutex.RUnlock()

    // Discover from registry
    services, err := sd.registry.Discover(serviceName)
    if err != nil {
        return nil, err
    }

    if len(services) == 0 {
        return nil, fmt.Errorf("no services found for %s", serviceName)
    }

    // Update cache
    sd.mutex.Lock()
    sd.cache[serviceName] = services
    sd.mutex.Unlock()

    return services[0], nil
}

func (sd *ServiceDiscovery) StartHealthCheck() {
    ticker := time.NewTicker(30 * time.Second)
    go func() {
        for range ticker.C {
            sd.refreshCache()
        }
    }()
}

func (sd *ServiceDiscovery) refreshCache() {
    sd.mutex.RLock()
    serviceNames := make([]string, 0, len(sd.cache))
    for name := range sd.cache {
        serviceNames = append(serviceNames, name)
    }
    sd.mutex.RUnlock()

    for _, serviceName := range serviceNames {
        services, err := sd.registry.Discover(serviceName)
        if err != nil {
            continue
        }

        sd.mutex.Lock()
        sd.cache[serviceName] = services
        sd.mutex.Unlock()
    }
}

// Example usage
func main() {
    // Create service registry
    registry, err := NewServiceRegistry()
    if err != nil {
        log.Fatal(err)
    }

    // Register a service
    service := &Service{
        ID:      "user-service-1",
        Name:    "user-service",
        Address: "localhost",
        Port:    8080,
        Tags:    []string{"api", "user"},
    }

    if err := registry.Register(service); err != nil {
        log.Fatal(err)
    }

    // Create service discovery
    discovery := NewServiceDiscovery(registry)
    discovery.StartHealthCheck()

    // Discover service
    discoveredService, err := discovery.GetService("user-service")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Discovered service: %+v\n", discoveredService)
}
```

### **API Gateway Implementation**

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "net/http/httputil"
    "net/url"
    "strings"
    "sync"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/go-redis/redis/v8"
)

type APIGateway struct {
    routes        map[string]*Route
    rateLimiter   *RateLimiter
    authService   *AuthService
    discovery     *ServiceDiscovery
    mutex         sync.RWMutex
}

type Route struct {
    Path        string
    Methods     []string
    ServiceName string
    Proxy       *httputil.ReverseProxy
    Middleware  []gin.HandlerFunc
}

type RateLimiter struct {
    redisClient *redis.Client
    limits      map[string]int
    windows     map[string]time.Duration
}

type AuthService struct {
    jwtSecret string
    redisClient *redis.Client
}

func NewAPIGateway(discovery *ServiceDiscovery) *APIGateway {
    return &APIGateway{
        routes:      make(map[string]*Route),
        rateLimiter: NewRateLimiter(),
        authService: NewAuthService(),
        discovery:   discovery,
    }
}

func (gw *APIGateway) AddRoute(path string, methods []string, serviceName string, middleware ...gin.HandlerFunc) error {
    // Create reverse proxy
    proxy := &httputil.ReverseProxy{
        Director: func(req *http.Request) {
            // Get service instance
            service, err := gw.discovery.GetService(serviceName)
            if err != nil {
                log.Printf("Service discovery failed: %v", err)
                return
            }

            // Update request
            req.URL.Scheme = "http"
            req.URL.Host = fmt.Sprintf("%s:%d", service.Address, service.Port)
            req.Header.Set("X-Forwarded-For", req.RemoteAddr)
        },
        ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
            log.Printf("Proxy error: %v", err)
            w.WriteHeader(http.StatusBadGateway)
        },
    }

    route := &Route{
        Path:        path,
        Methods:     methods,
        ServiceName: serviceName,
        Proxy:       proxy,
        Middleware:  middleware,
    }

    gw.mutex.Lock()
    gw.routes[path] = route
    gw.mutex.Unlock()

    return nil
}

func (gw *APIGateway) HandleRequest(c *gin.Context) {
    path := c.Request.URL.Path
    method := c.Request.Method

    gw.mutex.RLock()
    route, exists := gw.routes[path]
    gw.mutex.RUnlock()

    if !exists {
        c.JSON(http.StatusNotFound, gin.H{"error": "Route not found"})
        return
    }

    // Check if method is allowed
    allowed := false
    for _, m := range route.Methods {
        if m == method {
            allowed = true
            break
        }
    }

    if !allowed {
        c.JSON(http.StatusMethodNotAllowed, gin.H{"error": "Method not allowed"})
        return
    }

    // Apply middleware
    for _, middleware := range route.Middleware {
        middleware(c)
        if c.IsAborted() {
            return
        }
    }

    // Proxy request
    route.Proxy.ServeHTTP(c.Writer, c.Request)
}

func NewRateLimiter() *RateLimiter {
    client := redis.NewClient(&redis.Options{
        Addr: "localhost:6379",
        DB:   0,
    })

    return &RateLimiter{
        redisClient: client,
        limits: map[string]int{
            "default": 100, // 100 requests per minute
            "api":     1000,
            "auth":    10,
        },
        windows: map[string]time.Duration{
            "default": time.Minute,
            "api":     time.Minute,
            "auth":    time.Minute,
        },
    }
}

func (rl *RateLimiter) IsAllowed(key string, limitType string) (bool, error) {
    limit := rl.limits["default"]
    window := rl.windows["default"]

    if l, exists := rl.limits[limitType]; exists {
        limit = l
    }
    if w, exists := rl.windows[limitType]; exists {
        window = w
    }

    // Use sliding window rate limiting
    now := time.Now()
    windowStart := now.Add(-window)

    // Count requests in window
    count, err := rl.redisClient.ZCount(context.Background(), key,
        fmt.Sprintf("%d", windowStart.Unix()),
        fmt.Sprintf("%d", now.Unix())).Result()

    if err != nil {
        return false, err
    }

    if count >= int64(limit) {
        return false, nil
    }

    // Add current request
    rl.redisClient.ZAdd(context.Background(), key, &redis.Z{
        Score:  float64(now.Unix()),
        Member: now.UnixNano(),
    })

    // Set expiration
    rl.redisClient.Expire(context.Background(), key, window)

    return true, nil
}

func NewAuthService() *AuthService {
    return &AuthService{
        jwtSecret: "your-secret-key",
        redisClient: redis.NewClient(&redis.Options{
            Addr: "localhost:6379",
            DB:   1,
        }),
    }
}

func (as *AuthService) ValidateToken(token string) (map[string]interface{}, error) {
    // Check Redis cache first
    cached, err := as.redisClient.Get(context.Background(), "token:"+token).Result()
    if err == nil {
        var claims map[string]interface{}
        if err := json.Unmarshal([]byte(cached), &claims); err == nil {
            return claims, nil
        }
    }

    // Validate JWT token
    claims, err := as.validateJWT(token)
    if err != nil {
        return nil, err
    }

    // Cache token
    claimsData, _ := json.Marshal(claims)
    as.redisClient.Set(context.Background(), "token:"+token, claimsData, time.Hour)

    return claims, nil
}

func (as *AuthService) validateJWT(token string) (map[string]interface{}, error) {
    // JWT validation logic would go here
    // For simplicity, return mock claims
    return map[string]interface{}{
        "user_id": "123",
        "role":    "user",
        "exp":     time.Now().Add(time.Hour).Unix(),
    }, nil
}

// Rate limiting middleware
func (gw *APIGateway) RateLimitMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        clientIP := c.ClientIP()
        limitType := "default"

        // Determine limit type based on path
        if strings.HasPrefix(c.Request.URL.Path, "/api/") {
            limitType = "api"
        } else if strings.HasPrefix(c.Request.URL.Path, "/auth/") {
            limitType = "auth"
        }

        allowed, err := gw.rateLimiter.IsAllowed(clientIP+":"+limitType, limitType)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Rate limit check failed"})
            c.Abort()
            return
        }

        if !allowed {
            c.JSON(http.StatusTooManyRequests, gin.H{"error": "Rate limit exceeded"})
            c.Abort()
            return
        }

        c.Next()
    }
}

// Authentication middleware
func (gw *APIGateway) AuthMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        if token == "" {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Authorization header required"})
            c.Abort()
            return
        }

        // Remove "Bearer " prefix
        if strings.HasPrefix(token, "Bearer ") {
            token = token[7:]
        }

        claims, err := gw.authService.ValidateToken(token)
        if err != nil {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
            c.Abort()
            return
        }

        // Set user context
        c.Set("user", claims)
        c.Next()
    }
}

// Example usage
func main() {
    // Create service discovery
    registry, err := NewServiceRegistry()
    if err != nil {
        log.Fatal(err)
    }

    discovery := NewServiceDiscovery(registry)
    discovery.StartHealthCheck()

    // Create API gateway
    gateway := NewAPIGateway(discovery)

    // Add routes
    gateway.AddRoute("/api/users", []string{"GET", "POST"}, "user-service",
        gateway.RateLimitMiddleware(), gateway.AuthMiddleware())
    gateway.AddRoute("/api/orders", []string{"GET", "POST"}, "order-service",
        gateway.RateLimitMiddleware(), gateway.AuthMiddleware())
    gateway.AddRoute("/auth/login", []string{"POST"}, "auth-service",
        gateway.RateLimitMiddleware())

    // Create Gin router
    r := gin.Default()

    // Add global middleware
    r.Use(gateway.RateLimitMiddleware())

    // Add routes
    r.Any("/*path", gateway.HandleRequest)

    // Start server
    log.Fatal(r.Run(":8080"))
}
```

---

## 🎯 **2. Event-Driven Architecture**

### **Event Sourcing and CQRS**

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "time"

    "github.com/go-redis/redis/v8"
)

type EventStore struct {
    redisClient *redis.Client
    streams     map[string]string
}

type Event struct {
    ID        string                 `json:"id"`
    Type      string                 `json:"type"`
    AggregateID string               `json:"aggregate_id"`
    Data      map[string]interface{} `json:"data"`
    Metadata  map[string]interface{} `json:"metadata"`
    Timestamp time.Time              `json:"timestamp"`
    Version   int                    `json:"version"`
}

type EventHandler interface {
    Handle(event *Event) error
}

type EventBus struct {
    eventStore *EventStore
    handlers   map[string][]EventHandler
    mutex      sync.RWMutex
}

func NewEventStore() *EventStore {
    client := redis.NewClient(&redis.Options{
        Addr: "localhost:6379",
        DB:   2,
    })

    return &EventStore{
        redisClient: client,
        streams:     make(map[string]string),
    }
}

func (es *EventStore) AppendEvent(streamName string, event *Event) error {
    eventData, err := json.Marshal(event)
    if err != nil {
        return err
    }

    // Use Redis Streams for event storage
    _, err = es.redisClient.XAdd(context.Background(), &redis.XAddArgs{
        Stream: streamName,
        Values: map[string]interface{}{
            "event": string(eventData),
        },
    }).Result()

    return err
}

func (es *EventStore) GetEvents(streamName string, from string, count int64) ([]*Event, error) {
    args := &redis.XReadArgs{
        Streams: []string{streamName, from},
        Count:   count,
    }

    streams, err := es.redisClient.XRead(context.Background(), args).Result()
    if err != nil {
        return nil, err
    }

    var events []*Event
    for _, stream := range streams {
        for _, message := range stream.Messages {
            eventData := message.Values["event"].(string)
            var event Event
            if err := json.Unmarshal([]byte(eventData), &event); err != nil {
                continue
            }
            events = append(events, &event)
        }
    }

    return events, nil
}

func NewEventBus(eventStore *EventStore) *EventBus {
    return &EventBus{
        eventStore: eventStore,
        handlers:   make(map[string][]EventHandler),
    }
}

func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()

    eb.handlers[eventType] = append(eb.handlers[eventType], handler)
}

func (eb *EventBus) Publish(event *Event) error {
    // Store event
    if err := eb.eventStore.AppendEvent("events", event); err != nil {
        return err
    }

    // Notify handlers
    eb.mutex.RLock()
    handlers := eb.handlers[event.Type]
    eb.mutex.RUnlock()

    for _, handler := range handlers {
        go func(h EventHandler) {
            if err := h.Handle(event); err != nil {
                log.Printf("Event handler error: %v", err)
            }
        }(handler)
    }

    return nil
}

// CQRS Command and Query Handlers
type CommandHandler interface {
    Handle(command interface{}) error
}

type QueryHandler interface {
    Handle(query interface{}) (interface{}, error)
}

type UserCommandHandler struct {
    eventBus *EventBus
}

type CreateUserCommand struct {
    UserID   string
    Username string
    Email    string
}

type UpdateUserCommand struct {
    UserID   string
    Username string
    Email    string
}

func (uch *UserCommandHandler) Handle(command interface{}) error {
    switch cmd := command.(type) {
    case *CreateUserCommand:
        return uch.handleCreateUser(cmd)
    case *UpdateUserCommand:
        return uch.handleUpdateUser(cmd)
    default:
        return fmt.Errorf("unknown command type: %T", command)
    }
}

func (uch *UserCommandHandler) handleCreateUser(cmd *CreateUserCommand) error {
    event := &Event{
        ID:          generateEventID(),
        Type:        "UserCreated",
        AggregateID: cmd.UserID,
        Data: map[string]interface{}{
            "username": cmd.Username,
            "email":    cmd.Email,
        },
        Timestamp: time.Now(),
        Version:   1,
    }

    return uch.eventBus.Publish(event)
}

func (uch *UserCommandHandler) handleUpdateUser(cmd *UpdateUserCommand) error {
    event := &Event{
        ID:          generateEventID(),
        Type:        "UserUpdated",
        AggregateID: cmd.UserID,
        Data: map[string]interface{}{
            "username": cmd.Username,
            "email":    cmd.Email,
        },
        Timestamp: time.Now(),
        Version:   2,
    }

    return uch.eventBus.Publish(event)
}

type UserQueryHandler struct {
    readModel *UserReadModel
}

type GetUserQuery struct {
    UserID string
}

type UserReadModel struct {
    Users map[string]*User
    mutex sync.RWMutex
}

type User struct {
    ID       string
    Username string
    Email    string
    Version  int
}

func NewUserReadModel() *UserReadModel {
    return &UserReadModel{
        Users: make(map[string]*User),
    }
}

func (urh *UserQueryHandler) Handle(query interface{}) (interface{}, error) {
    switch q := query.(type) {
    case *GetUserQuery:
        return urh.handleGetUser(q)
    default:
        return nil, fmt.Errorf("unknown query type: %T", query)
    }
}

func (urh *UserQueryHandler) handleGetUser(query *GetUserQuery) (*User, error) {
    urh.readModel.mutex.RLock()
    defer urh.readModel.mutex.RUnlock()

    user, exists := urh.readModel.Users[query.UserID]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }

    return user, nil
}

// Event handlers for read model updates
type UserEventHandler struct {
    readModel *UserReadModel
}

func (ueh *UserEventHandler) Handle(event *Event) error {
    ueh.readModel.mutex.Lock()
    defer ueh.readModel.mutex.Unlock()

    switch event.Type {
    case "UserCreated":
        user := &User{
            ID:       event.AggregateID,
            Username: event.Data["username"].(string),
            Email:    event.Data["email"].(string),
            Version:  event.Version,
        }
        ueh.readModel.Users[event.AggregateID] = user

    case "UserUpdated":
        if user, exists := ueh.readModel.Users[event.AggregateID]; exists {
            user.Username = event.Data["username"].(string)
            user.Email = event.Data["email"].(string)
            user.Version = event.Version
        }
    }

    return nil
}

// Example usage
func main() {
    // Create event store
    eventStore := NewEventStore()

    // Create event bus
    eventBus := NewEventBus(eventStore)

    // Create read model
    readModel := NewUserReadModel()

    // Create handlers
    commandHandler := &UserCommandHandler{eventBus: eventBus}
    queryHandler := &UserQueryHandler{readModel: readModel}
    eventHandler := &UserEventHandler{readModel: readModel}

    // Subscribe to events
    eventBus.Subscribe("UserCreated", eventHandler)
    eventBus.Subscribe("UserUpdated", eventHandler)

    // Handle commands
    createCmd := &CreateUserCommand{
        UserID:   "user-1",
        Username: "john_doe",
        Email:    "john@example.com",
    }

    if err := commandHandler.Handle(createCmd); err != nil {
        log.Fatal(err)
    }

    // Wait for event processing
    time.Sleep(100 * time.Millisecond)

    // Handle queries
    getUserQuery := &GetUserQuery{UserID: "user-1"}
    user, err := queryHandler.Handle(getUserQuery)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("User: %+v\n", user)
}
```

---

## 🎯 **3. Circuit Breaker and Resilience Patterns**

### **Circuit Breaker Implementation**

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

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
            log.Printf("Circuit breaker %s changed from %s to %s", name, from, to)
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

// Retry mechanism
type RetryConfig struct {
    MaxAttempts int
    InitialDelay time.Duration
    MaxDelay     time.Duration
    Multiplier   float64
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
            time.Sleep(delay)
            delay = time.Duration(float64(delay) * config.Multiplier)
            if delay > config.MaxDelay {
                delay = config.MaxDelay
            }
        }
    }

    return lastErr
}

// Timeout mechanism
func WithTimeout(ctx context.Context, timeout time.Duration, fn func() error) error {
    ctx, cancel := context.WithTimeout(ctx, timeout)
    defer cancel()

    done := make(chan error, 1)
    go func() {
        done <- fn()
    }()

    select {
    case err := <-done:
        return err
    case <-ctx.Done():
        return ctx.Err()
    }
}

// Example usage
func main() {
    // Create circuit breaker
    cb := NewCircuitBreaker("user-service", 3, 30*time.Second, 5*time.Second)

    // Simulate service call
    serviceCall := func() (interface{}, error) {
        // Simulate network call
        time.Sleep(100 * time.Millisecond)

        // Simulate occasional failures
        if time.Now().UnixNano()%3 == 0 {
            return nil, fmt.Errorf("service unavailable")
        }

        return "success", nil
    }

    // Execute with circuit breaker
    for i := 0; i < 10; i++ {
        result, err := cb.Execute(serviceCall)
        if err != nil {
            log.Printf("Attempt %d failed: %v", i+1, err)
        } else {
            log.Printf("Attempt %d succeeded: %v", i+1, result)
        }

        time.Sleep(1 * time.Second)
    }

    // Retry example
    retryConfig := RetryConfig{
        MaxAttempts:  3,
        InitialDelay: 100 * time.Millisecond,
        MaxDelay:     1 * time.Second,
        Multiplier:   2.0,
    }

    err := Retry(retryConfig, func() error {
        // Simulate flaky service
        if time.Now().UnixNano()%2 == 0 {
            return fmt.Errorf("temporary failure")
        }
        return nil
    })

    if err != nil {
        log.Printf("Retry failed: %v", err)
    } else {
        log.Println("Retry succeeded")
    }
}
```

---

## 🎯 **4. Distributed Tracing and Monitoring**

### **OpenTelemetry Integration**

```go
package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "time"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
    "go.opentelemetry.io/otel/trace"
)

type TracingService struct {
    tracer trace.Tracer
}

func NewTracingService(serviceName string) (*TracingService, error) {
    // Create Jaeger exporter
    exp, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint("http://localhost:14268/api/traces")))
    if err != nil {
        return nil, err
    }

    // Create resource
    res, err := resource.New(context.Background(),
        resource.WithAttributes(
            semconv.ServiceNameKey.String(serviceName),
            semconv.ServiceVersionKey.String("1.0.0"),
        ),
    )
    if err != nil {
        return nil, err
    }

    // Create tracer provider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exp),
        trace.WithResource(res),
    )

    // Set global tracer provider
    otel.SetTracerProvider(tp)

    // Set global propagator
    otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
        propagation.TraceContext{},
        propagation.Baggage{},
    ))

    return &TracingService{
        tracer: tp.Tracer(serviceName),
    }, nil
}

func (ts *TracingService) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
    return ts.tracer.Start(ctx, name, opts...)
}

func (ts *TracingService) AddSpanAttributes(span trace.Span, attrs map[string]interface{}) {
    for key, value := range attrs {
        span.SetAttributes(attribute.String(key, fmt.Sprintf("%v", value)))
    }
}

func (ts *TracingService) AddSpanEvent(span trace.Span, name string, attrs map[string]interface{}) {
    var attributes []attribute.KeyValue
    for key, value := range attrs {
        attributes = append(attributes, attribute.String(key, fmt.Sprintf("%v", value)))
    }
    span.AddEvent(name, trace.WithAttributes(attributes...))
}

// HTTP middleware for tracing
func TracingMiddleware(ts *TracingService) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            // Extract trace context from headers
            ctx := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))

            // Start span
            ctx, span := ts.StartSpan(ctx, r.URL.Path)
            defer span.End()

            // Add span attributes
            ts.AddSpanAttributes(span, map[string]interface{}{
                "http.method":     r.Method,
                "http.url":        r.URL.String(),
                "http.user_agent": r.UserAgent(),
                "http.remote_addr": r.RemoteAddr,
            })

            // Add span event
            ts.AddSpanEvent(span, "request_started", map[string]interface{}{
                "timestamp": time.Now().Unix(),
            })

            // Create response writer wrapper
            wrapped := &responseWriter{ResponseWriter: w, statusCode: 200}

            // Call next handler
            next.ServeHTTP(wrapped, r)

            // Add response attributes
            ts.AddSpanAttributes(span, map[string]interface{}{
                "http.status_code": wrapped.statusCode,
            })

            // Add span event
            ts.AddSpanEvent(span, "request_completed", map[string]interface{}{
                "timestamp": time.Now().Unix(),
            })
        })
    }
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}

// Database tracing
func (ts *TracingService) TraceDatabase(ctx context.Context, operation string, query string) (context.Context, trace.Span) {
    ctx, span := ts.StartSpan(ctx, "database."+operation)

    ts.AddSpanAttributes(span, map[string]interface{}{
        "db.operation": operation,
        "db.statement": query,
    })

    return ctx, span
}

// External service tracing
func (ts *TracingService) TraceExternalService(ctx context.Context, serviceName string, operation string) (context.Context, trace.Span) {
    ctx, span := ts.StartSpan(ctx, "external."+serviceName+"."+operation)

    ts.AddSpanAttributes(span, map[string]interface{}{
        "service.name":    serviceName,
        "service.operation": operation,
    })

    return ctx, span
}

// Example usage
func main() {
    // Create tracing service
    ts, err := NewTracingService("user-service")
    if err != nil {
        log.Fatal(err)
    }

    // Create HTTP server with tracing
    mux := http.NewServeMux()
    mux.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
        // Start span for business logic
        ctx, span := ts.StartSpan(r.Context(), "get_users")
        defer span.End()

        // Simulate database call
        ctx, dbSpan := ts.TraceDatabase(ctx, "SELECT", "SELECT * FROM users")
        time.Sleep(50 * time.Millisecond) // Simulate DB call
        dbSpan.End()

        // Simulate external service call
        ctx, extSpan := ts.TraceExternalService(ctx, "notification-service", "send_notification")
        time.Sleep(30 * time.Millisecond) // Simulate external call
        extSpan.End()

        w.Write([]byte("Users retrieved"))
    })

    // Add tracing middleware
    handler := TracingMiddleware(ts)(mux)

    // Start server
    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", handler))
}
```

---

## 🎯 **Key Takeaways**

### **1. Service Discovery and Registration**

- **Consul integration** for service discovery
- **Health checks** for service availability
- **Caching** for performance optimization
- **Load balancing** across service instances

### **2. API Gateway**

- **Route management** for microservices
- **Rate limiting** for traffic control
- **Authentication** and authorization
- **Request/response transformation**

### **3. Event-Driven Architecture**

- **Event sourcing** for audit trails
- **CQRS** for read/write separation
- **Event bus** for decoupled communication
- **Read models** for query optimization

### **4. Resilience Patterns**

- **Circuit breaker** for fault tolerance
- **Retry mechanisms** for transient failures
- **Timeout handling** for responsiveness
- **Bulkhead isolation** for resource protection

### **5. Distributed Tracing**

- **OpenTelemetry** integration
- **Span correlation** across services
- **Performance monitoring** and debugging
- **Distributed context propagation**

---

**🎉 This comprehensive guide provides deep insights into microservices architecture with practical Go implementations! 🚀**
