---
# Auto-generated front matter
Title: Advanced System Design Patterns
LastUpdated: 2025-11-06T20:45:57.712774
Tags: []
Status: draft
---

# Advanced System Design Patterns

Comprehensive advanced system design patterns for senior engineering roles.

## 🎯 Event-Driven Architecture Patterns

### Event Sourcing Pattern
```go
// Event Store Implementation
type EventStore struct {
    events    map[string][]Event
    snapshots map[string]*Snapshot
    mutex     sync.RWMutex
}

type Event struct {
    ID          string
    StreamID    string
    Type        string
    Data        map[string]interface{}
    Metadata    map[string]interface{}
    Version     int
    Timestamp   time.Time
}

type Snapshot struct {
    StreamID    string
    Version     int
    Data        map[string]interface{}
    Timestamp   time.Time
}

func (es *EventStore) AppendEvents(streamID string, events []Event, expectedVersion int) error {
    es.mutex.Lock()
    defer es.mutex.Unlock()
    
    currentEvents := es.events[streamID]
    if len(currentEvents) != expectedVersion {
        return ErrConcurrencyConflict
    }
    
    for i, event := range events {
        event.Version = expectedVersion + i + 1
        event.Timestamp = time.Now()
        currentEvents = append(currentEvents, event)
    }
    
    es.events[streamID] = currentEvents
    return nil
}

func (es *EventStore) GetEvents(streamID string, fromVersion int) ([]Event, error) {
    es.mutex.RLock()
    defer es.mutex.RUnlock()
    
    events := es.events[streamID]
    if fromVersion >= len(events) {
        return []Event{}, nil
    }
    
    return events[fromVersion:], nil
}

// Aggregate Root
type AggregateRoot struct {
    ID      string
    Version int
    Events  []Event
}

func (ar *AggregateRoot) ApplyEvent(event Event) {
    ar.Events = append(ar.Events, event)
    ar.Version++
}

func (ar *AggregateRoot) GetUncommittedEvents() []Event {
    return ar.Events
}

func (ar *AggregateRoot) MarkEventsAsCommitted() {
    ar.Events = []Event{}
}
```

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
            ID:   generateEventID(),
            Type: "UserCreated",
            Data: map[string]interface{}{
                "id":       user.ID,
                "name":     user.Name,
                "email":    user.Email,
            },
        },
    }
    
    // Store events
    if err := h.eventStore.AppendEvents(user.ID, events, 0); err != nil {
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

## 🔄 Distributed Systems Patterns

### Saga Pattern
```go
// Saga Implementation
type Saga struct {
    ID        string
    Steps     []SagaStep
    Status    SagaStatus
    Events    []Event
}

type SagaStep struct {
    ID           string
    Command      Command
    Compensation Command
    Status       StepStatus
}

type SagaStatus int

const (
    SagaPending SagaStatus = iota
    SagaRunning
    SagaCompleted
    SagaFailed
    SagaCompensated
)

func (s *Saga) Execute() error {
    s.Status = SagaRunning
    
    for i, step := range s.Steps {
        if err := s.executeStep(step); err != nil {
            // Compensate previous steps
            return s.compensate(i)
        }
    }
    
    s.Status = SagaCompleted
    return nil
}

func (s *Saga) executeStep(step SagaStep) error {
    // Execute command
    if err := s.commandBus.Execute(step.Command); err != nil {
        return err
    }
    
    step.Status = StepCompleted
    return nil
}

func (s *Saga) compensate(failedStepIndex int) error {
    // Compensate in reverse order
    for i := failedStepIndex - 1; i >= 0; i-- {
        step := s.Steps[i]
        if step.Status == StepCompleted {
            if err := s.commandBus.Execute(step.Compensation); err != nil {
                return err
            }
            step.Status = StepCompensated
        }
    }
    
    s.Status = SagaFailed
    return nil
}
```

### Circuit Breaker Pattern
```go
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
    StateOpen
    StateHalfOpen
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
```

## 🏗️ Microservices Patterns

### API Gateway Pattern
```go
// API Gateway Implementation
type APIGateway struct {
    routes      map[string]Route
    middleware  []Middleware
    loadBalancer LoadBalancer
    rateLimiter RateLimiter
    authService AuthService
    logger      Logger
}

type Route struct {
    Path        string
    Method      string
    Service     string
    Middleware  []Middleware
    RateLimit   RateLimit
    Auth        AuthConfig
}

type Middleware interface {
    Process(req *http.Request, next http.Handler) http.ResponseWriter
}

type RateLimitingMiddleware struct {
    rateLimiter RateLimiter
}

func (rlm *RateLimitingMiddleware) Process(req *http.Request, next http.Handler) http.ResponseWriter {
    // Get client IP
    clientIP := rlm.getClientIP(req)
    
    // Check rate limit
    if !rlm.rateLimiter.Allow(clientIP) {
        return rlm.rateLimiter.GetResponse()
    }
    
    // Continue to next middleware
    return next.ServeHTTP(req)
}

type AuthenticationMiddleware struct {
    authService AuthService
}

func (am *AuthenticationMiddleware) Process(req *http.Request, next http.Handler) http.ResponseWriter {
    // Extract token
    token := am.extractToken(req)
    if token == "" {
        return am.unauthorizedResponse()
    }
    
    // Validate token
    user, err := am.authService.ValidateToken(token)
    if err != nil {
        return am.unauthorizedResponse()
    }
    
    // Add user to context
    ctx := context.WithValue(req.Context(), "user", user)
    req = req.WithContext(ctx)
    
    // Continue to next middleware
    return next.ServeHTTP(req)
}

func (ag *APIGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // Find route
    route := ag.findRoute(r.URL.Path, r.Method)
    if route == nil {
        http.NotFound(w, r)
        return
    }
    
    // Apply middleware
    handler := ag.applyMiddleware(route, ag.proxyHandler)
    
    // Execute handler
    handler.ServeHTTP(w, r)
}
```

### Service Mesh Pattern
```go
// Service Mesh Implementation
type ServiceMesh struct {
    services    map[string]*Service
    sidecars    map[string]*Sidecar
    controlPlane *ControlPlane
    dataPlane   *DataPlane
}

type Service struct {
    Name        string
    Address     string
    Port        int
    Dependencies []string
    Health      HealthStatus
}

type Sidecar struct {
    ServiceID   string
    Proxy       *Proxy
    Config      *SidecarConfig
    Metrics     *MetricsCollector
}

type Proxy struct {
    Inbound    *InboundHandler
    Outbound   *OutboundHandler
    LoadBalancer *LoadBalancer
    CircuitBreaker *CircuitBreaker
}

type ControlPlane struct {
    serviceRegistry *ServiceRegistry
    configManager   *ConfigManager
    policyEngine    *PolicyEngine
}

func (sm *ServiceMesh) RegisterService(service *Service) error {
    // Register service
    if err := sm.controlPlane.serviceRegistry.Register(service); err != nil {
        return err
    }
    
    // Create sidecar
    sidecar := &Sidecar{
        ServiceID: service.Name,
        Proxy:     sm.createProxy(service),
        Config:    sm.getSidecarConfig(service),
        Metrics:   sm.createMetricsCollector(service),
    }
    
    sm.sidecars[service.Name] = sidecar
    
    // Start sidecar
    return sidecar.Start()
}

func (sm *ServiceMesh) createProxy(service *Service) *Proxy {
    return &Proxy{
        Inbound:        sm.createInboundHandler(service),
        Outbound:       sm.createOutboundHandler(service),
        LoadBalancer:   sm.createLoadBalancer(service),
        CircuitBreaker: sm.createCircuitBreaker(service),
    }
}
```

## 📊 Data Management Patterns

### Database Sharding Pattern
```go
// Shard Manager Implementation
type ShardManager struct {
    shards    map[string]*sql.DB
    router    ShardRouter
    balancer  LoadBalancer
    replicator *Replicator
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

func (sm *ShardManager) ExecuteCrossShardQuery(query string, args ...interface{}) ([]*sql.Rows, error) {
    var allRows []*sql.Rows
    
    // Execute query on all shards
    for shardName, shard := range sm.shards {
        rows, err := shard.Query(query, args...)
        if err != nil {
            return nil, err
        }
        allRows = append(allRows, rows)
    }
    
    return allRows, nil
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

### Caching Patterns
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

## 🔐 Security Patterns

### Zero Trust Architecture
```go
// Zero Trust Implementation
type ZeroTrustSystem struct {
    identityProvider *IdentityProvider
    policyEngine     *PolicyEngine
    deviceTrust      *DeviceTrust
    networkSecurity  *NetworkSecurity
    dataProtection   *DataProtection
}

type IdentityProvider struct {
    users        map[string]*User
    mfaProvider  *MFAProvider
    tokenService *TokenService
}

type User struct {
    ID           string
    Email        string
    Roles        []string
    Permissions  []string
    DeviceTrust  *DeviceTrust
    LastLogin    time.Time
}

type PolicyEngine struct {
    policies []Policy
    evaluator *PolicyEvaluator
}

type Policy struct {
    ID          string
    Name        string
    Conditions  []Condition
    Actions     []Action
    Priority    int
}

func (zt *ZeroTrustSystem) AuthenticateUser(userID, password string, deviceInfo *DeviceInfo) (*AuthResult, error) {
    // Verify user credentials
    user, err := zt.identityProvider.VerifyCredentials(userID, password)
    if err != nil {
        return nil, err
    }
    
    // Verify device trust
    if !zt.deviceTrust.IsTrusted(deviceInfo) {
        return nil, ErrDeviceNotTrusted
    }
    
    // Evaluate policies
    if !zt.policyEngine.Evaluate(user, deviceInfo) {
        return nil, ErrAccessDenied
    }
    
    // Generate token
    token, err := zt.identityProvider.tokenService.GenerateToken(user)
    if err != nil {
        return nil, err
    }
    
    return &AuthResult{
        User:  user,
        Token: token,
    }, nil
}
```

### OAuth 2.0 and JWT
```go
// OAuth 2.0 Implementation
type OAuth2Service struct {
    clientStore    ClientStore
    tokenStore     TokenStore
    userStore      UserStore
    cryptoService  CryptoService
}

type Client struct {
    ID           string
    Secret       string
    RedirectURIs []string
    Scopes       []string
    GrantTypes   []string
}

type Token struct {
    AccessToken  string
    RefreshToken string
    TokenType    string
    ExpiresIn    int
    Scope        string
}

func (o *OAuth2Service) AuthorizeCode(clientID, redirectURI, scope string) (string, error) {
    // Validate client
    client, err := o.clientStore.GetByID(clientID)
    if err != nil {
        return "", err
    }
    
    // Validate redirect URI
    if !o.validateRedirectURI(client, redirectURI) {
        return "", errors.New("invalid redirect URI")
    }
    
    // Generate authorization code
    code := o.generateAuthorizationCode(clientID, redirectURI, scope)
    
    // Store code
    if err := o.tokenStore.StoreAuthorizationCode(code, clientID, redirectURI, scope); err != nil {
        return "", err
    }
    
    return code, nil
}

func (o *OAuth2Service) ExchangeToken(code, clientID, clientSecret, redirectURI string) (*Token, error) {
    // Validate client credentials
    client, err := o.clientStore.GetByID(clientID)
    if err != nil {
        return nil, err
    }
    
    if client.Secret != clientSecret {
        return nil, errors.New("invalid client secret")
    }
    
    // Validate authorization code
    authCode, err := o.tokenStore.GetAuthorizationCode(code)
    if err != nil {
        return nil, err
    }
    
    if authCode.ClientID != clientID || authCode.RedirectURI != redirectURI {
        return nil, errors.New("invalid authorization code")
    }
    
    // Generate tokens
    accessToken := o.generateAccessToken(authCode.UserID, authCode.Scope)
    refreshToken := o.generateRefreshToken(authCode.UserID)
    
    token := &Token{
        AccessToken:  accessToken,
        RefreshToken: refreshToken,
        TokenType:    "Bearer",
        ExpiresIn:    3600,
        Scope:        authCode.Scope,
    }
    
    // Store tokens
    if err := o.tokenStore.StoreTokens(token, authCode.UserID); err != nil {
        return nil, err
    }
    
    return token, nil
}
```

## 🎯 Best Practices

### Pattern Selection
1. **Choose Appropriate Patterns**: Select patterns that fit your use case
2. **Consider Trade-offs**: Understand the benefits and costs
3. **Start Simple**: Begin with basic patterns and evolve
4. **Monitor and Iterate**: Continuously improve based on metrics

### Implementation Guidelines
1. **Consistent Implementation**: Apply patterns consistently across the system
2. **Documentation**: Document pattern usage and decisions
3. **Testing**: Test pattern implementations thoroughly
4. **Monitoring**: Monitor pattern effectiveness and performance

### Common Anti-Patterns
1. **Over-Engineering**: Don't use complex patterns for simple problems
2. **Pattern Mixing**: Avoid mixing incompatible patterns
3. **Ignoring Context**: Consider the specific context and requirements
4. **Premature Optimization**: Don't optimize before understanding the problem

---

**Last Updated**: December 2024  
**Category**: Advanced System Design Patterns  
**Complexity**: Expert Level
