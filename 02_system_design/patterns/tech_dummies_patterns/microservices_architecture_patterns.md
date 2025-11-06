---
# Auto-generated front matter
Title: Microservices Architecture Patterns
LastUpdated: 2025-11-06T20:45:57.716075
Tags: []
Status: draft
---

# Microservices Architecture Patterns - Tech Dummies Methodology

## Table of Contents
1. [Introduction](#introduction)
2. [Microservices Fundamentals](#microservices-fundamentals)
3. [Communication Patterns](#communication-patterns)
4. [Data Management Patterns](#data-management-patterns)
5. [Deployment Patterns](#deployment-patterns)
6. [Observability Patterns](#observability-patterns)
7. [Security Patterns](#security-patterns)
8. [Golang Implementation](#golang-implementation)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)

## Introduction

This guide is based on Tech Dummies' comprehensive approach to microservices architecture. It focuses on practical patterns, real-world implementations, and common challenges in building distributed systems.

### Key Principles
- **Single Responsibility**: Each service has one business capability
- **Decentralized**: Services are independently deployable
- **Fault Tolerant**: System continues working despite service failures
- **Observable**: Services provide insights into their behavior
- **Resilient**: System can recover from failures

## Microservices Fundamentals

### Service Decomposition
```go
// E-commerce Microservices Example
type ECommerceServices struct {
    // User Management
    UserService        *UserService
    AuthService        *AuthService
    ProfileService     *ProfileService
    
    // Product Management
    ProductService     *ProductService
    InventoryService   *InventoryService
    CatalogService     *CatalogService
    
    // Order Management
    OrderService       *OrderService
    PaymentService     *PaymentService
    ShippingService    *ShippingService
    
    // Support Services
    NotificationService *NotificationService
    AnalyticsService   *AnalyticsService
    AuditService       *AuditService
}

// Service Interface Definition
type ServiceInterface interface {
    Start() error
    Stop() error
    Health() HealthStatus
    Metrics() map[string]interface{}
}

// Base Service Implementation
type BaseService struct {
    Name        string
    Version     string
    Port        int
    Dependencies []string
    Health      HealthStatus
    Metrics     map[string]interface{}
    mutex       sync.RWMutex
}

func (bs *BaseService) Start() error {
    // Service startup logic
    bs.Health = HealthStatusHealthy
    return nil
}

func (bs *BaseService) Stop() error {
    // Service shutdown logic
    bs.Health = HealthStatusStopped
    return nil
}

func (bs *BaseService) Health() HealthStatus {
    bs.mutex.RLock()
    defer bs.mutex.RUnlock()
    return bs.Health
}
```

### Service Discovery
```go
// Service Registry Interface
type ServiceRegistry interface {
    Register(service ServiceInfo) error
    Deregister(serviceID string) error
    Discover(serviceName string) ([]ServiceInfo, error)
    Watch(serviceName string) (<-chan []ServiceInfo, error)
}

// Service Information
type ServiceInfo struct {
    ID          string
    Name        string
    Version     string
    Address     string
    Port        int
    Health      HealthStatus
    Tags        map[string]string
    LastSeen    time.Time
}

// Consul-based Service Registry
type ConsulRegistry struct {
    client *consul.Client
    config *consul.Config
}

func (cr *ConsulRegistry) Register(service ServiceInfo) error {
    registration := &consul.AgentServiceRegistration{
        ID:      service.ID,
        Name:    service.Name,
        Port:    service.Port,
        Address: service.Address,
        Tags:    convertTags(service.Tags),
        Check: &consul.AgentServiceCheck{
            HTTP:                           fmt.Sprintf("http://%s:%d/health", service.Address, service.Port),
            Interval:                       "10s",
            Timeout:                        "3s",
            DeregisterCriticalServiceAfter: "30s",
        },
    }
    
    return cr.client.Agent().ServiceRegister(registration)
}

func (cr *ConsulRegistry) Discover(serviceName string) ([]ServiceInfo, error) {
    services, _, err := cr.client.Health().Service(serviceName, "", true, nil)
    if err != nil {
        return nil, err
    }
    
    var serviceInfos []ServiceInfo
    for _, service := range services {
        serviceInfos = append(serviceInfos, ServiceInfo{
            ID:       service.Service.ID,
            Name:     service.Service.Service,
            Version:  service.Service.Tags[0], // Assuming version is first tag
            Address:  service.Service.Address,
            Port:     service.Service.Port,
            Health:   HealthStatusHealthy,
            Tags:     convertConsulTags(service.Service.Tags),
            LastSeen: time.Now(),
        })
    }
    
    return serviceInfos, nil
}
```

## Communication Patterns

### Synchronous Communication
```go
// HTTP Client with Circuit Breaker
type HTTPClient struct {
    client        *http.Client
    circuitBreaker *CircuitBreaker
    retryPolicy   *RetryPolicy
    timeout       time.Duration
}

type RetryPolicy struct {
    MaxRetries int
    Backoff    time.Duration
    Multiplier float64
}

func (hc *HTTPClient) Get(url string) (*http.Response, error) {
    return hc.circuitBreaker.Call(func() (*http.Response, error) {
        return hc.executeWithRetry(func() (*http.Response, error) {
            ctx, cancel := context.WithTimeout(context.Background(), hc.timeout)
            defer cancel()
            
            req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
            if err != nil {
                return nil, err
            }
            
            return hc.client.Do(req)
        })
    })
}

func (hc *HTTPClient) executeWithRetry(fn func() (*http.Response, error)) (*http.Response, error) {
    var lastErr error
    
    for i := 0; i <= hc.retryPolicy.MaxRetries; i++ {
        resp, err := fn()
        if err == nil {
            return resp, nil
        }
        
        lastErr = err
        
        if i < hc.retryPolicy.MaxRetries {
            backoff := time.Duration(float64(hc.retryPolicy.Backoff) * math.Pow(hc.retryPolicy.Multiplier, float64(i)))
            time.Sleep(backoff)
        }
    }
    
    return nil, lastErr
}

// gRPC Client with Load Balancing
type GRPCClient struct {
    conn     *grpc.ClientConn
    balancer *LoadBalancer
    services map[string]interface{}
}

func (gc *GRPCClient) Call(serviceName, method string, req interface{}) (interface{}, error) {
    service, exists := gc.services[serviceName]
    if !exists {
        return nil, fmt.Errorf("service %s not found", serviceName)
    }
    
    // Get service instance from load balancer
    instance := gc.balancer.GetInstance(serviceName)
    if instance == nil {
        return nil, fmt.Errorf("no healthy instances for service %s", serviceName)
    }
    
    // Make gRPC call
    return gc.makeGRPCCall(instance, method, req)
}
```

### Asynchronous Communication
```go
// Message Queue Interface
type MessageQueue interface {
    Publish(topic string, message interface{}) error
    Subscribe(topic string, handler MessageHandler) error
    Close() error
}

type MessageHandler func(message []byte) error

// RabbitMQ Implementation
type RabbitMQ struct {
    conn    *amqp.Connection
    channel *amqp.Channel
    config  *RabbitMQConfig
}

type RabbitMQConfig struct {
    URL      string
    Exchange string
    Queues   map[string]QueueConfig
}

type QueueConfig struct {
    Name       string
    Durable    bool
    AutoDelete bool
    Exclusive  bool
    NoWait     bool
}

func (rmq *RabbitMQ) Publish(topic string, message interface{}) error {
    body, err := json.Marshal(message)
    if err != nil {
        return err
    }
    
    return rmq.channel.Publish(
        rmq.config.Exchange,
        topic,
        false, // mandatory
        false, // immediate
        amqp.Publishing{
            ContentType: "application/json",
            Body:        body,
        },
    )
}

func (rmq *RabbitMQ) Subscribe(topic string, handler MessageHandler) error {
    queue, err := rmq.channel.QueueDeclare(
        topic,
        true,  // durable
        false, // auto-delete
        false, // exclusive
        false, // no-wait
        nil,   // arguments
    )
    if err != nil {
        return err
    }
    
    msgs, err := rmq.channel.Consume(
        queue.Name,
        "",    // consumer
        true,  // auto-ack
        false, // exclusive
        false, // no-local
        false, // no-wait
        nil,   // args
    )
    if err != nil {
        return err
    }
    
    go func() {
        for msg := range msgs {
            if err := handler(msg.Body); err != nil {
                log.Printf("Error processing message: %v", err)
            }
        }
    }()
    
    return nil
}

// Event Sourcing Pattern
type EventStore interface {
    Append(streamID string, events []Event) error
    GetEvents(streamID string, fromVersion int) ([]Event, error)
    GetSnapshot(streamID string) (*Snapshot, error)
    SaveSnapshot(streamID string, snapshot *Snapshot) error
}

type Event struct {
    ID        string
    StreamID  string
    Type      string
    Data      []byte
    Version   int
    Timestamp time.Time
}

type Snapshot struct {
    StreamID  string
    Version   int
    Data      []byte
    Timestamp time.Time
}

// CQRS Pattern
type CommandHandler interface {
    Handle(command Command) error
}

type QueryHandler interface {
    Handle(query Query) (interface{}, error)
}

type CommandBus struct {
    handlers map[string]CommandHandler
}

type QueryBus struct {
    handlers map[string]QueryHandler
}

func (cb *CommandBus) Register(commandType string, handler CommandHandler) {
    cb.handlers[commandType] = handler
}

func (cb *CommandBus) Execute(command Command) error {
    handler, exists := cb.handlers[command.Type()]
    if !exists {
        return fmt.Errorf("no handler for command type %s", command.Type())
    }
    
    return handler.Handle(command)
}
```

## Data Management Patterns

### Database per Service
```go
// Service-specific Database
type UserServiceDB struct {
    db *sql.DB
}

func (usdb *UserServiceDB) CreateUser(user *User) error {
    query := `
        INSERT INTO users (id, email, password_hash, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5)
    `
    
    _, err := usdb.db.Exec(query, user.ID, user.Email, user.PasswordHash, user.CreatedAt, user.UpdatedAt)
    return err
}

func (usdb *UserServiceDB) GetUser(id string) (*User, error) {
    query := `SELECT id, email, password_hash, created_at, updated_at FROM users WHERE id = $1`
    
    row := usdb.db.QueryRow(query, id)
    user := &User{}
    
    err := row.Scan(&user.ID, &user.Email, &user.PasswordHash, &user.CreatedAt, &user.UpdatedAt)
    if err != nil {
        return nil, err
    }
    
    return user, nil
}

// Saga Pattern for Distributed Transactions
type SagaOrchestrator struct {
    steps []SagaStep
    state SagaState
}

type SagaStep struct {
    Name        string
    Compensate  func() error
    Execute     func() error
}

type SagaState int

const (
    SagaStateStarted SagaState = iota
    SagaStateExecuting
    SagaStateCompleted
    SagaStateCompensating
    SagaStateFailed
)

func (so *SagaOrchestrator) Execute() error {
    so.state = SagaStateExecuting
    
    for i, step := range so.steps {
        if err := step.Execute(); err != nil {
            so.state = SagaStateCompensating
            return so.compensate(i)
        }
    }
    
    so.state = SagaStateCompleted
    return nil
}

func (so *SagaOrchestrator) compensate(failedStep int) error {
    for i := failedStep - 1; i >= 0; i-- {
        if err := so.steps[i].Compensate(); err != nil {
            log.Printf("Failed to compensate step %s: %v", so.steps[i].Name, err)
        }
    }
    
    so.state = SagaStateFailed
    return fmt.Errorf("saga execution failed at step %d", failedStep)
}
```

### Event Sourcing
```go
// Event Store Implementation
type EventStoreImpl struct {
    db *sql.DB
}

func (es *EventStoreImpl) Append(streamID string, events []Event) error {
    tx, err := es.db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    for _, event := range events {
        query := `
            INSERT INTO events (id, stream_id, type, data, version, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6)
        `
        
        _, err := tx.Exec(query, event.ID, event.StreamID, event.Type, event.Data, event.Version, event.Timestamp)
        if err != nil {
            return err
        }
    }
    
    return tx.Commit()
}

func (es *EventStoreImpl) GetEvents(streamID string, fromVersion int) ([]Event, error) {
    query := `
        SELECT id, stream_id, type, data, version, timestamp
        FROM events
        WHERE stream_id = $1 AND version >= $2
        ORDER BY version
    `
    
    rows, err := es.db.Query(query, streamID, fromVersion)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var events []Event
    for rows.Next() {
        var event Event
        err := rows.Scan(&event.ID, &event.StreamID, &event.Type, &event.Data, &event.Version, &event.Timestamp)
        if err != nil {
            return nil, err
        }
        events = append(events, event)
    }
    
    return events, nil
}

// Aggregate Root
type UserAggregate struct {
    ID        string
    Email     string
    Version   int
    Events    []Event
    mutex     sync.RWMutex
}

func (ua *UserAggregate) CreateUser(email string) error {
    ua.mutex.Lock()
    defer ua.mutex.Unlock()
    
    event := Event{
        ID:        generateID(),
        StreamID:  ua.ID,
        Type:      "UserCreated",
        Data:      []byte(fmt.Sprintf(`{"email":"%s"}`, email)),
        Version:   ua.Version + 1,
        Timestamp: time.Now(),
    }
    
    ua.Events = append(ua.Events, event)
    ua.Email = email
    ua.Version++
    
    return nil
}

func (ua *UserAggregate) UpdateEmail(newEmail string) error {
    ua.mutex.Lock()
    defer ua.mutex.Unlock()
    
    event := Event{
        ID:        generateID(),
        StreamID:  ua.ID,
        Type:      "UserEmailUpdated",
        Data:      []byte(fmt.Sprintf(`{"email":"%s"}`, newEmail)),
        Version:   ua.Version + 1,
        Timestamp: time.Now(),
    }
    
    ua.Events = append(ua.Events, event)
    ua.Email = newEmail
    ua.Version++
    
    return nil
}
```

## Deployment Patterns

### Containerization
```go
// Docker Configuration
type DockerConfig struct {
    Image       string
    Tag         string
    Port        int
    Environment map[string]string
    Volumes     []VolumeMount
    Resources   ResourceLimits
}

type VolumeMount struct {
    HostPath      string
    ContainerPath string
    ReadOnly      bool
}

type ResourceLimits struct {
    CPU    string
    Memory string
}

// Kubernetes Deployment
type K8sDeployment struct {
    Name        string
    Namespace   string
    Replicas    int32
    Image       string
    Port        int32
    Environment map[string]string
    Resources   ResourceRequirements
}

type ResourceRequirements struct {
    Requests ResourceList
    Limits   ResourceList
}

type ResourceList struct {
    CPU    string
    Memory string
}

// Service Mesh Configuration
type ServiceMeshConfig struct {
    SidecarProxy bool
    TrafficPolicy TrafficPolicy
    SecurityPolicy SecurityPolicy
    Observability ObservabilityConfig
}

type TrafficPolicy struct {
    LoadBalancing string
    CircuitBreaker bool
    RetryPolicy   RetryPolicy
}

type SecurityPolicy struct {
    mTLS        bool
    Authorization bool
    RateLimit   bool
}

type ObservabilityConfig struct {
    Tracing     bool
    Metrics     bool
    Logging     bool
    Distributed bool
}
```

### Blue-Green Deployment
```go
// Blue-Green Deployment Manager
type BlueGreenDeployment struct {
    BlueService  *Service
    GreenService *Service
    ActiveColor  string
    LoadBalancer *LoadBalancer
}

func (bgd *BlueGreenDeployment) Deploy(newVersion string) error {
    // Determine inactive color
    inactiveColor := "blue"
    if bgd.ActiveColor == "blue" {
        inactiveColor = "green"
    }
    
    // Deploy to inactive environment
    var inactiveService *Service
    if inactiveColor == "blue" {
        inactiveService = bgd.BlueService
    } else {
        inactiveService = bgd.GreenService
    }
    
    if err := inactiveService.Deploy(newVersion); err != nil {
        return err
    }
    
    // Health check
    if err := inactiveService.HealthCheck(); err != nil {
        return err
    }
    
    // Switch traffic
    if err := bgd.switchTraffic(inactiveColor); err != nil {
        return err
    }
    
    // Update active color
    bgd.ActiveColor = inactiveColor
    
    return nil
}

func (bgd *BlueGreenDeployment) switchTraffic(color string) error {
    // Update load balancer configuration
    return bgd.LoadBalancer.UpdateBackends(color)
}

// Canary Deployment
type CanaryDeployment struct {
    StableService *Service
    CanaryService *Service
    TrafficSplit  int // Percentage of traffic to canary
    Metrics       *MetricsCollector
}

func (cd *CanaryDeployment) Deploy(newVersion string) error {
    // Deploy canary version
    if err := cd.CanaryService.Deploy(newVersion); err != nil {
        return err
    }
    
    // Health check
    if err := cd.CanaryService.HealthCheck(); err != nil {
        return err
    }
    
    // Gradually increase traffic
    for i := 10; i <= 100; i += 10 {
        if err := cd.setTrafficSplit(i); err != nil {
            return err
        }
        
        // Monitor metrics
        if err := cd.monitorCanary(); err != nil {
            // Rollback if metrics are poor
            return cd.rollback()
        }
        
        time.Sleep(5 * time.Minute)
    }
    
    // Promote canary to stable
    return cd.promoteCanary()
}
```

## Observability Patterns

### Distributed Tracing
```go
// OpenTelemetry Integration
type TracingService struct {
    tracer trace.Tracer
    propagator propagation.TextMapPropagator
}

func (ts *TracingService) StartSpan(ctx context.Context, name string) (context.Context, trace.Span) {
    return ts.tracer.Start(ctx, name)
}

func (ts *TracingService) InjectTraceContext(ctx context.Context, headers http.Header) {
    ts.propagator.Inject(ctx, propagation.HeaderCarrier(headers))
}

func (ts *TracingService) ExtractTraceContext(ctx context.Context, headers http.Header) context.Context {
    return ts.propagator.Extract(ctx, propagation.HeaderCarrier(headers))
}

// Service Mesh Tracing
type ServiceMeshTracing struct {
    jaegerEndpoint string
    serviceName    string
    version        string
}

func (smt *ServiceMeshTracing) ConfigureTracing() error {
    // Configure Jaeger exporter
    exp, err := jaeger.New(jaeger.WithCollectorEndpoint(jaeger.WithEndpoint(smt.jaegerEndpoint)))
    if err != nil {
        return err
    }
    
    // Create trace provider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exp),
        trace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String(smt.serviceName),
            semconv.ServiceVersionKey.String(smt.version),
        )),
    )
    
    // Set global tracer provider
    otel.SetTracerProvider(tp)
    
    return nil
}
```

### Metrics Collection
```go
// Prometheus Metrics
type PrometheusMetrics struct {
    requestDuration *prometheus.HistogramVec
    requestCount    *prometheus.CounterVec
    errorCount      *prometheus.CounterVec
    activeConnections prometheus.Gauge
}

func NewPrometheusMetrics() *PrometheusMetrics {
    return &PrometheusMetrics{
        requestDuration: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "http_request_duration_seconds",
                Help: "HTTP request duration in seconds",
            },
            []string{"method", "endpoint", "status"},
        ),
        requestCount: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "http_requests_total",
                Help: "Total number of HTTP requests",
            },
            []string{"method", "endpoint", "status"},
        ),
        errorCount: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "http_errors_total",
                Help: "Total number of HTTP errors",
            },
            []string{"method", "endpoint", "error_type"},
        ),
        activeConnections: prometheus.NewGauge(
            prometheus.GaugeOpts{
                Name: "active_connections",
                Help: "Current number of active connections",
            },
        ),
    }
}

// Custom Business Metrics
type BusinessMetrics struct {
    userRegistrations *prometheus.CounterVec
    orderValue        *prometheus.HistogramVec
    paymentSuccess    *prometheus.CounterVec
    cartAbandonment   *prometheus.CounterVec
}

func NewBusinessMetrics() *BusinessMetrics {
    return &BusinessMetrics{
        userRegistrations: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "user_registrations_total",
                Help: "Total number of user registrations",
            },
            []string{"source", "country"},
        ),
        orderValue: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "order_value_dollars",
                Help: "Order value in dollars",
                Buckets: []float64{10, 50, 100, 500, 1000, 5000},
            },
            []string{"currency", "payment_method"},
        ),
        paymentSuccess: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "payment_success_total",
                Help: "Total number of successful payments",
            },
            []string{"payment_method", "currency"},
        ),
        cartAbandonment: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "cart_abandonment_total",
                Help: "Total number of cart abandonments",
            },
            []string{"step", "reason"},
        ),
    }
}
```

## Security Patterns

### API Gateway Security
```go
// API Gateway with Security
type APIGateway struct {
    routes        map[string]*Route
    authService   *AuthService
    rateLimiter   *RateLimiter
    corsHandler   *CORSHandler
    securityHeaders *SecurityHeaders
}

type Route struct {
    Path        string
    Method      string
    Handler     http.HandlerFunc
    Middleware  []Middleware
    AuthRequired bool
    RateLimit   int
}

type Middleware func(http.Handler) http.Handler

// Authentication Middleware
func (ag *APIGateway) AuthMiddleware() Middleware {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            token := r.Header.Get("Authorization")
            if token == "" {
                http.Error(w, "Unauthorized", http.StatusUnauthorized)
                return
            }
            
            user, err := ag.authService.ValidateToken(token)
            if err != nil {
                http.Error(w, "Unauthorized", http.StatusUnauthorized)
                return
            }
            
            // Add user to context
            ctx := context.WithValue(r.Context(), "user", user)
            next.ServeHTTP(w, r.WithContext(ctx))
        })
    }
}

// Rate Limiting Middleware
func (ag *APIGateway) RateLimitMiddleware() Middleware {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            clientIP := getClientIP(r)
            
            if !ag.rateLimiter.Allow(clientIP) {
                http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
                return
            }
            
            next.ServeHTTP(w, r)
        })
    }
}

// CORS Middleware
func (ag *APIGateway) CORSMiddleware() Middleware {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            w.Header().Set("Access-Control-Allow-Origin", "*")
            w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
            
            if r.Method == "OPTIONS" {
                w.WriteHeader(http.StatusOK)
                return
            }
            
            next.ServeHTTP(w, r)
        })
    }
}
```

### Service-to-Service Security
```go
// mTLS Configuration
type mTLSConfig struct {
    CertFile    string
    KeyFile     string
    CAFile      string
    ServerName  string
}

func (mtls *mTLSConfig) CreateTLSConfig() (*tls.Config, error) {
    cert, err := tls.LoadX509KeyPair(mtls.CertFile, mtls.KeyFile)
    if err != nil {
        return nil, err
    }
    
    caCert, err := ioutil.ReadFile(mtls.CAFile)
    if err != nil {
        return nil, err
    }
    
    caCertPool := x509.NewCertPool()
    caCertPool.AppendCertsFromPEM(caCert)
    
    return &tls.Config{
        Certificates: []tls.Certificate{cert},
        RootCAs:      caCertPool,
        ServerName:   mtls.ServerName,
    }, nil
}

// JWT Token Validation
type JWTValidator struct {
    publicKey *rsa.PublicKey
    issuer    string
    audience  string
}

func (jv *JWTValidator) ValidateToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return jv.publicKey, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(*Claims); ok && token.Valid {
        if claims.Issuer != jv.issuer {
            return nil, fmt.Errorf("invalid issuer")
        }
        if claims.Audience != jv.audience {
            return nil, fmt.Errorf("invalid audience")
        }
        return claims, nil
    }
    
    return nil, fmt.Errorf("invalid token")
}

type Claims struct {
    UserID    string   `json:"user_id"`
    Email     string   `json:"email"`
    Roles     []string `json:"roles"`
    Issuer    string   `json:"iss"`
    Audience  string   `json:"aud"`
    ExpiresAt int64    `json:"exp"`
    IssuedAt  int64    `json:"iat"`
    jwt.StandardClaims
}
```

## Golang Implementation

### Service Template
```go
// Microservice Template
type Microservice struct {
    name        string
    version     string
    port        int
    dependencies []string
    health      HealthStatus
    metrics     *PrometheusMetrics
    tracing     *TracingService
    logger      *Logger
    server      *http.Server
    mutex       sync.RWMutex
}

func NewMicroservice(name, version string, port int) *Microservice {
    return &Microservice{
        name:        name,
        version:     version,
        port:        port,
        dependencies: []string{},
        health:      HealthStatusStarting,
        metrics:     NewPrometheusMetrics(),
        tracing:     NewTracingService(),
        logger:      NewLogger(name),
    }
}

func (ms *Microservice) Start() error {
    // Register metrics
    prometheus.MustRegister(ms.metrics.requestDuration)
    prometheus.MustRegister(ms.metrics.requestCount)
    prometheus.MustRegister(ms.metrics.errorCount)
    prometheus.MustRegister(ms.metrics.activeConnections)
    
    // Setup routes
    mux := http.NewServeMux()
    mux.HandleFunc("/health", ms.healthHandler)
    mux.HandleFunc("/metrics", promhttp.Handler().ServeHTTP)
    mux.HandleFunc("/api/", ms.apiHandler)
    
    // Setup middleware
    handler := ms.setupMiddleware(mux)
    
    // Start server
    ms.server = &http.Server{
        Addr:    fmt.Sprintf(":%d", ms.port),
        Handler: handler,
    }
    
    ms.health = HealthStatusHealthy
    ms.logger.Info("Service started", "port", ms.port)
    
    return ms.server.ListenAndServe()
}

func (ms *Microservice) Stop() error {
    ms.health = HealthStatusStopping
    ms.logger.Info("Service stopping")
    
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    return ms.server.Shutdown(ctx)
}

func (ms *Microservice) setupMiddleware(handler http.Handler) http.Handler {
    // Logging middleware
    handler = ms.loggingMiddleware(handler)
    
    // Metrics middleware
    handler = ms.metricsMiddleware(handler)
    
    // Tracing middleware
    handler = ms.tracingMiddleware(handler)
    
    // CORS middleware
    handler = ms.corsMiddleware(handler)
    
    return handler
}
```

## Best Practices

### 1. Service Design
- **Single Responsibility**: Each service should have one business capability
- **Loose Coupling**: Services should be independent and communicate through well-defined interfaces
- **High Cohesion**: Related functionality should be grouped together
- **Stateless**: Services should not maintain state between requests

### 2. Communication
- **Synchronous**: Use for request-response patterns
- **Asynchronous**: Use for event-driven patterns
- **Circuit Breaker**: Implement fault tolerance
- **Retry Logic**: Handle transient failures
- **Timeout**: Prevent hanging requests

### 3. Data Management
- **Database per Service**: Each service owns its data
- **Event Sourcing**: Use for audit trails and replay
- **CQRS**: Separate read and write models
- **Saga Pattern**: Handle distributed transactions

### 4. Deployment
- **Containerization**: Use Docker for consistency
- **Orchestration**: Use Kubernetes for management
- **Blue-Green**: Zero-downtime deployments
- **Canary**: Gradual rollouts

### 5. Observability
- **Logging**: Structured logging with correlation IDs
- **Metrics**: Business and technical metrics
- **Tracing**: Distributed tracing across services
- **Alerting**: Proactive monitoring and alerting

## Common Pitfalls

### 1. Over-Engineering
- **Problem**: Building complex systems when simple solutions suffice
- **Solution**: Start simple and add complexity only when needed

### 2. Tight Coupling
- **Problem**: Services are too dependent on each other
- **Solution**: Use well-defined interfaces and event-driven communication

### 3. Data Consistency
- **Problem**: Maintaining consistency across services
- **Solution**: Use eventual consistency and compensate for inconsistencies

### 4. Performance Issues
- **Problem**: Poor performance due to network calls and data duplication
- **Solution**: Implement caching and optimize data access patterns

### 5. Monitoring Gaps
- **Problem**: Lack of visibility into system behavior
- **Solution**: Implement comprehensive logging, metrics, and tracing

## Conclusion

Microservices architecture provides a powerful approach to building scalable, maintainable systems. Key success factors:

1. **Proper Service Decomposition**: Break down monoliths into cohesive services
2. **Effective Communication**: Use appropriate patterns for different scenarios
3. **Data Management**: Handle distributed data challenges
4. **Deployment Strategy**: Implement reliable deployment patterns
5. **Observability**: Monitor and understand system behavior
6. **Security**: Implement comprehensive security measures

By following these patterns and best practices, you can build robust, scalable microservices that meet the demands of modern applications.
