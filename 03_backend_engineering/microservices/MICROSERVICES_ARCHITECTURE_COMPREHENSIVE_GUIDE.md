# 🏗️ Microservices Architecture Comprehensive Guide

> **Complete guide to microservices design, implementation, and best practices for scalable systems**

## 📚 Table of Contents

1. [Microservices Fundamentals](#-microservices-fundamentals)
2. [Service Design Patterns](#-service-design-patterns)
3. [Communication Patterns](#-communication-patterns)
4. [Data Management](#-data-management)
5. [Service Discovery](#-service-discovery)
6. [API Gateway](#-api-gateway)
7. [Event-Driven Architecture](#-event-driven-architecture)
8. [Testing Strategies](#-testing-strategies)
9. [Deployment & DevOps](#-deployment--devops)
10. [Monitoring & Observability](#-monitoring--observability)

---

## 🎯 Microservices Fundamentals

### What are Microservices?

```
┌─────────────────────────────────────────────────────────────┐
│                    Monolithic vs Microservices              │
├─────────────────────────────────────────────────────────────┤
│  Monolithic Application          Microservices Architecture │
│  ┌─────────────────────────┐    ┌─────┐ ┌─────┐ ┌─────┐    │
│  │                         │    │User │ │Order│ │Pay  │    │
│  │    Single Application   │    │Svc  │ │Svc  │ │Svc  │    │
│  │                         │    └─────┘ └─────┘ └─────┘    │
│  │  - All functionality    │    ┌─────┐ ┌─────┐ ┌─────┐    │
│  │  - Single database      │    │Notif│ │Auth │ │Anal │    │
│  │  - Single deployment    │    │Svc  │ │Svc  │ │Svc  │    │
│  │  - Single technology    │    └─────┘ └─────┘ └─────┘    │
│  └─────────────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

### Microservices Characteristics

1. **Single Responsibility**: Each service has one business capability
2. **Decentralized**: Services are independently deployable
3. **Technology Diversity**: Each service can use different technologies
4. **Fault Isolation**: Failure in one service doesn't affect others
5. **Scalability**: Services can be scaled independently

---

## 🎨 Service Design Patterns

### 1. Domain-Driven Design (DDD)

```go
// User Service - Domain Model
package user

import (
    "context"
    "errors"
    "time"
)

// Domain Entities
type User struct {
    ID        string    `json:"id"`
    Email     string    `json:"email"`
    Name      string    `json:"name"`
    Status    Status    `json:"status"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

type Status string

const (
    StatusActive   Status = "active"
    StatusInactive Status = "inactive"
    StatusPending  Status = "pending"
)

// Value Objects
type Email struct {
    value string
}

func NewEmail(email string) (*Email, error) {
    if !isValidEmail(email) {
        return nil, errors.New("invalid email format")
    }
    return &Email{value: email}, nil
}

func (e *Email) String() string {
    return e.value
}

// Domain Services
type UserService struct {
    repo UserRepository
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{repo: repo}
}

func (us *UserService) CreateUser(ctx context.Context, req CreateUserRequest) (*User, error) {
    // Business logic validation
    if req.Name == "" {
        return nil, errors.New("name is required")
    }
    
    email, err := NewEmail(req.Email)
    if err != nil {
        return nil, err
    }
    
    // Create user entity
    user := &User{
        ID:        generateID(),
        Email:     email.String(),
        Name:      req.Name,
        Status:    StatusPending,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    // Save to repository
    if err := us.repo.Save(ctx, user); err != nil {
        return nil, err
    }
    
    return user, nil
}

func (us *UserService) ActivateUser(ctx context.Context, userID string) error {
    user, err := us.repo.FindByID(ctx, userID)
    if err != nil {
        return err
    }
    
    if user.Status != StatusPending {
        return errors.New("user is not in pending status")
    }
    
    user.Status = StatusActive
    user.UpdatedAt = time.Now()
    
    return us.repo.Save(ctx, user)
}

// Repository Interface
type UserRepository interface {
    Save(ctx context.Context, user *User) error
    FindByID(ctx context.Context, id string) (*User, error)
    FindByEmail(ctx context.Context, email string) (*User, error)
    Delete(ctx context.Context, id string) error
}

type CreateUserRequest struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}

func generateID() string {
    // Implementation for generating unique ID
    return "user_" + time.Now().Format("20060102150405")
}

func isValidEmail(email string) bool {
    // Simple email validation
    return len(email) > 0 && contains(email, "@")
}

func contains(s, substr string) bool {
    return len(s) >= len(substr) && s[:len(substr)] == substr
}
```

### 2. API-First Design

```yaml
# OpenAPI Specification for User Service
openapi: 3.0.0
info:
  title: User Service API
  version: 1.0.0
  description: User management microservice

servers:
  - url: https://api.example.com/users
    description: Production server
  - url: https://staging-api.example.com/users
    description: Staging server

paths:
  /users:
    get:
      summary: List users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 10
      responses:
        '200':
          description: List of users
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
    
    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          description: Bad request
        '409':
          description: User already exists

  /users/{id}:
    get:
      summary: Get user by ID
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: User details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found
    
    put:
      summary: Update user
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateUserRequest'
      responses:
        '200':
          description: User updated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found
    
    delete:
      summary: Delete user
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '204':
          description: User deleted
        '404':
          description: User not found

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
        email:
          type: string
          format: email
        name:
          type: string
        status:
          type: string
          enum: [active, inactive, pending]
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time
    
    CreateUserRequest:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
        email:
          type: string
          format: email
    
    UpdateUserRequest:
      type: object
      properties:
        name:
          type: string
        status:
          type: string
          enum: [active, inactive, pending]
    
    Pagination:
      type: object
      properties:
        page:
          type: integer
        limit:
          type: integer
        total:
          type: integer
        pages:
          type: integer
```

---

## 📡 Communication Patterns

### 1. Synchronous Communication (HTTP/gRPC)

```go
// HTTP Client for inter-service communication
package client

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type HTTPClient struct {
    baseURL    string
    httpClient *http.Client
}

func NewHTTPClient(baseURL string) *HTTPClient {
    return &HTTPClient{
        baseURL: baseURL,
        httpClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

func (c *HTTPClient) GetUser(ctx context.Context, userID string) (*User, error) {
    url := fmt.Sprintf("%s/users/%s", c.baseURL, userID)
    
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")
    
    resp, err := c.httpClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
    }
    
    var user User
    if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
        return nil, err
    }
    
    return &user, nil
}

func (c *HTTPClient) CreateUser(ctx context.Context, req CreateUserRequest) (*User, error) {
    url := fmt.Sprintf("%s/users", c.baseURL)
    
    body, err := json.Marshal(req)
    if err != nil {
        return nil, err
    }
    
    httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(body))
    if err != nil {
        return nil, err
    }
    
    httpReq.Header.Set("Content-Type", "application/json")
    httpReq.Header.Set("Accept", "application/json")
    
    resp, err := c.httpClient.Do(httpReq)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusCreated {
        return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
    }
    
    var user User
    if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
        return nil, err
    }
    
    return &user, nil
}

// gRPC Client
package grpc

import (
    "context"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
)

type UserServiceClient struct {
    conn   *grpc.ClientConn
    client UserServiceClient
}

func NewUserServiceClient(address string) (*UserServiceClient, error) {
    conn, err := grpc.Dial(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        return nil, err
    }
    
    client := NewUserServiceClient(conn)
    
    return &UserServiceClient{
        conn:   conn,
        client: client,
    }, nil
}

func (c *UserServiceClient) GetUser(ctx context.Context, req *GetUserRequest) (*User, error) {
    return c.client.GetUser(ctx, req)
}

func (c *UserServiceClient) CreateUser(ctx context.Context, req *CreateUserRequest) (*User, error) {
    return c.client.CreateUser(ctx, req)
}

func (c *UserServiceClient) Close() error {
    return c.conn.Close()
}
```

### 2. Asynchronous Communication (Message Queues)

```go
// Event-driven communication with Kafka
package events

import (
    "context"
    "encoding/json"
    "github.com/Shopify/sarama"
)

type EventPublisher struct {
    producer sarama.SyncProducer
}

func NewEventPublisher(brokers []string) (*EventPublisher, error) {
    config := sarama.NewConfig()
    config.Producer.Return.Successes = true
    config.Producer.RequiredAcks = sarama.WaitForAll
    config.Producer.Retry.Max = 3
    
    producer, err := sarama.NewSyncProducer(brokers, config)
    if err != nil {
        return nil, err
    }
    
    return &EventPublisher{producer: producer}, nil
}

func (ep *EventPublisher) PublishUserCreated(ctx context.Context, user *User) error {
    event := UserCreatedEvent{
        UserID:    user.ID,
        Email:     user.Email,
        Name:      user.Name,
        CreatedAt: user.CreatedAt,
    }
    
    data, err := json.Marshal(event)
    if err != nil {
        return err
    }
    
    message := &sarama.ProducerMessage{
        Topic: "user.created",
        Key:   sarama.StringEncoder(user.ID),
        Value: sarama.ByteEncoder(data),
        Headers: []sarama.RecordHeader{
            {
                Key:   []byte("event_type"),
                Value: []byte("user.created"),
            },
            {
                Key:   []byte("version"),
                Value: []byte("1.0"),
            },
        },
    }
    
    _, _, err = ep.producer.SendMessage(message)
    return err
}

func (ep *EventPublisher) PublishUserUpdated(ctx context.Context, user *User) error {
    event := UserUpdatedEvent{
        UserID:    user.ID,
        Email:     user.Email,
        Name:      user.Name,
        Status:    string(user.Status),
        UpdatedAt: user.UpdatedAt,
    }
    
    data, err := json.Marshal(event)
    if err != nil {
        return err
    }
    
    message := &sarama.ProducerMessage{
        Topic: "user.updated",
        Key:   sarama.StringEncoder(user.ID),
        Value: sarama.ByteEncoder(data),
        Headers: []sarama.RecordHeader{
            {
                Key:   []byte("event_type"),
                Value: []byte("user.updated"),
            },
            {
                Key:   []byte("version"),
                Value: []byte("1.0"),
            },
        },
    }
    
    _, _, err = ep.producer.SendMessage(message)
    return err
}

func (ep *EventPublisher) Close() error {
    return ep.producer.Close()
}

// Event Consumer
type EventConsumer struct {
    consumer sarama.ConsumerGroup
    handler  EventHandler
}

func NewEventConsumer(brokers []string, groupID string, handler EventHandler) (*EventConsumer, error) {
    config := sarama.NewConfig()
    config.Consumer.Group.Rebalance.Strategy = sarama.BalanceStrategyRoundRobin
    config.Consumer.Offsets.Initial = sarama.OffsetNewest
    
    consumer, err := sarama.NewConsumerGroup(brokers, groupID, config)
    if err != nil {
        return nil, err
    }
    
    return &EventConsumer{
        consumer: consumer,
        handler:  handler,
    }, nil
}

func (ec *EventConsumer) Start(ctx context.Context, topics []string) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            if err := ec.consumer.Consume(ctx, topics, ec); err != nil {
                return err
            }
        }
    }
}

func (ec *EventConsumer) Setup(sarama.ConsumerGroupSession) error   { return nil }
func (ec *EventConsumer) Cleanup(sarama.ConsumerGroupSession) error { return nil }

func (ec *EventConsumer) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
    for {
        select {
        case message := <-claim.Messages():
            if message == nil {
                return nil
            }
            
            if err := ec.handleMessage(session.Context(), message); err != nil {
                return err
            }
            
            session.MarkMessage(message, "")
            
        case <-session.Context().Done():
            return nil
        }
    }
}

func (ec *EventConsumer) handleMessage(ctx context.Context, message *sarama.ConsumerMessage) error {
    eventType := getHeaderValue(message.Headers, "event_type")
    
    switch eventType {
    case "user.created":
        var event UserCreatedEvent
        if err := json.Unmarshal(message.Value, &event); err != nil {
            return err
        }
        return ec.handler.HandleUserCreated(ctx, &event)
        
    case "user.updated":
        var event UserUpdatedEvent
        if err := json.Unmarshal(message.Value, &event); err != nil {
            return err
        }
        return ec.handler.HandleUserUpdated(ctx, &event)
        
    default:
        return fmt.Errorf("unknown event type: %s", eventType)
    }
}

func getHeaderValue(headers []sarama.RecordHeader, key string) string {
    for _, header := range headers {
        if string(header.Key) == key {
            return string(header.Value)
        }
    }
    return ""
}

// Event Handler Interface
type EventHandler interface {
    HandleUserCreated(ctx context.Context, event *UserCreatedEvent) error
    HandleUserUpdated(ctx context.Context, event *UserUpdatedEvent) error
}

// Event Types
type UserCreatedEvent struct {
    UserID    string    `json:"user_id"`
    Email     string    `json:"email"`
    Name      string    `json:"name"`
    CreatedAt time.Time `json:"created_at"`
}

type UserUpdatedEvent struct {
    UserID    string    `json:"user_id"`
    Email     string    `json:"email"`
    Name      string    `json:"name"`
    Status    string    `json:"status"`
    UpdatedAt time.Time `json:"updated_at"`
}
```

---

## 🗄️ Data Management

### 1. Database per Service

```go
// User Service Database
package repository

import (
    "context"
    "database/sql"
    "fmt"
    _ "github.com/lib/pq"
)

type UserRepositoryImpl struct {
    db *sql.DB
}

func NewUserRepositoryImpl(dsn string) (*UserRepositoryImpl, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }
    
    if err := db.Ping(); err != nil {
        return nil, err
    }
    
    return &UserRepositoryImpl{db: db}, nil
}

func (r *UserRepositoryImpl) Save(ctx context.Context, user *User) error {
    query := `
        INSERT INTO users (id, email, name, status, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (id) DO UPDATE SET
            email = EXCLUDED.email,
            name = EXCLUDED.name,
            status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at
    `
    
    _, err := r.db.ExecContext(ctx, query,
        user.ID, user.Email, user.Name, user.Status,
        user.CreatedAt, user.UpdatedAt)
    
    return err
}

func (r *UserRepositoryImpl) FindByID(ctx context.Context, id string) (*User, error) {
    query := `
        SELECT id, email, name, status, created_at, updated_at
        FROM users
        WHERE id = $1
    `
    
    row := r.db.QueryRowContext(ctx, query, id)
    
    var user User
    err := row.Scan(
        &user.ID, &user.Email, &user.Name, &user.Status,
        &user.CreatedAt, &user.UpdatedAt,
    )
    
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("user not found: %s", id)
        }
        return nil, err
    }
    
    return &user, nil
}

func (r *UserRepositoryImpl) FindByEmail(ctx context.Context, email string) (*User, error) {
    query := `
        SELECT id, email, name, status, created_at, updated_at
        FROM users
        WHERE email = $1
    `
    
    row := r.db.QueryRowContext(ctx, query, email)
    
    var user User
    err := row.Scan(
        &user.ID, &user.Email, &user.Name, &user.Status,
        &user.CreatedAt, &user.UpdatedAt,
    )
    
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, fmt.Errorf("user not found: %s", email)
        }
        return nil, err
    }
    
    return &user, nil
}

func (r *UserRepositoryImpl) Delete(ctx context.Context, id string) error {
    query := `DELETE FROM users WHERE id = $1`
    
    result, err := r.db.ExecContext(ctx, query, id)
    if err != nil {
        return err
    }
    
    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return err
    }
    
    if rowsAffected == 0 {
        return fmt.Errorf("user not found: %s", id)
    }
    
    return nil
}

func (r *UserRepositoryImpl) Close() error {
    return r.db.Close()
}
```

### 2. Saga Pattern for Distributed Transactions

```go
// Saga Pattern Implementation
package saga

import (
    "context"
    "fmt"
    "time"
)

type Saga struct {
    steps []SagaStep
}

type SagaStep struct {
    Name           string
    Execute        func(ctx context.Context) error
    Compensate     func(ctx context.Context) error
    RetryPolicy    RetryPolicy
    Timeout        time.Duration
}

type RetryPolicy struct {
    MaxRetries int
    Backoff    time.Duration
}

type SagaResult struct {
    Success bool
    Error   error
    Steps   []StepResult
}

type StepResult struct {
    StepName string
    Success  bool
    Error    error
    Duration time.Duration
}

func NewSaga() *Saga {
    return &Saga{
        steps: make([]SagaStep, 0),
    }
}

func (s *Saga) AddStep(step SagaStep) *Saga {
    s.steps = append(s.steps, step)
    return s
}

func (s *Saga) Execute(ctx context.Context) *SagaResult {
    result := &SagaResult{
        Success: true,
        Steps:   make([]StepResult, 0, len(s.steps)),
    }
    
    executedSteps := make([]SagaStep, 0)
    
    for i, step := range s.steps {
        stepResult := s.executeStep(ctx, step)
        result.Steps = append(result.Steps, stepResult)
        
        if !stepResult.Success {
            result.Success = false
            result.Error = stepResult.Error
            
            // Compensate for executed steps
            s.compensateSteps(ctx, executedSteps)
            break
        }
        
        executedSteps = append(executedSteps, step)
    }
    
    return result
}

func (s *Saga) executeStep(ctx context.Context, step SagaStep) StepResult {
    start := time.Now()
    
    for attempt := 0; attempt <= step.RetryPolicy.MaxRetries; attempt++ {
        if attempt > 0 {
            time.Sleep(step.RetryPolicy.Backoff * time.Duration(attempt))
        }
        
        // Create timeout context
        stepCtx := ctx
        if step.Timeout > 0 {
            var cancel context.CancelFunc
            stepCtx, cancel = context.WithTimeout(ctx, step.Timeout)
            defer cancel()
        }
        
        err := step.Execute(stepCtx)
        if err == nil {
            return StepResult{
                StepName: step.Name,
                Success:  true,
                Duration: time.Since(start),
            }
        }
        
        if attempt == step.RetryPolicy.MaxRetries {
            return StepResult{
                StepName: step.Name,
                Success:  false,
                Error:    err,
                Duration: time.Since(start),
            }
        }
    }
    
    return StepResult{
        StepName: step.Name,
        Success:  false,
        Error:    fmt.Errorf("max retries exceeded"),
        Duration: time.Since(start),
    }
}

func (s *Saga) compensateSteps(ctx context.Context, steps []SagaStep) {
    // Compensate in reverse order
    for i := len(steps) - 1; i >= 0; i-- {
        step := steps[i]
        if step.Compensate != nil {
            if err := step.Compensate(ctx); err != nil {
                // Log compensation error but continue
                fmt.Printf("Compensation failed for step %s: %v\n", step.Name, err)
            }
        }
    }
}

// Example: Order Processing Saga
func CreateOrderSaga(userService UserService, orderService OrderService, paymentService PaymentService) *Saga {
    return NewSaga().
        AddStep(SagaStep{
            Name: "ReserveInventory",
            Execute: func(ctx context.Context) error {
                return orderService.ReserveInventory(ctx, "order123")
            },
            Compensate: func(ctx context.Context) error {
                return orderService.ReleaseInventory(ctx, "order123")
            },
            RetryPolicy: RetryPolicy{MaxRetries: 3, Backoff: time.Second},
            Timeout:     30 * time.Second,
        }).
        AddStep(SagaStep{
            Name: "ProcessPayment",
            Execute: func(ctx context.Context) error {
                return paymentService.ProcessPayment(ctx, "order123", 100.00)
            },
            Compensate: func(ctx context.Context) error {
                return paymentService.RefundPayment(ctx, "order123")
            },
            RetryPolicy: RetryPolicy{MaxRetries: 3, Backoff: time.Second},
            Timeout:     30 * time.Second,
        }).
        AddStep(SagaStep{
            Name: "CreateOrder",
            Execute: func(ctx context.Context) error {
                return orderService.CreateOrder(ctx, "order123")
            },
            Compensate: func(ctx context.Context) error {
                return orderService.CancelOrder(ctx, "order123")
            },
            RetryPolicy: RetryPolicy{MaxRetries: 3, Backoff: time.Second},
            Timeout:     30 * time.Second,
        })
}

// Service Interfaces
type UserService interface {
    GetUser(ctx context.Context, userID string) (*User, error)
}

type OrderService interface {
    ReserveInventory(ctx context.Context, orderID string) error
    ReleaseInventory(ctx context.Context, orderID string) error
    CreateOrder(ctx context.Context, orderID string) error
    CancelOrder(ctx context.Context, orderID string) error
}

type PaymentService interface {
    ProcessPayment(ctx context.Context, orderID string, amount float64) error
    RefundPayment(ctx context.Context, orderID string) error
}
```

---

## 🔍 Service Discovery

### 1. Service Registry

```go
// Service Registry Implementation
package registry

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type ServiceInstance struct {
    ID       string            `json:"id"`
    Name     string            `json:"name"`
    Address  string            `json:"address"`
    Port     int               `json:"port"`
    Metadata map[string]string `json:"metadata"`
    Health   HealthStatus      `json:"health"`
    LastSeen time.Time         `json:"last_seen"`
}

type HealthStatus string

const (
    HealthUp   HealthStatus = "UP"
    HealthDown HealthStatus = "DOWN"
)

type ServiceRegistry struct {
    instances map[string][]ServiceInstance
    mutex     sync.RWMutex
    ttl       time.Duration
}

func NewServiceRegistry(ttl time.Duration) *ServiceRegistry {
    registry := &ServiceRegistry{
        instances: make(map[string][]ServiceInstance),
        ttl:       ttl,
    }
    
    // Start cleanup goroutine
    go registry.cleanup()
    
    return registry
}

func (sr *ServiceRegistry) Register(instance ServiceInstance) error {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    instance.LastSeen = time.Now()
    instance.Health = HealthUp
    
    serviceName := instance.Name
    instances := sr.instances[serviceName]
    
    // Check if instance already exists
    for i, existing := range instances {
        if existing.ID == instance.ID {
            instances[i] = instance
            sr.instances[serviceName] = instances
            return nil
        }
    }
    
    // Add new instance
    instances = append(instances, instance)
    sr.instances[serviceName] = instances
    
    return nil
}

func (sr *ServiceRegistry) Deregister(serviceName, instanceID string) error {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    instances := sr.instances[serviceName]
    for i, instance := range instances {
        if instance.ID == instanceID {
            instances = append(instances[:i], instances[i+1:]...)
            sr.instances[serviceName] = instances
            return nil
        }
    }
    
    return fmt.Errorf("instance not found: %s", instanceID)
}

func (sr *ServiceRegistry) GetInstances(serviceName string) ([]ServiceInstance, error) {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    instances, exists := sr.instances[serviceName]
    if !exists {
        return nil, fmt.Errorf("service not found: %s", serviceName)
    }
    
    // Filter healthy instances
    healthyInstances := make([]ServiceInstance, 0)
    for _, instance := range instances {
        if instance.Health == HealthUp && time.Since(instance.LastSeen) < sr.ttl {
            healthyInstances = append(healthyInstances, instance)
        }
    }
    
    if len(healthyInstances) == 0 {
        return nil, fmt.Errorf("no healthy instances found for service: %s", serviceName)
    }
    
    return healthyInstances, nil
}

func (sr *ServiceRegistry) cleanup() {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()
    
    for range ticker.C {
        sr.mutex.Lock()
        for serviceName, instances := range sr.instances {
            healthyInstances := make([]ServiceInstance, 0)
            for _, instance := range instances {
                if time.Since(instance.LastSeen) < sr.ttl {
                    healthyInstances = append(healthyInstances, instance)
                }
            }
            sr.instances[serviceName] = healthyInstances
        }
        sr.mutex.Unlock()
    }
}
```

### 2. Load Balancer

```go
// Load Balancer Implementation
package loadbalancer

import (
    "context"
    "fmt"
    "math/rand"
    "time"
)

type LoadBalancer interface {
    SelectInstance(ctx context.Context, instances []ServiceInstance) (*ServiceInstance, error)
}

type RoundRobinLoadBalancer struct {
    current int
}

func NewRoundRobinLoadBalancer() *RoundRobinLoadBalancer {
    return &RoundRobinLoadBalancer{current: 0}
}

func (rr *RoundRobinLoadBalancer) SelectInstance(ctx context.Context, instances []ServiceInstance) (*ServiceInstance, error) {
    if len(instances) == 0 {
        return nil, fmt.Errorf("no instances available")
    }
    
    instance := instances[rr.current%len(instances)]
    rr.current++
    
    return &instance, nil
}

type RandomLoadBalancer struct {
    rand *rand.Rand
}

func NewRandomLoadBalancer() *RandomLoadBalancer {
    return &RandomLoadBalancer{
        rand: rand.New(rand.NewSource(time.Now().UnixNano())),
    }
}

func (r *RandomLoadBalancer) SelectInstance(ctx context.Context, instances []ServiceInstance) (*ServiceInstance, error) {
    if len(instances) == 0 {
        return nil, fmt.Errorf("no instances available")
    }
    
    index := r.rand.Intn(len(instances))
    return &instances[index], nil
}

type WeightedRoundRobinLoadBalancer struct {
    current int
    weights []int
}

func NewWeightedRoundRobinLoadBalancer(weights []int) *WeightedRoundRobinLoadBalancer {
    return &WeightedRoundRobinLoadBalancer{
        current: 0,
        weights: weights,
    }
}

func (wrr *WeightedRoundRobinLoadBalancer) SelectInstance(ctx context.Context, instances []ServiceInstance) (*ServiceInstance, error) {
    if len(instances) == 0 {
        return nil, fmt.Errorf("no instances available")
    }
    
    if len(wrr.weights) != len(instances) {
        // Fallback to round robin if weights don't match
        instance := instances[wrr.current%len(instances)]
        wrr.current++
        return &instance, nil
    }
    
    // Find instance with highest weight
    maxWeight := 0
    selectedIndex := 0
    
    for i, weight := range wrr.weights {
        if weight > maxWeight {
            maxWeight = weight
            selectedIndex = i
        }
    }
    
    // Decrease weight of selected instance
    wrr.weights[selectedIndex]--
    
    return &instances[selectedIndex], nil
}
```

---

## 🌐 API Gateway

### 1. API Gateway Implementation

```go
// API Gateway with routing and middleware
package gateway

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "strings"
    "time"
)

type APIGateway struct {
    routes      []Route
    middleware  []Middleware
    registry    ServiceRegistry
    loadBalancer LoadBalancer
}

type Route struct {
    Method      string
    Path        string
    ServiceName string
    PathPrefix  string
}

type Middleware interface {
    Process(ctx context.Context, req *http.Request, next http.Handler) http.ResponseWriter
}

type ServiceRegistry interface {
    GetInstances(serviceName string) ([]ServiceInstance, error)
}

type LoadBalancer interface {
    SelectInstance(ctx context.Context, instances []ServiceInstance) (*ServiceInstance, error)
}

func NewAPIGateway(registry ServiceRegistry, loadBalancer LoadBalancer) *APIGateway {
    return &APIGateway{
        routes:       make([]Route, 0),
        middleware:   make([]Middleware, 0),
        registry:     registry,
        loadBalancer: loadBalancer,
    }
}

func (gw *APIGateway) AddRoute(route Route) {
    gw.routes = append(gw.routes, route)
}

func (gw *APIGateway) AddMiddleware(middleware Middleware) {
    gw.middleware = append(gw.middleware, middleware)
}

func (gw *APIGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    
    // Find matching route
    route := gw.findRoute(r.Method, r.URL.Path)
    if route == nil {
        http.NotFound(w, r)
        return
    }
    
    // Get service instances
    instances, err := gw.registry.GetInstances(route.ServiceName)
    if err != nil {
        http.Error(w, "Service unavailable", http.StatusServiceUnavailable)
        return
    }
    
    // Select instance using load balancer
    instance, err := gw.loadBalancer.SelectInstance(ctx, instances)
    if err != nil {
        http.Error(w, "No healthy instances", http.StatusServiceUnavailable)
        return
    }
    
    // Build target URL
    targetURL := fmt.Sprintf("http://%s:%d%s", instance.Address, instance.Port, r.URL.Path)
    
    // Create proxy request
    proxyReq, err := http.NewRequestWithContext(ctx, r.Method, targetURL, r.Body)
    if err != nil {
        http.Error(w, "Internal server error", http.StatusInternalServerError)
        return
    }
    
    // Copy headers
    for key, values := range r.Header {
        for _, value := range values {
            proxyReq.Header.Add(key, value)
        }
    }
    
    // Execute middleware chain
    gw.executeMiddleware(ctx, proxyReq, w)
}

func (gw *APIGateway) findRoute(method, path string) *Route {
    for _, route := range gw.routes {
        if route.Method == method && strings.HasPrefix(path, route.PathPrefix) {
            return &route
        }
    }
    return nil
}

func (gw *APIGateway) executeMiddleware(ctx context.Context, req *http.Request, w http.ResponseWriter) {
    if len(gw.middleware) == 0 {
        gw.proxyRequest(ctx, req, w)
        return
    }
    
    current := 0
    var next http.Handler
    
    next = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if current < len(gw.middleware) {
            middleware := gw.middleware[current]
            current++
            middleware.Process(ctx, r, next)
        } else {
            gw.proxyRequest(ctx, r, w)
        }
    })
    
    next.ServeHTTP(w, req)
}

func (gw *APIGateway) proxyRequest(ctx context.Context, req *http.Request, w http.ResponseWriter) {
    client := &http.Client{Timeout: 30 * time.Second}
    
    resp, err := client.Do(req)
    if err != nil {
        http.Error(w, "Service unavailable", http.StatusServiceUnavailable)
        return
    }
    defer resp.Body.Close()
    
    // Copy response headers
    for key, values := range resp.Header {
        for _, value := range values {
            w.Header().Add(key, value)
        }
    }
    
    w.WriteHeader(resp.StatusCode)
    
    // Copy response body
    if _, err := w.Write([]byte{}); err != nil {
        // Handle error
    }
}

// Middleware Examples
type LoggingMiddleware struct{}

func (lm *LoggingMiddleware) Process(ctx context.Context, req *http.Request, next http.Handler) http.ResponseWriter {
    start := time.Now()
    
    // Create response writer wrapper
    rw := &ResponseWriterWrapper{
        ResponseWriter: req.Response,
        statusCode:     200,
    }
    
    next.ServeHTTP(rw, req)
    
    duration := time.Since(start)
    fmt.Printf("%s %s %d %v\n", req.Method, req.URL.Path, rw.statusCode, duration)
    
    return rw
}

type RateLimitMiddleware struct {
    limiter RateLimiter
}

func (rlm *RateLimitMiddleware) Process(ctx context.Context, req *http.Request, next http.Handler) http.ResponseWriter {
    clientIP := getClientIP(req)
    
    if !rlm.limiter.Allow(clientIP) {
        w := req.Response
        w.WriteHeader(http.StatusTooManyRequests)
        w.Write([]byte("Rate limit exceeded"))
        return w
    }
    
    return next.ServeHTTP(req.Response, req)
}

type AuthMiddleware struct {
    authService AuthService
}

func (am *AuthMiddleware) Process(ctx context.Context, req *http.Request, next http.Handler) http.ResponseWriter {
    token := req.Header.Get("Authorization")
    if token == "" {
        w := req.Response
        w.WriteHeader(http.StatusUnauthorized)
        w.Write([]byte("Authorization header required"))
        return w
    }
    
    user, err := am.authService.ValidateToken(ctx, token)
    if err != nil {
        w := req.Response
        w.WriteHeader(http.StatusUnauthorized)
        w.Write([]byte("Invalid token"))
        return w
    }
    
    // Add user to context
    ctx = context.WithValue(ctx, "user", user)
    req = req.WithContext(ctx)
    
    return next.ServeHTTP(req.Response, req)
}

// Response Writer Wrapper
type ResponseWriterWrapper struct {
    http.ResponseWriter
    statusCode int
}

func (rw *ResponseWriterWrapper) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}

func (rw *ResponseWriterWrapper) Write(data []byte) (int, error) {
    return rw.ResponseWriter.Write(data)
}

// Helper functions
func getClientIP(req *http.Request) string {
    ip := req.Header.Get("X-Forwarded-For")
    if ip == "" {
        ip = req.Header.Get("X-Real-IP")
    }
    if ip == "" {
        ip = req.RemoteAddr
    }
    return ip
}

// Interfaces
type RateLimiter interface {
    Allow(clientIP string) bool
}

type AuthService interface {
    ValidateToken(ctx context.Context, token string) (*User, error)
}
```

---

## 🎯 Best Practices Summary

### 1. Service Design
- **Single Responsibility**: Each service has one business capability
- **API-First**: Design APIs before implementation
- **Domain-Driven Design**: Use DDD principles for service boundaries
- **Versioning**: Implement proper API versioning strategy

### 2. Communication
- **Synchronous**: Use HTTP/gRPC for request-response patterns
- **Asynchronous**: Use message queues for event-driven communication
- **Circuit Breaker**: Implement fault tolerance patterns
- **Retry Logic**: Implement exponential backoff for retries

### 3. Data Management
- **Database per Service**: Each service owns its data
- **Saga Pattern**: Use sagas for distributed transactions
- **Event Sourcing**: Consider event sourcing for audit trails
- **CQRS**: Separate read and write models when needed

### 4. Deployment
- **Containerization**: Use Docker for consistent deployments
- **Orchestration**: Use Kubernetes for container orchestration
- **CI/CD**: Implement automated deployment pipelines
- **Blue-Green**: Use blue-green deployments for zero downtime

### 5. Monitoring
- **Distributed Tracing**: Implement distributed tracing
- **Metrics**: Collect and monitor service metrics
- **Logging**: Implement structured logging
- **Alerting**: Set up proper alerting for failures

---

**🏗️ Master these microservices patterns to build scalable, maintainable, and resilient distributed systems! 🚀**
