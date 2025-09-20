# Architecture Design

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [Design Patterns](#design-patterns)
4. [Scalability Patterns](#scalability-patterns)
5. [Microservices Architecture](#microservices-architecture)
6. [Event-Driven Architecture](#event-driven-architecture)
7. [Domain-Driven Design](#domain-driven-design)
8. [Implementations](#implementations)
9. [Follow-up Questions](#follow-up-questions)
10. [Sources](#sources)
11. [Projects](#projects)

## Overview

### Learning Objectives

- Master advanced architecture principles and patterns
- Design scalable and maintainable systems
- Implement microservices and event-driven architectures
- Apply domain-driven design methodologies
- Create architecture documentation and governance
- Lead architectural decision-making processes

### What is Architecture Design?

Architecture Design involves creating high-level system designs that define structure, components, interactions, and principles to guide system development and evolution.

## Architecture Principles

### 1. SOLID Principles

#### Single Responsibility Principle
```go
package main

import "fmt"

// Bad: Multiple responsibilities
type UserService struct {
    db     Database
    email  EmailService
    logger Logger
}

func (us *UserService) CreateUser(user User) error {
    // Database operation
    err := us.db.Save(user)
    if err != nil {
        return err
    }
    
    // Email sending
    err = us.email.SendWelcomeEmail(user.Email)
    if err != nil {
        return err
    }
    
    // Logging
    us.logger.Log("User created: " + user.Email)
    
    return nil
}

// Good: Single responsibility
type UserRepository struct {
    db Database
}

func (ur *UserRepository) Save(user User) error {
    return ur.db.Save(user)
}

type UserNotificationService struct {
    email EmailService
}

func (uns *UserNotificationService) SendWelcomeEmail(user User) error {
    return uns.email.SendWelcomeEmail(user.Email)
}

type UserAuditService struct {
    logger Logger
}

func (uas *UserAuditService) LogUserCreation(user User) {
    uas.logger.Log("User created: " + user.Email)
}

type UserService struct {
    repository   *UserRepository
    notification *UserNotificationService
    audit       *UserAuditService
}

func (us *UserService) CreateUser(user User) error {
    // Each service has a single responsibility
    if err := us.repository.Save(user); err != nil {
        return err
    }
    
    if err := us.notification.SendWelcomeEmail(user); err != nil {
        return err
    }
    
    us.audit.LogUserCreation(user)
    return nil
}

type User struct {
    ID    string
    Email string
    Name  string
}

type Database interface {
    Save(user User) error
}

type EmailService interface {
    SendWelcomeEmail(email string) error
}

type Logger interface {
    Log(message string)
}

func main() {
    fmt.Println("Single Responsibility Principle example")
}
```

#### Open/Closed Principle
```go
package main

import "fmt"

// Bad: Not open for extension
type PaymentProcessor struct {
    paymentType string
}

func (pp *PaymentProcessor) ProcessPayment(amount float64) error {
    switch pp.paymentType {
    case "credit_card":
        return pp.processCreditCard(amount)
    case "paypal":
        return pp.processPayPal(amount)
    case "stripe":
        return pp.processStripe(amount)
    default:
        return fmt.Errorf("unsupported payment type")
    }
}

func (pp *PaymentProcessor) processCreditCard(amount float64) error {
    fmt.Printf("Processing credit card payment: $%.2f\n", amount)
    return nil
}

func (pp *PaymentProcessor) processPayPal(amount float64) error {
    fmt.Printf("Processing PayPal payment: $%.2f\n", amount)
    return nil
}

func (pp *PaymentProcessor) processStripe(amount float64) error {
    fmt.Printf("Processing Stripe payment: $%.2f\n", amount)
    return nil
}

// Good: Open for extension, closed for modification
type PaymentMethod interface {
    ProcessPayment(amount float64) error
}

type CreditCardPayment struct{}

func (ccp *CreditCardPayment) ProcessPayment(amount float64) error {
    fmt.Printf("Processing credit card payment: $%.2f\n", amount)
    return nil
}

type PayPalPayment struct{}

func (ppp *PayPalPayment) ProcessPayment(amount float64) error {
    fmt.Printf("Processing PayPal payment: $%.2f\n", amount)
    return nil
}

type StripePayment struct{}

func (sp *StripePayment) ProcessPayment(amount float64) error {
    fmt.Printf("Processing Stripe payment: $%.2f\n", amount)
    return nil
}

type PaymentProcessor struct {
    paymentMethod PaymentMethod
}

func (pp *PaymentProcessor) SetPaymentMethod(method PaymentMethod) {
    pp.paymentMethod = method
}

func (pp *PaymentProcessor) ProcessPayment(amount float64) error {
    return pp.paymentMethod.ProcessPayment(amount)
}

// Easy to extend with new payment methods
type BitcoinPayment struct{}

func (bp *BitcoinPayment) ProcessPayment(amount float64) error {
    fmt.Printf("Processing Bitcoin payment: $%.2f\n", amount)
    return nil
}

func main() {
    processor := &PaymentProcessor{}
    
    // Use different payment methods
    processor.SetPaymentMethod(&CreditCardPayment{})
    processor.ProcessPayment(100.0)
    
    processor.SetPaymentMethod(&PayPalPayment{})
    processor.ProcessPayment(200.0)
    
    processor.SetPaymentMethod(&BitcoinPayment{})
    processor.ProcessPayment(300.0)
}
```

### 2. Architecture Patterns

#### Layered Architecture
```go
package main

import "fmt"

// Presentation Layer
type UserController struct {
    userService *UserService
}

func (uc *UserController) CreateUser(request CreateUserRequest) CreateUserResponse {
    user := User{
        Name:  request.Name,
        Email: request.Email,
    }
    
    err := uc.userService.CreateUser(user)
    if err != nil {
        return CreateUserResponse{
            Success: false,
            Error:   err.Error(),
        }
    }
    
    return CreateUserResponse{
        Success: true,
        Message: "User created successfully",
    }
}

// Business Logic Layer
type UserService struct {
    userRepository *UserRepository
    emailService   *EmailService
}

func (us *UserService) CreateUser(user User) error {
    // Business logic
    if user.Email == "" {
        return fmt.Errorf("email is required")
    }
    
    // Check if user already exists
    existingUser, err := us.userRepository.FindByEmail(user.Email)
    if err == nil && existingUser != nil {
        return fmt.Errorf("user already exists")
    }
    
    // Save user
    err = us.userRepository.Save(user)
    if err != nil {
        return err
    }
    
    // Send welcome email
    return us.emailService.SendWelcomeEmail(user.Email)
}

// Data Access Layer
type UserRepository struct {
    db Database
}

func (ur *UserRepository) Save(user User) error {
    return ur.db.Save(user)
}

func (ur *UserRepository) FindByEmail(email string) (*User, error) {
    return ur.db.FindByEmail(email)
}

// Infrastructure Layer
type Database interface {
    Save(user User) error
    FindByEmail(email string) (*User, error)
}

type EmailService struct {
    smtpClient SMTPClient
}

func (es *EmailService) SendWelcomeEmail(email string) error {
    return es.smtpClient.SendEmail(email, "Welcome!", "Welcome to our platform!")
}

type SMTPClient interface {
    SendEmail(to, subject, body string) error
}

// DTOs
type CreateUserRequest struct {
    Name  string
    Email string
}

type CreateUserResponse struct {
    Success bool
    Message string
    Error   string
}

type User struct {
    ID    string
    Name  string
    Email string
}

func main() {
    fmt.Println("Layered Architecture example")
}
```

## Design Patterns

### 1. Creational Patterns

#### Factory Pattern
```go
package main

import "fmt"

// Product interface
type Database interface {
    Connect() error
    Query(sql string) ([]map[string]interface{}, error)
    Close() error
}

// Concrete products
type MySQLDatabase struct {
    host     string
    port     int
    username string
    password string
    database string
}

func (mdb *MySQLDatabase) Connect() error {
    fmt.Printf("Connecting to MySQL at %s:%d\n", mdb.host, mdb.port)
    return nil
}

func (mdb *MySQLDatabase) Query(sql string) ([]map[string]interface{}, error) {
    fmt.Printf("Executing MySQL query: %s\n", sql)
    return []map[string]interface{}{}, nil
}

func (mdb *MySQLDatabase) Close() error {
    fmt.Println("Closing MySQL connection")
    return nil
}

type PostgreSQLDatabase struct {
    host     string
    port     int
    username string
    password string
    database string
}

func (pdb *PostgreSQLDatabase) Connect() error {
    fmt.Printf("Connecting to PostgreSQL at %s:%d\n", pdb.host, pdb.port)
    return nil
}

func (pdb *PostgreSQLDatabase) Query(sql string) ([]map[string]interface{}, error) {
    fmt.Printf("Executing PostgreSQL query: %s\n", sql)
    return []map[string]interface{}{}, nil
}

func (pdb *PostgreSQLDatabase) Close() error {
    fmt.Println("Closing PostgreSQL connection")
    return nil
}

// Factory
type DatabaseFactory struct{}

func (df *DatabaseFactory) CreateDatabase(dbType string, config DatabaseConfig) Database {
    switch dbType {
    case "mysql":
        return &MySQLDatabase{
            host:     config.Host,
            port:     config.Port,
            username: config.Username,
            password: config.Password,
            database: config.Database,
        }
    case "postgresql":
        return &PostgreSQLDatabase{
            host:     config.Host,
            port:     config.Port,
            username: config.Username,
            password: config.Password,
            database: config.Database,
        }
    default:
        return nil
    }
}

type DatabaseConfig struct {
    Host     string
    Port     int
    Username string
    Password string
    Database string
}

func main() {
    factory := &DatabaseFactory{}
    
    config := DatabaseConfig{
        Host:     "localhost",
        Port:     3306,
        Username: "user",
        Password: "password",
        Database: "testdb",
    }
    
    // Create different database types
    mysqlDB := factory.CreateDatabase("mysql", config)
    mysqlDB.Connect()
    mysqlDB.Query("SELECT * FROM users")
    mysqlDB.Close()
    
    postgresDB := factory.CreateDatabase("postgresql", config)
    postgresDB.Connect()
    postgresDB.Query("SELECT * FROM users")
    postgresDB.Close()
}
```

### 2. Structural Patterns

#### Adapter Pattern
```go
package main

import "fmt"

// Legacy system interface
type LegacyPaymentSystem interface {
    ProcessLegacyPayment(amount float64, currency string) bool
}

// Legacy implementation
type LegacyPaymentProcessor struct{}

func (lpp *LegacyPaymentProcessor) ProcessLegacyPayment(amount float64, currency string) bool {
    fmt.Printf("Legacy system processing payment: %.2f %s\n", amount, currency)
    return true
}

// New system interface
type ModernPaymentSystem interface {
    ProcessPayment(amount float64) error
}

// Adapter to make legacy system compatible with new interface
type PaymentAdapter struct {
    legacySystem LegacyPaymentSystem
}

func (pa *PaymentAdapter) ProcessPayment(amount float64) error {
    success := pa.legacySystem.ProcessLegacyPayment(amount, "USD")
    if !success {
        return fmt.Errorf("payment processing failed")
    }
    return nil
}

// New system that uses the adapter
type PaymentService struct {
    paymentSystem ModernPaymentSystem
}

func (ps *PaymentService) ProcessPayment(amount float64) error {
    return ps.paymentSystem.ProcessPayment(amount)
}

func main() {
    // Create legacy system
    legacySystem := &LegacyPaymentProcessor{}
    
    // Create adapter
    adapter := &PaymentAdapter{legacySystem: legacySystem}
    
    // Use adapter in new system
    paymentService := &PaymentService{paymentSystem: adapter}
    
    // Process payment through new interface
    err := paymentService.ProcessPayment(100.0)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Println("Payment processed successfully")
    }
}
```

## Scalability Patterns

### 1. Load Balancing

#### Load Balancer Implementation
```go
package main

import (
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type Server struct {
    ID       string
    URL      string
    Weight   int
    Active   bool
    Requests int
    mutex    sync.RWMutex
}

func (s *Server) IsHealthy() bool {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    return s.Active
}

func (s *Server) IncrementRequests() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    s.Requests++
}

func (s *Server) GetRequestCount() int {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    return s.Requests
}

type LoadBalancer struct {
    servers []*Server
    strategy LoadBalancingStrategy
}

type LoadBalancingStrategy interface {
    SelectServer(servers []*Server) *Server
}

// Round Robin Strategy
type RoundRobinStrategy struct {
    current int
    mutex   sync.Mutex
}

func (rr *RoundRobinStrategy) SelectServer(servers []*Server) *Server {
    rr.mutex.Lock()
    defer rr.mutex.Unlock()
    
    healthyServers := []*Server{}
    for _, server := range servers {
        if server.IsHealthy() {
            healthyServers = append(healthyServers, server)
        }
    }
    
    if len(healthyServers) == 0 {
        return nil
    }
    
    server := healthyServers[rr.current%len(healthyServers)]
    rr.current++
    return server
}

// Weighted Round Robin Strategy
type WeightedRoundRobinStrategy struct {
    current int
    mutex   sync.Mutex
}

func (wrr *WeightedRoundRobinStrategy) SelectServer(servers []*Server) *Server {
    wrr.mutex.Lock()
    defer wrr.mutex.Unlock()
    
    healthyServers := []*Server{}
    for _, server := range servers {
        if server.IsHealthy() {
            healthyServers = append(healthyServers, server)
        }
    }
    
    if len(healthyServers) == 0 {
        return nil
    }
    
    // Calculate total weight
    totalWeight := 0
    for _, server := range healthyServers {
        totalWeight += server.Weight
    }
    
    // Select server based on weight
    currentWeight := 0
    for _, server := range healthyServers {
        currentWeight += server.Weight
        if wrr.current < currentWeight {
            wrr.current++
            return server
        }
    }
    
    // Fallback to first server
    return healthyServers[0]
}

// Least Connections Strategy
type LeastConnectionsStrategy struct{}

func (lc *LeastConnectionsStrategy) SelectServer(servers []*Server) *Server {
    var selectedServer *Server
    minConnections := int(^uint(0) >> 1) // Max int
    
    for _, server := range servers {
        if server.IsHealthy() {
            connections := server.GetRequestCount()
            if connections < minConnections {
                minConnections = connections
                selectedServer = server
            }
        }
    }
    
    return selectedServer
}

func NewLoadBalancer(strategy LoadBalancingStrategy) *LoadBalancer {
    return &LoadBalancer{
        servers:  []*Server{},
        strategy: strategy,
    }
}

func (lb *LoadBalancer) AddServer(server *Server) {
    lb.servers = append(lb.servers, server)
}

func (lb *LoadBalancer) RouteRequest() *Server {
    server := lb.strategy.SelectServer(lb.servers)
    if server != nil {
        server.IncrementRequests()
    }
    return server
}

func main() {
    // Create servers
    servers := []*Server{
        {ID: "server1", URL: "http://server1.com", Weight: 3, Active: true},
        {ID: "server2", URL: "http://server2.com", Weight: 2, Active: true},
        {ID: "server3", URL: "http://server3.com", Weight: 1, Active: true},
    }
    
    // Test Round Robin
    fmt.Println("Round Robin Load Balancing:")
    lb := NewLoadBalancer(&RoundRobinStrategy{})
    for _, server := range servers {
        lb.AddServer(server)
    }
    
    for i := 0; i < 10; i++ {
        server := lb.RouteRequest()
        if server != nil {
            fmt.Printf("Request %d routed to %s\n", i+1, server.ID)
        }
    }
    
    // Test Weighted Round Robin
    fmt.Println("\nWeighted Round Robin Load Balancing:")
    lb2 := NewLoadBalancer(&WeightedRoundRobinStrategy{})
    for _, server := range servers {
        lb2.AddServer(server)
    }
    
    for i := 0; i < 10; i++ {
        server := lb2.RouteRequest()
        if server != nil {
            fmt.Printf("Request %d routed to %s\n", i+1, server.ID)
        }
    }
    
    // Test Least Connections
    fmt.Println("\nLeast Connections Load Balancing:")
    lb3 := NewLoadBalancer(&LeastConnectionsStrategy{})
    for _, server := range servers {
        lb3.AddServer(server)
    }
    
    for i := 0; i < 10; i++ {
        server := lb3.RouteRequest()
        if server != nil {
            fmt.Printf("Request %d routed to %s\n", i+1, server.ID)
        }
    }
}
```

### 2. Caching Strategies

#### Multi-Level Cache
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type CacheItem struct {
    Value     interface{}
    ExpiresAt time.Time
    AccessCount int
    LastAccess time.Time
}

type CacheLevel int

const (
    L1 CacheLevel = iota
    L2
    L3
)

type MultiLevelCache struct {
    l1Cache    map[string]*CacheItem
    l2Cache    map[string]*CacheItem
    l3Cache    map[string]*CacheItem
    mutex      sync.RWMutex
    l1Capacity int
    l2Capacity int
    l3Capacity int
}

func NewMultiLevelCache(l1Cap, l2Cap, l3Cap int) *MultiLevelCache {
    return &MultiLevelCache{
        l1Cache:    make(map[string]*CacheItem),
        l2Cache:    make(map[string]*CacheItem),
        l3Cache:    make(map[string]*CacheItem),
        l1Capacity: l1Cap,
        l2Capacity: l2Cap,
        l3Capacity: l3Cap,
    }
}

func (mlc *MultiLevelCache) Get(key string) (interface{}, bool) {
    mlc.mutex.Lock()
    defer mlc.mutex.Unlock()
    
    // Check L1 cache first
    if item, exists := mlc.l1Cache[key]; exists {
        if time.Now().Before(item.ExpiresAt) {
            item.AccessCount++
            item.LastAccess = time.Now()
            return item.Value, true
        }
        delete(mlc.l1Cache, key)
    }
    
    // Check L2 cache
    if item, exists := mlc.l2Cache[key]; exists {
        if time.Now().Before(item.ExpiresAt) {
            item.AccessCount++
            item.LastAccess = time.Now()
            // Promote to L1
            mlc.promoteToL1(key, item)
            return item.Value, true
        }
        delete(mlc.l2Cache, key)
    }
    
    // Check L3 cache
    if item, exists := mlc.l3Cache[key]; exists {
        if time.Now().Before(item.ExpiresAt) {
            item.AccessCount++
            item.LastAccess = time.Now()
            // Promote to L2
            mlc.promoteToL2(key, item)
            return item.Value, true
        }
        delete(mlc.l3Cache, key)
    }
    
    return nil, false
}

func (mlc *MultiLevelCache) Set(key string, value interface{}, ttl time.Duration) {
    mlc.mutex.Lock()
    defer mlc.mutex.Unlock()
    
    item := &CacheItem{
        Value:      value,
        ExpiresAt:  time.Now().Add(ttl),
        AccessCount: 1,
        LastAccess: time.Now(),
    }
    
    // Always start in L1
    mlc.setInL1(key, item)
}

func (mlc *MultiLevelCache) setInL1(key string, item *CacheItem) {
    // Check if L1 is full
    if len(mlc.l1Cache) >= mlc.l1Capacity {
        // Evict least recently used item to L2
        mlc.evictLRUFromL1()
    }
    
    mlc.l1Cache[key] = item
}

func (mlc *MultiLevelCache) promoteToL1(key string, item *CacheItem) {
    // Remove from L2
    delete(mlc.l2Cache, key)
    
    // Check if L1 is full
    if len(mlc.l1Cache) >= mlc.l1Capacity {
        mlc.evictLRUFromL1()
    }
    
    mlc.l1Cache[key] = item
}

func (mlc *MultiLevelCache) promoteToL2(key string, item *CacheItem) {
    // Remove from L3
    delete(mlc.l3Cache, key)
    
    // Check if L2 is full
    if len(mlc.l2Cache) >= mlc.l2Capacity {
        mlc.evictLRUFromL2()
    }
    
    mlc.l2Cache[key] = item
}

func (mlc *MultiLevelCache) evictLRUFromL1() {
    var lruKey string
    var lruTime time.Time
    
    for key, item := range mlc.l1Cache {
        if lruKey == "" || item.LastAccess.Before(lruTime) {
            lruKey = key
            lruTime = item.LastAccess
        }
    }
    
    if lruKey != "" {
        item := mlc.l1Cache[lruKey]
        delete(mlc.l1Cache, lruKey)
        
        // Move to L2
        if len(mlc.l2Cache) >= mlc.l2Capacity {
            mlc.evictLRUFromL2()
        }
        mlc.l2Cache[lruKey] = item
    }
}

func (mlc *MultiLevelCache) evictLRUFromL2() {
    var lruKey string
    var lruTime time.Time
    
    for key, item := range mlc.l2Cache {
        if lruKey == "" || item.LastAccess.Before(lruTime) {
            lruKey = key
            lruTime = item.LastAccess
        }
    }
    
    if lruKey != "" {
        item := mlc.l2Cache[lruKey]
        delete(mlc.l2Cache, lruKey)
        
        // Move to L3
        if len(mlc.l3Cache) >= mlc.l3Capacity {
            mlc.evictLRUFromL3()
        }
        mlc.l3Cache[lruKey] = item
    }
}

func (mlc *MultiLevelCache) evictLRUFromL3() {
    var lruKey string
    var lruTime time.Time
    
    for key, item := range mlc.l3Cache {
        if lruKey == "" || item.LastAccess.Before(lruTime) {
            lruKey = key
            lruTime = item.LastAccess
        }
    }
    
    if lruKey != "" {
        delete(mlc.l3Cache, lruKey)
    }
}

func (mlc *MultiLevelCache) GetStats() map[string]interface{} {
    mlc.mutex.RLock()
    defer mlc.mutex.RUnlock()
    
    return map[string]interface{}{
        "l1_size": len(mlc.l1Cache),
        "l2_size": len(mlc.l2Cache),
        "l3_size": len(mlc.l3Cache),
        "l1_capacity": mlc.l1Capacity,
        "l2_capacity": mlc.l2Capacity,
        "l3_capacity": mlc.l3Capacity,
    }
}

func main() {
    cache := NewMultiLevelCache(3, 5, 10)
    
    // Set some values
    cache.Set("key1", "value1", 5*time.Second)
    cache.Set("key2", "value2", 10*time.Second)
    cache.Set("key3", "value3", 15*time.Second)
    cache.Set("key4", "value4", 20*time.Second)
    
    // Test retrieval
    if value, exists := cache.Get("key1"); exists {
        fmt.Printf("Found key1: %v\n", value)
    }
    
    if value, exists := cache.Get("key2"); exists {
        fmt.Printf("Found key2: %v\n", value)
    }
    
    // Print stats
    stats := cache.GetStats()
    fmt.Printf("Cache stats: %+v\n", stats)
}
```

## Microservices Architecture

### 1. Service Discovery

#### Service Registry
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type ServiceInstance struct {
    ID       string
    Name     string
    Address  string
    Port     int
    Health   string
    LastSeen time.Time
}

type ServiceRegistry struct {
    services map[string][]*ServiceInstance
    mutex    sync.RWMutex
}

func NewServiceRegistry() *ServiceRegistry {
    return &ServiceRegistry{
        services: make(map[string][]*ServiceInstance),
    }
}

func (sr *ServiceRegistry) Register(instance *ServiceInstance) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    instance.LastSeen = time.Now()
    sr.services[instance.Name] = append(sr.services[instance.Name], instance)
    
    fmt.Printf("Registered service: %s at %s:%d\n", instance.Name, instance.Address, instance.Port)
}

func (sr *ServiceRegistry) Deregister(serviceName, instanceID string) {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    if instances, exists := sr.services[serviceName]; exists {
        for i, instance := range instances {
            if instance.ID == instanceID {
                sr.services[serviceName] = append(instances[:i], instances[i+1:]...)
                fmt.Printf("Deregistered service: %s instance %s\n", serviceName, instanceID)
                break
            }
        }
    }
}

func (sr *ServiceRegistry) Discover(serviceName string) []*ServiceInstance {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    if instances, exists := sr.services[serviceName]; exists {
        // Filter healthy instances
        healthyInstances := []*ServiceInstance{}
        for _, instance := range instances {
            if instance.Health == "healthy" {
                healthyInstances = append(healthyInstances, instance)
            }
        }
        return healthyInstances
    }
    
    return []*ServiceInstance{}
}

func (sr *ServiceRegistry) HealthCheck() {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    now := time.Now()
    for serviceName, instances := range sr.services {
        for i := len(instances) - 1; i >= 0; i-- {
            instance := instances[i]
            if now.Sub(instance.LastSeen) > 30*time.Second {
                // Remove stale instances
                sr.services[serviceName] = append(instances[:i], instances[i+1:]...)
                fmt.Printf("Removed stale instance: %s\n", instance.ID)
            }
        }
    }
}

func (sr *ServiceRegistry) GetServices() map[string][]*ServiceInstance {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    result := make(map[string][]*ServiceInstance)
    for name, instances := range sr.services {
        result[name] = append([]*ServiceInstance(nil), instances...)
    }
    return result
}

func main() {
    registry := NewServiceRegistry()
    
    // Register some services
    registry.Register(&ServiceInstance{
        ID:      "user-service-1",
        Name:    "user-service",
        Address: "192.168.1.10",
        Port:    8080,
        Health:  "healthy",
    })
    
    registry.Register(&ServiceInstance{
        ID:      "user-service-2",
        Name:    "user-service",
        Address: "192.168.1.11",
        Port:    8080,
        Health:  "healthy",
    })
    
    registry.Register(&ServiceInstance{
        ID:      "order-service-1",
        Name:    "order-service",
        Address: "192.168.1.12",
        Port:    8081,
        Health:  "healthy",
    })
    
    // Discover services
    userServices := registry.Discover("user-service")
    fmt.Printf("Found %d user service instances\n", len(userServices))
    
    orderServices := registry.Discover("order-service")
    fmt.Printf("Found %d order service instances\n", len(orderServices))
    
    // Print all services
    services := registry.GetServices()
    for name, instances := range services {
        fmt.Printf("Service %s has %d instances\n", name, len(instances))
    }
}
```

## Follow-up Questions

### 1. Architecture Principles
**Q: What's the difference between layered and hexagonal architecture?**
A: Layered architecture organizes code into horizontal layers, while hexagonal architecture organizes around business logic with ports and adapters.

### 2. Design Patterns
**Q: When should you use the factory pattern vs. the builder pattern?**
A: Use factory pattern when creating objects of the same type with different configurations, and builder pattern when creating complex objects step by step.

### 3. Scalability
**Q: How do you choose between horizontal and vertical scaling?**
A: Horizontal scaling (adding more servers) is generally preferred for better fault tolerance and cost-effectiveness, while vertical scaling (upgrading hardware) is simpler but has limits.

## Sources

### Books
- **Clean Architecture** by Robert C. Martin
- **Patterns of Enterprise Application Architecture** by Martin Fowler
- **Building Microservices** by Sam Newman

### Online Resources
- **AWS Architecture Center** - Cloud architecture patterns
- **Microsoft Architecture Center** - Enterprise architecture
- **Google Cloud Architecture** - Scalable system design

## Projects

### 1. Microservices Platform
**Objective**: Design and implement a complete microservices platform
**Requirements**: Service discovery, API gateway, monitoring, logging
**Deliverables**: Production-ready microservices platform

### 2. Architecture Decision Records
**Objective**: Create a system for documenting architectural decisions
**Requirements**: Decision templates, review process, version control
**Deliverables**: Complete ADR system

### 3. Scalability Testing Framework
**Objective**: Build a framework for testing system scalability
**Requirements**: Load testing, performance monitoring, bottleneck identification
**Deliverables**: Comprehensive testing framework

---

**Next**: [Innovation Research](../../../README.md) | **Previous**: [Technical Leadership](../../../README.md) | **Up**: [Phase 3](README.md/)
