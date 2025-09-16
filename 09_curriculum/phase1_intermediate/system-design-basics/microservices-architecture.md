# Microservices Architecture

## Overview

This module covers microservices architecture concepts including service decomposition, communication patterns, service discovery, and deployment strategies. These concepts are essential for building scalable, maintainable distributed systems.

## Table of Contents

1. [Service Decomposition](#service-decomposition)
2. [Communication Patterns](#communication-patterns)
3. [Service Discovery](#service-discovery)
4. [Deployment Strategies](#deployment-strategies)
5. [Applications](#applications)
6. [Complexity Analysis](#complexity-analysis)
7. [Follow-up Questions](#follow-up-questions)

## Service Decomposition

### Theory

Service decomposition involves breaking down a monolithic application into smaller, independent services. Each service should have a single responsibility and be loosely coupled with other services.

### Service Decomposition Implementation

#### Golang Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type Service struct {
    ID          string
    Name        string
    Port        int
    Dependencies []string
    Health      string
    mutex       sync.RWMutex
}

type MicroservicesArchitecture struct {
    Services map[string]*Service
    mutex    sync.RWMutex
}

func NewMicroservicesArchitecture() *MicroservicesArchitecture {
    return &MicroservicesArchitecture{
        Services: make(map[string]*Service),
    }
}

func (ma *MicroservicesArchitecture) AddService(id, name string, port int, dependencies []string) *Service {
    service := &Service{
        ID:           id,
        Name:         name,
        Port:         port,
        Dependencies: dependencies,
        Health:       "healthy",
    }
    
    ma.mutex.Lock()
    ma.Services[id] = service
    ma.mutex.Unlock()
    
    fmt.Printf("Added service: %s (%s) on port %d\n", id, name, port)
    return service
}

func (ma *MicroservicesArchitecture) RemoveService(id string) bool {
    ma.mutex.Lock()
    defer ma.mutex.Unlock()
    
    if _, exists := ma.Services[id]; exists {
        delete(ma.Services, id)
        fmt.Printf("Removed service: %s\n", id)
        return true
    }
    
    return false
}

func (ma *MicroservicesArchitecture) GetService(id string) *Service {
    ma.mutex.RLock()
    defer ma.mutex.RUnlock()
    
    return ma.Services[id]
}

func (ma *MicroservicesArchitecture) GetServices() map[string]*Service {
    ma.mutex.RLock()
    defer ma.mutex.RUnlock()
    
    services := make(map[string]*Service)
    for id, service := range ma.Services {
        services[id] = service
    }
    
    return services
}

func (ma *MicroservicesArchitecture) UpdateServiceHealth(id, health string) {
    ma.mutex.RLock()
    service, exists := ma.Services[id]
    ma.mutex.RUnlock()
    
    if exists {
        service.mutex.Lock()
        service.Health = health
        service.mutex.Unlock()
        fmt.Printf("Updated service %s health to: %s\n", id, health)
    }
}

func (ma *MicroservicesArchitecture) GetServiceDependencies(id string) []string {
    ma.mutex.RLock()
    defer ma.mutex.RUnlock()
    
    if service, exists := ma.Services[id]; exists {
        return service.Dependencies
    }
    
    return nil
}

func (ma *MicroservicesArchitecture) CheckDependencies(id string) bool {
    dependencies := ma.GetServiceDependencies(id)
    if dependencies == nil {
        return true
    }
    
    ma.mutex.RLock()
    defer ma.mutex.RUnlock()
    
    for _, depID := range dependencies {
        if service, exists := ma.Services[depID]; exists {
            if service.Health != "healthy" {
                return false
            }
        } else {
            return false
        }
    }
    
    return true
}

func (ma *MicroservicesArchitecture) GetServiceStatus() map[string]interface{} {
    ma.mutex.RLock()
    defer ma.mutex.RUnlock()
    
    status := make(map[string]interface{})
    for id, service := range ma.Services {
        service.mutex.RLock()
        status[id] = map[string]interface{}{
            "name":         service.Name,
            "port":         service.Port,
            "health":       service.Health,
            "dependencies": service.Dependencies,
        }
        service.mutex.RUnlock()
    }
    
    return status
}

func (ma *MicroservicesArchitecture) StartService(id string) {
    service := ma.GetService(id)
    if service == nil {
        fmt.Printf("Service %s not found\n", id)
        return
    }
    
    // Check dependencies
    if !ma.CheckDependencies(id) {
        fmt.Printf("Service %s cannot start: dependencies not healthy\n", id)
        return
    }
    
    // Start service (simplified)
    fmt.Printf("Starting service %s on port %d\n", id, service.Port)
    ma.UpdateServiceHealth(id, "healthy")
}

func (ma *MicroservicesArchitecture) StopService(id string) {
    service := ma.GetService(id)
    if service == nil {
        fmt.Printf("Service %s not found\n", id)
        return
    }
    
    fmt.Printf("Stopping service %s\n", id)
    ma.UpdateServiceHealth(id, "stopped")
}

func main() {
    ma := NewMicroservicesArchitecture()
    
    fmt.Println("Microservices Architecture Demo:")
    
    // Add services
    ma.AddService("user-service", "User Service", 8080, nil)
    ma.AddService("order-service", "Order Service", 8081, []string{"user-service"})
    ma.AddService("payment-service", "Payment Service", 8082, []string{"user-service"})
    ma.AddService("notification-service", "Notification Service", 8083, []string{"user-service", "order-service"})
    
    // Start services
    ma.StartService("user-service")
    ma.StartService("order-service")
    ma.StartService("payment-service")
    ma.StartService("notification-service")
    
    // Show status
    status := ma.GetServiceStatus()
    fmt.Printf("Service status: %v\n", status)
    
    // Stop a dependency
    ma.StopService("user-service")
    
    // Try to start dependent service
    ma.StartService("order-service")
    
    // Show updated status
    status = ma.GetServiceStatus()
    fmt.Printf("Updated service status: %v\n", status)
}
```

## Communication Patterns

### Theory

Microservices communicate through various patterns including synchronous (HTTP/RPC) and asynchronous (message queues) communication. The choice of pattern depends on the use case and requirements.

### Communication Patterns Implementation

#### Golang Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type Message struct {
    ID        string
    Type      string
    Payload   interface{}
    Timestamp time.Time
    mutex     sync.RWMutex
}

type MessageQueue struct {
    Messages []*Message
    mutex    sync.RWMutex
}

func NewMessageQueue() *MessageQueue {
    return &MessageQueue{
        Messages: make([]*Message, 0),
    }
}

func (mq *MessageQueue) Publish(messageType string, payload interface{}) *Message {
    message := &Message{
        ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
        Type:      messageType,
        Payload:   payload,
        Timestamp: time.Now(),
    }
    
    mq.mutex.Lock()
    mq.Messages = append(mq.Messages, message)
    mq.mutex.Unlock()
    
    fmt.Printf("Published message: %s (%s)\n", message.ID, message.Type)
    return message
}

func (mq *MessageQueue) Consume(messageType string) *Message {
    mq.mutex.Lock()
    defer mq.mutex.Unlock()
    
    for i, message := range mq.Messages {
        if message.Type == messageType {
            // Remove message from queue
            mq.Messages = append(mq.Messages[:i], mq.Messages[i+1:]...)
            fmt.Printf("Consumed message: %s (%s)\n", message.ID, message.Type)
            return message
        }
    }
    
    return nil
}

func (mq *MessageQueue) GetMessages() []*Message {
    mq.mutex.RLock()
    defer mq.mutex.RUnlock()
    
    messages := make([]*Message, len(mq.Messages))
    copy(messages, mq.Messages)
    return messages
}

type ServiceClient struct {
    ServiceID string
    BaseURL   string
    mutex     sync.RWMutex
}

func NewServiceClient(serviceID, baseURL string) *ServiceClient {
    return &ServiceClient{
        ServiceID: serviceID,
        BaseURL:   baseURL,
    }
}

func (sc *ServiceClient) Call(endpoint string, data interface{}) (interface{}, error) {
    // Simulate HTTP call
    fmt.Printf("Service %s calling %s\n", sc.ServiceID, endpoint)
    
    // In a real implementation, you'd make an actual HTTP request
    response := map[string]interface{}{
        "service":  sc.ServiceID,
        "endpoint": endpoint,
        "data":     data,
        "timestamp": time.Now().Format(time.RFC3339),
    }
    
    return response, nil
}

func (sc *ServiceClient) CallAsync(endpoint string, data interface{}, callback func(interface{}, error)) {
    go func() {
        response, err := sc.Call(endpoint, data)
        callback(response, err)
    }()
}

type CommunicationManager struct {
    Clients     map[string]*ServiceClient
    MessageQueue *MessageQueue
    mutex       sync.RWMutex
}

func NewCommunicationManager() *CommunicationManager {
    return &CommunicationManager{
        Clients:     make(map[string]*ServiceClient),
        MessageQueue: NewMessageQueue(),
    }
}

func (cm *CommunicationManager) AddClient(serviceID, baseURL string) *ServiceClient {
    client := NewServiceClient(serviceID, baseURL)
    
    cm.mutex.Lock()
    cm.Clients[serviceID] = client
    cm.mutex.Unlock()
    
    fmt.Printf("Added client for service: %s\n", serviceID)
    return client
}

func (cm *CommunicationManager) GetClient(serviceID string) *ServiceClient {
    cm.mutex.RLock()
    defer cm.mutex.RUnlock()
    
    return cm.Clients[serviceID]
}

func (cm *CommunicationManager) SynchronousCall(fromService, toService, endpoint string, data interface{}) (interface{}, error) {
    client := cm.GetClient(toService)
    if client == nil {
        return nil, fmt.Errorf("service %s not found", toService)
    }
    
    fmt.Printf("Synchronous call: %s -> %s\n", fromService, toService)
    return client.Call(endpoint, data)
}

func (cm *CommunicationManager) AsynchronousCall(fromService, toService, endpoint string, data interface{}, callback func(interface{}, error)) {
    client := cm.GetClient(toService)
    if client == nil {
        callback(nil, fmt.Errorf("service %s not found", toService))
        return
    }
    
    fmt.Printf("Asynchronous call: %s -> %s\n", fromService, toService)
    client.CallAsync(endpoint, data, callback)
}

func (cm *CommunicationManager) PublishMessage(messageType string, payload interface{}) *Message {
    return cm.MessageQueue.Publish(messageType, payload)
}

func (cm *CommunicationManager) SubscribeToMessages(messageType string, handler func(*Message)) {
    go func() {
        for {
            message := cm.MessageQueue.Consume(messageType)
            if message != nil {
                handler(message)
            }
            time.Sleep(100 * time.Millisecond)
        }
    }()
}

func main() {
    cm := NewCommunicationManager()
    
    fmt.Println("Communication Patterns Demo:")
    
    // Add service clients
    cm.AddClient("user-service", "http://user-service:8080")
    cm.AddClient("order-service", "http://order-service:8081")
    cm.AddClient("payment-service", "http://payment-service:8082")
    
    // Synchronous communication
    response, err := cm.SynchronousCall("order-service", "user-service", "/users/123", map[string]interface{}{
        "action": "get_user",
    })
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Response: %v\n", response)
    }
    
    // Asynchronous communication
    cm.AsynchronousCall("order-service", "payment-service", "/payments", map[string]interface{}{
        "amount": 100.00,
        "currency": "USD",
    }, func(response interface{}, err error) {
        if err != nil {
            fmt.Printf("Async error: %v\n", err)
        } else {
            fmt.Printf("Async response: %v\n", response)
        }
    })
    
    // Message queue communication
    cm.PublishMessage("user.created", map[string]interface{}{
        "user_id": 123,
        "email": "user@example.com",
    })
    
    cm.PublishMessage("order.created", map[string]interface{}{
        "order_id": 456,
        "user_id": 123,
        "amount": 100.00,
    })
    
    // Subscribe to messages
    cm.SubscribeToMessages("user.created", func(message *Message) {
        fmt.Printf("Handled user.created message: %v\n", message.Payload)
    })
    
    cm.SubscribeToMessages("order.created", func(message *Message) {
        fmt.Printf("Handled order.created message: %v\n", message.Payload)
    })
    
    // Keep the program running
    time.Sleep(2 * time.Second)
}
```

## Service Discovery

### Theory

Service discovery allows microservices to find and communicate with each other without hardcoded endpoints. It provides dynamic service registration and discovery capabilities.

### Service Discovery Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "net/http"
    "sync"
    "time"
)

type ServiceRegistry struct {
    Services map[string]*ServiceInfo
    mutex    sync.RWMutex
}

type ServiceInfo struct {
    ID          string
    Name        string
    Host        string
    Port        int
    Health      string
    LastSeen    time.Time
    mutex       sync.RWMutex
}

func NewServiceRegistry() *ServiceRegistry {
    return &ServiceRegistry{
        Services: make(map[string]*ServiceInfo),
    }
}

func (sr *ServiceRegistry) Register(serviceID, name, host string, port int) *ServiceInfo {
    service := &ServiceInfo{
        ID:       serviceID,
        Name:     name,
        Host:     host,
        Port:     port,
        Health:   "healthy",
        LastSeen: time.Now(),
    }
    
    sr.mutex.Lock()
    sr.Services[serviceID] = service
    sr.mutex.Unlock()
    
    fmt.Printf("Registered service: %s (%s) at %s:%d\n", serviceID, name, host, port)
    return service
}

func (sr *ServiceRegistry) Unregister(serviceID string) bool {
    sr.mutex.Lock()
    defer sr.mutex.Unlock()
    
    if _, exists := sr.Services[serviceID]; exists {
        delete(sr.Services, serviceID)
        fmt.Printf("Unregistered service: %s\n", serviceID)
        return true
    }
    
    return false
}

func (sr *ServiceRegistry) GetService(serviceID string) *ServiceInfo {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    return sr.Services[serviceID]
}

func (sr *ServiceRegistry) GetServices() map[string]*ServiceInfo {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    services := make(map[string]*ServiceInfo)
    for id, service := range sr.Services {
        services[id] = service
    }
    
    return services
}

func (sr *ServiceRegistry) UpdateHealth(serviceID, health string) {
    sr.mutex.RLock()
    service, exists := sr.Services[serviceID]
    sr.mutex.RUnlock()
    
    if exists {
        service.mutex.Lock()
        service.Health = health
        service.LastSeen = time.Now()
        service.mutex.Unlock()
        fmt.Printf("Updated health for service %s: %s\n", serviceID, health)
    }
}

func (sr *ServiceRegistry) FindServices(name string) []*ServiceInfo {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    var services []*ServiceInfo
    for _, service := range sr.Services {
        if service.Name == name {
            services = append(services, service)
        }
    }
    
    return services
}

func (sr *ServiceRegistry) HealthCheck() {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    now := time.Now()
    for _, service := range sr.Services {
        service.mutex.Lock()
        if now.Sub(service.LastSeen) > 30*time.Second {
            service.Health = "unhealthy"
            fmt.Printf("Service %s marked as unhealthy\n", service.ID)
        }
        service.mutex.Unlock()
    }
}

func (sr *ServiceRegistry) GetHealthyServices() map[string]*ServiceInfo {
    sr.mutex.RLock()
    defer sr.mutex.RUnlock()
    
    healthy := make(map[string]*ServiceInfo)
    for id, service := range sr.Services {
        service.mutex.RLock()
        if service.Health == "healthy" {
            healthy[id] = service
        }
        service.mutex.RUnlock()
    }
    
    return healthy
}

func (sr *ServiceRegistry) StartHealthCheck() {
    ticker := time.NewTicker(10 * time.Second)
    go func() {
        for range ticker.C {
            sr.HealthCheck()
        }
    }()
}

func (sr *ServiceRegistry) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    
    switch r.URL.Path {
    case "/services":
        services := sr.GetServices()
        json.NewEncoder(w).Encode(services)
    case "/healthy":
        healthy := sr.GetHealthyServices()
        json.NewEncoder(w).Encode(healthy)
    default:
        http.NotFound(w, r)
    }
}

func main() {
    registry := NewServiceRegistry()
    
    fmt.Println("Service Discovery Demo:")
    
    // Register services
    registry.Register("user-service-1", "user-service", "localhost", 8080)
    registry.Register("user-service-2", "user-service", "localhost", 8081)
    registry.Register("order-service-1", "order-service", "localhost", 8082)
    registry.Register("payment-service-1", "payment-service", "localhost", 8083)
    
    // Update health
    registry.UpdateHealth("user-service-1", "healthy")
    registry.UpdateHealth("user-service-2", "unhealthy")
    registry.UpdateHealth("order-service-1", "healthy")
    registry.UpdateHealth("payment-service-1", "healthy")
    
    // Find services by name
    userServices := registry.FindServices("user-service")
    fmt.Printf("Found %d user services\n", len(userServices))
    
    // Get healthy services
    healthy := registry.GetHealthyServices()
    fmt.Printf("Healthy services: %d\n", len(healthy))
    
    // Start health check
    registry.StartHealthCheck()
    
    // Start HTTP server
    http.HandleFunc("/", registry.ServeHTTP)
    fmt.Println("Service registry starting on :8080")
    go http.ListenAndServe(":8080", nil)
    
    // Keep the program running
    time.Sleep(5 * time.Second)
}
```

## Follow-up Questions

### 1. Service Decomposition
**Q: What are the key principles for decomposing a monolithic application into microservices?**
A: Decompose by business capability, ensure loose coupling and high cohesion, design for failure, implement proper data management, and consider team boundaries and communication patterns.

### 2. Communication Patterns
**Q: When would you use synchronous vs asynchronous communication between microservices?**
A: Use synchronous communication for real-time operations that require immediate responses. Use asynchronous communication for event-driven architectures, long-running processes, and when you need to decouple services.

### 3. Service Discovery
**Q: What are the benefits of using a service discovery mechanism?**
A: Service discovery provides dynamic service registration, automatic health checking, load balancing, fault tolerance, and eliminates the need for hardcoded service endpoints.

## Complexity Analysis

| Operation | Service Decomposition | Communication Patterns | Service Discovery |
|-----------|----------------------|----------------------|------------------|
| Add Service | O(1) | O(1) | O(1) |
| Remove Service | O(1) | O(1) | O(1) |
| Find Service | O(n) | O(1) | O(n) |
| Health Check | O(n) | N/A | O(n) |

## Applications

1. **Service Decomposition**: Legacy system modernization, scalable application architecture
2. **Communication Patterns**: Event-driven systems, real-time applications, distributed systems
3. **Service Discovery**: Microservices architecture, cloud-native applications, container orchestration
4. **Microservices Architecture**: Large-scale applications, team autonomy, technology diversity

---

**Next**: [Phase 2 Advanced](../phase2_advanced/README.md) | **Previous**: [System Design Basics](../README.md) | **Up**: [Phase 1](../README.md)
