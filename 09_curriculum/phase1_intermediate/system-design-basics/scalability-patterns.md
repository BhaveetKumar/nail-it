# Scalability Patterns

## Overview

This module covers scalability patterns including horizontal scaling, vertical scaling, load balancing, caching strategies, and database sharding. These concepts are essential for building systems that can handle growing loads.

## Table of Contents

1. [Horizontal Scaling](#horizontal-scaling/)
2. [Vertical Scaling](#vertical-scaling/)
3. [Load Balancing](#load-balancing/)
4. [Caching Strategies](#caching-strategies/)
5. [Database Sharding](#database-sharding/)
6. [Applications](#applications/)
7. [Complexity Analysis](#complexity-analysis/)
8. [Follow-up Questions](#follow-up-questions/)

## Horizontal Scaling

### Theory

Horizontal scaling involves adding more machines to handle increased load, while vertical scaling involves upgrading existing machines with more powerful hardware. Horizontal scaling is generally more cost-effective and provides better fault tolerance.

### Horizontal Scaling Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "net/http"
    "sync"
    "time"
)

type ServiceInstance struct {
    ID       string
    Host     string
    Port     int
    Status   string
    Load     int
    mutex    sync.RWMutex
}

type LoadBalancer struct {
    Instances []*ServiceInstance
    Strategy  string
    mutex     sync.RWMutex
}

func NewLoadBalancer(strategy string) *LoadBalancer {
    return &LoadBalancer{
        Instances: make([]*ServiceInstance, 0),
        Strategy:  strategy,
    }
}

func (lb *LoadBalancer) AddInstance(host string, port int) *ServiceInstance {
    instance := &ServiceInstance{
        ID:     fmt.Sprintf("%s:%d", host, port),
        Host:   host,
        Port:   port,
        Status: "healthy",
        Load:   0,
    }
    
    lb.mutex.Lock()
    lb.Instances = append(lb.Instances, instance)
    lb.mutex.Unlock()
    
    fmt.Printf("Added instance: %s\n", instance.ID)
    return instance
}

func (lb *LoadBalancer) RemoveInstance(id string) bool {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    for i, instance := range lb.Instances {
        if instance.ID == id {
            lb.Instances = append(lb.Instances[:i], lb.Instances[i+1:]...)
            fmt.Printf("Removed instance: %s\n", id)
            return true
        }
    }
    
    return false
}

func (lb *LoadBalancer) GetInstance() *ServiceInstance {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    if len(lb.Instances) == 0 {
        return nil
    }
    
    switch lb.Strategy {
    case "round_robin":
        return lb.getRoundRobinInstance()
    case "least_connections":
        return lb.getLeastConnectionsInstance()
    case "weighted_round_robin":
        return lb.getWeightedRoundRobinInstance()
    default:
        return lb.Instances[0]
    }
}

func (lb *LoadBalancer) getRoundRobinInstance() *ServiceInstance {
    // Simple round-robin implementation
    if len(lb.Instances) == 0 {
        return nil
    }
    
    // In a real implementation, you'd use an atomic counter
    return lb.Instances[0]
}

func (lb *LoadBalancer) getLeastConnectionsInstance() *ServiceInstance {
    if len(lb.Instances) == 0 {
        return nil
    }
    
    best := lb.Instances[0]
    for _, instance := range lb.Instances[1:] {
        if instance.Load < best.Load {
            best = instance
        }
    }
    
    return best
}

func (lb *LoadBalancer) getWeightedRoundRobinInstance() *ServiceInstance {
    // Weighted round-robin implementation
    if len(lb.Instances) == 0 {
        return nil
    }
    
    // In a real implementation, you'd calculate weights based on instance capacity
    return lb.Instances[0]
}

func (lb *LoadBalancer) UpdateLoad(instanceID string, load int) {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    for _, instance := range lb.Instances {
        if instance.ID == instanceID {
            instance.mutex.Lock()
            instance.Load = load
            instance.mutex.Unlock()
            break
        }
    }
}

func (lb *LoadBalancer) HealthCheck() {
    for _, instance := range lb.Instances {
        go func(instance *ServiceInstance) {
            // Simulate health check
            time.Sleep(100 * time.Millisecond)
            
            // In a real implementation, you'd make an HTTP request
            instance.mutex.Lock()
            if instance.Load > 100 {
                instance.Status = "unhealthy"
            } else {
                instance.Status = "healthy"
            }
            instance.mutex.Unlock()
        }(instance)
    }
}

func (lb *LoadBalancer) GetStatus() {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    fmt.Println("Load Balancer Status:")
    for _, instance := range lb.Instances {
        instance.mutex.RLock()
        fmt.Printf("  %s: %s (load: %d)\n", instance.ID, instance.Status, instance.Load)
        instance.mutex.RUnlock()
    }
}

func main() {
    lb := NewLoadBalancer("round_robin")
    
    fmt.Println("Horizontal Scaling Demo:")
    
    // Add instances
    lb.AddInstance("server1", 8080)
    lb.AddInstance("server2", 8080)
    lb.AddInstance("server3", 8080)
    
    // Simulate load
    lb.UpdateLoad("server1:8080", 50)
    lb.UpdateLoad("server2:8080", 30)
    lb.UpdateLoad("server3:8080", 70)
    
    // Get instance
    instance := lb.GetInstance()
    if instance != nil {
        fmt.Printf("Selected instance: %s\n", instance.ID)
    }
    
    // Health check
    lb.HealthCheck()
    time.Sleep(200 * time.Millisecond)
    
    // Show status
    lb.GetStatus()
    
    // Remove an instance
    lb.RemoveInstance("server3:8080")
    
    // Show updated status
    lb.GetStatus()
}
```

## Vertical Scaling

### Theory

Vertical scaling involves upgrading existing hardware to handle increased load. While simpler than horizontal scaling, it has limitations and can become expensive.

### Vertical Scaling Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

type SystemResources struct {
    CPU    float64
    Memory float64
    Disk   float64
    Network float64
}

type VerticalScaler struct {
    CurrentResources SystemResources
    MaxResources     SystemResources
    mutex           sync.RWMutex
}

func NewVerticalScaler() *VerticalScaler {
    return &VerticalScaler{
        CurrentResources: SystemResources{
            CPU:     0.0,
            Memory:  0.0,
            Disk:    0.0,
            Network: 0.0,
        },
        MaxResources: SystemResources{
            CPU:     100.0,
            Memory:  100.0,
            Disk:    100.0,
            Network: 100.0,
        },
    }
}

func (vs *VerticalScaler) GetCurrentResources() SystemResources {
    vs.mutex.RLock()
    defer vs.mutex.RUnlock()
    
    return vs.CurrentResources
}

func (vs *VerticalScaler) UpdateResources(cpu, memory, disk, network float64) {
    vs.mutex.Lock()
    defer vs.mutex.Unlock()
    
    vs.CurrentResources.CPU = cpu
    vs.CurrentResources.Memory = memory
    vs.CurrentResources.Disk = disk
    vs.CurrentResources.Network = network
}

func (vs *VerticalScaler) CheckScalingNeed() bool {
    vs.mutex.RLock()
    defer vs.mutex.RUnlock()
    
    // Check if any resource is above 80% utilization
    return vs.CurrentResources.CPU > 80.0 ||
           vs.CurrentResources.Memory > 80.0 ||
           vs.CurrentResources.Disk > 80.0 ||
           vs.CurrentResources.Network > 80.0
}

func (vs *VerticalScaler) ScaleUp() bool {
    vs.mutex.Lock()
    defer vs.mutex.Unlock()
    
    // In a real implementation, this would trigger hardware upgrade
    fmt.Println("Scaling up system resources...")
    
    // Simulate resource increase
    vs.MaxResources.CPU += 20.0
    vs.MaxResources.Memory += 20.0
    vs.MaxResources.Disk += 20.0
    vs.MaxResources.Network += 20.0
    
    fmt.Printf("New max resources: CPU=%.1f%%, Memory=%.1f%%, Disk=%.1f%%, Network=%.1f%%\n",
               vs.MaxResources.CPU, vs.MaxResources.Memory, vs.MaxResources.Disk, vs.MaxResources.Network)
    
    return true
}

func (vs *VerticalScaler) ScaleDown() bool {
    vs.mutex.Lock()
    defer vs.mutex.Unlock()
    
    // In a real implementation, this would trigger hardware downgrade
    fmt.Println("Scaling down system resources...")
    
    // Simulate resource decrease
    if vs.MaxResources.CPU > 20.0 {
        vs.MaxResources.CPU -= 20.0
    }
    if vs.MaxResources.Memory > 20.0 {
        vs.MaxResources.Memory -= 20.0
    }
    if vs.MaxResources.Disk > 20.0 {
        vs.MaxResources.Disk -= 20.0
    }
    if vs.MaxResources.Network > 20.0 {
        vs.MaxResources.Network -= 20.0
    }
    
    fmt.Printf("New max resources: CPU=%.1f%%, Memory=%.1f%%, Disk=%.1f%%, Network=%.1f%%\n",
               vs.MaxResources.CPU, vs.MaxResources.Memory, vs.MaxResources.Disk, vs.MaxResources.Network)
    
    return true
}

func (vs *VerticalScaler) GetResourceUtilization() map[string]float64 {
    vs.mutex.RLock()
    defer vs.mutex.RUnlock()
    
    return map[string]float64{
        "cpu":     vs.CurrentResources.CPU,
        "memory":  vs.CurrentResources.Memory,
        "disk":    vs.CurrentResources.Disk,
        "network": vs.CurrentResources.Network,
    }
}

func (vs *VerticalScaler) MonitorResources() {
    ticker := time.NewTicker(1 * time.Second)
    go func() {
        for range ticker.C {
            // Simulate resource monitoring
            cpu := float64(runtime.NumCPU()) * 10.0
            memory := float64(runtime.MemStats{}.Alloc) / 1024 / 1024 // MB
            
            vs.UpdateResources(cpu, memory, 50.0, 30.0)
            
            if vs.CheckScalingNeed() {
                fmt.Println("High resource utilization detected!")
                vs.ScaleUp()
            }
        }
    }()
}

func main() {
    scaler := NewVerticalScaler()
    
    fmt.Println("Vertical Scaling Demo:")
    
    // Start monitoring
    scaler.MonitorResources()
    
    // Simulate some load
    for i := 0; i < 5; i++ {
        time.Sleep(2 * time.Second)
        
        resources := scaler.GetResourceUtilization()
        fmt.Printf("Resource utilization: CPU=%.1f%%, Memory=%.1f%%, Disk=%.1f%%, Network=%.1f%%\n",
                   resources["cpu"], resources["memory"], resources["disk"], resources["network"])
        
        if scaler.CheckScalingNeed() {
            fmt.Println("Scaling needed!")
        }
    }
    
    // Keep the program running
    time.Sleep(5 * time.Second)
}
```

## Load Balancing

### Theory

Load balancing distributes incoming requests across multiple servers to ensure no single server is overwhelmed. It improves availability, reliability, and performance.

### Load Balancing Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "net/http"
    "sync"
    "time"
)

type LoadBalancer struct {
    Servers    []*Server
    Strategy   string
    mutex      sync.RWMutex
    current    int
}

type Server struct {
    ID       string
    URL      string
    Weight   int
    Active   bool
    Requests int
    mutex    sync.RWMutex
}

func NewLoadBalancer(strategy string) *LoadBalancer {
    return &LoadBalancer{
        Servers:  make([]*Server, 0),
        Strategy: strategy,
        current:  0,
    }
}

func (lb *LoadBalancer) AddServer(id, url string, weight int) *Server {
    server := &Server{
        ID:       id,
        URL:      url,
        Weight:   weight,
        Active:   true,
        Requests: 0,
    }
    
    lb.mutex.Lock()
    lb.Servers = append(lb.Servers, server)
    lb.mutex.Unlock()
    
    fmt.Printf("Added server: %s (%s) with weight %d\n", id, url, weight)
    return server
}

func (lb *LoadBalancer) RemoveServer(id string) bool {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    for i, server := range lb.Servers {
        if server.ID == id {
            lb.Servers = append(lb.Servers[:i], lb.Servers[i+1:]...)
            fmt.Printf("Removed server: %s\n", id)
            return true
        }
    }
    
    return false
}

func (lb *LoadBalancer) GetServer() *Server {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    if len(lb.Servers) == 0 {
        return nil
    }
    
    switch lb.Strategy {
    case "round_robin":
        return lb.getRoundRobinServer()
    case "least_connections":
        return lb.getLeastConnectionsServer()
    case "weighted_round_robin":
        return lb.getWeightedRoundRobinServer()
    default:
        return lb.Servers[0]
    }
}

func (lb *LoadBalancer) getRoundRobinServer() *Server {
    if len(lb.Servers) == 0 {
        return nil
    }
    
    // Find next active server
    for i := 0; i < len(lb.Servers); i++ {
        server := lb.Servers[lb.current]
        lb.current = (lb.current + 1) % len(lb.Servers)
        
        if server.Active {
            return server
        }
    }
    
    return nil
}

func (lb *LoadBalancer) getLeastConnectionsServer() *Server {
    if len(lb.Servers) == 0 {
        return nil
    }
    
    best := lb.Servers[0]
    for _, server := range lb.Servers[1:] {
        if server.Active && server.Requests < best.Requests {
            best = server
        }
    }
    
    return best
}

func (lb *LoadBalancer) getWeightedRoundRobinServer() *Server {
    if len(lb.Servers) == 0 {
        return nil
    }
    
    // Simple weighted round-robin implementation
    totalWeight := 0
    for _, server := range lb.Servers {
        if server.Active {
            totalWeight += server.Weight
        }
    }
    
    if totalWeight == 0 {
        return nil
    }
    
    // Find server based on weight
    currentWeight := 0
    for _, server := range lb.Servers {
        if server.Active {
            currentWeight += server.Weight
            if lb.current < currentWeight {
                lb.current = (lb.current + 1) % totalWeight
                return server
            }
        }
    }
    
    return lb.Servers[0]
}

func (lb *LoadBalancer) HandleRequest(w http.ResponseWriter, r *http.Request) {
    server := lb.GetServer()
    if server == nil {
        http.Error(w, "No servers available", http.StatusServiceUnavailable)
        return
    }
    
    // Increment request count
    server.mutex.Lock()
    server.Requests++
    server.mutex.Unlock()
    
    // Forward request to server
    fmt.Printf("Forwarding request to server: %s\n", server.ID)
    
    // In a real implementation, you'd forward the actual request
    w.Header().Set("Content-Type", "application/json")
    fmt.Fprintf(w, `{"server": "%s", "requests": %d}`, server.ID, server.Requests)
}

func (lb *LoadBalancer) HealthCheck() {
    for _, server := range lb.Servers {
        go func(server *Server) {
            // Simulate health check
            time.Sleep(100 * time.Millisecond)
            
            // In a real implementation, you'd make an HTTP request
            server.mutex.Lock()
            if server.Requests > 100 {
                server.Active = false
                fmt.Printf("Server %s marked as inactive\n", server.ID)
            } else {
                server.Active = true
            }
            server.mutex.Unlock()
        }(server)
    }
}

func (lb *LoadBalancer) GetStatus() {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    fmt.Println("Load Balancer Status:")
    for _, server := range lb.Servers {
        server.mutex.RLock()
        status := "inactive"
        if server.Active {
            status = "active"
        }
        fmt.Printf("  %s: %s (requests: %d, weight: %d)\n", 
                   server.ID, status, server.Requests, server.Weight)
        server.mutex.RUnlock()
    }
}

func main() {
    lb := NewLoadBalancer("round_robin")
    
    fmt.Println("Load Balancing Demo:")
    
    // Add servers
    lb.AddServer("server1", "http://server1:8080", 1)
    lb.AddServer("server2", "http://server2:8080", 1)
    lb.AddServer("server3", "http://server3:8080", 2)
    
    // Simulate some requests
    for i := 0; i < 10; i++ {
        server := lb.GetServer()
        if server != nil {
            fmt.Printf("Request %d: %s\n", i+1, server.ID)
        }
    }
    
    // Health check
    lb.HealthCheck()
    time.Sleep(200 * time.Millisecond)
    
    // Show status
    lb.GetStatus()
    
    // Remove a server
    lb.RemoveServer("server3")
    
    // Show updated status
    lb.GetStatus()
}
```

## Caching Strategies

### Theory

Caching improves performance by storing frequently accessed data in faster storage. Common caching strategies include write-through, write-behind, and cache-aside patterns.

### Caching Implementation

#### Golang Implementation

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
    CreatedAt time.Time
}

type Cache struct {
    Items    map[string]*CacheItem
    mutex    sync.RWMutex
    MaxSize  int
    TTL      time.Duration
}

func NewCache(maxSize int, ttl time.Duration) *Cache {
    cache := &Cache{
        Items:   make(map[string]*CacheItem),
        MaxSize: maxSize,
        TTL:     ttl,
    }
    
    // Start cleanup goroutine
    go cache.cleanup()
    
    return cache
}

func (c *Cache) Set(key string, value interface{}) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    // Check if cache is full
    if len(c.Items) >= c.MaxSize {
        c.evictOldest()
    }
    
    item := &CacheItem{
        Value:     value,
        ExpiresAt: time.Now().Add(c.TTL),
        CreatedAt: time.Now(),
    }
    
    c.Items[key] = item
    fmt.Printf("Cached key: %s\n", key)
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mutex.RLock()
    item, exists := c.Items[key]
    c.mutex.RUnlock()
    
    if !exists {
        return nil, false
    }
    
    // Check if item has expired
    if time.Now().After(item.ExpiresAt) {
        c.Delete(key)
        return nil, false
    }
    
    return item.Value, true
}

func (c *Cache) Delete(key string) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    delete(c.Items, key)
    fmt.Printf("Deleted key: %s\n", key)
}

func (c *Cache) evictOldest() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, item := range c.Items {
        if oldestKey == "" || item.CreatedAt.Before(oldestTime) {
            oldestKey = key
            oldestTime = item.CreatedAt
        }
    }
    
    if oldestKey != "" {
        delete(c.Items, oldestKey)
        fmt.Printf("Evicted oldest key: %s\n", oldestKey)
    }
}

func (c *Cache) cleanup() {
    ticker := time.NewTicker(1 * time.Minute)
    for range ticker.C {
        c.mutex.Lock()
        now := time.Now()
        for key, item := range c.Items {
            if now.After(item.ExpiresAt) {
                delete(c.Items, key)
                fmt.Printf("Cleaned up expired key: %s\n", key)
            }
        }
        c.mutex.Unlock()
    }
}

func (c *Cache) GetStats() map[string]interface{} {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    
    return map[string]interface{}{
        "size":      len(c.Items),
        "max_size":  c.MaxSize,
        "ttl":       c.TTL.String(),
    }
}

func main() {
    cache := NewCache(3, 5*time.Second)
    
    fmt.Println("Caching Strategies Demo:")
    
    // Set some values
    cache.Set("key1", "value1")
    cache.Set("key2", "value2")
    cache.Set("key3", "value3")
    
    // Get values
    if value, exists := cache.Get("key1"); exists {
        fmt.Printf("Retrieved key1: %v\n", value)
    }
    
    if value, exists := cache.Get("key2"); exists {
        fmt.Printf("Retrieved key2: %v\n", value)
    }
    
    // Try to get non-existent key
    if value, exists := cache.Get("key4"); exists {
        fmt.Printf("Retrieved key4: %v\n", value)
    } else {
        fmt.Println("Key4 not found")
    }
    
    // Add more items to trigger eviction
    cache.Set("key4", "value4")
    cache.Set("key5", "value5")
    
    // Show stats
    stats := cache.GetStats()
    fmt.Printf("Cache stats: %v\n", stats)
    
    // Wait for expiration
    fmt.Println("Waiting for expiration...")
    time.Sleep(6 * time.Second)
    
    // Try to get expired key
    if value, exists := cache.Get("key1"); exists {
        fmt.Printf("Retrieved key1 after expiration: %v\n", value)
    } else {
        fmt.Println("Key1 expired and was cleaned up")
    }
}
```

## Follow-up Questions

### 1. Horizontal Scaling
**Q: What are the advantages and disadvantages of horizontal scaling?**
A: Advantages: Better fault tolerance, cost-effective, can handle more load. Disadvantages: More complex to implement, requires load balancing, potential data consistency issues.

### 2. Vertical Scaling
**Q: When would you choose vertical scaling over horizontal scaling?**
A: Choose vertical scaling when you have a single-threaded application, limited budget for infrastructure changes, or when horizontal scaling is not feasible due to application constraints.

### 3. Load Balancing
**Q: What are the different load balancing algorithms and when would you use each?**
A: Round-robin for equal capacity servers, least connections for varying request processing times, weighted round-robin for servers with different capacities, and IP hash for session affinity.

## Complexity Analysis

| Operation | Horizontal Scaling | Vertical Scaling | Load Balancing | Caching |
|-----------|-------------------|------------------|----------------|---------|
| Add Instance | O(1) | N/A | O(1) | N/A |
| Remove Instance | O(n) | N/A | O(n) | N/A |
| Get Instance | O(1) | N/A | O(1) | N/A |
| Set Cache | N/A | N/A | N/A | O(1) |
| Get Cache | N/A | N/A | N/A | O(1) |

## Applications

1. **Horizontal Scaling**: Web applications, microservices, distributed systems
2. **Vertical Scaling**: Single-threaded applications, legacy systems
3. **Load Balancing**: High-traffic websites, API gateways, microservices
4. **Caching**: Database queries, API responses, static content

---

**Next**: [Microservices Architecture](microservices-architecture.md/) | **Previous**: [System Design Basics](README.md/) | **Up**: [System Design Basics](README.md/)
