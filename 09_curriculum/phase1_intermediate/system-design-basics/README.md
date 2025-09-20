# System Design Basics

## Table of Contents

1. [Overview](#overview/)
2. [Scalability Patterns](#scalability-patterns/)
3. [Load Balancing](#load-balancing/)
4. [Caching Strategies](#caching-strategies/)
5. [Database Scaling](#database-scaling/)
6. [Microservices Architecture](#microservices-architecture/)
7. [Message Queues](#message-queues/)
8. [Implementations](#implementations/)
9. [Follow-up Questions](#follow-up-questions/)
10. [Sources](#sources/)
11. [Projects](#projects/)

## Overview

### Learning Objectives

- Understand scalability patterns and trade-offs
- Implement load balancing strategies
- Design effective caching systems
- Scale databases horizontally and vertically
- Design microservices architectures
- Implement message queue systems

### What is System Design?

System Design involves creating scalable, reliable, and maintainable software systems that can handle growing user demands and data volumes while maintaining performance and availability.

## Scalability Patterns

### 1. Horizontal vs Vertical Scaling

#### Vertical Scaling (Scale Up)
```go
// Vertical scaling example - increasing server resources
type ServerConfig struct {
    CPU    int    `json:"cpu_cores"`
    Memory int    `json:"memory_gb"`
    Storage int   `json:"storage_gb"`
    Network string `json:"network_speed"`
}

func (s *ServerConfig) UpgradeResources() *ServerConfig {
    return &ServerConfig{
        CPU:     s.CPU * 2,        // Double CPU cores
        Memory:  s.Memory * 2,     // Double memory
        Storage: s.Storage * 2,    // Double storage
        Network: "10Gbps",         // Upgrade network
    }
}

// Pros: Simple, no code changes needed
// Cons: Limited by hardware, single point of failure
```

#### Horizontal Scaling (Scale Out)
```go
// Horizontal scaling example - adding more servers
type LoadBalancer struct {
    servers []Server
    strategy LoadBalancingStrategy
}

type Server struct {
    ID       string
    Address  string
    Health   bool
    Load     float64
    Capacity int
}

func (lb *LoadBalancer) AddServer(server Server) {
    lb.servers = append(lb.servers, server)
    fmt.Printf("Added server %s to load balancer\n", server.ID)
}

func (lb *LoadBalancer) RemoveServer(serverID string) {
    for i, server := range lb.servers {
        if server.ID == serverID {
            lb.servers = append(lb.servers[:i], lb.servers[i+1:]...)
            fmt.Printf("Removed server %s from load balancer\n", serverID)
            break
        }
    }
}

// Pros: Unlimited scaling, fault tolerance
// Cons: More complex, requires load balancing
```

### 2. Database Sharding

#### Sharding Implementation
```go
package main

import (
    "crypto/md5"
    "fmt"
    "strconv"
)

type ShardManager struct {
    shards []Shard
    shardCount int
}

type Shard struct {
    ID       int
    Database string
    Host     string
    Port     int
}

type User struct {
    ID    string
    Name  string
    Email string
    ShardID int
}

func NewShardManager(shardCount int) *ShardManager {
    shards := make([]Shard, shardCount)
    for i := 0; i < shardCount; i++ {
        shards[i] = Shard{
            ID:       i,
            Database: fmt.Sprintf("shard_%d", i),
            Host:     fmt.Sprintf("shard%d.example.com", i),
            Port:     5432,
        }
    }
    
    return &ShardManager{
        shards:     shards,
        shardCount: shardCount,
    }
}

func (sm *ShardManager) GetShard(key string) *Shard {
    // Use consistent hashing to determine shard
    hash := md5.Sum([]byte(key))
    shardIndex := int(hash[0]) % sm.shardCount
    return &sm.shards[shardIndex]
}

func (sm *ShardManager) CreateUser(user User) error {
    shard := sm.GetShard(user.ID)
    user.ShardID = shard.ID
    
    fmt.Printf("Creating user %s in shard %d (%s)\n", 
        user.ID, shard.ID, shard.Database)
    
    // In real implementation, this would connect to the shard database
    return nil
}

func (sm *ShardManager) GetUser(userID string) (*User, error) {
    shard := sm.GetShard(userID)
    
    fmt.Printf("Fetching user %s from shard %d (%s)\n", 
        userID, shard.ID, shard.Database)
    
    // In real implementation, this would query the shard database
    return &User{ID: userID, ShardID: shard.ID}, nil
}

func main() {
    shardManager := NewShardManager(4)
    
    // Create users - they'll be distributed across shards
    users := []User{
        {ID: "user1", Name: "Alice", Email: "alice@example.com"},
        {ID: "user2", Name: "Bob", Email: "bob@example.com"},
        {ID: "user3", Name: "Charlie", Email: "charlie@example.com"},
        {ID: "user4", Name: "Diana", Email: "diana@example.com"},
    }
    
    for _, user := range users {
        shardManager.CreateUser(user)
    }
    
    // Fetch users
    for _, user := range users {
        shardManager.GetUser(user.ID)
    }
}
```

## Load Balancing

### 1. Load Balancing Algorithms

#### Round Robin Load Balancer
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type LoadBalancer struct {
    servers []Server
    current int
    mutex   sync.Mutex
}

type Server struct {
    ID       string
    Address  string
    Health   bool
    ResponseTime time.Duration
}

type Request struct {
    ID   string
    Data string
}

func NewLoadBalancer() *LoadBalancer {
    return &LoadBalancer{
        servers: make([]Server, 0),
        current: 0,
    }
}

func (lb *LoadBalancer) AddServer(server Server) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    lb.servers = append(lb.servers, server)
    fmt.Printf("Added server %s to load balancer\n", server.ID)
}

func (lb *LoadBalancer) GetNextServer() *Server {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    if len(lb.servers) == 0 {
        return nil
    }
    
    // Find next healthy server
    attempts := 0
    for attempts < len(lb.servers) {
        server := &lb.servers[lb.current]
        lb.current = (lb.current + 1) % len(lb.servers)
        
        if server.Health {
            return server
        }
        attempts++
    }
    
    return nil // No healthy servers
}

func (lb *LoadBalancer) ProcessRequest(req Request) {
    server := lb.GetNextServer()
    if server == nil {
        fmt.Printf("Request %s failed: No healthy servers available\n", req.ID)
        return
    }
    
    fmt.Printf("Processing request %s on server %s\n", req.ID, server.ID)
    
    // Simulate processing time
    time.Sleep(server.ResponseTime)
    fmt.Printf("Request %s completed on server %s\n", req.ID, server.ID)
}

func main() {
    lb := NewLoadBalancer()
    
    // Add servers with different response times
    lb.AddServer(Server{ID: "server1", Address: "192.168.1.1", Health: true, ResponseTime: 100 * time.Millisecond})
    lb.AddServer(Server{ID: "server2", Address: "192.168.1.2", Health: true, ResponseTime: 150 * time.Millisecond})
    lb.AddServer(Server{ID: "server3", Address: "192.168.1.3", Health: true, ResponseTime: 200 * time.Millisecond})
    
    // Process some requests
    for i := 0; i < 10; i++ {
        req := Request{
            ID:   fmt.Sprintf("req-%d", i),
            Data: fmt.Sprintf("data-%d", i),
        }
        lb.ProcessRequest(req)
    }
}
```

#### Weighted Round Robin
```go
type WeightedServer struct {
    Server
    Weight int
    CurrentWeight int
}

type WeightedLoadBalancer struct {
    servers []WeightedServer
    mutex   sync.Mutex
}

func NewWeightedLoadBalancer() *WeightedLoadBalancer {
    return &WeightedLoadBalancer{
        servers: make([]WeightedServer, 0),
    }
}

func (wlb *WeightedLoadBalancer) AddServer(server Server, weight int) {
    wlb.mutex.Lock()
    defer wlb.mutex.Unlock()
    
    weightedServer := WeightedServer{
        Server: server,
        Weight: weight,
        CurrentWeight: 0,
    }
    
    wlb.servers = append(wlb.servers, weightedServer)
    fmt.Printf("Added server %s with weight %d\n", server.ID, weight)
}

func (wlb *WeightedLoadBalancer) GetNextServer() *Server {
    wlb.mutex.Lock()
    defer wlb.mutex.Unlock()
    
    if len(wlb.servers) == 0 {
        return nil
    }
    
    var bestServer *WeightedServer
    totalWeight := 0
    
    // Find server with highest current weight
    for i := range wlb.servers {
        if !wlb.servers[i].Health {
            continue
        }
        
        wlb.servers[i].CurrentWeight += wlb.servers[i].Weight
        totalWeight += wlb.servers[i].Weight
        
        if bestServer == nil || wlb.servers[i].CurrentWeight > bestServer.CurrentWeight {
            bestServer = &wlb.servers[i]
        }
    }
    
    if bestServer != nil {
        bestServer.CurrentWeight -= totalWeight
        return &bestServer.Server
    }
    
    return nil
}
```

## Caching Strategies

### 1. Cache-Aside Pattern

#### Redis Cache Implementation
```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "time"

    "github.com/go-redis/redis/v8"
)

type CacheService struct {
    redis  *redis.Client
    db     Database
    ttl    time.Duration
}

type Database interface {
    GetUser(id string) (*User, error)
    SetUser(user *User) error
}

type User struct {
    ID    string `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

func NewCacheService(redisClient *redis.Client, db Database, ttl time.Duration) *CacheService {
    return &CacheService{
        redis: redisClient,
        db:    db,
        ttl:   ttl,
    }
}

func (cs *CacheService) GetUser(ctx context.Context, id string) (*User, error) {
    // Try to get from cache first
    cacheKey := fmt.Sprintf("user:%s", id)
    cachedData, err := cs.redis.Get(ctx, cacheKey).Result()
    
    if err == nil {
        // Cache hit
        var user User
        if err := json.Unmarshal([]byte(cachedData), &user); err == nil {
            fmt.Printf("Cache hit for user %s\n", id)
            return &user, nil
        }
    }
    
    // Cache miss - get from database
    fmt.Printf("Cache miss for user %s, fetching from database\n", id)
    user, err := cs.db.GetUser(id)
    if err != nil {
        return nil, err
    }
    
    // Store in cache for next time
    userData, err := json.Marshal(user)
    if err == nil {
        cs.redis.Set(ctx, cacheKey, userData, cs.ttl)
        fmt.Printf("Stored user %s in cache\n", id)
    }
    
    return user, nil
}

func (cs *CacheService) SetUser(ctx context.Context, user *User) error {
    // Update database
    if err := cs.db.SetUser(user); err != nil {
        return err
    }
    
    // Update cache
    cacheKey := fmt.Sprintf("user:%s", user.ID)
    userData, err := json.Marshal(user)
    if err != nil {
        return err
    }
    
    return cs.redis.Set(ctx, cacheKey, userData, cs.ttl).Err()
}

func (cs *CacheService) InvalidateUser(ctx context.Context, id string) error {
    cacheKey := fmt.Sprintf("user:%s", id)
    return cs.redis.Del(ctx, cacheKey).Err()
}
```

### 2. Write-Through Cache

#### Write-Through Implementation
```go
type WriteThroughCache struct {
    cache Cache
    db    Database
}

func (wtc *WriteThroughCache) Write(key string, value interface{}) error {
    // Write to database first
    if err := wtc.db.Write(key, value); err != nil {
        return err
    }
    
    // Then write to cache
    return wtc.cache.Set(key, value)
}

func (wtc *WriteThroughCache) Read(key string) (interface{}, error) {
    // Try cache first
    if value, err := wtc.cache.Get(key); err == nil {
        return value, nil
    }
    
    // Cache miss - get from database
    value, err := wtc.db.Read(key)
    if err != nil {
        return nil, err
    }
    
    // Store in cache
    wtc.cache.Set(key, value)
    return value, nil
}
```

## Database Scaling

### 1. Read Replicas

#### Read Replica Implementation
```go
type DatabaseCluster struct {
    master   *sql.DB
    replicas []*sql.DB
    currentReplica int
    mutex    sync.RWMutex
}

func NewDatabaseCluster(master *sql.DB, replicas []*sql.DB) *DatabaseCluster {
    return &DatabaseCluster{
        master:   master,
        replicas: replicas,
        currentReplica: 0,
    }
}

func (dc *DatabaseCluster) Write(query string, args ...interface{}) error {
    // All writes go to master
    _, err := dc.master.Exec(query, args...)
    return err
}

func (dc *DatabaseCluster) Read(query string, args ...interface{}) (*sql.Rows, error) {
    dc.mutex.RLock()
    defer dc.mutex.RUnlock()
    
    if len(dc.replicas) == 0 {
        // No replicas, use master for reads
        return dc.master.Query(query, args...)
    }
    
    // Use round-robin for read replicas
    replica := dc.replicas[dc.currentReplica]
    dc.currentReplica = (dc.currentReplica + 1) % len(dc.replicas)
    
    return replica.Query(query, args...)
}

func (dc *DatabaseCluster) AddReplica(replica *sql.DB) {
    dc.mutex.Lock()
    defer dc.mutex.Unlock()
    
    dc.replicas = append(dc.replicas, replica)
    fmt.Printf("Added read replica, total replicas: %d\n", len(dc.replicas))
}
```

### 2. Database Partitioning

#### Range Partitioning
```go
type PartitionedTable struct {
    partitions map[string]*sql.DB
    partitionKey string
}

func NewPartitionedTable(partitionKey string) *PartitionedTable {
    return &PartitionedTable{
        partitions: make(map[string]*sql.DB),
        partitionKey: partitionKey,
    }
}

func (pt *PartitionedTable) GetPartition(key string) string {
    // Simple hash-based partitioning
    hash := md5.Sum([]byte(key))
    partitionIndex := int(hash[0]) % 4 // 4 partitions
    return fmt.Sprintf("partition_%d", partitionIndex)
}

func (pt *PartitionedTable) Insert(table string, data map[string]interface{}) error {
    partition := pt.GetPartition(data[pt.partitionKey].(string))
    
    db, exists := pt.partitions[partition]
    if !exists {
        return fmt.Errorf("partition %s not found", partition)
    }
    
    // Build INSERT query for the specific partition
    query := fmt.Sprintf("INSERT INTO %s_%s", table, partition)
    // ... build query with data
    
    _, err := db.Exec(query)
    return err
}
```

## Microservices Architecture

### 1. Service Discovery

#### Consul Service Discovery
```go
package main

import (
    "fmt"
    "log"
    "time"

    "github.com/hashicorp/consul/api"
)

type ServiceRegistry struct {
    client *api.Client
    serviceID string
    serviceName string
    port int
}

func NewServiceRegistry(consulAddr, serviceName string, port int) (*ServiceRegistry, error) {
    config := api.DefaultConfig()
    config.Address = consulAddr
    
    client, err := api.NewClient(config)
    if err != nil {
        return nil, err
    }
    
    return &ServiceRegistry{
        client:      client,
        serviceName: serviceName,
        port:        port,
    }, nil
}

func (sr *ServiceRegistry) Register(serviceID string) error {
    sr.serviceID = serviceID
    
    registration := &api.AgentServiceRegistration{
        ID:      serviceID,
        Name:    sr.serviceName,
        Port:    sr.port,
        Address: "localhost",
        Check: &api.AgentServiceCheck{
            HTTP:                           fmt.Sprintf("http://localhost:%d/health", sr.port),
            Interval:                       "10s",
            Timeout:                        "3s",
            DeregisterCriticalServiceAfter: "30s",
        },
    }
    
    return sr.client.Agent().ServiceRegister(registration)
}

func (sr *ServiceRegistry) Deregister() error {
    return sr.client.Agent().ServiceDeregister(sr.serviceID)
}

func (sr *ServiceRegistry) DiscoverServices(serviceName string) ([]*api.ServiceEntry, error) {
    services, _, err := sr.client.Health().Service(serviceName, "", true, nil)
    return services, err
}

func (sr *ServiceRegistry) GetHealthyService(serviceName string) (*api.ServiceEntry, error) {
    services, err := sr.DiscoverServices(serviceName)
    if err != nil {
        return nil, err
    }
    
    if len(services) == 0 {
        return nil, fmt.Errorf("no healthy services found for %s", serviceName)
    }
    
    // Return first healthy service
    return services[0], nil
}
```

### 2. API Gateway

#### Simple API Gateway
```go
type APIGateway struct {
    services map[string]ServiceConfig
    loadBalancer LoadBalancer
}

type ServiceConfig struct {
    Name     string
    Endpoints []string
    HealthCheck string
}

func (gw *APIGateway) RegisterService(config ServiceConfig) {
    gw.services[config.Name] = config
    fmt.Printf("Registered service: %s\n", config.Name)
}

func (gw *APIGateway) RouteRequest(serviceName, path string) (string, error) {
    service, exists := gw.services[serviceName]
    if !exists {
        return "", fmt.Errorf("service %s not found", serviceName)
    }
    
    // Use load balancer to select endpoint
    endpoint := gw.loadBalancer.GetNextServer()
    if endpoint == nil {
        return "", fmt.Errorf("no healthy endpoints for service %s", serviceName)
    }
    
    return fmt.Sprintf("http://%s%s", endpoint.Address, path), nil
}
```

## Message Queues

### 1. RabbitMQ Implementation

#### Producer
```go
package main

import (
    "fmt"
    "log"
    "time"

    "github.com/streadway/amqp"
)

type MessageProducer struct {
    conn    *amqp.Connection
    channel *amqp.Channel
    queue   amqp.Queue
}

func NewMessageProducer(amqpURL, queueName string) (*MessageProducer, error) {
    conn, err := amqp.Dial(amqpURL)
    if err != nil {
        return nil, err
    }
    
    ch, err := conn.Channel()
    if err != nil {
        return nil, err
    }
    
    q, err := ch.QueueDeclare(
        queueName, // name
        true,      // durable
        false,     // delete when unused
        false,     // exclusive
        false,     // no-wait
        nil,       // arguments
    )
    if err != nil {
        return nil, err
    }
    
    return &MessageProducer{
        conn:    conn,
        channel: ch,
        queue:   q,
    }, nil
}

func (mp *MessageProducer) PublishMessage(message string) error {
    err := mp.channel.Publish(
        "",           // exchange
        mp.queue.Name, // routing key
        false,        // mandatory
        false,        // immediate
        amqp.Publishing{
            ContentType: "text/plain",
            Body:        []byte(message),
            Timestamp:   time.Now(),
        },
    )
    
    if err != nil {
        return err
    }
    
    fmt.Printf("Published message: %s\n", message)
    return nil
}

func (mp *MessageProducer) Close() {
    mp.channel.Close()
    mp.conn.Close()
}
```

#### Consumer
```go
type MessageConsumer struct {
    conn    *amqp.Connection
    channel *amqp.Channel
    queue   amqp.Queue
}

func NewMessageConsumer(amqpURL, queueName string) (*MessageConsumer, error) {
    conn, err := amqp.Dial(amqpURL)
    if err != nil {
        return nil, err
    }
    
    ch, err := conn.Channel()
    if err != nil {
        return nil, err
    }
    
    q, err := ch.QueueDeclare(
        queueName, // name
        true,      // durable
        false,     // delete when unused
        false,     // exclusive
        false,     // no-wait
        nil,       // arguments
    )
    if err != nil {
        return nil, err
    }
    
    return &MessageConsumer{
        conn:    conn,
        channel: ch,
        queue:   q,
    }, nil
}

func (mc *MessageConsumer) ConsumeMessages() error {
    msgs, err := mc.channel.Consume(
        mc.queue.Name, // queue
        "",            // consumer
        true,          // auto-ack
        false,         // exclusive
        false,         // no-local
        false,         // no-wait
        nil,           // args
    )
    if err != nil {
        return err
    }
    
    go func() {
        for msg := range msgs {
            fmt.Printf("Received message: %s\n", msg.Body)
            // Process message here
        }
    }()
    
    return nil
}
```

## Follow-up Questions

### 1. Scalability
**Q: What's the difference between horizontal and vertical scaling?**
A: Vertical scaling increases resources on existing servers (CPU, memory), while horizontal scaling adds more servers to handle increased load.

### 2. Load Balancing
**Q: What are the different load balancing algorithms?**
A: Round Robin, Weighted Round Robin, Least Connections, Least Response Time, IP Hash, and Random selection.

### 3. Caching
**Q: What's the difference between cache-aside and write-through patterns?**
A: Cache-aside requires application logic to manage cache, while write-through writes to both cache and database simultaneously.

## Sources

### Books
- **Designing Data-Intensive Applications** by Martin Kleppmann
- **System Design Interview** by Alex Xu
- **Microservices Patterns** by Chris Richardson

### Online Resources
- **High Scalability** - System design case studies
- **AWS Architecture Center** - Cloud architecture patterns
- **Google Cloud Architecture** - Scalable system designs

## Projects

### 1. Scalable Web Service
**Objective**: Build a horizontally scalable web service
**Requirements**: Load balancing, caching, database scaling
**Deliverables**: Complete scalable web application

### 2. Microservices Platform
**Objective**: Create a microservices architecture
**Requirements**: Service discovery, API gateway, message queues
**Deliverables**: Microservices platform with monitoring

### 3. Distributed Cache System
**Objective**: Implement a distributed caching solution
**Requirements**: Cache consistency, replication, failover
**Deliverables**: High-performance distributed cache

---

**Next**: [Phase 2: Advanced](../../../README.md) | **Previous**: [API Design](../../../README.md) | **Up**: [Phase 1](README.md/)

