# 🏗️ System Design Concepts - Complete Guide

> **Comprehensive guide to system design with Go implementations and FAANG interview questions**

## 📋 Table of Contents

1. [System Design Fundamentals](#system-design-fundamentals)
2. [Load Balancing](#load-balancing)
3. [Caching Strategies](#caching-strategies)
4. [Database Design](#database-design)
5. [Microservices Architecture](#microservices-architecture)
6. [Message Queues](#message-queues)
7. [API Design](#api-design)
8. [Security & Authentication](#security--authentication)
9. [FAANG Interview Questions](#faang-interview-questions)

---

## 🎯 System Design Fundamentals

### **1. Scalability Patterns**

#### **Horizontal vs Vertical Scaling**

**Detailed Explanation:**

Scalability is the ability of a system to handle increased load by adding resources. There are two primary approaches to scaling:

**Vertical Scaling (Scale Up):**

- Adding more power to existing machines (CPU, RAM, storage)
- Simpler architecture and management
- No network communication overhead
- Single point of failure
- Limited by hardware maximums
- More expensive per unit of performance

**Horizontal Scaling (Scale Out):**

- Adding more machines to the system
- Better fault tolerance and availability
- Can scale beyond single machine limits
- Requires load balancing and distributed systems knowledge
- Network communication overhead
- More complex to manage

**When to Use Each Approach:**

**Vertical Scaling is ideal for:**

- Small to medium applications
- When simplicity is more important than scalability
- Applications with single-threaded bottlenecks
- Legacy systems that are difficult to distribute

**Horizontal Scaling is ideal for:**

- High-traffic applications
- Systems requiring high availability
- Applications that need to scale beyond single machine limits
- Modern cloud-native applications

**Key Considerations:**

- **Cost**: Horizontal scaling can be more cost-effective at scale
- **Complexity**: Vertical scaling is simpler to implement
- **Fault Tolerance**: Horizontal scaling provides better resilience
- **Performance**: Vertical scaling avoids network overhead
- **Maintenance**: Horizontal scaling requires more operational overhead

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Vertical Scaling - Single powerful machine
// Pros: Simple, no network overhead, consistent performance
// Cons: Single point of failure, limited by hardware, expensive upgrades
type VerticalScaler struct {
    capacity int        // Maximum requests this machine can handle
    currentLoad int     // Current number of active requests
    mutex sync.Mutex    // Protects concurrent access to currentLoad
}

// Constructor for vertical scaler
func NewVerticalScaler(capacity int) *VerticalScaler {
    return &VerticalScaler{capacity: capacity}
}

// ProcessRequest simulates handling a request on a single machine
func (vs *VerticalScaler) ProcessRequest() bool {
    vs.mutex.Lock()         // Acquire lock for thread safety
    defer vs.mutex.Unlock() // Ensure lock is released

    // Check if machine has capacity
    if vs.currentLoad < vs.capacity {
        vs.currentLoad++                    // Increment load
        time.Sleep(100 * time.Millisecond) // Simulate processing time
        vs.currentLoad--                    // Decrement load
        return true                         // Request processed successfully
    }
    return false // Machine at capacity, request rejected
}

// Horizontal Scaling - Multiple machines
// Pros: Fault tolerance, cost-effective, unlimited scaling potential
// Cons: Network overhead, data consistency challenges, complex management
type HorizontalScaler struct {
    servers []*Server      // Array of server instances
    loadBalancer *LoadBalancer // Distributes requests across servers
}

// Server represents a single machine in the horizontal scaling setup
type Server struct {
    ID int              // Unique identifier for the server
    capacity int        // Maximum requests this server can handle
    currentLoad int     // Current number of active requests
    mutex sync.Mutex    // Protects concurrent access to currentLoad
}

// Constructor for individual server
func NewServer(id, capacity int) *Server {
    return &Server{ID: id, capacity: capacity}
}

// ProcessRequest simulates handling a request on this server
func (s *Server) ProcessRequest() bool {
    s.mutex.Lock()         // Acquire lock for thread safety
    defer s.mutex.Unlock() // Ensure lock is released

    // Check if server has capacity
    if s.currentLoad < s.capacity {
        s.currentLoad++                    // Increment load
        time.Sleep(100 * time.Millisecond) // Simulate processing time
        s.currentLoad--                    // Decrement load
        return true                        // Request processed successfully
    }
    return false // Server at capacity, request rejected
}

// LoadBalancer distributes incoming requests across multiple servers
// Implements Round Robin algorithm for request distribution
type LoadBalancer struct {
    servers []*Server  // List of available servers
    current int        // Index of current server for round robin
    mutex sync.Mutex   // Protects concurrent access to current index
}

// Constructor for load balancer
func NewLoadBalancer(servers []*Server) *LoadBalancer {
    return &LoadBalancer{servers: servers}
}

// GetServer returns the next server using round robin algorithm
func (lb *LoadBalancer) GetServer() *Server {
    lb.mutex.Lock()         // Acquire lock for thread safety
    defer lb.mutex.Unlock() // Ensure lock is released

    // Get current server and move to next
    server := lb.servers[lb.current]
    lb.current = (lb.current + 1) % len(lb.servers) // Round robin: cycle through servers
    return server
}

// ProcessRequest handles a request using horizontal scaling
func (hs *HorizontalScaler) ProcessRequest() bool {
    // Get next available server from load balancer
    server := hs.loadBalancer.GetServer()
    // Process request on the selected server
    return server.ProcessRequest()
}

func main() {
    // Vertical scaling example - single machine with capacity 10
    vs := NewVerticalScaler(10)
    fmt.Println("Vertical Scaling (Single Machine):")
    for i := 0; i < 15; i++ {
        if vs.ProcessRequest() {
            fmt.Printf("Request %d processed\n", i)
        } else {
            fmt.Printf("Request %d rejected (capacity exceeded)\n", i)
        }
    }

    // Horizontal scaling example - 3 machines with capacity 5 each
    servers := []*Server{
        NewServer(1, 5), // Server 1 with capacity 5
        NewServer(2, 5), // Server 2 with capacity 5
        NewServer(3, 5), // Server 3 with capacity 5
    }
    lb := NewLoadBalancer(servers)
    hs := &HorizontalScaler{servers: servers, loadBalancer: lb}

    fmt.Println("\nHorizontal Scaling (3 Machines):")
    for i := 0; i < 15; i++ {
        if hs.ProcessRequest() {
            fmt.Printf("Request %d processed\n", i)
        } else {
            fmt.Printf("Request %d rejected (capacity exceeded)\n", i)
        }
    }
}
```

**Key Concepts Explained:**

- **Vertical Scaling**: Single powerful machine, simple but limited
- **Horizontal Scaling**: Multiple machines, complex but scalable
- **Load Balancing**: Distributes requests across multiple servers
- **Round Robin**: Simple load balancing algorithm
- **Thread Safety**: Mutex protects shared resources
- **Capacity Management**: Tracks current load vs maximum capacity

### **2. CAP Theorem Implementation**

**Detailed Explanation:**

The CAP theorem, formulated by Eric Brewer, states that in a distributed system, you can only guarantee two out of three properties:

**Consistency (C):**

- All nodes see the same data at the same time
- After an update, all subsequent reads return the updated value
- Strong consistency ensures data integrity
- Examples: ACID databases, financial systems

**Availability (A):**

- The system remains operational and responsive
- Every request receives a response (success or failure)
- No downtime or service unavailability
- Examples: CDNs, web services

**Partition Tolerance (P):**

- The system continues to operate despite network failures
- Handles network splits and communication failures
- Essential for distributed systems
- Examples: Internet-based systems, cloud services

**CAP Trade-offs in Practice:**

**CP Systems (Consistency + Partition Tolerance):**

- Prioritize data consistency over availability
- Examples: MongoDB, HBase, traditional RDBMS
- Use case: Financial systems, inventory management

**AP Systems (Availability + Partition Tolerance):**

- Prioritize availability over consistency
- Examples: Cassandra, DynamoDB, CouchDB
- Use case: Social media, content delivery

**CA Systems (Consistency + Availability):**

- Only possible in non-distributed systems
- Single-node systems or systems with perfect network
- Examples: Single-server databases

**Real-World Implications:**

1. **Network Partitions are Inevitable**: In distributed systems, network failures will occur
2. **Choose Based on Use Case**: Different applications have different requirements
3. **Eventual Consistency**: Many systems use eventual consistency as a compromise
4. **Consistency Levels**: Modern systems offer different consistency levels

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Consistency, Availability, Partition Tolerance
type CAPSystem struct {
    nodes map[string]*Node
    consistencyLevel string
    mutex sync.RWMutex
}

type Node struct {
    ID string
    Data map[string]string
    mutex sync.RWMutex
}

func NewCAPSystem(consistencyLevel string) *CAPSystem {
    return &CAPSystem{
        nodes: make(map[string]*Node),
        consistencyLevel: consistencyLevel,
    }
}

func (cs *CAPSystem) AddNode(id string) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()

    cs.nodes[id] = &Node{
        ID: id,
        Data: make(map[string]string),
    }
}

func (cs *CAPSystem) Write(key, value string) error {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()

    switch cs.consistencyLevel {
    case "strong":
        return cs.strongConsistencyWrite(key, value)
    case "eventual":
        return cs.eventualConsistencyWrite(key, value)
    default:
        return fmt.Errorf("unknown consistency level")
    }
}

func (cs *CAPSystem) strongConsistencyWrite(key, value string) error {
    // Write to all nodes synchronously
    for _, node := range cs.nodes {
        node.mutex.Lock()
        node.Data[key] = value
        node.mutex.Unlock()
    }
    return nil
}

func (cs *CAPSystem) eventualConsistencyWrite(key, value string) error {
    // Write to one node, replicate asynchronously
    for _, node := range cs.nodes {
        node.mutex.Lock()
        node.Data[key] = value
        node.mutex.Unlock()
        break // Write to first available node
    }

    // Replicate asynchronously
    go cs.replicateAsync(key, value)
    return nil
}

func (cs *CAPSystem) replicateAsync(key, value string) {
    time.Sleep(100 * time.Millisecond) // Simulate network delay

    cs.mutex.RLock()
    defer cs.mutex.RUnlock()

    for _, node := range cs.nodes {
        node.mutex.Lock()
        node.Data[key] = value
        node.mutex.Unlock()
    }
}

func (cs *CAPSystem) Read(key string) (string, error) {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()

    switch cs.consistencyLevel {
    case "strong":
        return cs.strongConsistencyRead(key)
    case "eventual":
        return cs.eventualConsistencyRead(key)
    default:
        return "", fmt.Errorf("unknown consistency level")
    }
}

func (cs *CAPSystem) strongConsistencyRead(key string) (string, error) {
    // Read from all nodes and ensure consistency
    var values []string
    for _, node := range cs.nodes {
        node.mutex.RLock()
        if value, exists := node.Data[key]; exists {
            values = append(values, value)
        }
        node.mutex.RUnlock()
    }

    if len(values) == 0 {
        return "", fmt.Errorf("key not found")
    }

    // Check if all values are the same
    for _, value := range values {
        if value != values[0] {
            return "", fmt.Errorf("inconsistent data")
        }
    }

    return values[0], nil
}

func (cs *CAPSystem) eventualConsistencyRead(key string) (string, error) {
    // Read from any available node
    for _, node := range cs.nodes {
        node.mutex.RLock()
        if value, exists := node.Data[key]; exists {
            node.mutex.RUnlock()
            return value, nil
        }
        node.mutex.RUnlock()
    }

    return "", fmt.Errorf("key not found")
}

func main() {
    // Strong consistency system
    strongSystem := NewCAPSystem("strong")
    strongSystem.AddNode("node1")
    strongSystem.AddNode("node2")
    strongSystem.AddNode("node3")

    strongSystem.Write("key1", "value1")
    value, _ := strongSystem.Read("key1")
    fmt.Printf("Strong consistency read: %s\n", value)

    // Eventual consistency system
    eventualSystem := NewCAPSystem("eventual")
    eventualSystem.AddNode("node1")
    eventualSystem.AddNode("node2")
    eventualSystem.AddNode("node3")

    eventualSystem.Write("key1", "value1")
    value, _ = eventualSystem.Read("key1")
    fmt.Printf("Eventual consistency read: %s\n", value)
}
```

**Discussion Questions & Answers:**

**Q1: How do you choose between consistency and availability in a distributed system?**

**Answer:** The choice depends on your application's requirements:

- **Choose Consistency (CP)**: When data integrity is critical (financial systems, inventory management, user authentication)
- **Choose Availability (AP)**: When system uptime is more important than perfect consistency (social media feeds, content delivery, real-time analytics)
- **Consider Eventual Consistency**: A compromise that provides availability with eventual data consistency

**Q2: Can you have all three properties (CAP) in a distributed system?**

**Answer:** No, the CAP theorem proves that you cannot have all three properties simultaneously in a distributed system. However, you can:

- Have all three in non-distributed systems (CA systems)
- Use different consistency levels for different operations
- Implement eventual consistency as a compromise
- Design systems that gracefully degrade during partitions

**Q3: How does the CAP theorem apply to modern cloud systems?**

**Answer:** Modern cloud systems often use:

- **Multi-region deployments** with eventual consistency
- **Consistency levels** (strong, eventual, session, bounded staleness)
- **Conflict resolution** mechanisms for eventual consistency
- **Circuit breakers** to handle partition scenarios
- **Caching layers** to improve availability while maintaining consistency

**Q4: What are the practical implications of choosing AP over CP?**

**Answer:** Choosing AP (Availability + Partition Tolerance) means:

- **Benefits**: High availability, better user experience, handles network failures gracefully
- **Trade-offs**: Data may be temporarily inconsistent, requires conflict resolution, more complex application logic
- **Use Cases**: Social media, content delivery, real-time systems, analytics platforms

**Q5: How do you handle data consistency in an AP system?**

**Answer:** Several strategies for handling consistency in AP systems:

- **Eventual Consistency**: Data becomes consistent over time
- **Conflict Resolution**: Use timestamps, vector clocks, or application logic
- **Read Repair**: Fix inconsistencies during read operations
- **Anti-Entropy**: Background processes to sync data
- **CRDTs**: Conflict-free Replicated Data Types for automatic conflict resolution

---

## ⚖️ Load Balancing

### **3. Load Balancing Algorithms**

**Detailed Explanation:**

Load balancing is a critical component of scalable systems that distributes incoming requests across multiple servers to ensure optimal resource utilization, high availability, and improved performance.

**Why Load Balancing is Important:**

1. **Scalability**: Distribute load across multiple servers
2. **High Availability**: Route traffic away from failed servers
3. **Performance**: Optimize response times and throughput
4. **Resource Utilization**: Ensure all servers are used efficiently
5. **Fault Tolerance**: Handle server failures gracefully

**Load Balancing Algorithms:**

**Round Robin:**

- Distributes requests sequentially across servers
- Simple and fair distribution
- Good for servers with similar capacity
- May not consider server load or response time

**Weighted Round Robin:**

- Assigns different weights to servers based on capacity
- Servers with higher weights receive more requests
- Better for heterogeneous server environments
- More complex than simple round robin

**Least Connections:**

- Routes requests to the server with the fewest active connections
- Good for long-lived connections
- Considers current server load
- May not account for server capacity differences

**IP Hash:**

- Uses client IP to determine server assignment
- Ensures same client always goes to same server
- Good for session affinity requirements
- May cause uneven distribution

**Health Checks:**

- Monitors server health and availability
- Removes unhealthy servers from rotation
- Adds servers back when they recover
- Essential for production systems

**Load Balancing Strategies:**

**Layer 4 (Transport Layer):**

- Routes based on IP and port
- Faster processing
- Less intelligent routing
- Examples: HAProxy, F5

**Layer 7 (Application Layer):**

- Routes based on HTTP headers, content
- More intelligent routing
- Slower processing
- Examples: NGINX, AWS ALB

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type LoadBalancer interface {
    GetServer() *Server
}

type Server struct {
    ID string
    Weight int
    CurrentConnections int
    mutex sync.Mutex
}

func (s *Server) ProcessRequest() {
    s.mutex.Lock()
    s.CurrentConnections++
    s.mutex.Unlock()

    time.Sleep(100 * time.Millisecond) // Simulate processing

    s.mutex.Lock()
    s.CurrentConnections--
    s.mutex.Unlock()
}

// Round Robin Load Balancer
type RoundRobinLB struct {
    servers []*Server
    current int
    mutex sync.Mutex
}

func NewRoundRobinLB(servers []*Server) *RoundRobinLB {
    return &RoundRobinLB{servers: servers}
}

func (rr *RoundRobinLB) GetServer() *Server {
    rr.mutex.Lock()
    defer rr.mutex.Unlock()

    server := rr.servers[rr.current]
    rr.current = (rr.current + 1) % len(rr.servers)
    return server
}

// Weighted Round Robin Load Balancer
type WeightedRoundRobinLB struct {
    servers []*Server
    current int
    currentWeight int
    mutex sync.Mutex
}

func NewWeightedRoundRobinLB(servers []*Server) *WeightedRoundRobinLB {
    return &WeightedRoundRobinLB{servers: servers}
}

func (wrr *WeightedRoundRobinLB) GetServer() *Server {
    wrr.mutex.Lock()
    defer wrr.mutex.Unlock()

    for {
        wrr.current = (wrr.current + 1) % len(wrr.servers)
        if wrr.current == 0 {
            wrr.currentWeight -= wrr.gcd()
            if wrr.currentWeight <= 0 {
                wrr.currentWeight = wrr.maxWeight()
            }
        }

        if wrr.servers[wrr.current].Weight >= wrr.currentWeight {
            return wrr.servers[wrr.current]
        }
    }
}

func (wrr *WeightedRoundRobinLB) gcd() int {
    weights := make([]int, len(wrr.servers))
    for i, server := range wrr.servers {
        weights[i] = server.Weight
    }

    result := weights[0]
    for i := 1; i < len(weights); i++ {
        result = gcd(result, weights[i])
    }
    return result
}

func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func (wrr *WeightedRoundRobinLB) maxWeight() int {
    max := wrr.servers[0].Weight
    for _, server := range wrr.servers {
        if server.Weight > max {
            max = server.Weight
        }
    }
    return max
}

// Least Connections Load Balancer
type LeastConnectionsLB struct {
    servers []*Server
    mutex sync.Mutex
}

func NewLeastConnectionsLB(servers []*Server) *LeastConnectionsLB {
    return &LeastConnectionsLB{servers: servers}
}

func (lc *LeastConnectionsLB) GetServer() *Server {
    lc.mutex.Lock()
    defer lc.mutex.Unlock()

    minConnections := lc.servers[0].CurrentConnections
    selectedServer := lc.servers[0]

    for _, server := range lc.servers {
        if server.CurrentConnections < minConnections {
            minConnections = server.CurrentConnections
            selectedServer = server
        }
    }

    return selectedServer
}

func main() {
    servers := []*Server{
        {ID: "server1", Weight: 3},
        {ID: "server2", Weight: 2},
        {ID: "server3", Weight: 1},
    }

    // Test Round Robin
    fmt.Println("Round Robin Load Balancing:")
    rrLB := NewRoundRobinLB(servers)
    for i := 0; i < 6; i++ {
        server := rrLB.GetServer()
        fmt.Printf("Request %d -> %s\n", i, server.ID)
    }

    // Test Weighted Round Robin
    fmt.Println("\nWeighted Round Robin Load Balancing:")
    wrrLB := NewWeightedRoundRobinLB(servers)
    for i := 0; i < 6; i++ {
        server := wrrLB.GetServer()
        fmt.Printf("Request %d -> %s\n", i, server.ID)
    }

    // Test Least Connections
    fmt.Println("\nLeast Connections Load Balancing:")
    lcLB := NewLeastConnectionsLB(servers)
    for i := 0; i < 6; i++ {
        server := lcLB.GetServer()
        fmt.Printf("Request %d -> %s\n", i, server.ID)
        server.ProcessRequest()
    }
}
```

**Discussion Questions & Answers:**

**Q1: How do you choose the right load balancing algorithm for your system?**

**Answer:** Consider these factors:

- **Server Capacity**: Use weighted algorithms for heterogeneous servers
- **Connection Type**: Use least connections for long-lived connections
- **Session Requirements**: Use IP hash for session affinity
- **Simplicity**: Use round robin for simple, uniform servers
- **Performance**: Consider the overhead of each algorithm
- **Monitoring**: Choose algorithms that provide good observability

**Q2: What are the trade-offs between Layer 4 and Layer 7 load balancing?**

**Answer:**
**Layer 4 (Transport Layer):**

- **Pros**: Faster processing, lower latency, simpler configuration
- **Cons**: Less intelligent routing, no content-based decisions
- **Use Cases**: High-throughput applications, simple routing needs

**Layer 7 (Application Layer):**

- **Pros**: Intelligent routing, content-based decisions, better security
- **Cons**: Higher latency, more complex configuration, higher resource usage
- **Use Cases**: Complex routing requirements, API gateways, microservices

**Q3: How do you handle server failures in a load balancer?**

**Answer:** Implement comprehensive failure handling:

- **Health Checks**: Regular monitoring of server health
- **Circuit Breakers**: Temporarily remove failing servers
- **Graceful Degradation**: Reduce load on failing servers
- **Automatic Recovery**: Add servers back when they recover
- **Monitoring**: Alert on failure patterns and trends
- **Fallback Strategies**: Route to backup servers or show maintenance pages

**Q4: What are the challenges of load balancing in a microservices architecture?**

**Answer:** Key challenges include:

- **Service Discovery**: Finding available service instances
- **Health Monitoring**: Tracking health of many small services
- **Load Balancing**: Distributing load across service instances
- **Circuit Breaking**: Preventing cascade failures
- **Service Mesh**: Managing communication between services
- **Configuration**: Managing load balancer configs for many services

**Q5: How do you implement sticky sessions in a load balancer?**

**Answer:** Several approaches for session affinity:

- **IP Hash**: Route based on client IP address
- **Cookie-based**: Use application cookies to track sessions
- **Session ID**: Include session identifier in requests
- **Consistent Hashing**: Use consistent hashing for better distribution
- **Database Sessions**: Store sessions in shared database
- **Redis Sessions**: Use Redis for session storage

---

## 💾 Caching Strategies

### **4. Cache Implementation**

**Detailed Explanation:**

Caching is a fundamental technique for improving system performance by storing frequently accessed data in fast storage. It reduces latency, decreases database load, and improves user experience.

**Why Caching is Important:**

1. **Performance**: Reduces response time by serving data from memory
2. **Scalability**: Reduces load on backend systems
3. **Cost Efficiency**: Reduces database queries and infrastructure costs
4. **User Experience**: Faster page loads and interactions
5. **Reliability**: Provides fallback when backend systems are slow

**Cache Types:**

**In-Memory Cache:**

- Stored in application memory (Redis, Memcached)
- Fastest access but limited by memory size
- Lost when application restarts
- Good for frequently accessed data

**Distributed Cache:**

- Shared across multiple application instances
- Survives application restarts
- Can be scaled independently
- Examples: Redis Cluster, Hazelcast

**CDN Cache:**

- Caches static content at edge locations
- Reduces latency for global users
- Examples: CloudFlare, AWS CloudFront

**Database Cache:**

- Caches query results
- Reduces database load
- Examples: MySQL Query Cache, PostgreSQL Buffer Cache

**Cache Strategies:**

**Cache-Aside (Lazy Loading):**

- Application checks cache first
- Loads from database if cache miss
- Updates cache after loading
- Most common pattern

**Write-Through:**

- Writes to both cache and database
- Ensures data consistency
- Higher write latency
- Good for critical data

**Write-Behind (Write-Back):**

- Writes to cache immediately
- Writes to database asynchronously
- Fastest writes
- Risk of data loss

**Refresh-Ahead:**

- Refreshes cache before expiration
- Reduces cache misses
- More complex implementation
- Good for predictable access patterns

**Cache Eviction Policies:**

**LRU (Least Recently Used):**

- Evicts least recently accessed items
- Good for temporal locality
- Simple to implement

**LFU (Least Frequently Used):**

- Evicts least frequently accessed items
- Good for frequency-based access
- More complex to implement

**TTL (Time To Live):**

- Evicts items after fixed time
- Simple and predictable
- May evict still-useful data

**Random:**

- Evicts random items
- Simple implementation
- May evict important data

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type CacheItem struct {
    Key string
    Value interface{}
    Expiry time.Time
    AccessCount int
    LastAccessed time.Time
}

type Cache struct {
    items map[string]*CacheItem
    mutex sync.RWMutex
    maxSize int
    ttl time.Duration
}

func NewCache(maxSize int, ttl time.Duration) *Cache {
    cache := &Cache{
        items: make(map[string]*CacheItem),
        maxSize: maxSize,
        ttl: ttl,
    }

    // Start cleanup goroutine
    go cache.cleanup()

    return cache
}

func (c *Cache) Set(key string, value interface{}) {
    c.mutex.Lock()
    defer c.mutex.Unlock()

    // Check if cache is full
    if len(c.items) >= c.maxSize {
        c.evictLRU()
    }

    c.items[key] = &CacheItem{
        Key: key,
        Value: value,
        Expiry: time.Now().Add(c.ttl),
        AccessCount: 1,
        LastAccessed: time.Now(),
    }
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mutex.Lock()
    defer c.mutex.Unlock()

    item, exists := c.items[key]
    if !exists {
        return nil, false
    }

    // Check if expired
    if time.Now().After(item.Expiry) {
        delete(c.items, key)
        return nil, false
    }

    // Update access information
    item.AccessCount++
    item.LastAccessed = time.Now()

    return item.Value, true
}

func (c *Cache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time

    for key, item := range c.items {
        if oldestKey == "" || item.LastAccessed.Before(oldestTime) {
            oldestKey = key
            oldestTime = item.LastAccessed
        }
    }

    if oldestKey != "" {
        delete(c.items, oldestKey)
    }
}

func (c *Cache) cleanup() {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()

    for range ticker.C {
        c.mutex.Lock()
        now := time.Now()
        for key, item := range c.items {
            if now.After(item.Expiry) {
                delete(c.items, key)
            }
        }
        c.mutex.Unlock()
    }
}

func (c *Cache) Size() int {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    return len(c.items)
}

func main() {
    cache := NewCache(3, 5*time.Second)

    // Set some values
    cache.Set("key1", "value1")
    cache.Set("key2", "value2")
    cache.Set("key3", "value3")

    // Get values
    if value, exists := cache.Get("key1"); exists {
        fmt.Printf("key1: %v\n", value)
    }

    // Test LRU eviction
    cache.Set("key4", "value4") // This should evict key1

    if _, exists := cache.Get("key1"); !exists {
        fmt.Println("key1 was evicted (LRU)")
    }

    // Test expiration
    time.Sleep(6 * time.Second)
    if _, exists := cache.Get("key2"); !exists {
        fmt.Println("key2 expired")
    }

    fmt.Printf("Cache size: %d\n", cache.Size())
}
```

**Discussion Questions & Answers:**

**Q1: How do you choose the right caching strategy for your application?**

**Answer:** Consider these factors:

- **Data Access Patterns**: Use cache-aside for read-heavy, write-through for write-heavy
- **Consistency Requirements**: Use write-through for strong consistency, write-behind for eventual consistency
- **Performance Needs**: Use write-behind for fastest writes, cache-aside for balanced performance
- **Data Criticality**: Use write-through for critical data, write-behind for less critical data
- **System Complexity**: Start with cache-aside, add complexity as needed

**Q2: What are the trade-offs between different cache eviction policies?**

**Answer:**
**LRU (Least Recently Used):**

- **Pros**: Good for temporal locality, simple to implement
- **Cons**: May evict frequently used items if not accessed recently
- **Use Cases**: Web applications, user sessions

**LFU (Least Frequently Used):**

- **Pros**: Good for frequency-based access patterns
- **Cons**: More complex implementation, may keep old popular items
- **Use Cases**: Content recommendation systems, popular items

**TTL (Time To Live):**

- **Pros**: Simple and predictable, good for time-sensitive data
- **Cons**: May evict still-useful data, requires tuning
- **Use Cases**: API responses, temporary data

**Q3: How do you handle cache invalidation in a distributed system?**

**Answer:** Several strategies for cache invalidation:

- **TTL-based**: Let items expire naturally
- **Event-driven**: Invalidate on data changes
- **Version-based**: Use version numbers to detect stale data
- **Tag-based**: Invalidate by tags or categories
- **Write-through**: Update cache on every write
- **Cache-aside**: Let application handle invalidation

**Q4: What are the challenges of caching in a microservices architecture?**

**Answer:** Key challenges include:

- **Cache Consistency**: Keeping caches in sync across services
- **Cache Dependencies**: Managing cache dependencies between services
- **Cache Warming**: Preloading caches after service restarts
- **Cache Partitioning**: Distributing cache data across services
- **Cache Monitoring**: Tracking cache performance across services
- **Cache Security**: Securing cached data across service boundaries

**Q5: How do you implement cache warming strategies?**

**Answer:** Several approaches for cache warming:

- **Lazy Loading**: Load data on first access
- **Eager Loading**: Preload data during startup
- **Scheduled Warming**: Refresh cache at regular intervals
- **Event-driven Warming**: Warm cache on specific events
- **Predictive Warming**: Use ML to predict what to cache
- **Hybrid Approach**: Combine multiple strategies

---

## 🗄️ Database Design

### **5. Database Sharding**

**Detailed Explanation:**

Database sharding is a horizontal partitioning technique that splits a large database into smaller, more manageable pieces called shards. Each shard contains a subset of the data and can be stored on different servers.

**Why Sharding is Important:**

1. **Scalability**: Distribute data across multiple servers
2. **Performance**: Reduce query response time by accessing smaller datasets
3. **Availability**: Isolate failures to specific shards
4. **Maintenance**: Easier to manage smaller databases
5. **Cost**: Use smaller, cheaper servers instead of one large server

**Sharding Strategies:**

**Horizontal Sharding:**

- Split tables by rows
- Each shard contains different rows
- Most common approach
- Good for large tables

**Vertical Sharding:**

- Split tables by columns
- Each shard contains different columns
- Less common
- Good for wide tables

**Sharding Keys:**

**Hash-based Sharding:**

- Use hash function on shard key
- Even distribution of data
- Difficult to add/remove shards
- Examples: User ID, Order ID

**Range-based Sharding:**

- Split data by ranges
- Easy to add/remove shards
- May cause uneven distribution
- Examples: Date ranges, ID ranges

**Directory-based Sharding:**

- Use lookup table for shard mapping
- Flexible and easy to change
- Single point of failure
- Examples: Geographic regions

**Consistent Hashing:**

- Distribute data in a ring
- Easy to add/remove shards
- Minimal data movement
- Examples: Distributed caches

**Sharding Challenges:**

**Cross-shard Queries:**

- Queries spanning multiple shards
- Complex to implement
- May require application-level joins
- Consider denormalization

**Data Rebalancing:**

- Moving data when adding/removing shards
- Expensive operation
- May require downtime
- Plan for data migration

**Transaction Management:**

- ACID properties across shards
- Two-phase commit for consistency
- Eventual consistency for performance
- Consider distributed transactions

**Shard Monitoring:**

- Track performance per shard
- Monitor data distribution
- Alert on shard failures
- Plan for capacity growth

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "sync"
)

type Shard struct {
    ID string
    Data map[string]interface{}
    mutex sync.RWMutex
}

type ShardManager struct {
    shards []*Shard
    hashRing []string
    mutex sync.RWMutex
}

func NewShardManager(shardCount int) *ShardManager {
    sm := &ShardManager{
        shards: make([]*Shard, shardCount),
        hashRing: make([]string, 0),
    }

    for i := 0; i < shardCount; i++ {
        shardID := fmt.Sprintf("shard%d", i)
        sm.shards[i] = &Shard{
            ID: shardID,
            Data: make(map[string]interface{}),
        }
        sm.hashRing = append(sm.hashRing, shardID)
    }

    sort.Strings(sm.hashRing)
    return sm
}

func (sm *ShardManager) GetShard(key string) *Shard {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()

    hash := sm.hash(key)

    // Find the first shard with hash >= key hash
    for _, shardID := range sm.hashRing {
        shardHash := sm.hash(shardID)
        if shardHash >= hash {
            for _, shard := range sm.shards {
                if shard.ID == shardID {
                    return shard
                }
            }
        }
    }

    // If no shard found, return the first one
    return sm.shards[0]
}

func (sm *ShardManager) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

func (sm *ShardManager) Set(key string, value interface{}) {
    shard := sm.GetShard(key)
    shard.mutex.Lock()
    defer shard.mutex.Unlock()

    shard.Data[key] = value
    fmt.Printf("Set %s = %v in %s\n", key, value, shard.ID)
}

func (sm *ShardManager) Get(key string) (interface{}, bool) {
    shard := sm.GetShard(key)
    shard.mutex.RLock()
    defer shard.mutex.RUnlock()

    value, exists := shard.Data[key]
    if exists {
        fmt.Printf("Get %s = %v from %s\n", key, value, shard.ID)
    }
    return value, exists
}

func (sm *ShardManager) Delete(key string) {
    shard := sm.GetShard(key)
    shard.mutex.Lock()
    defer shard.mutex.Unlock()

    delete(shard.Data, key)
    fmt.Printf("Deleted %s from %s\n", key, shard.ID)
}

func main() {
    sm := NewShardManager(3)

    // Set some data
    sm.Set("user1", "John Doe")
    sm.Set("user2", "Jane Smith")
    sm.Set("user3", "Bob Johnson")
    sm.Set("user4", "Alice Brown")

    // Get data
    if value, exists := sm.Get("user1"); exists {
        fmt.Printf("Retrieved: %v\n", value)
    }

    // Delete data
    sm.Delete("user2")

    // Try to get deleted data
    if _, exists := sm.Get("user2"); !exists {
        fmt.Println("user2 not found (deleted)")
    }
}
```

**Discussion Questions & Answers:**

**Q1: How do you choose the right sharding strategy for your database?**

**Answer:** Consider these factors:

- **Data Distribution**: Use hash-based for even distribution, range-based for sequential data
- **Query Patterns**: Use range-based for range queries, hash-based for point queries
- **Scalability Needs**: Use consistent hashing for easy scaling, hash-based for fixed shards
- **Data Relationships**: Consider how related data is accessed together
- **Maintenance Complexity**: Balance between performance and operational complexity

**Q2: What are the challenges of maintaining ACID properties across shards?**

**Answer:** Key challenges include:

- **Distributed Transactions**: Use two-phase commit for consistency
- **Performance Impact**: Distributed transactions are slower than local ones
- **Failure Handling**: Partial failures can leave system in inconsistent state
- **Deadlocks**: More complex deadlock detection across shards
- **Network Latency**: Communication between shards adds latency
- **Consider Eventual Consistency**: May be acceptable for some use cases

**Q3: How do you handle cross-shard queries efficiently?**

**Answer:** Several strategies for cross-shard queries:

- **Denormalization**: Duplicate data across shards to avoid joins
- **Application-level Joins**: Fetch data from multiple shards and join in application
- **Read Replicas**: Use read replicas for cross-shard queries
- **Materialized Views**: Pre-compute cross-shard aggregations
- **Event Sourcing**: Use events to maintain cross-shard relationships
- **CQRS**: Separate read and write models

**Q4: What are the trade-offs between different sharding key strategies?**

**Answer:**
**Hash-based Sharding:**

- **Pros**: Even distribution, good for point queries
- **Cons**: Difficult to add/remove shards, poor for range queries
- **Use Cases**: User data, order data

**Range-based Sharding:**

- **Pros**: Easy to add/remove shards, good for range queries
- **Cons**: Uneven distribution, hot spots
- **Use Cases**: Time-series data, sequential data

**Consistent Hashing:**

- **Pros**: Easy scaling, minimal data movement
- **Cons**: Complex implementation, may cause uneven distribution
- **Use Cases**: Distributed caches, content delivery

**Q5: How do you monitor and maintain a sharded database?**

**Answer:** Comprehensive monitoring strategy:

- **Shard Health**: Monitor each shard's performance and availability
- **Data Distribution**: Track data distribution across shards
- **Query Performance**: Monitor query performance per shard
- **Capacity Planning**: Plan for shard capacity and growth
- **Automated Alerts**: Set up alerts for shard failures and performance issues
- **Regular Maintenance**: Plan for data rebalancing and shard maintenance

---

## 🎯 FAANG Interview Questions

### **Google Interview Questions**

#### **1. Design a URL Shortener**

**Question**: "Design a URL shortener service like bit.ly that can handle billions of URLs."

**Answer**:

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sync"
    "time"
)

type URLShortener struct {
    urls map[string]string
    shortToLong map[string]string
    mutex sync.RWMutex
    counter int64
}

func NewURLShortener() *URLShortener {
    return &URLShortener{
        urls: make(map[string]string),
        shortToLong: make(map[string]string),
    }
}

func (us *URLShortener) ShortenURL(longURL string) string {
    us.mutex.Lock()
    defer us.mutex.Unlock()

    // Check if URL already exists
    if shortURL, exists := us.urls[longURL]; exists {
        return shortURL
    }

    // Generate short URL
    us.counter++
    shortURL := us.encodeBase62(us.counter)

    // Store mapping
    us.urls[longURL] = shortURL
    us.shortToLong[shortURL] = longURL

    return shortURL
}

func (us *URLShortener) ExpandURL(shortURL string) (string, bool) {
    us.mutex.RLock()
    defer us.mutex.RUnlock()

    longURL, exists := us.shortToLong[shortURL]
    return longURL, exists
}

func (us *URLShortener) encodeBase62(num int64) string {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    result := ""

    for num > 0 {
        result = string(charset[num%62]) + result
        num /= 62
    }

    return result
}

func main() {
    shortener := NewURLShortener()

    // Shorten URLs
    short1 := shortener.ShortenURL("https://www.google.com")
    short2 := shortener.ShortenURL("https://www.github.com")
    short3 := shortener.ShortenURL("https://www.stackoverflow.com")

    fmt.Printf("Shortened URLs:\n")
    fmt.Printf("https://www.google.com -> %s\n", short1)
    fmt.Printf("https://www.github.com -> %s\n", short2)
    fmt.Printf("https://www.stackoverflow.com -> %s\n", short3)

    // Expand URLs
    if longURL, exists := shortener.ExpandURL(short1); exists {
        fmt.Printf("Expanded %s -> %s\n", short1, longURL)
    }
}
```

### **Meta Interview Questions**

#### **2. Design a News Feed System**

**Question**: "Design a news feed system similar to Facebook's that can handle millions of users."

**Answer**:

```go
package main

import (
    "fmt"
    "sort"
    "sync"
    "time"
)

type Post struct {
    ID string
    UserID string
    Content string
    Timestamp time.Time
    Likes int
    Comments int
}

type User struct {
    ID string
    Friends []string
    Posts []string
}

type NewsFeedService struct {
    users map[string]*User
    posts map[string]*Post
    feeds map[string][]*Post
    mutex sync.RWMutex
}

func NewNewsFeedService() *NewsFeedService {
    return &NewsFeedService{
        users: make(map[string]*User),
        posts: make(map[string]*Post),
        feeds: make(map[string][]*Post),
    }
}

func (nfs *NewsFeedService) AddUser(userID string) {
    nfs.mutex.Lock()
    defer nfs.mutex.Unlock()

    nfs.users[userID] = &User{
        ID: userID,
        Friends: make([]string, 0),
        Posts: make([]string, 0),
    }
}

func (nfs *NewsFeedService) AddFriend(userID, friendID string) {
    nfs.mutex.Lock()
    defer nfs.mutex.Unlock()

    if user, exists := nfs.users[userID]; exists {
        user.Friends = append(user.Friends, friendID)
    }
}

func (nfs *NewsFeedService) CreatePost(userID, content string) string {
    nfs.mutex.Lock()
    defer nfs.mutex.Unlock()

    postID := fmt.Sprintf("post_%d", time.Now().UnixNano())
    post := &Post{
        ID: postID,
        UserID: userID,
        Content: content,
        Timestamp: time.Now(),
        Likes: 0,
        Comments: 0,
    }

    nfs.posts[postID] = post

    if user, exists := nfs.users[userID]; exists {
        user.Posts = append(user.Posts, postID)
    }

    // Update feeds of friends
    nfs.updateFeeds(userID, post)

    return postID
}

func (nfs *NewsFeedService) updateFeeds(userID string, post *Post) {
    if user, exists := nfs.users[userID]; exists {
        for _, friendID := range user.Friends {
            nfs.feeds[friendID] = append(nfs.feeds[friendID], post)
        }
    }
}

func (nfs *NewsFeedService) GetNewsFeed(userID string, limit int) []*Post {
    nfs.mutex.RLock()
    defer nfs.mutex.RUnlock()

    feed, exists := nfs.feeds[userID]
    if !exists {
        return []*Post{}
    }

    // Sort by timestamp (newest first)
    sortedFeed := make([]*Post, len(feed))
    copy(sortedFeed, feed)
    sort.Slice(sortedFeed, func(i, j int) bool {
        return sortedFeed[i].Timestamp.After(sortedFeed[j].Timestamp)
    })

    // Return limited results
    if len(sortedFeed) > limit {
        return sortedFeed[:limit]
    }

    return sortedFeed
}

func (nfs *NewsFeedService) LikePost(postID string) {
    nfs.mutex.Lock()
    defer nfs.mutex.Unlock()

    if post, exists := nfs.posts[postID]; exists {
        post.Likes++
    }
}

func main() {
    nfs := NewNewsFeedService()

    // Add users
    nfs.AddUser("user1")
    nfs.AddUser("user2")
    nfs.AddUser("user3")

    // Add friendships
    nfs.AddFriend("user1", "user2")
    nfs.AddFriend("user1", "user3")
    nfs.AddFriend("user2", "user1")
    nfs.AddFriend("user3", "user1")

    // Create posts
    nfs.CreatePost("user1", "Hello, world!")
    nfs.CreatePost("user2", "Go is awesome!")
    nfs.CreatePost("user3", "Learning system design")

    // Get news feed for user1
    feed := nfs.GetNewsFeed("user1", 10)
    fmt.Println("News Feed for user1:")
    for _, post := range feed {
        fmt.Printf("- %s: %s (Likes: %d)\n", post.UserID, post.Content, post.Likes)
    }

    // Like a post
    nfs.LikePost("post_1")

    // Get updated feed
    feed = nfs.GetNewsFeed("user1", 10)
    fmt.Println("\nUpdated News Feed for user1:")
    for _, post := range feed {
        fmt.Printf("- %s: %s (Likes: %d)\n", post.UserID, post.Content, post.Likes)
    }
}
```

---

## 🎯 **LLD & HLD Interview Questions (Educative.io Style)**

### **High-Level Design (HLD) Questions**

#### **1. Design a URL Shortener (like bit.ly)**

**Problem Statement**: Design a URL shortener service that can handle millions of URLs and provide analytics.

**Requirements**:

- Shorten long URLs to short URLs
- Redirect short URLs to original URLs
- Handle 100M URLs per day
- Analytics on URL usage
- Custom short URLs (optional)

**Solution**:

```go
package main

import (
    "crypto/md5"
    "encoding/hex"
    "fmt"
    "math/rand"
    "sync"
    "time"
)

// URLShortener represents the main service
type URLShortener struct {
    urlMap    map[string]string  // shortURL -> longURL
    analytics map[string]*Analytics
    mutex     sync.RWMutex
    baseURL   string
}

// Analytics tracks URL usage statistics
type Analytics struct {
    ShortURL     string
    LongURL      string
    CreatedAt    time.Time
    AccessCount  int64
    LastAccessed time.Time
    mutex        sync.RWMutex
}

// NewURLShortener creates a new URL shortener service
func NewURLShortener(baseURL string) *URLShortener {
    return &URLShortener{
        urlMap:    make(map[string]string),
        analytics: make(map[string]*Analytics),
        baseURL:   baseURL,
    }
}

// ShortenURL creates a short URL from a long URL
func (us *URLShortener) ShortenURL(longURL string) string {
    us.mutex.Lock()
    defer us.mutex.Unlock()

    // Generate short code using MD5 hash
    hash := md5.Sum([]byte(longURL + time.Now().String()))
    shortCode := hex.EncodeToString(hash[:])[:8] // Use first 8 characters

    // Store mapping
    us.urlMap[shortCode] = longURL

    // Initialize analytics
    us.analytics[shortCode] = &Analytics{
        ShortURL:  shortCode,
        LongURL:   longURL,
        CreatedAt: time.Now(),
    }

    return us.baseURL + "/" + shortCode
}

// Redirect resolves short URL to long URL
func (us *URLShortener) Redirect(shortCode string) (string, error) {
    us.mutex.RLock()
    longURL, exists := us.urlMap[shortCode]
    us.mutex.RUnlock()

    if !exists {
        return "", fmt.Errorf("short URL not found")
    }

    // Update analytics
    us.updateAnalytics(shortCode)

    return longURL, nil
}

// updateAnalytics updates access statistics
func (us *URLShortener) updateAnalytics(shortCode string) {
    us.mutex.Lock()
    defer us.mutex.Unlock()

    if analytics, exists := us.analytics[shortCode]; exists {
        analytics.mutex.Lock()
        analytics.AccessCount++
        analytics.LastAccessed = time.Now()
        analytics.mutex.Unlock()
    }
}

// GetAnalytics returns analytics for a short URL
func (us *URLShortener) GetAnalytics(shortCode string) (*Analytics, error) {
    us.mutex.RLock()
    defer us.mutex.RUnlock()

    analytics, exists := us.analytics[shortCode]
    if !exists {
        return nil, fmt.Errorf("analytics not found")
    }

    return analytics, nil
}

func main() {
    shortener := NewURLShortener("https://short.ly")

    // Shorten URLs
    shortURL1 := shortener.ShortenURL("https://www.google.com/search?q=golang")
    shortURL2 := shortener.ShortenURL("https://github.com/golang/go")

    fmt.Printf("Shortened URLs:\n")
    fmt.Printf("URL 1: %s\n", shortURL1)
    fmt.Printf("URL 2: %s\n", shortURL2)

    // Simulate redirects
    for i := 0; i < 5; i++ {
        shortCode := shortURL1[len(shortURL1)-8:] // Extract short code
        longURL, err := shortener.Redirect(shortCode)
        if err == nil {
            fmt.Printf("Redirect: %s -> %s\n", shortCode, longURL)
        }
    }

    // Get analytics
    shortCode := shortURL1[len(shortURL1)-8:]
    analytics, err := shortener.GetAnalytics(shortCode)
    if err == nil {
        fmt.Printf("Analytics for %s: %d accesses\n", shortCode, analytics.AccessCount)
    }
}
```

**Key Design Decisions**:

- **Hash-based Shortening**: MD5 hash for consistent short codes
- **In-memory Storage**: Simple map for demonstration (use Redis/DB in production)
- **Analytics Tracking**: Real-time usage statistics
- **Thread Safety**: Mutex for concurrent access

**Scalability Considerations**:

- **Database**: Use distributed database (Cassandra) for URL storage
- **Caching**: Redis for frequently accessed URLs
- **Load Balancing**: Multiple instances behind load balancer
- **CDN**: Cache redirects at edge locations

#### **2. Design a Chat System (like WhatsApp)**

**Problem Statement**: Design a real-time chat system supporting 1M concurrent users.

**Requirements**:

- Send/receive messages in real-time
- Support group chats
- Message history
- Online/offline status
- Handle 1M concurrent users

**Solution**:

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// ChatSystem represents the main chat service
type ChatSystem struct {
    users      map[string]*User
    chats      map[string]*Chat
    connections map[string]*Connection
    mutex      sync.RWMutex
}

// User represents a chat user
type User struct {
    ID       string
    Username string
    Status   UserStatus
    LastSeen time.Time
    mutex    sync.RWMutex
}

// UserStatus represents user online/offline status
type UserStatus int

const (
    Online UserStatus = iota
    Offline
    Away
)

// Chat represents a chat room (1-on-1 or group)
type Chat struct {
    ID       string
    Type     ChatType
    Members  map[string]bool
    Messages []*Message
    mutex    sync.RWMutex
}

// ChatType represents the type of chat
type ChatType int

const (
    Direct ChatType = iota
    Group
)

// Message represents a chat message
type Message struct {
    ID        string
    ChatID    string
    SenderID  string
    Content   string
    Timestamp time.Time
    Type      MessageType
}

// MessageType represents the type of message
type MessageType int

const (
    Text MessageType = iota
    Image
    File
)

// Connection represents a WebSocket connection
type Connection struct {
    UserID     string
    Conn       interface{} // WebSocket connection
    LastPing   time.Time
    mutex      sync.RWMutex
}

// NewChatSystem creates a new chat system
func NewChatSystem() *ChatSystem {
    return &ChatSystem{
        users:      make(map[string]*User),
        chats:      make(map[string]*Chat),
        connections: make(map[string]*Connection),
    }
}

// RegisterUser registers a new user
func (cs *ChatSystem) RegisterUser(userID, username string) *User {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()

    user := &User{
        ID:       userID,
        Username: username,
        Status:   Offline,
        LastSeen: time.Now(),
    }

    cs.users[userID] = user
    return user
}

// CreateChat creates a new chat
func (cs *ChatSystem) CreateChat(chatID string, chatType ChatType, memberIDs []string) *Chat {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()

    members := make(map[string]bool)
    for _, memberID := range memberIDs {
        members[memberID] = true
    }

    chat := &Chat{
        ID:      chatID,
        Type:    chatType,
        Members: members,
        Messages: make([]*Message, 0),
    }

    cs.chats[chatID] = chat
    return chat
}

// SendMessage sends a message to a chat
func (cs *ChatSystem) SendMessage(chatID, senderID, content string) (*Message, error) {
    cs.mutex.RLock()
    chat, exists := cs.chats[chatID]
    cs.mutex.RUnlock()

    if !exists {
        return nil, fmt.Errorf("chat not found")
    }

    // Check if sender is member of chat
    chat.mutex.RLock()
    if !chat.Members[senderID] {
        chat.mutex.RUnlock()
        return nil, fmt.Errorf("user not member of chat")
    }
    chat.mutex.RUnlock()

    // Create message
    message := &Message{
        ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
        ChatID:    chatID,
        SenderID:  senderID,
        Content:   content,
        Timestamp: time.Now(),
        Type:      Text,
    }

    // Add message to chat
    chat.mutex.Lock()
    chat.Messages = append(chat.Messages, message)
    chat.mutex.Unlock()

    // Broadcast message to online members
    cs.broadcastMessage(chat, message)

    return message, nil
}

// broadcastMessage sends message to all online members
func (cs *ChatSystem) broadcastMessage(chat *Chat, message *Message) {
    chat.mutex.RLock()
    members := make([]string, 0, len(chat.Members))
    for memberID := range chat.Members {
        members = append(members, memberID)
    }
    chat.mutex.RUnlock()

    for _, memberID := range members {
        if memberID != message.SenderID {
            cs.sendToUser(memberID, message)
        }
    }
}

// sendToUser sends message to a specific user
func (cs *ChatSystem) sendToUser(userID string, message *Message) {
    cs.mutex.RLock()
    connection, exists := cs.connections[userID]
    cs.mutex.RUnlock()

    if exists && connection != nil {
        // In real implementation, send via WebSocket
        fmt.Printf("Sending message to user %s: %s\n", userID, message.Content)
    }
}

// GetChatHistory returns message history for a chat
func (cs *ChatSystem) GetChatHistory(chatID string, limit int) ([]*Message, error) {
    cs.mutex.RLock()
    chat, exists := cs.chats[chatID]
    cs.mutex.RUnlock()

    if !exists {
        return nil, fmt.Errorf("chat not found")
    }

    chat.mutex.RLock()
    defer chat.mutex.RUnlock()

    messages := chat.Messages
    if len(messages) > limit {
        messages = messages[len(messages)-limit:]
    }

    return messages, nil
}

// UpdateUserStatus updates user online/offline status
func (cs *ChatSystem) UpdateUserStatus(userID string, status UserStatus) {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()

    if user, exists := cs.users[userID]; exists {
        user.mutex.Lock()
        user.Status = status
        user.LastSeen = time.Now()
        user.mutex.Unlock()
    }
}

func main() {
    chatSystem := NewChatSystem()

    // Register users
    user1 := chatSystem.RegisterUser("user1", "Alice")
    user2 := chatSystem.RegisterUser("user2", "Bob")
    user3 := chatSystem.RegisterUser("user3", "Charlie")

    // Create group chat
    chat := chatSystem.CreateChat("chat1", Group, []string{"user1", "user2", "user3"})

    // Update user status
    chatSystem.UpdateUserStatus("user1", Online)
    chatSystem.UpdateUserStatus("user2", Online)

    // Send messages
    chatSystem.SendMessage("chat1", "user1", "Hello everyone!")
    chatSystem.SendMessage("chat1", "user2", "Hi Alice!")
    chatSystem.SendMessage("chat1", "user3", "Hey guys!")

    // Get chat history
    history, _ := chatSystem.GetChatHistory("chat1", 10)
    fmt.Println("Chat History:")
    for _, msg := range history {
        fmt.Printf("[%s] %s: %s\n", msg.Timestamp.Format("15:04:05"), msg.SenderID, msg.Content)
    }
}
```

**Key Design Decisions**:

- **WebSocket Connections**: Real-time bidirectional communication
- **Message Broadcasting**: Send to all online members
- **In-memory Storage**: Simple maps (use Redis/DB in production)
- **Thread Safety**: Mutex for concurrent access

**Scalability Considerations**:

- **Message Queues**: Use Apache Kafka for message delivery
- **Database**: Store messages in distributed database
- **Load Balancing**: Multiple chat servers behind load balancer
- **Caching**: Redis for active chat sessions

### **Low-Level Design (LLD) Questions**

#### **3. Design a Rate Limiter**

**Problem Statement**: Design a rate limiter that can handle different rate limiting strategies.

**Requirements**:

- Token bucket algorithm
- Sliding window algorithm
- Support multiple rate limiting strategies
- Thread-safe implementation

**Solution**:

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// RateLimiter interface for different rate limiting strategies
type RateLimiter interface {
    Allow(key string) bool
    GetRemainingTokens(key string) int
}

// TokenBucket implements token bucket rate limiting
type TokenBucket struct {
    capacity     int           // Maximum tokens
    refillRate   int           // Tokens per second
    tokens       map[string]*TokenBucketState
    mutex        sync.RWMutex
}

// TokenBucketState tracks token bucket state for a key
type TokenBucketState struct {
    tokens     int
    lastRefill time.Time
    mutex      sync.Mutex
}

// NewTokenBucket creates a new token bucket rate limiter
func NewTokenBucket(capacity, refillRate int) *TokenBucket {
    return &TokenBucket{
        capacity:   capacity,
        refillRate: refillRate,
        tokens:     make(map[string]*TokenBucketState),
    }
}

// Allow checks if request is allowed
func (tb *TokenBucket) Allow(key string) bool {
    tb.mutex.Lock()
    defer tb.mutex.Unlock()

    state, exists := tb.tokens[key]
    if !exists {
        state = &TokenBucketState{
            tokens:     tb.capacity,
            lastRefill: time.Now(),
        }
        tb.tokens[key] = state
    }

    state.mutex.Lock()
    defer state.mutex.Unlock()

    // Refill tokens based on time elapsed
    now := time.Now()
    elapsed := now.Sub(state.lastRefill)
    tokensToAdd := int(elapsed.Seconds()) * tb.refillRate

    if tokensToAdd > 0 {
        state.tokens = min(tb.capacity, state.tokens+tokensToAdd)
        state.lastRefill = now
    }

    // Check if tokens available
    if state.tokens > 0 {
        state.tokens--
        return true
    }

    return false
}

// GetRemainingTokens returns remaining tokens for a key
func (tb *TokenBucket) GetRemainingTokens(key string) int {
    tb.mutex.RLock()
    defer tb.mutex.RUnlock()

    state, exists := tb.tokens[key]
    if !exists {
        return tb.capacity
    }

    state.mutex.Lock()
    defer state.mutex.Unlock()

    // Refill tokens
    now := time.Now()
    elapsed := now.Sub(state.lastRefill)
    tokensToAdd := int(elapsed.Seconds()) * tb.refillRate

    if tokensToAdd > 0 {
        state.tokens = min(tb.capacity, state.tokens+tokensToAdd)
        state.lastRefill = now
    }

    return state.tokens
}

// SlidingWindow implements sliding window rate limiting
type SlidingWindow struct {
    windowSize time.Duration
    maxRequests int
    requests    map[string]*SlidingWindowState
    mutex       sync.RWMutex
}

// SlidingWindowState tracks sliding window state for a key
type SlidingWindowState struct {
    requests []time.Time
    mutex    sync.Mutex
}

// NewSlidingWindow creates a new sliding window rate limiter
func NewSlidingWindow(windowSize time.Duration, maxRequests int) *SlidingWindow {
    return &SlidingWindow{
        windowSize:   windowSize,
        maxRequests:  maxRequests,
        requests:     make(map[string]*SlidingWindowState),
    }
}

// Allow checks if request is allowed
func (sw *SlidingWindow) Allow(key string) bool {
    sw.mutex.Lock()
    defer sw.mutex.Unlock()

    state, exists := sw.requests[key]
    if !exists {
        state = &SlidingWindowState{
            requests: make([]time.Time, 0),
        }
        sw.requests[key] = state
    }

    state.mutex.Lock()
    defer state.mutex.Unlock()

    now := time.Now()
    windowStart := now.Add(-sw.windowSize)

    // Remove old requests outside the window
    validRequests := make([]time.Time, 0)
    for _, reqTime := range state.requests {
        if reqTime.After(windowStart) {
            validRequests = append(validRequests, reqTime)
        }
    }
    state.requests = validRequests

    // Check if under limit
    if len(state.requests) < sw.maxRequests {
        state.requests = append(state.requests, now)
        return true
    }

    return false
}

// GetRemainingTokens returns remaining requests for a key
func (sw *SlidingWindow) GetRemainingTokens(key string) int {
    sw.mutex.RLock()
    defer sw.mutex.RUnlock()

    state, exists := sw.requests[key]
    if !exists {
        return sw.maxRequests
    }

    state.mutex.Lock()
    defer state.mutex.Unlock()

    now := time.Now()
    windowStart := now.Add(-sw.windowSize)

    // Count valid requests
    validCount := 0
    for _, reqTime := range state.requests {
        if reqTime.After(windowStart) {
            validCount++
        }
    }

    return max(0, sw.maxRequests-validCount)
}

// RateLimiterFactory creates rate limiters
type RateLimiterFactory struct{}

// CreateRateLimiter creates a rate limiter based on type
func (rf *RateLimiterFactory) CreateRateLimiter(limiterType string, params map[string]interface{}) RateLimiter {
    switch limiterType {
    case "token_bucket":
        capacity := params["capacity"].(int)
        refillRate := params["refill_rate"].(int)
        return NewTokenBucket(capacity, refillRate)
    case "sliding_window":
        windowSize := params["window_size"].(time.Duration)
        maxRequests := params["max_requests"].(int)
        return NewSlidingWindow(windowSize, maxRequests)
    default:
        return nil
    }
}

// Helper functions
func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func main() {
    factory := &RateLimiterFactory{}

    // Create token bucket rate limiter
    tokenBucketParams := map[string]interface{}{
        "capacity":    10,
        "refill_rate": 2,
    }
    tokenBucket := factory.CreateRateLimiter("token_bucket", tokenBucketParams)

    // Create sliding window rate limiter
    slidingWindowParams := map[string]interface{}{
        "window_size":   time.Minute,
        "max_requests":  5,
    }
    slidingWindow := factory.CreateRateLimiter("sliding_window", slidingWindowParams)

    // Test token bucket
    fmt.Println("Token Bucket Rate Limiter:")
    for i := 0; i < 15; i++ {
        allowed := tokenBucket.Allow("user1")
        remaining := tokenBucket.GetRemainingTokens("user1")
        fmt.Printf("Request %d: Allowed=%t, Remaining=%d\n", i+1, allowed, remaining)
        time.Sleep(100 * time.Millisecond)
    }

    fmt.Println("\nSliding Window Rate Limiter:")
    for i := 0; i < 10; i++ {
        allowed := slidingWindow.Allow("user2")
        remaining := slidingWindow.GetRemainingTokens("user2")
        fmt.Printf("Request %d: Allowed=%t, Remaining=%d\n", i+1, allowed, remaining)
        time.Sleep(200 * time.Millisecond)
    }
}
```

**Key Design Decisions**:

- **Interface-based Design**: RateLimiter interface for different strategies
- **Thread Safety**: Mutex for concurrent access
- **Factory Pattern**: Create rate limiters based on type
- **State Management**: Track state for each key

**Rate Limiting Strategies**:

- **Token Bucket**: Refills tokens at fixed rate
- **Sliding Window**: Tracks requests in time window
- **Fixed Window**: Simple counter in time window
- **Leaky Bucket**: Smooths burst traffic

#### **4. Design a Distributed Cache**

**Problem Statement**: Design a distributed cache system with consistent hashing and replication.

**Requirements**:

- Consistent hashing for key distribution
- Replication for fault tolerance
- TTL support for expiration
- Thread-safe operations

**Solution**:

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sync"
    "time"
)

// DistributedCache represents the main cache system
type DistributedCache struct {
    nodes     []*CacheNode
    replicas  int
    ring      *ConsistentHashRing
    mutex     sync.RWMutex
}

// CacheNode represents a cache node
type CacheNode struct {
    ID       string
    Address  string
    Data     map[string]*CacheItem
    mutex    sync.RWMutex
}

// CacheItem represents a cached item
type CacheItem struct {
    Value     interface{}
    ExpiresAt time.Time
    mutex     sync.RWMutex
}

// ConsistentHashRing implements consistent hashing
type ConsistentHashRing struct {
    nodes    map[uint32]string
    sortedKeys []uint32
    mutex    sync.RWMutex
}

// NewDistributedCache creates a new distributed cache
func NewDistributedCache(replicas int) *DistributedCache {
    return &DistributedCache{
        nodes:    make([]*CacheNode, 0),
        replicas: replicas,
        ring:     NewConsistentHashRing(),
    }
}

// AddNode adds a new cache node
func (dc *DistributedCache) AddNode(nodeID, address string) {
    dc.mutex.Lock()
    defer dc.mutex.Unlock()

    node := &CacheNode{
        ID:      nodeID,
        Address: address,
        Data:    make(map[string]*CacheItem),
    }

    dc.nodes = append(dc.nodes, node)
    dc.ring.AddNode(nodeID, dc.replicas)
}

// RemoveNode removes a cache node
func (dc *DistributedCache) RemoveNode(nodeID string) {
    dc.mutex.Lock()
    defer dc.mutex.Unlock()

    // Remove from nodes slice
    for i, node := range dc.nodes {
        if node.ID == nodeID {
            dc.nodes = append(dc.nodes[:i], dc.nodes[i+1:]...)
            break
        }
    }

    dc.ring.RemoveNode(nodeID, dc.replicas)
}

// Get retrieves a value from cache
func (dc *DistributedCache) Get(key string) (interface{}, bool) {
    dc.mutex.RLock()
    defer dc.mutex.RUnlock()

    // Find primary node
    primaryNodeID := dc.ring.GetNode(key)
    primaryNode := dc.getNodeByID(primaryNodeID)

    if primaryNode == nil {
        return nil, false
    }

    // Try primary node first
    if value, found := dc.getFromNode(primaryNode, key); found {
        return value, true
    }

    // Try replica nodes
    replicaNodes := dc.ring.GetReplicaNodes(key, dc.replicas-1)
    for _, nodeID := range replicaNodes {
        if nodeID != primaryNodeID {
            node := dc.getNodeByID(nodeID)
            if node != nil {
                if value, found := dc.getFromNode(node, key); found {
                    return value, true
                }
            }
        }
    }

    return nil, false
}

// Set stores a value in cache
func (dc *DistributedCache) Set(key string, value interface{}, ttl time.Duration) {
    dc.mutex.RLock()
    defer dc.mutex.RUnlock()

    // Find nodes for replication
    nodes := dc.ring.GetReplicaNodes(key, dc.replicas)

    for _, nodeID := range nodes {
        node := dc.getNodeByID(nodeID)
        if node != nil {
            dc.setInNode(node, key, value, ttl)
        }
    }
}

// getFromNode retrieves value from a specific node
func (dc *DistributedCache) getFromNode(node *CacheNode, key string) (interface{}, bool) {
    node.mutex.RLock()
    defer node.mutex.RUnlock()

    item, exists := node.Data[key]
    if !exists {
        return nil, false
    }

    item.mutex.RLock()
    defer item.mutex.RUnlock()

    // Check expiration
    if time.Now().After(item.ExpiresAt) {
        // Remove expired item
        node.mutex.RUnlock()
        node.mutex.Lock()
        delete(node.Data, key)
        node.mutex.Unlock()
        node.mutex.RLock()
        return nil, false
    }

    return item.Value, true
}

// setInNode stores value in a specific node
func (dc *DistributedCache) setInNode(node *CacheNode, key string, value interface{}, ttl time.Duration) {
    node.mutex.Lock()
    defer node.mutex.Unlock()

    item := &CacheItem{
        Value:     value,
        ExpiresAt: time.Now().Add(ttl),
    }

    node.Data[key] = item
}

// getNodeByID finds node by ID
func (dc *DistributedCache) getNodeByID(nodeID string) *CacheNode {
    for _, node := range dc.nodes {
        if node.ID == nodeID {
            return node
        }
    }
    return nil
}

// NewConsistentHashRing creates a new consistent hash ring
func NewConsistentHashRing() *ConsistentHashRing {
    return &ConsistentHashRing{
        nodes: make(map[uint32]string),
    }
}

// AddNode adds a node to the hash ring
func (chr *ConsistentHashRing) AddNode(nodeID string, replicas int) {
    chr.mutex.Lock()
    defer chr.mutex.Unlock()

    for i := 0; i < replicas; i++ {
        hash := chr.hash(fmt.Sprintf("%s:%d", nodeID, i))
        chr.nodes[hash] = nodeID
    }

    chr.updateSortedKeys()
}

// RemoveNode removes a node from the hash ring
func (chr *ConsistentHashRing) RemoveNode(nodeID string, replicas int) {
    chr.mutex.Lock()
    defer chr.mutex.Unlock()

    for i := 0; i < replicas; i++ {
        hash := chr.hash(fmt.Sprintf("%s:%d", nodeID, i))
        delete(chr.nodes, hash)
    }

    chr.updateSortedKeys()
}

// GetNode returns the node for a key
func (chr *ConsistentHashRing) GetNode(key string) string {
    chr.mutex.RLock()
    defer chr.mutex.RUnlock()

    if len(chr.nodes) == 0 {
        return ""
    }

    hash := chr.hash(key)

    // Find first node with hash >= key hash
    for _, nodeHash := range chr.sortedKeys {
        if nodeHash >= hash {
            return chr.nodes[nodeHash]
        }
    }

    // Wrap around to first node
    return chr.nodes[chr.sortedKeys[0]]
}

// GetReplicaNodes returns replica nodes for a key
func (chr *ConsistentHashRing) GetReplicaNodes(key string, count int) []string {
    chr.mutex.RLock()
    defer chr.mutex.RUnlock()

    if len(chr.nodes) == 0 {
        return []string{}
    }

    hash := chr.hash(key)
    nodes := make([]string, 0, count)
    seen := make(map[string]bool)

    // Find starting position
    startIndex := 0
    for i, nodeHash := range chr.sortedKeys {
        if nodeHash >= hash {
            startIndex = i
            break
        }
    }

    // Collect unique nodes
    for i := 0; i < len(chr.sortedKeys) && len(nodes) < count; i++ {
        index := (startIndex + i) % len(chr.sortedKeys)
        nodeHash := chr.sortedKeys[index]
        nodeID := chr.nodes[nodeHash]

        if !seen[nodeID] {
            nodes = append(nodes, nodeID)
            seen[nodeID] = true
        }
    }

    return nodes
}

// hash generates a hash for a key
func (chr *ConsistentHashRing) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

// updateSortedKeys updates the sorted keys array
func (chr *ConsistentHashRing) updateSortedKeys() {
    chr.sortedKeys = make([]uint32, 0, len(chr.nodes))
    for hash := range chr.nodes {
        chr.sortedKeys = append(chr.sortedKeys, hash)
    }

    // Sort keys
    for i := 0; i < len(chr.sortedKeys); i++ {
        for j := i + 1; j < len(chr.sortedKeys); j++ {
            if chr.sortedKeys[i] > chr.sortedKeys[j] {
                chr.sortedKeys[i], chr.sortedKeys[j] = chr.sortedKeys[j], chr.sortedKeys[i]
            }
        }
    }
}

func main() {
    // Create distributed cache
    cache := NewDistributedCache(3) // 3 replicas

    // Add nodes
    cache.AddNode("node1", "192.168.1.1:8080")
    cache.AddNode("node2", "192.168.1.2:8080")
    cache.AddNode("node3", "192.168.1.3:8080")

    // Set values
    cache.Set("key1", "value1", 5*time.Minute)
    cache.Set("key2", "value2", 10*time.Minute)
    cache.Set("key3", "value3", 15*time.Minute)

    // Get values
    if value, found := cache.Get("key1"); found {
        fmt.Printf("key1: %v\n", value)
    }

    if value, found := cache.Get("key2"); found {
        fmt.Printf("key2: %v\n", value)
    }

    if value, found := cache.Get("key3"); found {
        fmt.Printf("key3: %v\n", value)
    }

    // Test node removal
    fmt.Println("\nRemoving node2...")
    cache.RemoveNode("node2")

    // Try to get values after node removal
    if value, found := cache.Get("key1"); found {
        fmt.Printf("key1 after node removal: %v\n", value)
    }
}
```

**Key Design Decisions**:

- **Consistent Hashing**: Even distribution of keys across nodes
- **Replication**: Multiple copies for fault tolerance
- **TTL Support**: Automatic expiration of cached items
- **Thread Safety**: Mutex for concurrent access

**Scalability Features**:

- **Horizontal Scaling**: Add/remove nodes dynamically
- **Load Distribution**: Consistent hashing ensures even load
- **Fault Tolerance**: Replication handles node failures
- **Memory Management**: TTL prevents memory leaks

#### **5. Design a Payment Gateway (like Razorpay)**

**Problem Statement**: Design a payment gateway system that can handle millions of transactions with high availability and security.

**Requirements**:

- Process payments from multiple payment methods (cards, UPI, net banking)
- Handle 1M+ transactions per day
- 99.99% availability
- Real-time fraud detection
- PCI DSS compliance
- Support for refunds and settlements

**High-Level Architecture**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web/Mobile    │    │   Merchant      │    │   Bank/UPI      │
│   Application   │    │   Dashboard     │    │   Networks      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Load Balancer                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway                                  │
│  • Authentication  • Rate Limiting  • Request Validation       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                Payment Processing Service                       │
│  • Transaction Management  • Payment Method Routing            │
│  • Fraud Detection  • Settlement Processing                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Database Layer                               │
│  • Transaction DB  • User DB  • Merchant DB  • Audit Logs     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                External Integrations                            │
│  • Bank APIs  • UPI Networks  • Card Networks  • Fraud APIs    │
└─────────────────────────────────────────────────────────────────┘
```

**Solution**:

```go
package main

import (
    "crypto/rand"
    "encoding/json"
    "fmt"
    "math/big"
    "sync"
    "time"
)

// PaymentGateway represents the main payment gateway service
type PaymentGateway struct {
    transactions    map[string]*Transaction
    merchants       map[string]*Merchant
    paymentMethods  map[string]*PaymentMethod
    fraudDetector   *FraudDetector
    settlementEngine *SettlementEngine
    mutex           sync.RWMutex
}

// Transaction represents a payment transaction
type Transaction struct {
    ID              string            `json:"id"`
    MerchantID      string            `json:"merchant_id"`
    Amount          int64             `json:"amount"` // Amount in paise
    Currency        string            `json:"currency"`
    PaymentMethod   string            `json:"payment_method"`
    Status          TransactionStatus `json:"status"`
    CreatedAt       time.Time         `json:"created_at"`
    UpdatedAt       time.Time         `json:"updated_at"`
    PaymentDetails  *PaymentDetails   `json:"payment_details"`
    FraudScore      float64           `json:"fraud_score"`
    RefundAmount    int64             `json:"refund_amount"`
    SettlementID    string            `json:"settlement_id"`
    mutex           sync.RWMutex
}

// TransactionStatus represents the status of a transaction
type TransactionStatus int

const (
    Pending TransactionStatus = iota
    Processing
    Success
    Failed
    Refunded
    PartiallyRefunded
)

// PaymentDetails contains payment method specific details
type PaymentDetails struct {
    CardNumber    string `json:"card_number,omitempty"`
    CardType      string `json:"card_type,omitempty"`
    UPIID         string `json:"upi_id,omitempty"`
    BankCode      string `json:"bank_code,omitempty"`
    WalletType    string `json:"wallet_type,omitempty"`
    NetBankingCode string `json:"net_banking_code,omitempty"`
}

// Merchant represents a merchant account
type Merchant struct {
    ID              string    `json:"id"`
    Name            string    `json:"name"`
    Email           string    `json:"email"`
    APIKey          string    `json:"api_key"`
    WebhookURL      string    `json:"webhook_url"`
    IsActive        bool      `json:"is_active"`
    CreatedAt       time.Time `json:"created_at"`
    mutex           sync.RWMutex
}

// PaymentMethod represents a payment method
type PaymentMethod struct {
    ID          string `json:"id"`
    Name        string `json:"name"`
    Type        string `json:"type"` // card, upi, netbanking, wallet
    IsActive    bool   `json:"is_active"`
    FeePercent  float64 `json:"fee_percent"`
    mutex       sync.RWMutex
}

// FraudDetector handles fraud detection
type FraudDetector struct {
    rules    []FraudRule
    mutex    sync.RWMutex
}

// FraudRule represents a fraud detection rule
type FraudRule struct {
    ID          string
    Name        string
    Condition   func(*Transaction) bool
    Score       float64
    IsActive    bool
}

// SettlementEngine handles settlement processing
type SettlementEngine struct {
    settlements map[string]*Settlement
    mutex       sync.RWMutex
}

// Settlement represents a settlement record
type Settlement struct {
    ID              string    `json:"id"`
    MerchantID      string    `json:"merchant_id"`
    Amount          int64     `json:"amount"`
    Fee             int64     `json:"fee"`
    NetAmount       int64     `json:"net_amount"`
    Status          string    `json:"status"`
    SettlementDate  time.Time `json:"settlement_date"`
    mutex           sync.RWMutex
}

// NewPaymentGateway creates a new payment gateway
func NewPaymentGateway() *PaymentGateway {
    return &PaymentGateway{
        transactions:    make(map[string]*Transaction),
        merchants:       make(map[string]*Merchant),
        paymentMethods:  make(map[string]*PaymentMethod),
        fraudDetector:   NewFraudDetector(),
        settlementEngine: NewSettlementEngine(),
    }
}

// RegisterMerchant registers a new merchant
func (pg *PaymentGateway) RegisterMerchant(name, email, webhookURL string) *Merchant {
    pg.mutex.Lock()
    defer pg.mutex.Unlock()

    merchantID := pg.generateID()
    apiKey := pg.generateAPIKey()

    merchant := &Merchant{
        ID:         merchantID,
        Name:       name,
        Email:      email,
        APIKey:     apiKey,
        WebhookURL: webhookURL,
        IsActive:   true,
        CreatedAt:  time.Now(),
    }

    pg.merchants[merchantID] = merchant
    return merchant
}

// CreatePayment creates a new payment transaction
func (pg *PaymentGateway) CreatePayment(merchantID string, amount int64, currency string, paymentMethod string, paymentDetails *PaymentDetails) (*Transaction, error) {
    pg.mutex.Lock()
    defer pg.mutex.Unlock()

    // Validate merchant
    merchant, exists := pg.merchants[merchantID]
    if !exists || !merchant.IsActive {
        return nil, fmt.Errorf("invalid or inactive merchant")
    }

    // Validate payment method
    method, exists := pg.paymentMethods[paymentMethod]
    if !exists || !method.IsActive {
        return nil, fmt.Errorf("invalid or inactive payment method")
    }

    // Create transaction
    transaction := &Transaction{
        ID:             pg.generateID(),
        MerchantID:     merchantID,
        Amount:         amount,
        Currency:       currency,
        PaymentMethod:  paymentMethod,
        Status:         Pending,
        CreatedAt:      time.Now(),
        UpdatedAt:      time.Now(),
        PaymentDetails: paymentDetails,
    }

    pg.transactions[transaction.ID] = transaction

    // Process payment asynchronously
    go pg.processPayment(transaction)

    return transaction, nil
}

// processPayment processes a payment transaction
func (pg *PaymentGateway) processPayment(transaction *Transaction) {
    transaction.mutex.Lock()
    transaction.Status = Processing
    transaction.UpdatedAt = time.Now()
    transaction.mutex.Unlock()

    // Fraud detection
    fraudScore := pg.fraudDetector.DetectFraud(transaction)
    transaction.mutex.Lock()
    transaction.FraudScore = fraudScore
    transaction.mutex.Unlock()

    // If fraud score is high, decline transaction
    if fraudScore > 0.8 {
        transaction.mutex.Lock()
        transaction.Status = Failed
        transaction.UpdatedAt = time.Now()
        transaction.mutex.Unlock()
        return
    }

    // Simulate payment processing
    time.Sleep(100 * time.Millisecond)

    // Simulate success/failure based on some logic
    success := pg.simulatePaymentProcessing(transaction)

    transaction.mutex.Lock()
    if success {
        transaction.Status = Success
    } else {
        transaction.Status = Failed
    }
    transaction.UpdatedAt = time.Now()
    transaction.mutex.Unlock()

    // Send webhook notification
    pg.sendWebhookNotification(transaction)

    // Process settlement if successful
    if success {
        go pg.settlementEngine.ProcessSettlement(transaction)
    }
}

// simulatePaymentProcessing simulates payment processing
func (pg *PaymentGateway) simulatePaymentProcessing(transaction *Transaction) bool {
    // Simulate 95% success rate
    randNum, _ := rand.Int(rand.Reader, big.NewInt(100))
    return randNum.Int64() < 95
}

// sendWebhookNotification sends webhook notification to merchant
func (pg *PaymentGateway) sendWebhookNotification(transaction *Transaction) {
    pg.mutex.RLock()
    merchant, exists := pg.merchants[transaction.MerchantID]
    pg.mutex.RUnlock()

    if !exists || merchant.WebhookURL == "" {
        return
    }

    // In real implementation, send HTTP POST to webhook URL
    fmt.Printf("Sending webhook to %s for transaction %s\n", merchant.WebhookURL, transaction.ID)
}

// GetTransaction retrieves a transaction by ID
func (pg *PaymentGateway) GetTransaction(transactionID string) (*Transaction, error) {
    pg.mutex.RLock()
    defer pg.mutex.RUnlock()

    transaction, exists := pg.transactions[transactionID]
    if !exists {
        return nil, fmt.Errorf("transaction not found")
    }

    return transaction, nil
}

// RefundTransaction processes a refund
func (pg *PaymentGateway) RefundTransaction(transactionID string, refundAmount int64) error {
    pg.mutex.Lock()
    defer pg.mutex.Unlock()

    transaction, exists := pg.transactions[transactionID]
    if !exists {
        return fmt.Errorf("transaction not found")
    }

    transaction.mutex.Lock()
    defer transaction.mutex.Unlock()

    if transaction.Status != Success {
        return fmt.Errorf("can only refund successful transactions")
    }

    if refundAmount > transaction.Amount {
        return fmt.Errorf("refund amount cannot exceed transaction amount")
    }

    if transaction.RefundAmount+refundAmount > transaction.Amount {
        return fmt.Errorf("total refund amount cannot exceed transaction amount")
    }

    transaction.RefundAmount += refundAmount
    transaction.UpdatedAt = time.Now()

    if transaction.RefundAmount == transaction.Amount {
        transaction.Status = Refunded
    } else {
        transaction.Status = PartiallyRefunded
    }

    return nil
}

// NewFraudDetector creates a new fraud detector
func NewFraudDetector() *FraudDetector {
    fd := &FraudDetector{
        rules: make([]FraudRule, 0),
    }

    // Add fraud detection rules
    fd.AddRule(FraudRule{
        ID:   "high_amount",
        Name: "High Amount Transaction",
        Condition: func(t *Transaction) bool {
            return t.Amount > 100000 // More than ₹1000
        },
        Score:    0.3,
        IsActive: true,
    })

    fd.AddRule(FraudRule{
        ID:   "multiple_failed",
        Name: "Multiple Failed Transactions",
        Condition: func(t *Transaction) bool {
            // In real implementation, check merchant's failed transaction history
            return false
        },
        Score:    0.5,
        IsActive: true,
    })

    return fd
}

// AddRule adds a fraud detection rule
func (fd *FraudDetector) AddRule(rule FraudRule) {
    fd.mutex.Lock()
    defer fd.mutex.Unlock()

    fd.rules = append(fd.rules, rule)
}

// DetectFraud detects fraud in a transaction
func (fd *FraudDetector) DetectFraud(transaction *Transaction) float64 {
    fd.mutex.RLock()
    defer fd.mutex.RUnlock()

    totalScore := 0.0

    for _, rule := range fd.rules {
        if rule.IsActive && rule.Condition(transaction) {
            totalScore += rule.Score
        }
    }

    return totalScore
}

// NewSettlementEngine creates a new settlement engine
func NewSettlementEngine() *SettlementEngine {
    return &SettlementEngine{
        settlements: make(map[string]*Settlement),
    }
}

// ProcessSettlement processes settlement for a transaction
func (se *SettlementEngine) ProcessSettlement(transaction *Transaction) {
    se.mutex.Lock()
    defer se.mutex.Unlock()

    settlementID := se.generateID()
    fee := int64(float64(transaction.Amount) * 0.02) // 2% fee
    netAmount := transaction.Amount - fee

    settlement := &Settlement{
        ID:             settlementID,
        MerchantID:     transaction.MerchantID,
        Amount:         transaction.Amount,
        Fee:            fee,
        NetAmount:      netAmount,
        Status:         "pending",
        SettlementDate: time.Now().Add(24 * time.Hour), // T+1 settlement
    }

    se.settlements[settlementID] = settlement

    // Update transaction with settlement ID
    transaction.mutex.Lock()
    transaction.SettlementID = settlementID
    transaction.mutex.Unlock()

    fmt.Printf("Settlement %s created for transaction %s, amount: ₹%.2f\n",
        settlementID, transaction.ID, float64(netAmount)/100)
}

// Helper functions
func (pg *PaymentGateway) generateID() string {
    randNum, _ := rand.Int(rand.Reader, big.NewInt(1000000))
    return fmt.Sprintf("txn_%d", randNum.Int64())
}

func (pg *PaymentGateway) generateAPIKey() string {
    randNum, _ := rand.Int(rand.Reader, big.NewInt(1000000000))
    return fmt.Sprintf("rzp_test_%d", randNum.Int64())
}

func (se *SettlementEngine) generateID() string {
    randNum, _ := rand.Int(rand.Reader, big.NewInt(1000000))
    return fmt.Sprintf("settle_%d", randNum.Int64())
}

func main() {
    // Create payment gateway
    pg := NewPaymentGateway()

    // Register payment methods
    pg.paymentMethods["card"] = &PaymentMethod{
        ID:         "card",
        Name:       "Credit/Debit Card",
        Type:       "card",
        IsActive:   true,
        FeePercent: 2.0,
    }

    pg.paymentMethods["upi"] = &PaymentMethod{
        ID:         "upi",
        Name:       "UPI",
        Type:       "upi",
        IsActive:   true,
        FeePercent: 0.5,
    }

    // Register merchant
    merchant := pg.RegisterMerchant("Test Store", "test@store.com", "https://store.com/webhook")
    fmt.Printf("Registered merchant: %s with API key: %s\n", merchant.Name, merchant.APIKey)

    // Create payment
    paymentDetails := &PaymentDetails{
        CardNumber: "4111111111111111",
        CardType:   "visa",
    }

    transaction, err := pg.CreatePayment(merchant.ID, 5000, "INR", "card", paymentDetails)
    if err != nil {
        fmt.Printf("Error creating payment: %v\n", err)
        return
    }

    fmt.Printf("Created transaction: %s, amount: ₹%.2f\n", transaction.ID, float64(transaction.Amount)/100)

    // Wait for processing
    time.Sleep(200 * time.Millisecond)

    // Get transaction status
    updatedTransaction, _ := pg.GetTransaction(transaction.ID)
    fmt.Printf("Transaction status: %d, fraud score: %.2f\n", updatedTransaction.Status, updatedTransaction.FraudScore)

    // Process refund if successful
    if updatedTransaction.Status == Success {
        err = pg.RefundTransaction(transaction.ID, 2500) // Partial refund
        if err != nil {
            fmt.Printf("Error processing refund: %v\n", err)
        } else {
            fmt.Printf("Refund processed for transaction %s\n", transaction.ID)
        }
    }
}
```

**Key Design Decisions**:

- **Microservices Architecture**: Separate services for different concerns
- **Asynchronous Processing**: Non-blocking payment processing
- **Fraud Detection**: Real-time fraud scoring with multiple rules
- **Webhook Notifications**: Event-driven architecture for merchant notifications
- **Settlement Engine**: Automated T+1 settlement processing
- **Thread Safety**: Mutex for concurrent access

**Scalability Considerations**:

- **Database Sharding**: Shard transactions by merchant ID
- **Message Queues**: Use Kafka for payment processing events
- **Caching**: Redis for frequently accessed merchant data
- **Load Balancing**: Multiple payment processing instances
- **Circuit Breakers**: Handle external API failures gracefully

**Security Features**:

- **PCI DSS Compliance**: Secure handling of card data
- **API Authentication**: API key-based authentication
- **Fraud Detection**: Multiple fraud detection rules
- **Audit Logging**: Complete transaction audit trail
- **Encryption**: Encrypt sensitive data at rest and in transit

---

## 📚 Additional Resources

### **Books**

- [System Design Interview](https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF/) - Alex Xu
- [Designing Data-Intensive Applications](https://dataintensive.net/) - Martin Kleppmann
- [Building Microservices](https://www.oreilly.com/library/view/building-microservices/9781491950340/) - Sam Newman

### **Online Resources**

- [System Design Primer](https://github.com/donnemartin/system-design-primer/) - GitHub repository
- [High Scalability](https://highscalability.com/) - Real-world system design case studies
- [AWS Architecture Center](https://aws.amazon.com/architecture/) - Cloud architecture patterns

### **Video Resources**

- [ByteByteGo](https://www.youtube.com/c/ByteByteGo/) - System design explanations
- [Gaurav Sen](https://www.youtube.com/c/GauravSensei/) - System design and algorithms
- [Exponent](https://www.youtube.com/c/ExponentTV/) - Mock interviews and system design

---

_This comprehensive guide covers essential system design concepts with practical Go implementations and real-world interview questions from top tech companies._
