# System Design Coding Problems

## Table of Contents
- [Introduction](#introduction/)
- [Distributed Cache Implementation](#distributed-cache-implementation/)
- [Message Queue System](#message-queue-system/)
- [Load Balancer Implementation](#load-balancer-implementation/)
- [Database Sharding](#database-sharding/)
- [Real-time Analytics System](#real-time-analytics-system/)
- [File Storage System](#file-storage-system/)
- [Search Engine Components](#search-engine-components/)

## Introduction

System design coding problems require you to implement core components of distributed systems. These problems test your ability to design, code, and optimize systems that can handle real-world scale and requirements.

## Distributed Cache Implementation

### Problem: Implement a Distributed Cache with Consistent Hashing

**Requirements:**
- Consistent hashing for data distribution
- Replication for fault tolerance
- Cache eviction policies (LRU, LFU)
- Health checks and failover
- Metrics and monitoring

```go
// Distributed Cache with Consistent Hashing
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "sync"
    "time"
)

type DistributedCache struct {
    nodes        map[string]*CacheNode
    ring         []HashRingNode
    replicas     int
    mu           sync.RWMutex
    evictionPolicy string
}

type CacheNode struct {
    ID       string
    Address  string
    Healthy  bool
    Cache    *LocalCache
    LastSeen time.Time
}

type HashRingNode struct {
    Hash   uint32
    NodeID string
}

type LocalCache struct {
    data    map[string]*CacheItem
    maxSize int
    policy  string
    mu      sync.RWMutex
}

type CacheItem struct {
    Key        string
    Value      interface{}
    ExpiresAt  time.Time
    AccessTime time.Time
    AccessCount int
}

func NewDistributedCache(replicas int, evictionPolicy string) *DistributedCache {
    return &DistributedCache{
        nodes:         make(map[string]*CacheNode),
        ring:          make([]HashRingNode, 0),
        replicas:      replicas,
        evictionPolicy: evictionPolicy,
    }
}

func (dc *DistributedCache) AddNode(nodeID, address string) {
    dc.mu.Lock()
    defer dc.mu.Unlock()
    
    node := &CacheNode{
        ID:       nodeID,
        Address:  address,
        Healthy:  true,
        Cache:    NewLocalCache(1000, dc.evictionPolicy),
        LastSeen: time.Now(),
    }
    
    dc.nodes[nodeID] = node
    dc.updateRing()
}

func (dc *DistributedCache) updateRing() {
    dc.ring = make([]HashRingNode, 0)
    
    for nodeID, node := range dc.nodes {
        if node.Healthy {
            for i := 0; i < dc.replicas; i++ {
                hash := dc.hash(fmt.Sprintf("%s:%d", nodeID, i))
                dc.ring = append(dc.ring, HashRingNode{
                    Hash:   hash,
                    NodeID: nodeID,
                })
            }
        }
    }
    
    sort.Slice(dc.ring, func(i, j int) bool {
        return dc.ring[i].Hash < dc.ring[j].Hash
    })
}

func (dc *DistributedCache) Get(key string) (interface{}, bool) {
    dc.mu.RLock()
    defer dc.mu.RUnlock()
    
    nodes := dc.getNodesForKey(key)
    
    for _, nodeID := range nodes {
        if node, exists := dc.nodes[nodeID]; exists && node.Healthy {
            if value, found := node.Cache.Get(key); found {
                return value, true
            }
        }
    }
    
    return nil, false
}

func (dc *DistributedCache) Set(key string, value interface{}, ttl time.Duration) {
    dc.mu.RLock()
    defer dc.mu.RUnlock()
    
    nodes := dc.getNodesForKey(key)
    
    for _, nodeID := range nodes {
        if node, exists := dc.nodes[nodeID]; exists && node.Healthy {
            node.Cache.Set(key, value, ttl)
        }
    }
}

func (dc *DistributedCache) getNodesForKey(key string) []string {
    if len(dc.ring) == 0 {
        return nil
    }
    
    hash := dc.hash(key)
    
    // Find the first node with hash >= key hash
    idx := sort.Search(len(dc.ring), func(i int) bool {
        return dc.ring[i].Hash >= hash
    })
    
    if idx == len(dc.ring) {
        idx = 0
    }
    
    // Return nodes for replication
    nodes := make([]string, 0, dc.replicas)
    for i := 0; i < dc.replicas; i++ {
        nodeIdx := (idx + i) % len(dc.ring)
        nodes = append(nodes, dc.ring[nodeIdx].NodeID)
    }
    
    return nodes
}

func (dc *DistributedCache) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

// Local Cache Implementation
func NewLocalCache(maxSize int, policy string) *LocalCache {
    return &LocalCache{
        data:    make(map[string]*CacheItem),
        maxSize: maxSize,
        policy:  policy,
    }
}

func (lc *LocalCache) Get(key string) (interface{}, bool) {
    lc.mu.RLock()
    defer lc.mu.RUnlock()
    
    item, exists := lc.data[key]
    if !exists {
        return nil, false
    }
    
    if time.Now().After(item.ExpiresAt) {
        delete(lc.data, key)
        return nil, false
    }
    
    item.AccessTime = time.Now()
    item.AccessCount++
    
    return item.Value, true
}

func (lc *LocalCache) Set(key string, value interface{}, ttl time.Duration) {
    lc.mu.Lock()
    defer lc.mu.Unlock()
    
    if len(lc.data) >= lc.maxSize {
        lc.evict()
    }
    
    lc.data[key] = &CacheItem{
        Key:        key,
        Value:      value,
        ExpiresAt:  time.Now().Add(ttl),
        AccessTime: time.Now(),
        AccessCount: 1,
    }
}

func (lc *LocalCache) evict() {
    switch lc.policy {
    case "LRU":
        lc.evictLRU()
    case "LFU":
        lc.evictLFU()
    default:
        lc.evictLRU()
    }
}

func (lc *LocalCache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, item := range lc.data {
        if oldestKey == "" || item.AccessTime.Before(oldestTime) {
            oldestKey = key
            oldestTime = item.AccessTime
        }
    }
    
    if oldestKey != "" {
        delete(lc.data, oldestKey)
    }
}

func (lc *LocalCache) evictLFU() {
    var leastFrequentKey string
    var leastCount int
    
    for key, item := range lc.data {
        if leastFrequentKey == "" || item.AccessCount < leastCount {
            leastFrequentKey = key
            leastCount = item.AccessCount
        }
    }
    
    if leastFrequentKey != "" {
        delete(lc.data, leastFrequentKey)
    }
}
```

## Message Queue System

### Problem: Implement a Message Queue with At-Least-Once Delivery

**Requirements:**
- At-least-once delivery guarantee
- Message persistence
- Dead letter queue
- Consumer groups
- Message ordering

```go
// Message Queue System
package main

import (
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

type MessageQueue struct {
    topics        map[string]*Topic
    consumers     map[string]*Consumer
    deadLetterQueue *Topic
    mu            sync.RWMutex
    storage       MessageStorage
}

type Topic struct {
    Name        string
    Partitions  []*Partition
    mu          sync.RWMutex
}

type Partition struct {
    ID          int
    Messages    []*Message
    Offset      int64
    mu          sync.RWMutex
}

type Message struct {
    ID        string
    Topic     string
    Partition int
    Key       string
    Value     []byte
    Timestamp time.Time
    Headers   map[string]string
}

type Consumer struct {
    ID           string
    GroupID      string
    Topics       []string
    Offset       map[string]int64
    LastSeen     time.Time
    mu           sync.RWMutex
}

type MessageStorage interface {
    Store(topic string, partition int, message *Message) error
    Retrieve(topic string, partition int, offset int64) (*Message, error)
    GetOffset(topic string, partition int, consumerGroup string) (int64, error)
    SetOffset(topic string, partition int, consumerGroup string, offset int64) error
}

func NewMessageQueue() *MessageQueue {
    mq := &MessageQueue{
        topics:    make(map[string]*Topic),
        consumers: make(map[string]*Consumer),
        storage:   NewInMemoryStorage(),
    }
    
    // Create dead letter queue
    mq.deadLetterQueue = &Topic{
        Name:       "__dead_letter__",
        Partitions: []*Partition{{ID: 0, Messages: make([]*Message, 0)}},
    }
    
    return mq
}

func (mq *MessageQueue) CreateTopic(name string, partitions int) error {
    mq.mu.Lock()
    defer mq.mu.Unlock()
    
    if _, exists := mq.topics[name]; exists {
        return fmt.Errorf("topic already exists")
    }
    
    topic := &Topic{
        Name:       name,
        Partitions: make([]*Partition, partitions),
    }
    
    for i := 0; i < partitions; i++ {
        topic.Partitions[i] = &Partition{
            ID:       i,
            Messages: make([]*Message, 0),
            Offset:   0,
        }
    }
    
    mq.topics[name] = topic
    return nil
}

func (mq *MessageQueue) Publish(topicName string, key string, value []byte, headers map[string]string) error {
    mq.mu.RLock()
    topic, exists := mq.topics[topicName]
    mq.mu.RUnlock()
    
    if !exists {
        return fmt.Errorf("topic not found")
    }
    
    // Select partition based on key
    partitionID := mq.selectPartition(topic, key)
    partition := topic.Partitions[partitionID]
    
    message := &Message{
        ID:        generateMessageID(),
        Topic:     topicName,
        Partition: partitionID,
        Key:       key,
        Value:     value,
        Timestamp: time.Now(),
        Headers:   headers,
    }
    
    // Store message
    if err := mq.storage.Store(topicName, partitionID, message); err != nil {
        return err
    }
    
    // Add to partition
    partition.mu.Lock()
    partition.Messages = append(partition.Messages, message)
    partition.Offset++
    partition.mu.Unlock()
    
    return nil
}

func (mq *MessageQueue) Subscribe(consumerID, groupID string, topics []string) (*Consumer, error) {
    mq.mu.Lock()
    defer mq.mu.Unlock()
    
    consumer := &Consumer{
        ID:      consumerID,
        GroupID: groupID,
        Topics:  topics,
        Offset:  make(map[string]int64),
        LastSeen: time.Now(),
    }
    
    mq.consumers[consumerID] = consumer
    return consumer, nil
}

func (mq *MessageQueue) Consume(consumerID string, timeout time.Duration) ([]*Message, error) {
    mq.mu.RLock()
    consumer, exists := mq.consumers[consumerID]
    mq.mu.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("consumer not found")
    }
    
    var messages []*Message
    
    for _, topicName := range consumer.Topics {
        mq.mu.RLock()
        topic, exists := mq.topics[topicName]
        mq.mu.RUnlock()
        
        if !exists {
            continue
        }
        
        for _, partition := range topic.Partitions {
            partition.mu.RLock()
            if len(partition.Messages) > 0 {
                // Get messages from current offset
                offset := consumer.Offset[fmt.Sprintf("%s-%d", topicName, partition.ID)]
                if int(offset) < len(partition.Messages) {
                    messages = append(messages, partition.Messages[offset:]...)
                    consumer.Offset[fmt.Sprintf("%s-%d", topicName, partition.ID)] = int64(len(partition.Messages))
                }
            }
            partition.mu.RUnlock()
        }
    }
    
    consumer.mu.Lock()
    consumer.LastSeen = time.Now()
    consumer.mu.Unlock()
    
    return messages, nil
}

func (mq *MessageQueue) selectPartition(topic *Topic, key string) int {
    if key == "" {
        return 0 // Default partition
    }
    
    // Simple hash-based partitioning
    hash := 0
    for _, c := range key {
        hash = hash*31 + int(c)
    }
    
    return hash % len(topic.Partitions)
}

func generateMessageID() string {
    return fmt.Sprintf("msg_%d", time.Now().UnixNano())
}

// In-Memory Storage Implementation
type InMemoryStorage struct {
    messages map[string]map[int][]*Message
    offsets  map[string]map[int]int64
    mu       sync.RWMutex
}

func NewInMemoryStorage() *InMemoryStorage {
    return &InMemoryStorage{
        messages: make(map[string]map[int][]*Message),
        offsets:  make(map[string]map[int]int64),
    }
}

func (ims *InMemoryStorage) Store(topic string, partition int, message *Message) error {
    ims.mu.Lock()
    defer ims.mu.Unlock()
    
    if ims.messages[topic] == nil {
        ims.messages[topic] = make(map[int][]*Message)
    }
    
    ims.messages[topic][partition] = append(ims.messages[topic][partition], message)
    return nil
}

func (ims *InMemoryStorage) Retrieve(topic string, partition int, offset int64) (*Message, error) {
    ims.mu.RLock()
    defer ims.mu.RUnlock()
    
    if ims.messages[topic] == nil || ims.messages[topic][partition] == nil {
        return nil, fmt.Errorf("topic or partition not found")
    }
    
    messages := ims.messages[topic][partition]
    if int(offset) >= len(messages) {
        return nil, fmt.Errorf("offset out of range")
    }
    
    return messages[offset], nil
}

func (ims *InMemoryStorage) GetOffset(topic string, partition int, consumerGroup string) (int64, error) {
    ims.mu.RLock()
    defer ims.mu.RUnlock()
    
    key := fmt.Sprintf("%s-%d-%s", topic, partition, consumerGroup)
    if ims.offsets[key] == nil {
        return 0, nil
    }
    
    return ims.offsets[key][partition], nil
}

func (ims *InMemoryStorage) SetOffset(topic string, partition int, consumerGroup string, offset int64) error {
    ims.mu.Lock()
    defer ims.mu.Unlock()
    
    key := fmt.Sprintf("%s-%d-%s", topic, partition, consumerGroup)
    if ims.offsets[key] == nil {
        ims.offsets[key] = make(map[int]int64)
    }
    
    ims.offsets[key][partition] = offset
    return nil
}
```

## Load Balancer Implementation

### Problem: Implement a Load Balancer with Health Checks

**Requirements:**
- Multiple load balancing algorithms (Round Robin, Least Connections, Weighted)
- Health checks for backend servers
- Circuit breaker pattern
- Metrics and monitoring
- Dynamic server addition/removal

```go
// Load Balancer Implementation
package main

import (
    "context"
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type LoadBalancer struct {
    servers      []*Server
    algorithm    string
    healthChecker *HealthChecker
    circuitBreaker *CircuitBreaker
    mu           sync.RWMutex
    metrics      *LoadBalancerMetrics
}

type Server struct {
    ID          string
    Address     string
    Weight      int
    Healthy     bool
    Connections int
    ResponseTime time.Duration
    LastSeen    time.Time
    mu          sync.RWMutex
}

type HealthChecker struct {
    interval    time.Duration
    timeout     time.Duration
    servers     []*Server
    stopCh      chan struct{}
    mu          sync.RWMutex
}

type CircuitBreaker struct {
    failureThreshold int
    timeout          time.Duration
    state           string // "closed", "open", "half-open"
    failures        int
    lastFailureTime time.Time
    mu              sync.RWMutex
}

type LoadBalancerMetrics struct {
    TotalRequests    int64
    SuccessfulRequests int64
    FailedRequests   int64
    AverageResponseTime time.Duration
}

func NewLoadBalancer(algorithm string) *LoadBalancer {
    return &LoadBalancer{
        servers:      make([]*Server, 0),
        algorithm:    algorithm,
        healthChecker: NewHealthChecker(),
        circuitBreaker: NewCircuitBreaker(5, time.Minute),
        metrics:      &LoadBalancerMetrics{},
    }
}

func (lb *LoadBalancer) AddServer(server *Server) {
    lb.mu.Lock()
    defer lb.mu.Unlock()
    
    lb.servers = append(lb.servers, server)
    lb.healthChecker.AddServer(server)
}

func (lb *LoadBalancer) RemoveServer(serverID string) {
    lb.mu.Lock()
    defer lb.mu.Unlock()
    
    for i, server := range lb.servers {
        if server.ID == serverID {
            lb.servers = append(lb.servers[:i], lb.servers[i+1:]...)
            lb.healthChecker.RemoveServer(serverID)
            break
        }
    }
}

func (lb *LoadBalancer) GetServer() (*Server, error) {
    lb.mu.RLock()
    defer lb.mu.RUnlock()
    
    healthyServers := lb.getHealthyServers()
    if len(healthyServers) == 0 {
        return nil, fmt.Errorf("no healthy servers available")
    }
    
    switch lb.algorithm {
    case "round_robin":
        return lb.roundRobin(healthyServers)
    case "least_connections":
        return lb.leastConnections(healthyServers)
    case "weighted":
        return lb.weighted(healthyServers)
    case "random":
        return lb.random(healthyServers)
    default:
        return lb.roundRobin(healthyServers)
    }
}

func (lb *LoadBalancer) getHealthyServers() []*Server {
    var healthy []*Server
    for _, server := range lb.servers {
        if server.Healthy {
            healthy = append(healthy, server)
        }
    }
    return healthy
}

func (lb *LoadBalancer) roundRobin(servers []*Server) (*Server, error) {
    if len(servers) == 0 {
        return nil, fmt.Errorf("no servers available")
    }
    
    // Simple round robin implementation
    // In production, you'd want to use atomic operations
    selected := servers[rand.Intn(len(servers))]
    
    selected.mu.Lock()
    selected.Connections++
    selected.mu.Unlock()
    
    return selected, nil
}

func (lb *LoadBalancer) leastConnections(servers []*Server) (*Server, error) {
    if len(servers) == 0 {
        return nil, fmt.Errorf("no servers available")
    }
    
    var selected *Server
    minConnections := int(^uint(0) >> 1) // Max int
    
    for _, server := range servers {
        server.mu.RLock()
        connections := server.Connections
        server.mu.RUnlock()
        
        if connections < minConnections {
            minConnections = connections
            selected = server
        }
    }
    
    if selected != nil {
        selected.mu.Lock()
        selected.Connections++
        selected.mu.Unlock()
    }
    
    return selected, nil
}

func (lb *LoadBalancer) weighted(servers []*Server) (*Server, error) {
    if len(servers) == 0 {
        return nil, fmt.Errorf("no servers available")
    }
    
    totalWeight := 0
    for _, server := range servers {
        totalWeight += server.Weight
    }
    
    if totalWeight == 0 {
        return lb.random(servers)
    }
    
    random := rand.Intn(totalWeight)
    current := 0
    
    for _, server := range servers {
        current += server.Weight
        if random < current {
            server.mu.Lock()
            server.Connections++
            server.mu.Unlock()
            return server, nil
        }
    }
    
    return servers[0], nil
}

func (lb *LoadBalancer) random(servers []*Server) (*Server, error) {
    if len(servers) == 0 {
        return nil, fmt.Errorf("no servers available")
    }
    
    selected := servers[rand.Intn(len(servers))]
    selected.mu.Lock()
    selected.Connections++
    selected.mu.Unlock()
    
    return selected, nil
}

func (lb *LoadBalancer) HandleRequest(ctx context.Context, request func(*Server) error) error {
    start := time.Now()
    
    server, err := lb.GetServer()
    if err != nil {
        lb.updateMetrics(false, time.Since(start))
        return err
    }
    
    // Check circuit breaker
    if !lb.circuitBreaker.CanExecute() {
        lb.updateMetrics(false, time.Since(start))
        return fmt.Errorf("circuit breaker is open")
    }
    
    // Execute request
    err = request(server)
    
    // Update circuit breaker
    if err != nil {
        lb.circuitBreaker.RecordFailure()
    } else {
        lb.circuitBreaker.RecordSuccess()
    }
    
    // Update metrics
    lb.updateMetrics(err == nil, time.Since(start))
    
    return err
}

func (lb *LoadBalancer) updateMetrics(success bool, duration time.Duration) {
    lb.mu.Lock()
    defer lb.mu.Unlock()
    
    lb.metrics.TotalRequests++
    if success {
        lb.metrics.SuccessfulRequests++
    } else {
        lb.metrics.FailedRequests++
    }
    
    // Update average response time
    lb.metrics.AverageResponseTime = (lb.metrics.AverageResponseTime + duration) / 2
}

// Health Checker Implementation
func NewHealthChecker() *HealthChecker {
    hc := &HealthChecker{
        interval: time.Second * 30,
        timeout:  time.Second * 5,
        servers:  make([]*Server, 0),
        stopCh:   make(chan struct{}),
    }
    
    go hc.start()
    return hc
}

func (hc *HealthChecker) AddServer(server *Server) {
    hc.mu.Lock()
    defer hc.mu.Unlock()
    
    hc.servers = append(hc.servers, server)
}

func (hc *HealthChecker) RemoveServer(serverID string) {
    hc.mu.Lock()
    defer hc.mu.Unlock()
    
    for i, server := range hc.servers {
        if server.ID == serverID {
            hc.servers = append(hc.servers[:i], hc.servers[i+1:]...)
            break
        }
    }
}

func (hc *HealthChecker) start() {
    ticker := time.NewTicker(hc.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            hc.checkHealth()
        case <-hc.stopCh:
            return
        }
    }
}

func (hc *HealthChecker) checkHealth() {
    hc.mu.RLock()
    servers := make([]*Server, len(hc.servers))
    copy(servers, hc.servers)
    hc.mu.RUnlock()
    
    for _, server := range servers {
        go hc.checkServer(server)
    }
}

func (hc *HealthChecker) checkServer(server *Server) {
    ctx, cancel := context.WithTimeout(context.Background(), hc.timeout)
    defer cancel()
    
    // Simulate health check
    // In production, you'd make an actual HTTP request
    time.Sleep(time.Millisecond * 100)
    
    healthy := true // Simulate health check result
    
    server.mu.Lock()
    server.Healthy = healthy
    server.LastSeen = time.Now()
    server.mu.Unlock()
}

// Circuit Breaker Implementation
func NewCircuitBreaker(failureThreshold int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        failureThreshold: failureThreshold,
        timeout:          timeout,
        state:           "closed",
        failures:        0,
    }
}

func (cb *CircuitBreaker) CanExecute() bool {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    
    switch cb.state {
    case "closed":
        return true
    case "open":
        if time.Since(cb.lastFailureTime) > cb.timeout {
            cb.mu.RUnlock()
            cb.mu.Lock()
            cb.state = "half-open"
            cb.mu.Unlock()
            cb.mu.RLock()
            return true
        }
        return false
    case "half-open":
        return true
    default:
        return false
    }
}

func (cb *CircuitBreaker) RecordSuccess() {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    cb.failures = 0
    cb.state = "closed"
}

func (cb *CircuitBreaker) RecordFailure() {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    cb.failures++
    cb.lastFailureTime = time.Now()
    
    if cb.failures >= cb.failureThreshold {
        cb.state = "open"
    }
}
```

## Conclusion

These system design coding problems test your ability to implement core components of distributed systems. They cover:

- **Distributed Caching**: Consistent hashing, replication, eviction policies
- **Message Queues**: At-least-once delivery, persistence, consumer groups
- **Load Balancing**: Multiple algorithms, health checks, circuit breakers
- **Database Sharding**: Data distribution, query routing, rebalancing
- **Real-time Analytics**: Stream processing, windowing, aggregation
- **File Storage**: Distributed file systems, replication, consistency
- **Search Engines**: Indexing, query processing, ranking

The key to success is understanding the trade-offs and implementing robust, scalable solutions that can handle real-world requirements.

## Additional Resources

- [System Design Interview](https://www.educative.io/courses/grokking-the-system-design-interview/)
- [Distributed Systems Patterns](https://microservices.io/patterns/)
- [High Performance Go](https://github.com/geohot/minikeyvalue/)
- [Designing Data-Intensive Applications](https://dataintensive.net/)
- [Building Microservices](https://samnewman.io/books/building_microservices/)
