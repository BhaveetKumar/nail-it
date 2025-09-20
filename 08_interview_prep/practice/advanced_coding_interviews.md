# Advanced Coding Interview Problems

## Table of Contents
- [Introduction](#introduction)
- [System Design Coding Problems](#system-design-coding-problems)
- [Distributed Systems Problems](#distributed-systems-problems)
- [Real-Time Systems Problems](#real-time-systems-problems)
- [Machine Learning Problems](#machine-learning-problems)
- [Financial Systems Problems](#financial-systems-problems)
- [Performance Optimization Problems](#performance-optimization-problems)
- [Security and Cryptography Problems](#security-and-cryptography-problems)
- [Database and Storage Problems](#database-and-storage-problems)
- [Concurrency and Parallelism Problems](#concurrency-and-parallelism-problems)

## Introduction

Advanced coding interview problems test your ability to solve complex, real-world problems that combine multiple concepts. These problems often require system design thinking, algorithm optimization, and practical implementation skills.

## System Design Coding Problems

### Problem 1: Design a Distributed Cache

**Problem Statement**: Implement a distributed cache system that can handle millions of requests per second with sub-millisecond latency.

**Requirements**:
- Support get, set, delete operations
- Handle node failures gracefully
- Maintain data consistency
- Support data replication
- Provide monitoring and metrics

**Solution**:

```go
package main

import (
    "context"
    "crypto/md5"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
)

type DistributedCache struct {
    nodes      map[string]*CacheNode
    ring       *ConsistentHashRing
    replicas   int
    mu         sync.RWMutex
    eviction   *EvictionManager
    monitoring *Monitoring
}

type CacheNode struct {
    ID       string
    Address  string
    Port     int
    Storage  *CacheStorage
    Client   *http.Client
    Health   *HealthChecker
}

type CacheStorage struct {
    data       map[string]*CacheEntry
    maxSize    int64
    currentSize int64
    mu         sync.RWMutex
}

type CacheEntry struct {
    Key        string
    Value      []byte
    Timestamp  time.Time
    TTL        time.Duration
    Version    int64
    AccessCount int64
}

func NewDistributedCache(replicas int) *DistributedCache {
    return &DistributedCache{
        nodes:      make(map[string]*CacheNode),
        ring:       NewConsistentHashRing(replicas),
        replicas:   replicas,
        eviction:   NewEvictionManager(),
        monitoring: NewMonitoring(),
    }
}

func (dc *DistributedCache) Get(key string) ([]byte, error) {
    // Get nodes for key
    nodes := dc.ring.GetNodes(key, dc.replicas)
    
    // Try to get from primary node
    for _, node := range nodes {
        if node.Health.IsHealthy() {
            value, err := node.Storage.Get(key)
            if err == nil {
                // Update access count
                node.Storage.UpdateAccessCount(key)
                return value, nil
            }
        }
    }
    
    return nil, fmt.Errorf("key %s not found", key)
}

func (dc *DistributedCache) Set(key string, value []byte, ttl time.Duration) error {
    // Get nodes for key
    nodes := dc.ring.GetNodes(key, dc.replicas)
    
    // Set on all replica nodes
    var wg sync.WaitGroup
    errors := make(chan error, len(nodes))
    
    for _, node := range nodes {
        if node.Health.IsHealthy() {
            wg.Add(1)
            go func(n *CacheNode) {
                defer wg.Done()
                if err := n.Storage.Set(key, value, ttl); err != nil {
                    errors <- err
                }
            }(node)
        }
    }
    
    wg.Wait()
    close(errors)
    
    // Check for errors
    for err := range errors {
        if err != nil {
            return err
        }
    }
    
    return nil
}

// Consistent Hash Ring Implementation
type ConsistentHashRing struct {
    nodes    []*HashNode
    replicas int
    mu       sync.RWMutex
}

type HashNode struct {
    ID     string
    Hash   uint32
    Node   *CacheNode
}

func NewConsistentHashRing(replicas int) *ConsistentHashRing {
    return &ConsistentHashRing{
        nodes:    make([]*HashNode, 0),
        replicas: replicas,
    }
}

func (chr *ConsistentHashRing) AddNode(node *CacheNode) error {
    chr.mu.Lock()
    defer chr.mu.Unlock()
    
    // Add virtual nodes
    for i := 0; i < chr.replicas; i++ {
        virtualNode := &HashNode{
            ID:   fmt.Sprintf("%s-%d", node.ID, i),
            Hash: chr.hash(fmt.Sprintf("%s-%d", node.ID, i)),
            Node: node,
        }
        chr.nodes = append(chr.nodes, virtualNode)
    }
    
    // Sort nodes by hash
    chr.sortNodes()
    
    return nil
}

func (chr *ConsistentHashRing) GetNodes(key string, count int) []*CacheNode {
    chr.mu.RLock()
    defer chr.mu.RUnlock()
    
    if len(chr.nodes) == 0 {
        return nil
    }
    
    hash := chr.hash(key)
    nodes := make([]*CacheNode, 0, count)
    seen := make(map[string]bool)
    
    // Find the first node with hash >= key hash
    start := chr.findNodeIndex(hash)
    
    for i := 0; i < len(chr.nodes) && len(nodes) < count; i++ {
        idx := (start + i) % len(chr.nodes)
        node := chr.nodes[idx]
        
        if !seen[node.Node.ID] {
            nodes = append(nodes, node.Node)
            seen[node.Node.ID] = true
        }
    }
    
    return nodes
}

func (chr *ConsistentHashRing) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

func (chr *ConsistentHashRing) findNodeIndex(hash uint32) int {
    // Binary search for the first node with hash >= key hash
    left, right := 0, len(chr.nodes)
    
    for left < right {
        mid := (left + right) / 2
        if chr.nodes[mid].Hash < hash {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return left % len(chr.nodes)
}

func (chr *ConsistentHashRing) sortNodes() {
    // Sort nodes by hash value
    for i := 0; i < len(chr.nodes); i++ {
        for j := i + 1; j < len(chr.nodes); j++ {
            if chr.nodes[i].Hash > chr.nodes[j].Hash {
                chr.nodes[i], chr.nodes[j] = chr.nodes[j], chr.nodes[i]
            }
        }
    }
}
```

### Problem 2: Design a Rate Limiter

**Problem Statement**: Implement a distributed rate limiter that can handle millions of requests per second with different rate limiting strategies.

**Requirements**:
- Support multiple rate limiting algorithms (token bucket, sliding window, etc.)
- Handle distributed systems
- Support different rate limits per user/endpoint
- Provide monitoring and metrics
- Handle burst traffic

**Solution**:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

type RateLimiter struct {
    strategies  map[string]*RateLimitStrategy
    storage     *RateLimitStorage
    monitoring  *RateLimitMonitoring
    mu          sync.RWMutex
}

type RateLimitStrategy struct {
    Name        string
    Algorithm   string
    Function    func(*RateLimitRequest) bool
    Parameters  map[string]interface{}
}

type RateLimitRequest struct {
    UserID      string
    Endpoint    string
    Timestamp   time.Time
    Count       int
}

type RateLimitStorage struct {
    buckets     map[string]*TokenBucket
    windows     map[string]*SlidingWindow
    mu          sync.RWMutex
}

type TokenBucket struct {
    Capacity    int
    Tokens      int
    LastRefill  time.Time
    RefillRate  int
    mu          sync.Mutex
}

type SlidingWindow struct {
    WindowSize  time.Duration
    Requests    []time.Time
    MaxRequests int
    mu          sync.Mutex
}

func NewRateLimiter() *RateLimiter {
    return &RateLimiter{
        strategies: make(map[string]*RateLimitStrategy),
        storage:    NewRateLimitStorage(),
        monitoring: NewRateLimitMonitoring(),
    }
}

func (rl *RateLimiter) AddStrategy(strategy *RateLimitStrategy) {
    rl.mu.Lock()
    defer rl.mu.Unlock()
    rl.strategies[strategy.Name] = strategy
}

func (rl *RateLimiter) IsAllowed(request *RateLimitRequest) bool {
    rl.mu.RLock()
    strategy, exists := rl.strategies[request.Endpoint]
    rl.mu.RUnlock()
    
    if !exists {
        return true // No rate limit configured
    }
    
    allowed := strategy.Function(request)
    
    // Update monitoring
    rl.monitoring.RecordRequest(request, allowed)
    
    return allowed
}

// Token Bucket Strategy
func NewTokenBucketStrategy(capacity, refillRate int) *RateLimitStrategy {
    return &RateLimitStrategy{
        Name:       "token_bucket",
        Algorithm:  "token_bucket",
        Parameters: map[string]interface{}{
            "capacity":    capacity,
            "refill_rate": refillRate,
        },
        Function: func(request *RateLimitRequest) bool {
            return tokenBucketCheck(request, capacity, refillRate)
        },
    }
}

func tokenBucketCheck(request *RateLimitRequest, capacity, refillRate int) bool {
    bucket := getOrCreateTokenBucket(request.UserID, capacity, refillRate)
    
    bucket.mu.Lock()
    defer bucket.mu.Unlock()
    
    // Refill tokens
    now := time.Now()
    if !bucket.LastRefill.IsZero() {
        elapsed := now.Sub(bucket.LastRefill)
        tokensToAdd := int(elapsed.Seconds()) * refillRate
        bucket.Tokens = min(bucket.Capacity, bucket.Tokens+tokensToAdd)
    }
    bucket.LastRefill = now
    
    // Check if enough tokens
    if bucket.Tokens >= request.Count {
        bucket.Tokens -= request.Count
        return true
    }
    
    return false
}

func getOrCreateTokenBucket(userID string, capacity, refillRate int) *TokenBucket {
    // This would typically use a distributed storage
    // For simplicity, using in-memory storage
    return &TokenBucket{
        Capacity:   capacity,
        Tokens:     capacity,
        RefillRate: refillRate,
        LastRefill: time.Now(),
    }
}

// Sliding Window Strategy
func NewSlidingWindowStrategy(windowSize time.Duration, maxRequests int) *RateLimitStrategy {
    return &RateLimitStrategy{
        Name:       "sliding_window",
        Algorithm:  "sliding_window",
        Parameters: map[string]interface{}{
            "window_size":   windowSize,
            "max_requests":  maxRequests,
        },
        Function: func(request *RateLimitRequest) bool {
            return slidingWindowCheck(request, windowSize, maxRequests)
        },
    }
}

func slidingWindowCheck(request *RateLimitRequest, windowSize time.Duration, maxRequests int) bool {
    window := getOrCreateSlidingWindow(request.UserID, windowSize, maxRequests)
    
    window.mu.Lock()
    defer window.mu.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-windowSize)
    
    // Remove old requests
    var validRequests []time.Time
    for _, reqTime := range window.Requests {
        if reqTime.After(cutoff) {
            validRequests = append(validRequests, reqTime)
        }
    }
    window.Requests = validRequests
    
    // Check if under limit
    if len(window.Requests) < maxRequests {
        window.Requests = append(window.Requests, now)
        return true
    }
    
    return false
}

func getOrCreateSlidingWindow(userID string, windowSize time.Duration, maxRequests int) *SlidingWindow {
    // This would typically use a distributed storage
    // For simplicity, using in-memory storage
    return &SlidingWindow{
        WindowSize:  windowSize,
        MaxRequests: maxRequests,
        Requests:    make([]time.Time, 0),
    }
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

## Distributed Systems Problems

### Problem 3: Implement a Distributed Lock

**Problem Statement**: Implement a distributed lock system that can be used across multiple processes and machines.

**Requirements**:
- Support exclusive locks
- Handle lock expiration
- Support lock renewal
- Handle network partitions
- Provide monitoring and metrics

**Solution**:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

type DistributedLock struct {
    key        string
    value      string
    ttl        time.Duration
    client     *LockClient
    mu         sync.RWMutex
    locked     bool
    renewTicker *time.Ticker
    stopChan   chan struct{}
}

type LockClient struct {
    storage    *LockStorage
    monitoring *Monitoring
    mu         sync.RWMutex
}

type LockStorage struct {
    locks    map[string]*LockInfo
    mu       sync.RWMutex
}

type LockInfo struct {
    Key       string
    Value     string
    ExpiresAt time.Time
    Owner     string
}

func NewDistributedLock(key string, ttl time.Duration, client *LockClient) *DistributedLock {
    return &DistributedLock{
        key:    key,
        value:  generateLockValue(),
        ttl:    ttl,
        client: client,
        locked: false,
    }
}

func (dl *DistributedLock) Lock(ctx context.Context) error {
    dl.mu.Lock()
    defer dl.mu.Unlock()
    
    if dl.locked {
        return fmt.Errorf("lock already held")
    }
    
    // Try to acquire lock
    acquired, err := dl.client.AcquireLock(dl.key, dl.value, dl.ttl)
    if err != nil {
        return err
    }
    
    if !acquired {
        return fmt.Errorf("failed to acquire lock")
    }
    
    dl.locked = true
    
    // Start renewal process
    dl.startRenewal()
    
    return nil
}

func (dl *DistributedLock) Unlock() error {
    dl.mu.Lock()
    defer dl.mu.Unlock()
    
    if !dl.locked {
        return fmt.Errorf("lock not held")
    }
    
    // Stop renewal
    dl.stopRenewal()
    
    // Release lock
    if err := dl.client.ReleaseLock(dl.key, dl.value); err != nil {
        return err
    }
    
    dl.locked = false
    return nil
}

func (dl *DistributedLock) startRenewal() {
    dl.renewTicker = time.NewTicker(dl.ttl / 2)
    dl.stopChan = make(chan struct{})
    
    go func() {
        for {
            select {
            case <-dl.renewTicker.C:
                if err := dl.renewLock(); err != nil {
                    log.Printf("Failed to renew lock: %v", err)
                }
            case <-dl.stopChan:
                return
            }
        }
    }()
}

func (dl *DistributedLock) stopRenewal() {
    if dl.renewTicker != nil {
        dl.renewTicker.Stop()
    }
    if dl.stopChan != nil {
        close(dl.stopChan)
    }
}

func (dl *DistributedLock) renewLock() error {
    return dl.client.RenewLock(dl.key, dl.value, dl.ttl)
}

func (dl *DistributedLock) IsLocked() bool {
    dl.mu.RLock()
    defer dl.mu.RUnlock()
    return dl.locked
}

// Lock Client Implementation
func NewLockClient() *LockClient {
    return &LockClient{
        storage:    NewLockStorage(),
        monitoring: NewMonitoring(),
    }
}

func (lc *LockClient) AcquireLock(key, value string, ttl time.Duration) (bool, error) {
    lc.mu.Lock()
    defer lc.mu.Unlock()
    
    // Check if lock exists
    if lockInfo, exists := lc.storage.locks[key]; exists {
        // Check if lock is expired
        if time.Now().After(lockInfo.ExpiresAt) {
            // Lock is expired, remove it
            delete(lc.storage.locks, key)
        } else {
            // Lock is still valid
            return false, nil
        }
    }
    
    // Acquire lock
    lockInfo := &LockInfo{
        Key:       key,
        Value:     value,
        ExpiresAt: time.Now().Add(ttl),
        Owner:     value,
    }
    
    lc.storage.locks[key] = lockInfo
    
    // Update monitoring
    lc.monitoring.RecordLockAcquired(key)
    
    return true, nil
}

func (lc *LockClient) ReleaseLock(key, value string) error {
    lc.mu.Lock()
    defer lc.mu.Unlock()
    
    lockInfo, exists := lc.storage.locks[key]
    if !exists {
        return fmt.Errorf("lock %s not found", key)
    }
    
    // Check if we own the lock
    if lockInfo.Value != value {
        return fmt.Errorf("lock %s not owned by %s", key, value)
    }
    
    // Release lock
    delete(lc.storage.locks, key)
    
    // Update monitoring
    lc.monitoring.RecordLockReleased(key)
    
    return nil
}

func (lc *LockClient) RenewLock(key, value string, ttl time.Duration) error {
    lc.mu.Lock()
    defer lc.mu.Unlock()
    
    lockInfo, exists := lc.storage.locks[key]
    if !exists {
        return fmt.Errorf("lock %s not found", key)
    }
    
    // Check if we own the lock
    if lockInfo.Value != value {
        return fmt.Errorf("lock %s not owned by %s", key, value)
    }
    
    // Renew lock
    lockInfo.ExpiresAt = time.Now().Add(ttl)
    
    return nil
}

func NewLockStorage() *LockStorage {
    return &LockStorage{
        locks: make(map[string]*LockInfo),
    }
}

func generateLockValue() string {
    return fmt.Sprintf("%d", time.Now().UnixNano())
}
```

## Real-Time Systems Problems

### Problem 4: Design a Real-Time Chat System

**Problem Statement**: Implement a real-time chat system that can handle millions of concurrent users with sub-second message delivery.

**Requirements**:
- Support multiple chat rooms
- Real-time message delivery
- Message persistence
- User presence
- Message history
- Scalable architecture

**Solution**:

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
    "github.com/gorilla/websocket"
)

type ChatSystem struct {
    rooms        map[string]*ChatRoom
    users        map[string]*User
    connections  map[string]*Connection
    messageQueue *MessageQueue
    database     *ChatDatabase
    mu           sync.RWMutex
}

type ChatRoom struct {
    ID          string
    Name        string
    Users       map[string]*User
    Messages    []*Message
    CreatedAt   time.Time
    UpdatedAt   time.Time
    mu          sync.RWMutex
}

type User struct {
    ID          string
    Username    string
    Status      string
    LastSeen    time.Time
    Connections []*Connection
    mu          sync.RWMutex
}

type Connection struct {
    ID       string
    UserID   string
    RoomID   string
    WS       *websocket.Conn
    Send     chan *Message
    mu       sync.RWMutex
}

type Message struct {
    ID        string
    RoomID    string
    UserID    string
    Username  string
    Content   string
    Type      string
    Timestamp time.Time
}

type MessageQueue struct {
    messages chan *Message
    workers  int
    mu       sync.RWMutex
}

func NewChatSystem() *ChatSystem {
    return &ChatSystem{
        rooms:        make(map[string]*ChatRoom),
        users:        make(map[string]*User),
        connections:  make(map[string]*Connection),
        messageQueue: NewMessageQueue(10),
        database:     NewChatDatabase(),
    }
}

func (cs *ChatSystem) CreateRoom(roomID, name string) error {
    cs.mu.Lock()
    defer cs.mu.Unlock()
    
    if _, exists := cs.rooms[roomID]; exists {
        return fmt.Errorf("room %s already exists", roomID)
    }
    
    room := &ChatRoom{
        ID:        roomID,
        Name:      name,
        Users:     make(map[string]*User),
        Messages:  make([]*Message, 0),
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    cs.rooms[roomID] = room
    
    return nil
}

func (cs *ChatSystem) JoinRoom(roomID, userID string, conn *websocket.Conn) error {
    cs.mu.Lock()
    defer cs.mu.Unlock()
    
    room, exists := cs.rooms[roomID]
    if !exists {
        return fmt.Errorf("room %s not found", roomID)
    }
    
    user, exists := cs.users[userID]
    if !exists {
        return fmt.Errorf("user %s not found", userID)
    }
    
    // Create connection
    connection := &Connection{
        ID:     generateConnectionID(),
        UserID: userID,
        RoomID: roomID,
        WS:     conn,
        Send:   make(chan *Message, 256),
    }
    
    // Add to room
    room.mu.Lock()
    room.Users[userID] = user
    room.UpdatedAt = time.Now()
    room.mu.Unlock()
    
    // Add to user
    user.mu.Lock()
    user.Connections = append(user.Connections, connection)
    user.mu.Unlock()
    
    // Store connection
    cs.connections[connection.ID] = connection
    
    // Start connection handlers
    go cs.handleConnection(connection)
    
    return nil
}

func (cs *ChatSystem) handleConnection(conn *Connection) {
    defer func() {
        cs.removeConnection(conn)
    }()
    
    // Start writer
    go cs.writePump(conn)
    
    // Start reader
    cs.readPump(conn)
}

func (cs *ChatSystem) readPump(conn *Connection) {
    defer conn.WS.Close()
    
    conn.WS.SetReadLimit(512)
    conn.WS.SetReadDeadline(time.Now().Add(60 * time.Second))
    conn.WS.SetPongHandler(func(string) error {
        conn.WS.SetReadDeadline(time.Now().Add(60 * time.Second))
        return nil
    })
    
    for {
        var msg Message
        err := conn.WS.ReadJSON(&msg)
        if err != nil {
            if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
                log.Printf("WebSocket error: %v", err)
            }
            break
        }
        
        // Process message
        if err := cs.processMessage(conn, &msg); err != nil {
            log.Printf("Error processing message: %v", err)
        }
    }
}

func (cs *ChatSystem) writePump(conn *Connection) {
    ticker := time.NewTicker(54 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case message := <-conn.Send:
            conn.WS.SetWriteDeadline(time.Now().Add(10 * time.Second))
            if err := conn.WS.WriteJSON(message); err != nil {
                log.Printf("WebSocket write error: %v", err)
                return
            }
        case <-ticker.C:
            conn.WS.SetWriteDeadline(time.Now().Add(10 * time.Second))
            if err := conn.WS.WriteMessage(websocket.PingMessage, nil); err != nil {
                return
            }
        }
    }
}

func (cs *ChatSystem) processMessage(conn *Connection, msg *Message) error {
    // Set message properties
    msg.ID = generateMessageID()
    msg.RoomID = conn.RoomID
    msg.UserID = conn.UserID
    msg.Timestamp = time.Now()
    
    // Get user
    user, exists := cs.users[conn.UserID]
    if !exists {
        return fmt.Errorf("user not found")
    }
    
    msg.Username = user.Username
    
    // Add to room
    room := cs.rooms[conn.RoomID]
    room.mu.Lock()
    room.Messages = append(room.Messages, msg)
    room.UpdatedAt = time.Now()
    room.mu.Unlock()
    
    // Store in database
    if err := cs.database.StoreMessage(msg); err != nil {
        log.Printf("Error storing message: %v", err)
    }
    
    // Broadcast to room
    cs.broadcastToRoom(conn.RoomID, msg)
    
    return nil
}

func (cs *ChatSystem) broadcastToRoom(roomID string, msg *Message) {
    cs.mu.RLock()
    room := cs.rooms[roomID]
    cs.mu.RUnlock()
    
    if room == nil {
        return
    }
    
    room.mu.RLock()
    for _, user := range room.Users {
        for _, conn := range user.Connections {
            if conn.RoomID == roomID {
                select {
                case conn.Send <- msg:
                default:
                    // Connection is full, skip
                }
            }
        }
    }
    room.mu.RUnlock()
}

func (cs *ChatSystem) removeConnection(conn *Connection) {
    cs.mu.Lock()
    delete(cs.connections, conn.ID)
    cs.mu.Unlock()
    
    // Remove from user
    if user, exists := cs.users[conn.UserID]; exists {
        user.mu.Lock()
        for i, c := range user.Connections {
            if c.ID == conn.ID {
                user.Connections = append(user.Connections[:i], user.Connections[i+1:]...)
                break
            }
        }
        user.mu.Unlock()
    }
    
    // Remove from room if no more connections
    if user, exists := cs.users[conn.UserID]; exists {
        if len(user.Connections) == 0 {
            cs.mu.Lock()
            if room, exists := cs.rooms[conn.RoomID]; exists {
                room.mu.Lock()
                delete(room.Users, conn.UserID)
                room.UpdatedAt = time.Now()
                room.mu.Unlock()
            }
            cs.mu.Unlock()
        }
    }
    
    conn.WS.Close()
}
```

## Conclusion

Advanced coding interview problems test your ability to:

1. **Design Systems**: Architecture, scalability, and reliability
2. **Implement Solutions**: Code quality, efficiency, and maintainability
3. **Handle Complexity**: Distributed systems, real-time processing, and ML systems
4. **Optimize Performance**: Caching, load balancing, and resource management
5. **Ensure Reliability**: Error handling, monitoring, and testing

Mastering these problems will prepare you for senior engineering roles that require both system design and implementation skills.

## Additional Resources

- [System Design Interview](https://www.systemdesigninterview.com/)
- [Distributed Systems](https://www.distributedsystems.com/)
- [Real-Time Systems](https://www.realtimesystems.com/)
- [Machine Learning Systems](https://www.mlsystems.com/)
- [Performance Engineering](https://www.performanceengineering.com/)
- [Load Testing](https://www.loadtesting.com/)
- [Monitoring and Observability](https://www.observability.com/)
- [Caching Strategies](https://www.cachingstrategies.com/)