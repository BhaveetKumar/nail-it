# Advanced Coding Problems

Comprehensive advanced coding problems for senior engineering interviews.

## 🎯 Advanced Algorithm Problems

### Problem 1: Implement a Distributed Cache with LRU Eviction
**Difficulty**: Hard  
**Time Complexity**: O(1) for get and put operations  
**Space Complexity**: O(capacity)

```go
// Distributed LRU Cache
type DistributedLRUCache struct {
    capacity    int
    shards      map[string]*LRUShard
    hashRing    *ConsistentHash
    replicator  *Replicator
    mutex       sync.RWMutex
}

type LRUShard struct {
    capacity int
    cache    map[string]*Node
    head     *Node
    tail     *Node
    mutex    sync.RWMutex
}

type Node struct {
    key   string
    value interface{}
    prev  *Node
    next  *Node
}

func NewDistributedLRUCache(capacity int, shardCount int) *DistributedLRUCache {
    cache := &DistributedLRUCache{
        capacity: capacity,
        shards:   make(map[string]*LRUShard),
        hashRing: NewConsistentHash(),
    }
    
    // Create shards
    for i := 0; i < shardCount; i++ {
        shardID := fmt.Sprintf("shard_%d", i)
        cache.shards[shardID] = NewLRUShard(capacity / shardCount)
        cache.hashRing.AddNode(shardID)
    }
    
    return cache
}

func (dc *DistributedLRUCache) Get(key string) (interface{}, bool) {
    shardID := dc.hashRing.GetNode(key)
    shard := dc.shards[shardID]
    
    return shard.Get(key)
}

func (dc *DistributedLRUCache) Put(key string, value interface{}) {
    shardID := dc.hashRing.GetNode(key)
    shard := dc.shards[shardID]
    
    shard.Put(key, value)
    
    // Replicate to other shards
    go dc.replicator.Replicate(key, value, shardID)
}

func (ls *LRUShard) Get(key string) (interface{}, bool) {
    ls.mutex.Lock()
    defer ls.mutex.Unlock()
    
    if node, exists := ls.cache[key]; exists {
        // Move to head
        ls.moveToHead(node)
        return node.value, true
    }
    
    return nil, false
}

func (ls *LRUShard) Put(key string, value interface{}) {
    ls.mutex.Lock()
    defer ls.mutex.Unlock()
    
    if node, exists := ls.cache[key]; exists {
        // Update existing node
        node.value = value
        ls.moveToHead(node)
    } else {
        // Create new node
        newNode := &Node{
            key:   key,
            value: value,
        }
        
        // Add to head
        ls.addToHead(newNode)
        ls.cache[key] = newNode
        
        // Check capacity
        if len(ls.cache) > ls.capacity {
            // Remove tail
            tail := ls.removeTail()
            delete(ls.cache, tail.key)
        }
    }
}

func (ls *LRUShard) moveToHead(node *Node) {
    ls.removeNode(node)
    ls.addToHead(node)
}

func (ls *LRUShard) addToHead(node *Node) {
    node.prev = ls.head
    node.next = ls.head.next
    ls.head.next.prev = node
    ls.head.next = node
}

func (ls *LRUShard) removeNode(node *Node) {
    node.prev.next = node.next
    node.next.prev = node.prev
}

func (ls *LRUShard) removeTail() *Node {
    lastNode := ls.tail.prev
    ls.removeNode(lastNode)
    return lastNode
}
```

### Problem 2: Implement a Rate Limiter with Multiple Strategies
**Difficulty**: Hard  
**Time Complexity**: O(1) for each request  
**Space Complexity**: O(n) where n is the number of users

```go
// Multi-Strategy Rate Limiter
type RateLimiter struct {
    strategies map[string]RateLimitStrategy
    mutex      sync.RWMutex
}

type RateLimitStrategy interface {
    Allow(userID string) bool
    GetRemaining(userID string) int
    GetResetTime(userID string) time.Time
}

// Token Bucket Strategy
type TokenBucketStrategy struct {
    capacity     int
    refillRate   int
    tokens       map[string]*TokenBucket
    mutex        sync.RWMutex
}

type TokenBucket struct {
    tokens     int
    lastRefill time.Time
    capacity   int
    refillRate int
}

func (tbs *TokenBucketStrategy) Allow(userID string) bool {
    tbs.mutex.Lock()
    defer tbs.mutex.Unlock()
    
    bucket, exists := tbs.tokens[userID]
    if !exists {
        bucket = &TokenBucket{
            tokens:     tbs.capacity,
            lastRefill: time.Now(),
            capacity:   tbs.capacity,
            refillRate: tbs.refillRate,
        }
        tbs.tokens[userID] = bucket
    }
    
    now := time.Now()
    tokensToAdd := int(now.Sub(bucket.lastRefill).Seconds()) * bucket.refillRate
    bucket.tokens = min(bucket.capacity, bucket.tokens+tokensToAdd)
    bucket.lastRefill = now
    
    if bucket.tokens > 0 {
        bucket.tokens--
        return true
    }
    
    return false
}

// Sliding Window Strategy
type SlidingWindowStrategy struct {
    windowSize  time.Duration
    maxRequests int
    requests    map[string][]time.Time
    mutex       sync.RWMutex
}

func (sws *SlidingWindowStrategy) Allow(userID string) bool {
    sws.mutex.Lock()
    defer sws.mutex.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-sws.windowSize)
    
    // Remove old requests
    if userRequests, exists := sws.requests[userID]; exists {
        var validRequests []time.Time
        for _, reqTime := range userRequests {
            if reqTime.After(cutoff) {
                validRequests = append(validRequests, reqTime)
            }
        }
        sws.requests[userID] = validRequests
    }
    
    // Check if under limit
    if len(sws.requests[userID]) < sws.maxRequests {
        sws.requests[userID] = append(sws.requests[userID], now)
        return true
    }
    
    return false
}

// Fixed Window Strategy
type FixedWindowStrategy struct {
    windowSize  time.Duration
    maxRequests int
    windows     map[string]*Window
    mutex       sync.RWMutex
}

type Window struct {
    start     time.Time
    requests  int
    resetTime time.Time
}

func (fws *FixedWindowStrategy) Allow(userID string) bool {
    fws.mutex.Lock()
    defer fws.mutex.Unlock()
    
    now := time.Now()
    
    if window, exists := fws.windows[userID]; exists {
        // Check if window needs reset
        if now.After(window.resetTime) {
            window.start = now
            window.requests = 0
            window.resetTime = now.Add(fws.windowSize)
        }
        
        // Check if under limit
        if window.requests < fws.maxRequests {
            window.requests++
            return true
        }
        
        return false
    } else {
        // Create new window
        fws.windows[userID] = &Window{
            start:     now,
            requests:  1,
            resetTime: now.Add(fws.windowSize),
        }
        return true
    }
}

func NewRateLimiter() *RateLimiter {
    return &RateLimiter{
        strategies: make(map[string]RateLimitStrategy),
    }
}

func (rl *RateLimiter) AddStrategy(name string, strategy RateLimitStrategy) {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()
    rl.strategies[name] = strategy
}

func (rl *RateLimiter) Allow(userID string, strategyName string) bool {
    rl.mutex.RLock()
    strategy, exists := rl.strategies[strategyName]
    rl.mutex.RUnlock()
    
    if !exists {
        return false
    }
    
    return strategy.Allow(userID)
}
```

### Problem 3: Implement a Message Queue with Priority and Dead Letter Queue
**Difficulty**: Hard  
**Time Complexity**: O(log n) for enqueue, O(1) for dequeue  
**Space Complexity**: O(n)

```go
// Priority Message Queue
type PriorityMessageQueue struct {
    queues        map[int]*Queue
    deadLetterQueue *Queue
    mutex         sync.RWMutex
    maxRetries    int
}

type Message struct {
    ID        string
    Priority  int
    Content   interface{}
    Retries   int
    Timestamp time.Time
    ExpiresAt time.Time
}

type Queue struct {
    messages []*Message
    mutex    sync.Mutex
}

func NewPriorityMessageQueue(maxRetries int) *PriorityMessageQueue {
    return &PriorityMessageQueue{
        queues:         make(map[int]*Queue),
        deadLetterQueue: &Queue{},
        maxRetries:     maxRetries,
    }
}

func (pmq *PriorityMessageQueue) Enqueue(message *Message) error {
    pmq.mutex.Lock()
    defer pmq.mutex.Unlock()
    
    // Check if message has expired
    if !message.ExpiresAt.IsZero() && time.Now().After(message.ExpiresAt) {
        return ErrMessageExpired
    }
    
    // Get or create queue for priority
    queue, exists := pmq.queues[message.Priority]
    if !exists {
        queue = &Queue{}
        pmq.queues[message.Priority] = queue
    }
    
    // Add message to queue
    queue.mutex.Lock()
    queue.messages = append(queue.messages, message)
    queue.mutex.Unlock()
    
    return nil
}

func (pmq *PriorityMessageQueue) Dequeue() (*Message, error) {
    pmq.mutex.RLock()
    defer pmq.mutex.RUnlock()
    
    // Find highest priority queue with messages
    for priority := 10; priority >= 1; priority-- {
        if queue, exists := pmq.queues[priority]; exists {
            queue.mutex.Lock()
            if len(queue.messages) > 0 {
                message := queue.messages[0]
                queue.messages = queue.messages[1:]
                queue.mutex.Unlock()
                return message, nil
            }
            queue.mutex.Unlock()
        }
    }
    
    return nil, ErrQueueEmpty
}

func (pmq *PriorityMessageQueue) ProcessMessage(message *Message, processor MessageProcessor) error {
    err := processor.Process(message)
    
    if err != nil {
        // Increment retry count
        message.Retries++
        
        if message.Retries >= pmq.maxRetries {
            // Move to dead letter queue
            pmq.deadLetterQueue.mutex.Lock()
            pmq.deadLetterQueue.messages = append(pmq.deadLetterQueue.messages, message)
            pmq.deadLetterQueue.mutex.Unlock()
            return ErrMessageMovedToDeadLetter
        } else {
            // Re-queue for retry
            return pmq.Enqueue(message)
        }
    }
    
    return nil
}

type MessageProcessor interface {
    Process(message *Message) error
}

// Dead Letter Queue Handler
func (pmq *PriorityMessageQueue) ProcessDeadLetterQueue(handler DeadLetterHandler) error {
    pmq.deadLetterQueue.mutex.Lock()
    defer pmq.deadLetterQueue.mutex.Unlock()
    
    for _, message := range pmq.deadLetterQueue.messages {
        if err := handler.Handle(message); err != nil {
            return err
        }
    }
    
    // Clear dead letter queue
    pmq.deadLetterQueue.messages = []*Message{}
    return nil
}

type DeadLetterHandler interface {
    Handle(message *Message) error
}
```

## 🔧 System Design Coding Problems

### Problem 4: Implement a Distributed Lock
**Difficulty**: Hard  
**Time Complexity**: O(1) for acquire/release  
**Space Complexity**: O(n) where n is the number of locks

```go
// Distributed Lock Implementation
type DistributedLock struct {
    lockID      string
    owner       string
    ttl         time.Duration
    redis       *redis.Client
    mutex       sync.Mutex
    stopChan    chan struct{}
}

func NewDistributedLock(lockID string, owner string, ttl time.Duration, redis *redis.Client) *DistributedLock {
    return &DistributedLock{
        lockID:   lockID,
        owner:    owner,
        ttl:      ttl,
        redis:    redis,
        stopChan: make(chan struct{}),
    }
}

func (dl *DistributedLock) Acquire() (bool, error) {
    dl.mutex.Lock()
    defer dl.mutex.Unlock()
    
    // Try to acquire lock
    result := dl.redis.SetNX(dl.lockID, dl.owner, dl.ttl)
    if result.Err() != nil {
        return false, result.Err()
    }
    
    if result.Val() {
        // Start renewal goroutine
        go dl.renewLock()
        return true, nil
    }
    
    return false, nil
}

func (dl *DistributedLock) Release() error {
    dl.mutex.Lock()
    defer dl.mutex.Unlock()
    
    // Stop renewal goroutine
    close(dl.stopChan)
    
    // Release lock using Lua script to ensure atomicity
    script := `
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
    `
    
    result := dl.redis.Eval(script, []string{dl.lockID}, dl.owner)
    return result.Err()
}

func (dl *DistributedLock) renewLock() {
    ticker := time.NewTicker(dl.ttl / 2)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            // Renew lock
            script := `
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("expire", KEYS[1], ARGV[2])
                else
                    return 0
                end
            `
            
            result := dl.redis.Eval(script, []string{dl.lockID}, dl.owner, int(dl.ttl.Seconds()))
            if result.Err() != nil || result.Val() == 0 {
                // Lock lost, stop renewal
                return
            }
        case <-dl.stopChan:
            return
        }
    }
}

// Lock Manager
type LockManager struct {
    redis *redis.Client
    locks map[string]*DistributedLock
    mutex sync.RWMutex
}

func NewLockManager(redis *redis.Client) *LockManager {
    return &LockManager{
        redis: redis,
        locks: make(map[string]*DistributedLock),
    }
}

func (lm *LockManager) AcquireLock(lockID string, owner string, ttl time.Duration) (*DistributedLock, error) {
    lock := NewDistributedLock(lockID, owner, ttl, lm.redis)
    
    acquired, err := lock.Acquire()
    if err != nil {
        return nil, err
    }
    
    if !acquired {
        return nil, ErrLockNotAcquired
    }
    
    lm.mutex.Lock()
    lm.locks[lockID] = lock
    lm.mutex.Unlock()
    
    return lock, nil
}

func (lm *LockManager) ReleaseLock(lockID string) error {
    lm.mutex.Lock()
    lock, exists := lm.locks[lockID]
    delete(lm.locks, lockID)
    lm.mutex.Unlock()
    
    if !exists {
        return ErrLockNotFound
    }
    
    return lock.Release()
}
```

### Problem 5: Implement a Consistent Hash Ring
**Difficulty**: Hard  
**Time Complexity**: O(log n) for node lookup  
**Space Complexity**: O(n) where n is the number of nodes

```go
// Consistent Hash Ring Implementation
type ConsistentHash struct {
    ring     map[uint32]string
    nodes    []string
    replicas int
    mutex    sync.RWMutex
}

func NewConsistentHash(replicas int) *ConsistentHash {
    return &ConsistentHash{
        ring:     make(map[uint32]string),
        replicas: replicas,
    }
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

func (ch *ConsistentHash) RemoveNode(node string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    for i := 0; i < ch.replicas; i++ {
        hash := ch.hash(fmt.Sprintf("%s:%d", node, i))
        delete(ch.ring, hash)
    }
    
    // Remove from nodes list
    for i, n := range ch.nodes {
        if n == node {
            ch.nodes = append(ch.nodes[:i], ch.nodes[i+1:]...)
            break
        }
    }
    
    ch.sortRing()
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    if len(ch.ring) == 0 {
        return ""
    }
    
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

func (ch *ConsistentHash) GetNodes(key string, count int) []string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    if len(ch.ring) == 0 {
        return []string{}
    }
    
    hash := ch.hash(key)
    nodes := make([]string, 0, count)
    seen := make(map[string]bool)
    
    // Find nodes starting from hash
    for _, nodeHash := range ch.getSortedHashes() {
        if nodeHash >= hash {
            node := ch.ring[nodeHash]
            if !seen[node] {
                nodes = append(nodes, node)
                seen[node] = true
                if len(nodes) >= count {
                    break
                }
            }
        }
    }
    
    // Wrap around if needed
    if len(nodes) < count {
        for _, nodeHash := range ch.getSortedHashes() {
            node := ch.ring[nodeHash]
            if !seen[node] {
                nodes = append(nodes, node)
                seen[node] = true
                if len(nodes) >= count {
                    break
                }
            }
        }
    }
    
    return nodes
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := fnv.New32a()
    h.Write([]byte(key))
    return h.Sum32()
}

func (ch *ConsistentHash) getSortedHashes() []uint32 {
    hashes := make([]uint32, 0, len(ch.ring))
    for hash := range ch.ring {
        hashes = append(hashes, hash)
    }
    sort.Slice(hashes, func(i, j int) bool {
        return hashes[i] < hashes[j]
    })
    return hashes
}

func (ch *ConsistentHash) sortRing() {
    // Ring is already sorted by hash values
    // This method can be used for additional sorting if needed
}
```

## 🎯 Best Practices

### Problem-Solving Approach
1. **Understand the Problem**: Clarify requirements and constraints
2. **Design the Solution**: Plan the architecture and data structures
3. **Implement Step by Step**: Build incrementally and test
4. **Optimize**: Improve time and space complexity
5. **Handle Edge Cases**: Consider error conditions and edge cases

### Coding Standards
1. **Clean Code**: Write readable and maintainable code
2. **Error Handling**: Proper error handling and validation
3. **Testing**: Write unit tests for your solutions
4. **Documentation**: Add comments for complex logic
5. **Performance**: Consider time and space complexity

### Common Patterns
1. **Two Pointers**: For array and string problems
2. **Sliding Window**: For substring and subarray problems
3. **Hash Map**: For frequency counting and lookups
4. **Stack/Queue**: For parsing and traversal problems
5. **Dynamic Programming**: For optimization problems

---

**Last Updated**: December 2024  
**Category**: Advanced Coding Problems  
**Complexity**: Expert Level
