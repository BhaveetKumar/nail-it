# Expert Coding Problems

Comprehensive expert-level coding problems for senior engineering interviews.

## 🎯 Advanced Algorithm Problems

### Problem 1: Distributed Rate Limiter
**Difficulty**: Hard  
**Time Complexity**: O(1) for each request  
**Space Complexity**: O(n)

```go
// Distributed Rate Limiter with Token Bucket
type DistributedRateLimiter struct {
    redis      *redis.Client
    capacity   int
    refillRate int
    keyPrefix  string
}

func (drl *DistributedRateLimiter) Allow(userID string) bool {
    key := drl.keyPrefix + ":" + userID
    
    script := `
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refillRate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'lastRefill')
        local tokens = tonumber(bucket[1]) or capacity
        local lastRefill = tonumber(bucket[2]) or now
        
        local tokensToAdd = math.floor((now - lastRefill) * refillRate)
        tokens = math.min(capacity, tokens + tokensToAdd)
        
        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', key, 'tokens', tokens, 'lastRefill', now)
            redis.call('EXPIRE', key, 3600)
            return {1, tokens}
        else
            redis.call('HMSET', key, 'tokens', tokens, 'lastRefill', now)
            redis.call('EXPIRE', key, 3600)
            return {0, tokens}
        end
    `
    
    result := drl.redis.Eval(script, []string{key}, 
        drl.capacity, drl.refillRate, time.Now().Unix())
    
    if result.Err() != nil {
        return false
    }
    
    values := result.Val().([]interface{})
    return values[0].(int64) == 1
}
```

### Problem 2: Consistent Hash Ring
**Difficulty**: Hard  
**Time Complexity**: O(log n) for node lookup  
**Space Complexity**: O(n)

```go
// Consistent Hash Ring Implementation
type ConsistentHash struct {
    ring     map[uint32]string
    nodes    []string
    replicas int
    mutex    sync.RWMutex
}

func (ch *ConsistentHash) AddNode(node string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    for i := 0; i < ch.replicas; i++ {
        hash := ch.hash(fmt.Sprintf("%s:%d", node, i))
        ch.ring[hash] = node
    }
    
    ch.nodes = append(ch.nodes, node)
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    if len(ch.ring) == 0 {
        return ""
    }
    
    hash := ch.hash(key)
    
    for _, nodeHash := range ch.getSortedHashes() {
        if nodeHash >= hash {
            return ch.ring[nodeHash]
        }
    }
    
    return ch.ring[ch.getSortedHashes()[0]]
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := fnv.New32a()
    h.Write([]byte(key))
    return h.Sum32()
}
```

### Problem 3: Priority Message Queue
**Difficulty**: Hard  
**Time Complexity**: O(log n) for enqueue, O(1) for dequeue

```go
// Priority Message Queue
type PriorityMessageQueue struct {
    queues map[int]*PriorityQueue
    mutex  sync.RWMutex
}

type Message struct {
    ID       string
    Priority int
    Content  interface{}
    Retries  int
}

type PriorityQueue struct {
    heap  []*Message
    mutex sync.Mutex
}

func (pq *PriorityQueue) Push(msg *Message) {
    pq.mutex.Lock()
    defer pq.mutex.Unlock()
    
    pq.heap = append(pq.heap, msg)
    pq.heapifyUp(len(pq.heap) - 1)
}

func (pq *PriorityQueue) Pop() *Message {
    pq.mutex.Lock()
    defer pq.mutex.Unlock()
    
    if len(pq.heap) == 0 {
        return nil
    }
    
    if len(pq.heap) == 1 {
        msg := pq.heap[0]
        pq.heap = pq.heap[:0]
        return msg
    }
    
    msg := pq.heap[0]
    pq.heap[0] = pq.heap[len(pq.heap)-1]
    pq.heap = pq.heap[:len(pq.heap)-1]
    pq.heapifyDown(0)
    
    return msg
}

func (pq *PriorityQueue) heapifyUp(index int) {
    for index > 0 {
        parent := (index - 1) / 2
        if pq.heap[parent].Priority >= pq.heap[index].Priority {
            break
        }
        pq.heap[parent], pq.heap[index] = pq.heap[index], pq.heap[parent]
        index = parent
    }
}

func (pq *PriorityQueue) heapifyDown(index int) {
    for {
        left := 2*index + 1
        right := 2*index + 2
        largest := index
        
        if left < len(pq.heap) && pq.heap[left].Priority > pq.heap[largest].Priority {
            largest = left
        }
        
        if right < len(pq.heap) && pq.heap[right].Priority > pq.heap[largest].Priority {
            largest = right
        }
        
        if largest == index {
            break
        }
        
        pq.heap[index], pq.heap[largest] = pq.heap[largest], pq.heap[index]
        index = largest
    }
}
```

## 🚀 System Design Coding Problems

### Problem 4: Load Balancer
**Difficulty**: Hard  
**Time Complexity**: O(1) for request routing

```go
// Load Balancer with Health Checks
type LoadBalancer struct {
    servers      []*Server
    algorithm    LoadBalanceAlgorithm
    mutex        sync.RWMutex
}

type Server struct {
    ID     string
    Health HealthStatus
    Weight int
}

type LoadBalanceAlgorithm interface {
    SelectServer(servers []*Server) *Server
}

// Round Robin Algorithm
type RoundRobinAlgorithm struct {
    current int
    mutex   sync.Mutex
}

func (rra *RoundRobinAlgorithm) SelectServer(servers []*Server) *Server {
    rra.mutex.Lock()
    defer rra.mutex.Unlock()
    
    healthyServers := rra.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    server := healthyServers[rra.current%len(healthyServers)]
    rra.current++
    return server
}

func (rra *RoundRobinAlgorithm) getHealthyServers(servers []*Server) []*Server {
    var healthy []*Server
    for _, server := range servers {
        if server.Health == Healthy {
            healthy = append(healthy, server)
        }
    }
    return healthy
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

---

**Last Updated**: December 2024  
**Category**: Expert Coding Problems  
**Complexity**: Expert Level
