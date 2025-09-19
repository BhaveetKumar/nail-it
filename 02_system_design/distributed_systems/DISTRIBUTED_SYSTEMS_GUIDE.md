# 🌐 Distributed Systems Guide

> **Essential guide to distributed systems concepts and patterns**

## 📚 Table of Contents

1. [Fundamentals](#fundamentals)
2. [Consensus Algorithms](#consensus-algorithms)
3. [Load Balancing](#load-balancing)
4. [Fault Tolerance](#fault-tolerance)
5. [Data Storage](#data-storage)

---

## 🎯 Fundamentals

### Key Characteristics
- **Concurrency**: Multiple processes running simultaneously
- **No Global Clock**: Different nodes have different time references
- **Independent Failures**: Nodes can fail independently
- **Scalability**: System can grow by adding more nodes

### CAP Theorem
- **Consistency**: All nodes see the same data
- **Availability**: System remains operational
- **Partition Tolerance**: System continues despite network failures

---

## 🤝 Consensus Algorithms

### Raft Algorithm

```go
type RaftNode struct {
    ID          string
    State       NodeState
    CurrentTerm int
    VotedFor    string
    Log         []LogEntry
}

type NodeState int
const (
    Follower NodeState = iota
    Candidate
    Leader
)

func (rn *RaftNode) StartElection() {
    rn.State = Candidate
    rn.CurrentTerm++
    rn.VotedFor = rn.ID
    
    // Send vote requests to all peers
    for _, peer := range rn.Peers {
        go rn.sendVoteRequest(peer)
    }
}
```

### Paxos Algorithm

```go
type PaxosNode struct {
    ID        string
    Proposers map[string]*Proposer
    Acceptors map[string]*Acceptor
    Learners  map[string]*Learner
}

func (pn *PaxosNode) Propose(value interface{}) error {
    // Phase 1: Prepare
    promises := pn.prepare()
    
    // Phase 2: Accept
    accepts := pn.accept()
    
    // Phase 3: Learn
    pn.learn(value)
    
    return nil
}
```

---

## ⚖️ Load Balancing

### Round Robin

```go
type RoundRobinLoadBalancer struct {
    servers []string
    current int
    mutex   sync.Mutex
}

func (rr *RoundRobinLoadBalancer) GetServer() string {
    rr.mutex.Lock()
    defer rr.mutex.Unlock()
    
    server := rr.servers[rr.current]
    rr.current = (rr.current + 1) % len(rr.servers)
    return server
}
```

### Weighted Round Robin

```go
type WeightedServer struct {
    Server  string
    Weight  int
    Current int
}

func (wrr *WeightedRoundRobinLoadBalancer) GetServer() string {
    maxWeight := 0
    selectedIndex := 0
    
    for i, server := range wrr.servers {
        server.Current += server.Weight
        if server.Current > maxWeight {
            maxWeight = server.Current
            selectedIndex = i
        }
    }
    
    wrr.servers[selectedIndex].Current -= wrr.servers[selectedIndex].Weight
    return wrr.servers[selectedIndex].Server
}
```

---

## 🛡️ Fault Tolerance

### Circuit Breaker

```go
type CircuitBreaker struct {
    maxFailures   int
    resetTimeout  time.Duration
    state         State
    failures      int
    lastFailTime  time.Time
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    if cb.state == StateOpen {
        if time.Since(cb.lastFailTime) > cb.resetTimeout {
            cb.state = StateHalfOpen
        } else {
            return errors.New("circuit breaker is open")
        }
    }
    
    err := fn()
    if err != nil {
        cb.failures++
        if cb.failures >= cb.maxFailures {
            cb.state = StateOpen
        }
        return err
    }
    
    cb.failures = 0
    cb.state = StateClosed
    return nil
}
```

### Retry Pattern

```go
func Retry(maxRetries int, backoff time.Duration, fn func() error) error {
    for attempt := 0; attempt <= maxRetries; attempt++ {
        if attempt > 0 {
            time.Sleep(backoff * time.Duration(attempt))
        }
        
        err := fn()
        if err == nil {
            return nil
        }
    }
    return errors.New("max retries exceeded")
}
```

---

## 💾 Data Storage

### Consistent Hashing

```go
type ConsistentHash struct {
    nodes    []Node
    replicas int
}

type Node struct {
    ID   string
    Hash uint32
}

func (ch *ConsistentHash) GetNode(key string) string {
    hash := ch.hash(key)
    idx := sort.Search(len(ch.nodes), func(i int) bool {
        return ch.nodes[i].Hash >= hash
    })
    
    if idx == len(ch.nodes) {
        idx = 0
    }
    
    return ch.nodes[idx].ID
}
```

### Database Sharding

```go
type ShardedDatabase struct {
    shards   map[string]*sql.DB
    hashRing *ConsistentHash
}

func (sd *ShardedDatabase) GetShard(key string) (*sql.DB, error) {
    shardID := sd.hashRing.GetNode(key)
    shard, exists := sd.shards[shardID]
    if !exists {
        return nil, fmt.Errorf("shard not found: %s", shardID)
    }
    return shard, nil
}
```

---

## 🎯 Best Practices

### 1. Design Principles
- **Idempotency**: Make operations idempotent
- **Graceful Degradation**: System works with reduced functionality
- **Eventual Consistency**: Accept temporary inconsistency

### 2. Communication
- **Synchronous**: HTTP/gRPC for request-response
- **Asynchronous**: Message queues for events
- **Circuit Breaker**: Implement fault tolerance

### 3. Monitoring
- **Distributed Tracing**: Track requests across services
- **Metrics**: Collect system metrics
- **Logging**: Structured logging
- **Alerting**: Proper alerting for failures

---

**🌐 Master these distributed systems concepts to build scalable, reliable systems! 🚀**
