# Advanced Mock Interview Scenarios

Comprehensive mock interview scenarios for senior engineering roles.

## 🎯 Senior Software Engineer Scenarios

### Scenario 1: System Design - Design a Distributed Cache
**Duration**: 45 minutes  
**Level**: Senior Software Engineer  
**Company**: Google/Meta/Amazon

**Problem Statement**:
Design a distributed cache system that can handle 1 billion requests per day with 99.9% availability. The cache should support TTL, eviction policies, and be horizontally scalable.

**Expected Discussion Points**:
1. **Requirements Clarification**:
   - What's the read/write ratio?
   - What's the average data size?
   - What's the acceptable latency?
   - What eviction policies are needed?

2. **High-Level Design**:
   - Consistent hashing for sharding
   - Master-slave replication
   - Load balancer for distribution
   - Cache warming strategies

3. **Detailed Components**:
   - Cache node architecture
   - Data replication strategy
   - Failure handling and recovery
   - Monitoring and metrics

**Sample Solution**:
```go
// Distributed Cache Node
type CacheNode struct {
    ID          string
    Address     string
    Data        map[string]*CacheItem
    Replicas    []*CacheNode
    Ring        *ConsistentHash
    mutex       sync.RWMutex
}

type CacheItem struct {
    Key        string
    Value      interface{}
    TTL        time.Duration
    CreatedAt  time.Time
    AccessCount int64
}

func (cn *CacheNode) Get(key string) (interface{}, bool) {
    cn.mutex.RLock()
    defer cn.mutex.RUnlock()
    
    item, exists := cn.Data[key]
    if !exists {
        return nil, false
    }
    
    // Check TTL
    if time.Since(item.CreatedAt) > item.TTL {
        delete(cn.Data, key)
        return nil, false
    }
    
    // Update access count for LRU
    item.AccessCount++
    
    return item.Value, true
}

func (cn *CacheNode) Set(key string, value interface{}, ttl time.Duration) {
    cn.mutex.Lock()
    defer cn.mutex.Unlock()
    
    item := &CacheItem{
        Key:        key,
        Value:      value,
        TTL:        ttl,
        CreatedAt:  time.Now(),
        AccessCount: 1,
    }
    
    cn.Data[key] = item
    
    // Replicate to other nodes
    cn.replicateToReplicas(key, item)
}
```

### Scenario 2: Coding - Implement a Rate Limiter
**Duration**: 30 minutes  
**Level**: Senior Software Engineer  
**Company**: Stripe/PayPal

**Problem Statement**:
Implement a rate limiter that can handle multiple rate limiting strategies (token bucket, sliding window, fixed window) and support different limits for different users.

**Expected Solution**:
```go
type RateLimiter interface {
    Allow(userID string) bool
    GetRemaining(userID string) int
    GetResetTime(userID string) time.Time
}

type TokenBucketLimiter struct {
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

func (tbl *TokenBucketLimiter) Allow(userID string) bool {
    tbl.mutex.Lock()
    defer tbl.mutex.Unlock()
    
    bucket, exists := tbl.tokens[userID]
    if !exists {
        bucket = &TokenBucket{
            tokens:     tbl.capacity,
            lastRefill: time.Now(),
            capacity:   tbl.capacity,
            refillRate: tbl.refillRate,
        }
        tbl.tokens[userID] = bucket
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
```

## 🚀 Staff Software Engineer Scenarios

### Scenario 3: Architecture Design - Microservices Migration
**Duration**: 60 minutes  
**Level**: Staff Software Engineer  
**Company**: Netflix/Uber

**Problem Statement**:
You need to migrate a monolithic e-commerce application to microservices. The current system handles 10M users, processes 1M orders per day, and has 50+ developers working on it.

**Expected Discussion Points**:
1. **Migration Strategy**:
   - Strangler Fig pattern
   - Database per service
   - Event-driven communication
   - API gateway implementation

2. **Service Decomposition**:
   - Domain boundaries
   - Service dependencies
   - Data consistency
   - Transaction management

3. **Operational Considerations**:
   - Monitoring and observability
   - Deployment strategies
   - Error handling
   - Performance optimization

**Sample Architecture**:
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │   User Service  │    │  Order Service  │
│   (Kong)        │────│   (Node.js)     │────│   (Go)          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Event Bus      │
                    │  (Kafka)        │
                    └─────────────────┘
```

### Scenario 4: Technical Leadership - Performance Optimization
**Duration**: 45 minutes  
**Level**: Staff Software Engineer  
**Company**: LinkedIn/Twitter

**Problem Statement**:
Your team's API response times have increased from 100ms to 2s over the past 6 months. You need to identify the root cause and implement a solution while maintaining system stability.

**Expected Approach**:
1. **Investigation**:
   - Performance profiling
   - Database query analysis
   - Memory usage patterns
   - Network latency analysis

2. **Root Cause Analysis**:
   - N+1 query problems
   - Missing indexes
   - Memory leaks
   - Inefficient algorithms

3. **Solution Implementation**:
   - Query optimization
   - Caching strategies
   - Code refactoring
   - Infrastructure improvements

## 🎯 Principal Engineer Scenarios

### Scenario 5: System Design - Global CDN
**Duration**: 75 minutes  
**Level**: Principal Engineer  
**Company**: Cloudflare/AWS

**Problem Statement**:
Design a global CDN that can serve 100TB of content daily to users worldwide with sub-100ms latency. The system should handle dynamic content, support various content types, and be cost-effective.

**Expected Discussion Points**:
1. **Global Architecture**:
   - Edge server placement
   - Content distribution strategy
   - Caching hierarchies
   - Geographic routing

2. **Technical Challenges**:
   - Cache invalidation
   - Content compression
   - Security considerations
   - DDoS protection

3. **Scalability and Performance**:
   - Load balancing
   - Auto-scaling
   - Performance monitoring
   - Cost optimization

### Scenario 6: Technical Strategy - Technology Modernization
**Duration**: 60 minutes  
**Level**: Principal Engineer  
**Company**: Microsoft/IBM

**Problem Statement**:
Your organization has a legacy system built 15 years ago that's becoming increasingly difficult to maintain. You need to propose a modernization strategy that balances technical debt reduction with business continuity.

**Expected Approach**:
1. **Assessment**:
   - Current system analysis
   - Technical debt quantification
   - Risk assessment
   - Cost-benefit analysis

2. **Strategy Development**:
   - Modernization roadmap
   - Technology selection
   - Migration approach
   - Risk mitigation

3. **Implementation Planning**:
   - Resource allocation
   - Timeline development
   - Success metrics
   - Change management

## 🔧 Coding Interview Scenarios

### Scenario 7: Algorithm Design - Distributed Consensus
**Duration**: 45 minutes  
**Level**: Senior+  
**Company**: Consul/etcd

**Problem Statement**:
Implement a simplified version of the Raft consensus algorithm that can handle leader election and log replication in a distributed system.

**Expected Solution**:
```go
type RaftNode struct {
    ID          string
    State       NodeState
    CurrentTerm int
    VotedFor    string
    Log         []LogEntry
    CommitIndex int
    LastApplied int
    NextIndex   map[string]int
    MatchIndex  map[string]int
}

type LogEntry struct {
    Term    int
    Command interface{}
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
    
    votes := 1 // Vote for self
    
    for _, peer := range rn.Peers {
        go func(peer string) {
            if rn.requestVote(peer) {
                rn.mutex.Lock()
                votes++
                if votes > len(rn.Peers)/2 {
                    rn.becomeLeader()
                }
                rn.mutex.Unlock()
            }
        }(peer)
    }
}

func (rn *RaftNode) becomeLeader() {
    rn.State = Leader
    for peer := range rn.Peers {
        rn.NextIndex[peer] = len(rn.Log)
        rn.MatchIndex[peer] = 0
    }
    
    // Start sending heartbeats
    go rn.sendHeartbeats()
}
```

### Scenario 8: System Design - Real-time Analytics
**Duration**: 50 minutes  
**Level**: Senior+  
**Company**: Databricks/Snowflake

**Problem Statement**:
Design a real-time analytics system that can process 1M events per second and provide sub-second query responses for time-series data.

**Expected Discussion Points**:
1. **Data Pipeline**:
   - Event ingestion (Kafka/Pulsar)
   - Stream processing (Flink/Storm)
   - Data storage (ClickHouse/InfluxDB)
   - Query engine (Presto/Trino)

2. **Performance Optimization**:
   - Data partitioning
   - Indexing strategies
   - Caching layers
   - Query optimization

3. **Scalability**:
   - Horizontal scaling
   - Load balancing
   - Resource management
   - Monitoring

## 🎯 Behavioral Interview Scenarios

### Scenario 9: Leadership Challenge
**Duration**: 30 minutes  
**Level**: Staff+  
**Company**: Any

**Question**: "Tell me about a time when you had to lead a team through a major technical transformation while maintaining business continuity."

**Expected Response Structure**:
- **Situation**: Context and background
- **Task**: Your responsibilities and objectives
- **Action**: Specific steps taken
- **Result**: Outcomes and learnings

**Sample Response**:
"Last year, I led the migration of our payment processing system from a monolithic architecture to microservices. The system processed $2B in transactions annually, so downtime wasn't an option.

I implemented a strangler fig pattern, gradually replacing functionality while maintaining the existing system. We used feature flags to control traffic flow and implemented comprehensive monitoring to ensure system stability.

The migration took 8 months and resulted in 40% faster deployment times, 60% reduction in bugs, and improved team productivity. We maintained 99.99% uptime throughout the process."

### Scenario 10: Technical Decision Making
**Duration**: 25 minutes  
**Level**: Senior+  
**Company**: Any

**Question**: "Describe a time when you had to make a difficult technical decision that affected the entire engineering organization."

**Expected Response Structure**:
- **Situation**: The technical challenge
- **Task**: Decision-making responsibility
- **Action**: Analysis and decision process
- **Result**: Impact and outcomes

**Sample Response**:
"We needed to choose between two database technologies for our new analytics platform. Option A was more mature but had licensing costs, while Option B was open-source but less proven at scale.

I conducted a comprehensive evaluation including performance benchmarks, cost analysis, and team expertise assessment. I also consulted with other engineering teams and created a proof-of-concept for both options.

We chose Option B, which saved $500K annually in licensing costs and improved our team's open-source expertise. The decision also aligned with our company's technology strategy."

## 🎯 Best Practices for Mock Interviews

### Preparation Tips
1. **Study the Company**: Research their technology stack and challenges
2. **Practice Timing**: Ensure you can complete solutions within time limits
3. **Prepare Examples**: Have 5-10 detailed examples ready
4. **Mock with Peers**: Practice with colleagues or mentors
5. **Record Yourself**: Review your performance and improve

### During the Interview
1. **Clarify Requirements**: Ask questions about constraints and expectations
2. **Think Out Loud**: Explain your thought process
3. **Start Simple**: Begin with basic solutions, then optimize
4. **Consider Edge Cases**: Discuss error handling and edge cases
5. **Ask for Feedback**: Seek clarification when needed

### Common Mistakes to Avoid
1. **Jumping to Solutions**: Take time to understand the problem
2. **Ignoring Scalability**: Consider performance and scale
3. **Poor Communication**: Practice explaining technical concepts clearly
4. **Not Testing**: Walk through your solution with examples
5. **Giving Up**: Persist through challenges and ask for help

---

**Last Updated**: December 2024  
**Category**: Advanced Mock Interview Scenarios  
**Complexity**: Senior+ Level
