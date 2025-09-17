# Senior Engineer Mock Interviews

Comprehensive mock interview scenarios for senior engineering roles.

## 🎯 Senior Software Engineer Mock Interview

### Interview Format: System Design + Coding + Behavioral
**Duration**: 90 minutes  
**Company**: FAANG+  
**Level**: Senior Software Engineer

#### Round 1: System Design (30 minutes)
**Problem**: Design a real-time chat application like WhatsApp

**Expected Discussion**:
1. **Requirements Clarification**:
   - 1 billion users globally
   - 100 million daily active users
   - Real-time messaging with delivery status
   - Group chats up to 256 members
   - File sharing (images, videos, documents)
   - End-to-end encryption
   - Offline message delivery

2. **High-Level Architecture**:
   - Client applications (mobile, web)
   - API Gateway for load balancing
   - Message service for real-time delivery
   - User service for authentication
   - Notification service for push notifications
   - Media service for file handling
   - Database sharding strategy

3. **Detailed Design**:
   - WebSocket connections for real-time communication
   - Message queuing with Kafka
   - Database design with sharding
   - Caching strategy with Redis
   - CDN for media files
   - Security and encryption

#### Round 2: Coding (30 minutes)
**Problem**: Implement a rate limiter with multiple strategies

```go
type RateLimiter struct {
    strategies map[string]RateLimitStrategy
    mutex      sync.RWMutex
}

type RateLimitStrategy interface {
    Allow(userID string) bool
    GetRemaining(userID string) int
    GetResetTime(userID string) time.Time
}

// Token Bucket Implementation
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
```

#### Round 3: Behavioral (30 minutes)
**Questions**:
1. Tell me about a time when you had to make a difficult technical decision that affected your team.
2. Describe a situation where you had to learn a new technology quickly to solve a problem.
3. How do you handle technical debt in your projects?
4. Tell me about a time when you had to work with a difficult team member.

## 🚀 Staff Engineer Mock Interview

### Interview Format: Technical Leadership + System Design
**Duration**: 120 minutes  
**Company**: Tech Giant  
**Level**: Staff Engineer

#### Round 1: Technical Leadership (40 minutes)
**Scenario**: You're leading a team of 15 engineers to migrate a monolithic application to microservices. The application serves 10 million users and processes 100,000 requests per second.

**Discussion Points**:
1. **Migration Strategy**:
   - How would you approach the migration?
   - What criteria would you use to identify service boundaries?
   - How would you ensure zero downtime during migration?

2. **Team Management**:
   - How would you organize the team for the migration?
   - What skills would you need to develop in your team?
   - How would you handle resistance to change?

3. **Risk Management**:
   - What are the main risks of the migration?
   - How would you mitigate these risks?
   - What rollback strategies would you implement?

#### Round 2: System Design (40 minutes)
**Problem**: Design a distributed file storage system like Google Drive

**Expected Discussion**:
1. **Requirements**:
   - Store 1 billion files
   - Support files up to 10GB
   - Real-time collaboration
   - Version history
   - Global availability
   - 99.99% durability

2. **Architecture**:
   - File upload/download service
   - Metadata service
   - Version control service
   - Collaboration service
   - CDN for global distribution
   - Backup and replication

3. **Data Storage**:
   - File chunking strategy
   - Distributed storage across data centers
   - Replication and redundancy
   - Metadata storage

#### Round 3: Coding (40 minutes)
**Problem**: Implement a consistent hash ring for distributed caching

```go
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

func (ch *ConsistentHash) hash(key string) uint32 {
    h := fnv.New32a()
    h.Write([]byte(key))
    return h.Sum32()
}
```

## 🎯 Principal Engineer Mock Interview

### Interview Format: Architecture + Strategy + Leadership
**Duration**: 150 minutes  
**Company**: Enterprise  
**Level**: Principal Engineer

#### Round 1: Architecture Strategy (50 minutes)
**Scenario**: You're the Principal Engineer responsible for the technical strategy of a 500-person engineering organization. The company is planning to expand globally and needs to support 10x growth in the next 2 years.

**Discussion Points**:
1. **Technical Strategy**:
   - How would you approach the technical strategy for 10x growth?
   - What architectural changes would you recommend?
   - How would you balance technical debt with new features?

2. **Global Expansion**:
   - What technical challenges would you face with global expansion?
   - How would you design for global scale?
   - What compliance and regulatory considerations would you have?

3. **Team Organization**:
   - How would you organize 500 engineers for maximum efficiency?
   - What technical leadership structure would you recommend?
   - How would you ensure knowledge sharing across teams?

#### Round 2: System Design (50 minutes)
**Problem**: Design a global content delivery network (CDN)

**Expected Discussion**:
1. **Requirements**:
   - Serve 1 billion users globally
   - 100TB of content
   - 99.99% availability
   - Sub-100ms latency globally
   - Support for video streaming
   - Real-time content updates

2. **Architecture**:
   - Edge servers distribution
   - Origin server architecture
   - Caching strategies
   - Load balancing
   - Content routing
   - Monitoring and analytics

3. **Global Distribution**:
   - Data center placement strategy
   - Content replication
   - Traffic routing algorithms
   - Failover mechanisms

#### Round 3: Leadership and Innovation (50 minutes)
**Questions**:
1. How would you drive innovation in a large engineering organization?
2. Describe your approach to technical decision-making at scale.
3. How would you handle technical disagreements between teams?
4. Tell me about a time when you had to make a controversial technical decision.

## 🎯 Best Practices for Mock Interviews

### Preparation Strategies
1. **Practice Regularly**: Schedule regular mock interview sessions
2. **Time Management**: Practice with actual time constraints
3. **Record Yourself**: Record mock interviews to identify areas for improvement
4. **Get Feedback**: Seek feedback from experienced interviewers
5. **Study Patterns**: Learn common interview patterns and questions

### Interview Techniques
1. **Clarify Requirements**: Always ask clarifying questions
2. **Think Out Loud**: Explain your thought process
3. **Start Simple**: Begin with basic solutions and iterate
4. **Consider Edge Cases**: Think about error conditions and edge cases
5. **Ask Questions**: Engage with the interviewer

### Common Mistakes to Avoid
1. **Jumping to Solutions**: Take time to understand the problem
2. **Ignoring Constraints**: Consider all given constraints
3. **Not Testing**: Always test your solutions
4. **Poor Communication**: Practice clear and concise communication
5. **Giving Up**: Persist even when you're stuck

---

**Last Updated**: December 2024  
**Category**: Senior Engineer Mock Interviews  
**Complexity**: Senior+ Level
