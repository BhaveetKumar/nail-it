# 🎯 Comprehensive Mock Interview Scenarios

> **Realistic interview scenarios for Razorpay and top tech companies**

## 📚 Table of Contents

1. [Coding Interviews](#-coding-interviews)
2. [System Design Interviews](#-system-design-interviews)
3. [Technical Deep Dive](#-technical-deep-dive)
4. [Behavioral Interviews](#-behavioral-interviews)
5. [Leadership Interviews](#-leadership-interviews)
6. [Razorpay-Specific Scenarios](#-razorpay-specific-scenarios)
7. [Interview Evaluation Rubrics](#-interview-evaluation-rubrics)

---

## 💻 Coding Interviews

### Senior Software Engineer - Round 1

**Duration**: 45 minutes  
**Format**: Live coding on shared screen  
**Language**: Go (preferred) or any language of choice

#### Problem 1: Payment Processing System

**Problem Statement**:
Design a payment processing system that can handle multiple payment methods (credit card, UPI, net banking) with the following requirements:

1. **Process Payment**: Validate and process payments
2. **Refund Payment**: Handle refunds with proper validation
3. **Payment History**: Retrieve payment history for a user
4. **Fraud Detection**: Basic fraud detection based on amount and frequency

**Expected Solution**:
```go
package main

import (
    "context"
    "errors"
    "fmt"
    "sync"
    "time"
)

type PaymentMethod string

const (
    CreditCard PaymentMethod = "credit_card"
    UPI         PaymentMethod = "upi"
    NetBanking  PaymentMethod = "net_banking"
)

type PaymentStatus string

const (
    Pending   PaymentStatus = "pending"
    Processing PaymentStatus = "processing"
    Completed  PaymentStatus = "completed"
    Failed     PaymentStatus = "failed"
    Refunded   PaymentStatus = "refunded"
)

type Payment struct {
    ID            string        `json:"id"`
    UserID        string        `json:"user_id"`
    Amount        float64       `json:"amount"`
    Currency      string        `json:"currency"`
    Method        PaymentMethod `json:"method"`
    Status        PaymentStatus `json:"status"`
    CreatedAt     time.Time     `json:"created_at"`
    ProcessedAt   *time.Time    `json:"processed_at,omitempty"`
    RefundedAt    *time.Time    `json:"refunded_at,omitempty"`
    Metadata      map[string]interface{} `json:"metadata"`
}

type PaymentService struct {
    payments map[string]*Payment
    mutex    sync.RWMutex
    fraudDetector *FraudDetector
}

type FraudDetector struct {
    userPayments map[string][]*Payment
    mutex        sync.RWMutex
}

func NewPaymentService() *PaymentService {
    return &PaymentService{
        payments: make(map[string]*Payment),
        fraudDetector: &FraudDetector{
            userPayments: make(map[string][]*Payment),
        },
    }
}

func (ps *PaymentService) ProcessPayment(ctx context.Context, req *ProcessPaymentRequest) (*Payment, error) {
    // Validate request
    if err := ps.validatePaymentRequest(req); err != nil {
        return nil, err
    }
    
    // Check for fraud
    if ps.fraudDetector.IsFraudulent(req.UserID, req.Amount) {
        return nil, errors.New("payment flagged as potentially fraudulent")
    }
    
    // Create payment
    payment := &Payment{
        ID:        generatePaymentID(),
        UserID:    req.UserID,
        Amount:    req.Amount,
        Currency:  req.Currency,
        Method:    req.Method,
        Status:    Pending,
        CreatedAt: time.Now(),
        Metadata:  req.Metadata,
    }
    
    // Store payment
    ps.mutex.Lock()
    ps.payments[payment.ID] = payment
    ps.mutex.Unlock()
    
    // Process payment asynchronously
    go ps.processPaymentAsync(ctx, payment)
    
    return payment, nil
}

func (ps *PaymentService) processPaymentAsync(ctx context.Context, payment *Payment) {
    // Simulate payment processing
    time.Sleep(100 * time.Millisecond)
    
    ps.mutex.Lock()
    payment.Status = Processing
    ps.mutex.Unlock()
    
    // Simulate payment gateway call
    success := ps.callPaymentGateway(payment)
    
    ps.mutex.Lock()
    if success {
        payment.Status = Completed
        now := time.Now()
        payment.ProcessedAt = &now
    } else {
        payment.Status = Failed
    }
    ps.mutex.Unlock()
    
    // Update fraud detector
    ps.fraudDetector.RecordPayment(payment)
}

func (ps *PaymentService) RefundPayment(ctx context.Context, paymentID string, amount float64) error {
    ps.mutex.Lock()
    payment, exists := ps.payments[paymentID]
    ps.mutex.Unlock()
    
    if !exists {
        return errors.New("payment not found")
    }
    
    if payment.Status != Completed {
        return errors.New("can only refund completed payments")
    }
    
    if amount > payment.Amount {
        return errors.New("refund amount cannot exceed payment amount")
    }
    
    ps.mutex.Lock()
    payment.Status = Refunded
    now := time.Now()
    payment.RefundedAt = &now
    ps.mutex.Unlock()
    
    return nil
}

func (ps *PaymentService) GetPaymentHistory(ctx context.Context, userID string, limit int) ([]*Payment, error) {
    ps.mutex.RLock()
    defer ps.mutex.RUnlock()
    
    var userPayments []*Payment
    for _, payment := range ps.payments {
        if payment.UserID == userID {
            userPayments = append(userPayments, payment)
        }
    }
    
    // Sort by created date (newest first)
    sort.Slice(userPayments, func(i, j int) bool {
        return userPayments[i].CreatedAt.After(userPayments[j].CreatedAt)
    })
    
    if limit > 0 && len(userPayments) > limit {
        userPayments = userPayments[:limit]
    }
    
    return userPayments, nil
}

func (ps *PaymentService) validatePaymentRequest(req *ProcessPaymentRequest) error {
    if req.UserID == "" {
        return errors.New("user ID is required")
    }
    if req.Amount <= 0 {
        return errors.New("amount must be positive")
    }
    if req.Currency == "" {
        return errors.New("currency is required")
    }
    if req.Method == "" {
        return errors.New("payment method is required")
    }
    return nil
}

func (ps *PaymentService) callPaymentGateway(payment *Payment) bool {
    // Simulate payment gateway response
    // In real implementation, this would call external payment gateway
    return true
}

type ProcessPaymentRequest struct {
    UserID   string                 `json:"user_id"`
    Amount   float64                `json:"amount"`
    Currency string                 `json:"currency"`
    Method   PaymentMethod          `json:"method"`
    Metadata map[string]interface{} `json:"metadata"`
}

func (fd *FraudDetector) IsFraudulent(userID string, amount float64) bool {
    fd.mutex.RLock()
    payments := fd.userPayments[userID]
    fd.mutex.RUnlock()
    
    // Simple fraud detection: flag if user has made more than 10 payments in last hour
    // or if amount is greater than 1 lakh
    recentPayments := 0
    oneHourAgo := time.Now().Add(-1 * time.Hour)
    
    for _, payment := range payments {
        if payment.CreatedAt.After(oneHourAgo) {
            recentPayments++
        }
    }
    
    return recentPayments > 10 || amount > 100000
}

func (fd *FraudDetector) RecordPayment(payment *Payment) {
    fd.mutex.Lock()
    defer fd.mutex.Unlock()
    
    fd.userPayments[payment.UserID] = append(fd.userPayments[payment.UserID], payment)
}

func generatePaymentID() string {
    return fmt.Sprintf("pay_%d", time.Now().UnixNano())
}

func main() {
    service := NewPaymentService()
    
    // Example usage
    req := &ProcessPaymentRequest{
        UserID:   "user_123",
        Amount:   1000.0,
        Currency: "INR",
        Method:   UPI,
        Metadata: map[string]interface{}{
            "upi_id": "user@paytm",
        },
    }
    
    payment, err := service.ProcessPayment(context.Background(), req)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("Payment created: %+v\n", payment)
}
```

**Follow-up Questions**:
1. How would you handle payment failures and retries?
2. How would you implement idempotency for payment processing?
3. How would you scale this system to handle millions of payments?
4. How would you implement real-time fraud detection?

#### Problem 2: Rate Limiting

**Problem Statement**:
Implement a rate limiter that can handle different rate limiting strategies:
- Fixed window
- Sliding window
- Token bucket
- Leaky bucket

**Expected Solution**:
```go
package main

import (
    "context"
    "sync"
    "time"
)

type RateLimiter interface {
    Allow(ctx context.Context, key string) bool
}

// Fixed Window Rate Limiter
type FixedWindowLimiter struct {
    requests map[string][]time.Time
    window   time.Duration
    limit    int
    mutex    sync.RWMutex
}

func NewFixedWindowLimiter(window time.Duration, limit int) *FixedWindowLimiter {
    return &FixedWindowLimiter{
        requests: make(map[string][]time.Time),
        window:   window,
        limit:    limit,
    }
}

func (fwl *FixedWindowLimiter) Allow(ctx context.Context, key string) bool {
    fwl.mutex.Lock()
    defer fwl.mutex.Unlock()
    
    now := time.Now()
    windowStart := now.Truncate(fwl.window)
    
    // Clean old requests
    var validRequests []time.Time
    for _, reqTime := range fwl.requests[key] {
        if reqTime.After(windowStart) {
            validRequests = append(validRequests, reqTime)
        }
    }
    
    if len(validRequests) >= fwl.limit {
        return false
    }
    
    validRequests = append(validRequests, now)
    fwl.requests[key] = validRequests
    return true
}

// Sliding Window Rate Limiter
type SlidingWindowLimiter struct {
    requests map[string][]time.Time
    window   time.Duration
    limit    int
    mutex    sync.RWMutex
}

func NewSlidingWindowLimiter(window time.Duration, limit int) *SlidingWindowLimiter {
    return &SlidingWindowLimiter{
        requests: make(map[string][]time.Time),
        window:   window,
        limit:    limit,
    }
}

func (swl *SlidingWindowLimiter) Allow(ctx context.Context, key string) bool {
    swl.mutex.Lock()
    defer swl.mutex.Unlock()
    
    now := time.Now()
    windowStart := now.Add(-swl.window)
    
    // Clean old requests
    var validRequests []time.Time
    for _, reqTime := range swl.requests[key] {
        if reqTime.After(windowStart) {
            validRequests = append(validRequests, reqTime)
        }
    }
    
    if len(validRequests) >= swl.limit {
        return false
    }
    
    validRequests = append(validRequests, now)
    swl.requests[key] = validRequests
    return true
}

// Token Bucket Rate Limiter
type TokenBucketLimiter struct {
    buckets map[string]*TokenBucket
    mutex   sync.RWMutex
}

type TokenBucket struct {
    tokens     int
    capacity   int
    refillRate int
    lastRefill time.Time
}

func NewTokenBucketLimiter(capacity, refillRate int) *TokenBucketLimiter {
    return &TokenBucketLimiter{
        buckets: make(map[string]*TokenBucket),
    }
}

func (tbl *TokenBucketLimiter) Allow(ctx context.Context, key string) bool {
    tbl.mutex.Lock()
    defer tbl.mutex.Unlock()
    
    bucket, exists := tbl.buckets[key]
    if !exists {
        bucket = &TokenBucket{
            tokens:     tbl.capacity,
            capacity:   tbl.capacity,
            refillRate: tbl.refillRate,
            lastRefill: time.Now(),
        }
        tbl.buckets[key] = bucket
    }
    
    now := time.Now()
    timePassed := now.Sub(bucket.lastRefill)
    tokensToAdd := int(timePassed.Seconds()) * bucket.refillRate
    
    bucket.tokens = min(bucket.capacity, bucket.tokens+tokensToAdd)
    bucket.lastRefill = now
    
    if bucket.tokens > 0 {
        bucket.tokens--
        return true
    }
    
    return false
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// Leaky Bucket Rate Limiter
type LeakyBucketLimiter struct {
    buckets map[string]*LeakyBucket
    mutex   sync.RWMutex
}

type LeakyBucket struct {
    capacity     int
    currentLevel int
    leakRate     int
    lastLeak     time.Time
}

func NewLeakyBucketLimiter(capacity, leakRate int) *LeakyBucketLimiter {
    return &LeakyBucketLimiter{
        buckets: make(map[string]*LeakyBucket),
    }
}

func (lbl *LeakyBucketLimiter) Allow(ctx context.Context, key string) bool {
    lbl.mutex.Lock()
    defer lbl.mutex.Unlock()
    
    bucket, exists := lbl.buckets[key]
    if !exists {
        bucket = &LeakyBucket{
            capacity:     lbl.capacity,
            currentLevel: 0,
            leakRate:     lbl.leakRate,
            lastLeak:     time.Now(),
        }
        lbl.buckets[key] = bucket
    }
    
    now := time.Now()
    timePassed := now.Sub(bucket.lastLeak)
    leaked := int(timePassed.Seconds()) * bucket.leakRate
    
    bucket.currentLevel = max(0, bucket.currentLevel-leaked)
    bucket.lastLeak = now
    
    if bucket.currentLevel < bucket.capacity {
        bucket.currentLevel++
        return true
    }
    
    return false
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

**Follow-up Questions**:
1. How would you implement distributed rate limiting?
2. How would you handle rate limiting for different user tiers?
3. How would you implement rate limiting with Redis?
4. How would you handle rate limiting for APIs with different endpoints?

---

## 🏗️ System Design Interviews

### Staff Software Engineer - Round 2

**Duration**: 60 minutes  
**Format**: Whiteboard/online drawing tool  
**Focus**: High-level design and scalability

#### Problem: Design a Real-time Notification System

**Requirements**:
- Send notifications to millions of users
- Support multiple notification types (email, SMS, push, in-app)
- Real-time delivery (within 5 seconds)
- Handle different user preferences
- Support batching and rate limiting
- 99.9% uptime

**Expected Discussion Points**:

1. **Requirements Clarification**:
   - What types of notifications? (transactional, marketing, system alerts)
   - What's the expected volume? (1M notifications/hour)
   - What are the delivery guarantees? (at-least-once, at-most-once)
   - What are the user preference options?

2. **High-Level Design**:
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Web Portal    │    │   Admin Panel   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    API Gateway            │
                    │  (Authentication,         │
                    │   Rate Limiting)          │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │   Notification Service    │
                    │  (Core Business Logic)    │
                    └─────────────┬─────────────┘
                                 │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────┴───────┐    ┌─────────┴───────┐    ┌─────────┴───────┐
│  Email Service  │    │   SMS Service   │    │  Push Service   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

3. **Detailed Components**:

**Notification Service**:
```go
type NotificationService struct {
    userRepo        UserRepository
    templateRepo    TemplateRepository
    deliveryService DeliveryService
    queueService    QueueService
    analytics       AnalyticsService
}

type Notification struct {
    ID          string                 `json:"id"`
    UserID      string                 `json:"user_id"`
    Type        NotificationType       `json:"type"`
    Channel     DeliveryChannel        `json:"channel"`
    TemplateID  string                 `json:"template_id"`
    Data        map[string]interface{} `json:"data"`
    Priority    Priority               `json:"priority"`
    ScheduledAt time.Time              `json:"scheduled_at"`
    Status      NotificationStatus     `json:"status"`
}

type DeliveryChannel string

const (
    Email   DeliveryChannel = "email"
    SMS     DeliveryChannel = "sms"
    Push    DeliveryChannel = "push"
    InApp   DeliveryChannel = "in_app"
)

func (ns *NotificationService) SendNotification(ctx context.Context, req *SendNotificationRequest) error {
    // Validate user preferences
    preferences, err := ns.userRepo.GetNotificationPreferences(req.UserID)
    if err != nil {
        return err
    }
    
    if !preferences.IsChannelEnabled(req.Channel) {
        return errors.New("channel not enabled for user")
    }
    
    // Create notification
    notification := &Notification{
        ID:         generateNotificationID(),
        UserID:     req.UserID,
        Type:       req.Type,
        Channel:    req.Channel,
        TemplateID: req.TemplateID,
        Data:       req.Data,
        Priority:   req.Priority,
        Status:     Pending,
    }
    
    // Queue for processing
    return ns.queueService.Enqueue(ctx, notification)
}
```

**Message Queue Architecture**:
- **Kafka**: For high-throughput message streaming
- **Partitioning**: By user ID for ordering
- **Consumer Groups**: Different groups for different channels

**Database Design**:
```sql
-- Users table
CREATE TABLE users (
    id VARCHAR(255) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Notification preferences
CREATE TABLE notification_preferences (
    user_id VARCHAR(255) PRIMARY KEY,
    email_enabled BOOLEAN DEFAULT TRUE,
    sms_enabled BOOLEAN DEFAULT TRUE,
    push_enabled BOOLEAN DEFAULT TRUE,
    in_app_enabled BOOLEAN DEFAULT TRUE,
    quiet_hours_start TIME,
    quiet_hours_end TIME,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Notifications table
CREATE TABLE notifications (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    template_id VARCHAR(255),
    data JSON,
    priority INT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    scheduled_at TIMESTAMP,
    delivered_at TIMESTAMP,
    failed_at TIMESTAMP,
    retry_count INT DEFAULT 0,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_scheduled_at (scheduled_at)
);

-- Delivery attempts
CREATE TABLE delivery_attempts (
    id VARCHAR(255) PRIMARY KEY,
    notification_id VARCHAR(255) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (notification_id) REFERENCES notifications(id)
);
```

4. **Scalability Considerations**:
- **Horizontal Scaling**: Multiple notification service instances
- **Database Sharding**: By user ID
- **Caching**: Redis for user preferences and templates
- **CDN**: For static content and templates
- **Load Balancing**: Round-robin with health checks

5. **Monitoring & Observability**:
- **Metrics**: Delivery rates, latency, error rates
- **Logging**: Structured logging with correlation IDs
- **Tracing**: Distributed tracing across services
- **Alerting**: SLA violations, error rate spikes

**Follow-up Questions**:
1. How would you handle notification batching?
2. How would you implement retry logic with exponential backoff?
3. How would you handle dead letter queues?
4. How would you implement notification templates?
5. How would you handle internationalization?

---

## 🔍 Technical Deep Dive

### Principal Engineer - Round 3

**Duration**: 90 minutes  
**Format**: Deep technical discussion  
**Focus**: Architecture decisions, trade-offs, and implementation details

#### Problem: Design a Distributed Caching System

**Requirements**:
- Sub-millisecond read latency
- 99.99% availability
- Handle 1M+ requests per second
- Support different eviction policies
- Handle cache invalidation
- Support multiple data types

**Expected Discussion**:

1. **Architecture Design**:
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client 1      │    │   Client 2      │    │   Client N      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    Load Balancer          │
                    │  (Consistent Hashing)     │
                    └─────────────┬─────────────┘
                                 │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────┴───────┐    ┌─────────┴───────┐    ┌─────────┴───────┐
│  Cache Node 1   │    │  Cache Node 2   │    │  Cache Node N   │
│  (Redis)        │    │  (Redis)        │    │  (Redis)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

2. **Consistent Hashing Implementation**:
```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "strconv"
)

type ConsistentHash struct {
    nodes    []Node
    replicas int
    ring     map[uint32]Node
}

type Node struct {
    ID   string
    Hash uint32
}

func NewConsistentHash(replicas int) *ConsistentHash {
    return &ConsistentHash{
        replicas: replicas,
        ring:     make(map[uint32]Node),
    }
}

func (ch *ConsistentHash) AddNode(nodeID string) {
    for i := 0; i < ch.replicas; i++ {
        hash := ch.hash(fmt.Sprintf("%s:%d", nodeID, i))
        node := Node{ID: nodeID, Hash: hash}
        ch.ring[hash] = node
        ch.nodes = append(ch.nodes, node)
    }
    sort.Slice(ch.nodes, func(i, j int) bool {
        return ch.nodes[i].Hash < ch.nodes[j].Hash
    })
}

func (ch *ConsistentHash) GetNode(key string) Node {
    hash := ch.hash(key)
    
    // Find the first node with hash >= key hash
    for _, node := range ch.nodes {
        if node.Hash >= hash {
            return node
        }
    }
    
    // Wrap around to the first node
    return ch.nodes[0]
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}
```

3. **Cache Implementation**:
```go
type CacheService struct {
    nodes    []CacheNode
    hashRing *ConsistentHash
    config   CacheConfig
}

type CacheNode struct {
    ID     string
    Client *redis.Client
}

type CacheConfig struct {
    DefaultTTL    time.Duration
    MaxMemory     int64
    EvictionPolicy string
}

func (cs *CacheService) Get(ctx context.Context, key string) (interface{}, error) {
    node := cs.hashRing.GetNode(key)
    
    // Try to get from cache
    value, err := node.Client.Get(ctx, key).Result()
    if err == redis.Nil {
        return nil, ErrCacheMiss
    }
    if err != nil {
        return nil, err
    }
    
    return value, nil
}

func (cs *CacheService) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    node := cs.hashRing.GetNode(key)
    
    if ttl == 0 {
        ttl = cs.config.DefaultTTL
    }
    
    return node.Client.Set(ctx, key, value, ttl).Err()
}

func (cs *CacheService) Delete(ctx context.Context, key string) error {
    node := cs.hashRing.GetNode(key)
    return node.Client.Del(ctx, key).Err()
}
```

4. **Cache Invalidation Strategies**:
- **TTL-based**: Automatic expiration
- **Event-driven**: Invalidate on data changes
- **Version-based**: Include version in cache key
- **Tag-based**: Invalidate by tags

5. **Monitoring & Metrics**:
- **Hit Rate**: Percentage of cache hits
- **Latency**: P50, P95, P99 latencies
- **Memory Usage**: Per-node memory consumption
- **Error Rate**: Failed operations

**Follow-up Questions**:
1. How would you handle cache warming?
2. How would you implement cache compression?
3. How would you handle cache consistency across regions?
4. How would you implement cache analytics?

---

## 🤝 Behavioral Interviews

### Engineering Manager - Round 4

**Duration**: 45 minutes  
**Format**: Behavioral questions with STAR method  
**Focus**: Leadership, conflict resolution, and team management

#### Question 1: Leadership Experience

**Question**: "Tell me about a time when you had to lead a team through a difficult technical challenge."

**Expected Response Structure (STAR)**:

**Situation**: 
"In my previous role at [Company], our payment processing system was experiencing 15% failure rates during peak hours, causing significant revenue loss and customer complaints. The issue was affecting our core business and needed immediate resolution."

**Task**: 
"I was tasked with leading a cross-functional team of 8 engineers (backend, frontend, DevOps, and QA) to identify the root cause and implement a solution within 2 weeks while maintaining system stability."

**Action**: 
"I took the following approach:
1. **Immediate Response**: Set up a war room with 24/7 monitoring and established a communication channel for real-time updates
2. **Root Cause Analysis**: Led daily standups to analyze logs, metrics, and user reports. We discovered the issue was in our database connection pooling under high load
3. **Team Coordination**: Assigned specific responsibilities - backend team focused on connection pool optimization, DevOps team on infrastructure scaling, QA team on load testing
4. **Risk Mitigation**: Implemented circuit breakers and fallback mechanisms to prevent cascading failures
5. **Communication**: Provided daily updates to stakeholders and maintained transparency about progress and challenges"

**Result**: 
"We successfully reduced the failure rate from 15% to 0.1% within 10 days. The solution included:
- Optimized database connection pooling (reduced connection time from 500ms to 50ms)
- Implemented horizontal scaling (increased capacity by 300%)
- Added comprehensive monitoring and alerting
- Established post-incident review process

The team's morale improved significantly, and we received recognition from leadership. The incident also led to the development of better monitoring tools that prevented similar issues in the future."

#### Question 2: Conflict Resolution

**Question**: "Describe a situation where you had to resolve a conflict between team members."

**Expected Response Structure (STAR)**:

**Situation**: 
"Two senior engineers on my team had a disagreement about the architecture for our new microservices system. One engineer (let's call him John) advocated for a service mesh approach with Istio, while another (Sarah) preferred a simpler API gateway solution. The conflict was affecting team productivity and project timeline."

**Task**: 
"I needed to facilitate a resolution that would:
- Address both engineers' concerns and expertise
- Ensure the best technical solution for our use case
- Maintain team harmony and collaboration
- Keep the project on track"

**Action**: 
"I took the following steps:
1. **Individual Meetings**: Met with each engineer separately to understand their perspectives, concerns, and underlying motivations
2. **Technical Evaluation**: Organized a technical review session where both engineers presented their approaches with pros/cons, performance benchmarks, and implementation complexity
3. **Stakeholder Input**: Involved the product team and DevOps team to understand business requirements and operational constraints
4. **Collaborative Solution**: Facilitated a workshop where both engineers worked together to design a hybrid approach that incorporated the best of both solutions
5. **Decision Framework**: Established clear criteria for future architectural decisions to prevent similar conflicts"

**Result**: 
"We implemented a hybrid solution that:
- Used API gateway for external traffic (Sarah's approach)
- Implemented service mesh for internal service-to-service communication (John's approach)
- Leveraged both engineers' expertise in their respective areas
- Delivered the project on time with improved performance

Both engineers felt heard and valued, and the collaboration actually strengthened their working relationship. The framework we established has been used for subsequent architectural decisions."

#### Question 3: Team Development

**Question**: "How do you develop and mentor junior engineers on your team?"

**Expected Response Structure (STAR)**:

**Situation**: 
"When I joined as Engineering Manager, I inherited a team with 3 junior engineers who had been with the company for 6-12 months but were struggling with confidence and technical growth. They were hesitant to take on challenging tasks and often needed extensive guidance on basic technical decisions."

**Task**: 
"I needed to create a structured approach to develop these engineers into confident, independent contributors while maintaining team productivity and project delivery."

**Action**: 
"I implemented a comprehensive mentoring program:

1. **Individual Development Plans**: Created personalized growth plans for each engineer based on their interests and career goals
2. **Technical Mentoring**: Assigned senior engineers as mentors and established regular 1:1 sessions
3. **Learning Opportunities**: 
   - Encouraged participation in code reviews and technical discussions
   - Assigned progressively challenging tasks with clear success criteria
   - Provided access to online courses and conference attendance
4. **Safe Learning Environment**: Created a culture where asking questions and making mistakes was encouraged
5. **Regular Feedback**: Implemented weekly check-ins and quarterly performance reviews
6. **Career Pathing**: Clearly defined promotion criteria and growth opportunities"

**Result**: 
"Within 6 months:
- All 3 junior engineers were independently leading small features
- 2 engineers received promotions to mid-level
- Team velocity increased by 40% as engineers became more confident
- Knowledge sharing improved significantly with engineers presenting at team tech talks
- Employee satisfaction scores increased from 6.5/10 to 9.2/10

The mentoring program became a template for other teams in the organization, and I was recognized for developing a strong engineering culture."

---

## 🎯 Razorpay-Specific Scenarios

### Senior Software Engineer - Razorpay

**Duration**: 60 minutes  
**Format**: Technical + Behavioral  
**Focus**: Fintech domain knowledge and Razorpay-specific challenges

#### Problem: Design a UPI Payment System

**Requirements**:
- Handle UPI payment requests
- Integrate with NPCI (National Payments Corporation of India)
- Support multiple UPI apps (PhonePe, Google Pay, Paytm)
- Handle real-time settlement
- Implement fraud detection
- Ensure compliance with RBI guidelines

**Expected Discussion**:

1. **UPI Architecture**:
```go
type UPIPaymentService struct {
    npciClient      *NPCIClient
    bankConnector   *BankConnector
    fraudDetector   *FraudDetector
    settlementEngine *SettlementEngine
    auditLogger     *AuditLogger
    rateLimiter     *RateLimiter
    circuitBreaker  *CircuitBreaker
}

type UPIPaymentRequest struct {
    TransactionID string  `json:"transaction_id"`
    Amount        float64 `json:"amount"`
    UPIID         string  `json:"upi_id"`
    MerchantID    string  `json:"merchant_id"`
    OrderID       string  `json:"order_id"`
    Description   string  `json:"description"`
    ExpiryTime    int64   `json:"expiry_time"`
}

type UPIPaymentResponse struct {
    TransactionID string           `json:"transaction_id"`
    Status        PaymentStatus    `json:"status"`
    UPIReference  string           `json:"upi_reference"`
    BankReference string           `json:"bank_reference"`
    ErrorCode     string           `json:"error_code,omitempty"`
    ErrorMessage  string           `json:"error_message,omitempty"`
    Timestamp     time.Time        `json:"timestamp"`
}

func (ups *UPIPaymentService) ProcessPayment(ctx context.Context, req *UPIPaymentRequest) (*UPIPaymentResponse, error) {
    // Validate request
    if err := ups.validateUPIRequest(req); err != nil {
        return nil, err
    }
    
    // Check fraud
    if ups.fraudDetector.IsFraudulent(req) {
        return &UPIPaymentResponse{
            TransactionID: req.TransactionID,
            Status:        Failed,
            ErrorCode:     "FRAUD_DETECTED",
            ErrorMessage:  "Transaction flagged as potentially fraudulent",
            Timestamp:     time.Now(),
        }, nil
    }
    
    // Rate limiting
    if !ups.rateLimiter.Allow(ctx, req.UPIID) {
        return &UPIPaymentResponse{
            TransactionID: req.TransactionID,
            Status:        Failed,
            ErrorCode:     "RATE_LIMITED",
            ErrorMessage:  "Too many requests",
            Timestamp:     time.Now(),
        }, nil
    }
    
    // Process with NPCI
    npciResp, err := ups.npciClient.ProcessPayment(ctx, req)
    if err != nil {
        return &UPIPaymentResponse{
            TransactionID: req.TransactionID,
            Status:        Failed,
            ErrorCode:     "NPCI_ERROR",
            ErrorMessage:  err.Error(),
            Timestamp:     time.Now(),
        }, nil
    }
    
    // Handle response
    response := &UPIPaymentResponse{
        TransactionID: req.TransactionID,
        Status:        mapNPCIStatus(npciResp.Status),
        UPIReference:  npciResp.UPIReference,
        BankReference: npciResp.BankReference,
        Timestamp:     time.Now(),
    }
    
    // Log for audit
    ups.auditLogger.LogPayment(req, response)
    
    // Trigger settlement if successful
    if response.Status == Completed {
        go ups.settlementEngine.ProcessSettlement(ctx, req.TransactionID)
    }
    
    return response, nil
}
```

2. **Fraud Detection**:
```go
type FraudDetector struct {
    rules []FraudRule
    mlModel MLModel
}

type FraudRule interface {
    Evaluate(ctx context.Context, req *UPIPaymentRequest) (bool, string)
}

type AmountRule struct {
    maxAmount float64
}

func (ar *AmountRule) Evaluate(ctx context.Context, req *UPIPaymentRequest) (bool, string) {
    if req.Amount > ar.maxAmount {
        return true, "Amount exceeds maximum limit"
    }
    return false, ""
}

type FrequencyRule struct {
    maxTransactions int
    timeWindow      time.Duration
}

func (fr *FrequencyRule) Evaluate(ctx context.Context, req *UPIPaymentRequest) (bool, string) {
    // Check transaction frequency for UPI ID
    count := fr.getTransactionCount(req.UPIID, fr.timeWindow)
    if count > fr.maxTransactions {
        return true, "Too many transactions in time window"
    }
    return false, ""
}

func (fd *FraudDetector) IsFraudulent(req *UPIPaymentRequest) bool {
    for _, rule := range fd.rules {
        if isFraud, reason := rule.Evaluate(context.Background(), req); isFraud {
            log.Printf("Fraud detected: %s", reason)
            return true
        }
    }
    
    // ML-based detection
    if fd.mlModel != nil {
        return fd.mlModel.Predict(req) > 0.8
    }
    
    return false
}
```

3. **Settlement System**:
```go
type SettlementEngine struct {
    bankConnector *BankConnector
    db           *Database
    queue        *Queue
}

func (se *SettlementEngine) ProcessSettlement(ctx context.Context, transactionID string) error {
    // Get transaction details
    tx, err := se.db.GetTransaction(transactionID)
    if err != nil {
        return err
    }
    
    // Create settlement record
    settlement := &Settlement{
        ID:            generateSettlementID(),
        TransactionID: transactionID,
        Amount:        tx.Amount,
        MerchantID:    tx.MerchantID,
        Status:        Pending,
        CreatedAt:     time.Now(),
    }
    
    // Queue for processing
    return se.queue.Enqueue(ctx, settlement)
}

func (se *SettlementEngine) ProcessSettlementBatch(ctx context.Context, settlements []*Settlement) error {
    // Group by bank
    bankGroups := make(map[string][]*Settlement)
    for _, settlement := range settlements {
        bankID := se.getBankID(settlement.MerchantID)
        bankGroups[bankID] = append(bankGroups[bankID], settlement)
    }
    
    // Process each bank group
    for bankID, group := range bankGroups {
        if err := se.processBankSettlement(ctx, bankID, group); err != nil {
            log.Printf("Failed to process settlement for bank %s: %v", bankID, err)
        }
    }
    
    return nil
}
```

**Follow-up Questions**:
1. How would you handle UPI app failures?
2. How would you implement real-time fraud detection?
3. How would you ensure compliance with RBI guidelines?
4. How would you handle settlement failures?
5. How would you scale this system for high volume?

---

## 📊 Interview Evaluation Rubrics

### Coding Interview Rubric

| Criteria | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|----------|---------------|----------|------------------|----------------------|
| **Problem Understanding** | Quickly grasps requirements, asks clarifying questions | Understands most requirements, asks some questions | Basic understanding, few questions | Misunderstands requirements |
| **Solution Design** | Clean, efficient, scalable design | Good design with minor issues | Basic design with some problems | Poor design, major issues |
| **Code Quality** | Clean, readable, well-structured | Mostly clean with minor issues | Basic structure, some issues | Poor structure, many issues |
| **Testing** | Comprehensive test cases | Good test coverage | Basic testing | Minimal or no testing |
| **Communication** | Clear explanation, good questions | Mostly clear communication | Basic communication | Poor communication |

### System Design Interview Rubric

| Criteria | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|----------|---------------|----------|------------------|----------------------|
| **Requirements Gathering** | Asks all relevant questions | Asks most important questions | Asks basic questions | Misses key requirements |
| **High-Level Design** | Clear, scalable architecture | Good architecture with minor gaps | Basic architecture | Poor or missing architecture |
| **Component Design** | Detailed, well-thought components | Good component design | Basic component design | Poor component design |
| **Scalability** | Addresses all scalability concerns | Addresses most concerns | Basic scalability thinking | Misses scalability |
| **Trade-offs** | Discusses all relevant trade-offs | Discusses most trade-offs | Basic trade-off discussion | Misses key trade-offs |

### Behavioral Interview Rubric

| Criteria | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|----------|---------------|----------|------------------|----------------------|
| **STAR Method** | Perfect STAR structure | Good STAR with minor gaps | Basic STAR structure | Poor or missing STAR |
| **Specificity** | Very specific examples | Mostly specific examples | Some specific examples | Vague or generic examples |
| **Leadership** | Strong leadership examples | Good leadership examples | Basic leadership examples | Weak leadership examples |
| **Problem Solving** | Excellent problem-solving approach | Good problem-solving | Basic problem-solving | Poor problem-solving |
| **Communication** | Clear, engaging communication | Good communication | Basic communication | Poor communication |

---

## 🎯 Interview Preparation Tips

### Before the Interview

1. **Research the Company**:
   - Understand Razorpay's business model
   - Learn about their tech stack
   - Read recent news and updates
   - Understand their products and services

2. **Practice Coding**:
   - Solve problems on LeetCode, HackerRank
   - Practice system design on Pramp, InterviewBit
   - Review common algorithms and data structures
   - Practice explaining your thought process

3. **Prepare Behavioral Examples**:
   - Prepare 5-7 STAR examples
   - Cover different scenarios (leadership, conflict, failure, success)
   - Practice telling stories concisely
   - Be ready to discuss technical challenges

### During the Interview

1. **Coding Interviews**:
   - Think out loud
   - Ask clarifying questions
   - Start with brute force, then optimize
   - Test your solution with examples
   - Discuss time and space complexity

2. **System Design Interviews**:
   - Clarify requirements first
   - Start with high-level design
   - Dive into details gradually
   - Discuss trade-offs and alternatives
   - Consider scalability and reliability

3. **Behavioral Interviews**:
   - Use the STAR method
   - Be specific and detailed
   - Show your thought process
   - Demonstrate leadership and teamwork
   - Be honest about challenges and failures

### After the Interview

1. **Follow Up**:
   - Send thank you email within 24 hours
   - Reference specific discussion points
   - Express continued interest
   - Ask about next steps

2. **Reflection**:
   - Note what went well
   - Identify areas for improvement
   - Update your preparation strategy
   - Practice weak areas

---

**🎉 Ready to ace your interviews? Use these scenarios to practice and prepare for success! 🚀**


##  Leadership Interviews

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-leadership-interviews -->

Placeholder content. Please replace with proper section.
