# Razorpay Mock Interview Scenarios

## Table of Contents
- [Introduction](#introduction)
- [Technical Deep Dive Scenarios](#technical-deep-dive-scenarios)
- [System Design Scenarios](#system-design-scenarios)
- [Coding Challenges](#coding-challenges)
- [Behavioral Scenarios](#behavioral-scenarios)
- [Leadership Scenarios](#leadership-scenarios)
- [Product and Business Scenarios](#product-and-business-scenarios)
- [Architecture and Scalability Scenarios](#architecture-and-scalability-scenarios)

## Introduction

This guide provides comprehensive mock interview scenarios specifically tailored for Razorpay interviews. These scenarios cover technical depth, system design, coding challenges, and behavioral questions that are commonly asked at Razorpay.

## Technical Deep Dive Scenarios

### Scenario 1: Payment Processing Deep Dive

**Interviewer**: "Walk me through how a payment transaction flows through Razorpay's system from initiation to completion."

**Expected Discussion Points**:
- Payment gateway integration
- PCI DSS compliance
- Fraud detection and prevention
- Transaction state management
- Error handling and retry mechanisms
- Webhook processing
- Reconciliation and settlement

**Follow-up Questions**:
1. "How would you handle a payment that gets stuck in a pending state?"
2. "What happens if the webhook delivery fails?"
3. "How do you ensure data consistency across multiple services?"
4. "Explain the difference between authorization and capture."

**Sample Answer Framework**:
```
1. Payment Initiation
   - Customer initiates payment
   - Frontend calls Razorpay API
   - Payment request validation

2. Payment Processing
   - Route to appropriate payment method
   - Bank/UPI/Card processing
   - Real-time fraud checks

3. Transaction State Management
   - Database updates
   - Event publishing
   - State machine transitions

4. Webhook Delivery
   - Reliable webhook delivery
   - Retry mechanisms
   - Idempotency handling

5. Reconciliation
   - Bank reconciliation
   - Settlement processing
   - Dispute handling
```

### Scenario 2: Database Design for Financial Data

**Interviewer**: "Design a database schema for storing payment transactions that can handle 1M+ transactions per day."

**Expected Discussion Points**:
- ACID properties for financial data
- Database sharding strategies
- Indexing for performance
- Data retention policies
- Audit trails
- Compliance requirements

**Follow-up Questions**:
1. "How would you handle schema migrations for financial data?"
2. "What's your strategy for handling peak loads during festivals?"
3. "How do you ensure data integrity across shards?"
4. "Explain your approach to data archival."

**Sample Schema Design**:
```sql
-- Core transaction table
CREATE TABLE transactions (
    id UUID PRIMARY KEY,
    merchant_id UUID NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    status VARCHAR(20) NOT NULL,
    payment_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_merchant_created (merchant_id, created_at),
    INDEX idx_status_created (status, created_at)
);

-- Payment method details
CREATE TABLE payment_methods (
    id UUID PRIMARY KEY,
    transaction_id UUID REFERENCES transactions(id),
    method_type VARCHAR(50) NOT NULL,
    method_details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit trail
CREATE TABLE transaction_audit (
    id UUID PRIMARY KEY,
    transaction_id UUID REFERENCES transactions(id),
    old_status VARCHAR(20),
    new_status VARCHAR(20),
    changed_by UUID,
    changed_at TIMESTAMP DEFAULT NOW(),
    reason TEXT
);
```

### Scenario 3: Microservices Architecture

**Interviewer**: "Explain how you would design a microservices architecture for Razorpay's payment processing system."

**Expected Discussion Points**:
- Service decomposition strategy
- Inter-service communication
- Data consistency patterns
- Service discovery and load balancing
- Monitoring and observability
- Deployment and scaling strategies

**Follow-up Questions**:
1. "How do you handle distributed transactions?"
2. "What's your strategy for service versioning?"
3. "How do you ensure service reliability?"
4. "Explain your approach to service mesh."

**Sample Architecture**:
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Payment Service │    │  Merchant Service│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Auth Service   │    │  Notification   │    │  Analytics      │
│                 │    │  Service        │    │  Service        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Database       │    │  Message Queue  │    │  Cache Layer    │
│  Layer          │    │  (Kafka)        │    │  (Redis)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## System Design Scenarios

### Scenario 4: Design a Real-time Payment Dashboard

**Interviewer**: "Design a real-time dashboard that shows payment metrics for merchants with 100K+ transactions per day."

**Requirements**:
- Real-time data updates
- Multiple metric types (success rate, volume, revenue)
- Historical data analysis
- Custom date ranges
- Export functionality
- Mobile responsive

**Expected Discussion Points**:
- Data pipeline architecture
- Real-time streaming (Kafka, Apache Flink)
- Time-series database (InfluxDB, TimescaleDB)
- Caching strategy
- WebSocket implementation
- Data aggregation strategies

**Follow-up Questions**:
1. "How would you handle data consistency in real-time?"
2. "What's your strategy for handling data spikes?"
3. "How do you ensure dashboard performance with large datasets?"
4. "Explain your approach to data visualization."

**Sample Architecture**:
```
Payment Events → Kafka → Stream Processing → Time Series DB → API → Dashboard
                     ↓
                Real-time Analytics
                     ↓
                WebSocket Updates
```

### Scenario 5: Design a Fraud Detection System

**Interviewer**: "Design a fraud detection system that can identify suspicious transactions in real-time."

**Requirements**:
- Real-time processing (< 100ms)
- 99.9% accuracy
- Low false positive rate
- Machine learning integration
- Rule-based and ML-based detection
- Historical analysis

**Expected Discussion Points**:
- Feature engineering
- Model training and deployment
- Real-time scoring
- Rule engine design
- Data pipeline for ML
- Model monitoring and retraining

**Follow-up Questions**:
1. "How do you handle model drift?"
2. "What's your strategy for handling new fraud patterns?"
3. "How do you balance accuracy vs. performance?"
4. "Explain your approach to feature store."

**Sample ML Pipeline**:
```
Transaction Data → Feature Engineering → Model Scoring → Decision Engine → Action
                      ↓
                Model Training Pipeline
                      ↓
                Feature Store (Redis/DB)
```

### Scenario 6: Design a Webhook Delivery System

**Interviewer**: "Design a reliable webhook delivery system that ensures merchants receive payment notifications."

**Requirements**:
- At-least-once delivery
- Retry mechanism with exponential backoff
- Dead letter queue
- Webhook signature verification
- Rate limiting
- Monitoring and alerting

**Expected Discussion Points**:
- Message queue design
- Retry strategies
- Idempotency handling
- Webhook security
- Monitoring and metrics
- Error handling

**Follow-up Questions**:
1. "How do you handle webhook endpoint failures?"
2. "What's your strategy for webhook ordering?"
3. "How do you ensure webhook security?"
4. "Explain your approach to webhook testing."

**Sample Implementation**:
```go
type WebhookDelivery struct {
    queue        MessageQueue
    retryPolicy  RetryPolicy
    httpClient   HTTPClient
    signature    SignatureVerifier
}

func (wd *WebhookDelivery) Deliver(webhook WebhookEvent) error {
    // Add to queue with retry policy
    return wd.queue.Publish(webhook)
}

func (wd *WebhookDelivery) Process() {
    for event := range wd.queue.Consume() {
        if err := wd.sendWebhook(event); err != nil {
            wd.handleFailure(event, err)
        }
    }
}
```

## Coding Challenges

### Challenge 1: Implement a Payment State Machine

**Problem**: Implement a payment state machine that handles state transitions for payment processing.

**Requirements**:
- States: PENDING, AUTHORIZED, CAPTURED, FAILED, REFUNDED
- Transitions: PENDING → AUTHORIZED → CAPTURED
- PENDING → FAILED
- CAPTURED → REFUNDED
- Thread-safe operations
- State persistence
- Event logging

**Sample Implementation**:
```go
type PaymentState int

const (
    PENDING PaymentState = iota
    AUTHORIZED
    CAPTURED
    FAILED
    REFUNDED
)

type PaymentStateMachine struct {
    currentState PaymentState
    transitions  map[PaymentState][]PaymentState
    mu           sync.RWMutex
    logger       Logger
}

func NewPaymentStateMachine() *PaymentStateMachine {
    return &PaymentStateMachine{
        currentState: PENDING,
        transitions: map[PaymentState][]PaymentState{
            PENDING:    {AUTHORIZED, FAILED},
            AUTHORIZED: {CAPTURED, FAILED},
            CAPTURED:   {REFUNDED},
        },
    }
}

func (psm *PaymentStateMachine) Transition(newState PaymentState) error {
    psm.mu.Lock()
    defer psm.mu.Unlock()
    
    if !psm.isValidTransition(newState) {
        return fmt.Errorf("invalid transition from %v to %v", psm.currentState, newState)
    }
    
    oldState := psm.currentState
    psm.currentState = newState
    
    psm.logger.LogStateTransition(oldState, newState)
    
    return nil
}

func (psm *PaymentStateMachine) isValidTransition(newState PaymentState) bool {
    validTransitions, exists := psm.transitions[psm.currentState]
    if !exists {
        return false
    }
    
    for _, validState := range validTransitions {
        if validState == newState {
            return true
        }
    }
    
    return false
}
```

### Challenge 2: Implement a Rate Limiter

**Problem**: Implement a rate limiter that can handle 1M+ requests per second with different rate limits per merchant.

**Requirements**:
- Token bucket algorithm
- Per-merchant rate limiting
- Redis backend for distributed systems
- Configurable rate limits
- Burst handling
- Metrics and monitoring

**Sample Implementation**:
```go
type RateLimiter struct {
    redis    redis.Client
    limits   map[string]RateLimit
    mu       sync.RWMutex
}

type RateLimit struct {
    RequestsPerSecond int
    BurstSize         int
}

func (rl *RateLimiter) Allow(merchantID string) (bool, error) {
    limit, exists := rl.getLimit(merchantID)
    if !exists {
        return true, nil // No limit set
    }
    
    key := fmt.Sprintf("rate_limit:%s", merchantID)
    
    // Use Redis Lua script for atomic operations
    script := `
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local burst = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local current = redis.call('GET', key)
        if current == false then
            current = burst
        else
            current = tonumber(current)
        end
        
        if current > 0 then
            redis.call('DECR', key)
            redis.call('EXPIRE', key, 1)
            return 1
        else
            return 0
        end
    `
    
    result, err := rl.redis.Eval(script, []string{key}, 
        limit.RequestsPerSecond, limit.BurstSize, time.Now().Unix()).Result()
    
    if err != nil {
        return false, err
    }
    
    return result.(int64) == 1, nil
}
```

### Challenge 3: Implement a Distributed Lock

**Problem**: Implement a distributed lock for coordinating operations across multiple services.

**Requirements**:
- Redis-based implementation
- Automatic expiration
- Lock renewal
- Deadlock prevention
- High availability
- Metrics and monitoring

**Sample Implementation**:
```go
type DistributedLock struct {
    redis    redis.Client
    key      string
    value    string
    ttl      time.Duration
    renewCh  chan struct{}
    stopCh   chan struct{}
    mu       sync.RWMutex
}

func (dl *DistributedLock) Lock(ctx context.Context) error {
    dl.mu.Lock()
    defer dl.mu.Unlock()
    
    // Try to acquire lock
    success, err := dl.redis.SetNX(ctx, dl.key, dl.value, dl.ttl).Result()
    if err != nil {
        return err
    }
    
    if !success {
        return fmt.Errorf("failed to acquire lock")
    }
    
    // Start renewal process
    go dl.startRenewal()
    
    return nil
}

func (dl *DistributedLock) Unlock(ctx context.Context) error {
    dl.mu.Lock()
    defer dl.mu.Unlock()
    
    // Stop renewal process
    close(dl.stopCh)
    
    // Use Lua script for atomic unlock
    script := `
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
    `
    
    result, err := dl.redis.Eval(ctx, script, []string{dl.key}, dl.value).Result()
    if err != nil {
        return err
    }
    
    if result.(int64) == 0 {
        return fmt.Errorf("lock not owned by this client")
    }
    
    return nil
}
```

## Behavioral Scenarios

### Scenario 7: Handling Production Incidents

**Interviewer**: "Describe a time when you had to handle a critical production incident. How did you approach it?"

**Expected Discussion Points**:
- Incident response process
- Communication with stakeholders
- Root cause analysis
- Post-incident review
- Prevention measures
- Leadership during crisis

**Follow-up Questions**:
1. "How do you prioritize incidents?"
2. "What's your approach to incident communication?"
3. "How do you prevent similar incidents?"
4. "Explain your post-incident review process."

**Sample Answer Framework**:
```
1. Immediate Response
   - Assess severity and impact
   - Form incident response team
   - Communicate with stakeholders

2. Investigation
   - Gather logs and metrics
   - Identify root cause
   - Implement temporary fix

3. Resolution
   - Deploy permanent fix
   - Verify system stability
   - Monitor for side effects

4. Post-Incident
   - Conduct post-mortem
   - Document lessons learned
   - Implement prevention measures
```

### Scenario 8: Leading Technical Decisions

**Interviewer**: "Tell me about a time when you had to make a difficult technical decision that affected the entire team."

**Expected Discussion Points**:
- Decision-making process
- Stakeholder involvement
- Risk assessment
- Communication strategy
- Implementation approach
- Results and learnings

**Follow-up Questions**:
1. "How do you handle conflicting opinions?"
2. "What's your approach to technical debt?"
3. "How do you ensure team buy-in?"
4. "Explain your decision-making framework."

### Scenario 9: Mentoring and Team Development

**Interviewer**: "Describe how you've helped develop junior engineers on your team."

**Expected Discussion Points**:
- Mentoring approach
- Skill development strategies
- Knowledge sharing
- Career guidance
- Team building
- Performance management

**Follow-up Questions**:
1. "How do you identify development needs?"
2. "What's your approach to knowledge transfer?"
3. "How do you measure mentoring success?"
4. "Explain your team development strategy."

## Leadership Scenarios

### Scenario 10: Technical Leadership

**Interviewer**: "How would you lead a team to migrate a monolithic payment system to microservices?"

**Expected Discussion Points**:
- Migration strategy
- Team organization
- Risk management
- Communication plan
- Success metrics
- Rollback strategy

**Follow-up Questions**:
1. "How do you handle resistance to change?"
2. "What's your approach to technical debt?"
3. "How do you ensure system reliability during migration?"
4. "Explain your change management process."

### Scenario 11: Cross-functional Collaboration

**Interviewer**: "Describe how you would work with product, design, and business teams to deliver a new payment feature."

**Expected Discussion Points**:
- Collaboration framework
- Communication strategies
- Requirement gathering
- Technical feasibility
- Timeline management
- Stakeholder alignment

**Follow-up Questions**:
1. "How do you handle conflicting priorities?"
2. "What's your approach to requirement changes?"
3. "How do you ensure technical quality?"
4. "Explain your stakeholder management strategy."

## Product and Business Scenarios

### Scenario 12: Product Engineering

**Interviewer**: "How would you design a new payment method integration (e.g., Buy Now Pay Later)?"

**Expected Discussion Points**:
- Product requirements analysis
- Technical architecture
- Integration challenges
- Compliance requirements
- Testing strategy
- Launch planning

**Follow-up Questions**:
1. "How do you ensure regulatory compliance?"
2. "What's your approach to testing new payment methods?"
3. "How do you handle integration failures?"
4. "Explain your risk assessment process."

### Scenario 13: Business Impact

**Interviewer**: "How would you measure the success of a new payment feature?"

**Expected Discussion Points**:
- Key performance indicators
- Business metrics
- Technical metrics
- User experience metrics
- A/B testing strategy
- Data analysis

**Follow-up Questions**:
1. "How do you define success metrics?"
2. "What's your approach to A/B testing?"
3. "How do you handle metric interpretation?"
4. "Explain your data-driven decision process."

## Architecture and Scalability Scenarios

### Scenario 14: System Scalability

**Interviewer**: "How would you scale Razorpay's payment system to handle 10x current traffic?"

**Expected Discussion Points**:
- Current system analysis
- Bottleneck identification
- Scaling strategies
- Infrastructure planning
- Performance optimization
- Cost considerations

**Follow-up Questions**:
1. "How do you identify system bottlenecks?"
2. "What's your approach to horizontal scaling?"
3. "How do you ensure system reliability during scaling?"
4. "Explain your capacity planning process."

### Scenario 15: Technology Migration

**Interviewer**: "How would you migrate from a relational database to a NoSQL solution for payment data?"

**Expected Discussion Points**:
- Migration strategy
- Data modeling
- Performance considerations
- Consistency requirements
- Rollback planning
- Team training

**Follow-up Questions**:
1. "How do you ensure data consistency?"
2. "What's your approach to schema migration?"
3. "How do you handle performance differences?"
4. "Explain your data migration process."

## Conclusion

These mock interview scenarios provide comprehensive coverage of the types of questions and challenges you might face in a Razorpay interview. The key to success is:

1. **Technical Depth**: Demonstrate deep understanding of payment systems, distributed systems, and backend engineering
2. **System Thinking**: Show ability to design scalable, reliable systems
3. **Problem Solving**: Approach problems methodically and consider trade-offs
4. **Communication**: Explain complex concepts clearly and concisely
5. **Leadership**: Show ability to lead teams and make difficult decisions
6. **Business Acumen**: Understand the business impact of technical decisions

Practice these scenarios regularly and be prepared to dive deep into any aspect of your answers. The interviewers will likely ask follow-up questions to test your understanding and experience.

## Additional Resources

- [Razorpay Engineering Blog](https://razorpay.com/blog/)
- [Payment System Design](https://www.oreilly.com/library/view/building-secure-and/9781492054886/)
- [Distributed Systems Patterns](https://microservices.io/patterns/)
- [System Design Interview](https://www.educative.io/courses/grokking-the-system-design-interview)
- [Backend Engineering Best Practices](https://github.com/donnemartin/system-design-primer)
