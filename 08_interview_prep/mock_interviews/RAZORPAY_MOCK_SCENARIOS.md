# 🎯 Razorpay Mock Interview Scenarios

> **Realistic interview scenarios for comprehensive practice**

## 📋 **Mock Interview Structure**

### **Round 1: Coding Interview (45 minutes)**
- 2 coding problems (medium/hard)
- Focus on algorithms and data structures
- Go implementation preferred

### **Round 2: System Design (60 minutes)**
- Design a scalable system
- Focus on payment/fintech scenarios
- Discuss trade-offs and scaling

### **Round 3: Technical Deep Dive (45 minutes)**
- Go runtime and performance
- System architecture decisions
- Technical leadership scenarios

### **Round 4: Behavioral (30 minutes)**
- Leadership and impact examples
- Conflict resolution
- Technical challenges

---

## 🚀 **Mock Interview Scenarios**

### **Scenario 1: Senior Backend Engineer**

#### **Round 1: Coding Problems**

**Problem 1: Payment Transaction Validation (Medium)**
```
Given a list of payment transactions, implement a function to validate them:

1. Each transaction has: id, amount, user_id, timestamp, status
2. Validate that:
   - Amount is positive
   - User exists in the system
   - Transaction is not duplicate
   - Timestamp is not in the future
3. Return list of invalid transactions

Time: 20 minutes
Follow-up: Optimize for large datasets
```

**Problem 2: Rate Limiter Implementation (Hard)**
```
Implement a sliding window rate limiter:

1. Allow N requests per time window
2. Support multiple users
3. Handle concurrent requests
4. Use Go channels and goroutines

Time: 25 minutes
Follow-up: How would you make it distributed?
```

#### **Round 2: System Design**

**Design a Payment Gateway for 1M TPS**

**Requirements:**
- Handle 1M transactions per second
- Support multiple payment methods (UPI, cards, net banking)
- 99.99% availability
- <100ms response time
- Fraud detection
- Real-time settlement

**Discussion Points:**
- How would you handle peak traffic during festivals?
- What if UPI is down?
- How would you ensure data consistency?
- How would you scale the fraud detection system?

#### **Round 3: Technical Deep Dive**

**Go Runtime Questions:**
1. How does Go's garbage collector work?
2. How would you optimize memory usage in a high-throughput payment system?
3. Explain the difference between channels and mutexes
4. How would you handle a memory leak in production?

**System Architecture:**
1. How would you migrate from a monolith to microservices?
2. What's your approach to handling database migrations?
3. How would you implement circuit breakers in Go?
4. Describe your approach to monitoring and alerting

#### **Round 4: Behavioral**

**Questions:**
1. Tell me about a time you led a complex technical project
2. How do you handle conflicts in your team?
3. Describe a situation where you had to make a difficult technical decision
4. What's your biggest technical failure and what did you learn?

---

### **Scenario 2: Staff Backend Engineer**

#### **Round 1: Coding Problems**

**Problem 1: Distributed Cache Implementation (Hard)**
```
Implement a distributed cache with the following features:

1. Consistent hashing for sharding
2. Replication for fault tolerance
3. Cache eviction (LRU)
4. Health checks for nodes
5. Handle node failures gracefully

Time: 30 minutes
Follow-up: How would you handle split-brain scenarios?
```

**Problem 2: Event Sourcing System (Hard)**
```
Implement an event sourcing system for payment events:

1. Store events in append-only log
2. Replay events to rebuild state
3. Handle concurrent writes
4. Support event versioning
5. Implement snapshots for performance

Time: 30 minutes
Follow-up: How would you handle event ordering?
```

#### **Round 2: System Design**

**Design a Real-Time Fraud Detection System**

**Requirements:**
- Process 100K transactions per second
- Detect fraud in <50ms
- Support multiple fraud detection rules
- Handle rule updates without downtime
- Provide real-time risk scores
- Support A/B testing of rules

**Discussion Points:**
- How would you handle false positives?
- What if the ML model is down?
- How would you ensure rule consistency across shards?
- How would you handle rule conflicts?

#### **Round 3: Technical Deep Dive**

**Advanced Go Questions:**
1. How would you implement a lock-free data structure in Go?
2. Explain Go's memory model and its implications
3. How would you profile and optimize a Go application?
4. What are the trade-offs between different concurrency patterns?

**System Architecture:**
1. How would you design a multi-tenant system?
2. What's your approach to handling data consistency in distributed systems?
3. How would you implement eventual consistency?
4. Describe your approach to handling backpressure in event streaming

#### **Round 4: Behavioral**

**Questions:**
1. How do you approach technical debt in a fast-growing startup?
2. Describe a time you had to influence without authority
3. How do you balance innovation with stability?
4. Tell me about a time you had to learn something new quickly

---

## 🎯 **Practice Guidelines**

### **Coding Round Tips**
1. **Start with brute force** solution
2. **Optimize step by step**
3. **Test with examples**
4. **Handle edge cases**
5. **Analyze time and space complexity**

### **System Design Tips**
1. **Ask clarifying questions**
2. **Start with high-level design**
3. **Drill down into details**
4. **Discuss trade-offs**
5. **Consider scalability and performance**

### **Technical Deep Dive Tips**
1. **Be specific** about your experience
2. **Explain your reasoning**
3. **Discuss challenges** and solutions
4. **Show leadership** through examples
5. **Demonstrate learning** from failures

### **Behavioral Tips**
1. **Use STAR method**
2. **Be specific** and quantified
3. **Show impact** and results
4. **Demonstrate growth** and learning
5. **Align with company values**

---

## 🚀 **Mock Interview Schedule**

### **Week 1: Foundation**
- **Day 1-2**: Coding practice
- **Day 3-4**: System design practice
- **Day 5-6**: Technical deep dive
- **Day 7**: Full mock interview

### **Week 2: Advanced**
- **Day 8-9**: Advanced coding problems
- **Day 10-11**: Complex system design
- **Day 12-13**: Leadership scenarios
- **Day 14**: Final mock interview

---

**🎉 Practice these scenarios regularly to build confidence and improve your interview performance! 🚀**
