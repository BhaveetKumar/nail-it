---
# Auto-generated front matter
Title: Quick Reference Flashcards
LastUpdated: 2025-11-06T20:45:58.335351
Tags: []
Status: draft
---

# 🎯 Quick Reference Flashcards - Razorpay Interview

> **Essential concepts and formulas for quick review during interview preparation**

## 🚀 **Go Runtime & Concurrency**

### **Goroutine Scheduler**
```
M:N Scheduler Model
- G: Goroutine (lightweight thread)
- M: Machine (OS thread)
- P: Processor (logical CPU)

GOMAXPROCS = Number of OS threads
Default: runtime.NumCPU()
```

### **Memory Management**
```
Stack: 2KB initial, grows as needed
Heap: Managed by garbage collector
GC Target: 100% (default)
GC Tuning: debug.SetGCPercent(50)
Memory Limit: debug.SetMemoryLimit(2GB)
```

### **Concurrency Patterns**
```
Worker Pool: runtime.NumCPU() * 2 workers
Channel Buffer: 100-1000 for I/O bound
Context: For cancellation and timeouts
Select: Non-blocking channel operations
```

## 🏗️ **System Design Patterns**

### **Scalability Patterns**
```
Horizontal Scaling: Add more servers
Vertical Scaling: Increase server capacity
Load Balancing: Round robin, weighted, least connections
Database Sharding: Hash-based, range-based, directory-based
Caching: L1 (memory), L2 (Redis), L3 (database)
```

### **Caching Strategies**
```
Cache-Aside: App manages cache
Write-Through: Write to cache + DB
Write-Behind: Write to cache, async to DB
Refresh-Ahead: Proactive cache refresh
TTL: Time-to-live for cache expiration
```

### **Database Design**
```
ACID Properties:
- Atomicity: All or nothing
- Consistency: Valid state transitions
- Isolation: Concurrent transactions
- Durability: Committed data persists

Consistency Models:
- Strong: Immediate consistency
- Eventual: Eventually consistent
- Weak: No consistency guarantees
```

## 💳 **Payment Systems**

### **UPI Payment Flow**
```
1. User initiates payment
2. Validate UPI ID format
3. Check fraud risk score
4. Route to NPCI
5. Bank processing
6. Settlement initiation
7. Notification to user
```

### **Fraud Detection**
```
Risk Factors:
- Transaction amount
- User behavior patterns
- Device fingerprinting
- Location analysis
- Time-based patterns

ML Models:
- Real-time scoring
- Feature engineering
- Model serving
- A/B testing
```

### **Settlement Process**
```
1. Transaction completion
2. Debit payer account
3. Credit payee account
4. Update ledgers
5. Generate settlement report
6. Send notifications
7. Audit logging
```

## 🔧 **Performance Optimization**

### **Go Performance**
```
Object Pooling: sync.Pool for expensive objects
String Interning: Share common strings
Connection Pooling: Reuse DB connections
Batch Processing: Process multiple items together
Profiling: pprof for CPU, memory, goroutines
```

### **Database Optimization**
```
Indexing: B-tree, hash, composite indexes
Query Optimization: EXPLAIN plans
Connection Pooling: 10-100 connections
Read Replicas: Distribute read load
Partitioning: Horizontal and vertical
```

### **Caching Optimization**
```
Cache Hit Ratio: >90% target
Cache Size: 10-20% of data size
Eviction Policy: LRU, LFU, TTL
Warm-up: Preload frequently accessed data
Invalidation: Cache-aside pattern
```

## 🏢 **Microservices Architecture**

### **Service Communication**
```
Synchronous: HTTP/REST, gRPC
Asynchronous: Message queues, event streaming
Service Discovery: Consul, etcd, Kubernetes
Load Balancing: Client-side, server-side
Circuit Breaker: Prevent cascade failures
```

### **Event-Driven Architecture**
```
Event Sourcing: Store events, not state
CQRS: Separate read/write models
Saga Pattern: Distributed transactions
Event Store: Append-only event log
Projections: Read model from events
```

### **API Design**
```
REST Principles:
- Stateless
- Resource-based URLs
- HTTP methods (GET, POST, PUT, DELETE)
- Status codes
- JSON responses

gRPC Benefits:
- Binary protocol
- HTTP/2 support
- Code generation
- Streaming support
```

## 🔒 **Security & Compliance**

### **Payment Security**
```
PCI DSS Compliance:
- Data encryption
- Secure networks
- Access control
- Regular monitoring
- Vulnerability management

Encryption:
- AES-256 for data at rest
- TLS 1.3 for data in transit
- Tokenization for sensitive data
- Key management
```

### **Authentication & Authorization**
```
JWT Tokens: Stateless authentication
OAuth 2.0: Authorization framework
RBAC: Role-based access control
API Keys: Service-to-service auth
Rate Limiting: Prevent abuse
```

## 📊 **Monitoring & Observability**

### **Key Metrics**
```
Availability: 99.9% target
Latency: P50, P95, P99 percentiles
Throughput: Requests per second
Error Rate: <0.1% target
CPU Usage: <70% average
Memory Usage: <80% average
```

### **Monitoring Tools**
```
Metrics: Prometheus, Grafana
Logging: ELK Stack, Fluentd
Tracing: Jaeger, Zipkin
APM: New Relic, DataDog
Alerting: PagerDuty, Slack
```

## 🎭 **Behavioral Interview**

### **STAR Method**
```
Situation: Set the context
Task: Describe your responsibility
Action: Explain what you did
Result: Share the outcome

Key Points:
- Be specific and quantified
- Show leadership and impact
- Demonstrate problem-solving
- Learn from failures
```

### **Common Questions**
```
Leadership:
- "Tell me about a time you led a project"
- "How do you handle team conflicts?"
- "Describe a difficult technical decision"

Problem Solving:
- "Tell me about a complex problem you solved"
- "How do you handle ambiguity?"
- "Describe a time you failed and learned"

Impact:
- "What's your biggest technical achievement?"
- "How do you measure success?"
- "Tell me about a time you went above and beyond"
```

## 🏢 **Razorpay-Specific**

### **Company Values**
```
Customer First: Always prioritize customers
Ownership: Take end-to-end responsibility
Innovation: Think differently, solve creatively
Excellence: High standards, continuous improvement
Integrity: Honest, transparent, ethical
```

### **Products & Services**
```
Payment Gateway: Online payment processing
RazorpayX: Business banking platform
Razorpay Capital: Lending solutions
Razorpay Payroll: HR and payroll management
Razorpay Tax: Tax compliance solutions
```

### **Technical Stack**
```
Backend: Go, Node.js, Python
Databases: PostgreSQL, Redis, MongoDB
Infrastructure: AWS, Kubernetes, Docker
Monitoring: Prometheus, Grafana, Jaeger
CI/CD: GitHub Actions, Jenkins
```

## 🎯 **Interview Tips**

### **System Design Approach**
```
1. Requirements (5 min)
   - Functional requirements
   - Non-functional requirements
   - Scale and constraints

2. High-Level Design (10 min)
   - Draw system diagram
   - Identify major components
   - Show data flow

3. Detailed Design (15 min)
   - Component interactions
   - Database design
   - API design

4. Scale & Optimize (10 min)
   - Identify bottlenecks
   - Scaling strategies
   - Trade-offs discussion
```

### **Coding Best Practices**
```
1. Understand the problem
2. Ask clarifying questions
3. Start with brute force
4. Optimize step by step
5. Test with examples
6. Handle edge cases
7. Analyze complexity
```

### **Communication Tips**
```
- Think out loud
- Ask questions
- Explain your approach
- Handle feedback gracefully
- Stay calm and focused
- Be specific with examples
```

## 📚 **Quick Formulas**

### **Capacity Planning**
```
QPS = (Peak Users × Actions per User) / Peak Hours
Storage = (Data per User × Total Users) × Growth Factor
Bandwidth = (Data per Request × QPS) × Safety Factor
```

### **Database Sizing**
```
Read Replicas = Read QPS / Single DB Capacity
Shards = Total Data / Single Shard Capacity
Cache Size = Hot Data × Replication Factor
```

### **Caching Math**
```
Cache Hit Ratio = Cache Hits / Total Requests
Cache Miss Ratio = 1 - Cache Hit Ratio
Effective Latency = (Hit Ratio × Cache Latency) + (Miss Ratio × DB Latency)
```

---

**🎉 Use these flashcards for quick review before interviews! 🚀**

**Print this out or keep it handy for last-minute preparation!**
