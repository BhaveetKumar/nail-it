# ⚡ Razorpay Interview Quick Reference

> **Essential concepts and patterns for Round 2 and Round 3 interviews**

## 🏗️ **System Design Quick Reference**

### **Scalability Patterns**

```
Horizontal Scaling: Add more servers
Vertical Scaling: Increase server capacity
Load Balancing: Round Robin, Weighted, Least Connections
Database Sharding: Hash-based, Range-based, Directory-based
Caching: L1 (Memory), L2 (Redis), L3 (Database)
CDN: Reduce latency, global distribution
```

### **Database Design Patterns**

```
SQL: ACID compliance, complex queries, strong consistency
NoSQL: High throughput, flexible schema, eventual consistency
Sharding: Distribute data across multiple databases
Replication: Master-slave, master-master, read replicas
Consistency: Strong, eventual, weak consistency models
```

### **Caching Strategies**

```
Cache-Aside: Application manages cache population
Write-Through: Write to cache and database simultaneously
Write-Behind: Write to cache, async to database
Refresh-Ahead: Proactive cache refresh
TTL: Time-to-live for cache expiration
```

### **Load Balancing Algorithms**

```
Round Robin: Distribute requests evenly
Weighted Round Robin: Consider server capacity
Least Connections: Route to server with fewest connections
IP Hash: Route based on client IP
Least Response Time: Route to fastest responding server
```

### **Microservices Patterns**

```
API Gateway: Single entry point, routing, authentication
Service Discovery: Eureka, Consul, etcd
Circuit Breaker: Prevent cascading failures
Bulkhead: Isolate resources
Saga Pattern: Manage distributed transactions
Event Sourcing: Store events instead of state
```

---

## 🔧 **Go Runtime Quick Reference**

### **Goroutine Management**

```
GOMAXPROCS: Number of OS threads (typically CPU cores)
Goroutine Stack: 2KB initial, grows as needed (2x factor)
Work Stealing: Efficient work distribution across processors
Context: Cancellation, timeouts, request-scoped values
Channel: Communication between goroutines
```

### **Memory Management**

```
Stack Allocation: Fast, automatic cleanup, limited size
Heap Allocation: Managed by GC, larger capacity
Escape Analysis: Compiler determines allocation location
GC Algorithm: Concurrent, tri-color mark and sweep
GC Tuning: GOGC (default 100), GOMEMLIMIT
```

### **Concurrency Patterns**

```
Worker Pool: Limit goroutine count
Pipeline: Chain of processing stages
Fan-out/Fan-in: Distribute work, collect results
Rate Limiting: Control request rate
Circuit Breaker: Prevent cascading failures
```

### **Performance Optimization**

```
Object Pooling: Reuse expensive objects
String Interning: Share common strings
Connection Pooling: Reuse database connections
Profiling: pprof for CPU, memory, goroutines
Benchmarking: Measure performance improvements
```

---

## 🎯 **Razorpay-Specific Concepts**

### **Payment Gateway Architecture**

```
Components: API Gateway, Payment Service, Fraud Detection, Settlement
Data Flow: Request → Validation → Processing → Settlement → Notification
Scalability: Horizontal scaling, database sharding, caching
Security: PCI DSS compliance, encryption, authentication
Monitoring: Real-time metrics, alerting, logging
```

### **UPI Payment Processing**

```
NPCI Compliance: National Payments Corporation of India
Bank Connectors: Integration with multiple banks
Settlement Engine: Real-time settlement processing
Reconciliation: Match transactions across systems
Audit Trail: Complete transaction history
```

### **Fraud Detection System**

```
Real-time Processing: Sub-10ms latency requirements
ML Models: Lightweight models for fast prediction
Feature Engineering: Real-time and historical features
Rules Engine: Business logic and domain rules
Hybrid Approach: ML + rules for better accuracy
```

### **Risk Management**

```
Risk Scoring: Real-time risk assessment
Feature Store: Centralized feature management
Model Serving: High-performance model inference
A/B Testing: Model comparison and validation
Monitoring: Model performance and drift detection
```

---

## 🚀 **Performance Engineering Quick Reference**

### **Latency Optimization**

```
Caching: Multi-level caching strategy
Connection Pooling: Reuse database connections
Async Processing: Non-blocking operations
Batch Operations: Group multiple operations
Parallel Processing: Concurrent execution
```

### **Memory Optimization**

```
Object Pooling: Reuse objects to reduce GC pressure
String Interning: Share common strings
Slice Reuse: Reuse slices instead of creating new ones
Reduce Pointers: Fewer pointers mean less GC work
Use Value Types: Value types allocated on stack
```

### **Database Optimization**

```
Indexing: Proper indexes for query optimization
Query Optimization: Efficient SQL queries
Connection Pooling: Reuse database connections
Read Replicas: Distribute read load
Sharding: Distribute data across multiple databases
```

### **Network Optimization**

```
Connection Reuse: HTTP keep-alive, connection pooling
Compression: Gzip, Brotli compression
CDN: Content delivery network for static assets
Load Balancing: Distribute traffic across servers
Caching: Reduce network requests
```

---

## 🎯 **Leadership & Architecture Quick Reference**

### **System Migration Strategies**

```
Strangler Fig: Gradually replace legacy system
Database Migration: Schema changes, data migration
Technology Migration: Language, framework changes
Risk Mitigation: Rollback plans, monitoring
Phased Approach: Incremental migration
```

### **Incident Response Framework**

```
Detection: Monitor system health and alerts
Assessment: Classify severity and impact
Communication: Notify stakeholders and users
Resolution: Fix the issue and restore service
Post-mortem: Analyze root cause and improvements
```

### **Team Management**

```
Daily Standups: Regular team communication
Code Reviews: Quality assurance and knowledge sharing
Mentoring: Develop junior team members
Cross-functional Collaboration: Work with other teams
Risk Management: Identify and mitigate project risks
```

### **Architecture Decision Records (ADRs)**

```
Context: Why this decision is needed
Decision: What was decided
Consequences: Positive and negative outcomes
Alternatives: Other options considered
Trade-offs: Pros and cons of the decision
```

---

## 📊 **Key Metrics & SLAs**

### **Performance Metrics**

```
Throughput: Requests per second (RPS)
Latency: Response time (p50, p95, p99)
Availability: Uptime percentage (99.9%, 99.99%)
Error Rate: Percentage of failed requests
CPU Usage: Processor utilization
Memory Usage: RAM consumption
```

### **Business Metrics**

```
Transaction Volume: Number of transactions
Success Rate: Percentage of successful transactions
Revenue: Financial impact
User Experience: Customer satisfaction
Compliance: Regulatory requirements
```

### **Technical Metrics**

```
Database Performance: Query time, connection pool
Cache Hit Rate: Percentage of cache hits
Network Latency: Round-trip time
GC Pause Time: Garbage collection impact
Error Rates: Application and infrastructure errors
```

---

## 🎯 **Interview Success Tips**

### **System Design Tips**

1. **Ask clarifying questions** about requirements
2. **Start with high-level design** and get feedback
3. **Drill down into details** based on interest
4. **Discuss trade-offs** and alternatives
5. **Consider scalability** and performance

### **Technical Deep Dive Tips**

1. **Be specific** about technical decisions
2. **Explain reasoning** behind choices
3. **Discuss challenges** and solutions
4. **Show leadership** through examples
5. **Demonstrate learning** from experiences

### **Behavioral Tips**

1. **Use STAR method** for examples
2. **Be specific** with numbers and outcomes
3. **Show growth** and learning
4. **Stay positive** and solution-focused
5. **Ask thoughtful questions** about the role

---

## 🚀 **Final Checklist**

### **Before Interview**

- [ ] Review key concepts and patterns
- [ ] Practice explaining your projects
- [ ] Prepare questions for interviewers
- [ ] Get good sleep and stay hydrated

### **During Interview**

- [ ] Think out loud and verbalize process
- [ ] Ask clarifying questions
- [ ] Start simple and iterate
- [ ] Discuss trade-offs and alternatives
- [ ] Be specific with examples

### **After Interview**

- [ ] Send thank you email
- [ ] Reflect on performance
- [ ] Note areas for improvement
- [ ] Follow up appropriately

---

**🎉 You're ready to ace your Razorpay interviews! Stay confident and demonstrate your passion for technology and leadership. Good luck! 🚀**
