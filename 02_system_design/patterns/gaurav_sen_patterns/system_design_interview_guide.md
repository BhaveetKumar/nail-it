# System Design Interview Guide - Gaurav Sen Methodology

## Table of Contents
1. [Introduction](#introduction)
2. [The 4-Step Approach](#the-4-step-approach)
3. [Requirements Gathering](#requirements-gathering)
4. [High-Level Design](#high-level-design)
5. [Detailed Design](#detailed-design)
6. [Scaling the System](#scaling-the-system)
7. [Common Patterns](#common-patterns)
8. [Gaurav Sen's Design Principles](#gaurav-sens-design-principles)
9. [Interview Tips](#interview-tips)
10. [Practice Scenarios](#practice-scenarios)

## Introduction

This guide is based on Gaurav Sen's proven methodology for system design interviews. His approach focuses on practical, scalable solutions that demonstrate deep understanding of distributed systems concepts.

### Key Philosophy
- **Start Simple**: Begin with basic components, then scale
- **Think in Layers**: Application, Service, Data, Infrastructure
- **Consider Trade-offs**: Every decision has pros and cons
- **Be Practical**: Focus on real-world constraints and solutions

## The 4-Step Approach

### Step 1: Requirements Gathering (5-10 minutes)
- **Functional Requirements**: What the system should do
- **Non-Functional Requirements**: Performance, scalability, reliability
- **Constraints**: Users, data, requests per second
- **Assumptions**: Clarify ambiguous requirements

### Step 2: High-Level Design (10-15 minutes)
- **Core Components**: Main services and their responsibilities
- **Data Flow**: How data moves through the system
- **APIs**: Key endpoints and their contracts
- **Database Schema**: Basic data models

### Step 3: Detailed Design (15-20 minutes)
- **Deep Dive**: Detailed implementation of each component
- **Data Storage**: Database choices, partitioning, replication
- **Caching**: Where and how to cache data
- **Load Balancing**: How to distribute traffic

### Step 4: Scaling the System (10-15 minutes)
- **Bottlenecks**: Identify and address performance issues
- **Horizontal Scaling**: Add more servers/services
- **Vertical Scaling**: Increase server capacity
- **Optimization**: Performance improvements

## Requirements Gathering

### Functional Requirements
```go
// Example: Design a URL Shortener
type URLShortenerRequirements struct {
    // Core Features
    ShortenURL    func(originalURL string) (shortURL string, error)
    RedirectURL   func(shortURL string) (originalURL string, error)
    GetStats      func(shortURL string) (stats URLStats, error)
    
    // User Features
    CreateAccount func(email, password string) (userID string, error)
    Login         func(email, password string) (token string, error)
    GetUserURLs   func(userID string) ([]URLInfo, error)
}

type URLStats struct {
    ClickCount    int
    CreatedAt     time.Time
    LastAccessed  time.Time
    UserID        string
}
```

### Non-Functional Requirements
```go
type SystemConstraints struct {
    // Scale
    UsersPerSecond    int    // 1000 users per second
    URLsPerSecond     int    // 10,000 URL creations per second
    ReadsPerSecond    int    // 100,000 redirects per second
    
    // Storage
    URLCount          int64  // 100 million URLs
    URLSize           int    // 500 bytes per URL
    TotalStorage      int64  // 50 GB total storage
    
    // Performance
    ResponseTime      int    // 200ms for URL creation
    RedirectTime      int    // 100ms for URL redirect
    Availability      float64 // 99.9% uptime
    
    // Consistency
    ConsistencyLevel  string // Eventual consistency acceptable
}
```

### Clarifying Questions
```go
type ClarifyingQuestions struct {
    // Scale Questions
    "How many users do we expect?"                    string
    "What's the read-to-write ratio?"                 string
    "Do we need real-time analytics?"                 string
    
    // Feature Questions
    "Should URLs expire?"                             string
    "Do we need custom short URLs?"                   string
    "Should we support bulk URL creation?"            string
    
    // Technical Questions
    "What's the acceptable latency?"                  string
    "Do we need global distribution?"                 string
    "Should we support mobile apps?"                  string
}
```

## High-Level Design

### Core Components
```go
// URL Shortener High-Level Architecture
type URLShortenerSystem struct {
    // Client Layer
    WebClient    *WebClient
    MobileClient *MobileClient
    API          *APIGateway
    
    // Application Layer
    URLService   *URLService
    UserService  *UserService
    AnalyticsService *AnalyticsService
    
    // Data Layer
    URLDatabase  *URLDatabase
    UserDatabase *UserDatabase
    Cache        *RedisCache
    
    // Infrastructure
    LoadBalancer *LoadBalancer
    CDN          *CDN
    Monitoring   *Monitoring
}
```

### API Design
```go
// RESTful API Design
type URLShortenerAPI struct {
    // URL Management
    POST   /api/v1/urls          // Create short URL
    GET    /api/v1/urls/{id}     // Get URL info
    DELETE /api/v1/urls/{id}     // Delete URL
    
    // User Management
    POST   /api/v1/users         // Create user
    POST   /api/v1/auth/login    // Login
    GET    /api/v1/users/{id}/urls // Get user URLs
    
    // Analytics
    GET    /api/v1/urls/{id}/stats // Get URL statistics
    GET    /api/v1/analytics/dashboard // Get dashboard data
}

// gRPC API Design
type URLShortenerGRPC struct {
    // URL Service
    rpc CreateURL(CreateURLRequest) returns (CreateURLResponse);
    rpc GetURL(GetURLRequest) returns (GetURLResponse);
    rpc DeleteURL(DeleteURLRequest) returns (DeleteURLResponse);
    
    // User Service
    rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
    rpc Authenticate(AuthRequest) returns (AuthResponse);
    
    // Analytics Service
    rpc GetStats(StatsRequest) returns (StatsResponse);
    rpc GetDashboard(DashboardRequest) returns (DashboardResponse);
}
```

### Data Flow
```go
// URL Creation Flow
func (s *URLShortenerSystem) CreateURL(originalURL string, userID string) (string, error) {
    // 1. Validate URL
    if !isValidURL(originalURL) {
        return "", errors.New("invalid URL")
    }
    
    // 2. Generate short code
    shortCode := s.generateShortCode()
    
    // 3. Store in database
    urlRecord := URLRecord{
        ShortCode:   shortCode,
        OriginalURL: originalURL,
        UserID:      userID,
        CreatedAt:   time.Now(),
    }
    
    if err := s.urlDatabase.Create(urlRecord); err != nil {
        return "", err
    }
    
    // 4. Cache the mapping
    s.cache.Set(shortCode, originalURL, 24*time.Hour)
    
    // 5. Return short URL
    return s.buildShortURL(shortCode), nil
}

// URL Redirect Flow
func (s *URLShortenerSystem) RedirectURL(shortCode string) (string, error) {
    // 1. Check cache first
    if originalURL, found := s.cache.Get(shortCode); found {
        s.analyticsService.RecordClick(shortCode)
        return originalURL, nil
    }
    
    // 2. Query database
    urlRecord, err := s.urlDatabase.GetByShortCode(shortCode)
    if err != nil {
        return "", err
    }
    
    // 3. Cache the result
    s.cache.Set(shortCode, urlRecord.OriginalURL, 24*time.Hour)
    
    // 4. Record analytics
    s.analyticsService.RecordClick(shortCode)
    
    // 5. Return original URL
    return urlRecord.OriginalURL, nil
}
```

## Detailed Design

### Database Design
```go
// URL Table Schema
type URLTable struct {
    ID          int64     `db:"id" json:"id"`
    ShortCode   string    `db:"short_code" json:"short_code"`
    OriginalURL string    `db:"original_url" json:"original_url"`
    UserID      string    `db:"user_id" json:"user_id"`
    CreatedAt   time.Time `db:"created_at" json:"created_at"`
    ExpiresAt   *time.Time `db:"expires_at" json:"expires_at"`
    ClickCount  int64     `db:"click_count" json:"click_count"`
}

// User Table Schema
type UserTable struct {
    ID        string    `db:"id" json:"id"`
    Email     string    `db:"email" json:"email"`
    Password  string    `db:"password" json:"password"`
    CreatedAt time.Time `db:"created_at" json:"created_at"`
    IsActive  bool      `db:"is_active" json:"is_active"`
}

// Analytics Table Schema
type AnalyticsTable struct {
    ID         int64     `db:"id" json:"id"`
    ShortCode  string    `db:"short_code" json:"short_code"`
    ClickedAt  time.Time `db:"clicked_at" json:"clicked_at"`
    IPAddress  string    `db:"ip_address" json:"ip_address"`
    UserAgent  string    `db:"user_agent" json:"user_agent"`
    Referer    string    `db:"referer" json:"referer"`
}
```

### Caching Strategy
```go
// Multi-Level Caching
type CachingStrategy struct {
    // L1: Application Cache (In-Memory)
    AppCache *sync.Map
    
    // L2: Redis Cache (Distributed)
    RedisCache *redis.Client
    
    // L3: CDN Cache (Global)
    CDNCache *CDNClient
}

func (cs *CachingStrategy) GetURL(shortCode string) (string, error) {
    // L1: Check application cache
    if url, found := cs.AppCache.Load(shortCode); found {
        return url.(string), nil
    }
    
    // L2: Check Redis cache
    if url, err := cs.RedisCache.Get(shortCode).Result(); err == nil {
        // Store in L1 cache
        cs.AppCache.Store(shortCode, url)
        return url, nil
    }
    
    // L3: Check CDN cache
    if url, err := cs.CDNCache.Get(shortCode); err == nil {
        // Store in L2 and L1 caches
        cs.RedisCache.Set(shortCode, url, 24*time.Hour)
        cs.AppCache.Store(shortCode, url)
        return url, nil
    }
    
    return "", errors.New("URL not found in cache")
}
```

### Load Balancing
```go
// Load Balancer Configuration
type LoadBalancerConfig struct {
    Algorithm string // round_robin, least_connections, weighted_round_robin
    HealthCheck HealthCheckConfig
    Servers []ServerConfig
}

type HealthCheckConfig struct {
    Path     string
    Interval time.Duration
    Timeout  time.Duration
    Retries  int
}

type ServerConfig struct {
    Address string
    Weight  int
    Health  bool
}

// Load Balancer Implementation
type LoadBalancer struct {
    config     LoadBalancerConfig
    servers    []ServerConfig
    current    int
    mutex      sync.RWMutex
}

func (lb *LoadBalancer) GetNextServer() *ServerConfig {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    switch lb.config.Algorithm {
    case "round_robin":
        return lb.roundRobin()
    case "least_connections":
        return lb.leastConnections()
    case "weighted_round_robin":
        return lb.weightedRoundRobin()
    default:
        return lb.roundRobin()
    }
}
```

## Scaling the System

### Horizontal Scaling
```go
// Database Sharding Strategy
type ShardingStrategy struct {
    Shards map[int]*DatabaseShard
    HashFunction func(string) int
    ShardCount int
}

type DatabaseShard struct {
    ID     int
    DB     *sql.DB
    Range  ShardRange
}

type ShardRange struct {
    Start int64
    End   int64
}

// Consistent Hashing for Sharding
func (ss *ShardingStrategy) GetShard(shortCode string) *DatabaseShard {
    hash := ss.HashFunction(shortCode)
    shardID := hash % ss.ShardCount
    return ss.Shards[shardID]
}

// Read Replicas for Scaling Reads
type ReadReplicaStrategy struct {
    Master  *DatabaseShard
    Replicas []*DatabaseShard
    Current int
}

func (rrs *ReadReplicaStrategy) GetReadDB() *DatabaseShard {
    rrs.Current = (rrs.Current + 1) % len(rrs.Replicas)
    return rrs.Replicas[rrs.Current]
}
```

### Caching at Scale
```go
// Distributed Caching with Redis Cluster
type RedisCluster struct {
    Nodes []*RedisNode
    HashRing *ConsistentHash
}

type RedisNode struct {
    Address string
    Client  *redis.Client
    Slots   []int
}

// Cache Warming Strategy
func (rc *RedisCluster) WarmCache(shortCodes []string) {
    for _, shortCode := range shortCodes {
        // Get URL from database
        url, err := rc.getURLFromDB(shortCode)
        if err != nil {
            continue
        }
        
        // Store in cache
        node := rc.HashRing.GetNode(shortCode)
        node.Client.Set(shortCode, url, 24*time.Hour)
    }
}
```

### CDN Integration
```go
// CDN Configuration
type CDNConfig struct {
    Provider    string // CloudFlare, AWS CloudFront, etc.
    EdgeLocations []string
    CacheTTL    time.Duration
    Compression bool
}

// CDN Integration
func (s *URLShortenerSystem) ServeFromCDN(shortCode string) (string, error) {
    // Check if URL exists in CDN
    if url, err := s.cdn.Get(shortCode); err == nil {
        return url, nil
    }
    
    // Get from origin server
    url, err := s.getURLFromOrigin(shortCode)
    if err != nil {
        return "", err
    }
    
    // Cache in CDN
    s.cdn.Set(shortCode, url, s.cdnConfig.CacheTTL)
    
    return url, nil
}
```

## Common Patterns

### Circuit Breaker Pattern
```go
type CircuitBreaker struct {
    State       CircuitState
    FailureCount int
    Threshold   int
    Timeout     time.Duration
    LastFailure time.Time
    mutex       sync.RWMutex
}

type CircuitState int

const (
    StateClosed CircuitState = iota
    StateOpen
    StateHalfOpen
)

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    if cb.State == StateOpen {
        if time.Since(cb.LastFailure) > cb.Timeout {
            cb.State = StateHalfOpen
        } else {
            return errors.New("circuit breaker is open")
        }
    }
    
    err := fn()
    if err != nil {
        cb.FailureCount++
        cb.LastFailure = time.Now()
        
        if cb.FailureCount >= cb.Threshold {
            cb.State = StateOpen
        }
        return err
    }
    
    cb.FailureCount = 0
    cb.State = StateClosed
    return nil
}
```

### Rate Limiting
```go
// Token Bucket Rate Limiter
type TokenBucket struct {
    Capacity    int
    Tokens      int
    RefillRate  int
    LastRefill  time.Time
    mutex       sync.Mutex
}

func (tb *TokenBucket) Allow() bool {
    tb.mutex.Lock()
    defer tb.mutex.Unlock()
    
    // Refill tokens
    now := time.Now()
    tokensToAdd := int(now.Sub(tb.LastRefill).Seconds()) * tb.RefillRate
    tb.Tokens = min(tb.Capacity, tb.Tokens+tokensToAdd)
    tb.LastRefill = now
    
    // Check if request is allowed
    if tb.Tokens > 0 {
        tb.Tokens--
        return true
    }
    
    return false
}

// Sliding Window Rate Limiter
type SlidingWindow struct {
    WindowSize time.Duration
    MaxRequests int
    Requests   []time.Time
    mutex      sync.Mutex
}

func (sw *SlidingWindow) Allow() bool {
    sw.mutex.Lock()
    defer sw.mutex.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-sw.WindowSize)
    
    // Remove old requests
    var validRequests []time.Time
    for _, req := range sw.Requests {
        if req.After(cutoff) {
            validRequests = append(validRequests, req)
        }
    }
    sw.Requests = validRequests
    
    // Check if request is allowed
    if len(sw.Requests) < sw.MaxRequests {
        sw.Requests = append(sw.Requests, now)
        return true
    }
    
    return false
}
```

## Gaurav Sen's Design Principles

### 1. Start Simple, Scale Gradually
```go
// Phase 1: Basic Implementation
type BasicURLShortener struct {
    DB    *sql.DB
    Cache *redis.Client
}

// Phase 2: Add Caching
type CachedURLShortener struct {
    DB    *sql.DB
    Cache *redis.Client
    CDN   *CDNClient
}

// Phase 3: Add Sharding
type ShardedURLShortener struct {
    Shards []*DatabaseShard
    Cache  *redis.Client
    CDN    *CDNClient
    LB     *LoadBalancer
}
```

### 2. Think in Terms of Trade-offs
```go
// Consistency vs Availability Trade-off
type ConsistencyLevel int

const (
    StrongConsistency ConsistencyLevel = iota
    EventualConsistency
    WeakConsistency
)

// Performance vs Cost Trade-off
type PerformanceTier int

const (
    BasicTier PerformanceTier = iota
    StandardTier
    PremiumTier
)

// Latency vs Throughput Trade-off
type OptimizationStrategy struct {
    LatencyOptimized bool
    ThroughputOptimized bool
    MemoryOptimized bool
}
```

### 3. Consider Real-World Constraints
```go
// Budget Constraints
type BudgetConstraints struct {
    MaxServers     int
    MaxStorage     int64
    MaxBandwidth   int64
    MaxCostPerMonth float64
}

// Technical Constraints
type TechnicalConstraints struct {
    MaxLatency     time.Duration
    MinAvailability float64
    MaxDataLoss    float64
    Compliance     []string
}

// Operational Constraints
type OperationalConstraints struct {
    TeamSize       int
    MaintenanceWindow time.Duration
    DeploymentFrequency time.Duration
    MonitoringCapability bool
}
```

## Interview Tips

### 1. Communication
- **Think Out Loud**: Explain your thought process
- **Ask Questions**: Clarify requirements and constraints
- **Draw Diagrams**: Visualize the system architecture
- **Be Interactive**: Engage with the interviewer

### 2. Problem-Solving Approach
- **Start Broad**: High-level design first
- **Drill Down**: Detailed design second
- **Consider Edge Cases**: Error handling, failure scenarios
- **Discuss Trade-offs**: Pros and cons of each decision

### 3. Technical Depth
- **Know Your Basics**: Databases, caching, load balancing
- **Understand Scaling**: Horizontal vs vertical scaling
- **Consider Performance**: Latency, throughput, bottlenecks
- **Think About Reliability**: Fault tolerance, disaster recovery

### 4. Business Understanding
- **User Experience**: How the system serves users
- **Cost Optimization**: Efficient resource utilization
- **Time to Market**: Quick iteration and deployment
- **Maintenance**: Long-term system health

## Practice Scenarios

### 1. Design a Chat System
**Requirements**: Real-time messaging, group chats, message history
**Key Components**: WebSocket servers, message queues, databases
**Scaling Challenges**: Connection management, message delivery

### 2. Design a Video Streaming Service
**Requirements**: Video upload, transcoding, streaming, recommendations
**Key Components**: CDN, transcoding pipeline, recommendation engine
**Scaling Challenges**: Bandwidth, storage, processing power

### 3. Design a Social Media Feed
**Requirements**: User posts, timeline generation, real-time updates
**Key Components**: Feed generation service, notification service, content delivery
**Scaling Challenges**: Timeline generation, real-time updates

### 4. Design a Payment System
**Requirements**: Payment processing, fraud detection, compliance
**Key Components**: Payment gateway, fraud detection, audit logging
**Scaling Challenges**: Security, compliance, high availability

### 5. Design a Search Engine
**Requirements**: Web crawling, indexing, search, ranking
**Key Components**: Crawler, indexer, search service, ranking algorithm
**Scaling Challenges**: Data processing, search performance

## Conclusion

Gaurav Sen's methodology emphasizes practical, scalable solutions that demonstrate deep understanding of distributed systems. Key takeaways:

1. **Start Simple**: Begin with basic components, then scale
2. **Think in Layers**: Application, Service, Data, Infrastructure
3. **Consider Trade-offs**: Every decision has pros and cons
4. **Be Practical**: Focus on real-world constraints and solutions
5. **Communicate Well**: Explain your thought process clearly
6. **Ask Questions**: Clarify requirements and constraints
7. **Draw Diagrams**: Visualize the system architecture
8. **Discuss Trade-offs**: Pros and cons of each decision

This approach will help you excel in system design interviews by demonstrating both technical depth and practical problem-solving skills.
