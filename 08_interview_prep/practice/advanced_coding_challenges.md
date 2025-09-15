# Advanced Coding Challenges for Backend Engineers

## Table of Contents
- [Introduction](#introduction)
- [System Design Coding Challenges](#system-design-coding-challenges)
- [Concurrency and Parallelism](#concurrency-and-parallelism)
- [Distributed Systems Challenges](#distributed-systems-challenges)
- [Database and Storage Challenges](#database-and-storage-challenges)
- [Performance Optimization Challenges](#performance-optimization-challenges)
- [Real-time Systems Challenges](#real-time-systems-challenges)
- [Security and Cryptography Challenges](#security-and-cryptography-challenges)
- [Machine Learning Integration Challenges](#machine-learning-integration-challenges)
- [Microservices Challenges](#microservices-challenges)

## Introduction

Advanced coding challenges for backend engineers go beyond basic algorithms and data structures. These challenges test your ability to design, implement, and optimize complex systems that handle real-world scale and requirements.

### Challenge Categories

1. **System Design Implementation**: Building scalable systems from scratch
2. **Concurrency**: Handling multiple threads and processes efficiently
3. **Distributed Systems**: Managing distributed state and communication
4. **Database Design**: Optimizing data storage and retrieval
5. **Performance**: Achieving high throughput and low latency
6. **Real-time Systems**: Processing streaming data efficiently
7. **Security**: Implementing secure systems and protocols
8. **ML Integration**: Building ML-powered backend systems

## System Design Coding Challenges

### Challenge 1: Design a URL Shortener

**Requirements:**
- Shorten URLs to 6-character codes
- Handle 100M URLs per day
- 99.9% uptime
- Support custom short codes
- Analytics and click tracking

**Implementation:**

```go
// URL Shortener Service
package main

import (
    "crypto/rand"
    "encoding/base64"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type URLShortener struct {
    mu        sync.RWMutex
    urls      map[string]*URLData
    analytics map[string]*Analytics
    counter   int64
}

type URLData struct {
    OriginalURL string
    ShortCode   string
    CreatedAt   time.Time
    ExpiresAt   *time.Time
    UserID      string
}

type Analytics struct {
    Clicks     int64
    LastClick  time.Time
    UserAgents map[string]int
    Countries  map[string]int
}

func NewURLShortener() *URLShortener {
    return &URLShortener{
        urls:      make(map[string]*URLData),
        analytics: make(map[string]*Analytics),
    }
}

func (us *URLShortener) ShortenURL(originalURL, customCode, userID string) (string, error) {
    us.mu.Lock()
    defer us.mu.Unlock()
    
    var shortCode string
    var err error
    
    if customCode != "" {
        if _, exists := us.urls[customCode]; exists {
            return "", fmt.Errorf("custom code already exists")
        }
        shortCode = customCode
    } else {
        shortCode, err = us.generateShortCode()
        if err != nil {
            return "", err
        }
    }
    
    urlData := &URLData{
        OriginalURL: originalURL,
        ShortCode:   shortCode,
        CreatedAt:   time.Now(),
        UserID:      userID,
    }
    
    us.urls[shortCode] = urlData
    us.analytics[shortCode] = &Analytics{
        UserAgents: make(map[string]int),
        Countries:  make(map[string]int),
    }
    
    return shortCode, nil
}

func (us *URLShortener) generateShortCode() (string, error) {
    for i := 0; i < 10; i++ {
        bytes := make([]byte, 4)
        if _, err := rand.Read(bytes); err != nil {
            return "", err
        }
        
        code := base64.URLEncoding.EncodeToString(bytes)[:6]
        if _, exists := us.urls[code]; !exists {
            return code, nil
        }
    }
    return "", fmt.Errorf("failed to generate unique code")
}

func (us *URLShortener) ResolveURL(shortCode string) (string, error) {
    us.mu.RLock()
    defer us.mu.RUnlock()
    
    urlData, exists := us.urls[shortCode]
    if !exists {
        return "", fmt.Errorf("short code not found")
    }
    
    if urlData.ExpiresAt != nil && time.Now().After(*urlData.ExpiresAt) {
        return "", fmt.Errorf("URL has expired")
    }
    
    // Update analytics
    go us.updateAnalytics(shortCode)
    
    return urlData.OriginalURL, nil
}

func (us *URLShortener) updateAnalytics(shortCode string) {
    us.mu.Lock()
    defer us.mu.Unlock()
    
    if analytics, exists := us.analytics[shortCode]; exists {
        analytics.Clicks++
        analytics.LastClick = time.Now()
    }
}

// HTTP Handlers
func (us *URLShortener) HandleShorten(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    var request struct {
        URL        string `json:"url"`
        CustomCode string `json:"custom_code,omitempty"`
        UserID     string `json:"user_id"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    shortCode, err := us.ShortenURL(request.URL, request.CustomCode, request.UserID)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    
    response := map[string]string{
        "short_code": shortCode,
        "short_url":  fmt.Sprintf("https://short.ly/%s", shortCode),
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (us *URLShortener) HandleRedirect(w http.ResponseWriter, r *http.Request) {
    shortCode := r.URL.Path[1:] // Remove leading slash
    
    originalURL, err := us.ResolveURL(shortCode)
    if err != nil {
        http.Error(w, "Not found", http.StatusNotFound)
        return
    }
    
    http.Redirect(w, r, originalURL, http.StatusMovedPermanently)
}
```

### Challenge 2: Design a Rate Limiter

**Requirements:**
- Support multiple rate limiting algorithms
- Handle 1M requests per second
- Per-user and per-IP rate limiting
- Sliding window implementation
- Redis backend for distributed systems

```go
// Rate Limiter Implementation
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type RateLimiter interface {
    Allow(ctx context.Context, key string) (bool, error)
    Reset(ctx context.Context, key string) error
}

// Token Bucket Rate Limiter
type TokenBucketLimiter struct {
    capacity     int
    refillRate   int
    tokens       map[string]*TokenBucket
    mu           sync.RWMutex
    refillTicker *time.Ticker
}

type TokenBucket struct {
    tokens     int
    lastRefill time.Time
    capacity   int
    refillRate int
}

func NewTokenBucketLimiter(capacity, refillRate int) *TokenBucketLimiter {
    limiter := &TokenBucketLimiter{
        capacity:   capacity,
        refillRate: refillRate,
        tokens:     make(map[string]*TokenBucket),
    }
    
    // Start background refill process
    go limiter.startRefillProcess()
    
    return limiter
}

func (tbl *TokenBucketLimiter) Allow(ctx context.Context, key string) (bool, error) {
    tbl.mu.Lock()
    defer tbl.mu.Unlock()
    
    bucket, exists := tbl.tokens[key]
    if !exists {
        bucket = &TokenBucket{
            tokens:     tbl.capacity,
            lastRefill: time.Now(),
            capacity:   tbl.capacity,
            refillRate: tbl.refillRate,
        }
        tbl.tokens[key] = bucket
    }
    
    // Refill tokens
    now := time.Now()
    timePassed := now.Sub(bucket.lastRefill)
    tokensToAdd := int(timePassed.Seconds()) * bucket.refillRate
    
    if tokensToAdd > 0 {
        bucket.tokens = min(bucket.capacity, bucket.tokens+tokensToAdd)
        bucket.lastRefill = now
    }
    
    if bucket.tokens > 0 {
        bucket.tokens--
        return true, nil
    }
    
    return false, nil
}

func (tbl *TokenBucketLimiter) startRefillProcess() {
    ticker := time.NewTicker(time.Second)
    defer ticker.Stop()
    
    for range ticker.C {
        tbl.mu.Lock()
        for _, bucket := range tbl.tokens {
            now := time.Now()
            timePassed := now.Sub(bucket.lastRefill)
            tokensToAdd := int(timePassed.Seconds()) * bucket.refillRate
            
            if tokensToAdd > 0 {
                bucket.tokens = min(bucket.capacity, bucket.tokens+tokensToAdd)
                bucket.lastRefill = now
            }
        }
        tbl.mu.Unlock()
    }
}

// Sliding Window Rate Limiter
type SlidingWindowLimiter struct {
    windowSize time.Duration
    maxRequests int
    requests    map[string][]time.Time
    mu          sync.RWMutex
}

func NewSlidingWindowLimiter(windowSize time.Duration, maxRequests int) *SlidingWindowLimiter {
    return &SlidingWindowLimiter{
        windowSize:   windowSize,
        maxRequests:  maxRequests,
        requests:     make(map[string][]time.Time),
    }
}

func (swl *SlidingWindowLimiter) Allow(ctx context.Context, key string) (bool, error) {
    swl.mu.Lock()
    defer swl.mu.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-swl.windowSize)
    
    // Clean old requests
    if requests, exists := swl.requests[key]; exists {
        var validRequests []time.Time
        for _, reqTime := range requests {
            if reqTime.After(cutoff) {
                validRequests = append(validRequests, reqTime)
            }
        }
        swl.requests[key] = validRequests
    }
    
    // Check if we can add a new request
    if len(swl.requests[key]) >= swl.maxRequests {
        return false, nil
    }
    
    // Add new request
    swl.requests[key] = append(swl.requests[key], now)
    return true, nil
}

// Distributed Rate Limiter with Redis
type RedisRateLimiter struct {
    client      redis.Client
    windowSize  time.Duration
    maxRequests int
}

func NewRedisRateLimiter(client redis.Client, windowSize time.Duration, maxRequests int) *RedisRateLimiter {
    return &RedisRateLimiter{
        client:      client,
        windowSize:  windowSize,
        maxRequests: maxRequests,
    }
}

func (rrl *RedisRateLimiter) Allow(ctx context.Context, key string) (bool, error) {
    now := time.Now()
    windowStart := now.Add(-rrl.windowSize)
    
    // Use Redis pipeline for atomic operations
    pipe := rrl.client.Pipeline()
    
    // Remove old entries
    pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart.Unix()))
    
    // Count current requests
    countCmd := pipe.ZCard(ctx, key)
    
    // Add current request
    pipe.ZAdd(ctx, key, &redis.Z{
        Score:  float64(now.Unix()),
        Member: now.UnixNano(),
    })
    
    // Set expiration
    pipe.Expire(ctx, key, rrl.windowSize)
    
    _, err := pipe.Exec(ctx)
    if err != nil {
        return false, err
    }
    
    count := countCmd.Val()
    return count < int64(rrl.maxRequests), nil
}
```

## Concurrency and Parallelism

### Challenge 3: Implement a Thread-Safe Cache

**Requirements:**
- Thread-safe operations
- LRU eviction policy
- TTL support
- Metrics and monitoring
- High performance

```go
// Thread-Safe LRU Cache with TTL
package main

import (
    "container/list"
    "sync"
    "time"
)

type CacheItem struct {
    Key        string
    Value      interface{}
    ExpiresAt  time.Time
    ListElement *list.Element
}

type ThreadSafeCache struct {
    mu       sync.RWMutex
    items    map[string]*CacheItem
    list     *list.List
    capacity int
    ttl      time.Duration
    metrics  *CacheMetrics
}

type CacheMetrics struct {
    Hits       int64
    Misses     int64
    Evictions  int64
    Sets       int64
    Gets       int64
}

func NewThreadSafeCache(capacity int, ttl time.Duration) *ThreadSafeCache {
    return &ThreadSafeCache{
        items:    make(map[string]*CacheItem),
        list:     list.New(),
        capacity: capacity,
        ttl:      ttl,
        metrics:  &CacheMetrics{},
    }
}

func (c *ThreadSafeCache) Get(key string) (interface{}, bool) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    c.metrics.Gets++
    
    item, exists := c.items[key]
    if !exists {
        c.metrics.Misses++
        return nil, false
    }
    
    // Check if expired
    if time.Now().After(item.ExpiresAt) {
        c.removeItem(item)
        c.metrics.Misses++
        return nil, false
    }
    
    // Move to front (LRU)
    c.list.MoveToFront(item.ListElement)
    c.metrics.Hits++
    
    return item.Value, true
}

func (c *ThreadSafeCache) Set(key string, value interface{}) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    c.metrics.Sets++
    
    // Check if key already exists
    if item, exists := c.items[key]; exists {
        item.Value = value
        item.ExpiresAt = time.Now().Add(c.ttl)
        c.list.MoveToFront(item.ListElement)
        return
    }
    
    // Check capacity
    if len(c.items) >= c.capacity {
        c.evictLRU()
    }
    
    // Add new item
    element := c.list.PushFront(key)
    item := &CacheItem{
        Key:         key,
        Value:       value,
        ExpiresAt:   time.Now().Add(c.ttl),
        ListElement: element,
    }
    
    c.items[key] = item
}

func (c *ThreadSafeCache) evictLRU() {
    if c.list.Len() == 0 {
        return
    }
    
    // Remove least recently used item
    element := c.list.Back()
    key := element.Value.(string)
    
    if item, exists := c.items[key]; exists {
        c.removeItem(item)
        c.metrics.Evictions++
    }
}

func (c *ThreadSafeCache) removeItem(item *CacheItem) {
    delete(c.items, item.Key)
    c.list.Remove(item.ListElement)
}

func (c *ThreadSafeCache) Cleanup() {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    now := time.Now()
    var expiredKeys []string
    
    for key, item := range c.items {
        if now.After(item.ExpiresAt) {
            expiredKeys = append(expiredKeys, key)
        }
    }
    
    for _, key := range expiredKeys {
        if item, exists := c.items[key]; exists {
            c.removeItem(item)
        }
    }
}

func (c *ThreadSafeCache) GetMetrics() *CacheMetrics {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    return &CacheMetrics{
        Hits:      c.metrics.Hits,
        Misses:    c.metrics.Misses,
        Evictions: c.metrics.Evictions,
        Sets:      c.metrics.Sets,
        Gets:      c.metrics.Gets,
    }
}

// Background cleanup goroutine
func (c *ThreadSafeCache) StartCleanup() {
    go func() {
        ticker := time.NewTicker(time.Minute)
        defer ticker.Stop()
        
        for range ticker.C {
            c.Cleanup()
        }
    }()
}
```

### Challenge 4: Implement a Worker Pool

**Requirements:**
- Configurable number of workers
- Job queuing and processing
- Graceful shutdown
- Error handling and retry
- Metrics and monitoring

```go
// Worker Pool Implementation
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type Job interface {
    Execute() error
    ID() string
    RetryCount() int
    MaxRetries() int
}

type WorkerPool struct {
    workers    int
    jobQueue   chan Job
    resultChan chan JobResult
    wg         sync.WaitGroup
    ctx        context.Context
    cancel     context.CancelFunc
    metrics    *PoolMetrics
    mu         sync.RWMutex
}

type JobResult struct {
    JobID string
    Error error
    Duration time.Duration
}

type PoolMetrics struct {
    JobsProcessed int64
    JobsFailed    int64
    JobsRetried   int64
    TotalDuration time.Duration
}

func NewWorkerPool(workers int, queueSize int) *WorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &WorkerPool{
        workers:    workers,
        jobQueue:   make(chan Job, queueSize),
        resultChan: make(chan JobResult, queueSize),
        ctx:        ctx,
        cancel:     cancel,
        metrics:    &PoolMetrics{},
    }
}

func (wp *WorkerPool) Start() {
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker(i)
    }
}

func (wp *WorkerPool) worker(id int) {
    defer wp.wg.Done()
    
    for {
        select {
        case job := <-wp.jobQueue:
            wp.processJob(job)
        case <-wp.ctx.Done():
            return
        }
    }
}

func (wp *WorkerPool) processJob(job Job) {
    start := time.Now()
    
    err := job.Execute()
    duration := time.Since(start)
    
    result := JobResult{
        JobID:    job.ID(),
        Error:    err,
        Duration: duration,
    }
    
    wp.updateMetrics(err, duration)
    
    // Handle retry logic
    if err != nil && job.RetryCount() < job.MaxRetries() {
        wp.retryJob(job)
        return
    }
    
    select {
    case wp.resultChan <- result:
    case <-wp.ctx.Done():
        return
    }
}

func (wp *WorkerPool) retryJob(job Job) {
    // Increment retry count and requeue
    if retryableJob, ok := job.(RetryableJob); ok {
        retryableJob.IncrementRetry()
        wp.SubmitJob(job)
    }
}

func (wp *WorkerPool) SubmitJob(job Job) error {
    select {
    case wp.jobQueue <- job:
        return nil
    case <-wp.ctx.Done():
        return fmt.Errorf("worker pool is shutting down")
    default:
        return fmt.Errorf("job queue is full")
    }
}

func (wp *WorkerPool) GetResults() <-chan JobResult {
    return wp.resultChan
}

func (wp *WorkerPool) updateMetrics(err error, duration time.Duration) {
    wp.mu.Lock()
    defer wp.mu.Unlock()
    
    wp.metrics.JobsProcessed++
    wp.metrics.TotalDuration += duration
    
    if err != nil {
        wp.metrics.JobsFailed++
    }
}

func (wp *WorkerPool) GetMetrics() *PoolMetrics {
    wp.mu.RLock()
    defer wp.mu.RUnlock()
    
    return &PoolMetrics{
        JobsProcessed: wp.metrics.JobsProcessed,
        JobsFailed:    wp.metrics.JobsFailed,
        JobsRetried:   wp.metrics.JobsRetried,
        TotalDuration: wp.metrics.TotalDuration,
    }
}

func (wp *WorkerPool) Shutdown() {
    wp.cancel()
    close(wp.jobQueue)
    wp.wg.Wait()
    close(wp.resultChan)
}

// Example job implementation
type ExampleJob struct {
    id         string
    data       string
    retryCount int
    maxRetries int
}

func (ej *ExampleJob) Execute() error {
    // Simulate work
    time.Sleep(time.Millisecond * 100)
    
    // Simulate occasional failure
    if time.Now().UnixNano()%10 == 0 {
        return fmt.Errorf("simulated error")
    }
    
    return nil
}

func (ej *ExampleJob) ID() string {
    return ej.id
}

func (ej *ExampleJob) RetryCount() int {
    return ej.retryCount
}

func (ej *ExampleJob) MaxRetries() int {
    return ej.maxRetries
}

func (ej *ExampleJob) IncrementRetry() {
    ej.retryCount++
}
```

## Distributed Systems Challenges

### Challenge 5: Implement a Distributed Lock

**Requirements:**
- Redis-based distributed lock
- Automatic expiration
- Lock renewal
- Deadlock prevention
- High availability

```go
// Distributed Lock Implementation
package main

import (
    "context"
    "fmt"
    "time"
    "github.com/go-redis/redis/v8"
)

type DistributedLock struct {
    client    *redis.Client
    key       string
    value     string
    ttl       time.Duration
    renewCh   chan struct{}
    stopCh    chan struct{}
    isLocked  bool
    mu        sync.RWMutex
}

func NewDistributedLock(client *redis.Client, key string, ttl time.Duration) *DistributedLock {
    return &DistributedLock{
        client:  client,
        key:     key,
        value:   generateLockValue(),
        ttl:     ttl,
        renewCh: make(chan struct{}),
        stopCh:  make(chan struct{}),
    }
}

func (dl *DistributedLock) Lock(ctx context.Context) error {
    dl.mu.Lock()
    defer dl.mu.Unlock()
    
    if dl.isLocked {
        return fmt.Errorf("lock already held")
    }
    
    // Try to acquire lock
    success, err := dl.client.SetNX(ctx, dl.key, dl.value, dl.ttl).Result()
    if err != nil {
        return err
    }
    
    if !success {
        return fmt.Errorf("failed to acquire lock")
    }
    
    dl.isLocked = true
    
    // Start renewal process
    go dl.startRenewal()
    
    return nil
}

func (dl *DistributedLock) Unlock(ctx context.Context) error {
    dl.mu.Lock()
    defer dl.mu.Unlock()
    
    if !dl.isLocked {
        return fmt.Errorf("lock not held")
    }
    
    // Stop renewal process
    close(dl.stopCh)
    
    // Use Lua script to ensure atomic unlock
    script := `
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
    `
    
    result, err := dl.client.Eval(ctx, script, []string{dl.key}, dl.value).Result()
    if err != nil {
        return err
    }
    
    if result.(int64) == 0 {
        return fmt.Errorf("lock not owned by this client")
    }
    
    dl.isLocked = false
    return nil
}

func (dl *DistributedLock) startRenewal() {
    ticker := time.NewTicker(dl.ttl / 3) // Renew every 1/3 of TTL
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            if !dl.renewLock() {
                return
            }
        case <-dl.stopCh:
            return
        }
    }
}

func (dl *DistributedLock) renewLock() bool {
    dl.mu.RLock()
    if !dl.isLocked {
        dl.mu.RUnlock()
        return false
    }
    dl.mu.RUnlock()
    
    // Use Lua script to renew lock
    script := `
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
    `
    
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()
    
    result, err := dl.client.Eval(ctx, script, []string{dl.key}, dl.value, int(dl.ttl.Seconds())).Result()
    if err != nil || result.(int64) == 0 {
        dl.mu.Lock()
        dl.isLocked = false
        dl.mu.Unlock()
        return false
    }
    
    return true
}

func generateLockValue() string {
    return fmt.Sprintf("%d-%d", time.Now().UnixNano(), rand.Int63())
}

// Lock Manager for multiple locks
type LockManager struct {
    client *redis.Client
    locks  map[string]*DistributedLock
    mu     sync.RWMutex
}

func NewLockManager(client *redis.Client) *LockManager {
    return &LockManager{
        client: client,
        locks:  make(map[string]*DistributedLock),
    }
}

func (lm *LockManager) AcquireLock(key string, ttl time.Duration) (*DistributedLock, error) {
    lm.mu.Lock()
    defer lm.mu.Unlock()
    
    if lock, exists := lm.locks[key]; exists {
        return lock, fmt.Errorf("lock already exists")
    }
    
    lock := NewDistributedLock(lm.client, key, ttl)
    err := lock.Lock(context.Background())
    if err != nil {
        return nil, err
    }
    
    lm.locks[key] = lock
    return lock, nil
}

func (lm *LockManager) ReleaseLock(key string) error {
    lm.mu.Lock()
    defer lm.mu.Unlock()
    
    lock, exists := lm.locks[key]
    if !exists {
        return fmt.Errorf("lock not found")
    }
    
    err := lock.Unlock(context.Background())
    if err != nil {
        return err
    }
    
    delete(lm.locks, key)
    return nil
}
```

## Database and Storage Challenges

### Challenge 6: Implement a Database Connection Pool

**Requirements:**
- Connection pooling
- Health checks
- Load balancing
- Metrics and monitoring
- Graceful shutdown

```go
// Database Connection Pool
package main

import (
    "context"
    "database/sql"
    "fmt"
    "sync"
    "time"
)

type ConnectionPool struct {
    factory      ConnectionFactory
    connections  chan *PooledConnection
    maxConns     int
    minConns     int
    maxIdleTime  time.Duration
    maxLifetime  time.Duration
    mu           sync.RWMutex
    stats        *PoolStats
    closed       bool
    closeCh      chan struct{}
}

type ConnectionFactory interface {
    Create() (*sql.DB, error)
    Close(*sql.DB) error
    Ping(*sql.DB) error
}

type PooledConnection struct {
    conn        *sql.DB
    createdAt   time.Time
    lastUsed    time.Time
    inUse       bool
    mu          sync.RWMutex
}

type PoolStats struct {
    TotalConns    int
    IdleConns     int
    InUseConns    int
    WaitCount     int64
    WaitDuration  time.Duration
    OpenCount     int64
    CloseCount    int64
}

func NewConnectionPool(factory ConnectionFactory, maxConns, minConns int, maxIdleTime, maxLifetime time.Duration) *ConnectionPool {
    pool := &ConnectionPool{
        factory:     factory,
        connections: make(chan *PooledConnection, maxConns),
        maxConns:    maxConns,
        minConns:    minConns,
        maxIdleTime: maxIdleTime,
        maxLifetime: maxLifetime,
        stats:       &PoolStats{},
        closeCh:     make(chan struct{}),
    }
    
    // Initialize minimum connections
    for i := 0; i < minConns; i++ {
        conn, err := pool.createConnection()
        if err == nil {
            pool.connections <- conn
        }
    }
    
    // Start background maintenance
    go pool.maintenance()
    
    return pool
}

func (cp *ConnectionPool) Get(ctx context.Context) (*PooledConnection, error) {
    start := time.Now()
    
    select {
    case conn := <-cp.connections:
        cp.mu.Lock()
        cp.stats.WaitDuration += time.Since(start)
        cp.mu.Unlock()
        
        // Check if connection is still valid
        if cp.isConnectionValid(conn) {
            conn.mu.Lock()
            conn.inUse = true
            conn.lastUsed = time.Now()
            conn.mu.Unlock()
            
            cp.mu.Lock()
            cp.stats.InUseConns++
            cp.stats.IdleConns--
            cp.mu.Unlock()
            
            return conn, nil
        } else {
            // Connection is invalid, create a new one
            cp.closeConnection(conn)
            return cp.createAndGetConnection(ctx)
        }
    case <-ctx.Done():
        cp.mu.Lock()
        cp.stats.WaitCount++
        cp.mu.Unlock()
        return nil, ctx.Err()
    default:
        // No idle connections, try to create a new one
        return cp.createAndGetConnection(ctx)
    }
}

func (cp *ConnectionPool) createAndGetConnection(ctx context.Context) (*PooledConnection, error) {
    cp.mu.Lock()
    defer cp.mu.Unlock()
    
    if cp.stats.TotalConns >= cp.maxConns {
        return nil, fmt.Errorf("connection pool exhausted")
    }
    
    conn, err := cp.createConnection()
    if err != nil {
        return nil, err
    }
    
    conn.mu.Lock()
    conn.inUse = true
    conn.lastUsed = time.Now()
    conn.mu.Unlock()
    
    cp.stats.TotalConns++
    cp.stats.InUseConns++
    
    return conn, nil
}

func (cp *ConnectionPool) Put(conn *PooledConnection) {
    if conn == nil {
        return
    }
    
    conn.mu.Lock()
    conn.inUse = false
    conn.lastUsed = time.Now()
    conn.mu.Unlock()
    
    cp.mu.Lock()
    cp.stats.InUseConns--
    cp.mu.Unlock()
    
    // Check if connection should be closed
    if cp.shouldCloseConnection(conn) {
        cp.closeConnection(conn)
        return
    }
    
    // Return to pool
    select {
    case cp.connections <- conn:
        cp.mu.Lock()
        cp.stats.IdleConns++
        cp.mu.Unlock()
    default:
        // Pool is full, close connection
        cp.closeConnection(conn)
    }
}

func (cp *ConnectionPool) createConnection() (*PooledConnection, error) {
    db, err := cp.factory.Create()
    if err != nil {
        return nil, err
    }
    
    conn := &PooledConnection{
        conn:      db,
        createdAt: time.Now(),
        lastUsed:  time.Now(),
    }
    
    cp.mu.Lock()
    cp.stats.OpenCount++
    cp.mu.Unlock()
    
    return conn, nil
}

func (cp *ConnectionPool) closeConnection(conn *PooledConnection) {
    cp.factory.Close(conn.conn)
    
    cp.mu.Lock()
    cp.stats.TotalConns--
    cp.stats.CloseCount++
    cp.mu.Unlock()
}

func (cp *ConnectionPool) isConnectionValid(conn *PooledConnection) bool {
    return cp.factory.Ping(conn.conn) == nil
}

func (cp *ConnectionPool) shouldCloseConnection(conn *PooledConnection) bool {
    now := time.Now()
    
    // Check max lifetime
    if cp.maxLifetime > 0 && now.Sub(conn.createdAt) > cp.maxLifetime {
        return true
    }
    
    // Check max idle time
    if cp.maxIdleTime > 0 && now.Sub(conn.lastUsed) > cp.maxIdleTime {
        return true
    }
    
    return false
}

func (cp *ConnectionPool) maintenance() {
    ticker := time.NewTicker(time.Minute)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            cp.cleanupIdleConnections()
        case <-cp.closeCh:
            return
        }
    }
}

func (cp *ConnectionPool) cleanupIdleConnections() {
    cp.mu.Lock()
    defer cp.mu.Unlock()
    
    // Close idle connections that exceed max idle time
    for {
        select {
        case conn := <-cp.connections:
            if cp.shouldCloseConnection(conn) {
                cp.closeConnection(conn)
            } else {
                // Put it back
                select {
                case cp.connections <- conn:
                default:
                    cp.closeConnection(conn)
                }
            }
        default:
            return
        }
    }
}

func (cp *ConnectionPool) GetStats() *PoolStats {
    cp.mu.RLock()
    defer cp.mu.RUnlock()
    
    return &PoolStats{
        TotalConns:    cp.stats.TotalConns,
        IdleConns:     cp.stats.IdleConns,
        InUseConns:    cp.stats.InUseConns,
        WaitCount:     cp.stats.WaitCount,
        WaitDuration:  cp.stats.WaitDuration,
        OpenCount:     cp.stats.OpenCount,
        CloseCount:    cp.stats.CloseCount,
    }
}

func (cp *ConnectionPool) Close() {
    cp.mu.Lock()
    defer cp.mu.Unlock()
    
    if cp.closed {
        return
    }
    
    cp.closed = true
    close(cp.closeCh)
    
    // Close all connections
    for {
        select {
        case conn := <-cp.connections:
            cp.closeConnection(conn)
        default:
            return
        }
    }
}
```

## Performance Optimization Challenges

### Challenge 7: Implement a High-Performance HTTP Server

**Requirements:**
- Handle 100K+ requests per second
- Low latency (< 1ms)
- Connection pooling
- Request batching
- Metrics and monitoring

```go
// High-Performance HTTP Server
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type HighPerformanceServer struct {
    server      *http.Server
    router      *http.ServeMux
    middleware  []Middleware
    metrics     *ServerMetrics
    mu          sync.RWMutex
    requestPool sync.Pool
    responsePool sync.Pool
}

type Middleware func(http.Handler) http.Handler

type ServerMetrics struct {
    RequestsTotal    int64
    RequestsPerSec   int64
    AvgResponseTime  time.Duration
    ActiveConnections int64
    ErrorCount       int64
    StartTime        time.Time
}

func NewHighPerformanceServer(addr string) *HighPerformanceServer {
    hps := &HighPerformanceServer{
        router: http.NewServeMux(),
        metrics: &ServerMetrics{
            StartTime: time.Now(),
        },
        requestPool: sync.Pool{
            New: func() interface{} {
                return &RequestContext{}
            },
        },
        responsePool: sync.Pool{
            New: func() interface{} {
                return &ResponseContext{}
            },
        },
    }
    
    hps.server = &http.Server{
        Addr:         addr,
        Handler:      hps.buildHandler(),
        ReadTimeout:  time.Second * 30,
        WriteTimeout: time.Second * 30,
        IdleTimeout:  time.Second * 60,
    }
    
    return hps
}

func (hps *HighPerformanceServer) buildHandler() http.Handler {
    handler := hps.router
    
    // Apply middleware in reverse order
    for i := len(hps.middleware) - 1; i >= 0; i-- {
        handler = hps.middleware[i](handler)
    }
    
    return handler
}

func (hps *HighPerformanceServer) AddMiddleware(middleware Middleware) {
    hps.middleware = append(hps.middleware, middleware)
}

func (hps *HighPerformanceServer) HandleFunc(pattern string, handler func(*RequestContext) *ResponseContext) {
    hps.router.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Get context from pool
        reqCtx := hps.requestPool.Get().(*RequestContext)
        reqCtx.Reset(r)
        
        // Process request
        respCtx := handler(reqCtx)
        
        // Write response
        hps.writeResponse(w, respCtx)
        
        // Update metrics
        hps.updateMetrics(time.Since(start), respCtx.StatusCode)
        
        // Return to pool
        hps.requestPool.Put(reqCtx)
        hps.responsePool.Put(respCtx)
    })
}

func (hps *HighPerformanceServer) writeResponse(w http.ResponseWriter, respCtx *ResponseContext) {
    w.Header().Set("Content-Type", respCtx.ContentType)
    w.WriteHeader(respCtx.StatusCode)
    w.Write(respCtx.Body)
}

func (hps *HighPerformanceServer) updateMetrics(duration time.Duration, statusCode int) {
    hps.mu.Lock()
    defer hps.mu.Unlock()
    
    hps.metrics.RequestsTotal++
    hps.metrics.AvgResponseTime = (hps.metrics.AvgResponseTime + duration) / 2
    
    if statusCode >= 400 {
        hps.metrics.ErrorCount++
    }
}

func (hps *HighPerformanceServer) Start() error {
    return hps.server.ListenAndServe()
}

func (hps *HighPerformanceServer) Shutdown(ctx context.Context) error {
    return hps.server.Shutdown(ctx)
}

func (hps *HighPerformanceServer) GetMetrics() *ServerMetrics {
    hps.mu.RLock()
    defer hps.mu.RUnlock()
    
    uptime := time.Since(hps.metrics.StartTime)
    hps.metrics.RequestsPerSec = hps.metrics.RequestsTotal / int64(uptime.Seconds())
    
    return &ServerMetrics{
        RequestsTotal:    hps.metrics.RequestsTotal,
        RequestsPerSec:   hps.metrics.RequestsPerSec,
        AvgResponseTime:  hps.metrics.AvgResponseTime,
        ActiveConnections: hps.metrics.ActiveConnections,
        ErrorCount:       hps.metrics.ErrorCount,
        StartTime:        hps.metrics.StartTime,
    }
}

// Request and Response contexts
type RequestContext struct {
    Request *http.Request
    Params  map[string]string
    Data    map[string]interface{}
}

func (rc *RequestContext) Reset(r *http.Request) {
    rc.Request = r
    rc.Params = make(map[string]string)
    rc.Data = make(map[string]interface{})
}

type ResponseContext struct {
    StatusCode  int
    ContentType string
    Body        []byte
    Headers     map[string]string
}

func (rc *ResponseContext) Reset() {
    rc.StatusCode = 200
    rc.ContentType = "application/json"
    rc.Body = rc.Body[:0]
    rc.Headers = make(map[string]string)
}

func (rc *ResponseContext) JSON(data interface{}) {
    body, _ := json.Marshal(data)
    rc.Body = body
    rc.ContentType = "application/json"
}

func (rc *ResponseContext) Text(text string) {
    rc.Body = []byte(text)
    rc.ContentType = "text/plain"
}

// Example usage
func main() {
    server := NewHighPerformanceServer(":8080")
    
    // Add middleware
    server.AddMiddleware(LoggingMiddleware)
    server.AddMiddleware(CORSMiddleware)
    server.AddMiddleware(RateLimitMiddleware)
    
    // Add routes
    server.HandleFunc("/api/users", handleUsers)
    server.HandleFunc("/api/health", handleHealth)
    
    // Start server
    if err := server.Start(); err != nil {
        panic(err)
    }
}

func handleUsers(reqCtx *RequestContext) *ResponseContext {
    respCtx := &ResponseContext{}
    
    // Process request
    users := []map[string]interface{}{
        {"id": 1, "name": "John Doe"},
        {"id": 2, "name": "Jane Smith"},
    }
    
    respCtx.JSON(users)
    return respCtx
}

func handleHealth(reqCtx *RequestContext) *ResponseContext {
    respCtx := &ResponseContext{}
    respCtx.JSON(map[string]string{"status": "healthy"})
    return respCtx
}

// Middleware implementations
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        fmt.Printf("%s %s %v\n", r.Method, r.URL.Path, time.Since(start))
    })
}

func CORSMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        
        next.ServeHTTP(w, r)
    })
}

func RateLimitMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Implement rate limiting logic
        next.ServeHTTP(w, r)
    })
}
```

## Conclusion

These advanced coding challenges test your ability to design and implement complex backend systems. They cover:

- **System Design**: Building scalable services from scratch
- **Concurrency**: Handling multiple threads and processes
- **Distributed Systems**: Managing distributed state and communication
- **Database Design**: Optimizing data storage and retrieval
- **Performance**: Achieving high throughput and low latency

The key to success is understanding the underlying principles and being able to implement them efficiently. Practice these challenges regularly to develop the skills needed for senior backend engineering roles.

## Additional Resources

- [System Design Interview](https://www.educative.io/courses/grokking-the-system-design-interview)
- [High Performance Go](https://github.com/geohot/minikeyvalue)
- [Distributed Systems Patterns](https://microservices.io/patterns/)
- [Database Internals](https://www.oreilly.com/library/view/database-internals/9781492040330/)
- [Designing Data-Intensive Applications](https://dataintensive.net/)
