# ⚡ Performance Engineering Comprehensive Guide

> **Complete guide to performance optimization, monitoring, and scaling for backend systems**

## 📚 Table of Contents

1. [Performance Fundamentals](#-performance-fundamentals)
2. [Profiling & Monitoring](#-profiling--monitoring)
3. [Database Performance](#-database-performance)
4. [Caching Strategies](#-caching-strategies)
5. [Memory Optimization](#-memory-optimization)
6. [Concurrency & Parallelism](#-concurrency--parallelism)
7. [Network Optimization](#-network-optimization)
8. [Load Testing](#-load-testing)
9. [Performance Patterns](#-performance-patterns)
10. [Real-world Case Studies](#-real-world-case-studies)

---

## 🎯 Performance Fundamentals

### Key Performance Metrics

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Latency       │    │   Throughput    │    │   Resource      │
│   (Response     │    │   (Requests     │    │   Utilization   │
│    Time)        │    │    per Second)  │    │   (CPU, Memory) │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │ P50: 100ms          │ RPS: 10,000          │ CPU: 70%
          │ P95: 500ms          │ TPS: 5,000           │ Memory: 60%
          │ P99: 1000ms         │ BPS: 50MB/s          │ Disk: 40%
          │                     │                      │
          └─────────────────────┼──────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Performance         │
                    │   Optimization        │
                    │   Strategies          │
                    └───────────────────────┘
```

### Performance Optimization Principles

1. **Measure First**: Always profile before optimizing
2. **Bottleneck Identification**: Find the slowest component
3. **Incremental Improvement**: Optimize one area at a time
4. **Trade-off Analysis**: Balance performance vs complexity
5. **Continuous Monitoring**: Track performance over time

---

## 🔍 Profiling & Monitoring

### Node.js Profiling

```javascript
// CPU Profiling with clinic.js
const clinic = require('@nearform/clinic');

// Memory profiling
const v8 = require('v8');

function startMemoryProfiling() {
    const heapStats = v8.getHeapStatistics();
    console.log('Heap Statistics:', heapStats);
    
    // Take heap snapshot
    const fs = require('fs');
    const snapshot = v8.getHeapSnapshot();
    const fileName = `heap-${Date.now()}.heapsnapshot`;
    const fileStream = fs.createWriteStream(fileName);
    snapshot.pipe(fileStream);
}

// Performance monitoring middleware
const performanceMiddleware = (req, res, next) => {
    const start = process.hrtime.bigint();
    
    res.on('finish', () => {
        const end = process.hrtime.bigint();
        const duration = Number(end - start) / 1000000; // Convert to milliseconds
        
        console.log(`${req.method} ${req.url} - ${res.statusCode} - ${duration}ms`);
        
        // Log slow requests
        if (duration > 1000) {
            console.warn(`Slow request detected: ${req.method} ${req.url} - ${duration}ms`);
        }
    });
    
    next();
};

// Memory leak detection
class MemoryLeakDetector {
    constructor() {
        this.initialMemory = process.memoryUsage();
        this.interval = setInterval(() => {
            this.checkMemoryUsage();
        }, 30000); // Check every 30 seconds
    }
    
    checkMemoryUsage() {
        const currentMemory = process.memoryUsage();
        const memoryIncrease = currentMemory.heapUsed - this.initialMemory.heapUsed;
        
        if (memoryIncrease > 100 * 1024 * 1024) { // 100MB increase
            console.warn('Potential memory leak detected:', {
                initial: this.initialMemory.heapUsed,
                current: currentMemory.heapUsed,
                increase: memoryIncrease
            });
        }
    }
    
    stop() {
        clearInterval(this.interval);
    }
}

// Usage
const leakDetector = new MemoryLeakDetector();
```

### Go Profiling

```go
package main

import (
    "fmt"
    "log"
    "net/http"
    _ "net/http/pprof"
    "runtime"
    "time"
)

// CPU profiling
func startCPUProfile() {
    f, err := os.Create("cpu.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    
    if err := pprof.StartCPUProfile(f); err != nil {
        log.Fatal(err)
    }
    defer pprof.StopCPUProfile()
}

// Memory profiling
func startMemoryProfile() {
    f, err := os.Create("mem.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    
    runtime.GC() // Get up-to-date statistics
    if err := pprof.WriteHeapProfile(f); err != nil {
        log.Fatal(err)
    }
}

// Performance monitoring middleware
func PerformanceMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Wrap response writer to capture status code
        ww := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        next.ServeHTTP(ww, r)
        
        duration := time.Since(start)
        
        // Log performance metrics
        log.Printf("%s %s - %d - %v", r.Method, r.URL.Path, ww.statusCode, duration)
        
        // Alert on slow requests
        if duration > time.Second {
            log.Printf("Slow request: %s %s - %v", r.Method, r.URL.Path, duration)
        }
    })
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}

// Memory monitoring
func monitorMemory() {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()
    
    for range ticker.C {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        
        log.Printf("Memory usage: Alloc=%d KB, TotalAlloc=%d KB, Sys=%d KB, NumGC=%d",
            m.Alloc/1024, m.TotalAlloc/1024, m.Sys/1024, m.NumGC)
        
        // Alert on high memory usage
        if m.Alloc > 100*1024*1024 { // 100MB
            log.Printf("High memory usage detected: %d KB", m.Alloc/1024)
        }
    }
}

func main() {
    // Start pprof server
    go func() {
        log.Println("Starting pprof server on :6060")
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()
    
    // Start memory monitoring
    go monitorMemory()
    
    // Your application code
    mux := http.NewServeMux()
    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte("Hello World"))
    })
    
    handler := PerformanceMiddleware(mux)
    log.Fatal(http.ListenAndServe(":8080", handler))
}
```

### Rust Profiling

```rust
use std::time::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Performance monitoring
pub struct PerformanceMonitor {
    request_count: AtomicU64,
    total_duration: AtomicU64,
    slow_requests: AtomicU64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            request_count: AtomicU64::new(0),
            total_duration: AtomicU64::new(0),
            slow_requests: AtomicU64::new(0),
        }
    }
    
    pub fn record_request(&self, duration: u64) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.total_duration.fetch_add(duration, Ordering::Relaxed);
        
        if duration > 1000 { // 1 second
            self.slow_requests.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    pub fn get_stats(&self) -> (u64, u64, u64) {
        let count = self.request_count.load(Ordering::Relaxed);
        let total = self.total_duration.load(Ordering::Relaxed);
        let slow = self.slow_requests.load(Ordering::Relaxed);
        
        (count, total, slow)
    }
}

// Memory profiling
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct ProfilingAllocator;

unsafe impl GlobalAlloc for ProfilingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
}

static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: ProfilingAllocator = ProfilingAllocator;

// Performance middleware
use actix_web::{web, App, HttpServer, middleware, HttpResponse};

async fn performance_middleware(
    req: actix_web::HttpRequest,
    next: actix_web::middleware::Next,
) -> Result<actix_web::HttpResponse, actix_web::Error> {
    let start = Instant::now();
    let res = next.run(req).await?;
    let duration = start.elapsed().as_millis() as u64;
    
    // Record performance metrics
    let monitor = req.app_data::<Arc<PerformanceMonitor>>().unwrap();
    monitor.record_request(duration);
    
    Ok(res)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let monitor = Arc::new(PerformanceMonitor::new());
    
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(monitor.clone()))
            .wrap(performance_middleware)
            .route("/", web::get().to(handler))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

async fn handler() -> Result<HttpResponse, actix_web::Error> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "message": "Hello World"
    })))
}
```

---

## 🗄️ Database Performance

### Query Optimization

#### SQL Query Optimization

```sql
-- Index optimization
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_user_id ON orders(user_id);
CREATE INDEX idx_order_created_at ON orders(created_at);

-- Composite indexes
CREATE INDEX idx_user_status_created ON users(status, created_at);

-- Partial indexes
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- Query optimization examples
-- Before: Full table scan
SELECT * FROM users WHERE email LIKE '%@gmail.com';

-- After: Using index
SELECT * FROM users WHERE email LIKE 'user%@gmail.com';

-- Before: N+1 queries
SELECT * FROM orders;
-- Then for each order: SELECT * FROM order_items WHERE order_id = ?

-- After: Single query with JOIN
SELECT o.*, oi.* 
FROM orders o 
LEFT JOIN order_items oi ON o.id = oi.order_id;

-- Before: Subquery
SELECT * FROM users 
WHERE id IN (SELECT user_id FROM orders WHERE total > 1000);

-- After: JOIN
SELECT DISTINCT u.* 
FROM users u 
INNER JOIN orders o ON u.id = o.user_id 
WHERE o.total > 1000;
```

#### Database Connection Pooling

```go
// Go database connection pooling
package main

import (
    "database/sql"
    "fmt"
    "time"
    _ "github.com/lib/pq"
)

type DatabaseConfig struct {
    Host            string
    Port            int
    User            string
    Password        string
    Database        string
    MaxOpenConns    int
    MaxIdleConns    int
    ConnMaxLifetime time.Duration
    ConnMaxIdleTime time.Duration
}

func NewDatabase(config DatabaseConfig) (*sql.DB, error) {
    dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
        config.Host, config.Port, config.User, config.Password, config.Database)
    
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(config.MaxOpenConns)
    db.SetMaxIdleConns(config.MaxIdleConns)
    db.SetConnMaxLifetime(config.ConnMaxLifetime)
    db.SetConnMaxIdleTime(config.ConnMaxIdleTime)
    
    // Test connection
    if err := db.Ping(); err != nil {
        return nil, err
    }
    
    return db, nil
}

// Usage
func main() {
    config := DatabaseConfig{
        Host:            "localhost",
        Port:            5432,
        User:            "user",
        Password:        "password",
        Database:        "mydb",
        MaxOpenConns:    25,
        MaxIdleConns:    5,
        ConnMaxLifetime: 5 * time.Minute,
        ConnMaxIdleTime: 1 * time.Minute,
    }
    
    db, err := NewDatabase(config)
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()
}
```

### Database Sharding

```go
// Database sharding implementation
package main

import (
    "hash/fnv"
    "strconv"
)

type ShardConfig struct {
    Host     string
    Port     int
    Database string
    Weight   int
}

type ShardedDatabase struct {
    shards []*sql.DB
    configs []ShardConfig
}

func NewShardedDatabase(configs []ShardConfig) (*ShardedDatabase, error) {
    var shards []*sql.DB
    
    for _, config := range configs {
        db, err := NewDatabase(DatabaseConfig{
            Host:     config.Host,
            Port:     config.Port,
            Database: config.Database,
        })
        if err != nil {
            return nil, err
        }
        shards = append(shards, db)
    }
    
    return &ShardedDatabase{
        shards:  shards,
        configs: configs,
    }, nil
}

func (sd *ShardedDatabase) GetShard(key string) *sql.DB {
    hash := fnv.New32a()
    hash.Write([]byte(key))
    shardIndex := int(hash.Sum32()) % len(sd.shards)
    return sd.shards[shardIndex]
}

func (sd *ShardedDatabase) Query(key string, query string, args ...interface{}) (*sql.Rows, error) {
    shard := sd.GetShard(key)
    return shard.Query(query, args...)
}
```

---

## 🚀 Caching Strategies

### Multi-Level Caching

```go
// Multi-level cache implementation
package main

import (
    "context"
    "time"
    "github.com/go-redis/redis/v8"
)

type CacheLevel int

const (
    L1Cache CacheLevel = iota // Memory cache
    L2Cache                   // Redis cache
    L3Cache                   // Database
)

type MultiLevelCache struct {
    l1Cache map[string]interface{}
    l2Cache *redis.Client
    l3Cache *sql.DB
    ttl     time.Duration
}

func NewMultiLevelCache(redisClient *redis.Client, db *sql.DB, ttl time.Duration) *MultiLevelCache {
    return &MultiLevelCache{
        l1Cache: make(map[string]interface{}),
        l2Cache: redisClient,
        l3Cache: db,
        ttl:     ttl,
    }
}

func (mlc *MultiLevelCache) Get(ctx context.Context, key string) (interface{}, error) {
    // Try L1 cache first
    if value, exists := mlc.l1Cache[key]; exists {
        return value, nil
    }
    
    // Try L2 cache (Redis)
    value, err := mlc.l2Cache.Get(ctx, key).Result()
    if err == nil {
        // Store in L1 cache
        mlc.l1Cache[key] = value
        return value, nil
    }
    
    // Try L3 cache (Database)
    var result interface{}
    err = mlc.l3Cache.QueryRowContext(ctx, "SELECT data FROM cache WHERE key = $1", key).Scan(&result)
    if err == nil {
        // Store in L2 and L1 caches
        mlc.l2Cache.Set(ctx, key, result, mlc.ttl)
        mlc.l1Cache[key] = result
        return result, nil
    }
    
    return nil, err
}

func (mlc *MultiLevelCache) Set(ctx context.Context, key string, value interface{}) error {
    // Store in all levels
    mlc.l1Cache[key] = value
    mlc.l2Cache.Set(ctx, key, value, mlc.ttl)
    
    _, err := mlc.l3Cache.ExecContext(ctx, 
        "INSERT INTO cache (key, data) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET data = $2",
        key, value)
    
    return err
}
```

### Cache-Aside Pattern

```go
// Cache-aside pattern implementation
type CacheAsideService struct {
    cache  *MultiLevelCache
    db     *sql.DB
    logger *log.Logger
}

func (cas *CacheAsideService) GetUser(ctx context.Context, userID string) (*User, error) {
    // Try cache first
    cacheKey := fmt.Sprintf("user:%s", userID)
    cached, err := cas.cache.Get(ctx, cacheKey)
    if err == nil {
        if user, ok := cached.(*User); ok {
            cas.logger.Printf("Cache hit for user %s", userID)
            return user, nil
        }
    }
    
    // Cache miss - get from database
    cas.logger.Printf("Cache miss for user %s", userID)
    user, err := cas.getUserFromDB(ctx, userID)
    if err != nil {
        return nil, err
    }
    
    // Store in cache
    cas.cache.Set(ctx, cacheKey, user)
    
    return user, nil
}

func (cas *CacheAsideService) UpdateUser(ctx context.Context, user *User) error {
    // Update database
    err := cas.updateUserInDB(ctx, user)
    if err != nil {
        return err
    }
    
    // Invalidate cache
    cacheKey := fmt.Sprintf("user:%s", user.ID)
    cas.cache.Delete(ctx, cacheKey)
    
    return nil
}
```

---

## 💾 Memory Optimization

### Memory Pool Pattern

```go
// Memory pool for reducing GC pressure
package main

import (
    "sync"
)

type ObjectPool struct {
    pool sync.Pool
}

func NewObjectPool() *ObjectPool {
    return &ObjectPool{
        pool: sync.Pool{
            New: func() interface{} {
                return &Buffer{
                    data: make([]byte, 0, 1024),
                }
            },
        },
    }
}

type Buffer struct {
    data []byte
}

func (b *Buffer) Reset() {
    b.data = b.data[:0]
}

func (b *Buffer) Write(p []byte) (n int, err error) {
    b.data = append(b.data, p...)
    return len(p), nil
}

func (b *Buffer) Bytes() []byte {
    return b.data
}

func (op *ObjectPool) Get() *Buffer {
    return op.pool.Get().(*Buffer)
}

func (op *ObjectPool) Put(buf *Buffer) {
    buf.Reset()
    op.pool.Put(buf)
}

// Usage
func processRequest(pool *ObjectPool, data []byte) {
    buf := pool.Get()
    defer pool.Put(buf)
    
    // Use buffer
    buf.Write(data)
    result := buf.Bytes()
    
    // Process result...
    _ = result
}
```

### String Interning

```go
// String interning to reduce memory usage
package main

import (
    "sync"
)

type StringInterner struct {
    strings map[string]string
    mutex   sync.RWMutex
}

func NewStringInterner() *StringInterner {
    return &StringInterner{
        strings: make(map[string]string),
    }
}

func (si *StringInterner) Intern(s string) string {
    si.mutex.RLock()
    if interned, exists := si.strings[s]; exists {
        si.mutex.RUnlock()
        return interned
    }
    si.mutex.RUnlock()
    
    si.mutex.Lock()
    defer si.mutex.Unlock()
    
    // Double-check after acquiring write lock
    if interned, exists := si.strings[s]; exists {
        return interned
    }
    
    si.strings[s] = s
    return s
}

// Usage
func main() {
    interner := NewStringInterner()
    
    // These will return the same string reference
    s1 := interner.Intern("hello")
    s2 := interner.Intern("hello")
    
    // s1 and s2 are the same string reference
    fmt.Println(s1 == s2) // true
}
```

---

## ⚡ Concurrency & Parallelism

### Worker Pool Pattern

```go
// Worker pool for concurrent processing
package main

import (
    "context"
    "sync"
)

type Job struct {
    ID   int
    Data interface{}
}

type Result struct {
    JobID int
    Data  interface{}
    Error error
}

type WorkerPool struct {
    workerCount int
    jobQueue    chan Job
    resultQueue chan Result
    ctx         context.Context
    cancel      context.CancelFunc
}

func NewWorkerPool(workerCount int) *WorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &WorkerPool{
        workerCount: workerCount,
        jobQueue:    make(chan Job, workerCount*2),
        resultQueue: make(chan Result, workerCount*2),
        ctx:         ctx,
        cancel:      cancel,
    }
}

func (wp *WorkerPool) Start() {
    var wg sync.WaitGroup
    
    for i := 0; i < wp.workerCount; i++ {
        wg.Add(1)
        go wp.worker(&wg)
    }
    
    go func() {
        wg.Wait()
        close(wp.resultQueue)
    }()
}

func (wp *WorkerPool) worker(wg *sync.WaitGroup) {
    defer wg.Done()
    
    for {
        select {
        case job := <-wp.jobQueue:
            result := wp.processJob(job)
            wp.resultQueue <- result
        case <-wp.ctx.Done():
            return
        }
    }
}

func (wp *WorkerPool) processJob(job Job) Result {
    // Process job
    return Result{
        JobID: job.ID,
        Data:  job.Data,
        Error: nil,
    }
}

func (wp *WorkerPool) Submit(job Job) {
    select {
    case wp.jobQueue <- job:
    case <-wp.ctx.Done():
    }
}

func (wp *WorkerPool) GetResult() <-chan Result {
    return wp.resultQueue
}

func (wp *WorkerPool) Stop() {
    wp.cancel()
}
```

### Pipeline Pattern

```go
// Pipeline for data processing
package main

import (
    "context"
    "sync"
)

type Pipeline struct {
    stages []Stage
    ctx    context.Context
    cancel context.CancelFunc
}

type Stage interface {
    Process(ctx context.Context, input <-chan interface{}) <-chan interface{}
}

type FilterStage struct {
    filter func(interface{}) bool
}

func (fs *FilterStage) Process(ctx context.Context, input <-chan interface{}) <-chan interface{} {
    output := make(chan interface{})
    
    go func() {
        defer close(output)
        for {
            select {
            case data, ok := <-input:
                if !ok {
                    return
                }
                if fs.filter(data) {
                    select {
                    case output <- data:
                    case <-ctx.Done():
                        return
                    }
                }
            case <-ctx.Done():
                return
            }
        }
    }()
    
    return output
}

type TransformStage struct {
    transform func(interface{}) interface{}
}

func (ts *TransformStage) Process(ctx context.Context, input <-chan interface{}) <-chan interface{} {
    output := make(chan interface{})
    
    go func() {
        defer close(output)
        for {
            select {
            case data, ok := <-input:
                if !ok {
                    return
                }
                transformed := ts.transform(data)
                select {
                case output <- transformed:
                case <-ctx.Done():
                    return
                }
            case <-ctx.Done():
                return
            }
        }
    }()
    
    return output
}

func NewPipeline(stages ...Stage) *Pipeline {
    ctx, cancel := context.WithCancel(context.Background())
    return &Pipeline{
        stages: stages,
        ctx:    ctx,
        cancel: cancel,
    }
}

func (p *Pipeline) Process(input <-chan interface{}) <-chan interface{} {
    current := input
    
    for _, stage := range p.stages {
        current = stage.Process(p.ctx, current)
    }
    
    return current
}

func (p *Pipeline) Stop() {
    p.cancel()
}
```

---

## 🌐 Network Optimization

### Connection Pooling

```go
// HTTP connection pooling
package main

import (
    "net/http"
    "time"
)

func NewHTTPClient() *http.Client {
    transport := &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
        DisableKeepAlives:   false,
    }
    
    return &http.Client{
        Transport: transport,
        Timeout:   30 * time.Second,
    }
}

// Connection reuse
type HTTPClientPool struct {
    clients []*http.Client
    current int
    mutex   sync.Mutex
}

func NewHTTPClientPool(size int) *HTTPClientPool {
    clients := make([]*http.Client, size)
    for i := 0; i < size; i++ {
        clients[i] = NewHTTPClient()
    }
    
    return &HTTPClientPool{
        clients: clients,
    }
}

func (hcp *HTTPClientPool) Get() *http.Client {
    hcp.mutex.Lock()
    defer hcp.mutex.Unlock()
    
    client := hcp.clients[hcp.current]
    hcp.current = (hcp.current + 1) % len(hcp.clients)
    return client
}
```

### Compression

```go
// Response compression
package main

import (
    "compress/gzip"
    "net/http"
    "strings"
)

func CompressionMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
            next.ServeHTTP(w, r)
            return
        }
        
        w.Header().Set("Content-Encoding", "gzip")
        w.Header().Set("Vary", "Accept-Encoding")
        
        gz := gzip.NewWriter(w)
        defer gz.Close()
        
        gzw := &gzipResponseWriter{ResponseWriter: w, Writer: gz}
        next.ServeHTTP(gzw, r)
    })
}

type gzipResponseWriter struct {
    http.ResponseWriter
    *gzip.Writer
}

func (gzw *gzipResponseWriter) Write(data []byte) (int, error) {
    return gzw.Writer.Write(data)
}
```

---

## 🧪 Load Testing

### Load Testing with Go

```go
// Load testing implementation
package main

import (
    "context"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type LoadTest struct {
    URL         string
    Concurrency int
    Duration    time.Duration
    Rate        int // requests per second
}

type LoadTestResult struct {
    TotalRequests    int
    SuccessfulRequests int
    FailedRequests   int
    AverageLatency   time.Duration
    P95Latency       time.Duration
    P99Latency       time.Duration
    RPS              float64
}

func (lt *LoadTest) Run() *LoadTestResult {
    ctx, cancel := context.WithTimeout(context.Background(), lt.Duration)
    defer cancel()
    
    var wg sync.WaitGroup
    results := make(chan time.Duration, lt.Concurrency*1000)
    
    // Start workers
    for i := 0; i < lt.Concurrency; i++ {
        wg.Add(1)
        go lt.worker(ctx, &wg, results)
    }
    
    // Collect results
    go func() {
        wg.Wait()
        close(results)
    }()
    
    var latencies []time.Duration
    for latency := range results {
        latencies = append(latencies, latency)
    }
    
    return lt.calculateResults(latencies)
}

func (lt *LoadTest) worker(ctx context.Context, wg *sync.WaitGroup, results chan<- time.Duration) {
    defer wg.Done()
    
    ticker := time.NewTicker(time.Second / time.Duration(lt.Rate/lt.Concurrency))
    defer ticker.Stop()
    
    client := &http.Client{Timeout: 10 * time.Second}
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            start := time.Now()
            resp, err := client.Get(lt.URL)
            latency := time.Since(start)
            
            if err == nil {
                resp.Body.Close()
            }
            
            results <- latency
        }
    }
}

func (lt *LoadTest) calculateResults(latencies []time.Duration) *LoadTestResult {
    if len(latencies) == 0 {
        return &LoadTestResult{}
    }
    
    // Sort latencies
    sort.Slice(latencies, func(i, j int) bool {
        return latencies[i] < latencies[j]
    })
    
    total := len(latencies)
    var sum time.Duration
    
    for _, latency := range latencies {
        sum += latency
    }
    
    return &LoadTestResult{
        TotalRequests:    total,
        SuccessfulRequests: total, // Simplified
        FailedRequests:   0,
        AverageLatency:   sum / time.Duration(total),
        P95Latency:       latencies[int(float64(total)*0.95)],
        P99Latency:       latencies[int(float64(total)*0.99)],
        RPS:              float64(total) / lt.Duration.Seconds(),
    }
}

// Usage
func main() {
    test := &LoadTest{
        URL:         "http://localhost:8080/api/data",
        Concurrency: 10,
        Duration:    30 * time.Second,
        Rate:        100,
    }
    
    result := test.Run()
    fmt.Printf("Load Test Results:\n")
    fmt.Printf("Total Requests: %d\n", result.TotalRequests)
    fmt.Printf("Average Latency: %v\n", result.AverageLatency)
    fmt.Printf("P95 Latency: %v\n", result.P95Latency)
    fmt.Printf("P99 Latency: %v\n", result.P99Latency)
    fmt.Printf("RPS: %.2f\n", result.RPS)
}
```

---

## 🎯 Performance Patterns

### Circuit Breaker Pattern

```go
// Circuit breaker for fault tolerance
package main

import (
    "context"
    "errors"
    "sync"
    "time"
)

type CircuitState int

const (
    StateClosed CircuitState = iota
    StateOpen
    StateHalfOpen
)

type CircuitBreaker struct {
    maxFailures   int
    resetTimeout  time.Duration
    state         CircuitState
    failures      int
    lastFailTime  time.Time
    mutex         sync.RWMutex
}

func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        maxFailures:  maxFailures,
        resetTimeout: resetTimeout,
        state:        StateClosed,
    }
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
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
        cb.lastFailTime = time.Now()
        
        if cb.failures >= cb.maxFailures {
            cb.state = StateOpen
        }
        return err
    }
    
    // Success
    cb.failures = 0
    cb.state = StateClosed
    return nil
}
```

### Bulkhead Pattern

```go
// Bulkhead pattern for resource isolation
package main

import (
    "context"
    "sync"
    "time"
)

type Bulkhead struct {
    maxConcurrency int
    semaphore      chan struct{}
    timeout        time.Duration
}

func NewBulkhead(maxConcurrency int, timeout time.Duration) *Bulkhead {
    return &Bulkhead{
        maxConcurrency: maxConcurrency,
        semaphore:      make(chan struct{}, maxConcurrency),
        timeout:        timeout,
    }
}

func (b *Bulkhead) Execute(ctx context.Context, fn func() error) error {
    select {
    case b.semaphore <- struct{}{}:
        defer func() { <-b.semaphore }()
        
        done := make(chan error, 1)
        go func() {
            done <- fn()
        }()
        
        select {
        case err := <-done:
            return err
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(b.timeout):
            return errors.New("bulkhead timeout")
        }
    case <-ctx.Done():
        return ctx.Err()
    case <-time.After(b.timeout):
        return errors.New("bulkhead full")
    }
}
```

---

## 📊 Real-world Case Studies

### Case Study 1: E-commerce API Optimization

**Problem**: API response time was 2-3 seconds, causing poor user experience.

**Solution**:
1. **Database Optimization**: Added indexes, optimized queries
2. **Caching**: Implemented Redis caching for product data
3. **Connection Pooling**: Optimized database connection pool
4. **Response Compression**: Added gzip compression

**Results**:
- Response time reduced to 200-300ms
- 90% reduction in database load
- 50% reduction in server resources

### Case Study 2: Payment Processing System

**Problem**: High latency during peak hours (Black Friday).

**Solution**:
1. **Horizontal Scaling**: Added more servers
2. **Load Balancing**: Implemented intelligent load balancing
3. **Caching**: Cached payment gateway responses
4. **Async Processing**: Moved non-critical operations to background

**Results**:
- 99.9% uptime during peak hours
- 60% reduction in response time
- 3x increase in throughput

---

## 🎯 Performance Best Practices

### 1. Measurement First
- Always profile before optimizing
- Set performance baselines
- Monitor continuously

### 2. Identify Bottlenecks
- Use profiling tools
- Monitor key metrics
- Test under load

### 3. Optimize Incrementally
- One component at a time
- Measure impact of each change
- A/B test optimizations

### 4. Consider Trade-offs
- Performance vs complexity
- Memory vs CPU
- Latency vs throughput

### 5. Plan for Scale
- Design for horizontal scaling
- Use appropriate data structures
- Implement proper caching

---

**⚡ Master these performance engineering techniques to build fast, scalable, and efficient backend systems! 🚀**
