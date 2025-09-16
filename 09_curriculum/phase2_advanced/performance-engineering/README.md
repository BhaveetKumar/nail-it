# Performance Engineering

## Table of Contents

1. [Overview](#overview/)
2. [Performance Measurement](#performance-measurement/)
3. [Memory Optimization](#memory-optimization/)
4. [CPU Optimization](#cpu-optimization/)
5. [I/O Optimization](#io-optimization/)
6. [Database Performance](#database-performance/)
7. [Caching Strategies](#caching-strategies/)
8. [Monitoring and Profiling](#monitoring-and-profiling/)
9. [Implementations](#implementations/)
10. [Follow-up Questions](#follow-up-questions/)
11. [Sources](#sources/)
12. [Projects](#projects/)

## Overview

### Learning Objectives

- Master performance measurement and profiling techniques
- Optimize memory usage and prevent leaks
- Improve CPU utilization and efficiency
- Optimize I/O operations and database queries
- Implement effective caching strategies
- Monitor and analyze system performance

### What is Performance Engineering?

Performance Engineering involves designing, implementing, and optimizing systems to achieve optimal performance, scalability, and resource utilization.

## Performance Measurement

### 1. Profiling Tools

#### Go Profiling
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

func cpuIntensiveTask() {
    sum := 0
    for i := 0; i < 1000000; i++ {
        sum += i * i
    }
}

func memoryIntensiveTask() {
    data := make([]int, 1000000)
    for i := range data {
        data[i] = i
    }
}

func main() {
    // Start pprof server
    go func() {
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()
    
    // CPU profiling
    fmt.Println("Starting CPU profiling...")
    for i := 0; i < 100; i++ {
        cpuIntensiveTask()
    }
    
    // Memory profiling
    fmt.Println("Starting memory profiling...")
    for i := 0; i < 100; i++ {
        memoryIntensiveTask()
    }
    
    // Print memory stats
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    fmt.Printf("Alloc = %d KB\n", m.Alloc/1024)
    fmt.Printf("TotalAlloc = %d KB\n", m.TotalAlloc/1024)
    fmt.Printf("Sys = %d KB\n", m.Sys/1024)
    fmt.Printf("NumGC = %d\n", m.NumGC)
    
    fmt.Println("Profiling data available at http://localhost:6060/debug/pprof/")
    time.Sleep(10 * time.Second)
}
```

#### Node.js Profiling
```javascript
const v8 = require('v8');
const fs = require('fs');

// Enable profiling
v8.setFlagsFromString('--prof');

function cpuIntensiveTask() {
    let sum = 0;
    for (let i = 0; i < 1000000; i++) {
        sum += i * i;
    }
    return sum;
}

function memoryIntensiveTask() {
    const data = [];
    for (let i = 0; i < 100000; i++) {
        data.push({ id: i, value: Math.random() });
    }
    return data;
}

// Performance measurement
console.time('CPU Task');
for (let i = 0; i < 100; i++) {
    cpuIntensiveTask();
}
console.timeEnd('CPU Task');

console.time('Memory Task');
for (let i = 0; i < 100; i++) {
    memoryIntensiveTask();
}
console.timeEnd('Memory Task');

// Memory usage
const memUsage = process.memoryUsage();
console.log('Memory Usage:', {
    rss: `${Math.round(memUsage.rss / 1024 / 1024)} MB`,
    heapTotal: `${Math.round(memUsage.heapTotal / 1024 / 1024)} MB`,
    heapUsed: `${Math.round(memUsage.heapUsed / 1024 / 1024)} MB`,
    external: `${Math.round(memUsage.external / 1024 / 1024)} MB`
});

// Generate heap snapshot
const heapSnapshot = v8.getHeapSnapshot();
const fileName = `heap-${Date.now()}.heapsnapshot`;
const fileStream = fs.createWriteStream(fileName);
heapSnapshot.pipe(fileStream);
console.log(`Heap snapshot saved to ${fileName}`);
```

## Memory Optimization

### 1. Memory Pool Implementation

#### Go Memory Pool
```go
package main

import (
    "fmt"
    "sync"
)

type Object struct {
    ID   int
    Data []byte
}

type ObjectPool struct {
    pool sync.Pool
}

func NewObjectPool() *ObjectPool {
    return &ObjectPool{
        pool: sync.Pool{
            New: func() interface{} {
                return &Object{
                    Data: make([]byte, 1024),
                }
            },
        },
    }
}

func (p *ObjectPool) Get() *Object {
    obj := p.pool.Get().(*Object)
    obj.ID = 0
    obj.Data = obj.Data[:0] // Reset slice length
    return obj
}

func (p *ObjectPool) Put(obj *Object) {
    p.pool.Put(obj)
}

func main() {
    pool := NewObjectPool()
    
    // Use objects from pool
    for i := 0; i < 1000; i++ {
        obj := pool.Get()
        obj.ID = i
        obj.Data = append(obj.Data, []byte("data")...)
        
        // Process object
        fmt.Printf("Processing object %d\n", obj.ID)
        
        // Return to pool
        pool.Put(obj)
    }
}
```

### 2. Memory Leak Detection

#### Go Memory Leak Detection
```go
package main

import (
    "fmt"
    "runtime"
    "time"
)

func leakyFunction() {
    // This creates a memory leak
    data := make([]int, 1000000)
    for i := range data {
        data[i] = i
    }
    // data is not used after this point but not garbage collected
}

func properFunction() {
    // This properly manages memory
    data := make([]int, 1000000)
    for i := range data {
        data[i] = i
    }
    // data is automatically garbage collected when function returns
}

func main() {
    var m1, m2 runtime.MemStats
    
    // Measure before
    runtime.GC()
    runtime.ReadMemStats(&m1)
    fmt.Printf("Before: Alloc = %d KB\n", m1.Alloc/1024)
    
    // Run leaky function
    for i := 0; i < 100; i++ {
        leakyFunction()
    }
    
    // Force garbage collection
    runtime.GC()
    runtime.ReadMemStats(&m2)
    fmt.Printf("After leaky: Alloc = %d KB\n", m2.Alloc/1024)
    
    // Run proper function
    for i := 0; i < 100; i++ {
        properFunction()
    }
    
    // Force garbage collection
    runtime.GC()
    runtime.ReadMemStats(&m2)
    fmt.Printf("After proper: Alloc = %d KB\n", m2.Alloc/1024)
}
```

## CPU Optimization

### 1. Parallel Processing

#### Go Worker Pool
```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

type Job struct {
    ID   int
    Data int
}

type Result struct {
    JobID int
    Value int
}

func worker(id int, jobs <-chan Job, results chan<- Result) {
    for job := range jobs {
        // Simulate CPU-intensive work
        result := 0
        for i := 0; i < job.Data; i++ {
            result += i * i
        }
        
        results <- Result{
            JobID: job.ID,
            Value: result,
        }
    }
}

func main() {
    numWorkers := runtime.NumCPU()
    numJobs := 1000
    
    jobs := make(chan Job, numJobs)
    results := make(chan Result, numJobs)
    
    // Start workers
    var wg sync.WaitGroup
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            worker(id, jobs, results)
        }(i)
    }
    
    // Send jobs
    start := time.Now()
    for i := 0; i < numJobs; i++ {
        jobs <- Job{ID: i, Data: 10000}
    }
    close(jobs)
    
    // Collect results
    go func() {
        wg.Wait()
        close(results)
    }()
    
    count := 0
    for result := range results {
        count++
        if count%100 == 0 {
            fmt.Printf("Processed %d jobs\n", count)
        }
    }
    
    elapsed := time.Since(start)
    fmt.Printf("Processed %d jobs in %v\n", count, elapsed)
    fmt.Printf("Jobs per second: %.2f\n", float64(count)/elapsed.Seconds())
}
```

### 2. SIMD Optimization

#### Vectorized Operations
```go
package main

import (
    "fmt"
    "math"
    "time"
)

func naiveSum(a, b []float64) []float64 {
    result := make([]float64, len(a))
    for i := range a {
        result[i] = a[i] + b[i]
    }
    return result
}

func optimizedSum(a, b []float64) []float64 {
    result := make([]float64, len(a))
    
    // Process in chunks for better cache locality
    chunkSize := 8
    for i := 0; i < len(a); i += chunkSize {
        end := i + chunkSize
        if end > len(a) {
            end = len(a)
        }
        
        for j := i; j < end; j++ {
            result[j] = a[j] + b[j]
        }
    }
    
    return result
}

func main() {
    size := 1000000
    a := make([]float64, size)
    b := make([]float64, size)
    
    for i := range a {
        a[i] = float64(i)
        b[i] = float64(i * 2)
    }
    
    // Benchmark naive approach
    start := time.Now()
    for i := 0; i < 100; i++ {
        naiveSum(a, b)
    }
    naiveTime := time.Since(start)
    
    // Benchmark optimized approach
    start = time.Now()
    for i := 0; i < 100; i++ {
        optimizedSum(a, b)
    }
    optimizedTime := time.Since(start)
    
    fmt.Printf("Naive approach: %v\n", naiveTime)
    fmt.Printf("Optimized approach: %v\n", optimizedTime)
    fmt.Printf("Speedup: %.2fx\n", float64(naiveTime)/float64(optimizedTime))
}
```

## I/O Optimization

### 1. Asynchronous I/O

#### Go Async I/O
```go
package main

import (
    "fmt"
    "io"
    "net/http"
    "sync"
    "time"
)

func fetchURL(url string, wg *sync.WaitGroup, results chan<- string) {
    defer wg.Done()
    
    start := time.Now()
    resp, err := http.Get(url)
    if err != nil {
        results <- fmt.Sprintf("Error fetching %s: %v", url, err)
        return
    }
    defer resp.Body.Close()
    
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        results <- fmt.Sprintf("Error reading %s: %v", url, err)
        return
    }
    
    elapsed := time.Since(start)
    results <- fmt.Sprintf("Fetched %s (%d bytes) in %v", url, len(body), elapsed)
}

func main() {
    urls := []string{
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/3",
    }
    
    var wg sync.WaitGroup
    results := make(chan string, len(urls))
    
    start := time.Now()
    
    for _, url := range urls {
        wg.Add(1)
        go fetchURL(url, &wg, results)
    }
    
    go func() {
        wg.Wait()
        close(results)
    }()
    
    for result := range results {
        fmt.Println(result)
    }
    
    elapsed := time.Since(start)
    fmt.Printf("Total time: %v\n", elapsed)
}
```

### 2. Database Query Optimization

#### SQL Query Optimization
```sql
-- Create optimized indexes
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_date ON orders(created_at);
CREATE INDEX idx_order_user_date ON orders(user_id, created_at);

-- Optimized query with proper joins
EXPLAIN ANALYZE
SELECT u.name, u.email, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2023-01-01'
  AND o.created_at >= '2023-01-01'
GROUP BY u.id, u.name, u.email
HAVING COUNT(o.id) > 5
ORDER BY order_count DESC
LIMIT 10;

-- Use prepared statements
PREPARE get_user_orders AS
SELECT o.*, u.name, u.email
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.user_id = $1 AND o.created_at >= $2;

-- Batch operations
INSERT INTO orders (user_id, product_id, quantity, price)
VALUES 
  (1, 101, 2, 29.99),
  (1, 102, 1, 19.99),
  (2, 101, 3, 29.99);
```

## Database Performance

### 1. Connection Pooling

#### Go Database Pool
```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "sync"
    "time"
    
    _ "github.com/lib/pq"
)

type DBPool struct {
    db *sql.DB
}

func NewDBPool(dsn string) (*DBPool, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)
    
    // Test connection
    if err := db.Ping(); err != nil {
        return nil, err
    }
    
    return &DBPool{db: db}, nil
}

func (p *DBPool) GetUser(id int) (*User, error) {
    user := &User{}
    err := p.db.QueryRow("SELECT id, name, email FROM users WHERE id = $1", id).
        Scan(&user.ID, &user.Name, &user.Email)
    return user, err
}

func (p *DBPool) GetUsers(limit, offset int) ([]*User, error) {
    rows, err := p.db.Query("SELECT id, name, email FROM users LIMIT $1 OFFSET $2", limit, offset)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []*User
    for rows.Next() {
        user := &User{}
        if err := rows.Scan(&user.ID, &user.Name, &user.Email); err != nil {
            return nil, err
        }
        users = append(users, user)
    }
    
    return users, nil
}

type User struct {
    ID    int
    Name  string
    Email string
}

func main() {
    dsn := "user=postgres password=password dbname=test sslmode=disable"
    pool, err := NewDBPool(dsn)
    if err != nil {
        log.Fatal(err)
    }
    
    // Concurrent database operations
    var wg sync.WaitGroup
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            user, err := pool.GetUser(id)
            if err != nil {
                log.Printf("Error getting user %d: %v", id, err)
                return
            }
            fmt.Printf("User: %+v\n", user)
        }(i)
    }
    
    wg.Wait()
}
```

## Caching Strategies

### 1. Multi-Level Cache

#### Go Cache Implementation
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type CacheItem struct {
    Value     interface{}
    ExpiresAt time.Time
}

type MultiLevelCache struct {
    L1Cache map[string]*CacheItem
    L2Cache map[string]*CacheItem
    mutex   sync.RWMutex
}

func NewMultiLevelCache() *MultiLevelCache {
    return &MultiLevelCache{
        L1Cache: make(map[string]*CacheItem),
        L2Cache: make(map[string]*CacheItem),
    }
}

func (c *MultiLevelCache) Get(key string) (interface{}, bool) {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    
    // Check L1 cache first
    if item, exists := c.L1Cache[key]; exists {
        if time.Now().Before(item.ExpiresAt) {
            return item.Value, true
        }
        delete(c.L1Cache, key)
    }
    
    // Check L2 cache
    if item, exists := c.L2Cache[key]; exists {
        if time.Now().Before(item.ExpiresAt) {
            // Promote to L1
            c.L1Cache[key] = item
            return item.Value, true
        }
        delete(c.L2Cache, key)
    }
    
    return nil, false
}

func (c *MultiLevelCache) Set(key string, value interface{}, ttl time.Duration) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    item := &CacheItem{
        Value:     value,
        ExpiresAt: time.Now().Add(ttl),
    }
    
    // Store in L1 cache
    c.L1Cache[key] = item
    
    // If L1 is full, move oldest to L2
    if len(c.L1Cache) > 1000 {
        // Simple eviction: move to L2
        for k, v := range c.L1Cache {
            if k != key {
                c.L2Cache[k] = v
                delete(c.L1Cache, k)
                break
            }
        }
    }
}

func (c *MultiLevelCache) Delete(key string) {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    delete(c.L1Cache, key)
    delete(c.L2Cache, key)
}

func main() {
    cache := NewMultiLevelCache()
    
    // Set some values
    cache.Set("key1", "value1", 5*time.Second)
    cache.Set("key2", "value2", 10*time.Second)
    
    // Get values
    if value, exists := cache.Get("key1"); exists {
        fmt.Printf("Found key1: %v\n", value)
    }
    
    if value, exists := cache.Get("key2"); exists {
        fmt.Printf("Found key2: %v\n", value)
    }
    
    // Wait for expiration
    time.Sleep(6 * time.Second)
    
    if value, exists := cache.Get("key1"); exists {
        fmt.Printf("Found key1 after expiration: %v\n", value)
    } else {
        fmt.Println("key1 expired")
    }
}
```

## Monitoring and Profiling

### 1. Performance Metrics

#### Go Metrics Collection
```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

type Metrics struct {
    RequestCount    int64
    ResponseTime    time.Duration
    ErrorCount      int64
    MemoryUsage     uint64
    GoroutineCount  int
    mutex           sync.RWMutex
}

func (m *Metrics) RecordRequest(duration time.Duration, isError bool) {
    m.mutex.Lock()
    defer m.mutex.Unlock()
    
    m.RequestCount++
    m.ResponseTime = duration
    if isError {
        m.ErrorCount++
    }
}

func (m *Metrics) UpdateSystemMetrics() {
    m.mutex.Lock()
    defer m.mutex.Unlock()
    
    var memStats runtime.MemStats
    runtime.ReadMemStats(&memStats)
    m.MemoryUsage = memStats.Alloc
    m.GoroutineCount = runtime.NumGoroutine()
}

func (m *Metrics) GetStats() map[string]interface{} {
    m.mutex.RLock()
    defer m.mutex.RUnlock()
    
    return map[string]interface{}{
        "request_count":     m.RequestCount,
        "response_time_ms":  m.ResponseTime.Milliseconds(),
        "error_count":       m.ErrorCount,
        "memory_usage_mb":   float64(m.MemoryUsage) / 1024 / 1024,
        "goroutine_count":   m.GoroutineCount,
        "error_rate":        float64(m.ErrorCount) / float64(m.RequestCount),
    }
}

func main() {
    metrics := &Metrics{}
    
    // Simulate some requests
    for i := 0; i < 100; i++ {
        start := time.Now()
        time.Sleep(time.Duration(i%10) * time.Millisecond)
        duration := time.Since(start)
        
        isError := i%20 == 0
        metrics.RecordRequest(duration, isError)
    }
    
    metrics.UpdateSystemMetrics()
    
    stats := metrics.GetStats()
    for key, value := range stats {
        fmt.Printf("%s: %v\n", key, value)
    }
}
```

## Follow-up Questions

### 1. Performance Measurement
**Q: What's the difference between profiling and benchmarking?**
A: Profiling analyzes where time is spent in a program, while benchmarking measures the performance of specific operations.

### 2. Memory Optimization
**Q: How do you detect memory leaks in Go?**
A: Use tools like pprof, monitor memory usage over time, and check for goroutine leaks.

### 3. CPU Optimization
**Q: What are the benefits of worker pools?**
A: Worker pools limit resource usage, improve throughput, and provide better control over concurrency.

## Sources

### Books
- **Systems Performance** by Brendan Gregg
- **High Performance MySQL** by Baron Schwartz
- **Go in Action** by Manning Publications

### Online Resources
- **Go Profiling** - Official Go profiling guide
- **Performance Optimization** - Best practices
- **Database Tuning** - Query optimization

## Projects

### 1. Performance Monitoring System
**Objective**: Build a comprehensive performance monitoring system
**Requirements**: Metrics collection, alerting, visualization
**Deliverables**: Complete monitoring platform

### 2. High-Performance Web Server
**Objective**: Create an optimized web server
**Requirements**: Async I/O, connection pooling, caching
**Deliverables**: Production-ready web server

### 3. Database Optimization Tool
**Objective**: Develop a database performance optimization tool
**Requirements**: Query analysis, index recommendations, monitoring
**Deliverables**: Complete optimization suite

---

**Next**: [Security Engineering](security-engineering/README.md/) | **Previous**: [Advanced Algorithms](advanced-algorithms/README.md/) | **Up**: [Phase 2](README.md/)

