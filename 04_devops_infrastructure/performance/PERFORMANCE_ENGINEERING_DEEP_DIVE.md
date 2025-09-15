# 🚀 **Performance Engineering Deep Dive**

## 📊 **Complete Guide to System Performance Optimization**

---

## 🎯 **1. Go Runtime Performance Optimization**

### **Memory Management and GC Tuning**

```go
package main

import (
    "runtime"
    "runtime/debug"
    "sync"
    "time"
    "unsafe"
)

// Memory Pool for Object Reuse
type ObjectPool struct {
    pool sync.Pool
    size int
}

type PooledObject struct {
    Data []byte
    ID   int
}

func NewObjectPool(size int) *ObjectPool {
    return &ObjectPool{
        pool: sync.Pool{
            New: func() interface{} {
                return &PooledObject{
                    Data: make([]byte, size),
                    ID:   0,
                }
            },
        },
        size: size,
    }
}

func (op *ObjectPool) Get() *PooledObject {
    obj := op.pool.Get().(*PooledObject)
    obj.ID = 0
    return obj
}

func (op *ObjectPool) Put(obj *PooledObject) {
    // Reset object state
    obj.ID = 0
    for i := range obj.Data {
        obj.Data[i] = 0
    }
    op.pool.Put(obj)
}

// String Interning for Memory Efficiency
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

// GC Tuning and Monitoring
type GCTuner struct {
    targetHeapSize int64
    maxHeapSize    int64
    gcPercent      int
}

func NewGCTuner(targetHeapSize, maxHeapSize int64) *GCTuner {
    return &GCTuner{
        targetHeapSize: targetHeapSize,
        maxHeapSize:    maxHeapSize,
        gcPercent:      100, // Default GOGC value
    }
}

func (gt *GCTuner) TuneGC() {
    // Set GOGC based on heap size
    if gt.targetHeapSize > 0 {
        gt.gcPercent = int((gt.targetHeapSize * 100) / gt.maxHeapSize)
        if gt.gcPercent < 10 {
            gt.gcPercent = 10
        }
        if gt.gcPercent > 1000 {
            gt.gcPercent = 1000
        }
        debug.SetGCPercent(gt.gcPercent)
    }
}

func (gt *GCTuner) MonitorGC() {
    var m runtime.MemStats
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()
    
    for range ticker.C {
        runtime.ReadMemStats(&m)
        
        fmt.Printf("Heap: %d KB, GC Cycles: %d, GC Pause: %v\n",
            m.HeapInuse/1024,
            m.NumGC,
            time.Duration(m.PauseTotalNs)/time.Millisecond,
        )
        
        // Adjust GC if needed
        if m.HeapInuse > uint64(gt.maxHeapSize) {
            gt.TuneGC()
        }
    }
}

// Lock-Free Data Structures
type LockFreeQueue struct {
    head unsafe.Pointer
    tail unsafe.Pointer
}

type node struct {
    value interface{}
    next  unsafe.Pointer
}

func NewLockFreeQueue() *LockFreeQueue {
    n := unsafe.Pointer(&node{})
    return &LockFreeQueue{
        head: n,
        tail: n,
    }
}

func (q *LockFreeQueue) Enqueue(value interface{}) {
    n := &node{value: value}
    
    for {
        tail := (*node)(atomic.LoadPointer(&q.tail))
        next := (*node)(atomic.LoadPointer(&tail.next))
        
        if tail == (*node)(atomic.LoadPointer(&q.tail)) {
            if next == nil {
                if atomic.CompareAndSwapPointer(&tail.next, unsafe.Pointer(next), unsafe.Pointer(n)) {
                    break
                }
            } else {
                atomic.CompareAndSwapPointer(&q.tail, unsafe.Pointer(tail), unsafe.Pointer(next))
            }
        }
    }
    
    atomic.CompareAndSwapPointer(&q.tail, unsafe.Pointer((*node)(atomic.LoadPointer(&q.tail))), unsafe.Pointer(n))
}

func (q *LockFreeQueue) Dequeue() (interface{}, bool) {
    for {
        head := (*node)(atomic.LoadPointer(&q.head))
        tail := (*node)(atomic.LoadPointer(&q.tail))
        next := (*node)(atomic.LoadPointer(&head.next))
        
        if head == (*node)(atomic.LoadPointer(&q.head)) {
            if head == tail {
                if next == nil {
                    return nil, false
                }
                atomic.CompareAndSwapPointer(&q.tail, unsafe.Pointer(tail), unsafe.Pointer(next))
            } else {
                if next == nil {
                    continue
                }
                value := next.value
                if atomic.CompareAndSwapPointer(&q.head, unsafe.Pointer(head), unsafe.Pointer(next)) {
                    return value, true
                }
            }
        }
    }
}

// Example usage
func main() {
    // Memory pool example
    pool := NewObjectPool(1024)
    
    obj := pool.Get()
    obj.ID = 123
    copy(obj.Data, []byte("Hello, World!"))
    
    // Use object...
    pool.Put(obj)
    
    // String interning example
    interner := NewStringInterner()
    s1 := interner.Intern("hello")
    s2 := interner.Intern("hello")
    fmt.Printf("Same string: %t\n", s1 == s2)
    
    // GC tuning example
    tuner := NewGCTuner(100*1024*1024, 200*1024*1024) // 100MB target, 200MB max
    tuner.TuneGC()
    go tuner.MonitorGC()
    
    // Lock-free queue example
    queue := NewLockFreeQueue()
    queue.Enqueue("item1")
    queue.Enqueue("item2")
    
    if value, ok := queue.Dequeue(); ok {
        fmt.Printf("Dequeued: %v\n", value)
    }
}
```

### **CPU Profiling and Optimization**

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "runtime/pprof"
    "sync"
    "time"
)

// CPU-intensive task for profiling
func cpuIntensiveTask(n int) int {
    result := 0
    for i := 0; i < n; i++ {
        result += i * i
    }
    return result
}

// Optimized version with better algorithm
func optimizedTask(n int) int {
    // Sum of squares formula: n(n+1)(2n+1)/6
    return n * (n + 1) * (2*n + 1) / 6
}

// Profiling wrapper
func profileCPU(fn func(), filename string) {
    f, err := os.Create(filename)
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    
    if err := pprof.StartCPUProfile(f); err != nil {
        log.Fatal(err)
    }
    defer pprof.StopCPUProfile()
    
    fn()
}

// Memory profiling
func profileMemory(fn func(), filename string) {
    f, err := os.Create(filename)
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    
    fn()
    
    if err := pprof.WriteHeapProfile(f); err != nil {
        log.Fatal(err)
    }
}

// Benchmark comparison
func benchmarkComparison() {
    const iterations = 1000000
    
    // Profile unoptimized version
    profileCPU(func() {
        for i := 0; i < iterations; i++ {
            cpuIntensiveTask(1000)
        }
    }, "cpu_unoptimized.prof")
    
    // Profile optimized version
    profileCPU(func() {
        for i := 0; i < iterations; i++ {
            optimizedTask(1000)
        }
    }, "cpu_optimized.prof")
    
    // Memory profiling
    profileMemory(func() {
        var data []int
        for i := 0; i < 100000; i++ {
            data = append(data, i)
        }
    }, "memory.prof")
}

// Concurrent processing optimization
type WorkerPool struct {
    workers    int
    jobQueue   chan func()
    resultQueue chan interface{}
    wg         sync.WaitGroup
}

func NewWorkerPool(workers int) *WorkerPool {
    return &WorkerPool{
        workers:     workers,
        jobQueue:    make(chan func(), 1000),
        resultQueue: make(chan interface{}, 1000),
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
    
    for job := range wp.jobQueue {
        job()
    }
}

func (wp *WorkerPool) Submit(job func()) {
    wp.jobQueue <- job
}

func (wp *WorkerPool) Close() {
    close(wp.jobQueue)
    wp.wg.Wait()
    close(wp.resultQueue)
}

// Cache optimization
type LRUCache struct {
    capacity int
    cache    map[string]*Node
    head     *Node
    tail     *Node
    mutex    sync.RWMutex
}

type Node struct {
    key   string
    value interface{}
    prev  *Node
    next  *Node
}

func NewLRUCache(capacity int) *LRUCache {
    lru := &LRUCache{
        capacity: capacity,
        cache:    make(map[string]*Node),
    }
    
    lru.head = &Node{}
    lru.tail = &Node{}
    lru.head.next = lru.tail
    lru.tail.prev = lru.head
    
    return lru
}

func (lru *LRUCache) Get(key string) (interface{}, bool) {
    lru.mutex.RLock()
    node, exists := lru.cache[key]
    lru.mutex.RUnlock()
    
    if !exists {
        return nil, false
    }
    
    lru.mutex.Lock()
    lru.moveToHead(node)
    lru.mutex.Unlock()
    
    return node.value, true
}

func (lru *LRUCache) Put(key string, value interface{}) {
    lru.mutex.Lock()
    defer lru.mutex.Unlock()
    
    if node, exists := lru.cache[key]; exists {
        node.value = value
        lru.moveToHead(node)
        return
    }
    
    if len(lru.cache) >= lru.capacity {
        lru.removeTail()
    }
    
    newNode := &Node{
        key:   key,
        value: value,
    }
    
    lru.cache[key] = newNode
    lru.addToHead(newNode)
}

func (lru *LRUCache) moveToHead(node *Node) {
    lru.removeNode(node)
    lru.addToHead(node)
}

func (lru *LRUCache) addToHead(node *Node) {
    node.prev = lru.head
    node.next = lru.head.next
    lru.head.next.prev = node
    lru.head.next = node
}

func (lru *LRUCache) removeNode(node *Node) {
    node.prev.next = node.next
    node.next.prev = node.prev
}

func (lru *LRUCache) removeTail() {
    lastNode := lru.tail.prev
    lru.removeNode(lastNode)
    delete(lru.cache, lastNode.key)
}

// Example usage
func main() {
    // Run benchmarks
    benchmarkComparison()
    
    // Worker pool example
    pool := NewWorkerPool(4)
    pool.Start()
    
    for i := 0; i < 100; i++ {
        i := i
        pool.Submit(func() {
            result := optimizedTask(i * 100)
            fmt.Printf("Worker result: %d\n", result)
        })
    }
    
    pool.Close()
    
    // LRU cache example
    cache := NewLRUCache(3)
    cache.Put("key1", "value1")
    cache.Put("key2", "value2")
    cache.Put("key3", "value3")
    
    if value, ok := cache.Get("key1"); ok {
        fmt.Printf("Cache hit: %v\n", value)
    }
}
```

---

## 🎯 **2. Database Performance Optimization**

### **Query Optimization and Indexing**

```go
package main

import (
    "database/sql"
    "fmt"
    "log"
    "time"
    
    _ "github.com/go-sql-driver/mysql"
)

type DatabaseOptimizer struct {
    db *sql.DB
}

func NewDatabaseOptimizer(dsn string) (*DatabaseOptimizer, error) {
    db, err := sql.Open("mysql", dsn)
    if err != nil {
        return nil, err
    }
    
    // Optimize connection settings
    db.SetMaxOpenConns(100)
    db.SetMaxIdleConns(10)
    db.SetConnMaxLifetime(time.Hour)
    
    return &DatabaseOptimizer{db: db}, nil
}

// Optimized query with proper indexing
func (do *DatabaseOptimizer) GetUsersByEmail(email string) ([]User, error) {
    // Use prepared statement for better performance
    query := `SELECT id, username, email, created_at 
              FROM users 
              WHERE email = ? 
              ORDER BY created_at DESC 
              LIMIT 100`
    
    stmt, err := do.db.Prepare(query)
    if err != nil {
        return nil, err
    }
    defer stmt.Close()
    
    rows, err := stmt.Query(email)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []User
    for rows.Next() {
        var user User
        err := rows.Scan(&user.ID, &user.Username, &user.Email, &user.CreatedAt)
        if err != nil {
            return nil, err
        }
        users = append(users, user)
    }
    
    return users, nil
}

// Batch operations for better performance
func (do *DatabaseOptimizer) BatchInsertUsers(users []User) error {
    if len(users) == 0 {
        return nil
    }
    
    // Use transaction for batch operations
    tx, err := do.db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    stmt, err := tx.Prepare(`INSERT INTO users (username, email, created_at) VALUES (?, ?, ?)`)
    if err != nil {
        return err
    }
    defer stmt.Close()
    
    for _, user := range users {
        _, err := stmt.Exec(user.Username, user.Email, time.Now())
        if err != nil {
            return err
        }
    }
    
    return tx.Commit()
}

// Connection pooling optimization
type ConnectionPool struct {
    db     *sql.DB
    config PoolConfig
}

type PoolConfig struct {
    MaxOpenConns    int
    MaxIdleConns    int
    ConnMaxLifetime time.Duration
    ConnMaxIdleTime time.Duration
}

func NewConnectionPool(dsn string, config PoolConfig) (*ConnectionPool, error) {
    db, err := sql.Open("mysql", dsn)
    if err != nil {
        return nil, err
    }
    
    db.SetMaxOpenConns(config.MaxOpenConns)
    db.SetMaxIdleConns(config.MaxIdleConns)
    db.SetConnMaxLifetime(config.ConnMaxLifetime)
    db.SetConnMaxIdleTime(config.ConnMaxIdleTime)
    
    return &ConnectionPool{db: db, config: config}, nil
}

// Query performance monitoring
func (do *DatabaseOptimizer) MonitorQueryPerformance(query string, args ...interface{}) (time.Duration, error) {
    start := time.Now()
    
    _, err := do.db.Exec(query, args...)
    if err != nil {
        return 0, err
    }
    
    duration := time.Since(start)
    
    // Log slow queries
    if duration > 100*time.Millisecond {
        log.Printf("Slow query detected: %s (duration: %v)", query, duration)
    }
    
    return duration, nil
}

// Index optimization suggestions
func (do *DatabaseOptimizer) AnalyzeTable(tableName string) error {
    query := fmt.Sprintf("ANALYZE TABLE %s", tableName)
    _, err := do.db.Exec(query)
    return err
}

func (do *DatabaseOptimizer) GetTableStats(tableName string) (map[string]interface{}, error) {
    query := `SELECT 
                table_rows,
                data_length,
                index_length,
                data_free
              FROM information_schema.tables 
              WHERE table_name = ?`
    
    row := do.db.QueryRow(query, tableName)
    
    var stats struct {
        TableRows  int64
        DataLength int64
        IndexLength int64
        DataFree   int64
    }
    
    err := row.Scan(&stats.TableRows, &stats.DataLength, &stats.IndexLength, &stats.DataFree)
    if err != nil {
        return nil, err
    }
    
    return map[string]interface{}{
        "table_rows":    stats.TableRows,
        "data_length":   stats.DataLength,
        "index_length":  stats.IndexLength,
        "data_free":     stats.DataFree,
        "total_size":    stats.DataLength + stats.IndexLength,
    }, nil
}

// Example usage
func main() {
    dsn := "user:password@tcp(localhost:3306)/testdb?parseTime=true"
    
    optimizer, err := NewDatabaseOptimizer(dsn)
    if err != nil {
        log.Fatal(err)
    }
    defer optimizer.db.Close()
    
    // Monitor query performance
    duration, err := optimizer.MonitorQueryPerformance(
        "SELECT * FROM users WHERE email = ?",
        "test@example.com",
    )
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Query executed in: %v\n", duration)
    
    // Analyze table
    if err := optimizer.AnalyzeTable("users"); err != nil {
        log.Fatal(err)
    }
    
    // Get table statistics
    stats, err := optimizer.GetTableStats("users")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Table stats: %+v\n", stats)
}
```

---

## 🎯 **3. Network Performance Optimization**

### **HTTP Client Optimization**

```go
package main

import (
    "context"
    "crypto/tls"
    "fmt"
    "io"
    "net"
    "net/http"
    "sync"
    "time"
)

type HTTPOptimizer struct {
    client *http.Client
    pool   *ConnectionPool
}

type ConnectionPool struct {
    maxConns        int
    maxConnsPerHost int
    idleTimeout     time.Duration
    keepAlive       time.Duration
}

func NewHTTPOptimizer() *HTTPOptimizer {
    // Custom transport with optimizations
    transport := &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
        DisableKeepAlives:   false,
        DisableCompression:  false,
        TLSClientConfig: &tls.Config{
            InsecureSkipVerify: false,
        },
        DialContext: (&net.Dialer{
            Timeout:   30 * time.Second,
            KeepAlive: 30 * time.Second,
        }).DialContext,
    }
    
    client := &http.Client{
        Transport: transport,
        Timeout:   30 * time.Second,
    }
    
    return &HTTPOptimizer{
        client: client,
        pool: &ConnectionPool{
            maxConns:        100,
            maxConnsPerHost: 10,
            idleTimeout:     90 * time.Second,
            keepAlive:       30 * time.Second,
        },
    }
}

// Concurrent HTTP requests
func (ho *HTTPOptimizer) ConcurrentRequests(urls []string) ([]*http.Response, error) {
    var wg sync.WaitGroup
    responses := make([]*http.Response, len(urls))
    errors := make([]error, len(urls))
    
    for i, url := range urls {
        wg.Add(1)
        go func(index int, url string) {
            defer wg.Done()
            
            resp, err := ho.client.Get(url)
            if err != nil {
                errors[index] = err
                return
            }
            
            responses[index] = resp
        }(i, url)
    }
    
    wg.Wait()
    
    // Check for errors
    for i, err := range errors {
        if err != nil {
            return nil, fmt.Errorf("request %d failed: %v", i, err)
        }
    }
    
    return responses, nil
}

// HTTP/2 optimization
func (ho *HTTPOptimizer) EnableHTTP2() {
    transport := ho.client.Transport.(*http.Transport)
    transport.ForceAttemptHTTP2 = true
}

// Connection pooling optimization
func (ho *HTTPOptimizer) OptimizeConnectionPool() {
    transport := ho.client.Transport.(*http.Transport)
    
    transport.MaxIdleConns = ho.pool.maxConns
    transport.MaxIdleConnsPerHost = ho.pool.maxConnsPerHost
    transport.IdleConnTimeout = ho.pool.idleTimeout
    
    dialer := &net.Dialer{
        Timeout:   30 * time.Second,
        KeepAlive: ho.pool.keepAlive,
    }
    
    transport.DialContext = dialer.DialContext
}

// Request batching
func (ho *HTTPOptimizer) BatchRequests(requests []Request) ([]Response, error) {
    var wg sync.WaitGroup
    responses := make([]Response, len(requests))
    errors := make([]error, len(requests))
    
    semaphore := make(chan struct{}, 10) // Limit concurrent requests
    
    for i, req := range requests {
        wg.Add(1)
        go func(index int, request Request) {
            defer wg.Done()
            
            semaphore <- struct{}{} // Acquire semaphore
            defer func() { <-semaphore }() // Release semaphore
            
            resp, err := ho.executeRequest(request)
            if err != nil {
                errors[index] = err
                return
            }
            
            responses[index] = resp
        }(i, req)
    }
    
    wg.Wait()
    
    // Check for errors
    for i, err := range errors {
        if err != nil {
            return nil, fmt.Errorf("request %d failed: %v", i, err)
        }
    }
    
    return responses, nil
}

type Request struct {
    URL    string
    Method string
    Body   io.Reader
    Header map[string]string
}

type Response struct {
    StatusCode int
    Body       []byte
    Header     http.Header
}

func (ho *HTTPOptimizer) executeRequest(req Request) (Response, error) {
    httpReq, err := http.NewRequest(req.Method, req.URL, req.Body)
    if err != nil {
        return Response{}, err
    }
    
    for key, value := range req.Header {
        httpReq.Header.Set(key, value)
    }
    
    resp, err := ho.client.Do(httpReq)
    if err != nil {
        return Response{}, err
    }
    defer resp.Body.Close()
    
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return Response{}, err
    }
    
    return Response{
        StatusCode: resp.StatusCode,
        Body:       body,
        Header:     resp.Header,
    }, nil
}

// Example usage
func main() {
    optimizer := NewHTTPOptimizer()
    optimizer.EnableHTTP2()
    optimizer.OptimizeConnectionPool()
    
    // Concurrent requests example
    urls := []string{
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/3",
    }
    
    start := time.Now()
    responses, err := optimizer.ConcurrentRequests(urls)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Concurrent requests completed in: %v\n", time.Since(start))
    
    // Close responses
    for _, resp := range responses {
        resp.Body.Close()
    }
}
```

---

## 🎯 **4. Caching Strategies and Optimization**

### **Multi-Level Caching System**

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "sync"
    "time"
    
    "github.com/go-redis/redis/v8"
)

type MultiLevelCache struct {
    l1Cache *L1Cache
    l2Cache *L2Cache
    l3Cache *L3Cache
}

type L1Cache struct {
    data map[string]*CacheItem
    mutex sync.RWMutex
    maxSize int
}

type L2Cache struct {
    client *redis.Client
    prefix string
}

type L3Cache struct {
    client *redis.Client
    prefix string
    cluster bool
}

type CacheItem struct {
    Value     interface{}
    ExpiresAt time.Time
    AccessCount int
    LastAccess time.Time
}

func NewMultiLevelCache() *MultiLevelCache {
    return &MultiLevelCache{
        l1Cache: NewL1Cache(1000),
        l2Cache: NewL2Cache("l2:"),
        l3Cache: NewL3Cache("l3:", true),
    }
}

func NewL1Cache(maxSize int) *L1Cache {
    return &L1Cache{
        data:    make(map[string]*CacheItem),
        maxSize: maxSize,
    }
}

func (l1 *L1Cache) Get(key string) (interface{}, bool) {
    l1.mutex.RLock()
    item, exists := l1.data[key]
    l1.mutex.RUnlock()
    
    if !exists {
        return nil, false
    }
    
    // Check expiration
    if time.Now().After(item.ExpiresAt) {
        l1.mutex.Lock()
        delete(l1.data, key)
        l1.mutex.Unlock()
        return nil, false
    }
    
    // Update access info
    l1.mutex.Lock()
    item.AccessCount++
    item.LastAccess = time.Now()
    l1.mutex.Unlock()
    
    return item.Value, true
}

func (l1 *L1Cache) Set(key string, value interface{}, ttl time.Duration) {
    l1.mutex.Lock()
    defer l1.mutex.Unlock()
    
    // Check if we need to evict
    if len(l1.data) >= l1.maxSize {
        l1.evictLRU()
    }
    
    l1.data[key] = &CacheItem{
        Value:      value,
        ExpiresAt:  time.Now().Add(ttl),
        AccessCount: 1,
        LastAccess: time.Now(),
    }
}

func (l1 *L1Cache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, item := range l1.data {
        if oldestKey == "" || item.LastAccess.Before(oldestTime) {
            oldestKey = key
            oldestTime = item.LastAccess
        }
    }
    
    if oldestKey != "" {
        delete(l1.data, oldestKey)
    }
}

func NewL2Cache(prefix string) *L2Cache {
    client := redis.NewClient(&redis.Options{
        Addr: "localhost:6379",
        DB:   0,
    })
    
    return &L2Cache{
        client: client,
        prefix: prefix,
    }
}

func (l2 *L2Cache) Get(key string) (interface{}, error) {
    val, err := l2.client.Get(context.Background(), l2.prefix+key).Result()
    if err != nil {
        return nil, err
    }
    
    var result interface{}
    err = json.Unmarshal([]byte(val), &result)
    return result, err
}

func (l2 *L2Cache) Set(key string, value interface{}, ttl time.Duration) error {
    data, err := json.Marshal(value)
    if err != nil {
        return err
    }
    
    return l2.client.Set(context.Background(), l2.prefix+key, data, ttl).Err()
}

func NewL3Cache(prefix string, cluster bool) *L3Cache {
    var client *redis.Client
    
    if cluster {
        client = redis.NewClusterClient(&redis.ClusterOptions{
            Addrs: []string{"localhost:7000", "localhost:7001", "localhost:7002"},
        })
    } else {
        client = redis.NewClient(&redis.Options{
            Addr: "localhost:6379",
            DB:   1,
        })
    }
    
    return &L3Cache{
        client: client,
        prefix: prefix,
        cluster: cluster,
    }
}

func (l3 *L3Cache) Get(key string) (interface{}, error) {
    val, err := l3.client.Get(context.Background(), l3.prefix+key).Result()
    if err != nil {
        return nil, err
    }
    
    var result interface{}
    err = json.Unmarshal([]byte(val), &result)
    return result, err
}

func (l3 *L3Cache) Set(key string, value interface{}, ttl time.Duration) error {
    data, err := json.Marshal(value)
    if err != nil {
        return err
    }
    
    return l3.client.Set(context.Background(), l3.prefix+key, data, ttl).Err()
}

// Multi-level cache operations
func (mlc *MultiLevelCache) Get(key string) (interface{}, error) {
    // Try L1 cache first
    if value, found := mlc.l1Cache.Get(key); found {
        return value, nil
    }
    
    // Try L2 cache
    if value, err := mlc.l2Cache.Get(key); err == nil {
        // Populate L1 cache
        mlc.l1Cache.Set(key, value, 5*time.Minute)
        return value, nil
    }
    
    // Try L3 cache
    if value, err := mlc.l3Cache.Get(key); err == nil {
        // Populate L1 and L2 caches
        mlc.l1Cache.Set(key, value, 5*time.Minute)
        mlc.l2Cache.Set(key, value, 30*time.Minute)
        return value, nil
    }
    
    return nil, fmt.Errorf("key not found in any cache level")
}

func (mlc *MultiLevelCache) Set(key string, value interface{}, ttl time.Duration) error {
    // Set in all cache levels
    mlc.l1Cache.Set(key, value, ttl)
    mlc.l2Cache.Set(key, value, ttl*2)
    mlc.l3Cache.Set(key, value, ttl*4)
    
    return nil
}

// Cache warming
func (mlc *MultiLevelCache) WarmCache(keys []string, fetcher func(string) (interface{}, error)) error {
    var wg sync.WaitGroup
    semaphore := make(chan struct{}, 10) // Limit concurrent operations
    
    for _, key := range keys {
        wg.Add(1)
        go func(k string) {
            defer wg.Done()
            
            semaphore <- struct{}{} // Acquire semaphore
            defer func() { <-semaphore }() // Release semaphore
            
            // Check if already in cache
            if _, err := mlc.Get(k); err == nil {
                return
            }
            
            // Fetch from source
            value, err := fetcher(k)
            if err != nil {
                return
            }
            
            // Set in cache
            mlc.Set(k, value, time.Hour)
        }(key)
    }
    
    wg.Wait()
    return nil
}

// Example usage
func main() {
    cache := NewMultiLevelCache()
    
    // Set a value
    err := cache.Set("user:1", map[string]string{
        "name":  "John Doe",
        "email": "john@example.com",
    }, time.Hour)
    
    if err != nil {
        log.Fatal(err)
    }
    
    // Get a value
    value, err := cache.Get("user:1")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Retrieved value: %+v\n", value)
    
    // Cache warming example
    keys := []string{"user:1", "user:2", "user:3"}
    fetcher := func(key string) (interface{}, error) {
        // Simulate fetching from database
        time.Sleep(100 * time.Millisecond)
        return map[string]string{
            "name":  "User " + key,
            "email": "user@example.com",
        }, nil
    }
    
    err = cache.WarmCache(keys, fetcher)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Cache warming completed")
}
```

---

## 🎯 **5. Monitoring and Observability**

### **Performance Metrics Collection**

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
)

type MetricsCollector struct {
    counters   map[string]int64
    gauges     map[string]float64
    histograms map[string]*Histogram
    mutex      sync.RWMutex
}

type Histogram struct {
    buckets map[string]int64
    count   int64
    sum     float64
}

func NewMetricsCollector() *MetricsCollector {
    return &MetricsCollector{
        counters:   make(map[string]int64),
        gauges:     make(map[string]float64),
        histograms: make(map[string]*Histogram),
    }
}

func (mc *MetricsCollector) IncrementCounter(name string, value int64) {
    mc.mutex.Lock()
    defer mc.mutex.Unlock()
    mc.counters[name] += value
}

func (mc *MetricsCollector) SetGauge(name string, value float64) {
    mc.mutex.Lock()
    defer mc.mutex.Unlock()
    mc.gauges[name] = value
}

func (mc *MetricsCollector) RecordHistogram(name string, value float64) {
    mc.mutex.Lock()
    defer mc.mutex.Unlock()
    
    if mc.histograms[name] == nil {
        mc.histograms[name] = &Histogram{
            buckets: make(map[string]int64),
        }
    }
    
    hist := mc.histograms[name]
    hist.count++
    hist.sum += value
    
    // Simple bucket logic
    switch {
    case value < 0.1:
        hist.buckets["0.1"]++
    case value < 0.5:
        hist.buckets["0.5"]++
    case value < 1.0:
        hist.buckets["1.0"]++
    case value < 5.0:
        hist.buckets["5.0"]++
    default:
        hist.buckets["+Inf"]++
    }
}

func (mc *MetricsCollector) GetMetrics() map[string]interface{} {
    mc.mutex.RLock()
    defer mc.mutex.RUnlock()
    
    metrics := make(map[string]interface{})
    
    // Counters
    counters := make(map[string]int64)
    for name, value := range mc.counters {
        counters[name] = value
    }
    metrics["counters"] = counters
    
    // Gauges
    gauges := make(map[string]float64)
    for name, value := range mc.gauges {
        gauges[name] = value
    }
    metrics["gauges"] = gauges
    
    // Histograms
    histograms := make(map[string]interface{})
    for name, hist := range mc.histograms {
        histData := map[string]interface{}{
            "count": hist.count,
            "sum":   hist.sum,
            "buckets": hist.buckets,
        }
        histograms[name] = histData
    }
    metrics["histograms"] = histograms
    
    return metrics
}

// Performance monitoring middleware
func PerformanceMiddleware(collector *MetricsCollector) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            start := time.Now()
            
            // Wrap response writer to capture status code
            wrapped := &responseWriter{ResponseWriter: w, statusCode: 200}
            
            next.ServeHTTP(wrapped, r)
            
            duration := time.Since(start)
            
            // Record metrics
            collector.IncrementCounter("http_requests_total", 1)
            collector.RecordHistogram("http_request_duration_seconds", duration.Seconds())
            collector.SetGauge("http_requests_in_flight", 1)
            
            // Record by status code
            statusCounter := fmt.Sprintf("http_requests_total{status=\"%d\"}", wrapped.statusCode)
            collector.IncrementCounter(statusCounter, 1)
        })
    }
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}

// Health check endpoint
func HealthCheckHandler(collector *MetricsCollector) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        metrics := collector.GetMetrics()
        
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]interface{}{
            "status": "healthy",
            "timestamp": time.Now().Unix(),
            "metrics": metrics,
        })
    }
}

// Example usage
func main() {
    collector := NewMetricsCollector()
    
    // Simulate some metrics
    collector.IncrementCounter("api_calls", 1)
    collector.SetGauge("memory_usage", 1024.5)
    collector.RecordHistogram("response_time", 0.5)
    
    // Get metrics
    metrics := collector.GetMetrics()
    fmt.Printf("Metrics: %+v\n", metrics)
    
    // HTTP server with performance monitoring
    mux := http.NewServeMux()
    mux.HandleFunc("/health", HealthCheckHandler(collector))
    mux.HandleFunc("/api", func(w http.ResponseWriter, r *http.Request) {
        time.Sleep(100 * time.Millisecond) // Simulate work
        w.Write([]byte("Hello, World!"))
    })
    
    // Add performance middleware
    handler := PerformanceMiddleware(collector)(mux)
    
    // Start server
    go func() {
        log.Fatal(http.ListenAndServe(":8080", handler))
    }()
    
    // Keep running
    select {}
}
```

---

## 🎯 **Key Takeaways**

### **1. Go Runtime Optimization**
- **Memory pools** for object reuse
- **String interning** for memory efficiency
- **GC tuning** based on heap size
- **Lock-free data structures** for high concurrency

### **2. Database Performance**
- **Connection pooling** for better resource utilization
- **Query optimization** with proper indexing
- **Batch operations** for better throughput
- **Performance monitoring** for slow query detection

### **3. Network Optimization**
- **HTTP/2** for better multiplexing
- **Connection pooling** for reduced latency
- **Concurrent requests** with proper limits
- **Request batching** for efficiency

### **4. Caching Strategies**
- **Multi-level caching** for different access patterns
- **Cache warming** for better hit rates
- **Eviction policies** (LRU, LFU) for memory management
- **Cache invalidation** strategies

### **5. Monitoring and Observability**
- **Metrics collection** for performance tracking
- **Health checks** for system status
- **Performance middleware** for request monitoring
- **Real-time monitoring** for proactive optimization

---

**🎉 This comprehensive guide provides deep insights into performance engineering with practical Go implementations! 🚀**
