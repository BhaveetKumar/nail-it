# 🗄️ Database Optimization Comprehensive Guide

> **Complete guide to database performance optimization, scaling, and best practices**

## 📚 Table of Contents

1. [Query Optimization](#-query-optimization)
2. [Indexing Strategies](#-indexing-strategies)
3. [Connection Pooling](#-connection-pooling)
4. [Database Sharding](#-database-sharding)
5. [Caching Strategies](#-caching-strategies)
6. [Performance Monitoring](#-performance-monitoring)
7. [NoSQL Optimization](#-nosql-optimization)
8. [Real-world Case Studies](#-real-world-case-studies)

---

## 🔍 Query Optimization

### SQL Query Optimization

```sql
-- Index optimization examples
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_user_id ON orders(user_id);
CREATE INDEX idx_order_created_at ON orders(created_at);

-- Composite indexes for complex queries
CREATE INDEX idx_user_status_created ON users(status, created_at);

-- Partial indexes for filtered queries
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

### Query Analysis Tools

```go
// Go database query analysis
package main

import (
    "database/sql"
    "fmt"
    "log"
    _ "github.com/lib/pq"
)

type QueryAnalyzer struct {
    db *sql.DB
}

func NewQueryAnalyzer(db *sql.DB) *QueryAnalyzer {
    return &QueryAnalyzer{db: db}
}

func (qa *QueryAnalyzer) AnalyzeQuery(query string) (*QueryPlan, error) {
    // Enable query analysis
    _, err := qa.db.Exec("SET enable_explain = true")
    if err != nil {
        return nil, err
    }
    
    // Get query plan
    rows, err := qa.db.Query("EXPLAIN ANALYZE " + query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var plan QueryPlan
    for rows.Next() {
        var line string
        if err := rows.Scan(&line); err != nil {
            return nil, err
        }
        plan.Lines = append(plan.Lines, line)
    }
    
    return &plan, nil
}

type QueryPlan struct {
    Lines []string
}

func (qp *QueryPlan) HasIndexScan() bool {
    for _, line := range qp.Lines {
        if contains(line, "Index Scan") {
            return true
        }
    }
    return false
}

func (qp *QueryPlan) HasSeqScan() bool {
    for _, line := range qp.Lines {
        if contains(line, "Seq Scan") {
            return true
        }
    }
    return false
}

func contains(s, substr string) bool {
    return len(s) >= len(substr) && s[:len(substr)] == substr
}
```

---

## 📊 Indexing Strategies

### Index Types and Usage

```sql
-- B-tree indexes (default)
CREATE INDEX idx_user_email ON users(email);

-- Hash indexes for equality lookups
CREATE INDEX idx_user_id_hash ON users USING hash(id);

-- GIN indexes for array and JSON data
CREATE INDEX idx_user_tags_gin ON users USING gin(tags);
CREATE INDEX idx_user_metadata_gin ON users USING gin(metadata);

-- GiST indexes for geometric data
CREATE INDEX idx_location_gist ON users USING gist(location);

-- Partial indexes
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- Expression indexes
CREATE INDEX idx_user_name_lower ON users(lower(name));

-- Covering indexes (include columns)
CREATE INDEX idx_user_covering ON users(id) INCLUDE (name, email);
```

### Index Monitoring

```go
// Index usage monitoring
package main

import (
    "database/sql"
    "fmt"
)

type IndexMonitor struct {
    db *sql.DB
}

func (im *IndexMonitor) GetIndexUsage() ([]IndexUsage, error) {
    query := `
        SELECT 
            schemaname,
            tablename,
            indexname,
            idx_scan,
            idx_tup_read,
            idx_tup_fetch
        FROM pg_stat_user_indexes
        ORDER BY idx_scan DESC
    `
    
    rows, err := im.db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var usages []IndexUsage
    for rows.Next() {
        var usage IndexUsage
        err := rows.Scan(
            &usage.SchemaName,
            &usage.TableName,
            &usage.IndexName,
            &usage.ScanCount,
            &usage.TupleRead,
            &usage.TupleFetch,
        )
        if err != nil {
            return nil, err
        }
        usages = append(usages, usage)
    }
    
    return usages, nil
}

type IndexUsage struct {
    SchemaName  string
    TableName   string
    IndexName   string
    ScanCount   int64
    TupleRead   int64
    TupleFetch  int64
}

func (im *IndexMonitor) FindUnusedIndexes() ([]string, error) {
    query := `
        SELECT indexname
        FROM pg_stat_user_indexes
        WHERE idx_scan = 0
        AND indexname NOT LIKE '%_pkey'
    `
    
    rows, err := im.db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var unused []string
    for rows.Next() {
        var indexName string
        if err := rows.Scan(&indexName); err != nil {
            return nil, err
        }
        unused = append(unused, indexName)
    }
    
    return unused, nil
}
```

---

## 🔗 Connection Pooling

### Go Connection Pool Implementation

```go
// Advanced connection pool
package main

import (
    "database/sql"
    "sync"
    "time"
    _ "github.com/lib/pq"
)

type ConnectionPool struct {
    db           *sql.DB
    maxOpen      int
    maxIdle      int
    maxLifetime  time.Duration
    maxIdleTime  time.Duration
    mutex        sync.RWMutex
    stats        PoolStats
}

type PoolStats struct {
    OpenConnections int
    InUse          int
    Idle           int
    WaitCount      int64
    WaitDuration   time.Duration
}

func NewConnectionPool(dsn string, maxOpen, maxIdle int, maxLifetime, maxIdleTime time.Duration) (*ConnectionPool, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(maxOpen)
    db.SetMaxIdleConns(maxIdle)
    db.SetConnMaxLifetime(maxLifetime)
    db.SetConnMaxIdleTime(maxIdleTime)
    
    // Test connection
    if err := db.Ping(); err != nil {
        return nil, err
    }
    
    return &ConnectionPool{
        db:          db,
        maxOpen:     maxOpen,
        maxIdle:     maxIdle,
        maxLifetime: maxLifetime,
        maxIdleTime: maxIdleTime,
    }, nil
}

func (cp *ConnectionPool) GetStats() PoolStats {
    cp.mutex.RLock()
    defer cp.mutex.RUnlock()
    
    stats := cp.db.Stats()
    return PoolStats{
        OpenConnections: stats.OpenConnections,
        InUse:          stats.InUse,
        Idle:           stats.Idle,
        WaitCount:      stats.WaitCount,
        WaitDuration:   stats.WaitDuration,
    }
}

func (cp *ConnectionPool) Query(query string, args ...interface{}) (*sql.Rows, error) {
    return cp.db.Query(query, args...)
}

func (cp *ConnectionPool) QueryRow(query string, args ...interface{}) *sql.Row {
    return cp.db.QueryRow(query, args...)
}

func (cp *ConnectionPool) Exec(query string, args ...interface{}) (sql.Result, error) {
    return cp.db.Exec(query, args...)
}

func (cp *ConnectionPool) Close() error {
    return cp.db.Close()
}

// Connection pool with health checks
type HealthyConnectionPool struct {
    *ConnectionPool
    healthCheckInterval time.Duration
    stopChan           chan struct{}
}

func NewHealthyConnectionPool(dsn string, maxOpen, maxIdle int, maxLifetime, maxIdleTime, healthCheckInterval time.Duration) (*HealthyConnectionPool, error) {
    pool, err := NewConnectionPool(dsn, maxOpen, maxIdle, maxLifetime, maxIdleTime)
    if err != nil {
        return nil, err
    }
    
    hcp := &HealthyConnectionPool{
        ConnectionPool:     pool,
        healthCheckInterval: healthCheckInterval,
        stopChan:           make(chan struct{}),
    }
    
    go hcp.healthCheck()
    return hcp, nil
}

func (hcp *HealthyConnectionPool) healthCheck() {
    ticker := time.NewTicker(hcp.healthCheckInterval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            if err := hcp.db.Ping(); err != nil {
                log.Printf("Database health check failed: %v", err)
            }
        case <-hcp.stopChan:
            return
        }
    }
}

func (hcp *HealthyConnectionPool) Stop() {
    close(hcp.stopChan)
    hcp.Close()
}
```

---

## 🔀 Database Sharding

### Horizontal Sharding Implementation

```go
// Database sharding with consistent hashing
package main

import (
    "crypto/md5"
    "database/sql"
    "fmt"
    "sort"
    "strconv"
)

type ShardConfig struct {
    Host     string
    Port     int
    Database string
    Weight   int
}

type ShardNode struct {
    ShardConfig
    Hash uint32
}

type ShardedDatabase struct {
    nodes    []ShardNode
    databases map[string]*sql.DB
    replicas  int
}

func NewShardedDatabase(configs []ShardConfig, replicas int) (*ShardedDatabase, error) {
    var nodes []ShardNode
    databases := make(map[string]*sql.DB)
    
    for _, config := range configs {
        // Create database connection
        dsn := fmt.Sprintf("host=%s port=%d user=admin password=password dbname=%s sslmode=disable",
            config.Host, config.Port, config.Database)
        
        db, err := sql.Open("postgres", dsn)
        if err != nil {
            return nil, err
        }
        
        // Create multiple virtual nodes for consistent hashing
        for i := 0; i < replicas; i++ {
            node := ShardNode{
                ShardConfig: config,
                Hash:        hashKey(fmt.Sprintf("%s:%d:%d", config.Host, config.Port, i)),
            }
            nodes = append(nodes, node)
        }
        
        databases[fmt.Sprintf("%s:%d", config.Host, config.Port)] = db
    }
    
    // Sort nodes by hash for binary search
    sort.Slice(nodes, func(i, j int) bool {
        return nodes[i].Hash < nodes[j].Hash
    })
    
    return &ShardedDatabase{
        nodes:     nodes,
        databases: databases,
        replicas:  replicas,
    }, nil
}

func (sd *ShardedDatabase) GetShard(key string) (*sql.DB, error) {
    hash := hashKey(key)
    
    // Binary search for the first node with hash >= key hash
    idx := sort.Search(len(sd.nodes), func(i int) bool {
        return sd.nodes[i].Hash >= hash
    })
    
    // Wrap around if we're at the end
    if idx == len(sd.nodes) {
        idx = 0
    }
    
    node := sd.nodes[idx]
    dbKey := fmt.Sprintf("%s:%d", node.Host, node.Port)
    
    if db, exists := sd.databases[dbKey]; exists {
        return db, nil
    }
    
    return nil, fmt.Errorf("shard not found for key: %s", key)
}

func (sd *ShardedDatabase) Query(key string, query string, args ...interface{}) (*sql.Rows, error) {
    shard, err := sd.GetShard(key)
    if err != nil {
        return nil, err
    }
    
    return shard.Query(query, args...)
}

func (sd *ShardedDatabase) QueryRow(key string, query string, args ...interface{}) *sql.Row {
    shard, err := sd.GetShard(key)
    if err != nil {
        return nil
    }
    
    return shard.QueryRow(query, args...)
}

func (sd *ShardedDatabase) Exec(key string, query string, args ...interface{}) (sql.Result, error) {
    shard, err := sd.GetShard(key)
    if err != nil {
        return nil, err
    }
    
    return shard.Exec(query, args...)
}

func hashKey(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

// Shard-aware repository
type UserRepository struct {
    shardedDB *ShardedDatabase
}

func NewUserRepository(shardedDB *ShardedDatabase) *UserRepository {
    return &UserRepository{shardedDB: shardedDB}
}

func (ur *UserRepository) GetUser(userID string) (*User, error) {
    query := "SELECT id, name, email FROM users WHERE id = $1"
    row := ur.shardedDB.QueryRow(userID, query, userID)
    
    var user User
    err := row.Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, err
    }
    
    return &user, nil
}

func (ur *UserRepository) CreateUser(user *User) error {
    query := "INSERT INTO users (id, name, email) VALUES ($1, $2, $3)"
    _, err := ur.shardedDB.Exec(user.ID, query, user.ID, user.Name, user.Email)
    return err
}

type User struct {
    ID    string
    Name  string
    Email string
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
    "encoding/json"
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
    l1Cache map[string]CacheItem
    l2Cache *redis.Client
    l3Cache *sql.DB
    ttl     time.Duration
    mutex   sync.RWMutex
}

type CacheItem struct {
    Value     interface{}
    ExpiresAt time.Time
}

func NewMultiLevelCache(redisClient *redis.Client, db *sql.DB, ttl time.Duration) *MultiLevelCache {
    return &MultiLevelCache{
        l1Cache: make(map[string]CacheItem),
        l2Cache: redisClient,
        l3Cache: db,
        ttl:     ttl,
    }
}

func (mlc *MultiLevelCache) Get(ctx context.Context, key string) (interface{}, error) {
    // Try L1 cache first
    mlc.mutex.RLock()
    if item, exists := mlc.l1Cache[key]; exists {
        if time.Now().Before(item.ExpiresAt) {
            mlc.mutex.RUnlock()
            return item.Value, nil
        }
        // Expired, remove from L1
        delete(mlc.l1Cache, key)
    }
    mlc.mutex.RUnlock()
    
    // Try L2 cache (Redis)
    value, err := mlc.l2Cache.Get(ctx, key).Result()
    if err == nil {
        var result interface{}
        if err := json.Unmarshal([]byte(value), &result); err == nil {
            // Store in L1 cache
            mlc.mutex.Lock()
            mlc.l1Cache[key] = CacheItem{
                Value:     result,
                ExpiresAt: time.Now().Add(mlc.ttl),
            }
            mlc.mutex.Unlock()
            return result, nil
        }
    }
    
    // Try L3 cache (Database)
    var result interface{}
    err = mlc.l3Cache.QueryRowContext(ctx, "SELECT data FROM cache WHERE key = $1", key).Scan(&result)
    if err == nil {
        // Store in L2 and L1 caches
        jsonData, _ := json.Marshal(result)
        mlc.l2Cache.Set(ctx, key, jsonData, mlc.ttl)
        
        mlc.mutex.Lock()
        mlc.l1Cache[key] = CacheItem{
            Value:     result,
            ExpiresAt: time.Now().Add(mlc.ttl),
        }
        mlc.mutex.Unlock()
        
        return result, nil
    }
    
    return nil, err
}

func (mlc *MultiLevelCache) Set(ctx context.Context, key string, value interface{}) error {
    // Store in all levels
    mlc.mutex.Lock()
    mlc.l1Cache[key] = CacheItem{
        Value:     value,
        ExpiresAt: time.Now().Add(mlc.ttl),
    }
    mlc.mutex.Unlock()
    
    jsonData, err := json.Marshal(value)
    if err != nil {
        return err
    }
    
    mlc.l2Cache.Set(ctx, key, jsonData, mlc.ttl)
    
    _, err = mlc.l3Cache.ExecContext(ctx, 
        "INSERT INTO cache (key, data) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET data = $2",
        key, jsonData)
    
    return err
}

func (mlc *MultiLevelCache) Delete(ctx context.Context, key string) error {
    // Remove from all levels
    mlc.mutex.Lock()
    delete(mlc.l1Cache, key)
    mlc.mutex.Unlock()
    
    mlc.l2Cache.Del(ctx, key)
    
    _, err := mlc.l3Cache.ExecContext(ctx, "DELETE FROM cache WHERE key = $1", key)
    return err
}
```

---

## 📈 Performance Monitoring

### Database Performance Metrics

```go
// Database performance monitoring
package main

import (
    "context"
    "database/sql"
    "time"
)

type DatabaseMonitor struct {
    db     *sql.DB
    logger *log.Logger
}

func NewDatabaseMonitor(db *sql.DB, logger *log.Logger) *DatabaseMonitor {
    return &DatabaseMonitor{db: db, logger: logger}
}

func (dm *DatabaseMonitor) StartMonitoring(ctx context.Context) {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            metrics := dm.collectMetrics()
            dm.logger.Printf("Database metrics: %+v", metrics)
            
            // Alert on high metrics
            if metrics.SlowQueries > 10 {
                dm.logger.Printf("WARNING: High number of slow queries: %d", metrics.SlowQueries)
            }
            
            if metrics.ActiveConnections > 80 {
                dm.logger.Printf("WARNING: High number of active connections: %d", metrics.ActiveConnections)
            }
        }
    }
}

func (dm *DatabaseMonitor) collectMetrics() DatabaseMetrics {
    var metrics DatabaseMetrics
    
    // Get connection stats
    stats := dm.db.Stats()
    metrics.OpenConnections = stats.OpenConnections
    metrics.InUse = stats.InUse
    metrics.Idle = stats.Idle
    metrics.WaitCount = stats.WaitCount
    metrics.WaitDuration = stats.WaitDuration
    
    // Get database-specific metrics
    dm.getSlowQueries(&metrics)
    dm.getLockWaits(&metrics)
    dm.getCacheHitRatio(&metrics)
    
    return metrics
}

func (dm *DatabaseMonitor) getSlowQueries(metrics *DatabaseMetrics) {
    query := `
        SELECT COUNT(*) 
        FROM pg_stat_statements 
        WHERE mean_exec_time > 1000
    `
    
    row := dm.db.QueryRow(query)
    row.Scan(&metrics.SlowQueries)
}

func (dm *DatabaseMonitor) getLockWaits(metrics *DatabaseMetrics) {
    query := `
        SELECT COUNT(*) 
        FROM pg_locks 
        WHERE NOT granted
    `
    
    row := dm.db.QueryRow(query)
    row.Scan(&metrics.LockWaits)
}

func (dm *DatabaseMonitor) getCacheHitRatio(metrics *DatabaseMetrics) {
    query := `
        SELECT 
            round(100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2) as hit_ratio
        FROM pg_stat_database 
        WHERE datname = current_database()
    `
    
    row := dm.db.QueryRow(query)
    row.Scan(&metrics.CacheHitRatio)
}

type DatabaseMetrics struct {
    OpenConnections int
    InUse          int
    Idle           int
    WaitCount      int64
    WaitDuration   time.Duration
    SlowQueries    int
    LockWaits      int
    CacheHitRatio  float64
}
```

---

## 🎯 Real-world Case Studies

### Case Study 1: E-commerce Database Optimization

**Problem**: Product search queries taking 5-10 seconds during peak hours.

**Solution**:
1. **Index Optimization**: Added composite indexes on frequently queried columns
2. **Query Rewriting**: Replaced subqueries with JOINs
3. **Connection Pooling**: Implemented proper connection pool configuration
4. **Caching**: Added Redis caching for product data

**Results**:
- Query time reduced to 50-100ms
- 90% reduction in database load
- 50% reduction in server resources

### Case Study 2: Payment Processing Sharding

**Problem**: Single database couldn't handle payment volume during Black Friday.

**Solution**:
1. **Horizontal Sharding**: Implemented consistent hashing for user-based sharding
2. **Read Replicas**: Added read replicas for each shard
3. **Connection Pooling**: Optimized connection pool per shard
4. **Monitoring**: Added comprehensive monitoring and alerting

**Results**:
- 10x increase in transaction capacity
- 99.9% uptime during peak hours
- Linear scalability with additional shards

---

## 🎯 Best Practices Summary

### 1. Query Optimization
- Use appropriate indexes
- Avoid N+1 queries
- Use JOINs instead of subqueries
- Limit result sets with pagination

### 2. Indexing
- Create indexes on frequently queried columns
- Use composite indexes for multi-column queries
- Monitor index usage and remove unused indexes
- Consider partial indexes for filtered queries

### 3. Connection Pooling
- Configure appropriate pool sizes
- Use connection health checks
- Monitor pool statistics
- Implement proper error handling

### 4. Sharding
- Choose appropriate sharding key
- Implement consistent hashing
- Plan for data rebalancing
- Handle cross-shard queries

### 5. Caching
- Implement multi-level caching
- Use appropriate cache TTLs
- Handle cache invalidation
- Monitor cache hit ratios

---

**🗄️ Master these database optimization techniques to build fast, scalable, and efficient data systems! 🚀**


##  Nosql Optimization

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-nosql-optimization -->

Placeholder content. Please replace with proper section.
