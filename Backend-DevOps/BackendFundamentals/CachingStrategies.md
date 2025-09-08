# 🚀 Caching Strategies: Redis, Memcached, CDN, and Cache Patterns

> **Master caching techniques for high-performance backend systems**

## 📚 Concept

Caching is the process of storing frequently accessed data in fast storage to reduce latency and improve performance. It's crucial for scalable backend systems.

### Cache Types

- **Application Cache**: In-memory caching (Redis, Memcached)
- **Database Cache**: Query result caching
- **CDN Cache**: Static content delivery
- **Browser Cache**: Client-side caching

## 🛠️ Hands-on Example

### Redis Implementation (Go)

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"

    "github.com/go-redis/redis/v8"
)

type CacheService struct {
    redis *redis.Client
}

func NewCacheService() *CacheService {
    rdb := redis.NewClient(&redis.Options{
        Addr:     "localhost:6379",
        Password: "",
        DB:       0,
    })

    return &CacheService{redis: rdb}
}

func (cs *CacheService) Set(key string, value interface{}, expiration time.Duration) error {
    jsonValue, err := json.Marshal(value)
    if err != nil {
        return err
    }

    return cs.redis.Set(context.Background(), key, jsonValue, expiration).Err()
}

func (cs *CacheService) Get(key string, dest interface{}) error {
    val, err := cs.redis.Get(context.Background(), key).Result()
    if err != nil {
        return err
    }

    return json.Unmarshal([]byte(val), dest)
}

func (cs *CacheService) Delete(key string) error {
    return cs.redis.Del(context.Background(), key).Err()
}

// Cache-aside pattern
func (cs *CacheService) GetUser(userID int) (*User, error) {
    cacheKey := fmt.Sprintf("user:%d", userID)
    var user User

    // Try cache first
    if err := cs.Get(cacheKey, &user); err == nil {
        return &user, nil
    }

    // Cache miss - fetch from database
    user, err := cs.fetchUserFromDB(userID)
    if err != nil {
        return nil, err
    }

    // Store in cache
    cs.Set(cacheKey, user, time.Hour)

    return &user, nil
}

// Write-through pattern
func (cs *CacheService) UpdateUser(userID int, updates map[string]interface{}) error {
    // Update database first
    if err := cs.updateUserInDB(userID, updates); err != nil {
        return err
    }

    // Update cache
    cacheKey := fmt.Sprintf("user:%d", userID)
    user, err := cs.fetchUserFromDB(userID)
    if err != nil {
        return err
    }

    cs.Set(cacheKey, user, time.Hour)
    return nil
}

// Write-behind pattern
func (cs *CacheService) UpdateUserAsync(userID int, updates map[string]interface{}) error {
    cacheKey := fmt.Sprintf("user:%d", userID)

    // Update cache immediately
    user, err := cs.fetchUserFromDB(userID)
    if err != nil {
        return err
    }

    // Apply updates to cached user
    for key, value := range updates {
        switch key {
        case "name":
            user.Name = value.(string)
        case "email":
            user.Email = value.(string)
        }
    }

    cs.Set(cacheKey, user, time.Hour)

    // Queue for database update
    go cs.updateUserInDB(userID, updates)

    return nil
}
```

### Cache Patterns

```go
// 1. Cache-Aside Pattern
func (cs *CacheService) GetProduct(productID int) (*Product, error) {
    cacheKey := fmt.Sprintf("product:%d", productID)
    var product Product

    // Check cache
    if err := cs.Get(cacheKey, &product); err == nil {
        return &product, nil
    }

    // Cache miss - fetch from database
    product, err := cs.fetchProductFromDB(productID)
    if err != nil {
        return nil, err
    }

    // Store in cache
    cs.Set(cacheKey, product, time.Hour)
    return &product, nil
}

// 2. Write-Through Pattern
func (cs *CacheService) CreateProduct(product *Product) error {
    // Write to database first
    if err := cs.createProductInDB(product); err != nil {
        return err
    }

    // Write to cache
    cacheKey := fmt.Sprintf("product:%d", product.ID)
    cs.Set(cacheKey, product, time.Hour)

    return nil
}

// 3. Write-Behind Pattern
func (cs *CacheService) UpdateProductAsync(productID int, updates map[string]interface{}) error {
    cacheKey := fmt.Sprintf("product:%d", productID)

    // Update cache immediately
    product, err := cs.GetProduct(productID)
    if err != nil {
        return err
    }

    // Apply updates
    for key, value := range updates {
        switch key {
        case "name":
            product.Name = value.(string)
        case "price":
            product.Price = value.(float64)
        }
    }

    cs.Set(cacheKey, product, time.Hour)

    // Queue database update
    go cs.updateProductInDB(productID, updates)

    return nil
}

// 4. Refresh-Ahead Pattern
func (cs *CacheService) GetProductWithRefresh(productID int) (*Product, error) {
    cacheKey := fmt.Sprintf("product:%d", productID)
    var product Product

    // Check cache
    if err := cs.Get(cacheKey, &product); err == nil {
        // Check if cache is about to expire
        ttl, _ := cs.redis.TTL(context.Background(), cacheKey).Result()
        if ttl < time.Minute*5 { // Refresh if less than 5 minutes left
            go cs.refreshProductCache(productID)
        }
        return &product, nil
    }

    // Cache miss - fetch from database
    product, err := cs.fetchProductFromDB(productID)
    if err != nil {
        return nil, err
    }

    // Store in cache
    cs.Set(cacheKey, product, time.Hour)
    return &product, nil
}

func (cs *CacheService) refreshProductCache(productID int) {
    product, err := cs.fetchProductFromDB(productID)
    if err != nil {
        return
    }

    cacheKey := fmt.Sprintf("product:%d", productID)
    cs.Set(cacheKey, product, time.Hour)
}
```

## 🚀 Best Practices

### 1. Cache Invalidation

```go
func (cs *CacheService) InvalidateUserCache(userID int) {
    // Invalidate user cache
    cs.Delete(fmt.Sprintf("user:%d", userID))

    // Invalidate related caches
    cs.Delete(fmt.Sprintf("user:%d:posts", userID))
    cs.Delete(fmt.Sprintf("user:%d:friends", userID))

    // Invalidate pattern-based caches
    keys, _ := cs.redis.Keys(context.Background(), fmt.Sprintf("user:%d:*", userID)).Result()
    if len(keys) > 0 {
        cs.redis.Del(context.Background(), keys...)
    }
}
```

### 2. Cache Warming

```go
func (cs *CacheService) WarmCache() {
    // Warm frequently accessed data
    go cs.warmUserCache()
    go cs.warmProductCache()
    go cs.warmCategoryCache()
}

func (cs *CacheService) warmUserCache() {
    users := cs.fetchActiveUsersFromDB()
    for _, user := range users {
        cacheKey := fmt.Sprintf("user:%d", user.ID)
        cs.Set(cacheKey, user, time.Hour)
    }
}
```

### 3. Distributed Caching

```go
type DistributedCache struct {
    redis *redis.Client
    local map[string]interface{}
    mutex sync.RWMutex
}

func (dc *DistributedCache) Get(key string) (interface{}, error) {
    // Check local cache first
    dc.mutex.RLock()
    if value, exists := dc.local[key]; exists {
        dc.mutex.RUnlock()
        return value, nil
    }
    dc.mutex.RUnlock()

    // Check Redis
    val, err := dc.redis.Get(context.Background(), key).Result()
    if err != nil {
        return nil, err
    }

    // Store in local cache
    dc.mutex.Lock()
    dc.local[key] = val
    dc.mutex.Unlock()

    return val, nil
}
```

## 🏢 Industry Insights

### Meta's Caching

- **Memcached**: Distributed caching layer
- **CDN**: Global content delivery
- **Application Cache**: In-memory caching
- **Database Cache**: Query result caching

### Google's Caching

- **Bigtable**: Distributed storage with caching
- **CDN**: Global edge caching
- **Application Cache**: Memcached clusters
- **Browser Cache**: HTTP caching headers

### Amazon's Caching

- **ElastiCache**: Managed Redis/Memcached
- **CloudFront**: Global CDN
- **Application Cache**: In-memory caching
- **Database Cache**: RDS query caching

## 🎯 Interview Questions

### Basic Level

1. **What are the main cache patterns?**

   - Cache-aside: Application manages cache
   - Write-through: Write to cache and database
   - Write-behind: Write to cache, async to database
   - Refresh-ahead: Proactive cache refresh

2. **What's the difference between Redis and Memcached?**

   - Redis: Persistent, data structures, pub/sub
   - Memcached: Simple key-value, in-memory only

3. **How do you handle cache invalidation?**
   - Time-based expiration
   - Event-based invalidation
   - Pattern-based invalidation
   - Version-based invalidation

### Intermediate Level

4. **How do you implement cache warming?**

   ```go
   func (cs *CacheService) WarmCache() {
       // Preload frequently accessed data
       go cs.warmUserCache()
       go cs.warmProductCache()
   }
   ```

5. **How do you handle cache consistency?**

   - Write-through for strong consistency
   - Event-driven invalidation
   - Version-based caching
   - Distributed locks for updates

6. **Explain cache eviction policies?**
   - LRU: Least Recently Used
   - LFU: Least Frequently Used
   - TTL: Time To Live
   - Random: Random eviction

### Advanced Level

7. **How do you implement distributed caching?**

   - Consistent hashing for sharding
   - Replication for high availability
   - Cache coherence protocols
   - Load balancing across cache nodes

8. **How do you handle cache stampede?**

   - Lock-based approach
   - Probabilistic early expiration
   - Background refresh
   - Circuit breaker pattern

9. **How do you implement cache compression?**
   - Gzip compression for large values
   - Binary serialization
   - Delta compression
   - Dictionary-based compression

---

**Next**: [Databases Integration](./DatabasesIntegration.md) - SQL vs NoSQL, connection pooling, migrations
