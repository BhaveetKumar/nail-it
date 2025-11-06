---
# Auto-generated front matter
Title: Scalability Patterns Deep Dive
LastUpdated: 2025-11-06T20:45:57.729019
Tags: []
Status: draft
---

# 📈 **Scalability Patterns Deep Dive**

## 📊 **Complete Guide to Building Highly Scalable Systems**

---

## 🎯 **1. Horizontal vs Vertical Scaling**

### **Scaling Strategies**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Vertical Scaling - Scale Up
type VerticalScaler struct {
    cpuCores    int
    memoryGB    int
    diskGB      int
    maxCapacity int
    mutex       sync.RWMutex
}

func NewVerticalScaler(cpuCores, memoryGB, diskGB int) *VerticalScaler {
    return &VerticalScaler{
        cpuCores:    cpuCores,
        memoryGB:    memoryGB,
        diskGB:      diskGB,
        maxCapacity: cpuCores * 1000, // Assume 1000 requests per core
    }
}

func (vs *VerticalScaler) ScaleUp() error {
    vs.mutex.Lock()
    defer vs.mutex.Unlock()

    // Simulate scaling up
    vs.cpuCores *= 2
    vs.memoryGB *= 2
    vs.diskGB *= 2
    vs.maxCapacity = vs.cpuCores * 1000

    fmt.Printf("Scaled up to %d cores, %dGB RAM, %dGB disk\n",
        vs.cpuCores, vs.memoryGB, vs.diskGB)

    return nil
}

func (vs *VerticalScaler) GetCapacity() int {
    vs.mutex.RLock()
    defer vs.mutex.RUnlock()
    return vs.maxCapacity
}

// Horizontal Scaling - Scale Out
type HorizontalScaler struct {
    instances []*Instance
    mutex     sync.RWMutex
}

type Instance struct {
    ID       string
    CPU      int
    Memory   int
    Capacity int
    Status   string
}

func NewHorizontalScaler() *HorizontalScaler {
    return &HorizontalScaler{
        instances: make([]*Instance, 0),
    }
}

func (hs *HorizontalScaler) AddInstance(cpu, memory int) *Instance {
    hs.mutex.Lock()
    defer hs.mutex.Unlock()

    instance := &Instance{
        ID:       fmt.Sprintf("instance_%d", len(hs.instances)+1),
        CPU:      cpu,
        Memory:   memory,
        Capacity: cpu * 1000,
        Status:   "running",
    }

    hs.instances = append(hs.instances, instance)
    fmt.Printf("Added instance %s with %d cores, %dGB RAM\n",
        instance.ID, instance.CPU, instance.Memory)

    return instance
}

func (hs *HorizontalScaler) RemoveInstance(instanceID string) error {
    hs.mutex.Lock()
    defer hs.mutex.Unlock()

    for i, instance := range hs.instances {
        if instance.ID == instanceID {
            hs.instances = append(hs.instances[:i], hs.instances[i+1:]...)
            fmt.Printf("Removed instance %s\n", instanceID)
            return nil
        }
    }

    return fmt.Errorf("instance %s not found", instanceID)
}

func (hs *HorizontalScaler) GetTotalCapacity() int {
    hs.mutex.RLock()
    defer hs.mutex.RUnlock()

    total := 0
    for _, instance := range hs.instances {
        if instance.Status == "running" {
            total += instance.Capacity
        }
    }

    return total
}

func (hs *HorizontalScaler) GetInstanceCount() int {
    hs.mutex.RLock()
    defer hs.mutex.RUnlock()
    return len(hs.instances)
}

// Auto-scaling based on metrics
type AutoScaler struct {
    horizontalScaler *HorizontalScaler
    verticalScaler   *VerticalScaler
    metrics          *Metrics
    config           *AutoScalingConfig
    mutex            sync.RWMutex
}

type AutoScalingConfig struct {
    MinInstances     int
    MaxInstances     int
    ScaleUpThreshold float64
    ScaleDownThreshold float64
    CooldownPeriod   time.Duration
    LastScaleTime    time.Time
}

type Metrics struct {
    CPUUsage    float64
    MemoryUsage float64
    RequestRate int
    ErrorRate   float64
    mutex       sync.RWMutex
}

func NewAutoScaler(config *AutoScalingConfig) *AutoScaler {
    return &AutoScaler{
        horizontalScaler: NewHorizontalScaler(),
        verticalScaler:   NewVerticalScaler(4, 8, 100),
        metrics:          &Metrics{},
        config:           config,
    }
}

func (as *AutoScaler) UpdateMetrics(cpu, memory float64, requestRate int, errorRate float64) {
    as.metrics.mutex.Lock()
    defer as.metrics.mutex.Unlock()

    as.metrics.CPUUsage = cpu
    as.metrics.MemoryUsage = memory
    as.metrics.RequestRate = requestRate
    as.metrics.ErrorRate = errorRate
}

func (as *AutoScaler) CheckScaling() error {
    as.mutex.Lock()
    defer as.mutex.Unlock()

    // Check cooldown period
    if time.Since(as.config.LastScaleTime) < as.config.CooldownPeriod {
        return nil
    }

    as.metrics.mutex.RLock()
    cpuUsage := as.metrics.CPUUsage
    memoryUsage := as.metrics.MemoryUsage
    requestRate := as.metrics.RequestRate
    errorRate := as.metrics.ErrorRate
    as.metrics.mutex.RUnlock()

    instanceCount := as.horizontalScaler.GetInstanceCount()

    // Scale up conditions
    if (cpuUsage > as.config.ScaleUpThreshold ||
        memoryUsage > as.config.ScaleUpThreshold ||
        requestRate > as.getCurrentCapacity()*0.8) &&
        instanceCount < as.config.MaxInstances {

        return as.scaleUp()
    }

    // Scale down conditions
    if (cpuUsage < as.config.ScaleDownThreshold &&
        memoryUsage < as.config.ScaleDownThreshold &&
        requestRate < as.getCurrentCapacity()*0.3) &&
        instanceCount > as.config.MinInstances {

        return as.scaleDown()
    }

    return nil
}

func (as *AutoScaler) scaleUp() error {
    // Add new instance
    as.horizontalScaler.AddInstance(4, 8)
    as.config.LastScaleTime = time.Now()

    fmt.Println("Scaled up: Added new instance")
    return nil
}

func (as *AutoScaler) scaleDown() error {
    // Remove least loaded instance
    instanceCount := as.horizontalScaler.GetInstanceCount()
    if instanceCount > 0 {
        instanceID := fmt.Sprintf("instance_%d", instanceCount)
        as.horizontalScaler.RemoveInstance(instanceID)
        as.config.LastScaleTime = time.Now()

        fmt.Println("Scaled down: Removed instance")
    }

    return nil
}

func (as *AutoScaler) getCurrentCapacity() int {
    return as.horizontalScaler.GetTotalCapacity()
}

// Example usage
func main() {
    // Vertical scaling example
    verticalScaler := NewVerticalScaler(4, 8, 100)
    fmt.Printf("Initial capacity: %d\n", verticalScaler.GetCapacity())

    verticalScaler.ScaleUp()
    fmt.Printf("After scale up: %d\n", verticalScaler.GetCapacity())

    // Horizontal scaling example
    horizontalScaler := NewHorizontalScaler()
    horizontalScaler.AddInstance(4, 8)
    horizontalScaler.AddInstance(4, 8)
    horizontalScaler.AddInstance(4, 8)

    fmt.Printf("Total capacity: %d\n", horizontalScaler.GetTotalCapacity())
    fmt.Printf("Instance count: %d\n", horizontalScaler.GetInstanceCount())

    // Auto-scaling example
    config := &AutoScalingConfig{
        MinInstances:       2,
        MaxInstances:       10,
        ScaleUpThreshold:   70.0,
        ScaleDownThreshold: 30.0,
        CooldownPeriod:     5 * time.Minute,
    }

    autoScaler := NewAutoScaler(config)

    // Simulate metrics updates
    go func() {
        for {
            // Simulate high load
            autoScaler.UpdateMetrics(80.0, 75.0, 5000, 0.01)
            autoScaler.CheckScaling()

            time.Sleep(10 * time.Second)
        }
    }()

    // Keep running
    time.Sleep(60 * time.Second)
}
```

---

## 🎯 **2. Load Balancing Strategies**

### **Advanced Load Balancing**

```go
package main

import (
    "fmt"
    "math/rand"
    "sync"
    "time"
)

// Load Balancer Interface
type LoadBalancer interface {
    SelectServer(servers []*Server) *Server
    GetName() string
}

// Round Robin Load Balancer
type RoundRobinLoadBalancer struct {
    current int
    mutex   sync.Mutex
}

func NewRoundRobinLoadBalancer() *RoundRobinLoadBalancer {
    return &RoundRobinLoadBalancer{current: 0}
}

func (rr *RoundRobinLoadBalancer) SelectServer(servers []*Server) *Server {
    if len(servers) == 0 {
        return nil
    }

    rr.mutex.Lock()
    defer rr.mutex.Unlock()

    server := servers[rr.current]
    rr.current = (rr.current + 1) % len(servers)
    return server
}

func (rr *RoundRobinLoadBalancer) GetName() string {
    return "Round Robin"
}

// Least Connections Load Balancer
type LeastConnectionsLoadBalancer struct{}

func NewLeastConnectionsLoadBalancer() *LeastConnectionsLoadBalancer {
    return &LeastConnectionsLoadBalancer{}
}

func (lc *LeastConnectionsLoadBalancer) SelectServer(servers []*Server) *Server {
    if len(servers) == 0 {
        return nil
    }

    minConnections := servers[0].GetConnectionCount()
    selectedServer := servers[0]

    for _, server := range servers[1:] {
        connections := server.GetConnectionCount()
        if connections < minConnections {
            minConnections = connections
            selectedServer = server
        }
    }

    return selectedServer
}

func (lc *LeastConnectionsLoadBalancer) GetName() string {
    return "Least Connections"
}

// Weighted Round Robin Load Balancer
type WeightedRoundRobinLoadBalancer struct {
    current int
    mutex   sync.Mutex
}

func NewWeightedRoundRobinLoadBalancer() *WeightedRoundRobinLoadBalancer {
    return &WeightedRoundRobinLoadBalancer{current: 0}
}

func (wrr *WeightedRoundRobinLoadBalancer) SelectServer(servers []*Server) *Server {
    if len(servers) == 0 {
        return nil
    }

    wrr.mutex.Lock()
    defer wrr.mutex.Unlock()

    totalWeight := 0
    for _, server := range servers {
        totalWeight += server.Weight
    }

    for i := range servers {
        servers[i].CurrentWeight += servers[i].Weight
        if servers[i].CurrentWeight >= totalWeight {
            servers[i].CurrentWeight -= totalWeight
            return servers[i]
        }
    }

    return servers[0]
}

func (wrr *WeightedRoundRobinLoadBalancer) GetName() string {
    return "Weighted Round Robin"
}

// IP Hash Load Balancer
type IPHashLoadBalancer struct{}

func NewIPHashLoadBalancer() *IPHashLoadBalancer {
    return &IPHashLoadBalancer{}
}

func (ih *IPHashLoadBalancer) SelectServer(servers []*Server) *Server {
    if len(servers) == 0 {
        return nil
    }

    // Simple hash function
    hash := 0
    for _, c := range "client_ip" { // In real implementation, use actual client IP
        hash += int(c)
    }

    index := hash % len(servers)
    return servers[index]
}

func (ih *IPHashLoadBalancer) GetName() string {
    return "IP Hash"
}

// Least Response Time Load Balancer
type LeastResponseTimeLoadBalancer struct{}

func NewLeastResponseTimeLoadBalancer() *LeastResponseTimeLoadBalancer {
    return &LeastResponseTimeLoadBalancer{}
}

func (lrt *LeastResponseTimeLoadBalancer) SelectServer(servers []*Server) *Server {
    if len(servers) == 0 {
        return nil
    }

    minResponseTime := servers[0].GetAverageResponseTime()
    selectedServer := servers[0]

    for _, server := range servers[1:] {
        responseTime := server.GetAverageResponseTime()
        if responseTime < minResponseTime {
            minResponseTime = responseTime
            selectedServer = server
        }
    }

    return selectedServer
}

func (lrt *LeastResponseTimeLoadBalancer) GetName() string {
    return "Least Response Time"
}

// Server representation
type Server struct {
    ID                string
    Address           string
    Weight            int
    CurrentWeight     int
    ConnectionCount   int
    ResponseTimes     []time.Duration
    mutex             sync.RWMutex
}

func NewServer(id, address string, weight int) *Server {
    return &Server{
        ID:            id,
        Address:       address,
        Weight:        weight,
        ResponseTimes: make([]time.Duration, 0),
    }
}

func (s *Server) GetConnectionCount() int {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    return s.ConnectionCount
}

func (s *Server) IncrementConnections() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    s.ConnectionCount++
}

func (s *Server) DecrementConnections() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    s.ConnectionCount--
}

func (s *Server) RecordResponseTime(duration time.Duration) {
    s.mutex.Lock()
    defer s.mutex.Unlock()

    s.ResponseTimes = append(s.ResponseTimes, duration)

    // Keep only last 100 response times
    if len(s.ResponseTimes) > 100 {
        s.ResponseTimes = s.ResponseTimes[1:]
    }
}

func (s *Server) GetAverageResponseTime() time.Duration {
    s.mutex.RLock()
    defer s.mutex.RUnlock()

    if len(s.ResponseTimes) == 0 {
        return 0
    }

    total := time.Duration(0)
    for _, rt := range s.ResponseTimes {
        total += rt
    }

    return total / time.Duration(len(s.ResponseTimes))
}

// Load Balancer Manager
type LoadBalancerManager struct {
    balancers map[string]LoadBalancer
    servers   []*Server
    mutex     sync.RWMutex
}

func NewLoadBalancerManager() *LoadBalancerManager {
    return &LoadBalancerManager{
        balancers: make(map[string]LoadBalancer),
        servers:   make([]*Server, 0),
    }
}

func (lbm *LoadBalancerManager) AddLoadBalancer(name string, balancer LoadBalancer) {
    lbm.mutex.Lock()
    defer lbm.mutex.Unlock()

    lbm.balancers[name] = balancer
}

func (lbm *LoadBalancerManager) AddServer(server *Server) {
    lbm.mutex.Lock()
    defer lbm.mutex.Unlock()

    lbm.servers = append(lbm.servers, server)
}

func (lbm *LoadBalancerManager) SelectServer(balancerName string) *Server {
    lbm.mutex.RLock()
    defer lbm.mutex.RUnlock()

    balancer, exists := lbm.balancers[balancerName]
    if !exists {
        return nil
    }

    return balancer.SelectServer(lbm.servers)
}

func (lbm *LoadBalancerManager) GetServerCount() int {
    lbm.mutex.RLock()
    defer lbm.mutex.RUnlock()
    return len(lbm.servers)
}

// Health Check Manager
type HealthCheckManager struct {
    servers []*Server
    mutex   sync.RWMutex
}

func NewHealthCheckManager() *HealthCheckManager {
    return &HealthCheckManager{
        servers: make([]*Server, 0),
    }
}

func (hcm *HealthCheckManager) AddServer(server *Server) {
    hcm.mutex.Lock()
    defer hcm.mutex.Unlock()

    hcm.servers = append(hcm.servers, server)
}

func (hcm *HealthCheckManager) StartHealthChecks() {
    ticker := time.NewTicker(5 * time.Second)
    go func() {
        for range ticker.C {
            hcm.checkAllServers()
        }
    }()
}

func (hcm *HealthCheckManager) checkAllServers() {
    hcm.mutex.RLock()
    servers := make([]*Server, len(hcm.servers))
    copy(servers, hcm.servers)
    hcm.mutex.RUnlock()

    for _, server := range servers {
        go hcm.checkServer(server)
    }
}

func (hcm *HealthCheckManager) checkServer(server *Server) {
    // Simulate health check
    start := time.Now()

    // Simulate network call
    time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)

    duration := time.Since(start)
    server.RecordResponseTime(duration)

    // Simulate occasional failures
    if rand.Float32() < 0.1 { // 10% failure rate
        fmt.Printf("Server %s is unhealthy\n", server.ID)
    } else {
        fmt.Printf("Server %s is healthy (response time: %v)\n", server.ID, duration)
    }
}

// Example usage
func main() {
    // Create load balancer manager
    manager := NewLoadBalancerManager()

    // Add load balancers
    manager.AddLoadBalancer("round_robin", NewRoundRobinLoadBalancer())
    manager.AddLoadBalancer("least_connections", NewLeastConnectionsLoadBalancer())
    manager.AddLoadBalancer("weighted_round_robin", NewWeightedRoundRobinLoadBalancer())
    manager.AddLoadBalancer("ip_hash", NewIPHashLoadBalancer())
    manager.AddLoadBalancer("least_response_time", NewLeastResponseTimeLoadBalancer())

    // Add servers
    servers := []*Server{
        NewServer("server1", "192.168.1.1:8080", 3),
        NewServer("server2", "192.168.1.2:8080", 2),
        NewServer("server3", "192.168.1.3:8080", 1),
    }

    for _, server := range servers {
        manager.AddServer(server)
    }

    // Start health checks
    healthManager := NewHealthCheckManager()
    for _, server := range servers {
        healthManager.AddServer(server)
    }
    healthManager.StartHealthChecks()

    // Test different load balancing strategies
    balancers := []string{"round_robin", "least_connections", "weighted_round_robin", "ip_hash", "least_response_time"}

    for _, balancerName := range balancers {
        fmt.Printf("\nTesting %s:\n", balancerName)

        for i := 0; i < 10; i++ {
            server := manager.SelectServer(balancerName)
            if server != nil {
                server.IncrementConnections()
                fmt.Printf("Request %d -> Server %s\n", i+1, server.ID)

                // Simulate request processing
                time.Sleep(100 * time.Millisecond)

                server.DecrementConnections()
            }
        }
    }

    // Keep running for health checks
    time.Sleep(30 * time.Second)
}
```

---

## 🎯 **3. Caching Strategies**

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

// Cache Interface
type Cache interface {
    Get(key string) (interface{}, error)
    Set(key string, value interface{}, ttl time.Duration) error
    Delete(key string) error
    Clear() error
    GetStats() map[string]interface{}
}

// L1 Cache (In-Memory)
type L1Cache struct {
    data    map[string]*CacheItem
    mutex   sync.RWMutex
    maxSize int
    stats   *CacheStats
}

type CacheItem struct {
    Value     interface{}
    ExpiresAt time.Time
    CreatedAt time.Time
    AccessCount int
}

type CacheStats struct {
    Hits       int64
    Misses     int64
    Evictions  int64
    mutex      sync.RWMutex
}

func NewL1Cache(maxSize int) *L1Cache {
    return &L1Cache{
        data:    make(map[string]*CacheItem),
        maxSize: maxSize,
        stats:   &CacheStats{},
    }
}

func (l1 *L1Cache) Get(key string) (interface{}, error) {
    l1.mutex.RLock()
    item, exists := l1.data[key]
    l1.mutex.RUnlock()

    if !exists {
        l1.stats.incrementMisses()
        return nil, fmt.Errorf("key not found")
    }

    // Check expiration
    if time.Now().After(item.ExpiresAt) {
        l1.mutex.Lock()
        delete(l1.data, key)
        l1.mutex.Unlock()
        l1.stats.incrementMisses()
        return nil, fmt.Errorf("key expired")
    }

    // Update access count
    l1.mutex.Lock()
    item.AccessCount++
    l1.mutex.Unlock()

    l1.stats.incrementHits()
    return item.Value, nil
}

func (l1 *L1Cache) Set(key string, value interface{}, ttl time.Duration) error {
    l1.mutex.Lock()
    defer l1.mutex.Unlock()

    // Check if we need to evict
    if len(l1.data) >= l1.maxSize {
        l1.evictLRU()
    }

    l1.data[key] = &CacheItem{
        Value:      value,
        ExpiresAt:  time.Now().Add(ttl),
        CreatedAt:  time.Now(),
        AccessCount: 1,
    }

    return nil
}

func (l1 *L1Cache) Delete(key string) error {
    l1.mutex.Lock()
    defer l1.mutex.Unlock()

    delete(l1.data, key)
    return nil
}

func (l1 *L1Cache) Clear() error {
    l1.mutex.Lock()
    defer l1.mutex.Unlock()

    l1.data = make(map[string]*CacheItem)
    return nil
}

func (l1 *L1Cache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time

    for key, item := range l1.data {
        if oldestKey == "" || item.CreatedAt.Before(oldestTime) {
            oldestKey = key
            oldestTime = item.CreatedAt
        }
    }

    if oldestKey != "" {
        delete(l1.data, oldestKey)
        l1.stats.incrementEvictions()
    }
}

func (l1 *L1Cache) GetStats() map[string]interface{} {
    l1.mutex.RLock()
    defer l1.mutex.RUnlock()

    return map[string]interface{}{
        "size":      len(l1.data),
        "max_size":  l1.maxSize,
        "hits":      l1.stats.getHits(),
        "misses":    l1.stats.getMisses(),
        "evictions": l1.stats.getEvictions(),
        "hit_rate":  l1.stats.getHitRate(),
    }
}

func (cs *CacheStats) incrementHits() {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    cs.Hits++
}

func (cs *CacheStats) incrementMisses() {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    cs.Misses++
}

func (cs *CacheStats) incrementEvictions() {
    cs.mutex.Lock()
    defer cs.mutex.Unlock()
    cs.Evictions++
}

func (cs *CacheStats) getHits() int64 {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    return cs.Hits
}

func (cs *CacheStats) getMisses() int64 {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    return cs.Misses
}

func (cs *CacheStats) getEvictions() int64 {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()
    return cs.Evictions
}

func (cs *CacheStats) getHitRate() float64 {
    cs.mutex.RLock()
    defer cs.mutex.RUnlock()

    total := cs.Hits + cs.Misses
    if total == 0 {
        return 0
    }

    return float64(cs.Hits) / float64(total)
}

// L2 Cache (Redis)
type L2Cache struct {
    client *redis.Client
    stats  *CacheStats
}

func NewL2Cache() *L2Cache {
    client := redis.NewClient(&redis.Options{
        Addr: "localhost:6379",
        DB:   0,
    })

    return &L2Cache{
        client: client,
        stats:  &CacheStats{},
    }
}

func (l2 *L2Cache) Get(key string) (interface{}, error) {
    val, err := l2.client.Get(context.Background(), key).Result()
    if err != nil {
        l2.stats.incrementMisses()
        return nil, err
    }

    var result interface{}
    err = json.Unmarshal([]byte(val), &result)
    if err != nil {
        l2.stats.incrementMisses()
        return nil, err
    }

    l2.stats.incrementHits()
    return result, nil
}

func (l2 *L2Cache) Set(key string, value interface{}, ttl time.Duration) error {
    data, err := json.Marshal(value)
    if err != nil {
        return err
    }

    return l2.client.Set(context.Background(), key, data, ttl).Err()
}

func (l2 *L2Cache) Delete(key string) error {
    return l2.client.Del(context.Background(), key).Err()
}

func (l2 *L2Cache) Clear() error {
    return l2.client.FlushDB(context.Background()).Err()
}

func (l2 *L2Cache) GetStats() map[string]interface{} {
    return map[string]interface{}{
        "hits":      l2.stats.getHits(),
        "misses":    l2.stats.getMisses(),
        "hit_rate":  l2.stats.getHitRate(),
    }
}

// Multi-Level Cache
type MultiLevelCache struct {
    l1Cache Cache
    l2Cache Cache
    mutex   sync.RWMutex
}

func NewMultiLevelCache(l1Size int) *MultiLevelCache {
    return &MultiLevelCache{
        l1Cache: NewL1Cache(l1Size),
        l2Cache: NewL2Cache(),
    }
}

func (mlc *MultiLevelCache) Get(key string) (interface{}, error) {
    // Try L1 cache first
    if value, err := mlc.l1Cache.Get(key); err == nil {
        return value, nil
    }

    // Try L2 cache
    if value, err := mlc.l2Cache.Get(key); err == nil {
        // Populate L1 cache
        mlc.l1Cache.Set(key, value, 5*time.Minute)
        return value, nil
    }

    return nil, fmt.Errorf("key not found in any cache level")
}

func (mlc *MultiLevelCache) Set(key string, value interface{}, ttl time.Duration) error {
    // Set in both cache levels
    mlc.l1Cache.Set(key, value, ttl)
    mlc.l2Cache.Set(key, value, ttl*2)

    return nil
}

func (mlc *MultiLevelCache) Delete(key string) error {
    mlc.l1Cache.Delete(key)
    mlc.l2Cache.Delete(key)
    return nil
}

func (mlc *MultiLevelCache) Clear() error {
    mlc.l1Cache.Clear()
    mlc.l2Cache.Clear()
    return nil
}

func (mlc *MultiLevelCache) GetStats() map[string]interface{} {
    l1Stats := mlc.l1Cache.GetStats()
    l2Stats := mlc.l2Cache.GetStats()

    return map[string]interface{}{
        "l1_cache": l1Stats,
        "l2_cache": l2Stats,
    }
}

// Cache Warming
type CacheWarmer struct {
    cache Cache
    mutex sync.RWMutex
}

func NewCacheWarmer(cache Cache) *CacheWarmer {
    return &CacheWarmer{
        cache: cache,
    }
}

func (cw *CacheWarmer) WarmCache(keys []string, fetcher func(string) (interface{}, error)) error {
    var wg sync.WaitGroup
    semaphore := make(chan struct{}, 10) // Limit concurrent operations

    for _, key := range keys {
        wg.Add(1)
        go func(k string) {
            defer wg.Done()

            semaphore <- struct{}{} // Acquire semaphore
            defer func() { <-semaphore }() // Release semaphore

            // Check if already in cache
            if _, err := cw.cache.Get(k); err == nil {
                return
            }

            // Fetch from source
            value, err := fetcher(k)
            if err != nil {
                return
            }

            // Set in cache
            cw.cache.Set(k, value, time.Hour)
        }(key)
    }

    wg.Wait()
    return nil
}

// Example usage
func main() {
    // Create multi-level cache
    cache := NewMultiLevelCache(1000)

    // Set some values
    cache.Set("user:1", map[string]string{
        "name":  "John Doe",
        "email": "john@example.com",
    }, time.Hour)

    cache.Set("user:2", map[string]string{
        "name":  "Jane Smith",
        "email": "jane@example.com",
    }, time.Hour)

    // Get values
    if value, err := cache.Get("user:1"); err == nil {
        fmt.Printf("Retrieved: %+v\n", value)
    }

    // Get cache statistics
    stats := cache.GetStats()
    fmt.Printf("Cache stats: %+v\n", stats)

    // Cache warming example
    warmer := NewCacheWarmer(cache)

    keys := []string{"user:3", "user:4", "user:5"}
    fetcher := func(key string) (interface{}, error) {
        // Simulate fetching from database
        time.Sleep(100 * time.Millisecond)
        return map[string]string{
            "name":  "User " + key,
            "email": "user@example.com",
        }, nil
    }

    err := warmer.WarmCache(keys, fetcher)
    if err != nil {
        fmt.Printf("Cache warming failed: %v\n", err)
    } else {
        fmt.Println("Cache warming completed")
    }
}
```

---

## 🎯 **4. Database Sharding and Partitioning**

### **Advanced Sharding Strategies**

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sync"
)

// Shard Manager
type ShardManager struct {
    shards    map[string]*Shard
    shardKeys []string
    mutex     sync.RWMutex
}

type Shard struct {
    ID       string
    Database *Database
    mutex    sync.RWMutex
}

type Database struct {
    ID   string
    Data map[string]interface{}
}

func NewShardManager() *ShardManager {
    return &ShardManager{
        shards:    make(map[string]*Shard),
        shardKeys: make([]string, 0),
    }
}

func (sm *ShardManager) AddShard(shardID string) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()

    shard := &Shard{
        ID: shardID,
        Database: &Database{
            ID:   shardID,
            Data: make(map[string]interface{}),
        },
    }

    sm.shards[shardID] = shard
    sm.shardKeys = append(sm.shardKeys, shardID)
}

func (sm *ShardManager) GetShard(key string) *Shard {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()

    if len(sm.shardKeys) == 0 {
        return nil
    }

    // Use consistent hashing
    hash := sm.hash(key)
    shardIndex := hash % len(sm.shardKeys)
    shardID := sm.shardKeys[shardIndex]

    return sm.shards[shardID]
}

func (sm *ShardManager) hash(key string) int {
    h := md5.Sum([]byte(key))
    result := 0
    for _, b := range h {
        result = (result << 8) | int(b)
    }
    if result < 0 {
        result = -result
    }
    return result
}

func (sm *ShardManager) Write(key string, value interface{}) error {
    shard := sm.GetShard(key)
    if shard == nil {
        return fmt.Errorf("no shard available")
    }

    shard.mutex.Lock()
    defer shard.mutex.Unlock()

    shard.Database.Data[key] = value
    return nil
}

func (sm *ShardManager) Read(key string) (interface{}, error) {
    shard := sm.GetShard(key)
    if shard == nil {
        return nil, fmt.Errorf("no shard available")
    }

    shard.mutex.RLock()
    defer shard.mutex.RUnlock()

    value, exists := shard.Database.Data[key]
    if !exists {
        return nil, fmt.Errorf("key not found")
    }

    return value, nil
}

func (sm *ShardManager) GetShardStats() map[string]int {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()

    stats := make(map[string]int)
    for shardID, shard := range sm.shards {
        shard.mutex.RLock()
        stats[shardID] = len(shard.Database.Data)
        shard.mutex.RUnlock()
    }

    return stats
}

// Cross-Shard Query Engine
type CrossShardQueryEngine struct {
    shardManager *ShardManager
    mutex        sync.RWMutex
}

func NewCrossShardQueryEngine(shardManager *ShardManager) *CrossShardQueryEngine {
    return &CrossShardQueryEngine{
        shardManager: shardManager,
    }
}

func (cqe *CrossShardQueryEngine) ExecuteQuery(query *CrossShardQuery) ([]interface{}, error) {
    cqe.mutex.RLock()
    defer cqe.mutex.RUnlock()

    var results []interface{}
    var wg sync.WaitGroup
    var mutex sync.Mutex

    // Execute query on all shards
    for _, shardID := range cqe.shardManager.shardKeys {
        wg.Add(1)
        go func(sid string) {
            defer wg.Done()

            shard := cqe.shardManager.shards[sid]
            shardResults := cqe.executeQueryOnShard(shard, query)

            mutex.Lock()
            results = append(results, shardResults...)
            mutex.Unlock()
        }(shardID)
    }

    wg.Wait()
    return results, nil
}

func (cqe *CrossShardQueryEngine) executeQueryOnShard(shard *Shard, query *CrossShardQuery) []interface{} {
    shard.mutex.RLock()
    defer shard.mutex.RUnlock()

    var results []interface{}
    for key, value := range shard.Database.Data {
        if query.Filter == nil || query.Filter(key, value) {
            results = append(results, value)
        }
    }

    return results
}

type CrossShardQuery struct {
    Filter func(string, interface{}) bool
}

// Data Rebalancing
type DataRebalancer struct {
    shardManager *ShardManager
    mutex        sync.RWMutex
}

func NewDataRebalancer(shardManager *ShardManager) *DataRebalancer {
    return &DataRebalancer{
        shardManager: shardManager,
    }
}

func (dr *DataRebalancer) Rebalance() error {
    dr.mutex.Lock()
    defer dr.mutex.Unlock()

    // Calculate current distribution
    stats := dr.shardManager.GetShardStats()

    // Find overloaded and underloaded shards
    var overloaded, underloaded []string
    totalData := 0

    for _, count := range stats {
        totalData += count
    }

    avgData := totalData / len(stats)

    for shardID, count := range stats {
        if count > avgData*1.2 {
            overloaded = append(overloaded, shardID)
        } else if count < avgData*0.8 {
            underloaded = append(underloaded, shardID)
        }
    }

    // Move data from overloaded to underloaded shards
    for _, fromShardID := range overloaded {
        for _, toShardID := range underloaded {
            if err := dr.moveData(fromShardID, toShardID); err != nil {
                return err
            }
        }
    }

    return nil
}

func (dr *DataRebalancer) moveData(fromShardID, toShardID string) error {
    fromShard := dr.shardManager.shards[fromShardID]
    toShard := dr.shardManager.shards[toShardID]

    // Get data to move
    fromShard.mutex.Lock()
    var dataToMove []string
    for key := range fromShard.Database.Data {
        if len(dataToMove) >= 100 { // Limit batch size
            break
        }
        dataToMove = append(dataToMove, key)
    }
    fromShard.mutex.Unlock()

    // Move data
    for _, key := range dataToMove {
        fromShard.mutex.Lock()
        value, exists := fromShard.Database.Data[key]
        if exists {
            delete(fromShard.Database.Data, key)
        }
        fromShard.mutex.Unlock()

        if exists {
            toShard.mutex.Lock()
            toShard.Database.Data[key] = value
            toShard.mutex.Unlock()
        }
    }

    return nil
}

// Example usage
func main() {
    // Create shard manager
    shardManager := NewShardManager()

    // Add shards
    shardManager.AddShard("shard1")
    shardManager.AddShard("shard2")
    shardManager.AddShard("shard3")

    // Write data
    for i := 0; i < 1000; i++ {
        key := fmt.Sprintf("key_%d", i)
        value := fmt.Sprintf("value_%d", i)

        if err := shardManager.Write(key, value); err != nil {
            fmt.Printf("Failed to write %s: %v\n", key, err)
        }
    }

    // Read data
    for i := 0; i < 10; i++ {
        key := fmt.Sprintf("key_%d", i)
        if value, err := shardManager.Read(key); err == nil {
            fmt.Printf("Read %s: %v\n", key, value)
        }
    }

    // Get shard statistics
    stats := shardManager.GetShardStats()
    fmt.Printf("Shard stats: %+v\n", stats)

    // Cross-shard query
    queryEngine := NewCrossShardQueryEngine(shardManager)

    query := &CrossShardQuery{
        Filter: func(key string, value interface{}) bool {
            return len(key) > 5 // Simple filter
        },
    }

    results, err := queryEngine.ExecuteQuery(query)
    if err != nil {
        fmt.Printf("Query failed: %v\n", err)
    } else {
        fmt.Printf("Query returned %d results\n", len(results))
    }

    // Data rebalancing
    rebalancer := NewDataRebalancer(shardManager)

    if err := rebalancer.Rebalance(); err != nil {
        fmt.Printf("Rebalancing failed: %v\n", err)
    } else {
        fmt.Println("Rebalancing completed")
    }

    // Get updated statistics
    stats = shardManager.GetShardStats()
    fmt.Printf("Updated shard stats: %+v\n", stats)
}
```

---

## 🎯 **Key Takeaways**

### **1. Scaling Strategies**

- **Vertical Scaling**: Increase resources of existing instances
- **Horizontal Scaling**: Add more instances
- **Auto-scaling**: Automatically adjust based on metrics
- **Hybrid Approach**: Combine both strategies

### **2. Load Balancing**

- **Round Robin**: Simple, even distribution
- **Least Connections**: Good for long-running connections
- **Weighted**: Consider server capacity
- **Least Response Time**: Optimize for performance

### **3. Caching Strategies**

- **Multi-level caching**: L1 (memory) + L2 (Redis)
- **Cache warming**: Pre-populate cache
- **Eviction policies**: LRU, LFU, TTL
- **Cache invalidation**: Keep data consistent

### **4. Database Sharding**

- **Consistent hashing**: Even distribution
- **Cross-shard queries**: Aggregate results
- **Data rebalancing**: Maintain balance
- **Shard management**: Add/remove shards

### **5. Best Practices**

- **Monitor metrics**: CPU, memory, latency, throughput
- **Plan for failure**: Implement circuit breakers
- **Test thoroughly**: Load testing, chaos engineering
- **Document everything**: Architecture decisions, runbooks

---

**🎉 This comprehensive guide provides deep insights into scalability patterns with practical Go implementations! 🚀**
