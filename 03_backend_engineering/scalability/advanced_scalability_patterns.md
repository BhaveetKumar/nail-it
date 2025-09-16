# 🚀 Advanced Scalability & Performance Patterns

## Table of Contents
1. [Horizontal vs Vertical Scaling](#horizontal-vs-vertical-scaling/)
2. [Load Balancing Strategies](#load-balancing-strategies/)
3. [Caching Patterns](#caching-patterns/)
4. [Database Scaling](#database-scaling/)
5. [Microservices Scaling](#microservices-scaling/)
6. [Event-Driven Architecture](#event-driven-architecture/)
7. [Performance Optimization](#performance-optimization/)
8. [Go Implementation Examples](#go-implementation-examples/)
9. [Interview Questions](#interview-questions/)

## Horizontal vs Vertical Scaling

### Scaling Strategies Implementation

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Vertical Scaling - Single Machine Optimization
type VerticalScaler struct {
    maxCPU    int
    maxMemory int64
    currentCPU int
    currentMemory int64
    mutex     sync.RWMutex
}

func NewVerticalScaler(maxCPU int, maxMemory int64) *VerticalScaler {
    return &VerticalScaler{
        maxCPU:    maxCPU,
        maxMemory: maxMemory,
    }
}

func (vs *VerticalScaler) OptimizeResourceUsage() {
    vs.mutex.Lock()
    defer vs.mutex.Unlock()
    
    // Simulate resource optimization
    if vs.currentCPU > vs.maxCPU*80/100 {
        vs.scaleUpCPU()
    }
    
    if vs.currentMemory > vs.maxMemory*80/100 {
        vs.scaleUpMemory()
    }
}

func (vs *VerticalScaler) scaleUpCPU() {
    // Simulate CPU scaling
    fmt.Println("Scaling up CPU resources")
    vs.currentCPU = vs.maxCPU
}

func (vs *VerticalScaler) scaleUpMemory() {
    // Simulate memory scaling
    fmt.Println("Scaling up memory resources")
    vs.currentMemory = vs.maxMemory
}

// Horizontal Scaling - Multiple Instances
type HorizontalScaler struct {
    instances    []*ServiceInstance
    minInstances int
    maxInstances int
    targetCPU    float64
    mutex        sync.RWMutex
}

type ServiceInstance struct {
    ID       string
    CPU      float64
    Memory   float64
    Requests int64
    IsHealthy bool
    CreatedAt time.Time
}

func NewHorizontalScaler(minInstances, maxInstances int, targetCPU float64) *HorizontalScaler {
    hs := &HorizontalScaler{
        instances:    make([]*ServiceInstance, 0),
        minInstances: minInstances,
        maxInstances: maxInstances,
        targetCPU:    targetCPU,
    }
    
    // Initialize with minimum instances
    for i := 0; i < minInstances; i++ {
        hs.addInstance()
    }
    
    return hs
}

func (hs *HorizontalScaler) addInstance() {
    instance := &ServiceInstance{
        ID:        fmt.Sprintf("instance-%d", len(hs.instances)+1),
        CPU:       0.0,
        Memory:    0.0,
        Requests:  0,
        IsHealthy: true,
        CreatedAt: time.Now(),
    }
    
    hs.instances = append(hs.instances, instance)
    fmt.Printf("Added instance: %s\n", instance.ID)
}

func (hs *HorizontalScaler) removeInstance() {
    if len(hs.instances) <= hs.minInstances {
        return
    }
    
    // Remove the least loaded instance
    var minIndex int
    minRequests := int64(^uint64(0) >> 1) // Max int64
    
    for i, instance := range hs.instances {
        if instance.Requests < minRequests {
            minRequests = instance.Requests
            minIndex = i
        }
    }
    
    removed := hs.instances[minIndex]
    hs.instances = append(hs.instances[:minIndex], hs.instances[minIndex+1:]...)
    fmt.Printf("Removed instance: %s\n", removed.ID)
}

func (hs *HorizontalScaler) Scale() {
    hs.mutex.Lock()
    defer hs.mutex.Unlock()
    
    avgCPU := hs.getAverageCPU()
    
    if avgCPU > hs.targetCPU && len(hs.instances) < hs.maxInstances {
        hs.addInstance()
    } else if avgCPU < hs.targetCPU*0.5 && len(hs.instances) > hs.minInstances {
        hs.removeInstance()
    }
}

func (hs *HorizontalScaler) getAverageCPU() float64 {
    if len(hs.instances) == 0 {
        return 0
    }
    
    totalCPU := 0.0
    for _, instance := range hs.instances {
        if instance.IsHealthy {
            totalCPU += instance.CPU
        }
    }
    
    return totalCPU / float64(len(hs.instances))
}

func (hs *HorizontalScaler) UpdateInstanceMetrics(instanceID string, cpu, memory float64, requests int64) {
    hs.mutex.Lock()
    defer hs.mutex.Unlock()
    
    for _, instance := range hs.instances {
        if instance.ID == instanceID {
            instance.CPU = cpu
            instance.Memory = memory
            instance.Requests = requests
            break
        }
    }
}
```

## Load Balancing Strategies

### Advanced Load Balancer

```go
package main

import (
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type LoadBalancer struct {
    instances []*ServiceInstance
    strategy  LoadBalancingStrategy
    mutex     sync.RWMutex
}

type LoadBalancingStrategy interface {
    SelectInstance(instances []*ServiceInstance) *ServiceInstance
}

// Round Robin Strategy
type RoundRobinStrategy struct {
    currentIndex int
    mutex        sync.Mutex
}

func (rr *RoundRobinStrategy) SelectInstance(instances []*ServiceInstance) *ServiceInstance {
    rr.mutex.Lock()
    defer rr.mutex.Unlock()
    
    if len(instances) == 0 {
        return nil
    }
    
    // Filter healthy instances
    healthyInstances := make([]*ServiceInstance, 0)
    for _, instance := range instances {
        if instance.IsHealthy {
            healthyInstances = append(healthyInstances, instance)
        }
    }
    
    if len(healthyInstances) == 0 {
        return nil
    }
    
    instance := healthyInstances[rr.currentIndex%len(healthyInstances)]
    rr.currentIndex++
    return instance
}

// Least Connections Strategy
type LeastConnectionsStrategy struct{}

func (lc *LeastConnectionsStrategy) SelectInstance(instances []*ServiceInstance) *ServiceInstance {
    var selected *ServiceInstance
    minConnections := int64(^uint64(0) >> 1) // Max int64
    
    for _, instance := range instances {
        if instance.IsHealthy && instance.Requests < minConnections {
            minConnections = instance.Requests
            selected = instance
        }
    }
    
    return selected
}

// Weighted Round Robin Strategy
type WeightedRoundRobinStrategy struct {
    currentIndex int
    currentWeight int
    mutex        sync.Mutex
}

func (wrr *WeightedRoundRobinStrategy) SelectInstance(instances []*ServiceInstance) *ServiceInstance {
    wrr.mutex.Lock()
    defer wrr.mutex.Unlock()
    
    if len(instances) == 0 {
        return nil
    }
    
    // Filter healthy instances
    healthyInstances := make([]*ServiceInstance, 0)
    for _, instance := range instances {
        if instance.IsHealthy {
            healthyInstances = append(healthyInstances, instance)
        }
    }
    
    if len(healthyInstances) == 0 {
        return nil
    }
    
    // Weighted selection (simplified - using CPU as weight)
    totalWeight := 0
    for _, instance := range healthyInstances {
        totalWeight += int(instance.CPU * 100) // Convert to integer weight
    }
    
    if totalWeight == 0 {
        return healthyInstances[0]
    }
    
    // Find instance with current weight
    currentWeight := wrr.currentWeight
    for _, instance := range healthyInstances {
        weight := int(instance.CPU * 100)
        if currentWeight < weight {
            wrr.currentWeight = currentWeight + totalWeight
            return instance
        }
        currentWeight -= weight
    }
    
    // Fallback to round robin
    instance := healthyInstances[wrr.currentIndex%len(healthyInstances)]
    wrr.currentIndex++
    return instance
}

// Random Strategy
type RandomStrategy struct{}

func (r *RandomStrategy) SelectInstance(instances []*ServiceInstance) *ServiceInstance {
    healthyInstances := make([]*ServiceInstance, 0)
    for _, instance := range instances {
        if instance.IsHealthy {
            healthyInstances = append(healthyInstances, instance)
        }
    }
    
    if len(healthyInstances) == 0 {
        return nil
    }
    
    return healthyInstances[rand.Intn(len(healthyInstances))]
}

func NewLoadBalancer(strategy LoadBalancingStrategy) *LoadBalancer {
    return &LoadBalancer{
        instances: make([]*ServiceInstance, 0),
        strategy:  strategy,
    }
}

func (lb *LoadBalancer) AddInstance(instance *ServiceInstance) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    lb.instances = append(lb.instances, instance)
}

func (lb *LoadBalancer) RemoveInstance(instanceID string) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    for i, instance := range lb.instances {
        if instance.ID == instanceID {
            lb.instances = append(lb.instances[:i], lb.instances[i+1:]...)
            break
        }
    }
}

func (lb *LoadBalancer) SelectInstance() *ServiceInstance {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    return lb.strategy.SelectInstance(lb.instances)
}

func (lb *LoadBalancer) HandleRequest() {
    instance := lb.SelectInstance()
    if instance == nil {
        fmt.Println("No healthy instances available")
        return
    }
    
    // Simulate request processing
    instance.Requests++
    fmt.Printf("Request routed to instance: %s (CPU: %.2f, Requests: %d)\n", 
        instance.ID, instance.CPU, instance.Requests)
    
    // Simulate request completion
    time.Sleep(100 * time.Millisecond)
    instance.Requests--
}
```

## Caching Patterns

### Multi-Level Caching System

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type CacheEntry struct {
    Value     interface{}
    ExpiresAt time.Time
    AccessCount int64
    LastAccess time.Time
}

type CacheLevel int

const (
    L1Cache CacheLevel = iota
    L2Cache
    L3Cache
)

type MultiLevelCache struct {
    l1Cache *L1Cache
    l2Cache *L2Cache
    l3Cache *L3Cache
    mutex   sync.RWMutex
}

type L1Cache struct {
    data map[string]*CacheEntry
    size int
    mutex sync.RWMutex
}

type L2Cache struct {
    data map[string]*CacheEntry
    size int
    mutex sync.RWMutex
}

type L3Cache struct {
    data map[string]*CacheEntry
    size int
    mutex sync.RWMutex
}

func NewMultiLevelCache(l1Size, l2Size, l3Size int) *MultiLevelCache {
    return &MultiLevelCache{
        l1Cache: &L1Cache{
            data: make(map[string]*CacheEntry),
            size: l1Size,
        },
        l2Cache: &L2Cache{
            data: make(map[string]*CacheEntry),
            size: l2Size,
        },
        l3Cache: &L3Cache{
            data: make(map[string]*CacheEntry),
            size: l3Size,
        },
    }
}

func (mlc *MultiLevelCache) Get(key string) (interface{}, bool) {
    // Try L1 cache first
    if value, found := mlc.l1Cache.Get(key); found {
        return value, true
    }
    
    // Try L2 cache
    if value, found := mlc.l2Cache.Get(key); found {
        // Promote to L1
        mlc.l1Cache.Set(key, value, 5*time.Minute)
        return value, true
    }
    
    // Try L3 cache
    if value, found := mlc.l3Cache.Get(key); found {
        // Promote to L2
        mlc.l2Cache.Set(key, value, 10*time.Minute)
        return value, true
    }
    
    return nil, false
}

func (mlc *MultiLevelCache) Set(key string, value interface{}, ttl time.Duration) {
    // Set in L1 cache
    mlc.l1Cache.Set(key, value, ttl)
    
    // Also set in L2 and L3 with longer TTL
    mlc.l2Cache.Set(key, value, ttl*2)
    mlc.l3Cache.Set(key, value, ttl*4)
}

func (l1 *L1Cache) Get(key string) (interface{}, bool) {
    l1.mutex.RLock()
    defer l1.mutex.RUnlock()
    
    entry, exists := l1.data[key]
    if !exists {
        return nil, false
    }
    
    if time.Now().After(entry.ExpiresAt) {
        delete(l1.data, key)
        return nil, false
    }
    
    entry.AccessCount++
    entry.LastAccess = time.Now()
    return entry.Value, true
}

func (l1 *L1Cache) Set(key string, value interface{}, ttl time.Duration) {
    l1.mutex.Lock()
    defer l1.mutex.Unlock()
    
    // Check if we need to evict
    if len(l1.data) >= l1.size {
        l1.evictLRU()
    }
    
    l1.data[key] = &CacheEntry{
        Value:      value,
        ExpiresAt:  time.Now().Add(ttl),
        AccessCount: 1,
        LastAccess: time.Now(),
    }
}

func (l1 *L1Cache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, entry := range l1.data {
        if oldestKey == "" || entry.LastAccess.Before(oldestTime) {
            oldestKey = key
            oldestTime = entry.LastAccess
        }
    }
    
    if oldestKey != "" {
        delete(l1.data, oldestKey)
    }
}

// Similar implementations for L2Cache and L3Cache
func (l2 *L2Cache) Get(key string) (interface{}, bool) {
    l2.mutex.RLock()
    defer l2.mutex.RUnlock()
    
    entry, exists := l2.data[key]
    if !exists {
        return nil, false
    }
    
    if time.Now().After(entry.ExpiresAt) {
        delete(l2.data, key)
        return nil, false
    }
    
    entry.AccessCount++
    entry.LastAccess = time.Now()
    return entry.Value, true
}

func (l2 *L2Cache) Set(key string, value interface{}, ttl time.Duration) {
    l2.mutex.Lock()
    defer l2.mutex.Unlock()
    
    if len(l2.data) >= l2.size {
        l2.evictLRU()
    }
    
    l2.data[key] = &CacheEntry{
        Value:      value,
        ExpiresAt:  time.Now().Add(ttl),
        AccessCount: 1,
        LastAccess: time.Now(),
    }
}

func (l2 *L2Cache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, entry := range l2.data {
        if oldestKey == "" || entry.LastAccess.Before(oldestTime) {
            oldestKey = key
            oldestTime = entry.LastAccess
        }
    }
    
    if oldestKey != "" {
        delete(l2.data, oldestKey)
    }
}

func (l3 *L3Cache) Get(key string) (interface{}, bool) {
    l3.mutex.RLock()
    defer l3.mutex.RUnlock()
    
    entry, exists := l3.data[key]
    if !exists {
        return nil, false
    }
    
    if time.Now().After(entry.ExpiresAt) {
        delete(l3.data, key)
        return nil, false
    }
    
    entry.AccessCount++
    entry.LastAccess = time.Now()
    return entry.Value, true
}

func (l3 *L3Cache) Set(key string, value interface{}, ttl time.Duration) {
    l3.mutex.Lock()
    defer l3.mutex.Unlock()
    
    if len(l3.data) >= l3.size {
        l3.evictLRU()
    }
    
    l3.data[key] = &CacheEntry{
        Value:      value,
        ExpiresAt:  time.Now().Add(ttl),
        AccessCount: 1,
        LastAccess: time.Now(),
    }
}

func (l3 *L3Cache) evictLRU() {
    var oldestKey string
    var oldestTime time.Time
    
    for key, entry := range l3.data {
        if oldestKey == "" || entry.LastAccess.Before(oldestTime) {
            oldestKey = key
            oldestTime = entry.LastAccess
        }
    }
    
    if oldestKey != "" {
        delete(l3.data, oldestKey)
    }
}
```

## Database Scaling

### Database Sharding Implementation

```go
package main

import (
    "fmt"
    "hash/crc32"
    "sync"
)

type Shard struct {
    ID       int
    Database *Database
    mutex    sync.RWMutex
}

type Database struct {
    Name string
    Data map[string]interface{}
}

type ShardedDatabase struct {
    shards []*Shard
    mutex  sync.RWMutex
}

func NewShardedDatabase(numShards int) *ShardedDatabase {
    shards := make([]*Shard, numShards)
    for i := 0; i < numShards; i++ {
        shards[i] = &Shard{
            ID: i,
            Database: &Database{
                Name: fmt.Sprintf("shard_%d", i),
                Data: make(map[string]interface{}),
            },
        }
    }
    
    return &ShardedDatabase{shards: shards}
}

func (sd *ShardedDatabase) getShard(key string) *Shard {
    hash := crc32.ChecksumIEEE([]byte(key))
    shardIndex := int(hash) % len(sd.shards)
    return sd.shards[shardIndex]
}

func (sd *ShardedDatabase) Put(key string, value interface{}) error {
    shard := sd.getShard(key)
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    
    shard.Database.Data[key] = value
    fmt.Printf("Stored key %s in shard %d\n", key, shard.ID)
    return nil
}

func (sd *ShardedDatabase) Get(key string) (interface{}, error) {
    shard := sd.getShard(key)
    shard.mutex.RLock()
    defer shard.mutex.RUnlock()
    
    value, exists := shard.Database.Data[key]
    if !exists {
        return nil, fmt.Errorf("key not found")
    }
    
    fmt.Printf("Retrieved key %s from shard %d\n", key, shard.ID)
    return value, nil
}

func (sd *ShardedDatabase) Delete(key string) error {
    shard := sd.getShard(key)
    shard.mutex.Lock()
    defer shard.mutex.Unlock()
    
    delete(shard.Database.Data, key)
    fmt.Printf("Deleted key %s from shard %d\n", key, shard.ID)
    return nil
}

// Read Replicas
type ReadReplica struct {
    ID       string
    Database *Database
    IsHealthy bool
    mutex    sync.RWMutex
}

type MasterSlaveDatabase struct {
    master   *Database
    replicas []*ReadReplica
    mutex    sync.RWMutex
}

func NewMasterSlaveDatabase(masterName string, replicaNames []string) *MasterSlaveDatabase {
    replicas := make([]*ReadReplica, len(replicaNames))
    for i, name := range replicaNames {
        replicas[i] = &ReadReplica{
            ID: name,
            Database: &Database{
                Name: name,
                Data: make(map[string]interface{}),
            },
            IsHealthy: true,
        }
    }
    
    return &MasterSlaveDatabase{
        master: &Database{
            Name: masterName,
            Data: make(map[string]interface{}),
        },
        replicas: replicas,
    }
}

func (msdb *MasterSlaveDatabase) Write(key string, value interface{}) error {
    msdb.mutex.Lock()
    defer msdb.mutex.Unlock()
    
    // Write to master
    msdb.master.Data[key] = value
    
    // Replicate to all healthy replicas
    for _, replica := range msdb.replicas {
        if replica.IsHealthy {
            replica.mutex.Lock()
            replica.Database.Data[key] = value
            replica.mutex.Unlock()
        }
    }
    
    fmt.Printf("Written key %s to master and replicas\n", key)
    return nil
}

func (msdb *MasterSlaveDatabase) Read(key string) (interface{}, error) {
    // Try to read from a healthy replica first
    for _, replica := range msdb.replicas {
        if replica.IsHealthy {
            replica.mutex.RLock()
            value, exists := replica.Database.Data[key]
            replica.mutex.RUnlock()
            
            if exists {
                fmt.Printf("Read key %s from replica %s\n", key, replica.ID)
                return value, nil
            }
        }
    }
    
    // Fallback to master
    msdb.mutex.RLock()
    value, exists := msdb.master.Data[key]
    msdb.mutex.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("key not found")
    }
    
    fmt.Printf("Read key %s from master\n", key)
    return value, nil
}
```

## Microservices Scaling

### Service Mesh Implementation

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type ServiceMesh struct {
    services map[string]*Service
    mutex    sync.RWMutex
}

type Service struct {
    Name        string
    Instances   []*ServiceInstance
    LoadBalancer *LoadBalancer
    CircuitBreaker *CircuitBreaker
    RetryPolicy *RetryPolicy
}

type CircuitBreaker struct {
    FailureThreshold int
    Timeout         time.Duration
    FailureCount    int
    LastFailureTime time.Time
    State          CircuitState
    mutex          sync.RWMutex
}

type CircuitState int

const (
    Closed CircuitState = iota
    Open
    HalfOpen
)

type RetryPolicy struct {
    MaxRetries int
    Backoff    time.Duration
    MaxBackoff time.Duration
}

func NewServiceMesh() *ServiceMesh {
    return &ServiceMesh{
        services: make(map[string]*Service),
    }
}

func (sm *ServiceMesh) RegisterService(name string, instances []*ServiceInstance) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    service := &Service{
        Name:        name,
        Instances:   instances,
        LoadBalancer: NewLoadBalancer(&RoundRobinStrategy{}),
        CircuitBreaker: &CircuitBreaker{
            FailureThreshold: 5,
            Timeout:         30 * time.Second,
            State:          Closed,
        },
        RetryPolicy: &RetryPolicy{
            MaxRetries: 3,
            Backoff:    100 * time.Millisecond,
            MaxBackoff: 5 * time.Second,
        },
    }
    
    // Add instances to load balancer
    for _, instance := range instances {
        service.LoadBalancer.AddInstance(instance)
    }
    
    sm.services[name] = service
}

func (sm *ServiceMesh) CallService(serviceName string, request interface{}) (interface{}, error) {
    sm.mutex.RLock()
    service, exists := sm.services[serviceName]
    sm.mutex.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("service not found: %s", serviceName)
    }
    
    // Check circuit breaker
    if !service.CircuitBreaker.CanExecute() {
        return nil, fmt.Errorf("circuit breaker is open for service: %s", serviceName)
    }
    
    // Execute with retry policy
    var lastErr error
    for attempt := 0; attempt <= service.RetryPolicy.MaxRetries; attempt++ {
        if attempt > 0 {
            backoff := service.RetryPolicy.Backoff * time.Duration(attempt)
            if backoff > service.RetryPolicy.MaxBackoff {
                backoff = service.RetryPolicy.MaxBackoff
            }
            time.Sleep(backoff)
        }
        
        result, err := sm.executeRequest(service, request)
        if err == nil {
            service.CircuitBreaker.OnSuccess()
            return result, nil
        }
        
        lastErr = err
        service.CircuitBreaker.OnFailure()
    }
    
    return nil, fmt.Errorf("service call failed after %d attempts: %v", 
        service.RetryPolicy.MaxRetries+1, lastErr)
}

func (sm *ServiceMesh) executeRequest(service *Service, request interface{}) (interface{}, error) {
    instance := service.LoadBalancer.SelectInstance()
    if instance == nil {
        return nil, fmt.Errorf("no healthy instances available")
    }
    
    // Simulate request execution
    time.Sleep(50 * time.Millisecond)
    
    // Simulate occasional failures
    if time.Now().UnixNano()%10 == 0 {
        return nil, fmt.Errorf("simulated failure")
    }
    
    return fmt.Sprintf("Response from %s", instance.ID), nil
}

func (cb *CircuitBreaker) CanExecute() bool {
    cb.mutex.RLock()
    defer cb.mutex.RUnlock()
    
    switch cb.State {
    case Closed:
        return true
    case Open:
        if time.Since(cb.LastFailureTime) > cb.Timeout {
            cb.State = HalfOpen
            return true
        }
        return false
    case HalfOpen:
        return true
    default:
        return false
    }
}

func (cb *CircuitBreaker) OnSuccess() {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    cb.FailureCount = 0
    cb.State = Closed
}

func (cb *CircuitBreaker) OnFailure() {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    cb.FailureCount++
    cb.LastFailureTime = time.Now()
    
    if cb.FailureCount >= cb.FailureThreshold {
        cb.State = Open
    }
}
```

## Performance Optimization

### Profiling and Monitoring

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

type PerformanceMonitor struct {
    metrics map[string]*Metric
    mutex   sync.RWMutex
}

type Metric struct {
    Name        string
    Value       float64
    Count       int64
    LastUpdated time.Time
    mutex       sync.RWMutex
}

func NewPerformanceMonitor() *PerformanceMonitor {
    pm := &PerformanceMonitor{
        metrics: make(map[string]*Metric),
    }
    
    // Start monitoring goroutine
    go pm.monitorSystemMetrics()
    
    return pm
}

func (pm *PerformanceMonitor) RecordMetric(name string, value float64) {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    metric, exists := pm.metrics[name]
    if !exists {
        metric = &Metric{Name: name}
        pm.metrics[name] = metric
    }
    
    metric.mutex.Lock()
    metric.Value = value
    metric.Count++
    metric.LastUpdated = time.Now()
    metric.mutex.Unlock()
}

func (pm *PerformanceMonitor) GetMetric(name string) (*Metric, bool) {
    pm.mutex.RLock()
    defer pm.mutex.RUnlock()
    
    metric, exists := pm.metrics[name]
    return metric, exists
}

func (pm *PerformanceMonitor) monitorSystemMetrics() {
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()
    
    for range ticker.C {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)
        
        // Record memory metrics
        pm.RecordMetric("memory_alloc_bytes", float64(m.Alloc))
        pm.RecordMetric("memory_total_alloc_bytes", float64(m.TotalAlloc))
        pm.RecordMetric("memory_sys_bytes", float64(m.Sys))
        pm.RecordMetric("memory_num_gc", float64(m.NumGC))
        
        // Record goroutine count
        pm.RecordMetric("goroutines", float64(runtime.NumGoroutine()))
    }
}

// Performance Profiler
type Profiler struct {
    startTime time.Time
    endTime   time.Time
    metrics   map[string]time.Duration
    mutex     sync.RWMutex
}

func NewProfiler() *Profiler {
    return &Profiler{
        metrics: make(map[string]time.Duration),
    }
}

func (p *Profiler) Start() {
    p.startTime = time.Now()
}

func (p *Profiler) End() {
    p.endTime = time.Now()
}

func (p *Profiler) RecordDuration(name string, duration time.Duration) {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    p.metrics[name] = duration
}

func (p *Profiler) GetTotalDuration() time.Duration {
    return p.endTime.Sub(p.startTime)
}

func (p *Profiler) GetReport() map[string]time.Duration {
    p.mutex.RLock()
    defer p.mutex.RUnlock()
    
    report := make(map[string]time.Duration)
    for name, duration := range p.metrics {
        report[name] = duration
    }
    return report
}

// Memory Pool for Object Reuse
type ObjectPool struct {
    pool chan interface{}
    new  func() interface{}
    mutex sync.Mutex
}

func NewObjectPool(size int, newFunc func() interface{}) *ObjectPool {
    return &ObjectPool{
        pool: make(chan interface{}, size),
        new:  newFunc,
    }
}

func (op *ObjectPool) Get() interface{} {
    select {
    case obj := <-op.pool:
        return obj
    default:
        return op.new()
    }
}

func (op *ObjectPool) Put(obj interface{}) {
    select {
    case op.pool <- obj:
    default:
        // Pool is full, discard object
    }
}
```

## Interview Questions

### Basic Concepts
1. **What is the difference between horizontal and vertical scaling?**
2. **How do you implement load balancing?**
3. **What are the different caching strategies?**
4. **How do you scale databases?**
5. **What is the purpose of circuit breakers?**

### Advanced Topics
1. **How would you implement auto-scaling?**
2. **What are the trade-offs of different load balancing algorithms?**
3. **How do you handle database sharding?**
4. **What are the challenges of microservices scaling?**
5. **How do you optimize performance in Go?**

### System Design
1. **Design a scalable web application.**
2. **How would you implement a distributed cache?**
3. **Design a high-performance API gateway.**
4. **How would you scale a real-time system?**
5. **Design a fault-tolerant distributed system.**

## Conclusion

Advanced scalability and performance patterns are essential for building high-performance systems. Key areas to master:

- **Scaling Strategies**: Horizontal vs vertical scaling
- **Load Balancing**: Different algorithms and strategies
- **Caching**: Multi-level caching, cache invalidation
- **Database Scaling**: Sharding, read replicas, partitioning
- **Microservices**: Service mesh, circuit breakers, retry policies
- **Performance**: Profiling, monitoring, optimization

Understanding these concepts helps in:
- Building scalable applications
- Optimizing performance
- Handling high traffic
- Designing fault-tolerant systems
- Preparing for technical interviews

This guide provides a comprehensive foundation for scalability patterns and their practical implementation in Go.
