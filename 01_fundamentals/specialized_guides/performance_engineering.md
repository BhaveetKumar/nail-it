---
# Auto-generated front matter
Title: Performance Engineering
LastUpdated: 2025-11-06T20:45:58.670752
Tags: []
Status: draft
---

# Performance Engineering

## Table of Contents
- [Introduction](#introduction)
- [Performance Profiling](#performance-profiling)
- [Memory Optimization](#memory-optimization)
- [CPU Optimization](#cpu-optimization)
- [I/O Optimization](#io-optimization)
- [Database Performance](#database-performance)
- [Caching Strategies](#caching-strategies)
- [Load Testing](#load-testing)
- [Performance Monitoring](#performance-monitoring)

## Introduction

Performance engineering focuses on optimizing system performance through profiling, analysis, and optimization techniques. This guide covers essential performance engineering concepts and practices.

## Performance Profiling

### Profiling Tools

```go
// Performance Profiler
type Profiler struct {
    cpuProfiler   *CPUProfiler
    memProfiler   *MemoryProfiler
    traceProfiler *TraceProfiler
    blockProfiler *BlockProfiler
}

type CPUProfiler struct {
    enabled       bool
    sampleRate    int
    duration      time.Duration
    output        string
}

type MemoryProfiler struct {
    enabled       bool
    heapProfile   bool
    allocProfile  bool
    goroutineProfile bool
}

func (p *Profiler) StartCPUProfile(duration time.Duration) error {
    file, err := os.Create("cpu.prof")
    if err != nil {
        return err
    }
    
    if err := pprof.StartCPUProfile(file); err != nil {
        return err
    }
    
    p.cpuProfiler.enabled = true
    p.cpuProfiler.duration = duration
    
    return nil
}

func (p *Profiler) StopCPUProfile() error {
    pprof.StopCPUProfile()
    p.cpuProfiler.enabled = false
    return nil
}

func (p *Profiler) WriteHeapProfile() error {
    file, err := os.Create("heap.prof")
    if err != nil {
        return err
    }
    defer file.Close()
    
    return pprof.WriteHeapProfile(file)
}
```

### Benchmarking

```go
// Benchmark Suite
type BenchmarkSuite struct {
    benchmarks   []*Benchmark
    results      []*BenchmarkResult
    environment  *BenchmarkEnvironment
}

type Benchmark struct {
    Name         string
    Function     func(*testing.B)
    Duration     time.Duration
    Iterations   int
    Memory       bool
    CPU          bool
}

type BenchmarkResult struct {
    Name         string
    Duration     time.Duration
    Operations   int64
    Bytes        int64
    Allocs       int64
    Memory       int64
    CPU          float64
}

func (bs *BenchmarkSuite) RunBenchmark(benchmark *Benchmark) (*BenchmarkResult, error) {
    start := time.Now()
    var memStats runtime.MemStats
    runtime.ReadMemStats(&memStats)
    
    // Run benchmark
    result := testing.Benchmark(benchmark.Function)
    
    runtime.ReadMemStats(&memStats)
    end := time.Now()
    
    return &BenchmarkResult{
        Name:       benchmark.Name,
        Duration:   end.Sub(start),
        Operations: result.N,
        Bytes:      result.Bytes,
        Allocs:     result.Allocs,
        Memory:     int64(memStats.Alloc),
        CPU:        float64(result.N) / end.Sub(start).Seconds(),
    }, nil
}
```

## Memory Optimization

### Memory Management

```go
// Memory Manager
type MemoryManager struct {
    pools         map[string]*ObjectPool
    allocators    []*Allocator
    gc            *GarbageCollector
    monitoring    *MemoryMonitoring
}

type ObjectPool struct {
    objects       chan interface{}
    factory       func() interface{}
    reset         func(interface{})
    maxSize       int
    currentSize   int
}

type Allocator struct {
    name          string
    strategy      string
    size          int
    alignment     int
    zeroed        bool
}

func (mm *MemoryManager) GetObject(poolName string) interface{} {
    pool, exists := mm.pools[poolName]
    if !exists {
        return nil
    }
    
    select {
    case obj := <-pool.objects:
        return obj
    default:
        if pool.currentSize < pool.maxSize {
            pool.currentSize++
            return pool.factory()
        }
        return nil
    }
}

func (mm *MemoryManager) PutObject(poolName string, obj interface{}) {
    pool, exists := mm.pools[poolName]
    if !exists {
        return
    }
    
    if pool.reset != nil {
        pool.reset(obj)
    }
    
    select {
    case pool.objects <- obj:
    default:
        // Pool is full, let GC handle it
    }
}
```

### Memory Leak Detection

```go
// Memory Leak Detector
type MemoryLeakDetector struct {
    snapshots     []*MemorySnapshot
    threshold     int64
    interval      time.Duration
    monitoring    *MemoryMonitoring
}

type MemorySnapshot struct {
    Timestamp     time.Time
    HeapSize      int64
    StackSize     int64
    Goroutines    int
    Objects       int64
    Allocations   int64
}

func (mld *MemoryLeakDetector) StartMonitoring() {
    ticker := time.NewTicker(mld.interval)
    go func() {
        for range ticker.C {
            snapshot := mld.takeSnapshot()
            mld.snapshots = append(mld.snapshots, snapshot)
            
            if mld.detectLeak(snapshot) {
                mld.handleLeak(snapshot)
            }
        }
    }()
}

func (mld *MemoryLeakDetector) takeSnapshot() *MemorySnapshot {
    var memStats runtime.MemStats
    runtime.ReadMemStats(&memStats)
    
    return &MemorySnapshot{
        Timestamp:   time.Now(),
        HeapSize:    int64(memStats.HeapAlloc),
        StackSize:   int64(memStats.StackInuse),
        Goroutines:  runtime.NumGoroutine(),
        Objects:     int64(memStats.HeapObjects),
        Allocations: int64(memStats.Mallocs),
    }
}

func (mld *MemoryLeakDetector) detectLeak(snapshot *MemorySnapshot) bool {
    if len(mld.snapshots) < 2 {
        return false
    }
    
    prev := mld.snapshots[len(mld.snapshots)-2]
    growth := snapshot.HeapSize - prev.HeapSize
    
    return growth > mld.threshold
}
```

## CPU Optimization

### CPU Profiling

```go
// CPU Optimizer
type CPUOptimizer struct {
    profiler      *CPUProfiler
    analyzer      *CPUAnalyzer
    optimizer     *CPUOptimizer
    monitoring    *CPUMonitoring
}

type CPUAnalyzer struct {
    hotspots      []*Hotspot
    bottlenecks   []*Bottleneck
    recommendations []*Recommendation
}

type Hotspot struct {
    Function      string
    Percentage    float64
    Samples       int
    Location      string
}

type Bottleneck struct {
    Type          string
    Description   string
    Impact        float64
    Solution      string
}

func (co *CPUOptimizer) AnalyzeProfile(profile *CPUProfile) *CPUAnalyzer {
    analyzer := &CPUAnalyzer{
        hotspots:      make([]*Hotspot, 0),
        bottlenecks:   make([]*Bottleneck, 0),
        recommendations: make([]*Recommendation, 0),
    }
    
    // Find hotspots
    for _, sample := range profile.Samples {
        if sample.Percentage > 5.0 { // 5% threshold
            analyzer.hotspots = append(analyzer.hotspots, &Hotspot{
                Function:   sample.Function,
                Percentage: sample.Percentage,
                Samples:    sample.Count,
                Location:   sample.Location,
            })
        }
    }
    
    // Identify bottlenecks
    analyzer.bottlenecks = co.identifyBottlenecks(profile)
    
    // Generate recommendations
    analyzer.recommendations = co.generateRecommendations(analyzer)
    
    return analyzer
}
```

### Concurrency Optimization

```go
// Concurrency Optimizer
type ConcurrencyOptimizer struct {
    workers       []*Worker
    pool          *WorkerPool
    scheduler     *Scheduler
    monitoring    *ConcurrencyMonitoring
}

type WorkerPool struct {
    workers       []*Worker
    queue         chan *Task
    maxWorkers    int
    activeWorkers int
    mu            sync.RWMutex
}

type Worker struct {
    ID            string
    Status        string
    Tasks         int64
    Duration      time.Duration
    LastActivity  time.Time
}

func (co *ConcurrencyOptimizer) OptimizeConcurrency(tasks []*Task) error {
    // Analyze task characteristics
    analysis := co.analyzeTasks(tasks)
    
    // Determine optimal worker count
    optimalWorkers := co.calculateOptimalWorkers(analysis)
    
    // Adjust worker pool
    co.adjustWorkerPool(optimalWorkers)
    
    // Distribute tasks
    return co.distributeTasks(tasks)
}

func (co *ConcurrencyOptimizer) analyzeTasks(tasks []*Task) *TaskAnalysis {
    analysis := &TaskAnalysis{
        Count:        len(tasks),
        TotalDuration: 0,
        AverageDuration: 0,
        MaxDuration:  0,
        MinDuration:  0,
        IOBound:      0,
        CPUBound:     0,
    }
    
    for _, task := range tasks {
        analysis.TotalDuration += task.Duration
        
        if task.Duration > analysis.MaxDuration {
            analysis.MaxDuration = task.Duration
        }
        
        if analysis.MinDuration == 0 || task.Duration < analysis.MinDuration {
            analysis.MinDuration = task.Duration
        }
        
        if task.Type == "io" {
            analysis.IOBound++
        } else {
            analysis.CPUBound++
        }
    }
    
    analysis.AverageDuration = analysis.TotalDuration / time.Duration(len(tasks))
    
    return analysis
}
```

## I/O Optimization

### I/O Profiling

```go
// I/O Profiler
type IOProfiler struct {
    fileIO        *FileIOProfiler
    networkIO     *NetworkIOProfiler
    databaseIO    *DatabaseIOProfiler
    monitoring    *IOMonitoring
}

type FileIOProfiler struct {
    operations    []*FileOperation
    statistics    *FileIOStatistics
}

type FileOperation struct {
    Type          string
    Path          string
    Size          int64
    Duration      time.Duration
    Timestamp     time.Time
    Error         error
}

type FileIOStatistics struct {
    TotalOps      int64
    TotalSize     int64
    TotalDuration time.Duration
    AverageSize   int64
    AverageDuration time.Duration
    ErrorRate     float64
}

func (iop *IOProfiler) ProfileFileOperation(op *FileOperation) {
    iop.fileIO.operations = append(iop.fileIO.operations, op)
    iop.updateStatistics()
}

func (iop *IOProfiler) updateStatistics() {
    stats := &FileIOStatistics{}
    
    for _, op := range iop.fileIO.operations {
        stats.TotalOps++
        stats.TotalSize += op.Size
        stats.TotalDuration += op.Duration
        
        if op.Error != nil {
            stats.ErrorRate++
        }
    }
    
    if stats.TotalOps > 0 {
        stats.AverageSize = stats.TotalSize / stats.TotalOps
        stats.AverageDuration = stats.TotalDuration / time.Duration(stats.TotalOps)
        stats.ErrorRate = stats.ErrorRate / float64(stats.TotalOps)
    }
    
    iop.fileIO.statistics = stats
}
```

### I/O Optimization Strategies

```go
// I/O Optimizer
type IOOptimizer struct {
    strategies    []*IOStrategy
    cache         *IOCache
    prefetcher    *Prefetcher
    batcher       *IOBatcher
}

type IOStrategy struct {
    Name          string
    Type          string
    Function      func(*IOOperation) error
    Conditions    []*IOCondition
}

type IOCache struct {
    data          map[string]*CacheEntry
    maxSize       int64
    currentSize   int64
    ttl           time.Duration
    mu            sync.RWMutex
}

type CacheEntry struct {
    Data          []byte
    Timestamp     time.Time
    AccessCount   int64
    Size          int64
}

func (ioo *IOOptimizer) OptimizeOperation(op *IOOperation) error {
    // Check cache first
    if cached, exists := ioo.cache.Get(op.Key); exists {
        op.Data = cached
        return nil
    }
    
    // Apply optimization strategies
    for _, strategy := range ioo.strategies {
        if ioo.matchesConditions(op, strategy.Conditions) {
            if err := strategy.Function(op); err != nil {
                return err
            }
        }
    }
    
    // Cache result
    ioo.cache.Set(op.Key, op.Data)
    
    return nil
}

func (ioo *IOOptimizer) matchesConditions(op *IOOperation, conditions []*IOCondition) bool {
    for _, condition := range conditions {
        if !ioo.evaluateCondition(op, condition) {
            return false
        }
    }
    return true
}

func (ioo *IOOptimizer) evaluateCondition(op *IOOperation, condition *IOCondition) bool {
    switch condition.Field {
    case "size":
        return op.Size >= condition.MinValue && op.Size <= condition.MaxValue
    case "type":
        return op.Type == condition.Value
    case "frequency":
        return op.Frequency >= condition.MinValue
    default:
        return false
    }
}
```

## Database Performance

### Query Optimization

```go
// Query Optimizer
type QueryOptimizer struct {
    analyzer      *QueryAnalyzer
    indexer       *IndexManager
    planner       *QueryPlanner
    executor      *QueryExecutor
    monitoring    *QueryMonitoring
}

type QueryAnalyzer struct {
    parser        *QueryParser
    validator     *QueryValidator
    statistics    *QueryStatistics
}

type QueryPlanner struct {
    rules         []*OptimizationRule
    costModel     *CostModel
    statistics    *StatisticsManager
}

type OptimizationRule struct {
    Name          string
    Function      func(*QueryPlan) *QueryPlan
    Cost          float64
    Applicable    func(*Query) bool
}

func (qo *QueryOptimizer) OptimizeQuery(query *Query) (*QueryPlan, error) {
    // Parse query
    parsed, err := qo.analyzer.parser.Parse(query.SQL)
    if err != nil {
        return nil, err
    }
    
    // Validate query
    if err := qo.analyzer.validator.Validate(parsed); err != nil {
        return nil, err
    }
    
    // Create initial plan
    plan := qo.createInitialPlan(parsed)
    
    // Apply optimization rules
    for _, rule := range qo.planner.rules {
        if rule.Applicable(parsed) {
            plan = rule.Function(plan)
        }
    }
    
    // Calculate cost
    plan.Cost = qo.planner.costModel.CalculateCost(plan)
    
    return plan, nil
}

func (qo *QueryOptimizer) createInitialPlan(parsed *ParsedQuery) *QueryPlan {
    plan := &QueryPlan{
        Operations: make([]*QueryOperation, 0),
        Cost:       0,
        EstimatedRows: 0,
    }
    
    // Add scan operation
    scanOp := &QueryOperation{
        Type:        "scan",
        Table:       parsed.Table,
        Index:       parsed.Index,
        Filter:      parsed.Filter,
    }
    plan.Operations = append(plan.Operations, scanOp)
    
    // Add join operations
    for _, join := range parsed.Joins {
        joinOp := &QueryOperation{
            Type:        "join",
            Table:       join.Table,
            Condition:   join.Condition,
            Type:        join.Type,
        }
        plan.Operations = append(plan.Operations, joinOp)
    }
    
    // Add projection
    if len(parsed.Columns) > 0 {
        projOp := &QueryOperation{
            Type:        "projection",
            Columns:     parsed.Columns,
        }
        plan.Operations = append(plan.Operations, projOp)
    }
    
    return plan
}
```

### Index Optimization

```go
// Index Manager
type IndexManager struct {
    indexes       map[string]*Index
    analyzer      *IndexAnalyzer
    recommender   *IndexRecommender
    monitor       *IndexMonitor
}

type Index struct {
    Name          string
    Table         string
    Columns       []string
    Type          string
    Size          int64
    Cardinality   int64
    Selectivity   float64
    Usage         *IndexUsage
}

type IndexUsage struct {
    Scans         int64
    Seeks         int64
    Lookups       int64
    Updates       int64
    LastUsed      time.Time
}

func (im *IndexManager) AnalyzeIndexUsage() *IndexAnalysis {
    analysis := &IndexAnalysis{
        UnusedIndexes: make([]*Index, 0),
        UnderusedIndexes: make([]*Index, 0),
        Recommendations: make([]*IndexRecommendation, 0),
    }
    
    for _, index := range im.indexes {
        if index.Usage.Scans == 0 {
            analysis.UnusedIndexes = append(analysis.UnusedIndexes, index)
        } else if index.Usage.Scans < 10 {
            analysis.UnderusedIndexes = append(analysis.UnderusedIndexes, index)
        }
    }
    
    // Generate recommendations
    analysis.Recommendations = im.recommender.GenerateRecommendations(analysis)
    
    return analysis
}

func (im *IndexManager) CreateIndex(table string, columns []string, indexType string) error {
    index := &Index{
        Name:        generateIndexName(table, columns),
        Table:       table,
        Columns:     columns,
        Type:        indexType,
        Usage:       &IndexUsage{},
    }
    
    // Create index in database
    if err := im.createIndexInDB(index); err != nil {
        return err
    }
    
    // Add to manager
    im.indexes[index.Name] = index
    
    return nil
}
```

## Caching Strategies

### Cache Implementation

```go
// Cache System
type CacheSystem struct {
    caches        map[string]*Cache
    policies      []*CachePolicy
    eviction      *EvictionManager
    monitoring    *CacheMonitoring
}

type Cache struct {
    Name          string
    Type          string
    MaxSize       int64
    CurrentSize   int64
    TTL           time.Duration
    Data          map[string]*CacheEntry
    Statistics    *CacheStatistics
    mu            sync.RWMutex
}

type CacheEntry struct {
    Key           string
    Value         interface{}
    CreatedAt     time.Time
    ExpiresAt     time.Time
    AccessCount   int64
    LastAccessed  time.Time
    Size          int64
}

type CachePolicy struct {
    Name          string
    TTL           time.Duration
    MaxSize       int64
    Eviction      string
    Compression   bool
    Encryption    bool
}

func (cs *CacheSystem) Get(cacheName, key string) (interface{}, bool) {
    cache, exists := cs.caches[cacheName]
    if !exists {
        return nil, false
    }
    
    cache.mu.RLock()
    entry, exists := cache.Data[key]
    cache.mu.RUnlock()
    
    if !exists {
        cs.monitoring.RecordMiss(cacheName, key)
        return nil, false
    }
    
    // Check expiration
    if time.Now().After(entry.ExpiresAt) {
        cs.Delete(cacheName, key)
        cs.monitoring.RecordExpired(cacheName, key)
        return nil, false
    }
    
    // Update access statistics
    entry.AccessCount++
    entry.LastAccessed = time.Now()
    
    cs.monitoring.RecordHit(cacheName, key)
    
    return entry.Value, true
}

func (cs *CacheSystem) Set(cacheName, key string, value interface{}, ttl time.Duration) error {
    cache, exists := cs.caches[cacheName]
    if !exists {
        return fmt.Errorf("cache %s not found", cacheName)
    }
    
    // Calculate size
    size := cs.calculateSize(value)
    
    // Check if we need to evict
    if cache.CurrentSize+size > cache.MaxSize {
        if err := cs.eviction.Evict(cache, size); err != nil {
            return err
        }
    }
    
    // Create entry
    entry := &CacheEntry{
        Key:          key,
        Value:        value,
        CreatedAt:    time.Now(),
        ExpiresAt:    time.Now().Add(ttl),
        AccessCount:  0,
        LastAccessed: time.Now(),
        Size:         size,
    }
    
    cache.mu.Lock()
    cache.Data[key] = entry
    cache.CurrentSize += size
    cache.mu.Unlock()
    
    cs.monitoring.RecordSet(cacheName, key, size)
    
    return nil
}
```

## Load Testing

### Load Test Framework

```go
// Load Test Framework
type LoadTestFramework struct {
    scenarios     []*LoadScenario
    generators    []*LoadGenerator
    collectors    []*MetricsCollector
    analyzers     []*LoadAnalyzer
    reporters     []*LoadReporter
}

type LoadScenario struct {
    Name          string
    Duration      time.Duration
    Users         int
    RampUp        time.Duration
    RampDown      time.Duration
    Functions     []*LoadFunction
    Assertions    []*LoadAssertion
}

type LoadFunction struct {
    Name          string
    Function      func() error
    Weight        float64
    ThinkTime     time.Duration
}

type LoadAssertion struct {
    Name          string
    Condition     string
    Threshold     float64
    Metric        string
}

func (ltf *LoadTestFramework) RunScenario(scenario *LoadScenario) (*LoadTestResult, error) {
    result := &LoadTestResult{
        Scenario:    scenario.Name,
        StartTime:   time.Now(),
        Results:     make([]*LoadTestResult, 0),
        Metrics:     make(map[string]*Metric),
    }
    
    // Start metrics collection
    for _, collector := range ltf.collectors {
        go collector.Start()
    }
    
    // Run load test
    if err := ltf.runLoadTest(scenario, result); err != nil {
        return nil, err
    }
    
    result.EndTime = time.Now()
    result.Duration = result.EndTime.Sub(result.StartTime)
    
    // Analyze results
    for _, analyzer := range ltf.analyzers {
        analysis := analyzer.Analyze(result)
        result.Analyses = append(result.Analyses, analysis)
    }
    
    // Generate reports
    for _, reporter := range ltf.reporters {
        if err := reporter.Report(result); err != nil {
            log.Printf("Error generating report: %v", err)
        }
    }
    
    return result, nil
}

func (ltf *LoadTestFramework) runLoadTest(scenario *LoadScenario, result *LoadTestResult) error {
    // Create user pool
    userPool := make(chan int, scenario.Users)
    for i := 0; i < scenario.Users; i++ {
        userPool <- i
    }
    
    // Ramp up
    rampUpDuration := scenario.RampUp / time.Duration(scenario.Users)
    for i := 0; i < scenario.Users; i++ {
        go func(userID int) {
            defer func() { userPool <- userID }()
            
            // Wait for ramp up
            time.Sleep(rampUpDuration * time.Duration(userID))
            
            // Run user scenario
            ltf.runUserScenario(scenario, userID, result)
        }(i)
    }
    
    // Wait for duration
    time.Sleep(scenario.Duration)
    
    return nil
}

func (ltf *LoadTestFramework) runUserScenario(scenario *LoadScenario, userID int, result *LoadTestResult) {
    start := time.Now()
    
    for {
        // Select function based on weight
        function := ltf.selectFunction(scenario.Functions)
        if function == nil {
            break
        }
        
        // Execute function
        funcStart := time.Now()
        err := function.Function()
        funcDuration := time.Since(funcStart)
        
        // Record result
        ltf.recordResult(result, function.Name, funcDuration, err)
        
        // Think time
        time.Sleep(function.ThinkTime)
        
        // Check if scenario should continue
        if time.Since(start) > scenario.Duration {
            break
        }
    }
}
```

## Performance Monitoring

### Performance Monitor

```go
// Performance Monitor
type PerformanceMonitor struct {
    metrics       *PerformanceMetrics
    collectors    []*MetricsCollector
    analyzers     []*PerformanceAnalyzer
    alerters      []*PerformanceAlerter
    dashboard     *PerformanceDashboard
}

type PerformanceMetrics struct {
    CPU           *CPUMetrics
    Memory        *MemoryMetrics
    I/O           *IOMetrics
    Network       *NetworkMetrics
    Database      *DatabaseMetrics
    Application   *ApplicationMetrics
}

type CPUMetrics struct {
    Usage         float64
    Load          float64
    Cores         int
    Processes     int
    Threads       int
}

type MemoryMetrics struct {
    Used          int64
    Available     int64
    Total         int64
    Heap          int64
    Stack         int64
    GC            *GCMetrics
}

type GCMetrics struct {
    Runs          int64
    Duration      time.Duration
    PauseTime     time.Duration
    Freed         int64
}

func (pm *PerformanceMonitor) StartMonitoring() error {
    // Start metrics collection
    for _, collector := range pm.collectors {
        go collector.Start()
    }
    
    // Start analysis
    for _, analyzer := range pm.analyzers {
        go analyzer.Start()
    }
    
    // Start alerting
    for _, alerter := range pm.alerters {
        go alerter.Start()
    }
    
    return nil
}

func (pm *PerformanceMonitor) CollectMetrics() error {
    // Collect CPU metrics
    pm.collectCPUMetrics()
    
    // Collect memory metrics
    pm.collectMemoryMetrics()
    
    // Collect I/O metrics
    pm.collectIOMetrics()
    
    // Collect network metrics
    pm.collectNetworkMetrics()
    
    // Collect database metrics
    pm.collectDatabaseMetrics()
    
    // Collect application metrics
    pm.collectApplicationMetrics()
    
    return nil
}

func (pm *PerformanceMonitor) collectCPUMetrics() {
    var memStats runtime.MemStats
    runtime.ReadMemStats(&memStats)
    
    pm.metrics.CPU = &CPUMetrics{
        Usage:     pm.calculateCPUUsage(),
        Load:      pm.calculateLoadAverage(),
        Cores:     runtime.NumCPU(),
        Processes: pm.getProcessCount(),
        Threads:   pm.getThreadCount(),
    }
}

func (pm *PerformanceMonitor) collectMemoryMetrics() {
    var memStats runtime.MemStats
    runtime.ReadMemStats(&memStats)
    
    pm.metrics.Memory = &MemoryMetrics{
        Used:      int64(memStats.HeapAlloc),
        Available: int64(memStats.HeapSys - memStats.HeapAlloc),
        Total:     int64(memStats.HeapSys),
        Heap:      int64(memStats.HeapAlloc),
        Stack:     int64(memStats.StackInuse),
        GC: &GCMetrics{
            Runs:      int64(memStats.NumGC),
            Duration:  time.Duration(memStats.PauseTotalNs),
            PauseTime: time.Duration(memStats.PauseTotalNs),
            Freed:     int64(memStats.TotalAlloc - memStats.HeapAlloc),
        },
    }
}
```

## Conclusion

Performance engineering is essential for building high-performance systems. Key areas to focus on include:

1. **Profiling**: CPU, memory, and I/O profiling tools and techniques
2. **Optimization**: Memory, CPU, and I/O optimization strategies
3. **Database Performance**: Query optimization and index management
4. **Caching**: Effective caching strategies and implementations
5. **Load Testing**: Comprehensive load testing frameworks
6. **Monitoring**: Performance monitoring and alerting systems

Mastering these areas will prepare you for building and maintaining high-performance backend systems.

## Additional Resources

- [Go Profiling](https://golang.org/pkg/runtime/pprof/)
- [Performance Testing](https://www.performance-testing.com/)
- [Database Optimization](https://www.dboptimization.com/)
- [Caching Strategies](https://www.cachingstrategies.com/)
- [Load Testing Tools](https://www.loadtestingtools.com/)
- [Performance Monitoring](https://www.performancemonitoring.com/)
