# ⚡ Performance Engineering Guide

> **Advanced performance optimization techniques and profiling for senior backend engineers**

## 🎯 **Overview**

Performance engineering is crucial for building scalable backend systems. This guide covers profiling techniques, memory optimization, CPU optimization, I/O optimization, caching strategies, load testing, and performance monitoring with practical Go implementations.

## 📚 **Table of Contents**

1. [Performance Profiling](#performance-profiling)
2. [Memory Optimization](#memory-optimization)
3. [CPU Optimization](#cpu-optimization)
4. [I/O Optimization](#io-optimization)
5. [Caching Strategies](#caching-strategies)
6. [Concurrency Optimization](#concurrency-optimization)
7. [Database Performance](#database-performance)
8. [Network Optimization](#network-optimization)
9. [Load Testing](#load-testing)
10. [Performance Monitoring](#performance-monitoring)
11. [Interview Questions](#interview-questions)

---

## 📊 **Performance Profiling**

### **Go Profiling Implementation**

```go
package profiling

import (
    "context"
    "fmt"
    "os"
    "runtime"
    "runtime/pprof"
    "runtime/trace"
    "sync"
    "time"
)

// Performance Profiler
type PerformanceProfiler struct {
    cpuProfile    *os.File
    memProfile    *os.File
    traceFile     *os.File
    profileDir    string
    isActive      bool
    mu            sync.RWMutex
    metrics       ProfileMetrics
}

type ProfileMetrics struct {
    StartTime       time.Time         `json:"start_time"`
    Duration        time.Duration     `json:"duration"`
    GoroutineCount  int               `json:"goroutine_count"`
    HeapAlloc       uint64            `json:"heap_alloc"`
    HeapSys         uint64            `json:"heap_sys"`
    HeapInuse       uint64            `json:"heap_inuse"`
    HeapObjects     uint64            `json:"heap_objects"`
    GCCycles        uint32            `json:"gc_cycles"`
    GCPauseTotal    time.Duration     `json:"gc_pause_total"`
    GCPauseAvg      time.Duration     `json:"gc_pause_avg"`
    CPUSamples      int               `json:"cpu_samples"`
    MemAllocations  map[string]uint64 `json:"memory_allocations"`
}

func NewPerformanceProfiler(profileDir string) *PerformanceProfiler {
    return &PerformanceProfiler{
        profileDir: profileDir,
        metrics: ProfileMetrics{
            MemAllocations: make(map[string]uint64),
        },
    }
}

// Start comprehensive profiling
func (pp *PerformanceProfiler) StartProfiling() error {
    pp.mu.Lock()
    defer pp.mu.Unlock()
    
    if pp.isActive {
        return fmt.Errorf("profiling already active")
    }
    
    // Create profile directory
    if err := os.MkdirAll(pp.profileDir, 0755); err != nil {
        return fmt.Errorf("failed to create profile directory: %w", err)
    }
    
    pp.metrics.StartTime = time.Now()
    
    // Start CPU profiling
    if err := pp.startCPUProfiling(); err != nil {
        return fmt.Errorf("failed to start CPU profiling: %w", err)
    }
    
    // Start trace profiling
    if err := pp.startTraceProfiling(); err != nil {
        return fmt.Errorf("failed to start trace profiling: %w", err)
    }
    
    pp.isActive = true
    
    // Start background metrics collection
    go pp.collectRuntimeMetrics()
    
    return nil
}

func (pp *PerformanceProfiler) startCPUProfiling() error {
    cpuFile, err := os.Create(fmt.Sprintf("%s/cpu.prof", pp.profileDir))
    if err != nil {
        return err
    }
    
    pp.cpuProfile = cpuFile
    return pprof.StartCPUProfile(cpuFile)
}

func (pp *PerformanceProfiler) startTraceProfiling() error {
    traceFile, err := os.Create(fmt.Sprintf("%s/trace.out", pp.profileDir))
    if err != nil {
        return err
    }
    
    pp.traceFile = traceFile
    return trace.Start(traceFile)
}

// Stop profiling and generate reports
func (pp *PerformanceProfiler) StopProfiling() error {
    pp.mu.Lock()
    defer pp.mu.Unlock()
    
    if !pp.isActive {
        return fmt.Errorf("profiling not active")
    }
    
    pp.metrics.Duration = time.Since(pp.metrics.StartTime)
    
    // Stop CPU profiling
    pprof.StopCPUProfile()
    if pp.cpuProfile != nil {
        pp.cpuProfile.Close()
    }
    
    // Stop trace profiling
    trace.Stop()
    if pp.traceFile != nil {
        pp.traceFile.Close()
    }
    
    // Generate memory profile
    if err := pp.generateMemoryProfile(); err != nil {
        return fmt.Errorf("failed to generate memory profile: %w", err)
    }
    
    // Generate goroutine profile
    if err := pp.generateGoroutineProfile(); err != nil {
        return fmt.Errorf("failed to generate goroutine profile: %w", err)
    }
    
    pp.isActive = false
    return nil
}

func (pp *PerformanceProfiler) generateMemoryProfile() error {
    memFile, err := os.Create(fmt.Sprintf("%s/mem.prof", pp.profileDir))
    if err != nil {
        return err
    }
    defer memFile.Close()
    
    runtime.GC() // Get up-to-date memory statistics
    return pprof.WriteHeapProfile(memFile)
}

func (pp *PerformanceProfiler) generateGoroutineProfile() error {
    goroutineFile, err := os.Create(fmt.Sprintf("%s/goroutine.prof", pp.profileDir))
    if err != nil {
        return err
    }
    defer goroutineFile.Close()
    
    return pprof.Lookup("goroutine").WriteTo(goroutineFile, 0)
}

// Collect runtime metrics during profiling
func (pp *PerformanceProfiler) collectRuntimeMetrics() {
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()
    
    for {
        pp.mu.RLock()
        if !pp.isActive {
            pp.mu.RUnlock()
            break
        }
        pp.mu.RUnlock()
        
        select {
        case <-ticker.C:
            pp.updateMetrics()
        }
    }
}

func (pp *PerformanceProfiler) updateMetrics() {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    pp.mu.Lock()
    defer pp.mu.Unlock()
    
    pp.metrics.GoroutineCount = runtime.NumGoroutine()
    pp.metrics.HeapAlloc = m.HeapAlloc
    pp.metrics.HeapSys = m.HeapSys
    pp.metrics.HeapInuse = m.HeapInuse
    pp.metrics.HeapObjects = m.HeapObjects
    pp.metrics.GCCycles = m.NumGC
    pp.metrics.GCPauseTotal = time.Duration(m.PauseTotalNs)
    
    if m.NumGC > 0 {
        pp.metrics.GCPauseAvg = time.Duration(m.PauseTotalNs / uint64(m.NumGC))
    }
}

// Get current profiling metrics
func (pp *PerformanceProfiler) GetMetrics() ProfileMetrics {
    pp.mu.RLock()
    defer pp.mu.RUnlock()
    
    // Update real-time metrics
    if pp.isActive {
        pp.updateMetrics()
    }
    
    return pp.metrics
}

// Custom profiling for specific functions
type FunctionProfiler struct {
    samples     map[string][]time.Duration
    allocations map[string][]uint64
    mu          sync.RWMutex
}

func NewFunctionProfiler() *FunctionProfiler {
    return &FunctionProfiler{
        samples:     make(map[string][]time.Duration),
        allocations: make(map[string][]uint64),
    }
}

// Profile a function execution
func (fp *FunctionProfiler) Profile(name string, fn func()) {
    var m1, m2 runtime.MemStats
    runtime.ReadMemStats(&m1)
    
    start := time.Now()
    fn()
    duration := time.Since(start)
    
    runtime.ReadMemStats(&m2)
    allocated := m2.TotalAlloc - m1.TotalAlloc
    
    fp.mu.Lock()
    defer fp.mu.Unlock()
    
    fp.samples[name] = append(fp.samples[name], duration)
    fp.allocations[name] = append(fp.allocations[name], allocated)
}

// Profile with context and return values
func (fp *FunctionProfiler) ProfileWithResult(name string, fn func() (interface{}, error)) (interface{}, error) {
    var m1, m2 runtime.MemStats
    runtime.ReadMemStats(&m1)
    
    start := time.Now()
    result, err := fn()
    duration := time.Since(start)
    
    runtime.ReadMemStats(&m2)
    allocated := m2.TotalAlloc - m1.TotalAlloc
    
    fp.mu.Lock()
    defer fp.mu.Unlock()
    
    fp.samples[name] = append(fp.samples[name], duration)
    fp.allocations[name] = append(fp.allocations[name], allocated)
    
    return result, err
}

// Get function statistics
func (fp *FunctionProfiler) GetStats(name string) FunctionStats {
    fp.mu.RLock()
    defer fp.mu.RUnlock()
    
    samples := fp.samples[name]
    allocations := fp.allocations[name]
    
    if len(samples) == 0 {
        return FunctionStats{}
    }
    
    stats := FunctionStats{
        Name:       name,
        CallCount:  len(samples),
        TotalTime:  0,
        MinTime:    samples[0],
        MaxTime:    samples[0],
        TotalAlloc: 0,
        MinAlloc:   allocations[0],
        MaxAlloc:   allocations[0],
    }
    
    for i, sample := range samples {
        stats.TotalTime += sample
        if sample < stats.MinTime {
            stats.MinTime = sample
        }
        if sample > stats.MaxTime {
            stats.MaxTime = sample
        }
        
        alloc := allocations[i]
        stats.TotalAlloc += alloc
        if alloc < stats.MinAlloc {
            stats.MinAlloc = alloc
        }
        if alloc > stats.MaxAlloc {
            stats.MaxAlloc = alloc
        }
    }
    
    stats.AvgTime = stats.TotalTime / time.Duration(len(samples))
    stats.AvgAlloc = stats.TotalAlloc / uint64(len(allocations))
    
    return stats
}

type FunctionStats struct {
    Name       string        `json:"name"`
    CallCount  int           `json:"call_count"`
    TotalTime  time.Duration `json:"total_time"`
    AvgTime    time.Duration `json:"avg_time"`
    MinTime    time.Duration `json:"min_time"`
    MaxTime    time.Duration `json:"max_time"`
    TotalAlloc uint64        `json:"total_alloc"`
    AvgAlloc   uint64        `json:"avg_alloc"`
    MinAlloc   uint64        `json:"min_alloc"`
    MaxAlloc   uint64        `json:"max_alloc"`
}

// Benchmark runner for comparative analysis
type BenchmarkRunner struct {
    iterations int
    warmupRuns int
    profiler   *FunctionProfiler
}

func NewBenchmarkRunner(iterations, warmupRuns int) *BenchmarkRunner {
    return &BenchmarkRunner{
        iterations: iterations,
        warmupRuns: warmupRuns,
        profiler:   NewFunctionProfiler(),
    }
}

func (br *BenchmarkRunner) RunBenchmark(name string, fn func()) BenchmarkResult {
    // Warmup runs
    for i := 0; i < br.warmupRuns; i++ {
        fn()
    }
    
    // Force GC before actual benchmark
    runtime.GC()
    runtime.GC() // Second GC to ensure clean state
    
    // Actual benchmark runs
    for i := 0; i < br.iterations; i++ {
        br.profiler.Profile(name, fn)
    }
    
    stats := br.profiler.GetStats(name)
    
    return BenchmarkResult{
        Name:           name,
        Iterations:     br.iterations,
        TotalDuration:  stats.TotalTime,
        AvgDuration:    stats.AvgTime,
        MinDuration:    stats.MinTime,
        MaxDuration:    stats.MaxTime,
        TotalAllocated: stats.TotalAlloc,
        AvgAllocated:   stats.AvgAlloc,
        AllocsPerOp:    float64(stats.TotalAlloc) / float64(br.iterations),
        NanosPerOp:     float64(stats.TotalTime.Nanoseconds()) / float64(br.iterations),
    }
}

type BenchmarkResult struct {
    Name           string        `json:"name"`
    Iterations     int           `json:"iterations"`
    TotalDuration  time.Duration `json:"total_duration"`
    AvgDuration    time.Duration `json:"avg_duration"`
    MinDuration    time.Duration `json:"min_duration"`
    MaxDuration    time.Duration `json:"max_duration"`
    TotalAllocated uint64        `json:"total_allocated"`
    AvgAllocated   uint64        `json:"avg_allocated"`
    AllocsPerOp    float64       `json:"allocs_per_op"`
    NanosPerOp     float64       `json:"nanos_per_op"`
}

func (br BenchmarkResult) String() string {
    return fmt.Sprintf(
        "Benchmark %s: %d iterations, %.2f ns/op, %.2f allocs/op, avg: %v",
        br.Name,
        br.Iterations,
        br.NanosPerOp,
        br.AllocsPerOp,
        br.AvgDuration,
    )
}

// Memory leak detector
type MemoryLeakDetector struct {
    initialStats runtime.MemStats
    samples      []runtime.MemStats
    threshold    uint64 // Memory growth threshold in bytes
    interval     time.Duration
    stopChan     chan struct{}
    mu           sync.RWMutex
}

func NewMemoryLeakDetector(threshold uint64, interval time.Duration) *MemoryLeakDetector {
    detector := &MemoryLeakDetector{
        threshold: threshold,
        interval:  interval,
        stopChan:  make(chan struct{}),
    }
    
    runtime.ReadMemStats(&detector.initialStats)
    return detector
}

func (mld *MemoryLeakDetector) StartMonitoring() {
    go mld.monitor()
}

func (mld *MemoryLeakDetector) monitor() {
    ticker := time.NewTicker(mld.interval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            var stats runtime.MemStats
            runtime.ReadMemStats(&stats)
            
            mld.mu.Lock()
            mld.samples = append(mld.samples, stats)
            
            // Check for potential memory leak
            growth := stats.HeapInuse - mld.initialStats.HeapInuse
            if growth > mld.threshold {
                fmt.Printf("MEMORY LEAK DETECTED: Heap growth: %d bytes (threshold: %d)\n", 
                    growth, mld.threshold)
                mld.analyzeMemoryPattern()
            }
            
            // Keep only last 100 samples
            if len(mld.samples) > 100 {
                mld.samples = mld.samples[1:]
            }
            mld.mu.Unlock()
            
        case <-mld.stopChan:
            return
        }
    }
}

func (mld *MemoryLeakDetector) analyzeMemoryPattern() {
    if len(mld.samples) < 5 {
        return
    }
    
    // Check if memory is consistently growing
    recentSamples := mld.samples[len(mld.samples)-5:]
    consistentGrowth := true
    
    for i := 1; i < len(recentSamples); i++ {
        if recentSamples[i].HeapInuse <= recentSamples[i-1].HeapInuse {
            consistentGrowth = false
            break
        }
    }
    
    if consistentGrowth {
        fmt.Println("WARNING: Consistent memory growth detected - possible memory leak")
        fmt.Printf("Heap in use trend: %d -> %d bytes\n", 
            recentSamples[0].HeapInuse, recentSamples[len(recentSamples)-1].HeapInuse)
    }
}

func (mld *MemoryLeakDetector) Stop() {
    close(mld.stopChan)
}

func (mld *MemoryLeakDetector) GetReport() MemoryReport {
    mld.mu.RLock()
    defer mld.mu.RUnlock()
    
    if len(mld.samples) == 0 {
        return MemoryReport{}
    }
    
    latest := mld.samples[len(mld.samples)-1]
    
    return MemoryReport{
        InitialHeapInuse: mld.initialStats.HeapInuse,
        CurrentHeapInuse: latest.HeapInuse,
        HeapGrowth:       latest.HeapInuse - mld.initialStats.HeapInuse,
        TotalAllocations: latest.TotalAlloc - mld.initialStats.TotalAlloc,
        GCCycles:         latest.NumGC - mld.initialStats.NumGC,
        SampleCount:      len(mld.samples),
    }
}

type MemoryReport struct {
    InitialHeapInuse uint64 `json:"initial_heap_inuse"`
    CurrentHeapInuse uint64 `json:"current_heap_inuse"`
    HeapGrowth       uint64 `json:"heap_growth"`
    TotalAllocations uint64 `json:"total_allocations"`
    GCCycles         uint32 `json:"gc_cycles"`
    SampleCount      int    `json:"sample_count"`
}
```

---

## 🧠 **Memory Optimization**

### **Advanced Memory Management**

```go
package memory

import (
    "reflect"
    "runtime"
    "sync"
    "unsafe"
)

// Object Pool for reducing GC pressure
type ObjectPool struct {
    pool    sync.Pool
    newFunc func() interface{}
    resetFunc func(interface{})
    stats   PoolStats
    mu      sync.RWMutex
}

type PoolStats struct {
    Gets     uint64 `json:"gets"`
    Puts     uint64 `json:"puts"`
    News     uint64 `json:"news"`
    Hits     uint64 `json:"hits"`
    Misses   uint64 `json:"misses"`
}

func NewObjectPool(newFunc func() interface{}, resetFunc func(interface{})) *ObjectPool {
    op := &ObjectPool{
        newFunc:   newFunc,
        resetFunc: resetFunc,
    }
    
    op.pool = sync.Pool{
        New: func() interface{} {
            op.mu.Lock()
            op.stats.News++
            op.mu.Unlock()
            return newFunc()
        },
    }
    
    return op
}

func (op *ObjectPool) Get() interface{} {
    obj := op.pool.Get()
    
    op.mu.Lock()
    op.stats.Gets++
    if obj != nil {
        op.stats.Hits++
    } else {
        op.stats.Misses++
    }
    op.mu.Unlock()
    
    return obj
}

func (op *ObjectPool) Put(obj interface{}) {
    if op.resetFunc != nil {
        op.resetFunc(obj)
    }
    
    op.pool.Put(obj)
    
    op.mu.Lock()
    op.stats.Puts++
    op.mu.Unlock()
}

func (op *ObjectPool) GetStats() PoolStats {
    op.mu.RLock()
    defer op.mu.RUnlock()
    return op.stats
}

// Memory arena for batch allocations
type MemoryArena struct {
    data     []byte
    offset   int
    size     int
    mu       sync.Mutex
}

func NewMemoryArena(size int) *MemoryArena {
    return &MemoryArena{
        data: make([]byte, size),
        size: size,
    }
}

func (ma *MemoryArena) Allocate(size int) []byte {
    ma.mu.Lock()
    defer ma.mu.Unlock()
    
    if ma.offset+size > ma.size {
        return nil // Arena full
    }
    
    slice := ma.data[ma.offset : ma.offset+size : ma.offset+size]
    ma.offset += size
    
    return slice
}

func (ma *MemoryArena) Reset() {
    ma.mu.Lock()
    defer ma.mu.Unlock()
    ma.offset = 0
}

func (ma *MemoryArena) Available() int {
    ma.mu.Lock()
    defer ma.mu.Unlock()
    return ma.size - ma.offset
}

// String interning to reduce memory usage
type StringInterner struct {
    strings map[string]string
    mu      sync.RWMutex
    stats   InternStats
}

type InternStats struct {
    TotalStrings    int `json:"total_strings"`
    UniqueStrings   int `json:"unique_strings"`
    MemorySaved     int `json:"memory_saved"`
    DuplicateCount  int `json:"duplicate_count"`
}

func NewStringInterner() *StringInterner {
    return &StringInterner{
        strings: make(map[string]string),
    }
}

func (si *StringInterner) Intern(s string) string {
    si.mu.RLock()
    if interned, exists := si.strings[s]; exists {
        si.mu.RUnlock()
        
        si.mu.Lock()
        si.stats.TotalStrings++
        si.stats.DuplicateCount++
        si.stats.MemorySaved += len(s)
        si.mu.Unlock()
        
        return interned
    }
    si.mu.RUnlock()
    
    si.mu.Lock()
    defer si.mu.Unlock()
    
    // Double-check after acquiring write lock
    if interned, exists := si.strings[s]; exists {
        si.stats.TotalStrings++
        si.stats.DuplicateCount++
        si.stats.MemorySaved += len(s)
        return interned
    }
    
    si.strings[s] = s
    si.stats.TotalStrings++
    si.stats.UniqueStrings++
    
    return s
}

func (si *StringInterner) GetStats() InternStats {
    si.mu.RLock()
    defer si.mu.RUnlock()
    return si.stats
}

// Memory-efficient data structures
type CompactSlice struct {
    data     unsafe.Pointer
    len      int
    cap      int
    elemSize uintptr
}

func NewCompactSlice(elemType reflect.Type, capacity int) *CompactSlice {
    elemSize := elemType.Size()
    data := make([]byte, capacity*int(elemSize))
    
    return &CompactSlice{
        data:     unsafe.Pointer(&data[0]),
        cap:      capacity,
        elemSize: elemSize,
    }
}

func (cs *CompactSlice) Get(index int) unsafe.Pointer {
    if index >= cs.len {
        return nil
    }
    
    offset := uintptr(index) * cs.elemSize
    return unsafe.Pointer(uintptr(cs.data) + offset)
}

func (cs *CompactSlice) Append(elem unsafe.Pointer) bool {
    if cs.len >= cs.cap {
        return false // No more capacity
    }
    
    offset := uintptr(cs.len) * cs.elemSize
    dst := unsafe.Pointer(uintptr(cs.data) + offset)
    
    // Copy element data
    for i := uintptr(0); i < cs.elemSize; i++ {
        *(*byte)(unsafe.Pointer(uintptr(dst) + i)) = *(*byte)(unsafe.Pointer(uintptr(elem) + i))
    }
    
    cs.len++
    return true
}

func (cs *CompactSlice) Len() int {
    return cs.len
}

// Memory usage analyzer
type MemoryAnalyzer struct {
    allocTracker map[string]*AllocationInfo
    mu           sync.RWMutex
}

type AllocationInfo struct {
    Count     int64  `json:"count"`
    TotalSize int64  `json:"total_size"`
    AvgSize   int64  `json:"avg_size"`
    MaxSize   int64  `json:"max_size"`
    MinSize   int64  `json:"min_size"`
}

func NewMemoryAnalyzer() *MemoryAnalyzer {
    return &MemoryAnalyzer{
        allocTracker: make(map[string]*AllocationInfo),
    }
}

func (ma *MemoryAnalyzer) TrackAllocation(typeName string, size int64) {
    ma.mu.Lock()
    defer ma.mu.Unlock()
    
    info, exists := ma.allocTracker[typeName]
    if !exists {
        info = &AllocationInfo{
            MinSize: size,
        }
        ma.allocTracker[typeName] = info
    }
    
    info.Count++
    info.TotalSize += size
    info.AvgSize = info.TotalSize / info.Count
    
    if size > info.MaxSize {
        info.MaxSize = size
    }
    if size < info.MinSize {
        info.MinSize = size
    }
}

func (ma *MemoryAnalyzer) GetTopAllocators(limit int) []AllocationSummary {
    ma.mu.RLock()
    defer ma.mu.RUnlock()
    
    var summaries []AllocationSummary
    for typeName, info := range ma.allocTracker {
        summaries = append(summaries, AllocationSummary{
            TypeName:    typeName,
            Count:       info.Count,
            TotalSize:   info.TotalSize,
            AvgSize:     info.AvgSize,
        })
    }
    
    // Sort by total size
    for i := 0; i < len(summaries)-1; i++ {
        for j := i + 1; j < len(summaries); j++ {
            if summaries[i].TotalSize < summaries[j].TotalSize {
                summaries[i], summaries[j] = summaries[j], summaries[i]
            }
        }
    }
    
    if limit > 0 && len(summaries) > limit {
        summaries = summaries[:limit]
    }
    
    return summaries
}

type AllocationSummary struct {
    TypeName    string `json:"type_name"`
    Count       int64  `json:"count"`
    TotalSize   int64  `json:"total_size"`
    AvgSize     int64  `json:"avg_size"`
}

// Zero-copy string operations
type ZeroCopyString struct {
    data []byte
}

func NewZeroCopyString(data []byte) ZeroCopyString {
    return ZeroCopyString{data: data}
}

func (zcs ZeroCopyString) String() string {
    return *(*string)(unsafe.Pointer(&zcs.data))
}

func (zcs ZeroCopyString) Substring(start, end int) ZeroCopyString {
    if start < 0 || end > len(zcs.data) || start > end {
        return ZeroCopyString{}
    }
    
    return ZeroCopyString{data: zcs.data[start:end]}
}

func (zcs ZeroCopyString) Len() int {
    return len(zcs.data)
}

// Memory pool for specific types
type TypedPool[T any] struct {
    pool      sync.Pool
    newFunc   func() *T
    resetFunc func(*T)
    stats     PoolStats
    mu        sync.RWMutex
}

func NewTypedPool[T any](newFunc func() *T, resetFunc func(*T)) *TypedPool[T] {
    tp := &TypedPool[T]{
        newFunc:   newFunc,
        resetFunc: resetFunc,
    }
    
    tp.pool = sync.Pool{
        New: func() interface{} {
            tp.mu.Lock()
            tp.stats.News++
            tp.mu.Unlock()
            return newFunc()
        },
    }
    
    return tp
}

func (tp *TypedPool[T]) Get() *T {
    obj := tp.pool.Get().(*T)
    
    tp.mu.Lock()
    tp.stats.Gets++
    tp.stats.Hits++
    tp.mu.Unlock()
    
    return obj
}

func (tp *TypedPool[T]) Put(obj *T) {
    if tp.resetFunc != nil {
        tp.resetFunc(obj)
    }
    
    tp.pool.Put(obj)
    
    tp.mu.Lock()
    tp.stats.Puts++
    tp.mu.Unlock()
}

func (tp *TypedPool[T]) GetStats() PoolStats {
    tp.mu.RLock()
    defer tp.mu.RUnlock()
    return tp.stats
}

// GC optimization utilities
type GCOptimizer struct {
    gcPercent    int
    memLimit     int64
    gcStats      runtime.MemStats
    gcHistory    []GCEvent
    mu           sync.Mutex
}

type GCEvent struct {
    Timestamp    time.Time     `json:"timestamp"`
    PauseDuration time.Duration `json:"pause_duration"`
    HeapSize     uint64        `json:"heap_size"`
    GCCycle      uint32        `json:"gc_cycle"`
}

func NewGCOptimizer() *GCOptimizer {
    return &GCOptimizer{
        gcPercent: runtime.GOMAXPROCS(0) * 100, // Default GC target
    }
}

func (gco *GCOptimizer) OptimizeGC() {
    runtime.ReadMemStats(&gco.gcStats)
    
    // Adaptive GC tuning based on allocation rate
    if gco.gcStats.NumGC > 0 {
        allocRate := float64(gco.gcStats.TotalAlloc) / float64(gco.gcStats.NumGC)
        
        if allocRate > 100*1024*1024 { // High allocation rate
            debug.SetGCPercent(50) // More aggressive GC
        } else if allocRate < 10*1024*1024 { // Low allocation rate
            debug.SetGCPercent(200) // Less aggressive GC
        } else {
            debug.SetGCPercent(100) // Default
        }
    }
}

func (gco *GCOptimizer) RecordGCEvent() {
    gco.mu.Lock()
    defer gco.mu.Unlock()
    
    var stats runtime.MemStats
    runtime.ReadMemStats(&stats)
    
    if len(gco.gcHistory) == 0 || stats.NumGC > gco.gcHistory[len(gco.gcHistory)-1].GCCycle {
        event := GCEvent{
            Timestamp:     time.Now(),
            PauseDuration: time.Duration(stats.PauseNs[(stats.NumGC+255)%256]),
            HeapSize:      stats.HeapInuse,
            GCCycle:       stats.NumGC,
        }
        
        gco.gcHistory = append(gco.gcHistory, event)
        
        // Keep only last 100 events
        if len(gco.gcHistory) > 100 {
            gco.gcHistory = gco.gcHistory[1:]
        }
    }
}

func (gco *GCOptimizer) GetGCMetrics() GCMetrics {
    gco.mu.Lock()
    defer gco.mu.Unlock()
    
    if len(gco.gcHistory) == 0 {
        return GCMetrics{}
    }
    
    var totalPause time.Duration
    var maxPause time.Duration
    minPause := gco.gcHistory[0].PauseDuration
    
    for _, event := range gco.gcHistory {
        totalPause += event.PauseDuration
        if event.PauseDuration > maxPause {
            maxPause = event.PauseDuration
        }
        if event.PauseDuration < minPause {
            minPause = event.PauseDuration
        }
    }
    
    return GCMetrics{
        EventCount:   len(gco.gcHistory),
        TotalPause:   totalPause,
        AveragePause: totalPause / time.Duration(len(gco.gcHistory)),
        MaxPause:     maxPause,
        MinPause:     minPause,
    }
}

type GCMetrics struct {
    EventCount   int           `json:"event_count"`
    TotalPause   time.Duration `json:"total_pause"`
    AveragePause time.Duration `json:"average_pause"`
    MaxPause     time.Duration `json:"max_pause"`
    MinPause     time.Duration `json:"min_pause"`
}
```

---

## 🚀 **CPU Optimization**

### **CPU Profiling and Optimization**

```go
package cpu

import (
    "context"
    "runtime"
    "sync"
    "sync/atomic"
    "time"
)

// CPU-bound task optimizer
type CPUOptimizer struct {
    numWorkers    int
    taskQueue     chan Task
    workerPool    []Worker
    stats         CPUStats
    stopChan      chan struct{}
    wg            sync.WaitGroup
}

type Task interface {
    Execute() (interface{}, error)
    GetPriority() int
    GetEstimatedDuration() time.Duration
}

type Worker struct {
    id       int
    tasksCh  chan Task
    resultCh chan TaskResult
    stats    WorkerStats
}

type TaskResult struct {
    Result   interface{}
    Error    error
    Duration time.Duration
    WorkerID int
}

type CPUStats struct {
    TasksProcessed   uint64        `json:"tasks_processed"`
    TotalCPUTime     time.Duration `json:"total_cpu_time"`
    AverageCPUTime   time.Duration `json:"average_cpu_time"`
    ActiveWorkers    int32         `json:"active_workers"`
    QueueSize        int32         `json:"queue_size"`
    ThroughputPerSec float64       `json:"throughput_per_sec"`
}

type WorkerStats struct {
    TasksCompleted uint64        `json:"tasks_completed"`
    TotalTime      time.Duration `json:"total_time"`
    IdleTime       time.Duration `json:"idle_time"`
    ErrorCount     uint64        `json:"error_count"`
}

func NewCPUOptimizer(numWorkers int, queueSize int) *CPUOptimizer {
    if numWorkers <= 0 {
        numWorkers = runtime.NumCPU()
    }
    
    optimizer := &CPUOptimizer{
        numWorkers: numWorkers,
        taskQueue:  make(chan Task, queueSize),
        workerPool: make([]Worker, numWorkers),
        stopChan:   make(chan struct{}),
    }
    
    return optimizer
}

func (co *CPUOptimizer) Start() {
    for i := 0; i < co.numWorkers; i++ {
        worker := Worker{
            id:       i,
            tasksCh:  make(chan Task, 10),
            resultCh: make(chan TaskResult, 10),
        }
        
        co.workerPool[i] = worker
        co.wg.Add(1)
        go co.runWorker(&worker)
    }
    
    // Start task dispatcher
    co.wg.Add(1)
    go co.dispatchTasks()
    
    // Start metrics collector
    go co.collectMetrics()
}

func (co *CPUOptimizer) runWorker(worker *Worker) {
    defer co.wg.Done()
    
    for {
        select {
        case task := <-worker.tasksCh:
            atomic.AddInt32(&co.stats.ActiveWorkers, 1)
            
            start := time.Now()
            result, err := task.Execute()
            duration := time.Since(start)
            
            atomic.AddInt32(&co.stats.ActiveWorkers, -1)
            atomic.AddUint64(&co.stats.TasksProcessed, 1)
            
            // Update worker stats
            worker.stats.TasksCompleted++
            worker.stats.TotalTime += duration
            if err != nil {
                worker.stats.ErrorCount++
            }
            
            // Send result if there's a result channel
            select {
            case worker.resultCh <- TaskResult{
                Result:   result,
                Error:    err,
                Duration: duration,
                WorkerID: worker.id,
            }:
            default:
                // Drop result if channel is full
            }
            
        case <-co.stopChan:
            return
        }
    }
}

func (co *CPUOptimizer) dispatchTasks() {
    defer co.wg.Done()
    
    for {
        select {
        case task := <-co.taskQueue:
            // Simple round-robin dispatch
            // In practice, you might want load-based or priority-based dispatch
            workerIndex := int(atomic.LoadUint64(&co.stats.TasksProcessed)) % co.numWorkers
            
            select {
            case co.workerPool[workerIndex].tasksCh <- task:
                atomic.AddInt32(&co.stats.QueueSize, -1)
            case <-time.After(100 * time.Millisecond):
                // Redistribute to any available worker
                for i := 0; i < co.numWorkers; i++ {
                    select {
                    case co.workerPool[i].tasksCh <- task:
                        atomic.AddInt32(&co.stats.QueueSize, -1)
                        goto dispatched
                    default:
                        continue
                    }
                }
                // If no worker available, put back in queue
                select {
                case co.taskQueue <- task:
                default:
                    // Queue full, drop task
                }
                dispatched:
            }
            
        case <-co.stopChan:
            return
        }
    }
}

func (co *CPUOptimizer) collectMetrics() {
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()
    
    lastTaskCount := uint64(0)
    
    for {
        select {
        case <-ticker.C:
            currentTasks := atomic.LoadUint64(&co.stats.TasksProcessed)
            co.stats.ThroughputPerSec = float64(currentTasks - lastTaskCount)
            lastTaskCount = currentTasks
            
        case <-co.stopChan:
            return
        }
    }
}

func (co *CPUOptimizer) SubmitTask(task Task) error {
    select {
    case co.taskQueue <- task:
        atomic.AddInt32(&co.stats.QueueSize, 1)
        return nil
    default:
        return fmt.Errorf("task queue full")
    }
}

func (co *CPUOptimizer) GetStats() CPUStats {
    return CPUStats{
        TasksProcessed:   atomic.LoadUint64(&co.stats.TasksProcessed),
        ActiveWorkers:    atomic.LoadInt32(&co.stats.ActiveWorkers),
        QueueSize:        atomic.LoadInt32(&co.stats.QueueSize),
        ThroughputPerSec: co.stats.ThroughputPerSec,
    }
}

func (co *CPUOptimizer) Stop() {
    close(co.stopChan)
    co.wg.Wait()
}

// SIMD-like operations for batch processing
type BatchProcessor struct {
    batchSize int
    processor func([]interface{}) []interface{}
}

func NewBatchProcessor(batchSize int, processor func([]interface{}) []interface{}) *BatchProcessor {
    return &BatchProcessor{
        batchSize: batchSize,
        processor: processor,
    }
}

func (bp *BatchProcessor) ProcessStream(input <-chan interface{}, output chan<- interface{}) {
    batch := make([]interface{}, 0, bp.batchSize)
    
    for item := range input {
        batch = append(batch, item)
        
        if len(batch) >= bp.batchSize {
            results := bp.processor(batch)
            for _, result := range results {
                output <- result
            }
            batch = batch[:0] // Reset batch
        }
    }
    
    // Process remaining items
    if len(batch) > 0 {
        results := bp.processor(batch)
        for _, result := range results {
            output <- result
        }
    }
    
    close(output)
}

// Cache-friendly data access patterns
type CacheFriendlyProcessor struct {
    blockSize    int
    dataSize     int
    processorFunc func([]float64) []float64
}

func NewCacheFriendlyProcessor(blockSize int) *CacheFriendlyProcessor {
    return &CacheFriendlyProcessor{
        blockSize: blockSize,
    }
}

// Process data in cache-friendly blocks
func (cfp *CacheFriendlyProcessor) ProcessMatrix(matrix [][]float64) [][]float64 {
    rows := len(matrix)
    cols := len(matrix[0])
    result := make([][]float64, rows)
    
    for i := range result {
        result[i] = make([]float64, cols)
    }
    
    // Process in blocks to improve cache locality
    for blockRow := 0; blockRow < rows; blockRow += cfp.blockSize {
        for blockCol := 0; blockCol < cols; blockCol += cfp.blockSize {
            maxRow := min(blockRow+cfp.blockSize, rows)
            maxCol := min(blockCol+cfp.blockSize, cols)
            
            // Process block
            for i := blockRow; i < maxRow; i++ {
                for j := blockCol; j < maxCol; j++ {
                    result[i][j] = cfp.processElement(matrix[i][j])
                }
            }
        }
    }
    
    return result
}

func (cfp *CacheFriendlyProcessor) processElement(value float64) float64 {
    // Example processing - would be replaced with actual computation
    return value * 2.0
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// CPU affinity and NUMA awareness
type NUMAOptimizer struct {
    numNodes     int
    coresPerNode int
    nodeWorkers  [][]Worker
}

func NewNUMAOptimizer() *NUMAOptimizer {
    // In practice, you'd detect actual NUMA topology
    return &NUMAOptimizer{
        numNodes:     2, // Assume 2 NUMA nodes
        coresPerNode: runtime.NumCPU() / 2,
    }
}

func (no *NUMAOptimizer) OptimizeForNUMA(tasks []Task) {
    // Distribute tasks across NUMA nodes based on memory locality
    for i, task := range tasks {
        nodeIndex := i % no.numNodes
        // Assign task to specific NUMA node
        no.assignToNode(task, nodeIndex)
    }
}

func (no *NUMAOptimizer) assignToNode(task Task, nodeIndex int) {
    // Implementation would use OS-specific APIs to set CPU affinity
    // This is a simplified example
    runtime.LockOSThread()
    defer runtime.UnlockOSThread()
    
    // Execute task on specific NUMA node
    task.Execute()
}
```

---

## 💾 **I/O Optimization**

### **Asynchronous I/O and Buffering**

```go
package io

import (
    "bufio"
    "context"
    "fmt"
    "io"
    "os"
    "sync"
    "sync/atomic"
    "time"
)

// Async I/O Manager
type AsyncIOManager struct {
    readPool     sync.Pool
    writePool    sync.Pool
    bufferSize   int
    maxConcurrency int
    semaphore    chan struct{}
    stats        IOStats
}

type IOStats struct {
    TotalReads     uint64        `json:"total_reads"`
    TotalWrites    uint64        `json:"total_writes"`
    BytesRead      uint64        `json:"bytes_read"`
    BytesWritten   uint64        `json:"bytes_written"`
    AvgReadTime    time.Duration `json:"avg_read_time"`
    AvgWriteTime   time.Duration `json:"avg_write_time"`
    ErrorCount     uint64        `json:"error_count"`
    ActiveOperations int32       `json:"active_operations"`
}

func NewAsyncIOManager(bufferSize, maxConcurrency int) *AsyncIOManager {
    aio := &AsyncIOManager{
        bufferSize:     bufferSize,
        maxConcurrency: maxConcurrency,
        semaphore:      make(chan struct{}, maxConcurrency),
    }
    
    aio.readPool = sync.Pool{
        New: func() interface{} {
            return make([]byte, bufferSize)
        },
    }
    
    aio.writePool = sync.Pool{
        New: func() interface{} {
            return bufio.NewWriterSize(nil, bufferSize)
        },
    }
    
    return aio
}

// Async read operation
func (aio *AsyncIOManager) ReadAsync(ctx context.Context, reader io.Reader) <-chan ReadResult {
    resultCh := make(chan ReadResult, 1)
    
    go func() {
        defer close(resultCh)
        
        // Acquire semaphore
        select {
        case aio.semaphore <- struct{}{}:
            defer func() { <-aio.semaphore }()
        case <-ctx.Done():
            resultCh <- ReadResult{Error: ctx.Err()}
            return
        }
        
        atomic.AddInt32(&aio.stats.ActiveOperations, 1)
        defer atomic.AddInt32(&aio.stats.ActiveOperations, -1)
        
        start := time.Now()
        buffer := aio.readPool.Get().([]byte)
        defer aio.readPool.Put(buffer)
        
        n, err := reader.Read(buffer)
        duration := time.Since(start)
        
        // Update stats
        atomic.AddUint64(&aio.stats.TotalReads, 1)
        atomic.AddUint64(&aio.stats.BytesRead, uint64(n))
        if err != nil {
            atomic.AddUint64(&aio.stats.ErrorCount, 1)
        }
        
        // Update average read time
        aio.updateAvgReadTime(duration)
        
        result := ReadResult{
            Data:     make([]byte, n),
            BytesRead: n,
            Duration: duration,
            Error:    err,
        }
        
        if n > 0 {
            copy(result.Data, buffer[:n])
        }
        
        resultCh <- result
    }()
    
    return resultCh
}

// Async write operation
func (aio *AsyncIOManager) WriteAsync(ctx context.Context, writer io.Writer, data []byte) <-chan WriteResult {
    resultCh := make(chan WriteResult, 1)
    
    go func() {
        defer close(resultCh)
        
        // Acquire semaphore
        select {
        case aio.semaphore <- struct{}{}:
            defer func() { <-aio.semaphore }()
        case <-ctx.Done():
            resultCh <- WriteResult{Error: ctx.Err()}
            return
        }
        
        atomic.AddInt32(&aio.stats.ActiveOperations, 1)
        defer atomic.AddInt32(&aio.stats.ActiveOperations, -1)
        
        start := time.Now()
        bufferedWriter := aio.writePool.Get().(*bufio.Writer)
        bufferedWriter.Reset(writer)
        defer func() {
            bufferedWriter.Flush()
            aio.writePool.Put(bufferedWriter)
        }()
        
        n, err := bufferedWriter.Write(data)
        if err == nil {
            err = bufferedWriter.Flush()
        }
        
        duration := time.Since(start)
        
        // Update stats
        atomic.AddUint64(&aio.stats.TotalWrites, 1)
        atomic.AddUint64(&aio.stats.BytesWritten, uint64(n))
        if err != nil {
            atomic.AddUint64(&aio.stats.ErrorCount, 1)
        }
        
        aio.updateAvgWriteTime(duration)
        
        resultCh <- WriteResult{
            BytesWritten: n,
            Duration:     duration,
            Error:        err,
        }
    }()
    
    return resultCh
}

func (aio *AsyncIOManager) updateAvgReadTime(duration time.Duration) {
    // Simple exponential moving average
    currentAvg := time.Duration(atomic.LoadInt64((*int64)(&aio.stats.AvgReadTime)))
    newAvg := (currentAvg*9 + duration) / 10
    atomic.StoreInt64((*int64)(&aio.stats.AvgReadTime), int64(newAvg))
}

func (aio *AsyncIOManager) updateAvgWriteTime(duration time.Duration) {
    currentAvg := time.Duration(atomic.LoadInt64((*int64)(&aio.stats.AvgWriteTime)))
    newAvg := (currentAvg*9 + duration) / 10
    atomic.StoreInt64((*int64)(&aio.stats.AvgWriteTime), int64(newAvg))
}

type ReadResult struct {
    Data      []byte
    BytesRead int
    Duration  time.Duration
    Error     error
}

type WriteResult struct {
    BytesWritten int
    Duration     time.Duration
    Error        error
}

// High-performance file operations
type OptimizedFileManager struct {
    directIO     bool
    bufferSize   int
    readAhead    int
    writeBuffer  []byte
    writeOffset  int
    mu           sync.Mutex
}

func NewOptimizedFileManager(bufferSize int, directIO bool) *OptimizedFileManager {
    return &OptimizedFileManager{
        directIO:    directIO,
        bufferSize:  bufferSize,
        writeBuffer: make([]byte, bufferSize),
    }
}

// Sequential read with prefetching
func (ofm *OptimizedFileManager) SequentialRead(filename string, chunkSize int) (<-chan []byte, <-chan error) {
    dataCh := make(chan []byte, 10)
    errorCh := make(chan error, 1)
    
    go func() {
        defer close(dataCh)
        defer close(errorCh)
        
        file, err := os.OpenFile(filename, os.O_RDONLY, 0644)
        if err != nil {
            errorCh <- err
            return
        }
        defer file.Close()
        
        // Enable read-ahead if supported
        if ofm.readAhead > 0 {
            // Platform-specific read-ahead optimization would go here
        }
        
        reader := bufio.NewReaderSize(file, ofm.bufferSize)
        buffer := make([]byte, chunkSize)
        
        for {
            n, err := reader.Read(buffer)
            if n > 0 {
                chunk := make([]byte, n)
                copy(chunk, buffer[:n])
                dataCh <- chunk
            }
            
            if err == io.EOF {
                break
            }
            if err != nil {
                errorCh <- err
                break
            }
        }
    }()
    
    return dataCh, errorCh
}

// Batched write operations
func (ofm *OptimizedFileManager) BatchWrite(filename string, data []byte) error {
    ofm.mu.Lock()
    defer ofm.mu.Unlock()
    
    // Check if we need to flush the buffer
    if len(ofm.writeBuffer)-ofm.writeOffset < len(data) {
        if err := ofm.flushWriteBuffer(filename); err != nil {
            return err
        }
    }
    
    // Add data to buffer
    copy(ofm.writeBuffer[ofm.writeOffset:], data)
    ofm.writeOffset += len(data)
    
    return nil
}

func (ofm *OptimizedFileManager) flushWriteBuffer(filename string) error {
    if ofm.writeOffset == 0 {
        return nil
    }
    
    file, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
    if err != nil {
        return err
    }
    defer file.Close()
    
    writer := bufio.NewWriterSize(file, ofm.bufferSize)
    defer writer.Flush()
    
    _, err = writer.Write(ofm.writeBuffer[:ofm.writeOffset])
    ofm.writeOffset = 0
    
    return err
}

// Memory-mapped file operations
type MMapFileManager struct {
    files map[string]*MMapFile
    mu    sync.RWMutex
}

type MMapFile struct {
    data   []byte
    size   int64
    offset int64
    mu     sync.RWMutex
}

func NewMMapFileManager() *MMapFileManager {
    return &MMapFileManager{
        files: make(map[string]*MMapFile),
    }
}

// This is a simplified example - actual implementation would use syscalls
func (mmfm *MMapFileManager) OpenMMap(filename string) error {
    mmfm.mu.Lock()
    defer mmfm.mu.Unlock()
    
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    stat, err := file.Stat()
    if err != nil {
        return err
    }
    
    // In practice, you'd use mmap syscall here
    data := make([]byte, stat.Size())
    _, err = file.Read(data)
    if err != nil {
        return err
    }
    
    mmfm.files[filename] = &MMapFile{
        data: data,
        size: stat.Size(),
    }
    
    return nil
}

func (mmfm *MMapFileManager) Read(filename string, offset int64, length int) ([]byte, error) {
    mmfm.mu.RLock()
    defer mmfm.mu.RUnlock()
    
    mmapFile, exists := mmfm.files[filename]
    if !exists {
        return nil, fmt.Errorf("file not mapped: %s", filename)
    }
    
    mmapFile.mu.RLock()
    defer mmapFile.mu.RUnlock()
    
    if offset+int64(length) > mmapFile.size {
        return nil, fmt.Errorf("read beyond file size")
    }
    
    result := make([]byte, length)
    copy(result, mmapFile.data[offset:offset+int64(length)])
    
    return result, nil
}

// Network I/O optimization
type NetworkOptimizer struct {
    connPool    sync.Pool
    bufferPool  sync.Pool
    keepAlive   time.Duration
    readTimeout time.Duration
    writeTimeout time.Duration
}

func NewNetworkOptimizer(keepAlive, readTimeout, writeTimeout time.Duration) *NetworkOptimizer {
    no := &NetworkOptimizer{
        keepAlive:    keepAlive,
        readTimeout:  readTimeout,
        writeTimeout: writeTimeout,
    }
    
    no.bufferPool = sync.Pool{
        New: func() interface{} {
            return make([]byte, 32768) // 32KB buffer
        },
    }
    
    return no
}

func (no *NetworkOptimizer) OptimizeConnection(conn net.Conn) error {
    if tcpConn, ok := conn.(*net.TCPConn); ok {
        // Enable TCP_NODELAY to reduce latency
        tcpConn.SetNoDelay(true)
        
        // Set keep-alive
        tcpConn.SetKeepAlive(true)
        tcpConn.SetKeepAlivePeriod(no.keepAlive)
        
        // Set buffer sizes
        tcpConn.SetReadBuffer(65536)  // 64KB
        tcpConn.SetWriteBuffer(65536) // 64KB
    }
    
    return nil
}

func (no *NetworkOptimizer) BufferedRead(conn net.Conn) ([]byte, error) {
    buffer := no.bufferPool.Get().([]byte)
    defer no.bufferPool.Put(buffer)
    
    conn.SetReadDeadline(time.Now().Add(no.readTimeout))
    defer conn.SetReadDeadline(time.Time{})
    
    n, err := conn.Read(buffer)
    if err != nil {
        return nil, err
    }
    
    result := make([]byte, n)
    copy(result, buffer[:n])
    
    return result, nil
}
```

---

## ❓ **Interview Questions**

### **Advanced Performance Engineering Questions**

#### **1. Memory Profiling and Optimization**

**Q: How would you identify and fix memory leaks in a Go application?**

**A: Comprehensive memory leak detection approach:**

```go
// Memory leak detection strategy
func DetectMemoryLeaks() {
    // 1. Enable memory profiling
    profiler := NewPerformanceProfiler("./profiles")
    profiler.StartProfiling()
    defer profiler.StopProfiling()
    
    // 2. Use heap profiling to identify allocations
    var m1, m2 runtime.MemStats
    runtime.ReadMemStats(&m1)
    
    // Run suspicious code
    runSuspiciousCode()
    
    runtime.GC()
    runtime.ReadMemStats(&m2)
    
    // 3. Analyze memory growth
    if m2.HeapInuse > m1.HeapInuse*2 {
        fmt.Printf("Potential memory leak detected: %d -> %d bytes\n", 
            m1.HeapInuse, m2.HeapInuse)
    }
    
    // 4. Use object pools for frequent allocations
    var bufferPool = sync.Pool{
        New: func() interface{} {
            return make([]byte, 1024)
        },
    }
    
    // 5. Monitor goroutine leaks
    if runtime.NumGoroutine() > 1000 {
        fmt.Println("Potential goroutine leak detected")
    }
}

// Common memory leak patterns and fixes
func FixCommonLeaks() {
    // Fix 1: Slice leaks - reslice to avoid holding references
    largeSlice := make([]byte, 1000000)
    smallSlice := largeSlice[0:10]
    // BAD: smallSlice holds reference to entire largeSlice
    
    // GOOD: Copy to new slice
    fixedSlice := make([]byte, 10)
    copy(fixedSlice, largeSlice[0:10])
    
    // Fix 2: Timer leaks - always stop timers
    timer := time.NewTimer(time.Hour)
    defer timer.Stop() // IMPORTANT: Always stop timers
    
    // Fix 3: Channel leaks - close channels
    ch := make(chan int, 10)
    defer close(ch) // IMPORTANT: Close channels
}
```

#### **2. CPU Optimization Strategies**

**Q: How would you optimize CPU-bound operations in a multi-core environment?**

**A: Multi-core CPU optimization techniques:**

```go
// CPU optimization for multi-core systems
func OptimizeCPUUsage() {
    numCPU := runtime.NumCPU()
    runtime.GOMAXPROCS(numCPU) // Use all available cores
    
    // 1. Parallel processing with worker pools
    optimizer := NewCPUOptimizer(numCPU, 1000)
    optimizer.Start()
    defer optimizer.Stop()
    
    // 2. Cache-friendly data access patterns
    data := generateLargeDataset()
    processInCacheFriendlyBlocks(data, 64) // 64-element blocks
    
    // 3. Reduce context switching
    batchProcessor := NewBatchProcessor(100, processBatch)
    
    // 4. Use CPU affinity for NUMA systems
    optimizeForNUMA(tasks)
    
    // 5. Profile CPU usage
    // go tool pprof cpu.prof
}

// Cache-friendly processing
func processInCacheFriendlyBlocks(data [][]float64, blockSize int) {
    rows, cols := len(data), len(data[0])
    
    for i := 0; i < rows; i += blockSize {
        for j := 0; j < cols; j += blockSize {
            // Process block to improve cache locality
            processBlock(data, i, j, blockSize)
        }
    }
}

// Avoid false sharing in multi-threaded operations
type CacheLinePadded struct {
    value uint64
    _     [56]byte // Padding to avoid false sharing (64-byte cache line)
}
```

#### **3. I/O Performance Optimization**

**Q: Design a high-performance I/O system for handling millions of concurrent operations.**

**A: High-performance I/O architecture:**

```go
// High-performance I/O system design
type HighPerformanceIOSystem struct {
    asyncManager    *AsyncIOManager
    connectionPool  *ConnectionPool
    bufferManager   *BufferManager
    ioScheduler     *IOScheduler
}

// Connection pooling for network I/O
type ConnectionPool struct {
    idle     chan net.Conn
    active   map[net.Conn]bool
    factory  func() (net.Conn, error)
    maxIdle  int
    maxActive int
    mu       sync.RWMutex
}

func (cp *ConnectionPool) Get() (net.Conn, error) {
    cp.mu.Lock()
    defer cp.mu.Unlock()
    
    // Try to get idle connection
    select {
    case conn := <-cp.idle:
        cp.active[conn] = true
        return conn, nil
    default:
        // Create new connection if under limit
        if len(cp.active) < cp.maxActive {
            conn, err := cp.factory()
            if err != nil {
                return nil, err
            }
            cp.active[conn] = true
            return conn, nil
        }
        return nil, errors.New("connection pool exhausted")
    }
}

// I/O request scheduling
type IOScheduler struct {
    readQueue    PriorityQueue
    writeQueue   PriorityQueue
    workers      []IOWorker
    scheduler    chan IORequest
}

type IORequest struct {
    Priority    int
    Type        IOType
    Data        []byte
    Callback    func(result IOResult)
    Deadline    time.Time
}

// Batch I/O operations
func (hpio *HighPerformanceIOSystem) BatchIOOperations(requests []IORequest) {
    // Group requests by type and destination
    readRequests := make(map[string][]IORequest)
    writeRequests := make(map[string][]IORequest)
    
    for _, req := range requests {
        destination := req.GetDestination()
        if req.Type == IORead {
            readRequests[destination] = append(readRequests[destination], req)
        } else {
            writeRequests[destination] = append(writeRequests[destination], req)
        }
    }
    
    // Execute batched operations
    for dest, reqs := range readRequests {
        go hpio.executeBatchRead(dest, reqs)
    }
    
    for dest, reqs := range writeRequests {
        go hpio.executeBatchWrite(dest, reqs)
    }
}
```

#### **4. Caching Strategy Design**

**Q: Design a multi-level caching system with intelligent eviction policies.**

**A: Advanced caching architecture:**

```go
// Multi-level cache with intelligent eviction
type IntelligentCache struct {
    l1Cache    *LRUCache        // Hot data (fast access)
    l2Cache    *LFUCache        // Warm data (frequency-based)
    l3Cache    *TTLCache        // Cold data (time-based eviction)
    promoter   *CachePromoter   // Handles promotion between levels
    metrics    *CacheMetrics    // Performance tracking
}

// Adaptive eviction policy
type AdaptiveEvictionPolicy struct {
    hitRatioThreshold  float64
    promoteThreshold   int
    demoteThreshold    int
    accessHistory      *AccessHistoryTracker
}

func (aep *AdaptiveEvictionPolicy) ShouldPromote(key string, level int) bool {
    history := aep.accessHistory.GetAccessHistory(key)
    
    // Promote if access frequency increases
    if history.RecentAccessRate > history.HistoricalAccessRate*1.5 {
        return true
    }
    
    // Promote if access pattern shows locality
    if history.AccessLocality > 0.8 {
        return true
    }
    
    return false
}

// Cache warming strategy
func (ic *IntelligentCache) WarmCache(ctx context.Context) {
    // 1. Preload frequently accessed data
    popularKeys := ic.metrics.GetMostAccessedKeys(1000)
    
    // 2. Load data asynchronously
    semaphore := make(chan struct{}, 10) // Limit concurrency
    
    for _, key := range popularKeys {
        select {
        case semaphore <- struct{}{}:
            go func(k string) {
                defer func() { <-semaphore }()
                
                data, err := ic.loadFromSource(k)
                if err == nil {
                    ic.l3Cache.Set(k, data, time.Hour)
                }
            }(key)
        case <-ctx.Done():
            return
        }
    }
}

// Performance monitoring and auto-tuning
func (ic *IntelligentCache) AutoTune() {
    metrics := ic.metrics.GetMetrics()
    
    // Adjust cache sizes based on hit ratios
    if metrics.L1HitRatio < 0.8 {
        ic.l1Cache.Resize(ic.l1Cache.Size() * 2)
    }
    
    if metrics.L2HitRatio < 0.6 {
        ic.l2Cache.Resize(ic.l2Cache.Size() * 2)
    }
    
    // Adjust eviction policies based on access patterns
    ic.adaptEvictionPolicy(metrics)
}
```

#### **5. Performance Monitoring System**

**Q: Design a comprehensive performance monitoring system for a distributed application.**

**A: Distributed performance monitoring architecture:**

```go
// Comprehensive performance monitoring
type PerformanceMonitoringSystem struct {
    collectors []MetricCollector
    aggregator *MetricAggregator
    alerting   *AlertingSystem
    storage    *MetricStorage
    dashboard  *DashboardService
}

// Real-time metric collection
type RealTimeCollector struct {
    cpuCollector    *CPUMetricCollector
    memoryCollector *MemoryMetricCollector
    ioCollector     *IOMetricCollector
    networkCollector *NetworkMetricCollector
    applicationCollector *ApplicationMetricCollector
}

func (rtc *RealTimeCollector) CollectMetrics() *SystemMetrics {
    return &SystemMetrics{
        CPU:         rtc.cpuCollector.Collect(),
        Memory:      rtc.memoryCollector.Collect(),
        IO:          rtc.ioCollector.Collect(),
        Network:     rtc.networkCollector.Collect(),
        Application: rtc.applicationCollector.Collect(),
        Timestamp:   time.Now(),
    }
}

// Predictive performance analysis
type PerformancePredictor struct {
    historicalData []SystemMetrics
    model          *PredictionModel
    alertThresholds map[string]float64
}

func (pp *PerformancePredictor) PredictBottlenecks() []PerformanceBottleneck {
    var bottlenecks []PerformanceBottleneck
    
    // Analyze trends
    cpuTrend := pp.analyzeTrend("cpu_usage")
    memoryTrend := pp.analyzeTrend("memory_usage")
    ioTrend := pp.analyzeTrend("io_wait")
    
    // Predict future bottlenecks
    if cpuTrend.Slope > 0.1 && cpuTrend.R2 > 0.8 {
        eta := pp.calculateETA(cpuTrend, 90.0) // 90% CPU threshold
        bottlenecks = append(bottlenecks, PerformanceBottleneck{
            Type: "CPU",
            Severity: "HIGH",
            ETA: eta,
            Recommendation: "Scale horizontally or optimize CPU-bound operations",
        })
    }
    
    return bottlenecks
}

// Auto-scaling based on performance metrics
type AutoScaler struct {
    scaleUpThreshold   float64
    scaleDownThreshold float64
    cooldownPeriod     time.Duration
    lastScaleAction    time.Time
}

func (as *AutoScaler) ShouldScale(metrics *SystemMetrics) ScaleDecision {
    if time.Since(as.lastScaleAction) < as.cooldownPeriod {
        return ScaleDecision{Action: "NONE", Reason: "Cooldown period"}
    }
    
    if metrics.CPU.Usage > as.scaleUpThreshold {
        return ScaleDecision{
            Action: "SCALE_UP",
            Reason: fmt.Sprintf("CPU usage %.2f%% > threshold %.2f%%", 
                metrics.CPU.Usage, as.scaleUpThreshold),
            Replicas: calculateOptimalReplicas(metrics),
        }
    }
    
    if metrics.CPU.Usage < as.scaleDownThreshold {
        return ScaleDecision{
            Action: "SCALE_DOWN", 
            Reason: "CPU usage below threshold",
            Replicas: calculateOptimalReplicas(metrics),
        }
    }
    
    return ScaleDecision{Action: "NONE"}
}
```

This comprehensive Performance Engineering Guide provides advanced implementations for profiling, memory optimization, CPU optimization, I/O optimization, and performance monitoring. The guide includes production-ready Go code with object pooling, memory arenas, async I/O, cache-friendly algorithms, and comprehensive performance monitoring systems that demonstrate the performance engineering expertise expected from senior backend engineers in technical interviews.