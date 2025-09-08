# Go Benchmarking and Profiling Guide

This guide provides comprehensive strategies for benchmarking and profiling Go applications, with a focus on backend services and fintech applications.

## Benchmarking Basics

### 1. Writing Benchmarks
```go
// Basic benchmark function
func BenchmarkPaymentProcessor_ProcessPayment(b *testing.B) {
    processor := NewPaymentProcessor()
    payment := &Payment{
        ID:     "bench-payment",
        Amount: 100.50,
        Status: "pending",
    }
    
    b.ResetTimer() // Reset timer after setup
    for i := 0; i < b.N; i++ {
        err := processor.ProcessPayment(payment)
        if err != nil {
            b.Fatal(err)
        }
    }
}

// Benchmark with different input sizes
func BenchmarkPaymentRepository_Create(b *testing.B) {
    db := setupTestDB(b)
    defer cleanupTestDB(b, db)
    
    repo := NewPaymentRepository(db)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        payment := &Payment{
            ID:     fmt.Sprintf("bench-payment-%d", i),
            Amount: 100.50,
            Status: "pending",
        }
        
        err := repo.Create(context.Background(), payment)
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

### 2. Benchmark Variations
```go
// Benchmark with different parameters
func BenchmarkPaymentValidator_ValidateAmount(b *testing.B) {
    validator := NewPaymentValidator()
    
    benchmarks := []struct {
        name   string
        amount float64
    }{
        {"SmallAmount", 10.50},
        {"MediumAmount", 1000.00},
        {"LargeAmount", 100000.00},
    }
    
    for _, bm := range benchmarks {
        b.Run(bm.name, func(b *testing.B) {
            for i := 0; i < b.N; i++ {
                err := validator.ValidateAmount(bm.amount)
                if err != nil {
                    b.Fatal(err)
                }
            }
        })
    }
}

// Benchmark with memory allocation tracking
func BenchmarkPaymentService_ProcessPayment(b *testing.B) {
    service := NewPaymentService()
    payment := &Payment{
        ID:     "bench-payment",
        Amount: 100.50,
        Status: "pending",
    }
    
    b.ResetTimer()
    b.ReportAllocs() // Report memory allocations
    
    for i := 0; i < b.N; i++ {
        err := service.ProcessPayment(payment)
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

### 3. Running Benchmarks
```bash
# Run all benchmarks
go test -bench=. ./...

# Run specific benchmark
go test -bench=BenchmarkPaymentProcessor ./...

# Run benchmarks with memory allocation info
go test -bench=. -benchmem ./...

# Run benchmarks multiple times for stability
go test -bench=. -count=5 ./...

# Run benchmarks with CPU profile
go test -bench=. -cpuprofile=cpu.prof ./...

# Run benchmarks with memory profile
go test -bench=. -memprofile=mem.prof ./...

# Run benchmarks with trace
go test -bench=. -trace=trace.out ./...
```

## Profiling with pprof

### 1. CPU Profiling
```go
// CPU profiling in code
func main() {
    // Start CPU profiling
    f, err := os.Create("cpu.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    
    if err := pprof.StartCPUProfile(f); err != nil {
        log.Fatal(err)
    }
    defer pprof.StopCPUProfile()
    
    // Your application code here
    runApplication()
}

// HTTP endpoint for CPU profiling
func setupProfiling() {
    http.HandleFunc("/debug/pprof/", pprof.Index)
    http.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
    http.HandleFunc("/debug/pprof/profile", pprof.Profile)
    http.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
    http.HandleFunc("/debug/pprof/trace", pprof.Trace)
}

// Example usage
func TestPaymentService_CPUProfile(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping CPU profile test in short mode")
    }
    
    // Start CPU profiling
    f, err := os.Create("payment_cpu.prof")
    if err != nil {
        t.Fatal(err)
    }
    defer f.Close()
    
    if err := pprof.StartCPUProfile(f); err != nil {
        t.Fatal(err)
    }
    defer pprof.StopCPUProfile()
    
    // Run the code to profile
    service := NewPaymentService()
    for i := 0; i < 10000; i++ {
        payment := &Payment{
            ID:     fmt.Sprintf("profile-payment-%d", i),
            Amount: 100.50,
            Status: "pending",
        }
        service.ProcessPayment(payment)
    }
}
```

### 2. Memory Profiling
```go
// Memory profiling in code
func main() {
    // Start memory profiling
    f, err := os.Create("mem.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    
    // Your application code here
    runApplication()
    
    // Write memory profile
    runtime.GC() // Force garbage collection
    if err := pprof.WriteHeapProfile(f); err != nil {
        log.Fatal(err)
    }
}

// Memory profiling with HTTP endpoint
func setupMemoryProfiling() {
    http.HandleFunc("/debug/pprof/heap", pprof.Handler("heap").ServeHTTP)
    http.HandleFunc("/debug/pprof/allocs", pprof.Handler("allocs").ServeHTTP)
}

// Example memory profiling test
func TestPaymentService_MemoryProfile(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping memory profile test in short mode")
    }
    
    service := NewPaymentService()
    
    // Run the code to profile
    for i := 0; i < 10000; i++ {
        payment := &Payment{
            ID:     fmt.Sprintf("mem-payment-%d", i),
            Amount: 100.50,
            Status: "pending",
        }
        service.ProcessPayment(payment)
    }
    
    // Force garbage collection
    runtime.GC()
    
    // Write memory profile
    f, err := os.Create("payment_mem.prof")
    if err != nil {
        t.Fatal(err)
    }
    defer f.Close()
    
    if err := pprof.WriteHeapProfile(f); err != nil {
        t.Fatal(err)
    }
}
```

### 3. Goroutine Profiling
```go
// Goroutine profiling
func setupGoroutineProfiling() {
    http.HandleFunc("/debug/pprof/goroutine", pprof.Handler("goroutine").ServeHTTP)
}

// Example goroutine profiling
func TestPaymentService_GoroutineProfile(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping goroutine profile test in short mode")
    }
    
    service := NewPaymentService()
    
    // Create many goroutines
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            payment := &Payment{
                ID:     fmt.Sprintf("goroutine-payment-%d", id),
                Amount: 100.50,
                Status: "pending",
            }
            service.ProcessPayment(payment)
        }(i)
    }
    
    wg.Wait()
    
    // Write goroutine profile
    f, err := os.Create("payment_goroutine.prof")
    if err != nil {
        t.Fatal(err)
    }
    defer f.Close()
    
    if err := pprof.Lookup("goroutine").WriteTo(f, 0); err != nil {
        t.Fatal(err)
    }
}
```

## Profiling Analysis

### 1. Using go tool pprof
```bash
# Analyze CPU profile
go tool pprof cpu.prof

# Analyze memory profile
go tool pprof mem.prof

# Analyze goroutine profile
go tool pprof goroutine.prof

# Interactive commands in pprof
(pprof) top10          # Top 10 functions by CPU time
(pprof) list function  # Show source code with profiling info
(pprof) web            # Open web interface
(pprof) png            # Generate PNG graph
(pprof) svg            # Generate SVG graph
(pprof) tree           # Show call tree
(pprof) peek function  # Show callers and callees
(pprof) disasm function # Show assembly code
```

### 2. Web Interface
```bash
# Start web interface
go tool pprof -http=:8080 cpu.prof

# Or use the built-in web interface
go tool pprof -http=:8080 http://localhost:8080/debug/pprof/profile
```

### 3. Flame Graphs
```bash
# Generate flame graph
go tool pprof -http=:8080 cpu.prof

# Or use external tools
go tool pprof -raw cpu.prof | flamegraph.pl > flame.svg
```

## Performance Optimization

### 1. Identifying Bottlenecks
```go
// Example: Optimizing payment processing
func BenchmarkPaymentProcessor_Original(b *testing.B) {
    processor := NewPaymentProcessor()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        payment := &Payment{
            ID:     fmt.Sprintf("payment-%d", i),
            Amount: 100.50,
            Status: "pending",
        }
        
        err := processor.ProcessPayment(payment)
        if err != nil {
            b.Fatal(err)
        }
    }
}

// Optimized version
func BenchmarkPaymentProcessor_Optimized(b *testing.B) {
    processor := NewPaymentProcessor()
    
    // Pre-allocate payment slice
    payments := make([]*Payment, b.N)
    for i := 0; i < b.N; i++ {
        payments[i] = &Payment{
            ID:     fmt.Sprintf("payment-%d", i),
            Amount: 100.50,
            Status: "pending",
        }
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        err := processor.ProcessPayment(payments[i])
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

### 2. Memory Optimization
```go
// Example: Reducing memory allocations
func BenchmarkPaymentValidator_Original(b *testing.B) {
    validator := NewPaymentValidator()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        // This creates a new string each time
        paymentID := fmt.Sprintf("payment-%d", i)
        err := validator.ValidatePaymentID(paymentID)
        if err != nil {
            b.Fatal(err)
        }
    }
}

// Optimized version with string builder
func BenchmarkPaymentValidator_Optimized(b *testing.B) {
    validator := NewPaymentValidator()
    var sb strings.Builder
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        sb.Reset()
        sb.WriteString("payment-")
        sb.WriteString(strconv.Itoa(i))
        paymentID := sb.String()
        
        err := validator.ValidatePaymentID(paymentID)
        if err != nil {
            b.Fatal(err)
        }
    }
}

// Even more optimized with pre-allocated buffer
func BenchmarkPaymentValidator_PreAllocated(b *testing.B) {
    validator := NewPaymentValidator()
    buffer := make([]byte, 0, 20) // Pre-allocate buffer
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        buffer = buffer[:0] // Reset buffer
        buffer = append(buffer, "payment-"...)
        buffer = strconv.AppendInt(buffer, int64(i), 10)
        paymentID := string(buffer)
        
        err := validator.ValidatePaymentID(paymentID)
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

### 3. Concurrency Optimization
```go
// Example: Optimizing concurrent payment processing
func BenchmarkPaymentProcessor_Sequential(b *testing.B) {
    processor := NewPaymentProcessor()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        payment := &Payment{
            ID:     fmt.Sprintf("payment-%d", i),
            Amount: 100.50,
            Status: "pending",
        }
        
        err := processor.ProcessPayment(payment)
        if err != nil {
            b.Fatal(err)
        }
    }
}

// Concurrent version
func BenchmarkPaymentProcessor_Concurrent(b *testing.B) {
    processor := NewPaymentProcessor()
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        i := 0
        for pb.Next() {
            payment := &Payment{
                ID:     fmt.Sprintf("payment-%d", i),
                Amount: 100.50,
                Status: "pending",
            }
            
            err := processor.ProcessPayment(payment)
            if err != nil {
                b.Fatal(err)
            }
            i++
        }
    })
}
```

## Advanced Profiling

### 1. Custom Profiling
```go
// Custom profiling for specific operations
type Profiler struct {
    startTime time.Time
    duration  time.Duration
    count     int64
}

func (p *Profiler) Start() {
    p.startTime = time.Now()
}

func (p *Profiler) Stop() {
    p.duration += time.Since(p.startTime)
    p.count++
}

func (p *Profiler) AverageDuration() time.Duration {
    if p.count == 0 {
        return 0
    }
    return p.duration / time.Duration(p.count)
}

// Usage in benchmarks
func BenchmarkPaymentService_WithCustomProfiling(b *testing.B) {
    service := NewPaymentService()
    profiler := &Profiler{}
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        profiler.Start()
        
        payment := &Payment{
            ID:     fmt.Sprintf("payment-%d", i),
            Amount: 100.50,
            Status: "pending",
        }
        
        err := service.ProcessPayment(payment)
        if err != nil {
            b.Fatal(err)
        }
        
        profiler.Stop()
    }
    
    b.ReportMetric(float64(profiler.AverageDuration().Nanoseconds()), "ns/op")
}
```

### 2. Memory Pool Optimization
```go
// Using sync.Pool for object reuse
var paymentPool = sync.Pool{
    New: func() interface{} {
        return &Payment{}
    },
}

func BenchmarkPaymentProcessor_WithPool(b *testing.B) {
    processor := NewPaymentProcessor()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        // Get payment from pool
        payment := paymentPool.Get().(*Payment)
        
        // Reset payment fields
        payment.ID = fmt.Sprintf("payment-%d", i)
        payment.Amount = 100.50
        payment.Status = "pending"
        
        err := processor.ProcessPayment(payment)
        if err != nil {
            b.Fatal(err)
        }
        
        // Return payment to pool
        paymentPool.Put(payment)
    }
}
```

### 3. Trace Analysis
```go
// Using runtime/trace for detailed analysis
func TestPaymentService_Trace(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping trace test in short mode")
    }
    
    // Start tracing
    f, err := os.Create("payment.trace")
    if err != nil {
        t.Fatal(err)
    }
    defer f.Close()
    
    if err := trace.Start(f); err != nil {
        t.Fatal(err)
    }
    defer trace.Stop()
    
    // Run the code to trace
    service := NewPaymentService()
    
    var wg sync.WaitGroup
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            payment := &Payment{
                ID:     fmt.Sprintf("trace-payment-%d", id),
                Amount: 100.50,
                Status: "pending",
            }
            service.ProcessPayment(payment)
        }(i)
    }
    
    wg.Wait()
}

// Analyze trace
// go tool trace payment.trace
```

## Continuous Performance Monitoring

### 1. Performance Regression Testing
```go
// Performance regression test
func TestPaymentService_PerformanceRegression(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping performance regression test in short mode")
    }
    
    service := NewPaymentService()
    
    // Measure performance
    start := time.Now()
    for i := 0; i < 10000; i++ {
        payment := &Payment{
            ID:     fmt.Sprintf("perf-payment-%d", i),
            Amount: 100.50,
            Status: "pending",
        }
        err := service.ProcessPayment(payment)
        if err != nil {
            t.Fatal(err)
        }
    }
    duration := time.Since(start)
    
    // Check if performance is within acceptable limits
    maxDuration := 5 * time.Second
    if duration > maxDuration {
        t.Errorf("Performance regression: operation took %v, expected < %v", 
            duration, maxDuration)
    }
    
    t.Logf("Performance test completed in %v", duration)
}
```

### 2. Benchmark Comparison
```bash
# Compare benchmarks
go test -bench=. -benchmem ./... > current.txt
go test -bench=. -benchmem ./... > previous.txt
benchcmp previous.txt current.txt
```

### 3. Automated Performance Testing
```yaml
# GitHub Actions for performance testing
name: Performance Tests

on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Go
      uses: actions/setup-go@v3
      with:
        go-version: 1.21
    
    - name: Run benchmarks
      run: |
        go test -bench=. -benchmem ./... > benchmark.txt
        cat benchmark.txt
    
    - name: Check performance regression
      run: |
        # Compare with previous benchmark results
        if [ -f previous_benchmark.txt ]; then
          benchcmp previous_benchmark.txt benchmark.txt
        fi
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark.txt
```

## Best Practices

### 1. Benchmarking Best Practices
- Always use `b.ResetTimer()` after setup
- Use `b.ReportAllocs()` to track memory allocations
- Run benchmarks multiple times for stability
- Use `b.RunParallel()` for concurrent benchmarks
- Profile before optimizing

### 2. Profiling Best Practices
- Profile in production-like environments
- Use multiple profiling types (CPU, memory, goroutines)
- Analyze profiles with web interface
- Look for hotspots and memory leaks
- Optimize the biggest bottlenecks first

### 3. Performance Testing Best Practices
- Set up continuous performance monitoring
- Track performance regressions
- Use realistic test data
- Test under load
- Monitor memory usage and garbage collection

This comprehensive guide provides the foundation for effective benchmarking and profiling of Go applications, with particular emphasis on backend services and fintech applications. Use these techniques to identify performance bottlenecks, optimize critical paths, and ensure your applications perform well under load.
