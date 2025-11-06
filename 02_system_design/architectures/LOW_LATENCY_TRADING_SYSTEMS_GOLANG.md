---
# Auto-generated front matter
Title: Low Latency Trading Systems Golang
LastUpdated: 2025-11-06T20:45:57.719172
Tags: []
Status: draft
---

# 🚀 **Low Latency Trading Systems in Go**

## 📊 **Based on C++ Low Latency Trading System Talk - Go Implementation**

---

## 🎯 **Core Principles from the Talk**

### **1. Avoid Node-Based Containers**

**Principle**: "Most of the time you do not want node containers" - avoid `std::map`, `std::set`, `std::unordered_map`, `std::unordered_set`, `std::list`

**Why**: Node-based containers have poor cache locality and cause memory fragmentation.

**Go Equivalent**: Use slices instead of maps when possible, avoid linked lists.

### **2. Problem Well Stated is Half Solved**

**Principle**: Understand your specific problem domain and leverage its properties for performance.

### **3. Leverage Specific Properties**

**Principle**: Use domain-specific optimizations rather than generic solutions.

### **4. Simplicity + Speed**

**Principle**: "When it's fast and it's extremely simple, you can stop."

### **5. Mechanical Sympathy**

**Principle**: Design algorithms that work in harmony with hardware (cache locality, prefetching, branch prediction).

### **6. Be Mindful of What You Use**

**Principle**: Don't use complex systems when simple ones suffice.

### **7. Use the Right Tool for the Right Task**

**Principle**: Choose appropriate data structures and algorithms for specific use cases.

### **8. Stay Fast is Harder Than Being Fast**

**Principle**: Maintaining performance over time requires discipline and monitoring.

### **9. You're Not Alone**

**Principle**: Consider the entire system, not just your application.

---

## 🚀 **Order Book Implementation in Go**

### **Problem Analysis**

An order book is a fundamental data structure in trading systems that maintains:

- **Bids**: Prices at which people are willing to buy (ordered descending)
- **Asks**: Prices at which people are willing to sell (ordered ascending)
- **Price Levels**: Each price level contains volume and order count
- **Order Management**: Add, modify, delete orders by ID

### **Key Requirements**

- **Low Latency**: Sub-microsecond operations
- **High Throughput**: Handle thousands of updates per second
- **Memory Efficiency**: Minimize allocations
- **Cache Locality**: Optimize for CPU cache behavior

### **Go Implementation**

```go
package main

import (
    "fmt"
    "sort"
    "sync"
    "time"
)

// PriceLevel represents a single price level in the order book
type PriceLevel struct {
    Price  int64 // Price in cents to avoid floating point
    Volume int64 // Total volume at this price
    Count  int   // Number of orders at this price
}

// Order represents a single order
type Order struct {
    ID     int64
    Price  int64
    Volume int64
    Side   OrderSide
}

type OrderSide int

const (
    Bid OrderSide = iota
    Ask
)

// OrderBook maintains the order book state
type OrderBook struct {
    symbol string

    // Bid side (prices in descending order)
    bids []PriceLevel

    // Ask side (prices in ascending order)
    asks []PriceLevel

    // Order lookup by ID
    orders map[int64]*Order

    // Mutex for thread safety
    mutex sync.RWMutex
}

// NewOrderBook creates a new order book
func NewOrderBook(symbol string) *OrderBook {
    return &OrderBook{
        symbol: symbol,
        bids:   make([]PriceLevel, 0, 1000),
        asks:   make([]PriceLevel, 0, 1000),
        orders: make(map[int64]*Order, 1000),
    }
}

// AddOrder adds a new order to the book
func (ob *OrderBook) AddOrder(order *Order) error {
    ob.mutex.Lock()
    defer ob.mutex.Unlock()

    // Store order for lookup
    ob.orders[order.ID] = order

    if order.Side == Bid {
        ob.addBid(order)
    } else {
        ob.addAsk(order)
    }

    return nil
}

// addBid adds a bid order (prices in descending order)
func (ob *OrderBook) addBid(order *Order) {
    // Find insertion point using binary search
    idx := sort.Search(len(ob.bids), func(i int) bool {
        return ob.bids[i].Price <= order.Price
    })

    if idx < len(ob.bids) && ob.bids[idx].Price == order.Price {
        // Update existing price level
        ob.bids[idx].Volume += order.Volume
        ob.bids[idx].Count++
    } else {
        // Insert new price level
        newLevel := PriceLevel{
            Price:  order.Price,
            Volume: order.Volume,
            Count:  1,
        }

        // Insert at position idx
        ob.bids = append(ob.bids, PriceLevel{})
        copy(ob.bids[idx+1:], ob.bids[idx:])
        ob.bids[idx] = newLevel
    }
}

// addAsk adds an ask order (prices in ascending order)
func (ob *OrderBook) addAsk(order *Order) {
    // Find insertion point using binary search
    idx := sort.Search(len(ob.asks), func(i int) bool {
        return ob.asks[i].Price >= order.Price
    })

    if idx < len(ob.asks) && ob.asks[idx].Price == order.Price {
        // Update existing price level
        ob.asks[idx].Volume += order.Volume
        ob.asks[idx].Count++
    } else {
        // Insert new price level
        newLevel := PriceLevel{
            Price:  order.Price,
            Volume: order.Volume,
            Count:  1,
        }

        // Insert at position idx
        ob.asks = append(ob.asks, PriceLevel{})
        copy(ob.asks[idx+1:], ob.asks[idx:])
        ob.asks[idx] = newLevel
    }
}

// ModifyOrder modifies an existing order
func (ob *OrderBook) ModifyOrder(orderID int64, newVolume int64) error {
    ob.mutex.Lock()
    defer ob.mutex.Unlock()

    order, exists := ob.orders[orderID]
    if !exists {
        return fmt.Errorf("order %d not found", orderID)
    }

    // Calculate volume difference
    volumeDiff := newVolume - order.Volume
    order.Volume = newVolume

    // Update price level
    if order.Side == Bid {
        ob.updateBidLevel(order.Price, volumeDiff)
    } else {
        ob.updateAskLevel(order.Price, volumeDiff)
    }

    return nil
}

// updateBidLevel updates a bid price level
func (ob *OrderBook) updateBidLevel(price int64, volumeDiff int64) {
    idx := ob.findBidLevel(price)
    if idx >= 0 {
        ob.bids[idx].Volume += volumeDiff
        if ob.bids[idx].Volume <= 0 {
            // Remove empty level
            copy(ob.bids[idx:], ob.bids[idx+1:])
            ob.bids = ob.bids[:len(ob.bids)-1]
        }
    }
}

// updateAskLevel updates an ask price level
func (ob *OrderBook) updateAskLevel(price int64, volumeDiff int64) {
    idx := ob.findAskLevel(price)
    if idx >= 0 {
        ob.asks[idx].Volume += volumeDiff
        if ob.asks[idx].Volume <= 0 {
            // Remove empty level
            copy(ob.asks[idx:], ob.asks[idx+1:])
            ob.asks = ob.asks[:len(ob.asks)-1]
        }
    }
}

// findBidLevel finds the index of a bid price level
func (ob *OrderBook) findBidLevel(price int64) int {
    for i, level := range ob.bids {
        if level.Price == price {
            return i
        }
    }
    return -1
}

// findAskLevel finds the index of an ask price level
func (ob *OrderBook) findAskLevel(price int64) int {
    for i, level := range ob.asks {
        if level.Price == price {
            return i
        }
    }
    return -1
}

// DeleteOrder removes an order from the book
func (ob *OrderBook) DeleteOrder(orderID int64) error {
    ob.mutex.Lock()
    defer ob.mutex.Unlock()

    order, exists := ob.orders[orderID]
    if !exists {
        return fmt.Errorf("order %d not found", orderID)
    }

    // Update price level
    if order.Side == Bid {
        ob.updateBidLevel(order.Price, -order.Volume)
    } else {
        ob.updateAskLevel(order.Price, -order.Volume)
    }

    // Remove from orders map
    delete(ob.orders, orderID)

    return nil
}

// GetBestBid returns the best bid price and volume
func (ob *OrderBook) GetBestBid() (int64, int64) {
    ob.mutex.RLock()
    defer ob.mutex.RUnlock()

    if len(ob.bids) == 0 {
        return 0, 0
    }
    return ob.bids[0].Price, ob.bids[0].Volume
}

// GetBestAsk returns the best ask price and volume
func (ob *OrderBook) GetBestAsk() (int64, int64) {
    ob.mutex.RLock()
    defer ob.mutex.RUnlock()

    if len(ob.asks) == 0 {
        return 0, 0
    }
    return ob.asks[0].Price, ob.asks[0].Volume
}

// GetSpread returns the bid-ask spread
func (ob *OrderBook) GetSpread() int64 {
    bestBid, _ := ob.GetBestBid()
    bestAsk, _ := ob.GetBestAsk()

    if bestBid == 0 || bestAsk == 0 {
        return 0
    }

    return bestAsk - bestBid
}

// GetTopOfBook returns the top of book (best bid and ask)
func (ob *OrderBook) GetTopOfBook() (int64, int64, int64, int64) {
    ob.mutex.RLock()
    defer ob.mutex.RUnlock()

    var bidPrice, bidVolume, askPrice, askVolume int64

    if len(ob.bids) > 0 {
        bidPrice = ob.bids[0].Price
        bidVolume = ob.bids[0].Volume
    }

    if len(ob.asks) > 0 {
        askPrice = ob.asks[0].Price
        askVolume = ob.asks[0].Volume
    }

    return bidPrice, bidVolume, askPrice, askVolume
}
```

---

## 🚀 **Optimized Order Book with Linear Search**

### **Key Insight from the Talk**

The speaker discovered that **linear search is often faster than binary search** for order books because:

1. **Cache Locality**: Linear search has better cache behavior
2. **Branch Prediction**: Modern CPUs predict linear access patterns well
3. **Data Distribution**: Most updates happen at the top of the book
4. **Simplicity**: No complex branching logic

### **Optimized Implementation**

```go
// OptimizedOrderBook uses linear search for better performance
type OptimizedOrderBook struct {
    symbol string

    // Bid side (prices in descending order)
    bids []PriceLevel

    // Ask side (prices in ascending order)
    asks []PriceLevel

    // Order lookup by ID
    orders map[int64]*Order

    // Mutex for thread safety
    mutex sync.RWMutex
}

// NewOptimizedOrderBook creates an optimized order book
func NewOptimizedOrderBook(symbol string) *OptimizedOrderBook {
    return &OptimizedOrderBook{
        symbol: symbol,
        bids:   make([]PriceLevel, 0, 1000),
        asks:   make([]PriceLevel, 0, 1000),
        orders: make(map[int64]*Order, 1000),
    }
}

// addBidOptimized uses linear search for better cache locality
func (ob *OptimizedOrderBook) addBidOptimized(order *Order) {
    // Linear search from the beginning (most common case)
    for i, level := range ob.bids {
        if level.Price == order.Price {
            // Update existing level
            ob.bids[i].Volume += order.Volume
            ob.bids[i].Count++
            return
        }
        if level.Price < order.Price {
            // Insert new level at position i
            newLevel := PriceLevel{
                Price:  order.Price,
                Volume: order.Volume,
                Count:  1,
            }

            // Insert at position i
            ob.bids = append(ob.bids, PriceLevel{})
            copy(ob.bids[i+1:], ob.bids[i:])
            ob.bids[i] = newLevel
            return
        }
    }

    // Insert at the end
    ob.bids = append(ob.bids, PriceLevel{
        Price:  order.Price,
        Volume: order.Volume,
        Count:  1,
    })
}

// addAskOptimized uses linear search for better cache locality
func (ob *OptimizedOrderBook) addAskOptimized(order *Order) {
    // Linear search from the beginning (most common case)
    for i, level := range ob.asks {
        if level.Price == order.Price {
            // Update existing level
            ob.asks[i].Volume += order.Volume
            ob.asks[i].Count++
            return
        }
        if level.Price > order.Price {
            // Insert new level at position i
            newLevel := PriceLevel{
                Price:  order.Price,
                Volume: order.Volume,
                Count:  1,
            }

            // Insert at position i
            ob.asks = append(ob.asks, PriceLevel{})
            copy(ob.asks[i+1:], ob.asks[i:])
            ob.asks[i] = newLevel
            return
        }
    }

    // Insert at the end
    ob.asks = append(ob.asks, PriceLevel{
        Price:  order.Price,
        Volume: order.Volume,
        Count:  1,
    })
}

// findBidLevelOptimized uses linear search
func (ob *OptimizedOrderBook) findBidLevelOptimized(price int64) int {
    for i, level := range ob.bids {
        if level.Price == price {
            return i
        }
    }
    return -1
}

// findAskLevelOptimized uses linear search
func (ob *OptimizedOrderBook) findAskLevelOptimized(price int64) int {
    for i, level := range ob.asks {
        if level.Price == price {
            return i
        }
    }
    return -1
}
```

---

## 🚀 **Memory Pool for High-Performance Order Books**

### **Object Pooling Pattern**

Based on the talk's emphasis on avoiding dynamic allocations:

```go
// OrderPool provides object pooling for orders
type OrderPool struct {
    pool sync.Pool
}

// NewOrderPool creates a new order pool
func NewOrderPool() *OrderPool {
    return &OrderPool{
        pool: sync.Pool{
            New: func() interface{} {
                return &Order{
                    ID:     0,
                    Price:  0,
                    Volume: 0,
                    Side:   Bid,
                }
            },
        },
    }
}

// Get retrieves an order from the pool
func (op *OrderPool) Get() *Order {
    order := op.pool.Get().(*Order)
    order.Reset()
    return order
}

// Put returns an order to the pool
func (op *OrderPool) Put(order *Order) {
    op.pool.Put(order)
}

// Reset resets the order to zero values
func (o *Order) Reset() {
    o.ID = 0
    o.Price = 0
    o.Volume = 0
    o.Side = Bid
}

// PriceLevelPool provides object pooling for price levels
type PriceLevelPool struct {
    pool sync.Pool
}

// NewPriceLevelPool creates a new price level pool
func NewPriceLevelPool() *PriceLevelPool {
    return &PriceLevelPool{
        pool: sync.Pool{
            New: func() interface{} {
                return &PriceLevel{
                    Price:  0,
                    Volume: 0,
                    Count:  0,
                }
            },
        },
    }
}

// Get retrieves a price level from the pool
func (plp *PriceLevelPool) Get() *PriceLevel {
    level := plp.pool.Get().(*PriceLevel)
    level.Reset()
    return level
}

// Put returns a price level to the pool
func (plp *PriceLevelPool) Put(level *PriceLevel) {
    plp.pool.Put(level)
}

// Reset resets the price level to zero values
func (pl *PriceLevel) Reset() {
    pl.Price = 0
    pl.Volume = 0
    pl.Count = 0
}
```

---

## 🚀 **Concurrent Queue Implementation**

### **Based on the Talk's Queue Design**

The speaker discussed a bounded concurrent queue with specific properties:

```go
// ConcurrentQueue implements a bounded concurrent queue
type ConcurrentQueue struct {
    // Queue data
    data []byte

    // Head and tail pointers (atomic)
    head uint64
    tail uint64

    // Queue size
    size uint64

    // Mutex for thread safety
    mutex sync.RWMutex
}

// NewConcurrentQueue creates a new concurrent queue
func NewConcurrentQueue(size uint64) *ConcurrentQueue {
    return &ConcurrentQueue{
        data: make([]byte, size),
        size: size,
    }
}

// Write writes data to the queue
func (cq *ConcurrentQueue) Write(data []byte) error {
    cq.mutex.Lock()
    defer cq.mutex.Unlock()

    dataSize := uint64(len(data))
    if dataSize+8 > cq.size { // 8 bytes for size header
        return fmt.Errorf("data too large for queue")
    }

    // Check if there's enough space
    if cq.head+dataSize+8 > cq.tail+cq.size {
        return fmt.Errorf("queue full")
    }

    // Write size header
    cq.writeUint64(cq.head, dataSize)
    cq.head += 8

    // Write data
    copy(cq.data[cq.head:], data)
    cq.head += dataSize

    return nil
}

// Read reads data from the queue
func (cq *ConcurrentQueue) Read() ([]byte, error) {
    cq.mutex.RLock()
    defer cq.mutex.RUnlock()

    if cq.tail >= cq.head {
        return nil, fmt.Errorf("queue empty")
    }

    // Read size header
    dataSize := cq.readUint64(cq.tail)
    cq.tail += 8

    // Read data
    data := make([]byte, dataSize)
    copy(data, cq.data[cq.tail:cq.tail+dataSize])
    cq.tail += dataSize

    return data, nil
}

// writeUint64 writes a uint64 to the queue
func (cq *ConcurrentQueue) writeUint64(offset uint64, value uint64) {
    for i := 0; i < 8; i++ {
        cq.data[offset+uint64(i)] = byte(value >> (i * 8))
    }
}

// readUint64 reads a uint64 from the queue
func (cq *ConcurrentQueue) readUint64(offset uint64) uint64 {
    var value uint64
    for i := 0; i < 8; i++ {
        value |= uint64(cq.data[offset+uint64(i)]) << (i * 8)
    }
    return value
}
```

---

## 🚀 **Performance Monitoring and Profiling**

### **Intrusive Profiling**

Based on the talk's discussion of profiling low-latency systems:

```go
// Profiler provides intrusive profiling for low-latency systems
type Profiler struct {
    measurements map[string][]time.Duration
    mutex        sync.RWMutex
}

// NewProfiler creates a new profiler
func NewProfiler() *Profiler {
    return &Profiler{
        measurements: make(map[string][]time.Duration),
    }
}

// ProfileFunc profiles a function execution
func (p *Profiler) ProfileFunc(name string, fn func()) {
    start := time.Now()
    fn()
    duration := time.Since(start)

    p.mutex.Lock()
    defer p.mutex.Unlock()

    p.measurements[name] = append(p.measurements[name], duration)
}

// GetStats returns statistics for a function
func (p *Profiler) GetStats(name string) (min, max, avg time.Duration, count int) {
    p.mutex.RLock()
    defer p.mutex.RUnlock()

    measurements, exists := p.measurements[name]
    if !exists || len(measurements) == 0 {
        return 0, 0, 0, 0
    }

    min = measurements[0]
    max = measurements[0]
    var sum time.Duration

    for _, m := range measurements {
        if m < min {
            min = m
        }
        if m > max {
            max = m
        }
        sum += m
    }

    avg = sum / time.Duration(len(measurements))
    count = len(measurements)

    return min, max, avg, count
}

// PrintStats prints statistics for all functions
func (p *Profiler) PrintStats() {
    p.mutex.RLock()
    defer p.mutex.RUnlock()

    fmt.Println("Function Performance Statistics:")
    fmt.Println("================================")

    for name, measurements := range p.measurements {
        if len(measurements) == 0 {
            continue
        }

        min, max, avg, count := p.GetStats(name)
        fmt.Printf("%s: min=%v, max=%v, avg=%v, count=%d\n",
            name, min, max, avg, count)
    }
}

// ProfileOrderBookOperation profiles order book operations
func (ob *OrderBook) ProfileOrderBookOperation(operation string, fn func()) {
    start := time.Now()
    fn()
    duration := time.Since(start)

    // Log or store the measurement
    fmt.Printf("Operation %s took %v\n", operation, duration)
}
```

---

## 🚀 **Benchmarking and Performance Testing**

### **Comprehensive Benchmark Suite**

```go
// BenchmarkOrderBook benchmarks order book operations
func BenchmarkOrderBook(b *testing.B) {
    ob := NewOrderBook("AAPL")

    // Pre-populate with orders
    for i := 0; i < 1000; i++ {
        order := &Order{
            ID:     int64(i),
            Price:  int64(100 + i),
            Volume: int64(100),
            Side:   Bid,
        }
        ob.AddOrder(order)
    }

    b.ResetTimer()

    b.Run("AddOrder", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            order := &Order{
                ID:     int64(1000 + i),
                Price:  int64(200 + i),
                Volume: int64(100),
                Side:   Bid,
            }
            ob.AddOrder(order)
        }
    })

    b.Run("GetBestBid", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            ob.GetBestBid()
        }
    })

    b.Run("ModifyOrder", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            ob.ModifyOrder(int64(i%1000), int64(200))
        }
    })
}

// BenchmarkOptimizedOrderBook benchmarks optimized order book
func BenchmarkOptimizedOrderBook(b *testing.B) {
    ob := NewOptimizedOrderBook("AAPL")

    // Pre-populate with orders
    for i := 0; i < 1000; i++ {
        order := &Order{
            ID:     int64(i),
            Price:  int64(100 + i),
            Volume: int64(100),
            Side:   Bid,
        }
        ob.AddOrder(order)
    }

    b.ResetTimer()

    b.Run("AddOrderOptimized", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            order := &Order{
                ID:     int64(1000 + i),
                Price:  int64(200 + i),
                Volume: int64(100),
                Side:   Bid,
            }
            ob.AddOrder(order)
        }
    })
}

// BenchmarkConcurrentQueue benchmarks concurrent queue
func BenchmarkConcurrentQueue(b *testing.B) {
    cq := NewConcurrentQueue(1024 * 1024) // 1MB queue

    data := []byte("test data")

    b.ResetTimer()

    b.Run("Write", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            cq.Write(data)
        }
    })

    b.Run("Read", func(b *testing.B) {
        for i := 0; i < b.N; i++ {
            cq.Read()
        }
    })
}
```

---

## 🚀 **Memory Management and GC Optimization**

### **GC Tuning for Low Latency**

```go
// GCOptimizer provides GC optimization utilities
type GCOptimizer struct {
    // Memory pools
    orderPool      *OrderPool
    priceLevelPool *PriceLevelPool

    // String interning for common values
    stringInterner map[string]string
    stringMutex    sync.RWMutex
}

// NewGCOptimizer creates a new GC optimizer
func NewGCOptimizer() *GCOptimizer {
    return &GCOptimizer{
        orderPool:      NewOrderPool(),
        priceLevelPool: NewPriceLevelPool(),
        stringInterner: make(map[string]string),
    }
}

// InternString interns a string to reduce allocations
func (gco *GCOptimizer) InternString(s string) string {
    gco.stringMutex.RLock()
    if interned, exists := gco.stringInterner[s]; exists {
        gco.stringMutex.RUnlock()
        return interned
    }
    gco.stringMutex.RUnlock()

    gco.stringMutex.Lock()
    defer gco.stringMutex.Unlock()

    // Double-check pattern
    if interned, exists := gco.stringInterner[s]; exists {
        return interned
    }

    gco.stringInterner[s] = s
    return s
}

// GetOrder gets an order from the pool
func (gco *GCOptimizer) GetOrder() *Order {
    return gco.orderPool.Get()
}

// PutOrder returns an order to the pool
func (gco *GCOptimizer) PutOrder(order *Order) {
    gco.orderPool.Put(order)
}

// GetPriceLevel gets a price level from the pool
func (gco *GCOptimizer) GetPriceLevel() *PriceLevel {
    return gco.priceLevelPool.Get()
}

// PutPriceLevel returns a price level to the pool
func (gco *GCOptimizer) PutPriceLevel(level *PriceLevel) {
    gco.priceLevelPool.Put(level)
}

// OptimizeGC sets GC parameters for low latency
func OptimizeGC() {
    // Set GC target percentage (default: 100%)
    // Lower values = more frequent GC = lower latency
    debug.SetGCPercent(50)

    // Set memory limit (Go 1.19+)
    debug.SetMemoryLimit(2 << 30) // 2GB limit
}

// MonitorGC monitors GC performance
func MonitorGC() {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()

    for range ticker.C {
        var m runtime.MemStats
        runtime.ReadMemStats(&m)

        fmt.Printf("GC Stats - Alloc: %d, Sys: %d, NumGC: %d, PauseTotal: %v\n",
            m.Alloc, m.Sys, m.NumGC, time.Duration(m.PauseTotalNs))
    }
}
```

---

## 🚀 **Complete Trading System Example**

### **Main Trading System**

```go
// TradingSystem represents the main trading system
type TradingSystem struct {
    orderBooks map[string]*OptimizedOrderBook
    gcOptimizer *GCOptimizer
    profiler   *Profiler
    mutex      sync.RWMutex
}

// NewTradingSystem creates a new trading system
func NewTradingSystem() *TradingSystem {
    return &TradingSystem{
        orderBooks:  make(map[string]*OptimizedOrderBook),
        gcOptimizer: NewGCOptimizer(),
        profiler:    NewProfiler(),
    }
}

// GetOrderBook gets or creates an order book for a symbol
func (ts *TradingSystem) GetOrderBook(symbol string) *OptimizedOrderBook {
    ts.mutex.RLock()
    ob, exists := ts.orderBooks[symbol]
    ts.mutex.RUnlock()

    if exists {
        return ob
    }

    ts.mutex.Lock()
    defer ts.mutex.Unlock()

    // Double-check pattern
    if ob, exists := ts.orderBooks[symbol]; exists {
        return ob
    }

    ob = NewOptimizedOrderBook(symbol)
    ts.orderBooks[symbol] = ob
    return ob
}

// ProcessOrder processes an order
func (ts *TradingSystem) ProcessOrder(symbol string, order *Order) error {
    ob := ts.GetOrderBook(symbol)

    ts.profiler.ProfileFunc("ProcessOrder", func() {
        ob.AddOrder(order)
    })

    return nil
}

// GetMarketData returns market data for a symbol
func (ts *TradingSystem) GetMarketData(symbol string) (int64, int64, int64, int64) {
    ob := ts.GetOrderBook(symbol)

    ts.profiler.ProfileFunc("GetMarketData", func() {
        ob.GetTopOfBook()
    })

    return ob.GetTopOfBook()
}

// RunPerformanceTest runs a performance test
func (ts *TradingSystem) RunPerformanceTest() {
    fmt.Println("Running Performance Test...")

    // Create test orders
    for i := 0; i < 10000; i++ {
        order := ts.gcOptimizer.GetOrder()
        order.ID = int64(i)
        order.Price = int64(100 + i%1000)
        order.Volume = int64(100)
        order.Side = Bid

        ts.ProcessOrder("AAPL", order)

        // Return to pool
        ts.gcOptimizer.PutOrder(order)
    }

    // Print performance statistics
    ts.profiler.PrintStats()
}

// Main function
func main() {
    // Optimize GC for low latency
    OptimizeGC()

    // Start GC monitoring
    go MonitorGC()

    // Create trading system
    ts := NewTradingSystem()

    // Run performance test
    ts.RunPerformanceTest()

    // Keep running
    select {}
}
```

---

## 🎯 **Key Takeaways from the Talk**

### **1. Performance Principles**

- **Avoid node-based containers** - Use slices instead of maps when possible
- **Linear search can be faster than binary search** - Better cache locality
- **Object pooling** - Reduce GC pressure
- **String interning** - Reduce memory allocations
- **Mechanical sympathy** - Design for hardware behavior

### **2. Go-Specific Optimizations**

- **Use `sync.Pool`** for object pooling
- **Pre-allocate slices** with known capacity
- **Avoid unnecessary allocations** in hot paths
- **Use `sync.RWMutex`** for read-heavy workloads
- **Profile with `runtime.MemStats`** and custom profilers

### **3. System Design Considerations**

- **Cache locality** - Design data structures for CPU cache behavior
- **Branch prediction** - Favor predictable access patterns
- **Memory layout** - Use struct of arrays when beneficial
- **Concurrency** - Design for lock-free or low-contention access
- **Monitoring** - Implement comprehensive performance monitoring

### **4. Trading System Specifics**

- **Order book design** - Optimize for top-of-book access
- **Price level management** - Use linear search for small collections
- **Memory management** - Pool frequently allocated objects
- **Performance monitoring** - Track latency distributions
- **GC optimization** - Tune for low latency requirements

---

**🎉 This comprehensive Go implementation demonstrates how to apply low-latency trading system principles from the C++ talk to Go, focusing on performance, memory efficiency, and mechanical sympathy! 🚀**
