# 🚀 Go Expertise: From Scratch to Super Duper Pro

> **Complete guide to mastering Go (Golang) from fundamentals to advanced concepts with real-world examples and FAANG interview questions**

## 📋 Table of Contents

1. [Go Fundamentals](#go-fundamentals)
2. [Advanced Go Concepts](#advanced-go-concepts)
3. [Design Patterns in Go](#design-patterns-in-go)
4. [Go Architecture & Best Practices](#go-architecture--best-practices)
5. [Debugging & Error Handling](#debugging--error-handling)
6. [Performance Optimization](#performance-optimization)
7. [Concurrency Deep Dive](#concurrency-deep-dive)
8. [FAANG Interview Questions](#faang-interview-questions)

---

## 🎯 Go Fundamentals

### **1. Go Basics & Syntax**

#### **Variables and Constants**

```go
package main

import "fmt"

func main() {
    // Variable declarations
    var name string = "Go"
    var age int = 10
    var isAwesome bool = true

    // Short declaration
    city := "San Francisco"

    // Multiple declarations
    var (
        firstName = "John"
        lastName  = "Doe"
        email     = "john@example.com"
    )

    // Constants
    const pi = 3.14159
    const (
        StatusOK    = 200
        StatusError = 500
    )

    fmt.Printf("Name: %s, Age: %d, City: %s\n", name, age, city)
}
```

#### **Data Types**

```go
package main

import "fmt"

func main() {
    // Basic types
    var i int = 42
    var f float64 = 3.14
    var s string = "Hello, Go!"
    var b bool = true

    // Complex types
    var arr [5]int = [5]int{1, 2, 3, 4, 5}
    var slice []int = []int{1, 2, 3, 4, 5}
    var m map[string]int = map[string]int{"a": 1, "b": 2}

    // Pointers
    var ptr *int = &i
    fmt.Printf("Value: %d, Pointer: %p\n", *ptr, ptr)

    // Structs
    type Person struct {
        Name string
        Age  int
    }

    person := Person{Name: "Alice", Age: 30}
    fmt.Printf("Person: %+v\n", person)
}
```

### **2. Functions & Methods**

#### **Function Basics**

```go
package main

import "fmt"

// Basic function
func add(a, b int) int {
    return a + b
}

// Multiple return values
func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

// Named return values
func calculate(a, b int) (sum, product int) {
    sum = a + b
    product = a * b
    return // naked return
}

// Variadic functions
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

// Function as parameter
func applyOperation(a, b int, op func(int, int) int) int {
    return op(a, b)
}

func main() {
    result := add(5, 3)
    fmt.Printf("Add: %d\n", result)

    quotient, err := divide(10, 2)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Divide: %d\n", quotient)
    }

    s, p := calculate(4, 5)
    fmt.Printf("Sum: %d, Product: %d\n", s, p)

    total := sum(1, 2, 3, 4, 5)
    fmt.Printf("Sum of variadic: %d\n", total)

    // Anonymous function
    multiply := func(x, y int) int { return x * y }
    result = applyOperation(3, 4, multiply)
    fmt.Printf("Apply operation: %d\n", result)
}
```

#### **Methods**

```go
package main

import "fmt"

type Rectangle struct {
    Width  float64
    Height float64
}

// Value receiver method
func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

// Pointer receiver method
func (r *Rectangle) Scale(factor float64) {
    r.Width *= factor
    r.Height *= factor
}

// Method on non-struct type
type MyInt int

func (m MyInt) IsEven() bool {
    return m%2 == 0
}

func main() {
    rect := Rectangle{Width: 10, Height: 5}
    fmt.Printf("Area: %.2f\n", rect.Area())

    rect.Scale(2)
    fmt.Printf("Scaled area: %.2f\n", rect.Area())

    num := MyInt(4)
    fmt.Printf("Is even: %t\n", num.IsEven())
}
```

### **3. Interfaces**

#### **Interface Basics**

```go
package main

import "fmt"

// Interface definition
type Shape interface {
    Area() float64
    Perimeter() float64
}

type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return 3.14159 * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * 3.14159 * c.Radius
}

type Square struct {
    Side float64
}

func (s Square) Area() float64 {
    return s.Side * s.Side
}

func (s Square) Perimeter() float64 {
    return 4 * s.Side
}

// Interface usage
func printShapeInfo(s Shape) {
    fmt.Printf("Area: %.2f, Perimeter: %.2f\n", s.Area(), s.Perimeter())
}

func main() {
    circle := Circle{Radius: 5}
    square := Square{Side: 4}

    printShapeInfo(circle)
    printShapeInfo(square)

    // Type assertion
    if c, ok := circle.(Circle); ok {
        fmt.Printf("Circle radius: %.2f\n", c.Radius)
    }

    // Type switch
    shapes := []Shape{circle, square}
    for _, shape := range shapes {
        switch s := shape.(type) {
        case Circle:
            fmt.Printf("Circle with radius: %.2f\n", s.Radius)
        case Square:
            fmt.Printf("Square with side: %.2f\n", s.Side)
        }
    }
}
```

---

## 🔥 Advanced Go Concepts

### **4. Goroutines & Channels**

#### **Goroutines**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

func worker(id int, jobs <-chan int, results chan<- int) {
    for j := range jobs {
        fmt.Printf("Worker %d processing job %d\n", id, j)
        time.Sleep(time.Second)
        results <- j * 2
    }
}

func main() {
    jobs := make(chan int, 100)
    results := make(chan int, 100)

    // Start workers
    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }

    // Send jobs
    for j := 1; j <= 5; j++ {
        jobs <- j
    }
    close(jobs)

    // Collect results
    for a := 1; a <= 5; a++ {
        result := <-results
        fmt.Printf("Result: %d\n", result)
    }
}
```

#### **Channels**

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    // Unbuffered channel
    ch := make(chan string)

    go func() {
        ch <- "Hello from goroutine!"
    }()

    msg := <-ch
    fmt.Println(msg)

    // Buffered channel
    buffered := make(chan int, 2)
    buffered <- 1
    buffered <- 2
    // buffered <- 3 // This would block

    fmt.Println(<-buffered)
    fmt.Println(<-buffered)

    // Channel directions
    go sendOnly(buffered)
    go receiveOnly(buffered)

    time.Sleep(time.Second)
}

func sendOnly(ch chan<- int) {
    ch <- 42
}

func receiveOnly(ch <-chan int) {
    value := <-ch
    fmt.Printf("Received: %d\n", value)
}
```

### **5. Select Statement**

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)

    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "from ch1"
    }()

    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "from ch2"
    }()

    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:
            fmt.Println(msg1)
        case msg2 := <-ch2:
            fmt.Println(msg2)
        case <-time.After(3 * time.Second):
            fmt.Println("timeout")
        }
    }
}
```

### **6. Context Package**

```go
package main

import (
    "context"
    "fmt"
    "time"
)

func longRunningTask(ctx context.Context) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            fmt.Println("Working...")
            time.Sleep(500 * time.Millisecond)
        }
    }
}

func main() {
    // With timeout
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    go func() {
        if err := longRunningTask(ctx); err != nil {
            fmt.Printf("Task failed: %v\n", err)
        }
    }()

    time.Sleep(3 * time.Second)
}
```

---

## 🎨 Design Patterns in Go

### **7. Singleton Pattern**

```go
package main

import (
    "fmt"
    "sync"
)

type Singleton struct {
    data string
}

var (
    instance *Singleton
    once     sync.Once
)

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{data: "initialized"}
    })
    return instance
}

func main() {
    s1 := GetInstance()
    s2 := GetInstance()

    fmt.Printf("s1 == s2: %t\n", s1 == s2)
    fmt.Printf("s1.data: %s\n", s1.data)
}
```

### **8. Factory Pattern**

```go
package main

import "fmt"

type Animal interface {
    Speak() string
}

type Dog struct{}
func (d Dog) Speak() string { return "Woof!" }

type Cat struct{}
func (c Cat) Speak() string { return "Meow!" }

type AnimalFactory struct{}

func (af AnimalFactory) CreateAnimal(animalType string) Animal {
    switch animalType {
    case "dog":
        return Dog{}
    case "cat":
        return Cat{}
    default:
        return nil
    }
}

func main() {
    factory := AnimalFactory{}

    dog := factory.CreateAnimal("dog")
    cat := factory.CreateAnimal("cat")

    fmt.Println(dog.Speak())
    fmt.Println(cat.Speak())
}
```

### **9. Observer Pattern**

```go
package main

import "fmt"

type Observer interface {
    Update(message string)
}

type Subject struct {
    observers []Observer
}

func (s *Subject) AddObserver(o Observer) {
    s.observers = append(s.observers, o)
}

func (s *Subject) NotifyObservers(message string) {
    for _, observer := range s.observers {
        observer.Update(message)
    }
}

type ConcreteObserver struct {
    name string
}

func (co ConcreteObserver) Update(message string) {
    fmt.Printf("%s received: %s\n", co.name, message)
}

func main() {
    subject := &Subject{}

    observer1 := ConcreteObserver{name: "Observer1"}
    observer2 := ConcreteObserver{name: "Observer2"}

    subject.AddObserver(observer1)
    subject.AddObserver(observer2)

    subject.NotifyObservers("Hello, observers!")
}
```

---

## 🏗️ Go Architecture & Best Practices

### **10. Project Structure**

```
my-go-project/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── config/
│   ├── handlers/
│   ├── models/
│   └── services/
├── pkg/
│   └── utils/
├── api/
│   └── openapi.yaml
├── migrations/
├── tests/
├── go.mod
├── go.sum
├── Dockerfile
└── README.md
```

### **11. Error Handling Best Practices**

```go
package main

import (
    "errors"
    "fmt"
    "log"
)

// Custom error types
type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation error on field %s: %s", e.Field, e.Message)
}

type BusinessError struct {
    Code    int
    Message string
}

func (e BusinessError) Error() string {
    return fmt.Sprintf("business error %d: %s", e.Code, e.Message)
}

// Error wrapping
func processUser(userID string) error {
    if userID == "" {
        return fmt.Errorf("processUser: %w", ValidationError{
            Field:   "userID",
            Message: "cannot be empty",
        })
    }

    if userID == "invalid" {
        return fmt.Errorf("processUser: %w", BusinessError{
            Code:    1001,
            Message: "user not found",
        })
    }

    return nil
}

// Error handling with context
func handleUserRequest(userID string) error {
    if err := processUser(userID); err != nil {
        // Log the error with context
        log.Printf("Failed to process user %s: %v", userID, err)

        // Check error type
        var validationErr ValidationError
        if errors.As(err, &validationErr) {
            return fmt.Errorf("invalid request: %w", err)
        }

        var businessErr BusinessError
        if errors.As(err, &businessErr) {
            return fmt.Errorf("business logic error: %w", err)
        }

        return fmt.Errorf("unexpected error: %w", err)
    }

    return nil
}

func main() {
    if err := handleUserRequest(""); err != nil {
        fmt.Printf("Error: %v\n", err)
    }

    if err := handleUserRequest("invalid"); err != nil {
        fmt.Printf("Error: %v\n", err)
    }
}
```

### **12. Configuration Management**

```go
package main

import (
    "encoding/json"
    "flag"
    "fmt"
    "os"
)

type Config struct {
    Server   ServerConfig   `json:"server"`
    Database DatabaseConfig `json:"database"`
    Redis    RedisConfig    `json:"redis"`
}

type ServerConfig struct {
    Host string `json:"host"`
    Port int    `json:"port"`
}

type DatabaseConfig struct {
    Host     string `json:"host"`
    Port     int    `json:"port"`
    Username string `json:"username"`
    Password string `json:"password"`
    Database string `json:"database"`
}

type RedisConfig struct {
    Host string `json:"host"`
    Port int    `json:"port"`
}

func LoadConfig() (*Config, error) {
    configFile := flag.String("config", "config.json", "Configuration file path")
    flag.Parse()

    file, err := os.Open(*configFile)
    if err != nil {
        return nil, fmt.Errorf("failed to open config file: %w", err)
    }
    defer file.Close()

    var config Config
    decoder := json.NewDecoder(file)
    if err := decoder.Decode(&config); err != nil {
        return nil, fmt.Errorf("failed to decode config: %w", err)
    }

    return &config, nil
}

func main() {
    config, err := LoadConfig()
    if err != nil {
        fmt.Printf("Failed to load config: %v\n", err)
        return
    }

    fmt.Printf("Server: %s:%d\n", config.Server.Host, config.Server.Port)
    fmt.Printf("Database: %s:%d\n", config.Database.Host, config.Database.Port)
}
```

---

## 🐛 Debugging & Error Handling

### **13. Debugging Techniques**

```go
package main

import (
    "fmt"
    "log"
    "runtime"
    "time"
)

// Debug logging
func debugLog(message string) {
    _, file, line, ok := runtime.Caller(1)
    if ok {
        log.Printf("[DEBUG] %s:%d - %s", file, line, message)
    }
}

// Performance timing
func timeFunction(fn func()) time.Duration {
    start := time.Now()
    fn()
    return time.Since(start)
}

// Memory usage
func printMemUsage() {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    fmt.Printf("Alloc = %d KB", bToKb(m.Alloc))
    fmt.Printf("\tTotalAlloc = %d KB", bToKb(m.TotalAlloc))
    fmt.Printf("\tSys = %d KB", bToKb(m.Sys))
    fmt.Printf("\tNumGC = %d\n", m.NumGC)
}

func bToKb(b uint64) uint64 {
    return b / 1024
}

func main() {
    debugLog("Starting application")

    duration := timeFunction(func() {
        // Simulate some work
        time.Sleep(100 * time.Millisecond)
    })

    fmt.Printf("Function took: %v\n", duration)
    printMemUsage()
}
```

### **14. Profiling**

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
    for i := 0; i < 1000000; i++ {
        _ = i * i
    }
}

func memoryIntensiveTask() {
    data := make([]byte, 1024*1024) // 1MB
    for i := range data {
        data[i] = byte(i % 256)
    }
}

func main() {
    // Start pprof server
    go func() {
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()

    // Simulate some work
    for i := 0; i < 10; i++ {
        cpuIntensiveTask()
        memoryIntensiveTask()
        runtime.GC() // Force garbage collection
        time.Sleep(100 * time.Millisecond)
    }

    fmt.Println("Profiling server running on http://localhost:6060/debug/pprof/")
    fmt.Println("Use 'go tool pprof http://localhost:6060/debug/pprof/profile' for CPU profiling")
    fmt.Println("Use 'go tool pprof http://localhost:6060/debug/pprof/heap' for memory profiling")

    // Keep the program running
    select {}
}
```

---

## ⚡ Performance Optimization

### **15. Memory Optimization**

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

// Object pooling
type ObjectPool struct {
    pool sync.Pool
}

func NewObjectPool() *ObjectPool {
    return &ObjectPool{
        pool: sync.Pool{
            New: func() interface{} {
                return make([]byte, 1024)
            },
        },
    }
}

func (op *ObjectPool) Get() []byte {
    return op.pool.Get().([]byte)
}

func (op *ObjectPool) Put(obj []byte) {
    op.pool.Put(obj)
}

// String builder optimization
func buildStringOptimized(parts []string) string {
    var builder strings.Builder
    builder.Grow(1000) // Pre-allocate capacity

    for _, part := range parts {
        builder.WriteString(part)
    }

    return builder.String()
}

// Slice pre-allocation
func processDataOptimized(data []int) []int {
    result := make([]int, 0, len(data)) // Pre-allocate capacity

    for _, item := range data {
        if item%2 == 0 {
            result = append(result, item*2)
        }
    }

    return result
}

func main() {
    // Object pooling example
    pool := NewObjectPool()

    obj := pool.Get()
    // Use the object
    fmt.Printf("Got object of length: %d\n", len(obj))
    pool.Put(obj)

    // String building example
    parts := []string{"Hello", " ", "World", "!"}
    result := buildStringOptimized(parts)
    fmt.Printf("Built string: %s\n", result)

    // Slice optimization example
    data := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    processed := processDataOptimized(data)
    fmt.Printf("Processed data: %v\n", processed)
}
```

### **16. Concurrency Optimization**

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

// Worker pool pattern
type WorkerPool struct {
    workers    int
    jobs       chan func()
    wg         sync.WaitGroup
}

func NewWorkerPool(workers int) *WorkerPool {
    return &WorkerPool{
        workers: workers,
        jobs:    make(chan func(), workers*2),
    }
}

func (wp *WorkerPool) Start() {
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker()
    }
}

func (wp *WorkerPool) worker() {
    defer wp.wg.Done()
    for job := range wp.jobs {
        job()
    }
}

func (wp *WorkerPool) Submit(job func()) {
    wp.jobs <- job
}

func (wp *WorkerPool) Stop() {
    close(wp.jobs)
    wp.wg.Wait()
}

// Rate limiting
type RateLimiter struct {
    tokens chan struct{}
    ticker *time.Ticker
}

func NewRateLimiter(rate int, per time.Duration) *RateLimiter {
    rl := &RateLimiter{
        tokens: make(chan struct{}, rate),
        ticker: time.NewTicker(per / time.Duration(rate)),
    }

    go rl.refill()
    return rl
}

func (rl *RateLimiter) refill() {
    for range rl.ticker.C {
        select {
        case rl.tokens <- struct{}{}:
        default:
        }
    }
}

func (rl *RateLimiter) Allow() bool {
    select {
    case <-rl.tokens:
        return true
    default:
        return false
    }
}

func (rl *RateLimiter) Stop() {
    rl.ticker.Stop()
}

func main() {
    // Worker pool example
    pool := NewWorkerPool(runtime.NumCPU())
    pool.Start()

    for i := 0; i < 10; i++ {
        i := i // Capture loop variable
        pool.Submit(func() {
            fmt.Printf("Processing job %d\n", i)
            time.Sleep(100 * time.Millisecond)
        })
    }

    pool.Stop()

    // Rate limiting example
    limiter := NewRateLimiter(5, time.Second)
    defer limiter.Stop()

    for i := 0; i < 10; i++ {
        if limiter.Allow() {
            fmt.Printf("Request %d allowed\n", i)
        } else {
            fmt.Printf("Request %d rate limited\n", i)
        }
        time.Sleep(100 * time.Millisecond)
    }
}
```

---

## 🔄 Concurrency Deep Dive

### **17. Advanced Concurrency Patterns**

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Fan-out, Fan-in pattern
func fanOut(input <-chan int, workers int) []<-chan int {
    outputs := make([]<-chan int, workers)

    for i := 0; i < workers; i++ {
        output := make(chan int)
        outputs[i] = output

        go func() {
            defer close(output)
            for value := range input {
                // Simulate processing
                time.Sleep(100 * time.Millisecond)
                output <- value * 2
            }
        }()
    }

    return outputs
}

func fanIn(inputs []<-chan int) <-chan int {
    output := make(chan int)
    var wg sync.WaitGroup

    for _, input := range inputs {
        wg.Add(1)
        go func(ch <-chan int) {
            defer wg.Done()
            for value := range ch {
                output <- value
            }
        }(input)
    }

    go func() {
        wg.Wait()
        close(output)
    }()

    return output
}

// Pipeline pattern
func pipeline(input <-chan int) <-chan int {
    output := make(chan int)

    go func() {
        defer close(output)
        for value := range input {
            // Stage 1: Multiply by 2
            value *= 2

            // Stage 2: Add 1
            value += 1

            output <- value
        }
    }()

    return output
}

// Circuit breaker pattern
type CircuitBreaker struct {
    maxFailures int
    timeout     time.Duration
    failures    int
    lastFailure time.Time
    state       string
    mutex       sync.RWMutex
}

func NewCircuitBreaker(maxFailures int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        maxFailures: maxFailures,
        timeout:     timeout,
        state:       "closed",
    }
}

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()

    if cb.state == "open" {
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = "half-open"
        } else {
            return fmt.Errorf("circuit breaker is open")
        }
    }

    err := fn()
    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()
        if cb.failures >= cb.maxFailures {
            cb.state = "open"
        }
        return err
    }

    cb.failures = 0
    cb.state = "closed"
    return nil
}

func main() {
    // Fan-out, Fan-in example
    input := make(chan int)
    go func() {
        defer close(input)
        for i := 1; i <= 10; i++ {
            input <- i
        }
    }()

    outputs := fanOut(input, 3)
    result := fanIn(outputs)

    for value := range result {
        fmt.Printf("Processed: %d\n", value)
    }

    // Pipeline example
    input2 := make(chan int)
    go func() {
        defer close(input2)
        for i := 1; i <= 5; i++ {
            input2 <- i
        }
    }()

    output := pipeline(input2)
    for value := range output {
        fmt.Printf("Pipeline result: %d\n", value)
    }

    // Circuit breaker example
    cb := NewCircuitBreaker(3, 5*time.Second)

    for i := 0; i < 5; i++ {
        err := cb.Call(func() error {
            if i < 3 {
                return fmt.Errorf("simulated error")
            }
            return nil
        })

        if err != nil {
            fmt.Printf("Call %d failed: %v\n", i, err)
        } else {
            fmt.Printf("Call %d succeeded\n", i)
        }
    }
}
```

---

## 🎯 FAANG Interview Questions

### **Google Interview Questions**

#### **1. Implement a Concurrent Map**

**Question**: "Implement a thread-safe map that can handle concurrent reads and writes efficiently."

**Answer**:

```go
package main

import (
    "fmt"
    "sync"
)

type ConcurrentMap struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func NewConcurrentMap() *ConcurrentMap {
    return &ConcurrentMap{
        data: make(map[string]interface{}),
    }
}

func (cm *ConcurrentMap) Get(key string) (interface{}, bool) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    value, exists := cm.data[key]
    return value, exists
}

func (cm *ConcurrentMap) Set(key string, value interface{}) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    cm.data[key] = value
}

func (cm *ConcurrentMap) Delete(key string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    delete(cm.data, key)
}

func (cm *ConcurrentMap) Size() int {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    return len(cm.data)
}

func main() {
    cm := NewConcurrentMap()

    // Concurrent writes
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(i int) {
            defer wg.Done()
            cm.Set(fmt.Sprintf("key%d", i), fmt.Sprintf("value%d", i))
        }(i)
    }

    wg.Wait()

    // Concurrent reads
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(i int) {
            defer wg.Done()
            if value, exists := cm.Get(fmt.Sprintf("key%d", i)); exists {
                fmt.Printf("Key%d: %v\n", i, value)
            }
        }(i)
    }

    wg.Wait()
    fmt.Printf("Map size: %d\n", cm.Size())
}
```

#### **2. Implement a Rate Limiter**

**Question**: "Design a rate limiter that can handle different rate limits for different users."

**Answer**:

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type RateLimiter struct {
    limiters map[string]*UserLimiter
    mu       sync.RWMutex
}

type UserLimiter struct {
    tokens   int
    lastTime time.Time
    rate     int
    capacity int
    mu       sync.Mutex
}

func NewRateLimiter() *RateLimiter {
    return &RateLimiter{
        limiters: make(map[string]*UserLimiter),
    }
}

func (rl *RateLimiter) GetLimiter(userID string, rate, capacity int) *UserLimiter {
    rl.mu.Lock()
    defer rl.mu.Unlock()

    if limiter, exists := rl.limiters[userID]; exists {
        return limiter
    }

    limiter := &UserLimiter{
        tokens:   capacity,
        lastTime: time.Now(),
        rate:     rate,
        capacity: capacity,
    }

    rl.limiters[userID] = limiter
    return limiter
}

func (ul *UserLimiter) Allow() bool {
    ul.mu.Lock()
    defer ul.mu.Unlock()

    now := time.Now()
    elapsed := now.Sub(ul.lastTime)

    // Add tokens based on elapsed time
    tokensToAdd := int(elapsed.Seconds()) * ul.rate
    ul.tokens = min(ul.capacity, ul.tokens+tokensToAdd)
    ul.lastTime = now

    if ul.tokens > 0 {
        ul.tokens--
        return true
    }

    return false
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func main() {
    rl := NewRateLimiter()

    // Test rate limiting
    user1 := rl.GetLimiter("user1", 2, 5) // 2 tokens per second, capacity 5
    user2 := rl.GetLimiter("user2", 1, 3) // 1 token per second, capacity 3

    for i := 0; i < 10; i++ {
        if user1.Allow() {
            fmt.Printf("User1 request %d: allowed\n", i)
        } else {
            fmt.Printf("User1 request %d: rate limited\n", i)
        }

        if user2.Allow() {
            fmt.Printf("User2 request %d: allowed\n", i)
        } else {
            fmt.Printf("User2 request %d: rate limited\n", i)
        }

        time.Sleep(500 * time.Millisecond)
    }
}
```

### **Meta Interview Questions**

#### **3. Implement a Message Queue**

**Question**: "Design a message queue system that can handle high throughput and ensure message delivery."

**Answer**:

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Message struct {
    ID        string
    Content   string
    Timestamp time.Time
    Retries   int
}

type MessageQueue struct {
    messages    chan Message
    subscribers map[string][]chan Message
    mu          sync.RWMutex
    wg          sync.WaitGroup
}

func NewMessageQueue(bufferSize int) *MessageQueue {
    return &MessageQueue{
        messages:    make(chan Message, bufferSize),
        subscribers: make(map[string][]chan Message),
    }
}

func (mq *MessageQueue) Subscribe(topic string) <-chan Message {
    mq.mu.Lock()
    defer mq.mu.Unlock()

    ch := make(chan Message, 10)
    mq.subscribers[topic] = append(mq.subscribers[topic], ch)
    return ch
}

func (mq *MessageQueue) Publish(topic string, content string) {
    message := Message{
        ID:        fmt.Sprintf("%d", time.Now().UnixNano()),
        Content:   content,
        Timestamp: time.Now(),
        Retries:   0,
    }

    mq.messages <- message
}

func (mq *MessageQueue) Start() {
    mq.wg.Add(1)
    go mq.processMessages()
}

func (mq *MessageQueue) processMessages() {
    defer mq.wg.Done()

    for message := range mq.messages {
        mq.mu.RLock()
        subscribers := mq.subscribers["default"] // Simplified: all messages go to default topic
        mq.mu.RUnlock()

        for _, ch := range subscribers {
            select {
            case ch <- message:
            default:
                // Subscriber is busy, skip
            }
        }
    }
}

func (mq *MessageQueue) Stop() {
    close(mq.messages)
    mq.wg.Wait()
}

func main() {
    mq := NewMessageQueue(100)
    mq.Start()

    // Subscribe to messages
    ch1 := mq.Subscribe("default")
    ch2 := mq.Subscribe("default")

    // Publish messages
    go func() {
        for i := 0; i < 5; i++ {
            mq.Publish("default", fmt.Sprintf("Message %d", i))
            time.Sleep(100 * time.Millisecond)
        }
    }()

    // Consume messages
    go func() {
        for message := range ch1 {
            fmt.Printf("Consumer 1 received: %s\n", message.Content)
        }
    }()

    go func() {
        for message := range ch2 {
            fmt.Printf("Consumer 2 received: %s\n", message.Content)
        }
    }()

    time.Sleep(2 * time.Second)
    mq.Stop()
}
```

### **Amazon Interview Questions**

#### **4. Implement a Distributed Cache**

**Question**: "Design a distributed cache system with consistent hashing and replication."

**Answer**:

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "sync"
    "time"
)

type CacheNode struct {
    ID   string
    Data map[string]interface{}
    mu   sync.RWMutex
}

type ConsistentHash struct {
    nodes    []string
    replicas int
    mu       sync.RWMutex
}

func NewConsistentHash(replicas int) *ConsistentHash {
    return &ConsistentHash{
        replicas: replicas,
    }
}

func (ch *ConsistentHash) AddNode(node string) {
    ch.mu.Lock()
    defer ch.mu.Unlock()

    for i := 0; i < ch.replicas; i++ {
        hash := ch.hash(fmt.Sprintf("%s:%d", node, i))
        ch.nodes = append(ch.nodes, fmt.Sprintf("%d:%s", hash, node))
    }

    sort.Strings(ch.nodes)
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mu.RLock()
    defer ch.mu.RUnlock()

    if len(ch.nodes) == 0 {
        return ""
    }

    hash := ch.hash(key)
    idx := sort.Search(len(ch.nodes), func(i int) bool {
        return ch.nodes[i] >= fmt.Sprintf("%d:", hash)
    })

    if idx == len(ch.nodes) {
        idx = 0
    }

    nodeHash := ch.nodes[idx]
    return nodeHash[11:] // Remove hash prefix
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

type DistributedCache struct {
    nodes  map[string]*CacheNode
    hash   *ConsistentHash
    mu     sync.RWMutex
}

func NewDistributedCache() *DistributedCache {
    return &DistributedCache{
        nodes: make(map[string]*CacheNode),
        hash:  NewConsistentHash(3),
    }
}

func (dc *DistributedCache) AddNode(nodeID string) {
    dc.mu.Lock()
    defer dc.mu.Unlock()

    node := &CacheNode{
        ID:   nodeID,
        Data: make(map[string]interface{}),
    }

    dc.nodes[nodeID] = node
    dc.hash.AddNode(nodeID)
}

func (dc *DistributedCache) Get(key string) (interface{}, bool) {
    dc.mu.RLock()
    nodeID := dc.hash.GetNode(key)
    dc.mu.RUnlock()

    if nodeID == "" {
        return nil, false
    }

    dc.mu.RLock()
    node := dc.nodes[nodeID]
    dc.mu.RUnlock()

    node.mu.RLock()
    defer node.mu.RUnlock()

    value, exists := node.Data[key]
    return value, exists
}

func (dc *DistributedCache) Set(key string, value interface{}) {
    dc.mu.RLock()
    nodeID := dc.hash.GetNode(key)
    dc.mu.RUnlock()

    if nodeID == "" {
        return
    }

    dc.mu.RLock()
    node := dc.nodes[nodeID]
    dc.mu.RUnlock()

    node.mu.Lock()
    defer node.mu.Unlock()

    node.Data[key] = value
}

func main() {
    cache := NewDistributedCache()

    // Add nodes
    cache.AddNode("node1")
    cache.AddNode("node2")
    cache.AddNode("node3")

    // Set values
    cache.Set("key1", "value1")
    cache.Set("key2", "value2")
    cache.Set("key3", "value3")

    // Get values
    if value, exists := cache.Get("key1"); exists {
        fmt.Printf("key1: %v\n", value)
    }

    if value, exists := cache.Get("key2"); exists {
        fmt.Printf("key2: %v\n", value)
    }

    if value, exists := cache.Get("key3"); exists {
        fmt.Printf("key3: %v\n", value)
    }
}
```

---

## 📚 Additional Resources

### **Books**

- [The Go Programming Language](https://www.gopl.io/) - Alan Donovan & Brian Kernighan
- [Effective Go](https://golang.org/doc/effective_go.html) - Official Go documentation
- [Go in Action](https://www.manning.com/books/go-in-action) - William Kennedy

### **Online Resources**

- [Go by Example](https://gobyexample.com/) - Hands-on introduction to Go
- [Go Playground](https://play.golang.org/) - Online Go compiler
- [Go Blog](https://blog.golang.org/) - Official Go blog

### **Video Resources**

- [Gopher Academy](https://www.youtube.com/c/GopherAcademy) - Go conferences and talks
- [JustForFunc](https://www.youtube.com/c/JustForFunc) - Go programming videos
- [Go Time](https://changelog.com/gotime) - Go podcast

---

_This comprehensive guide covers Go from fundamentals to advanced concepts, including real-world examples and FAANG interview questions to help you master Go programming._
