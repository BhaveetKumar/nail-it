# Golang Coding Patterns for Interviews

## 📚 Table of Contents

1. [Concurrency Patterns](#concurrency-patterns)
2. [Error Handling](#error-handling)
3. [Interface Design](#interface-design)
4. [Data Structures](#data-structures)
5. [Algorithm Patterns](#algorithm-patterns)
6. [Testing Patterns](#testing-patterns)
7. [Performance Optimization](#performance-optimization)

## Concurrency Patterns

### 1. Goroutines and Channels

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Basic Goroutine Pattern
func basicGoroutine() {
    go func() {
        fmt.Println("Running in goroutine")
    }()

    // Wait for goroutine to complete
    time.Sleep(100 * time.Millisecond)
}

// Channel Communication
func channelPattern() {
    ch := make(chan string, 1)

    go func() {
        ch <- "Hello from goroutine"
    }()

    msg := <-ch
    fmt.Println(msg)
}

// Worker Pool Pattern
type WorkerPool struct {
    workers    int
    jobQueue   chan Job
    resultQueue chan Result
    wg         sync.WaitGroup
}

type Job struct {
    ID   int
    Data interface{}
}

type Result struct {
    JobID int
    Data  interface{}
    Error error
}

func NewWorkerPool(workers int) *WorkerPool {
    return &WorkerPool{
        workers:     workers,
        jobQueue:    make(chan Job, 100),
        resultQueue: make(chan Result, 100),
    }
}

func (wp *WorkerPool) Start(ctx context.Context) {
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker(ctx, i)
    }
}

func (wp *WorkerPool) worker(ctx context.Context, id int) {
    defer wp.wg.Done()

    for {
        select {
        case job := <-wp.jobQueue:
            result := wp.processJob(job)
            wp.resultQueue <- result
        case <-ctx.Done():
            return
        }
    }
}

func (wp *WorkerPool) processJob(job Job) Result {
    // Simulate work
    time.Sleep(100 * time.Millisecond)

    return Result{
        JobID: job.ID,
        Data:  fmt.Sprintf("Processed job %d", job.ID),
        Error: nil,
    }
}

func (wp *WorkerPool) Submit(job Job) {
    wp.jobQueue <- job
}

func (wp *WorkerPool) GetResult() Result {
    return <-wp.resultQueue
}

func (wp *WorkerPool) Close() {
    close(wp.jobQueue)
    wp.wg.Wait()
    close(wp.resultQueue)
}
```

### 2. Context Pattern

```go
// Context for cancellation and timeouts
func contextPattern() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // Pass context to goroutines
    go doWork(ctx)

    select {
    case <-ctx.Done():
        fmt.Println("Context cancelled:", ctx.Err())
    case <-time.After(10 * time.Second):
        fmt.Println("Work completed")
    }
}

func doWork(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            fmt.Println("Work cancelled")
            return
        default:
            // Do work
            time.Sleep(100 * time.Millisecond)
        }
    }
}
```

### 3. Select Statement Patterns

```go
// Non-blocking channel operations
func nonBlockingPattern() {
    ch := make(chan string)

    select {
    case msg := <-ch:
        fmt.Println("Received:", msg)
    default:
        fmt.Println("No message available")
    }
}

// Timeout pattern
func timeoutPattern() {
    ch := make(chan string)

    select {
    case msg := <-ch:
        fmt.Println("Received:", msg)
    case <-time.After(1 * time.Second):
        fmt.Println("Timeout")
    }
}

// Multiple channel operations
func multipleChannelsPattern() {
    ch1 := make(chan string)
    ch2 := make(chan string)

    go func() {
        ch1 <- "from ch1"
    }()

    go func() {
        ch2 <- "from ch2"
    }()

    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:
            fmt.Println("Received from ch1:", msg1)
        case msg2 := <-ch2:
            fmt.Println("Received from ch2:", msg2)
        }
    }
}
```

## Error Handling

### 1. Error Wrapping and Unwrapping

```go
import (
    "errors"
    "fmt"
)

// Custom error types
type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation error on field %s: %s", e.Field, e.Message)
}

// Error wrapping
func processUser(user User) error {
    if err := validateUser(user); err != nil {
        return fmt.Errorf("failed to process user: %w", err)
    }

    if err := saveUser(user); err != nil {
        return fmt.Errorf("failed to save user: %w", err)
    }

    return nil
}

func validateUser(user User) error {
    if user.Name == "" {
        return ValidationError{Field: "name", Message: "name is required"}
    }
    return nil
}

// Error unwrapping and checking
func handleError(err error) {
    var validationErr ValidationError
    if errors.As(err, &validationErr) {
        fmt.Printf("Validation error: %s\n", validationErr.Message)
    } else {
        fmt.Printf("Other error: %v\n", err)
    }
}
```

### 2. Error Groups

```go
import "golang.org/x/sync/errgroup"

func errorGroupPattern() {
    var g errgroup.Group

    // Run multiple goroutines
    g.Go(func() error {
        return doWork1()
    })

    g.Go(func() error {
        return doWork2()
    })

    g.Go(func() error {
        return doWork3()
    })

    // Wait for all goroutines to complete
    if err := g.Wait(); err != nil {
        fmt.Printf("Error occurred: %v\n", err)
    }
}

func doWork1() error {
    time.Sleep(100 * time.Millisecond)
    return nil
}

func doWork2() error {
    time.Sleep(200 * time.Millisecond)
    return errors.New("work2 failed")
}

func doWork3() error {
    time.Sleep(150 * time.Millisecond)
    return nil
}
```

## Interface Design

### 1. Interface Segregation

```go
// Large interface (avoid this)
type BadInterface interface {
    Read() error
    Write() error
    Close() error
    Lock() error
    Unlock() error
    Compress() error
    Decompress() error
}

// Segregated interfaces (preferred)
type Reader interface {
    Read() error
}

type Writer interface {
    Write() error
}

type Closer interface {
    Close() error
}

type Locker interface {
    Lock() error
    Unlock() error
}

// Compose interfaces as needed
type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}
```

### 2. Interface Composition

```go
// Base interfaces
type Animal interface {
    Speak() string
}

type Walker interface {
    Walk() string
}

type Swimmer interface {
    Swim() string
}

// Composed interfaces
type LandAnimal interface {
    Animal
    Walker
}

type WaterAnimal interface {
    Animal
    Swimmer
}

type Amphibian interface {
    Animal
    Walker
    Swimmer
}

// Implementation
type Dog struct{}

func (d Dog) Speak() string { return "Woof!" }
func (d Dog) Walk() string  { return "Walking on land" }

type Fish struct{}

func (f Fish) Speak() string { return "Blub!" }
func (f Fish) Swim() string  { return "Swimming in water" }

type Frog struct{}

func (f Frog) Speak() string { return "Ribbit!" }
func (f Frog) Walk() string  { return "Walking on land" }
func (f Frog) Swim() string  { return "Swimming in water" }
```

### 3. Interface Assertions

```go
func interfaceAssertions() {
    var animal Animal = Dog{}

    // Type assertion
    if dog, ok := animal.(Dog); ok {
        fmt.Println("It's a dog:", dog.Speak())
    }

    // Type switch
    switch v := animal.(type) {
    case Dog:
        fmt.Println("Dog:", v.Speak())
    case Fish:
        fmt.Println("Fish:", v.Speak())
    case Frog:
        fmt.Println("Frog:", v.Speak())
    default:
        fmt.Println("Unknown animal")
    }

    // Interface assertion
    if walker, ok := animal.(Walker); ok {
        fmt.Println("Can walk:", walker.Walk())
    }
}
```

## Data Structures

### 1. Custom Data Structures

```go
// Generic Stack
type Stack[T any] struct {
    items []T
}

func NewStack[T any]() *Stack[T] {
    return &Stack[T]{items: make([]T, 0)}
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }

    index := len(s.items) - 1
    item := s.items[index]
    s.items = s.items[:index]
    return item, true
}

func (s *Stack[T]) Peek() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }

    return s.items[len(s.items)-1], true
}

func (s *Stack[T]) IsEmpty() bool {
    return len(s.items) == 0
}

// Generic Queue
type Queue[T any] struct {
    items []T
}

func NewQueue[T any]() *Queue[T] {
    return &Queue[T]{items: make([]T, 0)}
}

func (q *Queue[T]) Enqueue(item T) {
    q.items = append(q.items, item)
}

func (q *Queue[T]) Dequeue() (T, bool) {
    if len(q.items) == 0 {
        var zero T
        return zero, false
    }

    item := q.items[0]
    q.items = q.items[1:]
    return item, true
}

func (q *Queue[T]) IsEmpty() bool {
    return len(q.items) == 0
}
```

### 2. LRU Cache Implementation

```go
type LRUCache struct {
    capacity int
    cache    map[string]*Node
    head     *Node
    tail     *Node
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

    // Initialize dummy head and tail
    lru.head = &Node{}
    lru.tail = &Node{}
    lru.head.next = lru.tail
    lru.tail.prev = lru.head

    return lru
}

func (lru *LRUCache) Get(key string) (interface{}, bool) {
    if node, exists := lru.cache[key]; exists {
        lru.moveToHead(node)
        return node.value, true
    }
    return nil, false
}

func (lru *LRUCache) Put(key string, value interface{}) {
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
    if lru.tail.prev != lru.head {
        lastNode := lru.tail.prev
        lru.removeNode(lastNode)
        delete(lru.cache, lastNode.key)
    }
}
```

## Algorithm Patterns

### 1. Two Pointers

```go
// Two Sum problem
func twoSum(nums []int, target int) []int {
    left, right := 0, len(nums)-1

    for left < right {
        sum := nums[left] + nums[right]
        if sum == target {
            return []int{left, right}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }

    return nil
}

// Remove duplicates from sorted array
func removeDuplicates(nums []int) int {
    if len(nums) == 0 {
        return 0
    }

    slow := 0
    for fast := 1; fast < len(nums); fast++ {
        if nums[fast] != nums[slow] {
            slow++
            nums[slow] = nums[fast]
        }
    }

    return slow + 1
}
```

### 2. Sliding Window

```go
// Maximum sum of subarray of size k
func maxSumSubarray(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }

    windowSum := 0
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }

    maxSum := windowSum
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        maxSum = max(maxSum, windowSum)
    }

    return maxSum
}

// Longest substring without repeating characters
func lengthOfLongestSubstring(s string) int {
    charMap := make(map[byte]int)
    left, maxLen := 0, 0

    for right := 0; right < len(s); right++ {
        if lastIndex, exists := charMap[s[right]]; exists && lastIndex >= left {
            left = lastIndex + 1
        }

        charMap[s[right]] = right
        maxLen = max(maxLen, right-left+1)
    }

    return maxLen
}
```

### 3. Dynamic Programming

```go
// Fibonacci with memoization
func fibonacci(n int) int {
    memo := make(map[int]int)
    return fibMemo(n, memo)
}

func fibMemo(n int, memo map[int]int) int {
    if n <= 1 {
        return n
    }

    if val, exists := memo[n]; exists {
        return val
    }

    memo[n] = fibMemo(n-1, memo) + fibMemo(n-2, memo)
    return memo[n]
}

// Longest Common Subsequence
func longestCommonSubsequence(text1, text2 string) int {
    m, n := len(text1), len(text2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }

    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if text1[i-1] == text2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }

    return dp[m][n]
}
```

## Testing Patterns

### 1. Table-Driven Tests

```go
func TestTwoSum(t *testing.T) {
    tests := []struct {
        name     string
        nums     []int
        target   int
        expected []int
    }{
        {
            name:     "valid case",
            nums:     []int{2, 7, 11, 15},
            target:   9,
            expected: []int{0, 1},
        },
        {
            name:     "no solution",
            nums:     []int{2, 7, 11, 15},
            target:   3,
            expected: nil,
        },
        {
            name:     "duplicate numbers",
            nums:     []int{3, 3},
            target:   6,
            expected: []int{0, 1},
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := twoSum(tt.nums, tt.target)
            if !slicesEqual(result, tt.expected) {
                t.Errorf("twoSum() = %v, want %v", result, tt.expected)
            }
        })
    }
}

func slicesEqual(a, b []int) bool {
    if len(a) != len(b) {
        return false
    }
    for i := range a {
        if a[i] != b[i] {
            return false
        }
    }
    return true
}
```

### 2. Mock Testing

```go
// Interface for dependency injection
type UserService interface {
    GetUser(id string) (*User, error)
    CreateUser(user *User) error
}

// Mock implementation
type MockUserService struct {
    users map[string]*User
}

func NewMockUserService() *MockUserService {
    return &MockUserService{
        users: make(map[string]*User),
    }
}

func (m *MockUserService) GetUser(id string) (*User, error) {
    if user, exists := m.users[id]; exists {
        return user, nil
    }
    return nil, errors.New("user not found")
}

func (m *MockUserService) CreateUser(user *User) error {
    m.users[user.ID] = user
    return nil
}

// Test with mock
func TestUserHandler(t *testing.T) {
    mockService := NewMockUserService()
    handler := NewUserHandler(mockService)

    user := &User{ID: "1", Name: "John"}
    err := handler.CreateUser(user)
    if err != nil {
        t.Fatalf("CreateUser failed: %v", err)
    }

    retrieved, err := handler.GetUser("1")
    if err != nil {
        t.Fatalf("GetUser failed: %v", err)
    }

    if retrieved.Name != "John" {
        t.Errorf("Expected John, got %s", retrieved.Name)
    }
}
```

## Performance Optimization

### 1. Memory Pool Pattern

```go
type ObjectPool struct {
    pool chan interface{}
    new  func() interface{}
}

func NewObjectPool(size int, newFunc func() interface{}) *ObjectPool {
    pool := &ObjectPool{
        pool: make(chan interface{}, size),
        new:  newFunc,
    }

    // Pre-populate pool
    for i := 0; i < size; i++ {
        pool.pool <- newFunc()
    }

    return pool
}

func (p *ObjectPool) Get() interface{} {
    select {
    case obj := <-p.pool:
        return obj
    default:
        return p.new()
    }
}

func (p *ObjectPool) Put(obj interface{}) {
    select {
    case p.pool <- obj:
    default:
        // Pool is full, discard object
    }
}
```

### 2. String Builder Pattern

```go
func buildStringEfficiently(parts []string) string {
    var builder strings.Builder
    builder.Grow(len(parts) * 10) // Pre-allocate capacity

    for _, part := range parts {
        builder.WriteString(part)
    }

    return builder.String()
}
```

### 3. Goroutine Pool for CPU-bound Tasks

```go
type CPUPool struct {
    workers int
    jobs    chan func()
    wg      sync.WaitGroup
}

func NewCPUPool(workers int) *CPUPool {
    pool := &CPUPool{
        workers: workers,
        jobs:    make(chan func(), 100),
    }

    for i := 0; i < workers; i++ {
        pool.wg.Add(1)
        go pool.worker()
    }

    return pool
}

func (p *CPUPool) worker() {
    defer p.wg.Done()

    for job := range p.jobs {
        job()
    }
}

func (p *CPUPool) Submit(job func()) {
    p.jobs <- job
}

func (p *CPUPool) Close() {
    close(p.jobs)
    p.wg.Wait()
}
```

## Common Interview Patterns

### 1. Builder Pattern

```go
type QueryBuilder struct {
    table    string
    columns  []string
    where    []string
    orderBy  string
    limit    int
    offset   int
}

func NewQueryBuilder() *QueryBuilder {
    return &QueryBuilder{}
}

func (qb *QueryBuilder) Table(table string) *QueryBuilder {
    qb.table = table
    return qb
}

func (qb *QueryBuilder) Select(columns ...string) *QueryBuilder {
    qb.columns = columns
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    qb.where = append(qb.where, condition)
    return qb
}

func (qb *QueryBuilder) OrderBy(column string) *QueryBuilder {
    qb.orderBy = column
    return qb
}

func (qb *QueryBuilder) Limit(limit int) *QueryBuilder {
    qb.limit = limit
    return qb
}

func (qb *QueryBuilder) Build() string {
    var query strings.Builder

    query.WriteString("SELECT ")
    if len(qb.columns) == 0 {
        query.WriteString("*")
    } else {
        query.WriteString(strings.Join(qb.columns, ", "))
    }

    query.WriteString(" FROM ")
    query.WriteString(qb.table)

    if len(qb.where) > 0 {
        query.WriteString(" WHERE ")
        query.WriteString(strings.Join(qb.where, " AND "))
    }

    if qb.orderBy != "" {
        query.WriteString(" ORDER BY ")
        query.WriteString(qb.orderBy)
    }

    if qb.limit > 0 {
        query.WriteString(" LIMIT ")
        query.WriteString(strconv.Itoa(qb.limit))
    }

    return query.String()
}
```

### 2. Observer Pattern

```go
type Event struct {
    Type string
    Data interface{}
}

type Observer interface {
    Update(event Event)
}

type Subject interface {
    Attach(observer Observer)
    Detach(observer Observer)
    Notify(event Event)
}

type EventBus struct {
    observers []Observer
    mutex     sync.RWMutex
}

func NewEventBus() *EventBus {
    return &EventBus{
        observers: make([]Observer, 0),
    }
}

func (eb *EventBus) Attach(observer Observer) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    eb.observers = append(eb.observers, observer)
}

func (eb *EventBus) Detach(observer Observer) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()

    for i, obs := range eb.observers {
        if obs == observer {
            eb.observers = append(eb.observers[:i], eb.observers[i+1:]...)
            break
        }
    }
}

func (eb *EventBus) Notify(event Event) {
    eb.mutex.RLock()
    defer eb.mutex.RUnlock()

    for _, observer := range eb.observers {
        go observer.Update(event)
    }
}
```

This comprehensive guide covers the essential Golang patterns and idioms that are commonly tested in technical interviews. Each pattern includes practical examples and best practices for writing clean, efficient, and maintainable Go code.
