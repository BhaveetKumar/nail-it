# Advanced Coding Patterns

## Table of Contents
- [Introduction](#introduction/)
- [Design Patterns](#design-patterns/)
- [Concurrency Patterns](#concurrency-patterns/)
- [Performance Patterns](#performance-patterns/)
- [Error Handling Patterns](#error-handling-patterns/)
- [Testing Patterns](#testing-patterns/)

## Introduction

Advanced coding patterns demonstrate your understanding of sophisticated software engineering practices and your ability to write maintainable, scalable, and efficient code.

## Design Patterns

### Singleton Pattern with Thread Safety

```go
// Thread-safe Singleton
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

// Lazy initialization with mutex
var (
    instance2 *Singleton
    mu        sync.Mutex
)

func GetInstance2() *Singleton {
    if instance2 == nil {
        mu.Lock()
        defer mu.Unlock()
        if instance2 == nil {
            instance2 = &Singleton{data: "lazy initialized"}
        }
    }
    return instance2
}
```

### Observer Pattern

```go
// Observer Pattern
type Event struct {
    Type      string
    Data      interface{}
    Timestamp time.Time
}

type Observer interface {
    Update(event Event)
}

type Subject struct {
    observers []Observer
    mutex     sync.RWMutex
}

func (s *Subject) Attach(observer Observer) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    s.observers = append(s.observers, observer)
}

func (s *Subject) Detach(observer Observer) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    for i, obs := range s.observers {
        if obs == observer {
            s.observers = append(s.observers[:i], s.observers[i+1:]...)
            break
        }
    }
}

func (s *Subject) Notify(event Event) {
    s.mutex.RLock()
    observers := make([]Observer, len(s.observers))
    copy(observers, s.observers)
    s.mutex.RUnlock()
    
    for _, observer := range observers {
        go observer.Update(event)
    }
}

// Concrete Observer
type Logger struct {
    name string
}

func (l *Logger) Update(event Event) {
    fmt.Printf("[%s] Event: %s, Data: %v, Time: %v\n", 
        l.name, event.Type, event.Data, event.Timestamp)
}
```

### Factory Pattern with Registry

```go
// Factory Pattern with Registry
type Product interface {
    Name() string
    Process() error
}

type ConcreteProductA struct{}
func (p *ConcreteProductA) Name() string { return "ProductA" }
func (p *ConcreteProductA) Process() error { 
    fmt.Println("Processing Product A")
    return nil
}

type ConcreteProductB struct{}
func (p *ConcreteProductB) Name() string { return "ProductB" }
func (p *ConcreteProductB) Process() error { 
    fmt.Println("Processing Product B")
    return nil
}

type ProductFactory struct {
    creators map[string]func() Product
}

func NewProductFactory() *ProductFactory {
    factory := &ProductFactory{
        creators: make(map[string]func() Product),
    }
    
    // Register product creators
    factory.Register("A", func() Product { return &ConcreteProductA{} })
    factory.Register("B", func() Product { return &ConcreteProductB{} })
    
    return factory
}

func (f *ProductFactory) Register(productType string, creator func() Product) {
    f.creators[productType] = creator
}

func (f *ProductFactory) Create(productType string) (Product, error) {
    creator, exists := f.creators[productType]
    if !exists {
        return nil, fmt.Errorf("unknown product type: %s", productType)
    }
    return creator(), nil
}
```

### Builder Pattern

```go
// Builder Pattern
type Query struct {
    selectFields []string
    table        string
    whereClause  string
    orderBy      string
    limit        int
    offset       int
}

type QueryBuilder struct {
    query *Query
}

func NewQueryBuilder() *QueryBuilder {
    return &QueryBuilder{
        query: &Query{},
    }
}

func (qb *QueryBuilder) Select(fields ...string) *QueryBuilder {
    qb.query.selectFields = append(qb.query.selectFields, fields...)
    return qb
}

func (qb *QueryBuilder) From(table string) *QueryBuilder {
    qb.query.table = table
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    qb.query.whereClause = condition
    return qb
}

func (qb *QueryBuilder) OrderBy(field string) *QueryBuilder {
    qb.query.orderBy = field
    return qb
}

func (qb *QueryBuilder) Limit(limit int) *QueryBuilder {
    qb.query.limit = limit
    return qb
}

func (qb *QueryBuilder) Offset(offset int) *QueryBuilder {
    qb.query.offset = offset
    return qb
}

func (qb *QueryBuilder) Build() string {
    var query strings.Builder
    
    // SELECT
    query.WriteString("SELECT ")
    if len(qb.query.selectFields) == 0 {
        query.WriteString("*")
    } else {
        query.WriteString(strings.Join(qb.query.selectFields, ", "))
    }
    
    // FROM
    query.WriteString(" FROM ")
    query.WriteString(qb.query.table)
    
    // WHERE
    if qb.query.whereClause != "" {
        query.WriteString(" WHERE ")
        query.WriteString(qb.query.whereClause)
    }
    
    // ORDER BY
    if qb.query.orderBy != "" {
        query.WriteString(" ORDER BY ")
        query.WriteString(qb.query.orderBy)
    }
    
    // LIMIT
    if qb.query.limit > 0 {
        query.WriteString(fmt.Sprintf(" LIMIT %d", qb.query.limit))
    }
    
    // OFFSET
    if qb.query.offset > 0 {
        query.WriteString(fmt.Sprintf(" OFFSET %d", qb.query.offset))
    }
    
    return query.String()
}
```

## Concurrency Patterns

### Worker Pool Pattern

```go
// Worker Pool Pattern
type Job struct {
    ID       int
    Data     interface{}
    Result   chan interface{}
    Error    chan error
}

type WorkerPool struct {
    workers    int
    jobQueue   chan Job
    quit       chan bool
    wg         sync.WaitGroup
}

func NewWorkerPool(workers int, queueSize int) *WorkerPool {
    return &WorkerPool{
        workers:  workers,
        jobQueue: make(chan Job, queueSize),
        quit:     make(chan bool),
    }
}

func (wp *WorkerPool) Start() {
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker(i)
    }
}

func (wp *WorkerPool) worker(id int) {
    defer wp.wg.Done()
    
    for {
        select {
        case job := <-wp.jobQueue:
            wp.processJob(job)
        case <-wp.quit:
            return
        }
    }
}

func (wp *WorkerPool) processJob(job Job) {
    // Simulate work
    time.Sleep(time.Millisecond * 100)
    
    // Process the job
    result := fmt.Sprintf("Processed job %d by worker", job.ID)
    
    select {
    case job.Result <- result:
    default:
        job.Error <- fmt.Errorf("failed to send result for job %d", job.ID)
    }
}

func (wp *WorkerPool) Submit(job Job) {
    wp.jobQueue <- job
}

func (wp *WorkerPool) Stop() {
    close(wp.quit)
    wp.wg.Wait()
    close(wp.jobQueue)
}
```

### Pipeline Pattern

```go
// Pipeline Pattern
type Pipeline struct {
    stages []Stage
    input  chan interface{}
    output chan interface{}
}

type Stage interface {
    Process(input interface{}) (interface{}, error)
}

type StringToUpper struct{}
func (s *StringToUpper) Process(input interface{}) (interface{}, error) {
    str, ok := input.(string)
    if !ok {
        return nil, fmt.Errorf("expected string, got %T", input)
    }
    return strings.ToUpper(str), nil
}

type StringLength struct{}
func (s *StringLength) Process(input interface{}) (interface{}, error) {
    str, ok := input.(string)
    if !ok {
        return nil, fmt.Errorf("expected string, got %T", input)
    }
    return len(str), nil
}

func NewPipeline(inputChan chan interface{}, stages ...Stage) *Pipeline {
    return &Pipeline{
        stages: stages,
        input:  inputChan,
        output: make(chan interface{}, 100),
    }
}

func (p *Pipeline) Run() {
    defer close(p.output)
    
    for input := range p.input {
        result := input
        
        for _, stage := range p.stages {
            processed, err := stage.Process(result)
            if err != nil {
                fmt.Printf("Error in pipeline: %v\n", err)
                continue
            }
            result = processed
        }
        
        p.output <- result
    }
}

func (p *Pipeline) GetOutput() <-chan interface{} {
    return p.output
}
```

### Circuit Breaker Pattern

```go
// Circuit Breaker Pattern
type State int

const (
    Closed State = iota
    Open
    HalfOpen
)

type CircuitBreaker struct {
    maxFailures     int
    resetTimeout    time.Duration
    failureCount    int
    lastFailTime    time.Time
    state           State
    mutex           sync.RWMutex
}

func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        maxFailures:  maxFailures,
        resetTimeout: resetTimeout,
        state:        Closed,
    }
}

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    if cb.state == Open {
        if time.Since(cb.lastFailTime) >= cb.resetTimeout {
            cb.state = HalfOpen
        } else {
            return fmt.Errorf("circuit breaker is open")
        }
    }
    
    err := fn()
    
    if err != nil {
        cb.onFailure()
        return err
    }
    
    cb.onSuccess()
    return nil
}

func (cb *CircuitBreaker) onFailure() {
    cb.failureCount++
    cb.lastFailTime = time.Now()
    
    if cb.failureCount >= cb.maxFailures {
        cb.state = Open
    }
}

func (cb *CircuitBreaker) onSuccess() {
    cb.failureCount = 0
    cb.state = Closed
}

func (cb *CircuitBreaker) GetState() State {
    cb.mutex.RLock()
    defer cb.mutex.RUnlock()
    return cb.state
}
```

## Performance Patterns

### Object Pool Pattern

```go
// Object Pool Pattern
type Pool struct {
    objects chan interface{}
    factory func() interface{}
    mutex   sync.RWMutex
    size    int
}

func NewPool(factory func() interface{}, size int) *Pool {
    pool := &Pool{
        objects: make(chan interface{}, size),
        factory: factory,
        size:    size,
    }
    
    // Pre-populate pool
    for i := 0; i < size; i++ {
        pool.Put(factory())
    }
    
    return pool
}

func (p *Pool) Get() interface{} {
    select {
    case obj := <-p.objects:
        return obj
    default:
        return p.factory()
    }
}

func (p *Pool) Put(obj interface{}) {
    select {
    case p.objects <- obj:
    default:
        // Pool is full, discard the object
    }
}

func (p *Pool) Size() int {
    p.mutex.RLock()
    defer p.mutex.RUnlock()
    return len(p.objects)
}

// Usage example
type Buffer struct {
    data []byte
}

func NewBuffer() interface{} {
    return &Buffer{data: make([]byte, 0, 1024)}
}

func (b *Buffer) Reset() {
    b.data = b.data[:0]
}
```

### Memoization Pattern

```go
// Memoization Pattern
type Memoizer struct {
    cache map[string]interface{}
    mutex sync.RWMutex
}

func NewMemoizer() *Memoizer {
    return &Memoizer{
        cache: make(map[string]interface{}),
    }
}

func (m *Memoizer) Memoize(key string, fn func() (interface{}, error)) (interface{}, error) {
    m.mutex.RLock()
    if result, exists := m.cache[key]; exists {
        m.mutex.RUnlock()
        return result, nil
    }
    m.mutex.RUnlock()
    
    m.mutex.Lock()
    defer m.mutex.Unlock()
    
    // Double-check after acquiring write lock
    if result, exists := m.cache[key]; exists {
        return result, nil
    }
    
    result, err := fn()
    if err != nil {
        return nil, err
    }
    
    m.cache[key] = result
    return result, nil
}

func (m *Memoizer) Clear() {
    m.mutex.Lock()
    defer m.mutex.Unlock()
    m.cache = make(map[string]interface{})
}

func (m *Memoizer) Size() int {
    m.mutex.RLock()
    defer m.mutex.RUnlock()
    return len(m.cache)
}
```

### Lazy Loading Pattern

```go
// Lazy Loading Pattern
type LazyLoader struct {
    factory func() (interface{}, error)
    value   interface{}
    loaded  bool
    mutex   sync.Mutex
}

func NewLazyLoader(factory func() (interface{}, error)) *LazyLoader {
    return &LazyLoader{factory: factory}
}

func (ll *LazyLoader) Get() (interface{}, error) {
    if ll.loaded {
        return ll.value, nil
    }
    
    ll.mutex.Lock()
    defer ll.mutex.Unlock()
    
    if ll.loaded {
        return ll.value, nil
    }
    
    value, err := ll.factory()
    if err != nil {
        return nil, err
    }
    
    ll.value = value
    ll.loaded = true
    return value, nil
}

func (ll *LazyLoader) IsLoaded() bool {
    ll.mutex.Lock()
    defer ll.mutex.Unlock()
    return ll.loaded
}
```

## Error Handling Patterns

### Error Wrapping Pattern

```go
// Error Wrapping Pattern
type AppError struct {
    Code    string
    Message string
    Cause   error
    Context map[string]interface{}
}

func (e *AppError) Error() string {
    if e.Cause != nil {
        return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Cause)
    }
    return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

func (e *AppError) Unwrap() error {
    return e.Cause
}

func (e *AppError) WithContext(key string, value interface{}) *AppError {
    e.Context[key] = value
    return e
}

func NewError(code, message string, cause error) *AppError {
    return &AppError{
        Code:    code,
        Message: message,
        Cause:   cause,
        Context: make(map[string]interface{}),
    }
}

// Usage
func ProcessFile(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return NewError("FILE_OPEN_FAILED", "failed to open file", err).
            WithContext("filename", filename)
    }
    defer file.Close()
    
    // Process file...
    return nil
}
```

### Retry Pattern

```go
// Retry Pattern
type RetryConfig struct {
    MaxAttempts int
    InitialDelay time.Duration
    MaxDelay     time.Duration
    Multiplier   float64
    Jitter       bool
}

func NewRetryConfig(maxAttempts int, initialDelay time.Duration) *RetryConfig {
    return &RetryConfig{
        MaxAttempts:  maxAttempts,
        InitialDelay: initialDelay,
        MaxDelay:     time.Minute,
        Multiplier:   2.0,
        Jitter:       true,
    }
}

func (rc *RetryConfig) Execute(fn func() error) error {
    var lastErr error
    delay := rc.InitialDelay
    
    for attempt := 1; attempt <= rc.MaxAttempts; attempt++ {
        err := fn()
        if err == nil {
            return nil
        }
        
        lastErr = err
        
        if attempt == rc.MaxAttempts {
            break
        }
        
        // Calculate delay with exponential backoff
        time.Sleep(delay)
        delay = time.Duration(float64(delay) * rc.Multiplier)
        if delay > rc.MaxDelay {
            delay = rc.MaxDelay
        }
        
        // Add jitter
        if rc.Jitter {
            jitter := time.Duration(rand.Intn(int(delay.Milliseconds()))) * time.Millisecond
            delay += jitter
        }
    }
    
    return fmt.Errorf("operation failed after %d attempts: %w", rc.MaxAttempts, lastErr)
}
```

### Timeout Pattern

```go
// Timeout Pattern
func WithTimeout(duration time.Duration, fn func() error) error {
    ctx, cancel := context.WithTimeout(context.Background(), duration)
    defer cancel()
    
    done := make(chan error, 1)
    
    go func() {
        done <- fn()
    }()
    
    select {
    case err := <-done:
        return err
    case <-ctx.Done():
        return fmt.Errorf("operation timed out after %v", duration)
    }
}

// Usage
func ProcessRequest() error {
    return WithTimeout(5*time.Second, func() error {
        // Perform long-running operation
        time.Sleep(10 * time.Second)
        return nil
    })
}
```

## Testing Patterns

### Test Doubles

```go
// Test Doubles - Mock, Stub, Fake patterns
type UserService interface {
    GetUser(id int) (*User, error)
    CreateUser(user *User) error
}

type MockUserService struct {
    users map[int]*User
    err   error
}

func NewMockUserService() *MockUserService {
    return &MockUserService{
        users: make(map[int]*User),
    }
}

func (m *MockUserService) GetUser(id int) (*User, error) {
    if m.err != nil {
        return nil, m.err
    }
    return m.users[id], nil
}

func (m *MockUserService) CreateUser(user *User) error {
    if m.err != nil {
        return m.err
    }
    m.users[user.ID] = user
    return nil
}

func (m *MockUserService) SetError(err error) {
    m.err = err
}

// Test Table Pattern
func TestUserService(t *testing.T) {
    tests := []struct {
        name     string
        userID   int
        want     *User
        wantErr  bool
        setupMock func(*MockUserService)
    }{
        {
            name:    "existing user",
            userID:  1,
            want:    &User{ID: 1, Name: "John"},
            wantErr: false,
            setupMock: func(m *MockUserService) {
                m.users[1] = &User{ID: 1, Name: "John"}
            },
        },
        {
            name:    "non-existing user",
            userID:  2,
            want:    nil,
            wantErr: true,
            setupMock: func(m *MockUserService) {
                m.SetError(fmt.Errorf("user not found"))
            },
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            mock := NewMockUserService()
            tt.setupMock(mock)
            
            got, err := mock.GetUser(tt.userID)
            
            if tt.wantErr {
                if err == nil {
                    t.Errorf("expected error but got none")
                }
                return
            }
            
            if err != nil {
                t.Errorf("unexpected error: %v", err)
                return
            }
            
            if !reflect.DeepEqual(got, tt.want) {
                t.Errorf("got %v, want %v", got, tt.want)
            }
        })
    }
}
```

### Property-Based Testing

```go
// Property-Based Testing
func TestReverseProperty(t *testing.T) {
    tests := []struct {
        name     string
        property func([]int) bool
    }{
        {
            name: "reverse twice is identity",
            property: func(slice []int) bool {
                original := make([]int, len(slice))
                copy(original, slice)
                
                reversed := reverse(slice)
                doubleReversed := reverse(reversed)
                
                return reflect.DeepEqual(original, doubleReversed)
            },
        },
        {
            name: "reverse preserves length",
            property: func(slice []int) bool {
                reversed := reverse(slice)
                return len(slice) == len(reversed)
            },
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Generate random test cases
            for i := 0; i < 1000; i++ {
                slice := generateRandomSlice(rand.Intn(100))
                
                if !tt.property(slice) {
                    t.Errorf("property failed for slice: %v", slice)
                }
            }
        })
    }
}

func generateRandomSlice(size int) []int {
    slice := make([]int, size)
    for i := range slice {
        slice[i] = rand.Intn(1000)
    }
    return slice
}

func reverse(slice []int) []int {
    result := make([]int, len(slice))
    for i, v := range slice {
        result[len(slice)-1-i] = v
    }
    return result
}
```

### Contract Testing

```go
// Contract Testing
type ServiceContract interface {
    Process(input string) (string, error)
    Health() error
}

func TestServiceContract(t *testing.T) {
    // Test that all implementations satisfy the contract
    implementations := []struct {
        name string
        impl ServiceContract
    }{
        {"ServiceA", &ServiceA{}},
        {"ServiceB", &ServiceB{}},
        {"ServiceC", &ServiceC{}},
    }
    
    for _, impl := range implementations {
        t.Run(impl.name, func(t *testing.T) {
            testContract(t, impl.impl)
        })
    }
}

func testContract(t *testing.T, service ServiceContract) {
    // Test Process method
    result, err := service.Process("test")
    if err != nil {
        t.Errorf("Process failed: %v", err)
    }
    if result == "" {
        t.Error("Process returned empty result")
    }
    
    // Test Health method
    if err := service.Health(); err != nil {
        t.Errorf("Health check failed: %v", err)
    }
}
```

## Conclusion

Advanced coding patterns demonstrate:

1. **Design Skills**: Understanding of design patterns and their applications
2. **Concurrency Mastery**: Handling concurrent operations safely and efficiently
3. **Performance Optimization**: Writing efficient, scalable code
4. **Error Handling**: Robust error handling and recovery mechanisms
5. **Testing Expertise**: Comprehensive testing strategies and techniques
6. **Code Quality**: Writing maintainable, readable, and well-structured code
7. **Problem Solving**: Applying appropriate patterns to solve real-world problems

Mastering these patterns demonstrates your readiness for senior engineering roles and complex software development challenges.

## Additional Resources

- [Design Patterns](https://www.designpatterns.com/)
- [Concurrency Patterns](https://www.concurrencypatterns.com/)
- [Performance Patterns](https://www.performancepatterns.com/)
- [Error Handling](https://www.errorhandling.com/)
- [Testing Patterns](https://www.testingpatterns.com/)
- [Code Quality](https://www.codequality.com/)
- [Software Architecture](https://www.softwarearchitecture.com/)
