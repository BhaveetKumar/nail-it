# Advanced Testing Comprehensive

Comprehensive guide to advanced testing strategies for senior backend engineers.

## 🎯 Advanced Testing Strategies

### Property-Based Testing
```go
// Advanced Property-Based Testing with QuickCheck
package testing

import (
    "fmt"
    "math/rand"
    "testing"
    "time"
)

// Property: Reverse of reverse should equal original
func TestReverseProperty(t *testing.T) {
    quickCheck := NewQuickCheck(1000) // 1000 test cases
    
    quickCheck.Property("reverse of reverse equals original", func(slice []int) bool {
        reversed := reverse(reverse(slice))
        return equal(slice, reversed)
    })
}

// Property: Sorting is idempotent
func TestSortProperty(t *testing.T) {
    quickCheck := NewQuickCheck(1000)
    
    quickCheck.Property("sorting is idempotent", func(slice []int) bool {
        sorted1 := sort(slice)
        sorted2 := sort(sorted1)
        return equal(sorted1, sorted2)
    })
}

// Property: Binary search finds correct element
func TestBinarySearchProperty(t *testing.T) {
    quickCheck := NewQuickCheck(1000)
    
    quickCheck.Property("binary search finds correct element", func(slice []int, target int) bool {
        if len(slice) == 0 {
            return true // Empty slice case
        }
        
        sorted := sort(slice)
        index := binarySearch(sorted, target)
        
        if index == -1 {
            // Target not found, verify it's not in slice
            return !contains(sorted, target)
        } else {
            // Target found, verify it's at correct position
            return sorted[index] == target
        }
    })
}

// Property: Hash table operations
func TestHashTableProperty(t *testing.T) {
    quickCheck := NewQuickCheck(1000)
    
    quickCheck.Property("hash table operations", func(operations []Operation) bool {
        ht := NewHashTable()
        
        for _, op := range operations {
            switch op.Type {
            case "put":
                ht.Put(op.Key, op.Value)
            case "get":
                value, exists := ht.Get(op.Key)
                if op.ExpectedExists {
                    return exists && value == op.ExpectedValue
                } else {
                    return !exists
                }
            case "delete":
                ht.Delete(op.Key)
            }
        }
        
        return true
    })
}

type Operation struct {
    Type           string
    Key            string
    Value          int
    ExpectedExists bool
    ExpectedValue  int
}

// QuickCheck Implementation
type QuickCheck struct {
    numTests int
    rng      *rand.Rand
}

func NewQuickCheck(numTests int) *QuickCheck {
    return &QuickCheck{
        numTests: numTests,
        rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
    }
}

func (qc *QuickCheck) Property(name string, property func(interface{}) bool) {
    for i := 0; i < qc.numTests; i++ {
        // Generate random input
        input := qc.generateInput()
        
        // Test property
        if !property(input) {
            panic(fmt.Sprintf("Property %s failed with input: %v", name, input))
        }
    }
}

func (qc *QuickCheck) generateInput() interface{} {
    // Simplified input generation
    // In practice, you'd use a more sophisticated generator
    return qc.rng.Intn(1000)
}

// Helper functions
func reverse(slice []int) []int {
    result := make([]int, len(slice))
    for i, v := range slice {
        result[len(slice)-1-i] = v
    }
    return result
}

func equal(a, b []int) bool {
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

func sort(slice []int) []int {
    result := make([]int, len(slice))
    copy(result, slice)
    
    // Simple bubble sort for demonstration
    for i := 0; i < len(result)-1; i++ {
        for j := 0; j < len(result)-i-1; j++ {
            if result[j] > result[j+1] {
                result[j], result[j+1] = result[j+1], result[j]
            }
        }
    }
    
    return result
}

func binarySearch(slice []int, target int) int {
    left, right := 0, len(slice)-1
    
    for left <= right {
        mid := (left + right) / 2
        if slice[mid] == target {
            return mid
        } else if slice[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}

func contains(slice []int, target int) bool {
    for _, v := range slice {
        if v == target {
            return true
        }
    }
    return false
}
```

### Chaos Engineering
```go
// Advanced Chaos Engineering Implementation
package testing

import (
    "context"
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type ChaosEngine struct {
    experiments map[string]*Experiment
    mutex       sync.RWMutex
    running     bool
    stopChan    chan struct{}
}

type Experiment struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Probability float64                `json:"probability"`
    Duration    time.Duration          `json:"duration"`
    Actions     []ChaosAction          `json:"actions"`
    Conditions  []ChaosCondition       `json:"conditions"`
    Metrics     map[string]interface{} `json:"metrics"`
    Status      ExperimentStatus       `json:"status"`
}

type ChaosAction interface {
    Execute(ctx context.Context) error
    Revert(ctx context.Context) error
    Name() string
}

type ChaosCondition interface {
    Check(ctx context.Context) bool
    Name() string
}

type ExperimentStatus int

const (
    StatusStopped ExperimentStatus = iota
    StatusRunning
    StatusPaused
    StatusFailed
)

// Network Latency Action
type NetworkLatencyAction struct {
    target    string
    latency   time.Duration
    jitter    time.Duration
    duration  time.Duration
    original  time.Duration
}

func NewNetworkLatencyAction(target string, latency, jitter, duration time.Duration) *NetworkLatencyAction {
    return &NetworkLatencyAction{
        target:   target,
        latency:  latency,
        jitter:   jitter,
        duration: duration,
    }
}

func (nla *NetworkLatencyAction) Execute(ctx context.Context) error {
    // Simulate network latency injection
    // In practice, you'd use tools like tc (traffic control) on Linux
    fmt.Printf("Injecting %v latency to %s\n", nla.latency, nla.target)
    
    // Store original latency for revert
    nla.original = 0 // Placeholder
    
    // Simulate the action
    time.Sleep(nla.duration)
    
    return nil
}

func (nla *NetworkLatencyAction) Revert(ctx context.Context) error {
    // Revert network latency
    fmt.Printf("Reverting latency for %s\n", nla.target)
    return nil
}

func (nla *NetworkLatencyAction) Name() string {
    return "network_latency"
}

// CPU Stress Action
type CPUStressAction struct {
    duration time.Duration
    cores    int
}

func NewCPUStressAction(duration time.Duration, cores int) *CPUStressAction {
    return &CPUStressAction{
        duration: duration,
        cores:    cores,
    }
}

func (csa *CPUStressAction) Execute(ctx context.Context) error {
    fmt.Printf("Stressing CPU for %v on %d cores\n", csa.duration, csa.cores)
    
    // Simulate CPU stress
    done := make(chan struct{})
    for i := 0; i < csa.cores; i++ {
        go func() {
            for {
                select {
                case <-done:
                    return
                default:
                    // CPU intensive operation
                    _ = rand.Intn(1000000)
                }
            }
        }()
    }
    
    time.Sleep(csa.duration)
    close(done)
    
    return nil
}

func (csa *CPUStressAction) Revert(ctx context.Context) error {
    fmt.Println("Reverting CPU stress")
    return nil
}

func (csa *CPUStressAction) Name() string {
    return "cpu_stress"
}

// Memory Leak Action
type MemoryLeakAction struct {
    duration time.Duration
    rate     int // MB per second
}

func NewMemoryLeakAction(duration time.Duration, rate int) *MemoryLeakAction {
    return &MemoryLeakAction{
        duration: duration,
        rate:     rate,
    }
}

func (mla *MemoryLeakAction) Execute(ctx context.Context) error {
    fmt.Printf("Simulating memory leak for %v at %d MB/s\n", mla.duration, mla.rate)
    
    // Simulate memory leak
    var memory [][]byte
    ticker := time.NewTicker(time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
            // Allocate memory
            chunk := make([]byte, mla.rate*1024*1024)
            memory = append(memory, chunk)
        case <-time.After(mla.duration):
            return nil
        }
    }
}

func (mla *MemoryLeakAction) Revert(ctx context.Context) error {
    fmt.Println("Reverting memory leak")
    return nil
}

func (mla *MemoryLeakAction) Name() string {
    return "memory_leak"
}

// Service Failure Action
type ServiceFailureAction struct {
    serviceName string
    duration    time.Duration
    failureRate float64
}

func NewServiceFailureAction(serviceName string, duration time.Duration, failureRate float64) *ServiceFailureAction {
    return &ServiceFailureAction{
        serviceName: serviceName,
        duration:    duration,
        failureRate: failureRate,
    }
}

func (sfa *ServiceFailureAction) Execute(ctx context.Context) error {
    fmt.Printf("Simulating %s failure for %v with %f failure rate\n", 
        sfa.serviceName, sfa.duration, sfa.failureRate)
    
    // Simulate service failure
    ticker := time.NewTicker(100 * time.Millisecond)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
            if rand.Float64() < sfa.failureRate {
                // Simulate failure
                fmt.Printf("Service %s failed\n", sfa.serviceName)
            }
        case <-time.After(sfa.duration):
            return nil
        }
    }
}

func (sfa *ServiceFailureAction) Revert(ctx context.Context) error {
    fmt.Printf("Reverting service failure for %s\n", sfa.serviceName)
    return nil
}

func (sfa *ServiceFailureAction) Name() string {
    return "service_failure"
}

// Chaos Engine Implementation
func NewChaosEngine() *ChaosEngine {
    return &ChaosEngine{
        experiments: make(map[string]*Experiment),
        stopChan:    make(chan struct{}),
    }
}

func (ce *ChaosEngine) AddExperiment(experiment *Experiment) {
    ce.mutex.Lock()
    defer ce.mutex.Unlock()
    
    ce.experiments[experiment.Name] = experiment
}

func (ce *ChaosEngine) StartExperiment(name string) error {
    ce.mutex.Lock()
    defer ce.mutex.Unlock()
    
    experiment, exists := ce.experiments[name]
    if !exists {
        return fmt.Errorf("experiment %s not found", name)
    }
    
    if experiment.Status == StatusRunning {
        return fmt.Errorf("experiment %s is already running", name)
    }
    
    experiment.Status = StatusRunning
    go ce.runExperiment(experiment)
    
    return nil
}

func (ce *ChaosEngine) StopExperiment(name string) error {
    ce.mutex.Lock()
    defer ce.mutex.Unlock()
    
    experiment, exists := ce.experiments[name]
    if !exists {
        return fmt.Errorf("experiment %s not found", name)
    }
    
    if experiment.Status != StatusRunning {
        return fmt.Errorf("experiment %s is not running", name)
    }
    
    experiment.Status = StatusStopped
    return nil
}

func (ce *ChaosEngine) runExperiment(experiment *Experiment) {
    ctx, cancel := context.WithTimeout(context.Background(), experiment.Duration)
    defer cancel()
    
    // Check conditions
    for _, condition := range experiment.Conditions {
        if !condition.Check(ctx) {
            fmt.Printf("Condition %s failed for experiment %s\n", condition.Name(), experiment.Name)
            experiment.Status = StatusFailed
            return
        }
    }
    
    // Execute actions
    for _, action := range experiment.Actions {
        if rand.Float64() < experiment.Probability {
            if err := action.Execute(ctx); err != nil {
                fmt.Printf("Action %s failed: %v\n", action.Name(), err)
                experiment.Status = StatusFailed
                return
            }
        }
    }
    
    // Wait for duration
    select {
    case <-ctx.Done():
        // Timeout reached
    case <-ce.stopChan:
        // Stopped by user
    }
    
    // Revert actions
    for _, action := range experiment.Actions {
        if err := action.Revert(ctx); err != nil {
            fmt.Printf("Failed to revert action %s: %v\n", action.Name(), err)
        }
    }
    
    experiment.Status = StatusStopped
}

func (ce *ChaosEngine) GetExperimentStatus(name string) (ExperimentStatus, error) {
    ce.mutex.RLock()
    defer ce.mutex.RUnlock()
    
    experiment, exists := ce.experiments[name]
    if !exists {
        return StatusStopped, fmt.Errorf("experiment %s not found", name)
    }
    
    return experiment.Status, nil
}

func (ce *ChaosEngine) ListExperiments() map[string]*Experiment {
    ce.mutex.RLock()
    defer ce.mutex.RUnlock()
    
    result := make(map[string]*Experiment)
    for name, experiment := range ce.experiments {
        result[name] = experiment
    }
    
    return result
}
```

### Advanced Test Doubles
```go
// Advanced Test Doubles Implementation
package testing

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Mock with Behavior Verification
type MockUserService struct {
    calls       []Call
    expectations []Expectation
    mutex       sync.RWMutex
}

type Call struct {
    Method string
    Args   []interface{}
    Time   time.Time
}

type Expectation struct {
    Method string
    Args   []interface{}
    Return []interface{}
    Error  error
    Times  int
    Actual int
}

func NewMockUserService() *MockUserService {
    return &MockUserService{
        calls:       make([]Call, 0),
        expectations: make([]Expectation, 0),
    }
}

func (mus *MockUserService) Expect(method string, args []interface{}, returnVals []interface{}, err error) *MockUserService {
    mus.mutex.Lock()
    defer mus.mutex.Unlock()
    
    expectation := Expectation{
        Method: method,
        Args:   args,
        Return: returnVals,
        Error:  err,
        Times:  1,
        Actual: 0,
    }
    
    mus.expectations = append(mus.expectations, expectation)
    return mus
}

func (mus *MockUserService) ExpectTimes(method string, args []interface{}, returnVals []interface{}, err error, times int) *MockUserService {
    mus.mutex.Lock()
    defer mus.mutex.Unlock()
    
    expectation := Expectation{
        Method: method,
        Args:   args,
        Return: returnVals,
        Error:  err,
        Times:  times,
        Actual: 0,
    }
    
    mus.expectations = append(mus.expectations, expectation)
    return mus
}

func (mus *MockUserService) GetUser(ctx context.Context, id string) (*User, error) {
    mus.mutex.Lock()
    defer mus.mutex.Unlock()
    
    // Record call
    call := Call{
        Method: "GetUser",
        Args:   []interface{}{id},
        Time:   time.Now(),
    }
    mus.calls = append(mus.calls, call)
    
    // Find matching expectation
    for i, exp := range mus.expectations {
        if exp.Method == "GetUser" && mus.argsMatch(exp.Args, []interface{}{id}) {
            mus.expectations[i].Actual++
            
            if exp.Return != nil && len(exp.Return) > 0 {
                if user, ok := exp.Return[0].(*User); ok {
                    return user, exp.Error
                }
            }
            return nil, exp.Error
        }
    }
    
    return nil, fmt.Errorf("unexpected call to GetUser with id: %s", id)
}

func (mus *MockUserService) CreateUser(ctx context.Context, user *User) error {
    mus.mutex.Lock()
    defer mus.mutex.Unlock()
    
    // Record call
    call := Call{
        Method: "CreateUser",
        Args:   []interface{}{user},
        Time:   time.Now(),
    }
    mus.calls = append(mus.calls, call)
    
    // Find matching expectation
    for i, exp := range mus.expectations {
        if exp.Method == "CreateUser" && mus.argsMatch(exp.Args, []interface{}{user}) {
            mus.expectations[i].Actual++
            return exp.Error
        }
    }
    
    return fmt.Errorf("unexpected call to CreateUser with user: %v", user)
}

func (mus *MockUserService) argsMatch(expected, actual []interface{}) bool {
    if len(expected) != len(actual) {
        return false
    }
    
    for i, exp := range expected {
        if !mus.deepEqual(exp, actual[i]) {
            return false
        }
    }
    
    return true
}

func (mus *MockUserService) deepEqual(a, b interface{}) bool {
    // Simplified deep equality check
    // In practice, you'd use a more sophisticated comparison
    return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
}

func (mus *MockUserService) Verify() error {
    mus.mutex.RLock()
    defer mus.mutex.RUnlock()
    
    for _, exp := range mus.expectations {
        if exp.Actual != exp.Times {
            return fmt.Errorf("expected %s to be called %d times, but was called %d times", 
                exp.Method, exp.Times, exp.Actual)
        }
    }
    
    return nil
}

func (mus *MockUserService) GetCalls() []Call {
    mus.mutex.RLock()
    defer mus.mutex.RUnlock()
    
    return append([]Call(nil), mus.calls...)
}

// Spy Implementation
type SpyEmailService struct {
    sentEmails []Email
    mutex      sync.RWMutex
}

type Email struct {
    To      string
    Subject string
    Body    string
    Time    time.Time
}

func NewSpyEmailService() *SpyEmailService {
    return &SpyEmailService{
        sentEmails: make([]Email, 0),
    }
}

func (ses *SpyEmailService) SendEmail(to, subject, body string) error {
    ses.mutex.Lock()
    defer ses.mutex.Unlock()
    
    email := Email{
        To:      to,
        Subject: subject,
        Body:    body,
        Time:    time.Now(),
    }
    
    ses.sentEmails = append(ses.sentEmails, email)
    return nil
}

func (ses *SpyEmailService) GetSentEmails() []Email {
    ses.mutex.RLock()
    defer ses.mutex.RUnlock()
    
    return append([]Email(nil), ses.sentEmails...)
}

func (ses *SpyEmailService) GetEmailsTo(to string) []Email {
    ses.mutex.RLock()
    defer ses.mutex.RUnlock()
    
    var result []Email
    for _, email := range ses.sentEmails {
        if email.To == to {
            result = append(result, email)
        }
    }
    
    return result
}

func (ses *SpyEmailService) GetEmailsWithSubject(subject string) []Email {
    ses.mutex.RLock()
    defer ses.mutex.RUnlock()
    
    var result []Email
    for _, email := range ses.sentEmails {
        if email.Subject == subject {
            result = append(result, email)
        }
    }
    
    return result
}

func (ses *SpyEmailService) Clear() {
    ses.mutex.Lock()
    defer ses.mutex.Unlock()
    
    ses.sentEmails = ses.sentEmails[:0]
}

// Fake Implementation
type FakeUserRepository struct {
    users map[string]*User
    mutex sync.RWMutex
}

func NewFakeUserRepository() *FakeUserRepository {
    return &FakeUserRepository{
        users: make(map[string]*User),
    }
}

func (fur *FakeUserRepository) GetUser(ctx context.Context, id string) (*User, error) {
    fur.mutex.RLock()
    defer fur.mutex.RUnlock()
    
    user, exists := fur.users[id]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }
    
    return user, nil
}

func (fur *FakeUserRepository) CreateUser(ctx context.Context, user *User) error {
    fur.mutex.Lock()
    defer fur.mutex.Unlock()
    
    if _, exists := fur.users[user.ID]; exists {
        return fmt.Errorf("user already exists")
    }
    
    fur.users[user.ID] = user
    return nil
}

func (fur *FakeUserRepository) UpdateUser(ctx context.Context, user *User) error {
    fur.mutex.Lock()
    defer fur.mutex.Unlock()
    
    if _, exists := fur.users[user.ID]; !exists {
        return fmt.Errorf("user not found")
    }
    
    fur.users[user.ID] = user
    return nil
}

func (fur *FakeUserRepository) DeleteUser(ctx context.Context, id string) error {
    fur.mutex.Lock()
    defer fur.mutex.Unlock()
    
    if _, exists := fur.users[id]; !exists {
        return fmt.Errorf("user not found")
    }
    
    delete(fur.users, id)
    return nil
}

func (fur *FakeUserRepository) ListUsers(ctx context.Context) ([]*User, error) {
    fur.mutex.RLock()
    defer fur.mutex.RUnlock()
    
    var users []*User
    for _, user := range fur.users {
        users = append(users, user)
    }
    
    return users, nil
}

func (fur *FakeUserRepository) GetUserCount() int {
    fur.mutex.RLock()
    defer fur.mutex.RUnlock()
    
    return len(fur.users)
}

func (fur *FakeUserRepository) Clear() {
    fur.mutex.Lock()
    defer fur.mutex.Unlock()
    
    fur.users = make(map[string]*User)
}

// Stub Implementation
type StubPaymentService struct {
    shouldFail bool
    delay      time.Duration
}

func NewStubPaymentService(shouldFail bool, delay time.Duration) *StubPaymentService {
    return &StubPaymentService{
        shouldFail: shouldFail,
        delay:      delay,
    }
}

func (sps *StubPaymentService) ProcessPayment(ctx context.Context, amount float64, currency string) error {
    if sps.delay > 0 {
        time.Sleep(sps.delay)
    }
    
    if sps.shouldFail {
        return fmt.Errorf("payment processing failed")
    }
    
    return nil
}

func (sps *StubPaymentService) RefundPayment(ctx context.Context, transactionID string) error {
    if sps.delay > 0 {
        time.Sleep(sps.delay)
    }
    
    if sps.shouldFail {
        return fmt.Errorf("refund processing failed")
    }
    
    return nil
}
```

## 🎯 Best Practices

### Testing Principles
1. **Test Pyramid**: Unit tests > Integration tests > E2E tests
2. **AAA Pattern**: Arrange, Act, Assert
3. **Single Responsibility**: One test, one behavior
4. **Independence**: Tests should not depend on each other
5. **Repeatability**: Tests should produce the same results every time

### Test Design
1. **Descriptive Names**: Use clear, descriptive test names
2. **Given-When-Then**: Structure tests with clear sections
3. **Edge Cases**: Test boundary conditions and edge cases
4. **Error Paths**: Test error conditions and failure scenarios
5. **Performance**: Include performance tests where appropriate

### Test Maintenance
1. **Refactoring**: Refactor tests when code changes
2. **Documentation**: Document complex test scenarios
3. **Review**: Review tests during code review
4. **Metrics**: Track test coverage and quality metrics
5. **Automation**: Automate test execution and reporting

---

**Last Updated**: December 2024  
**Category**: Advanced Testing Comprehensive  
**Complexity**: Expert Level
