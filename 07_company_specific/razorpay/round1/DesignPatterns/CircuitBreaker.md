---
# Auto-generated front matter
Title: Circuitbreaker
LastUpdated: 2025-11-06T20:45:58.523001
Tags: []
Status: draft
---

# Circuit Breaker Pattern

## Pattern Name & Intent

**Circuit Breaker** is a design pattern that prevents cascading failures in distributed systems by monitoring the success/failure rate of operations and "opening the circuit" when the failure rate exceeds a threshold, thus preventing further calls to the failing service.

**Key Intent:**

- Prevent cascading failures in distributed systems
- Provide fast failure detection and recovery
- Improve system resilience and fault tolerance
- Reduce latency during service failures
- Enable graceful degradation of functionality

## When to Use

**Use Circuit Breaker when:**

1. **External Service Dependencies**: Your service depends on external APIs or services
2. **Network Operations**: Operations involve network calls that may fail or timeout
3. **Resource Protection**: Need to protect limited resources from overload
4. **Fail-Fast Requirements**: Want to fail fast instead of waiting for timeouts
5. **Cascading Failure Prevention**: Need to prevent failures from propagating
6. **Graceful Degradation**: Want to provide fallback mechanisms

**Don't use when:**

- Operations are purely local (no network calls)
- Failure rates are consistently low
- System can handle all failures gracefully
- Overhead of circuit breaker exceeds benefits

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Gateway Integration

```go
// Protecting payment gateway calls
type PaymentGatewayCircuitBreaker struct {
    razorpayCircuit *CircuitBreaker
    stripeCircuit   *CircuitBreaker
    fallbackGateway string
}

func (p *PaymentGatewayCircuitBreaker) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    // Try primary gateway with circuit breaker
    if req.Gateway == "razorpay" {
        return p.razorpayCircuit.Execute(func() (interface{}, error) {
            return p.callRazorpay(req)
        })
    }

    // Try Stripe with circuit breaker
    return p.stripeCircuit.Execute(func() (interface{}, error) {
        return p.callStripe(req)
    })
}

func (p *PaymentGatewayCircuitBreaker) ProcessPaymentWithFallback(req PaymentRequest) (*PaymentResponse, error) {
    resp, err := p.ProcessPayment(req)
    if err != nil && p.isCircuitOpen(req.Gateway) {
        // Use fallback gateway
        req.Gateway = p.fallbackGateway
        return p.ProcessPayment(req)
    }
    return resp, err
}
```

### 2. Banking Service Mesh

```go
// Circuit breakers for microservices
type BankingServiceMesh struct {
    accountServiceCB    *CircuitBreaker
    transactionServiceCB *CircuitBreaker
    notificationServiceCB *CircuitBreaker
    auditServiceCB      *CircuitBreaker
}

func (b *BankingServiceMesh) TransferFunds(req TransferRequest) error {
    // Check account balance with circuit breaker
    balance, err := b.accountServiceCB.Execute(func() (interface{}, error) {
        return b.accountService.GetBalance(req.FromAccount)
    })
    if err != nil {
        return fmt.Errorf("failed to check balance: %w", err)
    }

    if balance.(decimal.Decimal).LessThan(req.Amount) {
        return fmt.Errorf("insufficient funds")
    }

    // Process transfer with circuit breaker
    _, err = b.transactionServiceCB.Execute(func() (interface{}, error) {
        return nil, b.transactionService.Transfer(req)
    })
    if err != nil {
        return fmt.Errorf("transfer failed: %w", err)
    }

    // Send notification (non-critical, can fail)
    b.notificationServiceCB.Execute(func() (interface{}, error) {
        return nil, b.notificationService.SendTransferNotification(req)
    })

    // Audit logging (critical for compliance)
    return b.auditServiceCB.Execute(func() (interface{}, error) {
        return nil, b.auditService.LogTransaction(req)
    }).(error)
}
```

### 3. Trading Platform Circuit Protection

```go
// Circuit breakers for trading operations
type TradingPlatform struct {
    marketDataCB    *CircuitBreaker
    orderExecutionCB *CircuitBreaker
    riskManagementCB *CircuitBreaker
    settlementCB    *CircuitBreaker
}

func (t *TradingPlatform) PlaceOrder(order *Order) (*OrderResponse, error) {
    // Get market data with circuit breaker
    marketData, err := t.marketDataCB.Execute(func() (interface{}, error) {
        return t.marketDataService.GetQuote(order.Symbol)
    })
    if err != nil {
        // Use stale market data if circuit is open
        marketData = t.getCachedMarketData(order.Symbol)
    }

    // Risk check with circuit breaker
    _, err = t.riskManagementCB.Execute(func() (interface{}, error) {
        return nil, t.riskService.ValidateOrder(order, marketData)
    })
    if err != nil {
        return nil, fmt.Errorf("risk validation failed: %w", err)
    }

    // Execute order with circuit breaker
    return t.orderExecutionCB.Execute(func() (interface{}, error) {
        return t.orderService.ExecuteOrder(order)
    }).(*OrderResponse), nil
}
```

### 4. Fraud Detection System

```go
// Circuit breaker for fraud detection services
type FraudDetectionSystem struct {
    mlModelCB       *CircuitBreaker
    ruleEngineCB    *CircuitBreaker
    externalDataCB  *CircuitBreaker
    blacklistCB     *CircuitBreaker
}

func (f *FraudDetectionSystem) AnalyzeTransaction(tx *Transaction) (*FraudScore, error) {
    var scores []float64

    // ML Model analysis (high latency)
    if mlScore, err := f.mlModelCB.Execute(func() (interface{}, error) {
        return f.mlModel.Predict(tx)
    }); err == nil {
        scores = append(scores, mlScore.(float64))
    }

    // Rule engine analysis (fast)
    if ruleScore, err := f.ruleEngineCB.Execute(func() (interface{}, error) {
        return f.ruleEngine.Evaluate(tx)
    }); err == nil {
        scores = append(scores, ruleScore.(float64))
    }

    // External data check (may be slow)
    if extScore, err := f.externalDataCB.Execute(func() (interface{}, error) {
        return f.externalData.CheckReputation(tx.UserID)
    }); err == nil {
        scores = append(scores, extScore.(float64))
    }

    // Blacklist check (critical, must succeed)
    isBlacklisted, err := f.blacklistCB.Execute(func() (interface{}, error) {
        return f.blacklistService.IsBlacklisted(tx.UserID)
    })
    if err != nil {
        return nil, fmt.Errorf("blacklist check failed: %w", err)
    }

    return &FraudScore{
        Score:        f.calculateAggregateScore(scores),
        IsBlacklisted: isBlacklisted.(bool),
        Timestamp:    time.Now(),
    }, nil
}
```

## Go Implementation

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"
    "errors"
    "math/rand"
)

// Circuit States
type CircuitState int

const (
    CircuitStateClosed CircuitState = iota
    CircuitStateOpen
    CircuitStateHalfOpen
)

func (s CircuitState) String() string {
    switch s {
    case CircuitStateClosed:
        return "CLOSED"
    case CircuitStateOpen:
        return "OPEN"
    case CircuitStateHalfOpen:
        return "HALF_OPEN"
    default:
        return "UNKNOWN"
    }
}

// Circuit Breaker Configuration
type CircuitBreakerConfig struct {
    Name                    string        `json:"name"`
    FailureThreshold        int           `json:"failure_threshold"`        // Number of failures to open circuit
    SuccessThreshold        int           `json:"success_threshold"`        // Number of successes to close circuit in half-open
    Timeout                 time.Duration `json:"timeout"`                  // How long to wait before trying half-open
    MaxRequests             int           `json:"max_requests"`             // Max requests allowed in half-open state
    RequestVolumeThreshold  int           `json:"request_volume_threshold"` // Minimum requests before considering failure rate
    SleepWindow             time.Duration `json:"sleep_window"`             // Time window for failure rate calculation
    ErrorThreshold          float64       `json:"error_threshold"`          // Error rate threshold (0.0 - 1.0)
    SlowCallThreshold       time.Duration `json:"slow_call_threshold"`      // Threshold for considering calls slow
    SlowCallRateThreshold   float64       `json:"slow_call_rate_threshold"` // Slow call rate threshold
}

// Default configuration
func DefaultCircuitBreakerConfig(name string) CircuitBreakerConfig {
    return CircuitBreakerConfig{
        Name:                   name,
        FailureThreshold:       5,
        SuccessThreshold:       3,
        Timeout:                time.Second * 60,
        MaxRequests:            10,
        RequestVolumeThreshold: 10,
        SleepWindow:            time.Second * 10,
        ErrorThreshold:         0.5,
        SlowCallThreshold:      time.Second * 5,
        SlowCallRateThreshold:  0.5,
    }
}

// Circuit Breaker Statistics
type CircuitBreakerStats struct {
    State                CircuitState  `json:"state"`
    TotalRequests        int64         `json:"total_requests"`
    SuccessfulRequests   int64         `json:"successful_requests"`
    FailedRequests       int64         `json:"failed_requests"`
    RejectedRequests     int64         `json:"rejected_requests"`
    SlowRequests         int64         `json:"slow_requests"`
    AverageResponseTime  time.Duration `json:"average_response_time"`
    LastFailureTime      *time.Time    `json:"last_failure_time,omitempty"`
    LastSuccessTime      *time.Time    `json:"last_success_time,omitempty"`
    StateTransitionTime  time.Time     `json:"state_transition_time"`
    FailureRate          float64       `json:"failure_rate"`
    SlowCallRate         float64       `json:"slow_call_rate"`
    HalfOpenRequests     int           `json:"half_open_requests"`
    HalfOpenSuccesses    int           `json:"half_open_successes"`
}

// Call Result
type CallResult struct {
    Success      bool
    Duration     time.Duration
    Error        error
    SlowCall     bool
    Timestamp    time.Time
}

// Circuit Breaker
type CircuitBreaker struct {
    config       CircuitBreakerConfig
    state        CircuitState
    stats        CircuitBreakerStats
    recentCalls  []CallResult
    mu           sync.RWMutex
    stateChanged chan CircuitState

    // Callbacks
    onStateChange func(from, to CircuitState)
    onCallResult  func(result CallResult)
}

// Constructor
func NewCircuitBreaker(config CircuitBreakerConfig) *CircuitBreaker {
    cb := &CircuitBreaker{
        config:       config,
        state:        CircuitStateClosed,
        recentCalls:  make([]CallResult, 0),
        stateChanged: make(chan CircuitState, 10),
        stats: CircuitBreakerStats{
            State:               CircuitStateClosed,
            StateTransitionTime: time.Now(),
        },
    }

    return cb
}

// Set callbacks
func (cb *CircuitBreaker) OnStateChange(callback func(from, to CircuitState)) {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    cb.onStateChange = callback
}

func (cb *CircuitBreaker) OnCallResult(callback func(result CallResult)) {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    cb.onCallResult = callback
}

// Execute function with circuit breaker protection
func (cb *CircuitBreaker) Execute(fn func() (interface{}, error)) (interface{}, error) {
    return cb.ExecuteWithContext(context.Background(), fn)
}

func (cb *CircuitBreaker) ExecuteWithContext(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
    // Check if call should be allowed
    if !cb.allowRequest() {
        cb.recordRejection()
        return nil, fmt.Errorf("circuit breaker is %s", cb.state)
    }

    start := time.Now()

    // Execute the function
    result, err := cb.executeCall(ctx, fn)

    duration := time.Since(start)

    // Record the call result
    callResult := CallResult{
        Success:   err == nil,
        Duration:  duration,
        Error:     err,
        SlowCall:  duration > cb.config.SlowCallThreshold,
        Timestamp: time.Now(),
    }

    cb.recordCallResult(callResult)

    return result, err
}

func (cb *CircuitBreaker) executeCall(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
    // Create a channel to receive the result
    resultChan := make(chan struct {
        result interface{}
        err    error
    }, 1)

    // Execute function in a goroutine
    go func() {
        defer func() {
            if r := recover(); r != nil {
                resultChan <- struct {
                    result interface{}
                    err    error
                }{nil, fmt.Errorf("panic recovered: %v", r)}
            }
        }()

        result, err := fn()
        resultChan <- struct {
            result interface{}
            err    error
        }{result, err}
    }()

    // Wait for result or context cancellation
    select {
    case res := <-resultChan:
        return res.result, res.err
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}

func (cb *CircuitBreaker) allowRequest() bool {
    cb.mu.RLock()
    defer cb.mu.RUnlock()

    switch cb.state {
    case CircuitStateClosed:
        return true
    case CircuitStateOpen:
        // Check if timeout has passed
        if time.Since(cb.stats.StateTransitionTime) > cb.config.Timeout {
            // Transition to half-open
            cb.mu.RUnlock()
            cb.mu.Lock()
            if cb.state == CircuitStateOpen { // Double-check after acquiring write lock
                cb.transitionToHalfOpen()
            }
            cb.mu.Unlock()
            cb.mu.RLock()
            return cb.state == CircuitStateHalfOpen
        }
        return false
    case CircuitStateHalfOpen:
        // Allow limited requests in half-open state
        return cb.stats.HalfOpenRequests < cb.config.MaxRequests
    default:
        return false
    }
}

func (cb *CircuitBreaker) recordCallResult(result CallResult) {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    // Update statistics
    cb.stats.TotalRequests++

    if result.Success {
        cb.stats.SuccessfulRequests++
        now := time.Now()
        cb.stats.LastSuccessTime = &now

        if cb.state == CircuitStateHalfOpen {
            cb.stats.HalfOpenSuccesses++
            // Check if we should close the circuit
            if cb.stats.HalfOpenSuccesses >= cb.config.SuccessThreshold {
                cb.transitionToClosed()
            }
        }
    } else {
        cb.stats.FailedRequests++
        now := time.Now()
        cb.stats.LastFailureTime = &now

        if cb.state == CircuitStateHalfOpen {
            // Any failure in half-open state opens the circuit
            cb.transitionToOpen()
        }
    }

    if result.SlowCall {
        cb.stats.SlowRequests++
    }

    if cb.state == CircuitStateHalfOpen {
        cb.stats.HalfOpenRequests++
    }

    // Add to recent calls window
    cb.recentCalls = append(cb.recentCalls, result)

    // Remove old calls outside the window
    cutoff := time.Now().Add(-cb.config.SleepWindow)
    for i, call := range cb.recentCalls {
        if call.Timestamp.After(cutoff) {
            cb.recentCalls = cb.recentCalls[i:]
            break
        }
    }

    // Update rates
    cb.updateRates()

    // Check if circuit should be opened (only if closed)
    if cb.state == CircuitStateClosed && cb.shouldOpenCircuit() {
        cb.transitionToOpen()
    }

    // Update average response time
    if cb.stats.TotalRequests > 0 {
        totalTime := time.Duration(cb.stats.TotalRequests-1) * cb.stats.AverageResponseTime + result.Duration
        cb.stats.AverageResponseTime = totalTime / time.Duration(cb.stats.TotalRequests)
    } else {
        cb.stats.AverageResponseTime = result.Duration
    }

    // Call callback if set
    if cb.onCallResult != nil {
        go cb.onCallResult(result)
    }
}

func (cb *CircuitBreaker) recordRejection() {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    cb.stats.RejectedRequests++
}

func (cb *CircuitBreaker) updateRates() {
    if len(cb.recentCalls) == 0 {
        cb.stats.FailureRate = 0
        cb.stats.SlowCallRate = 0
        return
    }

    failures := 0
    slowCalls := 0

    for _, call := range cb.recentCalls {
        if !call.Success {
            failures++
        }
        if call.SlowCall {
            slowCalls++
        }
    }

    cb.stats.FailureRate = float64(failures) / float64(len(cb.recentCalls))
    cb.stats.SlowCallRate = float64(slowCalls) / float64(len(cb.recentCalls))
}

func (cb *CircuitBreaker) shouldOpenCircuit() bool {
    // Need minimum request volume
    if len(cb.recentCalls) < cb.config.RequestVolumeThreshold {
        return false
    }

    // Check failure rate
    if cb.stats.FailureRate >= cb.config.ErrorThreshold {
        return true
    }

    // Check slow call rate
    if cb.stats.SlowCallRate >= cb.config.SlowCallRateThreshold {
        return true
    }

    return false
}

func (cb *CircuitBreaker) transitionToOpen() {
    if cb.state != CircuitStateOpen {
        oldState := cb.state
        cb.state = CircuitStateOpen
        cb.stats.State = CircuitStateOpen
        cb.stats.StateTransitionTime = time.Now()
        cb.stats.HalfOpenRequests = 0
        cb.stats.HalfOpenSuccesses = 0

        log.Printf("Circuit breaker '%s' opened", cb.config.Name)

        if cb.onStateChange != nil {
            go cb.onStateChange(oldState, CircuitStateOpen)
        }
    }
}

func (cb *CircuitBreaker) transitionToHalfOpen() {
    if cb.state != CircuitStateHalfOpen {
        oldState := cb.state
        cb.state = CircuitStateHalfOpen
        cb.stats.State = CircuitStateHalfOpen
        cb.stats.StateTransitionTime = time.Now()
        cb.stats.HalfOpenRequests = 0
        cb.stats.HalfOpenSuccesses = 0

        log.Printf("Circuit breaker '%s' half-opened", cb.config.Name)

        if cb.onStateChange != nil {
            go cb.onStateChange(oldState, CircuitStateHalfOpen)
        }
    }
}

func (cb *CircuitBreaker) transitionToClosed() {
    if cb.state != CircuitStateClosed {
        oldState := cb.state
        cb.state = CircuitStateClosed
        cb.stats.State = CircuitStateClosed
        cb.stats.StateTransitionTime = time.Now()
        cb.stats.HalfOpenRequests = 0
        cb.stats.HalfOpenSuccesses = 0

        // Reset failure tracking
        cb.recentCalls = make([]CallResult, 0)
        cb.stats.FailureRate = 0
        cb.stats.SlowCallRate = 0

        log.Printf("Circuit breaker '%s' closed", cb.config.Name)

        if cb.onStateChange != nil {
            go cb.onStateChange(oldState, CircuitStateClosed)
        }
    }
}

// Get current state
func (cb *CircuitBreaker) GetState() CircuitState {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    return cb.state
}

// Get statistics
func (cb *CircuitBreaker) GetStats() CircuitBreakerStats {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    return cb.stats
}

// Get configuration
func (cb *CircuitBreaker) GetConfig() CircuitBreakerConfig {
    return cb.config
}

// Reset circuit breaker
func (cb *CircuitBreaker) Reset() {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    oldState := cb.state
    cb.state = CircuitStateClosed
    cb.stats = CircuitBreakerStats{
        State:               CircuitStateClosed,
        StateTransitionTime: time.Now(),
    }
    cb.recentCalls = make([]CallResult, 0)

    if cb.onStateChange != nil && oldState != CircuitStateClosed {
        go cb.onStateChange(oldState, CircuitStateClosed)
    }
}

// Circuit Breaker Registry
type CircuitBreakerRegistry struct {
    breakers map[string]*CircuitBreaker
    mu       sync.RWMutex
}

func NewCircuitBreakerRegistry() *CircuitBreakerRegistry {
    return &CircuitBreakerRegistry{
        breakers: make(map[string]*CircuitBreaker),
    }
}

func (r *CircuitBreakerRegistry) Register(name string, breaker *CircuitBreaker) {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.breakers[name] = breaker
}

func (r *CircuitBreakerRegistry) Get(name string) (*CircuitBreaker, bool) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    breaker, exists := r.breakers[name]
    return breaker, exists
}

func (r *CircuitBreakerRegistry) GetOrCreate(name string, config CircuitBreakerConfig) *CircuitBreaker {
    r.mu.Lock()
    defer r.mu.Unlock()

    if breaker, exists := r.breakers[name]; exists {
        return breaker
    }

    breaker := NewCircuitBreaker(config)
    r.breakers[name] = breaker
    return breaker
}

func (r *CircuitBreakerRegistry) List() map[string]*CircuitBreaker {
    r.mu.RLock()
    defer r.mu.RUnlock()

    result := make(map[string]*CircuitBreaker)
    for name, breaker := range r.breakers {
        result[name] = breaker
    }
    return result
}

func (r *CircuitBreakerRegistry) Remove(name string) {
    r.mu.Lock()
    defer r.mu.Unlock()
    delete(r.breakers, name)
}

// Service examples
type PaymentService struct {
    circuitBreaker *CircuitBreaker
    gateway        PaymentGateway
}

type PaymentGateway interface {
    ProcessPayment(req PaymentRequest) (*PaymentResponse, error)
}

type PaymentRequest struct {
    ID       string  `json:"id"`
    Amount   float64 `json:"amount"`
    Currency string  `json:"currency"`
    CardID   string  `json:"card_id"`
}

type PaymentResponse struct {
    TransactionID string `json:"transaction_id"`
    Status        string `json:"status"`
    ProcessedAt   time.Time `json:"processed_at"`
}

func NewPaymentService(gateway PaymentGateway) *PaymentService {
    config := DefaultCircuitBreakerConfig("payment-gateway")
    config.FailureThreshold = 3
    config.Timeout = time.Second * 30

    cb := NewCircuitBreaker(config)

    // Set up monitoring callbacks
    cb.OnStateChange(func(from, to CircuitState) {
        log.Printf("Payment gateway circuit breaker: %s -> %s", from, to)
    })

    return &PaymentService{
        circuitBreaker: cb,
        gateway:        gateway,
    }
}

func (s *PaymentService) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    result, err := s.circuitBreaker.Execute(func() (interface{}, error) {
        return s.gateway.ProcessPayment(req)
    })

    if err != nil {
        return nil, fmt.Errorf("payment processing failed: %w", err)
    }

    return result.(*PaymentResponse), nil
}

// Mock payment gateway for demonstration
type MockPaymentGateway struct {
    failureRate float64
    slowRate    float64
    baseLatency time.Duration
}

func NewMockPaymentGateway(failureRate, slowRate float64, baseLatency time.Duration) *MockPaymentGateway {
    return &MockPaymentGateway{
        failureRate: failureRate,
        slowRate:    slowRate,
        baseLatency: baseLatency,
    }
}

func (g *MockPaymentGateway) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    // Simulate base latency
    time.Sleep(g.baseLatency)

    // Simulate slow calls
    if rand.Float64() < g.slowRate {
        time.Sleep(time.Second * 6) // Slow call
    }

    // Simulate failures
    if rand.Float64() < g.failureRate {
        return nil, errors.New("payment gateway error")
    }

    return &PaymentResponse{
        TransactionID: fmt.Sprintf("TXN_%d", time.Now().UnixNano()),
        Status:        "SUCCESS",
        ProcessedAt:   time.Now(),
    }, nil
}

// Example usage and monitoring
func main() {
    fmt.Println("=== Circuit Breaker Pattern Demo ===\n")

    // Create registry
    registry := NewCircuitBreakerRegistry()

    // Create mock gateway with 30% failure rate and 20% slow calls
    gateway := NewMockPaymentGateway(0.3, 0.2, time.Millisecond*100)

    // Create payment service
    paymentService := NewPaymentService(gateway)

    // Register circuit breaker
    registry.Register("payment-gateway", paymentService.circuitBreaker)

    // Simulate payment requests
    fmt.Println("Simulating payment requests...")

    successCount := 0
    failureCount := 0
    rejectedCount := 0

    for i := 0; i < 50; i++ {
        req := PaymentRequest{
            ID:       fmt.Sprintf("PAY_%d", i+1),
            Amount:   100.50,
            Currency: "USD",
            CardID:   "CARD_123",
        }

        resp, err := paymentService.ProcessPayment(req)
        if err != nil {
            if fmt.Sprintf("%v", err) == "circuit breaker is OPEN" ||
               fmt.Sprintf("%v", err) == "circuit breaker is HALF_OPEN" {
                rejectedCount++
                fmt.Printf("Request %d: REJECTED (Circuit %s)\n", i+1, paymentService.circuitBreaker.GetState())
            } else {
                failureCount++
                fmt.Printf("Request %d: FAILED (%v)\n", i+1, err)
            }
        } else {
            successCount++
            fmt.Printf("Request %d: SUCCESS (%s)\n", i+1, resp.TransactionID)
        }

        // Print stats every 10 requests
        if (i+1)%10 == 0 {
            stats := paymentService.circuitBreaker.GetStats()
            fmt.Printf("\n--- Stats after %d requests ---\n", i+1)
            fmt.Printf("Circuit State: %s\n", stats.State)
            fmt.Printf("Success Rate: %.2f%%\n", float64(stats.SuccessfulRequests)/float64(stats.TotalRequests)*100)
            fmt.Printf("Failure Rate: %.2f%%\n", stats.FailureRate*100)
            fmt.Printf("Slow Call Rate: %.2f%%\n", stats.SlowCallRate*100)
            fmt.Printf("Average Response Time: %v\n", stats.AverageResponseTime)
            fmt.Printf("Rejected Requests: %d\n", stats.RejectedRequests)
            fmt.Println()
        }

        // Small delay between requests
        time.Sleep(time.Millisecond * 100)
    }

    // Final summary
    fmt.Printf("\n=== Final Summary ===\n")
    fmt.Printf("Total Requests: 50\n")
    fmt.Printf("Successful: %d\n", successCount)
    fmt.Printf("Failed: %d\n", failureCount)
    fmt.Printf("Rejected by Circuit Breaker: %d\n", rejectedCount)

    stats := paymentService.circuitBreaker.GetStats()
    fmt.Printf("\nFinal Circuit State: %s\n", stats.State)
    fmt.Printf("Total Circuit Requests: %d\n", stats.TotalRequests)
    fmt.Printf("Circuit Success Rate: %.2f%%\n", float64(stats.SuccessfulRequests)/float64(stats.TotalRequests)*100)

    // Demonstrate recovery
    if stats.State == CircuitStateOpen {
        fmt.Printf("\nWaiting for circuit to recover...\n")
        time.Sleep(time.Second * 61) // Wait for timeout

        fmt.Println("Trying requests after timeout...")
        for i := 0; i < 5; i++ {
            req := PaymentRequest{
                ID:       fmt.Sprintf("RECOVERY_%d", i+1),
                Amount:   50.25,
                Currency: "USD",
                CardID:   "CARD_456",
            }

            resp, err := paymentService.ProcessPayment(req)
            if err != nil {
                fmt.Printf("Recovery Request %d: FAILED (%v)\n", i+1, err)
            } else {
                fmt.Printf("Recovery Request %d: SUCCESS (%s)\n", i+1, resp.TransactionID)
            }

            time.Sleep(time.Millisecond * 200)
        }

        finalStats := paymentService.circuitBreaker.GetStats()
        fmt.Printf("\nFinal State After Recovery: %s\n", finalStats.State)
    }

    fmt.Println("\n=== Circuit Breaker Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Simple Circuit Breaker**

```go
type SimpleCircuitBreaker struct {
    failures    int
    threshold   int
    state       CircuitState
    lastFailure time.Time
    timeout     time.Duration
}

func (cb *SimpleCircuitBreaker) Execute(fn func() error) error {
    if cb.state == CircuitStateOpen {
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = CircuitStateHalfOpen
        } else {
            return fmt.Errorf("circuit breaker is open")
        }
    }

    err := fn()

    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()

        if cb.failures >= cb.threshold {
            cb.state = CircuitStateOpen
        }
    } else {
        cb.failures = 0
        cb.state = CircuitStateClosed
    }

    return err
}
```

2. **Bulkhead Circuit Breaker**

```go
type BulkheadCircuitBreaker struct {
    breakers map[string]*CircuitBreaker
    semaphore chan struct{}
}

func (b *BulkheadCircuitBreaker) Execute(resource string, fn func() error) error {
    // Acquire semaphore slot
    select {
    case b.semaphore <- struct{}{}:
        defer func() { <-b.semaphore }()
    default:
        return fmt.Errorf("bulkhead capacity exceeded")
    }

    // Get resource-specific circuit breaker
    breaker := b.breakers[resource]
    return breaker.Execute(fn)
}
```

3. **Adaptive Circuit Breaker**

```go
type AdaptiveCircuitBreaker struct {
    *CircuitBreaker
    errorRateWindow time.Duration
    adaptiveThreshold bool
}

func (a *AdaptiveCircuitBreaker) updateThreshold() {
    if a.adaptiveThreshold {
        // Adjust threshold based on recent performance
        recentSuccessRate := a.calculateRecentSuccessRate()

        if recentSuccessRate > 0.95 {
            a.config.ErrorThreshold = 0.7 // More tolerant
        } else if recentSuccessRate < 0.8 {
            a.config.ErrorThreshold = 0.3 // Less tolerant
        }
    }
}
```

4. **Hystrix-style Circuit Breaker**

```go
type HystrixCircuitBreaker struct {
    *CircuitBreaker
    threadPool    chan struct{}
    fallback      func() (interface{}, error)
    metricsStream chan Metrics
}

func (h *HystrixCircuitBreaker) Execute(fn func() (interface{}, error)) (interface{}, error) {
    // Try to acquire thread pool slot
    select {
    case h.threadPool <- struct{}{}:
        defer func() { <-h.threadPool }()
    default:
        // Thread pool exhausted, call fallback
        if h.fallback != nil {
            return h.fallback()
        }
        return nil, fmt.Errorf("thread pool exhausted")
    }

    // Execute with circuit breaker
    result, err := h.CircuitBreaker.Execute(fn)

    if err != nil && h.fallback != nil {
        return h.fallback()
    }

    return result, err
}
```

### Trade-offs

**Pros:**

- **Fail Fast**: Prevents long waits during service failures
- **Resource Protection**: Protects resources from overload
- **Automatic Recovery**: Automatically attempts recovery
- **Observability**: Provides metrics and monitoring
- **Resilience**: Improves overall system resilience

**Cons:**

- **False Positives**: May open circuit during temporary issues
- **Complexity**: Adds complexity to system architecture
- **Configuration**: Requires careful tuning of thresholds
- **Latency**: Adds small overhead to each call
- **State Management**: Needs persistent state for distributed systems

**When to Choose Circuit Breaker vs Alternatives:**

| Scenario                | Pattern                 | Reason                            |
| ----------------------- | ----------------------- | --------------------------------- |
| External service calls  | Circuit Breaker         | Protects against cascade failures |
| Internal method calls   | Timeout                 | Lower overhead                    |
| Batch processing        | Bulkhead                | Resource isolation                |
| Critical operations     | Retry + Circuit Breaker | Comprehensive resilience          |
| Non-critical operations | Timeout only            | Simpler implementation            |

## Testable Example

```go
package main

import (
    "context"
    "fmt"
    "testing"
    "time"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestCircuitBreaker_ClosedState(t *testing.T) {
    config := DefaultCircuitBreakerConfig("test")
    config.FailureThreshold = 3

    cb := NewCircuitBreaker(config)

    // Should be closed initially
    assert.Equal(t, CircuitStateClosed, cb.GetState())

    // Successful calls should keep circuit closed
    for i := 0; i < 5; i++ {
        result, err := cb.Execute(func() (interface{}, error) {
            return "success", nil
        })

        assert.NoError(t, err)
        assert.Equal(t, "success", result)
        assert.Equal(t, CircuitStateClosed, cb.GetState())
    }

    stats := cb.GetStats()
    assert.Equal(t, int64(5), stats.TotalRequests)
    assert.Equal(t, int64(5), stats.SuccessfulRequests)
    assert.Equal(t, int64(0), stats.FailedRequests)
}

func TestCircuitBreaker_OpenState(t *testing.T) {
    config := DefaultCircuitBreakerConfig("test")
    config.FailureThreshold = 3
    config.RequestVolumeThreshold = 3
    config.ErrorThreshold = 0.5

    cb := NewCircuitBreaker(config)

    // Generate enough failures to open circuit
    for i := 0; i < 5; i++ {
        _, err := cb.Execute(func() (interface{}, error) {
            return nil, fmt.Errorf("simulated error")
        })

        assert.Error(t, err)
        assert.Contains(t, err.Error(), "simulated error")
    }

    // Circuit should be open now
    assert.Equal(t, CircuitStateOpen, cb.GetState())

    // Subsequent calls should be rejected
    result, err := cb.Execute(func() (interface{}, error) {
        t.Fatal("This should not be called when circuit is open")
        return nil, nil
    })

    assert.Error(t, err)
    assert.Nil(t, result)
    assert.Contains(t, err.Error(), "circuit breaker is OPEN")

    stats := cb.GetStats()
    assert.True(t, stats.RejectedRequests > 0)
}

func TestCircuitBreaker_HalfOpenState(t *testing.T) {
    config := DefaultCircuitBreakerConfig("test")
    config.FailureThreshold = 2
    config.SuccessThreshold = 2
    config.Timeout = time.Millisecond * 100 // Short timeout for testing
    config.MaxRequests = 3
    config.RequestVolumeThreshold = 2

    cb := NewCircuitBreaker(config)

    // Open the circuit
    for i := 0; i < 3; i++ {
        cb.Execute(func() (interface{}, error) {
            return nil, fmt.Errorf("error")
        })
    }

    assert.Equal(t, CircuitStateOpen, cb.GetState())

    // Wait for timeout
    time.Sleep(time.Millisecond * 150)

    // Next call should transition to half-open
    result, err := cb.Execute(func() (interface{}, error) {
        return "success", nil
    })

    assert.NoError(t, err)
    assert.Equal(t, "success", result)
    assert.Equal(t, CircuitStateHalfOpen, cb.GetState())

    // Another successful call should close the circuit
    result, err = cb.Execute(func() (interface{}, error) {
        return "success", nil
    })

    assert.NoError(t, err)
    assert.Equal(t, "success", result)
    assert.Equal(t, CircuitStateClosed, cb.GetState())
}

func TestCircuitBreaker_HalfOpenToOpen(t *testing.T) {
    config := DefaultCircuitBreakerConfig("test")
    config.FailureThreshold = 2
    config.Timeout = time.Millisecond * 100
    config.RequestVolumeThreshold = 2

    cb := NewCircuitBreaker(config)

    // Open the circuit
    for i := 0; i < 3; i++ {
        cb.Execute(func() (interface{}, error) {
            return nil, fmt.Errorf("error")
        })
    }

    // Wait for timeout
    time.Sleep(time.Millisecond * 150)

    // Fail in half-open state
    _, err := cb.Execute(func() (interface{}, error) {
        return nil, fmt.Errorf("still failing")
    })

    assert.Error(t, err)
    assert.Equal(t, CircuitStateOpen, cb.GetState())
}

func TestCircuitBreaker_SlowCalls(t *testing.T) {
    config := DefaultCircuitBreakerConfig("test")
    config.SlowCallThreshold = time.Millisecond * 100
    config.SlowCallRateThreshold = 0.5
    config.RequestVolumeThreshold = 3

    cb := NewCircuitBreaker(config)

    // Make slow calls
    for i := 0; i < 4; i++ {
        cb.Execute(func() (interface{}, error) {
            time.Sleep(time.Millisecond * 150) // Slow call
            return "slow success", nil
        })
    }

    // Circuit should open due to slow calls
    assert.Equal(t, CircuitStateOpen, cb.GetState())

    stats := cb.GetStats()
    assert.True(t, stats.SlowCallRate >= config.SlowCallRateThreshold)
}

func TestCircuitBreaker_ContextCancellation(t *testing.T) {
    config := DefaultCircuitBreakerConfig("test")
    cb := NewCircuitBreaker(config)

    ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond*50)
    defer cancel()

    start := time.Now()
    result, err := cb.ExecuteWithContext(ctx, func() (interface{}, error) {
        time.Sleep(time.Millisecond * 100) // Longer than context timeout
        return "should not reach here", nil
    })
    duration := time.Since(start)

    assert.Error(t, err)
    assert.Nil(t, result)
    assert.Contains(t, err.Error(), "context deadline exceeded")
    assert.True(t, duration < time.Millisecond*100) // Should timeout before function completes
}

func TestCircuitBreaker_PanicRecovery(t *testing.T) {
    config := DefaultCircuitBreakerConfig("test")
    cb := NewCircuitBreaker(config)

    result, err := cb.Execute(func() (interface{}, error) {
        panic("test panic")
    })

    assert.Error(t, err)
    assert.Nil(t, result)
    assert.Contains(t, err.Error(), "panic recovered")
}

func TestCircuitBreaker_Callbacks(t *testing.T) {
    config := DefaultCircuitBreakerConfig("test")
    config.FailureThreshold = 2
    config.RequestVolumeThreshold = 2

    cb := NewCircuitBreaker(config)

    var stateChanges []string
    var callResults []bool

    cb.OnStateChange(func(from, to CircuitState) {
        stateChanges = append(stateChanges, fmt.Sprintf("%s->%s", from, to))
    })

    cb.OnCallResult(func(result CallResult) {
        callResults = append(callResults, result.Success)
    })

    // Make some calls to trigger state changes
    cb.Execute(func() (interface{}, error) { return nil, fmt.Errorf("error") })
    cb.Execute(func() (interface{}, error) { return nil, fmt.Errorf("error") })
    cb.Execute(func() (interface{}, error) { return "success", nil })

    // Wait for callbacks to be called
    time.Sleep(time.Millisecond * 10)

    assert.Len(t, stateChanges, 1)
    assert.Equal(t, "CLOSED->OPEN", stateChanges[0])
    assert.Len(t, callResults, 2) // Only executed calls, not rejected ones
    assert.False(t, callResults[0])
    assert.False(t, callResults[1])
}

func TestCircuitBreaker_Reset(t *testing.T) {
    config := DefaultCircuitBreakerConfig("test")
    config.FailureThreshold = 2
    config.RequestVolumeThreshold = 2

    cb := NewCircuitBreaker(config)

    // Open the circuit
    for i := 0; i < 3; i++ {
        cb.Execute(func() (interface{}, error) {
            return nil, fmt.Errorf("error")
        })
    }

    assert.Equal(t, CircuitStateOpen, cb.GetState())

    // Reset the circuit
    cb.Reset()

    assert.Equal(t, CircuitStateClosed, cb.GetState())

    stats := cb.GetStats()
    assert.Equal(t, int64(0), stats.TotalRequests)
    assert.Equal(t, int64(0), stats.FailedRequests)
    assert.Equal(t, int64(0), stats.RejectedRequests)
}

func TestCircuitBreakerRegistry(t *testing.T) {
    registry := NewCircuitBreakerRegistry()

    // Test GetOrCreate
    config := DefaultCircuitBreakerConfig("test1")
    cb1 := registry.GetOrCreate("test1", config)
    cb2 := registry.GetOrCreate("test1", config) // Should return same instance

    assert.Same(t, cb1, cb2)

    // Test Get
    cb3, exists := registry.Get("test1")
    assert.True(t, exists)
    assert.Same(t, cb1, cb3)

    cb4, exists := registry.Get("nonexistent")
    assert.False(t, exists)
    assert.Nil(t, cb4)

    // Test List
    registry.Register("test2", NewCircuitBreaker(DefaultCircuitBreakerConfig("test2")))

    breakers := registry.List()
    assert.Len(t, breakers, 2)
    assert.Contains(t, breakers, "test1")
    assert.Contains(t, breakers, "test2")

    // Test Remove
    registry.Remove("test1")

    _, exists = registry.Get("test1")
    assert.False(t, exists)

    breakers = registry.List()
    assert.Len(t, breakers, 1)
    assert.Contains(t, breakers, "test2")
}

func TestPaymentService_CircuitBreaker(t *testing.T) {
    // Create mock gateway that fails 50% of the time
    gateway := NewMockPaymentGateway(0.5, 0.0, time.Millisecond*10)
    service := NewPaymentService(gateway)

    successCount := 0
    failureCount := 0
    rejectedCount := 0

    // Make 20 requests
    for i := 0; i < 20; i++ {
        req := PaymentRequest{
            ID:       fmt.Sprintf("PAY_%d", i+1),
            Amount:   100.0,
            Currency: "USD",
            CardID:   "CARD_123",
        }

        _, err := service.ProcessPayment(req)
        if err != nil {
            if fmt.Sprintf("%v", err) == "circuit breaker is OPEN" {
                rejectedCount++
            } else {
                failureCount++
            }
        } else {
            successCount++
        }
    }

    // Should have some failures and potentially some rejections
    assert.True(t, failureCount > 0)
    assert.True(t, successCount+failureCount+rejectedCount == 20)

    stats := service.circuitBreaker.GetStats()
    assert.True(t, stats.TotalRequests > 0)
    assert.True(t, stats.FailedRequests > 0)
}

func BenchmarkCircuitBreaker_Execute(b *testing.B) {
    config := DefaultCircuitBreakerConfig("benchmark")
    cb := NewCircuitBreaker(config)

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        cb.Execute(func() (interface{}, error) {
            return "success", nil
        })
    }
}

func BenchmarkCircuitBreaker_ExecuteWithFailures(b *testing.B) {
    config := DefaultCircuitBreakerConfig("benchmark")
    cb := NewCircuitBreaker(config)

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        cb.Execute(func() (interface{}, error) {
            if i%4 == 0 { // 25% failure rate
                return nil, fmt.Errorf("simulated error")
            }
            return "success", nil
        })
    }
}
```

## Integration Tips

### 1. HTTP Client Integration

```go
type CircuitBreakerHTTPClient struct {
    client         *http.Client
    circuitBreaker *CircuitBreaker
}

func NewCircuitBreakerHTTPClient(client *http.Client, config CircuitBreakerConfig) *CircuitBreakerHTTPClient {
    return &CircuitBreakerHTTPClient{
        client:         client,
        circuitBreaker: NewCircuitBreaker(config),
    }
}

func (c *CircuitBreakerHTTPClient) Do(req *http.Request) (*http.Response, error) {
    result, err := c.circuitBreaker.ExecuteWithContext(req.Context(), func() (interface{}, error) {
        resp, err := c.client.Do(req)
        if err != nil {
            return nil, err
        }

        // Consider 5xx status codes as failures
        if resp.StatusCode >= 500 {
            resp.Body.Close()
            return nil, fmt.Errorf("server error: %d", resp.StatusCode)
        }

        return resp, nil
    })

    if err != nil {
        return nil, err
    }

    return result.(*http.Response), nil
}
```

### 2. Database Connection Integration

```go
type CircuitBreakerDB struct {
    db             *sql.DB
    circuitBreaker *CircuitBreaker
}

func NewCircuitBreakerDB(db *sql.DB, config CircuitBreakerConfig) *CircuitBreakerDB {
    return &CircuitBreakerDB{
        db:             db,
        circuitBreaker: NewCircuitBreaker(config),
    }
}

func (c *CircuitBreakerDB) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    result, err := c.circuitBreaker.ExecuteWithContext(ctx, func() (interface{}, error) {
        return c.db.QueryContext(ctx, query, args...)
    })

    if err != nil {
        return nil, err
    }

    return result.(*sql.Rows), nil
}

func (c *CircuitBreakerDB) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
    result, err := c.circuitBreaker.ExecuteWithContext(ctx, func() (interface{}, error) {
        return c.db.ExecContext(ctx, query, args...)
    })

    if err != nil {
        return nil, err
    }

    return result.(sql.Result), nil
}
```

### 3. gRPC Integration

```go
func CircuitBreakerUnaryInterceptor(cb *CircuitBreaker) grpc.UnaryClientInterceptor {
    return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
        _, err := cb.ExecuteWithContext(ctx, func() (interface{}, error) {
            return nil, invoker(ctx, method, req, reply, cc, opts...)
        })

        return err
    }
}

func CircuitBreakerStreamInterceptor(cb *CircuitBreaker) grpc.StreamClientInterceptor {
    return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
        result, err := cb.ExecuteWithContext(ctx, func() (interface{}, error) {
            return streamer(ctx, desc, cc, method, opts...)
        })

        if err != nil {
            return nil, err
        }

        return result.(grpc.ClientStream), nil
    }
}
```

### 4. Monitoring Integration

```go
type PrometheusCircuitBreaker struct {
    *CircuitBreaker
    stateGauge      prometheus.Gauge
    requestsCounter *prometheus.CounterVec
    durationHist    prometheus.Histogram
}

func NewPrometheusCircuitBreaker(name string, config CircuitBreakerConfig) *PrometheusCircuitBreaker {
    cb := NewCircuitBreaker(config)

    stateGauge := prometheus.NewGauge(prometheus.GaugeOpts{
        Name: "circuit_breaker_state",
        Help: "Current state of the circuit breaker (0=closed, 1=open, 2=half-open)",
        ConstLabels: prometheus.Labels{"circuit": name},
    })

    requestsCounter := prometheus.NewCounterVec(prometheus.CounterOpts{
        Name: "circuit_breaker_requests_total",
        Help: "Total number of requests through the circuit breaker",
        ConstLabels: prometheus.Labels{"circuit": name},
    }, []string{"result"})

    durationHist := prometheus.NewHistogram(prometheus.HistogramOpts{
        Name: "circuit_breaker_request_duration_seconds",
        Help: "Request duration through the circuit breaker",
        ConstLabels: prometheus.Labels{"circuit": name},
    })

    prometheus.MustRegister(stateGauge, requestsCounter, durationHist)

    pcb := &PrometheusCircuitBreaker{
        CircuitBreaker:  cb,
        stateGauge:      stateGauge,
        requestsCounter: requestsCounter,
        durationHist:    durationHist,
    }

    // Set up callbacks
    cb.OnStateChange(func(from, to CircuitState) {
        pcb.stateGauge.Set(float64(to))
    })

    cb.OnCallResult(func(result CallResult) {
        status := "success"
        if !result.Success {
            status = "failure"
        }

        pcb.requestsCounter.WithLabelValues(status).Inc()
        pcb.durationHist.Observe(result.Duration.Seconds())
    })

    return pcb
}
```

## Common Interview Questions

### 1. **How does Circuit Breaker differ from Retry pattern?**

**Answer:**
| Aspect | Circuit Breaker | Retry |
|--------|-----------------|-------|
| **Purpose** | Prevent cascading failures | Overcome transient failures |
| **Behavior** | Fails fast when service is down | Keeps trying failed operations |
| **State** | Maintains state (open/closed/half-open) | Stateless |
| **Use Case** | Protect against sustained failures | Handle temporary glitches |
| **Resource Usage** | Reduces load on failing service | Increases load on failing service |

**Example:**

```go
// Retry - keeps trying
func withRetry(fn func() error, maxAttempts int) error {
    for i := 0; i < maxAttempts; i++ {
        if err := fn(); err == nil {
            return nil // Success
        }
        time.Sleep(time.Second * time.Duration(i+1))
    }
    return fmt.Errorf("failed after %d attempts", maxAttempts)
}

// Circuit Breaker - fails fast
func withCircuitBreaker(cb *CircuitBreaker, fn func() error) error {
    _, err := cb.Execute(func() (interface{}, error) {
        return nil, fn()
    })
    return err
}
```

**Best Practice**: Use them together

```go
func resilientCall(cb *CircuitBreaker, fn func() error) error {
    return cb.Execute(func() (interface{}, error) {
        // Retry within circuit breaker
        return nil, withRetry(fn, 3)
    })
}
```

### 2. **How do you tune Circuit Breaker parameters?**

**Answer:**
Circuit breaker tuning requires understanding your system's characteristics:

1. **Failure Threshold**: Based on expected error rate

```go
// For high-reliability services
config.FailureThreshold = 3      // Open after 3 failures
config.ErrorThreshold = 0.1     // 10% error rate

// For less reliable external services
config.FailureThreshold = 10     // More tolerance
config.ErrorThreshold = 0.5     // 50% error rate
```

2. **Timeout Duration**: Based on service recovery time

```go
// Fast-recovering services
config.Timeout = time.Second * 30

// Slow-recovering services
config.Timeout = time.Minute * 5
```

3. **Request Volume Threshold**: Based on traffic patterns

```go
// High-traffic services
config.RequestVolumeThreshold = 100

// Low-traffic services
config.RequestVolumeThreshold = 5
```

4. **Monitoring-based Tuning**:

```go
type AdaptiveConfig struct {
    baseConfig CircuitBreakerConfig
    metrics    *SystemMetrics
}

func (a *AdaptiveConfig) GetConfig() CircuitBreakerConfig {
    config := a.baseConfig

    if a.metrics.AvgResponseTime > time.Second*5 {
        config.SlowCallThreshold = time.Second * 3
    }

    if a.metrics.ErrorRate > 0.3 {
        config.ErrorThreshold = 0.2 // Be more sensitive
    }

    return config
}
```

### 3. **How do you handle Circuit Breaker in a distributed system?**

**Answer:**
Distributed circuit breakers require coordination:

1. **Per-Instance Circuit Breakers**:

```go
// Each service instance has its own circuit breaker
type DistributedService struct {
    localCircuitBreaker *CircuitBreaker
    instanceID          string
}

func (s *DistributedService) CallExternalService() error {
    return s.localCircuitBreaker.Execute(func() error {
        return s.externalService.Call()
    })
}
```

2. **Shared State Circuit Breaker**:

```go
type SharedStateCircuitBreaker struct {
    stateStore StateStore // Redis, etcd, etc.
    localCache *CircuitBreakerStats
    syncInterval time.Duration
}

func (s *SharedStateCircuitBreaker) Execute(fn func() error) error {
    // Check shared state
    sharedState := s.stateStore.GetState()

    if sharedState.State == CircuitStateOpen {
        return fmt.Errorf("circuit breaker is open (shared state)")
    }

    // Execute locally and update shared state
    err := fn()
    s.stateStore.UpdateStats(CallResult{
        Success:   err == nil,
        Timestamp: time.Now(),
    })

    return err
}
```

3. **Event-Driven Circuit Breaker**:

```go
type EventDrivenCircuitBreaker struct {
    localCircuitBreaker *CircuitBreaker
    eventBus           EventBus
}

func (e *EventDrivenCircuitBreaker) Execute(fn func() error) error {
    result := e.localCircuitBreaker.Execute(fn)

    // Publish circuit breaker events
    if e.localCircuitBreaker.GetState() == CircuitStateOpen {
        e.eventBus.Publish(CircuitOpenedEvent{
            Service:   "payment-gateway",
            Instance:  e.instanceID,
            Timestamp: time.Now(),
        })
    }

    return result
}

// Other instances listen for circuit breaker events
func (e *EventDrivenCircuitBreaker) OnCircuitOpened(event CircuitOpenedEvent) {
    if event.Service == "payment-gateway" {
        // Consider opening local circuit if many instances report failures
        e.evaluateGlobalState()
    }
}
```

### 4. **How do you implement fallback mechanisms with Circuit Breaker?**

**Answer:**
Fallback mechanisms provide alternative behavior when circuit is open:

1. **Static Fallback**:

```go
type FallbackCircuitBreaker struct {
    circuitBreaker *CircuitBreaker
    fallbackValue  interface{}
}

func (f *FallbackCircuitBreaker) Execute(fn func() (interface{}, error)) (interface{}, error) {
    result, err := f.circuitBreaker.Execute(fn)

    if err != nil && f.circuitBreaker.GetState() == CircuitStateOpen {
        return f.fallbackValue, nil
    }

    return result, err
}
```

2. **Dynamic Fallback**:

```go
type DynamicFallbackCircuitBreaker struct {
    circuitBreaker *CircuitBreaker
    fallbackFn     func() (interface{}, error)
}

func (d *DynamicFallbackCircuitBreaker) Execute(fn func() (interface{}, error)) (interface{}, error) {
    result, err := d.circuitBreaker.Execute(fn)

    if err != nil && d.circuitBreaker.GetState() == CircuitStateOpen {
        if d.fallbackFn != nil {
            return d.fallbackFn()
        }
    }

    return result, err
}
```

3. **Cache-Based Fallback**:

```go
type CacheFallbackCircuitBreaker struct {
    circuitBreaker *CircuitBreaker
    cache          Cache
    cacheKeyFn     func() string
}

func (c *CacheFallbackCircuitBreaker) Execute(fn func() (interface{}, error)) (interface{}, error) {
    result, err := c.circuitBreaker.Execute(fn)

    if err == nil {
        // Cache successful result
        c.cache.Set(c.cacheKeyFn(), result, time.Hour)
        return result, nil
    }

    if c.circuitBreaker.GetState() == CircuitStateOpen {
        // Return cached value if circuit is open
        if cached, found := c.cache.Get(c.cacheKeyFn()); found {
            return cached, nil
        }
    }

    return result, err
}
```

4. **Multi-Level Fallback**:

```go
type MultiLevelFallback struct {
    primary   *CircuitBreaker
    secondary *CircuitBreaker
    tertiary  *CircuitBreaker
}

func (m *MultiLevelFallback) Execute(fn func() (interface{}, error)) (interface{}, error) {
    // Try primary
    if result, err := m.primary.Execute(fn); err == nil {
        return result, nil
    }

    // Try secondary fallback
    if result, err := m.secondary.Execute(m.secondaryFn); err == nil {
        return result, nil
    }

    // Try tertiary fallback
    return m.tertiary.Execute(m.tertiaryFn)
}
```

### 5. **How do you test Circuit Breaker implementations?**

**Answer:**
Testing circuit breakers requires simulating various failure scenarios:

1. **Unit Tests for State Transitions**:

```go
func TestCircuitBreakerStates(t *testing.T) {
    cb := NewCircuitBreaker(testConfig)

    // Test closed -> open
    for i := 0; i < 5; i++ {
        cb.Execute(failingFunction)
    }
    assert.Equal(t, CircuitStateOpen, cb.GetState())

    // Test open -> half-open
    time.Sleep(cb.config.Timeout)
    cb.Execute(successFunction)
    assert.Equal(t, CircuitStateHalfOpen, cb.GetState())

    // Test half-open -> closed
    for i := 0; i < cb.config.SuccessThreshold; i++ {
        cb.Execute(successFunction)
    }
    assert.Equal(t, CircuitStateClosed, cb.GetState())
}
```

2. **Integration Tests with Mock Services**:

```go
type MockService struct {
    failureRate    float64
    responseTime   time.Duration
    shouldTimeout  bool
}

func TestPaymentServiceWithCircuitBreaker(t *testing.T) {
    mockGateway := &MockService{
        failureRate:  0.6, // 60% failure rate
        responseTime: time.Second * 2,
    }

    service := NewPaymentServiceWithCircuitBreaker(mockGateway)

    // Test that circuit opens after failures
    var rejectedCount int
    for i := 0; i < 20; i++ {
        _, err := service.ProcessPayment(testRequest)
        if strings.Contains(err.Error(), "circuit breaker") {
            rejectedCount++
        }
    }

    assert.True(t, rejectedCount > 0)
}
```

3. **Load Testing**:

```go
func TestCircuitBreakerUnderLoad(t *testing.T) {
    cb := NewCircuitBreaker(testConfig)

    var wg sync.WaitGroup
    var successCount, failureCount int64

    // Simulate 100 concurrent requests
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()

            _, err := cb.Execute(randomFunction)
            if err == nil {
                atomic.AddInt64(&successCount, 1)
            } else {
                atomic.AddInt64(&failureCount, 1)
            }
        }()
    }

    wg.Wait()

    assert.True(t, successCount+failureCount == 100)
    assert.True(t, cb.GetStats().TotalRequests <= 100) // Some may be rejected
}
```

4. **Chaos Testing**:

```go
func TestCircuitBreakerChaos(t *testing.T) {
    cb := NewCircuitBreaker(testConfig)

    scenarios := []struct {
        name        string
        duration    time.Duration
        failureRate float64
        latency     time.Duration
    }{
        {"normal", time.Second * 10, 0.1, time.Millisecond * 100},
        {"high_failure", time.Second * 5, 0.8, time.Millisecond * 100},
        {"high_latency", time.Second * 5, 0.1, time.Second * 2},
        {"recovery", time.Second * 10, 0.1, time.Millisecond * 100},
    }

    for _, scenario := range scenarios {
        t.Run(scenario.name, func(t *testing.T) {
            runChaosScenario(t, cb, scenario)
        })
    }
}
```
