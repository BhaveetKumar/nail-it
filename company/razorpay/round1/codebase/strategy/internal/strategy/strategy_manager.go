package strategy

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// StrategyManagerImpl implements StrategyManager interface
type StrategyManagerImpl struct {
	strategies       map[string]interface{}
	defaultStrategy  string
	fallbackStrategy string
	timeout          time.Duration
	retryCount       int
	circuitBreaker   *CircuitBreaker
	metrics          StrategyMetrics
	mu               sync.RWMutex
}

// NewStrategyManager creates a new strategy manager
func NewStrategyManager(defaultStrategy, fallbackStrategy string, timeout time.Duration, retryCount int, metrics StrategyMetrics) *StrategyManagerImpl {
	return &StrategyManagerImpl{
		strategies:       make(map[string]interface{}),
		defaultStrategy:  defaultStrategy,
		fallbackStrategy: fallbackStrategy,
		timeout:          timeout,
		retryCount:       retryCount,
		circuitBreaker:   NewCircuitBreaker(5, 30*time.Second, 3),
		metrics:          metrics,
	}
}

// GetStrategy retrieves a strategy by name
func (sm *StrategyManagerImpl) GetStrategy(strategyName string) (interface{}, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	strategy, exists := sm.strategies[strategyName]
	if !exists {
		return nil, fmt.Errorf("strategy not found: %s", strategyName)
	}

	return strategy, nil
}

// GetAvailableStrategies returns list of available strategies
func (sm *StrategyManagerImpl) GetAvailableStrategies() []string {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	strategies := make([]string, 0, len(sm.strategies))
	for name := range sm.strategies {
		strategies = append(strategies, name)
	}

	return strategies
}

// RegisterStrategy registers a new strategy
func (sm *StrategyManagerImpl) RegisterStrategy(strategyName string, strategy interface{}) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if strategyName == "" {
		return fmt.Errorf("strategy name cannot be empty")
	}

	if strategy == nil {
		return fmt.Errorf("strategy cannot be nil")
	}

	sm.strategies[strategyName] = strategy
	return nil
}

// UnregisterStrategy unregisters a strategy
func (sm *StrategyManagerImpl) UnregisterStrategy(strategyName string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if strategyName == "" {
		return fmt.Errorf("strategy name cannot be empty")
	}

	if _, exists := sm.strategies[strategyName]; !exists {
		return fmt.Errorf("strategy not found: %s", strategyName)
	}

	delete(sm.strategies, strategyName)
	return nil
}

// GetDefaultStrategy returns the default strategy
func (sm *StrategyManagerImpl) GetDefaultStrategy() (interface{}, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	if sm.defaultStrategy == "" {
		return nil, fmt.Errorf("no default strategy set")
	}

	strategy, exists := sm.strategies[sm.defaultStrategy]
	if !exists {
		return nil, fmt.Errorf("default strategy not found: %s", sm.defaultStrategy)
	}

	return strategy, nil
}

// SetDefaultStrategy sets the default strategy
func (sm *StrategyManagerImpl) SetDefaultStrategy(strategyName string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if strategyName == "" {
		return fmt.Errorf("strategy name cannot be empty")
	}

	if _, exists := sm.strategies[strategyName]; !exists {
		return fmt.Errorf("strategy not found: %s", strategyName)
	}

	sm.defaultStrategy = strategyName
	return nil
}

// ExecuteStrategy executes a strategy with circuit breaker protection
func (sm *StrategyManagerImpl) ExecuteStrategy(ctx context.Context, strategyName string, executeFunc func() (interface{}, error)) (interface{}, error) {
	start := time.Now()

	// Check circuit breaker
	if !sm.circuitBreaker.CanExecute() {
		sm.metrics.RecordStrategyCall(strategyName, time.Since(start), false)
		return nil, fmt.Errorf("circuit breaker is open for strategy: %s", strategyName)
	}

	// Execute strategy with retry
	var result interface{}
	var err error

	for i := 0; i <= sm.retryCount; i++ {
		result, err = executeFunc()
		if err == nil {
			sm.circuitBreaker.RecordSuccess()
			sm.metrics.RecordStrategyCall(strategyName, time.Since(start), true)
			return result, nil
		}

		if i < sm.retryCount {
			time.Sleep(time.Duration(i+1) * 100 * time.Millisecond)
		}
	}

	sm.circuitBreaker.RecordFailure()
	sm.metrics.RecordStrategyCall(strategyName, time.Since(start), false)
	return nil, fmt.Errorf("strategy execution failed after %d retries: %w", sm.retryCount, err)
}

// GetStrategyHealth returns health status of a strategy
func (sm *StrategyManagerImpl) GetStrategyHealth(strategyName string) (*StrategyHealth, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	strategy, exists := sm.strategies[strategyName]
	if !exists {
		return nil, fmt.Errorf("strategy not found: %s", strategyName)
	}

	// Check if strategy is available
	var isAvailable bool
	switch s := strategy.(type) {
	case PaymentStrategy:
		isAvailable = s.IsAvailable()
	case NotificationStrategy:
		isAvailable = s.IsAvailable()
	case PricingStrategy:
		isAvailable = s.IsAvailable()
	case AuthenticationStrategy:
		isAvailable = s.IsAvailable()
	case CachingStrategy:
		isAvailable = s.IsAvailable()
	case LoggingStrategy:
		isAvailable = s.IsAvailable()
	case DataProcessingStrategy:
		isAvailable = s.IsAvailable()
	default:
		isAvailable = true
	}

	status := "healthy"
	if !isAvailable {
		status = "unhealthy"
	}

	health := &StrategyHealth{
		StrategyName: strategyName,
		Status:       status,
		Message:      fmt.Sprintf("Strategy %s is %s", strategyName, status),
		LastCheck:    time.Now(),
		Metrics:      make(map[string]interface{}),
	}

	// Add metrics if available
	if sm.metrics != nil {
		metrics, err := sm.metrics.GetStrategyMetrics(strategyName)
		if err == nil {
			health.Metrics["total_calls"] = metrics.TotalCalls
			health.Metrics["success_rate"] = metrics.SuccessRate
			health.Metrics["average_duration"] = metrics.AverageDuration
		}
	}

	return health, nil
}

// GetCircuitBreakerStatus returns circuit breaker status
func (sm *StrategyManagerImpl) GetCircuitBreakerStatus() map[string]interface{} {
	return map[string]interface{}{
		"state":         sm.circuitBreaker.GetState(),
		"failure_count": sm.circuitBreaker.GetFailureCount(),
		"success_count": sm.circuitBreaker.GetSuccessCount(),
		"last_failure":  sm.circuitBreaker.GetLastFailureTime(),
		"next_retry":    sm.circuitBreaker.GetNextRetryTime(),
	}
}

// ResetCircuitBreaker resets the circuit breaker
func (sm *StrategyManagerImpl) ResetCircuitBreaker() {
	sm.circuitBreaker.Reset()
}

// CircuitBreaker implements circuit breaker pattern
type CircuitBreaker struct {
	failureThreshold int
	recoveryTimeout  time.Duration
	halfOpenMaxCalls int

	state        string
	failureCount int
	successCount int
	lastFailure  time.Time
	nextRetry    time.Time
	mu           sync.RWMutex
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(failureThreshold int, recoveryTimeout time.Duration, halfOpenMaxCalls int) *CircuitBreaker {
	return &CircuitBreaker{
		failureThreshold: failureThreshold,
		recoveryTimeout:  recoveryTimeout,
		halfOpenMaxCalls: halfOpenMaxCalls,
		state:            "closed",
	}
}

// CanExecute checks if the circuit breaker allows execution
func (cb *CircuitBreaker) CanExecute() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	switch cb.state {
	case "closed":
		return true
	case "open":
		return time.Now().After(cb.nextRetry)
	case "half-open":
		return cb.successCount < cb.halfOpenMaxCalls
	default:
		return false
	}
}

// RecordSuccess records a successful execution
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.successCount++

	if cb.state == "half-open" {
		if cb.successCount >= cb.halfOpenMaxCalls {
			cb.state = "closed"
			cb.failureCount = 0
			cb.successCount = 0
		}
	}
}

// RecordFailure records a failed execution
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount++
	cb.lastFailure = time.Now()

	if cb.state == "closed" {
		if cb.failureCount >= cb.failureThreshold {
			cb.state = "open"
			cb.nextRetry = time.Now().Add(cb.recoveryTimeout)
		}
	} else if cb.state == "half-open" {
		cb.state = "open"
		cb.nextRetry = time.Now().Add(cb.recoveryTimeout)
	}
}

// GetState returns the current state
func (cb *CircuitBreaker) GetState() string {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// GetFailureCount returns the failure count
func (cb *CircuitBreaker) GetFailureCount() int {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.failureCount
}

// GetSuccessCount returns the success count
func (cb *CircuitBreaker) GetSuccessCount() int {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.successCount
}

// GetLastFailureTime returns the last failure time
func (cb *CircuitBreaker) GetLastFailureTime() time.Time {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.lastFailure
}

// GetNextRetryTime returns the next retry time
func (cb *CircuitBreaker) GetNextRetryTime() time.Time {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.nextRetry
}

// Reset resets the circuit breaker
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.state = "closed"
	cb.failureCount = 0
	cb.successCount = 0
	cb.lastFailure = time.Time{}
	cb.nextRetry = time.Time{}
}
