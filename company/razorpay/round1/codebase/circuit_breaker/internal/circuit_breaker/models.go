package circuit_breaker

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Common errors
var (
	ErrCircuitBreakerNotFound      = errors.New("circuit breaker not found")
	ErrCircuitBreakerAlreadyExists = errors.New("circuit breaker already exists")
	ErrCircuitBreakerInactive      = errors.New("circuit breaker is inactive")
	ErrCircuitBreakerOpen          = errors.New("circuit breaker is open")
	ErrCircuitBreakerHalfOpen      = errors.New("circuit breaker is half-open")
	ErrInvalidConfiguration        = errors.New("invalid configuration")
	ErrValidationFailed            = errors.New("validation failed")
	ErrServiceInactive             = errors.New("service is inactive")
	ErrOperationTimeout            = errors.New("operation timeout")
	ErrOperationFailed             = errors.New("operation failed")
	ErrTooManyRequests             = errors.New("too many requests")
	ErrSlowCall                    = errors.New("slow call")
)

// ConcreteCircuitBreaker represents a concrete implementation of CircuitBreaker
type ConcreteCircuitBreaker struct {
	config    CircuitConfig
	state     CircuitState
	stats     CircuitStats
	mutex     sync.RWMutex
	createdAt time.Time
	updatedAt time.Time
	active    bool
}

// NewConcreteCircuitBreaker creates a new concrete circuit breaker
func NewConcreteCircuitBreaker(config CircuitConfig) *ConcreteCircuitBreaker {
	now := time.Now()

	return &ConcreteCircuitBreaker{
		config: config,
		state:  CircuitStateClosed,
		stats: CircuitStats{
			ID:                   config.ID,
			State:                CircuitStateClosed,
			TotalRequests:        0,
			SuccessfulRequests:   0,
			FailedRequests:       0,
			SlowRequests:         0,
			RejectedRequests:     0,
			ErrorRate:            0.0,
			SlowCallRate:         0.0,
			LastFailureTime:      nil,
			LastSuccessTime:      nil,
			StateChangeTime:      now,
			ConsecutiveFailures:  0,
			ConsecutiveSuccesses: 0,
			Metadata:             make(map[string]interface{}),
		},
		createdAt: now,
		updatedAt: now,
		active:    config.Active,
	}
}

// GetID returns the circuit breaker ID
func (cb *ConcreteCircuitBreaker) GetID() string {
	return cb.config.ID
}

// GetName returns the circuit breaker name
func (cb *ConcreteCircuitBreaker) GetName() string {
	return cb.config.Name
}

// GetDescription returns the circuit breaker description
func (cb *ConcreteCircuitBreaker) GetDescription() string {
	return cb.config.Description
}

// GetState returns the current state
func (cb *ConcreteCircuitBreaker) GetState() CircuitState {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.state
}

// GetConfig returns the configuration
func (cb *ConcreteCircuitBreaker) GetConfig() CircuitConfig {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.config
}

// GetStats returns the statistics
func (cb *ConcreteCircuitBreaker) GetStats() CircuitStats {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.stats
}

// GetCreatedAt returns the creation time
func (cb *ConcreteCircuitBreaker) GetCreatedAt() time.Time {
	return cb.createdAt
}

// GetUpdatedAt returns the last update time
func (cb *ConcreteCircuitBreaker) GetUpdatedAt() time.Time {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.updatedAt
}

// GetMetadata returns the metadata
func (cb *ConcreteCircuitBreaker) GetMetadata() map[string]interface{} {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.stats.Metadata
}

// SetMetadata sets the metadata
func (cb *ConcreteCircuitBreaker) SetMetadata(key string, value interface{}) {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	if cb.stats.Metadata == nil {
		cb.stats.Metadata = make(map[string]interface{})
	}
	cb.stats.Metadata[key] = value
	cb.updatedAt = time.Now()
}

// IsActive returns whether the circuit breaker is active
func (cb *ConcreteCircuitBreaker) IsActive() bool {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	return cb.active
}

// SetActive sets the active status
func (cb *ConcreteCircuitBreaker) SetActive(active bool) {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	cb.active = active
	cb.updatedAt = time.Now()
}

// Execute executes an operation through the circuit breaker
func (cb *ConcreteCircuitBreaker) Execute(ctx context.Context, operation func() (interface{}, error)) (interface{}, error) {
	if !cb.IsActive() {
		return nil, ErrCircuitBreakerInactive
	}

	// Check if circuit breaker allows the request
	if !cb.allowRequest() {
		cb.recordRejectedRequest()
		return nil, ErrCircuitBreakerOpen
	}

	// Execute the operation
	start := time.Now()
	result, err := operation()
	duration := time.Since(start)

	// Record the result
	cb.recordResult(result, err, duration)

	return result, err
}

// ExecuteAsync executes an operation asynchronously through the circuit breaker
func (cb *ConcreteCircuitBreaker) ExecuteAsync(ctx context.Context, operation func() (interface{}, error)) <-chan CircuitResult {
	resultChan := make(chan CircuitResult, 1)

	go func() {
		defer close(resultChan)

		if !cb.IsActive() {
			resultChan <- CircuitResult{
				Value:     nil,
				Error:     ErrCircuitBreakerInactive,
				Duration:  0,
				Timestamp: time.Now(),
			}
			return
		}

		// Check if circuit breaker allows the request
		if !cb.allowRequest() {
			cb.recordRejectedRequest()
			resultChan <- CircuitResult{
				Value:     nil,
				Error:     ErrCircuitBreakerOpen,
				Duration:  0,
				Timestamp: time.Now(),
			}
			return
		}

		// Execute the operation
		start := time.Now()
		result, err := operation()
		duration := time.Since(start)

		// Record the result
		cb.recordResult(result, err, duration)

		resultChan <- CircuitResult{
			Value:     result,
			Error:     err,
			Duration:  duration,
			Timestamp: time.Now(),
		}
	}()

	return resultChan
}

// Reset resets the circuit breaker
func (cb *ConcreteCircuitBreaker) Reset() error {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	if !cb.active {
		return ErrCircuitBreakerInactive
	}

	cb.state = CircuitStateClosed
	cb.stats.State = CircuitStateClosed
	cb.stats.TotalRequests = 0
	cb.stats.SuccessfulRequests = 0
	cb.stats.FailedRequests = 0
	cb.stats.SlowRequests = 0
	cb.stats.RejectedRequests = 0
	cb.stats.ErrorRate = 0.0
	cb.stats.SlowCallRate = 0.0
	cb.stats.LastFailureTime = nil
	cb.stats.LastSuccessTime = nil
	cb.stats.StateChangeTime = time.Now()
	cb.stats.ConsecutiveFailures = 0
	cb.stats.ConsecutiveSuccesses = 0
	cb.updatedAt = time.Now()

	return nil
}

// Validate validates the circuit breaker configuration
func (cb *ConcreteCircuitBreaker) Validate() error {
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()

	if cb.config.ID == "" {
		return fmt.Errorf("circuit breaker ID is required")
	}
	if cb.config.Name == "" {
		return fmt.Errorf("circuit breaker name is required")
	}
	if cb.config.Description == "" {
		return fmt.Errorf("circuit breaker description is required")
	}
	if cb.config.FailureThreshold <= 0 {
		return fmt.Errorf("failure threshold must be positive")
	}
	if cb.config.SuccessThreshold <= 0 {
		return fmt.Errorf("success threshold must be positive")
	}
	if cb.config.Timeout <= 0 {
		return fmt.Errorf("timeout must be positive")
	}
	if cb.config.ResetTimeout <= 0 {
		return fmt.Errorf("reset timeout must be positive")
	}
	if cb.config.MaxRequests <= 0 {
		return fmt.Errorf("max requests must be positive")
	}
	if cb.config.RequestVolumeThreshold <= 0 {
		return fmt.Errorf("request volume threshold must be positive")
	}
	if cb.config.SleepWindow <= 0 {
		return fmt.Errorf("sleep window must be positive")
	}
	if cb.config.ErrorThreshold < 0 || cb.config.ErrorThreshold > 1 {
		return fmt.Errorf("error threshold must be between 0 and 1")
	}
	if cb.config.SlowCallThreshold <= 0 {
		return fmt.Errorf("slow call threshold must be positive")
	}
	if cb.config.SlowCallRatioThreshold < 0 || cb.config.SlowCallRatioThreshold > 1 {
		return fmt.Errorf("slow call ratio threshold must be between 0 and 1")
	}

	return nil
}

// Cleanup performs cleanup operations
func (cb *ConcreteCircuitBreaker) Cleanup(ctx context.Context) error {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	if !cb.active {
		return ErrCircuitBreakerInactive
	}

	// Reset statistics
	cb.stats.TotalRequests = 0
	cb.stats.SuccessfulRequests = 0
	cb.stats.FailedRequests = 0
	cb.stats.SlowRequests = 0
	cb.stats.RejectedRequests = 0
	cb.stats.ErrorRate = 0.0
	cb.stats.SlowCallRate = 0.0
	cb.stats.ConsecutiveFailures = 0
	cb.stats.ConsecutiveSuccesses = 0

	cb.updatedAt = time.Now()
	return nil
}

// allowRequest checks if the circuit breaker allows the request
func (cb *ConcreteCircuitBreaker) allowRequest() bool {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	switch cb.state {
	case CircuitStateClosed:
		return true
	case CircuitStateOpen:
		// Check if enough time has passed to try half-open
		if time.Since(cb.stats.StateChangeTime) >= cb.config.ResetTimeout {
			cb.state = CircuitStateHalfOpen
			cb.stats.State = CircuitStateHalfOpen
			cb.stats.StateChangeTime = time.Now()
			return true
		}
		return false
	case CircuitStateHalfOpen:
		// Allow limited requests in half-open state
		return cb.stats.TotalRequests < cb.config.MaxRequests
	default:
		return false
	}
}

// recordResult records the result of an operation
func (cb *ConcreteCircuitBreaker) recordResult(result interface{}, err error, duration time.Duration) {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	now := time.Now()
	cb.stats.TotalRequests++
	cb.updatedAt = now

	// Check if it's a slow call
	isSlowCall := duration > cb.config.SlowCallThreshold
	if isSlowCall {
		cb.stats.SlowRequests++
	}

	// Check if it's a success or failure
	if err == nil {
		cb.stats.SuccessfulRequests++
		cb.stats.ConsecutiveSuccesses++
		cb.stats.ConsecutiveFailures = 0
		cb.stats.LastSuccessTime = &now

		// Update error rate
		cb.updateErrorRate()

		// Check if we should transition to closed state
		if cb.state == CircuitStateHalfOpen && cb.stats.ConsecutiveSuccesses >= cb.config.SuccessThreshold {
			cb.state = CircuitStateClosed
			cb.stats.State = CircuitStateClosed
			cb.stats.StateChangeTime = now
		}
	} else {
		cb.stats.FailedRequests++
		cb.stats.ConsecutiveFailures++
		cb.stats.ConsecutiveSuccesses = 0
		cb.stats.LastFailureTime = &now

		// Update error rate
		cb.updateErrorRate()

		// Check if we should transition to open state
		if cb.shouldOpen() {
			cb.state = CircuitStateOpen
			cb.stats.State = CircuitStateOpen
			cb.stats.StateChangeTime = now
		}
	}

	// Update slow call rate
	cb.updateSlowCallRate()
}

// recordRejectedRequest records a rejected request
func (cb *ConcreteCircuitBreaker) recordRejectedRequest() {
	cb.mutex.Lock()
	defer cb.mutex.Unlock()

	cb.stats.RejectedRequests++
	cb.updatedAt = time.Now()
}

// shouldOpen checks if the circuit breaker should open
func (cb *ConcreteCircuitBreaker) shouldOpen() bool {
	// Check if we have enough requests to make a decision
	if cb.stats.TotalRequests < cb.config.RequestVolumeThreshold {
		return false
	}

	// Check if error rate exceeds threshold
	if cb.stats.ErrorRate >= cb.config.ErrorThreshold {
		return true
	}

	// Check if slow call rate exceeds threshold
	if cb.stats.SlowCallRate >= cb.config.SlowCallRatioThreshold {
		return true
	}

	// Check if consecutive failures exceed threshold
	if cb.stats.ConsecutiveFailures >= cb.config.FailureThreshold {
		return true
	}

	return false
}

// updateErrorRate updates the error rate
func (cb *ConcreteCircuitBreaker) updateErrorRate() {
	if cb.stats.TotalRequests > 0 {
		cb.stats.ErrorRate = float64(cb.stats.FailedRequests) / float64(cb.stats.TotalRequests)
	} else {
		cb.stats.ErrorRate = 0.0
	}
}

// updateSlowCallRate updates the slow call rate
func (cb *ConcreteCircuitBreaker) updateSlowCallRate() {
	if cb.stats.TotalRequests > 0 {
		cb.stats.SlowCallRate = float64(cb.stats.SlowRequests) / float64(cb.stats.TotalRequests)
	} else {
		cb.stats.SlowCallRate = 0.0
	}
}
