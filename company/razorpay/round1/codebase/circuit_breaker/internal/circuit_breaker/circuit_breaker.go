package circuit_breaker

import (
	"context"
	"sync"
	"time"
)

// CircuitBreakerManagerImpl implements the CircuitBreakerManager interface
type CircuitBreakerManagerImpl struct {
	circuitBreakers map[string]CircuitBreaker
	mutex           sync.RWMutex
	createdAt       time.Time
	updatedAt       time.Time
	active          bool
}

// NewCircuitBreakerManager creates a new circuit breaker manager
func NewCircuitBreakerManager() *CircuitBreakerManagerImpl {
	return &CircuitBreakerManagerImpl{
		circuitBreakers: make(map[string]CircuitBreaker),
		createdAt:       time.Now(),
		updatedAt:       time.Now(),
		active:          true,
	}
}

// CreateCircuitBreaker creates a new circuit breaker
func (m *CircuitBreakerManagerImpl) CreateCircuitBreaker(config CircuitConfig) (CircuitBreaker, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.active {
		return nil, ErrServiceInactive
	}

	if _, exists := m.circuitBreakers[config.ID]; exists {
		return nil, ErrCircuitBreakerAlreadyExists
	}

	circuitBreaker := NewConcreteCircuitBreaker(config)
	m.circuitBreakers[config.ID] = circuitBreaker
	m.updatedAt = time.Now()

	return circuitBreaker, nil
}

// DestroyCircuitBreaker destroys a circuit breaker
func (m *CircuitBreakerManagerImpl) DestroyCircuitBreaker(id string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.active {
		return ErrServiceInactive
	}

	circuitBreaker, exists := m.circuitBreakers[id]
	if !exists {
		return ErrCircuitBreakerNotFound
	}

	// Cleanup circuit breaker
	if err := circuitBreaker.Cleanup(context.Background()); err != nil {
		// Log error but continue
	}

	delete(m.circuitBreakers, id)
	m.updatedAt = time.Now()

	return nil
}

// GetCircuitBreaker retrieves a circuit breaker by ID
func (m *CircuitBreakerManagerImpl) GetCircuitBreaker(id string) (CircuitBreaker, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	if !m.active {
		return nil, ErrServiceInactive
	}

	circuitBreaker, exists := m.circuitBreakers[id]
	if !exists {
		return nil, ErrCircuitBreakerNotFound
	}

	return circuitBreaker, nil
}

// ListCircuitBreakers returns a list of all circuit breaker IDs
func (m *CircuitBreakerManagerImpl) ListCircuitBreakers() []string {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	ids := make([]string, 0, len(m.circuitBreakers))
	for id := range m.circuitBreakers {
		ids = append(ids, id)
	}

	return ids
}

// GetCircuitBreakerStats returns statistics for a specific circuit breaker
func (m *CircuitBreakerManagerImpl) GetCircuitBreakerStats(id string) map[string]interface{} {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	circuitBreaker, exists := m.circuitBreakers[id]
	if !exists {
		return map[string]interface{}{
			"error": "circuit breaker not found",
		}
	}

	stats := circuitBreaker.GetStats()
	return map[string]interface{}{
		"id":                    stats.ID,
		"state":                 stats.State,
		"total_requests":        stats.TotalRequests,
		"successful_requests":   stats.SuccessfulRequests,
		"failed_requests":       stats.FailedRequests,
		"slow_requests":         stats.SlowRequests,
		"rejected_requests":     stats.RejectedRequests,
		"error_rate":            stats.ErrorRate,
		"slow_call_rate":        stats.SlowCallRate,
		"last_failure_time":     stats.LastFailureTime,
		"last_success_time":     stats.LastSuccessTime,
		"state_change_time":     stats.StateChangeTime,
		"consecutive_failures":  stats.ConsecutiveFailures,
		"consecutive_successes": stats.ConsecutiveSuccesses,
		"metadata":              stats.Metadata,
	}
}

// GetAllCircuitBreakerStats returns statistics for all circuit breakers
func (m *CircuitBreakerManagerImpl) GetAllCircuitBreakerStats() map[string]interface{} {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	stats := make(map[string]interface{})
	for id, circuitBreaker := range m.circuitBreakers {
		stats[id] = m.GetCircuitBreakerStats(id)
	}

	return stats
}

// IsCircuitBreakerActive checks if a circuit breaker is active
func (m *CircuitBreakerManagerImpl) IsCircuitBreakerActive(id string) bool {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	circuitBreaker, exists := m.circuitBreakers[id]
	if !exists {
		return false
	}

	return circuitBreaker.IsActive()
}

// SetCircuitBreakerActive sets the active status of a circuit breaker
func (m *CircuitBreakerManagerImpl) SetCircuitBreakerActive(id string, active bool) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.active {
		return ErrServiceInactive
	}

	circuitBreaker, exists := m.circuitBreakers[id]
	if !exists {
		return ErrCircuitBreakerNotFound
	}

	circuitBreaker.SetActive(active)
	m.updatedAt = time.Now()

	return nil
}

// GetManagerStats returns manager statistics
func (m *CircuitBreakerManagerImpl) GetManagerStats() map[string]interface{} {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	closedBreakers := 0
	openBreakers := 0
	halfOpenBreakers := 0
	totalRequests := int64(0)
	successfulRequests := int64(0)
	failedRequests := int64(0)
	rejectedRequests := int64(0)

	for _, circuitBreaker := range m.circuitBreakers {
		stats := circuitBreaker.GetStats()

		switch stats.State {
		case CircuitStateClosed:
			closedBreakers++
		case CircuitStateOpen:
			openBreakers++
		case CircuitStateHalfOpen:
			halfOpenBreakers++
		}

		totalRequests += stats.TotalRequests
		successfulRequests += stats.SuccessfulRequests
		failedRequests += stats.FailedRequests
		rejectedRequests += stats.RejectedRequests
	}

	return map[string]interface{}{
		"active":                 m.active,
		"created_at":             m.createdAt,
		"updated_at":             m.updatedAt,
		"circuit_breakers_count": len(m.circuitBreakers),
		"closed_breakers":        closedBreakers,
		"open_breakers":          openBreakers,
		"half_open_breakers":     halfOpenBreakers,
		"total_requests":         totalRequests,
		"successful_requests":    successfulRequests,
		"failed_requests":        failedRequests,
		"rejected_requests":      rejectedRequests,
		"circuit_breakers":       m.ListCircuitBreakers(),
	}
}

// GetHealthStatus returns the health status of the manager
func (m *CircuitBreakerManagerImpl) GetHealthStatus() map[string]interface{} {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"circuit_breaker_manager": map[string]interface{}{
				"status": "healthy",
				"active": m.active,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !m.active {
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_manager"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_manager"].(map[string]interface{})["message"] = "Circuit breaker manager is inactive"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (m *CircuitBreakerManagerImpl) Cleanup(ctx context.Context) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.active {
		return ErrServiceInactive
	}

	// Cleanup all circuit breakers
	for _, circuitBreaker := range m.circuitBreakers {
		if err := circuitBreaker.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	m.updatedAt = time.Now()
	return nil
}

// CircuitBreakerExecutorImpl implements the CircuitBreakerExecutor interface
type CircuitBreakerExecutorImpl struct {
	manager   CircuitBreakerManager
	createdAt time.Time
	updatedAt time.Time
	active    bool
	mutex     sync.RWMutex
}

// NewCircuitBreakerExecutor creates a new circuit breaker executor
func NewCircuitBreakerExecutor(manager CircuitBreakerManager) *CircuitBreakerExecutorImpl {
	return &CircuitBreakerExecutorImpl{
		manager:   manager,
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// Execute executes an operation through a circuit breaker
func (e *CircuitBreakerExecutorImpl) Execute(ctx context.Context, id string, operation func() (interface{}, error)) (interface{}, error) {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	if !e.active {
		return nil, ErrServiceInactive
	}

	circuitBreaker, err := e.manager.GetCircuitBreaker(id)
	if err != nil {
		return nil, err
	}

	return circuitBreaker.Execute(ctx, operation)
}

// ExecuteAsync executes an operation asynchronously through a circuit breaker
func (e *CircuitBreakerExecutorImpl) ExecuteAsync(ctx context.Context, id string, operation func() (interface{}, error)) <-chan CircuitResult {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	if !e.active {
		resultChan := make(chan CircuitResult, 1)
		resultChan <- CircuitResult{
			Value:     nil,
			Error:     ErrServiceInactive,
			Duration:  0,
			Timestamp: time.Now(),
		}
		close(resultChan)
		return resultChan
	}

	circuitBreaker, err := e.manager.GetCircuitBreaker(id)
	if err != nil {
		resultChan := make(chan CircuitResult, 1)
		resultChan <- CircuitResult{
			Value:     nil,
			Error:     err,
			Duration:  0,
			Timestamp: time.Now(),
		}
		close(resultChan)
		return resultChan
	}

	return circuitBreaker.ExecuteAsync(ctx, operation)
}

// GetExecutorStats returns executor statistics
func (e *CircuitBreakerExecutorImpl) GetExecutorStats() map[string]interface{} {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	return map[string]interface{}{
		"active":        e.active,
		"created_at":    e.createdAt,
		"updated_at":    e.updatedAt,
		"manager_stats": e.manager.GetManagerStats(),
	}
}

// GetHealthStatus returns the health status of the executor
func (e *CircuitBreakerExecutorImpl) GetHealthStatus() map[string]interface{} {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"circuit_breaker_executor": map[string]interface{}{
				"status": "healthy",
				"active": e.active,
			},
			"circuit_breaker_manager": map[string]interface{}{
				"status": "healthy",
				"active": e.manager != nil,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !e.active {
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_executor"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_executor"].(map[string]interface{})["message"] = "Circuit breaker executor is inactive"
	}

	if e.manager == nil {
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_manager"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_manager"].(map[string]interface{})["message"] = "Circuit breaker manager is not available"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (e *CircuitBreakerExecutorImpl) Cleanup(ctx context.Context) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if !e.active {
		return ErrServiceInactive
	}

	// Cleanup manager
	if e.manager != nil {
		if err := e.manager.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	e.updatedAt = time.Now()
	return nil
}
