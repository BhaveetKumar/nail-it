package circuit_breaker

import (
	"context"
	"sync"
	"time"
)

// CircuitBreakerServiceImpl implements the CircuitBreakerService interface
type CircuitBreakerServiceImpl struct {
	config     *ServiceConfig
	manager    CircuitBreakerManager
	executor   CircuitBreakerExecutor
	createdAt  time.Time
	updatedAt  time.Time
	active     bool
	mutex      sync.RWMutex
}

// NewCircuitBreakerService creates a new circuit breaker service
func NewCircuitBreakerService(config *ServiceConfig) *CircuitBreakerServiceImpl {
	manager := NewCircuitBreakerManager()
	executor := NewCircuitBreakerExecutor(manager)

	return &CircuitBreakerServiceImpl{
		config:    config,
		manager:   manager,
		executor:  executor,
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// CreateCircuitBreaker creates a new circuit breaker
func (s *CircuitBreakerServiceImpl) CreateCircuitBreaker(config CircuitConfig) (CircuitBreaker, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	// Validate configuration
	if err := s.validateCircuitBreakerConfig(config); err != nil {
		return nil, err
	}

	// Set default values if not provided
	s.setDefaultValues(&config)

	// Create circuit breaker
	circuitBreaker, err := s.manager.CreateCircuitBreaker(config)
	if err != nil {
		return nil, err
	}

	s.updatedAt = time.Now()
	return circuitBreaker, nil
}

// DestroyCircuitBreaker destroys a circuit breaker
func (s *CircuitBreakerServiceImpl) DestroyCircuitBreaker(id string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	if err := s.manager.DestroyCircuitBreaker(id); err != nil {
		return err
	}

	s.updatedAt = time.Now()
	return nil
}

// GetCircuitBreaker retrieves a circuit breaker by ID
func (s *CircuitBreakerServiceImpl) GetCircuitBreaker(id string) (CircuitBreaker, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	return s.manager.GetCircuitBreaker(id)
}

// ListCircuitBreakers returns a list of all circuit breaker IDs
func (s *CircuitBreakerServiceImpl) ListCircuitBreakers() []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return []string{}
	}

	return s.manager.ListCircuitBreakers()
}

// GetCircuitBreakerStats returns statistics for a specific circuit breaker
func (s *CircuitBreakerServiceImpl) GetCircuitBreakerStats(id string) map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.manager.GetCircuitBreakerStats(id)
}

// GetAllCircuitBreakerStats returns statistics for all circuit breakers
func (s *CircuitBreakerServiceImpl) GetAllCircuitBreakerStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.manager.GetAllCircuitBreakerStats()
}

// IsCircuitBreakerActive checks if a circuit breaker is active
func (s *CircuitBreakerServiceImpl) IsCircuitBreakerActive(id string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return false
	}

	return s.manager.IsCircuitBreakerActive(id)
}

// SetCircuitBreakerActive sets the active status of a circuit breaker
func (s *CircuitBreakerServiceImpl) SetCircuitBreakerActive(id string, active bool) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	if err := s.manager.SetCircuitBreakerActive(id, active); err != nil {
		return err
	}

	s.updatedAt = time.Now()
	return nil
}

// ExecuteWithCircuitBreaker executes an operation through a circuit breaker
func (s *CircuitBreakerServiceImpl) ExecuteWithCircuitBreaker(ctx context.Context, id string, operation func() (interface{}, error)) (interface{}, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	return s.executor.Execute(ctx, id, operation)
}

// ExecuteWithCircuitBreakerAsync executes an operation asynchronously through a circuit breaker
func (s *CircuitBreakerServiceImpl) ExecuteWithCircuitBreakerAsync(ctx context.Context, id string, operation func() (interface{}, error)) <-chan CircuitResult {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
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

	return s.executor.ExecuteAsync(ctx, id, operation)
}

// GetServiceStats returns service statistics
func (s *CircuitBreakerServiceImpl) GetServiceStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	managerStats := s.manager.GetManagerStats()
	executorStats := s.executor.GetExecutorStats()

	return map[string]interface{}{
		"service_name":           s.config.Name,
		"version":                s.config.Version,
		"active":                 s.active,
		"created_at":             s.createdAt,
		"updated_at":             s.updatedAt,
		"manager_stats":          managerStats,
		"executor_stats":         executorStats,
		"metadata":               s.config.Metadata,
	}
}

// GetHealthStatus returns the health status of the service
func (s *CircuitBreakerServiceImpl) GetHealthStatus() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"circuit_breaker_service": map[string]interface{}{
				"status": "healthy",
				"active": s.active,
			},
			"circuit_breaker_manager": map[string]interface{}{
				"status": "healthy",
				"active": s.manager != nil,
			},
			"circuit_breaker_executor": map[string]interface{}{
				"status": "healthy",
				"active": s.executor != nil,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !s.active {
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_service"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_service"].(map[string]interface{})["message"] = "Circuit breaker service is inactive"
	}

	if s.manager == nil {
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_manager"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_manager"].(map[string]interface{})["message"] = "Circuit breaker manager is not available"
	}

	if s.executor == nil {
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_executor"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["circuit_breaker_executor"].(map[string]interface{})["message"] = "Circuit breaker executor is not available"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (s *CircuitBreakerServiceImpl) Cleanup(ctx context.Context) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	// Cleanup manager
	if s.manager != nil {
		if err := s.manager.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	// Cleanup executor
	if s.executor != nil {
		if err := s.executor.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	s.updatedAt = time.Now()
	return nil
}

// validateCircuitBreakerConfig validates a circuit breaker configuration
func (s *CircuitBreakerServiceImpl) validateCircuitBreakerConfig(config CircuitConfig) error {
	if config.ID == "" {
		return fmt.Errorf("circuit breaker ID is required")
	}
	if config.Name == "" {
		return fmt.Errorf("circuit breaker name is required")
	}
	if config.Description == "" {
		return fmt.Errorf("circuit breaker description is required")
	}
	if config.FailureThreshold <= 0 {
		return fmt.Errorf("failure threshold must be positive")
	}
	if config.SuccessThreshold <= 0 {
		return fmt.Errorf("success threshold must be positive")
	}
	if config.Timeout <= 0 {
		return fmt.Errorf("timeout must be positive")
	}
	if config.ResetTimeout <= 0 {
		return fmt.Errorf("reset timeout must be positive")
	}
	if config.MaxRequests <= 0 {
		return fmt.Errorf("max requests must be positive")
	}
	if config.RequestVolumeThreshold <= 0 {
		return fmt.Errorf("request volume threshold must be positive")
	}
	if config.SleepWindow <= 0 {
		return fmt.Errorf("sleep window must be positive")
	}
	if config.ErrorThreshold < 0 || config.ErrorThreshold > 1 {
		return fmt.Errorf("error threshold must be between 0 and 1")
	}
	if config.SlowCallThreshold <= 0 {
		return fmt.Errorf("slow call threshold must be positive")
	}
	if config.SlowCallRatioThreshold < 0 || config.SlowCallRatioThreshold > 1 {
		return fmt.Errorf("slow call ratio threshold must be between 0 and 1")
	}

	return nil
}

// setDefaultValues sets default values for the configuration
func (s *CircuitBreakerServiceImpl) setDefaultValues(config *CircuitConfig) {
	if config.FailureThreshold == 0 {
		config.FailureThreshold = s.config.DefaultFailureThreshold
	}
	if config.SuccessThreshold == 0 {
		config.SuccessThreshold = s.config.DefaultSuccessThreshold
	}
	if config.Timeout == 0 {
		config.Timeout = s.config.DefaultTimeout
	}
	if config.ResetTimeout == 0 {
		config.ResetTimeout = s.config.DefaultResetTimeout
	}
	if config.MaxRequests == 0 {
		config.MaxRequests = 10
	}
	if config.RequestVolumeThreshold == 0 {
		config.RequestVolumeThreshold = 10
	}
	if config.SleepWindow == 0 {
		config.SleepWindow = 60 * time.Second
	}
	if config.ErrorThreshold == 0 {
		config.ErrorThreshold = 0.5
	}
	if config.SlowCallThreshold == 0 {
		config.SlowCallThreshold = 5 * time.Second
	}
	if config.SlowCallRatioThreshold == 0 {
		config.SlowCallRatioThreshold = 0.5
	}
	if config.Metadata == nil {
		config.Metadata = make(map[string]interface{})
	}
}
