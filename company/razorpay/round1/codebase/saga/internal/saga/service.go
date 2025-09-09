package saga

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// SagaServiceImpl implements the SagaService interface
type SagaServiceImpl struct {
	config       *ServiceConfig
	manager      SagaManager
	executor     SagaExecutor
	stepExecutor StepExecutor
	createdAt    time.Time
	updatedAt    time.Time
	active       bool
	mutex        sync.RWMutex
}

// NewSagaService creates a new saga service
func NewSagaService(config *ServiceConfig) *SagaServiceImpl {
	manager := NewSagaManager()
	executor := NewSagaExecutor(config, manager)
	stepExecutor := NewStepExecutor(config)

	return &SagaServiceImpl{
		config:       config,
		manager:      manager,
		executor:     executor,
		stepExecutor: stepExecutor,
		createdAt:    time.Now(),
		updatedAt:    time.Now(),
		active:       true,
	}
}

// CreateSaga creates a new saga
func (s *SagaServiceImpl) CreateSaga(config SagaConfig) (Saga, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	// Validate configuration
	if err := s.validateSagaConfig(config); err != nil {
		return nil, err
	}

	// Create saga
	saga, err := s.manager.CreateSaga(config)
	if err != nil {
		return nil, err
	}

	s.updatedAt = time.Now()
	return saga, nil
}

// DestroySaga destroys a saga
func (s *SagaServiceImpl) DestroySaga(sagaID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	if err := s.manager.DestroySaga(sagaID); err != nil {
		return err
	}

	s.updatedAt = time.Now()
	return nil
}

// GetSaga retrieves a saga by ID
func (s *SagaServiceImpl) GetSaga(sagaID string) (Saga, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	return s.manager.GetSaga(sagaID)
}

// ListSagas returns a list of all saga IDs
func (s *SagaServiceImpl) ListSagas() []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return []string{}
	}

	return s.manager.ListSagas()
}

// GetSagaStats returns statistics for a specific saga
func (s *SagaServiceImpl) GetSagaStats(sagaID string) map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.manager.GetSagaStats(sagaID)
}

// GetAllSagaStats returns statistics for all sagas
func (s *SagaServiceImpl) GetAllSagaStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.manager.GetAllSagaStats()
}

// IsSagaActive checks if a saga is active
func (s *SagaServiceImpl) IsSagaActive(sagaID string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return false
	}

	return s.manager.IsSagaActive(sagaID)
}

// SetSagaActive sets the active status of a saga
func (s *SagaServiceImpl) SetSagaActive(sagaID string, active bool) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	if err := s.manager.SetSagaActive(sagaID, active); err != nil {
		return err
	}

	s.updatedAt = time.Now()
	return nil
}

// ExecuteSaga executes a saga
func (s *SagaServiceImpl) ExecuteSaga(ctx context.Context, sagaID string) error {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return ErrServiceInactive
	}

	saga, err := s.manager.GetSaga(sagaID)
	if err != nil {
		return err
	}

	return s.executor.ExecuteSaga(ctx, saga)
}

// CompensateSaga compensates a saga
func (s *SagaServiceImpl) CompensateSaga(ctx context.Context, sagaID string) error {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return ErrServiceInactive
	}

	saga, err := s.manager.GetSaga(sagaID)
	if err != nil {
		return err
	}

	return s.executor.CompensateSaga(ctx, saga)
}

// GetSagaStatus returns the status of a saga
func (s *SagaServiceImpl) GetSagaStatus(ctx context.Context, sagaID string) (SagaStatus, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return "", ErrServiceInactive
	}

	return s.executor.GetSagaStatus(ctx, sagaID)
}

// GetServiceStats returns service statistics
func (s *SagaServiceImpl) GetServiceStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	managerStats := s.manager.GetManagerStats()
	executorStats := s.executor.GetExecutorStats()
	stepExecutorStats := s.stepExecutor.GetExecutorStats()

	return map[string]interface{}{
		"service_name":        s.config.Name,
		"version":             s.config.Version,
		"active":              s.active,
		"created_at":          s.createdAt,
		"updated_at":          s.updatedAt,
		"manager_stats":       managerStats,
		"executor_stats":      executorStats,
		"step_executor_stats": stepExecutorStats,
		"metadata":            s.config.Metadata,
	}
}

// GetHealthStatus returns the health status of the service
func (s *SagaServiceImpl) GetHealthStatus() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"saga_service": map[string]interface{}{
				"status": "healthy",
				"active": s.active,
			},
			"saga_manager": map[string]interface{}{
				"status": "healthy",
				"active": s.manager != nil,
			},
			"saga_executor": map[string]interface{}{
				"status": "healthy",
				"active": s.executor != nil,
			},
			"step_executor": map[string]interface{}{
				"status": "healthy",
				"active": s.stepExecutor != nil,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !s.active {
		healthStatus["checks"].(map[string]interface{})["saga_service"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["saga_service"].(map[string]interface{})["message"] = "Saga service is inactive"
	}

	if s.manager == nil {
		healthStatus["checks"].(map[string]interface{})["saga_manager"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["saga_manager"].(map[string]interface{})["message"] = "Saga manager is not available"
	}

	if s.executor == nil {
		healthStatus["checks"].(map[string]interface{})["saga_executor"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["saga_executor"].(map[string]interface{})["message"] = "Saga executor is not available"
	}

	if s.stepExecutor == nil {
		healthStatus["checks"].(map[string]interface{})["step_executor"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["step_executor"].(map[string]interface{})["message"] = "Step executor is not available"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (s *SagaServiceImpl) Cleanup(ctx context.Context) error {
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

	// Cleanup step executor
	if s.stepExecutor != nil {
		if err := s.stepExecutor.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	s.updatedAt = time.Now()
	return nil
}

// validateSagaConfig validates a saga configuration
func (s *SagaServiceImpl) validateSagaConfig(config SagaConfig) error {
	if config.ID == "" {
		return fmt.Errorf("saga ID is required")
	}
	if config.Name == "" {
		return fmt.Errorf("saga name is required")
	}
	if config.Description == "" {
		return fmt.Errorf("saga description is required")
	}
	if len(config.Steps) == 0 {
		return fmt.Errorf("saga must have at least one step")
	}

	// Validate steps
	for i, stepConfig := range config.Steps {
		if err := s.validateStepConfig(stepConfig); err != nil {
			return fmt.Errorf("step %d validation failed: %w", i, err)
		}
	}

	return nil
}

// validateStepConfig validates a step configuration
func (s *SagaServiceImpl) validateStepConfig(config SagaStepConfig) error {
	if config.ID == "" {
		return fmt.Errorf("step ID is required")
	}
	if config.Name == "" {
		return fmt.Errorf("step name is required")
	}
	if config.Description == "" {
		return fmt.Errorf("step description is required")
	}
	if config.Action == "" {
		return fmt.Errorf("step action is required")
	}
	if config.Compensation == "" {
		return fmt.Errorf("step compensation is required")
	}
	if config.MaxRetries < 0 {
		return fmt.Errorf("max retries cannot be negative")
	}
	if config.Timeout < 0 {
		return fmt.Errorf("timeout cannot be negative")
	}

	return nil
}

// SagaService defines the interface for saga service operations
type SagaService interface {
	CreateSaga(config SagaConfig) (Saga, error)
	DestroySaga(sagaID string) error
	GetSaga(sagaID string) (Saga, error)
	ListSagas() []string
	GetSagaStats(sagaID string) map[string]interface{}
	GetAllSagaStats() map[string]interface{}
	IsSagaActive(sagaID string) bool
	SetSagaActive(sagaID string, active bool) error
	ExecuteSaga(ctx context.Context, sagaID string) error
	CompensateSaga(ctx context.Context, sagaID string) error
	GetSagaStatus(ctx context.Context, sagaID string) (SagaStatus, error)
	GetServiceStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}
