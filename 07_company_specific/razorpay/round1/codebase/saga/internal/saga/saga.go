package saga

import (
	"context"
	"sync"
	"time"
)

// SagaManagerImpl implements the SagaManager interface
type SagaManagerImpl struct {
	sagas     map[string]Saga
	mutex     sync.RWMutex
	createdAt time.Time
	updatedAt time.Time
	active    bool
}

// NewSagaManager creates a new saga manager
func NewSagaManager() *SagaManagerImpl {
	return &SagaManagerImpl{
		sagas:     make(map[string]Saga),
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// CreateSaga creates a new saga
func (m *SagaManagerImpl) CreateSaga(config SagaConfig) (Saga, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.active {
		return nil, ErrServiceInactive
	}

	if _, exists := m.sagas[config.ID]; exists {
		return nil, ErrSagaAlreadyExists
	}

	saga := NewConcreteSaga(config)
	m.sagas[config.ID] = saga
	m.updatedAt = time.Now()

	return saga, nil
}

// DestroySaga destroys a saga
func (m *SagaManagerImpl) DestroySaga(sagaID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.active {
		return ErrServiceInactive
	}

	saga, exists := m.sagas[sagaID]
	if !exists {
		return ErrSagaNotFound
	}

	// Cleanup saga
	if err := saga.Cleanup(context.Background()); err != nil {
		// Log error but continue
	}

	delete(m.sagas, sagaID)
	m.updatedAt = time.Now()

	return nil
}

// GetSaga retrieves a saga by ID
func (m *SagaManagerImpl) GetSaga(sagaID string) (Saga, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	if !m.active {
		return nil, ErrServiceInactive
	}

	saga, exists := m.sagas[sagaID]
	if !exists {
		return nil, ErrSagaNotFound
	}

	return saga, nil
}

// ListSagas returns a list of all saga IDs
func (m *SagaManagerImpl) ListSagas() []string {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	sagaIDs := make([]string, 0, len(m.sagas))
	for sagaID := range m.sagas {
		sagaIDs = append(sagaIDs, sagaID)
	}

	return sagaIDs
}

// GetSagaStats returns statistics for a specific saga
func (m *SagaManagerImpl) GetSagaStats(sagaID string) map[string]interface{} {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	saga, exists := m.sagas[sagaID]
	if !exists {
		return map[string]interface{}{
			"error": "saga not found",
		}
	}

	return saga.GetStats()
}

// GetAllSagaStats returns statistics for all sagas
func (m *SagaManagerImpl) GetAllSagaStats() map[string]interface{} {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	stats := make(map[string]interface{})
	for sagaID, saga := range m.sagas {
		stats[sagaID] = saga.GetStats()
	}

	return stats
}

// IsSagaActive checks if a saga is active
func (m *SagaManagerImpl) IsSagaActive(sagaID string) bool {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	saga, exists := m.sagas[sagaID]
	if !exists {
		return false
	}

	return saga.IsActive()
}

// SetSagaActive sets the active status of a saga
func (m *SagaManagerImpl) SetSagaActive(sagaID string, active bool) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.active {
		return ErrServiceInactive
	}

	saga, exists := m.sagas[sagaID]
	if !exists {
		return ErrSagaNotFound
	}

	saga.SetActive(active)
	m.updatedAt = time.Now()

	return nil
}

// GetManagerStats returns manager statistics
func (m *SagaManagerImpl) GetManagerStats() map[string]interface{} {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	return map[string]interface{}{
		"active":      m.active,
		"created_at":  m.createdAt,
		"updated_at":  m.updatedAt,
		"sagas_count": len(m.sagas),
		"sagas":       m.ListSagas(),
	}
}

// GetHealthStatus returns the health status of the manager
func (m *SagaManagerImpl) GetHealthStatus() map[string]interface{} {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"saga_manager": map[string]interface{}{
				"status": "healthy",
				"active": m.active,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !m.active {
		healthStatus["checks"].(map[string]interface{})["saga_manager"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["saga_manager"].(map[string]interface{})["message"] = "Saga manager is inactive"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (m *SagaManagerImpl) Cleanup(ctx context.Context) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.active {
		return ErrServiceInactive
	}

	// Cleanup all sagas
	for _, saga := range m.sagas {
		if err := saga.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	m.updatedAt = time.Now()
	return nil
}

// SagaExecutorImpl implements the SagaExecutor interface
type SagaExecutorImpl struct {
	config    *ServiceConfig
	manager   SagaManager
	createdAt time.Time
	updatedAt time.Time
	active    bool
	mutex     sync.RWMutex
}

// NewSagaExecutor creates a new saga executor
func NewSagaExecutor(config *ServiceConfig, manager SagaManager) *SagaExecutorImpl {
	return &SagaExecutorImpl{
		config:    config,
		manager:   manager,
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// ExecuteSaga executes a saga
func (e *SagaExecutorImpl) ExecuteSaga(ctx context.Context, saga Saga) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if !e.active {
		return ErrServiceInactive
	}

	if !saga.IsActive() {
		return ErrSagaInactive
	}

	// Execute the saga
	if err := saga.Execute(ctx); err != nil {
		return err
	}

	e.updatedAt = time.Now()
	return nil
}

// CompensateSaga compensates a saga
func (e *SagaExecutorImpl) CompensateSaga(ctx context.Context, saga Saga) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if !e.active {
		return ErrServiceInactive
	}

	if !saga.IsActive() {
		return ErrSagaInactive
	}

	// Compensate the saga
	if err := saga.Compensate(ctx); err != nil {
		return err
	}

	e.updatedAt = time.Now()
	return nil
}

// GetSagaStatus returns the status of a saga
func (e *SagaExecutorImpl) GetSagaStatus(ctx context.Context, sagaID string) (SagaStatus, error) {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	if !e.active {
		return "", ErrServiceInactive
	}

	saga, err := e.manager.GetSaga(sagaID)
	if err != nil {
		return "", err
	}

	return saga.GetStatus(), nil
}

// GetSagaStats returns statistics for a saga
func (e *SagaExecutorImpl) GetSagaStats(ctx context.Context, sagaID string) map[string]interface{} {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	if !e.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return e.manager.GetSagaStats(sagaID)
}

// IsSagaActive checks if a saga is active
func (e *SagaExecutorImpl) IsSagaActive(ctx context.Context, sagaID string) bool {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	if !e.active {
		return false
	}

	return e.manager.IsSagaActive(sagaID)
}

// SetSagaActive sets the active status of a saga
func (e *SagaExecutorImpl) SetSagaActive(ctx context.Context, sagaID string, active bool) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if !e.active {
		return ErrServiceInactive
	}

	if err := e.manager.SetSagaActive(sagaID, active); err != nil {
		return err
	}

	e.updatedAt = time.Now()
	return nil
}

// GetExecutorStats returns executor statistics
func (e *SagaExecutorImpl) GetExecutorStats() map[string]interface{} {
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
func (e *SagaExecutorImpl) GetHealthStatus() map[string]interface{} {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"saga_executor": map[string]interface{}{
				"status": "healthy",
				"active": e.active,
			},
			"saga_manager": map[string]interface{}{
				"status": "healthy",
				"active": e.manager != nil,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !e.active {
		healthStatus["checks"].(map[string]interface{})["saga_executor"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["saga_executor"].(map[string]interface{})["message"] = "Saga executor is inactive"
	}

	if e.manager == nil {
		healthStatus["checks"].(map[string]interface{})["saga_manager"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["saga_manager"].(map[string]interface{})["message"] = "Saga manager is not available"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (e *SagaExecutorImpl) Cleanup(ctx context.Context) error {
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

// StepExecutorImpl implements the StepExecutor interface
type StepExecutorImpl struct {
	config    *ServiceConfig
	createdAt time.Time
	updatedAt time.Time
	active    bool
	mutex     sync.RWMutex
}

// NewStepExecutor creates a new step executor
func NewStepExecutor(config *ServiceConfig) *StepExecutorImpl {
	return &StepExecutorImpl{
		config:    config,
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// ExecuteStep executes a saga step
func (e *StepExecutorImpl) ExecuteStep(ctx context.Context, step SagaStep) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if !e.active {
		return ErrServiceInactive
	}

	if !step.IsActive() {
		return ErrStepInactive
	}

	// Execute the step
	if err := step.Execute(ctx); err != nil {
		return err
	}

	e.updatedAt = time.Now()
	return nil
}

// CompensateStep compensates a saga step
func (e *StepExecutorImpl) CompensateStep(ctx context.Context, step SagaStep) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if !e.active {
		return ErrServiceInactive
	}

	if !step.IsActive() {
		return ErrStepInactive
	}

	// Compensate the step
	if err := step.Compensate(ctx); err != nil {
		return err
	}

	e.updatedAt = time.Now()
	return nil
}

// GetStepStatus returns the status of a step
func (e *StepExecutorImpl) GetStepStatus(ctx context.Context, stepID string) (StepStatus, error) {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	if !e.active {
		return "", ErrServiceInactive
	}

	// This would typically require a step registry
	// For now, return an error
	return "", ErrStepNotFound
}

// GetStepStats returns statistics for a step
func (e *StepExecutorImpl) GetStepStats(ctx context.Context, stepID string) map[string]interface{} {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	if !e.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	// This would typically require a step registry
	// For now, return an error
	return map[string]interface{}{
		"error": "step not found",
	}
}

// IsStepActive checks if a step is active
func (e *StepExecutorImpl) IsStepActive(ctx context.Context, stepID string) bool {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	if !e.active {
		return false
	}

	// This would typically require a step registry
	// For now, return false
	return false
}

// SetStepActive sets the active status of a step
func (e *StepExecutorImpl) SetStepActive(ctx context.Context, stepID string, active bool) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if !e.active {
		return ErrServiceInactive
	}

	// This would typically require a step registry
	// For now, return an error
	return ErrStepNotFound
}

// GetExecutorStats returns executor statistics
func (e *StepExecutorImpl) GetExecutorStats() map[string]interface{} {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	return map[string]interface{}{
		"active":     e.active,
		"created_at": e.createdAt,
		"updated_at": e.updatedAt,
	}
}

// GetHealthStatus returns the health status of the executor
func (e *StepExecutorImpl) GetHealthStatus() map[string]interface{} {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"step_executor": map[string]interface{}{
				"status": "healthy",
				"active": e.active,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !e.active {
		healthStatus["checks"].(map[string]interface{})["step_executor"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["step_executor"].(map[string]interface{})["message"] = "Step executor is inactive"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (e *StepExecutorImpl) Cleanup(ctx context.Context) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if !e.active {
		return ErrServiceInactive
	}

	e.updatedAt = time.Now()
	return nil
}
