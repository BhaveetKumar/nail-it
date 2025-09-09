package saga

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Common errors
var (
	ErrSagaNotFound         = errors.New("saga not found")
	ErrSagaAlreadyExists    = errors.New("saga already exists")
	ErrSagaInactive         = errors.New("saga is inactive")
	ErrStepNotFound         = errors.New("step not found")
	ErrStepAlreadyExists    = errors.New("step already exists")
	ErrStepInactive         = errors.New("step is inactive")
	ErrInvalidSagaType      = errors.New("invalid saga type")
	ErrInvalidStepType      = errors.New("invalid step type")
	ErrInvalidConfiguration = errors.New("invalid configuration")
	ErrValidationFailed     = errors.New("validation failed")
	ErrServiceInactive      = errors.New("service is inactive")
	ErrExecutionFailed      = errors.New("execution failed")
	ErrCompensationFailed   = errors.New("compensation failed")
	ErrRetryExhausted       = errors.New("retry exhausted")
	ErrTimeoutExceeded      = errors.New("timeout exceeded")
	ErrSagaNotExecutable    = errors.New("saga is not executable")
	ErrStepNotExecutable    = errors.New("step is not executable")
)

// BaseSaga represents a base saga implementation
type BaseSaga struct {
	ID          string                 `json:"id" yaml:"id"`
	Name        string                 `json:"name" yaml:"name"`
	Description string                 `json:"description" yaml:"description"`
	Status      SagaStatus             `json:"status" yaml:"status"`
	Steps       []SagaStep             `json:"steps" yaml:"steps"`
	CurrentStep int                    `json:"current_step" yaml:"current_step"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
}

// GetID returns the saga ID
func (s *BaseSaga) GetID() string {
	return s.ID
}

// GetName returns the saga name
func (s *BaseSaga) GetName() string {
	return s.Name
}

// GetDescription returns the saga description
func (s *BaseSaga) GetDescription() string {
	return s.Description
}

// GetStatus returns the saga status
func (s *BaseSaga) GetStatus() SagaStatus {
	return s.Status
}

// GetSteps returns the saga steps
func (s *BaseSaga) GetSteps() []SagaStep {
	return s.Steps
}

// GetCurrentStep returns the current step index
func (s *BaseSaga) GetCurrentStep() int {
	return s.CurrentStep
}

// GetCreatedAt returns the creation time
func (s *BaseSaga) GetCreatedAt() time.Time {
	return s.CreatedAt
}

// GetUpdatedAt returns the last update time
func (s *BaseSaga) GetUpdatedAt() time.Time {
	return s.UpdatedAt
}

// GetMetadata returns the saga metadata
func (s *BaseSaga) GetMetadata() map[string]interface{} {
	return s.Metadata
}

// SetMetadata sets the saga metadata
func (s *BaseSaga) SetMetadata(key string, value interface{}) {
	if s.Metadata == nil {
		s.Metadata = make(map[string]interface{})
	}
	s.Metadata[key] = value
	s.UpdatedAt = time.Now()
}

// IsActive returns whether the saga is active
func (s *BaseSaga) IsActive() bool {
	return s.Active
}

// SetActive sets the active status
func (s *BaseSaga) SetActive(active bool) {
	s.Active = active
	s.UpdatedAt = time.Now()
}

// AddStep adds a step to the saga
func (s *BaseSaga) AddStep(step SagaStep) error {
	if !s.Active {
		return ErrSagaInactive
	}

	// Check if step already exists
	for _, existingStep := range s.Steps {
		if existingStep.GetID() == step.GetID() {
			return ErrStepAlreadyExists
		}
	}

	s.Steps = append(s.Steps, step)
	s.UpdatedAt = time.Now()
	return nil
}

// RemoveStep removes a step from the saga
func (s *BaseSaga) RemoveStep(stepID string) error {
	if !s.Active {
		return ErrSagaInactive
	}

	for i, step := range s.Steps {
		if step.GetID() == stepID {
			s.Steps = append(s.Steps[:i], s.Steps[i+1:]...)
			s.UpdatedAt = time.Now()
			return nil
		}
	}

	return ErrStepNotFound
}

// GetStep retrieves a step by ID
func (s *BaseSaga) GetStep(stepID string) (SagaStep, error) {
	for _, step := range s.Steps {
		if step.GetID() == stepID {
			return step, nil
		}
	}
	return nil, ErrStepNotFound
}

// GetStepByIndex retrieves a step by index
func (s *BaseSaga) GetStepByIndex(index int) (SagaStep, error) {
	if index < 0 || index >= len(s.Steps) {
		return nil, ErrStepNotFound
	}
	return s.Steps[index], nil
}

// GetStats returns saga statistics
func (s *BaseSaga) GetStats() map[string]interface{} {
	completedSteps := 0
	failedSteps := 0
	retryCount := 0

	for _, step := range s.Steps {
		switch step.GetStatus() {
		case StepStatusCompleted:
			completedSteps++
		case StepStatusFailed:
			failedSteps++
		}
		retryCount += step.GetRetryCount()
	}

	return map[string]interface{}{
		"id":              s.ID,
		"name":            s.Name,
		"description":     s.Description,
		"status":          s.Status,
		"current_step":    s.CurrentStep,
		"steps_count":     len(s.Steps),
		"completed_steps": completedSteps,
		"failed_steps":    failedSteps,
		"retry_count":     retryCount,
		"active":          s.Active,
		"created_at":      s.CreatedAt,
		"updated_at":      s.UpdatedAt,
		"metadata":        s.Metadata,
	}
}

// Validate validates the saga
func (s *BaseSaga) Validate() error {
	if s.ID == "" {
		return fmt.Errorf("saga ID is required")
	}
	if s.Name == "" {
		return fmt.Errorf("saga name is required")
	}
	if s.Description == "" {
		return fmt.Errorf("saga description is required")
	}
	if len(s.Steps) == 0 {
		return fmt.Errorf("saga must have at least one step")
	}

	// Validate steps
	for i, step := range s.Steps {
		if err := step.Validate(); err != nil {
			return fmt.Errorf("step %d validation failed: %w", i, err)
		}
	}

	return nil
}

// Reset resets the saga
func (s *BaseSaga) Reset() error {
	if !s.Active {
		return ErrSagaInactive
	}

	s.Status = SagaStatusPending
	s.CurrentStep = 0

	// Reset all steps
	for _, step := range s.Steps {
		if err := step.Reset(); err != nil {
			return err
		}
	}

	s.UpdatedAt = time.Now()
	return nil
}

// Cleanup performs cleanup operations
func (s *BaseSaga) Cleanup(ctx context.Context) error {
	if !s.Active {
		return ErrSagaInactive
	}

	// Cleanup all steps
	for _, step := range s.Steps {
		if err := step.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	s.UpdatedAt = time.Now()
	return nil
}

// ConcreteSaga represents a concrete implementation of Saga
type ConcreteSaga struct {
	BaseSaga
	Version string `json:"version" yaml:"version"`
}

// NewConcreteSaga creates a new ConcreteSaga
func NewConcreteSaga(config SagaConfig) *ConcreteSaga {
	steps := make([]SagaStep, 0, len(config.Steps))
	for _, stepConfig := range config.Steps {
		step := NewConcreteSagaStep(stepConfig)
		steps = append(steps, step)
	}

	return &ConcreteSaga{
		BaseSaga: BaseSaga{
			ID:          config.ID,
			Name:        config.Name,
			Description: config.Description,
			Status:      SagaStatusPending,
			Steps:       steps,
			CurrentStep: 0,
			Metadata:    config.Metadata,
			Active:      config.Active,
			CreatedAt:   config.CreatedAt,
			UpdatedAt:   config.UpdatedAt,
		},
		Version: config.Version,
	}
}

// Execute executes the saga
func (s *ConcreteSaga) Execute(ctx context.Context) error {
	if !s.Active {
		return ErrSagaInactive
	}

	if s.Status != SagaStatusPending && s.Status != SagaStatusPaused {
		return ErrSagaNotExecutable
	}

	s.Status = SagaStatusRunning
	s.UpdatedAt = time.Now()

	// Execute steps in order
	for i := s.CurrentStep; i < len(s.Steps); i++ {
		step := s.Steps[i]
		s.CurrentStep = i

		if err := step.Execute(ctx); err != nil {
			s.Status = SagaStatusFailed
			s.UpdatedAt = time.Now()
			return fmt.Errorf("step %d execution failed: %w", i, err)
		}
	}

	s.Status = SagaStatusCompleted
	s.UpdatedAt = time.Now()
	return nil
}

// Compensate compensates the saga
func (s *ConcreteSaga) Compensate(ctx context.Context) error {
	if !s.Active {
		return ErrSagaInactive
	}

	if s.Status != SagaStatusFailed && s.Status != SagaStatusRunning {
		return fmt.Errorf("saga cannot be compensated in status: %s", s.Status)
	}

	s.Status = SagaStatusRunning
	s.UpdatedAt = time.Now()

	// Compensate steps in reverse order
	for i := s.CurrentStep; i >= 0; i-- {
		step := s.Steps[i]

		if err := step.Compensate(ctx); err != nil {
			s.Status = SagaStatusFailed
			s.UpdatedAt = time.Now()
			return fmt.Errorf("step %d compensation failed: %w", i, err)
		}
	}

	s.Status = SagaStatusCompensated
	s.UpdatedAt = time.Now()
	return nil
}

// GetStats returns saga statistics
func (s *ConcreteSaga) GetStats() map[string]interface{} {
	stats := s.BaseSaga.GetStats()
	stats["version"] = s.Version
	return stats
}

// BaseSagaStep represents a base saga step implementation
type BaseSagaStep struct {
	ID           string                 `json:"id" yaml:"id"`
	Name         string                 `json:"name" yaml:"name"`
	Description  string                 `json:"description" yaml:"description"`
	Order        int                    `json:"order" yaml:"order"`
	Action       string                 `json:"action" yaml:"action"`
	Compensation string                 `json:"compensation" yaml:"compensation"`
	Status       StepStatus             `json:"status" yaml:"status"`
	RetryCount   int                    `json:"retry_count" yaml:"retry_count"`
	MaxRetries   int                    `json:"max_retries" yaml:"max_retries"`
	Timeout      time.Duration          `json:"timeout" yaml:"timeout"`
	Metadata     map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active       bool                   `json:"active" yaml:"active"`
	CreatedAt    time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at" yaml:"updated_at"`
}

// GetID returns the step ID
func (s *BaseSagaStep) GetID() string {
	return s.ID
}

// GetName returns the step name
func (s *BaseSagaStep) GetName() string {
	return s.Name
}

// GetDescription returns the step description
func (s *BaseSagaStep) GetDescription() string {
	return s.Description
}

// GetOrder returns the step order
func (s *BaseSagaStep) GetOrder() int {
	return s.Order
}

// GetAction returns the step action
func (s *BaseSagaStep) GetAction() string {
	return s.Action
}

// GetCompensation returns the step compensation
func (s *BaseSagaStep) GetCompensation() string {
	return s.Compensation
}

// GetStatus returns the step status
func (s *BaseSagaStep) GetStatus() StepStatus {
	return s.Status
}

// GetRetryCount returns the retry count
func (s *BaseSagaStep) GetRetryCount() int {
	return s.RetryCount
}

// GetMaxRetries returns the max retries
func (s *BaseSagaStep) GetMaxRetries() int {
	return s.MaxRetries
}

// GetTimeout returns the timeout
func (s *BaseSagaStep) GetTimeout() time.Duration {
	return s.Timeout
}

// GetCreatedAt returns the creation time
func (s *BaseSagaStep) GetCreatedAt() time.Time {
	return s.CreatedAt
}

// GetUpdatedAt returns the last update time
func (s *BaseSagaStep) GetUpdatedAt() time.Time {
	return s.UpdatedAt
}

// GetMetadata returns the step metadata
func (s *BaseSagaStep) GetMetadata() map[string]interface{} {
	return s.Metadata
}

// SetMetadata sets the step metadata
func (s *BaseSagaStep) SetMetadata(key string, value interface{}) {
	if s.Metadata == nil {
		s.Metadata = make(map[string]interface{})
	}
	s.Metadata[key] = value
	s.UpdatedAt = time.Now()
}

// IsActive returns whether the step is active
func (s *BaseSagaStep) IsActive() bool {
	return s.Active
}

// SetActive sets the active status
func (s *BaseSagaStep) SetActive(active bool) {
	s.Active = active
	s.UpdatedAt = time.Now()
}

// GetStats returns step statistics
func (s *BaseSagaStep) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"id":           s.ID,
		"name":         s.Name,
		"description":  s.Description,
		"order":        s.Order,
		"action":       s.Action,
		"compensation": s.Compensation,
		"status":       s.Status,
		"retry_count":  s.RetryCount,
		"max_retries":  s.MaxRetries,
		"timeout":      s.Timeout,
		"active":       s.Active,
		"created_at":   s.CreatedAt,
		"updated_at":   s.UpdatedAt,
		"metadata":     s.Metadata,
	}
}

// Validate validates the step
func (s *BaseSagaStep) Validate() error {
	if s.ID == "" {
		return fmt.Errorf("step ID is required")
	}
	if s.Name == "" {
		return fmt.Errorf("step name is required")
	}
	if s.Description == "" {
		return fmt.Errorf("step description is required")
	}
	if s.Action == "" {
		return fmt.Errorf("step action is required")
	}
	if s.Compensation == "" {
		return fmt.Errorf("step compensation is required")
	}
	if s.MaxRetries < 0 {
		return fmt.Errorf("max retries cannot be negative")
	}
	if s.Timeout < 0 {
		return fmt.Errorf("timeout cannot be negative")
	}
	return nil
}

// Reset resets the step
func (s *BaseSagaStep) Reset() error {
	if !s.Active {
		return ErrStepInactive
	}

	s.Status = StepStatusPending
	s.RetryCount = 0
	s.UpdatedAt = time.Now()
	return nil
}

// Cleanup performs cleanup operations
func (s *BaseSagaStep) Cleanup(ctx context.Context) error {
	if !s.Active {
		return ErrStepInactive
	}

	s.UpdatedAt = time.Now()
	return nil
}

// ConcreteSagaStep represents a concrete implementation of SagaStep
type ConcreteSagaStep struct {
	BaseSagaStep
	ExecuteFunc      func(ctx context.Context) error `json:"-" yaml:"-"`
	CompensateFunc   func(ctx context.Context) error `json:"-" yaml:"-"`
}

// NewConcreteSagaStep creates a new ConcreteSagaStep
func NewConcreteSagaStep(config SagaStepConfig) *ConcreteSagaStep {
	return &ConcreteSagaStep{
		BaseSagaStep: BaseSagaStep{
			ID:           config.ID,
			Name:         config.Name,
			Description:  config.Description,
			Order:        config.Order,
			Action:       config.Action,
			Compensation: config.Compensation,
			Status:       StepStatusPending,
			RetryCount:   0,
			MaxRetries:   config.MaxRetries,
			Timeout:      config.Timeout,
			Metadata:     config.Metadata,
			Active:       config.Active,
			CreatedAt:    config.CreatedAt,
			UpdatedAt:    config.UpdatedAt,
		},
		ExecuteFunc:    nil,
		CompensateFunc: nil,
	}
}

// Execute executes the step
func (s *ConcreteSagaStep) Execute(ctx context.Context) error {
	if !s.Active {
		return ErrStepInactive
	}

	if s.Status != StepStatusPending && s.Status != StepStatusFailed {
		return ErrStepNotExecutable
	}

	s.Status = StepStatusRunning
	s.UpdatedAt = time.Now()

	// Create timeout context
	timeoutCtx, cancel := context.WithTimeout(ctx, s.Timeout)
	defer cancel()

	// Execute the step
	if s.ExecuteFunc != nil {
		if err := s.ExecuteFunc(timeoutCtx); err != nil {
			s.Status = StepStatusFailed
			s.RetryCount++
			s.UpdatedAt = time.Now()

			// Check if we should retry
			if s.RetryCount <= s.MaxRetries {
				return fmt.Errorf("step execution failed (retry %d/%d): %w", s.RetryCount, s.MaxRetries, err)
			}

			return fmt.Errorf("step execution failed after %d retries: %w", s.MaxRetries, err)
		}
	}

	s.Status = StepStatusCompleted
	s.UpdatedAt = time.Now()
	return nil
}

// Compensate compensates the step
func (s *ConcreteSagaStep) Compensate(ctx context.Context) error {
	if !s.Active {
		return ErrStepInactive
	}

	if s.Status != StepStatusCompleted && s.Status != StepStatusFailed {
		return fmt.Errorf("step cannot be compensated in status: %s", s.Status)
	}

	s.Status = StepStatusRunning
	s.UpdatedAt = time.Now()

	// Create timeout context
	timeoutCtx, cancel := context.WithTimeout(ctx, s.Timeout)
	defer cancel()

	// Compensate the step
	if s.CompensateFunc != nil {
		if err := s.CompensateFunc(timeoutCtx); err != nil {
			s.Status = StepStatusFailed
			s.UpdatedAt = time.Now()
			return fmt.Errorf("step compensation failed: %w", err)
		}
	}

	s.Status = StepStatusCompensated
	s.UpdatedAt = time.Now()
	return nil
}

// SetExecuteFunc sets the execute function
func (s *ConcreteSagaStep) SetExecuteFunc(fn func(ctx context.Context) error) {
	s.ExecuteFunc = fn
}

// SetCompensateFunc sets the compensate function
func (s *ConcreteSagaStep) SetCompensateFunc(fn func(ctx context.Context) error) {
	s.CompensateFunc = fn
}
