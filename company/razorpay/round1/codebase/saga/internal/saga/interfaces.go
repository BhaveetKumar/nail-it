package saga

import (
	"context"
	"time"
)

// Saga defines the interface for a saga
type Saga interface {
	GetID() string
	GetName() string
	GetDescription() string
	GetStatus() SagaStatus
	GetSteps() []SagaStep
	GetCurrentStep() int
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	IsActive() bool
	SetActive(active bool)
	Execute(ctx context.Context) error
	Compensate(ctx context.Context) error
	AddStep(step SagaStep) error
	RemoveStep(stepID string) error
	GetStep(stepID string) (SagaStep, error)
	GetStepByIndex(index int) (SagaStep, error)
	GetStats() map[string]interface{}
	Validate() error
	Reset() error
	Cleanup(ctx context.Context) error
}

// SagaStep defines the interface for a saga step
type SagaStep interface {
	GetID() string
	GetName() string
	GetDescription() string
	GetOrder() int
	GetAction() string
	GetCompensation() string
	GetStatus() StepStatus
	GetRetryCount() int
	GetMaxRetries() int
	GetTimeout() time.Duration
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	IsActive() bool
	SetActive(active bool)
	Execute(ctx context.Context) error
	Compensate(ctx context.Context) error
	Validate() error
	GetStats() map[string]interface{}
	Reset() error
}

// SagaExecutor defines the interface for executing sagas
type SagaExecutor interface {
	ExecuteSaga(ctx context.Context, saga Saga) error
	CompensateSaga(ctx context.Context, saga Saga) error
	GetSagaStatus(ctx context.Context, sagaID string) (SagaStatus, error)
	GetSagaStats(ctx context.Context, sagaID string) map[string]interface{}
	IsSagaActive(ctx context.Context, sagaID string) bool
	SetSagaActive(ctx context.Context, sagaID string, active bool) error
	GetExecutorStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// SagaManager manages sagas
type SagaManager interface {
	CreateSaga(config SagaConfig) (Saga, error)
	DestroySaga(sagaID string) error
	GetSaga(sagaID string) (Saga, error)
	ListSagas() []string
	GetSagaStats(sagaID string) map[string]interface{}
	GetAllSagaStats() map[string]interface{}
	IsSagaActive(sagaID string) bool
	SetSagaActive(sagaID string, active bool) error
	GetManagerStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// StepExecutor defines the interface for executing saga steps
type StepExecutor interface {
	ExecuteStep(ctx context.Context, step SagaStep) error
	CompensateStep(ctx context.Context, step SagaStep) error
	GetStepStatus(ctx context.Context, stepID string) (StepStatus, error)
	GetStepStats(ctx context.Context, stepID string) map[string]interface{}
	IsStepActive(ctx context.Context, stepID string) bool
	SetStepActive(ctx context.Context, stepID string, active bool) error
	GetExecutorStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// SagaStatus represents the status of a saga
type SagaStatus string

const (
	SagaStatusPending    SagaStatus = "pending"
	SagaStatusRunning    SagaStatus = "running"
	SagaStatusCompleted  SagaStatus = "completed"
	SagaStatusFailed     SagaStatus = "failed"
	SagaStatusCompensated SagaStatus = "compensated"
	SagaStatusCancelled  SagaStatus = "cancelled"
	SagaStatusPaused     SagaStatus = "paused"
)

// StepStatus represents the status of a saga step
type StepStatus string

const (
	StepStatusPending    StepStatus = "pending"
	StepStatusRunning    StepStatus = "running"
	StepStatusCompleted  StepStatus = "completed"
	StepStatusFailed     StepStatus = "failed"
	StepStatusCompensated StepStatus = "compensated"
	StepStatusSkipped    StepStatus = "skipped"
	StepStatusCancelled  StepStatus = "cancelled"
)

// SagaConfig holds configuration for a saga
type SagaConfig struct {
	ID          string                 `json:"id" yaml:"id"`
	Name        string                 `json:"name" yaml:"name"`
	Description string                 `json:"description" yaml:"description"`
	Version     string                 `json:"version" yaml:"version"`
	Steps       []SagaStepConfig       `json:"steps" yaml:"steps"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
}

// SagaStepConfig holds configuration for a saga step
type SagaStepConfig struct {
	ID           string                 `json:"id" yaml:"id"`
	Name         string                 `json:"name" yaml:"name"`
	Description  string                 `json:"description" yaml:"description"`
	Order        int                    `json:"order" yaml:"order"`
	Action       string                 `json:"action" yaml:"action"`
	Compensation string                 `json:"compensation" yaml:"compensation"`
	MaxRetries   int                    `json:"max_retries" yaml:"max_retries"`
	Timeout      time.Duration          `json:"timeout" yaml:"timeout"`
	Metadata     map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active       bool                   `json:"active" yaml:"active"`
	CreatedAt    time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at" yaml:"updated_at"`
}

// ServiceConfig holds configuration for the service
type ServiceConfig struct {
	Name                    string                 `json:"name" yaml:"name"`
	Version                 string                 `json:"version" yaml:"version"`
	Description             string                 `json:"description" yaml:"description"`
	MaxSagas                int                    `json:"max_sagas" yaml:"max_sagas"`
	MaxSteps                int                    `json:"max_steps" yaml:"max_steps"`
	CleanupInterval         time.Duration          `json:"cleanup_interval" yaml:"cleanup_interval"`
	ValidationEnabled       bool                   `json:"validation_enabled" yaml:"validation_enabled"`
	CachingEnabled          bool                   `json:"caching_enabled" yaml:"caching_enabled"`
	MonitoringEnabled       bool                   `json:"monitoring_enabled" yaml:"monitoring_enabled"`
	AuditingEnabled         bool                   `json:"auditing_enabled" yaml:"auditing_enabled"`
	RetryEnabled            bool                   `json:"retry_enabled" yaml:"retry_enabled"`
	CompensationEnabled     bool                   `json:"compensation_enabled" yaml:"compensation_enabled"`
	SupportedSagaTypes      []string               `json:"supported_saga_types" yaml:"supported_saga_types"`
	SupportedStepTypes      []string               `json:"supported_step_types" yaml:"supported_step_types"`
	ValidationRules         map[string]interface{} `json:"validation_rules" yaml:"validation_rules"`
	Metadata                map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// SagaStats holds statistics for a saga
type SagaStats struct {
	SagaID        string                 `json:"saga_id" yaml:"saga_id"`
	Status        SagaStatus             `json:"status" yaml:"status"`
	Active        bool                   `json:"active" yaml:"active"`
	CreatedAt     time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at" yaml:"updated_at"`
	StepsCount    int                    `json:"steps_count" yaml:"steps_count"`
	CompletedSteps int                    `json:"completed_steps" yaml:"completed_steps"`
	FailedSteps   int                    `json:"failed_steps" yaml:"failed_steps"`
	RetryCount    int                    `json:"retry_count" yaml:"retry_count"`
	Metadata      map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// StepStats holds statistics for a saga step
type StepStats struct {
	StepID      string                 `json:"step_id" yaml:"step_id"`
	Status      StepStatus             `json:"status" yaml:"status"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
	RetryCount  int                    `json:"retry_count" yaml:"retry_count"`
	ExecuteTime time.Duration          `json:"execute_time" yaml:"execute_time"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// ServiceStats holds statistics for the service
type ServiceStats struct {
	ServiceName      string                 `json:"service_name" yaml:"service_name"`
	Version          string                 `json:"version" yaml:"version"`
	Active           bool                   `json:"active" yaml:"active"`
	CreatedAt        time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt        time.Time              `json:"updated_at" yaml:"updated_at"`
	SagasCount       int                    `json:"sagas_count" yaml:"sagas_count"`
	StepsCount       int                    `json:"steps_count" yaml:"steps_count"`
	CompletedSagas   int                    `json:"completed_sagas" yaml:"completed_sagas"`
	FailedSagas      int                    `json:"failed_sagas" yaml:"failed_sagas"`
	CompensatedSagas int                    `json:"compensated_sagas" yaml:"compensated_sagas"`
	Metadata         map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// HealthStatus holds health status information
type HealthStatus struct {
	Status    string                 `json:"status" yaml:"status"`
	Checks    map[string]interface{} `json:"checks" yaml:"checks"`
	Timestamp time.Time              `json:"timestamp" yaml:"timestamp"`
}
