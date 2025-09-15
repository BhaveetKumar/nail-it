package circuit_breaker

import (
	"context"
	"time"
)

// CircuitBreaker defines the interface for a circuit breaker
type CircuitBreaker interface {
	GetID() string
	GetName() string
	GetDescription() string
	GetState() CircuitState
	GetConfig() CircuitConfig
	GetStats() CircuitStats
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	IsActive() bool
	SetActive(active bool)
	Execute(ctx context.Context, operation func() (interface{}, error)) (interface{}, error)
	ExecuteAsync(ctx context.Context, operation func() (interface{}, error)) <-chan CircuitResult
	Reset() error
	Validate() error
	Cleanup(ctx context.Context) error
}

// CircuitState represents the state of a circuit breaker
type CircuitState string

const (
	CircuitStateClosed   CircuitState = "closed"
	CircuitStateOpen     CircuitState = "open"
	CircuitStateHalfOpen CircuitState = "half-open"
)

// CircuitConfig holds configuration for a circuit breaker
type CircuitConfig struct {
	ID                     string                 `json:"id" yaml:"id"`
	Name                   string                 `json:"name" yaml:"name"`
	Description            string                 `json:"description" yaml:"description"`
	FailureThreshold       int                    `json:"failure_threshold" yaml:"failure_threshold"`
	SuccessThreshold       int                    `json:"success_threshold" yaml:"success_threshold"`
	Timeout                time.Duration          `json:"timeout" yaml:"timeout"`
	ResetTimeout           time.Duration          `json:"reset_timeout" yaml:"reset_timeout"`
	MaxRequests            int                    `json:"max_requests" yaml:"max_requests"`
	RequestVolumeThreshold int                    `json:"request_volume_threshold" yaml:"request_volume_threshold"`
	SleepWindow            time.Duration          `json:"sleep_window" yaml:"sleep_window"`
	ErrorThreshold         float64                `json:"error_threshold" yaml:"error_threshold"`
	SlowCallThreshold      time.Duration          `json:"slow_call_threshold" yaml:"slow_call_threshold"`
	SlowCallRatioThreshold float64                `json:"slow_call_ratio_threshold" yaml:"slow_call_ratio_threshold"`
	Metadata               map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active                 bool                   `json:"active" yaml:"active"`
	CreatedAt              time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt              time.Time              `json:"updated_at" yaml:"updated_at"`
}

// CircuitStats holds statistics for a circuit breaker
type CircuitStats struct {
	ID                   string                 `json:"id" yaml:"id"`
	State                CircuitState           `json:"state" yaml:"state"`
	TotalRequests        int64                  `json:"total_requests" yaml:"total_requests"`
	SuccessfulRequests   int64                  `json:"successful_requests" yaml:"successful_requests"`
	FailedRequests       int64                  `json:"failed_requests" yaml:"failed_requests"`
	SlowRequests         int64                  `json:"slow_requests" yaml:"slow_requests"`
	RejectedRequests     int64                  `json:"rejected_requests" yaml:"rejected_requests"`
	ErrorRate            float64                `json:"error_rate" yaml:"error_rate"`
	SlowCallRate         float64                `json:"slow_call_rate" yaml:"slow_call_rate"`
	LastFailureTime      *time.Time             `json:"last_failure_time" yaml:"last_failure_time"`
	LastSuccessTime      *time.Time             `json:"last_success_time" yaml:"last_success_time"`
	StateChangeTime      time.Time              `json:"state_change_time" yaml:"state_change_time"`
	ConsecutiveFailures  int                    `json:"consecutive_failures" yaml:"consecutive_failures"`
	ConsecutiveSuccesses int                    `json:"consecutive_successes" yaml:"consecutive_successes"`
	Metadata             map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// CircuitResult represents the result of a circuit breaker operation
type CircuitResult struct {
	Value     interface{}   `json:"value" yaml:"value"`
	Error     error         `json:"error" yaml:"error"`
	Duration  time.Duration `json:"duration" yaml:"duration"`
	Timestamp time.Time     `json:"timestamp" yaml:"timestamp"`
}

// CircuitBreakerManager manages circuit breakers
type CircuitBreakerManager interface {
	CreateCircuitBreaker(config CircuitConfig) (CircuitBreaker, error)
	DestroyCircuitBreaker(id string) error
	GetCircuitBreaker(id string) (CircuitBreaker, error)
	ListCircuitBreakers() []string
	GetCircuitBreakerStats(id string) map[string]interface{}
	GetAllCircuitBreakerStats() map[string]interface{}
	IsCircuitBreakerActive(id string) bool
	SetCircuitBreakerActive(id string, active bool) error
	GetManagerStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// CircuitBreakerService provides high-level operations for circuit breaker management
type CircuitBreakerService interface {
	CreateCircuitBreaker(config CircuitConfig) (CircuitBreaker, error)
	DestroyCircuitBreaker(id string) error
	GetCircuitBreaker(id string) (CircuitBreaker, error)
	ListCircuitBreakers() []string
	GetCircuitBreakerStats(id string) map[string]interface{}
	GetAllCircuitBreakerStats() map[string]interface{}
	IsCircuitBreakerActive(id string) bool
	SetCircuitBreakerActive(id string, active bool) error
	ExecuteWithCircuitBreaker(ctx context.Context, id string, operation func() (interface{}, error)) (interface{}, error)
	ExecuteWithCircuitBreakerAsync(ctx context.Context, id string, operation func() (interface{}, error)) <-chan CircuitResult
	GetServiceStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// CircuitBreakerExecutor executes operations through circuit breakers
type CircuitBreakerExecutor interface {
	Execute(ctx context.Context, id string, operation func() (interface{}, error)) (interface{}, error)
	ExecuteAsync(ctx context.Context, id string, operation func() (interface{}, error)) <-chan CircuitResult
	GetExecutorStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// ServiceConfig holds configuration for the service
type ServiceConfig struct {
	Name                    string                 `json:"name" yaml:"name"`
	Version                 string                 `json:"version" yaml:"version"`
	Description             string                 `json:"description" yaml:"description"`
	MaxCircuitBreakers      int                    `json:"max_circuit_breakers" yaml:"max_circuit_breakers"`
	CleanupInterval         time.Duration          `json:"cleanup_interval" yaml:"cleanup_interval"`
	ValidationEnabled       bool                   `json:"validation_enabled" yaml:"validation_enabled"`
	CachingEnabled          bool                   `json:"caching_enabled" yaml:"caching_enabled"`
	MonitoringEnabled       bool                   `json:"monitoring_enabled" yaml:"monitoring_enabled"`
	AuditingEnabled         bool                   `json:"auditing_enabled" yaml:"auditing_enabled"`
	DefaultFailureThreshold int                    `json:"default_failure_threshold" yaml:"default_failure_threshold"`
	DefaultSuccessThreshold int                    `json:"default_success_threshold" yaml:"default_success_threshold"`
	DefaultTimeout          time.Duration          `json:"default_timeout" yaml:"default_timeout"`
	DefaultResetTimeout     time.Duration          `json:"default_reset_timeout" yaml:"default_reset_timeout"`
	SupportedTypes          []string               `json:"supported_types" yaml:"supported_types"`
	ValidationRules         map[string]interface{} `json:"validation_rules" yaml:"validation_rules"`
	Metadata                map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// ServiceStats holds statistics for the service
type ServiceStats struct {
	ServiceName          string                 `json:"service_name" yaml:"service_name"`
	Version              string                 `json:"version" yaml:"version"`
	Active               bool                   `json:"active" yaml:"active"`
	CreatedAt            time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt            time.Time              `json:"updated_at" yaml:"updated_at"`
	CircuitBreakersCount int                    `json:"circuit_breakers_count" yaml:"circuit_breakers_count"`
	ClosedBreakers       int                    `json:"closed_breakers" yaml:"closed_breakers"`
	OpenBreakers         int                    `json:"open_breakers" yaml:"open_breakers"`
	HalfOpenBreakers     int                    `json:"half_open_breakers" yaml:"half_open_breakers"`
	TotalRequests        int64                  `json:"total_requests" yaml:"total_requests"`
	SuccessfulRequests   int64                  `json:"successful_requests" yaml:"successful_requests"`
	FailedRequests       int64                  `json:"failed_requests" yaml:"failed_requests"`
	RejectedRequests     int64                  `json:"rejected_requests" yaml:"rejected_requests"`
	Metadata             map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// HealthStatus holds health status information
type HealthStatus struct {
	Status    string                 `json:"status" yaml:"status"`
	Checks    map[string]interface{} `json:"checks" yaml:"checks"`
	Timestamp time.Time              `json:"timestamp" yaml:"timestamp"`
}
