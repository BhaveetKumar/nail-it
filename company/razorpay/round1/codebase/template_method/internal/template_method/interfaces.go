package template_method

import (
	"context"
	"time"
)

// TemplateMethod defines the interface for template methods
type TemplateMethod interface {
	Execute() error
	GetName() string
	GetDescription() string
	GetSteps() []Step
	GetCurrentStep() int
	GetStatus() string
	GetData() interface{}
	SetData(data interface{}) error
	GetResult() interface{}
	SetResult(result interface{}) error
	GetStartTime() time.Time
	GetEndTime() time.Time
	GetDuration() time.Duration
	IsCompleted() bool
	IsFailed() bool
	IsRunning() bool
	GetError() error
	SetError(err error)
	GetMetadata() map[string]interface{}
	SetMetadata(metadata map[string]interface{})
}

// Step defines the interface for template method steps
type Step interface {
	GetName() string
	GetDescription() string
	GetType() string
	GetStatus() string
	GetData() interface{}
	SetData(data interface{}) error
	GetResult() interface{}
	SetResult(result interface{}) error
	GetStartTime() time.Time
	GetEndTime() time.Time
	GetDuration() time.Duration
	IsCompleted() bool
	IsFailed() bool
	IsRunning() bool
	GetError() error
	SetError(err error)
	GetDependencies() []string
	SetDependencies(dependencies []string)
	CanExecute() bool
	Execute() error
	Validate() error
	GetMetadata() map[string]interface{}
	SetMetadata(metadata map[string]interface{})
}

// TemplateMethodManager manages template methods
type TemplateMethodManager interface {
	CreateTemplateMethod(name string, template TemplateMethod) error
	GetTemplateMethod(name string) (TemplateMethod, error)
	RemoveTemplateMethod(name string) error
	ListTemplateMethods() []string
	GetTemplateMethodCount() int
	GetTemplateMethodStats() map[string]interface{}
	Cleanup() error
}

// TemplateMethodExecutor executes template methods
type TemplateMethodExecutor interface {
	ExecuteTemplateMethod(template TemplateMethod) error
	ExecuteStep(step Step) error
	ExecuteSteps(steps []Step) error
	ValidateTemplateMethod(template TemplateMethod) error
	ValidateStep(step Step) error
	GetExecutionStats() map[string]interface{}
	GetExecutionHistory() []ExecutionRecord
	ClearExecutionHistory() error
}

// ExecutionRecord represents an execution record
type ExecutionRecord interface {
	GetID() string
	GetTemplateMethodName() string
	GetStartTime() time.Time
	GetEndTime() time.Time
	GetDuration() time.Duration
	GetStatus() string
	GetError() error
	GetData() interface{}
	GetResult() interface{}
	GetMetadata() map[string]interface{}
	SetEndTime(endTime time.Time)
	SetStatus(status string)
	SetError(err error)
	SetResult(result interface{})
	SetMetadata(metadata map[string]interface{})
	IsCompleted() bool
	IsFailed() bool
	IsRunning() bool
}

// TemplateMethodValidator validates template methods
type TemplateMethodValidator interface {
	ValidateTemplateMethod(template TemplateMethod) error
	ValidateStep(step Step) error
	ValidateSteps(steps []Step) error
	GetValidationRules() map[string]interface{}
	SetValidationRules(rules map[string]interface{})
	GetValidationErrors() []string
	ClearValidationErrors()
}

// TemplateMethodCache caches template methods
type TemplateMethodCache interface {
	Get(key string) (TemplateMethod, bool)
	Set(key string, template TemplateMethod, ttl time.Duration) error
	Delete(key string) error
	Clear() error
	Size() int
	Keys() []string
	GetStats() map[string]interface{}
	GetHitRate() float64
	GetMissRate() float64
	GetEvictionCount() int64
	GetExpirationCount() int64
}

// TemplateMethodMonitor monitors template methods
type TemplateMethodMonitor interface {
	RecordTemplateMethodExecution(template TemplateMethod) error
	RecordStepExecution(step Step) error
	GetTemplateMethodMetrics() map[string]interface{}
	GetStepMetrics() map[string]interface{}
	GetExecutionMetrics() map[string]interface{}
	GetPerformanceMetrics() map[string]interface{}
	GetErrorMetrics() map[string]interface{}
	GetResourceMetrics() map[string]interface{}
	GetHealthMetrics() map[string]interface{}
	GetAlertMetrics() []Alert
	ResetMetrics() error
}

// Alert represents an alert
type Alert interface {
	GetID() string
	GetType() string
	GetSeverity() string
	GetMessage() string
	GetTimestamp() time.Time
	GetStatus() string
	GetSource() string
	GetData() map[string]interface{}
	SetStatus(status string)
	Resolve() error
	IsActive() bool
	IsResolved() bool
}

// TemplateMethodScheduler schedules template methods
type TemplateMethodScheduler interface {
	ScheduleTemplateMethod(template TemplateMethod, schedule string) error
	ScheduleStep(step Step, schedule string) error
	GetScheduledTasks() ([]ScheduledTask, error)
	CancelTask(taskID string) error
	GetTaskStatus(taskID string) (string, error)
	GetTaskStats() map[string]interface{}
	GetTaskHistory() ([]TaskHistory, error)
}

// ScheduledTask represents a scheduled task
type ScheduledTask interface {
	GetID() string
	GetName() string
	GetType() string
	GetSchedule() string
	GetStatus() string
	GetNextRun() time.Time
	GetLastRun() time.Time
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetStatus(status string)
	SetNextRun(nextRun time.Time)
	SetLastRun(lastRun time.Time)
	SetUpdatedAt(updatedAt time.Time)
	SetMetadata(metadata map[string]interface{})
	IsActive() bool
	SetActive(active bool)
	GetRetryCount() int
	SetRetryCount(retryCount int)
	GetMaxRetries() int
	SetMaxRetries(maxRetries int)
}

// TaskHistory represents task history
type TaskHistory interface {
	GetID() string
	GetTaskID() string
	GetStatus() string
	GetStartTime() time.Time
	GetEndTime() time.Time
	GetDuration() time.Duration
	GetError() string
	GetResult() map[string]interface{}
	GetMetadata() map[string]interface{}
	SetStatus(status string)
	SetEndTime(endTime time.Time)
	SetError(error string)
	SetResult(result map[string]interface{})
	SetMetadata(metadata map[string]interface{})
	IsSuccessful() bool
	IsFailed() bool
	IsRunning() bool
	IsCompleted() bool
}

// TemplateMethodAuditor audits template methods
type TemplateMethodAuditor interface {
	AuditTemplateMethodExecution(template TemplateMethod, userID string) error
	AuditStepExecution(step Step, userID string) error
	GetAuditLogs() ([]AuditLog, error)
	GetAuditLogsByUser(userID string) ([]AuditLog, error)
	GetAuditLogsByTemplateMethod(templateMethodName string) ([]AuditLog, error)
	GetAuditLogsByDateRange(start, end time.Time) ([]AuditLog, error)
	GetAuditLogsByAction(action string) ([]AuditLog, error)
	GetAuditStats() map[string]interface{}
	ClearAuditLogs() error
	ExportAuditLogs(format string) ([]byte, error)
}

// AuditLog represents an audit log
type AuditLog interface {
	GetID() string
	GetUserID() string
	GetAction() string
	GetResource() string
	GetDetails() map[string]interface{}
	GetIP() string
	GetUserAgent() string
	GetTimestamp() time.Time
	GetSessionID() string
	GetRequestID() string
	GetResponseCode() int
	GetResponseTime() time.Duration
	SetResponseCode(code int)
	SetResponseTime(responseTime time.Duration)
	IsSuccessful() bool
	SetSuccessful(successful bool)
	GetError() string
	SetError(error string)
	GetMetadata() map[string]interface{}
	SetMetadata(metadata map[string]interface{})
}

// TemplateMethodConfig defines the interface for template method configuration
type TemplateMethodConfig interface {
	GetName() string
	GetVersion() string
	GetDescription() string
	GetMaxTemplateMethods() int
	GetMaxSteps() int
	GetMaxExecutionTime() time.Duration
	GetMaxRetries() int
	GetRetryDelay() time.Duration
	GetRetryBackoff() float64
	GetValidationEnabled() bool
	GetCachingEnabled() bool
	GetMonitoringEnabled() bool
	GetAuditingEnabled() bool
	GetSchedulingEnabled() bool
	GetSupportedTypes() []string
	GetDefaultType() string
	GetValidationRules() map[string]interface{}
	GetMetadata() map[string]interface{}
	SetMetadata(metadata map[string]interface{})
	GetDatabase() DatabaseConfig
	GetCache() CacheConfig
	GetMessageQueue() MessageQueueConfig
	GetWebSocket() WebSocketConfig
	GetSecurity() SecurityConfig
	GetMonitoring() MonitoringConfig
	GetLogging() LoggingConfig
}

// DatabaseConfig defines the interface for database configuration
type DatabaseConfig interface {
	GetMySQL() MySQLConfig
	GetMongoDB() MongoDBConfig
	GetRedis() RedisConfig
}

// MySQLConfig defines the interface for MySQL configuration
type MySQLConfig interface {
	GetHost() string
	GetPort() int
	GetUsername() string
	GetPassword() string
	GetDatabase() string
	GetMaxConnections() int
	GetMaxIdleConnections() int
	GetConnectionMaxLifetime() time.Duration
}

// MongoDBConfig defines the interface for MongoDB configuration
type MongoDBConfig interface {
	GetURI() string
	GetDatabase() string
	GetMaxPoolSize() int
	GetMinPoolSize() int
	GetMaxIdleTime() time.Duration
}

// RedisConfig defines the interface for Redis configuration
type RedisConfig interface {
	GetHost() string
	GetPort() int
	GetPassword() string
	GetDB() int
	GetMaxRetries() int
	GetPoolSize() int
	GetMinIdleConnections() int
}

// CacheConfig defines the interface for cache configuration
type CacheConfig interface {
	GetEnabled() bool
	GetType() string
	GetTTL() time.Duration
	GetMaxSize() int64
	GetCleanupInterval() time.Duration
	GetEvictionPolicy() string
}

// MessageQueueConfig defines the interface for message queue configuration
type MessageQueueConfig interface {
	GetEnabled() bool
	GetBrokers() []string
	GetTopics() []string
	GetConsumerGroup() string
	GetAutoCommit() bool
	GetCommitInterval() time.Duration
}

// WebSocketConfig defines the interface for WebSocket configuration
type WebSocketConfig interface {
	GetEnabled() bool
	GetPort() int
	GetReadBufferSize() int
	GetWriteBufferSize() int
	GetHandshakeTimeout() time.Duration
	GetPingPeriod() time.Duration
	GetPongWait() time.Duration
	GetWriteWait() time.Duration
	GetMaxMessageSize() int
	GetMaxConnections() int
}

// SecurityConfig defines the interface for security configuration
type SecurityConfig interface {
	GetEnabled() bool
	GetJWTSecret() string
	GetTokenExpiry() time.Duration
	GetAllowedOrigins() []string
	GetRateLimitEnabled() bool
	GetRateLimitRequests() int
	GetRateLimitWindow() time.Duration
	GetCORSEnabled() bool
	GetCORSOrigins() []string
}

// MonitoringConfig defines the interface for monitoring configuration
type MonitoringConfig interface {
	GetEnabled() bool
	GetPort() int
	GetPath() string
	GetCollectInterval() time.Duration
	GetHealthCheckInterval() time.Duration
	GetMetricsRetention() time.Duration
	GetAlertThresholds() map[string]interface{}
}

// LoggingConfig defines the interface for logging configuration
type LoggingConfig interface {
	GetLevel() string
	GetFormat() string
	GetOutput() string
	GetFilePath() string
	GetMaxSize() int
	GetMaxBackups() int
	GetMaxAge() int
	GetCompress() bool
}
