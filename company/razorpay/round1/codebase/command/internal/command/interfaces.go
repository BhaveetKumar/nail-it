package command

import (
	"context"
	"time"
)

// Command defines the interface for all commands
type Command interface {
	Execute(ctx context.Context) (*CommandResult, error)
	Undo(ctx context.Context) (*CommandResult, error)
	GetCommandID() string
	GetCommandType() string
	GetDescription() string
	GetCreatedAt() time.Time
	GetExecutedAt() time.Time
	IsExecuted() bool
	CanUndo() bool
	Validate() error
}

// CommandResult represents the result of command execution
type CommandResult struct {
	CommandID    string                 `json:"command_id"`
	Success      bool                   `json:"success"`
	Data         interface{}            `json:"data,omitempty"`
	Error        string                 `json:"error,omitempty"`
	ExecutedAt   time.Time              `json:"executed_at"`
	Duration     time.Duration          `json:"duration"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// CommandHandler handles command execution
type CommandHandler interface {
	Handle(ctx context.Context, command Command) (*CommandResult, error)
	CanHandle(commandType string) bool
	GetHandlerName() string
	GetSupportedCommands() []string
}

// CommandInvoker invokes commands
type CommandInvoker interface {
	Execute(ctx context.Context, command Command) (*CommandResult, error)
	ExecuteAsync(ctx context.Context, command Command) (<-chan *CommandResult, error)
	ExecuteBatch(ctx context.Context, commands []Command) ([]*CommandResult, error)
	GetExecutionHistory() []*CommandExecution
	ClearHistory()
}

// CommandReceiver receives and processes commands
type CommandReceiver interface {
	Receive(ctx context.Context, command Command) error
	Process(ctx context.Context, command Command) (*CommandResult, error)
	GetReceiverName() string
	IsAvailable() bool
}

// CommandQueue manages command queuing
type CommandQueue interface {
	Enqueue(ctx context.Context, command Command) error
	Dequeue(ctx context.Context) (Command, error)
	Peek(ctx context.Context) (Command, error)
	Size() int
	IsEmpty() bool
	Clear()
	GetQueueName() string
}

// CommandScheduler schedules command execution
type CommandScheduler interface {
	Schedule(ctx context.Context, command Command, executeAt time.Time) error
	ScheduleRecurring(ctx context.Context, command Command, interval time.Duration) error
	Cancel(ctx context.Context, commandID string) error
	GetScheduledCommands() []*ScheduledCommand
	GetSchedulerName() string
}

// CommandAuditor audits command execution
type CommandAuditor interface {
	Audit(ctx context.Context, command Command, result *CommandResult) error
	GetAuditLog(commandID string) (*AuditLog, error)
	GetAuditLogs(limit, offset int) ([]*AuditLog, error)
	GetAuditLogsByType(commandType string, limit, offset int) ([]*AuditLog, error)
	GetAuditLogsByTimeRange(start, end time.Time) ([]*AuditLog, error)
}

// CommandValidator validates commands
type CommandValidator interface {
	Validate(ctx context.Context, command Command) error
	GetValidationRules(commandType string) []ValidationRule
	AddValidationRule(commandType string, rule ValidationRule) error
	RemoveValidationRule(commandType string, ruleID string) error
}

// CommandMetrics collects command execution metrics
type CommandMetrics interface {
	RecordCommandExecution(commandType string, duration time.Duration, success bool)
	GetCommandMetrics(commandType string) (*CommandMetricsData, error)
	GetAllMetrics() (map[string]*CommandMetricsData, error)
	ResetMetrics(commandType string) error
	ResetAllMetrics() error
}

// CommandConfig holds configuration for commands
type CommandConfig struct {
	MaxRetries        int           `json:"max_retries"`
	RetryDelay        time.Duration `json:"retry_delay"`
	Timeout           time.Duration `json:"timeout"`
	EnableAuditing    bool          `json:"enable_auditing"`
	EnableMetrics     bool          `json:"enable_metrics"`
	EnableValidation  bool          `json:"enable_validation"`
	EnableScheduling  bool          `json:"enable_scheduling"`
	EnableQueuing     bool          `json:"enable_queuing"`
	MaxQueueSize      int           `json:"max_queue_size"`
	MaxHistorySize    int           `json:"max_history_size"`
	CircuitBreaker    CircuitBreakerConfig `json:"circuit_breaker"`
}

// CircuitBreakerConfig holds circuit breaker configuration
type CircuitBreakerConfig struct {
	Enabled           bool          `json:"enabled"`
	FailureThreshold  int           `json:"failure_threshold"`
	RecoveryTimeout   time.Duration `json:"recovery_timeout"`
	HalfOpenMaxCalls  int           `json:"half_open_max_calls"`
}

// CommandExecution represents a command execution record
type CommandExecution struct {
	CommandID    string        `json:"command_id"`
	CommandType  string        `json:"command_type"`
	Description  string        `json:"description"`
	Status       string        `json:"status"`
	StartTime    time.Time     `json:"start_time"`
	EndTime      time.Time     `json:"end_time"`
	Duration     time.Duration `json:"duration"`
	Result       *CommandResult `json:"result,omitempty"`
	Error        string        `json:"error,omitempty"`
	RetryCount   int           `json:"retry_count"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// ScheduledCommand represents a scheduled command
type ScheduledCommand struct {
	CommandID    string        `json:"command_id"`
	Command      Command       `json:"command"`
	ScheduledAt  time.Time     `json:"scheduled_at"`
	ExecuteAt    time.Time     `json:"execute_at"`
	Interval     time.Duration `json:"interval,omitempty"`
	Recurring    bool          `json:"recurring"`
	Status       string        `json:"status"`
	CreatedAt    time.Time     `json:"created_at"`
	UpdatedAt    time.Time     `json:"updated_at"`
}

// AuditLog represents an audit log entry
type AuditLog struct {
	LogID        string        `json:"log_id"`
	CommandID    string        `json:"command_id"`
	CommandType  string        `json:"command_type"`
	Action       string        `json:"action"`
	Status       string        `json:"status"`
	UserID       string        `json:"user_id,omitempty"`
	IPAddress    string        `json:"ip_address,omitempty"`
	UserAgent    string        `json:"user_agent,omitempty"`
	Timestamp    time.Time     `json:"timestamp"`
	Duration     time.Duration `json:"duration"`
	Data         interface{}   `json:"data,omitempty"`
	Error        string        `json:"error,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// ValidationRule represents a validation rule
type ValidationRule struct {
	RuleID      string                 `json:"rule_id"`
	RuleType    string                 `json:"rule_type"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Enabled     bool                   `json:"enabled"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// CommandMetricsData holds metrics for a command type
type CommandMetricsData struct {
	CommandType     string        `json:"command_type"`
	TotalExecutions int64         `json:"total_executions"`
	SuccessfulExecutions int64    `json:"successful_executions"`
	FailedExecutions int64        `json:"failed_executions"`
	AverageDuration time.Duration `json:"average_duration"`
	MinDuration     time.Duration `json:"min_duration"`
	MaxDuration     time.Duration `json:"max_duration"`
	LastExecution   time.Time     `json:"last_execution"`
	SuccessRate     float64       `json:"success_rate"`
	Availability    float64       `json:"availability"`
}

// CommandStatus represents command execution status
type CommandStatus string

const (
	CommandStatusPending   CommandStatus = "pending"
	CommandStatusExecuting CommandStatus = "executing"
	CommandStatusCompleted CommandStatus = "completed"
	CommandStatusFailed    CommandStatus = "failed"
	CommandStatusCancelled CommandStatus = "cancelled"
	CommandStatusRetrying  CommandStatus = "retrying"
)

// String returns the string representation of CommandStatus
func (cs CommandStatus) String() string {
	return string(cs)
}

// CommandPriority represents command execution priority
type CommandPriority int

const (
	CommandPriorityLow    CommandPriority = 1
	CommandPriorityNormal CommandPriority = 2
	CommandPriorityHigh   CommandPriority = 3
	CommandPriorityCritical CommandPriority = 4
)

// String returns the string representation of CommandPriority
func (cp CommandPriority) String() string {
	switch cp {
	case CommandPriorityLow:
		return "low"
	case CommandPriorityNormal:
		return "normal"
	case CommandPriorityHigh:
		return "high"
	case CommandPriorityCritical:
		return "critical"
	default:
		return "unknown"
	}
}
