package mediator

import (
	"context"
	"time"
)

// Mediator defines the interface for the mediator pattern
type Mediator interface {
	RegisterColleague(colleague Colleague) error
	UnregisterColleague(colleagueID string) error
	SendMessage(senderID string, recipientID string, message interface{}) error
	BroadcastMessage(senderID string, message interface{}) error
	GetColleagues() []Colleague
	GetColleague(colleagueID string) (Colleague, error)
}

// Colleague defines the interface for colleagues in the mediator pattern
type Colleague interface {
	GetID() string
	GetName() string
	GetType() string
	ReceiveMessage(senderID string, message interface{}) error
	SendMessage(recipientID string, message interface{}) error
	BroadcastMessage(message interface{}) error
	SetMediator(mediator Mediator)
	GetMediator() Mediator
	IsActive() bool
	SetActive(active bool)
	GetLastActivity() time.Time
	UpdateActivity()
}

// Message defines the interface for messages
type Message interface {
	GetID() string
	GetType() string
	GetContent() interface{}
	GetSenderID() string
	GetRecipientID() string
	GetTimestamp() time.Time
	GetPriority() int
	SetPriority(priority int)
	IsBroadcast() bool
	SetBroadcast(broadcast bool)
}

// MessageHandler defines the interface for message handlers
type MessageHandler interface {
	HandleMessage(message Message) error
	CanHandle(messageType string) bool
	GetPriority() int
}

// Event defines the interface for events
type Event interface {
	GetID() string
	GetType() string
	GetData() interface{}
	GetTimestamp() time.Time
	GetSource() string
}

// EventHandler defines the interface for event handlers
type EventHandler interface {
	HandleEvent(event Event) error
	CanHandle(eventType string) bool
	GetPriority() int
}

// Command defines the interface for commands
type Command interface {
	GetID() string
	GetType() string
	GetData() interface{}
	GetTimestamp() time.Time
	GetSource() string
	Execute() error
	Undo() error
	CanUndo() bool
}

// CommandHandler defines the interface for command handlers
type CommandHandler interface {
	HandleCommand(command Command) error
	CanHandle(commandType string) bool
	GetPriority() int
}

// Query defines the interface for queries
type Query interface {
	GetID() string
	GetType() string
	GetData() interface{}
	GetTimestamp() time.Time
	GetSource() string
	Execute() (interface{}, error)
}

// QueryHandler defines the interface for query handlers
type QueryHandler interface {
	HandleQuery(query Query) (interface{}, error)
	CanHandle(queryType string) bool
	GetPriority() int
}

// Notification defines the interface for notifications
type Notification interface {
	GetID() string
	GetType() string
	GetContent() interface{}
	GetRecipientID() string
	GetTimestamp() time.Time
	GetPriority() int
	IsRead() bool
	SetRead(read bool)
	GetReadAt() time.Time
	SetReadAt(readAt time.Time)
}

// NotificationHandler defines the interface for notification handlers
type NotificationHandler interface {
	HandleNotification(notification Notification) error
	CanHandle(notificationType string) bool
	GetPriority() int
}

// Workflow defines the interface for workflows
type Workflow interface {
	GetID() string
	GetName() string
	GetSteps() []WorkflowStep
	GetCurrentStep() int
	GetStatus() string
	GetData() interface{}
	Execute() error
	Pause() error
	Resume() error
	Cancel() error
	Complete() error
	GetProgress() float64
	GetEstimatedTime() time.Duration
	GetActualTime() time.Duration
}

// WorkflowStep defines the interface for workflow steps
type WorkflowStep interface {
	GetID() string
	GetName() string
	GetType() string
	GetStatus() string
	GetData() interface{}
	Execute() error
	CanExecute() bool
	GetDependencies() []string
	GetEstimatedTime() time.Duration
	GetActualTime() time.Duration
}

// WorkflowHandler defines the interface for workflow handlers
type WorkflowHandler interface {
	HandleWorkflow(workflow Workflow) error
	CanHandle(workflowType string) bool
	GetPriority() int
}

// Service defines the interface for services
type Service interface {
	GetID() string
	GetName() string
	GetType() string
	GetStatus() string
	GetHealth() bool
	GetMetrics() map[string]interface{}
	Start() error
	Stop() error
	Restart() error
	GetDependencies() []string
	GetDependents() []string
}

// ServiceHandler defines the interface for service handlers
type ServiceHandler interface {
	HandleService(service Service) error
	CanHandle(serviceType string) bool
	GetPriority() int
}

// Resource defines the interface for resources
type Resource interface {
	GetID() string
	GetName() string
	GetType() string
	GetStatus() string
	GetCapacity() int
	GetUsed() int
	GetAvailable() int
	GetUtilization() float64
	Acquire() error
	Release() error
	IsAvailable() bool
	GetWaitTime() time.Duration
}

// ResourceHandler defines the interface for resource handlers
type ResourceHandler interface {
	HandleResource(resource Resource) error
	CanHandle(resourceType string) bool
	GetPriority() int
}

// Task defines the interface for tasks
type Task interface {
	GetID() string
	GetName() string
	GetType() string
	GetStatus() string
	GetPriority() int
	GetData() interface{}
	GetDependencies() []string
	GetDependents() []string
	Execute() error
	CanExecute() bool
	GetEstimatedTime() time.Duration
	GetActualTime() time.Duration
	GetProgress() float64
}

// TaskHandler defines the interface for task handlers
type TaskHandler interface {
	HandleTask(task Task) error
	CanHandle(taskType string) bool
	GetPriority() int
}

// Job defines the interface for jobs
type Job interface {
	GetID() string
	GetName() string
	GetType() string
	GetStatus() string
	GetPriority() int
	GetData() interface{}
	GetSchedule() string
	GetNextRun() time.Time
	GetLastRun() time.Time
	Execute() error
	CanExecute() bool
	GetEstimatedTime() time.Duration
	GetActualTime() time.Duration
	GetProgress() float64
}

// JobHandler defines the interface for job handlers
type JobHandler interface {
	HandleJob(job Job) error
	CanHandle(jobType string) bool
	GetPriority() int
}

// Cache defines the interface for cache operations
type Cache interface {
	Get(key string) (interface{}, bool)
	Set(key string, value interface{}, ttl time.Duration) error
	Delete(key string) error
	Clear() error
	Size() int
	Keys() []string
	GetStats() map[string]interface{}
}

// Database defines the interface for database operations
type Database interface {
	Connect() error
	Disconnect() error
	IsConnected() bool
	GetStats() map[string]interface{}
	Execute(query string, args ...interface{}) (interface{}, error)
	Query(query string, args ...interface{}) ([]map[string]interface{}, error)
	Transaction(fn func(Database) error) error
}

// MessageQueue defines the interface for message queue operations
type MessageQueue interface {
	Connect() error
	Disconnect() error
	IsConnected() bool
	Publish(topic string, message interface{}) error
	Subscribe(topic string, handler func(interface{}) error) error
	Unsubscribe(topic string) error
	GetStats() map[string]interface{}
}

// WebSocket defines the interface for WebSocket operations
type WebSocket interface {
	Connect() error
	Disconnect() error
	IsConnected() bool
	Send(message interface{}) error
	Receive() (interface{}, error)
	Broadcast(message interface{}) error
	GetStats() map[string]interface{}
}

// Security defines the interface for security operations
type Security interface {
	Authenticate(token string) (string, error)
	Authorize(userID string, resource string, action string) bool
	AuditLog(action string, userID string, details map[string]interface{}) error
	GetUserPermissions(userID string) ([]string, error)
	ValidateInput(input interface{}) error
}

// Monitoring defines the interface for monitoring operations
type Monitoring interface {
	RecordMetric(name string, value float64, labels map[string]string) error
	RecordEvent(name string, data map[string]interface{}) error
	GetMetrics() map[string]interface{}
	GetHealth() map[string]interface{}
	GetAlerts() []map[string]interface{}
}

// Logging defines the interface for logging operations
type Logging interface {
	Debug(message string, fields ...map[string]interface{})
	Info(message string, fields ...map[string]interface{})
	Warn(message string, fields ...map[string]interface{})
	Error(message string, fields ...map[string]interface{})
	Fatal(message string, fields ...map[string]interface{})
	GetLogs(level string, limit int) ([]map[string]interface{}, error)
}

// Configuration defines the interface for configuration operations
type Configuration interface {
	Get(key string) (interface{}, error)
	Set(key string, value interface{}) error
	GetString(key string) (string, error)
	GetInt(key string) (int, error)
	GetBool(key string) (bool, error)
	GetFloat(key string) (float64, error)
	GetDuration(key string) (time.Duration, error)
	GetStringSlice(key string) ([]string, error)
	GetMap(key string) (map[string]interface{}, error)
	Reload() error
	Watch(key string, callback func(interface{})) error
}

// Validation defines the interface for validation operations
type Validation interface {
	Validate(data interface{}, rules map[string]interface{}) error
	ValidateField(field string, value interface{}, rules map[string]interface{}) error
	ValidateStruct(struct interface{}) error
	ValidateMap(data map[string]interface{}, rules map[string]interface{}) error
	GetErrors() []string
	ClearErrors()
}

// Transformation defines the interface for transformation operations
type Transformation interface {
	Transform(data interface{}, transformer string) (interface{}, error)
	TransformField(field string, value interface{}, transformer string) (interface{}, error)
	TransformStruct(struct interface{}, transformer string) (interface{}, error)
	TransformMap(data map[string]interface{}, transformer string) (map[string]interface{}, error)
	GetAvailableTransformers() []string
}

// Serialization defines the interface for serialization operations
type Serialization interface {
	Serialize(data interface{}) ([]byte, error)
	Deserialize(data []byte, target interface{}) error
	SerializeToJSON(data interface{}) ([]byte, error)
	DeserializeFromJSON(data []byte, target interface{}) error
	SerializeToXML(data interface{}) ([]byte, error)
	DeserializeFromXML(data []byte, target interface{}) error
	SerializeToYAML(data interface{}) ([]byte, error)
	DeserializeFromYAML(data []byte, target interface{}) error
}

// Compression defines the interface for compression operations
type Compression interface {
	Compress(data []byte) ([]byte, error)
	Decompress(data []byte) ([]byte, error)
	CompressString(data string) (string, error)
	DecompressString(data string) (string, error)
	GetCompressionRatio() float64
	GetCompressionTime() time.Duration
}

// Encryption defines the interface for encryption operations
type Encryption interface {
	Encrypt(data []byte, key []byte) ([]byte, error)
	Decrypt(data []byte, key []byte) ([]byte, error)
	EncryptString(data string, key string) (string, error)
	DecryptString(data string, key string) (string, error)
	GenerateKey() ([]byte, error)
	Hash(data []byte) ([]byte, error)
	HashString(data string) (string, error)
}

// RateLimiting defines the interface for rate limiting operations
type RateLimiting interface {
	Allow(key string, limit int, window time.Duration) bool
	GetRemaining(key string, limit int, window time.Duration) int
	GetResetTime(key string, limit int, window time.Duration) time.Time
	GetStats() map[string]interface{}
	Reset(key string) error
	Clear() error
}

// CircuitBreaker defines the interface for circuit breaker operations
type CircuitBreaker interface {
	Call(fn func() error) error
	GetState() string
	GetStats() map[string]interface{}
	Reset() error
	IsOpen() bool
	IsClosed() bool
	IsHalfOpen() bool
}

// Retry defines the interface for retry operations
type Retry interface {
	Execute(fn func() error) error
	ExecuteWithBackoff(fn func() error) error
	GetStats() map[string]interface{}
	Reset() error
	IsExhausted() bool
	GetAttempts() int
	GetMaxAttempts() int
}

// Timeout defines the interface for timeout operations
type Timeout interface {
	Execute(fn func() error, timeout time.Duration) error
	ExecuteWithContext(ctx context.Context, fn func() error) error
	GetStats() map[string]interface{}
	Reset() error
	IsExpired() bool
	GetRemaining() time.Duration
}

// HealthCheck defines the interface for health check operations
type HealthCheck interface {
	Check() error
	GetStatus() string
	GetDetails() map[string]interface{}
	GetLastCheck() time.Time
	GetInterval() time.Duration
	SetInterval(interval time.Duration)
	Start() error
	Stop() error
	IsRunning() bool
}

// Metrics defines the interface for metrics operations
type Metrics interface {
	IncrementCounter(name string, labels map[string]string)
	RecordHistogram(name string, value float64, labels map[string]string)
	RecordGauge(name string, value float64, labels map[string]string)
	RecordTiming(name string, duration time.Duration, labels map[string]string)
	GetMetrics() map[string]interface{}
	GetCounter(name string) (int64, error)
	GetHistogram(name string) (map[string]interface{}, error)
	GetGauge(name string) (float64, error)
	GetTiming(name string) (map[string]interface{}, error)
}

// Alerting defines the interface for alerting operations
type Alerting interface {
	SendAlert(alert Alert) error
	GetAlerts() []Alert
	GetActiveAlerts() []Alert
	GetResolvedAlerts() []Alert
	ResolveAlert(alertID string) error
	GetAlertStats() map[string]interface{}
	SetThreshold(metric string, threshold float64) error
	GetThreshold(metric string) (float64, error)
}

// Alert defines the interface for alerts
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
