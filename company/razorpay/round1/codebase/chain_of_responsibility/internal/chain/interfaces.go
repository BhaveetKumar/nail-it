package chain

import (
	"context"
	"time"
)

// Handler defines the interface for chain of responsibility handlers
type Handler interface {
	SetNext(handler Handler)
	Handle(ctx context.Context, request *Request) (*Response, error)
	CanHandle(request *Request) bool
	GetName() string
	GetPriority() int
	IsEnabled() bool
	SetEnabled(enabled bool)
	GetStatistics() HandlerStatistics
}

// Request represents a request in the chain
type Request struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Priority    int                    `json:"priority"`
	Data        map[string]interface{} `json:"data"`
	Metadata    map[string]interface{} `json:"metadata"`
	UserID      string                 `json:"user_id"`
	SessionID   string                 `json:"session_id"`
	Timestamp   time.Time              `json:"timestamp"`
	Source      string                 `json:"source"`
	Destination string                 `json:"destination"`
	Headers     map[string]string      `json:"headers"`
	Context     context.Context        `json:"-"`
}

// Response represents a response from the chain
type Response struct {
	ID          string                 `json:"id"`
	RequestID   string                 `json:"request_id"`
	Status      string                 `json:"status"`
	Data        map[string]interface{} `json:"data"`
	Metadata    map[string]interface{} `json:"metadata"`
	HandlerName string                 `json:"handler_name"`
	ProcessedAt time.Time              `json:"processed_at"`
	Duration    time.Duration          `json:"duration"`
	Error       string                 `json:"error,omitempty"`
	NextHandler string                 `json:"next_handler,omitempty"`
}

// ChainManager defines the interface for managing the chain
type ChainManager interface {
	AddHandler(handler Handler) error
	RemoveHandler(name string) error
	GetHandler(name string) (Handler, error)
	GetAllHandlers() []Handler
	ProcessRequest(ctx context.Context, request *Request) (*Response, error)
	GetChainStatistics() ChainStatistics
	OptimizeChain() error
	ValidateChain() error
}

// Cache defines the interface for caching operations
type Cache interface {
	Get(key string) (interface{}, bool)
	Set(key string, value interface{}, expiration time.Duration)
	Delete(key string)
	Clear()
	GetStats() CacheStats
	GetSize() int
	GetMaxSize() int
}

// Logger defines the interface for logging operations
type Logger interface {
	Info(msg string, fields ...interface{})
	Error(msg string, fields ...interface{})
	Debug(msg string, fields ...interface{})
	Warn(msg string, fields ...interface{})
}

// Metrics defines the interface for metrics collection
type Metrics interface {
	IncrementCounter(name string, labels map[string]string)
	RecordHistogram(name string, value float64, labels map[string]string)
	RecordGauge(name string, value float64, labels map[string]string)
	RecordTiming(name string, duration time.Duration, labels map[string]string)
}

// Database defines the interface for database operations
type Database interface {
	Save(ctx context.Context, collection string, data interface{}) error
	Find(ctx context.Context, collection string, query interface{}) (interface{}, error)
	Update(ctx context.Context, collection string, id string, data interface{}) error
	Delete(ctx context.Context, collection string, id string) error
	Transaction(ctx context.Context, fn func(ctx context.Context) error) error
}

// MessageQueue defines the interface for message queue operations
type MessageQueue interface {
	Publish(ctx context.Context, topic string, message interface{}) error
	Subscribe(ctx context.Context, topic string, handler func(interface{}) error) error
	Close() error
	GetStats() MessageQueueStats
}

// WebSocket defines the interface for WebSocket operations
type WebSocket interface {
	Send(ctx context.Context, clientID string, message interface{}) error
	Broadcast(ctx context.Context, message interface{}) error
	Register(ctx context.Context, clientID string, conn interface{}) error
	Unregister(ctx context.Context, clientID string) error
	GetStats() WebSocketStats
}

// Security defines the interface for security operations
type Security interface {
	ValidateInput(input interface{}) error
	SanitizeInput(input interface{}) interface{}
	CheckPermission(ctx context.Context, userID string, resource string, action string) bool
	AuditLog(ctx context.Context, action string, userID string, details map[string]interface{})
}

// Configuration defines the interface for configuration operations
type Configuration interface {
	GetConfig(ctx context.Context, key string) (interface{}, error)
	SetConfig(ctx context.Context, key string, value interface{}) error
	GetAllConfigs(ctx context.Context) (map[string]interface{}, error)
	ReloadConfig(ctx context.Context) error
}

// Health defines the interface for health check operations
type Health interface {
	CheckHealth(ctx context.Context) (*HealthStatus, error)
	CheckServiceHealth(ctx context.Context, serviceName string) (*ServiceHealth, error)
	GetAllServicesHealth(ctx context.Context) ([]*ServiceHealth, error)
}

// Monitoring defines the interface for monitoring operations
type Monitoring interface {
	RecordRequest(ctx context.Context, service string, duration time.Duration, success bool)
	RecordError(ctx context.Context, service string, err error)
	GetServiceMetrics(ctx context.Context, service string) (*ServiceMetrics, error)
	GetSystemMetrics(ctx context.Context) (*SystemMetrics, error)
}

// Serialization defines the interface for serialization operations
type Serialization interface {
	Serialize(data interface{}) ([]byte, error)
	Deserialize(data []byte, target interface{}) error
	GetContentType() string
}

// Compression defines the interface for compression operations
type Compression interface {
	Compress(data []byte) ([]byte, error)
	Decompress(compressedData []byte) ([]byte, error)
	GetCompressionRatio(originalSize, compressedSize int) float64
}

// Encryption defines the interface for encryption operations
type Encryption interface {
	Encrypt(data []byte) ([]byte, error)
	Decrypt(encryptedData []byte) ([]byte, error)
	Hash(data []byte) ([]byte, error)
	VerifyHash(data []byte, hash []byte) bool
}

// Validation defines the interface for validation operations
type Validation interface {
	Validate(ctx context.Context, data interface{}) error
	ValidateSchema(ctx context.Context, data interface{}, schema interface{}) error
	GetValidationRules() map[string]interface{}
}

// Notification defines the interface for notification operations
type Notification interface {
	Send(ctx context.Context, recipient string, message interface{}) error
	SendBulk(ctx context.Context, recipients []string, message interface{}) error
	GetDeliveryStatus(ctx context.Context, messageID string) (string, error)
}

// Analytics defines the interface for analytics operations
type Analytics interface {
	Track(ctx context.Context, event string, properties map[string]interface{}) error
	TrackUser(ctx context.Context, userID string, event string, properties map[string]interface{}) error
	GetMetrics(ctx context.Context, query interface{}) (interface{}, error)
}

// Audit defines the interface for audit operations
type Audit interface {
	Log(ctx context.Context, action string, userID string, resource string, details map[string]interface{}) error
	GetAuditLogs(ctx context.Context, query interface{}) ([]AuditLog, error)
	GetUserActivity(ctx context.Context, userID string) ([]AuditLog, error)
}
