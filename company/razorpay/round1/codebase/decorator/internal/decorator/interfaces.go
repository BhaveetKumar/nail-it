package decorator

import (
	"context"
	"time"
)

// Component defines the interface for components that can be decorated
type Component interface {
	Execute(ctx context.Context, request interface{}) (interface{}, error)
	GetName() string
	GetDescription() string
}

// Decorator defines the interface for decorators
type Decorator interface {
	Component
	SetComponent(component Component)
	GetComponent() Component
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

// Cache defines the interface for caching operations
type Cache interface {
	Get(key string) (interface{}, bool)
	Set(key string, value interface{}, expiration time.Duration)
	Delete(key string)
	Clear()
	GetStats() CacheStats
}

// Security defines the interface for security operations
type Security interface {
	ValidateInput(input interface{}) error
	SanitizeInput(input interface{}) interface{}
	CheckPermission(ctx context.Context, userID string, resource string, action string) bool
	AuditLog(ctx context.Context, action string, userID string, details map[string]interface{})
}

// RateLimiter defines the interface for rate limiting operations
type RateLimiter interface {
	Allow(key string) bool
	Wait(ctx context.Context, key string) error
	GetLimit() int
	GetRemaining(key string) int
	Reset(key string)
}

// CircuitBreaker defines the interface for circuit breaker operations
type CircuitBreaker interface {
	Execute(ctx context.Context, operation func() (interface{}, error)) (interface{}, error)
	GetState() string
	Reset()
	GetStats() CircuitBreakerStats
}

// Retry defines the interface for retry operations
type Retry interface {
	Execute(ctx context.Context, operation func() (interface{}, error)) (interface{}, error)
	GetMaxAttempts() int
	GetDelay(attempt int) time.Duration
	ShouldRetry(attempt int, err error) bool
}

// Monitoring defines the interface for monitoring operations
type Monitoring interface {
	RecordRequest(ctx context.Context, component string, duration time.Duration, success bool)
	RecordError(ctx context.Context, component string, err error)
	RecordCustomMetric(name string, value float64, labels map[string]string)
	GetComponentMetrics(component string) (*ComponentMetrics, error)
}

// Validation defines the interface for validation operations
type Validation interface {
	Validate(ctx context.Context, data interface{}) error
	ValidateSchema(ctx context.Context, data interface{}, schema interface{}) error
	GetValidationRules() map[string]interface{}
}

// Encryption defines the interface for encryption operations
type Encryption interface {
	Encrypt(data []byte) ([]byte, error)
	Decrypt(encryptedData []byte) ([]byte, error)
	Hash(data []byte) ([]byte, error)
	VerifyHash(data []byte, hash []byte) bool
}

// Compression defines the interface for compression operations
type Compression interface {
	Compress(data []byte) ([]byte, error)
	Decompress(compressedData []byte) ([]byte, error)
	GetCompressionRatio(originalSize, compressedSize int) float64
}

// Serialization defines the interface for serialization operations
type Serialization interface {
	Serialize(data interface{}) ([]byte, error)
	Deserialize(data []byte, target interface{}) error
	GetContentType() string
}

// Database defines the interface for database operations
type Database interface {
	Save(ctx context.Context, data interface{}) error
	Find(ctx context.Context, query interface{}) (interface{}, error)
	Update(ctx context.Context, id string, data interface{}) error
	Delete(ctx context.Context, id string) error
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
