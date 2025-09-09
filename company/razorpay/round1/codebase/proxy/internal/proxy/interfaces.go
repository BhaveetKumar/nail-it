package proxy

import (
	"context"
	"time"
)

// Service defines the interface for services that can be proxied
type Service interface {
	Process(ctx context.Context, request interface{}) (interface{}, error)
	GetName() string
	IsHealthy(ctx context.Context) bool
}

// Cache defines the interface for caching operations
type Cache interface {
	Get(key string) (interface{}, bool)
	Set(key string, value interface{}, expiration time.Duration)
	Delete(key string)
	Clear()
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
}

// CircuitBreaker defines the interface for circuit breaker operations
type CircuitBreaker interface {
	Execute(ctx context.Context, operation func() (interface{}, error)) (interface{}, error)
	GetState() string
	Reset()
}

// RateLimiter defines the interface for rate limiting operations
type RateLimiter interface {
	Allow(key string) bool
	Wait(ctx context.Context, key string) error
	GetLimit() int
	GetRemaining(key string) int
}

// Authentication defines the interface for authentication operations
type Authentication interface {
	Authenticate(ctx context.Context, token string) (*User, error)
	Authorize(ctx context.Context, user *User, resource string, action string) bool
	GenerateToken(user *User) (string, error)
}

// Authorization defines the interface for authorization operations
type Authorization interface {
	CheckPermission(ctx context.Context, user *User, resource string, action string) bool
	GetUserRoles(ctx context.Context, userID string) ([]string, error)
	HasRole(ctx context.Context, userID string, role string) bool
}

// LoadBalancer defines the interface for load balancing operations
type LoadBalancer interface {
	SelectService(services []Service) Service
	UpdateServiceHealth(service Service, healthy bool)
	GetHealthyServices() []Service
}

// RetryPolicy defines the interface for retry operations
type RetryPolicy interface {
	ShouldRetry(attempt int, err error) bool
	GetDelay(attempt int) time.Duration
	GetMaxAttempts() int
}

// Monitoring defines the interface for monitoring operations
type Monitoring interface {
	RecordRequest(ctx context.Context, service string, duration time.Duration, success bool)
	RecordError(ctx context.Context, service string, err error)
	GetServiceMetrics(service string) (*ServiceMetrics, error)
}

// Security defines the interface for security operations
type Security interface {
	ValidateInput(input interface{}) error
	SanitizeInput(input interface{}) interface{}
	CheckRateLimit(ctx context.Context, key string) bool
	AuditLog(ctx context.Context, action string, userID string, details map[string]interface{})
}

// Database defines the interface for database operations
type Database interface {
	Save(ctx context.Context, data interface{}) error
	Find(ctx context.Context, query interface{}) (interface{}, error)
	Update(ctx context.Context, id string, data interface{}) error
	Delete(ctx context.Context, id string) error
}

// MessageQueue defines the interface for message queue operations
type MessageQueue interface {
	Publish(ctx context.Context, topic string, message interface{}) error
	Subscribe(ctx context.Context, topic string, handler func(interface{}) error) error
	Close() error
}

// WebSocket defines the interface for WebSocket operations
type WebSocket interface {
	Send(ctx context.Context, clientID string, message interface{}) error
	Broadcast(ctx context.Context, message interface{}) error
	Register(ctx context.Context, clientID string, conn interface{}) error
	Unregister(ctx context.Context, clientID string) error
}
