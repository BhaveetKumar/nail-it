package decorator

import (
	"time"
)

// Request represents a generic request
type Request struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id"`
	Component string                 `json:"component"`
	Method    string                 `json:"method"`
	Data      interface{}            `json:"data"`
	Headers   map[string]string      `json:"headers"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
}

// Response represents a generic response
type Response struct {
	ID        string                 `json:"id"`
	RequestID string                 `json:"request_id"`
	Status    int                    `json:"status"`
	Data      interface{}            `json:"data"`
	Error     string                 `json:"error,omitempty"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
	Duration  time.Duration          `json:"duration"`
}

// ComponentMetrics represents metrics for a component
type ComponentMetrics struct {
	ComponentName     string    `json:"component_name"`
	TotalRequests     int64     `json:"total_requests"`
	SuccessfulRequests int64    `json:"successful_requests"`
	FailedRequests    int64     `json:"failed_requests"`
	AverageLatency    float64   `json:"average_latency"`
	MaxLatency        float64   `json:"max_latency"`
	MinLatency        float64   `json:"min_latency"`
	SuccessRate       float64   `json:"success_rate"`
	LastRequest       time.Time `json:"last_request"`
	LastError         time.Time `json:"last_error"`
	CacheHits         int64     `json:"cache_hits"`
	CacheMisses       int64     `json:"cache_misses"`
	RateLimitHits     int64     `json:"rate_limit_hits"`
	CircuitBreakerTrips int64   `json:"circuit_breaker_trips"`
}

// CacheStats represents cache statistics
type CacheStats struct {
	Hits       int64   `json:"hits"`
	Misses     int64   `json:"misses"`
	HitRate    float64 `json:"hit_rate"`
	Size       int64   `json:"size"`
	MaxSize    int64   `json:"max_size"`
	Evictions  int64   `json:"evictions"`
	LastAccess time.Time `json:"last_access"`
}

// CircuitBreakerStats represents circuit breaker statistics
type CircuitBreakerStats struct {
	State           string    `json:"state"`
	TotalRequests   int64     `json:"total_requests"`
	SuccessfulRequests int64  `json:"successful_requests"`
	FailedRequests  int64     `json:"failed_requests"`
	LastFailure     time.Time `json:"last_failure"`
	LastSuccess     time.Time `json:"last_success"`
	FailureRate     float64   `json:"failure_rate"`
	NextAttempt     time.Time `json:"next_attempt"`
}

// MessageQueueStats represents message queue statistics
type MessageQueueStats struct {
	TotalMessages    int64     `json:"total_messages"`
	PublishedMessages int64    `json:"published_messages"`
	ConsumedMessages int64     `json:"consumed_messages"`
	FailedMessages   int64     `json:"failed_messages"`
	QueueSize        int64     `json:"queue_size"`
	LastMessage      time.Time `json:"last_message"`
}

// WebSocketStats represents WebSocket statistics
type WebSocketStats struct {
	TotalConnections    int64     `json:"total_connections"`
	ActiveConnections   int64     `json:"active_connections"`
	TotalMessages       int64     `json:"total_messages"`
	MessagesPerSecond   float64   `json:"messages_per_second"`
	LastConnection      time.Time `json:"last_connection"`
	LastDisconnection   time.Time `json:"last_disconnection"`
}

// AuditLog represents an audit log entry
type AuditLog struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id"`
	Action    string                 `json:"action"`
	Resource  string                 `json:"resource"`
	Details   map[string]interface{} `json:"details"`
	IP        string                 `json:"ip"`
	UserAgent string                 `json:"user_agent"`
	Timestamp time.Time              `json:"timestamp"`
}

// DecoratorConfig represents configuration for decorators
type DecoratorConfig struct {
	Logging      LoggingConfig      `json:"logging"`
	Metrics      MetricsConfig      `json:"metrics"`
	Cache        CacheConfig        `json:"cache"`
	Security     SecurityConfig     `json:"security"`
	RateLimit    RateLimitConfig    `json:"rate_limit"`
	CircuitBreaker CircuitBreakerConfig `json:"circuit_breaker"`
	Retry        RetryConfig        `json:"retry"`
	Monitoring   MonitoringConfig   `json:"monitoring"`
	Validation   ValidationConfig   `json:"validation"`
	Encryption   EncryptionConfig   `json:"encryption"`
	Compression  CompressionConfig  `json:"compression"`
	Serialization SerializationConfig `json:"serialization"`
	Notification NotificationConfig `json:"notification"`
	Analytics    AnalyticsConfig    `json:"analytics"`
	Audit        AuditConfig        `json:"audit"`
}

// LoggingConfig represents logging configuration
type LoggingConfig struct {
	Enabled     bool     `json:"enabled"`
	Level       string   `json:"level"`
	Format      string   `json:"format"`
	Output      string   `json:"output"`
	Fields      []string `json:"fields"`
	IncludeData bool     `json:"include_data"`
}

// MetricsConfig represents metrics configuration
type MetricsConfig struct {
	Enabled         bool          `json:"enabled"`
	Port            int           `json:"port"`
	Path            string        `json:"path"`
	CollectInterval time.Duration `json:"collect_interval"`
	Labels          []string      `json:"labels"`
}

// CacheConfig represents cache configuration
type CacheConfig struct {
	Enabled         bool          `json:"enabled"`
	Type            string        `json:"type"`
	TTL             time.Duration `json:"ttl"`
	MaxSize         int64         `json:"max_size"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
	Compression     bool          `json:"compression"`
}

// SecurityConfig represents security configuration
type SecurityConfig struct {
	Enabled           bool     `json:"enabled"`
	ValidateInput     bool     `json:"validate_input"`
	SanitizeInput     bool     `json:"sanitize_input"`
	CheckPermissions  bool     `json:"check_permissions"`
	AuditLogging      bool     `json:"audit_logging"`
	AllowedOrigins    []string `json:"allowed_origins"`
	MaxRequestSize    int64    `json:"max_request_size"`
}

// RateLimitConfig represents rate limiting configuration
type RateLimitConfig struct {
	Enabled     bool          `json:"enabled"`
	RequestsPerMinute int     `json:"requests_per_minute"`
	BurstSize   int           `json:"burst_size"`
	WindowSize  time.Duration `json:"window_size"`
	KeyFunc     string        `json:"key_func"`
}

// CircuitBreakerConfig represents circuit breaker configuration
type CircuitBreakerConfig struct {
	Enabled           bool          `json:"enabled"`
	FailureThreshold  int           `json:"failure_threshold"`
	SuccessThreshold  int           `json:"success_threshold"`
	Timeout           time.Duration `json:"timeout"`
	MaxRequests       int           `json:"max_requests"`
}

// RetryConfig represents retry configuration
type RetryConfig struct {
	Enabled       bool          `json:"enabled"`
	MaxAttempts   int           `json:"max_attempts"`
	InitialDelay  time.Duration `json:"initial_delay"`
	MaxDelay      time.Duration `json:"max_delay"`
	BackoffFactor float64       `json:"backoff_factor"`
}

// MonitoringConfig represents monitoring configuration
type MonitoringConfig struct {
	Enabled       bool          `json:"enabled"`
	Port          int           `json:"port"`
	LogLevel      string        `json:"log_level"`
	CollectInterval time.Duration `json:"collect_interval"`
	CustomMetrics []string      `json:"custom_metrics"`
}

// ValidationConfig represents validation configuration
type ValidationConfig struct {
	Enabled     bool                   `json:"enabled"`
	Rules       map[string]interface{} `json:"rules"`
	Schemas     map[string]interface{} `json:"schemas"`
	StrictMode  bool                   `json:"strict_mode"`
}

// EncryptionConfig represents encryption configuration
type EncryptionConfig struct {
	Enabled     bool   `json:"enabled"`
	Algorithm   string `json:"algorithm"`
	KeySize     int    `json:"key_size"`
	SaltLength  int    `json:"salt_length"`
	Iterations  int    `json:"iterations"`
}

// CompressionConfig represents compression configuration
type CompressionConfig struct {
	Enabled     bool   `json:"enabled"`
	Algorithm   string `json:"algorithm"`
	Level       int    `json:"level"`
	MinSize     int    `json:"min_size"`
}

// SerializationConfig represents serialization configuration
type SerializationConfig struct {
	Enabled     bool   `json:"enabled"`
	Format      string `json:"format"`
	Compression bool   `json:"compression"`
	Encryption  bool   `json:"encryption"`
}

// NotificationConfig represents notification configuration
type NotificationConfig struct {
	Enabled     bool     `json:"enabled"`
	Channels    []string `json:"channels"`
	RetryCount  int      `json:"retry_count"`
	RetryDelay  time.Duration `json:"retry_delay"`
	BatchSize   int      `json:"batch_size"`
}

// AnalyticsConfig represents analytics configuration
type AnalyticsConfig struct {
	Enabled     bool     `json:"enabled"`
	Events      []string `json:"events"`
	BatchSize   int      `json:"batch_size"`
	FlushInterval time.Duration `json:"flush_interval"`
}

// AuditConfig represents audit configuration
type AuditConfig struct {
	Enabled     bool     `json:"enabled"`
	Events      []string `json:"events"`
	Retention   time.Duration `json:"retention"`
	Compression bool     `json:"compression"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string `json:"error"`
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
	Timestamp time.Time `json:"timestamp"`
}

// HealthCheck represents a health check result
type HealthCheck struct {
	Component string    `json:"component"`
	Healthy   bool      `json:"healthy"`
	Status    string    `json:"status"`
	Message   string    `json:"message"`
	Timestamp time.Time `json:"timestamp"`
	Latency   time.Duration `json:"latency"`
}
