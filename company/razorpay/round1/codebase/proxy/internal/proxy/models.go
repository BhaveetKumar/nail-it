package proxy

import (
	"time"
)

// User represents a user in the system
type User struct {
	ID       string   `json:"id"`
	Username string   `json:"username"`
	Email    string   `json:"email"`
	Roles    []string `json:"roles"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Request represents a generic request
type Request struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id"`
	Service   string                 `json:"service"`
	Method    string                 `json:"method"`
	Path      string                 `json:"path"`
	Headers   map[string]string      `json:"headers"`
	Body      interface{}            `json:"body"`
	Query     map[string]string      `json:"query"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
}

// Response represents a generic response
type Response struct {
	ID        string                 `json:"id"`
	RequestID string                 `json:"request_id"`
	Status    int                    `json:"status"`
	Headers   map[string]string      `json:"headers"`
	Body      interface{}            `json:"body"`
	Error     string                 `json:"error,omitempty"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
	Duration  time.Duration          `json:"duration"`
}

// ServiceMetrics represents metrics for a service
type ServiceMetrics struct {
	ServiceName     string    `json:"service_name"`
	TotalRequests   int64     `json:"total_requests"`
	SuccessfulRequests int64  `json:"successful_requests"`
	FailedRequests  int64     `json:"failed_requests"`
	AverageLatency  float64   `json:"average_latency"`
	MaxLatency      float64   `json:"max_latency"`
	MinLatency      float64   `json:"min_latency"`
	SuccessRate     float64   `json:"success_rate"`
	LastRequest     time.Time `json:"last_request"`
	LastError       time.Time `json:"last_error"`
}

// ProxyConfig represents configuration for the proxy
type ProxyConfig struct {
	Name            string            `json:"name"`
	Port            int               `json:"port"`
	Host            string            `json:"host"`
	Services        []ServiceConfig   `json:"services"`
	Cache           CacheConfig       `json:"cache"`
	RateLimit       RateLimitConfig   `json:"rate_limit"`
	CircuitBreaker  CircuitBreakerConfig `json:"circuit_breaker"`
	Security        SecurityConfig    `json:"security"`
	Monitoring      MonitoringConfig  `json:"monitoring"`
	LoadBalancing   LoadBalancingConfig `json:"load_balancing"`
	Retry           RetryConfig       `json:"retry"`
}

// ServiceConfig represents configuration for a service
type ServiceConfig struct {
	Name        string            `json:"name"`
	URL         string            `json:"url"`
	HealthCheck string            `json:"health_check"`
	Timeout     time.Duration     `json:"timeout"`
	RetryCount  int               `json:"retry_count"`
	Weight      int               `json:"weight"`
	Enabled     bool              `json:"enabled"`
	Headers     map[string]string `json:"headers"`
}

// CacheConfig represents cache configuration
type CacheConfig struct {
	Enabled     bool          `json:"enabled"`
	Type        string        `json:"type"`
	TTL         time.Duration `json:"ttl"`
	MaxSize     int           `json:"max_size"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
}

// RateLimitConfig represents rate limiting configuration
type RateLimitConfig struct {
	Enabled     bool          `json:"enabled"`
	RequestsPerMinute int     `json:"requests_per_minute"`
	BurstSize   int           `json:"burst_size"`
	WindowSize  time.Duration `json:"window_size"`
}

// CircuitBreakerConfig represents circuit breaker configuration
type CircuitBreakerConfig struct {
	Enabled           bool          `json:"enabled"`
	FailureThreshold  int           `json:"failure_threshold"`
	SuccessThreshold  int           `json:"success_threshold"`
	Timeout           time.Duration `json:"timeout"`
	MaxRequests       int           `json:"max_requests"`
}

// SecurityConfig represents security configuration
type SecurityConfig struct {
	Enabled           bool     `json:"enabled"`
	RequireAuth       bool     `json:"require_auth"`
	AllowedOrigins    []string `json:"allowed_origins"`
	AllowedMethods    []string `json:"allowed_methods"`
	AllowedHeaders    []string `json:"allowed_headers"`
	MaxRequestSize    int64    `json:"max_request_size"`
	ValidateInput     bool     `json:"validate_input"`
	SanitizeInput     bool     `json:"sanitize_input"`
}

// MonitoringConfig represents monitoring configuration
type MonitoringConfig struct {
	Enabled       bool          `json:"enabled"`
	MetricsPort   int           `json:"metrics_port"`
	LogLevel      string        `json:"log_level"`
	LogFormat     string        `json:"log_format"`
	CollectInterval time.Duration `json:"collect_interval"`
}

// LoadBalancingConfig represents load balancing configuration
type LoadBalancingConfig struct {
	Enabled     bool   `json:"enabled"`
	Algorithm   string `json:"algorithm"`
	HealthCheck bool   `json:"health_check"`
	Interval    time.Duration `json:"interval"`
}

// RetryConfig represents retry configuration
type RetryConfig struct {
	Enabled       bool          `json:"enabled"`
	MaxAttempts   int           `json:"max_attempts"`
	InitialDelay  time.Duration `json:"initial_delay"`
	MaxDelay      time.Duration `json:"max_delay"`
	BackoffFactor float64       `json:"backoff_factor"`
}

// ProxyStats represents proxy statistics
type ProxyStats struct {
	TotalRequests     int64     `json:"total_requests"`
	ActiveConnections int64     `json:"active_connections"`
	CacheHits         int64     `json:"cache_hits"`
	CacheMisses       int64     `json:"cache_misses"`
	RateLimitHits     int64     `json:"rate_limit_hits"`
	CircuitBreakerTrips int64   `json:"circuit_breaker_trips"`
	LastReset         time.Time `json:"last_reset"`
}

// HealthCheck represents a health check result
type HealthCheck struct {
	Service   string    `json:"service"`
	Healthy   bool      `json:"healthy"`
	Status    string    `json:"status"`
	Message   string    `json:"message"`
	Timestamp time.Time `json:"timestamp"`
	Latency   time.Duration `json:"latency"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string `json:"error"`
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
	Timestamp time.Time `json:"timestamp"`
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
