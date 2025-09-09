package decorator

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// BaseDecorator provides a base implementation for decorators
type BaseDecorator struct {
	component Component
	name      string
	description string
}

// NewBaseDecorator creates a new base decorator
func NewBaseDecorator(name, description string) *BaseDecorator {
	return &BaseDecorator{
		name:        name,
		description: description,
	}
}

// SetComponent sets the wrapped component
func (bd *BaseDecorator) SetComponent(component Component) {
	bd.component = component
}

// GetComponent returns the wrapped component
func (bd *BaseDecorator) GetComponent() Component {
	return bd.component
}

// GetName returns the decorator name
func (bd *BaseDecorator) GetName() string {
	return bd.name
}

// GetDescription returns the decorator description
func (bd *BaseDecorator) GetDescription() string {
	return bd.description
}

// Execute executes the wrapped component
func (bd *BaseDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if bd.component == nil {
		return nil, fmt.Errorf("no component set")
	}
	return bd.component.Execute(ctx, request)
}

// LoggingDecorator adds logging functionality
type LoggingDecorator struct {
	*BaseDecorator
	logger Logger
	config LoggingConfig
}

// NewLoggingDecorator creates a new logging decorator
func NewLoggingDecorator(logger Logger, config LoggingConfig) *LoggingDecorator {
	return &LoggingDecorator{
		BaseDecorator: NewBaseDecorator("logging", "Adds logging functionality"),
		logger:        logger,
		config:        config,
	}
}

// Execute executes with logging
func (ld *LoggingDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !ld.config.Enabled {
		return ld.BaseDecorator.Execute(ctx, request)
	}

	start := time.Now()
	ld.logger.Info("Request started", "component", ld.GetName(), "request", request)

	result, err := ld.BaseDecorator.Execute(ctx, request)

	duration := time.Since(start)
	if err != nil {
		ld.logger.Error("Request failed", "component", ld.GetName(), "error", err, "duration", duration)
	} else {
		ld.logger.Info("Request completed", "component", ld.GetName(), "duration", duration)
	}

	return result, err
}

// MetricsDecorator adds metrics collection
type MetricsDecorator struct {
	*BaseDecorator
	metrics Metrics
	config  MetricsConfig
}

// NewMetricsDecorator creates a new metrics decorator
func NewMetricsDecorator(metrics Metrics, config MetricsConfig) *MetricsDecorator {
	return &MetricsDecorator{
		BaseDecorator: NewBaseDecorator("metrics", "Adds metrics collection"),
		metrics:       metrics,
		config:        config,
	}
}

// Execute executes with metrics collection
func (md *MetricsDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !md.config.Enabled {
		return md.BaseDecorator.Execute(ctx, request)
	}

	start := time.Now()
	md.metrics.IncrementCounter("requests_total", map[string]string{"component": md.GetName()})

	result, err := md.BaseDecorator.Execute(ctx, request)

	duration := time.Since(start)
	md.metrics.RecordTiming("request_duration", duration, map[string]string{"component": md.GetName()})

	if err != nil {
		md.metrics.IncrementCounter("requests_failed", map[string]string{"component": md.GetName()})
	} else {
		md.metrics.IncrementCounter("requests_success", map[string]string{"component": md.GetName()})
	}

	return result, err
}

// CacheDecorator adds caching functionality
type CacheDecorator struct {
	*BaseDecorator
	cache  Cache
	config CacheConfig
	mu     sync.RWMutex
}

// NewCacheDecorator creates a new cache decorator
func NewCacheDecorator(cache Cache, config CacheConfig) *CacheDecorator {
	return &CacheDecorator{
		BaseDecorator: NewBaseDecorator("cache", "Adds caching functionality"),
		cache:         cache,
		config:        config,
	}
}

// Execute executes with caching
func (cd *CacheDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !cd.config.Enabled {
		return cd.BaseDecorator.Execute(ctx, request)
	}

	// Generate cache key
	cacheKey := cd.generateCacheKey(request)

	// Try to get from cache
	if cached, found := cd.cache.Get(cacheKey); found {
		cd.mu.Lock()
		// Update cache stats
		cd.mu.Unlock()
		return cached, nil
	}

	// Execute and cache result
	result, err := cd.BaseDecorator.Execute(ctx, request)
	if err == nil {
		cd.cache.Set(cacheKey, result, cd.config.TT)
	}

	return result, err
}

// generateCacheKey generates a cache key for the request
func (cd *CacheDecorator) generateCacheKey(request interface{}) string {
	return fmt.Sprintf("%s:%v", cd.GetName(), request)
}

// SecurityDecorator adds security functionality
type SecurityDecorator struct {
	*BaseDecorator
	security Security
	config   SecurityConfig
}

// NewSecurityDecorator creates a new security decorator
func NewSecurityDecorator(security Security, config SecurityConfig) *SecurityDecorator {
	return &SecurityDecorator{
		BaseDecorator: NewBaseDecorator("security", "Adds security functionality"),
		security:      security,
		config:        config,
	}
}

// Execute executes with security checks
func (sd *SecurityDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !sd.config.Enabled {
		return sd.BaseDecorator.Execute(ctx, request)
	}

	// Validate input
	if sd.config.ValidateInput {
		if err := sd.security.ValidateInput(request); err != nil {
			return nil, fmt.Errorf("input validation failed: %w", err)
		}
	}

	// Sanitize input
	if sd.config.SanitizeInput {
		request = sd.security.SanitizeInput(request)
	}

	// Execute with security
	result, err := sd.BaseDecorator.Execute(ctx, request)

	// Audit log
	if sd.config.AuditLogging {
		sd.security.AuditLog(ctx, "execute", "system", map[string]interface{}{
			"component": sd.GetName(),
			"success":   err == nil,
		})
	}

	return result, err
}

// RateLimitDecorator adds rate limiting functionality
type RateLimitDecorator struct {
	*BaseDecorator
	rateLimiter RateLimiter
	config      RateLimitConfig
}

// NewRateLimitDecorator creates a new rate limit decorator
func NewRateLimitDecorator(rateLimiter RateLimiter, config RateLimitConfig) *RateLimitDecorator {
	return &RateLimitDecorator{
		BaseDecorator: NewBaseDecorator("rate_limit", "Adds rate limiting functionality"),
		rateLimiter:   rateLimiter,
		config:        config,
	}
}

// Execute executes with rate limiting
func (rld *RateLimitDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !rld.config.Enabled {
		return rld.BaseDecorator.Execute(ctx, request)
	}

	// Generate rate limit key
	key := rld.generateRateLimitKey(request)

	// Check rate limit
	if !rld.rateLimiter.Allow(key) {
		return nil, fmt.Errorf("rate limit exceeded")
	}

	return rld.BaseDecorator.Execute(ctx, request)
}

// generateRateLimitKey generates a rate limit key for the request
func (rld *RateLimitDecorator) generateRateLimitKey(request interface{}) string {
	return fmt.Sprintf("%s:%v", rld.GetName(), request)
}

// CircuitBreakerDecorator adds circuit breaker functionality
type CircuitBreakerDecorator struct {
	*BaseDecorator
	circuitBreaker CircuitBreaker
	config         CircuitBreakerConfig
}

// NewCircuitBreakerDecorator creates a new circuit breaker decorator
func NewCircuitBreakerDecorator(circuitBreaker CircuitBreaker, config CircuitBreakerConfig) *CircuitBreakerDecorator {
	return &CircuitBreakerDecorator{
		BaseDecorator:  NewBaseDecorator("circuit_breaker", "Adds circuit breaker functionality"),
		circuitBreaker: circuitBreaker,
		config:         config,
	}
}

// Execute executes with circuit breaker
func (cbd *CircuitBreakerDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !cbd.config.Enabled {
		return cbd.BaseDecorator.Execute(ctx, request)
	}

	return cbd.circuitBreaker.Execute(ctx, func() (interface{}, error) {
		return cbd.BaseDecorator.Execute(ctx, request)
	})
}

// RetryDecorator adds retry functionality
type RetryDecorator struct {
	*BaseDecorator
	retry  Retry
	config RetryConfig
}

// NewRetryDecorator creates a new retry decorator
func NewRetryDecorator(retry Retry, config RetryConfig) *RetryDecorator {
	return &RetryDecorator{
		BaseDecorator: NewBaseDecorator("retry", "Adds retry functionality"),
		retry:         retry,
		config:        config,
	}
}

// Execute executes with retry logic
func (rd *RetryDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !rd.config.Enabled {
		return rd.BaseDecorator.Execute(ctx, request)
	}

	return rd.retry.Execute(ctx, func() (interface{}, error) {
		return rd.BaseDecorator.Execute(ctx, request)
	})
}

// MonitoringDecorator adds monitoring functionality
type MonitoringDecorator struct {
	*BaseDecorator
	monitoring Monitoring
	config     MonitoringConfig
}

// NewMonitoringDecorator creates a new monitoring decorator
func NewMonitoringDecorator(monitoring Monitoring, config MonitoringConfig) *MonitoringDecorator {
	return &MonitoringDecorator{
		BaseDecorator: NewBaseDecorator("monitoring", "Adds monitoring functionality"),
		monitoring:    monitoring,
		config:        config,
	}
}

// Execute executes with monitoring
func (md *MonitoringDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !md.config.Enabled {
		return md.BaseDecorator.Execute(ctx, request)
	}

	start := time.Now()
	result, err := md.BaseDecorator.Execute(ctx, request)
	duration := time.Since(start)

	md.monitoring.RecordRequest(ctx, md.GetName(), duration, err == nil)
	if err != nil {
		md.monitoring.RecordError(ctx, md.GetName(), err)
	}

	return result, err
}

// ValidationDecorator adds validation functionality
type ValidationDecorator struct {
	*BaseDecorator
	validation Validation
	config     ValidationConfig
}

// NewValidationDecorator creates a new validation decorator
func NewValidationDecorator(validation Validation, config ValidationConfig) *ValidationDecorator {
	return &ValidationDecorator{
		BaseDecorator: NewBaseDecorator("validation", "Adds validation functionality"),
		validation:    validation,
		config:        config,
	}
}

// Execute executes with validation
func (vd *ValidationDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !vd.config.Enabled {
		return vd.BaseDecorator.Execute(ctx, request)
	}

	// Validate request
	if err := vd.validation.Validate(ctx, request); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	return vd.BaseDecorator.Execute(ctx, request)
}

// EncryptionDecorator adds encryption functionality
type EncryptionDecorator struct {
	*BaseDecorator
	encryption Encryption
	config     EncryptionConfig
}

// NewEncryptionDecorator creates a new encryption decorator
func NewEncryptionDecorator(encryption Encryption, config EncryptionConfig) *EncryptionDecorator {
	return &EncryptionDecorator{
		BaseDecorator: NewBaseDecorator("encryption", "Adds encryption functionality"),
		encryption:    encryption,
		config:        config,
	}
}

// Execute executes with encryption
func (ed *EncryptionDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !ed.config.Enabled {
		return ed.BaseDecorator.Execute(ctx, request)
	}

	// Encrypt request data if it's a byte slice
	if data, ok := request.([]byte); ok {
		encrypted, err := ed.encryption.Encrypt(data)
		if err != nil {
			return nil, fmt.Errorf("encryption failed: %w", err)
		}
		request = encrypted
	}

	result, err := ed.BaseDecorator.Execute(ctx, request)
	if err != nil {
		return nil, err
	}

	// Decrypt result if it's a byte slice
	if data, ok := result.([]byte); ok {
		decrypted, err := ed.encryption.Decrypt(data)
		if err != nil {
			return nil, fmt.Errorf("decryption failed: %w", err)
		}
		result = decrypted
	}

	return result, err
}

// CompressionDecorator adds compression functionality
type CompressionDecorator struct {
	*BaseDecorator
	compression Compression
	config      CompressionConfig
}

// NewCompressionDecorator creates a new compression decorator
func NewCompressionDecorator(compression Compression, config CompressionConfig) *CompressionDecorator {
	return &CompressionDecorator{
		BaseDecorator: NewBaseDecorator("compression", "Adds compression functionality"),
		compression:   compression,
		config:        config,
	}
}

// Execute executes with compression
func (cd *CompressionDecorator) Execute(ctx context.Context, request interface{}) (interface{}, error) {
	if !cd.config.Enabled {
		return cd.BaseDecorator.Execute(ctx, request)
	}

	// Compress request data if it's a byte slice
	if data, ok := request.([]byte); ok && len(data) >= cd.config.MinSize {
		compressed, err := cd.compression.Compress(data)
		if err != nil {
			return nil, fmt.Errorf("compression failed: %w", err)
		}
		request = compressed
	}

	result, err := cd.BaseDecorator.Execute(ctx, request)
	if err != nil {
		return nil, err
	}

	// Decompress result if it's a byte slice
	if data, ok := result.([]byte); ok {
		decompressed, err := cd.compression.Decompress(data)
		if err != nil {
			return nil, fmt.Errorf("decompression failed: %w", err)
		}
		result = decompressed
	}

	return result, err
}
