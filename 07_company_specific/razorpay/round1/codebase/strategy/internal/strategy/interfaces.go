package strategy

import (
	"context"
	"time"
)

// PaymentStrategy defines the interface for payment processing strategies
type PaymentStrategy interface {
	ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error)
	ValidatePayment(ctx context.Context, request PaymentRequest) error
	GetStrategyName() string
	GetSupportedCurrencies() []string
	GetProcessingTime() time.Duration
	IsAvailable() bool
}

// NotificationStrategy defines the interface for notification strategies
type NotificationStrategy interface {
	SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error)
	ValidateNotification(ctx context.Context, request NotificationRequest) error
	GetStrategyName() string
	GetSupportedChannels() []string
	GetDeliveryTime() time.Duration
	IsAvailable() bool
}

// PricingStrategy defines the interface for pricing strategies
type PricingStrategy interface {
	CalculatePrice(ctx context.Context, request PricingRequest) (*PricingResponse, error)
	ValidatePricing(ctx context.Context, request PricingRequest) error
	GetStrategyName() string
	GetSupportedProducts() []string
	GetCalculationTime() time.Duration
	IsAvailable() bool
}

// AuthenticationStrategy defines the interface for authentication strategies
type AuthenticationStrategy interface {
	Authenticate(ctx context.Context, request AuthRequest) (*AuthResponse, error)
	ValidateAuth(ctx context.Context, request AuthRequest) error
	GetStrategyName() string
	GetSupportedMethods() []string
	GetAuthTime() time.Duration
	IsAvailable() bool
}

// CachingStrategy defines the interface for caching strategies
type CachingStrategy interface {
	Get(ctx context.Context, key string) (interface{}, error)
	Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Clear(ctx context.Context) error
	GetStrategyName() string
	GetSupportedTypes() []string
	GetAccessTime() time.Duration
	IsAvailable() bool
}

// LoggingStrategy defines the interface for logging strategies
type LoggingStrategy interface {
	Log(ctx context.Context, level LogLevel, message string, fields map[string]interface{}) error
	GetStrategyName() string
	GetSupportedLevels() []LogLevel
	GetLogTime() time.Duration
	IsAvailable() bool
}

// DataProcessingStrategy defines the interface for data processing strategies
type DataProcessingStrategy interface {
	ProcessData(ctx context.Context, data interface{}) (interface{}, error)
	ValidateData(ctx context.Context, data interface{}) error
	GetStrategyName() string
	GetSupportedFormats() []string
	GetProcessingTime() time.Duration
	IsAvailable() bool
}

// StrategyManager manages multiple strategies of the same type
type StrategyManager interface {
	GetStrategy(strategyName string) (interface{}, error)
	GetAvailableStrategies() []string
	RegisterStrategy(strategyName string, strategy interface{}) error
	UnregisterStrategy(strategyName string) error
	GetDefaultStrategy() (interface{}, error)
	SetDefaultStrategy(strategyName string) error
}

// StrategyFactory creates strategies based on configuration
type StrategyFactory interface {
	CreatePaymentStrategy(strategyType string) (PaymentStrategy, error)
	CreateNotificationStrategy(strategyType string) (NotificationStrategy, error)
	CreatePricingStrategy(strategyType string) (PricingStrategy, error)
	CreateAuthenticationStrategy(strategyType string) (AuthenticationStrategy, error)
	CreateCachingStrategy(strategyType string) (CachingStrategy, error)
	CreateLoggingStrategy(strategyType string) (LoggingStrategy, error)
	CreateDataProcessingStrategy(strategyType string) (DataProcessingStrategy, error)
}

// StrategySelector selects the best strategy based on context
type StrategySelector interface {
	SelectPaymentStrategy(ctx context.Context, request PaymentRequest) (PaymentStrategy, error)
	SelectNotificationStrategy(ctx context.Context, request NotificationRequest) (NotificationStrategy, error)
	SelectPricingStrategy(ctx context.Context, request PricingRequest) (PricingStrategy, error)
	SelectAuthenticationStrategy(ctx context.Context, request AuthRequest) (AuthenticationStrategy, error)
	SelectCachingStrategy(ctx context.Context, key string) (CachingStrategy, error)
	SelectLoggingStrategy(ctx context.Context, level LogLevel) (LoggingStrategy, error)
	SelectDataProcessingStrategy(ctx context.Context, data interface{}) (DataProcessingStrategy, error)
}

// StrategyMetrics collects metrics for strategy performance
type StrategyMetrics interface {
	RecordStrategyCall(strategyName string, duration time.Duration, success bool)
	GetStrategyMetrics(strategyName string) (*StrategyMetricsData, error)
	GetAllMetrics() (map[string]*StrategyMetricsData, error)
	ResetMetrics(strategyName string) error
	ResetAllMetrics() error
}

// StrategyConfig holds configuration for strategies
type StrategyConfig struct {
	DefaultStrategy  string               `json:"default_strategy"`
	Strategies       map[string]Config    `json:"strategies"`
	FallbackStrategy string               `json:"fallback_strategy"`
	Timeout          time.Duration        `json:"timeout"`
	RetryCount       int                  `json:"retry_count"`
	CircuitBreaker   CircuitBreakerConfig `json:"circuit_breaker"`
}

// Config holds configuration for a specific strategy
type Config struct {
	Enabled    bool              `json:"enabled"`
	Priority   int               `json:"priority"`
	Timeout    time.Duration     `json:"timeout"`
	RetryCount int               `json:"retry_count"`
	Parameters map[string]string `json:"parameters"`
	Fallback   string            `json:"fallback"`
}

// CircuitBreakerConfig holds circuit breaker configuration
type CircuitBreakerConfig struct {
	Enabled          bool          `json:"enabled"`
	FailureThreshold int           `json:"failure_threshold"`
	RecoveryTimeout  time.Duration `json:"recovery_timeout"`
	HalfOpenMaxCalls int           `json:"half_open_max_calls"`
}

// StrategyMetricsData holds metrics for a strategy
type StrategyMetricsData struct {
	StrategyName    string        `json:"strategy_name"`
	TotalCalls      int64         `json:"total_calls"`
	SuccessfulCalls int64         `json:"successful_calls"`
	FailedCalls     int64         `json:"failed_calls"`
	AverageDuration time.Duration `json:"average_duration"`
	MinDuration     time.Duration `json:"min_duration"`
	MaxDuration     time.Duration `json:"max_duration"`
	LastCallTime    time.Time     `json:"last_call_time"`
	SuccessRate     float64       `json:"success_rate"`
	Availability    float64       `json:"availability"`
}

// LogLevel represents logging levels
type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
	LogLevelFatal
)

// String returns the string representation of LogLevel
func (l LogLevel) String() string {
	switch l {
	case LogLevelDebug:
		return "DEBUG"
	case LogLevelInfo:
		return "INFO"
	case LogLevelWarn:
		return "WARN"
	case LogLevelError:
		return "ERROR"
	case LogLevelFatal:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}
