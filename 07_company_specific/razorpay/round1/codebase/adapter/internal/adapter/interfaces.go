package adapter

import (
	"context"
	"time"
)

// PaymentGateway defines the interface for payment gateways
type PaymentGateway interface {
	ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error)
	RefundPayment(ctx context.Context, request RefundRequest) (*RefundResponse, error)
	GetPaymentStatus(ctx context.Context, paymentID string) (*PaymentStatus, error)
	GetGatewayName() string
	IsAvailable() bool
}

// NotificationService defines the interface for notification services
type NotificationService interface {
	SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error)
	GetNotificationStatus(ctx context.Context, notificationID string) (*NotificationStatus, error)
	GetServiceName() string
	IsAvailable() bool
}

// DatabaseAdapter defines the interface for database adapters
type DatabaseAdapter interface {
	Connect(ctx context.Context) error
	Disconnect(ctx context.Context) error
	Query(ctx context.Context, query string, args ...interface{}) ([]map[string]interface{}, error)
	Execute(ctx context.Context, query string, args ...interface{}) (int64, error)
	BeginTransaction(ctx context.Context) (Transaction, error)
	GetAdapterName() string
	IsConnected() bool
}

// Transaction defines the interface for database transactions
type Transaction interface {
	Commit(ctx context.Context) error
	Rollback(ctx context.Context) error
	Query(ctx context.Context, query string, args ...interface{}) ([]map[string]interface{}, error)
	Execute(ctx context.Context, query string, args ...interface{}) (int64, error)
}

// CacheAdapter defines the interface for cache adapters
type CacheAdapter interface {
	Get(ctx context.Context, key string) (interface{}, error)
	Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Clear(ctx context.Context) error
	GetAdapterName() string
	IsConnected() bool
}

// MessageQueueAdapter defines the interface for message queue adapters
type MessageQueueAdapter interface {
	Publish(ctx context.Context, topic string, message interface{}) error
	Subscribe(ctx context.Context, topic string, handler MessageHandler) error
	Unsubscribe(ctx context.Context, topic string) error
	GetAdapterName() string
	IsConnected() bool
}

// MessageHandler handles messages from message queues
type MessageHandler interface {
	HandleMessage(ctx context.Context, message interface{}) error
	GetHandlerName() string
}

// FileStorageAdapter defines the interface for file storage adapters
type FileStorageAdapter interface {
	Upload(ctx context.Context, file File) (*UploadResponse, error)
	Download(ctx context.Context, fileID string) (*File, error)
	Delete(ctx context.Context, fileID string) error
	List(ctx context.Context, prefix string) ([]FileInfo, error)
	GetAdapterName() string
	IsAvailable() bool
}

// AuthenticationAdapter defines the interface for authentication adapters
type AuthenticationAdapter interface {
	Authenticate(ctx context.Context, credentials Credentials) (*AuthResponse, error)
	ValidateToken(ctx context.Context, token string) (*TokenValidation, error)
	RefreshToken(ctx context.Context, refreshToken string) (*AuthResponse, error)
	GetAdapterName() string
	IsAvailable() bool
}

// AdapterManager manages multiple adapters
type AdapterManager interface {
	RegisterAdapter(adapterType string, adapter interface{}) error
	UnregisterAdapter(adapterType string, adapterName string) error
	GetAdapter(adapterType string, adapterName string) (interface{}, error)
	GetAdapters(adapterType string) ([]interface{}, error)
	GetAllAdapters() map[string][]interface{}
	GetAdapterHealth(adapterType string, adapterName string) (*AdapterHealth, error)
}

// AdapterFactory creates adapters based on configuration
type AdapterFactory interface {
	CreatePaymentGateway(gatewayType string) (PaymentGateway, error)
	CreateNotificationService(serviceType string) (NotificationService, error)
	CreateDatabaseAdapter(dbType string) (DatabaseAdapter, error)
	CreateCacheAdapter(cacheType string) (CacheAdapter, error)
	CreateMessageQueueAdapter(mqType string) (MessageQueueAdapter, error)
	CreateFileStorageAdapter(storageType string) (FileStorageAdapter, error)
	CreateAuthenticationAdapter(authType string) (AuthenticationAdapter, error)
}

// AdapterMetrics collects adapter performance metrics
type AdapterMetrics interface {
	RecordAdapterCall(adapterType string, adapterName string, duration time.Duration, success bool)
	GetAdapterMetrics(adapterType string, adapterName string) (*AdapterMetricsData, error)
	GetAllMetrics() (map[string]map[string]*AdapterMetricsData, error)
	ResetMetrics(adapterType string, adapterName string) error
	ResetAllMetrics() error
}

// AdapterConfig holds configuration for adapters
type AdapterConfig struct {
	Adapters       map[string]AdapterTypeConfig `json:"adapters"`
	DefaultTimeout time.Duration                `json:"default_timeout"`
	MaxRetries     int                          `json:"max_retries"`
	RetryDelay     time.Duration                `json:"retry_delay"`
	CircuitBreaker CircuitBreakerConfig         `json:"circuit_breaker"`
}

// AdapterTypeConfig holds configuration for a specific adapter type
type AdapterTypeConfig struct {
	DefaultAdapter string                           `json:"default_adapter"`
	Adapters       map[string]AdapterInstanceConfig `json:"adapters"`
}

// AdapterInstanceConfig holds configuration for a specific adapter instance
type AdapterInstanceConfig struct {
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

// AdapterHealth represents the health status of an adapter
type AdapterHealth struct {
	AdapterType string                 `json:"adapter_type"`
	AdapterName string                 `json:"adapter_name"`
	Status      string                 `json:"status"`
	Message     string                 `json:"message"`
	LastCheck   time.Time              `json:"last_check"`
	Metrics     map[string]interface{} `json:"metrics"`
	Error       string                 `json:"error,omitempty"`
}

// AdapterMetricsData holds metrics for an adapter
type AdapterMetricsData struct {
	AdapterType     string        `json:"adapter_type"`
	AdapterName     string        `json:"adapter_name"`
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

// AdapterStatus represents adapter status
type AdapterStatus string

const (
	AdapterStatusActive      AdapterStatus = "active"
	AdapterStatusInactive    AdapterStatus = "inactive"
	AdapterStatusError       AdapterStatus = "error"
	AdapterStatusUnavailable AdapterStatus = "unavailable"
)

// String returns the string representation of AdapterStatus
func (as AdapterStatus) String() string {
	return string(as)
}
