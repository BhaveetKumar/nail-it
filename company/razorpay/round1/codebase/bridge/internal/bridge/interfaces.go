package bridge

import (
	"context"
	"time"
)

// MessageSender defines the interface for message sending
type MessageSender interface {
	Send(ctx context.Context, message Message) error
	GetSenderType() string
	IsAvailable() bool
}

// MessageReceiver defines the interface for message receiving
type MessageReceiver interface {
	Receive(ctx context.Context) (Message, error)
	GetReceiverType() string
	IsAvailable() bool
}

// Message defines the interface for messages
type Message interface {
	GetID() string
	GetType() string
	GetContent() interface{}
	GetMetadata() map[string]string
	GetTimestamp() time.Time
	SetID(id string)
	SetType(msgType string)
	SetContent(content interface{})
	SetMetadata(metadata map[string]string)
	SetTimestamp(timestamp time.Time)
}

// NotificationBridge defines the interface for notification bridges
type NotificationBridge interface {
	SendNotification(ctx context.Context, notification Notification) error
	GetNotificationStatus(ctx context.Context, notificationID string) (*NotificationStatus, error)
	GetBridgeType() string
	IsAvailable() bool
}

// PaymentBridge defines the interface for payment bridges
type PaymentBridge interface {
	ProcessPayment(ctx context.Context, payment Payment) error
	RefundPayment(ctx context.Context, refund Refund) error
	GetPaymentStatus(ctx context.Context, paymentID string) (*PaymentStatus, error)
	GetBridgeType() string
	IsAvailable() bool
}

// DatabaseBridge defines the interface for database bridges
type DatabaseBridge interface {
	Connect(ctx context.Context) error
	Disconnect(ctx context.Context) error
	Query(ctx context.Context, query string, args ...interface{}) ([]map[string]interface{}, error)
	Execute(ctx context.Context, query string, args ...interface{}) (int64, error)
	BeginTransaction(ctx context.Context) (Transaction, error)
	GetBridgeType() string
	IsConnected() bool
}

// Transaction defines the interface for database transactions
type Transaction interface {
	Commit(ctx context.Context) error
	Rollback(ctx context.Context) error
	Query(ctx context.Context, query string, args ...interface{}) ([]map[string]interface{}, error)
	Execute(ctx context.Context, query string, args ...interface{}) (int64, error)
}

// CacheBridge defines the interface for cache bridges
type CacheBridge interface {
	Get(ctx context.Context, key string) (interface{}, error)
	Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Clear(ctx context.Context) error
	GetBridgeType() string
	IsConnected() bool
}

// FileStorageBridge defines the interface for file storage bridges
type FileStorageBridge interface {
	Upload(ctx context.Context, file File) (*UploadResponse, error)
	Download(ctx context.Context, fileID string) (*File, error)
	Delete(ctx context.Context, fileID string) error
	List(ctx context.Context, prefix string) ([]FileInfo, error)
	GetBridgeType() string
	IsAvailable() bool
}

// AuthenticationBridge defines the interface for authentication bridges
type AuthenticationBridge interface {
	Authenticate(ctx context.Context, credentials Credentials) (*AuthResponse, error)
	ValidateToken(ctx context.Context, token string) (*TokenValidation, error)
	RefreshToken(ctx context.Context, refreshToken string) (*AuthResponse, error)
	GetBridgeType() string
	IsAvailable() bool
}

// BridgeManager manages multiple bridges
type BridgeManager interface {
	RegisterBridge(bridgeType string, bridge interface{}) error
	UnregisterBridge(bridgeType string, bridgeName string) error
	GetBridge(bridgeType string, bridgeName string) (interface{}, error)
	GetBridges(bridgeType string) ([]interface{}, error)
	GetAllBridges() map[string][]interface{}
	GetBridgeHealth(bridgeType string, bridgeName string) (*BridgeHealth, error)
}

// BridgeFactory creates bridges based on configuration
type BridgeFactory interface {
	CreateNotificationBridge(bridgeType string) (NotificationBridge, error)
	CreatePaymentBridge(bridgeType string) (PaymentBridge, error)
	CreateDatabaseBridge(bridgeType string) (DatabaseBridge, error)
	CreateCacheBridge(bridgeType string) (CacheBridge, error)
	CreateFileStorageBridge(bridgeType string) (FileStorageBridge, error)
	CreateAuthenticationBridge(bridgeType string) (AuthenticationBridge, error)
}

// BridgeMetrics collects bridge performance metrics
type BridgeMetrics interface {
	RecordBridgeCall(bridgeType string, bridgeName string, duration time.Duration, success bool)
	GetBridgeMetrics(bridgeType string, bridgeName string) (*BridgeMetricsData, error)
	GetAllMetrics() (map[string]map[string]*BridgeMetricsData, error)
	ResetMetrics(bridgeType string, bridgeName string) error
	ResetAllMetrics() error
}

// BridgeConfig holds configuration for bridges
type BridgeConfig struct {
	Bridges        map[string]BridgeTypeConfig `json:"bridges"`
	DefaultTimeout time.Duration               `json:"default_timeout"`
	MaxRetries     int                         `json:"max_retries"`
	RetryDelay     time.Duration               `json:"retry_delay"`
	CircuitBreaker CircuitBreakerConfig        `json:"circuit_breaker"`
}

// BridgeTypeConfig holds configuration for a specific bridge type
type BridgeTypeConfig struct {
	DefaultBridge string                          `json:"default_bridge"`
	Bridges       map[string]BridgeInstanceConfig `json:"bridges"`
}

// BridgeInstanceConfig holds configuration for a specific bridge instance
type BridgeInstanceConfig struct {
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

// BridgeHealth represents the health status of a bridge
type BridgeHealth struct {
	BridgeType string                 `json:"bridge_type"`
	BridgeName string                 `json:"bridge_name"`
	Status     string                 `json:"status"`
	Message    string                 `json:"message"`
	LastCheck  time.Time              `json:"last_check"`
	Metrics    map[string]interface{} `json:"metrics"`
	Error      string                 `json:"error,omitempty"`
}

// BridgeMetricsData holds metrics for a bridge
type BridgeMetricsData struct {
	BridgeType      string        `json:"bridge_type"`
	BridgeName      string        `json:"bridge_name"`
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

// BridgeStatus represents bridge status
type BridgeStatus string

const (
	BridgeStatusActive      BridgeStatus = "active"
	BridgeStatusInactive    BridgeStatus = "inactive"
	BridgeStatusError       BridgeStatus = "error"
	BridgeStatusUnavailable BridgeStatus = "unavailable"
)

// String returns the string representation of BridgeStatus
func (bs BridgeStatus) String() string {
	return string(bs)
}
