package facade

import (
	"context"
	"time"
)

// PaymentService defines the interface for payment operations
type PaymentService interface {
	ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error)
	RefundPayment(ctx context.Context, transactionID string, amount float64) error
	GetPaymentStatus(ctx context.Context, transactionID string) (*PaymentStatus, error)
	GetPaymentHistory(ctx context.Context, userID string) ([]*PaymentHistory, error)
}

// NotificationService defines the interface for notification operations
type NotificationService interface {
	SendEmail(ctx context.Context, request EmailRequest) error
	SendSMS(ctx context.Context, request SMSRequest) error
	SendPushNotification(ctx context.Context, request PushRequest) error
	SendBulkNotification(ctx context.Context, request BulkNotificationRequest) error
}

// UserService defines the interface for user operations
type UserService interface {
	CreateUser(ctx context.Context, request CreateUserRequest) (*User, error)
	GetUser(ctx context.Context, userID string) (*User, error)
	UpdateUser(ctx context.Context, userID string, request UpdateUserRequest) (*User, error)
	DeleteUser(ctx context.Context, userID string) error
	AuthenticateUser(ctx context.Context, request AuthRequest) (*AuthResponse, error)
}

// OrderService defines the interface for order operations
type OrderService interface {
	CreateOrder(ctx context.Context, request CreateOrderRequest) (*Order, error)
	GetOrder(ctx context.Context, orderID string) (*Order, error)
	UpdateOrder(ctx context.Context, orderID string, request UpdateOrderRequest) (*Order, error)
	CancelOrder(ctx context.Context, orderID string) error
	GetOrderHistory(ctx context.Context, userID string) ([]*Order, error)
}

// InventoryService defines the interface for inventory operations
type InventoryService interface {
	CheckAvailability(ctx context.Context, productID string, quantity int) (bool, error)
	ReserveProduct(ctx context.Context, productID string, quantity int) error
	ReleaseProduct(ctx context.Context, productID string, quantity int) error
	UpdateStock(ctx context.Context, productID string, quantity int) error
	GetProductInfo(ctx context.Context, productID string) (*Product, error)
}

// AnalyticsService defines the interface for analytics operations
type AnalyticsService interface {
	TrackEvent(ctx context.Context, event Event) error
	GetUserAnalytics(ctx context.Context, userID string) (*UserAnalytics, error)
	GetProductAnalytics(ctx context.Context, productID string) (*ProductAnalytics, error)
	GetSalesReport(ctx context.Context, request ReportRequest) (*SalesReport, error)
}

// AuditService defines the interface for audit operations
type AuditService interface {
	LogAction(ctx context.Context, action AuditAction) error
	GetAuditLogs(ctx context.Context, request AuditLogRequest) ([]*AuditLog, error)
	GetUserActivity(ctx context.Context, userID string) ([]*AuditLog, error)
}

// CacheService defines the interface for cache operations
type CacheService interface {
	Get(ctx context.Context, key string) (interface{}, error)
	Set(ctx context.Context, key string, value interface{}, expiration time.Duration) error
	Delete(ctx context.Context, key string) error
	Clear(ctx context.Context) error
	GetStats() CacheStats
}

// DatabaseService defines the interface for database operations
type DatabaseService interface {
	Save(ctx context.Context, collection string, data interface{}) error
	Find(ctx context.Context, collection string, query interface{}) (interface{}, error)
	Update(ctx context.Context, collection string, id string, data interface{}) error
	Delete(ctx context.Context, collection string, id string) error
	Transaction(ctx context.Context, fn func(ctx context.Context) error) error
}

// MessageQueueService defines the interface for message queue operations
type MessageQueueService interface {
	Publish(ctx context.Context, topic string, message interface{}) error
	Subscribe(ctx context.Context, topic string, handler func(interface{}) error) error
	Close() error
	GetStats() MessageQueueStats
}

// WebSocketService defines the interface for WebSocket operations
type WebSocketService interface {
	Send(ctx context.Context, clientID string, message interface{}) error
	Broadcast(ctx context.Context, message interface{}) error
	Register(ctx context.Context, clientID string, conn interface{}) error
	Unregister(ctx context.Context, clientID string) error
	GetStats() WebSocketStats
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

// SecurityService defines the interface for security operations
type SecurityService interface {
	ValidateToken(ctx context.Context, token string) (*TokenClaims, error)
	GenerateToken(ctx context.Context, userID string) (string, error)
	CheckPermission(ctx context.Context, userID string, resource string, action string) bool
	EncryptData(ctx context.Context, data []byte) ([]byte, error)
	DecryptData(ctx context.Context, encryptedData []byte) ([]byte, error)
}

// ConfigurationService defines the interface for configuration operations
type ConfigurationService interface {
	GetConfig(ctx context.Context, key string) (interface{}, error)
	SetConfig(ctx context.Context, key string, value interface{}) error
	GetAllConfigs(ctx context.Context) (map[string]interface{}, error)
	ReloadConfig(ctx context.Context) error
}

// HealthService defines the interface for health check operations
type HealthService interface {
	CheckHealth(ctx context.Context) (*HealthStatus, error)
	CheckServiceHealth(ctx context.Context, serviceName string) (*ServiceHealth, error)
	GetAllServicesHealth(ctx context.Context) ([]*ServiceHealth, error)
}

// MonitoringService defines the interface for monitoring operations
type MonitoringService interface {
	RecordRequest(ctx context.Context, service string, duration time.Duration, success bool)
	RecordError(ctx context.Context, service string, err error)
	GetServiceMetrics(ctx context.Context, service string) (*ServiceMetrics, error)
	GetSystemMetrics(ctx context.Context) (*SystemMetrics, error)
}
