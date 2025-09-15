package facade

import (
	"time"
)

// PaymentRequest represents a payment request
type PaymentRequest struct {
	ID          string                 `json:"id"`
	Amount      float64                `json:"amount"`
	Currency    string                 `json:"currency"`
	CustomerID  string                 `json:"customer_id"`
	MerchantID  string                 `json:"merchant_id"`
	Description string                 `json:"description"`
	PaymentMethod string               `json:"payment_method"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
}

// PaymentResponse represents a payment response
type PaymentResponse struct {
	ID            string                 `json:"id"`
	Status        string                 `json:"status"`
	TransactionID string                 `json:"transaction_id"`
	Amount        float64                `json:"amount"`
	Currency      string                 `json:"currency"`
	Gateway       string                 `json:"gateway"`
	Metadata      map[string]interface{} `json:"metadata"`
	CreatedAt     time.Time              `json:"created_at"`
	ProcessedAt   time.Time              `json:"processed_at"`
}

// PaymentStatus represents payment status
type PaymentStatus struct {
	TransactionID string    `json:"transaction_id"`
	Status        string    `json:"status"`
	Amount        float64   `json:"amount"`
	Currency      string    `json:"currency"`
	LastUpdated   time.Time `json:"last_updated"`
}

// PaymentHistory represents payment history
type PaymentHistory struct {
	ID            string    `json:"id"`
	Amount        float64   `json:"amount"`
	Currency      string    `json:"currency"`
	Status        string    `json:"status"`
	Description   string    `json:"description"`
	CreatedAt     time.Time `json:"created_at"`
}

// EmailRequest represents an email request
type EmailRequest struct {
	To      string                 `json:"to"`
	Subject string                 `json:"subject"`
	Body    string                 `json:"body"`
	Type    string                 `json:"type"`
	Data    map[string]interface{} `json:"data"`
}

// SMSRequest represents an SMS request
type SMSRequest struct {
	To      string                 `json:"to"`
	Message string                 `json:"message"`
	Type    string                 `json:"type"`
	Data    map[string]interface{} `json:"data"`
}

// PushRequest represents a push notification request
type PushRequest struct {
	UserID  string                 `json:"user_id"`
	Title   string                 `json:"title"`
	Message string                 `json:"message"`
	Type    string                 `json:"type"`
	Data    map[string]interface{} `json:"data"`
}

// BulkNotificationRequest represents a bulk notification request
type BulkNotificationRequest struct {
	Recipients []string               `json:"recipients"`
	Type       string                 `json:"type"`
	Title      string                 `json:"title"`
	Message    string                 `json:"message"`
	Data       map[string]interface{} `json:"data"`
}

// CreateUserRequest represents a create user request
type CreateUserRequest struct {
	Username string                 `json:"username"`
	Email    string                 `json:"email"`
	Password string                 `json:"password"`
	Profile  map[string]interface{} `json:"profile"`
}

// UpdateUserRequest represents an update user request
type UpdateUserRequest struct {
	Username string                 `json:"username,omitempty"`
	Email    string                 `json:"email,omitempty"`
	Profile  map[string]interface{} `json:"profile,omitempty"`
}

// User represents a user
type User struct {
	ID        string                 `json:"id"`
	Username  string                 `json:"username"`
	Email     string                 `json:"email"`
	Profile   map[string]interface{} `json:"profile"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

// AuthRequest represents an authentication request
type AuthRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// AuthResponse represents an authentication response
type AuthResponse struct {
	Token     string    `json:"token"`
	User      *User     `json:"user"`
	ExpiresAt time.Time `json:"expires_at"`
}

// CreateOrderRequest represents a create order request
type CreateOrderRequest struct {
	UserID      string                 `json:"user_id"`
	Items       []OrderItem            `json:"items"`
	ShippingAddress map[string]interface{} `json:"shipping_address"`
	BillingAddress  map[string]interface{} `json:"billing_address"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// UpdateOrderRequest represents an update order request
type UpdateOrderRequest struct {
	Status         string                 `json:"status,omitempty"`
	ShippingAddress map[string]interface{} `json:"shipping_address,omitempty"`
	BillingAddress  map[string]interface{} `json:"billing_address,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

// OrderItem represents an order item
type OrderItem struct {
	ProductID string  `json:"product_id"`
	Quantity  int     `json:"quantity"`
	Price     float64 `json:"price"`
}

// Order represents an order
type Order struct {
	ID              string                 `json:"id"`
	UserID          string                 `json:"user_id"`
	Items           []OrderItem            `json:"items"`
	Total           float64                `json:"total"`
	Status          string                 `json:"status"`
	ShippingAddress map[string]interface{} `json:"shipping_address"`
	BillingAddress  map[string]interface{} `json:"billing_address"`
	Metadata        map[string]interface{} `json:"metadata"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// Product represents a product
type Product struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Price       float64                `json:"price"`
	Stock       int                    `json:"stock"`
	Category    string                 `json:"category"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// Event represents an analytics event
type Event struct {
	ID         string                 `json:"id"`
	UserID     string                 `json:"user_id"`
	EventType  string                 `json:"event_type"`
	Properties map[string]interface{} `json:"properties"`
	Timestamp  time.Time              `json:"timestamp"`
}

// UserAnalytics represents user analytics
type UserAnalytics struct {
	UserID           string    `json:"user_id"`
	TotalEvents      int64     `json:"total_events"`
	LastActivity     time.Time `json:"last_activity"`
	FavoriteCategory string    `json:"favorite_category"`
	TotalSpent       float64   `json:"total_spent"`
}

// ProductAnalytics represents product analytics
type ProductAnalytics struct {
	ProductID     string    `json:"product_id"`
	Views         int64     `json:"views"`
	Purchases     int64     `json:"purchases"`
	Revenue       float64   `json:"revenue"`
	LastViewed    time.Time `json:"last_viewed"`
	ConversionRate float64  `json:"conversion_rate"`
}

// ReportRequest represents a report request
type ReportRequest struct {
	StartDate time.Time `json:"start_date"`
	EndDate   time.Time `json:"end_date"`
	GroupBy   string    `json:"group_by"`
	Filters   map[string]interface{} `json:"filters"`
}

// SalesReport represents a sales report
type SalesReport struct {
	Period     string    `json:"period"`
	TotalSales float64   `json:"total_sales"`
	TotalOrders int64    `json:"total_orders"`
	AverageOrderValue float64 `json:"average_order_value"`
	TopProducts []Product `json:"top_products"`
	GeneratedAt time.Time `json:"generated_at"`
}

// AuditAction represents an audit action
type AuditAction struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id"`
	Action    string                 `json:"action"`
	Resource  string                 `json:"resource"`
	Details   map[string]interface{} `json:"details"`
	IP        string                 `json:"ip"`
	UserAgent string                 `json:"user_agent"`
	Timestamp time.Time              `json:"timestamp"`
}

// AuditLogRequest represents an audit log request
type AuditLogRequest struct {
	UserID    string    `json:"user_id,omitempty"`
	Action    string    `json:"action,omitempty"`
	Resource  string    `json:"resource,omitempty"`
	StartDate time.Time `json:"start_date,omitempty"`
	EndDate   time.Time `json:"end_date,omitempty"`
	Limit     int       `json:"limit,omitempty"`
	Offset    int       `json:"offset,omitempty"`
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

// CacheStats represents cache statistics
type CacheStats struct {
	Hits       int64     `json:"hits"`
	Misses     int64     `json:"misses"`
	HitRate    float64   `json:"hit_rate"`
	Size       int64     `json:"size"`
	MaxSize    int64     `json:"max_size"`
	Evictions  int64     `json:"evictions"`
	LastAccess time.Time `json:"last_access"`
}

// MessageQueueStats represents message queue statistics
type MessageQueueStats struct {
	TotalMessages     int64     `json:"total_messages"`
	PublishedMessages int64     `json:"published_messages"`
	ConsumedMessages  int64     `json:"consumed_messages"`
	FailedMessages    int64     `json:"failed_messages"`
	QueueSize         int64     `json:"queue_size"`
	LastMessage       time.Time `json:"last_message"`
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

// TokenClaims represents JWT token claims
type TokenClaims struct {
	UserID    string    `json:"user_id"`
	Username  string    `json:"username"`
	Email     string    `json:"email"`
	Roles     []string  `json:"roles"`
	ExpiresAt time.Time `json:"expires_at"`
}

// HealthStatus represents overall health status
type HealthStatus struct {
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
	Services  []ServiceHealth `json:"services"`
}

// ServiceHealth represents service health
type ServiceHealth struct {
	Name      string    `json:"name"`
	Status    string    `json:"status"`
	Healthy   bool      `json:"healthy"`
	Message   string    `json:"message"`
	Latency   time.Duration `json:"latency"`
	Timestamp time.Time `json:"timestamp"`
}

// ServiceMetrics represents service metrics
type ServiceMetrics struct {
	ServiceName        string    `json:"service_name"`
	TotalRequests      int64     `json:"total_requests"`
	SuccessfulRequests int64     `json:"successful_requests"`
	FailedRequests     int64     `json:"failed_requests"`
	AverageLatency     float64   `json:"average_latency"`
	MaxLatency         float64   `json:"max_latency"`
	MinLatency         float64   `json:"min_latency"`
	SuccessRate        float64   `json:"success_rate"`
	LastRequest        time.Time `json:"last_request"`
	LastError          time.Time `json:"last_error"`
}

// SystemMetrics represents system metrics
type SystemMetrics struct {
	CPUUsage    float64 `json:"cpu_usage"`
	MemoryUsage float64 `json:"memory_usage"`
	DiskUsage   float64 `json:"disk_usage"`
	NetworkIO   float64 `json:"network_io"`
	Timestamp   time.Time `json:"timestamp"`
}

// FacadeConfig represents facade configuration
type FacadeConfig struct {
	Name        string            `json:"name"`
	Version     string            `json:"version"`
	Description string            `json:"description"`
	Services    map[string]ServiceConfig `json:"services"`
	Database    DatabaseConfig    `json:"database"`
	Cache       CacheConfig       `json:"cache"`
	MessageQueue MessageQueueConfig `json:"message_queue"`
	WebSocket   WebSocketConfig   `json:"websocket"`
	Security    SecurityConfig    `json:"security"`
	Monitoring  MonitoringConfig  `json:"monitoring"`
	Logging     LoggingConfig     `json:"logging"`
}

// ServiceConfig represents service configuration
type ServiceConfig struct {
	Name        string            `json:"name"`
	Enabled     bool              `json:"enabled"`
	URL         string            `json:"url"`
	Timeout     time.Duration     `json:"timeout"`
	RetryCount  int               `json:"retry_count"`
	Headers     map[string]string `json:"headers"`
}

// DatabaseConfig represents database configuration
type DatabaseConfig struct {
	MySQL    MySQLConfig    `json:"mysql"`
	MongoDB  MongoDBConfig  `json:"mongodb"`
	Redis    RedisConfig    `json:"redis"`
}

// MySQLConfig represents MySQL configuration
type MySQLConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Username string `json:"username"`
	Password string `json:"password"`
	Database string `json:"database"`
}

// MongoDBConfig represents MongoDB configuration
type MongoDBConfig struct {
	URI      string `json:"uri"`
	Database string `json:"database"`
}

// RedisConfig represents Redis configuration
type RedisConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Password string `json:"password"`
	DB       int    `json:"db"`
}

// CacheConfig represents cache configuration
type CacheConfig struct {
	Enabled         bool          `json:"enabled"`
	Type            string        `json:"type"`
	TTL             time.Duration `json:"ttl"`
	MaxSize         int64         `json:"max_size"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
}

// MessageQueueConfig represents message queue configuration
type MessageQueueConfig struct {
	Enabled bool     `json:"enabled"`
	Brokers []string `json:"brokers"`
	Topics  []string `json:"topics"`
}

// WebSocketConfig represents WebSocket configuration
type WebSocketConfig struct {
	Enabled           bool          `json:"enabled"`
	Port              int           `json:"port"`
	ReadBufferSize    int           `json:"read_buffer_size"`
	WriteBufferSize   int           `json:"write_buffer_size"`
	HandshakeTimeout  time.Duration `json:"handshake_timeout"`
}

// SecurityConfig represents security configuration
type SecurityConfig struct {
	Enabled           bool     `json:"enabled"`
	JWTSecret         string   `json:"jwt_secret"`
	TokenExpiry       time.Duration `json:"token_expiry"`
	AllowedOrigins    []string `json:"allowed_origins"`
	RateLimitEnabled  bool     `json:"rate_limit_enabled"`
	RateLimitRequests int      `json:"rate_limit_requests"`
	RateLimitWindow   time.Duration `json:"rate_limit_window"`
}

// MonitoringConfig represents monitoring configuration
type MonitoringConfig struct {
	Enabled         bool          `json:"enabled"`
	Port            int           `json:"port"`
	Path            string        `json:"path"`
	CollectInterval time.Duration `json:"collect_interval"`
}

// LoggingConfig represents logging configuration
type LoggingConfig struct {
	Level  string `json:"level"`
	Format string `json:"format"`
	Output string `json:"output"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string `json:"error"`
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
	Timestamp time.Time `json:"timestamp"`
}
