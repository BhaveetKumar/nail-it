package flyweight

import (
	"time"
)

// ProductFlyweight represents a shared product flyweight
type ProductFlyweight struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	Category     string                 `json:"category"`
	Brand        string                 `json:"brand"`
	BasePrice    float64                `json:"base_price"`
	Currency     string                 `json:"currency"`
	Attributes   map[string]interface{} `json:"attributes"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
	IsShared     bool                   `json:"is_shared"`
}

// GetIntrinsicState returns the intrinsic state of the flyweight
func (pf *ProductFlyweight) GetIntrinsicState() map[string]interface{} {
	return map[string]interface{}{
		"id":          pf.ID,
		"type":        pf.Type,
		"name":        pf.Name,
		"description": pf.Description,
		"category":    pf.Category,
		"brand":       pf.Brand,
		"base_price":  pf.BasePrice,
		"currency":    pf.Currency,
		"attributes":  pf.Attributes,
	}
}

// GetType returns the type of the flyweight
func (pf *ProductFlyweight) GetType() string {
	return pf.Type
}

// GetID returns the ID of the flyweight
func (pf *ProductFlyweight) GetID() string {
	return pf.ID
}

// IsShared returns whether the flyweight is shared
func (pf *ProductFlyweight) IsShared() bool {
	return pf.IsShared
}

// GetCreatedAt returns the creation time
func (pf *ProductFlyweight) GetCreatedAt() time.Time {
	return pf.CreatedAt
}

// GetLastAccessed returns the last accessed time
func (pf *ProductFlyweight) GetLastAccessed() time.Time {
	return pf.LastAccessed
}

// UpdateLastAccessed updates the last accessed time
func (pf *ProductFlyweight) UpdateLastAccessed() {
	pf.LastAccessed = time.Now()
}

// UserFlyweight represents a shared user flyweight
type UserFlyweight struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Username     string                 `json:"username"`
	Email        string                 `json:"email"`
	Profile      map[string]interface{} `json:"profile"`
	Preferences  map[string]interface{} `json:"preferences"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
	IsShared     bool                   `json:"is_shared"`
}

// GetIntrinsicState returns the intrinsic state of the flyweight
func (uf *UserFlyweight) GetIntrinsicState() map[string]interface{} {
	return map[string]interface{}{
		"id":          uf.ID,
		"type":        uf.Type,
		"username":    uf.Username,
		"email":       uf.Email,
		"profile":     uf.Profile,
		"preferences": uf.Preferences,
	}
}

// GetType returns the type of the flyweight
func (uf *UserFlyweight) GetType() string {
	return uf.Type
}

// GetID returns the ID of the flyweight
func (uf *UserFlyweight) GetID() string {
	return uf.ID
}

// IsShared returns whether the flyweight is shared
func (uf *UserFlyweight) IsShared() bool {
	return uf.IsShared
}

// GetCreatedAt returns the creation time
func (uf *UserFlyweight) GetCreatedAt() time.Time {
	return uf.CreatedAt
}

// GetLastAccessed returns the last accessed time
func (uf *UserFlyweight) GetLastAccessed() time.Time {
	return uf.LastAccessed
}

// UpdateLastAccessed updates the last accessed time
func (uf *UserFlyweight) UpdateLastAccessed() {
	uf.LastAccessed = time.Now()
}

// OrderFlyweight represents a shared order flyweight
type OrderFlyweight struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Status       string                 `json:"status"`
	Priority     string                 `json:"priority"`
	Metadata     map[string]interface{} `json:"metadata"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
	IsShared     bool                   `json:"is_shared"`
}

// GetIntrinsicState returns the intrinsic state of the flyweight
func (of *OrderFlyweight) GetIntrinsicState() map[string]interface{} {
	return map[string]interface{}{
		"id":       of.ID,
		"type":     of.Type,
		"status":   of.Status,
		"priority": of.Priority,
		"metadata": of.Metadata,
	}
}

// GetType returns the type of the flyweight
func (of *OrderFlyweight) GetType() string {
	return of.Type
}

// GetID returns the ID of the flyweight
func (of *OrderFlyweight) GetID() string {
	return of.ID
}

// IsShared returns whether the flyweight is shared
func (of *OrderFlyweight) IsShared() bool {
	return of.IsShared
}

// GetCreatedAt returns the creation time
func (of *OrderFlyweight) GetCreatedAt() time.Time {
	return of.CreatedAt
}

// GetLastAccessed returns the last accessed time
func (of *OrderFlyweight) GetLastAccessed() time.Time {
	return of.LastAccessed
}

// UpdateLastAccessed updates the last accessed time
func (of *OrderFlyweight) UpdateLastAccessed() {
	of.LastAccessed = time.Now()
}

// NotificationFlyweight represents a shared notification flyweight
type NotificationFlyweight struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Template     string                 `json:"template"`
	Subject      string                 `json:"subject"`
	Body         string                 `json:"body"`
	Channels     []string               `json:"channels"`
	Metadata     map[string]interface{} `json:"metadata"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
	IsShared     bool                   `json:"is_shared"`
}

// GetIntrinsicState returns the intrinsic state of the flyweight
func (nf *NotificationFlyweight) GetIntrinsicState() map[string]interface{} {
	return map[string]interface{}{
		"id":       nf.ID,
		"type":     nf.Type,
		"template": nf.Template,
		"subject":  nf.Subject,
		"body":     nf.Body,
		"channels": nf.Channels,
		"metadata": nf.Metadata,
	}
}

// GetType returns the type of the flyweight
func (nf *NotificationFlyweight) GetType() string {
	return nf.Type
}

// GetID returns the ID of the flyweight
func (nf *NotificationFlyweight) GetID() string {
	return nf.ID
}

// IsShared returns whether the flyweight is shared
func (nf *NotificationFlyweight) IsShared() bool {
	return nf.IsShared
}

// GetCreatedAt returns the creation time
func (nf *NotificationFlyweight) GetCreatedAt() time.Time {
	return nf.CreatedAt
}

// GetLastAccessed returns the last accessed time
func (nf *NotificationFlyweight) GetLastAccessed() time.Time {
	return nf.LastAccessed
}

// UpdateLastAccessed updates the last accessed time
func (nf *NotificationFlyweight) UpdateLastAccessed() {
	nf.LastAccessed = time.Now()
}

// ConfigurationFlyweight represents a shared configuration flyweight
type ConfigurationFlyweight struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Key          string                 `json:"key"`
	Value        interface{}            `json:"value"`
	Description  string                 `json:"description"`
	Category     string                 `json:"category"`
	Metadata     map[string]interface{} `json:"metadata"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
	IsShared     bool                   `json:"is_shared"`
}

// GetIntrinsicState returns the intrinsic state of the flyweight
func (cf *ConfigurationFlyweight) GetIntrinsicState() map[string]interface{} {
	return map[string]interface{}{
		"id":          cf.ID,
		"type":        cf.Type,
		"key":         cf.Key,
		"value":       cf.Value,
		"description": cf.Description,
		"category":    cf.Category,
		"metadata":    cf.Metadata,
	}
}

// GetType returns the type of the flyweight
func (cf *ConfigurationFlyweight) GetType() string {
	return cf.Type
}

// GetID returns the ID of the flyweight
func (cf *ConfigurationFlyweight) GetID() string {
	return cf.ID
}

// IsShared returns whether the flyweight is shared
func (cf *ConfigurationFlyweight) IsShared() bool {
	return cf.IsShared
}

// GetCreatedAt returns the creation time
func (cf *ConfigurationFlyweight) GetCreatedAt() time.Time {
	return cf.CreatedAt
}

// GetLastAccessed returns the last accessed time
func (cf *ConfigurationFlyweight) GetLastAccessed() time.Time {
	return cf.LastAccessed
}

// UpdateLastAccessed updates the last accessed time
func (cf *ConfigurationFlyweight) UpdateLastAccessed() {
	cf.LastAccessed = time.Now()
}

// FactoryStats represents factory statistics
type FactoryStats struct {
	TotalFlyweights    int64     `json:"total_flyweights"`
	SharedFlyweights   int64     `json:"shared_flyweights"`
	UnsharedFlyweights int64     `json:"unshared_flyweights"`
	MemoryUsage        int64     `json:"memory_usage"`
	CacheHits          int64     `json:"cache_hits"`
	CacheMisses        int64     `json:"cache_misses"`
	HitRate            float64   `json:"hit_rate"`
	LastCleanup        time.Time `json:"last_cleanup"`
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

// HealthStatus represents overall health status
type HealthStatus struct {
	Status    string          `json:"status"`
	Timestamp time.Time       `json:"timestamp"`
	Services  []ServiceHealth `json:"services"`
}

// ServiceHealth represents service health
type ServiceHealth struct {
	Name      string        `json:"name"`
	Status    string        `json:"status"`
	Healthy   bool          `json:"healthy"`
	Message   string        `json:"message"`
	Latency   time.Duration `json:"latency"`
	Timestamp time.Time     `json:"timestamp"`
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
	CPUUsage    float64   `json:"cpu_usage"`
	MemoryUsage float64   `json:"memory_usage"`
	DiskUsage   float64   `json:"disk_usage"`
	NetworkIO   float64   `json:"network_io"`
	Timestamp   time.Time `json:"timestamp"`
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

// FlyweightConfig represents flyweight configuration
type FlyweightConfig struct {
	Name        string            `json:"name"`
	Version     string            `json:"version"`
	Description string            `json:"description"`
	MaxSize     int64             `json:"max_size"`
	TTL         time.Duration     `json:"ttl"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
	Types       []string          `json:"types"`
	Database    DatabaseConfig    `json:"database"`
	Cache       CacheConfig       `json:"cache"`
	MessageQueue MessageQueueConfig `json:"message_queue"`
	WebSocket   WebSocketConfig   `json:"websocket"`
	Security    SecurityConfig    `json:"security"`
	Monitoring  MonitoringConfig  `json:"monitoring"`
	Logging     LoggingConfig     `json:"logging"`
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
