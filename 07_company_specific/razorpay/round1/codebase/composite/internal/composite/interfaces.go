package composite

import (
	"context"
	"time"
)

// Component defines the interface for composite pattern components
type Component interface {
	GetID() string
	GetName() string
	GetType() string
	GetParent() Component
	SetParent(parent Component)
	Add(child Component) error
	Remove(child Component) error
	GetChildren() []Component
	GetChild(id string) (Component, error)
	HasChildren() bool
	GetSize() int
	GetDepth() int
	GetPath() string
	Execute(ctx context.Context) (interface{}, error)
	Validate() error
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	IsLeaf() bool
	IsComposite() bool
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	Update()
}

// Composite defines the interface for composite components
type Composite interface {
	Component
	GetAllChildren() []Component
	GetChildrenByType(componentType string) []Component
	GetChildrenByDepth(depth int) []Component
	FindChild(predicate func(Component) bool) (Component, error)
	FindAllChildren(predicate func(Component) bool) []Component
	Traverse(visitor func(Component) error) error
	GetStatistics() ComponentStatistics
	Optimize() error
}

// Leaf defines the interface for leaf components
type Leaf interface {
	Component
	GetValue() interface{}
	SetValue(value interface{})
	GetProperties() map[string]interface{}
	SetProperty(key string, value interface{})
	GetWeight() float64
	SetWeight(weight float64)
}

// Visitor defines the interface for visitor pattern
type Visitor interface {
	VisitComponent(component Component) error
	VisitComposite(composite Composite) error
	VisitLeaf(leaf Leaf) error
}

// Iterator defines the interface for component iteration
type Iterator interface {
	HasNext() bool
	Next() Component
	Reset()
	GetCurrent() Component
	GetIndex() int
}

// Filter defines the interface for component filtering
type Filter interface {
	Filter(component Component) bool
	GetType() string
	GetDescription() string
}

// Sorter defines the interface for component sorting
type Sorter interface {
	Sort(components []Component) []Component
	GetType() string
	GetDescription() string
}

// Cache defines the interface for caching operations
type Cache interface {
	Get(key string) (interface{}, bool)
	Set(key string, value interface{}, expiration time.Duration)
	Delete(key string)
	Clear()
	GetStats() CacheStats
	GetSize() int
	GetMaxSize() int
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

// Database defines the interface for database operations
type Database interface {
	Save(ctx context.Context, collection string, data interface{}) error
	Find(ctx context.Context, collection string, query interface{}) (interface{}, error)
	Update(ctx context.Context, collection string, id string, data interface{}) error
	Delete(ctx context.Context, collection string, id string) error
	Transaction(ctx context.Context, fn func(ctx context.Context) error) error
}

// MessageQueue defines the interface for message queue operations
type MessageQueue interface {
	Publish(ctx context.Context, topic string, message interface{}) error
	Subscribe(ctx context.Context, topic string, handler func(interface{}) error) error
	Close() error
	GetStats() MessageQueueStats
}

// WebSocket defines the interface for WebSocket operations
type WebSocket interface {
	Send(ctx context.Context, clientID string, message interface{}) error
	Broadcast(ctx context.Context, message interface{}) error
	Register(ctx context.Context, clientID string, conn interface{}) error
	Unregister(ctx context.Context, clientID string) error
	GetStats() WebSocketStats
}

// Security defines the interface for security operations
type Security interface {
	ValidateInput(input interface{}) error
	SanitizeInput(input interface{}) interface{}
	CheckPermission(ctx context.Context, userID string, resource string, action string) bool
	AuditLog(ctx context.Context, action string, userID string, details map[string]interface{})
}

// Configuration defines the interface for configuration operations
type Configuration interface {
	GetConfig(ctx context.Context, key string) (interface{}, error)
	SetConfig(ctx context.Context, key string, value interface{}) error
	GetAllConfigs(ctx context.Context) (map[string]interface{}, error)
	ReloadConfig(ctx context.Context) error
}

// Health defines the interface for health check operations
type Health interface {
	CheckHealth(ctx context.Context) (*HealthStatus, error)
	CheckServiceHealth(ctx context.Context, serviceName string) (*ServiceHealth, error)
	GetAllServicesHealth(ctx context.Context) ([]*ServiceHealth, error)
}

// Monitoring defines the interface for monitoring operations
type Monitoring interface {
	RecordRequest(ctx context.Context, service string, duration time.Duration, success bool)
	RecordError(ctx context.Context, service string, err error)
	GetServiceMetrics(ctx context.Context, service string) (*ServiceMetrics, error)
	GetSystemMetrics(ctx context.Context) (*SystemMetrics, error)
}

// Serialization defines the interface for serialization operations
type Serialization interface {
	Serialize(data interface{}) ([]byte, error)
	Deserialize(data []byte, target interface{}) error
	GetContentType() string
}

// Compression defines the interface for compression operations
type Compression interface {
	Compress(data []byte) ([]byte, error)
	Decompress(compressedData []byte) ([]byte, error)
	GetCompressionRatio(originalSize, compressedSize int) float64
}

// Encryption defines the interface for encryption operations
type Encryption interface {
	Encrypt(data []byte) ([]byte, error)
	Decrypt(encryptedData []byte) ([]byte, error)
	Hash(data []byte) ([]byte, error)
	VerifyHash(data []byte, hash []byte) bool
}

// Validation defines the interface for validation operations
type Validation interface {
	Validate(ctx context.Context, data interface{}) error
	ValidateSchema(ctx context.Context, data interface{}, schema interface{}) error
	GetValidationRules() map[string]interface{}
}

// Notification defines the interface for notification operations
type Notification interface {
	Send(ctx context.Context, recipient string, message interface{}) error
	SendBulk(ctx context.Context, recipients []string, message interface{}) error
	GetDeliveryStatus(ctx context.Context, messageID string) (string, error)
}

// Analytics defines the interface for analytics operations
type Analytics interface {
	Track(ctx context.Context, event string, properties map[string]interface{}) error
	TrackUser(ctx context.Context, userID string, event string, properties map[string]interface{}) error
	GetMetrics(ctx context.Context, query interface{}) (interface{}, error)
}

// Audit defines the interface for audit operations
type Audit interface {
	Log(ctx context.Context, action string, userID string, resource string, details map[string]interface{}) error
	GetAuditLogs(ctx context.Context, query interface{}) ([]AuditLog, error)
	GetUserActivity(ctx context.Context, userID string) ([]AuditLog, error)
}
