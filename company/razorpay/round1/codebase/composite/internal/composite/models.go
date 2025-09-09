package composite

import (
	"errors"
	"time"
)

// BaseComponent provides common functionality for all components
type BaseComponent struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Type      string                 `json:"type"`
	Parent    Component              `json:"-"`
	Children  []Component            `json:"-"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

// GetID returns the component ID
func (bc *BaseComponent) GetID() string {
	return bc.ID
}

// GetName returns the component name
func (bc *BaseComponent) GetName() string {
	return bc.Name
}

// GetType returns the component type
func (bc *BaseComponent) GetType() string {
	return bc.Type
}

// GetParent returns the parent component
func (bc *BaseComponent) GetParent() Component {
	return bc.Parent
}

// SetParent sets the parent component
func (bc *BaseComponent) SetParent(parent Component) {
	bc.Parent = parent
	bc.Update()
}

// GetChildren returns the children components
func (bc *BaseComponent) GetChildren() []Component {
	return bc.Children
}

// HasChildren returns true if the component has children
func (bc *BaseComponent) HasChildren() bool {
	return len(bc.Children) > 0
}

// GetSize returns the total number of components in the tree
func (bc *BaseComponent) GetSize() int {
	size := 1
	for _, child := range bc.Children {
		size += child.GetSize()
	}
	return size
}

// GetDepth returns the depth of the component in the tree
func (bc *BaseComponent) GetDepth() int {
	if bc.Parent == nil {
		return 0
	}
	return bc.Parent.GetDepth() + 1
}

// GetPath returns the path from root to this component
func (bc *BaseComponent) GetPath() string {
	if bc.Parent == nil {
		return bc.Name
	}
	return bc.Parent.GetPath() + "/" + bc.Name
}

// GetMetadata returns the component metadata
func (bc *BaseComponent) GetMetadata() map[string]interface{} {
	return bc.Metadata
}

// SetMetadata sets a metadata value
func (bc *BaseComponent) SetMetadata(key string, value interface{}) {
	if bc.Metadata == nil {
		bc.Metadata = make(map[string]interface{})
	}
	bc.Metadata[key] = value
	bc.Update()
}

// IsLeaf returns false for base components
func (bc *BaseComponent) IsLeaf() bool {
	return false
}

// IsComposite returns false for base components
func (bc *BaseComponent) IsComposite() bool {
	return false
}

// GetCreatedAt returns the creation time
func (bc *BaseComponent) GetCreatedAt() time.Time {
	return bc.CreatedAt
}

// GetUpdatedAt returns the last update time
func (bc *BaseComponent) GetUpdatedAt() time.Time {
	return bc.UpdatedAt
}

// Update updates the last modified time
func (bc *BaseComponent) Update() {
	bc.UpdatedAt = time.Now()
}

// FolderComponent represents a folder in a file system
type FolderComponent struct {
	BaseComponent
	Path        string `json:"path"`
	Permissions string `json:"permissions"`
	Size        int64  `json:"size"`
}

// NewFolderComponent creates a new folder component
func NewFolderComponent(id, name, path string) *FolderComponent {
	now := time.Now()
	return &FolderComponent{
		BaseComponent: BaseComponent{
			ID:        id,
			Name:      name,
			Type:      "folder",
			Metadata:  make(map[string]interface{}),
			CreatedAt: now,
			UpdatedAt: now,
		},
		Path:        path,
		Permissions: "755",
		Size:        0,
	}
}

// Add adds a child component
func (fc *FolderComponent) Add(child Component) error {
	if child == nil {
		return ErrInvalidComponent
	}
	
	// Check if child already exists
	for _, existingChild := range fc.Children {
		if existingChild.GetID() == child.GetID() {
			return ErrComponentExists
		}
	}
	
	// Set parent and add to children
	child.SetParent(fc)
	fc.Children = append(fc.Children, child)
	fc.Update()
	
	return nil
}

// Remove removes a child component
func (fc *FolderComponent) Remove(child Component) error {
	if child == nil {
		return ErrInvalidComponent
	}
	
	for i, existingChild := range fc.Children {
		if existingChild.GetID() == child.GetID() {
			fc.Children = append(fc.Children[:i], fc.Children[i+1:]...)
			child.SetParent(nil)
			fc.Update()
			return nil
		}
	}
	
	return ErrComponentNotFound
}

// GetChild returns a child component by ID
func (fc *FolderComponent) GetChild(id string) (Component, error) {
	for _, child := range fc.Children {
		if child.GetID() == id {
			return child, nil
		}
	}
	return nil, ErrComponentNotFound
}

// Execute executes the folder component
func (fc *FolderComponent) Execute(ctx context.Context) (interface{}, error) {
	// Simulate folder operations
	result := map[string]interface{}{
		"id":          fc.ID,
		"name":        fc.Name,
		"type":        fc.Type,
		"path":        fc.Path,
		"permissions": fc.Permissions,
		"size":        fc.Size,
		"children":    len(fc.Children),
		"depth":       fc.GetDepth(),
	}
	
	return result, nil
}

// Validate validates the folder component
func (fc *FolderComponent) Validate() error {
	if fc.ID == "" {
		return ErrInvalidID
	}
	if fc.Name == "" {
		return ErrInvalidName
	}
	if fc.Path == "" {
		return ErrInvalidPath
	}
	return nil
}

// IsComposite returns true for folder components
func (fc *FolderComponent) IsComposite() bool {
	return true
}

// FileComponent represents a file in a file system
type FileComponent struct {
	BaseComponent
	Path        string `json:"path"`
	Permissions string `json:"permissions"`
	Size        int64  `json:"size"`
	Content     string `json:"content"`
	Extension   string `json:"extension"`
}

// NewFileComponent creates a new file component
func NewFileComponent(id, name, path string) *FileComponent {
	now := time.Now()
	return &FileComponent{
		BaseComponent: BaseComponent{
			ID:        id,
			Name:      name,
			Type:      "file",
			Metadata:  make(map[string]interface{}),
			CreatedAt: now,
			UpdatedAt: now,
		},
		Path:        path,
		Permissions: "644",
		Size:        0,
		Content:     "",
		Extension:   "",
	}
}

// Add is not supported for file components
func (fc *FileComponent) Add(child Component) error {
	return ErrNotSupported
}

// Remove is not supported for file components
func (fc *FileComponent) Remove(child Component) error {
	return ErrNotSupported
}

// GetChild is not supported for file components
func (fc *FileComponent) GetChild(id string) (Component, error) {
	return nil, ErrNotSupported
}

// Execute executes the file component
func (fc *FileComponent) Execute(ctx context.Context) (interface{}, error) {
	// Simulate file operations
	result := map[string]interface{}{
		"id":          fc.ID,
		"name":        fc.Name,
		"type":        fc.Type,
		"path":        fc.Path,
		"permissions": fc.Permissions,
		"size":        fc.Size,
		"content":     fc.Content,
		"extension":   fc.Extension,
		"depth":       fc.GetDepth(),
	}
	
	return result, nil
}

// Validate validates the file component
func (fc *FileComponent) Validate() error {
	if fc.ID == "" {
		return ErrInvalidID
	}
	if fc.Name == "" {
		return ErrInvalidName
	}
	if fc.Path == "" {
		return ErrInvalidPath
	}
	return nil
}

// IsLeaf returns true for file components
func (fc *FileComponent) IsLeaf() bool {
	return true
}

// GetValue returns the file content
func (fc *FileComponent) GetValue() interface{} {
	return fc.Content
}

// SetValue sets the file content
func (fc *FileComponent) SetValue(value interface{}) {
	if content, ok := value.(string); ok {
		fc.Content = content
		fc.Size = int64(len(content))
		fc.Update()
	}
}

// GetProperties returns the file properties
func (fc *FileComponent) GetProperties() map[string]interface{} {
	return map[string]interface{}{
		"path":        fc.Path,
		"permissions": fc.Permissions,
		"size":        fc.Size,
		"extension":   fc.Extension,
	}
}

// SetProperty sets a file property
func (fc *FileComponent) SetProperty(key string, value interface{}) {
	switch key {
	case "path":
		if path, ok := value.(string); ok {
			fc.Path = path
		}
	case "permissions":
		if permissions, ok := value.(string); ok {
			fc.Permissions = permissions
		}
	case "size":
		if size, ok := value.(int64); ok {
			fc.Size = size
		}
	case "extension":
		if extension, ok := value.(string); ok {
			fc.Extension = extension
		}
	}
	fc.Update()
}

// GetWeight returns the file weight (size)
func (fc *FileComponent) GetWeight() float64 {
	return float64(fc.Size)
}

// SetWeight sets the file weight
func (fc *FileComponent) SetWeight(weight float64) {
	fc.Size = int64(weight)
	fc.Update()
}

// MenuComponent represents a menu item in a navigation system
type MenuComponent struct {
	BaseComponent
	URL         string `json:"url"`
	Icon        string `json:"icon"`
	Order       int    `json:"order"`
	IsActive    bool   `json:"is_active"`
	IsVisible   bool   `json:"is_visible"`
}

// NewMenuComponent creates a new menu component
func NewMenuComponent(id, name, url string) *MenuComponent {
	now := time.Now()
	return &MenuComponent{
		BaseComponent: BaseComponent{
			ID:        id,
			Name:      name,
			Type:      "menu",
			Metadata:  make(map[string]interface{}),
			CreatedAt: now,
			UpdatedAt: now,
		},
		URL:       url,
		Icon:      "",
		Order:     0,
		IsActive:  false,
		IsVisible: true,
	}
}

// Add adds a child menu item
func (mc *MenuComponent) Add(child Component) error {
	if child == nil {
		return ErrInvalidComponent
	}
	
	// Check if child already exists
	for _, existingChild := range mc.Children {
		if existingChild.GetID() == child.GetID() {
			return ErrComponentExists
		}
	}
	
	// Set parent and add to children
	child.SetParent(mc)
	mc.Children = append(mc.Children, child)
	mc.Update()
	
	return nil
}

// Remove removes a child menu item
func (mc *MenuComponent) Remove(child Component) error {
	if child == nil {
		return ErrInvalidComponent
	}
	
	for i, existingChild := range mc.Children {
		if existingChild.GetID() == child.GetID() {
			mc.Children = append(mc.Children[:i], mc.Children[i+1:]...)
			child.SetParent(nil)
			mc.Update()
			return nil
		}
	}
	
	return ErrComponentNotFound
}

// GetChild returns a child menu item by ID
func (mc *MenuComponent) GetChild(id string) (Component, error) {
	for _, child := range mc.Children {
		if child.GetID() == child.GetID() {
			return child, nil
		}
	}
	return nil, ErrComponentNotFound
}

// Execute executes the menu component
func (mc *MenuComponent) Execute(ctx context.Context) (interface{}, error) {
	// Simulate menu operations
	result := map[string]interface{}{
		"id":         mc.ID,
		"name":       mc.Name,
		"type":       mc.Type,
		"url":        mc.URL,
		"icon":       mc.Icon,
		"order":      mc.Order,
		"is_active":  mc.IsActive,
		"is_visible": mc.IsVisible,
		"children":   len(mc.Children),
		"depth":      mc.GetDepth(),
	}
	
	return result, nil
}

// Validate validates the menu component
func (mc *MenuComponent) Validate() error {
	if mc.ID == "" {
		return ErrInvalidID
	}
	if mc.Name == "" {
		return ErrInvalidName
	}
	return nil
}

// IsComposite returns true for menu components
func (mc *MenuComponent) IsComposite() bool {
	return true
}

// ComponentStatistics represents statistics for a component
type ComponentStatistics struct {
	TotalComponents int            `json:"total_components"`
	LeafComponents  int            `json:"leaf_components"`
	CompositeComponents int        `json:"composite_components"`
	MaxDepth        int            `json:"max_depth"`
	AverageDepth    float64        `json:"average_depth"`
	TotalSize       int64          `json:"total_size"`
	TypeDistribution map[string]int `json:"type_distribution"`
	LastUpdated     time.Time      `json:"last_updated"`
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

// CompositeConfig represents composite configuration
type CompositeConfig struct {
	Name        string            `json:"name"`
	Version     string            `json:"version"`
	Description string            `json:"description"`
	MaxDepth    int               `json:"max_depth"`
	MaxChildren int               `json:"max_children"`
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

// Custom errors
var (
	ErrInvalidComponent    = errors.New("invalid component")
	ErrComponentExists     = errors.New("component already exists")
	ErrComponentNotFound   = errors.New("component not found")
	ErrNotSupported        = errors.New("operation not supported")
	ErrInvalidID           = errors.New("invalid ID")
	ErrInvalidName         = errors.New("invalid name")
	ErrInvalidPath         = errors.New("invalid path")
	ErrMaxDepthExceeded    = errors.New("maximum depth exceeded")
	ErrMaxChildrenExceeded = errors.New("maximum children exceeded")
)
