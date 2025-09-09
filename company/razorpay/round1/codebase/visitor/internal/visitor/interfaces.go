package visitor

import (
	"time"
)

// Element represents a visitable element in the visitor pattern
type Element interface {
	Accept(visitor Visitor) error
	GetID() string
	GetType() string
	GetName() string
	GetDescription() string
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	IsActive() bool
	SetActive(active bool)
}

// Visitor defines the visitor interface for different operations
type Visitor interface {
	VisitElement(element Element) error
	GetName() string
	GetType() string
	GetDescription() string
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	IsActive() bool
	SetActive(active bool)
}

// ConcreteElement represents a concrete implementation of Element
type ConcreteElement struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Active      bool                   `json:"active"`
}

// Accept implements the Element interface
func (ce *ConcreteElement) Accept(visitor Visitor) error {
	return visitor.VisitElement(ce)
}

// GetID returns the element ID
func (ce *ConcreteElement) GetID() string {
	return ce.ID
}

// GetType returns the element type
func (ce *ConcreteElement) GetType() string {
	return ce.Type
}

// GetName returns the element name
func (ce *ConcreteElement) GetName() string {
	return ce.Name
}

// GetDescription returns the element description
func (ce *ConcreteElement) GetDescription() string {
	return ce.Description
}

// GetMetadata returns the element metadata
func (ce *ConcreteElement) GetMetadata() map[string]interface{} {
	return ce.Metadata
}

// SetMetadata sets a metadata key-value pair
func (ce *ConcreteElement) SetMetadata(key string, value interface{}) {
	if ce.Metadata == nil {
		ce.Metadata = make(map[string]interface{})
	}
	ce.Metadata[key] = value
	ce.UpdatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (ce *ConcreteElement) GetCreatedAt() time.Time {
	return ce.CreatedAt
}

// GetUpdatedAt returns the last update time
func (ce *ConcreteElement) GetUpdatedAt() time.Time {
	return ce.UpdatedAt
}

// IsActive returns whether the element is active
func (ce *ConcreteElement) IsActive() bool {
	return ce.Active
}

// SetActive sets the active status
func (ce *ConcreteElement) SetActive(active bool) {
	ce.Active = active
	ce.UpdatedAt = time.Now()
}

// ConcreteVisitor represents a concrete implementation of Visitor
type ConcreteVisitor struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Active      bool                   `json:"active"`
}

// VisitElement implements the Visitor interface
func (cv *ConcreteVisitor) VisitElement(element Element) error {
	// Default implementation - can be overridden by specific visitors
	element.SetMetadata("visited_by", cv.Name)
	element.SetMetadata("visited_at", time.Now())
	return nil
}

// GetID returns the visitor ID
func (cv *ConcreteVisitor) GetID() string {
	return cv.ID
}

// GetType returns the visitor type
func (cv *ConcreteVisitor) GetType() string {
	return cv.Type
}

// GetName returns the visitor name
func (cv *ConcreteVisitor) GetName() string {
	return cv.Name
}

// GetDescription returns the visitor description
func (cv *ConcreteVisitor) GetDescription() string {
	return cv.Description
}

// GetMetadata returns the visitor metadata
func (cv *ConcreteVisitor) GetMetadata() map[string]interface{} {
	return cv.Metadata
}

// SetMetadata sets a metadata key-value pair
func (cv *ConcreteVisitor) SetMetadata(key string, value interface{}) {
	if cv.Metadata == nil {
		cv.Metadata = make(map[string]interface{})
	}
	cv.Metadata[key] = value
	cv.UpdatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (cv *ConcreteVisitor) GetCreatedAt() time.Time {
	return cv.CreatedAt
}

// GetUpdatedAt returns the last update time
func (cv *ConcreteVisitor) GetUpdatedAt() time.Time {
	return cv.UpdatedAt
}

// IsActive returns whether the visitor is active
func (cv *ConcreteVisitor) IsActive() bool {
	return cv.Active
}

// SetActive sets the active status
func (cv *ConcreteVisitor) SetActive(active bool) {
	cv.Active = active
	cv.UpdatedAt = time.Now()
}

// ElementCollection represents a collection of elements
type ElementCollection interface {
	AddElement(element Element) error
	RemoveElement(elementID string) error
	GetElement(elementID string) (Element, error)
	ListElements() []Element
	GetElementCount() int
	Accept(visitor Visitor) error
	GetID() string
	GetName() string
	GetDescription() string
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	IsActive() bool
	SetActive(active bool)
}

// ConcreteElementCollection represents a concrete implementation of ElementCollection
type ConcreteElementCollection struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Elements    map[string]Element     `json:"elements"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Active      bool                   `json:"active"`
}

// AddElement adds an element to the collection
func (cec *ConcreteElementCollection) AddElement(element Element) error {
	if cec.Elements == nil {
		cec.Elements = make(map[string]Element)
	}
	cec.Elements[element.GetID()] = element
	cec.UpdatedAt = time.Now()
	return nil
}

// RemoveElement removes an element from the collection
func (cec *ConcreteElementCollection) RemoveElement(elementID string) error {
	if cec.Elements == nil {
		return nil
	}
	delete(cec.Elements, elementID)
	cec.UpdatedAt = time.Now()
	return nil
}

// GetElement retrieves an element by ID
func (cec *ConcreteElementCollection) GetElement(elementID string) (Element, error) {
	if cec.Elements == nil {
		return nil, nil
	}
	element, exists := cec.Elements[elementID]
	if !exists {
		return nil, nil
	}
	return element, nil
}

// ListElements returns all elements in the collection
func (cec *ConcreteElementCollection) ListElements() []Element {
	if cec.Elements == nil {
		return []Element{}
	}
	elements := make([]Element, 0, len(cec.Elements))
	for _, element := range cec.Elements {
		elements = append(elements, element)
	}
	return elements
}

// GetElementCount returns the number of elements in the collection
func (cec *ConcreteElementCollection) GetElementCount() int {
	if cec.Elements == nil {
		return 0
	}
	return len(cec.Elements)
}

// Accept implements the Element interface for collections
func (cec *ConcreteElementCollection) Accept(visitor Visitor) error {
	// Visit the collection itself
	if err := visitor.VisitElement(cec); err != nil {
		return err
	}

	// Visit all elements in the collection
	for _, element := range cec.Elements {
		if err := element.Accept(visitor); err != nil {
			return err
		}
	}

	return nil
}

// GetID returns the collection ID
func (cec *ConcreteElementCollection) GetID() string {
	return cec.ID
}

// GetName returns the collection name
func (cec *ConcreteElementCollection) GetName() string {
	return cec.Name
}

// GetDescription returns the collection description
func (cec *ConcreteElementCollection) GetDescription() string {
	return cec.Description
}

// GetMetadata returns the collection metadata
func (cec *ConcreteElementCollection) GetMetadata() map[string]interface{} {
	return cec.Metadata
}

// SetMetadata sets a metadata key-value pair
func (cec *ConcreteElementCollection) SetMetadata(key string, value interface{}) {
	if cec.Metadata == nil {
		cec.Metadata = make(map[string]interface{})
	}
	cec.Metadata[key] = value
	cec.UpdatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (cec *ConcreteElementCollection) GetCreatedAt() time.Time {
	return cec.CreatedAt
}

// GetUpdatedAt returns the last update time
func (cec *ConcreteElementCollection) GetUpdatedAt() time.Time {
	return cec.UpdatedAt
}

// IsActive returns whether the collection is active
func (cec *ConcreteElementCollection) IsActive() bool {
	return cec.Active
}

// SetActive sets the active status
func (cec *ConcreteElementCollection) SetActive(active bool) {
	cec.Active = active
	cec.UpdatedAt = time.Now()
}

// VisitorManager manages visitors and their operations
type VisitorManager interface {
	CreateVisitor(name, visitorType, description string) (Visitor, error)
	GetVisitor(visitorID string) (Visitor, error)
	RemoveVisitor(visitorID string) error
	ListVisitors() []Visitor
	GetVisitorCount() int
	CreateElement(name, elementType, description string) (Element, error)
	GetElement(elementID string) (Element, error)
	RemoveElement(elementID string) error
	ListElements() []Element
	GetElementCount() int
	CreateElementCollection(name, description string) (ElementCollection, error)
	GetElementCollection(collectionID string) (ElementCollection, error)
	RemoveElementCollection(collectionID string) error
	ListElementCollections() []ElementCollection
	GetElementCollectionCount() int
	VisitElement(visitorID, elementID string) error
	VisitElementCollection(visitorID, collectionID string) error
	GetVisitHistory() []VisitRecord
	ClearVisitHistory() error
	GetVisitorStats() map[string]interface{}
	Cleanup() error
}

// VisitRecord represents a record of a visit
type VisitRecord struct {
	ID           string                 `json:"id"`
	VisitorID    string                 `json:"visitor_id"`
	VisitorName  string                 `json:"visitor_name"`
	VisitorType  string                 `json:"visitor_type"`
	ElementID    string                 `json:"element_id"`
	ElementName  string                 `json:"element_name"`
	ElementType  string                 `json:"element_type"`
	CollectionID string                 `json:"collection_id,omitempty"`
	VisitTime    time.Time              `json:"visit_time"`
	Duration     time.Duration          `json:"duration"`
	Success      bool                   `json:"success"`
	Error        string                 `json:"error,omitempty"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// VisitorConfig represents the configuration for the visitor service
type VisitorConfig struct {
	Name                  string                 `json:"name"`
	Version               string                 `json:"version"`
	Description           string                 `json:"description"`
	MaxVisitors           int                    `json:"max_visitors"`
	MaxElements           int                    `json:"max_elements"`
	MaxElementCollections int                    `json:"max_element_collections"`
	MaxVisitHistory       int                    `json:"max_visit_history"`
	VisitTimeout          time.Duration          `json:"visit_timeout"`
	CleanupInterval       time.Duration          `json:"cleanup_interval"`
	ValidationEnabled     bool                   `json:"validation_enabled"`
	CachingEnabled        bool                   `json:"caching_enabled"`
	MonitoringEnabled     bool                   `json:"monitoring_enabled"`
	AuditingEnabled       bool                   `json:"auditing_enabled"`
	SupportedVisitorTypes []string               `json:"supported_visitor_types"`
	SupportedElementTypes []string               `json:"supported_element_types"`
	DefaultVisitorType    string                 `json:"default_visitor_type"`
	DefaultElementType    string                 `json:"default_element_type"`
	ValidationRules       map[string]interface{} `json:"validation_rules"`
	Metadata              map[string]interface{} `json:"metadata"`
	Database              DatabaseConfig         `json:"database"`
	Cache                 CacheConfig            `json:"cache"`
	MessageQueue          MessageQueueConfig     `json:"message_queue"`
	WebSocket             WebSocketConfig        `json:"websocket"`
	Security              SecurityConfig         `json:"security"`
	Monitoring            MonitoringConfig       `json:"monitoring"`
	Logging               LoggingConfig          `json:"logging"`
}

// DatabaseConfig represents database configuration
type DatabaseConfig struct {
	MySQL   MySQLConfig   `json:"mysql"`
	MongoDB MongoDBConfig `json:"mongodb"`
	Redis   RedisConfig   `json:"redis"`
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
	MaxSize         int           `json:"max_size"`
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
	Enabled          bool          `json:"enabled"`
	Port             int           `json:"port"`
	ReadBufferSize   int           `json:"read_buffer_size"`
	WriteBufferSize  int           `json:"write_buffer_size"`
	HandshakeTimeout time.Duration `json:"handshake_timeout"`
}

// SecurityConfig represents security configuration
type SecurityConfig struct {
	Enabled           bool          `json:"enabled"`
	JWTSecret         string        `json:"jwt_secret"`
	TokenExpiry       time.Duration `json:"token_expiry"`
	AllowedOrigins    []string      `json:"allowed_origins"`
	RateLimitEnabled  bool          `json:"rate_limit_enabled"`
	RateLimitRequests int           `json:"rate_limit_requests"`
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
