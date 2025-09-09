package event_sourcing

import (
	"context"
	"errors"
	"time"
)

// Custom errors
var (
	ErrEventSourcingServiceInactive = errors.New("event sourcing service is inactive")
	ErrEventStoreNotFound           = errors.New("event store not found")
	ErrEventBusNotFound             = errors.New("event bus not found")
	ErrSnapshotStoreNotFound        = errors.New("snapshot store not found")
	ErrAggregateNotFound            = errors.New("aggregate not found")
	ErrEventNotFound                = errors.New("event not found")
	ErrSnapshotNotFound             = errors.New("snapshot not found")
	ErrMaxEventsReached             = errors.New("maximum number of events reached")
	ErrMaxAggregatesReached         = errors.New("maximum number of aggregates reached")
	ErrMaxSnapshotsReached          = errors.New("maximum number of snapshots reached")
	ErrInvalidEventType             = errors.New("invalid event type")
	ErrInvalidAggregateType         = errors.New("invalid aggregate type")
	ErrInvalidSnapshotType          = errors.New("invalid snapshot type")
	ErrEventPublishFailed           = errors.New("event publish failed")
	ErrEventSubscribeFailed         = errors.New("event subscribe failed")
	ErrEventUnsubscribeFailed       = errors.New("event unsubscribe failed")
	ErrSnapshotCreateFailed         = errors.New("snapshot create failed")
	ErrSnapshotRetrieveFailed       = errors.New("snapshot retrieve failed")
	ErrAggregateCreateFailed        = errors.New("aggregate create failed")
	ErrAggregateSaveFailed          = errors.New("aggregate save failed")
	ErrAggregateRetrieveFailed      = errors.New("aggregate retrieve failed")
	ErrEventRetrieveFailed          = errors.New("event retrieve failed")
	ErrCleanupFailed                = errors.New("cleanup failed")
)

// EventSourcingConfig represents the configuration for the event sourcing service
type EventSourcingConfig struct {
	Name                    string                 `json:"name"`
	Version                 string                 `json:"version"`
	Description             string                 `json:"description"`
	MaxEvents               int                    `json:"max_events"`
	MaxAggregates           int                    `json:"max_aggregates"`
	MaxSnapshots            int                    `json:"max_snapshots"`
	SnapshotInterval        time.Duration          `json:"snapshot_interval"`
	CleanupInterval         time.Duration          `json:"cleanup_interval"`
	ValidationEnabled       bool                   `json:"validation_enabled"`
	CachingEnabled          bool                   `json:"caching_enabled"`
	MonitoringEnabled       bool                   `json:"monitoring_enabled"`
	AuditingEnabled         bool                   `json:"auditing_enabled"`
	SupportedEventTypes     []string               `json:"supported_event_types"`
	SupportedAggregateTypes []string               `json:"supported_aggregate_types"`
	ValidationRules         map[string]interface{} `json:"validation_rules"`
	Metadata                map[string]interface{} `json:"metadata"`
	Database                DatabaseConfig         `json:"database"`
	Cache                   CacheConfig            `json:"cache"`
	MessageQueue            MessageQueueConfig     `json:"message_queue"`
	WebSocket               WebSocketConfig        `json:"websocket"`
	Security                SecurityConfig         `json:"security"`
	Monitoring              MonitoringConfig       `json:"monitoring"`
	Logging                 LoggingConfig          `json:"logging"`
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

// EventSourcingServiceManager manages the event sourcing service operations
type EventSourcingServiceManager struct {
	service *EventSourcingService
	config  *EventSourcingConfig
}

// NewEventSourcingServiceManager creates a new event sourcing service manager
func NewEventSourcingServiceManager(config *EventSourcingConfig) *EventSourcingServiceManager {
	return &EventSourcingServiceManager{
		service: NewEventSourcingService(config),
		config:  config,
	}
}

// CreateAggregate creates a new aggregate with validation
func (essm *EventSourcingServiceManager) CreateAggregate(ctx context.Context, aggregateType string, aggregateID string, initialState map[string]interface{}) (Aggregate, error) {
	// Validate input
	if aggregateType == "" {
		return nil, errors.New("aggregate type cannot be empty")
	}
	if aggregateID == "" {
		return nil, errors.New("aggregate ID cannot be empty")
	}

	// Validate aggregate type
	if essm.config.ValidationEnabled {
		if !essm.isValidAggregateType(aggregateType) {
			return nil, ErrInvalidAggregateType
		}
	}

	// Check aggregate limit
	if essm.service.GetAggregateCount() >= essm.config.MaxAggregates {
		return nil, ErrMaxAggregatesReached
	}

	// Create aggregate
	aggregate, err := essm.service.CreateAggregate(ctx, aggregateType, aggregateID, initialState)
	if err != nil {
		return nil, ErrAggregateCreateFailed
	}

	// Set metadata
	aggregate.SetMetadata("created_by", "event-sourcing-service-manager")
	aggregate.SetMetadata("creation_time", time.Now())

	return aggregate, nil
}

// GetAggregate retrieves an aggregate with validation
func (essm *EventSourcingServiceManager) GetAggregate(ctx context.Context, aggregateID string) (Aggregate, error) {
	// Validate input
	if aggregateID == "" {
		return nil, errors.New("aggregate ID cannot be empty")
	}

	// Get aggregate
	aggregate, err := essm.service.GetAggregate(ctx, aggregateID)
	if err != nil {
		return nil, ErrAggregateRetrieveFailed
	}

	// Update last accessed time
	aggregate.SetMetadata("last_accessed", time.Now())

	return aggregate, nil
}

// SaveAggregate saves an aggregate with validation
func (essm *EventSourcingServiceManager) SaveAggregate(ctx context.Context, aggregate Aggregate) error {
	// Validate input
	if aggregate == nil {
		return errors.New("aggregate cannot be nil")
	}

	// Validate aggregate type
	if essm.config.ValidationEnabled {
		if !essm.isValidAggregateType(aggregate.GetType()) {
			return ErrInvalidAggregateType
		}
	}

	// Save aggregate
	err := essm.service.SaveAggregate(ctx, aggregate)
	if err != nil {
		return ErrAggregateSaveFailed
	}

	return nil
}

// GetEvents retrieves events with validation
func (essm *EventSourcingServiceManager) GetEvents(ctx context.Context, aggregateID string, fromVersion int) ([]Event, error) {
	// Validate input
	if aggregateID == "" {
		return nil, errors.New("aggregate ID cannot be empty")
	}
	if fromVersion < 0 {
		return nil, errors.New("from version cannot be negative")
	}

	// Get events
	events, err := essm.service.GetEvents(ctx, aggregateID, fromVersion)
	if err != nil {
		return nil, ErrEventRetrieveFailed
	}

	return events, nil
}

// GetEventsByType retrieves events by type with validation
func (essm *EventSourcingServiceManager) GetEventsByType(ctx context.Context, eventType string, fromTimestamp time.Time) ([]Event, error) {
	// Validate input
	if eventType == "" {
		return nil, errors.New("event type cannot be empty")
	}

	// Validate event type
	if essm.config.ValidationEnabled {
		if !essm.isValidEventType(eventType) {
			return nil, ErrInvalidEventType
		}
	}

	// Get events
	events, err := essm.service.GetEventsByType(ctx, eventType, fromTimestamp)
	if err != nil {
		return nil, ErrEventRetrieveFailed
	}

	return events, nil
}

// GetEventsByAggregateType retrieves events by aggregate type with validation
func (essm *EventSourcingServiceManager) GetEventsByAggregateType(ctx context.Context, aggregateType string, fromTimestamp time.Time) ([]Event, error) {
	// Validate input
	if aggregateType == "" {
		return nil, errors.New("aggregate type cannot be empty")
	}

	// Validate aggregate type
	if essm.config.ValidationEnabled {
		if !essm.isValidAggregateType(aggregateType) {
			return nil, ErrInvalidAggregateType
		}
	}

	// Get events
	events, err := essm.service.GetEventsByAggregateType(ctx, aggregateType, fromTimestamp)
	if err != nil {
		return nil, ErrEventRetrieveFailed
	}

	return events, nil
}

// GetAllEvents retrieves all events with validation
func (essm *EventSourcingServiceManager) GetAllEvents(ctx context.Context, fromTimestamp time.Time) ([]Event, error) {
	// Get events
	events, err := essm.service.GetAllEvents(ctx, fromTimestamp)
	if err != nil {
		return nil, ErrEventRetrieveFailed
	}

	return events, nil
}

// PublishEvent publishes an event with validation
func (essm *EventSourcingServiceManager) PublishEvent(ctx context.Context, event Event) error {
	// Validate input
	if event == nil {
		return errors.New("event cannot be nil")
	}

	// Validate event type
	if essm.config.ValidationEnabled {
		if !essm.isValidEventType(event.GetType()) {
			return ErrInvalidEventType
		}
	}

	// Publish event
	err := essm.service.PublishEvent(ctx, event)
	if err != nil {
		return ErrEventPublishFailed
	}

	return nil
}

// SubscribeToEvent subscribes to an event type with validation
func (essm *EventSourcingServiceManager) SubscribeToEvent(ctx context.Context, eventType string, handler EventHandler) error {
	// Validate input
	if eventType == "" {
		return errors.New("event type cannot be empty")
	}
	if handler == nil {
		return errors.New("handler cannot be nil")
	}

	// Validate event type
	if essm.config.ValidationEnabled {
		if !essm.isValidEventType(eventType) {
			return ErrInvalidEventType
		}
	}

	// Subscribe to event
	err := essm.service.SubscribeToEvent(ctx, eventType, handler)
	if err != nil {
		return ErrEventSubscribeFailed
	}

	return nil
}

// UnsubscribeFromEvent unsubscribes from an event type with validation
func (essm *EventSourcingServiceManager) UnsubscribeFromEvent(ctx context.Context, eventType string, handler EventHandler) error {
	// Validate input
	if eventType == "" {
		return errors.New("event type cannot be empty")
	}
	if handler == nil {
		return errors.New("handler cannot be nil")
	}

	// Validate event type
	if essm.config.ValidationEnabled {
		if !essm.isValidEventType(eventType) {
			return ErrInvalidEventType
		}
	}

	// Unsubscribe from event
	err := essm.service.UnsubscribeFromEvent(ctx, eventType, handler)
	if err != nil {
		return ErrEventUnsubscribeFailed
	}

	return nil
}

// CreateSnapshot creates a snapshot with validation
func (essm *EventSourcingServiceManager) CreateSnapshot(ctx context.Context, aggregateID string, version int, data map[string]interface{}) error {
	// Validate input
	if aggregateID == "" {
		return errors.New("aggregate ID cannot be empty")
	}
	if version < 0 {
		return errors.New("version cannot be negative")
	}
	if data == nil {
		return errors.New("data cannot be nil")
	}

	// Create snapshot
	err := essm.service.CreateSnapshot(ctx, aggregateID, version, data)
	if err != nil {
		return ErrSnapshotCreateFailed
	}

	return nil
}

// GetSnapshot retrieves a snapshot with validation
func (essm *EventSourcingServiceManager) GetSnapshot(ctx context.Context, aggregateID string) (Snapshot, error) {
	// Validate input
	if aggregateID == "" {
		return nil, errors.New("aggregate ID cannot be empty")
	}

	// Get snapshot
	snapshot, err := essm.service.GetSnapshot(ctx, aggregateID)
	if err != nil {
		return nil, ErrSnapshotRetrieveFailed
	}

	return snapshot, nil
}

// GetLatestSnapshot retrieves the latest snapshot with validation
func (essm *EventSourcingServiceManager) GetLatestSnapshot(ctx context.Context, aggregateID string) (Snapshot, error) {
	// Validate input
	if aggregateID == "" {
		return nil, errors.New("aggregate ID cannot be empty")
	}

	// Get latest snapshot
	snapshot, err := essm.service.GetLatestSnapshot(ctx, aggregateID)
	if err != nil {
		return nil, ErrSnapshotRetrieveFailed
	}

	return snapshot, nil
}

// GetServiceStats returns service statistics
func (essm *EventSourcingServiceManager) GetServiceStats(ctx context.Context) map[string]interface{} {
	return essm.service.GetServiceStats(ctx)
}

// Cleanup performs cleanup operations with validation
func (essm *EventSourcingServiceManager) Cleanup(ctx context.Context, beforeTimestamp time.Time) error {
	// Validate input
	if beforeTimestamp.IsZero() {
		return errors.New("before timestamp cannot be zero")
	}

	// Perform cleanup
	err := essm.service.Cleanup(ctx, beforeTimestamp)
	if err != nil {
		return ErrCleanupFailed
	}

	return nil
}

// GetService returns the underlying event sourcing service
func (essm *EventSourcingServiceManager) GetService() *EventSourcingService {
	return essm.service
}

// GetConfig returns the service configuration
func (essm *EventSourcingServiceManager) GetConfig() *EventSourcingConfig {
	return essm.config
}

// SetConfig sets the service configuration
func (essm *EventSourcingServiceManager) SetConfig(config *EventSourcingConfig) {
	essm.config = config
	essm.service.SetConfig(config)
}

// isValidEventType checks if the event type is valid
func (essm *EventSourcingServiceManager) isValidEventType(eventType string) bool {
	for _, validType := range essm.config.SupportedEventTypes {
		if validType == eventType {
			return true
		}
	}
	return false
}

// isValidAggregateType checks if the aggregate type is valid
func (essm *EventSourcingServiceManager) isValidAggregateType(aggregateType string) bool {
	for _, validType := range essm.config.SupportedAggregateTypes {
		if validType == aggregateType {
			return true
		}
	}
	return false
}

// GetAggregateCount returns the number of aggregates
func (essm *EventSourcingServiceManager) GetAggregateCount() int {
	return essm.service.GetAggregateCount()
}

// GetAggregateTypes returns the list of aggregate types
func (essm *EventSourcingServiceManager) GetAggregateTypes() []string {
	return essm.service.GetAggregateTypes()
}

// GetHealthStatus returns the health status of the service
func (essm *EventSourcingServiceManager) GetHealthStatus() map[string]interface{} {
	return essm.service.GetHealthStatus()
}

// GetServiceInfo returns service information
func (essm *EventSourcingServiceManager) GetServiceInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":                  essm.config.Name,
		"version":               essm.config.Version,
		"description":           essm.config.Description,
		"aggregate_count":       essm.service.GetAggregateCount(),
		"aggregate_types":       essm.service.GetAggregateTypes(),
		"created_at":            essm.service.GetCreatedAt(),
		"updated_at":            essm.service.GetUpdatedAt(),
		"active":                essm.service.IsActive(),
		"metadata":              essm.service.GetMetadata(),
	}
}

// IsActive returns whether the service is active
func (essm *EventSourcingServiceManager) IsActive() bool {
	return essm.service.IsActive()
}

// SetActive sets the active status
func (essm *EventSourcingServiceManager) SetActive(active bool) {
	essm.service.SetActive(active)
}

// GetID returns the service ID
func (essm *EventSourcingServiceManager) GetID() string {
	return essm.service.GetID()
}

// GetName returns the service name
func (essm *EventSourcingServiceManager) GetName() string {
	return essm.service.GetName()
}

// GetDescription returns the service description
func (essm *EventSourcingServiceManager) GetDescription() string {
	return essm.service.GetDescription()
}

// GetType returns the service type
func (essm *EventSourcingServiceManager) GetType() string {
	return "event-sourcing-service-manager"
}

// GetMetadata returns the service metadata
func (essm *EventSourcingServiceManager) GetMetadata() map[string]interface{} {
	return essm.service.GetMetadata()
}

// SetMetadata sets the service metadata
func (essm *EventSourcingServiceManager) SetMetadata(key string, value interface{}) {
	essm.service.SetMetadata(key, value)
}

// GetCreatedAt returns the service creation time
func (essm *EventSourcingServiceManager) GetCreatedAt() time.Time {
	return essm.service.GetCreatedAt()
}

// GetUpdatedAt returns the service last update time
func (essm *EventSourcingServiceManager) GetUpdatedAt() time.Time {
	return essm.service.GetUpdatedAt()
}
