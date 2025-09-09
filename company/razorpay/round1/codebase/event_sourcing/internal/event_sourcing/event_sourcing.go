package event_sourcing

import (
	"context"
	"time"
)

// EventSourcingService implements the EventSourcingService interface
type EventSourcingService struct {
	config        *EventSourcingConfig
	eventStore    EventStore
	eventBus      EventBus
	snapshotStore SnapshotStore
	aggregates    map[string]Aggregate
	createdAt     time.Time
	updatedAt     time.Time
	active        bool
}

// NewEventSourcingService creates a new event sourcing service
func NewEventSourcingService(config *EventSourcingConfig) *EventSourcingService {
	return &EventSourcingService{
		config:        config,
		eventStore:    nil,
		eventBus:      nil,
		snapshotStore: nil,
		aggregates:    make(map[string]Aggregate),
		createdAt:     time.Now(),
		updatedAt:     time.Now(),
		active:        true,
	}
}

// CreateAggregate creates a new aggregate
func (ess *EventSourcingService) CreateAggregate(ctx context.Context, aggregateType string, aggregateID string, initialState map[string]interface{}) (Aggregate, error) {
	if !ess.active {
		return nil, ErrEventSourcingServiceInactive
	}

	var aggregate Aggregate
	switch aggregateType {
	case "user":
		aggregate = NewUserAggregate(aggregateID, aggregateID, "", "", "", "user")
	case "order":
		aggregate = NewOrderAggregate(aggregateID, aggregateID, "", "", 0, "USD")
	case "payment":
		aggregate = NewPaymentAggregate(aggregateID, aggregateID, "", "", 0, "USD", "card", "")
	default:
		aggregate = &ConcreteAggregate{
			ID:                aggregateID,
			Type:              aggregateType,
			Version:           0,
			Events:            make([]Event, 0),
			UncommittedEvents: make([]Event, 0),
			State:             initialState,
			CreatedAt:         time.Now(),
			UpdatedAt:         time.Now(),
			Active:            true,
			Metadata:          make(map[string]interface{}),
		}
	}

	ess.aggregates[aggregateID] = aggregate
	ess.updatedAt = time.Now()

	return aggregate, nil
}

// GetAggregate retrieves an aggregate by ID
func (ess *EventSourcingService) GetAggregate(ctx context.Context, aggregateID string) (Aggregate, error) {
	if !ess.active {
		return nil, ErrEventSourcingServiceInactive
	}

	// Check if aggregate is in memory
	if aggregate, exists := ess.aggregates[aggregateID]; exists {
		return aggregate, nil
	}

	// Load aggregate from event store
	events, err := ess.eventStore.GetEvents(ctx, aggregateID, 0)
	if err != nil {
		return nil, err
	}

	if len(events) == 0 {
		return nil, ErrAggregateNotFound
	}

	// Create aggregate from events
	aggregate := &ConcreteAggregate{
		ID:                aggregateID,
		Type:              events[0].GetAggregateType(),
		Version:           0,
		Events:            make([]Event, 0),
		UncommittedEvents: make([]Event, 0),
		State:             make(map[string]interface{}),
		CreatedAt:         time.Now(),
		UpdatedAt:         time.Now(),
		Active:            true,
		Metadata:          make(map[string]interface{}),
	}

	// Apply events to aggregate
	for _, event := range events {
		if err := aggregate.ApplyEvent(event); err != nil {
			return nil, err
		}
	}

	ess.aggregates[aggregateID] = aggregate
	ess.updatedAt = time.Now()

	return aggregate, nil
}

// SaveAggregate saves an aggregate
func (ess *EventSourcingService) SaveAggregate(ctx context.Context, aggregate Aggregate) error {
	if !ess.active {
		return ErrEventSourcingServiceInactive
	}

	// Save uncommitted events to event store
	uncommittedEvents := aggregate.GetUncommittedEvents()
	if len(uncommittedEvents) > 0 {
		err := ess.eventStore.SaveEvents(ctx, aggregate.GetID(), uncommittedEvents, aggregate.GetVersion()-len(uncommittedEvents))
		if err != nil {
			return err
		}

		// Mark events as committed
		aggregate.MarkEventsAsCommitted()

		// Publish events to event bus
		for _, event := range uncommittedEvents {
			if err := ess.eventBus.Publish(ctx, event); err != nil {
				return err
			}
		}
	}

	// Update aggregate in memory
	ess.aggregates[aggregate.GetID()] = aggregate
	ess.updatedAt = time.Now()

	return nil
}

// GetEvents retrieves events for an aggregate
func (ess *EventSourcingService) GetEvents(ctx context.Context, aggregateID string, fromVersion int) ([]Event, error) {
	if !ess.active {
		return nil, ErrEventSourcingServiceInactive
	}

	return ess.eventStore.GetEvents(ctx, aggregateID, fromVersion)
}

// GetEventsByType retrieves events by type
func (ess *EventSourcingService) GetEventsByType(ctx context.Context, eventType string, fromTimestamp time.Time) ([]Event, error) {
	if !ess.active {
		return nil, ErrEventSourcingServiceInactive
	}

	return ess.eventStore.GetEventsByType(ctx, eventType, fromTimestamp)
}

// GetEventsByAggregateType retrieves events by aggregate type
func (ess *EventSourcingService) GetEventsByAggregateType(ctx context.Context, aggregateType string, fromTimestamp time.Time) ([]Event, error) {
	if !ess.active {
		return nil, ErrEventSourcingServiceInactive
	}

	return ess.eventStore.GetEventsByAggregateType(ctx, aggregateType, fromTimestamp)
}

// GetAllEvents retrieves all events
func (ess *EventSourcingService) GetAllEvents(ctx context.Context, fromTimestamp time.Time) ([]Event, error) {
	if !ess.active {
		return nil, ErrEventSourcingServiceInactive
	}

	return ess.eventStore.GetAllEvents(ctx, fromTimestamp)
}

// PublishEvent publishes an event
func (ess *EventSourcingService) PublishEvent(ctx context.Context, event Event) error {
	if !ess.active {
		return ErrEventSourcingServiceInactive
	}

	return ess.eventBus.Publish(ctx, event)
}

// SubscribeToEvent subscribes to an event type
func (ess *EventSourcingService) SubscribeToEvent(ctx context.Context, eventType string, handler EventHandler) error {
	if !ess.active {
		return ErrEventSourcingServiceInactive
	}

	return ess.eventBus.Subscribe(ctx, eventType, handler)
}

// UnsubscribeFromEvent unsubscribes from an event type
func (ess *EventSourcingService) UnsubscribeFromEvent(ctx context.Context, eventType string, handler EventHandler) error {
	if !ess.active {
		return ErrEventSourcingServiceInactive
	}

	return ess.eventBus.Unsubscribe(ctx, eventType, handler)
}

// CreateSnapshot creates a snapshot
func (ess *EventSourcingService) CreateSnapshot(ctx context.Context, aggregateID string, version int, data map[string]interface{}) error {
	if !ess.active {
		return ErrEventSourcingServiceInactive
	}

	snapshot := &ConcreteSnapshot{
		ID:            generateID(),
		AggregateID:   aggregateID,
		AggregateType: "",
		Version:       version,
		Data:          data,
		Metadata:      make(map[string]interface{}),
		Timestamp:     time.Now(),
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
		Active:        true,
	}

	return ess.snapshotStore.SaveSnapshot(ctx, aggregateID, snapshot)
}

// GetSnapshot retrieves a snapshot
func (ess *EventSourcingService) GetSnapshot(ctx context.Context, aggregateID string) (Snapshot, error) {
	if !ess.active {
		return nil, ErrEventSourcingServiceInactive
	}

	return ess.snapshotStore.GetSnapshot(ctx, aggregateID)
}

// GetLatestSnapshot retrieves the latest snapshot
func (ess *EventSourcingService) GetLatestSnapshot(ctx context.Context, aggregateID string) (Snapshot, error) {
	if !ess.active {
		return nil, ErrEventSourcingServiceInactive
	}

	return ess.snapshotStore.GetLatestSnapshot(ctx, aggregateID)
}

// GetServiceStats returns service statistics
func (ess *EventSourcingService) GetServiceStats(ctx context.Context) map[string]interface{} {
	stats := map[string]interface{}{
		"id":                    ess.GetID(),
		"name":                  ess.GetName(),
		"description":           ess.GetDescription(),
		"active":                ess.IsActive(),
		"created_at":            ess.GetCreatedAt(),
		"updated_at":            ess.GetUpdatedAt(),
		"aggregate_count":       len(ess.aggregates),
		"event_store_stats":     make(map[string]interface{}),
		"event_bus_stats":       make(map[string]interface{}),
		"snapshot_store_stats":  make(map[string]interface{}),
		"metadata":              ess.GetMetadata(),
	}

	// Get event store stats
	if ess.eventStore != nil {
		stats["event_store_stats"] = ess.eventStore.GetStoreStats(ctx)
	}

	// Get event bus stats
	if ess.eventBus != nil {
		stats["event_bus_stats"] = ess.eventBus.GetBusStats(ctx)
	}

	// Get snapshot store stats
	if ess.snapshotStore != nil {
		stats["snapshot_store_stats"] = ess.snapshotStore.GetStoreStats(ctx)
	}

	return stats
}

// Cleanup performs cleanup operations
func (ess *EventSourcingService) Cleanup(ctx context.Context, beforeTimestamp time.Time) error {
	if !ess.active {
		return ErrEventSourcingServiceInactive
	}

	// Cleanup event store
	if ess.eventStore != nil {
		if err := ess.eventStore.Cleanup(ctx, beforeTimestamp); err != nil {
			return err
		}
	}

	// Cleanup event bus
	if ess.eventBus != nil {
		if err := ess.eventBus.Cleanup(ctx); err != nil {
			return err
		}
	}

	// Cleanup snapshot store
	if ess.snapshotStore != nil {
		if err := ess.snapshotStore.Cleanup(ctx, beforeTimestamp); err != nil {
			return err
		}
	}

	// Clear aggregates from memory
	ess.aggregates = make(map[string]Aggregate)
	ess.updatedAt = time.Now()

	return nil
}

// IsActive returns whether the service is active
func (ess *EventSourcingService) IsActive() bool {
	return ess.active
}

// SetActive sets the active status
func (ess *EventSourcingService) SetActive(active bool) {
	ess.active = active
	ess.updatedAt = time.Now()
}

// GetID returns the service ID
func (ess *EventSourcingService) GetID() string {
	return "event-sourcing-service"
}

// GetName returns the service name
func (ess *EventSourcingService) GetName() string {
	return ess.config.Name
}

// GetDescription returns the service description
func (ess *EventSourcingService) GetDescription() string {
	return ess.config.Description
}

// GetMetadata returns the service metadata
func (ess *EventSourcingService) GetMetadata() map[string]interface{} {
	return map[string]interface{}{
		"name":                    ess.config.Name,
		"version":                 ess.config.Version,
		"description":             ess.config.Description,
		"max_events":              ess.config.MaxEvents,
		"max_aggregates":          ess.config.MaxAggregates,
		"max_snapshots":           ess.config.MaxSnapshots,
		"snapshot_interval":       ess.config.SnapshotInterval,
		"cleanup_interval":        ess.config.CleanupInterval,
		"validation_enabled":      ess.config.ValidationEnabled,
		"caching_enabled":         ess.config.CachingEnabled,
		"monitoring_enabled":      ess.config.MonitoringEnabled,
		"auditing_enabled":        ess.config.AuditingEnabled,
		"supported_event_types":   ess.config.SupportedEventTypes,
		"supported_aggregate_types": ess.config.SupportedAggregateTypes,
		"validation_rules":        ess.config.ValidationRules,
		"metadata":                ess.config.Metadata,
	}
}

// SetMetadata sets the service metadata
func (ess *EventSourcingService) SetMetadata(key string, value interface{}) {
	if ess.config.Metadata == nil {
		ess.config.Metadata = make(map[string]interface{})
	}
	ess.config.Metadata[key] = value
	ess.updatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (ess *EventSourcingService) GetCreatedAt() time.Time {
	return ess.createdAt
}

// GetUpdatedAt returns the last update time
func (ess *EventSourcingService) GetUpdatedAt() time.Time {
	return ess.updatedAt
}

// GetConfig returns the service configuration
func (ess *EventSourcingService) GetConfig() *EventSourcingConfig {
	return ess.config
}

// SetConfig sets the service configuration
func (ess *EventSourcingService) SetConfig(config *EventSourcingConfig) {
	ess.config = config
	ess.updatedAt = time.Now()
}

// GetEventStore returns the event store
func (ess *EventSourcingService) GetEventStore() EventStore {
	return ess.eventStore
}

// SetEventStore sets the event store
func (ess *EventSourcingService) SetEventStore(eventStore EventStore) {
	ess.eventStore = eventStore
	ess.updatedAt = time.Now()
}

// GetEventBus returns the event bus
func (ess *EventSourcingService) GetEventBus() EventBus {
	return ess.eventBus
}

// SetEventBus sets the event bus
func (ess *EventSourcingService) SetEventBus(eventBus EventBus) {
	ess.eventBus = eventBus
	ess.updatedAt = time.Now()
}

// GetSnapshotStore returns the snapshot store
func (ess *EventSourcingService) GetSnapshotStore() SnapshotStore {
	return ess.snapshotStore
}

// SetSnapshotStore sets the snapshot store
func (ess *EventSourcingService) SetSnapshotStore(snapshotStore SnapshotStore) {
	ess.snapshotStore = snapshotStore
	ess.updatedAt = time.Now()
}

// GetAggregateCount returns the number of aggregates
func (ess *EventSourcingService) GetAggregateCount() int {
	return len(ess.aggregates)
}

// GetAggregateTypes returns the list of aggregate types
func (ess *EventSourcingService) GetAggregateTypes() []string {
	types := make([]string, 0)
	aggregateTypes := make(map[string]bool)

	for _, aggregate := range ess.aggregates {
		aggregateTypes[aggregate.GetType()] = true
	}

	for aggregateType := range aggregateTypes {
		types = append(types, aggregateType)
	}

	return types
}

// GetHealthStatus returns the health status of the service
func (ess *EventSourcingService) GetHealthStatus() map[string]interface{} {
	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"event_sourcing_service": map[string]interface{}{
				"status": "healthy",
				"active": ess.IsActive(),
			},
			"event_store": map[string]interface{}{
				"status": "healthy",
				"active": ess.eventStore != nil && ess.eventStore.IsActive(),
			},
			"event_bus": map[string]interface{}{
				"status": "healthy",
				"active": ess.eventBus != nil && ess.eventBus.IsActive(),
			},
			"snapshot_store": map[string]interface{}{
				"status": "healthy",
				"active": ess.snapshotStore != nil && ess.snapshotStore.IsActive(),
			},
			"aggregates": map[string]interface{}{
				"status": "healthy",
				"count":  ess.GetAggregateCount(),
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !ess.IsActive() {
		healthStatus["checks"].(map[string]interface{})["event_sourcing_service"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["event_sourcing_service"].(map[string]interface{})["message"] = "Event sourcing service is inactive"
	}

	if ess.eventStore == nil || !ess.eventStore.IsActive() {
		healthStatus["checks"].(map[string]interface{})["event_store"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["event_store"].(map[string]interface{})["message"] = "Event store is not available"
	}

	if ess.eventBus == nil || !ess.eventBus.IsActive() {
		healthStatus["checks"].(map[string]interface{})["event_bus"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["event_bus"].(map[string]interface{})["message"] = "Event bus is not available"
	}

	if ess.snapshotStore == nil || !ess.snapshotStore.IsActive() {
		healthStatus["checks"].(map[string]interface{})["snapshot_store"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["snapshot_store"].(map[string]interface{})["message"] = "Snapshot store is not available"
	}

	if ess.GetAggregateCount() >= ess.config.MaxAggregates {
		healthStatus["checks"].(map[string]interface{})["aggregates"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["aggregates"].(map[string]interface{})["message"] = "Maximum aggregates reached"
	}

	return healthStatus
}
