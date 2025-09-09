package event_sourcing

import (
	"context"
	"time"
)

// Event represents a domain event
type Event interface {
	GetID() string
	GetType() string
	GetAggregateID() string
	GetAggregateType() string
	GetVersion() int
	GetData() map[string]interface{}
	GetMetadata() map[string]interface{}
	GetTimestamp() time.Time
	GetCorrelationID() string
	GetCausationID() string
	SetCorrelationID(correlationID string)
	SetCausationID(causationID string)
	IsProcessed() bool
	SetProcessed(processed bool)
	GetProcessedAt() time.Time
	SetProcessedAt(processedAt time.Time)
}

// Aggregate represents a domain aggregate
type Aggregate interface {
	GetID() string
	GetType() string
	GetVersion() int
	GetEvents() []Event
	GetUncommittedEvents() []Event
	MarkEventsAsCommitted()
	ApplyEvent(event Event) error
	GetState() map[string]interface{}
	SetState(state map[string]interface{})
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	IsActive() bool
	SetActive(active bool)
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
}

// EventStore represents an event store interface
type EventStore interface {
	SaveEvents(ctx context.Context, aggregateID string, events []Event, expectedVersion int) error
	GetEvents(ctx context.Context, aggregateID string, fromVersion int) ([]Event, error)
	GetEventsByType(ctx context.Context, eventType string, fromTimestamp time.Time) ([]Event, error)
	GetEventsByAggregateType(ctx context.Context, aggregateType string, fromTimestamp time.Time) ([]Event, error)
	GetAllEvents(ctx context.Context, fromTimestamp time.Time) ([]Event, error)
	GetEventCount(ctx context.Context, aggregateID string) (int64, error)
	GetAggregateCount(ctx context.Context) (int64, error)
	GetEventTypeCount(ctx context.Context) (int64, error)
	GetStoreStats(ctx context.Context) map[string]interface{}
	Cleanup(ctx context.Context, beforeTimestamp time.Time) error
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

// EventBus represents an event bus interface
type EventBus interface {
	Publish(ctx context.Context, event Event) error
	Subscribe(ctx context.Context, eventType string, handler EventHandler) error
	Unsubscribe(ctx context.Context, eventType string, handler EventHandler) error
	GetSubscribers(ctx context.Context, eventType string) ([]EventHandler, error)
	GetEventTypes(ctx context.Context) ([]string, error)
	GetHandlerCount(ctx context.Context) (int64, error)
	GetBusStats(ctx context.Context) map[string]interface{}
	Cleanup(ctx context.Context) error
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

// EventHandler represents an event handler interface
type EventHandler interface {
	Handle(ctx context.Context, event Event) error
	GetHandlerType() string
	GetEventTypes() []string
	GetName() string
	GetDescription() string
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	IsActive() bool
	SetActive(active bool)
}

// SnapshotStore represents a snapshot store interface
type SnapshotStore interface {
	SaveSnapshot(ctx context.Context, aggregateID string, snapshot Snapshot) error
	GetSnapshot(ctx context.Context, aggregateID string) (Snapshot, error)
	GetLatestSnapshot(ctx context.Context, aggregateID string) (Snapshot, error)
	GetSnapshotsByType(ctx context.Context, aggregateType string) ([]Snapshot, error)
	GetSnapshotCount(ctx context.Context, aggregateID string) (int64, error)
	GetStoreStats(ctx context.Context) map[string]interface{}
	Cleanup(ctx context.Context, beforeTimestamp time.Time) error
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

// Snapshot represents a snapshot of an aggregate
type Snapshot interface {
	GetID() string
	GetAggregateID() string
	GetAggregateType() string
	GetVersion() int
	GetData() map[string]interface{}
	GetMetadata() map[string]interface{}
	GetTimestamp() time.Time
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	IsActive() bool
	SetActive(active bool)
}

// EventSourcingService represents the main event sourcing service
type EventSourcingService interface {
	CreateAggregate(ctx context.Context, aggregateType string, aggregateID string, initialState map[string]interface{}) (Aggregate, error)
	GetAggregate(ctx context.Context, aggregateID string) (Aggregate, error)
	SaveAggregate(ctx context.Context, aggregate Aggregate) error
	GetEvents(ctx context.Context, aggregateID string, fromVersion int) ([]Event, error)
	GetEventsByType(ctx context.Context, eventType string, fromTimestamp time.Time) ([]Event, error)
	GetEventsByAggregateType(ctx context.Context, aggregateType string, fromTimestamp time.Time) ([]Event, error)
	GetAllEvents(ctx context.Context, fromTimestamp time.Time) ([]Event, error)
	PublishEvent(ctx context.Context, event Event) error
	SubscribeToEvent(ctx context.Context, eventType string, handler EventHandler) error
	UnsubscribeFromEvent(ctx context.Context, eventType string, handler EventHandler) error
	CreateSnapshot(ctx context.Context, aggregateID string, version int, data map[string]interface{}) error
	GetSnapshot(ctx context.Context, aggregateID string) (Snapshot, error)
	GetLatestSnapshot(ctx context.Context, aggregateID string) (Snapshot, error)
	GetServiceStats(ctx context.Context) map[string]interface{}
	Cleanup(ctx context.Context, beforeTimestamp time.Time) error
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

// ConcreteEvent represents a concrete implementation of Event
type ConcreteEvent struct {
	ID            string                 `json:"id"`
	Type          string                 `json:"type"`
	AggregateID   string                 `json:"aggregate_id"`
	AggregateType string                 `json:"aggregate_type"`
	Version       int                    `json:"version"`
	Data          map[string]interface{} `json:"data"`
	Metadata      map[string]interface{} `json:"metadata"`
	Timestamp     time.Time              `json:"timestamp"`
	CorrelationID string                 `json:"correlation_id"`
	CausationID   string                 `json:"causation_id"`
	Processed     bool                   `json:"processed"`
	ProcessedAt   time.Time              `json:"processed_at"`
}

// GetID returns the event ID
func (ce *ConcreteEvent) GetID() string {
	return ce.ID
}

// GetType returns the event type
func (ce *ConcreteEvent) GetType() string {
	return ce.Type
}

// GetAggregateID returns the aggregate ID
func (ce *ConcreteEvent) GetAggregateID() string {
	return ce.AggregateID
}

// GetAggregateType returns the aggregate type
func (ce *ConcreteEvent) GetAggregateType() string {
	return ce.AggregateType
}

// GetVersion returns the event version
func (ce *ConcreteEvent) GetVersion() int {
	return ce.Version
}

// GetData returns the event data
func (ce *ConcreteEvent) GetData() map[string]interface{} {
	return ce.Data
}

// GetMetadata returns the event metadata
func (ce *ConcreteEvent) GetMetadata() map[string]interface{} {
	return ce.Metadata
}

// GetTimestamp returns the event timestamp
func (ce *ConcreteEvent) GetTimestamp() time.Time {
	return ce.Timestamp
}

// GetCorrelationID returns the correlation ID
func (ce *ConcreteEvent) GetCorrelationID() string {
	return ce.CorrelationID
}

// GetCausationID returns the causation ID
func (ce *ConcreteEvent) GetCausationID() string {
	return ce.CausationID
}

// SetCorrelationID sets the correlation ID
func (ce *ConcreteEvent) SetCorrelationID(correlationID string) {
	ce.CorrelationID = correlationID
}

// SetCausationID sets the causation ID
func (ce *ConcreteEvent) SetCausationID(causationID string) {
	ce.CausationID = causationID
}

// IsProcessed returns whether the event is processed
func (ce *ConcreteEvent) IsProcessed() bool {
	return ce.Processed
}

// SetProcessed sets the processed status
func (ce *ConcreteEvent) SetProcessed(processed bool) {
	ce.Processed = processed
}

// GetProcessedAt returns the processed timestamp
func (ce *ConcreteEvent) GetProcessedAt() time.Time {
	return ce.ProcessedAt
}

// SetProcessedAt sets the processed timestamp
func (ce *ConcreteEvent) SetProcessedAt(processedAt time.Time) {
	ce.ProcessedAt = processedAt
}

// ConcreteAggregate represents a concrete implementation of Aggregate
type ConcreteAggregate struct {
	ID                string                 `json:"id"`
	Type              string                 `json:"type"`
	Version           int                    `json:"version"`
	Events            []Event                `json:"events"`
	UncommittedEvents []Event                `json:"uncommitted_events"`
	State             map[string]interface{} `json:"state"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
	Active            bool                   `json:"active"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// GetID returns the aggregate ID
func (ca *ConcreteAggregate) GetID() string {
	return ca.ID
}

// GetType returns the aggregate type
func (ca *ConcreteAggregate) GetType() string {
	return ca.Type
}

// GetVersion returns the aggregate version
func (ca *ConcreteAggregate) GetVersion() int {
	return ca.Version
}

// GetEvents returns all events
func (ca *ConcreteAggregate) GetEvents() []Event {
	return ca.Events
}

// GetUncommittedEvents returns uncommitted events
func (ca *ConcreteAggregate) GetUncommittedEvents() []Event {
	return ca.UncommittedEvents
}

// MarkEventsAsCommitted marks events as committed
func (ca *ConcreteAggregate) MarkEventsAsCommitted() {
	ca.Events = append(ca.Events, ca.UncommittedEvents...)
	ca.UncommittedEvents = make([]Event, 0)
	ca.UpdatedAt = time.Now()
}

// ApplyEvent applies an event to the aggregate
func (ca *ConcreteAggregate) ApplyEvent(event Event) error {
	ca.UncommittedEvents = append(ca.UncommittedEvents, event)
	ca.Version++
	ca.UpdatedAt = time.Now()
	return nil
}

// GetState returns the aggregate state
func (ca *ConcreteAggregate) GetState() map[string]interface{} {
	return ca.State
}

// SetState sets the aggregate state
func (ca *ConcreteAggregate) SetState(state map[string]interface{}) {
	ca.State = state
	ca.UpdatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (ca *ConcreteAggregate) GetCreatedAt() time.Time {
	return ca.CreatedAt
}

// GetUpdatedAt returns the last update time
func (ca *ConcreteAggregate) GetUpdatedAt() time.Time {
	return ca.UpdatedAt
}

// IsActive returns whether the aggregate is active
func (ca *ConcreteAggregate) IsActive() bool {
	return ca.Active
}

// SetActive sets the active status
func (ca *ConcreteAggregate) SetActive(active bool) {
	ca.Active = active
	ca.UpdatedAt = time.Now()
}

// GetMetadata returns the aggregate metadata
func (ca *ConcreteAggregate) GetMetadata() map[string]interface{} {
	return ca.Metadata
}

// SetMetadata sets a metadata key-value pair
func (ca *ConcreteAggregate) SetMetadata(key string, value interface{}) {
	if ca.Metadata == nil {
		ca.Metadata = make(map[string]interface{})
	}
	ca.Metadata[key] = value
	ca.UpdatedAt = time.Now()
}

// ConcreteSnapshot represents a concrete implementation of Snapshot
type ConcreteSnapshot struct {
	ID            string                 `json:"id"`
	AggregateID   string                 `json:"aggregate_id"`
	AggregateType string                 `json:"aggregate_type"`
	Version       int                    `json:"version"`
	Data          map[string]interface{} `json:"data"`
	Metadata      map[string]interface{} `json:"metadata"`
	Timestamp     time.Time              `json:"timestamp"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
	Active        bool                   `json:"active"`
}

// GetID returns the snapshot ID
func (cs *ConcreteSnapshot) GetID() string {
	return cs.ID
}

// GetAggregateID returns the aggregate ID
func (cs *ConcreteSnapshot) GetAggregateID() string {
	return cs.AggregateID
}

// GetAggregateType returns the aggregate type
func (cs *ConcreteSnapshot) GetAggregateType() string {
	return cs.AggregateType
}

// GetVersion returns the snapshot version
func (cs *ConcreteSnapshot) GetVersion() int {
	return cs.Version
}

// GetData returns the snapshot data
func (cs *ConcreteSnapshot) GetData() map[string]interface{} {
	return cs.Data
}

// GetMetadata returns the snapshot metadata
func (cs *ConcreteSnapshot) GetMetadata() map[string]interface{} {
	return cs.Metadata
}

// GetTimestamp returns the snapshot timestamp
func (cs *ConcreteSnapshot) GetTimestamp() time.Time {
	return cs.Timestamp
}

// GetCreatedAt returns the creation time
func (cs *ConcreteSnapshot) GetCreatedAt() time.Time {
	return cs.CreatedAt
}

// GetUpdatedAt returns the last update time
func (cs *ConcreteSnapshot) GetUpdatedAt() time.Time {
	return cs.UpdatedAt
}

// IsActive returns whether the snapshot is active
func (cs *ConcreteSnapshot) IsActive() bool {
	return cs.Active
}

// SetActive sets the active status
func (cs *ConcreteSnapshot) SetActive(active bool) {
	cs.Active = active
	cs.UpdatedAt = time.Now()
}

// Utility function to generate unique IDs
func generateID() string {
	return time.Now().Format("20060102150405") + "-" + time.Now().Format("000000000")
}
