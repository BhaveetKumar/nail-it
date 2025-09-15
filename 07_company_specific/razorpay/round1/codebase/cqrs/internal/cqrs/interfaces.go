package cqrs

import (
	"context"
	"time"
)

// Command represents a command in the CQRS pattern
type Command interface {
	GetID() string
	GetType() string
	GetAggregateID() string
	GetAggregateType() string
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

// Query represents a query in the CQRS pattern
type Query interface {
	GetID() string
	GetType() string
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

// CommandHandler represents a command handler interface
type CommandHandler interface {
	Handle(ctx context.Context, command Command) error
	GetHandlerType() string
	GetCommandTypes() []string
	GetName() string
	GetDescription() string
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	IsActive() bool
	SetActive(active bool)
}

// QueryHandler represents a query handler interface
type QueryHandler interface {
	Handle(ctx context.Context, query Query) (interface{}, error)
	GetHandlerType() string
	GetQueryTypes() []string
	GetName() string
	GetDescription() string
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	IsActive() bool
	SetActive(active bool)
}

// CommandBus represents a command bus interface
type CommandBus interface {
	Send(ctx context.Context, command Command) error
	RegisterHandler(commandType string, handler CommandHandler) error
	UnregisterHandler(commandType string, handler CommandHandler) error
	GetHandlers(ctx context.Context, commandType string) ([]CommandHandler, error)
	GetCommandTypes(ctx context.Context) ([]string, error)
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

// QueryBus represents a query bus interface
type QueryBus interface {
	Send(ctx context.Context, query Query) (interface{}, error)
	RegisterHandler(queryType string, handler QueryHandler) error
	UnregisterHandler(queryType string, handler QueryHandler) error
	GetHandlers(ctx context.Context, queryType string) ([]QueryHandler, error)
	GetQueryTypes(ctx context.Context) ([]string, error)
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

// ReadModel represents a read model interface
type ReadModel interface {
	GetID() string
	GetType() string
	GetData() map[string]interface{}
	GetMetadata() map[string]interface{}
	GetVersion() int
	GetTimestamp() time.Time
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	IsActive() bool
	SetActive(active bool)
	Update(data map[string]interface{}) error
	Delete() error
}

// ReadModelStore represents a read model store interface
type ReadModelStore interface {
	Save(ctx context.Context, readModel ReadModel) error
	GetByID(ctx context.Context, id string) (ReadModel, error)
	GetByType(ctx context.Context, modelType string) ([]ReadModel, error)
	GetByQuery(ctx context.Context, query map[string]interface{}) ([]ReadModel, error)
	Delete(ctx context.Context, id string) error
	GetCount(ctx context.Context, modelType string) (int64, error)
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

// CQRSService represents the main CQRS service
type CQRSService interface {
	SendCommand(ctx context.Context, command Command) error
	SendQuery(ctx context.Context, query Query) (interface{}, error)
	PublishEvent(ctx context.Context, event Event) error
	RegisterCommandHandler(commandType string, handler CommandHandler) error
	RegisterQueryHandler(queryType string, handler QueryHandler) error
	RegisterEventHandler(eventType string, handler EventHandler) error
	UnregisterCommandHandler(commandType string, handler CommandHandler) error
	UnregisterQueryHandler(queryType string, handler QueryHandler) error
	UnregisterEventHandler(eventType string, handler EventHandler) error
	SaveReadModel(ctx context.Context, readModel ReadModel) error
	GetReadModel(ctx context.Context, id string) (ReadModel, error)
	GetReadModelsByType(ctx context.Context, modelType string) ([]ReadModel, error)
	GetReadModelsByQuery(ctx context.Context, query map[string]interface{}) ([]ReadModel, error)
	DeleteReadModel(ctx context.Context, id string) error
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

// ConcreteCommand represents a concrete implementation of Command
type ConcreteCommand struct {
	ID            string                 `json:"id"`
	Type          string                 `json:"type"`
	AggregateID   string                 `json:"aggregate_id"`
	AggregateType string                 `json:"aggregate_type"`
	Data          map[string]interface{} `json:"data"`
	Metadata      map[string]interface{} `json:"metadata"`
	Timestamp     time.Time              `json:"timestamp"`
	CorrelationID string                 `json:"correlation_id"`
	CausationID   string                 `json:"causation_id"`
	Processed     bool                   `json:"processed"`
	ProcessedAt   time.Time              `json:"processed_at"`
}

// GetID returns the command ID
func (cc *ConcreteCommand) GetID() string {
	return cc.ID
}

// GetType returns the command type
func (cc *ConcreteCommand) GetType() string {
	return cc.Type
}

// GetAggregateID returns the aggregate ID
func (cc *ConcreteCommand) GetAggregateID() string {
	return cc.AggregateID
}

// GetAggregateType returns the aggregate type
func (cc *ConcreteCommand) GetAggregateType() string {
	return cc.AggregateType
}

// GetData returns the command data
func (cc *ConcreteCommand) GetData() map[string]interface{} {
	return cc.Data
}

// GetMetadata returns the command metadata
func (cc *ConcreteCommand) GetMetadata() map[string]interface{} {
	return cc.Metadata
}

// GetTimestamp returns the command timestamp
func (cc *ConcreteCommand) GetTimestamp() time.Time {
	return cc.Timestamp
}

// GetCorrelationID returns the correlation ID
func (cc *ConcreteCommand) GetCorrelationID() string {
	return cc.CorrelationID
}

// GetCausationID returns the causation ID
func (cc *ConcreteCommand) GetCausationID() string {
	return cc.CausationID
}

// SetCorrelationID sets the correlation ID
func (cc *ConcreteCommand) SetCorrelationID(correlationID string) {
	cc.CorrelationID = correlationID
}

// SetCausationID sets the causation ID
func (cc *ConcreteCommand) SetCausationID(causationID string) {
	cc.CausationID = causationID
}

// IsProcessed returns whether the command is processed
func (cc *ConcreteCommand) IsProcessed() bool {
	return cc.Processed
}

// SetProcessed sets the processed status
func (cc *ConcreteCommand) SetProcessed(processed bool) {
	cc.Processed = processed
}

// GetProcessedAt returns the processed timestamp
func (cc *ConcreteCommand) GetProcessedAt() time.Time {
	return cc.ProcessedAt
}

// SetProcessedAt sets the processed timestamp
func (cc *ConcreteCommand) SetProcessedAt(processedAt time.Time) {
	cc.ProcessedAt = processedAt
}

// ConcreteQuery represents a concrete implementation of Query
type ConcreteQuery struct {
	ID            string                 `json:"id"`
	Type          string                 `json:"type"`
	Data          map[string]interface{} `json:"data"`
	Metadata      map[string]interface{} `json:"metadata"`
	Timestamp     time.Time              `json:"timestamp"`
	CorrelationID string                 `json:"correlation_id"`
	CausationID   string                 `json:"causation_id"`
	Processed     bool                   `json:"processed"`
	ProcessedAt   time.Time              `json:"processed_at"`
}

// GetID returns the query ID
func (cq *ConcreteQuery) GetID() string {
	return cq.ID
}

// GetType returns the query type
func (cq *ConcreteQuery) GetType() string {
	return cq.Type
}

// GetData returns the query data
func (cq *ConcreteQuery) GetData() map[string]interface{} {
	return cq.Data
}

// GetMetadata returns the query metadata
func (cq *ConcreteQuery) GetMetadata() map[string]interface{} {
	return cq.Metadata
}

// GetTimestamp returns the query timestamp
func (cq *ConcreteQuery) GetTimestamp() time.Time {
	return cq.Timestamp
}

// GetCorrelationID returns the correlation ID
func (cq *ConcreteQuery) GetCorrelationID() string {
	return cq.CorrelationID
}

// GetCausationID returns the causation ID
func (cq *ConcreteQuery) GetCausationID() string {
	return cq.CausationID
}

// SetCorrelationID sets the correlation ID
func (cq *ConcreteQuery) SetCorrelationID(correlationID string) {
	cq.CorrelationID = correlationID
}

// SetCausationID sets the causation ID
func (cq *ConcreteQuery) SetCausationID(causationID string) {
	cq.CausationID = causationID
}

// IsProcessed returns whether the query is processed
func (cq *ConcreteQuery) IsProcessed() bool {
	return cq.Processed
}

// SetProcessed sets the processed status
func (cq *ConcreteQuery) SetProcessed(processed bool) {
	cq.Processed = processed
}

// GetProcessedAt returns the processed timestamp
func (cq *ConcreteQuery) GetProcessedAt() time.Time {
	return cq.ProcessedAt
}

// SetProcessedAt sets the processed timestamp
func (cq *ConcreteQuery) SetProcessedAt(processedAt time.Time) {
	cq.ProcessedAt = processedAt
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

// ConcreteReadModel represents a concrete implementation of ReadModel
type ConcreteReadModel struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Data      map[string]interface{} `json:"data"`
	Metadata  map[string]interface{} `json:"metadata"`
	Version   int                    `json:"version"`
	Timestamp time.Time              `json:"timestamp"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
	Active    bool                   `json:"active"`
}

// GetID returns the read model ID
func (crm *ConcreteReadModel) GetID() string {
	return crm.ID
}

// GetType returns the read model type
func (crm *ConcreteReadModel) GetType() string {
	return crm.Type
}

// GetData returns the read model data
func (crm *ConcreteReadModel) GetData() map[string]interface{} {
	return crm.Data
}

// GetMetadata returns the read model metadata
func (crm *ConcreteReadModel) GetMetadata() map[string]interface{} {
	return crm.Metadata
}

// GetVersion returns the read model version
func (crm *ConcreteReadModel) GetVersion() int {
	return crm.Version
}

// GetTimestamp returns the read model timestamp
func (crm *ConcreteReadModel) GetTimestamp() time.Time {
	return crm.Timestamp
}

// GetCreatedAt returns the creation time
func (crm *ConcreteReadModel) GetCreatedAt() time.Time {
	return crm.CreatedAt
}

// GetUpdatedAt returns the last update time
func (crm *ConcreteReadModel) GetUpdatedAt() time.Time {
	return crm.UpdatedAt
}

// IsActive returns whether the read model is active
func (crm *ConcreteReadModel) IsActive() bool {
	return crm.Active
}

// SetActive sets the active status
func (crm *ConcreteReadModel) SetActive(active bool) {
	crm.Active = active
	crm.UpdatedAt = time.Now()
}

// Update updates the read model data
func (crm *ConcreteReadModel) Update(data map[string]interface{}) error {
	crm.Data = data
	crm.Version++
	crm.UpdatedAt = time.Now()
	return nil
}

// Delete marks the read model as deleted
func (crm *ConcreteReadModel) Delete() error {
	crm.Active = false
	crm.UpdatedAt = time.Now()
	return nil
}

// Utility function to generate unique IDs
func generateID() string {
	return time.Now().Format("20060102150405") + "-" + time.Now().Format("000000000")
}
