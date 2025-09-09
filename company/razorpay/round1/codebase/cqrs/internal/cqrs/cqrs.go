package cqrs

import (
	"context"
	"time"
)

// CQRSService implements the CQRSService interface
type CQRSService struct {
	config         *CQRSConfig
	commandBus     CommandBus
	queryBus       QueryBus
	eventBus       EventBus
	readModelStore ReadModelStore
	createdAt      time.Time
	updatedAt      time.Time
	active         bool
}

// NewCQRSService creates a new CQRS service
func NewCQRSService(config *CQRSConfig) *CQRSService {
	return &CQRSService{
		config:         config,
		commandBus:     nil,
		queryBus:       nil,
		eventBus:       nil,
		readModelStore: nil,
		createdAt:      time.Now(),
		updatedAt:      time.Now(),
		active:         true,
	}
}

// SendCommand sends a command through the command bus
func (cs *CQRSService) SendCommand(ctx context.Context, command Command) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.commandBus == nil {
		return ErrCommandBusNotFound
	}

	return cs.commandBus.Send(ctx, command)
}

// SendQuery sends a query through the query bus
func (cs *CQRSService) SendQuery(ctx context.Context, query Query) (interface{}, error) {
	if !cs.active {
		return nil, ErrCQRSServiceInactive
	}

	if cs.queryBus == nil {
		return nil, ErrQueryBusNotFound
	}

	return cs.queryBus.Send(ctx, query)
}

// PublishEvent publishes an event through the event bus
func (cs *CQRSService) PublishEvent(ctx context.Context, event Event) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.eventBus == nil {
		return ErrEventBusNotFound
	}

	return cs.eventBus.Publish(ctx, event)
}

// RegisterCommandHandler registers a command handler
func (cs *CQRSService) RegisterCommandHandler(commandType string, handler CommandHandler) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.commandBus == nil {
		return ErrCommandBusNotFound
	}

	return cs.commandBus.RegisterHandler(commandType, handler)
}

// RegisterQueryHandler registers a query handler
func (cs *CQRSService) RegisterQueryHandler(queryType string, handler QueryHandler) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.queryBus == nil {
		return ErrQueryBusNotFound
	}

	return cs.queryBus.RegisterHandler(queryType, handler)
}

// RegisterEventHandler registers an event handler
func (cs *CQRSService) RegisterEventHandler(eventType string, handler EventHandler) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.eventBus == nil {
		return ErrEventBusNotFound
	}

	return cs.eventBus.Subscribe(ctx, eventType, handler)
}

// UnregisterCommandHandler unregisters a command handler
func (cs *CQRSService) UnregisterCommandHandler(commandType string, handler CommandHandler) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.commandBus == nil {
		return ErrCommandBusNotFound
	}

	return cs.commandBus.UnregisterHandler(commandType, handler)
}

// UnregisterQueryHandler unregisters a query handler
func (cs *CQRSService) UnregisterQueryHandler(queryType string, handler QueryHandler) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.queryBus == nil {
		return ErrQueryBusNotFound
	}

	return cs.queryBus.UnregisterHandler(queryType, handler)
}

// UnregisterEventHandler unregisters an event handler
func (cs *CQRSService) UnregisterEventHandler(eventType string, handler EventHandler) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.eventBus == nil {
		return ErrEventBusNotFound
	}

	return cs.eventBus.Unsubscribe(ctx, eventType, handler)
}

// SaveReadModel saves a read model
func (cs *CQRSService) SaveReadModel(ctx context.Context, readModel ReadModel) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.readModelStore == nil {
		return ErrReadModelStoreNotFound
	}

	return cs.readModelStore.Save(ctx, readModel)
}

// GetReadModel retrieves a read model by ID
func (cs *CQRSService) GetReadModel(ctx context.Context, id string) (ReadModel, error) {
	if !cs.active {
		return nil, ErrCQRSServiceInactive
	}

	if cs.readModelStore == nil {
		return nil, ErrReadModelStoreNotFound
	}

	return cs.readModelStore.GetByID(ctx, id)
}

// GetReadModelsByType retrieves read models by type
func (cs *CQRSService) GetReadModelsByType(ctx context.Context, modelType string) ([]ReadModel, error) {
	if !cs.active {
		return nil, ErrCQRSServiceInactive
	}

	if cs.readModelStore == nil {
		return nil, ErrReadModelStoreNotFound
	}

	return cs.readModelStore.GetByType(ctx, modelType)
}

// GetReadModelsByQuery retrieves read models by query
func (cs *CQRSService) GetReadModelsByQuery(ctx context.Context, query map[string]interface{}) ([]ReadModel, error) {
	if !cs.active {
		return nil, ErrCQRSServiceInactive
	}

	if cs.readModelStore == nil {
		return nil, ErrReadModelStoreNotFound
	}

	return cs.readModelStore.GetByQuery(ctx, query)
}

// DeleteReadModel deletes a read model
func (cs *CQRSService) DeleteReadModel(ctx context.Context, id string) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	if cs.readModelStore == nil {
		return ErrReadModelStoreNotFound
	}

	return cs.readModelStore.Delete(ctx, id)
}

// GetServiceStats returns service statistics
func (cs *CQRSService) GetServiceStats(ctx context.Context) map[string]interface{} {
	stats := map[string]interface{}{
		"id":                     cs.GetID(),
		"name":                   cs.GetName(),
		"description":            cs.GetDescription(),
		"active":                 cs.IsActive(),
		"created_at":             cs.GetCreatedAt(),
		"updated_at":             cs.GetUpdatedAt(),
		"command_bus_stats":      make(map[string]interface{}),
		"query_bus_stats":        make(map[string]interface{}),
		"event_bus_stats":        make(map[string]interface{}),
		"read_model_store_stats": make(map[string]interface{}),
		"metadata":               cs.GetMetadata(),
	}

	// Get command bus stats
	if cs.commandBus != nil {
		stats["command_bus_stats"] = cs.commandBus.GetBusStats(ctx)
	}

	// Get query bus stats
	if cs.queryBus != nil {
		stats["query_bus_stats"] = cs.queryBus.GetBusStats(ctx)
	}

	// Get event bus stats
	if cs.eventBus != nil {
		stats["event_bus_stats"] = cs.eventBus.GetBusStats(ctx)
	}

	// Get read model store stats
	if cs.readModelStore != nil {
		stats["read_model_store_stats"] = cs.readModelStore.GetStoreStats(ctx)
	}

	return stats
}

// Cleanup performs cleanup operations
func (cs *CQRSService) Cleanup(ctx context.Context, beforeTimestamp time.Time) error {
	if !cs.active {
		return ErrCQRSServiceInactive
	}

	// Cleanup command bus
	if cs.commandBus != nil {
		if err := cs.commandBus.Cleanup(ctx); err != nil {
			return err
		}
	}

	// Cleanup query bus
	if cs.queryBus != nil {
		if err := cs.queryBus.Cleanup(ctx); err != nil {
			return err
		}
	}

	// Cleanup event bus
	if cs.eventBus != nil {
		if err := cs.eventBus.Cleanup(ctx); err != nil {
			return err
		}
	}

	// Cleanup read model store
	if cs.readModelStore != nil {
		if err := cs.readModelStore.Cleanup(ctx, beforeTimestamp); err != nil {
			return err
		}
	}

	cs.updatedAt = time.Now()

	return nil
}

// IsActive returns whether the service is active
func (cs *CQRSService) IsActive() bool {
	return cs.active
}

// SetActive sets the active status
func (cs *CQRSService) SetActive(active bool) {
	cs.active = active
	cs.updatedAt = time.Now()
}

// GetID returns the service ID
func (cs *CQRSService) GetID() string {
	return "cqrs-service"
}

// GetName returns the service name
func (cs *CQRSService) GetName() string {
	return cs.config.Name
}

// GetDescription returns the service description
func (cs *CQRSService) GetDescription() string {
	return cs.config.Description
}

// GetMetadata returns the service metadata
func (cs *CQRSService) GetMetadata() map[string]interface{} {
	return map[string]interface{}{
		"name":                       cs.config.Name,
		"version":                    cs.config.Version,
		"description":                cs.config.Description,
		"max_commands":               cs.config.MaxCommands,
		"max_queries":                cs.config.MaxQueries,
		"max_events":                 cs.config.MaxEvents,
		"max_read_models":            cs.config.MaxReadModels,
		"cleanup_interval":           cs.config.CleanupInterval,
		"validation_enabled":         cs.config.ValidationEnabled,
		"caching_enabled":            cs.config.CachingEnabled,
		"monitoring_enabled":         cs.config.MonitoringEnabled,
		"auditing_enabled":           cs.config.AuditingEnabled,
		"supported_command_types":    cs.config.SupportedCommandTypes,
		"supported_query_types":      cs.config.SupportedQueryTypes,
		"supported_event_types":      cs.config.SupportedEventTypes,
		"supported_read_model_types": cs.config.SupportedReadModelTypes,
		"validation_rules":           cs.config.ValidationRules,
		"metadata":                   cs.config.Metadata,
	}
}

// SetMetadata sets the service metadata
func (cs *CQRSService) SetMetadata(key string, value interface{}) {
	if cs.config.Metadata == nil {
		cs.config.Metadata = make(map[string]interface{})
	}
	cs.config.Metadata[key] = value
	cs.updatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (cs *CQRSService) GetCreatedAt() time.Time {
	return cs.createdAt
}

// GetUpdatedAt returns the last update time
func (cs *CQRSService) GetUpdatedAt() time.Time {
	return cs.updatedAt
}

// GetConfig returns the service configuration
func (cs *CQRSService) GetConfig() *CQRSConfig {
	return cs.config
}

// SetConfig sets the service configuration
func (cs *CQRSService) SetConfig(config *CQRSConfig) {
	cs.config = config
	cs.updatedAt = time.Now()
}

// GetCommandBus returns the command bus
func (cs *CQRSService) GetCommandBus() CommandBus {
	return cs.commandBus
}

// SetCommandBus sets the command bus
func (cs *CQRSService) SetCommandBus(commandBus CommandBus) {
	cs.commandBus = commandBus
	cs.updatedAt = time.Now()
}

// GetQueryBus returns the query bus
func (cs *CQRSService) GetQueryBus() QueryBus {
	return cs.queryBus
}

// SetQueryBus sets the query bus
func (cs *CQRSService) SetQueryBus(queryBus QueryBus) {
	cs.queryBus = queryBus
	cs.updatedAt = time.Now()
}

// GetEventBus returns the event bus
func (cs *CQRSService) GetEventBus() EventBus {
	return cs.eventBus
}

// SetEventBus sets the event bus
func (cs *CQRSService) SetEventBus(eventBus EventBus) {
	cs.eventBus = eventBus
	cs.updatedAt = time.Now()
}

// GetReadModelStore returns the read model store
func (cs *CQRSService) GetReadModelStore() ReadModelStore {
	return cs.readModelStore
}

// SetReadModelStore sets the read model store
func (cs *CQRSService) SetReadModelStore(readModelStore ReadModelStore) {
	cs.readModelStore = readModelStore
	cs.updatedAt = time.Now()
}

// GetHealthStatus returns the health status of the service
func (cs *CQRSService) GetHealthStatus() map[string]interface{} {
	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"cqrs_service": map[string]interface{}{
				"status": "healthy",
				"active": cs.IsActive(),
			},
			"command_bus": map[string]interface{}{
				"status": "healthy",
				"active": cs.commandBus != nil && cs.commandBus.IsActive(),
			},
			"query_bus": map[string]interface{}{
				"status": "healthy",
				"active": cs.queryBus != nil && cs.queryBus.IsActive(),
			},
			"event_bus": map[string]interface{}{
				"status": "healthy",
				"active": cs.eventBus != nil && cs.eventBus.IsActive(),
			},
			"read_model_store": map[string]interface{}{
				"status": "healthy",
				"active": cs.readModelStore != nil && cs.readModelStore.IsActive(),
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !cs.IsActive() {
		healthStatus["checks"].(map[string]interface{})["cqrs_service"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["cqrs_service"].(map[string]interface{})["message"] = "CQRS service is inactive"
	}

	if cs.commandBus == nil || !cs.commandBus.IsActive() {
		healthStatus["checks"].(map[string]interface{})["command_bus"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["command_bus"].(map[string]interface{})["message"] = "Command bus is not available"
	}

	if cs.queryBus == nil || !cs.queryBus.IsActive() {
		healthStatus["checks"].(map[string]interface{})["query_bus"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["query_bus"].(map[string]interface{})["message"] = "Query bus is not available"
	}

	if cs.eventBus == nil || !cs.eventBus.IsActive() {
		healthStatus["checks"].(map[string]interface{})["event_bus"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["event_bus"].(map[string]interface{})["message"] = "Event bus is not available"
	}

	if cs.readModelStore == nil || !cs.readModelStore.IsActive() {
		healthStatus["checks"].(map[string]interface{})["read_model_store"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["read_model_store"].(map[string]interface{})["message"] = "Read model store is not available"
	}

	return healthStatus
}
