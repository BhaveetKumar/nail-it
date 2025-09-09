package cqrs

import (
	"context"
	"time"
)

// Service implements the Service interface
type Service struct {
	config        *CQRSConfig
	commandBus    CommandBus
	queryBus      QueryBus
	eventBus      EventBus
	readModelStore ReadModelStore
	createdAt     time.Time
	updatedAt     time.Time
	active        bool
}

// NewService creates a new CQRS service
func NewService(config *CQRSConfig) *Service {
	return &Service{
		config:        config,
		commandBus:    nil,
		queryBus:      nil,
		eventBus:      nil,
		readModelStore: nil,
		createdAt:     time.Now(),
		updatedAt:     time.Now(),
		active:        true,
	}
}

// SendCommand sends a command through the command bus
func (s *Service) SendCommand(ctx context.Context, command Command) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.commandBus == nil {
		return ErrCommandBusNotFound
	}

	return s.commandBus.Send(ctx, command)
}

// SendQuery sends a query through the query bus
func (s *Service) SendQuery(ctx context.Context, query Query) (interface{}, error) {
	if !s.active {
		return nil, ErrCQRSServiceInactive
	}

	if s.queryBus == nil {
		return nil, ErrQueryBusNotFound
	}

	return s.queryBus.Send(ctx, query)
}

// PublishEvent publishes an event through the event bus
func (s *Service) PublishEvent(ctx context.Context, event Event) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.eventBus == nil {
		return ErrEventBusNotFound
	}

	return s.eventBus.Publish(ctx, event)
}

// RegisterCommandHandler registers a command handler
func (s *Service) RegisterCommandHandler(commandType string, handler CommandHandler) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.commandBus == nil {
		return ErrCommandBusNotFound
	}

	return s.commandBus.RegisterHandler(commandType, handler)
}

// RegisterQueryHandler registers a query handler
func (s *Service) RegisterQueryHandler(queryType string, handler QueryHandler) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.queryBus == nil {
		return ErrQueryBusNotFound
	}

	return s.queryBus.RegisterHandler(queryType, handler)
}

// RegisterEventHandler registers an event handler
func (s *Service) RegisterEventHandler(eventType string, handler EventHandler) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.eventBus == nil {
		return ErrEventBusNotFound
	}

	return s.eventBus.Subscribe(ctx, eventType, handler)
}

// UnregisterCommandHandler unregisters a command handler
func (s *Service) UnregisterCommandHandler(commandType string, handler CommandHandler) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.commandBus == nil {
		return ErrCommandBusNotFound
	}

	return s.commandBus.UnregisterHandler(commandType, handler)
}

// UnregisterQueryHandler unregisters a query handler
func (s *Service) UnregisterQueryHandler(queryType string, handler QueryHandler) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.queryBus == nil {
		return ErrQueryBusNotFound
	}

	return s.queryBus.UnregisterHandler(queryType, handler)
}

// UnregisterEventHandler unregisters an event handler
func (s *Service) UnregisterEventHandler(eventType string, handler EventHandler) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.eventBus == nil {
		return ErrEventBusNotFound
	}

	return s.eventBus.Unsubscribe(ctx, eventType, handler)
}

// SaveReadModel saves a read model
func (s *Service) SaveReadModel(ctx context.Context, readModel ReadModel) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.readModelStore == nil {
		return ErrReadModelStoreNotFound
	}

	return s.readModelStore.Save(ctx, readModel)
}

// GetReadModel retrieves a read model by ID
func (s *Service) GetReadModel(ctx context.Context, id string) (ReadModel, error) {
	if !s.active {
		return nil, ErrCQRSServiceInactive
	}

	if s.readModelStore == nil {
		return nil, ErrReadModelStoreNotFound
	}

	return s.readModelStore.GetByID(ctx, id)
}

// GetReadModelsByType retrieves read models by type
func (s *Service) GetReadModelsByType(ctx context.Context, modelType string) ([]ReadModel, error) {
	if !s.active {
		return nil, ErrCQRSServiceInactive
	}

	if s.readModelStore == nil {
		return nil, ErrReadModelStoreNotFound
	}

	return s.readModelStore.GetByType(ctx, modelType)
}

// GetReadModelsByQuery retrieves read models by query
func (s *Service) GetReadModelsByQuery(ctx context.Context, query map[string]interface{}) ([]ReadModel, error) {
	if !s.active {
		return nil, ErrCQRSServiceInactive
	}

	if s.readModelStore == nil {
		return nil, ErrReadModelStoreNotFound
	}

	return s.readModelStore.GetByQuery(ctx, query)
}

// DeleteReadModel deletes a read model
func (s *Service) DeleteReadModel(ctx context.Context, id string) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	if s.readModelStore == nil {
		return ErrReadModelStoreNotFound
	}

	return s.readModelStore.Delete(ctx, id)
}

// GetServiceStats returns service statistics
func (s *Service) GetServiceStats(ctx context.Context) map[string]interface{} {
	stats := map[string]interface{}{
		"id":                    s.GetID(),
		"name":                  s.GetName(),
		"description":           s.GetDescription(),
		"active":                s.IsActive(),
		"created_at":            s.GetCreatedAt(),
		"updated_at":            s.GetUpdatedAt(),
		"command_bus_stats":     make(map[string]interface{}),
		"query_bus_stats":       make(map[string]interface{}),
		"event_bus_stats":       make(map[string]interface{}),
		"read_model_store_stats": make(map[string]interface{}),
		"metadata":              s.GetMetadata(),
	}

	// Get command bus stats
	if s.commandBus != nil {
		stats["command_bus_stats"] = s.commandBus.GetBusStats(ctx)
	}

	// Get query bus stats
	if s.queryBus != nil {
		stats["query_bus_stats"] = s.queryBus.GetBusStats(ctx)
	}

	// Get event bus stats
	if s.eventBus != nil {
		stats["event_bus_stats"] = s.eventBus.GetBusStats(ctx)
	}

	// Get read model store stats
	if s.readModelStore != nil {
		stats["read_model_store_stats"] = s.readModelStore.GetStoreStats(ctx)
	}

	return stats
}

// Cleanup performs cleanup operations
func (s *Service) Cleanup(ctx context.Context, beforeTimestamp time.Time) error {
	if !s.active {
		return ErrCQRSServiceInactive
	}

	// Cleanup command bus
	if s.commandBus != nil {
		if err := s.commandBus.Cleanup(ctx); err != nil {
			return err
		}
	}

	// Cleanup query bus
	if s.queryBus != nil {
		if err := s.queryBus.Cleanup(ctx); err != nil {
			return err
		}
	}

	// Cleanup event bus
	if s.eventBus != nil {
		if err := s.eventBus.Cleanup(ctx); err != nil {
			return err
		}
	}

	// Cleanup read model store
	if s.readModelStore != nil {
		if err := s.readModelStore.Cleanup(ctx, beforeTimestamp); err != nil {
			return err
		}
	}

	s.updatedAt = time.Now()

	return nil
}

// IsActive returns whether the service is active
func (s *Service) IsActive() bool {
	return s.active
}

// SetActive sets the active status
func (s *Service) SetActive(active bool) {
	s.active = active
	s.updatedAt = time.Now()
}

// GetID returns the service ID
func (s *Service) GetID() string {
	return "cqrs-service"
}

// GetName returns the service name
func (s *Service) GetName() string {
	return s.config.Name
}

// GetDescription returns the service description
func (s *Service) GetDescription() string {
	return s.config.Description
}

// GetMetadata returns the service metadata
func (s *Service) GetMetadata() map[string]interface{} {
	return map[string]interface{}{
		"name":                    s.config.Name,
		"version":                 s.config.Version,
		"description":             s.config.Description,
		"max_commands":            s.config.MaxCommands,
		"max_queries":             s.config.MaxQueries,
		"max_events":              s.config.MaxEvents,
		"max_read_models":         s.config.MaxReadModels,
		"cleanup_interval":        s.config.CleanupInterval,
		"validation_enabled":      s.config.ValidationEnabled,
		"caching_enabled":         s.config.CachingEnabled,
		"monitoring_enabled":      s.config.MonitoringEnabled,
		"auditing_enabled":        s.config.AuditingEnabled,
		"supported_command_types": s.config.SupportedCommandTypes,
		"supported_query_types":   s.config.SupportedQueryTypes,
		"supported_event_types":   s.config.SupportedEventTypes,
		"supported_read_model_types": s.config.SupportedReadModelTypes,
		"validation_rules":        s.config.ValidationRules,
		"metadata":                s.config.Metadata,
	}
}

// SetMetadata sets the service metadata
func (s *Service) SetMetadata(key string, value interface{}) {
	if s.config.Metadata == nil {
		s.config.Metadata = make(map[string]interface{})
	}
	s.config.Metadata[key] = value
	s.updatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (s *Service) GetCreatedAt() time.Time {
	return s.createdAt
}

// GetUpdatedAt returns the last update time
func (s *Service) GetUpdatedAt() time.Time {
	return s.updatedAt
}

// GetConfig returns the service configuration
func (s *Service) GetConfig() *CQRSConfig {
	return s.config
}

// SetConfig sets the service configuration
func (s *Service) SetConfig(config *CQRSConfig) {
	s.config = config
	s.updatedAt = time.Now()
}

// GetCommandBus returns the command bus
func (s *Service) GetCommandBus() CommandBus {
	return s.commandBus
}

// SetCommandBus sets the command bus
func (s *Service) SetCommandBus(commandBus CommandBus) {
	s.commandBus = commandBus
	s.updatedAt = time.Now()
}

// GetQueryBus returns the query bus
func (s *Service) GetQueryBus() QueryBus {
	return s.queryBus
}

// SetQueryBus sets the query bus
func (s *Service) SetQueryBus(queryBus QueryBus) {
	s.queryBus = queryBus
	s.updatedAt = time.Now()
}

// GetEventBus returns the event bus
func (s *Service) GetEventBus() EventBus {
	return s.eventBus
}

// SetEventBus sets the event bus
func (s *Service) SetEventBus(eventBus EventBus) {
	s.eventBus = eventBus
	s.updatedAt = time.Now()
}

// GetReadModelStore returns the read model store
func (s *Service) GetReadModelStore() ReadModelStore {
	return s.readModelStore
}

// SetReadModelStore sets the read model store
func (s *Service) SetReadModelStore(readModelStore ReadModelStore) {
	s.readModelStore = readModelStore
	s.updatedAt = time.Now()
}

// GetHealthStatus returns the health status of the service
func (s *Service) GetHealthStatus() map[string]interface{} {
	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"cqrs_service": map[string]interface{}{
				"status": "healthy",
				"active": s.IsActive(),
			},
			"command_bus": map[string]interface{}{
				"status": "healthy",
				"active": s.commandBus != nil && s.commandBus.IsActive(),
			},
			"query_bus": map[string]interface{}{
				"status": "healthy",
				"active": s.queryBus != nil && s.queryBus.IsActive(),
			},
			"event_bus": map[string]interface{}{
				"status": "healthy",
				"active": s.eventBus != nil && s.eventBus.IsActive(),
			},
			"read_model_store": map[string]interface{}{
				"status": "healthy",
				"active": s.readModelStore != nil && s.readModelStore.IsActive(),
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !s.IsActive() {
		healthStatus["checks"].(map[string]interface{})["cqrs_service"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["cqrs_service"].(map[string]interface{})["message"] = "CQRS service is inactive"
	}

	if s.commandBus == nil || !s.commandBus.IsActive() {
		healthStatus["checks"].(map[string]interface{})["command_bus"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["command_bus"].(map[string]interface{})["message"] = "Command bus is not available"
	}

	if s.queryBus == nil || !s.queryBus.IsActive() {
		healthStatus["checks"].(map[string]interface{})["query_bus"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["query_bus"].(map[string]interface{})["message"] = "Query bus is not available"
	}

	if s.eventBus == nil || !s.eventBus.IsActive() {
		healthStatus["checks"].(map[string]interface{})["event_bus"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["event_bus"].(map[string]interface{})["message"] = "Event bus is not available"
	}

	if s.readModelStore == nil || !s.readModelStore.IsActive() {
		healthStatus["checks"].(map[string]interface{})["read_model_store"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["read_model_store"].(map[string]interface{})["message"] = "Read model store is not available"
	}

	return healthStatus
}
