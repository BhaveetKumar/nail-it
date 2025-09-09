package unit_of_work

import (
	"context"
	"time"
)

// UnitOfWorkService implements the UnitOfWork interface
type UnitOfWorkService struct {
	config           *UnitOfWorkConfig
	repositories     map[string]Repository
	newEntities      []Entity
	dirtyEntities    []Entity
	deletedEntities  []Entity
	cleanEntities    []Entity
	createdAt        time.Time
	updatedAt        time.Time
	active           bool
}

// NewUnitOfWorkService creates a new unit of work service
func NewUnitOfWorkService(config *UnitOfWorkConfig) *UnitOfWorkService {
	return &UnitOfWorkService{
		config:          config,
		repositories:    make(map[string]Repository),
		newEntities:     make([]Entity, 0),
		dirtyEntities:   make([]Entity, 0),
		deletedEntities: make([]Entity, 0),
		cleanEntities:   make([]Entity, 0),
		createdAt:       time.Now(),
		updatedAt:       time.Now(),
		active:          true,
	}
}

// RegisterNew registers a new entity
func (uows *UnitOfWorkService) RegisterNew(entity Entity) error {
	if !uows.active {
		return ErrUnitOfWorkInactive
	}

	entity.SetNew(true)
	uows.newEntities = append(uows.newEntities, entity)
	uows.updatedAt = time.Now()

	return nil
}

// RegisterDirty registers a dirty entity
func (uows *UnitOfWorkService) RegisterDirty(entity Entity) error {
	if !uows.active {
		return ErrUnitOfWorkInactive
	}

	entity.SetDirty(true)
	uows.dirtyEntities = append(uows.dirtyEntities, entity)
	uows.updatedAt = time.Now()

	return nil
}

// RegisterDeleted registers a deleted entity
func (uows *UnitOfWorkService) RegisterDeleted(entity Entity) error {
	if !uows.active {
		return ErrUnitOfWorkInactive
	}

	entity.SetDeleted(true)
	uows.deletedEntities = append(uows.deletedEntities, entity)
	uows.updatedAt = time.Now()

	return nil
}

// RegisterClean registers a clean entity
func (uows *UnitOfWorkService) RegisterClean(entity Entity) error {
	if !uows.active {
		return ErrUnitOfWorkInactive
	}

	entity.SetDirty(false)
	uows.cleanEntities = append(uows.cleanEntities, entity)
	uows.updatedAt = time.Now()

	return nil
}

// Commit commits the unit of work
func (uows *UnitOfWorkService) Commit() error {
	if !uows.active {
		return ErrUnitOfWorkInactive
	}

	// Process new entities
	for _, entity := range uows.newEntities {
		repository, err := uows.GetRepository(entity.GetType())
		if err != nil {
			return err
		}
		if repository == nil {
			return ErrRepositoryNotFound
		}
		if err := repository.Create(context.Background(), entity); err != nil {
			return err
		}
	}

	// Process dirty entities
	for _, entity := range uows.dirtyEntities {
		repository, err := uows.GetRepository(entity.GetType())
		if err != nil {
			return err
		}
		if repository == nil {
			return ErrRepositoryNotFound
		}
		if err := repository.Update(context.Background(), entity); err != nil {
			return err
		}
	}

	// Process deleted entities
	for _, entity := range uows.deletedEntities {
		repository, err := uows.GetRepository(entity.GetType())
		if err != nil {
			return err
		}
		if repository == nil {
			return ErrRepositoryNotFound
		}
		if err := repository.Delete(context.Background(), entity.GetID()); err != nil {
			return err
		}
	}

	// Clear all entities
	uows.Clear()
	uows.updatedAt = time.Now()

	return nil
}

// Rollback rolls back the unit of work
func (uows *UnitOfWorkService) Rollback() error {
	if !uows.active {
		return ErrUnitOfWorkInactive
	}

	// Clear all entities
	uows.Clear()
	uows.updatedAt = time.Now()

	return nil
}

// GetRepository retrieves a repository by entity type
func (uows *UnitOfWorkService) GetRepository(entityType string) (Repository, error) {
	repository, exists := uows.repositories[entityType]
	if !exists {
		return nil, nil
	}
	return repository, nil
}

// RegisterRepository registers a repository
func (uows *UnitOfWorkService) RegisterRepository(entityType string, repository Repository) error {
	if !uows.active {
		return ErrUnitOfWorkInactive
	}

	uows.repositories[entityType] = repository
	uows.updatedAt = time.Now()

	return nil
}

// GetEntities returns all entities
func (uows *UnitOfWorkService) GetEntities() map[string][]Entity {
	entities := make(map[string][]Entity)
	entities["new"] = uows.newEntities
	entities["dirty"] = uows.dirtyEntities
	entities["deleted"] = uows.deletedEntities
	entities["clean"] = uows.cleanEntities
	return entities
}

// GetNewEntities returns new entities
func (uows *UnitOfWorkService) GetNewEntities() []Entity {
	return uows.newEntities
}

// GetDirtyEntities returns dirty entities
func (uows *UnitOfWorkService) GetDirtyEntities() []Entity {
	return uows.dirtyEntities
}

// GetDeletedEntities returns deleted entities
func (uows *UnitOfWorkService) GetDeletedEntities() []Entity {
	return uows.deletedEntities
}

// GetCleanEntities returns clean entities
func (uows *UnitOfWorkService) GetCleanEntities() []Entity {
	return uows.cleanEntities
}

// Clear clears all entities
func (uows *UnitOfWorkService) Clear() error {
	uows.newEntities = make([]Entity, 0)
	uows.dirtyEntities = make([]Entity, 0)
	uows.deletedEntities = make([]Entity, 0)
	uows.cleanEntities = make([]Entity, 0)
	uows.updatedAt = time.Now()

	return nil
}

// IsActive returns whether the unit of work is active
func (uows *UnitOfWorkService) IsActive() bool {
	return uows.active
}

// SetActive sets the active status
func (uows *UnitOfWorkService) SetActive(active bool) {
	uows.active = active
	uows.updatedAt = time.Now()
}

// GetID returns the unit of work ID
func (uows *UnitOfWorkService) GetID() string {
	return "unit-of-work-service"
}

// GetName returns the unit of work name
func (uows *UnitOfWorkService) GetName() string {
	return uows.config.Name
}

// GetDescription returns the unit of work description
func (uows *UnitOfWorkService) GetDescription() string {
	return uows.config.Description
}

// GetMetadata returns the unit of work metadata
func (uows *UnitOfWorkService) GetMetadata() map[string]interface{} {
	return map[string]interface{}{
		"name":                    uows.config.Name,
		"version":                 uows.config.Version,
		"description":             uows.config.Description,
		"max_entities":            uows.config.MaxEntities,
		"max_repositories":        uows.config.MaxRepositories,
		"transaction_timeout":     uows.config.TransactionTimeout,
		"cleanup_interval":        uows.config.CleanupInterval,
		"validation_enabled":      uows.config.ValidationEnabled,
		"caching_enabled":         uows.config.CachingEnabled,
		"monitoring_enabled":      uows.config.MonitoringEnabled,
		"auditing_enabled":        uows.config.AuditingEnabled,
		"supported_entity_types": uows.config.SupportedEntityTypes,
		"default_entity_type":     uows.config.DefaultEntityType,
		"validation_rules":        uows.config.ValidationRules,
		"metadata":                uows.config.Metadata,
	}
}

// SetMetadata sets the unit of work metadata
func (uows *UnitOfWorkService) SetMetadata(key string, value interface{}) {
	if uows.config.Metadata == nil {
		uows.config.Metadata = make(map[string]interface{})
	}
	uows.config.Metadata[key] = value
	uows.updatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (uows *UnitOfWorkService) GetCreatedAt() time.Time {
	return uows.createdAt
}

// GetUpdatedAt returns the last update time
func (uows *UnitOfWorkService) GetUpdatedAt() time.Time {
	return uows.updatedAt
}

// GetConfig returns the service configuration
func (uows *UnitOfWorkService) GetConfig() *UnitOfWorkConfig {
	return uows.config
}

// SetConfig sets the service configuration
func (uows *UnitOfWorkService) SetConfig(config *UnitOfWorkConfig) {
	uows.config = config
	uows.updatedAt = time.Now()
}

// GetRepositoryCount returns the number of repositories
func (uows *UnitOfWorkService) GetRepositoryCount() int {
	return len(uows.repositories)
}

// GetEntityCount returns the total number of entities
func (uows *UnitOfWorkService) GetEntityCount() int {
	return len(uows.newEntities) + len(uows.dirtyEntities) + len(uows.deletedEntities) + len(uows.cleanEntities)
}

// GetNewEntityCount returns the number of new entities
func (uows *UnitOfWorkService) GetNewEntityCount() int {
	return len(uows.newEntities)
}

// GetDirtyEntityCount returns the number of dirty entities
func (uows *UnitOfWorkService) GetDirtyEntityCount() int {
	return len(uows.dirtyEntities)
}

// GetDeletedEntityCount returns the number of deleted entities
func (uows *UnitOfWorkService) GetDeletedEntityCount() int {
	return len(uows.deletedEntities)
}

// GetCleanEntityCount returns the number of clean entities
func (uows *UnitOfWorkService) GetCleanEntityCount() int {
	return len(uows.cleanEntities)
}

// GetRepositoryTypes returns the list of repository types
func (uows *UnitOfWorkService) GetRepositoryTypes() []string {
	types := make([]string, 0, len(uows.repositories))
	for entityType := range uows.repositories {
		types = append(types, entityType)
	}
	return types
}

// GetEntityTypes returns the list of entity types
func (uows *UnitOfWorkService) GetEntityTypes() []string {
	types := make([]string, 0)
	entityTypes := make(map[string]bool)

	// Collect entity types from all entity lists
	for _, entity := range uows.newEntities {
		entityTypes[entity.GetType()] = true
	}
	for _, entity := range uows.dirtyEntities {
		entityTypes[entity.GetType()] = true
	}
	for _, entity := range uows.deletedEntities {
		entityTypes[entity.GetType()] = true
	}
	for _, entity := range uows.cleanEntities {
		entityTypes[entity.GetType()] = true
	}

	for entityType := range entityTypes {
		types = append(types, entityType)
	}

	return types
}

// GetStats returns unit of work statistics
func (uows *UnitOfWorkService) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"id":                    uows.GetID(),
		"name":                  uows.GetName(),
		"description":           uows.GetDescription(),
		"active":                uows.IsActive(),
		"created_at":            uows.GetCreatedAt(),
		"updated_at":            uows.GetUpdatedAt(),
		"repository_count":      uows.GetRepositoryCount(),
		"entity_count":          uows.GetEntityCount(),
		"new_entity_count":      uows.GetNewEntityCount(),
		"dirty_entity_count":    uows.GetDirtyEntityCount(),
		"deleted_entity_count":  uows.GetDeletedEntityCount(),
		"clean_entity_count":    uows.GetCleanEntityCount(),
		"repository_types":      uows.GetRepositoryTypes(),
		"entity_types":          uows.GetEntityTypes(),
		"metadata":              uows.GetMetadata(),
	}
}

// GetHealthStatus returns the health status of the unit of work
func (uows *UnitOfWorkService) GetHealthStatus() map[string]interface{} {
	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"unit_of_work": map[string]interface{}{
				"status": "healthy",
				"active": uows.IsActive(),
			},
			"repositories": map[string]interface{}{
				"status": "healthy",
				"count":  uows.GetRepositoryCount(),
			},
			"entities": map[string]interface{}{
				"status": "healthy",
				"count":  uows.GetEntityCount(),
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !uows.IsActive() {
		healthStatus["checks"].(map[string]interface{})["unit_of_work"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["unit_of_work"].(map[string]interface{})["message"] = "Unit of work is inactive"
	}

	if uows.GetRepositoryCount() >= uows.config.MaxRepositories {
		healthStatus["checks"].(map[string]interface{})["repositories"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["repositories"].(map[string]interface{})["message"] = "Maximum repositories reached"
	}

	if uows.GetEntityCount() >= uows.config.MaxEntities {
		healthStatus["checks"].(map[string]interface{})["entities"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["entities"].(map[string]interface{})["message"] = "Maximum entities reached"
	}

	return healthStatus
}
