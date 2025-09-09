package unit_of_work

import (
	"context"
	"errors"
	"time"
)

// Custom errors
var (
	ErrUnitOfWorkInactive     = errors.New("unit of work is inactive")
	ErrRepositoryNotFound     = errors.New("repository not found")
	ErrEntityNotFound         = errors.New("entity not found")
	ErrMaxEntitiesReached     = errors.New("maximum number of entities reached")
	ErrMaxRepositoriesReached = errors.New("maximum number of repositories reached")
	ErrInvalidEntityType      = errors.New("invalid entity type")
	ErrInvalidRepositoryType  = errors.New("invalid repository type")
	ErrTransactionTimeout     = errors.New("transaction timeout")
	ErrValidationFailed       = errors.New("validation failed")
	ErrCommitFailed           = errors.New("commit failed")
	ErrRollbackFailed         = errors.New("rollback failed")
)

// UnitOfWorkConfig represents the configuration for the unit of work service
type UnitOfWorkConfig struct {
	Name                 string                 `json:"name"`
	Version              string                 `json:"version"`
	Description          string                 `json:"description"`
	MaxEntities          int                    `json:"max_entities"`
	MaxRepositories      int                    `json:"max_repositories"`
	TransactionTimeout   time.Duration          `json:"transaction_timeout"`
	CleanupInterval      time.Duration          `json:"cleanup_interval"`
	ValidationEnabled    bool                   `json:"validation_enabled"`
	CachingEnabled       bool                   `json:"caching_enabled"`
	MonitoringEnabled    bool                   `json:"monitoring_enabled"`
	AuditingEnabled      bool                   `json:"auditing_enabled"`
	SupportedEntityTypes []string               `json:"supported_entity_types"`
	DefaultEntityType    string                 `json:"default_entity_type"`
	ValidationRules      map[string]interface{} `json:"validation_rules"`
	Metadata             map[string]interface{} `json:"metadata"`
	Database             DatabaseConfig         `json:"database"`
	Cache                CacheConfig            `json:"cache"`
	MessageQueue         MessageQueueConfig     `json:"message_queue"`
	WebSocket            WebSocketConfig        `json:"websocket"`
	Security             SecurityConfig         `json:"security"`
	Monitoring           MonitoringConfig       `json:"monitoring"`
	Logging              LoggingConfig          `json:"logging"`
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

// UnitOfWorkServiceManager manages the unit of work service operations
type UnitOfWorkServiceManager struct {
	service *UnitOfWorkService
	config  *UnitOfWorkConfig
}

// NewUnitOfWorkServiceManager creates a new unit of work service manager
func NewUnitOfWorkServiceManager(config *UnitOfWorkConfig) *UnitOfWorkServiceManager {
	return &UnitOfWorkServiceManager{
		service: NewUnitOfWorkService(config),
		config:  config,
	}
}

// RegisterNew registers a new entity with validation
func (uowsm *UnitOfWorkServiceManager) RegisterNew(entity Entity) error {
	// Validate input
	if entity == nil {
		return errors.New("entity cannot be nil")
	}

	// Validate entity type
	if uowsm.config.ValidationEnabled {
		if !uowsm.isValidEntityType(entity.GetType()) {
			return ErrInvalidEntityType
		}
	}

	// Check entity limit
	if uowsm.service.GetEntityCount() >= uowsm.config.MaxEntities {
		return ErrMaxEntitiesReached
	}

	// Register entity
	return uowsm.service.RegisterNew(entity)
}

// RegisterDirty registers a dirty entity with validation
func (uowsm *UnitOfWorkServiceManager) RegisterDirty(entity Entity) error {
	// Validate input
	if entity == nil {
		return errors.New("entity cannot be nil")
	}

	// Validate entity type
	if uowsm.config.ValidationEnabled {
		if !uowsm.isValidEntityType(entity.GetType()) {
			return ErrInvalidEntityType
		}
	}

	// Check entity limit
	if uowsm.service.GetEntityCount() >= uowsm.config.MaxEntities {
		return ErrMaxEntitiesReached
	}

	// Register entity
	return uowsm.service.RegisterDirty(entity)
}

// RegisterDeleted registers a deleted entity with validation
func (uowsm *UnitOfWorkServiceManager) RegisterDeleted(entity Entity) error {
	// Validate input
	if entity == nil {
		return errors.New("entity cannot be nil")
	}

	// Validate entity type
	if uowsm.config.ValidationEnabled {
		if !uowsm.isValidEntityType(entity.GetType()) {
			return ErrInvalidEntityType
		}
	}

	// Check entity limit
	if uowsm.service.GetEntityCount() >= uowsm.config.MaxEntities {
		return ErrMaxEntitiesReached
	}

	// Register entity
	return uowsm.service.RegisterDeleted(entity)
}

// RegisterClean registers a clean entity with validation
func (uowsm *UnitOfWorkServiceManager) RegisterClean(entity Entity) error {
	// Validate input
	if entity == nil {
		return errors.New("entity cannot be nil")
	}

	// Validate entity type
	if uowsm.config.ValidationEnabled {
		if !uowsm.isValidEntityType(entity.GetType()) {
			return ErrInvalidEntityType
		}
	}

	// Check entity limit
	if uowsm.service.GetEntityCount() >= uowsm.config.MaxEntities {
		return ErrMaxEntitiesReached
	}

	// Register entity
	return uowsm.service.RegisterClean(entity)
}

// Commit commits the unit of work with timeout
func (uowsm *UnitOfWorkServiceManager) Commit() error {
	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), uowsm.config.TransactionTimeout)
	defer cancel()

	// Perform commit in goroutine with timeout
	done := make(chan error, 1)
	go func() {
		done <- uowsm.service.Commit()
	}()

	select {
	case err := <-done:
		if err != nil {
			return ErrCommitFailed
		}
		return nil
	case <-ctx.Done():
		return ErrTransactionTimeout
	}
}

// Rollback rolls back the unit of work with timeout
func (uowsm *UnitOfWorkServiceManager) Rollback() error {
	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), uowsm.config.TransactionTimeout)
	defer cancel()

	// Perform rollback in goroutine with timeout
	done := make(chan error, 1)
	go func() {
		done <- uowsm.service.Rollback()
	}()

	select {
	case err := <-done:
		if err != nil {
			return ErrRollbackFailed
		}
		return nil
	case <-ctx.Done():
		return ErrTransactionTimeout
	}
}

// GetRepository retrieves a repository with validation
func (uowsm *UnitOfWorkServiceManager) GetRepository(entityType string) (Repository, error) {
	// Validate input
	if entityType == "" {
		return nil, errors.New("entity type cannot be empty")
	}

	// Validate repository type
	if uowsm.config.ValidationEnabled {
		if !uowsm.isValidEntityType(entityType) {
			return nil, ErrInvalidRepositoryType
		}
	}

	// Get repository
	repository, err := uowsm.service.GetRepository(entityType)
	if err != nil {
		return nil, err
	}

	return repository, nil
}

// RegisterRepository registers a repository with validation
func (uowsm *UnitOfWorkServiceManager) RegisterRepository(entityType string, repository Repository) error {
	// Validate input
	if entityType == "" {
		return errors.New("entity type cannot be empty")
	}
	if repository == nil {
		return errors.New("repository cannot be nil")
	}

	// Validate repository type
	if uowsm.config.ValidationEnabled {
		if !uowsm.isValidEntityType(entityType) {
			return ErrInvalidRepositoryType
		}
	}

	// Check repository limit
	if uowsm.service.GetRepositoryCount() >= uowsm.config.MaxRepositories {
		return ErrMaxRepositoriesReached
	}

	// Register repository
	return uowsm.service.RegisterRepository(entityType, repository)
}

// GetEntities returns all entities with filtering
func (uowsm *UnitOfWorkServiceManager) GetEntities() map[string][]Entity {
	return uowsm.service.GetEntities()
}

// GetNewEntities returns new entities
func (uowsm *UnitOfWorkServiceManager) GetNewEntities() []Entity {
	return uowsm.service.GetNewEntities()
}

// GetDirtyEntities returns dirty entities
func (uowsm *UnitOfWorkServiceManager) GetDirtyEntities() []Entity {
	return uowsm.service.GetDirtyEntities()
}

// GetDeletedEntities returns deleted entities
func (uowsm *UnitOfWorkServiceManager) GetDeletedEntities() []Entity {
	return uowsm.service.GetDeletedEntities()
}

// GetCleanEntities returns clean entities
func (uowsm *UnitOfWorkServiceManager) GetCleanEntities() []Entity {
	return uowsm.service.GetCleanEntities()
}

// Clear clears all entities
func (uowsm *UnitOfWorkServiceManager) Clear() error {
	return uowsm.service.Clear()
}

// GetService returns the underlying unit of work service
func (uowsm *UnitOfWorkServiceManager) GetService() *UnitOfWorkService {
	return uowsm.service
}

// GetConfig returns the service configuration
func (uowsm *UnitOfWorkServiceManager) GetConfig() *UnitOfWorkConfig {
	return uowsm.config
}

// SetConfig sets the service configuration
func (uowsm *UnitOfWorkServiceManager) SetConfig(config *UnitOfWorkConfig) {
	uowsm.config = config
	uowsm.service.SetConfig(config)
}

// isValidEntityType checks if the entity type is valid
func (uowsm *UnitOfWorkServiceManager) isValidEntityType(entityType string) bool {
	for _, validType := range uowsm.config.SupportedEntityTypes {
		if validType == entityType {
			return true
		}
	}
	return false
}

// GetRepositoryCount returns the number of repositories
func (uowsm *UnitOfWorkServiceManager) GetRepositoryCount() int {
	return uowsm.service.GetRepositoryCount()
}

// GetEntityCount returns the total number of entities
func (uowsm *UnitOfWorkServiceManager) GetEntityCount() int {
	return uowsm.service.GetEntityCount()
}

// GetNewEntityCount returns the number of new entities
func (uowsm *UnitOfWorkServiceManager) GetNewEntityCount() int {
	return uowsm.service.GetNewEntityCount()
}

// GetDirtyEntityCount returns the number of dirty entities
func (uowsm *UnitOfWorkServiceManager) GetDirtyEntityCount() int {
	return uowsm.service.GetDirtyEntityCount()
}

// GetDeletedEntityCount returns the number of deleted entities
func (uowsm *UnitOfWorkServiceManager) GetDeletedEntityCount() int {
	return uowsm.service.GetDeletedEntityCount()
}

// GetCleanEntityCount returns the number of clean entities
func (uowsm *UnitOfWorkServiceManager) GetCleanEntityCount() int {
	return uowsm.service.GetCleanEntityCount()
}

// GetRepositoryTypes returns the list of repository types
func (uowsm *UnitOfWorkServiceManager) GetRepositoryTypes() []string {
	return uowsm.service.GetRepositoryTypes()
}

// GetEntityTypes returns the list of entity types
func (uowsm *UnitOfWorkServiceManager) GetEntityTypes() []string {
	return uowsm.service.GetEntityTypes()
}

// GetStats returns unit of work statistics
func (uowsm *UnitOfWorkServiceManager) GetStats() map[string]interface{} {
	return uowsm.service.GetStats()
}

// GetHealthStatus returns the health status of the unit of work
func (uowsm *UnitOfWorkServiceManager) GetHealthStatus() map[string]interface{} {
	return uowsm.service.GetHealthStatus()
}

// GetServiceInfo returns service information
func (uowsm *UnitOfWorkServiceManager) GetServiceInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":                 uowsm.config.Name,
		"version":              uowsm.config.Version,
		"description":          uowsm.config.Description,
		"repository_count":     uowsm.service.GetRepositoryCount(),
		"entity_count":         uowsm.service.GetEntityCount(),
		"new_entity_count":     uowsm.service.GetNewEntityCount(),
		"dirty_entity_count":   uowsm.service.GetDirtyEntityCount(),
		"deleted_entity_count": uowsm.service.GetDeletedEntityCount(),
		"clean_entity_count":   uowsm.service.GetCleanEntityCount(),
		"created_at":           uowsm.service.GetCreatedAt(),
		"updated_at":           uowsm.service.GetUpdatedAt(),
		"active":               uowsm.service.IsActive(),
		"metadata":             uowsm.service.GetMetadata(),
	}
}

// IsActive returns whether the unit of work is active
func (uowsm *UnitOfWorkServiceManager) IsActive() bool {
	return uowsm.service.IsActive()
}

// SetActive sets the active status
func (uowsm *UnitOfWorkServiceManager) SetActive(active bool) {
	uowsm.service.SetActive(active)
}

// GetID returns the service ID
func (uowsm *UnitOfWorkServiceManager) GetID() string {
	return uowsm.service.GetID()
}

// GetName returns the service name
func (uowsm *UnitOfWorkServiceManager) GetName() string {
	return uowsm.service.GetName()
}

// GetDescription returns the service description
func (uowsm *UnitOfWorkServiceManager) GetDescription() string {
	return uowsm.service.GetDescription()
}

// GetType returns the service type
func (uowsm *UnitOfWorkServiceManager) GetType() string {
	return "unit-of-work-service-manager"
}

// GetMetadata returns the service metadata
func (uowsm *UnitOfWorkServiceManager) GetMetadata() map[string]interface{} {
	return uowsm.service.GetMetadata()
}

// SetMetadata sets the service metadata
func (uowsm *UnitOfWorkServiceManager) SetMetadata(key string, value interface{}) {
	uowsm.service.SetMetadata(key, value)
}

// GetCreatedAt returns the service creation time
func (uowsm *UnitOfWorkServiceManager) GetCreatedAt() time.Time {
	return uowsm.service.GetCreatedAt()
}

// GetUpdatedAt returns the service last update time
func (uowsm *UnitOfWorkServiceManager) GetUpdatedAt() time.Time {
	return uowsm.service.GetUpdatedAt()
}
