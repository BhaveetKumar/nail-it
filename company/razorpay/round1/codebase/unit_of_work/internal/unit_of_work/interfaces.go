package unit_of_work

import (
	"context"
	"time"
)

// Entity represents a domain entity
type Entity interface {
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
	IsDirty() bool
	SetDirty(dirty bool)
	IsNew() bool
	SetNew(isNew bool)
	IsDeleted() bool
	SetDeleted(deleted bool)
}

// Repository represents a repository interface
type Repository interface {
	Create(ctx context.Context, entity Entity) error
	Update(ctx context.Context, entity Entity) error
	Delete(ctx context.Context, entityID string) error
	GetByID(ctx context.Context, entityID string) (Entity, error)
	List(ctx context.Context, filters map[string]interface{}) ([]Entity, error)
	Count(ctx context.Context, filters map[string]interface{}) (int64, error)
	Exists(ctx context.Context, entityID string) (bool, error)
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

// UnitOfWork represents the unit of work interface
type UnitOfWork interface {
	RegisterNew(entity Entity) error
	RegisterDirty(entity Entity) error
	RegisterDeleted(entity Entity) error
	RegisterClean(entity Entity) error
	Commit() error
	Rollback() error
	GetRepository(entityType string) (Repository, error)
	RegisterRepository(entityType string, repository Repository) error
	GetEntities() map[string][]Entity
	GetNewEntities() []Entity
	GetDirtyEntities() []Entity
	GetDeletedEntities() []Entity
	GetCleanEntities() []Entity
	Clear() error
	IsActive() bool
	SetActive(active bool)
	GetID() string
	GetName() string
	GetDescription() string
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
}

// TransactionManager manages transactions
type TransactionManager interface {
	Begin() (UnitOfWork, error)
	Commit(unitOfWork UnitOfWork) error
	Rollback(unitOfWork UnitOfWork) error
	GetActiveTransactions() []UnitOfWork
	GetTransactionCount() int
	GetMaxTransactions() int
	SetMaxTransactions(max int)
	GetTransactionTimeout() time.Duration
	SetTransactionTimeout(timeout time.Duration)
	Cleanup() error
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

// ConcreteEntity represents a concrete implementation of Entity
type ConcreteEntity struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Active      bool                   `json:"active"`
	Dirty       bool                   `json:"dirty"`
	New         bool                   `json:"new"`
	Deleted     bool                   `json:"deleted"`
}

// GetID returns the entity ID
func (ce *ConcreteEntity) GetID() string {
	return ce.ID
}

// GetType returns the entity type
func (ce *ConcreteEntity) GetType() string {
	return ce.Type
}

// GetName returns the entity name
func (ce *ConcreteEntity) GetName() string {
	return ce.Name
}

// GetDescription returns the entity description
func (ce *ConcreteEntity) GetDescription() string {
	return ce.Description
}

// GetMetadata returns the entity metadata
func (ce *ConcreteEntity) GetMetadata() map[string]interface{} {
	return ce.Metadata
}

// SetMetadata sets a metadata key-value pair
func (ce *ConcreteEntity) SetMetadata(key string, value interface{}) {
	if ce.Metadata == nil {
		ce.Metadata = make(map[string]interface{})
	}
	ce.Metadata[key] = value
	ce.UpdatedAt = time.Now()
	ce.Dirty = true
}

// GetCreatedAt returns the creation time
func (ce *ConcreteEntity) GetCreatedAt() time.Time {
	return ce.CreatedAt
}

// GetUpdatedAt returns the last update time
func (ce *ConcreteEntity) GetUpdatedAt() time.Time {
	return ce.UpdatedAt
}

// IsActive returns whether the entity is active
func (ce *ConcreteEntity) IsActive() bool {
	return ce.Active
}

// SetActive sets the active status
func (ce *ConcreteEntity) SetActive(active bool) {
	ce.Active = active
	ce.UpdatedAt = time.Now()
	ce.Dirty = true
}

// IsDirty returns whether the entity is dirty
func (ce *ConcreteEntity) IsDirty() bool {
	return ce.Dirty
}

// SetDirty sets the dirty status
func (ce *ConcreteEntity) SetDirty(dirty bool) {
	ce.Dirty = dirty
}

// IsNew returns whether the entity is new
func (ce *ConcreteEntity) IsNew() bool {
	return ce.New
}

// SetNew sets the new status
func (ce *ConcreteEntity) SetNew(isNew bool) {
	ce.New = isNew
}

// IsDeleted returns whether the entity is deleted
func (ce *ConcreteEntity) IsDeleted() bool {
	return ce.Deleted
}

// SetDeleted sets the deleted status
func (ce *ConcreteEntity) SetDeleted(deleted bool) {
	ce.Deleted = deleted
	ce.UpdatedAt = time.Now()
	ce.Dirty = true
}

// ConcreteRepository represents a concrete implementation of Repository
type ConcreteRepository struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Active      bool                   `json:"active"`
	Entities    map[string]Entity      `json:"entities"`
}

// Create creates a new entity
func (cr *ConcreteRepository) Create(ctx context.Context, entity Entity) error {
	if cr.Entities == nil {
		cr.Entities = make(map[string]Entity)
	}
	cr.Entities[entity.GetID()] = entity
	cr.UpdatedAt = time.Now()
	return nil
}

// Update updates an existing entity
func (cr *ConcreteRepository) Update(ctx context.Context, entity Entity) error {
	if cr.Entities == nil {
		cr.Entities = make(map[string]Entity)
	}
	cr.Entities[entity.GetID()] = entity
	cr.UpdatedAt = time.Now()
	return nil
}

// Delete deletes an entity
func (cr *ConcreteRepository) Delete(ctx context.Context, entityID string) error {
	if cr.Entities == nil {
		return nil
	}
	delete(cr.Entities, entityID)
	cr.UpdatedAt = time.Now()
	return nil
}

// GetByID retrieves an entity by ID
func (cr *ConcreteRepository) GetByID(ctx context.Context, entityID string) (Entity, error) {
	if cr.Entities == nil {
		return nil, nil
	}
	entity, exists := cr.Entities[entityID]
	if !exists {
		return nil, nil
	}
	return entity, nil
}

// List retrieves entities with filters
func (cr *ConcreteRepository) List(ctx context.Context, filters map[string]interface{}) ([]Entity, error) {
	if cr.Entities == nil {
		return []Entity{}, nil
	}
	entities := make([]Entity, 0, len(cr.Entities))
	for _, entity := range cr.Entities {
		entities = append(entities, entity)
	}
	return entities, nil
}

// Count returns the number of entities
func (cr *ConcreteRepository) Count(ctx context.Context, filters map[string]interface{}) (int64, error) {
	if cr.Entities == nil {
		return 0, nil
	}
	return int64(len(cr.Entities)), nil
}

// Exists checks if an entity exists
func (cr *ConcreteRepository) Exists(ctx context.Context, entityID string) (bool, error) {
	if cr.Entities == nil {
		return false, nil
	}
	_, exists := cr.Entities[entityID]
	return exists, nil
}

// GetType returns the repository type
func (cr *ConcreteRepository) GetType() string {
	return cr.Type
}

// GetName returns the repository name
func (cr *ConcreteRepository) GetName() string {
	return cr.Name
}

// GetDescription returns the repository description
func (cr *ConcreteRepository) GetDescription() string {
	return cr.Description
}

// GetMetadata returns the repository metadata
func (cr *ConcreteRepository) GetMetadata() map[string]interface{} {
	return cr.Metadata
}

// SetMetadata sets a metadata key-value pair
func (cr *ConcreteRepository) SetMetadata(key string, value interface{}) {
	if cr.Metadata == nil {
		cr.Metadata = make(map[string]interface{})
	}
	cr.Metadata[key] = value
	cr.UpdatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (cr *ConcreteRepository) GetCreatedAt() time.Time {
	return cr.CreatedAt
}

// GetUpdatedAt returns the last update time
func (cr *ConcreteRepository) GetUpdatedAt() time.Time {
	return cr.UpdatedAt
}

// IsActive returns whether the repository is active
func (cr *ConcreteRepository) IsActive() bool {
	return cr.Active
}

// SetActive sets the active status
func (cr *ConcreteRepository) SetActive(active bool) {
	cr.Active = active
	cr.UpdatedAt = time.Now()
}

// ConcreteUnitOfWork represents a concrete implementation of UnitOfWork
type ConcreteUnitOfWork struct {
	ID             string                 `json:"id"`
	Name           string                 `json:"name"`
	Description    string                 `json:"description"`
	Metadata       map[string]interface{} `json:"metadata"`
	CreatedAt      time.Time              `json:"created_at"`
	UpdatedAt      time.Time              `json:"updated_at"`
	Active         bool                   `json:"active"`
	Repositories   map[string]Repository  `json:"repositories"`
	NewEntities    []Entity               `json:"new_entities"`
	DirtyEntities  []Entity               `json:"dirty_entities"`
	DeletedEntities []Entity              `json:"deleted_entities"`
	CleanEntities  []Entity               `json:"clean_entities"`
}

// RegisterNew registers a new entity
func (cuow *ConcreteUnitOfWork) RegisterNew(entity Entity) error {
	entity.SetNew(true)
	cuow.NewEntities = append(cuow.NewEntities, entity)
	cuow.UpdatedAt = time.Now()
	return nil
}

// RegisterDirty registers a dirty entity
func (cuow *ConcreteUnitOfWork) RegisterDirty(entity Entity) error {
	entity.SetDirty(true)
	cuow.DirtyEntities = append(cuow.DirtyEntities, entity)
	cuow.UpdatedAt = time.Now()
	return nil
}

// RegisterDeleted registers a deleted entity
func (cuow *ConcreteUnitOfWork) RegisterDeleted(entity Entity) error {
	entity.SetDeleted(true)
	cuow.DeletedEntities = append(cuow.DeletedEntities, entity)
	cuow.UpdatedAt = time.Now()
	return nil
}

// RegisterClean registers a clean entity
func (cuow *ConcreteUnitOfWork) RegisterClean(entity Entity) error {
	entity.SetDirty(false)
	cuow.CleanEntities = append(cuow.CleanEntities, entity)
	cuow.UpdatedAt = time.Now()
	return nil
}

// Commit commits the unit of work
func (cuow *ConcreteUnitOfWork) Commit() error {
	// Process new entities
	for _, entity := range cuow.NewEntities {
		repository, err := cuow.GetRepository(entity.GetType())
		if err != nil {
			return err
		}
		if err := repository.Create(context.Background(), entity); err != nil {
			return err
		}
	}

	// Process dirty entities
	for _, entity := range cuow.DirtyEntities {
		repository, err := cuow.GetRepository(entity.GetType())
		if err != nil {
			return err
		}
		if err := repository.Update(context.Background(), entity); err != nil {
			return err
		}
	}

	// Process deleted entities
	for _, entity := range cuow.DeletedEntities {
		repository, err := cuow.GetRepository(entity.GetType())
		if err != nil {
			return err
		}
		if err := repository.Delete(context.Background(), entity.GetID()); err != nil {
			return err
		}
	}

	// Clear all entities
	cuow.Clear()
	cuow.UpdatedAt = time.Now()

	return nil
}

// Rollback rolls back the unit of work
func (cuow *ConcreteUnitOfWork) Rollback() error {
	// Clear all entities
	cuow.Clear()
	cuow.UpdatedAt = time.Now()
	return nil
}

// GetRepository retrieves a repository by entity type
func (cuow *ConcreteUnitOfWork) GetRepository(entityType string) (Repository, error) {
	repository, exists := cuow.Repositories[entityType]
	if !exists {
		return nil, nil
	}
	return repository, nil
}

// RegisterRepository registers a repository
func (cuow *ConcreteUnitOfWork) RegisterRepository(entityType string, repository Repository) error {
	if cuow.Repositories == nil {
		cuow.Repositories = make(map[string]Repository)
	}
	cuow.Repositories[entityType] = repository
	cuow.UpdatedAt = time.Now()
	return nil
}

// GetEntities returns all entities
func (cuow *ConcreteUnitOfWork) GetEntities() map[string][]Entity {
	entities := make(map[string][]Entity)
	entities["new"] = cuow.NewEntities
	entities["dirty"] = cuow.DirtyEntities
	entities["deleted"] = cuow.DeletedEntities
	entities["clean"] = cuow.CleanEntities
	return entities
}

// GetNewEntities returns new entities
func (cuow *ConcreteUnitOfWork) GetNewEntities() []Entity {
	return cuow.NewEntities
}

// GetDirtyEntities returns dirty entities
func (cuow *ConcreteUnitOfWork) GetDirtyEntities() []Entity {
	return cuow.DirtyEntities
}

// GetDeletedEntities returns deleted entities
func (cuow *ConcreteUnitOfWork) GetDeletedEntities() []Entity {
	return cuow.DeletedEntities
}

// GetCleanEntities returns clean entities
func (cuow *ConcreteUnitOfWork) GetCleanEntities() []Entity {
	return cuow.CleanEntities
}

// Clear clears all entities
func (cuow *ConcreteUnitOfWork) Clear() error {
	cuow.NewEntities = make([]Entity, 0)
	cuow.DirtyEntities = make([]Entity, 0)
	cuow.DeletedEntities = make([]Entity, 0)
	cuow.CleanEntities = make([]Entity, 0)
	cuow.UpdatedAt = time.Now()
	return nil
}

// IsActive returns whether the unit of work is active
func (cuow *ConcreteUnitOfWork) IsActive() bool {
	return cuow.Active
}

// SetActive sets the active status
func (cuow *ConcreteUnitOfWork) SetActive(active bool) {
	cuow.Active = active
	cuow.UpdatedAt = time.Now()
}

// GetID returns the unit of work ID
func (cuow *ConcreteUnitOfWork) GetID() string {
	return cuow.ID
}

// GetName returns the unit of work name
func (cuow *ConcreteUnitOfWork) GetName() string {
	return cuow.Name
}

// GetDescription returns the unit of work description
func (cuow *ConcreteUnitOfWork) GetDescription() string {
	return cuow.Description
}

// GetMetadata returns the unit of work metadata
func (cuow *ConcreteUnitOfWork) GetMetadata() map[string]interface{} {
	return cuow.Metadata
}

// SetMetadata sets a metadata key-value pair
func (cuow *ConcreteUnitOfWork) SetMetadata(key string, value interface{}) {
	if cuow.Metadata == nil {
		cuow.Metadata = make(map[string]interface{})
	}
	cuow.Metadata[key] = value
	cuow.UpdatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (cuow *ConcreteUnitOfWork) GetCreatedAt() time.Time {
	return cuow.CreatedAt
}

// GetUpdatedAt returns the last update time
func (cuow *ConcreteUnitOfWork) GetUpdatedAt() time.Time {
	return cuow.UpdatedAt
}

// ConcreteTransactionManager represents a concrete implementation of TransactionManager
type ConcreteTransactionManager struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Description       string                 `json:"description"`
	Metadata          map[string]interface{} `json:"metadata"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
	Active            bool                   `json:"active"`
	ActiveTransactions []UnitOfWork          `json:"active_transactions"`
	MaxTransactions   int                    `json:"max_transactions"`
	TransactionTimeout time.Duration         `json:"transaction_timeout"`
}

// Begin begins a new transaction
func (ctm *ConcreteTransactionManager) Begin() (UnitOfWork, error) {
	if len(ctm.ActiveTransactions) >= ctm.MaxTransactions {
		return nil, nil
	}

	unitOfWork := &ConcreteUnitOfWork{
		ID:             generateID(),
		Name:           "Unit of Work",
		Description:    "Auto-generated unit of work",
		Metadata:       make(map[string]interface{}),
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
		Active:         true,
		Repositories:   make(map[string]Repository),
		NewEntities:    make([]Entity, 0),
		DirtyEntities:  make([]Entity, 0),
		DeletedEntities: make([]Entity, 0),
		CleanEntities:  make([]Entity, 0),
	}

	ctm.ActiveTransactions = append(ctm.ActiveTransactions, unitOfWork)
	ctm.UpdatedAt = time.Now()

	return unitOfWork, nil
}

// Commit commits a transaction
func (ctm *ConcreteTransactionManager) Commit(unitOfWork UnitOfWork) error {
	if err := unitOfWork.Commit(); err != nil {
		return err
	}

	// Remove from active transactions
	for i, activeUOW := range ctm.ActiveTransactions {
		if activeUOW.GetID() == unitOfWork.GetID() {
			ctm.ActiveTransactions = append(ctm.ActiveTransactions[:i], ctm.ActiveTransactions[i+1:]...)
			break
		}
	}

	ctm.UpdatedAt = time.Now()
	return nil
}

// Rollback rolls back a transaction
func (ctm *ConcreteTransactionManager) Rollback(unitOfWork UnitOfWork) error {
	if err := unitOfWork.Rollback(); err != nil {
		return err
	}

	// Remove from active transactions
	for i, activeUOW := range ctm.ActiveTransactions {
		if activeUOW.GetID() == unitOfWork.GetID() {
			ctm.ActiveTransactions = append(ctm.ActiveTransactions[:i], ctm.ActiveTransactions[i+1:]...)
			break
		}
	}

	ctm.UpdatedAt = time.Now()
	return nil
}

// GetActiveTransactions returns active transactions
func (ctm *ConcreteTransactionManager) GetActiveTransactions() []UnitOfWork {
	return ctm.ActiveTransactions
}

// GetTransactionCount returns the number of active transactions
func (ctm *ConcreteTransactionManager) GetTransactionCount() int {
	return len(ctm.ActiveTransactions)
}

// GetMaxTransactions returns the maximum number of transactions
func (ctm *ConcreteTransactionManager) GetMaxTransactions() int {
	return ctm.MaxTransactions
}

// SetMaxTransactions sets the maximum number of transactions
func (ctm *ConcreteTransactionManager) SetMaxTransactions(max int) {
	ctm.MaxTransactions = max
	ctm.UpdatedAt = time.Now()
}

// GetTransactionTimeout returns the transaction timeout
func (ctm *ConcreteTransactionManager) GetTransactionTimeout() time.Duration {
	return ctm.TransactionTimeout
}

// SetTransactionTimeout sets the transaction timeout
func (ctm *ConcreteTransactionManager) SetTransactionTimeout(timeout time.Duration) {
	ctm.TransactionTimeout = timeout
	ctm.UpdatedAt = time.Now()
}

// Cleanup performs cleanup operations
func (ctm *ConcreteTransactionManager) Cleanup() error {
	ctm.ActiveTransactions = make([]UnitOfWork, 0)
	ctm.UpdatedAt = time.Now()
	return nil
}

// GetID returns the transaction manager ID
func (ctm *ConcreteTransactionManager) GetID() string {
	return ctm.ID
}

// GetName returns the transaction manager name
func (ctm *ConcreteTransactionManager) GetName() string {
	return ctm.Name
}

// GetDescription returns the transaction manager description
func (ctm *ConcreteTransactionManager) GetDescription() string {
	return ctm.Description
}

// GetMetadata returns the transaction manager metadata
func (ctm *ConcreteTransactionManager) GetMetadata() map[string]interface{} {
	return ctm.Metadata
}

// SetMetadata sets a metadata key-value pair
func (ctm *ConcreteTransactionManager) SetMetadata(key string, value interface{}) {
	if ctm.Metadata == nil {
		ctm.Metadata = make(map[string]interface{})
	}
	ctm.Metadata[key] = value
	ctm.UpdatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (ctm *ConcreteTransactionManager) GetCreatedAt() time.Time {
	return ctm.CreatedAt
}

// GetUpdatedAt returns the last update time
func (ctm *ConcreteTransactionManager) GetUpdatedAt() time.Time {
	return ctm.UpdatedAt
}

// IsActive returns whether the transaction manager is active
func (ctm *ConcreteTransactionManager) IsActive() bool {
	return ctm.Active
}

// SetActive sets the active status
func (ctm *ConcreteTransactionManager) SetActive(active bool) {
	ctm.Active = active
	ctm.UpdatedAt = time.Now()
}

// Utility function to generate unique IDs
func generateID() string {
	return time.Now().Format("20060102150405") + "-" + time.Now().Format("000000000")
}
