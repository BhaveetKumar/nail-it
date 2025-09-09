package repository

import (
	"context"
	"database/sql"
	"sync"

	"go.mongodb.org/mongo-driver/mongo"
	"repository-service/internal/config"
	"repository-service/internal/logger"
	"repository-service/internal/redis"
)

// RepositoryFactory creates repository instances
type RepositoryFactory struct {
	mysqlDB     *sql.DB
	mongoDB     *mongo.Database
	redisClient *redis.RedisManager
	config      *config.ConfigManager
	logger      *logger.Logger
	mutex       sync.RWMutex
}

var (
	repositoryFactory *RepositoryFactory
	factoryOnce       sync.Once
)

// GetRepositoryFactory returns the singleton instance of RepositoryFactory
func GetRepositoryFactory() *RepositoryFactory {
	factoryOnce.Do(func() {
		repositoryFactory = &RepositoryFactory{
			config: config.GetConfigManager(),
			logger: logger.GetLogger(),
		}
		repositoryFactory.initialize()
	})
	return repositoryFactory
}

// initialize initializes the repository factory
func (rf *RepositoryFactory) initialize() {
	rf.mutex.Lock()
	defer rf.mutex.Unlock()
	
	// Initialize database connections
	rf.mysqlDB = database.GetMySQLManager().GetDB()
	rf.mongoDB = database.GetMongoManager().GetDatabase()
	rf.redisClient = redis.GetRedisManager()
	
	rf.logger.Info("Repository factory initialized")
}

// CreateUserRepository creates a user repository
func (rf *RepositoryFactory) CreateUserRepository() Repository[*User] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMySQLRepository[*User](rf.mysqlDB, "users")
}

// CreatePaymentRepository creates a payment repository
func (rf *RepositoryFactory) CreatePaymentRepository() Repository[*Payment] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMySQLRepository[*Payment](rf.mysqlDB, "payments")
}

// CreateOrderRepository creates an order repository
func (rf *RepositoryFactory) CreateOrderRepository() Repository[*Order] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMySQLRepository[*Order](rf.mysqlDB, "orders")
}

// CreateProductRepository creates a product repository
func (rf *RepositoryFactory) CreateProductRepository() Repository[*Product] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMySQLRepository[*Product](rf.mysqlDB, "products")
}

// CreateMongoUserRepository creates a MongoDB user repository
func (rf *RepositoryFactory) CreateMongoUserRepository() Repository[*User] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*User](rf.mongoDB.Collection("users"))
}

// CreateMongoPaymentRepository creates a MongoDB payment repository
func (rf *RepositoryFactory) CreateMongoPaymentRepository() Repository[*Payment] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Payment](rf.mongoDB.Collection("payments"))
}

// CreateMongoOrderRepository creates a MongoDB order repository
func (rf *RepositoryFactory) CreateMongoOrderRepository() Repository[*Order] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Order](rf.mongoDB.Collection("orders"))
}

// CreateMongoProductRepository creates a MongoDB product repository
func (rf *RepositoryFactory) CreateMongoProductRepository() Repository[*Product] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Product](rf.mongoDB.Collection("products"))
}

// CreateCachedUserRepository creates a cached user repository
func (rf *RepositoryFactory) CreateCachedUserRepository() CachedRepository[*User] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	baseRepo := NewMySQLRepository[*User](rf.mysqlDB, "users")
	ttl := rf.config.GetCacheConfig().TTL
	
	return NewCachedRepository[*User](baseRepo, rf.redisClient, ttl)
}

// CreateCachedPaymentRepository creates a cached payment repository
func (rf *RepositoryFactory) CreateCachedPaymentRepository() CachedRepository[*Payment] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	baseRepo := NewMySQLRepository[*Payment](rf.mysqlDB, "payments")
	ttl := rf.config.GetCacheConfig().TTL
	
	return NewCachedRepository[*Payment](baseRepo, rf.redisClient, ttl)
}

// CreateCachedOrderRepository creates a cached order repository
func (rf *RepositoryFactory) CreateCachedOrderRepository() CachedRepository[*Order] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	baseRepo := NewMySQLRepository[*Order](rf.mysqlDB, "orders")
	ttl := rf.config.GetCacheConfig().TTL
	
	return NewCachedRepository[*Order](baseRepo, rf.redisClient, ttl)
}

// CreateCachedProductRepository creates a cached product repository
func (rf *RepositoryFactory) CreateCachedProductRepository() CachedRepository[*Product] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	baseRepo := NewMySQLRepository[*Product](rf.mysqlDB, "products")
	ttl := rf.config.GetCacheConfig().TTL
	
	return NewCachedRepository[*Product](baseRepo, rf.redisClient, ttl)
}

// CreateUnitOfWork creates a new unit of work
func (rf *RepositoryFactory) CreateUnitOfWork() UnitOfWork {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewUnitOfWork(rf.mysqlDB, rf.mongoDB, rf.redisClient)
}

// CreateTransactionalUserRepository creates a transactional user repository
func (rf *RepositoryFactory) CreateTransactionalUserRepository(uow *UnitOfWorkImpl) TransactionalRepository[*User] {
	baseRepo := NewMySQLRepository[*User](rf.mysqlDB, "users")
	return NewTransactionalRepository[*User](baseRepo, uow)
}

// CreateTransactionalPaymentRepository creates a transactional payment repository
func (rf *RepositoryFactory) CreateTransactionalPaymentRepository(uow *UnitOfWorkImpl) TransactionalRepository[*Payment] {
	baseRepo := NewMySQLRepository[*Payment](rf.mysqlDB, "payments")
	return NewTransactionalRepository[*Payment](baseRepo, uow)
}

// CreateTransactionalOrderRepository creates a transactional order repository
func (rf *RepositoryFactory) CreateTransactionalOrderRepository(uow *UnitOfWorkImpl) TransactionalRepository[*Order] {
	baseRepo := NewMySQLRepository[*Order](rf.mysqlDB, "orders")
	return NewTransactionalRepository[*Order](baseRepo, uow)
}

// CreateTransactionalProductRepository creates a transactional product repository
func (rf *RepositoryFactory) CreateTransactionalProductRepository(uow *UnitOfWorkImpl) TransactionalRepository[*Product] {
	baseRepo := NewMySQLRepository[*Product](rf.mysqlDB, "products")
	return NewTransactionalRepository[*Product](baseRepo, uow)
}

// CreateSpecificationUserRepository creates a specification-based user repository
func (rf *RepositoryFactory) CreateSpecificationUserRepository() SpecificationRepository[*User] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*User](rf.mongoDB.Collection("users"))
}

// CreateSpecificationPaymentRepository creates a specification-based payment repository
func (rf *RepositoryFactory) CreateSpecificationPaymentRepository() SpecificationRepository[*Payment] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Payment](rf.mongoDB.Collection("payments"))
}

// CreateSpecificationOrderRepository creates a specification-based order repository
func (rf *RepositoryFactory) CreateSpecificationOrderRepository() SpecificationRepository[*Order] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Order](rf.mongoDB.Collection("orders"))
}

// CreateSpecificationProductRepository creates a specification-based product repository
func (rf *RepositoryFactory) CreateSpecificationProductRepository() SpecificationRepository[*Product] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Product](rf.mongoDB.Collection("products"))
}

// CreateAuditUserRepository creates an audit-enabled user repository
func (rf *RepositoryFactory) CreateAuditUserRepository() AuditRepository[*User] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*User](rf.mongoDB.Collection("users"))
}

// CreateAuditPaymentRepository creates an audit-enabled payment repository
func (rf *RepositoryFactory) CreateAuditPaymentRepository() AuditRepository[*Payment] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Payment](rf.mongoDB.Collection("payments"))
}

// CreateAuditOrderRepository creates an audit-enabled order repository
func (rf *RepositoryFactory) CreateAuditOrderRepository() AuditRepository[*Order] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Order](rf.mongoDB.Collection("orders"))
}

// CreateAuditProductRepository creates an audit-enabled product repository
func (rf *RepositoryFactory) CreateAuditProductRepository() AuditRepository[*Product] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Product](rf.mongoDB.Collection("products"))
}

// CreateSoftDeleteUserRepository creates a soft-delete enabled user repository
func (rf *RepositoryFactory) CreateSoftDeleteUserRepository() SoftDeleteRepository[*User] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*User](rf.mongoDB.Collection("users"))
}

// CreateSoftDeletePaymentRepository creates a soft-delete enabled payment repository
func (rf *RepositoryFactory) CreateSoftDeletePaymentRepository() SoftDeleteRepository[*Payment] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Payment](rf.mongoDB.Collection("payments"))
}

// CreateSoftDeleteOrderRepository creates a soft-delete enabled order repository
func (rf *RepositoryFactory) CreateSoftDeleteOrderRepository() SoftDeleteRepository[*Order] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Order](rf.mongoDB.Collection("orders"))
}

// CreateSoftDeleteProductRepository creates a soft-delete enabled product repository
func (rf *RepositoryFactory) CreateSoftDeleteProductRepository() SoftDeleteRepository[*Product] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Product](rf.mongoDB.Collection("products"))
}

// CreatePaginatedUserRepository creates a paginated user repository
func (rf *RepositoryFactory) CreatePaginatedUserRepository() PaginatedRepository[*User] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*User](rf.mongoDB.Collection("users"))
}

// CreatePaginatedPaymentRepository creates a paginated payment repository
func (rf *RepositoryFactory) CreatePaginatedPaymentRepository() PaginatedRepository[*Payment] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Payment](rf.mongoDB.Collection("payments"))
}

// CreatePaginatedOrderRepository creates a paginated order repository
func (rf *RepositoryFactory) CreatePaginatedOrderRepository() PaginatedRepository[*Order] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Order](rf.mongoDB.Collection("orders"))
}

// CreatePaginatedProductRepository creates a paginated product repository
func (rf *RepositoryFactory) CreatePaginatedProductRepository() PaginatedRepository[*Product] {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	return NewMongoRepository[*Product](rf.mongoDB.Collection("products"))
}

// RepositoryType represents the type of repository to create
type RepositoryType string

const (
	RepositoryTypeMySQL      RepositoryType = "mysql"
	RepositoryTypeMongoDB    RepositoryType = "mongodb"
	RepositoryTypeCached     RepositoryType = "cached"
	RepositoryTypeTransactional RepositoryType = "transactional"
	RepositoryTypeSpecification RepositoryType = "specification"
	RepositoryTypeAudit      RepositoryType = "audit"
	RepositoryTypeSoftDelete RepositoryType = "soft_delete"
	RepositoryTypePaginated  RepositoryType = "paginated"
)

// CreateRepository creates a repository based on type
func (rf *RepositoryFactory) CreateRepository[T Entity](entityType string, repoType RepositoryType) (Repository[T], error) {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	switch repoType {
	case RepositoryTypeMySQL:
		return rf.createMySQLRepository[T](entityType)
	case RepositoryTypeMongoDB:
		return rf.createMongoRepository[T](entityType)
	case RepositoryTypeCached:
		return rf.createCachedRepository[T](entityType)
	case RepositoryTypeSpecification:
		return rf.createSpecificationRepository[T](entityType)
	case RepositoryTypeAudit:
		return rf.createAuditRepository[T](entityType)
	case RepositoryTypeSoftDelete:
		return rf.createSoftDeleteRepository[T](entityType)
	case RepositoryTypePaginated:
		return rf.createPaginatedRepository[T](entityType)
	default:
		return nil, &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Unsupported repository type",
		}
	}
}

// createMySQLRepository creates a MySQL repository
func (rf *RepositoryFactory) createMySQLRepository[T Entity](entityType string) (Repository[T], error) {
	switch entityType {
	case "user":
		return any(rf.CreateUserRepository()).(Repository[T]), nil
	case "payment":
		return any(rf.CreatePaymentRepository()).(Repository[T]), nil
	case "order":
		return any(rf.CreateOrderRepository()).(Repository[T]), nil
	case "product":
		return any(rf.CreateProductRepository()).(Repository[T]), nil
	default:
		return nil, &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Unsupported entity type",
		}
	}
}

// createMongoRepository creates a MongoDB repository
func (rf *RepositoryFactory) createMongoRepository[T Entity](entityType string) (Repository[T], error) {
	switch entityType {
	case "user":
		return any(rf.CreateMongoUserRepository()).(Repository[T]), nil
	case "payment":
		return any(rf.CreateMongoPaymentRepository()).(Repository[T]), nil
	case "order":
		return any(rf.CreateMongoOrderRepository()).(Repository[T]), nil
	case "product":
		return any(rf.CreateMongoProductRepository()).(Repository[T]), nil
	default:
		return nil, &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Unsupported entity type",
		}
	}
}

// createCachedRepository creates a cached repository
func (rf *RepositoryFactory) createCachedRepository[T Entity](entityType string) (Repository[T], error) {
	switch entityType {
	case "user":
		return any(rf.CreateCachedUserRepository()).(Repository[T]), nil
	case "payment":
		return any(rf.CreateCachedPaymentRepository()).(Repository[T]), nil
	case "order":
		return any(rf.CreateCachedOrderRepository()).(Repository[T]), nil
	case "product":
		return any(rf.CreateCachedProductRepository()).(Repository[T]), nil
	default:
		return nil, &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Unsupported entity type",
		}
	}
}

// createSpecificationRepository creates a specification repository
func (rf *RepositoryFactory) createSpecificationRepository[T Entity](entityType string) (Repository[T], error) {
	switch entityType {
	case "user":
		return any(rf.CreateSpecificationUserRepository()).(Repository[T]), nil
	case "payment":
		return any(rf.CreateSpecificationPaymentRepository()).(Repository[T]), nil
	case "order":
		return any(rf.CreateSpecificationOrderRepository()).(Repository[T]), nil
	case "product":
		return any(rf.CreateSpecificationProductRepository()).(Repository[T]), nil
	default:
		return nil, &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Unsupported entity type",
		}
	}
}

// createAuditRepository creates an audit repository
func (rf *RepositoryFactory) createAuditRepository[T Entity](entityType string) (Repository[T], error) {
	switch entityType {
	case "user":
		return any(rf.CreateAuditUserRepository()).(Repository[T]), nil
	case "payment":
		return any(rf.CreateAuditPaymentRepository()).(Repository[T]), nil
	case "order":
		return any(rf.CreateAuditOrderRepository()).(Repository[T]), nil
	case "product":
		return any(rf.CreateAuditProductRepository()).(Repository[T]), nil
	default:
		return nil, &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Unsupported entity type",
		}
	}
}

// createSoftDeleteRepository creates a soft delete repository
func (rf *RepositoryFactory) createSoftDeleteRepository[T Entity](entityType string) (Repository[T], error) {
	switch entityType {
	case "user":
		return any(rf.CreateSoftDeleteUserRepository()).(Repository[T]), nil
	case "payment":
		return any(rf.CreateSoftDeletePaymentRepository()).(Repository[T]), nil
	case "order":
		return any(rf.CreateSoftDeleteOrderRepository()).(Repository[T]), nil
	case "product":
		return any(rf.CreateSoftDeleteProductRepository()).(Repository[T]), nil
	default:
		return nil, &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Unsupported entity type",
		}
	}
}

// createPaginatedRepository creates a paginated repository
func (rf *RepositoryFactory) createPaginatedRepository[T Entity](entityType string) (Repository[T], error) {
	switch entityType {
	case "user":
		return any(rf.CreatePaginatedUserRepository()).(Repository[T]), nil
	case "payment":
		return any(rf.CreatePaginatedPaymentRepository()).(Repository[T]), nil
	case "order":
		return any(rf.CreatePaginatedOrderRepository()).(Repository[T]), nil
	case "product":
		return any(rf.CreatePaginatedProductRepository()).(Repository[T]), nil
	default:
		return nil, &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Unsupported entity type",
		}
	}
}

// GetAvailableRepositoryTypes returns the available repository types
func (rf *RepositoryFactory) GetAvailableRepositoryTypes() []RepositoryType {
	return []RepositoryType{
		RepositoryTypeMySQL,
		RepositoryTypeMongoDB,
		RepositoryTypeCached,
		RepositoryTypeTransactional,
		RepositoryTypeSpecification,
		RepositoryTypeAudit,
		RepositoryTypeSoftDelete,
		RepositoryTypePaginated,
	}
}

// GetAvailableEntityTypes returns the available entity types
func (rf *RepositoryFactory) GetAvailableEntityTypes() []string {
	return []string{"user", "payment", "order", "product"}
}

// HealthCheck performs a health check on all repositories
func (rf *RepositoryFactory) HealthCheck(ctx context.Context) error {
	rf.mutex.RLock()
	defer rf.mutex.RUnlock()
	
	// Check MySQL connection
	if err := rf.mysqlDB.PingContext(ctx); err != nil {
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "MySQL connection failed",
			Err:     err,
		}
	}
	
	// Check MongoDB connection
	if err := rf.mongoDB.Client().Ping(ctx, nil); err != nil {
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "MongoDB connection failed",
			Err:     err,
		}
	}
	
	// Check Redis connection
	if err := rf.redisClient.Ping(ctx); err != nil {
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Redis connection failed",
			Err:     err,
		}
	}
	
	return nil
}
