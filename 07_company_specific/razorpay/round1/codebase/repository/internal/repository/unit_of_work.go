package repository

import (
	"context"
	"database/sql"
	"sync"
	"time"

	"repository-service/internal/logger"
	"repository-service/internal/redis"

	"go.mongodb.org/mongo-driver/mongo"
)

// UnitOfWorkImpl implements UnitOfWork interface
type UnitOfWorkImpl struct {
	mysqlDB     *sql.DB
	mongoDB     *mongo.Database
	redisClient *redis.RedisManager
	tx          *sql.Tx
	session     mongo.Session
	userRepo    Repository[*User]
	paymentRepo Repository[*Payment]
	orderRepo   Repository[*Order]
	productRepo Repository[*Product]
	active      bool
	mutex       sync.RWMutex
	logger      *logger.Logger
}

// NewUnitOfWork creates a new unit of work
func NewUnitOfWork(mysqlDB *sql.DB, mongoDB *mongo.Database, redisClient *redis.RedisManager) *UnitOfWorkImpl {
	return &UnitOfWorkImpl{
		mysqlDB:     mysqlDB,
		mongoDB:     mongoDB,
		redisClient: redisClient,
		logger:      logger.GetLogger(),
	}
}

// Begin starts a new transaction
func (uow *UnitOfWorkImpl) Begin() error {
	uow.mutex.Lock()
	defer uow.mutex.Unlock()

	if uow.active {
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "Transaction already active",
		}
	}

	// Begin MySQL transaction
	tx, err := uow.mysqlDB.Begin()
	if err != nil {
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "Failed to begin MySQL transaction",
			Err:     err,
		}
	}

	// Begin MongoDB session
	session, err := uow.mongoDB.Client().StartSession()
	if err != nil {
		tx.Rollback()
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "Failed to begin MongoDB session",
			Err:     err,
		}
	}

	// Start MongoDB transaction
	err = session.StartTransaction()
	if err != nil {
		tx.Rollback()
		session.EndSession(context.Background())
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "Failed to start MongoDB transaction",
			Err:     err,
		}
	}

	uow.tx = tx
	uow.session = session
	uow.active = true

	// Create repositories with transaction context
	uow.userRepo = NewMySQLRepository[*User](uow.tx, "users")
	uow.paymentRepo = NewMySQLRepository[*Payment](uow.tx, "payments")
	uow.orderRepo = NewMySQLRepository[*Order](uow.tx, "orders")
	uow.productRepo = NewMySQLRepository[*Product](uow.tx, "products")

	uow.logger.Info("Unit of work transaction started")
	return nil
}

// Commit commits the transaction
func (uow *UnitOfWorkImpl) Commit() error {
	uow.mutex.Lock()
	defer uow.mutex.Unlock()

	if !uow.active {
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "No active transaction to commit",
		}
	}

	// Commit MongoDB transaction
	err := uow.session.CommitTransaction(context.Background())
	if err != nil {
		uow.logger.Error("Failed to commit MongoDB transaction", "error", err)
		// Still try to commit MySQL transaction
		uow.tx.Commit()
		uow.session.EndSession(context.Background())
		uow.active = false
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "Failed to commit MongoDB transaction",
			Err:     err,
		}
	}

	// Commit MySQL transaction
	err = uow.tx.Commit()
	if err != nil {
		uow.logger.Error("Failed to commit MySQL transaction", "error", err)
		uow.session.EndSession(context.Background())
		uow.active = false
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "Failed to commit MySQL transaction",
			Err:     err,
		}
	}

	// End MongoDB session
	uow.session.EndSession(context.Background())
	uow.active = false

	uow.logger.Info("Unit of work transaction committed successfully")
	return nil
}

// Rollback rolls back the transaction
func (uow *UnitOfWorkImpl) Rollback() error {
	uow.mutex.Lock()
	defer uow.mutex.Unlock()

	if !uow.active {
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "No active transaction to rollback",
		}
	}

	// Rollback MongoDB transaction
	err := uow.session.AbortTransaction(context.Background())
	if err != nil {
		uow.logger.Error("Failed to rollback MongoDB transaction", "error", err)
	}

	// Rollback MySQL transaction
	err = uow.tx.Rollback()
	if err != nil {
		uow.logger.Error("Failed to rollback MySQL transaction", "error", err)
	}

	// End MongoDB session
	uow.session.EndSession(context.Background())
	uow.active = false

	uow.logger.Info("Unit of work transaction rolled back")
	return nil
}

// IsActive returns whether the transaction is active
func (uow *UnitOfWorkImpl) IsActive() bool {
	uow.mutex.RLock()
	defer uow.mutex.RUnlock()
	return uow.active
}

// GetUserRepository returns the user repository
func (uow *UnitOfWorkImpl) GetUserRepository() Repository[*User] {
	uow.mutex.RLock()
	defer uow.mutex.RUnlock()
	return uow.userRepo
}

// GetPaymentRepository returns the payment repository
func (uow *UnitOfWorkImpl) GetPaymentRepository() Repository[*Payment] {
	uow.mutex.RLock()
	defer uow.mutex.RUnlock()
	return uow.paymentRepo
}

// GetOrderRepository returns the order repository
func (uow *UnitOfWorkImpl) GetOrderRepository() Repository[*Order] {
	uow.mutex.RLock()
	defer uow.mutex.RUnlock()
	return uow.orderRepo
}

// GetProductRepository returns the product repository
func (uow *UnitOfWorkImpl) GetProductRepository() Repository[*Product] {
	uow.mutex.RLock()
	defer uow.mutex.RUnlock()
	return uow.productRepo
}

// TransactionalRepository wraps a repository with transaction support
type TransactionalRepository[T Entity] struct {
	repository Repository[T]
	uow        *UnitOfWorkImpl
}

// NewTransactionalRepository creates a new transactional repository
func NewTransactionalRepository[T Entity](repository Repository[T], uow *UnitOfWorkImpl) *TransactionalRepository[T] {
	return &TransactionalRepository[T]{
		repository: repository,
		uow:        uow,
	}
}

// Create creates a new entity within a transaction
func (tr *TransactionalRepository[T]) Create(ctx context.Context, entity T) error {
	if !tr.uow.IsActive() {
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "No active transaction",
		}
	}
	return tr.repository.Create(ctx, entity)
}

// GetByID retrieves an entity by ID
func (tr *TransactionalRepository[T]) GetByID(ctx context.Context, id string) (T, error) {
	return tr.repository.GetByID(ctx, id)
}

// Update updates an entity within a transaction
func (tr *TransactionalRepository[T]) Update(ctx context.Context, entity T) error {
	if !tr.uow.IsActive() {
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "No active transaction",
		}
	}
	return tr.repository.Update(ctx, entity)
}

// Delete deletes an entity within a transaction
func (tr *TransactionalRepository[T]) Delete(ctx context.Context, id string) error {
	if !tr.uow.IsActive() {
		return &RepositoryError{
			Code:    ErrCodeTransaction,
			Message: "No active transaction",
		}
	}
	return tr.repository.Delete(ctx, id)
}

// GetAll retrieves all entities
func (tr *TransactionalRepository[T]) GetAll(ctx context.Context, limit, offset int) ([]T, error) {
	return tr.repository.GetAll(ctx, limit, offset)
}

// Count returns the total number of entities
func (tr *TransactionalRepository[T]) Count(ctx context.Context) (int64, error) {
	return tr.repository.Count(ctx)
}

// Exists checks if an entity exists by ID
func (tr *TransactionalRepository[T]) Exists(ctx context.Context, id string) (bool, error) {
	return tr.repository.Exists(ctx, id)
}

// FindBy finds entities by field and value
func (tr *TransactionalRepository[T]) FindBy(ctx context.Context, field string, value interface{}) ([]T, error) {
	return tr.repository.FindBy(ctx, field, value)
}

// FindByMultiple finds entities by multiple filters
func (tr *TransactionalRepository[T]) FindByMultiple(ctx context.Context, filters map[string]interface{}) ([]T, error) {
	return tr.repository.FindByMultiple(ctx, filters)
}

// BeginTransaction starts a new transaction
func (tr *TransactionalRepository[T]) BeginTransaction(ctx context.Context) (Transaction, error) {
	return tr.uow, nil
}

// WithTransaction executes a function within a transaction
func (tr *TransactionalRepository[T]) WithTransaction(ctx context.Context, fn func(Transaction) error) error {
	err := tr.uow.Begin()
	if err != nil {
		return err
	}

	defer func() {
		if r := recover(); r != nil {
			tr.uow.Rollback()
			panic(r)
		}
	}()

	err = fn(tr.uow)
	if err != nil {
		tr.uow.Rollback()
		return err
	}

	return tr.uow.Commit()
}

// TransactionManager manages transactions across multiple repositories
type TransactionManager struct {
	uow *UnitOfWorkImpl
}

// NewTransactionManager creates a new transaction manager
func NewTransactionManager(uow *UnitOfWorkImpl) *TransactionManager {
	return &TransactionManager{
		uow: uow,
	}
}

// ExecuteInTransaction executes a function within a transaction
func (tm *TransactionManager) ExecuteInTransaction(ctx context.Context, fn func(*UnitOfWorkImpl) error) error {
	err := tm.uow.Begin()
	if err != nil {
		return err
	}

	defer func() {
		if r := recover(); r != nil {
			tm.uow.Rollback()
			panic(r)
		}
	}()

	err = fn(tm.uow)
	if err != nil {
		tm.uow.Rollback()
		return err
	}

	return tm.uow.Commit()
}

// ExecuteInTransactionWithRetry executes a function within a transaction with retry logic
func (tm *TransactionManager) ExecuteInTransactionWithRetry(ctx context.Context, fn func(*UnitOfWorkImpl) error, maxRetries int) error {
	var lastErr error

	for i := 0; i <= maxRetries; i++ {
		err := tm.ExecuteInTransaction(ctx, fn)
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if error is retryable
		if !isRetryableError(err) {
			break
		}

		if i < maxRetries {
			// Wait before retry
			time.Sleep(time.Duration(i+1) * time.Second)
		}
	}

	return lastErr
}

// isRetryableError checks if an error is retryable
func isRetryableError(err error) bool {
	// Check for specific error types that are retryable
	// This is a simplified implementation
	return false
}

// TransactionContext provides transaction context
type TransactionContext struct {
	ctx context.Context
	uow *UnitOfWorkImpl
}

// NewTransactionContext creates a new transaction context
func NewTransactionContext(ctx context.Context, uow *UnitOfWorkImpl) *TransactionContext {
	return &TransactionContext{
		ctx: ctx,
		uow: uow,
	}
}

// GetContext returns the context
func (tc *TransactionContext) GetContext() context.Context {
	return tc.ctx
}

// GetUnitOfWork returns the unit of work
func (tc *TransactionContext) GetUnitOfWork() *UnitOfWorkImpl {
	return tc.uow
}

// Execute executes a function within the transaction context
func (tc *TransactionContext) Execute(fn func(*TransactionContext) error) error {
	return fn(tc)
}
