# Repository Pattern

Comprehensive guide to the Repository pattern for Razorpay interviews.

## 🎯 Repository Pattern Overview

The Repository pattern is a design pattern that abstracts data access logic and provides a more object-oriented view of the persistence layer. It acts as an in-memory collection of domain objects.

### Key Benefits
- **Separation of Concerns**: Separates business logic from data access logic
- **Testability**: Makes unit testing easier by allowing mock implementations
- **Flexibility**: Allows switching between different data sources
- **Maintainability**: Centralizes data access logic

## 🚀 Implementation Examples

### Basic Repository Interface
```go
// Repository Interface
type UserRepository interface {
    Create(user *User) error
    GetByID(id string) (*User, error)
    GetByEmail(email string) (*User, error)
    Update(user *User) error
    Delete(id string) error
    List(limit, offset int) ([]*User, error)
    Count() (int, error)
}

// User Entity
type User struct {
    ID        string    `json:"id" db:"id"`
    Email     string    `json:"email" db:"email"`
    Name      string    `json:"name" db:"name"`
    CreatedAt time.Time `json:"created_at" db:"created_at"`
    UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}
```

### Database Repository Implementation
```go
// Database Repository Implementation
type DatabaseUserRepository struct {
    db *sql.DB
}

func NewDatabaseUserRepository(db *sql.DB) *DatabaseUserRepository {
    return &DatabaseUserRepository{db: db}
}

func (r *DatabaseUserRepository) Create(user *User) error {
    query := `
        INSERT INTO users (id, email, name, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5)
    `
    
    _, err := r.db.Exec(query, user.ID, user.Email, user.Name, user.CreatedAt, user.UpdatedAt)
    return err
}

func (r *DatabaseUserRepository) GetByID(id string) (*User, error) {
    query := `
        SELECT id, email, name, created_at, updated_at
        FROM users
        WHERE id = $1
    `
    
    var user User
    err := r.db.QueryRow(query, id).Scan(
        &user.ID, &user.Email, &user.Name, &user.CreatedAt, &user.UpdatedAt,
    )
    
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, ErrUserNotFound
        }
        return nil, err
    }
    
    return &user, nil
}

func (r *DatabaseUserRepository) GetByEmail(email string) (*User, error) {
    query := `
        SELECT id, email, name, created_at, updated_at
        FROM users
        WHERE email = $1
    `
    
    var user User
    err := r.db.QueryRow(query, email).Scan(
        &user.ID, &user.Email, &user.Name, &user.CreatedAt, &user.UpdatedAt,
    )
    
    if err != nil {
        if err == sql.ErrNoRows {
            return nil, ErrUserNotFound
        }
        return nil, err
    }
    
    return &user, nil
}

func (r *DatabaseUserRepository) Update(user *User) error {
    query := `
        UPDATE users
        SET email = $2, name = $3, updated_at = $4
        WHERE id = $1
    `
    
    result, err := r.db.Exec(query, user.ID, user.Email, user.Name, user.UpdatedAt)
    if err != nil {
        return err
    }
    
    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return err
    }
    
    if rowsAffected == 0 {
        return ErrUserNotFound
    }
    
    return nil
}

func (r *DatabaseUserRepository) Delete(id string) error {
    query := `DELETE FROM users WHERE id = $1`
    
    result, err := r.db.Exec(query, id)
    if err != nil {
        return err
    }
    
    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return err
    }
    
    if rowsAffected == 0 {
        return ErrUserNotFound
    }
    
    return nil
}

func (r *DatabaseUserRepository) List(limit, offset int) ([]*User, error) {
    query := `
        SELECT id, email, name, created_at, updated_at
        FROM users
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
    `
    
    rows, err := r.db.Query(query, limit, offset)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    var users []*User
    for rows.Next() {
        var user User
        err := rows.Scan(
            &user.ID, &user.Email, &user.Name, &user.CreatedAt, &user.UpdatedAt,
        )
        if err != nil {
            return nil, err
        }
        users = append(users, &user)
    }
    
    return users, nil
}

func (r *DatabaseUserRepository) Count() (int, error) {
    query := `SELECT COUNT(*) FROM users`
    
    var count int
    err := r.db.QueryRow(query).Scan(&count)
    return count, err
}
```

### In-Memory Repository Implementation
```go
// In-Memory Repository Implementation (for testing)
type InMemoryUserRepository struct {
    users map[string]*User
    mutex sync.RWMutex
}

func NewInMemoryUserRepository() *InMemoryUserRepository {
    return &InMemoryUserRepository{
        users: make(map[string]*User),
    }
}

func (r *InMemoryUserRepository) Create(user *User) error {
    r.mutex.Lock()
    defer r.mutex.Unlock()
    
    if _, exists := r.users[user.ID]; exists {
        return ErrUserAlreadyExists
    }
    
    r.users[user.ID] = user
    return nil
}

func (r *InMemoryUserRepository) GetByID(id string) (*User, error) {
    r.mutex.RLock()
    defer r.mutex.RUnlock()
    
    user, exists := r.users[id]
    if !exists {
        return nil, ErrUserNotFound
    }
    
    return user, nil
}

func (r *InMemoryUserRepository) GetByEmail(email string) (*User, error) {
    r.mutex.RLock()
    defer r.mutex.RUnlock()
    
    for _, user := range r.users {
        if user.Email == email {
            return user, nil
        }
    }
    
    return nil, ErrUserNotFound
}

func (r *InMemoryUserRepository) Update(user *User) error {
    r.mutex.Lock()
    defer r.mutex.Unlock()
    
    if _, exists := r.users[user.ID]; !exists {
        return ErrUserNotFound
    }
    
    r.users[user.ID] = user
    return nil
}

func (r *InMemoryUserRepository) Delete(id string) error {
    r.mutex.Lock()
    defer r.mutex.Unlock()
    
    if _, exists := r.users[id]; !exists {
        return ErrUserNotFound
    }
    
    delete(r.users, id)
    return nil
}

func (r *InMemoryUserRepository) List(limit, offset int) ([]*User, error) {
    r.mutex.RLock()
    defer r.mutex.RUnlock()
    
    var users []*User
    count := 0
    skipped := 0
    
    for _, user := range r.users {
        if skipped < offset {
            skipped++
            continue
        }
        
        if count >= limit {
            break
        }
        
        users = append(users, user)
        count++
    }
    
    return users, nil
}

func (r *InMemoryUserRepository) Count() (int, error) {
    r.mutex.RLock()
    defer r.mutex.RUnlock()
    
    return len(r.users), nil
}
```

## 🔧 Advanced Repository Patterns

### Generic Repository
```go
// Generic Repository Interface
type Repository[T any, K comparable] interface {
    Create(entity *T) error
    GetByID(id K) (*T, error)
    Update(entity *T) error
    Delete(id K) error
    List(limit, offset int) ([]*T, error)
    Count() (int, error)
}

// Generic Database Repository
type GenericDatabaseRepository[T any, K comparable] struct {
    db        *sql.DB
    tableName string
    idField   string
}

func NewGenericDatabaseRepository[T any, K comparable](README.md) *GenericDatabaseRepository[T, K] {
    return &GenericDatabaseRepository[T, K]{
        db:        db,
        tableName: tableName,
        idField:   idField,
    }
}

func (r *GenericDatabaseRepository[T, K]) Create(entity *T) error {
    // Use reflection to build dynamic query
    // Implementation depends on specific requirements
    return nil
}

func (r *GenericDatabaseRepository[T, K]) GetByID(id K) (*T, error) {
    // Use reflection to build dynamic query
    // Implementation depends on specific requirements
    return nil, nil
}
```

### Cached Repository
```go
// Cached Repository Implementation
type CachedUserRepository struct {
    repository UserRepository
    cache      Cache
    ttl        time.Duration
}

func NewCachedUserRepository(repository UserRepository, cache Cache, ttl time.Duration) *CachedUserRepository {
    return &CachedUserRepository{
        repository: repository,
        cache:      cache,
        ttl:        ttl,
    }
}

func (r *CachedUserRepository) GetByID(id string) (*User, error) {
    // Try cache first
    cacheKey := fmt.Sprintf("user:%s", id)
    if cached, found := r.cache.Get(cacheKey); found {
        return cached.(*User), nil
    }
    
    // Get from repository
    user, err := r.repository.GetByID(id)
    if err != nil {
        return nil, err
    }
    
    // Cache the result
    r.cache.Set(cacheKey, user, r.ttl)
    
    return user, nil
}

func (r *CachedUserRepository) Update(user *User) error {
    // Update in repository
    err := r.repository.Update(user)
    if err != nil {
        return err
    }
    
    // Update cache
    cacheKey := fmt.Sprintf("user:%s", user.ID)
    r.cache.Set(cacheKey, user, r.ttl)
    
    return nil
}

func (r *CachedUserRepository) Delete(id string) error {
    // Delete from repository
    err := r.repository.Delete(id)
    if err != nil {
        return err
    }
    
    // Remove from cache
    cacheKey := fmt.Sprintf("user:%s", id)
    r.cache.Delete(cacheKey)
    
    return nil
}
```

### Unit of Work Pattern
```go
// Unit of Work Interface
type UnitOfWork interface {
    Users() UserRepository
    Orders() OrderRepository
    Commit() error
    Rollback() error
}

// Database Unit of Work
type DatabaseUnitOfWork struct {
    db     *sql.DB
    tx     *sql.Tx
    users  UserRepository
    orders OrderRepository
}

func NewDatabaseUnitOfWork(db *sql.DB) *DatabaseUnitOfWork {
    return &DatabaseUnitOfWork{db: db}
}

func (uow *DatabaseUnitOfWork) Begin() error {
    tx, err := uow.db.Begin()
    if err != nil {
        return err
    }
    
    uow.tx = tx
    uow.users = NewDatabaseUserRepository(tx)
    uow.orders = NewDatabaseOrderRepository(tx)
    
    return nil
}

func (uow *DatabaseUnitOfWork) Users() UserRepository {
    return uow.users
}

func (uow *DatabaseUnitOfWork) Orders() OrderRepository {
    return uow.orders
}

func (uow *DatabaseUnitOfWork) Commit() error {
    if uow.tx == nil {
        return ErrNoActiveTransaction
    }
    
    err := uow.tx.Commit()
    uow.tx = nil
    return err
}

func (uow *DatabaseUnitOfWork) Rollback() error {
    if uow.tx == nil {
        return ErrNoActiveTransaction
    }
    
    err := uow.tx.Rollback()
    uow.tx = nil
    return err
}
```

## 🎯 Razorpay-Specific Examples

### Payment Repository
```go
// Payment Repository for Razorpay
type PaymentRepository interface {
    Create(payment *Payment) error
    GetByID(id string) (*Payment, error)
    GetByOrderID(orderID string) ([]*Payment, error)
    UpdateStatus(id string, status PaymentStatus) error
    ListByMerchant(merchantID string, limit, offset int) ([]*Payment, error)
    GetStats(merchantID string, from, to time.Time) (*PaymentStats, error)
}

type Payment struct {
    ID          string        `json:"id" db:"id"`
    OrderID     string        `json:"order_id" db:"order_id"`
    MerchantID  string        `json:"merchant_id" db:"merchant_id"`
    Amount      int64         `json:"amount" db:"amount"`
    Currency    string        `json:"currency" db:"currency"`
    Status      PaymentStatus `json:"status" db:"status"`
    Method      string        `json:"method" db:"method"`
    CreatedAt   time.Time     `json:"created_at" db:"created_at"`
    UpdatedAt   time.Time     `json:"updated_at" db:"updated_at"`
}

type PaymentStatus string

const (
    PaymentStatusPending   PaymentStatus = "pending"
    PaymentStatusCaptured  PaymentStatus = "captured"
    PaymentStatusFailed    PaymentStatus = "failed"
    PaymentStatusRefunded  PaymentStatus = "refunded"
)

type PaymentStats struct {
    TotalAmount    int64 `json:"total_amount"`
    TotalCount     int   `json:"total_count"`
    SuccessCount   int   `json:"success_count"`
    FailedCount    int   `json:"failed_count"`
    RefundedCount  int   `json:"refunded_count"`
}
```

## 🎯 Best Practices

### Design Principles
1. **Single Responsibility**: Each repository should handle one entity type
2. **Interface Segregation**: Keep interfaces focused and specific
3. **Dependency Inversion**: Depend on abstractions, not concretions
4. **Open/Closed**: Open for extension, closed for modification

### Implementation Guidelines
1. **Error Handling**: Use consistent error types and handling
2. **Logging**: Add appropriate logging for debugging
3. **Validation**: Validate inputs before database operations
4. **Transactions**: Use transactions for related operations
5. **Caching**: Implement caching where appropriate

### Testing Strategies
1. **Unit Tests**: Test repository methods in isolation
2. **Integration Tests**: Test with real database
3. **Mock Tests**: Use in-memory implementations for fast tests
4. **Performance Tests**: Test with large datasets

---

**Last Updated**: December 2024  
**Category**: Repository Pattern  
**Complexity**: Intermediate Level
