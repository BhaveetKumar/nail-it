# Repository Pattern

## Pattern Name & Intent

**Repository** - Mediate between the domain and data mapping layers using a collection-like interface for accessing domain objects.

The Repository pattern provides an abstraction layer between the business logic and data access layers. It encapsulates the logic needed to access data sources, centralizing common data access functionality, providing better maintainability, and decoupling the infrastructure or technology used to access databases from the domain model layer.

## When to Use

### Appropriate Scenarios

- **Data Access Abstraction**: When you need to abstract data access logic
- **Testing**: When you need to mock data access for unit testing
- **Multiple Data Sources**: When data might come from different sources
- **Business Logic Separation**: When you want to separate business logic from data access
- **Caching**: When you need to implement caching strategies

### When NOT to Use

- **Simple CRUD**: When you only need basic CRUD operations
- **Performance Critical**: When abstraction adds unnecessary overhead
- **Tight Coupling**: When the repository becomes tightly coupled to specific entities

## Real-World Use Cases (Fintech/Payments)

### Payment Transaction Repository

```go
// Domain entity
type Payment struct {
    ID            string    `json:"id"`
    Amount        float64   `json:"amount"`
    Currency      string    `json:"currency"`
    Status        string    `json:"status"`
    UserID        string    `json:"user_id"`
    MerchantID    string    `json:"merchant_id"`
    CreatedAt     time.Time `json:"created_at"`
    UpdatedAt     time.Time `json:"updated_at"`
}

// Repository interface
type PaymentRepository interface {
    Create(ctx context.Context, payment *Payment) error
    GetByID(ctx context.Context, id string) (*Payment, error)
    GetByUserID(ctx context.Context, userID string, limit, offset int) ([]*Payment, error)
    GetByMerchantID(ctx context.Context, merchantID string, limit, offset int) ([]*Payment, error)
    Update(ctx context.Context, payment *Payment) error
    Delete(ctx context.Context, id string) error
    GetByStatus(ctx context.Context, status string) ([]*Payment, error)
    GetByDateRange(ctx context.Context, start, end time.Time) ([]*Payment, error)
}

// Database implementation
type DatabasePaymentRepository struct {
    db *sql.DB
}

func NewDatabasePaymentRepository(db *sql.DB) PaymentRepository {
    return &DatabasePaymentRepository{db: db}
}

func (r *DatabasePaymentRepository) Create(ctx context.Context, payment *Payment) error {
    query := `
        INSERT INTO payments (id, amount, currency, status, user_id, merchant_id, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    `
    _, err := r.db.ExecContext(ctx, query,
        payment.ID, payment.Amount, payment.Currency, payment.Status,
        payment.UserID, payment.MerchantID, payment.CreatedAt, payment.UpdatedAt)
    return err
}

func (r *DatabasePaymentRepository) GetByID(ctx context.Context, id string) (*Payment, error) {
    query := `
        SELECT id, amount, currency, status, user_id, merchant_id, created_at, updated_at
        FROM payments WHERE id = $1
    `
    row := r.db.QueryRowContext(ctx, query, id)

    payment := &Payment{}
    err := row.Scan(&payment.ID, &payment.Amount, &payment.Currency, &payment.Status,
        &payment.UserID, &payment.MerchantID, &payment.CreatedAt, &payment.UpdatedAt)

    if err != nil {
        if err == sql.ErrNoRows {
            return nil, ErrPaymentNotFound
        }
        return nil, err
    }

    return payment, nil
}
```

### User Account Repository

```go
type User struct {
    ID        string    `json:"id"`
    Email     string    `json:"email"`
    Name      string    `json:"name"`
    Status    string    `json:"status"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

type UserRepository interface {
    Create(ctx context.Context, user *User) error
    GetByID(ctx context.Context, id string) (*User, error)
    GetByEmail(ctx context.Context, email string) (*User, error)
    Update(ctx context.Context, user *User) error
    Delete(ctx context.Context, id string) error
    List(ctx context.Context, limit, offset int) ([]*User, error)
    GetByStatus(ctx context.Context, status string) ([]*User, error)
}

type DatabaseUserRepository struct {
    db *sql.DB
}

func (r *DatabaseUserRepository) GetByEmail(ctx context.Context, email string) (*User, error) {
    query := `SELECT id, email, name, status, created_at, updated_at FROM users WHERE email = $1`
    row := r.db.QueryRowContext(ctx, query, email)

    user := &User{}
    err := row.Scan(&user.ID, &user.Email, &user.Name, &user.Status, &user.CreatedAt, &user.UpdatedAt)

    if err != nil {
        if err == sql.ErrNoRows {
            return nil, ErrUserNotFound
        }
        return nil, err
    }

    return user, nil
}
```

## Go Implementation

### Generic Repository Interface

```go
package main

import (
    "context"
    "database/sql"
    "fmt"
    "time"
)

// Generic repository interface
type Repository[T any] interface {
    Create(ctx context.Context, entity *T) error
    GetByID(ctx context.Context, id string) (*T, error)
    Update(ctx context.Context, entity *T) error
    Delete(ctx context.Context, id string) error
    List(ctx context.Context, limit, offset int) ([]*T, error)
}

// Base repository with common functionality
type BaseRepository[T any] struct {
    db        *sql.DB
    tableName string
    idField   string
}

func NewBaseRepository[T any](db *sql.DB, tableName, idField string) *BaseRepository[T] {
    return &BaseRepository[T]{
        db:        db,
        tableName: tableName,
        idField:   idField,
    }
}

func (r *BaseRepository[T]) Delete(ctx context.Context, id string) error {
    query := fmt.Sprintf("DELETE FROM %s WHERE %s = $1", r.tableName, r.idField)
    _, err := r.db.ExecContext(ctx, query, id)
    return err
}

func (r *BaseRepository[T]) Exists(ctx context.Context, id string) (bool, error) {
    query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE %s = $1", r.tableName, r.idField)
    var count int
    err := r.db.QueryRowContext(ctx, query, id).Scan(&count)
    return count > 0, err
}
```

### Cached Repository Implementation

```go
type CachedRepository[T any] struct {
    repository Repository[T]
    cache      map[string]*T
    mutex      sync.RWMutex
    ttl        time.Duration
}

type CacheEntry[T any] struct {
    data      *T
    expiresAt time.Time
}

func NewCachedRepository[T any](repo Repository[T], ttl time.Duration) *CachedRepository[T] {
    return &CachedRepository[T]{
        repository: repo,
        cache:      make(map[string]*CacheEntry[T]),
        ttl:        ttl,
    }
}

func (r *CachedRepository[T]) GetByID(ctx context.Context, id string) (*T, error) {
    // Try cache first
    r.mutex.RLock()
    if entry, exists := r.cache[id]; exists {
        if time.Now().Before(entry.expiresAt) {
            r.mutex.RUnlock()
            return entry.data, nil
        }
        // Cache expired, remove it
        delete(r.cache, id)
    }
    r.mutex.RUnlock()

    // Get from repository
    data, err := r.repository.GetByID(ctx, id)
    if err != nil {
        return nil, err
    }

    // Cache the result
    r.mutex.Lock()
    r.cache[id] = &CacheEntry[T]{
        data:      data,
        expiresAt: time.Now().Add(r.ttl),
    }
    r.mutex.Unlock()

    return data, nil
}

func (r *CachedRepository[T]) Update(ctx context.Context, entity *T) error {
    err := r.repository.Update(ctx, entity)
    if err != nil {
        return err
    }

    // Invalidate cache
    r.mutex.Lock()
    // Assuming entity has an ID field - this would need reflection or interface
    delete(r.cache, "some_id") // This is simplified
    r.mutex.Unlock()

    return nil
}
```

### Repository with Unit of Work

```go
type UnitOfWork interface {
    PaymentRepository() PaymentRepository
    UserRepository() UserRepository
    Commit() error
    Rollback() error
}

type DatabaseUnitOfWork struct {
    db               *sql.DB
    tx               *sql.Tx
    paymentRepo      PaymentRepository
    userRepo         UserRepository
    committed        bool
    rolledBack       bool
}

func NewUnitOfWork(db *sql.DB) (UnitOfWork, error) {
    tx, err := db.Begin()
    if err != nil {
        return nil, err
    }

    uow := &DatabaseUnitOfWork{
        db:  db,
        tx:  tx,
    }

    uow.paymentRepo = NewDatabasePaymentRepository(tx)
    uow.userRepo = NewDatabaseUserRepository(tx)

    return uow, nil
}

func (uow *DatabaseUnitOfWork) PaymentRepository() PaymentRepository {
    return uow.paymentRepo
}

func (uow *DatabaseUnitOfWork) UserRepository() UserRepository {
    return uow.userRepo
}

func (uow *DatabaseUnitOfWork) Commit() error {
    if uow.committed || uow.rolledBack {
        return fmt.Errorf("transaction already completed")
    }

    err := uow.tx.Commit()
    if err == nil {
        uow.committed = true
    }
    return err
}

func (uow *DatabaseUnitOfWork) Rollback() error {
    if uow.committed || uow.rolledBack {
        return fmt.Errorf("transaction already completed")
    }

    err := uow.tx.Rollback()
    if err == nil {
        uow.rolledBack = true
    }
    return err
}
```

## Variants & Trade-offs

### Variants

#### 1. Generic Repository

```go
type GenericRepository[T any] interface {
    Create(ctx context.Context, entity *T) error
    GetByID(ctx context.Context, id string) (*T, error)
    Update(ctx context.Context, entity *T) error
    Delete(ctx context.Context, id string) error
}
```

**Pros**: Reusable across different entities
**Cons**: Less type safety, harder to implement entity-specific logic

#### 2. Specification Pattern

```go
type Specification interface {
    IsSatisfiedBy(entity interface{}) bool
    ToSQL() (string, []interface{})
}

type PaymentStatusSpecification struct {
    Status string
}

func (s *PaymentStatusSpecification) IsSatisfiedBy(entity interface{}) bool {
    if payment, ok := entity.(*Payment); ok {
        return payment.Status == s.Status
    }
    return false
}

func (s *PaymentStatusSpecification) ToSQL() (string, []interface{}) {
    return "status = ?", []interface{}{s.Status}
}

type PaymentRepository interface {
    FindBySpecification(ctx context.Context, spec Specification) ([]*Payment, error)
}
```

**Pros**: Flexible querying, composable conditions
**Cons**: More complex, potential performance issues

#### 3. Repository with Caching

```go
type CachedRepository struct {
    repository Repository
    cache      Cache
    ttl        time.Duration
}
```

**Pros**: Improved performance, reduced database load
**Cons**: Cache invalidation complexity, memory usage

### Trade-offs

| Aspect              | Pros                          | Cons                           |
| ------------------- | ----------------------------- | ------------------------------ |
| **Abstraction**     | Clean separation of concerns  | Additional layer of complexity |
| **Testing**         | Easy to mock and test         | More interfaces to maintain    |
| **Performance**     | Can implement caching easily  | Abstraction overhead           |
| **Flexibility**     | Easy to change data sources   | Can become over-engineered     |
| **Maintainability** | Centralized data access logic | More code to maintain          |

## Testable Example

```go
package main

import (
    "context"
    "testing"
    "time"
)

// Mock repository for testing
type MockPaymentRepository struct {
    payments map[string]*Payment
    createError error
    getError    error
}

func NewMockPaymentRepository() *MockPaymentRepository {
    return &MockPaymentRepository{
        payments: make(map[string]*Payment),
    }
}

func (m *MockPaymentRepository) Create(ctx context.Context, payment *Payment) error {
    if m.createError != nil {
        return m.createError
    }
    m.payments[payment.ID] = payment
    return nil
}

func (m *MockPaymentRepository) GetByID(ctx context.Context, id string) (*Payment, error) {
    if m.getError != nil {
        return nil, m.getError
    }
    if payment, exists := m.payments[id]; exists {
        return payment, nil
    }
    return nil, ErrPaymentNotFound
}

func (m *MockPaymentRepository) GetByUserID(ctx context.Context, userID string, limit, offset int) ([]*Payment, error) {
    var userPayments []*Payment
    for _, payment := range m.payments {
        if payment.UserID == userID {
            userPayments = append(userPayments, payment)
        }
    }
    return userPayments, nil
}

func (m *MockPaymentRepository) Update(ctx context.Context, payment *Payment) error {
    if _, exists := m.payments[payment.ID]; !exists {
        return ErrPaymentNotFound
    }
    m.payments[payment.ID] = payment
    return nil
}

func (m *MockPaymentRepository) Delete(ctx context.Context, id string) error {
    if _, exists := m.payments[id]; !exists {
        return ErrPaymentNotFound
    }
    delete(m.payments, id)
    return nil
}

func (m *MockPaymentRepository) GetByStatus(ctx context.Context, status string) ([]*Payment, error) {
    var statusPayments []*Payment
    for _, payment := range m.payments {
        if payment.Status == status {
            statusPayments = append(statusPayments, payment)
        }
    }
    return statusPayments, nil
}

func (m *MockPaymentRepository) GetByDateRange(ctx context.Context, start, end time.Time) ([]*Payment, error) {
    var rangePayments []*Payment
    for _, payment := range m.payments {
        if payment.CreatedAt.After(start) && payment.CreatedAt.Before(end) {
            rangePayments = append(rangePayments, payment)
        }
    }
    return rangePayments, nil
}

// Tests
func TestPaymentRepository_Create(t *testing.T) {
    repo := NewMockPaymentRepository()
    ctx := context.Background()

    payment := &Payment{
        ID:         "payment_123",
        Amount:     100.50,
        Currency:   "USD",
        Status:     "pending",
        UserID:     "user_456",
        MerchantID: "merchant_789",
        CreatedAt:  time.Now(),
        UpdatedAt:  time.Now(),
    }

    err := repo.Create(ctx, payment)
    if err != nil {
        t.Fatalf("Create() error = %v", err)
    }

    // Verify payment was created
    retrieved, err := repo.GetByID(ctx, payment.ID)
    if err != nil {
        t.Fatalf("GetByID() error = %v", err)
    }

    if retrieved.Amount != payment.Amount {
        t.Errorf("Expected amount %v, got %v", payment.Amount, retrieved.Amount)
    }
}

func TestPaymentRepository_GetByUserID(t *testing.T) {
    repo := NewMockPaymentRepository()
    ctx := context.Background()

    // Create test payments
    payment1 := &Payment{ID: "1", UserID: "user_123", Amount: 100}
    payment2 := &Payment{ID: "2", UserID: "user_123", Amount: 200}
    payment3 := &Payment{ID: "3", UserID: "user_456", Amount: 300}

    repo.Create(ctx, payment1)
    repo.Create(ctx, payment2)
    repo.Create(ctx, payment3)

    // Get payments for user_123
    payments, err := repo.GetByUserID(ctx, "user_123", 10, 0)
    if err != nil {
        t.Fatalf("GetByUserID() error = %v", err)
    }

    if len(payments) != 2 {
        t.Errorf("Expected 2 payments, got %d", len(payments))
    }
}

func TestPaymentRepository_Update(t *testing.T) {
    repo := NewMockPaymentRepository()
    ctx := context.Background()

    payment := &Payment{
        ID:     "payment_123",
        Amount: 100.50,
        Status: "pending",
    }

    repo.Create(ctx, payment)

    // Update payment
    payment.Status = "completed"
    err := repo.Update(ctx, payment)
    if err != nil {
        t.Fatalf("Update() error = %v", err)
    }

    // Verify update
    updated, err := repo.GetByID(ctx, payment.ID)
    if err != nil {
        t.Fatalf("GetByID() error = %v", err)
    }

    if updated.Status != "completed" {
        t.Errorf("Expected status 'completed', got %s", updated.Status)
    }
}

func TestPaymentRepository_Delete(t *testing.T) {
    repo := NewMockPaymentRepository()
    ctx := context.Background()

    payment := &Payment{ID: "payment_123", Amount: 100.50}
    repo.Create(ctx, payment)

    // Delete payment
    err := repo.Delete(ctx, payment.ID)
    if err != nil {
        t.Fatalf("Delete() error = %v", err)
    }

    // Verify deletion
    _, err = repo.GetByID(ctx, payment.ID)
    if err != ErrPaymentNotFound {
        t.Errorf("Expected ErrPaymentNotFound, got %v", err)
    }
}
```

## Integration Tips

### 1. With Dependency Injection

```go
type PaymentService struct {
    paymentRepo PaymentRepository
    userRepo    UserRepository
}

func NewPaymentService(paymentRepo PaymentRepository, userRepo UserRepository) *PaymentService {
    return &PaymentService{
        paymentRepo: paymentRepo,
        userRepo:    userRepo,
    }
}

func (s *PaymentService) ProcessPayment(ctx context.Context, payment *Payment) error {
    // Business logic here
    return s.paymentRepo.Create(ctx, payment)
}
```

### 2. With Context and Timeouts

```go
func (r *DatabasePaymentRepository) GetByID(ctx context.Context, id string) (*Payment, error) {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    query := `SELECT id, amount, currency, status FROM payments WHERE id = $1`
    row := r.db.QueryRowContext(ctx, query, id)

    // ... rest of implementation
}
```

### 3. With Error Handling

```go
var (
    ErrPaymentNotFound = errors.New("payment not found")
    ErrInvalidPayment  = errors.New("invalid payment data")
    ErrDuplicatePayment = errors.New("duplicate payment")
)

func (r *DatabasePaymentRepository) Create(ctx context.Context, payment *Payment) error {
    if payment.ID == "" {
        return ErrInvalidPayment
    }

    // Check for duplicates
    exists, err := r.Exists(ctx, payment.ID)
    if err != nil {
        return err
    }
    if exists {
        return ErrDuplicatePayment
    }

    // Create payment
    return r.createPayment(ctx, payment)
}
```

## Common Interview Questions

### 1. What is the Repository pattern and when would you use it?

**Answer**: The Repository pattern abstracts data access logic, providing a collection-like interface for domain objects. Use it when you need to separate business logic from data access, enable testing with mocks, or support multiple data sources.

### 2. How do you implement the Repository pattern in Go?

**Answer**: Define interfaces for data access operations, implement concrete repositories for specific data sources (database, cache, API), and use dependency injection to provide repositories to business logic.

### 3. What are the benefits and drawbacks of the Repository pattern?

**Answer**: Benefits include testability, separation of concerns, and flexibility to change data sources. Drawbacks include additional complexity, potential over-engineering, and abstraction overhead.

### 4. How do you handle transactions with the Repository pattern?

**Answer**: Use the Unit of Work pattern to coordinate multiple repository operations within a single transaction, ensuring data consistency across multiple entities.

### 5. How do you implement caching in a Repository?

**Answer**: Create a cached repository wrapper that checks cache first, falls back to the underlying repository, and updates cache on writes. Implement cache invalidation strategies for data consistency.
