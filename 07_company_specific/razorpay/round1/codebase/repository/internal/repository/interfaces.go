package repository

import (
	"context"
	"time"
)

// Entity represents a generic entity with common fields
type Entity interface {
	GetID() string
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
}

// Repository defines the contract for data access operations
type Repository[T Entity] interface {
	// Basic CRUD operations
	Create(ctx context.Context, entity T) error
	GetByID(ctx context.Context, id string) (T, error)
	Update(ctx context.Context, entity T) error
	Delete(ctx context.Context, id string) error
	
	// Query operations
	GetAll(ctx context.Context, limit, offset int) ([]T, error)
	Count(ctx context.Context) (int64, error)
	Exists(ctx context.Context, id string) (bool, error)
	
	// Search operations
	FindBy(ctx context.Context, field string, value interface{}) ([]T, error)
	FindByMultiple(ctx context.Context, filters map[string]interface{}) ([]T, error)
}

// CachedRepository extends Repository with caching capabilities
type CachedRepository[T Entity] interface {
	Repository[T]
	
	// Cache operations
	GetFromCache(ctx context.Context, id string) (T, error)
	SetCache(ctx context.Context, id string, entity T, expiration time.Duration) error
	InvalidateCache(ctx context.Context, id string) error
	ClearCache(ctx context.Context) error
}

// TransactionalRepository extends Repository with transaction support
type TransactionalRepository[T Entity] interface {
	Repository[T]
	
	// Transaction operations
	BeginTransaction(ctx context.Context) (Transaction, error)
	WithTransaction(ctx context.Context, fn func(Transaction) error) error
}

// Transaction represents a database transaction
type Transaction interface {
	Commit() error
	Rollback() error
	Repository[T Entity] // Embedded repository for transaction operations
}

// UnitOfWork manages multiple repository operations within a single transaction
type UnitOfWork interface {
	// Repository access
	GetUserRepository() Repository[*User]
	GetPaymentRepository() Repository[*Payment]
	GetOrderRepository() Repository[*Order]
	GetProductRepository() Repository[*Product]
	
	// Transaction management
	Begin() error
	Commit() error
	Rollback() error
	IsActive() bool
}

// Specification defines a query specification
type Specification interface {
	IsSatisfiedBy(entity Entity) bool
	ToSQL() (string, []interface{})
	ToMongoFilter() map[string]interface{}
}

// CompositeSpecification combines multiple specifications
type CompositeSpecification struct {
	specifications []Specification
	operator       string // "AND" or "OR"
}

// AndSpecification creates an AND specification
func AndSpecification(specs ...Specification) *CompositeSpecification {
	return &CompositeSpecification{
		specifications: specs,
		operator:       "AND",
	}
}

// OrSpecification creates an OR specification
func OrSpecification(specs ...Specification) *CompositeSpecification {
	return &CompositeSpecification{
		specifications: specs,
		operator:       "OR",
	}
}

func (cs *CompositeSpecification) IsSatisfiedBy(entity Entity) bool {
	if cs.operator == "AND" {
		for _, spec := range cs.specifications {
			if !spec.IsSatisfiedBy(entity) {
				return false
			}
		}
		return true
	} else { // OR
		for _, spec := range cs.specifications {
			if spec.IsSatisfiedBy(entity) {
				return true
			}
		}
		return false
	}
}

func (cs *CompositeSpecification) ToSQL() (string, []interface{}) {
	if len(cs.specifications) == 0 {
		return "", nil
	}
	
	var conditions []string
	var args []interface{}
	
	for _, spec := range cs.specifications {
		condition, specArgs := spec.ToSQL()
		if condition != "" {
			conditions = append(conditions, condition)
			args = append(args, specArgs...)
		}
	}
	
	if len(conditions) == 0 {
		return "", nil
	}
	
	operator := " " + cs.operator + " "
	return "(" + joinStrings(conditions, operator) + ")", args
}

func (cs *CompositeSpecification) ToMongoFilter() map[string]interface{} {
	if len(cs.specifications) == 0 {
		return map[string]interface{}{}
	}
	
	var filters []map[string]interface{}
	
	for _, spec := range cs.specifications {
		filter := spec.ToMongoFilter()
		if len(filter) > 0 {
			filters = append(filters, filter)
		}
	}
	
	if len(filters) == 0 {
		return map[string]interface{}{}
	}
	
	if cs.operator == "AND" {
		return map[string]interface{}{"$and": filters}
	} else { // OR
		return map[string]interface{}{"$or": filters}
	}
}

// SpecificationRepository extends Repository with specification support
type SpecificationRepository[T Entity] interface {
	Repository[T]
	
	// Specification-based queries
	FindBySpecification(ctx context.Context, spec Specification) ([]T, error)
	CountBySpecification(ctx context.Context, spec Specification) (int64, error)
}

// AuditRepository extends Repository with audit capabilities
type AuditRepository[T Entity] interface {
	Repository[T]
	
	// Audit operations
	GetAuditLog(ctx context.Context, entityID string) ([]AuditLog, error)
	CreateAuditLog(ctx context.Context, log AuditLog) error
}

// SoftDeleteRepository extends Repository with soft delete capabilities
type SoftDeleteRepository[T Entity] interface {
	Repository[T]
	
	// Soft delete operations
	SoftDelete(ctx context.Context, id string) error
	Restore(ctx context.Context, id string) error
	GetDeleted(ctx context.Context, limit, offset int) ([]T, error)
}

// PaginatedRepository extends Repository with pagination support
type PaginatedRepository[T Entity] interface {
	Repository[T]
	
	// Pagination operations
	GetPaginated(ctx context.Context, page, limit int) (*PaginatedResult[T], error)
	GetPaginatedBySpecification(ctx context.Context, spec Specification, page, limit int) (*PaginatedResult[T], error)
}

// PaginatedResult represents a paginated result
type PaginatedResult[T Entity] struct {
	Data       []T   `json:"data"`
	Page       int   `json:"page"`
	Limit      int   `json:"limit"`
	Total      int64 `json:"total"`
	TotalPages int   `json:"total_pages"`
	HasNext    bool  `json:"has_next"`
	HasPrev    bool  `json:"has_prev"`
}

// AuditLog represents an audit log entry
type AuditLog struct {
	ID         string                 `json:"id" bson:"id"`
	EntityType string                 `json:"entity_type" bson:"entity_type"`
	EntityID   string                 `json:"entity_id" bson:"entity_id"`
	Action     string                 `json:"action" bson:"action"`
	UserID     string                 `json:"user_id" bson:"user_id"`
	Changes    map[string]interface{} `json:"changes" bson:"changes"`
	CreatedAt  time.Time              `json:"created_at" bson:"created_at"`
}

// RepositoryError represents a repository-specific error
type RepositoryError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Err     error  `json:"-"`
}

func (re *RepositoryError) Error() string {
	if re.Err != nil {
		return re.Message + ": " + re.Err.Error()
	}
	return re.Message
}

func (re *RepositoryError) Unwrap() error {
	return re.Err
}

// Common repository error codes
const (
	ErrCodeNotFound     = "NOT_FOUND"
	ErrCodeDuplicate    = "DUPLICATE"
	ErrCodeValidation   = "VALIDATION"
	ErrCodeConnection   = "CONNECTION"
	ErrCodeTransaction  = "TRANSACTION"
	ErrCodeTimeout      = "TIMEOUT"
	ErrCodeConstraint   = "CONSTRAINT"
)

// Helper function to join strings
func joinStrings(strs []string, separator string) string {
	if len(strs) == 0 {
		return ""
	}
	if len(strs) == 1 {
		return strs[0]
	}
	
	result := strs[0]
	for i := 1; i < len(strs); i++ {
		result += separator + strs[i]
	}
	return result
}
