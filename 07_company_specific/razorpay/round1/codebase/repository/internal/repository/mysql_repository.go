package repository

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	"repository-service/internal/logger"
)

// MySQLRepository implements Repository interface for MySQL
type MySQLRepository[T Entity] struct {
	db     *sql.DB
	table  string
	logger *logger.Logger
}

// NewMySQLRepository creates a new MySQL repository
func NewMySQLRepository[T Entity](db *sql.DB, table string) *MySQLRepository[T] {
	return &MySQLRepository[T]{
		db:     db,
		table:  table,
		logger: logger.GetLogger(),
	}
}

// Create creates a new entity
func (r *MySQLRepository[T]) Create(ctx context.Context, entity T) error {
	query := r.buildInsertQuery(entity)
	args := r.extractInsertArgs(entity)
	
	_, err := r.db.ExecContext(ctx, query, args...)
	if err != nil {
		r.logger.Error("Failed to create entity", "table", r.table, "error", err)
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to create entity",
			Err:     err,
		}
	}
	
	r.logger.Debug("Entity created successfully", "table", r.table, "id", entity.GetID())
	return nil
}

// GetByID retrieves an entity by ID
func (r *MySQLRepository[T]) GetByID(ctx context.Context, id string) (T, error) {
	var entity T
	query := fmt.Sprintf("SELECT * FROM %s WHERE id = ?", r.table)
	
	row := r.db.QueryRowContext(ctx, query, id)
	err := r.scanRow(row, &entity)
	if err != nil {
		if err == sql.ErrNoRows {
			return entity, &RepositoryError{
				Code:    ErrCodeNotFound,
				Message: fmt.Sprintf("Entity with ID %s not found", id),
				Err:     err,
			}
		}
		r.logger.Error("Failed to get entity by ID", "table", r.table, "id", id, "error", err)
		return entity, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to get entity by ID",
			Err:     err,
		}
	}
	
	return entity, nil
}

// Update updates an entity
func (r *MySQLRepository[T]) Update(ctx context.Context, entity T) error {
	query := r.buildUpdateQuery(entity)
	args := r.extractUpdateArgs(entity)
	
	result, err := r.db.ExecContext(ctx, query, args...)
	if err != nil {
		r.logger.Error("Failed to update entity", "table", r.table, "id", entity.GetID(), "error", err)
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to update entity",
			Err:     err,
		}
	}
	
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to get rows affected",
			Err:     err,
		}
	}
	
	if rowsAffected == 0 {
		return &RepositoryError{
			Code:    ErrCodeNotFound,
			Message: fmt.Sprintf("Entity with ID %s not found", entity.GetID()),
		}
	}
	
	r.logger.Debug("Entity updated successfully", "table", r.table, "id", entity.GetID())
	return nil
}

// Delete deletes an entity by ID
func (r *MySQLRepository[T]) Delete(ctx context.Context, id string) error {
	query := fmt.Sprintf("DELETE FROM %s WHERE id = ?", r.table)
	
	result, err := r.db.ExecContext(ctx, query, id)
	if err != nil {
		r.logger.Error("Failed to delete entity", "table", r.table, "id", id, "error", err)
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to delete entity",
			Err:     err,
		}
	}
	
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to get rows affected",
			Err:     err,
		}
	}
	
	if rowsAffected == 0 {
		return &RepositoryError{
			Code:    ErrCodeNotFound,
			Message: fmt.Sprintf("Entity with ID %s not found", id),
		}
	}
	
	r.logger.Debug("Entity deleted successfully", "table", r.table, "id", id)
	return nil
}

// GetAll retrieves all entities with pagination
func (r *MySQLRepository[T]) GetAll(ctx context.Context, limit, offset int) ([]T, error) {
	query := fmt.Sprintf("SELECT * FROM %s ORDER BY created_at DESC LIMIT ? OFFSET ?", r.table)
	
	rows, err := r.db.QueryContext(ctx, query, limit, offset)
	if err != nil {
		r.logger.Error("Failed to get all entities", "table", r.table, "error", err)
		return nil, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to get all entities",
			Err:     err,
		}
	}
	defer rows.Close()
	
	return r.scanRows(rows)
}

// Count returns the total number of entities
func (r *MySQLRepository[T]) Count(ctx context.Context) (int64, error) {
	query := fmt.Sprintf("SELECT COUNT(*) FROM %s", r.table)
	
	var count int64
	err := r.db.QueryRowContext(ctx, query).Scan(&count)
	if err != nil {
		r.logger.Error("Failed to count entities", "table", r.table, "error", err)
		return 0, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to count entities",
			Err:     err,
		}
	}
	
	return count, nil
}

// Exists checks if an entity exists by ID
func (r *MySQLRepository[T]) Exists(ctx context.Context, id string) (bool, error) {
	query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE id = ?", r.table)
	
	var count int
	err := r.db.QueryRowContext(ctx, query, id).Scan(&count)
	if err != nil {
		r.logger.Error("Failed to check entity existence", "table", r.table, "id", id, "error", err)
		return false, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to check entity existence",
			Err:     err,
		}
	}
	
	return count > 0, nil
}

// FindBy finds entities by field and value
func (r *MySQLRepository[T]) FindBy(ctx context.Context, field string, value interface{}) ([]T, error) {
	query := fmt.Sprintf("SELECT * FROM %s WHERE %s = ? ORDER BY created_at DESC", r.table, field)
	
	rows, err := r.db.QueryContext(ctx, query, value)
	if err != nil {
		r.logger.Error("Failed to find entities by field", "table", r.table, "field", field, "error", err)
		return nil, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to find entities by field",
			Err:     err,
		}
	}
	defer rows.Close()
	
	return r.scanRows(rows)
}

// FindByMultiple finds entities by multiple filters
func (r *MySQLRepository[T]) FindByMultiple(ctx context.Context, filters map[string]interface{}) ([]T, error) {
	if len(filters) == 0 {
		return r.GetAll(ctx, 100, 0)
	}
	
	var conditions []string
	var args []interface{}
	
	for field, value := range filters {
		conditions = append(conditions, fmt.Sprintf("%s = ?", field))
		args = append(args, value)
	}
	
	query := fmt.Sprintf("SELECT * FROM %s WHERE %s ORDER BY created_at DESC", 
		r.table, strings.Join(conditions, " AND "))
	
	rows, err := r.db.QueryContext(ctx, query, args...)
	if err != nil {
		r.logger.Error("Failed to find entities by multiple filters", "table", r.table, "error", err)
		return nil, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to find entities by multiple filters",
			Err:     err,
		}
	}
	defer rows.Close()
	
	return r.scanRows(rows)
}

// buildInsertQuery builds an INSERT query for the entity
func (r *MySQLRepository[T]) buildInsertQuery(entity T) string {
	var fields []string
	var placeholders []string
	
	switch e := any(entity).(type) {
	case *User:
		fields = []string{"id", "email", "name", "status", "created_at", "updated_at"}
		placeholders = []string{"?", "?", "?", "?", "?", "?"}
	case *Payment:
		fields = []string{"id", "user_id", "amount", "currency", "status", "gateway", "transaction_id", "metadata", "created_at", "updated_at"}
		placeholders = []string{"?", "?", "?", "?", "?", "?", "?", "?", "?", "?"}
	case *Order:
		fields = []string{"id", "user_id", "payment_id", "total_amount", "currency", "status", "items", "metadata", "created_at", "updated_at"}
		placeholders = []string{"?", "?", "?", "?", "?", "?", "?", "?", "?", "?"}
	case *Product:
		fields = []string{"id", "name", "description", "price", "currency", "category", "stock", "status", "metadata", "created_at", "updated_at"}
		placeholders = []string{"?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"}
	default:
		// Generic implementation
		fields = []string{"id", "created_at", "updated_at"}
		placeholders = []string{"?", "?", "?"}
	}
	
	return fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)", 
		r.table, strings.Join(fields, ", "), strings.Join(placeholders, ", "))
}

// extractInsertArgs extracts arguments for INSERT query
func (r *MySQLRepository[T]) extractInsertArgs(entity T) []interface{} {
	now := time.Now()
	
	switch e := any(entity).(type) {
	case *User:
		return []interface{}{e.ID, e.Email, e.Name, e.Status, now, now}
	case *Payment:
		return []interface{}{e.ID, e.UserID, e.Amount, e.Currency, e.Status, e.Gateway, e.TransactionID, e.Metadata, now, now}
	case *Order:
		return []interface{}{e.ID, e.UserID, e.PaymentID, e.TotalAmount, e.Currency, e.Status, e.Items, e.Metadata, now, now}
	case *Product:
		return []interface{}{e.ID, e.Name, e.Description, e.Price, e.Currency, e.Category, e.Stock, e.Status, e.Metadata, now, now}
	default:
		return []interface{}{entity.GetID(), now, now}
	}
}

// buildUpdateQuery builds an UPDATE query for the entity
func (r *MySQLRepository[T]) buildUpdateQuery(entity T) string {
	var fields []string
	
	switch any(entity).(type) {
	case *User:
		fields = []string{"email = ?", "name = ?", "status = ?", "updated_at = ?"}
	case *Payment:
		fields = []string{"user_id = ?", "amount = ?", "currency = ?", "status = ?", "gateway = ?", "transaction_id = ?", "metadata = ?", "updated_at = ?"}
	case *Order:
		fields = []string{"user_id = ?", "payment_id = ?", "total_amount = ?", "currency = ?", "status = ?", "items = ?", "metadata = ?", "updated_at = ?"}
	case *Product:
		fields = []string{"name = ?", "description = ?", "price = ?", "currency = ?", "category = ?", "stock = ?", "status = ?", "metadata = ?", "updated_at = ?"}
	default:
		fields = []string{"updated_at = ?"}
	}
	
	return fmt.Sprintf("UPDATE %s SET %s WHERE id = ?", r.table, strings.Join(fields, ", "))
}

// extractUpdateArgs extracts arguments for UPDATE query
func (r *MySQLRepository[T]) extractUpdateArgs(entity T) []interface{} {
	now := time.Now()
	
	switch e := any(entity).(type) {
	case *User:
		return []interface{}{e.Email, e.Name, e.Status, now, e.ID}
	case *Payment:
		return []interface{}{e.UserID, e.Amount, e.Currency, e.Status, e.Gateway, e.TransactionID, e.Metadata, now, e.ID}
	case *Order:
		return []interface{}{e.UserID, e.PaymentID, e.TotalAmount, e.Currency, e.Status, e.Items, e.Metadata, now, e.ID}
	case *Product:
		return []interface{}{e.Name, e.Description, e.Price, e.Currency, e.Category, e.Stock, e.Status, e.Metadata, now, e.ID}
	default:
		return []interface{}{now, entity.GetID()}
	}
}

// scanRow scans a single row into an entity
func (r *MySQLRepository[T]) scanRow(row *sql.Row, entity *T) error {
	switch e := any(entity).(type) {
	case **User:
		var user User
		err := row.Scan(&user.ID, &user.Email, &user.Name, &user.Status, &user.CreatedAt, &user.UpdatedAt, &user.DeletedAt)
		if err != nil {
			return err
		}
		*e = &user
	case **Payment:
		var payment Payment
		err := row.Scan(&payment.ID, &payment.UserID, &payment.Amount, &payment.Currency, &payment.Status, &payment.Gateway, &payment.TransactionID, &payment.Metadata, &payment.CreatedAt, &payment.UpdatedAt, &payment.DeletedAt)
		if err != nil {
			return err
		}
		*e = &payment
	case **Order:
		var order Order
		err := row.Scan(&order.ID, &order.UserID, &order.PaymentID, &order.TotalAmount, &order.Currency, &order.Status, &order.Items, &order.Metadata, &order.CreatedAt, &order.UpdatedAt, &order.DeletedAt)
		if err != nil {
			return err
		}
		*e = &order
	case **Product:
		var product Product
		err := row.Scan(&product.ID, &product.Name, &product.Description, &product.Price, &product.Currency, &product.Category, &product.Stock, &product.Status, &product.Metadata, &product.CreatedAt, &product.UpdatedAt, &product.DeletedAt)
		if err != nil {
			return err
		}
		*e = &product
	default:
		return fmt.Errorf("unsupported entity type")
	}
	
	return nil
}

// scanRows scans multiple rows into entities
func (r *MySQLRepository[T]) scanRows(rows *sql.Rows) ([]T, error) {
	var entities []T
	
	for rows.Next() {
		var entity T
		err := r.scanRow(rows, &entity)
		if err != nil {
			return nil, err
		}
		entities = append(entities, entity)
	}
	
	if err := rows.Err(); err != nil {
		return nil, err
	}
	
	return entities, nil
}
