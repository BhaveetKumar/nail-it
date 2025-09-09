package repository

import (
	"context"
	"fmt"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"repository-service/internal/logger"
)

// MongoRepository implements Repository interface for MongoDB
type MongoRepository[T Entity] struct {
	collection *mongo.Collection
	logger     *logger.Logger
}

// NewMongoRepository creates a new MongoDB repository
func NewMongoRepository[T Entity](collection *mongo.Collection) *MongoRepository[T] {
	return &MongoRepository[T]{
		collection: collection,
		logger:     logger.GetLogger(),
	}
}

// Create creates a new entity
func (r *MongoRepository[T]) Create(ctx context.Context, entity T) error {
	now := time.Now()
	
	// Set timestamps
	switch e := any(entity).(type) {
	case *User:
		e.CreatedAt = now
		e.UpdatedAt = now
	case *Payment:
		e.CreatedAt = now
		e.UpdatedAt = now
	case *Order:
		e.CreatedAt = now
		e.UpdatedAt = now
	case *Product:
		e.CreatedAt = now
		e.UpdatedAt = now
	}
	
	_, err := r.collection.InsertOne(ctx, entity)
	if err != nil {
		r.logger.Error("Failed to create entity", "collection", r.collection.Name(), "error", err)
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to create entity",
			Err:     err,
		}
	}
	
	r.logger.Debug("Entity created successfully", "collection", r.collection.Name(), "id", entity.GetID())
	return nil
}

// GetByID retrieves an entity by ID
func (r *MongoRepository[T]) GetByID(ctx context.Context, id string) (T, error) {
	var entity T
	filter := bson.M{"id": id}
	
	err := r.collection.FindOne(ctx, filter).Decode(&entity)
	if err != nil {
		if err == mongo.ErrNoDocuments {
			return entity, &RepositoryError{
				Code:    ErrCodeNotFound,
				Message: fmt.Sprintf("Entity with ID %s not found", id),
				Err:     err,
			}
		}
		r.logger.Error("Failed to get entity by ID", "collection", r.collection.Name(), "id", id, "error", err)
		return entity, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to get entity by ID",
			Err:     err,
		}
	}
	
	return entity, nil
}

// Update updates an entity
func (r *MongoRepository[T]) Update(ctx context.Context, entity T) error {
	now := time.Now()
	
	// Set updated timestamp
	switch e := any(entity).(type) {
	case *User:
		e.UpdatedAt = now
	case *Payment:
		e.UpdatedAt = now
	case *Order:
		e.UpdatedAt = now
	case *Product:
		e.UpdatedAt = now
	}
	
	filter := bson.M{"id": entity.GetID()}
	update := bson.M{"$set": entity}
	
	result, err := r.collection.UpdateOne(ctx, filter, update)
	if err != nil {
		r.logger.Error("Failed to update entity", "collection", r.collection.Name(), "id", entity.GetID(), "error", err)
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to update entity",
			Err:     err,
		}
	}
	
	if result.MatchedCount == 0 {
		return &RepositoryError{
			Code:    ErrCodeNotFound,
			Message: fmt.Sprintf("Entity with ID %s not found", entity.GetID()),
		}
	}
	
	r.logger.Debug("Entity updated successfully", "collection", r.collection.Name(), "id", entity.GetID())
	return nil
}

// Delete deletes an entity by ID
func (r *MongoRepository[T]) Delete(ctx context.Context, id string) error {
	filter := bson.M{"id": id}
	
	result, err := r.collection.DeleteOne(ctx, filter)
	if err != nil {
		r.logger.Error("Failed to delete entity", "collection", r.collection.Name(), "id", id, "error", err)
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to delete entity",
			Err:     err,
		}
	}
	
	if result.DeletedCount == 0 {
		return &RepositoryError{
			Code:    ErrCodeNotFound,
			Message: fmt.Sprintf("Entity with ID %s not found", id),
		}
	}
	
	r.logger.Debug("Entity deleted successfully", "collection", r.collection.Name(), "id", id)
	return nil
}

// GetAll retrieves all entities with pagination
func (r *MongoRepository[T]) GetAll(ctx context.Context, limit, offset int) ([]T, error) {
	opts := options.Find().
		SetSort(bson.D{{"created_at", -1}}).
		SetLimit(int64(limit)).
		SetSkip(int64(offset))
	
	cursor, err := r.collection.Find(ctx, bson.M{}, opts)
	if err != nil {
		r.logger.Error("Failed to get all entities", "collection", r.collection.Name(), "error", err)
		return nil, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to get all entities",
			Err:     err,
		}
	}
	defer cursor.Close(ctx)
	
	return r.scanCursor(cursor, ctx)
}

// Count returns the total number of entities
func (r *MongoRepository[T]) Count(ctx context.Context) (int64, error) {
	count, err := r.collection.CountDocuments(ctx, bson.M{})
	if err != nil {
		r.logger.Error("Failed to count entities", "collection", r.collection.Name(), "error", err)
		return 0, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to count entities",
			Err:     err,
		}
	}
	
	return count, nil
}

// Exists checks if an entity exists by ID
func (r *MongoRepository[T]) Exists(ctx context.Context, id string) (bool, error) {
	filter := bson.M{"id": id}
	count, err := r.collection.CountDocuments(ctx, filter)
	if err != nil {
		r.logger.Error("Failed to check entity existence", "collection", r.collection.Name(), "id", id, "error", err)
		return false, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to check entity existence",
			Err:     err,
		}
	}
	
	return count > 0, nil
}

// FindBy finds entities by field and value
func (r *MongoRepository[T]) FindBy(ctx context.Context, field string, value interface{}) ([]T, error) {
	filter := bson.M{field: value}
	opts := options.Find().SetSort(bson.D{{"created_at", -1}})
	
	cursor, err := r.collection.Find(ctx, filter, opts)
	if err != nil {
		r.logger.Error("Failed to find entities by field", "collection", r.collection.Name(), "field", field, "error", err)
		return nil, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to find entities by field",
			Err:     err,
		}
	}
	defer cursor.Close(ctx)
	
	return r.scanCursor(cursor, ctx)
}

// FindByMultiple finds entities by multiple filters
func (r *MongoRepository[T]) FindByMultiple(ctx context.Context, filters map[string]interface{}) ([]T, error) {
	if len(filters) == 0 {
		return r.GetAll(ctx, 100, 0)
	}
	
	opts := options.Find().SetSort(bson.D{{"created_at", -1}})
	
	cursor, err := r.collection.Find(ctx, filters, opts)
	if err != nil {
		r.logger.Error("Failed to find entities by multiple filters", "collection", r.collection.Name(), "error", err)
		return nil, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to find entities by multiple filters",
			Err:     err,
		}
	}
	defer cursor.Close(ctx)
	
	return r.scanCursor(cursor, ctx)
}

// FindBySpecification finds entities by specification
func (r *MongoRepository[T]) FindBySpecification(ctx context.Context, spec Specification) ([]T, error) {
	filter := spec.ToMongoFilter()
	opts := options.Find().SetSort(bson.D{{"created_at", -1}})
	
	cursor, err := r.collection.Find(ctx, filter, opts)
	if err != nil {
		r.logger.Error("Failed to find entities by specification", "collection", r.collection.Name(), "error", err)
		return nil, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to find entities by specification",
			Err:     err,
		}
	}
	defer cursor.Close(ctx)
	
	return r.scanCursor(cursor, ctx)
}

// CountBySpecification counts entities by specification
func (r *MongoRepository[T]) CountBySpecification(ctx context.Context, spec Specification) (int64, error) {
	filter := spec.ToMongoFilter()
	
	count, err := r.collection.CountDocuments(ctx, filter)
	if err != nil {
		r.logger.Error("Failed to count entities by specification", "collection", r.collection.Name(), "error", err)
		return 0, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to count entities by specification",
			Err:     err,
		}
	}
	
	return count, nil
}

// GetPaginated retrieves paginated entities
func (r *MongoRepository[T]) GetPaginated(ctx context.Context, page, limit int) (*PaginatedResult[T], error) {
	offset := (page - 1) * limit
	
	// Get total count
	total, err := r.Count(ctx)
	if err != nil {
		return nil, err
	}
	
	// Get entities
	entities, err := r.GetAll(ctx, limit, offset)
	if err != nil {
		return nil, err
	}
	
	totalPages := int((total + int64(limit) - 1) / int64(limit))
	
	return &PaginatedResult[T]{
		Data:       entities,
		Page:       page,
		Limit:      limit,
		Total:      total,
		TotalPages: totalPages,
		HasNext:    page < totalPages,
		HasPrev:    page > 1,
	}, nil
}

// GetPaginatedBySpecification retrieves paginated entities by specification
func (r *MongoRepository[T]) GetPaginatedBySpecification(ctx context.Context, spec Specification, page, limit int) (*PaginatedResult[T], error) {
	offset := (page - 1) * limit
	
	// Get total count
	total, err := r.CountBySpecification(ctx, spec)
	if err != nil {
		return nil, err
	}
	
	// Get entities
	filter := spec.ToMongoFilter()
	opts := options.Find().
		SetSort(bson.D{{"created_at", -1}}).
		SetLimit(int64(limit)).
		SetSkip(int64(offset))
	
	cursor, err := r.collection.Find(ctx, filter, opts)
	if err != nil {
		return nil, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to get paginated entities by specification",
			Err:     err,
		}
	}
	defer cursor.Close(ctx)
	
	entities, err := r.scanCursor(cursor, ctx)
	if err != nil {
		return nil, err
	}
	
	totalPages := int((total + int64(limit) - 1) / int64(limit))
	
	return &PaginatedResult[T]{
		Data:       entities,
		Page:       page,
		Limit:      limit,
		Total:      total,
		TotalPages: totalPages,
		HasNext:    page < totalPages,
		HasPrev:    page > 1,
	}, nil
}

// SoftDelete performs a soft delete by setting deleted_at timestamp
func (r *MongoRepository[T]) SoftDelete(ctx context.Context, id string) error {
	filter := bson.M{"id": id}
	update := bson.M{"$set": bson.M{"deleted_at": time.Now()}}
	
	result, err := r.collection.UpdateOne(ctx, filter, update)
	if err != nil {
		r.logger.Error("Failed to soft delete entity", "collection", r.collection.Name(), "id", id, "error", err)
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to soft delete entity",
			Err:     err,
		}
	}
	
	if result.MatchedCount == 0 {
		return &RepositoryError{
			Code:    ErrCodeNotFound,
			Message: fmt.Sprintf("Entity with ID %s not found", id),
		}
	}
	
	r.logger.Debug("Entity soft deleted successfully", "collection", r.collection.Name(), "id", id)
	return nil
}

// Restore restores a soft-deleted entity
func (r *MongoRepository[T]) Restore(ctx context.Context, id string) error {
	filter := bson.M{"id": id}
	update := bson.M{"$unset": bson.M{"deleted_at": ""}}
	
	result, err := r.collection.UpdateOne(ctx, filter, update)
	if err != nil {
		r.logger.Error("Failed to restore entity", "collection", r.collection.Name(), "id", id, "error", err)
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to restore entity",
			Err:     err,
		}
	}
	
	if result.MatchedCount == 0 {
		return &RepositoryError{
			Code:    ErrCodeNotFound,
			Message: fmt.Sprintf("Entity with ID %s not found", id),
		}
	}
	
	r.logger.Debug("Entity restored successfully", "collection", r.collection.Name(), "id", id)
	return nil
}

// GetDeleted retrieves soft-deleted entities
func (r *MongoRepository[T]) GetDeleted(ctx context.Context, limit, offset int) ([]T, error) {
	filter := bson.M{"deleted_at": bson.M{"$ne": nil}}
	opts := options.Find().
		SetSort(bson.D{{"deleted_at", -1}}).
		SetLimit(int64(limit)).
		SetSkip(int64(offset))
	
	cursor, err := r.collection.Find(ctx, filter, opts)
	if err != nil {
		r.logger.Error("Failed to get deleted entities", "collection", r.collection.Name(), "error", err)
		return nil, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to get deleted entities",
			Err:     err,
		}
	}
	defer cursor.Close(ctx)
	
	return r.scanCursor(cursor, ctx)
}

// scanCursor scans a MongoDB cursor into entities
func (r *MongoRepository[T]) scanCursor(cursor *mongo.Cursor, ctx context.Context) ([]T, error) {
	var entities []T
	
	for cursor.Next(ctx) {
		var entity T
		err := cursor.Decode(&entity)
		if err != nil {
			return nil, err
		}
		entities = append(entities, entity)
	}
	
	if err := cursor.Err(); err != nil {
		return nil, err
	}
	
	return entities, nil
}

// CreateIndexes creates necessary indexes for the collection
func (r *MongoRepository[T]) CreateIndexes(ctx context.Context) error {
	indexes := []mongo.IndexModel{
		{
			Keys: bson.D{{"id", 1}},
			Options: options.Index().SetUnique(true),
		},
		{
			Keys: bson.D{{"created_at", -1}},
		},
		{
			Keys: bson.D{{"updated_at", -1}},
		},
	}
	
	// Add entity-specific indexes
	switch any(*new(T)).(type) {
	case User:
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"email", 1}},
			Options: options.Index().SetUnique(true),
		})
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"status", 1}},
		})
	case Payment:
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"user_id", 1}},
		})
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"status", 1}},
		})
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"gateway", 1}},
		})
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"transaction_id", 1}},
		})
	case Order:
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"user_id", 1}},
		})
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"status", 1}},
		})
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"payment_id", 1}},
		})
	case Product:
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"category", 1}},
		})
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"status", 1}},
		})
		indexes = append(indexes, mongo.IndexModel{
			Keys: bson.D{{"price", 1}},
		})
	}
	
	_, err := r.collection.Indexes().CreateMany(ctx, indexes)
	if err != nil {
		r.logger.Error("Failed to create indexes", "collection", r.collection.Name(), "error", err)
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to create indexes",
			Err:     err,
		}
	}
	
	r.logger.Info("Indexes created successfully", "collection", r.collection.Name())
	return nil
}
