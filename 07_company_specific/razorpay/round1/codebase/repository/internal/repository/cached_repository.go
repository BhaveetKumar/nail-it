package repository

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"repository-service/internal/logger"
	"repository-service/internal/redis"
)

// CachedRepository implements CachedRepository interface
type CachedRepository[T Entity] struct {
	repository Repository[T]
	cache      *redis.RedisManager
	ttl        time.Duration
	logger     *logger.Logger
}

// NewCachedRepository creates a new cached repository
func NewCachedRepository[T Entity](repository Repository[T], cache *redis.RedisManager, ttl time.Duration) *CachedRepository[T] {
	return &CachedRepository[T]{
		repository: repository,
		cache:      cache,
		ttl:        ttl,
		logger:     logger.GetLogger(),
	}
}

// Create creates a new entity and caches it
func (r *CachedRepository[T]) Create(ctx context.Context, entity T) error {
	err := r.repository.Create(ctx, entity)
	if err != nil {
		return err
	}
	
	// Cache the created entity
	err = r.SetCache(ctx, entity.GetID(), entity, r.ttl)
	if err != nil {
		r.logger.Warn("Failed to cache created entity", "id", entity.GetID(), "error", err)
	}
	
	return nil
}

// GetByID retrieves an entity by ID, checking cache first
func (r *CachedRepository[T]) GetByID(ctx context.Context, id string) (T, error) {
	// Try to get from cache first
	entity, err := r.GetFromCache(ctx, id)
	if err == nil {
		r.logger.Debug("Entity retrieved from cache", "id", id)
		return entity, nil
	}
	
	// If not in cache, get from repository
	entity, err = r.repository.GetByID(ctx, id)
	if err != nil {
		return entity, err
	}
	
	// Cache the retrieved entity
	err = r.SetCache(ctx, id, entity, r.ttl)
	if err != nil {
		r.logger.Warn("Failed to cache retrieved entity", "id", id, "error", err)
	}
	
	return entity, nil
}

// Update updates an entity and invalidates cache
func (r *CachedRepository[T]) Update(ctx context.Context, entity T) error {
	err := r.repository.Update(ctx, entity)
	if err != nil {
		return err
	}
	
	// Invalidate cache
	err = r.InvalidateCache(ctx, entity.GetID())
	if err != nil {
		r.logger.Warn("Failed to invalidate cache for updated entity", "id", entity.GetID(), "error", err)
	}
	
	// Optionally cache the updated entity
	err = r.SetCache(ctx, entity.GetID(), entity, r.ttl)
	if err != nil {
		r.logger.Warn("Failed to cache updated entity", "id", entity.GetID(), "error", err)
	}
	
	return nil
}

// Delete deletes an entity and invalidates cache
func (r *CachedRepository[T]) Delete(ctx context.Context, id string) error {
	err := r.repository.Delete(ctx, id)
	if err != nil {
		return err
	}
	
	// Invalidate cache
	err = r.InvalidateCache(ctx, id)
	if err != nil {
		r.logger.Warn("Failed to invalidate cache for deleted entity", "id", id, "error", err)
	}
	
	return nil
}

// GetAll retrieves all entities (not cached)
func (r *CachedRepository[T]) GetAll(ctx context.Context, limit, offset int) ([]T, error) {
	return r.repository.GetAll(ctx, limit, offset)
}

// Count returns the total number of entities (not cached)
func (r *CachedRepository[T]) Count(ctx context.Context) (int64, error) {
	return r.repository.Count(ctx)
}

// Exists checks if an entity exists by ID
func (r *CachedRepository[T]) Exists(ctx context.Context, id string) (bool, error) {
	// Try to get from cache first
	_, err := r.GetFromCache(ctx, id)
	if err == nil {
		return true, nil
	}
	
	// If not in cache, check repository
	return r.repository.Exists(ctx, id)
}

// FindBy finds entities by field and value (not cached)
func (r *CachedRepository[T]) FindBy(ctx context.Context, field string, value interface{}) ([]T, error) {
	return r.repository.FindBy(ctx, field, value)
}

// FindByMultiple finds entities by multiple filters (not cached)
func (r *CachedRepository[T]) FindByMultiple(ctx context.Context, filters map[string]interface{}) ([]T, error) {
	return r.repository.FindByMultiple(ctx, filters)
}

// GetFromCache retrieves an entity from cache
func (r *CachedRepository[T]) GetFromCache(ctx context.Context, id string) (T, error) {
	var entity T
	key := r.getCacheKey(id)
	
	cachedData, err := r.cache.Get(ctx, key)
	if err != nil {
		return entity, &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to get entity from cache",
			Err:     err,
		}
	}
	
	err = json.Unmarshal([]byte(cachedData), &entity)
	if err != nil {
		return entity, &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Failed to unmarshal cached entity",
			Err:     err,
		}
	}
	
	return entity, nil
}

// SetCache stores an entity in cache
func (r *CachedRepository[T]) SetCache(ctx context.Context, id string, entity T, expiration time.Duration) error {
	key := r.getCacheKey(id)
	
	entityData, err := json.Marshal(entity)
	if err != nil {
		return &RepositoryError{
			Code:    ErrCodeValidation,
			Message: "Failed to marshal entity for caching",
			Err:     err,
		}
	}
	
	err = r.cache.Set(ctx, key, entityData, expiration)
	if err != nil {
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to set entity in cache",
			Err:     err,
		}
	}
	
	return nil
}

// InvalidateCache removes an entity from cache
func (r *CachedRepository[T]) InvalidateCache(ctx context.Context, id string) error {
	key := r.getCacheKey(id)
	
	err := r.cache.Del(ctx, key)
	if err != nil {
		return &RepositoryError{
			Code:    ErrCodeConnection,
			Message: "Failed to invalidate cache",
			Err:     err,
		}
	}
	
	return nil
}

// ClearCache clears all cached entities
func (r *CachedRepository[T]) ClearCache(ctx context.Context) error {
	// This is a simplified implementation
	// In a real scenario, you might want to use Redis patterns to clear specific keys
	pattern := r.getCachePattern()
	
	// Note: This would require implementing a method to get all keys matching a pattern
	// and then deleting them. For now, we'll just log a warning.
	r.logger.Warn("ClearCache not fully implemented", "pattern", pattern)
	
	return nil
}

// getCacheKey generates a cache key for an entity
func (r *CachedRepository[T]) getCacheKey(id string) string {
	var entityType string
	
	switch any(*new(T)).(type) {
	case User:
		entityType = "user"
	case Payment:
		entityType = "payment"
	case Order:
		entityType = "order"
	case Product:
		entityType = "product"
	default:
		entityType = "entity"
	}
	
	return fmt.Sprintf("%s:%s", entityType, id)
}

// getCachePattern generates a cache pattern for clearing
func (r *CachedRepository[T]) getCachePattern() string {
	var entityType string
	
	switch any(*new(T)).(type) {
	case User:
		entityType = "user"
	case Payment:
		entityType = "payment"
	case Order:
		entityType = "order"
	case Product:
		entityType = "product"
	default:
		entityType = "entity"
	}
	
	return fmt.Sprintf("%s:*", entityType)
}

// CacheStats represents cache statistics
type CacheStats struct {
	HitCount  int64 `json:"hit_count"`
	MissCount int64 `json:"miss_count"`
	HitRate   float64 `json:"hit_rate"`
}

// GetCacheStats returns cache statistics
func (r *CachedRepository[T]) GetCacheStats(ctx context.Context) (*CacheStats, error) {
	// This would require implementing cache statistics tracking
	// For now, return empty stats
	return &CacheStats{
		HitCount:  0,
		MissCount: 0,
		HitRate:   0.0,
	}, nil
}

// WarmCache warms the cache with frequently accessed entities
func (r *CachedRepository[T]) WarmCache(ctx context.Context, limit int) error {
	// Get recent entities
	entities, err := r.repository.GetAll(ctx, limit, 0)
	if err != nil {
		return err
	}
	
	// Cache each entity
	for _, entity := range entities {
		err := r.SetCache(ctx, entity.GetID(), entity, r.ttl)
		if err != nil {
			r.logger.Warn("Failed to warm cache for entity", "id", entity.GetID(), "error", err)
		}
	}
	
	r.logger.Info("Cache warmed successfully", "count", len(entities))
	return nil
}

// CacheWithTTL caches an entity with a specific TTL
func (r *CachedRepository[T]) CacheWithTTL(ctx context.Context, id string, entity T, ttl time.Duration) error {
	return r.SetCache(ctx, id, entity, ttl)
}

// GetFromCacheWithFallback retrieves from cache with fallback to repository
func (r *CachedRepository[T]) GetFromCacheWithFallback(ctx context.Context, id string) (T, error) {
	// Try cache first
	entity, err := r.GetFromCache(ctx, id)
	if err == nil {
		return entity, nil
	}
	
	// Fallback to repository
	entity, err = r.repository.GetByID(ctx, id)
	if err != nil {
		return entity, err
	}
	
	// Cache the result
	err = r.SetCache(ctx, id, entity, r.ttl)
	if err != nil {
		r.logger.Warn("Failed to cache entity after fallback", "id", id, "error", err)
	}
	
	return entity, nil
}
