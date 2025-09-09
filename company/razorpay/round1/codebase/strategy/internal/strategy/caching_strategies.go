package strategy

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// RedisCachingStrategy implements CachingStrategy for Redis
type RedisCachingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewRedisCachingStrategy creates a new Redis caching strategy
func NewRedisCachingStrategy() *RedisCachingStrategy {
	return &RedisCachingStrategy{
		timeout:   10 * time.Millisecond,
		available: true,
	}
}

// Get retrieves value from Redis cache
func (r *RedisCachingStrategy) Get(ctx context.Context, key string) (interface{}, error) {
	// Simulate Redis get operation
	time.Sleep(r.timeout)
	
	// Mock data
	if key == "user:123" {
		return map[string]interface{}{
			"id":    "123",
			"name":  "John Doe",
			"email": "john@example.com",
		}, nil
	}
	
	return nil, fmt.Errorf("key not found: %s", key)
}

// Set stores value in Redis cache
func (r *RedisCachingStrategy) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	// Simulate Redis set operation
	time.Sleep(r.timeout)
	return nil
}

// Delete removes value from Redis cache
func (r *RedisCachingStrategy) Delete(ctx context.Context, key string) error {
	// Simulate Redis delete operation
	time.Sleep(r.timeout)
	return nil
}

// Clear clears all values from Redis cache
func (r *RedisCachingStrategy) Clear(ctx context.Context) error {
	// Simulate Redis clear operation
	time.Sleep(r.timeout)
	return nil
}

// GetStrategyName returns the strategy name
func (r *RedisCachingStrategy) GetStrategyName() string {
	return "redis"
}

// GetSupportedTypes returns supported types
func (r *RedisCachingStrategy) GetSupportedTypes() []string {
	return []string{"all"}
}

// GetAccessTime returns access time
func (r *RedisCachingStrategy) GetAccessTime() time.Duration {
	return r.timeout
}

// IsAvailable returns availability status
func (r *RedisCachingStrategy) IsAvailable() bool {
	return r.available
}

// MemoryCachingStrategy implements CachingStrategy for in-memory cache
type MemoryCachingStrategy struct {
	cache    map[string]interface{}
	ttl      map[string]time.Time
	timeout  time.Duration
	available bool
	mu       sync.RWMutex
}

// NewMemoryCachingStrategy creates a new in-memory caching strategy
func NewMemoryCachingStrategy() *MemoryCachingStrategy {
	return &MemoryCachingStrategy{
		cache:     make(map[string]interface{}),
		ttl:       make(map[string]time.Time),
		timeout:   1 * time.Millisecond,
		available: true,
	}
}

// Get retrieves value from memory cache
func (m *MemoryCachingStrategy) Get(ctx context.Context, key string) (interface{}, error) {
	// Simulate memory get operation
	time.Sleep(m.timeout)
	
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	// Check if key exists and is not expired
	if value, exists := m.cache[key]; exists {
		if ttl, ttlExists := m.ttl[key]; ttlExists {
			if time.Now().Before(ttl) {
				return value, nil
			}
			// Key expired, remove it
			delete(m.cache, key)
			delete(m.ttl, key)
		} else {
			return value, nil
		}
	}
	
	return nil, fmt.Errorf("key not found: %s", key)
}

// Set stores value in memory cache
func (m *MemoryCachingStrategy) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	// Simulate memory set operation
	time.Sleep(m.timeout)
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.cache[key] = value
	if ttl > 0 {
		m.ttl[key] = time.Now().Add(ttl)
	}
	
	return nil
}

// Delete removes value from memory cache
func (m *MemoryCachingStrategy) Delete(ctx context.Context, key string) error {
	// Simulate memory delete operation
	time.Sleep(m.timeout)
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	delete(m.cache, key)
	delete(m.ttl, key)
	
	return nil
}

// Clear clears all values from memory cache
func (m *MemoryCachingStrategy) Clear(ctx context.Context) error {
	// Simulate memory clear operation
	time.Sleep(m.timeout)
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.cache = make(map[string]interface{})
	m.ttl = make(map[string]time.Time)
	
	return nil
}

// GetStrategyName returns the strategy name
func (m *MemoryCachingStrategy) GetStrategyName() string {
	return "memory"
}

// GetSupportedTypes returns supported types
func (m *MemoryCachingStrategy) GetSupportedTypes() []string {
	return []string{"all"}
}

// GetAccessTime returns access time
func (m *MemoryCachingStrategy) GetAccessTime() time.Duration {
	return m.timeout
}

// IsAvailable returns availability status
func (m *MemoryCachingStrategy) IsAvailable() bool {
	return m.available
}

// DatabaseCachingStrategy implements CachingStrategy for database cache
type DatabaseCachingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewDatabaseCachingStrategy creates a new database caching strategy
func NewDatabaseCachingStrategy() *DatabaseCachingStrategy {
	return &DatabaseCachingStrategy{
		timeout:   50 * time.Millisecond,
		available: true,
	}
}

// Get retrieves value from database cache
func (d *DatabaseCachingStrategy) Get(ctx context.Context, key string) (interface{}, error) {
	// Simulate database get operation
	time.Sleep(d.timeout)
	
	// Mock data
	if key == "user:456" {
		return map[string]interface{}{
			"id":    "456",
			"name":  "Jane Smith",
			"email": "jane@example.com",
		}, nil
	}
	
	return nil, fmt.Errorf("key not found: %s", key)
}

// Set stores value in database cache
func (d *DatabaseCachingStrategy) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	// Simulate database set operation
	time.Sleep(d.timeout)
	return nil
}

// Delete removes value from database cache
func (d *DatabaseCachingStrategy) Delete(ctx context.Context, key string) error {
	// Simulate database delete operation
	time.Sleep(d.timeout)
	return nil
}

// Clear clears all values from database cache
func (d *DatabaseCachingStrategy) Clear(ctx context.Context) error {
	// Simulate database clear operation
	time.Sleep(d.timeout)
	return nil
}

// GetStrategyName returns the strategy name
func (d *DatabaseCachingStrategy) GetStrategyName() string {
	return "database"
}

// GetSupportedTypes returns supported types
func (d *DatabaseCachingStrategy) GetSupportedTypes() []string {
	return []string{"all"}
}

// GetAccessTime returns access time
func (d *DatabaseCachingStrategy) GetAccessTime() time.Duration {
	return d.timeout
}

// IsAvailable returns availability status
func (d *DatabaseCachingStrategy) IsAvailable() bool {
	return d.available
}

// HybridCachingStrategy implements CachingStrategy for hybrid cache (Redis + Memory)
type HybridCachingStrategy struct {
	redisStrategy   CachingStrategy
	memoryStrategy  CachingStrategy
	timeout         time.Duration
	available       bool
}

// NewHybridCachingStrategy creates a new hybrid caching strategy
func NewHybridCachingStrategy() *HybridCachingStrategy {
	return &HybridCachingStrategy{
		redisStrategy:  NewRedisCachingStrategy(),
		memoryStrategy: NewMemoryCachingStrategy(),
		timeout:        15 * time.Millisecond,
		available:      true,
	}
}

// Get retrieves value from hybrid cache (tries memory first, then Redis)
func (h *HybridCachingStrategy) Get(ctx context.Context, key string) (interface{}, error) {
	// Try memory cache first
	value, err := h.memoryStrategy.Get(ctx, key)
	if err == nil {
		return value, nil
	}
	
	// Try Redis cache
	value, err = h.redisStrategy.Get(ctx, key)
	if err == nil {
		// Store in memory cache for future access
		h.memoryStrategy.Set(ctx, key, value, 5*time.Minute)
		return value, nil
	}
	
	return nil, fmt.Errorf("key not found: %s", key)
}

// Set stores value in hybrid cache (both memory and Redis)
func (h *HybridCachingStrategy) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	// Set in both caches
	err1 := h.memoryStrategy.Set(ctx, key, value, ttl)
	err2 := h.redisStrategy.Set(ctx, key, value, ttl)
	
	if err1 != nil && err2 != nil {
		return fmt.Errorf("failed to set in both caches: %v, %v", err1, err2)
	}
	
	return nil
}

// Delete removes value from hybrid cache (both memory and Redis)
func (h *HybridCachingStrategy) Delete(ctx context.Context, key string) error {
	// Delete from both caches
	err1 := h.memoryStrategy.Delete(ctx, key)
	err2 := h.redisStrategy.Delete(ctx, key)
	
	if err1 != nil && err2 != nil {
		return fmt.Errorf("failed to delete from both caches: %v, %v", err1, err2)
	}
	
	return nil
}

// Clear clears all values from hybrid cache (both memory and Redis)
func (h *HybridCachingStrategy) Clear(ctx context.Context) error {
	// Clear both caches
	err1 := h.memoryStrategy.Clear(ctx)
	err2 := h.redisStrategy.Clear(ctx)
	
	if err1 != nil && err2 != nil {
		return fmt.Errorf("failed to clear both caches: %v, %v", err1, err2)
	}
	
	return nil
}

// GetStrategyName returns the strategy name
func (h *HybridCachingStrategy) GetStrategyName() string {
	return "hybrid"
}

// GetSupportedTypes returns supported types
func (h *HybridCachingStrategy) GetSupportedTypes() []string {
	return []string{"all"}
}

// GetAccessTime returns access time
func (h *HybridCachingStrategy) GetAccessTime() time.Duration {
	return h.timeout
}

// IsAvailable returns availability status
func (h *HybridCachingStrategy) IsAvailable() bool {
	return h.available
}
