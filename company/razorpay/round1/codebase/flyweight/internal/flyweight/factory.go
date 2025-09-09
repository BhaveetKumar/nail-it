package flyweight

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// FlyweightFactoryImpl implements the FlyweightFactory interface
type FlyweightFactoryImpl struct {
	flyweights map[string]Flyweight
	cache      Cache
	logger     Logger
	metrics    Metrics
	config     FlyweightConfig
	mu         sync.RWMutex
	stats      *FactoryStats
}

// NewFlyweightFactory creates a new flyweight factory
func NewFlyweightFactory(cache Cache, logger Logger, metrics Metrics, config FlyweightConfig) *FlyweightFactoryImpl {
	factory := &FlyweightFactoryImpl{
		flyweights: make(map[string]Flyweight),
		cache:      cache,
		logger:     logger,
		metrics:    metrics,
		config:     config,
		stats:      &FactoryStats{},
	}

	// Start cleanup routine
	go factory.startCleanupRoutine()

	return factory
}

// GetFlyweight retrieves a flyweight by key
func (ff *FlyweightFactoryImpl) GetFlyweight(key string) (Flyweight, error) {
	start := time.Now()
	
	// Check cache first
	if cached, found := ff.cache.Get(key); found {
		ff.mu.RLock()
		ff.stats.CacheHits++
		ff.mu.RUnlock()
		
		flyweight := cached.(Flyweight)
		flyweight.UpdateLastAccessed()
		
		ff.metrics.RecordTiming("flyweight_cache_hit_duration", time.Since(start), map[string]string{"type": flyweight.GetType()})
		ff.metrics.IncrementCounter("flyweight_cache_hits", map[string]string{"type": flyweight.GetType()})
		
		return flyweight, nil
	}
	
	// Check in-memory storage
	ff.mu.RLock()
	flyweight, exists := ff.flyweights[key]
	ff.mu.RUnlock()
	
	if exists {
		flyweight.UpdateLastAccessed()
		
		// Update cache
		ff.cache.Set(key, flyweight, ff.config.TTL)
		
		ff.mu.Lock()
		ff.stats.CacheHits++
		ff.mu.Unlock()
		
		ff.metrics.RecordTiming("flyweight_memory_hit_duration", time.Since(start), map[string]string{"type": flyweight.GetType()})
		ff.metrics.IncrementCounter("flyweight_memory_hits", map[string]string{"type": flyweight.GetType()})
		
		return flyweight, nil
	}
	
	// Cache miss
	ff.mu.Lock()
	ff.stats.CacheMisses++
	ff.mu.Unlock()
	
	ff.metrics.RecordTiming("flyweight_miss_duration", time.Since(start), map[string]string{"key": key})
	ff.metrics.IncrementCounter("flyweight_misses", map[string]string{"key": key})
	
	return nil, fmt.Errorf("flyweight not found: %s", key)
}

// CreateFlyweight creates a new flyweight
func (ff *FlyweightFactoryImpl) CreateFlyweight(key string, intrinsicState map[string]interface{}) (Flyweight, error) {
	start := time.Now()
	
	// Determine flyweight type
	flyweightType, ok := intrinsicState["type"].(string)
	if !ok {
		return nil, fmt.Errorf("flyweight type is required")
	}
	
	var flyweight Flyweight
	var err error
	
	// Create appropriate flyweight based on type
	switch flyweightType {
	case "product":
		flyweight, err = ff.createProductFlyweight(key, intrinsicState)
	case "user":
		flyweight, err = ff.createUserFlyweight(key, intrinsicState)
	case "order":
		flyweight, err = ff.createOrderFlyweight(key, intrinsicState)
	case "notification":
		flyweight, err = ff.createNotificationFlyweight(key, intrinsicState)
	case "configuration":
		flyweight, err = ff.createConfigurationFlyweight(key, intrinsicState)
	default:
		return nil, fmt.Errorf("unsupported flyweight type: %s", flyweightType)
	}
	
	if err != nil {
		ff.logger.Error("Failed to create flyweight", "key", key, "type", flyweightType, "error", err)
		return nil, err
	}
	
	// Store in memory
	ff.mu.Lock()
	ff.flyweights[key] = flyweight
	ff.stats.TotalFlyweights++
	if flyweight.IsShared() {
		ff.stats.SharedFlyweights++
	} else {
		ff.stats.UnsharedFlyweights++
	}
	ff.mu.Unlock()
	
	// Store in cache
	ff.cache.Set(key, flyweight, ff.config.TTL)
	
	ff.logger.Info("Flyweight created", "key", key, "type", flyweightType, "shared", flyweight.IsShared())
	
	ff.metrics.RecordTiming("flyweight_creation_duration", time.Since(start), map[string]string{"type": flyweightType})
	ff.metrics.IncrementCounter("flyweight_created", map[string]string{"type": flyweightType})
	
	return flyweight, nil
}

// createProductFlyweight creates a product flyweight
func (ff *FlyweightFactoryImpl) createProductFlyweight(key string, intrinsicState map[string]interface{}) (*ProductFlyweight, error) {
	now := time.Now()
	
	flyweight := &ProductFlyweight{
		ID:           key,
		Type:         "product",
		Name:         getString(intrinsicState, "name"),
		Description:  getString(intrinsicState, "description"),
		Category:     getString(intrinsicState, "category"),
		Brand:        getString(intrinsicState, "brand"),
		BasePrice:    getFloat64(intrinsicState, "base_price"),
		Currency:     getString(intrinsicState, "currency"),
		Attributes:   getMap(intrinsicState, "attributes"),
		CreatedAt:    now,
		LastAccessed: now,
		IsShared:     true, // Products are typically shared
	}
	
	return flyweight, nil
}

// createUserFlyweight creates a user flyweight
func (ff *FlyweightFactoryImpl) createUserFlyweight(key string, intrinsicState map[string]interface{}) (*UserFlyweight, error) {
	now := time.Now()
	
	flyweight := &UserFlyweight{
		ID:           key,
		Type:         "user",
		Username:     getString(intrinsicState, "username"),
		Email:        getString(intrinsicState, "email"),
		Profile:      getMap(intrinsicState, "profile"),
		Preferences:  getMap(intrinsicState, "preferences"),
		CreatedAt:    now,
		LastAccessed: now,
		IsShared:     false, // Users are typically not shared
	}
	
	return flyweight, nil
}

// createOrderFlyweight creates an order flyweight
func (ff *FlyweightFactoryImpl) createOrderFlyweight(key string, intrinsicState map[string]interface{}) (*OrderFlyweight, error) {
	now := time.Now()
	
	flyweight := &OrderFlyweight{
		ID:           key,
		Type:         "order",
		Status:       getString(intrinsicState, "status"),
		Priority:     getString(intrinsicState, "priority"),
		Metadata:     getMap(intrinsicState, "metadata"),
		CreatedAt:    now,
		LastAccessed: now,
		IsShared:     false, // Orders are typically not shared
	}
	
	return flyweight, nil
}

// createNotificationFlyweight creates a notification flyweight
func (ff *FlyweightFactoryImpl) createNotificationFlyweight(key string, intrinsicState map[string]interface{}) (*NotificationFlyweight, error) {
	now := time.Now()
	
	flyweight := &NotificationFlyweight{
		ID:           key,
		Type:         "notification",
		Template:     getString(intrinsicState, "template"),
		Subject:      getString(intrinsicState, "subject"),
		Body:         getString(intrinsicState, "body"),
		Channels:     getStringSlice(intrinsicState, "channels"),
		Metadata:     getMap(intrinsicState, "metadata"),
		CreatedAt:    now,
		LastAccessed: now,
		IsShared:     true, // Notification templates are typically shared
	}
	
	return flyweight, nil
}

// createConfigurationFlyweight creates a configuration flyweight
func (ff *FlyweightFactoryImpl) createConfigurationFlyweight(key string, intrinsicState map[string]interface{}) (*ConfigurationFlyweight, error) {
	now := time.Now()
	
	flyweight := &ConfigurationFlyweight{
		ID:           key,
		Type:         "configuration",
		Key:          getString(intrinsicState, "key"),
		Value:        intrinsicState["value"],
		Description:  getString(intrinsicState, "description"),
		Category:     getString(intrinsicState, "category"),
		Metadata:     getMap(intrinsicState, "metadata"),
		CreatedAt:    now,
		LastAccessed: now,
		IsShared:     true, // Configurations are typically shared
	}
	
	return flyweight, nil
}

// GetFlyweightCount returns the total number of flyweights
func (ff *FlyweightFactoryImpl) GetFlyweightCount() int {
	ff.mu.RLock()
	defer ff.mu.RUnlock()
	return len(ff.flyweights)
}

// GetFlyweightTypes returns the types of flyweights
func (ff *FlyweightFactoryImpl) GetFlyweightTypes() []string {
	ff.mu.RLock()
	defer ff.mu.RUnlock()
	
	typeSet := make(map[string]bool)
	for _, flyweight := range ff.flyweights {
		typeSet[flyweight.GetType()] = true
	}
	
	types := make([]string, 0, len(typeSet))
	for flyweightType := range typeSet {
		types = append(types, flyweightType)
	}
	
	return types
}

// ClearUnusedFlyweights removes unused flyweights
func (ff *FlyweightFactoryImpl) ClearUnusedFlyweights() {
	ff.mu.Lock()
	defer ff.mu.Unlock()
	
	now := time.Now()
	cutoff := now.Add(-ff.config.TTL)
	
	removedCount := 0
	for key, flyweight := range ff.flyweights {
		if flyweight.GetLastAccessed().Before(cutoff) {
			delete(ff.flyweights, key)
			ff.cache.Delete(key)
			removedCount++
			
			if flyweight.IsShared() {
				ff.stats.SharedFlyweights--
			} else {
				ff.stats.UnsharedFlyweights--
			}
			ff.stats.TotalFlyweights--
		}
	}
	
	ff.stats.LastCleanup = now
	
	ff.logger.Info("Cleared unused flyweights", "removed_count", removedCount, "remaining_count", len(ff.flyweights))
	ff.metrics.IncrementCounter("flyweight_cleanup", map[string]string{"removed_count": fmt.Sprintf("%d", removedCount)})
}

// GetFactoryStats returns factory statistics
func (ff *FlyweightFactoryImpl) GetFactoryStats() FactoryStats {
	ff.mu.RLock()
	defer ff.mu.RUnlock()
	
	// Calculate hit rate
	totalHits := ff.stats.CacheHits + ff.stats.CacheMisses
	if totalHits > 0 {
		ff.stats.HitRate = float64(ff.stats.CacheHits) / float64(totalHits) * 100
	}
	
	// Calculate memory usage (approximate)
	ff.stats.MemoryUsage = int64(len(ff.flyweights) * 1024) // Rough estimate
	
	return *ff.stats
}

// startCleanupRoutine starts the cleanup routine
func (ff *FlyweightFactoryImpl) startCleanupRoutine() {
	ticker := time.NewTicker(ff.config.CleanupInterval)
	defer ticker.Stop()
	
	for range ticker.C {
		ff.ClearUnusedFlyweights()
	}
}

// Helper functions

func getString(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

func getFloat64(m map[string]interface{}, key string) float64 {
	if val, ok := m[key].(float64); ok {
		return val
	}
	return 0.0
}

func getMap(m map[string]interface{}, key string) map[string]interface{} {
	if val, ok := m[key].(map[string]interface{}); ok {
		return val
	}
	return make(map[string]interface{})
}

func getStringSlice(m map[string]interface{}, key string) []string {
	if val, ok := m[key].([]string); ok {
		return val
	}
	if val, ok := m[key].([]interface{}); ok {
		result := make([]string, 0, len(val))
		for _, v := range val {
			if str, ok := v.(string); ok {
				result = append(result, str)
			}
		}
		return result
	}
	return []string{}
}
