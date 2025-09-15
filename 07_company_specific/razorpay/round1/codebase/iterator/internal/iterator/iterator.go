package iterator

import (
	"context"
	"sync"
	"time"
)

// IteratorManager manages all iterators
type IteratorManager struct {
	iterators map[string]Iterator
	mutex     sync.RWMutex
	config    *IteratorConfig
}

// NewIteratorManager creates a new iterator manager
func NewIteratorManager(config *IteratorConfig) *IteratorManager {
	return &IteratorManager{
		iterators: make(map[string]Iterator),
		config:    config,
	}
}

// CreateIterator creates a new iterator
func (im *IteratorManager) CreateIterator(name string, iterator Iterator) error {
	im.mutex.Lock()
	defer im.mutex.Unlock()
	
	if len(im.iterators) >= im.config.MaxIterators {
		return ErrMaxIteratorsReached
	}
	
	im.iterators[name] = iterator
	return nil
}

// GetIterator retrieves an iterator by name
func (im *IteratorManager) GetIterator(name string) (Iterator, error) {
	im.mutex.RLock()
	defer im.mutex.RUnlock()
	
	iterator, exists := im.iterators[name]
	if !exists {
		return nil, ErrIteratorNotFound
	}
	
	return iterator, nil
}

// RemoveIterator removes an iterator
func (im *IteratorManager) RemoveIterator(name string) error {
	im.mutex.Lock()
	defer im.mutex.Unlock()
	
	iterator, exists := im.iterators[name]
	if !exists {
		return ErrIteratorNotFound
	}
	
	iterator.Close()
	delete(im.iterators, name)
	return nil
}

// ListIterators returns all iterator names
func (im *IteratorManager) ListIterators() []string {
	im.mutex.RLock()
	defer im.mutex.RUnlock()
	
	names := make([]string, 0, len(im.iterators))
	for name := range im.iterators {
		names = append(names, name)
	}
	
	return names
}

// GetIteratorStats returns statistics for an iterator
func (im *IteratorManager) GetIteratorStats(name string) (*IteratorStatistics, error) {
	im.mutex.RLock()
	defer im.mutex.RUnlock()
	
	iterator, exists := im.iterators[name]
	if !exists {
		return nil, ErrIteratorNotFound
	}
	
	stats := &IteratorStatistics{
		TotalItems:     int64(iterator.GetSize()),
		ProcessedItems: int64(iterator.GetIndex()),
		LastAccess:     time.Now(),
		CreatedAt:      time.Now(),
	}
	
	return stats, nil
}

// CloseAll closes all iterators
func (im *IteratorManager) CloseAll() {
	im.mutex.Lock()
	defer im.mutex.Unlock()
	
	for _, iterator := range im.iterators {
		iterator.Close()
	}
	
	im.iterators = make(map[string]Iterator)
}

// IteratorService provides iterator operations
type IteratorService struct {
	manager *IteratorManager
	config  *IteratorConfig
}

// NewIteratorService creates a new iterator service
func NewIteratorService(config *IteratorConfig) *IteratorService {
	return &IteratorService{
		manager: NewIteratorManager(config),
		config:  config,
	}
}

// CreateSliceIterator creates a slice iterator
func (is *IteratorService) CreateSliceIterator(name string, items []interface{}) error {
	iterator := NewSliceIterator(items)
	return is.manager.CreateIterator(name, iterator)
}

// CreateMapIterator creates a map iterator
func (is *IteratorService) CreateMapIterator(name string, items map[string]interface{}) error {
	iterator := NewMapIterator(items)
	return is.manager.CreateIterator(name, iterator)
}

// CreateChannelIterator creates a channel iterator
func (is *IteratorService) CreateChannelIterator(name string, channel <-chan interface{}) error {
	iterator := NewChannelIterator(channel)
	return is.manager.CreateIterator(name, iterator)
}

// CreateDatabaseIterator creates a database iterator
func (is *IteratorService) CreateDatabaseIterator(name string, query interface{}, results []interface{}) error {
	iterator := NewDatabaseIterator(query, results)
	return is.manager.CreateIterator(name, iterator)
}

// CreateFileIterator creates a file iterator
func (is *IteratorService) CreateFileIterator(name string, filePath string, lines []string) error {
	iterator := NewFileIterator(filePath, lines)
	return is.manager.CreateIterator(name, iterator)
}

// CreateFilteredIterator creates a filtered iterator
func (is *IteratorService) CreateFilteredIterator(name string, baseIteratorName string, filter Filter) error {
	baseIterator, err := is.manager.GetIterator(baseIteratorName)
	if err != nil {
		return err
	}
	
	iterator := NewFilteredIterator(baseIterator, filter)
	return is.manager.CreateIterator(name, iterator)
}

// CreateSortedIterator creates a sorted iterator
func (is *IteratorService) CreateSortedIterator(name string, baseIteratorName string, sorter Sorter) error {
	baseIterator, err := is.manager.GetIterator(baseIteratorName)
	if err != nil {
		return err
	}
	
	iterator := NewSortedIterator(baseIterator, sorter)
	return is.manager.CreateIterator(name, iterator)
}

// CreateTransformedIterator creates a transformed iterator
func (is *IteratorService) CreateTransformedIterator(name string, baseIteratorName string, transformer Transformer) error {
	baseIterator, err := is.manager.GetIterator(baseIteratorName)
	if err != nil {
		return err
	}
	
	iterator := NewTransformedIterator(baseIterator, transformer)
	return is.manager.CreateIterator(name, iterator)
}

// GetIterator retrieves an iterator
func (is *IteratorService) GetIterator(name string) (Iterator, error) {
	return is.manager.GetIterator(name)
}

// RemoveIterator removes an iterator
func (is *IteratorService) RemoveIterator(name string) error {
	return is.manager.RemoveIterator(name)
}

// ListIterators returns all iterator names
func (is *IteratorService) ListIterators() []string {
	return is.manager.ListIterators()
}

// GetIteratorStats returns iterator statistics
func (is *IteratorService) GetIteratorStats(name string) (*IteratorStatistics, error) {
	return is.manager.GetIteratorStats(name)
}

// CloseAll closes all iterators
func (is *IteratorService) CloseAll() {
	is.manager.CloseAll()
}

// IteratorCollection manages collections
type IteratorCollection struct {
	collections map[string]Collection
	mutex       sync.RWMutex
}

// NewIteratorCollection creates a new iterator collection
func NewIteratorCollection() *IteratorCollection {
	return &IteratorCollection{
		collections: make(map[string]Collection),
	}
}

// AddCollection adds a collection
func (ic *IteratorCollection) AddCollection(name string, collection Collection) {
	ic.mutex.Lock()
	defer ic.mutex.Unlock()
	
	ic.collections[name] = collection
}

// GetCollection retrieves a collection
func (ic *IteratorCollection) GetCollection(name string) (Collection, error) {
	ic.mutex.RLock()
	defer ic.mutex.RUnlock()
	
	collection, exists := ic.collections[name]
	if !exists {
		return nil, ErrCollectionNotFound
	}
	
	return collection, nil
}

// RemoveCollection removes a collection
func (ic *IteratorCollection) RemoveCollection(name string) error {
	ic.mutex.Lock()
	defer ic.mutex.Unlock()
	
	_, exists := ic.collections[name]
	if !exists {
		return ErrCollectionNotFound
	}
	
	delete(ic.collections, name)
	return nil
}

// ListCollections returns all collection names
func (ic *IteratorCollection) ListCollections() []string {
	ic.mutex.RLock()
	defer ic.mutex.RUnlock()
	
	names := make([]string, 0, len(ic.collections))
	for name := range ic.collections {
		names = append(names, name)
	}
	
	return names
}

// CreateIteratorFromCollection creates an iterator from a collection
func (ic *IteratorCollection) CreateIteratorFromCollection(name string) (Iterator, error) {
	ic.mutex.RLock()
	defer ic.mutex.RUnlock()
	
	collection, exists := ic.collections[name]
	if !exists {
		return nil, ErrCollectionNotFound
	}
	
	return collection.CreateIterator(), nil
}

// IteratorCache provides caching for iterators
type IteratorCache struct {
	cache map[string]interface{}
	mutex sync.RWMutex
	ttl   time.Duration
}

// NewIteratorCache creates a new iterator cache
func NewIteratorCache(ttl time.Duration) *IteratorCache {
	return &IteratorCache{
		cache: make(map[string]interface{}),
		ttl:   ttl,
	}
}

// Set sets a value in the cache
func (ic *IteratorCache) Set(key string, value interface{}) {
	ic.mutex.Lock()
	defer ic.mutex.Unlock()
	
	ic.cache[key] = value
}

// Get gets a value from the cache
func (ic *IteratorCache) Get(key string) (interface{}, bool) {
	ic.mutex.RLock()
	defer ic.mutex.RUnlock()
	
	value, exists := ic.cache[key]
	return value, exists
}

// Delete deletes a value from the cache
func (ic *IteratorCache) Delete(key string) {
	ic.mutex.Lock()
	defer ic.mutex.Unlock()
	
	delete(ic.cache, key)
}

// Clear clears the cache
func (ic *IteratorCache) Clear() {
	ic.mutex.Lock()
	defer ic.mutex.Unlock()
	
	ic.cache = make(map[string]interface{})
}

// Size returns the cache size
func (ic *IteratorCache) Size() int {
	ic.mutex.RLock()
	defer ic.mutex.RUnlock()
	
	return len(ic.cache)
}

// IteratorMetrics provides metrics for iterators
type IteratorMetrics struct {
	TotalIterators    int64
	ActiveIterators   int64
	TotalItems        int64
	ProcessedItems    int64
	AverageLatency    float64
	MaxLatency        float64
	MinLatency        float64
	LastUpdate        time.Time
}

// GetMetrics returns current metrics
func (im *IteratorMetrics) GetMetrics() *IteratorMetrics {
	return im
}

// UpdateMetrics updates the metrics
func (im *IteratorMetrics) UpdateMetrics(iterator Iterator) {
	im.TotalIterators++
	im.ActiveIterators++
	im.TotalItems += int64(iterator.GetSize())
	im.ProcessedItems += int64(iterator.GetIndex())
	im.LastUpdate = time.Now()
}

// IteratorValidator validates iterators
type IteratorValidator struct {
	config *IteratorConfig
}

// NewIteratorValidator creates a new iterator validator
func NewIteratorValidator(config *IteratorConfig) *IteratorValidator {
	return &IteratorValidator{
		config: config,
	}
}

// ValidateIterator validates an iterator
func (iv *IteratorValidator) ValidateIterator(iterator Iterator) error {
	if iterator == nil {
		return ErrInvalidIterator
	}
	
	if !iterator.IsValid() {
		return ErrIteratorNotValid
	}
	
	return nil
}

// ValidateCollection validates a collection
func (iv *IteratorValidator) ValidateCollection(collection Collection) error {
	if collection == nil {
		return ErrInvalidCollection
	}
	
	if collection.IsEmpty() {
		return ErrEmptyCollection
	}
	
	return nil
}

// IteratorContext provides context for iterator operations
type IteratorContext struct {
	context.Context
	IteratorName string
	StartTime    time.Time
	Timeout      time.Duration
}

// NewIteratorContext creates a new iterator context
func NewIteratorContext(ctx context.Context, iteratorName string, timeout time.Duration) *IteratorContext {
	return &IteratorContext{
		Context:      ctx,
		IteratorName: iteratorName,
		StartTime:    time.Now(),
		Timeout:      timeout,
	}
}

// IsExpired checks if the context is expired
func (ic *IteratorContext) IsExpired() bool {
	return time.Since(ic.StartTime) > ic.Timeout
}

// GetElapsedTime returns the elapsed time
func (ic *IteratorContext) GetElapsedTime() time.Duration {
	return time.Since(ic.StartTime)
}

// IteratorPool manages a pool of iterators
type IteratorPool struct {
	pool   chan Iterator
	mutex  sync.RWMutex
	config *IteratorConfig
}

// NewIteratorPool creates a new iterator pool
func NewIteratorPool(config *IteratorConfig) *IteratorPool {
	return &IteratorPool{
		pool:   make(chan Iterator, config.MaxIterators),
		config: config,
	}
}

// Get gets an iterator from the pool
func (ip *IteratorPool) Get() (Iterator, error) {
	select {
	case iterator := <-ip.pool:
		return iterator, nil
	default:
		return nil, ErrPoolEmpty
	}
}

// Put puts an iterator back into the pool
func (ip *IteratorPool) Put(iterator Iterator) error {
	select {
	case ip.pool <- iterator:
		return nil
	default:
		return ErrPoolFull
	}
}

// Size returns the pool size
func (ip *IteratorPool) Size() int {
	return len(ip.pool)
}

// Close closes the pool
func (ip *IteratorPool) Close() {
	close(ip.pool)
}
