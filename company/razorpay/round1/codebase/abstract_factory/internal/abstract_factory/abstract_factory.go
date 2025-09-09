package abstract_factory

import (
	"context"
	"sync"
	"time"
)

// FactoryRegistryImpl implements the FactoryRegistry interface
type FactoryRegistryImpl struct {
	factories map[string]AbstractFactory
	mutex     sync.RWMutex
	createdAt time.Time
	updatedAt time.Time
	active    bool
}

// NewFactoryRegistry creates a new factory registry
func NewFactoryRegistry() *FactoryRegistryImpl {
	return &FactoryRegistryImpl{
		factories: make(map[string]AbstractFactory),
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// RegisterFactory registers a factory
func (r *FactoryRegistryImpl) RegisterFactory(factoryType string, factory AbstractFactory) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.active {
		return ErrRegistryInactive
	}

	if _, exists := r.factories[factoryType]; exists {
		return ErrFactoryAlreadyExists
	}

	r.factories[factoryType] = factory
	r.updatedAt = time.Now()

	return nil
}

// UnregisterFactory unregisters a factory
func (r *FactoryRegistryImpl) UnregisterFactory(factoryType string) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.active {
		return ErrRegistryInactive
	}

	if _, exists := r.factories[factoryType]; !exists {
		return ErrFactoryNotFound
	}

	delete(r.factories, factoryType)
	r.updatedAt = time.Now()

	return nil
}

// GetFactory retrieves a factory by type
func (r *FactoryRegistryImpl) GetFactory(factoryType string) (AbstractFactory, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	if !r.active {
		return nil, ErrRegistryInactive
	}

	factory, exists := r.factories[factoryType]
	if !exists {
		return nil, ErrFactoryNotFound
	}

	return factory, nil
}

// ListFactories returns a list of all registered factory types
func (r *FactoryRegistryImpl) ListFactories() []string {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	factoryTypes := make([]string, 0, len(r.factories))
	for factoryType := range r.factories {
		factoryTypes = append(factoryTypes, factoryType)
	}

	return factoryTypes
}

// GetFactoryStats returns statistics for a specific factory
func (r *FactoryRegistryImpl) GetFactoryStats(factoryType string) map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	factory, exists := r.factories[factoryType]
	if !exists {
		return map[string]interface{}{
			"error": "factory not found",
		}
	}

	return factory.GetFactoryInfo()
}

// GetAllFactoryStats returns statistics for all factories
func (r *FactoryRegistryImpl) GetAllFactoryStats() map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	stats := make(map[string]interface{})
	for factoryType, factory := range r.factories {
		stats[factoryType] = factory.GetFactoryInfo()
	}

	return stats
}

// IsFactoryRegistered checks if a factory is registered
func (r *FactoryRegistryImpl) IsFactoryRegistered(factoryType string) bool {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	_, exists := r.factories[factoryType]
	return exists
}

// GetRegistryStats returns registry statistics
func (r *FactoryRegistryImpl) GetRegistryStats() map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	return map[string]interface{}{
		"active":       r.active,
		"created_at":   r.createdAt,
		"updated_at":   r.updatedAt,
		"factories_count": len(r.factories),
		"factories":    r.ListFactories(),
	}
}

// Cleanup performs cleanup operations
func (r *FactoryRegistryImpl) Cleanup(ctx context.Context) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.active {
		return ErrRegistryInactive
	}

	// Cleanup all factories
	for _, factory := range r.factories {
		if err := factory.Cleanup(ctx); err != nil {
			// Log error but continue cleanup
			continue
		}
	}

	r.updatedAt = time.Now()
	return nil
}

// ProductRegistryImpl implements the ProductRegistry interface
type ProductRegistryImpl struct {
	products map[string]interface{}
	mutex    sync.RWMutex
	createdAt time.Time
	updatedAt time.Time
	active   bool
}

// NewProductRegistry creates a new product registry
func NewProductRegistry() *ProductRegistryImpl {
	return &ProductRegistryImpl{
		products: make(map[string]interface{}),
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:   true,
	}
}

// RegisterProduct registers a product
func (r *ProductRegistryImpl) RegisterProduct(productID string, product interface{}) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.active {
		return ErrRegistryInactive
	}

	if _, exists := r.products[productID]; exists {
		return ErrProductAlreadyExists
	}

	r.products[productID] = product
	r.updatedAt = time.Now()

	return nil
}

// UnregisterProduct unregisters a product
func (r *ProductRegistryImpl) UnregisterProduct(productID string) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.active {
		return ErrRegistryInactive
	}

	if _, exists := r.products[productID]; !exists {
		return ErrProductNotFound
	}

	delete(r.products, productID)
	r.updatedAt = time.Now()

	return nil
}

// GetProduct retrieves a product by ID
func (r *ProductRegistryImpl) GetProduct(productID string) (interface{}, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	if !r.active {
		return nil, ErrRegistryInactive
	}

	product, exists := r.products[productID]
	if !exists {
		return nil, ErrProductNotFound
	}

	return product, nil
}

// ListProducts returns a list of all registered product IDs
func (r *ProductRegistryImpl) ListProducts() []string {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	productIDs := make([]string, 0, len(r.products))
	for productID := range r.products {
		productIDs = append(productIDs, productID)
	}

	return productIDs
}

// GetProductStats returns statistics for a specific product
func (r *ProductRegistryImpl) GetProductStats(productID string) map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	product, exists := r.products[productID]
	if !exists {
		return map[string]interface{}{
			"error": "product not found",
		}
	}

	// Try to get stats from the product if it implements the interface
	if statsProvider, ok := product.(interface{ GetStats() map[string]interface{} }); ok {
		return statsProvider.GetStats()
	}

	return map[string]interface{}{
		"product_id": productID,
		"type":       "unknown",
	}
}

// GetAllProductStats returns statistics for all products
func (r *ProductRegistryImpl) GetAllProductStats() map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	stats := make(map[string]interface{})
	for productID, product := range r.products {
		if statsProvider, ok := product.(interface{ GetStats() map[string]interface{} }); ok {
			stats[productID] = statsProvider.GetStats()
		} else {
			stats[productID] = map[string]interface{}{
				"product_id": productID,
				"type":       "unknown",
			}
		}
	}

	return stats
}

// IsProductRegistered checks if a product is registered
func (r *ProductRegistryImpl) IsProductRegistered(productID string) bool {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	_, exists := r.products[productID]
	return exists
}

// GetRegistryStats returns registry statistics
func (r *ProductRegistryImpl) GetRegistryStats() map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	return map[string]interface{}{
		"active":       r.active,
		"created_at":   r.createdAt,
		"updated_at":   r.updatedAt,
		"products_count": len(r.products),
		"products":     r.ListProducts(),
	}
}

// Cleanup performs cleanup operations
func (r *ProductRegistryImpl) Cleanup(ctx context.Context) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.active {
		return ErrRegistryInactive
	}

	// Clear all products
	r.products = make(map[string]interface{})
	r.updatedAt = time.Now()

	return nil
}
