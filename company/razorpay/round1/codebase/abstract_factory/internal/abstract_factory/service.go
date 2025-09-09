package abstract_factory

import (
	"context"
	"sync"
	"time"
)

// FactoryServiceImpl implements the FactoryService interface
type FactoryServiceImpl struct {
	config          *ServiceConfig
	factoryRegistry FactoryRegistry
	productRegistry ProductRegistry
	createdAt       time.Time
	updatedAt       time.Time
	active          bool
	mutex           sync.RWMutex
}

// NewFactoryService creates a new factory service
func NewFactoryService(config *ServiceConfig) *FactoryServiceImpl {
	return &FactoryServiceImpl{
		config:          config,
		factoryRegistry: NewFactoryRegistry(),
		productRegistry: NewProductRegistry(),
		createdAt:       time.Now(),
		updatedAt:       time.Now(),
		active:          true,
	}
}

// CreateFactory creates a new factory
func (s *FactoryServiceImpl) CreateFactory(factoryType string, config map[string]interface{}) (AbstractFactory, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	// Check if factory type is supported
	if !s.isFactoryTypeSupported(factoryType) {
		return nil, ErrInvalidFactoryType
	}

	// Check if factory already exists
	if s.factoryRegistry.IsFactoryRegistered(factoryType) {
		return nil, ErrFactoryAlreadyExists
	}

	// Create factory configuration
	factoryConfig := &FactoryConfig{
		Type:        factoryType,
		Name:        s.getStringFromConfig(config, "name", factoryType),
		Description: s.getStringFromConfig(config, "description", "Factory for "+factoryType),
		Version:     s.getStringFromConfig(config, "version", "1.0.0"),
		Metadata:    s.getMapFromConfig(config, "metadata"),
		Active:      true,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// Create factory based on type
	var factory AbstractFactory
	switch factoryType {
	case "ConcreteFactory1":
		factory = NewConcreteFactory1(factoryConfig)
	case "ConcreteFactory2":
		factory = NewConcreteFactory2(factoryConfig)
	default:
		return nil, ErrInvalidFactoryType
	}

	// Register factory
	if err := s.factoryRegistry.RegisterFactory(factoryType, factory); err != nil {
		return nil, err
	}

	s.updatedAt = time.Now()
	return factory, nil
}

// DestroyFactory destroys a factory
func (s *FactoryServiceImpl) DestroyFactory(factoryType string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	// Get factory
	factory, err := s.factoryRegistry.GetFactory(factoryType)
	if err != nil {
		return err
	}

	// Cleanup factory
	if err := factory.Cleanup(context.Background()); err != nil {
		// Log error but continue
	}

	// Unregister factory
	if err := s.factoryRegistry.UnregisterFactory(factoryType); err != nil {
		return err
	}

	s.updatedAt = time.Now()
	return nil
}

// GetFactory retrieves a factory by type
func (s *FactoryServiceImpl) GetFactory(factoryType string) (AbstractFactory, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	return s.factoryRegistry.GetFactory(factoryType)
}

// ListFactories returns a list of all factory types
func (s *FactoryServiceImpl) ListFactories() []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return []string{}
	}

	return s.factoryRegistry.ListFactories()
}

// GetFactoryStats returns statistics for a specific factory
func (s *FactoryServiceImpl) GetFactoryStats(factoryType string) map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.factoryRegistry.GetFactoryStats(factoryType)
}

// GetAllFactoryStats returns statistics for all factories
func (s *FactoryServiceImpl) GetAllFactoryStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.factoryRegistry.GetAllFactoryStats()
}

// IsFactoryActive checks if a factory is active
func (s *FactoryServiceImpl) IsFactoryActive(factoryType string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return false
	}

	factory, err := s.factoryRegistry.GetFactory(factoryType)
	if err != nil {
		return false
	}

	return factory.IsActive()
}

// SetFactoryActive sets the active status of a factory
func (s *FactoryServiceImpl) SetFactoryActive(factoryType string, active bool) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	factory, err := s.factoryRegistry.GetFactory(factoryType)
	if err != nil {
		return err
	}

	factory.SetActive(active)
	s.updatedAt = time.Now()

	return nil
}

// GetServiceStats returns service statistics
func (s *FactoryServiceImpl) GetServiceStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return map[string]interface{}{
		"service_name":    s.config.Name,
		"version":         s.config.Version,
		"active":          s.active,
		"created_at":      s.createdAt,
		"updated_at":      s.updatedAt,
		"factories_count": len(s.factoryRegistry.ListFactories()),
		"products_count":  len(s.productRegistry.ListProducts()),
		"metadata":        s.config.Metadata,
	}
}

// GetHealthStatus returns the health status of the service
func (s *FactoryServiceImpl) GetHealthStatus() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"factory_service": map[string]interface{}{
				"status": "healthy",
				"active": s.active,
			},
			"factory_registry": map[string]interface{}{
				"status": "healthy",
				"active": s.factoryRegistry != nil,
			},
			"product_registry": map[string]interface{}{
				"status": "healthy",
				"active": s.productRegistry != nil,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !s.active {
		healthStatus["checks"].(map[string]interface{})["factory_service"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["factory_service"].(map[string]interface{})["message"] = "Factory service is inactive"
	}

	if s.factoryRegistry == nil {
		healthStatus["checks"].(map[string]interface{})["factory_registry"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["factory_registry"].(map[string]interface{})["message"] = "Factory registry is not available"
	}

	if s.productRegistry == nil {
		healthStatus["checks"].(map[string]interface{})["product_registry"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["product_registry"].(map[string]interface{})["message"] = "Product registry is not available"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (s *FactoryServiceImpl) Cleanup(ctx context.Context) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	// Cleanup factory registry
	if s.factoryRegistry != nil {
		if err := s.factoryRegistry.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	// Cleanup product registry
	if s.productRegistry != nil {
		if err := s.productRegistry.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	s.updatedAt = time.Now()
	return nil
}

// ProductServiceImpl implements the ProductService interface
type ProductServiceImpl struct {
	config          *ServiceConfig
	factoryRegistry FactoryRegistry
	productRegistry ProductRegistry
	createdAt       time.Time
	updatedAt       time.Time
	active          bool
	mutex           sync.RWMutex
}

// NewProductService creates a new product service
func NewProductService(config *ServiceConfig) *ProductServiceImpl {
	return &ProductServiceImpl{
		config:          config,
		factoryRegistry: NewFactoryRegistry(),
		productRegistry: NewProductRegistry(),
		createdAt:       time.Now(),
		updatedAt:       time.Now(),
		active:          true,
	}
}

// CreateProduct creates a new product using a factory
func (s *ProductServiceImpl) CreateProduct(factoryType string, productType string, config map[string]interface{}) (interface{}, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	// Check if product type is supported
	if !s.isProductTypeSupported(productType) {
		return nil, ErrInvalidProductType
	}

	// Get factory
	factory, err := s.factoryRegistry.GetFactory(factoryType)
	if err != nil {
		return nil, err
	}

	if !factory.IsActive() {
		return nil, ErrFactoryInactive
	}

	// Create product based on type
	var product interface{}
	switch productType {
	case "ProductA":
		product = factory.CreateProductA()
	case "ProductB":
		product = factory.CreateProductB()
	case "ProductC":
		product = factory.CreateProductC()
	default:
		return nil, ErrInvalidProductType
	}

	// Get product ID
	productID := s.getProductID(product)
	if productID == "" {
		return nil, ErrInvalidConfiguration
	}

	// Register product
	if err := s.productRegistry.RegisterProduct(productID, product); err != nil {
		return nil, err
	}

	s.updatedAt = time.Now()
	return product, nil
}

// DestroyProduct destroys a product
func (s *ProductServiceImpl) DestroyProduct(productID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	// Unregister product
	if err := s.productRegistry.UnregisterProduct(productID); err != nil {
		return err
	}

	s.updatedAt = time.Now()
	return nil
}

// GetProduct retrieves a product by ID
func (s *ProductServiceImpl) GetProduct(productID string) (interface{}, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	return s.productRegistry.GetProduct(productID)
}

// ListProducts returns a list of all product IDs
func (s *ProductServiceImpl) ListProducts() []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return []string{}
	}

	return s.productRegistry.ListProducts()
}

// GetProductStats returns statistics for a specific product
func (s *ProductServiceImpl) GetProductStats(productID string) map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.productRegistry.GetProductStats(productID)
}

// GetAllProductStats returns statistics for all products
func (s *ProductServiceImpl) GetAllProductStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.productRegistry.GetAllProductStats()
}

// IsProductActive checks if a product is active
func (s *ProductServiceImpl) IsProductActive(productID string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return false
	}

	product, err := s.productRegistry.GetProduct(productID)
	if err != nil {
		return false
	}

	if activeProvider, ok := product.(interface{ IsActive() bool }); ok {
		return activeProvider.IsActive()
	}

	return false
}

// SetProductActive sets the active status of a product
func (s *ProductServiceImpl) SetProductActive(productID string, active bool) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	product, err := s.productRegistry.GetProduct(productID)
	if err != nil {
		return err
	}

	if activeProvider, ok := product.(interface{ SetActive(bool) }); ok {
		activeProvider.SetActive(active)
	}

	s.updatedAt = time.Now()
	return nil
}

// GetServiceStats returns service statistics
func (s *ProductServiceImpl) GetServiceStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return map[string]interface{}{
		"service_name":    s.config.Name,
		"version":         s.config.Version,
		"active":          s.active,
		"created_at":      s.createdAt,
		"updated_at":      s.updatedAt,
		"factories_count": len(s.factoryRegistry.ListFactories()),
		"products_count":  len(s.productRegistry.ListProducts()),
		"metadata":        s.config.Metadata,
	}
}

// GetHealthStatus returns the health status of the service
func (s *ProductServiceImpl) GetHealthStatus() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"product_service": map[string]interface{}{
				"status": "healthy",
				"active": s.active,
			},
			"factory_registry": map[string]interface{}{
				"status": "healthy",
				"active": s.factoryRegistry != nil,
			},
			"product_registry": map[string]interface{}{
				"status": "healthy",
				"active": s.productRegistry != nil,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !s.active {
		healthStatus["checks"].(map[string]interface{})["product_service"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["product_service"].(map[string]interface{})["message"] = "Product service is inactive"
	}

	if s.factoryRegistry == nil {
		healthStatus["checks"].(map[string]interface{})["factory_registry"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["factory_registry"].(map[string]interface{})["message"] = "Factory registry is not available"
	}

	if s.productRegistry == nil {
		healthStatus["checks"].(map[string]interface{})["product_registry"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["product_registry"].(map[string]interface{})["message"] = "Product registry is not available"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (s *ProductServiceImpl) Cleanup(ctx context.Context) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	// Cleanup factory registry
	if s.factoryRegistry != nil {
		if err := s.factoryRegistry.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	// Cleanup product registry
	if s.productRegistry != nil {
		if err := s.productRegistry.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	s.updatedAt = time.Now()
	return nil
}

// Helper methods

func (s *FactoryServiceImpl) isFactoryTypeSupported(factoryType string) bool {
	for _, supportedType := range s.config.SupportedFactoryTypes {
		if supportedType == factoryType {
			return true
		}
	}
	return false
}

func (s *ProductServiceImpl) isProductTypeSupported(productType string) bool {
	for _, supportedType := range s.config.SupportedProductTypes {
		if supportedType == productType {
			return true
		}
	}
	return false
}

func (s *FactoryServiceImpl) getStringFromConfig(config map[string]interface{}, key, defaultValue string) string {
	if value, exists := config[key]; exists {
		if str, ok := value.(string); ok {
			return str
		}
	}
	return defaultValue
}

func (s *FactoryServiceImpl) getMapFromConfig(config map[string]interface{}, key string) map[string]interface{} {
	if value, exists := config[key]; exists {
		if m, ok := value.(map[string]interface{}); ok {
			return m
		}
	}
	return make(map[string]interface{})
}

func (s *ProductServiceImpl) getProductID(product interface{}) string {
	if idProvider, ok := product.(interface{ GetID() string }); ok {
		return idProvider.GetID()
	}
	return ""
}
