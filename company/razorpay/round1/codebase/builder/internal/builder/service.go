package builder

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// BuilderServiceImpl implements the BuilderService interface
type BuilderServiceImpl struct {
	config          *ServiceConfig
	builderRegistry BuilderRegistry
	productRegistry ProductRegistry
	createdAt       time.Time
	updatedAt       time.Time
	active          bool
	mutex           sync.RWMutex
}

// NewBuilderService creates a new builder service
func NewBuilderService(config *ServiceConfig) *BuilderServiceImpl {
	return &BuilderServiceImpl{
		config:          config,
		builderRegistry: NewBuilderRegistry(),
		productRegistry: NewProductRegistry(),
		createdAt:       time.Now(),
		updatedAt:       time.Now(),
		active:          true,
	}
}

// CreateBuilder creates a new builder
func (s *BuilderServiceImpl) CreateBuilder(builderType string, config map[string]interface{}) (Builder, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	// Check if builder type is supported
	if !s.isBuilderTypeSupported(builderType) {
		return nil, ErrInvalidBuilderType
	}

	// Check if builder already exists
	if s.builderRegistry.IsBuilderRegistered(builderType) {
		return nil, ErrBuilderAlreadyExists
	}

	// Create builder configuration
	builderConfig := &BuilderConfig{
		Type:        builderType,
		Name:        s.getStringFromConfig(config, "name", builderType),
		Description: s.getStringFromConfig(config, "description", "Builder for "+builderType),
		Version:     s.getStringFromConfig(config, "version", "1.0.0"),
		Metadata:    s.getMapFromConfig(config, "metadata"),
		Active:      true,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// Create builder based on type
	var builder Builder
	switch builderType {
	case "ConcreteBuilder":
		builder = NewConcreteBuilder(builderConfig)
	default:
		return nil, ErrInvalidBuilderType
	}

	// Register builder
	if err := s.builderRegistry.RegisterBuilder(builderType, builder); err != nil {
		return nil, err
	}

	s.updatedAt = time.Now()
	return builder, nil
}

// DestroyBuilder destroys a builder
func (s *BuilderServiceImpl) DestroyBuilder(builderType string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	// Get builder
	builder, err := s.builderRegistry.GetBuilder(builderType)
	if err != nil {
		return err
	}

	// Cleanup builder
	if err := builder.Cleanup(context.Background()); err != nil {
		// Log error but continue
	}

	// Unregister builder
	if err := s.builderRegistry.UnregisterBuilder(builderType); err != nil {
		return err
	}

	s.updatedAt = time.Now()
	return nil
}

// GetBuilder retrieves a builder by type
func (s *BuilderServiceImpl) GetBuilder(builderType string) (Builder, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	return s.builderRegistry.GetBuilder(builderType)
}

// ListBuilders returns a list of all builder types
func (s *BuilderServiceImpl) ListBuilders() []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return []string{}
	}

	return s.builderRegistry.ListBuilders()
}

// GetBuilderStats returns statistics for a specific builder
func (s *BuilderServiceImpl) GetBuilderStats(builderType string) map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.builderRegistry.GetBuilderStats(builderType)
}

// GetAllBuilderStats returns statistics for all builders
func (s *BuilderServiceImpl) GetAllBuilderStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	return s.builderRegistry.GetAllBuilderStats()
}

// IsBuilderActive checks if a builder is active
func (s *BuilderServiceImpl) IsBuilderActive(builderType string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return false
	}

	builder, err := s.builderRegistry.GetBuilder(builderType)
	if err != nil {
		return false
	}

	return builder.IsActive()
}

// SetBuilderActive sets the active status of a builder
func (s *BuilderServiceImpl) SetBuilderActive(builderType string, active bool) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	builder, err := s.builderRegistry.GetBuilder(builderType)
	if err != nil {
		return err
	}

	builder.SetBuilderActive(active)
	s.updatedAt = time.Now()

	return nil
}

// GetServiceStats returns service statistics
func (s *BuilderServiceImpl) GetServiceStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return map[string]interface{}{
		"service_name":   s.config.Name,
		"version":        s.config.Version,
		"active":         s.active,
		"created_at":     s.createdAt,
		"updated_at":     s.updatedAt,
		"builders_count": len(s.builderRegistry.ListBuilders()),
		"products_count": len(s.productRegistry.ListProducts()),
		"metadata":       s.config.Metadata,
	}
}

// GetHealthStatus returns the health status of the service
func (s *BuilderServiceImpl) GetHealthStatus() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"builder_service": map[string]interface{}{
				"status": "healthy",
				"active": s.active,
			},
			"builder_registry": map[string]interface{}{
				"status": "healthy",
				"active": s.builderRegistry != nil,
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
		healthStatus["checks"].(map[string]interface{})["builder_service"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["builder_service"].(map[string]interface{})["message"] = "Builder service is inactive"
	}

	if s.builderRegistry == nil {
		healthStatus["checks"].(map[string]interface{})["builder_registry"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["builder_registry"].(map[string]interface{})["message"] = "Builder registry is not available"
	}

	if s.productRegistry == nil {
		healthStatus["checks"].(map[string]interface{})["product_registry"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["product_registry"].(map[string]interface{})["message"] = "Product registry is not available"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (s *BuilderServiceImpl) Cleanup(ctx context.Context) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	// Cleanup builder registry
	if s.builderRegistry != nil {
		if err := s.builderRegistry.Cleanup(ctx); err != nil {
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
	builderRegistry BuilderRegistry
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
		builderRegistry: NewBuilderRegistry(),
		productRegistry: NewProductRegistry(),
		createdAt:       time.Now(),
		updatedAt:       time.Now(),
		active:          true,
	}
}

// CreateProduct creates a new product using a builder
func (s *ProductServiceImpl) CreateProduct(builderType string, config map[string]interface{}) (Product, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	// Get builder
	builder, err := s.builderRegistry.GetBuilder(builderType)
	if err != nil {
		return nil, err
	}

	if !builder.IsActive() {
		return nil, ErrBuilderInactive
	}

	// Reset builder
	builder.Reset()

	// Apply configuration
	if name, ok := config["name"].(string); ok {
		builder.SetName(name)
	}
	if description, ok := config["description"].(string); ok {
		builder.SetDescription(description)
	}
	if price, ok := config["price"].(float64); ok {
		builder.SetPrice(price)
	}
	if category, ok := config["category"].(string); ok {
		builder.SetCategory(category)
	}
	if tags, ok := config["tags"].([]string); ok {
		builder.SetTags(tags)
	}
	if metadata, ok := config["metadata"].(map[string]interface{}); ok {
		builder.SetMetadata(metadata)
	}
	if active, ok := config["active"].(bool); ok {
		builder.SetActive(active)
	}

	// Build the product
	product, err := builder.Build()
	if err != nil {
		return nil, err
	}

	// Register product
	if err := s.productRegistry.RegisterProduct(product.GetID(), product); err != nil {
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
func (s *ProductServiceImpl) GetProduct(productID string) (Product, error) {
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

	return product.IsActive()
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

	product.SetActive(active)
	s.updatedAt = time.Now()

	return nil
}

// GetServiceStats returns service statistics
func (s *ProductServiceImpl) GetServiceStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return map[string]interface{}{
		"service_name":   s.config.Name,
		"version":        s.config.Version,
		"active":         s.active,
		"created_at":     s.createdAt,
		"updated_at":     s.updatedAt,
		"builders_count": len(s.builderRegistry.ListBuilders()),
		"products_count": len(s.productRegistry.ListProducts()),
		"metadata":       s.config.Metadata,
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
			"builder_registry": map[string]interface{}{
				"status": "healthy",
				"active": s.builderRegistry != nil,
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

	if s.builderRegistry == nil {
		healthStatus["checks"].(map[string]interface{})["builder_registry"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["builder_registry"].(map[string]interface{})["message"] = "Builder registry is not available"
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

	// Cleanup builder registry
	if s.builderRegistry != nil {
		if err := s.builderRegistry.Cleanup(ctx); err != nil {
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

// DirectorServiceImpl implements the DirectorService interface
type DirectorServiceImpl struct {
	config    *ServiceConfig
	directors map[string]Director
	createdAt time.Time
	updatedAt time.Time
	active    bool
	mutex     sync.RWMutex
}

// NewDirectorService creates a new director service
func NewDirectorService(config *ServiceConfig) *DirectorServiceImpl {
	return &DirectorServiceImpl{
		config:    config,
		directors: make(map[string]Director),
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// CreateDirector creates a new director
func (s *DirectorServiceImpl) CreateDirector(config map[string]interface{}) (Director, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	// Create director configuration
	directorConfig := &DirectorConfig{
		ID:          s.getStringFromConfig(config, "id", fmt.Sprintf("director-%d", time.Now().UnixNano())),
		Name:        s.getStringFromConfig(config, "name", "Default Director"),
		Description: s.getStringFromConfig(config, "description", "Default director for building products"),
		Version:     s.getStringFromConfig(config, "version", "1.0.0"),
		Metadata:    s.getMapFromConfig(config, "metadata"),
		Active:      true,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// Create director
	director := NewDirector(directorConfig)

	// Register director
	s.directors[directorConfig.ID] = director
	s.updatedAt = time.Now()

	return director, nil
}

// DestroyDirector destroys a director
func (s *DirectorServiceImpl) DestroyDirector(directorID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	director, exists := s.directors[directorID]
	if !exists {
		return ErrDirectorNotFound
	}

	// Cleanup director
	if err := director.Cleanup(context.Background()); err != nil {
		// Log error but continue
	}

	// Remove director
	delete(s.directors, directorID)
	s.updatedAt = time.Now()

	return nil
}

// GetDirector retrieves a director by ID
func (s *DirectorServiceImpl) GetDirector(directorID string) (Director, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return nil, ErrServiceInactive
	}

	director, exists := s.directors[directorID]
	if !exists {
		return nil, ErrDirectorNotFound
	}

	return director, nil
}

// ListDirectors returns a list of all director IDs
func (s *DirectorServiceImpl) ListDirectors() []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return []string{}
	}

	directorIDs := make([]string, 0, len(s.directors))
	for directorID := range s.directors {
		directorIDs = append(directorIDs, directorID)
	}

	return directorIDs
}

// GetDirectorStats returns statistics for a specific director
func (s *DirectorServiceImpl) GetDirectorStats(directorID string) map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	director, exists := s.directors[directorID]
	if !exists {
		return map[string]interface{}{
			"error": "director not found",
		}
	}

	return director.GetDirectorInfo()
}

// GetAllDirectorStats returns statistics for all directors
func (s *DirectorServiceImpl) GetAllDirectorStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return map[string]interface{}{
			"error": "service is inactive",
		}
	}

	stats := make(map[string]interface{})
	for directorID, director := range s.directors {
		stats[directorID] = director.GetDirectorInfo()
	}

	return stats
}

// IsDirectorActive checks if a director is active
func (s *DirectorServiceImpl) IsDirectorActive(directorID string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if !s.active {
		return false
	}

	director, exists := s.directors[directorID]
	if !exists {
		return false
	}

	return director.IsActive()
}

// SetDirectorActive sets the active status of a director
func (s *DirectorServiceImpl) SetDirectorActive(directorID string, active bool) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	director, exists := s.directors[directorID]
	if !exists {
		return ErrDirectorNotFound
	}

	director.SetActive(active)
	s.updatedAt = time.Now()

	return nil
}

// GetServiceStats returns service statistics
func (s *DirectorServiceImpl) GetServiceStats() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return map[string]interface{}{
		"service_name":    s.config.Name,
		"version":         s.config.Version,
		"active":          s.active,
		"created_at":      s.createdAt,
		"updated_at":      s.updatedAt,
		"directors_count": len(s.directors),
		"metadata":        s.config.Metadata,
	}
}

// GetHealthStatus returns the health status of the service
func (s *DirectorServiceImpl) GetHealthStatus() map[string]interface{} {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"director_service": map[string]interface{}{
				"status": "healthy",
				"active": s.active,
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if !s.active {
		healthStatus["checks"].(map[string]interface{})["director_service"].(map[string]interface{})["status"] = "error"
		healthStatus["checks"].(map[string]interface{})["director_service"].(map[string]interface{})["message"] = "Director service is inactive"
	}

	return healthStatus
}

// Cleanup performs cleanup operations
func (s *DirectorServiceImpl) Cleanup(ctx context.Context) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !s.active {
		return ErrServiceInactive
	}

	// Cleanup all directors
	for _, director := range s.directors {
		if err := director.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	s.updatedAt = time.Now()
	return nil
}

// Helper methods

func (s *BuilderServiceImpl) isBuilderTypeSupported(builderType string) bool {
	for _, supportedType := range s.config.SupportedBuilderTypes {
		if supportedType == builderType {
			return true
		}
	}
	return false
}

func (s *BuilderServiceImpl) getStringFromConfig(config map[string]interface{}, key, defaultValue string) string {
	if value, exists := config[key]; exists {
		if str, ok := value.(string); ok {
			return str
		}
	}
	return defaultValue
}

func (s *BuilderServiceImpl) getMapFromConfig(config map[string]interface{}, key string) map[string]interface{} {
	if value, exists := config[key]; exists {
		if m, ok := value.(map[string]interface{}); ok {
			return m
		}
	}
	return make(map[string]interface{})
}

func (s *DirectorServiceImpl) getStringFromConfig(config map[string]interface{}, key, defaultValue string) string {
	if value, exists := config[key]; exists {
		if str, ok := value.(string); ok {
			return str
		}
	}
	return defaultValue
}

func (s *DirectorServiceImpl) getMapFromConfig(config map[string]interface{}, key string) map[string]interface{} {
	if value, exists := config[key]; exists {
		if m, ok := value.(map[string]interface{}); ok {
			return m
		}
	}
	return make(map[string]interface{})
}
