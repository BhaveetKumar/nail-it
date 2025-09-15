package builder

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// BuilderRegistryImpl implements the BuilderRegistry interface
type BuilderRegistryImpl struct {
	builders  map[string]Builder
	mutex     sync.RWMutex
	createdAt time.Time
	updatedAt time.Time
	active    bool
}

// NewBuilderRegistry creates a new builder registry
func NewBuilderRegistry() *BuilderRegistryImpl {
	return &BuilderRegistryImpl{
		builders:  make(map[string]Builder),
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// RegisterBuilder registers a builder
func (r *BuilderRegistryImpl) RegisterBuilder(builderType string, builder Builder) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.active {
		return ErrRegistryInactive
	}

	if _, exists := r.builders[builderType]; exists {
		return ErrBuilderAlreadyExists
	}

	r.builders[builderType] = builder
	r.updatedAt = time.Now()

	return nil
}

// UnregisterBuilder unregisters a builder
func (r *BuilderRegistryImpl) UnregisterBuilder(builderType string) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.active {
		return ErrRegistryInactive
	}

	if _, exists := r.builders[builderType]; !exists {
		return ErrBuilderNotFound
	}

	delete(r.builders, builderType)
	r.updatedAt = time.Now()

	return nil
}

// GetBuilder retrieves a builder by type
func (r *BuilderRegistryImpl) GetBuilder(builderType string) (Builder, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	if !r.active {
		return nil, ErrRegistryInactive
	}

	builder, exists := r.builders[builderType]
	if !exists {
		return nil, ErrBuilderNotFound
	}

	return builder, nil
}

// ListBuilders returns a list of all registered builder types
func (r *BuilderRegistryImpl) ListBuilders() []string {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	builderTypes := make([]string, 0, len(r.builders))
	for builderType := range r.builders {
		builderTypes = append(builderTypes, builderType)
	}

	return builderTypes
}

// GetBuilderStats returns statistics for a specific builder
func (r *BuilderRegistryImpl) GetBuilderStats(builderType string) map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	builder, exists := r.builders[builderType]
	if !exists {
		return map[string]interface{}{
			"error": "builder not found",
		}
	}

	return builder.GetBuilderInfo()
}

// GetAllBuilderStats returns statistics for all builders
func (r *BuilderRegistryImpl) GetAllBuilderStats() map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	stats := make(map[string]interface{})
	for builderType, builder := range r.builders {
		stats[builderType] = builder.GetBuilderInfo()
	}

	return stats
}

// IsBuilderRegistered checks if a builder is registered
func (r *BuilderRegistryImpl) IsBuilderRegistered(builderType string) bool {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	_, exists := r.builders[builderType]
	return exists
}

// GetRegistryStats returns registry statistics
func (r *BuilderRegistryImpl) GetRegistryStats() map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	return map[string]interface{}{
		"active":         r.active,
		"created_at":     r.createdAt,
		"updated_at":     r.updatedAt,
		"builders_count": len(r.builders),
		"builders":       r.ListBuilders(),
	}
}

// Cleanup performs cleanup operations
func (r *BuilderRegistryImpl) Cleanup(ctx context.Context) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if !r.active {
		return ErrRegistryInactive
	}

	// Cleanup all builders
	for _, builder := range r.builders {
		if err := builder.Cleanup(ctx); err != nil {
			// Log error but continue cleanup
			continue
		}
	}

	r.updatedAt = time.Now()
	return nil
}

// ProductRegistryImpl implements the ProductRegistry interface
type ProductRegistryImpl struct {
	products  map[string]Product
	mutex     sync.RWMutex
	createdAt time.Time
	updatedAt time.Time
	active    bool
}

// NewProductRegistry creates a new product registry
func NewProductRegistry() *ProductRegistryImpl {
	return &ProductRegistryImpl{
		products:  make(map[string]Product),
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// RegisterProduct registers a product
func (r *ProductRegistryImpl) RegisterProduct(productID string, product Product) error {
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
func (r *ProductRegistryImpl) GetProduct(productID string) (Product, error) {
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

	return product.GetStats()
}

// GetAllProductStats returns statistics for all products
func (r *ProductRegistryImpl) GetAllProductStats() map[string]interface{} {
	r.mutex.RLock()
	defer r.mutex.RUnlock()

	stats := make(map[string]interface{})
	for productID, product := range r.products {
		stats[productID] = product.GetStats()
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
		"active":         r.active,
		"created_at":     r.createdAt,
		"updated_at":     r.updatedAt,
		"products_count": len(r.products),
		"products":       r.ListProducts(),
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
	r.products = make(map[string]Product)
	r.updatedAt = time.Now()

	return nil
}

// DirectorImpl implements the Director interface
type DirectorImpl struct {
	config     *DirectorConfig
	builder    Builder
	buildSteps map[string][]BuildStep
	createdAt  time.Time
	updatedAt  time.Time
	active     bool
	mutex      sync.RWMutex
}

// NewDirector creates a new director
func NewDirector(config *DirectorConfig) *DirectorImpl {
	return &DirectorImpl{
		config:     config,
		builder:    nil,
		buildSteps: make(map[string][]BuildStep),
		createdAt:  time.Now(),
		updatedAt:  time.Now(),
		active:     true,
	}
}

// SetBuilder sets the builder
func (d *DirectorImpl) SetBuilder(builder Builder) Director {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	d.builder = builder
	d.updatedAt = time.Now()
	return d
}

// GetBuilder returns the current builder
func (d *DirectorImpl) GetBuilder() Builder {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	return d.builder
}

// BuildProduct builds a product using the configured builder
func (d *DirectorImpl) BuildProduct(productType string, config map[string]interface{}) (Product, error) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	if !d.active {
		return nil, ErrDirectorInactive
	}

	if d.builder == nil {
		return nil, fmt.Errorf("no builder configured")
	}

	if !d.builder.IsActive() {
		return nil, ErrBuilderInactive
	}

	// Reset builder
	d.builder.Reset()

	// Apply configuration
	if name, ok := config["name"].(string); ok {
		d.builder.SetName(name)
	}
	if description, ok := config["description"].(string); ok {
		d.builder.SetDescription(description)
	}
	if price, ok := config["price"].(float64); ok {
		d.builder.SetPrice(price)
	}
	if category, ok := config["category"].(string); ok {
		d.builder.SetCategory(category)
	}
	if tags, ok := config["tags"].([]string); ok {
		d.builder.SetTags(tags)
	}
	if metadata, ok := config["metadata"].(map[string]interface{}); ok {
		d.builder.SetMetadata(metadata)
	}
	if active, ok := config["active"].(bool); ok {
		d.builder.SetActive(active)
	}

	// Build the product
	product, err := d.builder.Build()
	if err != nil {
		return nil, err
	}

	d.updatedAt = time.Now()
	return product, nil
}

// BuildProductWithSteps builds a product using specific build steps
func (d *DirectorImpl) BuildProductWithSteps(steps []BuildStep) (Product, error) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	if !d.active {
		return nil, ErrDirectorInactive
	}

	if d.builder == nil {
		return nil, fmt.Errorf("no builder configured")
	}

	if !d.builder.IsActive() {
		return nil, ErrBuilderInactive
	}

	// Reset builder
	d.builder.Reset()

	// Execute build steps in order
	for _, step := range steps {
		if err := d.executeBuildStep(step); err != nil {
			return nil, fmt.Errorf("failed to execute build step %s: %w", step.ID, err)
		}
	}

	// Build the product
	product, err := d.builder.Build()
	if err != nil {
		return nil, err
	}

	d.updatedAt = time.Now()
	return product, nil
}

// GetBuildSteps returns build steps for a product type
func (d *DirectorImpl) GetBuildSteps(productType string) []BuildStep {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	steps, exists := d.buildSteps[productType]
	if !exists {
		return []BuildStep{}
	}

	return steps
}

// GetSupportedProductTypes returns supported product types
func (d *DirectorImpl) GetSupportedProductTypes() []string {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	types := make([]string, 0, len(d.buildSteps))
	for productType := range d.buildSteps {
		types = append(types, productType)
	}

	return types
}

// GetDirectorInfo returns director information
func (d *DirectorImpl) GetDirectorInfo() map[string]interface{} {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	return map[string]interface{}{
		"id":                      d.config.ID,
		"name":                    d.config.Name,
		"description":             d.config.Description,
		"version":                 d.config.Version,
		"active":                  d.active,
		"created_at":              d.createdAt,
		"updated_at":              d.updatedAt,
		"metadata":                d.config.Metadata,
		"builder_type":            d.getBuilderType(),
		"supported_product_types": d.GetSupportedProductTypes(),
	}
}

// IsActive returns whether the director is active
func (d *DirectorImpl) IsActive() bool {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	return d.active
}

// SetActive sets the active status
func (d *DirectorImpl) SetActive(active bool) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	d.active = active
	d.updatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (d *DirectorImpl) GetCreatedAt() time.Time {
	return d.createdAt
}

// GetUpdatedAt returns the last update time
func (d *DirectorImpl) GetUpdatedAt() time.Time {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	return d.updatedAt
}

// GetMetadata returns the director metadata
func (d *DirectorImpl) GetMetadata() map[string]interface{} {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	return d.config.Metadata
}

// SetMetadata sets the director metadata
func (d *DirectorImpl) SetMetadata(key string, value interface{}) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	if d.config.Metadata == nil {
		d.config.Metadata = make(map[string]interface{})
	}
	d.config.Metadata[key] = value
	d.updatedAt = time.Now()
}

// Cleanup performs cleanup operations
func (d *DirectorImpl) Cleanup(ctx context.Context) error {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	if !d.active {
		return ErrDirectorInactive
	}

	// Cleanup builder
	if d.builder != nil {
		if err := d.builder.Cleanup(ctx); err != nil {
			// Log error but continue
		}
	}

	d.updatedAt = time.Now()
	return nil
}

// executeBuildStep executes a build step
func (d *DirectorImpl) executeBuildStep(step BuildStep) error {
	switch step.Action {
	case "set_name":
		if name, ok := step.Parameters["name"].(string); ok {
			d.builder.SetName(name)
		}
	case "set_description":
		if description, ok := step.Parameters["description"].(string); ok {
			d.builder.SetDescription(description)
		}
	case "set_price":
		if price, ok := step.Parameters["price"].(float64); ok {
			d.builder.SetPrice(price)
		}
	case "set_category":
		if category, ok := step.Parameters["category"].(string); ok {
			d.builder.SetCategory(category)
		}
	case "set_tags":
		if tags, ok := step.Parameters["tags"].([]string); ok {
			d.builder.SetTags(tags)
		}
	case "set_metadata":
		if metadata, ok := step.Parameters["metadata"].(map[string]interface{}); ok {
			d.builder.SetMetadata(metadata)
		}
	case "set_active":
		if active, ok := step.Parameters["active"].(bool); ok {
			d.builder.SetActive(active)
		}
	default:
		return fmt.Errorf("unknown build step action: %s", step.Action)
	}

	return nil
}

// getBuilderType returns the builder type
func (d *DirectorImpl) getBuilderType() string {
	if d.builder == nil {
		return "none"
	}
	return d.builder.GetBuilderType()
}
