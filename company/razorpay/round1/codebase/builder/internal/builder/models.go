package builder

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Common errors
var (
	ErrBuilderNotFound       = errors.New("builder not found")
	ErrBuilderAlreadyExists  = errors.New("builder already exists")
	ErrBuilderInactive       = errors.New("builder is inactive")
	ErrProductNotFound       = errors.New("product not found")
	ErrProductAlreadyExists  = errors.New("product already exists")
	ErrProductInactive       = errors.New("product is inactive")
	ErrDirectorNotFound      = errors.New("director not found")
	ErrDirectorAlreadyExists = errors.New("director already exists")
	ErrDirectorInactive      = errors.New("director is inactive")
	ErrInvalidBuilderType    = errors.New("invalid builder type")
	ErrInvalidProductType    = errors.New("invalid product type")
	ErrInvalidConfiguration  = errors.New("invalid configuration")
	ErrValidationFailed      = errors.New("validation failed")
	ErrServiceInactive       = errors.New("service is inactive")
	ErrRegistryNotFound      = errors.New("registry not found")
	ErrRegistryInactive      = errors.New("registry is inactive")
	ErrBuildStepNotFound     = errors.New("build step not found")
	ErrBuildStepInvalid      = errors.New("build step is invalid")
	ErrBuildStepOrderInvalid = errors.New("build step order is invalid")
)

// BaseProduct represents a base product implementation
type BaseProduct struct {
	ID          string                 `json:"id" yaml:"id"`
	Name        string                 `json:"name" yaml:"name"`
	Description string                 `json:"description" yaml:"description"`
	Price       float64                `json:"price" yaml:"price"`
	Category    string                 `json:"category" yaml:"category"`
	Tags        []string               `json:"tags" yaml:"tags"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
}

// GetID returns the product ID
func (p *BaseProduct) GetID() string {
	return p.ID
}

// GetName returns the product name
func (p *BaseProduct) GetName() string {
	return p.Name
}

// GetDescription returns the product description
func (p *BaseProduct) GetDescription() string {
	return p.Description
}

// GetPrice returns the product price
func (p *BaseProduct) GetPrice() float64 {
	return p.Price
}

// GetCategory returns the product category
func (p *BaseProduct) GetCategory() string {
	return p.Category
}

// GetTags returns the product tags
func (p *BaseProduct) GetTags() []string {
	return p.Tags
}

// GetMetadata returns the product metadata
func (p *BaseProduct) GetMetadata() map[string]interface{} {
	return p.Metadata
}

// IsActive returns whether the product is active
func (p *BaseProduct) IsActive() bool {
	return p.Active
}

// GetCreatedAt returns the creation time
func (p *BaseProduct) GetCreatedAt() time.Time {
	return p.CreatedAt
}

// GetUpdatedAt returns the last update time
func (p *BaseProduct) GetUpdatedAt() time.Time {
	return p.UpdatedAt
}

// SetName sets the product name
func (p *BaseProduct) SetName(name string) {
	p.Name = name
	p.UpdatedAt = time.Now()
}

// SetDescription sets the product description
func (p *BaseProduct) SetDescription(description string) {
	p.Description = description
	p.UpdatedAt = time.Now()
}

// SetPrice sets the product price
func (p *BaseProduct) SetPrice(price float64) {
	p.Price = price
	p.UpdatedAt = time.Now()
}

// SetCategory sets the product category
func (p *BaseProduct) SetCategory(category string) {
	p.Category = category
	p.UpdatedAt = time.Now()
}

// SetTags sets the product tags
func (p *BaseProduct) SetTags(tags []string) {
	p.Tags = tags
	p.UpdatedAt = time.Now()
}

// SetMetadata sets the product metadata
func (p *BaseProduct) SetMetadata(metadata map[string]interface{}) {
	p.Metadata = metadata
	p.UpdatedAt = time.Now()
}

// SetActive sets the active status
func (p *BaseProduct) SetActive(active bool) {
	p.Active = active
	p.UpdatedAt = time.Now()
}

// SetUpdatedAt sets the updated time
func (p *BaseProduct) SetUpdatedAt(updatedAt time.Time) {
	p.UpdatedAt = updatedAt
}

// Validate validates the product
func (p *BaseProduct) Validate() error {
	if p.ID == "" {
		return fmt.Errorf("product ID is required")
	}
	if p.Name == "" {
		return fmt.Errorf("product name is required")
	}
	if p.Description == "" {
		return fmt.Errorf("product description is required")
	}
	if p.Price < 0 {
		return fmt.Errorf("product price cannot be negative")
	}
	if p.Category == "" {
		return fmt.Errorf("product category is required")
	}
	return nil
}

// GetStats returns product statistics
func (p *BaseProduct) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"id":          p.ID,
		"name":        p.Name,
		"description": p.Description,
		"price":       p.Price,
		"category":    p.Category,
		"tags":        p.Tags,
		"active":      p.Active,
		"created_at":  p.CreatedAt,
		"updated_at":  p.UpdatedAt,
		"metadata":    p.Metadata,
	}
}

// Process processes the product
func (p *BaseProduct) Process(ctx context.Context) error {
	if !p.Active {
		return ErrProductInactive
	}
	// Simulate processing
	time.Sleep(100 * time.Millisecond)
	p.UpdatedAt = time.Now()
	return nil
}

// Clone creates a copy of the product
func (p *BaseProduct) Clone() Product {
	return &BaseProduct{
		ID:          p.ID + "-clone",
		Name:        p.Name + " (Clone)",
		Description: p.Description,
		Price:       p.Price,
		Category:    p.Category,
		Tags:        append([]string(nil), p.Tags...),
		Metadata:    p.copyMetadata(),
		Active:      p.Active,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

// Equals checks if two products are equal
func (p *BaseProduct) Equals(other Product) bool {
	if other == nil {
		return false
	}
	return p.ID == other.GetID() &&
		p.Name == other.GetName() &&
		p.Description == other.GetDescription() &&
		p.Price == other.GetPrice() &&
		p.Category == other.GetCategory() &&
		p.Active == other.IsActive()
}

// ToString returns a string representation of the product
func (p *BaseProduct) ToString() string {
	return fmt.Sprintf("Product{ID: %s, Name: %s, Price: %.2f, Category: %s, Active: %t}",
		p.ID, p.Name, p.Price, p.Category, p.Active)
}

// copyMetadata creates a deep copy of metadata
func (p *BaseProduct) copyMetadata() map[string]interface{} {
	if p.Metadata == nil {
		return nil
	}
	copied := make(map[string]interface{})
	for k, v := range p.Metadata {
		copied[k] = v
	}
	return copied
}

// ConcreteProduct represents a concrete implementation of Product
type ConcreteProduct struct {
	BaseProduct
	Brand string `json:"brand" yaml:"brand"`
	Model string `json:"model" yaml:"model"`
	SKU   string `json:"sku" yaml:"sku"`
}

// NewConcreteProduct creates a new ConcreteProduct
func NewConcreteProduct(id, name, description string, price float64, category string) *ConcreteProduct {
	return &ConcreteProduct{
		BaseProduct: BaseProduct{
			ID:          id,
			Name:        name,
			Description: description,
			Price:       price,
			Category:    category,
			Tags:        []string{},
			Metadata:    make(map[string]interface{}),
			Active:      true,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
		Brand: "Default Brand",
		Model: "Default Model",
		SKU:   "SKU-001",
	}
}

// GetBrand returns the product brand
func (p *ConcreteProduct) GetBrand() string {
	return p.Brand
}

// GetModel returns the product model
func (p *ConcreteProduct) GetModel() string {
	return p.Model
}

// GetSKU returns the product SKU
func (p *ConcreteProduct) GetSKU() string {
	return p.SKU
}

// SetBrand sets the product brand
func (p *ConcreteProduct) SetBrand(brand string) {
	p.Brand = brand
	p.UpdatedAt = time.Now()
}

// SetModel sets the product model
func (p *ConcreteProduct) SetModel(model string) {
	p.Model = model
	p.UpdatedAt = time.Now()
}

// SetSKU sets the product SKU
func (p *ConcreteProduct) SetSKU(sku string) {
	p.SKU = sku
	p.UpdatedAt = time.Now()
}

// GetStats returns product statistics
func (p *ConcreteProduct) GetStats() map[string]interface{} {
	stats := p.BaseProduct.GetStats()
	stats["brand"] = p.Brand
	stats["model"] = p.Model
	stats["sku"] = p.SKU
	return stats
}

// Clone creates a copy of the product
func (p *ConcreteProduct) Clone() Product {
	return &ConcreteProduct{
		BaseProduct: BaseProduct{
			ID:          p.ID + "-clone",
			Name:        p.Name + " (Clone)",
			Description: p.Description,
			Price:       p.Price,
			Category:    p.Category,
			Tags:        append([]string(nil), p.Tags...),
			Metadata:    p.copyMetadata(),
			Active:      p.Active,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
		Brand: p.Brand,
		Model: p.Model,
		SKU:   p.SKU + "-clone",
	}
}

// ConcreteBuilder represents a concrete implementation of Builder
type ConcreteBuilder struct {
	config    *BuilderConfig
	product   *ConcreteProduct
	createdAt time.Time
	updatedAt time.Time
	active    bool
}

// NewConcreteBuilder creates a new ConcreteBuilder
func NewConcreteBuilder(config *BuilderConfig) *ConcreteBuilder {
	return &ConcreteBuilder{
		config:    config,
		product:   nil,
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// SetName sets the product name
func (b *ConcreteBuilder) SetName(name string) Builder {
	if b.product == nil {
		b.product = &ConcreteProduct{}
	}
	b.product.SetName(name)
	b.updatedAt = time.Now()
	return b
}

// SetDescription sets the product description
func (b *ConcreteBuilder) SetDescription(description string) Builder {
	if b.product == nil {
		b.product = &ConcreteProduct{}
	}
	b.product.SetDescription(description)
	b.updatedAt = time.Now()
	return b
}

// SetPrice sets the product price
func (b *ConcreteBuilder) SetPrice(price float64) Builder {
	if b.product == nil {
		b.product = &ConcreteProduct{}
	}
	b.product.SetPrice(price)
	b.updatedAt = time.Now()
	return b
}

// SetCategory sets the product category
func (b *ConcreteBuilder) SetCategory(category string) Builder {
	if b.product == nil {
		b.product = &ConcreteProduct{}
	}
	b.product.SetCategory(category)
	b.updatedAt = time.Now()
	return b
}

// SetTags sets the product tags
func (b *ConcreteBuilder) SetTags(tags []string) Builder {
	if b.product == nil {
		b.product = &ConcreteProduct{}
	}
	b.product.SetTags(tags)
	b.updatedAt = time.Now()
	return b
}

// SetMetadata sets the product metadata
func (b *ConcreteBuilder) SetMetadata(metadata map[string]interface{}) Builder {
	if b.product == nil {
		b.product = &ConcreteProduct{}
	}
	b.product.SetMetadata(metadata)
	b.updatedAt = time.Now()
	return b
}

// SetActive sets the product active status
func (b *ConcreteBuilder) SetActive(active bool) Builder {
	if b.product == nil {
		b.product = &ConcreteProduct{}
	}
	b.product.SetActive(active)
	b.updatedAt = time.Now()
	return b
}

// SetCreatedAt sets the product creation time
func (b *ConcreteBuilder) SetCreatedAt(createdAt time.Time) Builder {
	if b.product == nil {
		b.product = &ConcreteProduct{}
	}
	b.product.CreatedAt = createdAt
	b.updatedAt = time.Now()
	return b
}

// SetUpdatedAt sets the product update time
func (b *ConcreteBuilder) SetUpdatedAt(updatedAt time.Time) Builder {
	if b.product == nil {
		b.product = &ConcreteProduct{}
	}
	b.product.SetUpdatedAt(updatedAt)
	b.updatedAt = time.Now()
	return b
}

// Build builds the product
func (b *ConcreteBuilder) Build() (Product, error) {
	if !b.active {
		return nil, ErrBuilderInactive
	}

	if b.product == nil {
		return nil, fmt.Errorf("no product to build")
	}

	// Set default values if not set
	if b.product.ID == "" {
		b.product.ID = fmt.Sprintf("product-%d", time.Now().UnixNano())
	}
	if b.product.CreatedAt.IsZero() {
		b.product.CreatedAt = time.Now()
	}
	if b.product.UpdatedAt.IsZero() {
		b.product.UpdatedAt = time.Now()
	}

	// Validate the product
	if err := b.product.Validate(); err != nil {
		return nil, err
	}

	// Create a copy of the product
	product := b.product.Clone()
	b.updatedAt = time.Now()

	return product, nil
}

// Reset resets the builder
func (b *ConcreteBuilder) Reset() Builder {
	b.product = nil
	b.updatedAt = time.Now()
	return b
}

// GetCurrentState returns the current state of the builder
func (b *ConcreteBuilder) GetCurrentState() map[string]interface{} {
	state := map[string]interface{}{
		"builder_type": b.GetBuilderType(),
		"active":       b.active,
		"created_at":   b.createdAt,
		"updated_at":   b.updatedAt,
		"has_product":  b.product != nil,
	}

	if b.product != nil {
		state["product"] = b.product.GetStats()
	}

	return state
}

// Validate validates the builder
func (b *ConcreteBuilder) Validate() error {
	if !b.active {
		return ErrBuilderInactive
	}
	if b.product == nil {
		return fmt.Errorf("no product to validate")
	}
	return b.product.Validate()
}

// IsValid checks if the builder is valid
func (b *ConcreteBuilder) IsValid() bool {
	return b.active && b.product != nil && b.product.Validate() == nil
}

// GetBuilderType returns the builder type
func (b *ConcreteBuilder) GetBuilderType() string {
	return "ConcreteBuilder"
}

// GetBuilderInfo returns builder information
func (b *ConcreteBuilder) GetBuilderInfo() map[string]interface{} {
	return map[string]interface{}{
		"type":        b.GetBuilderType(),
		"name":        b.config.Name,
		"description": b.config.Description,
		"version":     b.config.Version,
		"active":      b.active,
		"created_at":  b.createdAt,
		"updated_at":  b.updatedAt,
		"metadata":    b.config.Metadata,
	}
}

// IsActive returns whether the builder is active
func (b *ConcreteBuilder) IsActive() bool {
	return b.active
}

// SetBuilderActive sets the active status
func (b *ConcreteBuilder) SetBuilderActive(active bool) {
	b.active = active
	b.updatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (b *ConcreteBuilder) GetCreatedAt() time.Time {
	return b.createdAt
}

// GetUpdatedAt returns the last update time
func (b *ConcreteBuilder) GetUpdatedAt() time.Time {
	return b.updatedAt
}

// GetMetadata returns the builder metadata
func (b *ConcreteBuilder) GetMetadata() map[string]interface{} {
	return b.config.Metadata
}

// SetBuilderMetadata sets the builder metadata
func (b *ConcreteBuilder) SetBuilderMetadata(key string, value interface{}) {
	if b.config.Metadata == nil {
		b.config.Metadata = make(map[string]interface{})
	}
	b.config.Metadata[key] = value
	b.updatedAt = time.Now()
}

// Cleanup performs cleanup operations
func (b *ConcreteBuilder) Cleanup(ctx context.Context) error {
	if !b.active {
		return ErrBuilderInactive
	}
	// Perform cleanup operations
	b.updatedAt = time.Now()
	return nil
}
