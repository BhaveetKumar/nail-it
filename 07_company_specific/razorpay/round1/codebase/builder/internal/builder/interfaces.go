package builder

import (
	"context"
	"time"
)

// Builder defines the interface for building complex objects
type Builder interface {
	SetName(name string) Builder
	SetDescription(description string) Builder
	SetPrice(price float64) Builder
	SetCategory(category string) Builder
	SetTags(tags []string) Builder
	SetMetadata(metadata map[string]interface{}) Builder
	SetActive(active bool) Builder
	SetCreatedAt(createdAt time.Time) Builder
	SetUpdatedAt(updatedAt time.Time) Builder
	Build() (Product, error)
	Reset() Builder
	GetCurrentState() map[string]interface{}
	Validate() error
	IsValid() bool
	GetBuilderType() string
	GetBuilderInfo() map[string]interface{}
	IsActive() bool
	SetBuilderActive(active bool)
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetBuilderMetadata(key string, value interface{})
	Cleanup(ctx context.Context) error
}

// Product defines the interface for the built product
type Product interface {
	GetID() string
	GetName() string
	GetDescription() string
	GetPrice() float64
	GetCategory() string
	GetTags() []string
	GetMetadata() map[string]interface{}
	IsActive() bool
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	SetName(name string)
	SetDescription(description string)
	SetPrice(price float64)
	SetCategory(category string)
	SetTags(tags []string)
	SetMetadata(metadata map[string]interface{})
	SetActive(active bool)
	SetUpdatedAt(updatedAt time.Time)
	Validate() error
	GetStats() map[string]interface{}
	Process(ctx context.Context) error
	Clone() Product
	Equals(other Product) bool
	ToString() string
}

// Director defines the interface for directing the building process
type Director interface {
	SetBuilder(builder Builder) Director
	GetBuilder() Builder
	BuildProduct(productType string, config map[string]interface{}) (Product, error)
	BuildProductWithSteps(steps []BuildStep) (Product, error)
	GetBuildSteps(productType string) []BuildStep
	GetSupportedProductTypes() []string
	GetDirectorInfo() map[string]interface{}
	IsActive() bool
	SetActive(active bool)
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	Cleanup(ctx context.Context) error
}

// BuildStep represents a step in the building process
type BuildStep struct {
	ID          string                 `json:"id" yaml:"id"`
	Name        string                 `json:"name" yaml:"name"`
	Description string                 `json:"description" yaml:"description"`
	Action      string                 `json:"action" yaml:"action"`
	Parameters  map[string]interface{} `json:"parameters" yaml:"parameters"`
	Required    bool                   `json:"required" yaml:"required"`
	Order       int                    `json:"order" yaml:"order"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
}

// BuilderRegistry manages the registration and retrieval of builders
type BuilderRegistry interface {
	RegisterBuilder(builderType string, builder Builder) error
	UnregisterBuilder(builderType string) error
	GetBuilder(builderType string) (Builder, error)
	ListBuilders() []string
	GetBuilderStats(builderType string) map[string]interface{}
	GetAllBuilderStats() map[string]interface{}
	IsBuilderRegistered(builderType string) bool
	GetRegistryStats() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// ProductRegistry manages the registration and retrieval of products
type ProductRegistry interface {
	RegisterProduct(productID string, product Product) error
	UnregisterProduct(productID string) error
	GetProduct(productID string) (Product, error)
	ListProducts() []string
	GetProductStats(productID string) map[string]interface{}
	GetAllProductStats() map[string]interface{}
	IsProductRegistered(productID string) bool
	GetRegistryStats() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// BuilderService provides high-level operations for builder management
type BuilderService interface {
	CreateBuilder(builderType string, config map[string]interface{}) (Builder, error)
	DestroyBuilder(builderType string) error
	GetBuilder(builderType string) (Builder, error)
	ListBuilders() []string
	GetBuilderStats(builderType string) map[string]interface{}
	GetAllBuilderStats() map[string]interface{}
	IsBuilderActive(builderType string) bool
	SetBuilderActive(builderType string, active bool) error
	GetServiceStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// ProductService provides high-level operations for product management
type ProductService interface {
	CreateProduct(builderType string, config map[string]interface{}) (Product, error)
	DestroyProduct(productID string) error
	GetProduct(productID string) (Product, error)
	ListProducts() []string
	GetProductStats(productID string) map[string]interface{}
	GetAllProductStats() map[string]interface{}
	IsProductActive(productID string) bool
	SetProductActive(productID string, active bool) error
	GetServiceStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// DirectorService provides high-level operations for director management
type DirectorService interface {
	CreateDirector(config map[string]interface{}) (Director, error)
	DestroyDirector(directorID string) error
	GetDirector(directorID string) (Director, error)
	ListDirectors() []string
	GetDirectorStats(directorID string) map[string]interface{}
	GetAllDirectorStats() map[string]interface{}
	IsDirectorActive(directorID string) bool
	SetDirectorActive(directorID string, active bool) error
	GetServiceStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// BuilderConfig holds configuration for a builder
type BuilderConfig struct {
	Type        string                 `json:"type" yaml:"type"`
	Name        string                 `json:"name" yaml:"name"`
	Description string                 `json:"description" yaml:"description"`
	Version     string                 `json:"version" yaml:"version"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
}

// ProductConfig holds configuration for a product
type ProductConfig struct {
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

// DirectorConfig holds configuration for a director
type DirectorConfig struct {
	ID          string                 `json:"id" yaml:"id"`
	Name        string                 `json:"name" yaml:"name"`
	Description string                 `json:"description" yaml:"description"`
	Version     string                 `json:"version" yaml:"version"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
}

// ServiceConfig holds configuration for the service
type ServiceConfig struct {
	Name                  string                 `json:"name" yaml:"name"`
	Version               string                 `json:"version" yaml:"version"`
	Description           string                 `json:"description" yaml:"description"`
	MaxBuilders           int                    `json:"max_builders" yaml:"max_builders"`
	MaxProducts           int                    `json:"max_products" yaml:"max_products"`
	MaxDirectors          int                    `json:"max_directors" yaml:"max_directors"`
	CleanupInterval       time.Duration          `json:"cleanup_interval" yaml:"cleanup_interval"`
	ValidationEnabled     bool                   `json:"validation_enabled" yaml:"validation_enabled"`
	CachingEnabled        bool                   `json:"caching_enabled" yaml:"caching_enabled"`
	MonitoringEnabled     bool                   `json:"monitoring_enabled" yaml:"monitoring_enabled"`
	AuditingEnabled       bool                   `json:"auditing_enabled" yaml:"auditing_enabled"`
	SupportedBuilderTypes []string               `json:"supported_builder_types" yaml:"supported_builder_types"`
	SupportedProductTypes []string               `json:"supported_product_types" yaml:"supported_product_types"`
	ValidationRules       map[string]interface{} `json:"validation_rules" yaml:"validation_rules"`
	Metadata              map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// BuilderStats holds statistics for a builder
type BuilderStats struct {
	BuilderType    string                 `json:"builder_type" yaml:"builder_type"`
	Active         bool                   `json:"active" yaml:"active"`
	CreatedAt      time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt      time.Time              `json:"updated_at" yaml:"updated_at"`
	ProductsBuilt  int                    `json:"products_built" yaml:"products_built"`
	ProductsActive int                    `json:"products_active" yaml:"products_active"`
	Metadata       map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// ProductStats holds statistics for a product
type ProductStats struct {
	ProductID   string                 `json:"product_id" yaml:"product_id"`
	ProductType string                 `json:"product_type" yaml:"product_type"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
	Processed   int                    `json:"processed" yaml:"processed"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// DirectorStats holds statistics for a director
type DirectorStats struct {
	DirectorID    string                 `json:"director_id" yaml:"director_id"`
	Active        bool                   `json:"active" yaml:"active"`
	CreatedAt     time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at" yaml:"updated_at"`
	ProductsBuilt int                    `json:"products_built" yaml:"products_built"`
	Metadata      map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// ServiceStats holds statistics for the service
type ServiceStats struct {
	ServiceName    string                 `json:"service_name" yaml:"service_name"`
	Version        string                 `json:"version" yaml:"version"`
	Active         bool                   `json:"active" yaml:"active"`
	CreatedAt      time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt      time.Time              `json:"updated_at" yaml:"updated_at"`
	BuildersCount  int                    `json:"builders_count" yaml:"builders_count"`
	ProductsCount  int                    `json:"products_count" yaml:"products_count"`
	DirectorsCount int                    `json:"directors_count" yaml:"directors_count"`
	Metadata       map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// HealthStatus holds health status information
type HealthStatus struct {
	Status    string                 `json:"status" yaml:"status"`
	Checks    map[string]interface{} `json:"checks" yaml:"checks"`
	Timestamp time.Time              `json:"timestamp" yaml:"timestamp"`
}
