package composite

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// CompositeService provides high-level operations using composite pattern
type CompositeService struct {
	components map[string]Component
	cache      Cache
	database   Database
	logger     Logger
	metrics    Metrics
	config     CompositeConfig
	mu         sync.RWMutex
}

// NewCompositeService creates a new composite service
func NewCompositeService(
	cache Cache,
	database Database,
	logger Logger,
	metrics Metrics,
	config CompositeConfig,
) *CompositeService {
	return &CompositeService{
		components: make(map[string]Component),
		cache:      cache,
		database:   database,
		logger:     logger,
		metrics:    metrics,
		config:     config,
	}
}

// CreateFileSystem creates a new file system component
func (cs *CompositeService) CreateFileSystem(ctx context.Context, request CreateFileSystemRequest) (*FileSystemComponent, error) {
	start := time.Now()
	
	// Create file system component
	fileSystem := NewFileSystemComponent(request.ID, request.Name, request.RootPath)
	
	// Store in memory
	cs.mu.Lock()
	cs.components[request.ID] = fileSystem
	cs.mu.Unlock()
	
	// Store in cache
	cs.cache.Set(request.ID, fileSystem, cs.config.Cache.TTL)
	
	// Store in database
	if err := cs.database.Save(ctx, "filesystems", fileSystem); err != nil {
		cs.logger.Error("Failed to save file system to database", "id", request.ID, "error", err)
	}
	
	duration := time.Since(start)
	cs.metrics.RecordTiming("filesystem_creation_duration", duration, map[string]string{"status": "success"})
	cs.metrics.IncrementCounter("filesystem_created", map[string]string{"status": "success"})
	
	cs.logger.Info("File system created", "id", request.ID, "name", request.Name, "root_path", request.RootPath)
	
	return fileSystem, nil
}

// CreateMenuSystem creates a new menu system component
func (cs *CompositeService) CreateMenuSystem(ctx context.Context, request CreateMenuSystemRequest) (*MenuSystemComponent, error) {
	start := time.Now()
	
	// Create menu system component
	menuSystem := NewMenuSystemComponent(request.ID, request.Name, request.BaseURL)
	
	// Store in memory
	cs.mu.Lock()
	cs.components[request.ID] = menuSystem
	cs.mu.Unlock()
	
	// Store in cache
	cs.cache.Set(request.ID, menuSystem, cs.config.Cache.TTL)
	
	// Store in database
	if err := cs.database.Save(ctx, "menu_systems", menuSystem); err != nil {
		cs.logger.Error("Failed to save menu system to database", "id", request.ID, "error", err)
	}
	
	duration := time.Since(start)
	cs.metrics.RecordTiming("menu_system_creation_duration", duration, map[string]string{"status": "success"})
	cs.metrics.IncrementCounter("menu_system_created", map[string]string{"status": "success"})
	
	cs.logger.Info("Menu system created", "id", request.ID, "name", request.Name, "base_url", request.BaseURL)
	
	return menuSystem, nil
}

// AddComponent adds a component to a parent component
func (cs *CompositeService) AddComponent(ctx context.Context, request AddComponentRequest) error {
	start := time.Now()
	
	// Get parent component
	parent, err := cs.GetComponent(ctx, request.ParentID)
	if err != nil {
		cs.logger.Error("Failed to get parent component", "parent_id", request.ParentID, "error", err)
		return fmt.Errorf("parent component not found: %w", err)
	}
	
	// Check if parent is composite
	if !parent.IsComposite() {
		return fmt.Errorf("parent component is not composite")
	}
	
	// Create child component based on type
	var child Component
	switch request.ComponentType {
	case "folder":
		child = NewFolderComponent(request.ChildID, request.ChildName, request.Path)
	case "file":
		child = NewFileComponent(request.ChildID, request.ChildName, request.Path)
	case "menu":
		child = NewMenuComponent(request.ChildID, request.ChildName, request.URL)
	default:
		return fmt.Errorf("unsupported component type: %s", request.ComponentType)
	}
	
	// Add child to parent
	if err := parent.Add(child); err != nil {
		cs.logger.Error("Failed to add child component", "parent_id", request.ParentID, "child_id", request.ChildID, "error", err)
		return fmt.Errorf("failed to add child component: %w", err)
	}
	
	// Store child in memory
	cs.mu.Lock()
	cs.components[request.ChildID] = child
	cs.mu.Unlock()
	
	// Store in cache
	cs.cache.Set(request.ChildID, child, cs.config.Cache.TTL)
	
	// Store in database
	if err := cs.database.Save(ctx, "components", child); err != nil {
		cs.logger.Error("Failed to save child component to database", "child_id", request.ChildID, "error", err)
	}
	
	duration := time.Since(start)
	cs.metrics.RecordTiming("component_addition_duration", duration, map[string]string{"type": request.ComponentType})
	cs.metrics.IncrementCounter("component_added", map[string]string{"type": request.ComponentType})
	
	cs.logger.Info("Component added", "parent_id", request.ParentID, "child_id", request.ChildID, "type", request.ComponentType)
	
	return nil
}

// RemoveComponent removes a component from its parent
func (cs *CompositeService) RemoveComponent(ctx context.Context, request RemoveComponentRequest) error {
	start := time.Now()
	
	// Get parent component
	parent, err := cs.GetComponent(ctx, request.ParentID)
	if err != nil {
		cs.logger.Error("Failed to get parent component", "parent_id", request.ParentID, "error", err)
		return fmt.Errorf("parent component not found: %w", err)
	}
	
	// Get child component
	child, err := cs.GetComponent(ctx, request.ChildID)
	if err != nil {
		cs.logger.Error("Failed to get child component", "child_id", request.ChildID, "error", err)
		return fmt.Errorf("child component not found: %w", err)
	}
	
	// Remove child from parent
	if err := parent.Remove(child); err != nil {
		cs.logger.Error("Failed to remove child component", "parent_id", request.ParentID, "child_id", request.ChildID, "error", err)
		return fmt.Errorf("failed to remove child component: %w", err)
	}
	
	// Remove from memory
	cs.mu.Lock()
	delete(cs.components, request.ChildID)
	cs.mu.Unlock()
	
	// Remove from cache
	cs.cache.Delete(request.ChildID)
	
	// Remove from database
	if err := cs.database.Delete(ctx, "components", request.ChildID); err != nil {
		cs.logger.Error("Failed to delete child component from database", "child_id", request.ChildID, "error", err)
	}
	
	duration := time.Since(start)
	cs.metrics.RecordTiming("component_removal_duration", duration, map[string]string{"type": child.GetType()})
	cs.metrics.IncrementCounter("component_removed", map[string]string{"type": child.GetType()})
	
	cs.logger.Info("Component removed", "parent_id", request.ParentID, "child_id", request.ChildID)
	
	return nil
}

// GetComponent retrieves a component by ID
func (cs *CompositeService) GetComponent(ctx context.Context, id string) (Component, error) {
	start := time.Now()
	
	// Check cache first
	if cached, found := cs.cache.Get(id); found {
		cs.metrics.IncrementCounter("component_cache_hit", map[string]string{"id": id})
		return cached.(Component), nil
	}
	
	// Check memory
	cs.mu.RLock()
	component, exists := cs.components[id]
	cs.mu.RUnlock()
	
	if exists {
		// Update cache
		cs.cache.Set(id, component, cs.config.Cache.TTL)
		cs.metrics.IncrementCounter("component_memory_hit", map[string]string{"id": id})
		return component, nil
	}
	
	// Load from database
	componentData, err := cs.database.Find(ctx, "components", map[string]interface{}{"id": id})
	if err != nil {
		cs.logger.Error("Failed to load component from database", "id", id, "error", err)
		return nil, fmt.Errorf("component not found: %w", err)
	}
	
	// Reconstruct component from database data
	component, err = cs.reconstructComponent(componentData)
	if err != nil {
		cs.logger.Error("Failed to reconstruct component", "id", id, "error", err)
		return nil, fmt.Errorf("failed to reconstruct component: %w", err)
	}
	
	// Store in memory and cache
	cs.mu.Lock()
	cs.components[id] = component
	cs.mu.Unlock()
	cs.cache.Set(id, component, cs.config.Cache.TTL)
	
	duration := time.Since(start)
	cs.metrics.RecordTiming("component_retrieval_duration", duration, map[string]string{"source": "database"})
	cs.metrics.IncrementCounter("component_retrieved", map[string]string{"source": "database"})
	
	cs.logger.Debug("Component retrieved", "id", id, "duration", duration)
	
	return component, nil
}

// ExecuteComponent executes a component and its children
func (cs *CompositeService) ExecuteComponent(ctx context.Context, id string) (interface{}, error) {
	start := time.Now()
	
	// Get component
	component, err := cs.GetComponent(ctx, id)
	if err != nil {
		cs.logger.Error("Failed to get component", "id", id, "error", err)
		return nil, fmt.Errorf("component not found: %w", err)
	}
	
	// Execute component
	result, err := component.Execute(ctx)
	if err != nil {
		cs.logger.Error("Failed to execute component", "id", id, "error", err)
		return nil, fmt.Errorf("failed to execute component: %w", err)
	}
	
	duration := time.Since(start)
	cs.metrics.RecordTiming("component_execution_duration", duration, map[string]string{"type": component.GetType()})
	cs.metrics.IncrementCounter("component_executed", map[string]string{"type": component.GetType()})
	
	cs.logger.Debug("Component executed", "id", id, "type", component.GetType(), "duration", duration)
	
	return result, nil
}

// GetComponentTree returns the complete component tree
func (cs *CompositeService) GetComponentTree(ctx context.Context, id string) (*ComponentTree, error) {
	start := time.Now()
	
	// Get root component
	root, err := cs.GetComponent(ctx, id)
	if err != nil {
		cs.logger.Error("Failed to get root component", "id", id, "error", err)
		return nil, fmt.Errorf("root component not found: %w", err)
	}
	
	// Build tree structure
	tree := &ComponentTree{
		Root:       root,
		TotalNodes: root.GetSize(),
		MaxDepth:   root.GetDepth(),
		CreatedAt:  time.Now(),
	}
	
	duration := time.Since(start)
	cs.metrics.RecordTiming("component_tree_retrieval_duration", duration, map[string]string{"root_id": id})
	cs.metrics.IncrementCounter("component_tree_retrieved", map[string]string{"root_id": id})
	
	cs.logger.Debug("Component tree retrieved", "id", id, "total_nodes", tree.TotalNodes, "max_depth", tree.MaxDepth, "duration", duration)
	
	return tree, nil
}

// GetComponentStatistics returns statistics for a component
func (cs *CompositeService) GetComponentStatistics(ctx context.Context, id string) (*ComponentStatistics, error) {
	start := time.Now()
	
	// Get component
	component, err := cs.GetComponent(ctx, id)
	if err != nil {
		cs.logger.Error("Failed to get component", "id", id, "error", err)
		return nil, fmt.Errorf("component not found: %w", err)
	}
	
	// Get statistics
	var stats ComponentStatistics
	if component.IsComposite() {
		if composite, ok := component.(Composite); ok {
			stats = composite.GetStatistics()
		}
	} else {
		stats = ComponentStatistics{
			TotalComponents:     1,
			LeafComponents:      1,
			CompositeComponents: 0,
			MaxDepth:            component.GetDepth(),
			AverageDepth:        float64(component.GetDepth()),
			TotalSize:           0,
			TypeDistribution:    map[string]int{component.GetType(): 1},
			LastUpdated:         time.Now(),
		}
	}
	
	duration := time.Since(start)
	cs.metrics.RecordTiming("component_statistics_duration", duration, map[string]string{"type": component.GetType()})
	cs.metrics.IncrementCounter("component_statistics_retrieved", map[string]string{"type": component.GetType()})
	
	cs.logger.Debug("Component statistics retrieved", "id", id, "type", component.GetType(), "duration", duration)
	
	return &stats, nil
}

// OptimizeComponent optimizes a component tree
func (cs *CompositeService) OptimizeComponent(ctx context.Context, id string) error {
	start := time.Now()
	
	// Get component
	component, err := cs.GetComponent(ctx, id)
	if err != nil {
		cs.logger.Error("Failed to get component", "id", id, "error", err)
		return fmt.Errorf("component not found: %w", err)
	}
	
	// Optimize if composite
	if component.IsComposite() {
		if composite, ok := component.(Composite); ok {
			if err := composite.Optimize(); err != nil {
				cs.logger.Error("Failed to optimize component", "id", id, "error", err)
				return fmt.Errorf("failed to optimize component: %w", err)
			}
		}
	}
	
	duration := time.Since(start)
	cs.metrics.RecordTiming("component_optimization_duration", duration, map[string]string{"type": component.GetType()})
	cs.metrics.IncrementCounter("component_optimized", map[string]string{"type": component.GetType()})
	
	cs.logger.Info("Component optimized", "id", id, "type", component.GetType(), "duration", duration)
	
	return nil
}

// Helper methods

func (cs *CompositeService) reconstructComponent(data interface{}) (Component, error) {
	// This is a simplified reconstruction - in a real implementation,
	// you would properly deserialize the data based on the component type
	// For now, we'll return a mock component
	return NewFolderComponent("reconstructed", "Reconstructed Component", "/reconstructed"), nil
}

// Request/Response models

type CreateFileSystemRequest struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	RootPath string `json:"root_path"`
}

type CreateMenuSystemRequest struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	BaseURL string `json:"base_url"`
}

type AddComponentRequest struct {
	ParentID      string `json:"parent_id"`
	ChildID       string `json:"child_id"`
	ChildName     string `json:"child_name"`
	ComponentType string `json:"component_type"`
	Path          string `json:"path,omitempty"`
	URL           string `json:"url,omitempty"`
}

type RemoveComponentRequest struct {
	ParentID string `json:"parent_id"`
	ChildID  string `json:"child_id"`
}

type ComponentTree struct {
	Root       Component `json:"root"`
	TotalNodes int       `json:"total_nodes"`
	MaxDepth   int       `json:"max_depth"`
	CreatedAt  time.Time `json:"created_at"`
}
