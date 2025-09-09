package composite

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// CompositeComponent represents a composite component that can contain other components
type CompositeComponent struct {
	BaseComponent
	MaxDepth    int `json:"max_depth"`
	MaxChildren int `json:"max_children"`
	mu          sync.RWMutex
}

// NewCompositeComponent creates a new composite component
func NewCompositeComponent(id, name, componentType string, maxDepth, maxChildren int) *CompositeComponent {
	now := time.Now()
	return &CompositeComponent{
		BaseComponent: BaseComponent{
			ID:        id,
			Name:      name,
			Type:      componentType,
			Metadata:  make(map[string]interface{}),
			CreatedAt: now,
			UpdatedAt: now,
		},
		MaxDepth:    maxDepth,
		MaxChildren: maxChildren,
	}
}

// Add adds a child component
func (cc *CompositeComponent) Add(child Component) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	
	if child == nil {
		return ErrInvalidComponent
	}
	
	// Check depth limit
	if cc.GetDepth() >= cc.MaxDepth {
		return ErrMaxDepthExceeded
	}
	
	// Check children limit
	if len(cc.Children) >= cc.MaxChildren {
		return ErrMaxChildrenExceeded
	}
	
	// Check if child already exists
	for _, existingChild := range cc.Children {
		if existingChild.GetID() == child.GetID() {
			return ErrComponentExists
		}
	}
	
	// Set parent and add to children
	child.SetParent(cc)
	cc.Children = append(cc.Children, child)
	cc.Update()
	
	return nil
}

// Remove removes a child component
func (cc *CompositeComponent) Remove(child Component) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	
	if child == nil {
		return ErrInvalidComponent
	}
	
	for i, existingChild := range cc.Children {
		if existingChild.GetID() == child.GetID() {
			cc.Children = append(cc.Children[:i], cc.Children[i+1:]...)
			child.SetParent(nil)
			cc.Update()
			return nil
		}
	}
	
	return ErrComponentNotFound
}

// GetChild returns a child component by ID
func (cc *CompositeComponent) GetChild(id string) (Component, error) {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	for _, child := range cc.Children {
		if child.GetID() == id {
			return child, nil
		}
	}
	return nil, ErrComponentNotFound
}

// GetAllChildren returns all children components
func (cc *CompositeComponent) GetAllChildren() []Component {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	allChildren := make([]Component, 0, len(cc.Children))
	for _, child := range cc.Children {
		allChildren = append(allChildren, child)
		if child.IsComposite() {
			if composite, ok := child.(Composite); ok {
				allChildren = append(allChildren, composite.GetAllChildren()...)
			}
		}
	}
	return allChildren
}

// GetChildrenByType returns children of a specific type
func (cc *CompositeComponent) GetChildrenByType(componentType string) []Component {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	var result []Component
	for _, child := range cc.Children {
		if child.GetType() == componentType {
			result = append(result, child)
		}
	}
	return result
}

// GetChildrenByDepth returns children at a specific depth
func (cc *CompositeComponent) GetChildrenByDepth(depth int) []Component {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	var result []Component
	for _, child := range cc.Children {
		if child.GetDepth() == depth {
			result = append(result, child)
		}
	}
	return result
}

// FindChild finds a child component using a predicate
func (cc *CompositeComponent) FindChild(predicate func(Component) bool) (Component, error) {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	for _, child := range cc.Children {
		if predicate(child) {
			return child, nil
		}
	}
	return nil, ErrComponentNotFound
}

// FindAllChildren finds all child components using a predicate
func (cc *CompositeComponent) FindAllChildren(predicate func(Component) bool) []Component {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	var result []Component
	for _, child := range cc.Children {
		if predicate(child) {
			result = append(result, child)
		}
	}
	return result
}

// Traverse traverses the component tree using a visitor
func (cc *CompositeComponent) Traverse(visitor func(Component) error) error {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	// Visit current component
	if err := visitor(cc); err != nil {
		return err
	}
	
	// Visit children
	for _, child := range cc.Children {
		if err := child.Execute(context.Background()); err != nil {
			return err
		}
		
		// Recursively traverse composite children
		if child.IsComposite() {
			if composite, ok := child.(Composite); ok {
				if err := composite.Traverse(visitor); err != nil {
					return err
				}
			}
		}
	}
	
	return nil
}

// GetStatistics returns component statistics
func (cc *CompositeComponent) GetStatistics() ComponentStatistics {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	stats := ComponentStatistics{
		TotalComponents:     cc.GetSize(),
		LeafComponents:      0,
		CompositeComponents: 1,
		MaxDepth:            cc.GetDepth(),
		AverageDepth:        0,
		TotalSize:           0,
		TypeDistribution:    make(map[string]int),
		LastUpdated:         time.Now(),
	}
	
	// Calculate statistics
	var totalDepth int
	cc.calculateStatistics(cc, &stats, &totalDepth, 0)
	
	if stats.TotalComponents > 0 {
		stats.AverageDepth = float64(totalDepth) / float64(stats.TotalComponents)
	}
	
	return stats
}

// calculateStatistics recursively calculates component statistics
func (cc *CompositeComponent) calculateStatistics(component Component, stats *ComponentStatistics, totalDepth *int, currentDepth int) {
	*totalDepth += currentDepth
	
	if component.IsLeaf() {
		stats.LeafComponents++
	} else {
		stats.CompositeComponents++
	}
	
	// Update type distribution
	componentType := component.GetType()
	stats.TypeDistribution[componentType]++
	
	// Update max depth
	if currentDepth > stats.MaxDepth {
		stats.MaxDepth = currentDepth
	}
	
	// Recursively process children
	for _, child := range component.GetChildren() {
		cc.calculateStatistics(child, stats, totalDepth, currentDepth+1)
	}
}

// Optimize optimizes the component tree structure
func (cc *CompositeComponent) Optimize() error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	
	// Sort children by name for better organization
	sort.Slice(cc.Children, func(i, j int) bool {
		return cc.Children[i].GetName() < cc.Children[j].GetName()
	})
	
	// Optimize child components
	for _, child := range cc.Children {
		if child.IsComposite() {
			if composite, ok := child.(Composite); ok {
				if err := composite.Optimize(); err != nil {
					return err
				}
			}
		}
	}
	
	cc.Update()
	return nil
}

// Execute executes the composite component
func (cc *CompositeComponent) Execute(ctx context.Context) (interface{}, error) {
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	
	// Simulate composite operations
	result := map[string]interface{}{
		"id":           cc.ID,
		"name":         cc.Name,
		"type":         cc.Type,
		"children":     len(cc.Children),
		"depth":        cc.GetDepth(),
		"size":         cc.GetSize(),
		"max_depth":    cc.MaxDepth,
		"max_children": cc.MaxChildren,
		"path":         cc.GetPath(),
	}
	
	return result, nil
}

// Validate validates the composite component
func (cc *CompositeComponent) Validate() error {
	if cc.ID == "" {
		return ErrInvalidID
	}
	if cc.Name == "" {
		return ErrInvalidName
	}
	if cc.MaxDepth < 0 {
		return fmt.Errorf("max depth must be non-negative")
	}
	if cc.MaxChildren < 0 {
		return fmt.Errorf("max children must be non-negative")
	}
	
	// Validate children
	for _, child := range cc.Children {
		if err := child.Validate(); err != nil {
			return fmt.Errorf("child validation failed: %w", err)
		}
	}
	
	return nil
}

// IsComposite returns true for composite components
func (cc *CompositeComponent) IsComposite() bool {
	return true
}

// FileSystemComponent represents a file system component
type FileSystemComponent struct {
	*CompositeComponent
	RootPath string `json:"root_path"`
}

// NewFileSystemComponent creates a new file system component
func NewFileSystemComponent(id, name, rootPath string) *FileSystemComponent {
	composite := NewCompositeComponent(id, name, "filesystem", 10, 1000)
	return &FileSystemComponent{
		CompositeComponent: composite,
		RootPath:           rootPath,
	}
}

// Execute executes the file system component
func (fsc *FileSystemComponent) Execute(ctx context.Context) (interface{}, error) {
	fsc.mu.RLock()
	defer fsc.mu.RUnlock()
	
	// Simulate file system operations
	result := map[string]interface{}{
		"id":           fsc.ID,
		"name":         fsc.Name,
		"type":         fsc.Type,
		"root_path":    fsc.RootPath,
		"children":     len(fsc.Children),
		"depth":        fsc.GetDepth(),
		"size":         fsc.GetSize(),
		"max_depth":    fsc.MaxDepth,
		"max_children": fsc.MaxChildren,
		"path":         fsc.GetPath(),
	}
	
	return result, nil
}

// Validate validates the file system component
func (fsc *FileSystemComponent) Validate() error {
	if err := fsc.CompositeComponent.Validate(); err != nil {
		return err
	}
	if fsc.RootPath == "" {
		return ErrInvalidPath
	}
	return nil
}

// MenuSystemComponent represents a menu system component
type MenuSystemComponent struct {
	*CompositeComponent
	BaseURL string `json:"base_url"`
}

// NewMenuSystemComponent creates a new menu system component
func NewMenuSystemComponent(id, name, baseURL string) *MenuSystemComponent {
	composite := NewCompositeComponent(id, name, "menu_system", 5, 100)
	return &MenuSystemComponent{
		CompositeComponent: composite,
		BaseURL:            baseURL,
	}
}

// Execute executes the menu system component
func (msc *MenuSystemComponent) Execute(ctx context.Context) (interface{}, error) {
	msc.mu.RLock()
	defer msc.mu.RUnlock()
	
	// Simulate menu system operations
	result := map[string]interface{}{
		"id":           msc.ID,
		"name":         msc.Name,
		"type":         msc.Type,
		"base_url":     msc.BaseURL,
		"children":     len(msc.Children),
		"depth":        msc.GetDepth(),
		"size":         msc.GetSize(),
		"max_depth":    msc.MaxDepth,
		"max_children": msc.MaxChildren,
		"path":         msc.GetPath(),
	}
	
	return result, nil
}

// Validate validates the menu system component
func (msc *MenuSystemComponent) Validate() error {
	if err := msc.CompositeComponent.Validate(); err != nil {
		return err
	}
	if msc.BaseURL == "" {
		return fmt.Errorf("base URL is required")
	}
	return nil
}
