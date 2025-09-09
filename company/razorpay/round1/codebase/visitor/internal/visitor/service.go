package visitor

import (
	"context"
	"errors"
	"time"
)

// Custom errors
var (
	ErrMaxVisitorsReached           = errors.New("maximum number of visitors reached")
	ErrMaxElementsReached           = errors.New("maximum number of elements reached")
	ErrMaxElementCollectionsReached = errors.New("maximum number of element collections reached")
	ErrVisitorNotFound              = errors.New("visitor not found")
	ErrElementNotFound              = errors.New("element not found")
	ErrElementCollectionNotFound    = errors.New("element collection not found")
	ErrInvalidVisitorType           = errors.New("invalid visitor type")
	ErrInvalidElementType           = errors.New("invalid element type")
	ErrVisitTimeout                 = errors.New("visit operation timed out")
	ErrValidationFailed             = errors.New("validation failed")
	ErrProcessingFailed             = errors.New("processing failed")
	ErrAnalyticsFailed              = errors.New("analytics failed")
)

// VisitorServiceManager manages the visitor service operations
type VisitorServiceManager struct {
	service *VisitorService
	config  *VisitorConfig
}

// NewVisitorServiceManager creates a new visitor service manager
func NewVisitorServiceManager(config *VisitorConfig) *VisitorServiceManager {
	return &VisitorServiceManager{
		service: NewVisitorService(config),
		config:  config,
	}
}

// CreateVisitor creates a new visitor with validation
func (vsm *VisitorServiceManager) CreateVisitor(name, visitorType, description string) (Visitor, error) {
	// Validate input
	if name == "" {
		return nil, errors.New("visitor name cannot be empty")
	}
	if visitorType == "" {
		visitorType = vsm.config.DefaultVisitorType
	}
	if description == "" {
		description = "Auto-generated visitor"
	}

	// Validate visitor type
	if vsm.config.ValidationEnabled {
		if !vsm.isValidVisitorType(visitorType) {
			return nil, ErrInvalidVisitorType
		}
	}

	// Create visitor
	visitor, err := vsm.service.CreateVisitor(name, visitorType, description)
	if err != nil {
		return nil, err
	}

	// Set metadata
	visitor.SetMetadata("created_by", "visitor-service-manager")
	visitor.SetMetadata("creation_time", time.Now())

	return visitor, nil
}

// GetVisitor retrieves a visitor with caching
func (vsm *VisitorServiceManager) GetVisitor(visitorID string) (Visitor, error) {
	// Validate input
	if visitorID == "" {
		return nil, errors.New("visitor ID cannot be empty")
	}

	// Get visitor
	visitor, err := vsm.service.GetVisitor(visitorID)
	if err != nil {
		return nil, err
	}

	// Update last accessed time
	visitor.SetMetadata("last_accessed", time.Now())

	return visitor, nil
}

// RemoveVisitor removes a visitor with cleanup
func (vsm *VisitorServiceManager) RemoveVisitor(visitorID string) error {
	// Validate input
	if visitorID == "" {
		return errors.New("visitor ID cannot be empty")
	}

	// Get visitor before removal
	visitor, err := vsm.service.GetVisitor(visitorID)
	if err != nil {
		return err
	}

	// Set removal metadata
	visitor.SetMetadata("removed_at", time.Now())
	visitor.SetMetadata("removed_by", "visitor-service-manager")

	// Remove visitor
	err = vsm.service.RemoveVisitor(visitorID)
	if err != nil {
		return err
	}

	return nil
}

// ListVisitors returns all visitors with filtering
func (vsm *VisitorServiceManager) ListVisitors() []Visitor {
	visitors := vsm.service.ListVisitors()

	// Filter active visitors if needed
	if vsm.config.ValidationEnabled {
		activeVisitors := make([]Visitor, 0)
		for _, visitor := range visitors {
			if visitor.IsActive() {
				activeVisitors = append(activeVisitors, visitor)
			}
		}
		return activeVisitors
	}

	return visitors
}

// CreateElement creates a new element with validation
func (vsm *VisitorServiceManager) CreateElement(name, elementType, description string) (Element, error) {
	// Validate input
	if name == "" {
		return nil, errors.New("element name cannot be empty")
	}
	if elementType == "" {
		elementType = vsm.config.DefaultElementType
	}
	if description == "" {
		description = "Auto-generated element"
	}

	// Validate element type
	if vsm.config.ValidationEnabled {
		if !vsm.isValidElementType(elementType) {
			return nil, ErrInvalidElementType
		}
	}

	// Create element
	element, err := vsm.service.CreateElement(name, elementType, description)
	if err != nil {
		return nil, err
	}

	// Set metadata
	element.SetMetadata("created_by", "visitor-service-manager")
	element.SetMetadata("creation_time", time.Now())

	return element, nil
}

// GetElement retrieves an element with caching
func (vsm *VisitorServiceManager) GetElement(elementID string) (Element, error) {
	// Validate input
	if elementID == "" {
		return nil, errors.New("element ID cannot be empty")
	}

	// Get element
	element, err := vsm.service.GetElement(elementID)
	if err != nil {
		return nil, err
	}

	// Update last accessed time
	element.SetMetadata("last_accessed", time.Now())

	return element, nil
}

// RemoveElement removes an element with cleanup
func (vsm *VisitorServiceManager) RemoveElement(elementID string) error {
	// Validate input
	if elementID == "" {
		return errors.New("element ID cannot be empty")
	}

	// Get element before removal
	element, err := vsm.service.GetElement(elementID)
	if err != nil {
		return err
	}

	// Set removal metadata
	element.SetMetadata("removed_at", time.Now())
	element.SetMetadata("removed_by", "visitor-service-manager")

	// Remove element
	err = vsm.service.RemoveElement(elementID)
	if err != nil {
		return err
	}

	return nil
}

// ListElements returns all elements with filtering
func (vsm *VisitorServiceManager) ListElements() []Element {
	elements := vsm.service.ListElements()

	// Filter active elements if needed
	if vsm.config.ValidationEnabled {
		activeElements := make([]Element, 0)
		for _, element := range elements {
			if element.IsActive() {
				activeElements = append(activeElements, element)
			}
		}
		return activeElements
	}

	return elements
}

// CreateElementCollection creates a new element collection with validation
func (vsm *VisitorServiceManager) CreateElementCollection(name, description string) (ElementCollection, error) {
	// Validate input
	if name == "" {
		return nil, errors.New("collection name cannot be empty")
	}
	if description == "" {
		description = "Auto-generated collection"
	}

	// Create collection
	collection, err := vsm.service.CreateElementCollection(name, description)
	if err != nil {
		return nil, err
	}

	// Set metadata
	collection.SetMetadata("created_by", "visitor-service-manager")
	collection.SetMetadata("creation_time", time.Now())

	return collection, nil
}

// GetElementCollection retrieves an element collection with caching
func (vsm *VisitorServiceManager) GetElementCollection(collectionID string) (ElementCollection, error) {
	// Validate input
	if collectionID == "" {
		return nil, errors.New("collection ID cannot be empty")
	}

	// Get collection
	collection, err := vsm.service.GetElementCollection(collectionID)
	if err != nil {
		return nil, err
	}

	// Update last accessed time
	collection.SetMetadata("last_accessed", time.Now())

	return collection, nil
}

// RemoveElementCollection removes an element collection with cleanup
func (vsm *VisitorServiceManager) RemoveElementCollection(collectionID string) error {
	// Validate input
	if collectionID == "" {
		return errors.New("collection ID cannot be empty")
	}

	// Get collection before removal
	collection, err := vsm.service.GetElementCollection(collectionID)
	if err != nil {
		return err
	}

	// Set removal metadata
	collection.SetMetadata("removed_at", time.Now())
	collection.SetMetadata("removed_by", "visitor-service-manager")

	// Remove collection
	err = vsm.service.RemoveElementCollection(collectionID)
	if err != nil {
		return err
	}

	return nil
}

// ListElementCollections returns all element collections with filtering
func (vsm *VisitorServiceManager) ListElementCollections() []ElementCollection {
	collections := vsm.service.ListElementCollections()

	// Filter active collections if needed
	if vsm.config.ValidationEnabled {
		activeCollections := make([]ElementCollection, 0)
		for _, collection := range collections {
			if collection.IsActive() {
				activeCollections = append(activeCollections, collection)
			}
		}
		return activeCollections
	}

	return collections
}

// VisitElement performs a visit operation with timeout and validation
func (vsm *VisitorServiceManager) VisitElement(visitorID, elementID string) error {
	// Validate input
	if visitorID == "" {
		return errors.New("visitor ID cannot be empty")
	}
	if elementID == "" {
		return errors.New("element ID cannot be empty")
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), vsm.config.VisitTimeout)
	defer cancel()

	// Perform visit in goroutine with timeout
	done := make(chan error, 1)
	go func() {
		done <- vsm.service.VisitElement(visitorID, elementID)
	}()

	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ErrVisitTimeout
	}
}

// VisitElementCollection performs a visit operation on an element collection with timeout and validation
func (vsm *VisitorServiceManager) VisitElementCollection(visitorID, collectionID string) error {
	// Validate input
	if visitorID == "" {
		return errors.New("visitor ID cannot be empty")
	}
	if collectionID == "" {
		return errors.New("collection ID cannot be empty")
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), vsm.config.VisitTimeout)
	defer cancel()

	// Perform visit in goroutine with timeout
	done := make(chan error, 1)
	go func() {
		done <- vsm.service.VisitElementCollection(visitorID, collectionID)
	}()

	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ErrVisitTimeout
	}
}

// GetVisitHistory returns the visit history with filtering
func (vsm *VisitorServiceManager) GetVisitHistory() []VisitRecord {
	return vsm.service.GetVisitHistory()
}

// ClearVisitHistory clears the visit history
func (vsm *VisitorServiceManager) ClearVisitHistory() error {
	return vsm.service.ClearVisitHistory()
}

// GetVisitorStats returns visitor statistics
func (vsm *VisitorServiceManager) GetVisitorStats() map[string]interface{} {
	return vsm.service.GetVisitorStats()
}

// Cleanup performs cleanup operations
func (vsm *VisitorServiceManager) Cleanup() error {
	return vsm.service.Cleanup()
}

// GetService returns the underlying visitor service
func (vsm *VisitorServiceManager) GetService() *VisitorService {
	return vsm.service
}

// GetConfig returns the service configuration
func (vsm *VisitorServiceManager) GetConfig() *VisitorConfig {
	return vsm.config
}

// SetConfig sets the service configuration
func (vsm *VisitorServiceManager) SetConfig(config *VisitorConfig) {
	vsm.config = config
	vsm.service.SetConfig(config)
}

// isValidVisitorType checks if the visitor type is valid
func (vsm *VisitorServiceManager) isValidVisitorType(visitorType string) bool {
	for _, validType := range vsm.config.SupportedVisitorTypes {
		if validType == visitorType {
			return true
		}
	}
	return false
}

// isValidElementType checks if the element type is valid
func (vsm *VisitorServiceManager) isValidElementType(elementType string) bool {
	for _, validType := range vsm.config.SupportedElementTypes {
		if validType == elementType {
			return true
		}
	}
	return false
}

// GetVisitorCount returns the number of visitors
func (vsm *VisitorServiceManager) GetVisitorCount() int {
	return vsm.service.GetVisitorCount()
}

// GetElementCount returns the number of elements
func (vsm *VisitorServiceManager) GetElementCount() int {
	return vsm.service.GetElementCount()
}

// GetElementCollectionCount returns the number of element collections
func (vsm *VisitorServiceManager) GetElementCollectionCount() int {
	return vsm.service.GetElementCollectionCount()
}

// GetServiceInfo returns service information
func (vsm *VisitorServiceManager) GetServiceInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":                     vsm.config.Name,
		"version":                  vsm.config.Version,
		"description":              vsm.config.Description,
		"visitor_count":            vsm.service.GetVisitorCount(),
		"element_count":            vsm.service.GetElementCount(),
		"element_collection_count": vsm.service.GetElementCollectionCount(),
		"visit_history_count":      len(vsm.service.GetVisitHistory()),
		"created_at":               vsm.service.GetCreatedAt(),
		"updated_at":               vsm.service.GetUpdatedAt(),
		"active":                   vsm.service.IsActive(),
		"metadata":                 vsm.service.GetMetadata(),
	}
}

// GetHealthStatus returns the health status of the service
func (vsm *VisitorServiceManager) GetHealthStatus() map[string]interface{} {
	stats := vsm.service.GetVisitorStats()

	healthStatus := map[string]interface{}{
		"status": "healthy",
		"checks": map[string]interface{}{
			"visitors": map[string]interface{}{
				"status": "healthy",
				"count":  stats["total_visitors"],
			},
			"elements": map[string]interface{}{
				"status": "healthy",
				"count":  stats["total_elements"],
			},
			"element_collections": map[string]interface{}{
				"status": "healthy",
				"count":  stats["total_element_collections"],
			},
			"visit_history": map[string]interface{}{
				"status": "healthy",
				"count":  stats["total_visits"],
			},
		},
		"timestamp": time.Now(),
	}

	// Check for potential issues
	if vsm.service.GetVisitorCount() >= vsm.config.MaxVisitors {
		healthStatus["checks"].(map[string]interface{})["visitors"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["visitors"].(map[string]interface{})["message"] = "Maximum visitors reached"
	}

	if vsm.service.GetElementCount() >= vsm.config.MaxElements {
		healthStatus["checks"].(map[string]interface{})["elements"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["elements"].(map[string]interface{})["message"] = "Maximum elements reached"
	}

	if vsm.service.GetElementCollectionCount() >= vsm.config.MaxElementCollections {
		healthStatus["checks"].(map[string]interface{})["element_collections"].(map[string]interface{})["status"] = "warning"
		healthStatus["checks"].(map[string]interface{})["element_collections"].(map[string]interface{})["message"] = "Maximum element collections reached"
	}

	return healthStatus
}
