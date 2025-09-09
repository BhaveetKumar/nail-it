package visitor

import (
	"time"
)

// VisitorService implements the VisitorManager interface
type VisitorService struct {
	config             *VisitorConfig
	visitors           map[string]Visitor
	elements           map[string]Element
	elementCollections map[string]ElementCollection
	visitHistory       []VisitRecord
	createdAt          time.Time
	updatedAt          time.Time
}

// NewVisitorService creates a new visitor service
func NewVisitorService(config *VisitorConfig) *VisitorService {
	return &VisitorService{
		config:             config,
		visitors:           make(map[string]Visitor),
		elements:           make(map[string]Element),
		elementCollections: make(map[string]ElementCollection),
		visitHistory:       make([]VisitRecord, 0),
		createdAt:          time.Now(),
		updatedAt:          time.Now(),
	}
}

// CreateVisitor creates a new visitor
func (vs *VisitorService) CreateVisitor(name, visitorType, description string) (Visitor, error) {
	if len(vs.visitors) >= vs.config.MaxVisitors {
		return nil, ErrMaxVisitorsReached
	}

	var visitor Visitor
	switch visitorType {
	case "validation":
		visitor = NewValidationVisitor(name, description)
	case "processing":
		visitor = NewProcessingVisitor(name, description)
	case "analytics":
		visitor = NewAnalyticsVisitor(name, description)
	default:
		visitor = &ConcreteVisitor{
			ID:          generateID(),
			Type:        visitorType,
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
		}
	}

	vs.visitors[visitor.GetID()] = visitor
	vs.updatedAt = time.Now()

	return visitor, nil
}

// GetVisitor retrieves a visitor by ID
func (vs *VisitorService) GetVisitor(visitorID string) (Visitor, error) {
	visitor, exists := vs.visitors[visitorID]
	if !exists {
		return nil, ErrVisitorNotFound
	}
	return visitor, nil
}

// RemoveVisitor removes a visitor
func (vs *VisitorService) RemoveVisitor(visitorID string) error {
	if _, exists := vs.visitors[visitorID]; !exists {
		return ErrVisitorNotFound
	}

	delete(vs.visitors, visitorID)
	vs.updatedAt = time.Now()

	return nil
}

// ListVisitors returns all visitors
func (vs *VisitorService) ListVisitors() []Visitor {
	visitors := make([]Visitor, 0, len(vs.visitors))
	for _, visitor := range vs.visitors {
		visitors = append(visitors, visitor)
	}
	return visitors
}

// GetVisitorCount returns the number of visitors
func (vs *VisitorService) GetVisitorCount() int {
	return len(vs.visitors)
}

// CreateElement creates a new element
func (vs *VisitorService) CreateElement(name, elementType, description string) (Element, error) {
	if len(vs.elements) >= vs.config.MaxElements {
		return nil, ErrMaxElementsReached
	}

	var element Element
	switch elementType {
	case "document":
		element = NewDocumentElement(name, description, "text/plain", "")
	case "data":
		element = NewDataElement(name, description, "string", "")
	case "service":
		element = NewServiceElement(name, description, "", "GET")
	default:
		element = &ConcreteElement{
			ID:          generateID(),
			Type:        elementType,
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
		}
	}

	vs.elements[element.GetID()] = element
	vs.updatedAt = time.Now()

	return element, nil
}

// GetElement retrieves an element by ID
func (vs *VisitorService) GetElement(elementID string) (Element, error) {
	element, exists := vs.elements[elementID]
	if !exists {
		return nil, ErrElementNotFound
	}
	return element, nil
}

// RemoveElement removes an element
func (vs *VisitorService) RemoveElement(elementID string) error {
	if _, exists := vs.elements[elementID]; !exists {
		return ErrElementNotFound
	}

	delete(vs.elements, elementID)
	vs.updatedAt = time.Now()

	return nil
}

// ListElements returns all elements
func (vs *VisitorService) ListElements() []Element {
	elements := make([]Element, 0, len(vs.elements))
	for _, element := range vs.elements {
		elements = append(elements, element)
	}
	return elements
}

// GetElementCount returns the number of elements
func (vs *VisitorService) GetElementCount() int {
	return len(vs.elements)
}

// CreateElementCollection creates a new element collection
func (vs *VisitorService) CreateElementCollection(name, description string) (ElementCollection, error) {
	if len(vs.elementCollections) >= vs.config.MaxElementCollections {
		return nil, ErrMaxElementCollectionsReached
	}

	collection := &ConcreteElementCollection{
		ID:          generateID(),
		Name:        name,
		Description: description,
		Elements:    make(map[string]Element),
		Metadata:    make(map[string]interface{}),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Active:      true,
	}

	vs.elementCollections[collection.GetID()] = collection
	vs.updatedAt = time.Now()

	return collection, nil
}

// GetElementCollection retrieves an element collection by ID
func (vs *VisitorService) GetElementCollection(collectionID string) (ElementCollection, error) {
	collection, exists := vs.elementCollections[collectionID]
	if !exists {
		return nil, ErrElementCollectionNotFound
	}
	return collection, nil
}

// RemoveElementCollection removes an element collection
func (vs *VisitorService) RemoveElementCollection(collectionID string) error {
	if _, exists := vs.elementCollections[collectionID]; !exists {
		return ErrElementCollectionNotFound
	}

	delete(vs.elementCollections, collectionID)
	vs.updatedAt = time.Now()

	return nil
}

// ListElementCollections returns all element collections
func (vs *VisitorService) ListElementCollections() []ElementCollection {
	collections := make([]ElementCollection, 0, len(vs.elementCollections))
	for _, collection := range vs.elementCollections {
		collections = append(collections, collection)
	}
	return collections
}

// GetElementCollectionCount returns the number of element collections
func (vs *VisitorService) GetElementCollectionCount() int {
	return len(vs.elementCollections)
}

// VisitElement performs a visit operation
func (vs *VisitorService) VisitElement(visitorID, elementID string) error {
	visitor, err := vs.GetVisitor(visitorID)
	if err != nil {
		return err
	}

	element, err := vs.GetElement(elementID)
	if err != nil {
		return err
	}

	startTime := time.Now()
	err = element.Accept(visitor)
	duration := time.Since(startTime)

	// Record the visit
	visitRecord := VisitRecord{
		ID:          generateID(),
		VisitorID:   visitorID,
		VisitorName: visitor.GetName(),
		VisitorType: visitor.GetType(),
		ElementID:   elementID,
		ElementName: element.GetName(),
		ElementType: element.GetType(),
		VisitTime:   startTime,
		Duration:    duration,
		Success:     err == nil,
		Error:       "",
		Metadata:    make(map[string]interface{}),
	}

	if err != nil {
		visitRecord.Error = err.Error()
	}

	vs.visitHistory = append(vs.visitHistory, visitRecord)

	// Trim history if it exceeds the maximum
	if len(vs.visitHistory) > vs.config.MaxVisitHistory {
		vs.visitHistory = vs.visitHistory[1:]
	}

	vs.updatedAt = time.Now()

	return err
}

// VisitElementCollection performs a visit operation on an element collection
func (vs *VisitorService) VisitElementCollection(visitorID, collectionID string) error {
	visitor, err := vs.GetVisitor(visitorID)
	if err != nil {
		return err
	}

	collection, err := vs.GetElementCollection(collectionID)
	if err != nil {
		return err
	}

	startTime := time.Now()
	err = collection.Accept(visitor)
	duration := time.Since(startTime)

	// Record the visit
	visitRecord := VisitRecord{
		ID:           generateID(),
		VisitorID:    visitorID,
		VisitorName:  visitor.GetName(),
		VisitorType:  visitor.GetType(),
		ElementID:    "",
		ElementName:  "",
		ElementType:  "",
		CollectionID: collectionID,
		VisitTime:    startTime,
		Duration:     duration,
		Success:      err == nil,
		Error:        "",
		Metadata:     make(map[string]interface{}),
	}

	if err != nil {
		visitRecord.Error = err.Error()
	}

	vs.visitHistory = append(vs.visitHistory, visitRecord)

	// Trim history if it exceeds the maximum
	if len(vs.visitHistory) > vs.config.MaxVisitHistory {
		vs.visitHistory = vs.visitHistory[1:]
	}

	vs.updatedAt = time.Now()

	return err
}

// GetVisitHistory returns the visit history
func (vs *VisitorService) GetVisitHistory() []VisitRecord {
	return vs.visitHistory
}

// ClearVisitHistory clears the visit history
func (vs *VisitorService) ClearVisitHistory() error {
	vs.visitHistory = make([]VisitRecord, 0)
	vs.updatedAt = time.Now()
	return nil
}

// GetVisitorStats returns visitor statistics
func (vs *VisitorService) GetVisitorStats() map[string]interface{} {
	stats := map[string]interface{}{
		"total_visitors":            len(vs.visitors),
		"total_elements":            len(vs.elements),
		"total_element_collections": len(vs.elementCollections),
		"total_visits":              len(vs.visitHistory),
		"successful_visits":         0,
		"failed_visits":             0,
		"average_visit_duration":    0,
		"visitors":                  make(map[string]interface{}),
		"elements":                  make(map[string]interface{}),
		"element_collections":       make(map[string]interface{}),
	}

	// Calculate visit statistics
	var totalDuration time.Duration
	successfulVisits := 0
	failedVisits := 0

	for _, visit := range vs.visitHistory {
		totalDuration += visit.Duration
		if visit.Success {
			successfulVisits++
		} else {
			failedVisits++
		}
	}

	stats["successful_visits"] = successfulVisits
	stats["failed_visits"] = failedVisits

	if len(vs.visitHistory) > 0 {
		stats["average_visit_duration"] = totalDuration / time.Duration(len(vs.visitHistory))
	}

	// Add visitor details
	visitorDetails := make(map[string]interface{})
	for _, visitor := range vs.visitors {
		visitorDetails[visitor.GetID()] = map[string]interface{}{
			"name":        visitor.GetName(),
			"type":        visitor.GetType(),
			"description": visitor.GetDescription(),
			"active":      visitor.IsActive(),
			"created_at":  visitor.GetCreatedAt(),
			"updated_at":  visitor.GetUpdatedAt(),
		}
	}
	stats["visitors"] = visitorDetails

	// Add element details
	elementDetails := make(map[string]interface{})
	for _, element := range vs.elements {
		elementDetails[element.GetID()] = map[string]interface{}{
			"name":        element.GetName(),
			"type":        element.GetType(),
			"description": element.GetDescription(),
			"active":      element.IsActive(),
			"created_at":  element.GetCreatedAt(),
			"updated_at":  element.GetUpdatedAt(),
		}
	}
	stats["elements"] = elementDetails

	// Add element collection details
	collectionDetails := make(map[string]interface{})
	for _, collection := range vs.elementCollections {
		collectionDetails[collection.GetID()] = map[string]interface{}{
			"name":          collection.GetName(),
			"description":   collection.GetDescription(),
			"active":        collection.IsActive(),
			"created_at":    collection.GetCreatedAt(),
			"updated_at":    collection.GetUpdatedAt(),
			"element_count": collection.GetElementCount(),
		}
	}
	stats["element_collections"] = collectionDetails

	return stats
}

// Cleanup performs cleanup operations
func (vs *VisitorService) Cleanup() error {
	// Clear all data
	vs.visitors = make(map[string]Visitor)
	vs.elements = make(map[string]Element)
	vs.elementCollections = make(map[string]ElementCollection)
	vs.visitHistory = make([]VisitRecord, 0)
	vs.updatedAt = time.Now()

	return nil
}

// GetConfig returns the service configuration
func (vs *VisitorService) GetConfig() *VisitorConfig {
	return vs.config
}

// SetConfig sets the service configuration
func (vs *VisitorService) SetConfig(config *VisitorConfig) {
	vs.config = config
	vs.updatedAt = time.Now()
}

// GetCreatedAt returns the service creation time
func (vs *VisitorService) GetCreatedAt() time.Time {
	return vs.createdAt
}

// GetUpdatedAt returns the service last update time
func (vs *VisitorService) GetUpdatedAt() time.Time {
	return vs.updatedAt
}

// IsActive returns whether the service is active
func (vs *VisitorService) IsActive() bool {
	return true
}

// SetActive sets the service active status
func (vs *VisitorService) SetActive(active bool) {
	vs.updatedAt = time.Now()
}

// GetMetadata returns the service metadata
func (vs *VisitorService) GetMetadata() map[string]interface{} {
	return map[string]interface{}{
		"name":                    vs.config.Name,
		"version":                 vs.config.Version,
		"description":             vs.config.Description,
		"max_visitors":            vs.config.MaxVisitors,
		"max_elements":            vs.config.MaxElements,
		"max_element_collections": vs.config.MaxElementCollections,
		"max_visit_history":       vs.config.MaxVisitHistory,
		"visit_timeout":           vs.config.VisitTimeout,
		"cleanup_interval":        vs.config.CleanupInterval,
		"validation_enabled":      vs.config.ValidationEnabled,
		"caching_enabled":         vs.config.CachingEnabled,
		"monitoring_enabled":      vs.config.MonitoringEnabled,
		"auditing_enabled":        vs.config.AuditingEnabled,
		"supported_visitor_types": vs.config.SupportedVisitorTypes,
		"supported_element_types": vs.config.SupportedElementTypes,
		"default_visitor_type":    vs.config.DefaultVisitorType,
		"default_element_type":    vs.config.DefaultElementType,
		"validation_rules":        vs.config.ValidationRules,
		"metadata":                vs.config.Metadata,
	}
}

// SetMetadata sets the service metadata
func (vs *VisitorService) SetMetadata(key string, value interface{}) {
	if vs.config.Metadata == nil {
		vs.config.Metadata = make(map[string]interface{})
	}
	vs.config.Metadata[key] = value
	vs.updatedAt = time.Now()
}

// GetID returns the service ID
func (vs *VisitorService) GetID() string {
	return "visitor-service"
}

// GetName returns the service name
func (vs *VisitorService) GetName() string {
	return vs.config.Name
}

// GetDescription returns the service description
func (vs *VisitorService) GetDescription() string {
	return vs.config.Description
}

// GetType returns the service type
func (vs *VisitorService) GetType() string {
	return "visitor-service"
}
