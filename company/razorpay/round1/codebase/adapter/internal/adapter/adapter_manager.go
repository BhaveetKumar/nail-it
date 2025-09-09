package adapter

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// AdapterManagerImpl implements AdapterManager interface
type AdapterManagerImpl struct {
	adapters map[string]map[string]interface{}
	mu       sync.RWMutex
}

// NewAdapterManager creates a new adapter manager
func NewAdapterManager() *AdapterManagerImpl {
	return &AdapterManagerImpl{
		adapters: make(map[string]map[string]interface{}),
	}
}

// RegisterAdapter registers an adapter
func (am *AdapterManagerImpl) RegisterAdapter(adapterType string, adapter interface{}) error {
	am.mu.Lock()
	defer am.mu.Unlock()
	
	if adapterType == "" {
		return fmt.Errorf("adapter type cannot be empty")
	}
	
	if adapter == nil {
		return fmt.Errorf("adapter cannot be nil")
	}
	
	// Get adapter name
	var adapterName string
	switch a := adapter.(type) {
	case PaymentGateway:
		adapterName = a.GetGatewayName()
	case NotificationService:
		adapterName = a.GetServiceName()
	case DatabaseAdapter:
		adapterName = a.GetAdapterName()
	case CacheAdapter:
		adapterName = a.GetAdapterName()
	case MessageQueueAdapter:
		adapterName = a.GetAdapterName()
	case FileStorageAdapter:
		adapterName = a.GetAdapterName()
	case AuthenticationAdapter:
		adapterName = a.GetAdapterName()
	default:
		return fmt.Errorf("unsupported adapter type: %T", adapter)
	}
	
	if adapterName == "" {
		return fmt.Errorf("adapter name cannot be empty")
	}
	
	// Initialize adapter type map if it doesn't exist
	if am.adapters[adapterType] == nil {
		am.adapters[adapterType] = make(map[string]interface{})
	}
	
	am.adapters[adapterType][adapterName] = adapter
	return nil
}

// UnregisterAdapter unregisters an adapter
func (am *AdapterManagerImpl) UnregisterAdapter(adapterType string, adapterName string) error {
	am.mu.Lock()
	defer am.mu.Unlock()
	
	if adapterType == "" {
		return fmt.Errorf("adapter type cannot be empty")
	}
	
	if adapterName == "" {
		return fmt.Errorf("adapter name cannot be empty")
	}
	
	if am.adapters[adapterType] == nil {
		return fmt.Errorf("adapter type not found: %s", adapterType)
	}
	
	if _, exists := am.adapters[adapterType][adapterName]; !exists {
		return fmt.Errorf("adapter not found: %s/%s", adapterType, adapterName)
	}
	
	delete(am.adapters[adapterType], adapterName)
	
	// Remove adapter type map if empty
	if len(am.adapters[adapterType]) == 0 {
		delete(am.adapters, adapterType)
	}
	
	return nil
}

// GetAdapter retrieves an adapter by type and name
func (am *AdapterManagerImpl) GetAdapter(adapterType string, adapterName string) (interface{}, error) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	if adapterType == "" {
		return nil, fmt.Errorf("adapter type cannot be empty")
	}
	
	if adapterName == "" {
		return nil, fmt.Errorf("adapter name cannot be empty")
	}
	
	if am.adapters[adapterType] == nil {
		return nil, fmt.Errorf("adapter type not found: %s", adapterType)
	}
	
	adapter, exists := am.adapters[adapterType][adapterName]
	if !exists {
		return nil, fmt.Errorf("adapter not found: %s/%s", adapterType, adapterName)
	}
	
	return adapter, nil
}

// GetAdapters retrieves all adapters of a specific type
func (am *AdapterManagerImpl) GetAdapters(adapterType string) ([]interface{}, error) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	if adapterType == "" {
		return nil, fmt.Errorf("adapter type cannot be empty")
	}
	
	if am.adapters[adapterType] == nil {
		return []interface{}{}, nil
	}
	
	var adapters []interface{}
	for _, adapter := range am.adapters[adapterType] {
		adapters = append(adapters, adapter)
	}
	
	return adapters, nil
}

// GetAllAdapters retrieves all adapters
func (am *AdapterManagerImpl) GetAllAdapters() map[string][]interface{} {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	result := make(map[string][]interface{})
	for adapterType, adapters := range am.adapters {
		var adapterList []interface{}
		for _, adapter := range adapters {
			adapterList = append(adapterList, adapter)
		}
		result[adapterType] = adapterList
	}
	
	return result
}

// GetAdapterHealth returns the health status of an adapter
func (am *AdapterManagerImpl) GetAdapterHealth(adapterType string, adapterName string) (*AdapterHealth, error) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	if adapterType == "" {
		return nil, fmt.Errorf("adapter type cannot be empty")
	}
	
	if adapterName == "" {
		return nil, fmt.Errorf("adapter name cannot be empty")
	}
	
	if am.adapters[adapterType] == nil {
		return nil, fmt.Errorf("adapter type not found: %s", adapterType)
	}
	
	adapter, exists := am.adapters[adapterType][adapterName]
	if !exists {
		return nil, fmt.Errorf("adapter not found: %s/%s", adapterType, adapterName)
	}
	
	// Check adapter availability
	var isAvailable bool
	switch a := adapter.(type) {
	case PaymentGateway:
		isAvailable = a.IsAvailable()
	case NotificationService:
		isAvailable = a.IsAvailable()
	case DatabaseAdapter:
		isAvailable = a.IsConnected()
	case CacheAdapter:
		isAvailable = a.IsConnected()
	case MessageQueueAdapter:
		isAvailable = a.IsConnected()
	case FileStorageAdapter:
		isAvailable = a.IsAvailable()
	case AuthenticationAdapter:
		isAvailable = a.IsAvailable()
	default:
		isAvailable = true
	}
	
	status := "active"
	if !isAvailable {
		status = "inactive"
	}
	
	health := &AdapterHealth{
		AdapterType: adapterType,
		AdapterName: adapterName,
		Status:      status,
		Message:     fmt.Sprintf("Adapter %s/%s is %s", adapterType, adapterName, status),
		LastCheck:   time.Now(),
		Metrics:     make(map[string]interface{}),
	}
	
	return health, nil
}

// GetAdapterCount returns the total number of adapters
func (am *AdapterManagerImpl) GetAdapterCount() int {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	count := 0
	for _, adapters := range am.adapters {
		count += len(adapters)
	}
	
	return count
}

// GetAdapterCountByType returns the number of adapters by type
func (am *AdapterManagerImpl) GetAdapterCountByType() map[string]int {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	counts := make(map[string]int)
	for adapterType, adapters := range am.adapters {
		counts[adapterType] = len(adapters)
	}
	
	return counts
}

// GetAdapterStats returns statistics about adapters
func (am *AdapterManagerImpl) GetAdapterStats() map[string]interface{} {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	stats := map[string]interface{}{
		"total_adapters": 0,
		"by_type":        make(map[string]int),
		"types":          make([]string, 0),
	}
	
	totalCount := 0
	byType := make(map[string]int)
	types := make([]string, 0)
	
	for adapterType, adapters := range am.adapters {
		count := len(adapters)
		totalCount += count
		byType[adapterType] = count
		types = append(types, adapterType)
	}
	
	stats["total_adapters"] = totalCount
	stats["by_type"] = byType
	stats["types"] = types
	
	return stats
}

// ClearAdapters clears all adapters
func (am *AdapterManagerImpl) ClearAdapters() {
	am.mu.Lock()
	defer am.mu.Unlock()
	
	am.adapters = make(map[string]map[string]interface{})
}

// GetAvailableAdapters returns available adapters
func (am *AdapterManagerImpl) GetAvailableAdapters(adapterType string) ([]interface{}, error) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	if adapterType == "" {
		return nil, fmt.Errorf("adapter type cannot be empty")
	}
	
	if am.adapters[adapterType] == nil {
		return []interface{}{}, nil
	}
	
	var availableAdapters []interface{}
	for _, adapter := range am.adapters[adapterType] {
		var isAvailable bool
		switch a := adapter.(type) {
		case PaymentGateway:
			isAvailable = a.IsAvailable()
		case NotificationService:
			isAvailable = a.IsAvailable()
		case DatabaseAdapter:
			isAvailable = a.IsConnected()
		case CacheAdapter:
			isAvailable = a.IsConnected()
		case MessageQueueAdapter:
			isAvailable = a.IsConnected()
		case FileStorageAdapter:
			isAvailable = a.IsAvailable()
		case AuthenticationAdapter:
			isAvailable = a.IsAvailable()
		default:
			isAvailable = true
		}
		
		if isAvailable {
			availableAdapters = append(availableAdapters, adapter)
		}
	}
	
	return availableAdapters, nil
}
