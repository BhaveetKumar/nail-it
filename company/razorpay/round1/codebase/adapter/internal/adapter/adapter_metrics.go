package adapter

import (
	"fmt"
	"sync"
	"time"
)

// AdapterMetricsImpl implements AdapterMetrics interface
type AdapterMetricsImpl struct {
	metrics map[string]map[string]*AdapterMetricsData
	mu      sync.RWMutex
}

// NewAdapterMetrics creates a new adapter metrics
func NewAdapterMetrics() *AdapterMetricsImpl {
	return &AdapterMetricsImpl{
		metrics: make(map[string]map[string]*AdapterMetricsData),
	}
}

// RecordAdapterCall records an adapter call
func (am *AdapterMetricsImpl) RecordAdapterCall(adapterType string, adapterName string, duration time.Duration, success bool) {
	am.mu.Lock()
	defer am.mu.Unlock()
	
	// Initialize adapter type map if it doesn't exist
	if am.metrics[adapterType] == nil {
		am.metrics[adapterType] = make(map[string]*AdapterMetricsData)
	}
	
	// Initialize adapter metrics if it doesn't exist
	if am.metrics[adapterType][adapterName] == nil {
		am.metrics[adapterType][adapterName] = &AdapterMetricsData{
			AdapterType: adapterType,
			AdapterName: adapterName,
			TotalCalls: 0,
			SuccessfulCalls: 0,
			FailedCalls: 0,
			AverageDuration: 0,
			MinDuration: duration,
			MaxDuration: duration,
			LastCallTime: time.Now(),
			SuccessRate: 0,
			Availability: 0,
		}
	}
	
	metrics := am.metrics[adapterType][adapterName]
	
	// Update metrics
	metrics.TotalCalls++
	if success {
		metrics.SuccessfulCalls++
	} else {
		metrics.FailedCalls++
	}
	
	// Update duration metrics
	if duration < metrics.MinDuration {
		metrics.MinDuration = duration
	}
	if duration > metrics.MaxDuration {
		metrics.MaxDuration = duration
	}
	
	// Calculate average duration
	totalDuration := metrics.AverageDuration * time.Duration(metrics.TotalCalls-1)
	metrics.AverageDuration = (totalDuration + duration) / time.Duration(metrics.TotalCalls)
	
	// Calculate success rate
	metrics.SuccessRate = float64(metrics.SuccessfulCalls) / float64(metrics.TotalCalls) * 100
	
	// Calculate availability (simplified)
	metrics.Availability = metrics.SuccessRate
	
	metrics.LastCallTime = time.Now()
}

// GetAdapterMetrics returns metrics for a specific adapter
func (am *AdapterMetricsImpl) GetAdapterMetrics(adapterType string, adapterName string) (*AdapterMetricsData, error) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	if adapterType == "" {
		return nil, fmt.Errorf("adapter type cannot be empty")
	}
	
	if adapterName == "" {
		return nil, fmt.Errorf("adapter name cannot be empty")
	}
	
	if am.metrics[adapterType] == nil {
		return nil, fmt.Errorf("adapter type not found: %s", adapterType)
	}
	
	metrics, exists := am.metrics[adapterType][adapterName]
	if !exists {
		return nil, fmt.Errorf("adapter not found: %s/%s", adapterType, adapterName)
	}
	
	// Return a copy to avoid race conditions
	return &AdapterMetricsData{
		AdapterType: metrics.AdapterType,
		AdapterName: metrics.AdapterName,
		TotalCalls: metrics.TotalCalls,
		SuccessfulCalls: metrics.SuccessfulCalls,
		FailedCalls: metrics.FailedCalls,
		AverageDuration: metrics.AverageDuration,
		MinDuration: metrics.MinDuration,
		MaxDuration: metrics.MaxDuration,
		LastCallTime: metrics.LastCallTime,
		SuccessRate: metrics.SuccessRate,
		Availability: metrics.Availability,
	}, nil
}

// GetAllMetrics returns metrics for all adapters
func (am *AdapterMetricsImpl) GetAllMetrics() (map[string]map[string]*AdapterMetricsData, error) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	// Return a copy to avoid race conditions
	result := make(map[string]map[string]*AdapterMetricsData)
	for adapterType, adapters := range am.metrics {
		result[adapterType] = make(map[string]*AdapterMetricsData)
		for adapterName, metrics := range adapters {
			result[adapterType][adapterName] = &AdapterMetricsData{
				AdapterType: metrics.AdapterType,
				AdapterName: metrics.AdapterName,
				TotalCalls: metrics.TotalCalls,
				SuccessfulCalls: metrics.SuccessfulCalls,
				FailedCalls: metrics.FailedCalls,
				AverageDuration: metrics.AverageDuration,
				MinDuration: metrics.MinDuration,
				MaxDuration: metrics.MaxDuration,
				LastCallTime: metrics.LastCallTime,
				SuccessRate: metrics.SuccessRate,
				Availability: metrics.Availability,
			}
		}
	}
	
	return result, nil
}

// ResetMetrics resets metrics for a specific adapter
func (am *AdapterMetricsImpl) ResetMetrics(adapterType string, adapterName string) error {
	am.mu.Lock()
	defer am.mu.Unlock()
	
	if adapterType == "" {
		return fmt.Errorf("adapter type cannot be empty")
	}
	
	if adapterName == "" {
		return fmt.Errorf("adapter name cannot be empty")
	}
	
	if am.metrics[adapterType] == nil {
		return fmt.Errorf("adapter type not found: %s", adapterType)
	}
	
	if _, exists := am.metrics[adapterType][adapterName]; !exists {
		return fmt.Errorf("adapter not found: %s/%s", adapterType, adapterName)
	}
	
	am.metrics[adapterType][adapterName] = &AdapterMetricsData{
		AdapterType: adapterType,
		AdapterName: adapterName,
		TotalCalls: 0,
		SuccessfulCalls: 0,
		FailedCalls: 0,
		AverageDuration: 0,
		MinDuration: 0,
		MaxDuration: 0,
		LastCallTime: time.Time{},
		SuccessRate: 0,
		Availability: 0,
	}
	
	return nil
}

// ResetAllMetrics resets metrics for all adapters
func (am *AdapterMetricsImpl) ResetAllMetrics() error {
	am.mu.Lock()
	defer am.mu.Unlock()
	
	am.metrics = make(map[string]map[string]*AdapterMetricsData)
	return nil
}

// GetTopAdapters returns top adapters by call count
func (am *AdapterMetricsImpl) GetTopAdapters(limit int) ([]*AdapterMetricsData, error) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	var adapters []*AdapterMetricsData
	for _, adapterType := range am.metrics {
		for _, metrics := range adapterType {
			adapters = append(adapters, &AdapterMetricsData{
				AdapterType: metrics.AdapterType,
				AdapterName: metrics.AdapterName,
				TotalCalls: metrics.TotalCalls,
				SuccessfulCalls: metrics.SuccessfulCalls,
				FailedCalls: metrics.FailedCalls,
				AverageDuration: metrics.AverageDuration,
				MinDuration: metrics.MinDuration,
				MaxDuration: metrics.MaxDuration,
				LastCallTime: metrics.LastCallTime,
				SuccessRate: metrics.SuccessRate,
				Availability: metrics.Availability,
			})
		}
	}
	
	// Sort by total calls
	for i := 0; i < len(adapters); i++ {
		for j := i + 1; j < len(adapters); j++ {
			if adapters[i].TotalCalls < adapters[j].TotalCalls {
				adapters[i], adapters[j] = adapters[j], adapters[i]
			}
		}
	}
	
	// Return top N adapters
	if limit > 0 && limit < len(adapters) {
		adapters = adapters[:limit]
	}
	
	return adapters, nil
}

// GetAdapterTrends returns adapter trends over time
func (am *AdapterMetricsImpl) GetAdapterTrends(adapterType string, adapterName string, duration time.Duration) ([]*AdapterTrend, error) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	
	// For now, return a simple trend based on current metrics
	// In a real implementation, you would store historical data
	var trends []*AdapterTrend
	
	if am.metrics[adapterType] != nil && am.metrics[adapterType][adapterName] != nil {
		metrics := am.metrics[adapterType][adapterName]
		trends = append(trends, &AdapterTrend{
			Timestamp:   metrics.LastCallTime,
			AdapterType: adapterType,
			AdapterName: adapterName,
			Count:       int(metrics.TotalCalls),
			SuccessRate: metrics.SuccessRate,
			AvgDuration: metrics.AverageDuration,
		})
	}
	
	return trends, nil
}
