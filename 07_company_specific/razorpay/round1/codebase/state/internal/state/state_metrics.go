package state

import (
	"fmt"
	"sync"
	"time"
)

// StateMetricsImpl implements StateMetrics interface
type StateMetricsImpl struct {
	metrics map[string]*StateMetricsData
	mu      sync.RWMutex
}

// NewStateMetrics creates a new state metrics
func NewStateMetrics() *StateMetricsImpl {
	return &StateMetricsImpl{
		metrics: make(map[string]*StateMetricsData),
	}
}

// RecordStateTransition records a state transition
func (sm *StateMetricsImpl) RecordStateTransition(entityType string, fromState string, toState string, duration time.Duration, success bool) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	key := fmt.Sprintf("%s_%s_%s", entityType, fromState, toState)
	metrics, exists := sm.metrics[key]
	if !exists {
		metrics = &StateMetricsData{
			EntityType:            entityType,
			TotalTransitions:      0,
			SuccessfulTransitions: 0,
			FailedTransitions:     0,
			AverageDuration:       0,
			MinDuration:           duration,
			MaxDuration:           duration,
			LastTransition:        time.Now(),
			SuccessRate:           0,
			Availability:          0,
		}
		sm.metrics[key] = metrics
	}

	// Update metrics
	metrics.TotalTransitions++
	if success {
		metrics.SuccessfulTransitions++
	} else {
		metrics.FailedTransitions++
	}

	// Update duration metrics
	if duration < metrics.MinDuration {
		metrics.MinDuration = duration
	}
	if duration > metrics.MaxDuration {
		metrics.MaxDuration = duration
	}

	// Calculate average duration
	totalDuration := metrics.AverageDuration * time.Duration(metrics.TotalTransitions-1)
	metrics.AverageDuration = (totalDuration + duration) / time.Duration(metrics.TotalTransitions)

	// Calculate success rate
	metrics.SuccessRate = float64(metrics.SuccessfulTransitions) / float64(metrics.TotalTransitions) * 100

	// Calculate availability (simplified)
	metrics.Availability = metrics.SuccessRate

	metrics.LastTransition = time.Now()
}

// GetStateMetrics returns metrics for a specific entity type
func (sm *StateMetricsImpl) GetStateMetrics(entityType string) (*StateMetricsData, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Aggregate metrics for the entity type
	var totalTransitions int64
	var successfulTransitions int64
	var failedTransitions int64
	var totalDuration time.Duration
	var minDuration time.Duration
	var maxDuration time.Duration
	var lastTransition time.Time
	var count int

	for key, metrics := range sm.metrics {
		if metrics.EntityType == entityType {
			totalTransitions += metrics.TotalTransitions
			successfulTransitions += metrics.SuccessfulTransitions
			failedTransitions += metrics.FailedTransitions
			totalDuration += metrics.AverageDuration * time.Duration(metrics.TotalTransitions)

			if count == 0 || metrics.MinDuration < minDuration {
				minDuration = metrics.MinDuration
			}
			if metrics.MaxDuration > maxDuration {
				maxDuration = metrics.MaxDuration
			}
			if metrics.LastTransition.After(lastTransition) {
				lastTransition = metrics.LastTransition
			}
			count++
		}
	}

	if count == 0 {
		return nil, fmt.Errorf("no metrics found for entity type: %s", entityType)
	}

	// Calculate aggregated metrics
	var averageDuration time.Duration
	if totalTransitions > 0 {
		averageDuration = totalDuration / time.Duration(totalTransitions)
	}

	successRate := float64(0)
	if totalTransitions > 0 {
		successRate = float64(successfulTransitions) / float64(totalTransitions) * 100
	}

	return &StateMetricsData{
		EntityType:            entityType,
		TotalTransitions:      totalTransitions,
		SuccessfulTransitions: successfulTransitions,
		FailedTransitions:     failedTransitions,
		AverageDuration:       averageDuration,
		MinDuration:           minDuration,
		MaxDuration:           maxDuration,
		LastTransition:        lastTransition,
		SuccessRate:           successRate,
		Availability:          successRate,
	}, nil
}

// GetAllMetrics returns metrics for all entity types
func (sm *StateMetricsImpl) GetAllMetrics() (map[string]*StateMetricsData, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Group metrics by entity type
	entityTypes := make(map[string]bool)
	for _, metrics := range sm.metrics {
		entityTypes[metrics.EntityType] = true
	}

	result := make(map[string]*StateMetricsData)
	for entityType := range entityTypes {
		metrics, err := sm.GetStateMetrics(entityType)
		if err == nil {
			result[entityType] = metrics
		}
	}

	return result, nil
}

// ResetMetrics resets metrics for a specific entity type
func (sm *StateMetricsImpl) ResetMetrics(entityType string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if entityType == "" {
		return fmt.Errorf("entity type cannot be empty")
	}

	// Remove all metrics for the entity type
	for key, metrics := range sm.metrics {
		if metrics.EntityType == entityType {
			delete(sm.metrics, key)
		}
	}

	return nil
}

// ResetAllMetrics resets metrics for all entity types
func (sm *StateMetricsImpl) ResetAllMetrics() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.metrics = make(map[string]*StateMetricsData)
	return nil
}

// GetTopTransitions returns top transitions by count
func (sm *StateMetricsImpl) GetTopTransitions(limit int) ([]*StateMetricsData, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	var transitions []*StateMetricsData
	for _, metrics := range sm.metrics {
		transitions = append(transitions, &StateMetricsData{
			EntityType:            metrics.EntityType,
			TotalTransitions:      metrics.TotalTransitions,
			SuccessfulTransitions: metrics.SuccessfulTransitions,
			FailedTransitions:     metrics.FailedTransitions,
			AverageDuration:       metrics.AverageDuration,
			MinDuration:           metrics.MinDuration,
			MaxDuration:           metrics.MaxDuration,
			LastTransition:        metrics.LastTransition,
			SuccessRate:           metrics.SuccessRate,
			Availability:          metrics.Availability,
		})
	}

	// Sort by total transitions
	for i := 0; i < len(transitions); i++ {
		for j := i + 1; j < len(transitions); j++ {
			if transitions[i].TotalTransitions < transitions[j].TotalTransitions {
				transitions[i], transitions[j] = transitions[j], transitions[i]
			}
		}
	}

	// Return top N transitions
	if limit > 0 && limit < len(transitions) {
		transitions = transitions[:limit]
	}

	return transitions, nil
}

// GetTransitionTrends returns transition trends over time
func (sm *StateMetricsImpl) GetTransitionTrends(entityType string, duration time.Duration) ([]*StateTrend, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// For now, return a simple trend based on current metrics
	// In a real implementation, you would store historical data
	var trends []*StateTrend

	for key, metrics := range sm.metrics {
		if metrics.EntityType == entityType {
			trends = append(trends, &StateTrend{
				Timestamp:   metrics.LastTransition,
				StateName:   key,
				EntityType:  entityType,
				Count:       int(metrics.TotalTransitions),
				SuccessRate: metrics.SuccessRate,
				AvgDuration: metrics.AverageDuration,
			})
		}
	}

	return trends, nil
}
