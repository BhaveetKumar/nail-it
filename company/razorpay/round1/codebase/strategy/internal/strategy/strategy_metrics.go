package strategy

import (
	"fmt"
	"sync"
	"time"
)

// StrategyMetricsImpl implements StrategyMetrics interface
type StrategyMetricsImpl struct {
	metrics map[string]*StrategyMetricsData
	mu      sync.RWMutex
}

// NewStrategyMetrics creates a new strategy metrics
func NewStrategyMetrics() *StrategyMetricsImpl {
	return &StrategyMetricsImpl{
		metrics: make(map[string]*StrategyMetricsData),
	}
}

// RecordStrategyCall records a strategy call
func (sm *StrategyMetricsImpl) RecordStrategyCall(strategyName string, duration time.Duration, success bool) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	metrics, exists := sm.metrics[strategyName]
	if !exists {
		metrics = &StrategyMetricsData{
			StrategyName:    strategyName,
			TotalCalls:      0,
			SuccessfulCalls: 0,
			FailedCalls:     0,
			AverageDuration: 0,
			MinDuration:     duration,
			MaxDuration:     duration,
			LastCallTime:    time.Now(),
			SuccessRate:     0,
			Availability:    0,
		}
		sm.metrics[strategyName] = metrics
	}
	
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

// GetStrategyMetrics returns metrics for a specific strategy
func (sm *StrategyMetricsImpl) GetStrategyMetrics(strategyName string) (*StrategyMetricsData, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	metrics, exists := sm.metrics[strategyName]
	if !exists {
		return nil, fmt.Errorf("metrics not found for strategy: %s", strategyName)
	}
	
	// Return a copy to avoid race conditions
	return &StrategyMetricsData{
		StrategyName:    metrics.StrategyName,
		TotalCalls:      metrics.TotalCalls,
		SuccessfulCalls: metrics.SuccessfulCalls,
		FailedCalls:     metrics.FailedCalls,
		AverageDuration: metrics.AverageDuration,
		MinDuration:     metrics.MinDuration,
		MaxDuration:     metrics.MaxDuration,
		LastCallTime:    metrics.LastCallTime,
		SuccessRate:     metrics.SuccessRate,
		Availability:    metrics.Availability,
	}, nil
}

// GetAllMetrics returns metrics for all strategies
func (sm *StrategyMetricsImpl) GetAllMetrics() (map[string]*StrategyMetricsData, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	// Return a copy to avoid race conditions
	result := make(map[string]*StrategyMetricsData)
	for name, metrics := range sm.metrics {
		result[name] = &StrategyMetricsData{
			StrategyName:    metrics.StrategyName,
			TotalCalls:      metrics.TotalCalls,
			SuccessfulCalls: metrics.SuccessfulCalls,
			FailedCalls:     metrics.FailedCalls,
			AverageDuration: metrics.AverageDuration,
			MinDuration:     metrics.MinDuration,
			MaxDuration:     metrics.MaxDuration,
			LastCallTime:    metrics.LastCallTime,
			SuccessRate:     metrics.SuccessRate,
			Availability:    metrics.Availability,
		}
	}
	
	return result, nil
}

// ResetMetrics resets metrics for a specific strategy
func (sm *StrategyMetricsImpl) ResetMetrics(strategyName string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if strategyName == "" {
		return fmt.Errorf("strategy name cannot be empty")
	}
	
	if _, exists := sm.metrics[strategyName]; !exists {
		return fmt.Errorf("metrics not found for strategy: %s", strategyName)
	}
	
	sm.metrics[strategyName] = &StrategyMetricsData{
		StrategyName:    strategyName,
		TotalCalls:      0,
		SuccessfulCalls: 0,
		FailedCalls:     0,
		AverageDuration: 0,
		MinDuration:     0,
		MaxDuration:     0,
		LastCallTime:    time.Time{},
		SuccessRate:     0,
		Availability:    0,
	}
	
	return nil
}

// ResetAllMetrics resets metrics for all strategies
func (sm *StrategyMetricsImpl) ResetAllMetrics() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	sm.metrics = make(map[string]*StrategyMetricsData)
	return nil
}

// GetTopPerformingStrategies returns top performing strategies
func (sm *StrategyMetricsImpl) GetTopPerformingStrategies(limit int) ([]*StrategyMetricsData, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	var strategies []*StrategyMetricsData
	for _, metrics := range sm.metrics {
		strategies = append(strategies, &StrategyMetricsData{
			StrategyName:    metrics.StrategyName,
			TotalCalls:      metrics.TotalCalls,
			SuccessfulCalls: metrics.SuccessfulCalls,
			FailedCalls:     metrics.FailedCalls,
			AverageDuration: metrics.AverageDuration,
			MinDuration:     metrics.MinDuration,
			MaxDuration:     metrics.MaxDuration,
			LastCallTime:    metrics.LastCallTime,
			SuccessRate:     metrics.SuccessRate,
			Availability:    metrics.Availability,
		})
	}
	
	// Sort by success rate
	for i := 0; i < len(strategies); i++ {
		for j := i + 1; j < len(strategies); j++ {
			if strategies[i].SuccessRate < strategies[j].SuccessRate {
				strategies[i], strategies[j] = strategies[j], strategies[i]
			}
		}
	}
	
	// Return top N strategies
	if limit > 0 && limit < len(strategies) {
		strategies = strategies[:limit]
	}
	
	return strategies, nil
}

// GetWorstPerformingStrategies returns worst performing strategies
func (sm *StrategyMetricsImpl) GetWorstPerformingStrategies(limit int) ([]*StrategyMetricsData, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	var strategies []*StrategyMetricsData
	for _, metrics := range sm.metrics {
		strategies = append(strategies, &StrategyMetricsData{
			StrategyName:    metrics.StrategyName,
			TotalCalls:      metrics.TotalCalls,
			SuccessfulCalls: metrics.SuccessfulCalls,
			FailedCalls:     metrics.FailedCalls,
			AverageDuration: metrics.AverageDuration,
			MinDuration:     metrics.MinDuration,
			MaxDuration:     metrics.MaxDuration,
			LastCallTime:    metrics.LastCallTime,
			SuccessRate:     metrics.SuccessRate,
			Availability:    metrics.Availability,
		})
	}
	
	// Sort by success rate (ascending)
	for i := 0; i < len(strategies); i++ {
		for j := i + 1; j < len(strategies); j++ {
			if strategies[i].SuccessRate > strategies[j].SuccessRate {
				strategies[i], strategies[j] = strategies[j], strategies[i]
			}
		}
	}
	
	// Return top N strategies
	if limit > 0 && limit < len(strategies) {
		strategies = strategies[:limit]
	}
	
	return strategies, nil
}

// GetStrategyHealthScore returns health score for a strategy
func (sm *StrategyMetricsImpl) GetStrategyHealthScore(strategyName string) (float64, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	metrics, exists := sm.metrics[strategyName]
	if !exists {
		return 0, fmt.Errorf("metrics not found for strategy: %s", strategyName)
	}
	
	// Calculate health score based on multiple factors
	successRate := metrics.SuccessRate
	availability := metrics.Availability
	
	// Penalize for high average duration
	durationPenalty := 0.0
	if metrics.AverageDuration > 1*time.Second {
		durationPenalty = 10.0
	} else if metrics.AverageDuration > 500*time.Millisecond {
		durationPenalty = 5.0
	}
	
	// Calculate final health score
	healthScore := (successRate + availability) / 2 - durationPenalty
	
	// Ensure score is between 0 and 100
	if healthScore < 0 {
		healthScore = 0
	} else if healthScore > 100 {
		healthScore = 100
	}
	
	return healthScore, nil
}

// GetOverallHealthScore returns overall health score for all strategies
func (sm *StrategyMetricsImpl) GetOverallHealthScore() (float64, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	if len(sm.metrics) == 0 {
		return 0, fmt.Errorf("no metrics available")
	}
	
	var totalScore float64
	var count int
	
	for _, metrics := range sm.metrics {
		successRate := metrics.SuccessRate
		availability := metrics.Availability
		
		// Penalize for high average duration
		durationPenalty := 0.0
		if metrics.AverageDuration > 1*time.Second {
			durationPenalty = 10.0
		} else if metrics.AverageDuration > 500*time.Millisecond {
			durationPenalty = 5.0
		}
		
		// Calculate health score
		healthScore := (successRate + availability) / 2 - durationPenalty
		
		// Ensure score is between 0 and 100
		if healthScore < 0 {
			healthScore = 0
		} else if healthScore > 100 {
			healthScore = 100
		}
		
		totalScore += healthScore
		count++
	}
	
	return totalScore / float64(count), nil
}

// GetStrategyTrends returns trends for a strategy over time
func (sm *StrategyMetricsImpl) GetStrategyTrends(strategyName string, duration time.Duration) ([]*StrategyTrend, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	metrics, exists := sm.metrics[strategyName]
	if !exists {
		return nil, fmt.Errorf("metrics not found for strategy: %s", strategyName)
	}
	
	// For now, return a simple trend based on current metrics
	// In a real implementation, you would store historical data
	trends := []*StrategyTrend{
		{
			Timestamp:    metrics.LastCallTime,
			SuccessRate:  metrics.SuccessRate,
			Availability: metrics.Availability,
			AvgDuration:  metrics.AverageDuration,
		},
	}
	
	return trends, nil
}

// StrategyTrend represents a trend data point
type StrategyTrend struct {
	Timestamp    time.Time     `json:"timestamp"`
	SuccessRate  float64       `json:"success_rate"`
	Availability float64       `json:"availability"`
	AvgDuration  time.Duration `json:"avg_duration"`
}
