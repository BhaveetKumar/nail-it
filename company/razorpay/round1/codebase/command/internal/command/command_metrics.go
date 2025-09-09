package command

import (
	"fmt"
	"sync"
	"time"
)

// CommandMetricsImpl implements CommandMetrics interface
type CommandMetricsImpl struct {
	metrics map[string]*CommandMetricsData
	mu      sync.RWMutex
}

// NewCommandMetrics creates a new command metrics
func NewCommandMetrics() *CommandMetricsImpl {
	return &CommandMetricsImpl{
		metrics: make(map[string]*CommandMetricsData),
	}
}

// RecordCommandExecution records command execution metrics
func (cm *CommandMetricsImpl) RecordCommandExecution(commandType string, duration time.Duration, success bool) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	metrics, exists := cm.metrics[commandType]
	if !exists {
		metrics = &CommandMetricsData{
			CommandType:          commandType,
			TotalExecutions:      0,
			SuccessfulExecutions: 0,
			FailedExecutions:     0,
			AverageDuration:      0,
			MinDuration:          duration,
			MaxDuration:          duration,
			LastExecution:        time.Now(),
			SuccessRate:          0,
			Availability:         0,
		}
		cm.metrics[commandType] = metrics
	}

	// Update metrics
	metrics.TotalExecutions++
	if success {
		metrics.SuccessfulExecutions++
	} else {
		metrics.FailedExecutions++
	}

	// Update duration metrics
	if duration < metrics.MinDuration {
		metrics.MinDuration = duration
	}
	if duration > metrics.MaxDuration {
		metrics.MaxDuration = duration
	}

	// Calculate average duration
	totalDuration := metrics.AverageDuration * time.Duration(metrics.TotalExecutions-1)
	metrics.AverageDuration = (totalDuration + duration) / time.Duration(metrics.TotalExecutions)

	// Calculate success rate
	metrics.SuccessRate = float64(metrics.SuccessfulExecutions) / float64(metrics.TotalExecutions) * 100

	// Calculate availability (simplified)
	metrics.Availability = metrics.SuccessRate

	metrics.LastExecution = time.Now()
}

// GetCommandMetrics returns metrics for a specific command type
func (cm *CommandMetricsImpl) GetCommandMetrics(commandType string) (*CommandMetricsData, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	metrics, exists := cm.metrics[commandType]
	if !exists {
		return nil, fmt.Errorf("metrics not found for command type: %s", commandType)
	}

	// Return a copy to avoid race conditions
	return &CommandMetricsData{
		CommandType:          metrics.CommandType,
		TotalExecutions:      metrics.TotalExecutions,
		SuccessfulExecutions: metrics.SuccessfulExecutions,
		FailedExecutions:     metrics.FailedExecutions,
		AverageDuration:      metrics.AverageDuration,
		MinDuration:          metrics.MinDuration,
		MaxDuration:          metrics.MaxDuration,
		LastExecution:        metrics.LastExecution,
		SuccessRate:          metrics.SuccessRate,
		Availability:         metrics.Availability,
	}, nil
}

// GetAllMetrics returns metrics for all command types
func (cm *CommandMetricsImpl) GetAllMetrics() (map[string]*CommandMetricsData, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Return a copy to avoid race conditions
	result := make(map[string]*CommandMetricsData)
	for name, metrics := range cm.metrics {
		result[name] = &CommandMetricsData{
			CommandType:          metrics.CommandType,
			TotalExecutions:      metrics.TotalExecutions,
			SuccessfulExecutions: metrics.SuccessfulExecutions,
			FailedExecutions:     metrics.FailedExecutions,
			AverageDuration:      metrics.AverageDuration,
			MinDuration:          metrics.MinDuration,
			MaxDuration:          metrics.MaxDuration,
			LastExecution:        metrics.LastExecution,
			SuccessRate:          metrics.SuccessRate,
			Availability:         metrics.Availability,
		}
	}

	return result, nil
}

// ResetMetrics resets metrics for a specific command type
func (cm *CommandMetricsImpl) ResetMetrics(commandType string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if commandType == "" {
		return fmt.Errorf("command type cannot be empty")
	}

	if _, exists := cm.metrics[commandType]; !exists {
		return fmt.Errorf("metrics not found for command type: %s", commandType)
	}

	cm.metrics[commandType] = &CommandMetricsData{
		CommandType:          commandType,
		TotalExecutions:      0,
		SuccessfulExecutions: 0,
		FailedExecutions:     0,
		AverageDuration:      0,
		MinDuration:          0,
		MaxDuration:          0,
		LastExecution:        time.Time{},
		SuccessRate:          0,
		Availability:         0,
	}

	return nil
}

// ResetAllMetrics resets metrics for all command types
func (cm *CommandMetricsImpl) ResetAllMetrics() error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.metrics = make(map[string]*CommandMetricsData)
	return nil
}
